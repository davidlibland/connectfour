import tensorflow as tf
from tensorflow.python.framework.errors_impl import PermissionDeniedError

from src.file_utils import get_dir, build_checkpoint_file_name
from src.game import BatchGameState, BatchGame
from src.play_state import PlayState
import os

from src import nn

tf.enable_eager_execution()


class Policy:
    def __init__(self, player: PlayState=PlayState.X, num_rows=5, num_cols=5, lr=1e-3, root_dir=None):
        self._player = player
        self._container = tf.contrib.eager.EagerVariableStore()
        self._optimizer = tf.train.AdamOptimizer(lr)
        num_policy_layers = 2
        num_policy_filters = 32
        num_reward_layers = 2
        num_reward_filters = 32
        with self._container.as_default():
            # Policy network params
            self._p_conv_params = []
            for i in range(num_policy_layers):
                num_in_filters = 3 if i == 0 else num_policy_filters
                w = tf.get_variable(
                    "p_conv_w_%i" % i,
                    shape=[3, 3, num_in_filters, num_policy_filters], dtype=tf.float32,
                    initializer=tf.random_normal_initializer(0, 0.05)
                )
                b = tf.get_variable(
                    "p_conv_b_%i" % i,
                    shape=[num_policy_filters], dtype=tf.float32,
                    initializer=tf.constant_initializer(0.)
                )
                self._p_conv_params.append((w, b))
            self._p_conv_1_w = tf.get_variable(
                "p_conv_1_w_params",
                shape=[num_cols, num_policy_filters, 1], dtype=tf.float32,
                initializer=tf.random_normal_initializer(0, 0.05)
            )
            self._p_conv_1_b = tf.get_variable(
                "p_conv_1_b_params",
                shape=[1], dtype=tf.float32,
                initializer=tf.constant_initializer(0.)
            )

            # reward network params
            self._r_conv_params = []
            conv_height = num_rows
            conv_width = num_cols
            for i in range(num_reward_layers):
                num_in_filters = 3 if i == 0 else num_reward_filters
                w = tf.get_variable(
                    "r_conv_w_%i" % i,
                    shape=[3, 3, num_in_filters, num_reward_filters], dtype=tf.float32,
                    initializer=tf.random_normal_initializer(0, 0.05)
                )
                b = tf.get_variable(
                    "r_conv_b_%i" % i,
                    shape=[num_reward_filters], dtype=tf.float32,
                    initializer=tf.constant_initializer(0.)
                )
                self._r_conv_params.append((w, b))
                conv_height -= 2
                conv_width -= 2
            self._r_dense_param = tf.get_variable(
                "r_params",
                shape=[conv_height, conv_width, num_reward_filters], dtype=tf.float32,
                initializer=tf.random_normal_initializer(0, 0.05)
            )
        self._global_step = tf.train.get_or_create_global_step()
        self._saver = tf.train.Saver(
            var_list=self._container.trainable_variables()+[self._global_step],
            save_relative_paths=True,
            sharded=False
        )
        if root_dir is None:
            root_dir = get_dir("Checkpoints")
        self._root_dir = root_dir
        self._summary_writer = tf.contrib.summary.create_file_writer("{}/logs".format(self._root_dir), flush_millis = 10000)
        self._summary_writer.set_as_default()

    def reward_logits(self, gs: BatchGameState):
        assert gs.turn == self._player, "Can't play on this turn"
        gs_array = gs.as_array()
        if self.player == PlayState.O:
            gs_array = tf.gather(gs_array, [0, 2, 1], axis=-1)
        x = tf.constant(gs_array, dtype=tf.float32)
        for w,b in self._r_conv_params:
            x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="VALID") + b
        logits = tf.einsum("nijk,ijk->n", x, self._r_dense_param)
        return logits

    def ln_pi(self, gs: BatchGameState):
        logits = self.logits(gs)
        return logits - tf.reduce_logsumexp(logits, axis=1)[:,tf.newaxis]

    def logits(self, gs: BatchGameState):
        assert gs.turn == self._player, "Can't play on this turn"
        gs_array = gs.as_array()
        if self.player == PlayState.O:
            gs_array = tf.gather(gs_array, [0, 2, 1], axis=-1)
        x = tf.constant(gs_array, dtype=tf.float32)
        for w,b in self._p_conv_params:
            x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME") + b
        y = tf.reduce_max(x, axis=1)
        logits = tf.nn.conv1d(y, self._p_conv_1_w, stride=1, padding="SAME") \
                 + self._p_conv_1_b
        return tf.reshape(logits, [gs.batch_size, -1])

    def learn_from_games(self, games: BatchGame, alpha=0.9, verbose=True):
        self._global_step.assign_add(1)
        last_state = games.cur_state
        rewards = tf.zeros(last_state.batch_size,dtype=tf.float32)
        for i, state in enumerate(games.reversed_states):
            if state.turn != self.player:
                rewards_list = [
                    1. if winner == self.player else
                        float(reward * alpha) if winner is None else 0.
                    for reward, winner in zip(rewards, state.winners())
                ]
                rewards = tf.constant(rewards_list, dtype=tf.float32)
                num_rewards = sum(1 for r in rewards if int(r) == 1)
                if verbose and num_rewards > 0:
                    print("Noticed %d rewards on turn -%d for %s"
                          % (num_rewards, i, self.player))
            else:
                # Update the reward function
                with tf.GradientTape() as gr_tape, self._container.as_default():
                    expected_reward_l = self.reward_logits(state)
                    loss = tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=rewards,
                        logits=expected_reward_l
                    )
                with tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.scalar("%s reward loss" % self.player.name, loss)
                rw_gradients = gr_tape.gradient(
                    loss,
                    self._container.trainable_variables()
                )
                self._optimizer.apply_gradients(
                    zip(rw_gradients, self._container.trainable_variables())
                )

                # Update the policy function
                with tf.GradientTape() as gr_tape, self._container.as_default():
                    td = rewards - tf.sigmoid(expected_reward_l)
                    un_weighted_ln_pi = -td[:,tf.newaxis]*self.ln_pi(state)
                    rw_ln_pi = un_weighted_ln_pi*(alpha**i)

                with tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.scalar("%s loss" % self.player.name, un_weighted_ln_pi)
                gradients = gr_tape.gradient(
                    rw_ln_pi,
                    self._container.trainable_variables()
                )
                self._optimizer.apply_gradients(
                    zip(gradients, self._container.trainable_variables())
                )

    @property
    def player(self):
        return self._player

    def save(self, descriptor):
        path = build_checkpoint_file_name(self._root_dir, descriptor)
        fp = self._saver.save(None, path, global_step=self._global_step)
        print("model saved at %s" % fp)

    @classmethod
    def load(cls, ckpt_dir, player=PlayState.X, num_rows=5, num_cols=5):
        root_dir = os.path.dirname(ckpt_dir)
        p = Policy(player, num_rows, num_cols, root_dir=root_dir)
        p._saver.restore(None, tf.train.latest_checkpoint(ckpt_dir))
        return p


class AI:
    def __init__(self, pi: Policy):
        self._pi = pi

    def next_moves(self, gs: BatchGameState):
        impossible_actions = tf.logical_not(gs.next_actions())
        logits = self._pi.logits(gs)
        return nn.sample_masked_multinomial(logits, impossible_actions, axis=1)

    @property
    def player(self):
        return self._pi.player

    def learn_from_games(self, game: BatchGame, alpha=0.5, verbose=True):
        self._pi.learn_from_games(game, alpha, verbose=verbose)

    def save(self, fp):
        self._pi.save(fp)

    @classmethod
    def load(cls, fp, player=PlayState.X):
        try:
            pi = Policy.load(fp, player)
        except (FileNotFoundError, ValueError, PermissionDeniedError):
            pi = Policy(player=player)
        return AI(pi)