import tensorflow as tf
from tensorflow.python.framework.errors_impl import PermissionDeniedError

from src.file_utils import get_dir, build_checkpoint_file_name
from src.game import BatchGameState, BatchGame
from src.play_state import PlayState, opponent
import os

from src import nn

tf.enable_eager_execution()


class Policy:
    def __init__(
        self,
        player: PlayState = PlayState.X,
        lr=1e-3,
        root_dir=None,
        descriptor="C4",
    ):
        self._player = player
        self._container = tf.contrib.eager.EagerVariableStore()
        self._optimizer = tf.train.AdamOptimizer(lr)
        num_policy_layers = 5
        num_policy_filters = 64
        num_reward_layers = 2
        num_reward_filters = 64
        with self._container.as_default():
            # Policy network params
            self._p_conv_params = []
            for i in range(num_policy_layers):
                num_in_filters = 3 if i == 0 else num_policy_filters
                w = tf.get_variable(
                    "p_conv_w_%s_%i" % (self.player, i),
                    shape=[3, 3, num_in_filters, num_policy_filters],
                    dtype=tf.float32,
                    initializer=tf.random_normal_initializer(0, 0.05),
                )
                b = tf.get_variable(
                    "p_conv_b_%s_%i" % (self.player, i),
                    shape=[num_policy_filters],
                    dtype=tf.float32,
                    initializer=tf.constant_initializer(0.0),
                )
                s = tf.get_variable(
                    "p_skip_%s_%i" % (self.player, i),
                    shape=[1, 1, 3, num_policy_filters],
                    dtype=tf.float32,
                    initializer=tf.random_normal_initializer(0, 0.05),
                )
                self._p_conv_params.append((w, b, s))
            self._p_conv_1_w = tf.get_variable(
                "p_conv_1_w_params_%s" % self.player,
                shape=[3, num_policy_filters, 1],
                dtype=tf.float32,
                initializer=tf.random_normal_initializer(0, 0.05),
            )
            self._p_conv_1_b = tf.get_variable(
                "p_conv_1_b_params_%s" % self.player,
                shape=[1],
                dtype=tf.float32,
                initializer=tf.constant_initializer(0.0),
            )

            # reward network params
            self._r_conv_params = []
            for i in range(num_reward_layers):
                num_in_filters = 3 if i == 0 else num_reward_filters
                w = tf.get_variable(
                    "r_conv_w_%s_%i" % (self.player, i),
                    shape=[3, 3, num_in_filters, num_reward_filters],
                    dtype=tf.float32,
                    initializer=tf.random_normal_initializer(0, 0.05),
                )
                b = tf.get_variable(
                    "r_conv_b_%s_%i" % (self.player, i),
                    shape=[num_reward_filters],
                    dtype=tf.float32,
                    initializer=tf.constant_initializer(0.0),
                )
                self._r_conv_params.append((w, b))
            self._r_dense_mu_w = tf.get_variable(
                "r_dense_mu_w_%s" % self.player,
                shape=[num_reward_filters],
                dtype=tf.float32,
                initializer=tf.random_normal_initializer(0.0),
            )
            self._r_dense_mu_b = tf.get_variable(
                "r_dense_mu_b_%s" % self.player,
                shape=[1],
                dtype=tf.float32,
                initializer=tf.random_normal_initializer(0.0),
            )
            self._r_dense_sig_w = tf.get_variable(
                "r_log_sig_w_%s" % self.player,
                shape=[num_reward_filters],
                dtype=tf.float32,
                initializer=tf.random_normal_initializer(0, 0.05),
            )
            self._r_dense_sig_b = tf.get_variable(
                "r_log_sig_b_%s" % self.player,
                shape=[1],
                dtype=tf.float32,
                initializer=tf.random_normal_initializer(0, 0.05),
            )
        self._global_step = tf.train.get_or_create_global_step()
        self._saver = tf.train.Saver(
            var_list=self._container.trainable_variables()
            + [self._global_step],
            save_relative_paths=True,
            sharded=False,
        )
        if root_dir is None:
            root_dir = get_dir("Checkpoints")
            print("Saving params in new dir:", root_dir)
        self._root_dir = root_dir
        self._summary_writer = tf.contrib.summary.create_file_writer(
            "{}/logs".format(self._root_dir), flush_millis=10000
        )
        self._summary_writer.set_as_default()
        self._descriptor = descriptor

    def reward_logits(self, gs: BatchGameState):
        assert gs.turn == self._player, "Can't play on this turn"
        gs_array = gs.as_array()
        if self.player == PlayState.O:
            gs_array = tf.gather(gs_array, [0, 2, 1], axis=-1)
        x = tf.constant(gs_array, dtype=tf.float32)
        for w, b in self._r_conv_params:
            x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="VALID") + b
            x = tf.nn.relu(x)
        y = tf.reduce_max(x, axis=[2, 3])
        mu = tf.einsum("nk,k->n", y, self._r_dense_mu_w) + self._r_dense_mu_b
        log_sig = (
            tf.einsum("nk,k->n", y, self._r_dense_sig_w) + self._r_dense_sig_b
        )
        return mu, log_sig

    def ln_pi(self, gs: BatchGameState):
        logits = self.logits(gs)
        return logits - tf.reduce_logsumexp(logits, axis=1)[:, tf.newaxis]

    def logits(self, gs: BatchGameState):
        assert gs.turn == self._player, "Can't play on this turn"
        gs_array = gs.as_array()
        if self.player == PlayState.O:
            gs_array = tf.gather(gs_array, [0, 2, 1], axis=-1)
        x = tf.constant(gs_array, dtype=tf.float32)
        y = x
        for w, b, s in self._p_conv_params:
            y = tf.nn.conv2d(y, w, strides=[1, 1, 1, 1], padding="SAME") + b
            y = tf.nn.relu(y)
            # Skip connections:
            y += tf.nn.conv2d(x, s, strides=[1, 1, 1, 1], padding="SAME")
        y = tf.reduce_max(y, axis=1)
        logits = (
            tf.nn.conv1d(y, self._p_conv_1_w, stride=1, padding="SAME")
            + self._p_conv_1_b
        )
        return tf.reshape(logits, [gs.batch_size, -1])

    def learn_from_games(self, games: BatchGame, alpha=1.0, verbose=True):
        self._global_step.assign_add(1)
        last_state = games.cur_state
        rewards = tf.zeros(last_state.batch_size, dtype=tf.float32)
        num_p_rewards = []
        num_m_rewards = []
        r_loss = []
        p_loss = []
        for i, state in enumerate(games.reversed_states):
            if state.turn == self.player and i > 0:
                # Update the reward function
                with tf.GradientTape() as gr_tape, self._container.as_default():
                    (
                        expected_reward_mu,
                        expected_reward_log_sig,
                    ) = self.reward_logits(state)
                    loss = tf.reduce_mean(
                        nn.gaussian_neg_log_likelihood(
                            mu=expected_reward_mu,
                            log_sig=expected_reward_log_sig,
                            x=rewards,
                        )
                    )
                    r_loss.append(float(loss))
                with tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.scalar(
                        "%s reward loss" % self.player.name, loss
                    )
                rw_gradients = gr_tape.gradient(
                    loss, self._container.trainable_variables()
                )
                self._optimizer.apply_gradients(
                    zip(rw_gradients, self._container.trainable_variables())
                )

                # Update the policy function
                with tf.GradientTape() as gr_tape, self._container.as_default():
                    expected_reward = expected_reward_mu + tf.random.normal(
                        shape=expected_reward_mu.shape
                    ) * tf.exp(expected_reward_log_sig)
                    td = tf.stop_gradient(rewards - expected_reward)
                    un_weighted_ln_pi = tf.reduce_mean(
                        -td[:, tf.newaxis] * self.ln_pi(state)
                    )
                    rw_ln_pi = un_weighted_ln_pi * (alpha**i)
                    p_loss.append(float(un_weighted_ln_pi))

                with tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.scalar(
                        "%s policy loss" % self.player.name, un_weighted_ln_pi
                    )
                gradients = gr_tape.gradient(
                    rw_ln_pi, self._container.trainable_variables()
                )
                self._optimizer.apply_gradients(
                    zip(gradients, self._container.trainable_variables())
                )
            winners = state.winners()
            rewards_list = [
                1.0
                if winner == self.player
                else -1.0
                if winner == self.opponent
                else float(old_reward * alpha)
                if winner is None
                else 0.0
                for old_reward, winner in zip(rewards, winners)
            ]
            rewards = tf.constant(rewards_list, dtype=tf.float32)
            num_p_rewards.append(sum(1 for w in winners if w == self.player))
            num_m_rewards.append(sum(1 for w in winners if w == self.opponent))
        total_p_rewards = sum(num_p_rewards)
        total_m_rewards = sum(num_m_rewards)
        with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar(
                "%s avg_policy_loss" % self.player.name, sum(p_loss) / (i + 1)
            )
            tf.contrib.summary.scalar(
                "%s avg_reward_loss" % self.player.name, sum(r_loss) / (i + 1)
            )
            tf.contrib.summary.scalar(
                "%s positive rewards" % self.player.name, total_p_rewards
            )
            tf.contrib.summary.scalar(
                "%s reward cnt" % self.player.name,
                total_p_rewards + total_m_rewards,
            )
        if verbose and total_p_rewards > 0 or total_m_rewards > 0:
            print(
                "Noticed %d: (+%d, -%d) rewards for %s"
                % (
                    total_p_rewards + total_m_rewards,
                    total_p_rewards,
                    total_m_rewards,
                    self.player,
                )
            )
            print(
                "Reward Loss: %s, Policy Loss: %s, Player: %s"
                % (sum(r_loss) / (i + 1), sum(p_loss) / (i + 1), self.player)
            )

    @property
    def player(self):
        return self._player

    @property
    def opponent(self):
        return opponent(self._player)

    def save(self, descriptor=None):
        if descriptor is None:
            descriptor = self._descriptor
        path = build_checkpoint_file_name(self._root_dir, descriptor)
        fp = self._saver.save(None, path, global_step=self._global_step)
        print("model saved at %s" % fp)

    @classmethod
    def load(cls, root_dir, descriptor="C4", player=PlayState.X):
        ckpt_dir = os.path.join(root_dir, descriptor)
        root_dir = os.path.dirname(ckpt_dir)
        print("ckpt:", ckpt_dir)
        print("root:", root_dir)
        p = Policy(player, root_dir=root_dir, descriptor=descriptor)
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

    def save(self, fp=None):
        self._pi.save(fp)

    @classmethod
    def load(cls, rp, dp, player=PlayState.X):
        try:
            pi = Policy.load(rp, dp, player)
        except (FileNotFoundError, ValueError, PermissionDeniedError) as exc:
            print("Unable to load policy:", exc)
            pi = Policy(player=player, descriptor=dp)
        return AI(pi)
