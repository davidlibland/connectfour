import tensorflow as tf
import numpy as np
from tensorflow.python.framework.errors_impl import PermissionDeniedError

from src.file_utils import get_dir, build_checkpoint_file_name
from src.batch_game import BatchGameState, BatchGame, Player
from src.game import GameState
import os
tf.enable_eager_execution()


class Policy:
    def __init__(self, player: Player=Player.X, num_rows=5, num_cols=5, lr=1e-3, root_dir=None):
        self._player = player
        self._container = tf.contrib.eager.EagerVariableStore()
        self._optimizer = tf.train.AdamOptimizer(lr)
        with self._container.as_default():
            self._params = tf.get_variable(
                "params_%s" % player.name,
                shape=[num_rows, num_cols, 3, num_cols], dtype=tf.float32,
                initializer=tf.random_normal_initializer(0, 0.05)
            )
            self._reward_p = tf.get_variable(
                "r_params_%s" % player.name,
                shape=[num_rows, num_cols, 3], dtype=tf.float32,
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
        gs_tens = tf.constant(gs.as_array(), dtype=tf.float32)
        logits = tf.einsum("nijk,ijk->n", gs_tens, self._reward_p)
        return logits

    def ln_pi(self, gs: BatchGameState):
        logits = self.logits(gs)
        return logits - tf.reduce_logsumexp(logits, axis=1)[:,tf.newaxis]

    def logits(self, gs: BatchGameState):
        assert gs.turn == self._player, "Can't play on this turn"
        gs_tens = tf.constant(gs.as_array(), dtype=tf.float32)
        logits = tf.einsum("nijk,ijkl->nl", gs_tens, self._params)
        return logits

    def learn_from_games(self, games: BatchGame, alpha=0.9, verbose=True):
        self._global_step.assign_add(1)
        last_state = games.cur_state
        rewards = tf.constant([
            1. if winner == self.player else 0.
            for winner in last_state.winners()
        ],dtype=tf.float32)
        for i, state in enumerate(games.reversed_states):
            if state.turn == self.player:
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
            rwrds = [
                1. if winner == self.player else float(reward * alpha) if winner == None else 0.
                for reward, winner in zip(rewards, state.winners())
            ]
            rewards = tf.constant(rwrds, dtype=tf.float32)

    @property
    def player(self):
        return self._player

    def save(self, descriptor):
        path = build_checkpoint_file_name(self._root_dir, descriptor)
        fp = self._saver.save(None, path, global_step=self._global_step)
        print("model saved at %s" % fp)

    @classmethod
    def load(cls, ckpt_dir, player=Player.X, num_rows=5, num_cols=5):
        root_dir = os.path.dirname(ckpt_dir)
        p = Policy(player, num_rows, num_cols, root_dir=root_dir)
        p._saver.restore(None, tf.train.latest_checkpoint(ckpt_dir))
        return p

class AI:
    def __init__(self, pi: Policy):
        self._pi = pi

    def next_moves(self, gs: BatchGameState):
        possible_actions = gs.next_actions()
        logits = self._pi.logits(gs)
        logit_range = tf.reduce_max(logits)-tf.reduce_min(logits)
        possible_actions *= -logit_range
        _m_logits = logits+possible_actions
        action_ix = tf.random.multinomial(_m_logits, 1)
        return action_ix

    def next_move(self, gs: GameState):
        possible_actions = list(gs.next_actions())
        bgs = BatchGameState.from_game_state(gs)
        logits = tf.gather(self._pi.logits(bgs)[0, :], possible_actions)
        action_ix = tf.random.multinomial(logits[tf.newaxis,:], 1)
        return possible_actions[action_ix]

    @property
    def player(self):
        return self._pi.player

    def learn_from_games(self, game: BatchGame, alpha=0.5, verbose=True):
        self._pi.learn_from_games(game, alpha, verbose=verbose)

    def save(self, fp):
        self._pi.save(fp)

    @classmethod
    def load(cls, fp, player=Player.X):
        try:
            pi = Policy.load(fp, player)
        except (FileNotFoundError, ValueError, PermissionDeniedError):
            pi = Policy(player=player)
        return AI(pi)