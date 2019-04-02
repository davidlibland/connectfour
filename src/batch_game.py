from functools import reduce
from typing import Optional, List

import numpy as np
import tensorflow as tf

from src.abstract import AbsBatchGameState, ABSGame
from src.game import Player, player_embedding, GameState, player_embedding_ix, player_extraction


class BatchGameState(AbsBatchGameState):
    @classmethod
    def from_game_state(cls, gs: GameState) -> "BatchGameState":
        return BatchGameState(gs.as_array()[np.newaxis, :, :, :], turn=gs.turn)

    def __init__(self, state=None, turn=Player.X, num_rows=5, num_cols=5, batch_size=32):
        if state is not None:
            self._batch_size, self._num_rows, self._num_cols, _ = state.shape
        else:
            self._batch_size = batch_size
            self._num_cols, self._num_rows = num_cols, num_rows
            state = self._blank_boards()
        self._board_state = state
        self._turn = turn

    def _blank_board(self):
        return np.array([
            [player_embedding(Player.BLANK)]*self._num_cols
        ]*self._num_rows)

    def _blank_boards(self):
        blank_board = self._blank_board()
        return np.array([blank_board]*self.batch_size)

    def winners(self, run_length=4) -> List[Optional[Player]]:
        results = np.array([None]*self.batch_size)
        # find draws:
        num_blank_spaces = tf.reduce_sum(self._board_state[:, :, :, 0], axis=[1, 2])
        draws = tf.equal(num_blank_spaces, 0)
        results[np.array(draws)] = Player.BLANK
        # find wins:
        win_types = []
        for filter in self._get_winning_filters(run_length):
            w = tf.nn.conv2d(self._board_state[:, :, :, :], filter,
                         strides=[1, 1, 1, 1], padding="VALID")
            win_types.append(tf.reduce_max(w, axis=[1,2]))
        wins = reduce(tf.maximum, win_types) >= run_length
        for p in [Player.X, Player.O]:
            results[np.array(wins[:, player_embedding_ix(p)])] = p
        return results

    def next_actions(self) -> np.array:
        # Find the columns with at least one blank entry
        blank_ix = player_embedding_ix(Player.BLANK)
        blank_space_indicator = self._board_state[:, :, :, blank_ix]
        num_blank_spaces = tf.reduce_sum(blank_space_indicator, axis=1)
        actions = tf.not_equal(num_blank_spaces, 0)
        return actions

    @property
    def turn(self) -> Player:
        return self._turn

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def _next_turn(self):
        if self._turn == Player.X:
            return Player.O
        return Player.X

    def play_at(self, js: List[int], reset_games: List[bool]=None
                ) -> "BatchGameState":
        new_state = self._board_state.copy()
        if reset_games is None:
            reset_games = [False]*self.batch_size
        for n, (j, reset) in enumerate(zip(js, reset_games)):
            if reset:
                new_state[n, :, :, :] = self._blank_board()
                continue
            assert 0 != tf.reduce_sum(self._board_state[n, :, j, player_embedding_ix(Player.BLANK)]), \
                "Must play valid move! Column %d is full!" \
                % j
            i = 1
            for i in range(1, self._num_rows+1):
                if self._board_state[n, -i, j, player_embedding_ix(Player.BLANK)] == 1:
                    break
            new_state[n, -i, j, :] = player_embedding(self._turn)
        return BatchGameState(new_state, self._next_turn)

    def __hash__(self):
        return hash(self.as_tuple())

    def as_tuple(self):
        to_val = lambda x: x.value
        return tuple([tuple([
            tuple([
                to_val(player_extraction(v)) for v in row
            ])
            for row in game
        ]) for game in self._board_state.tolist()])

    def __str__(self):
        game_strs = []
        for game in self.as_tuple():
            hor_line = "\n%s\n" % ("-"*(self._num_cols*2-1))
            game_strs.append(hor_line.join(map(lambda row: "|".join(row), game)))
        hor_line = "\n\n%s\n\n" % ("*"*(self._num_cols*2+1))
        return hor_line.join(game_strs)

    def __repr__(self):
        return str(self)

    def as_array(self):
        return self._board_state

    def _get_winning_filters(self, run_length: int=4):
        # get horizontal filter
        horiz = tf.einsum("ij,kl->ijkl", tf.ones([1, run_length]), tf.eye(3))
        # get vertical filter
        vert = tf.einsum("ij,kl->ijkl", tf.ones([run_length, 1]), tf.eye(3))
        # get diagonal filter
        diag = tf.einsum("ij,kl->ijkl", tf.eye(run_length), tf.eye(3))
        # get anti-diagonal filter
        anti_diag = tf.einsum("ij,kl->ijkl", tf.eye(run_length)[:,::-1], tf.eye(3))
        return [horiz, vert, diag, anti_diag]


BatchGame = ABSGame.factory(BatchGameState)
