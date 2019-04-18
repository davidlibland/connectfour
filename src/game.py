from functools import reduce
from typing import List
import random

import numpy as np
import torch
import torch.nn.functional as F
from toolz import memoize

from src.abstract import AbsBatchGameState, ABSGame
from src.play_state import (
    PlayState,
    play_state_embedding_ix,
    play_state_embedding,
    play_state_extraction,
)

MIN_WIDTH, MAX_WIDTH = 5, 10
MIN_HEIGHT, MAX_HEIGHT = 5, 10

class BatchGameState(AbsBatchGameState):
    def __init__(self, state=None, turn="Random", num_rows="Random", num_cols="Random", batch_size=32):
        if state is not None:
            self._batch_size, _, self._num_rows, self._num_cols = state.shape
        else:
            if not isinstance(num_rows, int) or num_rows <= 0:
                num_rows = random.randint(MIN_HEIGHT, MAX_HEIGHT)
            if not isinstance(num_cols, int) or num_cols <= 0:
                num_cols = random.randint(MIN_WIDTH, MAX_WIDTH)
            self._batch_size = batch_size
            self._num_cols, self._num_rows = num_cols, num_rows
            state = self._blank_boards()
        self._board_state = state
        if turn != PlayState.X and turn != PlayState.O:
            # Choose a random turn.
            turn = random.choice([PlayState.X, PlayState.O])
        self._turn = turn

    def _blank_board(self):
        return np.broadcast_to(
            play_state_embedding(PlayState.BLANK).reshape([3, 1, 1]),
            (3, self._num_rows, self._num_cols)
        )

    def _blank_boards(self):
        blank_board = self._blank_board()
        return np.array([blank_board]*self.batch_size)

    def winners(self, run_length=4) -> np.ndarray:
        """
        Returns an array of `Optional[PlayState]`
        None = "no winner"
        PlayState.DRAW = "draw"
        PlayState.X = "X won"
        PlayState.O = "O won"
        """
        results = np.array([None]*self.batch_size)
        # find draws:
        blank_spaces = self._board_state \
            [:, play_state_embedding_ix(PlayState.BLANK), :, :]
        num_blank_spaces = np.sum(blank_spaces, axis=(1, 2))
        draws = num_blank_spaces == 0
        results[np.array(draws)] = PlayState.DRAW
        # find wins:
        existent_runs = self.get_max_runs(max_run_length=run_length)
        wins = existent_runs >= run_length
        for p in [PlayState.X, PlayState.O]:
            results[np.array(wins[:, play_state_embedding_ix(p)])] = p
        return results

    def next_actions(self) -> torch.ByteTensor:
        # Find the columns with at least one blank entry
        blank_ix = play_state_embedding_ix(PlayState.BLANK)
        blank_space_indicator = torch.Tensor(self._board_state[:, blank_ix, :, :])
        num_blank_spaces = torch.sum(blank_space_indicator, dim=(1,))
        actions: torch.ByteTensor = num_blank_spaces != 0
        return actions

    @property
    def turn(self) -> PlayState:
        return self._turn

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def _next_turn(self):
        if self._turn == PlayState.X:
            return PlayState.O
        return PlayState.X

    def play_at(self, js: List[int], reset_games: List[bool]=None
                ) -> "BatchGameState":
        new_state = self._board_state.copy()
        if reset_games is None:
            reset_games = [False]*self.batch_size
        for n, (j, reset) in enumerate(zip(js, reset_games)):
            if reset:
                new_state[n, :, :, :] = self._blank_board()
                continue
            assert 0 != np.sum(self._board_state[n, play_state_embedding_ix(PlayState.BLANK), :, j]), \
                "Must play valid move! Column %d is full!" \
                % j
            i = 1
            for i in range(1, self._num_rows+1):
                if self._board_state[n, play_state_embedding_ix(PlayState.BLANK), -i, j] == 1:
                    break
            new_state[n, :, -i, j] = play_state_embedding(self._turn)
        return BatchGameState(new_state, self._next_turn)

    def __hash__(self):
        return hash(self.as_tuple())

    def as_tuple(self):
        to_val = lambda x: x.value
        return tuple([tuple([
            tuple([
                to_val(play_state_extraction(v)) for v in row
            ])
            for row in game
        ]) for game in np.einsum("ijkl->iklj", self._board_state).tolist()])

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

    def get_max_runs(self, max_run_length=4):
        """
        Processes the boards with win-shapes, and returns the max run-length
        for each player.

        :param run_length: No run length below this will be considered.
        :return: An array of shape (n, 3), where n is the batch size,
        and index [i,j] indicates the maximum run length for player j on game i.
        """
        win_types = []
        for filter in get_winning_filters(max_run_length):
            x = torch.Tensor(self._board_state[:, :, :, :])
            w = F.conv2d(x, filter, stride=1, padding=1)
            win_types.append(np.max(w.numpy(), axis=(2, 3)))
        return reduce(np.maximum, win_types)

    def split(self) -> List["BatchGameState"]:
        """Unbatches the game state."""
        states = [self._board_state[i:i+1,...] for i in range(self.batch_size)]
        return[
            BatchGameState(state=state, turn=self.turn)
            for state in states
        ]


# Helpers:

@memoize
def get_winning_filters(run_length: int=4):
    # get horizontal filter
    horiz = np.einsum("ij,kl->klij", np.ones([1, run_length]), np.eye(3))
    # get vertical filter
    vert = np.einsum("ij,kl->klij", np.ones([run_length, 1]), np.eye(3))
    # get diagonal filter
    diag = np.einsum("ij,kl->klij", np.eye(run_length), np.eye(3))
    # get anti-diagonal filter
    anti_diag = np.einsum("ij,kl->klij", np.eye(run_length)[:,::-1], np.eye(3))
    np_weights = [horiz, vert, diag, anti_diag]
    return [torch.Tensor(w) for w in np_weights]


BatchGame = ABSGame.factory(BatchGameState)
