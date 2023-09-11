from functools import reduce
from typing import Optional, List
import random

import numpy as np
import torch
import torch.nn as nn

from connectfour.abstract import AbsBatchGameState, ABSGame
from connectfour.play_state import (
    PlayState,
    play_state_embedding_ix,
    play_state_embedding,
    play_state_extraction,
)

MIN_WIDTH, MAX_WIDTH = 5, 10
MIN_HEIGHT, MAX_HEIGHT = 5, 10


class BatchGameState(AbsBatchGameState):
    """The connectfour game."""
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
        return torch.tile(
            torch.tensor(play_state_embedding(PlayState.BLANK))[:, None, None],
            (1, self._num_rows, self._num_cols)
        )

    def _blank_boards(self):
        blank_board = self._blank_board()
        return torch.tile(blank_board[None, :, :, :], (self.batch_size, 1, 1, 1))

    def winners(self, run_length=4) -> List[Optional[PlayState]]:
        results = np.array([None]*self.batch_size)
        # find draws:
        num_blank_spaces = torch.sum(self._board_state[:, 0, :, :], dim=[1, 2])
        draws = num_blank_spaces == 0
        results[np.array(draws)] = PlayState.DRAW
        # find wins:
        win_types = []
        for filter in self._get_winning_filters(run_length):
            w = nn.functional.conv2d(self._board_state.to(dtype=filter.dtype), filter,
                         stride=1, padding="valid")
            win_types.append(torch.amax(w, dim=(2, 3)))
        wins = reduce(torch.maximum, win_types) >= run_length
        for p in [PlayState.X, PlayState.O]:
            results[np.array(wins[:, play_state_embedding_ix(p)])] = p
        return results

    def next_actions(self) -> np.array:
        # Find the columns with at least one blank entry
        blank_ix = play_state_embedding_ix(PlayState.BLANK)
        blank_space_indicator = self._board_state[:, blank_ix, :, :]
        num_blank_spaces = torch.sum(blank_space_indicator, dim=1)
        actions = num_blank_spaces != 0
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
        new_state = torch.clone(self._board_state)
        # Determine the number of previous plays in each column by summing the one hot mask:
        num_plays = torch.einsum("ijki->i", self._board_state[:, play_state_embedding_ix(PlayState.BLANK)+1:, :, js])
        is_ = self._num_rows - 1 - num_plays
        # Set the one-hot-values at those locations
        for ix, v in enumerate(play_state_embedding(self._turn)):
            new_state[torch.arange(self._batch_size), ix, is_, js] = v

        # reset any other games.
        if reset_games is not None:
            new_state[reset_games, :, :, :] = torch.tile(
                self._blank_board(), (np.sum(reset_games), 1, 1, 1)
            )
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
        ]) for game in torch.permute(self._board_state, (0, 2, 3, 1)).tolist()])

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
        horiz = torch.einsum("ij,kl->ijkl", torch.eye(3), torch.ones([1, run_length]))
        # get vertical filter
        vert = torch.einsum("ij,kl->ijkl", torch.eye(3), torch.ones([run_length, 1]))
        # get diagonal filter
        diag = torch.einsum("ij,kl->ijkl", torch.eye(3), torch.eye(run_length))
        # get anti-diagonal filter
        anti_diag_ = torch.flip(torch.eye(run_length), (1,))
        anti_diag = torch.einsum("ij,kl->ijkl", torch.eye(3), anti_diag_)
        return [horiz, vert, diag, anti_diag]


BatchGame = ABSGame.factory(BatchGameState)
