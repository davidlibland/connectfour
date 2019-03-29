from enum import Enum
from itertools import product
from typing import Optional, Iterator

import numpy as np

from src.abstract import AbsGameState, ABSGame


class Player(Enum):
    X = "X"
    O = "O"
    BLANK = " "


def player_embedding(p: Player):
    if p == Player.BLANK:
        return np.array([1,0,0], dtype=np.float32)
    elif p == Player.X:
        return np.array([0,1,0], dtype=np.float32)
    elif p == Player.O:
        return np.array([0,0,1], dtype=np.float32)


class GameState(AbsGameState):
    def __init__(self, state=None, turn=Player.X, num_rows=5, num_cols=5):
        if state is not None:
            self._num_rows, self._num_cols = state.shape
        else:
            self._num_cols, self._num_rows = num_cols, num_rows
            state = self._blank_board()
        self._board_state = state
        self._turn = turn

    def _blank_board(self):
        return np.array([
            [Player.BLANK]*self._num_cols
        ]*self._num_rows)

    def winner(self, run_length=4) -> Optional[Player]:
        if Player.BLANK not in set(self._board_state.flatten()):
            return Player.BLANK
        for i, j in product(range(self._num_rows), range(self._num_cols)):
            if self._board_state[i, j] != Player.BLANK:
                cur_player = self._board_state[i, j]
                streak = tuple(np.array([cur_player]*run_length))
                # check downwards:
                if i <= self._num_rows-run_length:
                    chk = tuple(self._board_state[i:i+run_length, j])
                    if chk == streak:
                        return cur_player
                # check rightwards:
                if j <= self._num_cols-run_length:
                    chk = tuple(self._board_state[i, j:j+run_length])
                    if chk == streak:
                        return cur_player
                # check downright:
                if j <= self._num_cols-run_length and i <= self._num_rows-run_length:
                    diag = tuple(self._board_state[i:i + run_length, j:j + run_length].diagonal())
                    if diag == streak:
                        return cur_player
                # check downleft:
                if j-run_length+1 >= 0 and i <= self._num_rows-run_length:
                    diag = tuple(np.fliplr(self._board_state[i:i + run_length, j-run_length+1:j+1]).diagonal())
                    if diag == streak:
                        return cur_player
        # It's undecided
        return None

    def next_actions(self) -> Iterator[int]:
        for j in range(self._num_cols):
            if Player.BLANK in set(self._board_state[:,j]):
                yield j

    @property
    def turn(self) -> Player:
        return self._turn

    @property
    def _next_turn(self):
        if self._turn == Player.X:
            return Player.O
        return Player.X

    def play_at(self, j) -> "GameState":
        assert Player.BLANK in set(self._board_state[:, j]), \
            "Must play valid move! Column %d is full!" \
            % j
        new_state = self._board_state.copy()
        i = 1
        for i in range(1, self._num_rows+1):
            if self._board_state[-i, j] == Player.BLANK:
                break
        new_state[-i, j] = self._turn
        return GameState(new_state, self._next_turn)

    def __hash__(self):
        return hash(tuple(map(tuple, self._board_state.tolist())))

    def as_tuple(self):
        to_val = lambda x: x.value
        return tuple([
            tuple([
                to_val(v) for v in row
            ])
            for row in self._board_state.tolist()
        ])

    def __str__(self):
        hor_line = "\n%s\n" % ("-"*(self._num_cols*2-1))
        return hor_line.join(map(lambda row: "|".join(row),self.as_tuple()))

    def __repr__(self):
        return str(self)

    def as_array(self):
        return np.array([
            [player_embedding(p) for p in row]
            for row in self._board_state
        ])


Game = ABSGame.factory(GameState)
