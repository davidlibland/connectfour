from enum import Enum
from itertools import product
from typing import Optional, Iterator, List

import numpy as np

from src.abstract import AbsBatchGameState, ABSGame
from src.game import Player, player_embedding, GameState


class BatchGameState(AbsBatchGameState):
    @classmethod
    def from_game_state(cls, gs: GameState) -> "BatchGameState":
        return BatchGameState(gs._board_state[np.newaxis, :, :], turn=gs.turn)

    def __init__(self, state=None, turn=Player.X, num_rows=5, num_cols=5, batch_size=32):
        if state is not None:
            self._batch_size, self._num_rows, self._num_cols = state.shape
        else:
            self._batch_size = batch_size
            self._num_cols, self._num_rows = num_cols, num_rows
            state = self._blank_boards()
        self._board_state = state
        self._turn = turn

    def _blank_board(self):
        return np.array([
            [Player.BLANK]*self._num_cols
        ]*self._num_rows)

    def _blank_boards(self):
        blank_board = self._blank_board()
        return np.array([blank_board]*self.batch_size)

    def winners(self, run_length=4) -> List[Optional[Player]]:
        results = []
        for n in range(self.batch_size):
            if Player.BLANK not in set(self._board_state[n,:,:].flatten()):
                results.append(Player.BLANK)
            else:
                for i, j in product(range(self._num_rows), range(self._num_cols)):
                    if self._board_state[n, i, j] != Player.BLANK:
                        cur_player = self._board_state[n, i, j]
                        streak = tuple(np.array([cur_player]*run_length))
                        # check downwards:
                        if i <= self._num_rows-run_length:
                            chk = tuple(self._board_state[n, i:i+run_length, j])
                            if chk == streak:
                                results.append(cur_player)
                                break
                        # check rightwards:
                        if j <= self._num_cols-run_length:
                            chk = tuple(self._board_state[n, i, j:j+run_length])
                            if chk == streak:
                                results.append(cur_player)
                                break
                        # check downright:
                        if j <= self._num_cols-run_length and i <= self._num_rows-run_length:
                            diag = tuple(self._board_state[n, i:i + run_length, j:j + run_length].diagonal())
                            if diag == streak:
                                results.append(cur_player)
                                break
                        # check downleft:
                        if j-run_length+1 >= 0 and i <= self._num_rows-run_length:
                            diag = tuple(np.fliplr(self._board_state[n, i:i + run_length, j-run_length+1:j+1]).diagonal())
                            if diag == streak:
                                results.append(cur_player)
                                break
            if len(results) < n+1:
                # It's undecided
                results.append(None)
        return results

    def next_actions(self) -> np.array:
        actions = np.ones([self.batch_size, self._num_cols])
        for i in range(self.batch_size):
            for j in range(self._num_cols):
                if Player.BLANK in set(self._board_state[i, :, j]):
                    actions[i,j] = 0
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
                new_state[n] = self._blank_board()
                continue
            if Player.BLANK not in set(self._board_state[n,:,j]):
                continue
            assert Player.BLANK in set(self._board_state[n, :, j]), \
                "Must play valid move! Column %d is full!" \
                % j
            i = 1
            for i in range(1, self._num_rows+1):
                if self._board_state[n, -i, j] == Player.BLANK:
                    break
            new_state[n, -i, j] = self._turn
        return BatchGameState(new_state, self._next_turn)

    def __hash__(self):
        return hash(self.as_tuple())

    def as_tuple(self):
        to_val = lambda x: x.value
        return tuple([tuple([
            tuple([
                to_val(v) for v in row
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
        return np.array([[
            [player_embedding(p) for p in row]
            for row in game
        ] for game in self._board_state])


BatchGame = ABSGame.factory(BatchGameState)
