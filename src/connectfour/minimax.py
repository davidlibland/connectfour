"""Mininmax search"""
from functools import reduce
from typing import Dict, Literal, Tuple, Any, Optional

import numpy as np
import torch
from torch import nn

from connectfour.game import MutableBatchGameState
from connectfour.play_state import (
    PlayState,
    play_state_embedding_ix,
    play_state_extraction,
    play_state_embedding,
)
from connectfour.utils import get_winning_filters
from connectfour.board_parser import as_string, as_tuple


class HashableBoard:
    def __init__(self, board_state: torch.Tensor):
        self.board_state = board_state
        _, self._num_rows, self._num_cols = board_state.shape

    def __repr__(self):
        return str(self)

    def __str__(self):
        return as_string(self.board_state)

    def as_tuple(self):
        return as_tuple(self.board_state)

    @staticmethod
    def from_int(n, num_rows, num_cols):
        flat_state = torch.zeros((num_rows * num_cols), dtype=torch.long)
        for i in reversed(range(num_rows * num_cols)):
            j = n % 3
            flat_state[i] = j
            n //= 3
        state = torch.nn.functional.one_hot(
            flat_state.reshape([num_rows, num_cols]), 3
        ).permute(2, 0, 1)
        return HashableBoard(state)

    def to_int(self) -> int:
        n = 0
        for i, j in enumerate(
            reversed(
                torch.argmax(self.board_state.detach(), dim=0)
                .flatten()
                .cpu()
                .numpy()
                .astype(int)
            )
        ):
            n += int(j) * (3**i)
        return n

    def __hash__(self):
        return self.to_int()

    def __eq__(self, other):
        if not isinstance(other, HashableBoard):
            return False
        return self.as_tuple() == other.as_tuple()


def swap_players(board_state: torch.Tensor):
    """swaps the players in the boardstate"""
    return board_state[[0, 2, 1], :, :]


def next_actions(board_state: torch.Tensor) -> torch.Tensor:
    # Find the columns with at least one blank entry
    blank_ix = play_state_embedding_ix(PlayState.BLANK)
    blank_space_indicator = board_state[blank_ix, :, :]
    num_blank_spaces = torch.sum(blank_space_indicator, dim=0)
    actions = num_blank_spaces != 0
    return actions


def play_at(board_state: torch.Tensor, j: int) -> torch.Tensor:
    # Determine the number of previous plays in each column by summing the one hot mask:
    num_plays = torch.sum(
        board_state[play_state_embedding_ix(PlayState.BLANK) + 1 :, :, j]
    ).to(torch.int)
    _, num_rows, num_cols = board_state.shape
    is_ = num_rows - 1 - num_plays
    # Set the one-hot-values at those locations
    for ix, v in enumerate(play_state_embedding(PlayState.X)):
        board_state[ix, is_, j] = v
    return board_state


def winners(board_state: torch.Tensor, run_length=4) -> Optional[PlayState]:
    # find draws:
    num_blank_spaces = torch.sum(board_state[0, :, :])
    if num_blank_spaces == 0:
        return PlayState.DRAW
    # find wins:
    win_types = []
    for filter in get_winning_filters(run_length):
        w = nn.functional.conv2d(
            board_state[None, ...].to(filter),
            filter,
            stride=1,
            padding="valid",
        )
        win_types.append(torch.amax(w, dim=(2, 3)))
    wins = reduce(torch.maximum, win_types) >= run_length
    for p in [PlayState.X, PlayState.O]:
        if wins[0, play_state_embedding_ix(p)]:
            return p
    return None


class MiniMaxPolicyCorrector:
    """We always assume we are playing for X."""

    def __init__(self, num_rows, num_cols, depth=4, run_length=4):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.depth = depth
        self.run_length = run_length
        self.cache = {}

    def __call__(self, x: torch.Tensor, core_policy: nn.Module) -> torch.Tensor:
        """Produces (extreme) logits from minimax search"""
        n = x.shape[0]
        logits = []
        if x.device != next(core_policy.parameters()).device:
            core_policy = core_policy.to(x.device)
        for ix in range(n):
            # Pull the individual state:
            x_ = x[ix, ...]
            reward, move = minimax(
                x_, self.cache, depth=self.depth, run_length=self.run_length
            )
            if move is None:
                logit = core_policy(x_[None, ...])
            else:
                logit = torch.zeros(
                    [1, self.num_cols], device=x.device, dtype=torch.float
                )
                logit[0, move] = 100
            logits.append(logit)
        return torch.cat(logits, dim=0)


def minimax(
    state: torch.Tensor,
    cache: Dict[int, Tuple[float, Optional[int]]],
    depth: int,
    run_length: int,
) -> Tuple[float, Optional[int]]:
    """
    We compute the minimax reward, assuming we are playing as X
    """
    if HashableBoard(state) in cache:
        return cache[hash(HashableBoard(state))]
    if depth == 0:
        return 0, None
    _, num_rows, num_cols = state.shape
    action_rewards = {}
    action_mask = next_actions(state).flatten().cpu().numpy()
    for a in np.arange(num_cols)[action_mask]:
        next_state = play_at(torch.clone(state), a)
        win = winners(next_state, run_length=run_length)
        if win == PlayState.X:
            reward = 1
            confidence = 1
            cache[hash(HashableBoard(next_state))] = (reward, None)
        elif win == PlayState.O:
            reward = -1
            confidence = 1
            cache[hash(HashableBoard(next_state))] = (reward, None)
        elif win == PlayState.DRAW:
            reward = 0
            confidence = 1
            cache[hash(HashableBoard(next_state))] = (reward, None)
        else:
            # Negative reward of the O player
            neg_reward, sub_move = minimax(
                swap_players(next_state), cache, depth=depth - 1, run_length=run_length
            )
            confidence = int(sub_move is not None)
            reward = -neg_reward
        action_rewards[a] = (reward, confidence)
    player_reward, confidence, player_action = max(
        (r, c, a) for a, (r, c) in action_rewards.items()
    )
    if confidence > 0:
        cache[hash(HashableBoard(state))] = (player_reward, player_action)
    # We have some confidence, if we're confident about any move.
    any_confidence = max(c for a, (r, c) in action_rewards.items())
    return (player_reward, player_action if any_confidence > 0 else None)


def fill_cache(depth, num_rows, num_cols, run_length=4) -> Dict[int, Tuple[float, int]]:
    cache = dict()
    state = MutableBatchGameState(
        num_rows=num_rows, num_cols=num_cols, batch_size=1, turn=PlayState.X
    ).as_array()[0, ...]
    minimax(state=state, cache=cache, depth=depth, run_length=run_length)
    return cache
