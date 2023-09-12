"""Simple tests for the policy net"""
from functools import reduce

import pytest
import torch

from connectfour.game import BatchGameState
from connectfour.play_state import PlayState
from connectfour.policy import PolicyNet


@pytest.mark.parametrize("rows", [3, 4, 7])
@pytest.mark.parametrize("cols", [6, 7, 9])
def test_policy_shape(rows, cols):
    bgs = BatchGameState(
        batch_size=3, turn=PlayState.X, num_cols=cols, num_rows=rows
    )
    plays1 = [4, 4, 3, 3, 2]
    plays2 = [4, 4, 3, 3, 1]
    plays3 = [1, 5, 1, 0, 1]
    bgs = reduce(BatchGameState.play_at, zip(plays1, plays2, plays3), bgs)

    board = bgs.as_array()

    policy_net = PolicyNet(rows=rows, cols=cols)

    logits = policy_net(board.to(dtype=torch.float))

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (3, cols)
