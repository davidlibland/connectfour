"""Simple tests for the policy net"""
from functools import reduce

import numpy as np
import pytest
import torch

from connectfour.game import MutableBatchGameState
from connectfour.play_state import PlayState
from connectfour.policy import PolicyNet
from connectfour.embedding_net import EmbeddingNet


@pytest.mark.parametrize("rows", [3, 4, 7])
@pytest.mark.parametrize("cols", [6, 7, 9])
@pytest.mark.parametrize("batch_size", [3, 5, 7])
def test_policy_shape(rows, cols, batch_size):
    bgs = MutableBatchGameState(
        batch_size=batch_size, turn=PlayState.X, num_cols=cols, num_rows=rows
    )
    plays = np.random.randint(low=0, high=cols, size=(7, batch_size)).tolist()
    bgs = reduce(MutableBatchGameState.play_at, plays, bgs)

    board = bgs.as_array()

    embedding = EmbeddingNet(depth=2, out_channels=4, kernel_size=3)
    policy_net = PolicyNet(embedding=embedding, kernel_size=3)

    logits = policy_net(board.to(dtype=torch.float))

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (batch_size, cols)


@pytest.mark.parametrize("rows", [3, 4, 7])
@pytest.mark.parametrize("cols", [6, 7, 9])
@pytest.mark.parametrize("batch_size", [3, 5, 7])
def test_policy_initialization(rows, cols, batch_size):
    bgs = MutableBatchGameState(
        batch_size=batch_size, turn=PlayState.X, num_cols=cols, num_rows=rows
    )

    board = bgs.as_array()

    embedding = EmbeddingNet(depth=2, out_channels=4, kernel_size=3)
    policy_net = PolicyNet(embedding=embedding, kernel_size=3)

    logits = policy_net(board.to(dtype=torch.float))

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (batch_size, cols)
    assert (logits.max() - logits.min()).detach().cpu().numpy() < 1
