"""Simple tests for the value net"""
from functools import reduce

import numpy as np
import pytest
import torch

from connectfour.embedding_net import EmbeddingNet
from connectfour.game import BatchGameState
from connectfour.play_state import PlayState
from connectfour.value_net import ValueNet


@pytest.mark.parametrize("rows", [3, 4, 7])
@pytest.mark.parametrize("cols", [6, 7, 9])
@pytest.mark.parametrize("batch_size", [3, 5, 7])
def test_value_shape(rows, cols, batch_size):
    bgs = BatchGameState(
        batch_size=batch_size, turn=PlayState.X, num_cols=cols, num_rows=rows
    )
    plays = np.random.randint(low=0, high=cols, size=(7, batch_size)).tolist()
    bgs = reduce(BatchGameState.play_at, plays, bgs)

    board = bgs.as_array()

    embedding = EmbeddingNet(depth=2, out_channels=4, kernel_size=3)
    policy_net = ValueNet(embedding=embedding, n_rows=rows, n_cols=cols, kernel_size=3)

    logits = policy_net(board.to(dtype=torch.float))

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (batch_size,)
