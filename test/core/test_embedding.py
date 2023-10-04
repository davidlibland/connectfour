"""Tests the embedding"""
from functools import reduce

import numpy as np
import pytest
import torch

from connectfour.embedding_net import EmbeddingNet
from connectfour.game import BatchGameState
from connectfour.play_state import PlayState
from connectfour.value_net import ValueNet
from connectfour.embedding_net import EmbeddingNet


@pytest.mark.parametrize("rows", [3, 4, 7])
@pytest.mark.parametrize("cols", [6, 7, 9])
@pytest.mark.parametrize("batch_size", [3, 5, 7])
@pytest.mark.parametrize("out_channels", [1, 2, 5, 7])
@pytest.mark.parametrize("depth", [0, 1, 3])
def test_embedding_shape(rows, cols, batch_size, out_channels, depth):
    bgs = BatchGameState(
        batch_size=batch_size, turn=PlayState.X, num_cols=cols, num_rows=rows
    )
    plays = np.random.randint(low=0, high=cols, size=(7, batch_size)).tolist()
    bgs = reduce(BatchGameState.play_at, plays, bgs)

    board = bgs.as_array()

    embedding = EmbeddingNet(depth=depth, out_channels=out_channels, kernel_size=3)

    outs = embedding(board.to(dtype=torch.float))

    assert isinstance(outs, torch.Tensor)
    assert outs.shape == (batch_size, out_channels, rows, cols)


@pytest.mark.parametrize("rows", [3, 4, 7])
@pytest.mark.parametrize("cols", [6, 7, 9])
@pytest.mark.parametrize("batch_size", [3, 5, 7])
@pytest.mark.parametrize("out_channels", [1, 2, 5, 7])
@pytest.mark.parametrize("depth", [0, 1, 3])
def test_embedding_initialization(rows, cols, batch_size, out_channels, depth):
    bgs = BatchGameState(
        batch_size=batch_size, turn=PlayState.X, num_cols=cols, num_rows=rows
    )

    board = bgs.as_array()
    embedding = EmbeddingNet(depth=depth, out_channels=out_channels, kernel_size=3)

    i_outs = embedding.fan_out(board.to(dtype=torch.float))
    outs = embedding(board.to(dtype=torch.float))

    assert isinstance(outs, torch.Tensor)
    assert outs.shape == (batch_size, out_channels, rows, cols)
    assert abs(outs - i_outs).max().detach().cpu().numpy() < 0.1
