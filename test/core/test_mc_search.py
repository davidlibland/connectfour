"""Tests the mc-search algorithms"""
from functools import reduce

import numpy as np
import pytest
import torch

from connectfour.embedding_net import EmbeddingNet
from connectfour.game import MutableBatchGameState
from connectfour.play_state import PlayState, play_state_embedding_ix
from connectfour.value_net import ValueNet
from connectfour.policy import PolicyNet
from connectfour.mc_tree_search import fantasy_play, mc_tree_search


@pytest.mark.parametrize("rows", [5, 4, 7])
@pytest.mark.parametrize("cols", [6, 7, 9])
@pytest.mark.parametrize("batch_size", [3, 5, 7])
@pytest.mark.parametrize("depth", [3])
@pytest.mark.parametrize("run_length", [4])
def test_fantasy_play(rows, cols, batch_size, depth, run_length):
    bgs = MutableBatchGameState(
        batch_size=batch_size, turn=PlayState.X, num_cols=cols, num_rows=rows
    )
    plays = np.random.randint(low=0, high=cols, size=(7, batch_size)).tolist()
    bgs = reduce(MutableBatchGameState.play_at, plays, bgs)

    embedding = EmbeddingNet(depth=2, out_channels=4, kernel_size=3)
    policy_net = PolicyNet(embedding=embedding, kernel_size=3)
    value_net = ValueNet(embedding=embedding, n_rows=rows, n_cols=cols, kernel_size=3)

    result = fantasy_play(
        bgs,
        policy_net=policy_net,
        value_net=value_net,
        depth=depth,
        run_length=run_length,
        discount=0.9,
    )
    assert "value" in result
    assert "move" in result
    assert result["move"].shape == (batch_size,)
    assert result["value"].shape == (batch_size,)
    assert (result["value"] <= 1).all()
    assert (result["value"] >= -1).all()


@pytest.mark.parametrize("rows", [5, 4, 7])
@pytest.mark.parametrize("cols", [6, 7, 9])
@pytest.mark.parametrize("batch_size", [3, 5, 7])
@pytest.mark.parametrize("depth", [3])
@pytest.mark.parametrize("breadth", [3])
@pytest.mark.parametrize("run_length", [4])
def test_mc_search(rows, cols, batch_size, depth, breadth, run_length):
    bgs = MutableBatchGameState(
        batch_size=batch_size, turn=PlayState.X, num_cols=cols, num_rows=rows
    )
    plays = np.random.randint(low=0, high=cols, size=(7, batch_size)).tolist()
    bgs = reduce(MutableBatchGameState.play_at, plays, bgs)

    embedding = EmbeddingNet(depth=2, out_channels=4, kernel_size=3)
    policy_net = PolicyNet(embedding=embedding, kernel_size=3)
    value_net = ValueNet(embedding=embedding, n_rows=rows, n_cols=cols, kernel_size=3)
    value_net.fully_connected_value.bias.data = torch.randn_like(
        value_net.fully_connected_value.bias
    )

    result = mc_tree_search(
        bgs,
        policy_net=policy_net,
        value_net=value_net,
        depth=depth,
        breadth=breadth,
        run_length=run_length,
        discount=0.9,
    )
    assert "value" in result
    assert "move" in result
    assert "counts" in result
    assert result["move"].shape == (batch_size,)
    assert result["value"].shape == (batch_size,)
    assert (result["value"] <= 1).all()
    assert (result["value"] >= -1).all()
    assert not (result["value"] == 0).all()
    assert result["counts"].shape == (batch_size, cols)
    assert (result["counts"].sum(dim=1) >= 0).all()
