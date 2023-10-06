from collections import Counter
from functools import reduce

import torch

import connectfour.nn as cf_nn
from connectfour.game import MutableBatchGameState
from connectfour.play_state import PlayState, play_state_embedding_ix


def test_batch_winner():
    bgs = MutableBatchGameState(
        batch_size=3, turn=PlayState.X, num_cols=7, num_rows=7
    )
    plays1 = [4, 4, 3, 3, 2]
    plays2 = [4, 4, 3, 3, 1]
    plays3 = [1, 5, 1, 0, 1]
    bgs = reduce(
        MutableBatchGameState.play_at, zip(plays1, plays2, plays3), bgs
    )
    winners = bgs.winners(3)
    winners_numeric = bgs.winners_numeric(3)
    assert winners[0] == PlayState.X
    assert winners_numeric[0] == play_state_embedding_ix(PlayState.X)
    assert winners[1] is None
    assert winners_numeric[1] == play_state_embedding_ix(None)
    assert winners[2] == PlayState.X
    assert winners_numeric[2] == play_state_embedding_ix(PlayState.X)

    # Now reset a couple of the boards:
    bgs = bgs.play_at([0, 2, 0], [True, False, True])
    torch.testing.assert_close(bgs._board_state[0], bgs._blank_board())
    torch.testing.assert_close(bgs._board_state[2], bgs._blank_board())
    winners = bgs.winners(3)
    winners_numeric = bgs.winners_numeric(3)
    assert winners[0] is None
    assert winners_numeric[0] == play_state_embedding_ix(None)
    assert winners[1] is None
    assert winners_numeric[1] == play_state_embedding_ix(None)
    assert winners[2] is None
    assert winners_numeric[2] == play_state_embedding_ix(None)


def test_batch_next_actions():
    bgs = MutableBatchGameState(
        batch_size=2, turn=PlayState.X, num_cols=7, num_rows=3
    )
    plays1 = [4, 4, 4, 3, 3, 3, 1, 2]
    plays2 = [4, 4, 3, 3, 2, 2, 1, 1]
    bgs = reduce(MutableBatchGameState.play_at, zip(plays1, plays2), bgs)
    print(bgs)
    next_actions = bgs.next_actions()
    assert next_actions[0].tolist() == [
        True,
        True,
        True,
        False,
        False,
        True,
        True,
    ]
    assert next_actions[1].tolist() == [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
