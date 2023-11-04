"""Tests for the minimax code"""
from types import NoneType

from hypothesis import given
import hypothesis.strategies as h_strats
import pytest

from connectfour.minimax import *
from connectfour.board_parser import as_string, from_string, as_tuple


@pytest.mark.parametrize("num_rows", [5])
@pytest.mark.parametrize("num_cols", [5])
@pytest.mark.parametrize("run_length", [4])
def test_caching(num_rows, num_cols, run_length):
    cache = fill_cache(4, num_rows, num_cols, run_length=run_length)
    for state, val in cache.items():
        assert isinstance(state, HashableBoard)
        assert isinstance(val, tuple)
        assert len(val) == 2


@pytest.mark.parametrize("num_rows", [3, 4, 5])
@pytest.mark.parametrize("num_cols", [4, 3])
@pytest.mark.parametrize("run_length", [2])
def test_parser(num_rows, num_cols, run_length):
    cache = fill_cache(4, num_rows, num_cols, run_length=run_length)
    for hstate in cache.keys():
        state = HashableBoard.from_int(hstate, num_rows, num_cols).board_state
        s = as_string(state)
        state_ = from_string(s)
        assert (state_ == state).all(), (
            as_string(state) + " \n!= \n" + as_string(state_)
        )


def test_minimax():
    cache = {}
    board = """ | | | | | | 
-------------
 | | | | | | 
-------------
 | | | | | | 
-------------
 | | |O| | | 
-------------
 | |X|O| | | 
-------------
 |X|X|O|X| | """
    state = from_string(board)
    reward, move = minimax(state, cache, depth=2, run_length=4)
    assert move == 3
    assert reward == 0

    board = """ | | | | | | 
-------------
 | | | | | | 
-------------
 | | | | | | 
-------------
 | | |O| | | 
-------------
 | |X|O| | | 
-------------
 |X|X|O|X| | """
    state = swap_players(from_string(board))
    reward, move = minimax(state, cache, depth=2, run_length=4)
    assert move == 3
    assert reward == 1

    board = """ | | | | | | 
-------------
 | | | | | | 
-------------
 | | |X| | | 
-------------
 | |O|X| | | 
-------------
 | |O|O| | | 
-------------
 | |O|X|X|X|O"""
    state = from_string(board)
    reward, move = minimax(state, cache, depth=2, run_length=4)
    assert move == 2
    assert reward == 0


@given(
    h_strats.tuples(
        h_strats.integers(min_value=1, max_value=6),
        h_strats.integers(min_value=1, max_value=6),
    ).flatmap(
        lambda t: h_strats.tuples(
            h_strats.just(t[0]),
            h_strats.just(t[1]),
            h_strats.lists(
                h_strats.integers(min_value=0, max_value=2),
                min_size=t[0] * t[1],
                max_size=t[0] * t[1],
            ),
        )
    )
)
def test_hash(data):
    num_rows, num_cols, vals = data
    flat_state = torch.zeros((3, num_rows * num_cols))
    for i, j in enumerate(vals):
        flat_state[j, i] = 1
    state = flat_state.reshape([3, num_rows, num_cols])

    n = hash(HashableBoard(state))
    state_ = HashableBoard.from_int(n, num_rows, num_cols).board_state
    assert state_.shape == state.shape
    assert (state == state_).all()
