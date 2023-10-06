from collections import Counter
from functools import reduce

import torch

import connectfour.nn as cf_nn
from connectfour.game import BatchGameState
from connectfour.play_state import PlayState


def test_batch_winner():
    bgs = BatchGameState(
        batch_size=3, turn=PlayState.X, num_cols=7, num_rows=7
    )
    plays1 = [4, 4, 3, 3, 2]
    plays2 = [4, 4, 3, 3, 1]
    plays3 = [1, 5, 1, 0, 1]
    bgs = reduce(BatchGameState.play_at, zip(plays1, plays2, plays3), bgs)
    winners = bgs.winners(3)
    assert winners[0] == PlayState.X
    assert winners[1] is None
    assert winners[2] == PlayState.X

    # Now reset a couple of the boards:
    bgs = bgs.play_at([0, 2, 0], [True, False, True])
    torch.testing.assert_close(bgs._board_state[0], bgs._blank_board())
    torch.testing.assert_close(bgs._board_state[2], bgs._blank_board())
    winners = bgs.winners(3)
    assert winners[0] is None
    assert winners[1] is None
    assert winners[2] is None


def test_batch_next_actions():
    bgs = BatchGameState(
        batch_size=2, turn=PlayState.X, num_cols=7, num_rows=3
    )
    plays1 = [4, 4, 4, 3, 3, 3, 1, 2]
    plays2 = [4, 4, 3, 3, 2, 2, 1, 1]
    bgs = reduce(BatchGameState.play_at, zip(plays1, plays2), bgs)
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


def test_multinomial_sampler(ps=(0.6, 0.3, 0.1), n=5000, eps=0.5):
    logits = torch.log(torch.tensor([ps] * n))
    mask = torch.tensor([[False] * len(ps)] * n)
    samples = cf_nn.sample_masked_multinomial(logits, mask=mask, axis=1)
    counts = Counter(samples.numpy().tolist())
    for i, p in enumerate(ps):
        em_p = counts[i] / n
        assert abs(em_p - p) < eps, "Probs are not close: %s vs %s" % (em_p, p)


def test_gaussian_neg_log_likelihood(eps=1e-3):
    mu = torch.tensor([27.0, -25.0, 37.0, -12.0, 2.0])
    log_sig = torch.tensor([1.2, 1.1, 2.4, 0.2, 1.6])
    norm = torch.distributions.Normal(loc=mu, scale=torch.exp(log_sig))
    x = torch.tensor([22.29, -20.04, 33.13, -11.44, -11.76])
    expected_log_prob = norm.log_prob(x)
    actual_log_prob = -cf_nn.gaussian_neg_log_likelihood(mu, log_sig, x)
    assert torch.all(
        torch.less_equal(torch.abs(expected_log_prob - actual_log_prob), eps)
    )
