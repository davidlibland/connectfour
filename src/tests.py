from collections import Counter
from functools import reduce
import tensorflow as tf
tf.enable_eager_execution()

import src.nn as nn

from src.game import BatchGameState
from src.play_state import PlayState


def test_batch_winner():
    bgs = BatchGameState(batch_size=2)
    plays1 = [4,4,3,3,2]
    plays2 = [4,4,3,3,1]
    bgs = reduce(BatchGameState.play_at, zip(plays1,plays2), bgs)
    print(bgs)
    print(bgs.winners(3))
    assert bgs.winners(3)[0] == PlayState.X
    assert bgs.winners(3)[1] == None


def test_multinomial_sampler(ps=(.6, .3, .1), n=5000, eps=0.5):
    logits = tf.log(tf.constant([ps]*n))
    mask=[[False]*len(ps)]*n
    samples = nn.sample_masked_multinomial(logits, mask=mask, axis=1)
    counts = Counter(samples.numpy().tolist())
    for i, p in enumerate(ps):
        em_p = counts[i]/n
        assert abs(em_p - p) < eps, "Probs are not close: %s vs %s" % (em_p, p)


def test_gaussian_neg_log_likelihood(eps=1e-3):
    mu = tf.constant([27., -25., 37., -12., 2.])
    log_sig = tf.constant([1.2, 1.1, 2.4, 0.2, 1.6])
    norm = tf.distributions.Normal(loc=mu,scale=tf.exp(log_sig))
    x = tf.constant([ 22.29, -20.04,  33.13, -11.44, -11.76])
    expected_log_prob = norm.log_prob(x)
    actual_log_prob = -nn.gaussian_neg_log_likelihood(mu, log_sig, x)
    assert tf.reduce_all(tf.less_equal(tf.abs(expected_log_prob - actual_log_prob), eps))
