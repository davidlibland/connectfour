from collections import Counter
from functools import reduce
import torch

import src.nn as nn

from src.game import BatchGameState
from src.play_state import PlayState


def test_batch_winner():
    bgs = BatchGameState(batch_size=2, num_cols=5, num_rows=5, turn=PlayState.X)
    plays1 = [4,4,3,3,2]
    plays2 = [4,4,3,3,1]
    bgs = reduce(BatchGameState.play_at, zip(plays1,plays2), bgs)
    print(bgs)
    print(bgs.winners(3))
    assert bgs.winners(3)[0] == PlayState.X, "%s" % bgs.winners(3)[0]
    assert bgs.winners(3)[1] == None


def test_multinomial_sampler(ps=(.6, .3, .1), mask=(0, 1, 0), n=5000, eps=0.5):
    logits = torch.log(torch.Tensor([ps]*n))
    m_mask=torch.Tensor([[0] * len(ps)] * n).byte()
    samples = nn.sample_masked_multinomial(logits, mask=m_mask, axis=1)
    counts = Counter(samples.numpy().tolist())
    for i, p in enumerate(ps):
        em_p = counts[i]/n
        assert abs(em_p - p) < eps, "Probs are not close: %s vs %s" % (em_p, p)

    logits = torch.log(torch.Tensor([ps]*2*n))
    m_mask=torch.Tensor([[0] * len(ps)] * n + [list(mask)] * n).byte()
    samples = nn.sample_masked_multinomial(logits, mask=m_mask, axis=1)
    unmasked_cnts = Counter(samples.numpy().tolist()[:n])
    masked_cnts = Counter(samples.numpy().tolist()[n:])
    for i, p in enumerate(ps):
        em_p = unmasked_cnts[i]/n
        assert abs(em_p - p) < eps, "Probs are not close: %s vs %s" % (em_p, p)
    masked_ps = [p if not m else 0 for p, m in zip(ps, mask)]
    Z = sum(masked_ps)
    masked_ps = [p/Z for p in masked_ps]
    for i, p in enumerate(masked_ps):
        em_p = masked_cnts[i]/n
        assert abs(em_p - p) < eps, "Probs are not close: %s vs %s" % (em_p, p)
