import torch
import math


def sample_gumbel(shape):
    uniform_samples = torch.rand(shape)*(1. - 2e-5) + 1e-5
    return -torch.log(-torch.log(uniform_samples))


def sample_masked_multinomial(logits, mask, axis=None):
    gumbels = sample_gumbel(logits.shape)
    noisy_logits = logits + gumbels
    min_val = torch.broadcast_to(torch.min(noisy_logits) - 1, logits.shape)
    masked_logits = torch.where(mask, min_val, noisy_logits)
    return torch.argmax(masked_logits, dim=axis)


def gaussian_neg_log_likelihood(mu, log_sig, x):
    """1-dim gaussian"""
    l2_diff = (mu-x)**2
    scaled_l2 = l2_diff*torch.exp(-log_sig*2)/2
    log_z = -log_sig - math.log(2*math.pi)/2
    return scaled_l2 - log_z
