import torch
from torch import nn
import math


def sample_gumbel(shape):
    uniform_samples = torch.rand(shape) * (1.0 - 2e-5) + 1e-5
    return -torch.log(-torch.log(uniform_samples))


def sample_masked_multinomial(logits, mask, axis=None):
    """Samples from a multinomial with the given logits, excluding masked items."""
    gumbels = sample_gumbel(logits.shape)
    noisy_logits = logits + gumbels.to(logits)
    min_val = torch.broadcast_to(torch.min(noisy_logits) - 1, logits.shape)
    masked_logits = torch.where(mask, min_val, noisy_logits)
    return torch.argmax(masked_logits, dim=axis)


def gaussian_neg_log_likelihood(mu, log_sig, x):
    """1-dim gaussian"""
    l2_diff = (mu - x) ** 2
    scaled_l2 = l2_diff * torch.exp(-log_sig * 2) / 2
    log_z = -log_sig - math.log(2 * math.pi) / 2
    return scaled_l2 - log_z


class ResidualLayer(nn.Module):
    def __init__(self, height, width, n_channels, filter_size):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=filter_size,
            padding=(filter_size - 1) // 2,
        )
        self.ln1 = nn.LayerNorm([n_channels, height, width])
        self.act1 = nn.GELU()

    def forward(self, x):
        z = self.conv1(x)
        z = self.ln1(z)
        z = self.act1(z)
        return x + z
