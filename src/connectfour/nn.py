import math

import torch
import torch.nn.functional as F
from torch import nn

from connectfour.utils import get_winning_filters


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


class LayerNorm2d(nn.LayerNorm):
    r"""LayerNorm for channels_first tensors with 2d spatial dimensions (ie N, C, H, W)."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x) -> torch.Tensor:
        return F.layer_norm(
            x.permute(0, 2, 3, 1),
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
        ).permute(0, 3, 1, 2)


class ResidualLayer(nn.Module):
    def __init__(self, in_channels, filter_size, bottleneck_dim=8):
        super().__init__()
        self.attention = nn.Sequential(
            LayerNorm2d(in_channels),
            ConvAttention(
                input_dim=in_channels,
                embed_dim=bottleneck_dim,
                n_heads=in_channels,
                kernel_size=filter_size,
                padding=(filter_size - 1) // 2,
            ),
        )
        self.convolution = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=filter_size,
                padding=(filter_size - 1) // 2,
            ),
            LayerNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        x = x + self.attention(x)
        x = x + self.convolution(x)
        return x


class ConnectFourFeatures(nn.Module):
    def __init__(self, *run_lengths):
        super().__init__()
        filters = []
        for l in run_lengths:
            filters.extend(get_winning_filters(l))
        self.filter_names = []
        for i, filter in enumerate(filters):
            filter_name = f"filter_{i}"
            self.register_buffer(filter_name, filter)
            self.filter_names.append(filter_name)

    @staticmethod
    def compute_out_channels(in_channels, *run_lengths):
        return in_channels + 4 * 3 * len(run_lengths)

    def forward(self, x):
        stacks = [x]
        for filter_name in self.filter_names:
            filter = self.get_buffer(filter_name)
            filter_sizes = filter.shape[2:]
            padding = tuple((n - 1) // 2 for n in filter_sizes)
            w = nn.functional.conv2d(
                x,
                filter,
                stride=1,
                padding=padding,
            )
            stacks.append(w)
        return torch.concat(stacks, 1)


class ConvAttention(nn.Module):
    def __init__(
        self, input_dim, embed_dim, n_heads, kernel_size=1, padding=0
    ):
        super().__init__()
        self._embed_dim = embed_dim
        self._n_heads = n_heads
        self.query_embed = nn.Conv2d(
            input_dim,
            embed_dim * n_heads,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.key_embed = nn.Conv2d(
            input_dim,
            embed_dim * n_heads,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.value_embed = nn.Conv2d(
            input_dim,
            embed_dim * n_heads,
            kernel_size=kernel_size,
            padding=padding,
        )

        # Initialization:
        nn.init.normal_(self.query_embed.weight, mean=0, std=0.02)
        nn.init.normal_(self.key_embed.weight, mean=0, std=0.02)
        nn.init.kaiming_normal_(self.value_embed.weight)

        nn.init.zeros_(self.query_embed.bias)
        nn.init.zeros_(self.key_embed.bias)
        nn.init.zeros_(self.value_embed.bias)

    def _stack_channels(self, x):
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(
            [
                x.shape[0],
                x.shape[1],
                x.shape[2],
                self._embed_dim,
                self._n_heads,
            ]
        )
        return x.permute(0, 3, 4, 1, 2)

    def forward(self, x):
        scores_wide = (
            self.query_embed(x)
            * self.key_embed(x)
            / math.sqrt(self._embed_dim)
        )
        values_wide = self.value_embed(x)
        scores = self._stack_channels(scores_wide)
        values = self._stack_channels(values_wide)
        scores_soft = nn.functional.softmax(scores, 1)
        out = (values * scores_soft).sum(dim=1)
        assert out.shape == (x.shape[0], self._n_heads, x.shape[2], x.shape[3])
        return out
