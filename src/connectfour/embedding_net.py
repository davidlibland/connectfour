"""The embedding network"""
import math

import torch
import torch.nn as nn

from connectfour.nn import ResidualLayer, ConnectFourFeatures, LayerNorm2d
from connectfour.play_state import play_state_embedding_ix, PlayState


class EmbeddingNet(nn.Module):
    """This is a simple resnet"""

    def __init__(self, kernel_size, out_channels, depth, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert kernel_size % 2 == 1
        initial_kernel = 2 * (kernel_size - 1) + 1
        explode_channels = 2 * out_channels
        self.out_channels = out_channels
        self.fan_out = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=explode_channels,
                kernel_size=initial_kernel,
                padding=(initial_kernel - 1) // 2,
            ),
            LayerNorm2d(explode_channels),
            nn.GELU(),
            nn.Conv2d(
                in_channels=explode_channels,
                out_channels=out_channels,
                kernel_size=1,
            ),
        )
        self.layers = nn.Sequential(
            *[ResidualLayer(in_channels=out_channels, filter_size=3)] * depth,
        )

        # Initialization
        for res in self.layers:
            nn.init.kaiming_normal_(res.convolution[0].weight)
            nn.init.zeros_(res.convolution[0].bias)
            nn.init.zeros_(res.convolution[3].weight)
            nn.init.zeros_(res.convolution[3].bias)

            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                res.attention[1].value_embed.weight
            )
            nn.init.normal_(
                res.attention[1].value_embed.weight,
                mean=0,
                std=0.02 / (depth * math.sqrt(fan_in)),
            )

    def forward(self, x):
        """
        x: A (batch_size, 3, rows, cols) shaped tensor
        """
        x = self.fan_out(x)
        return self.layers(x)
