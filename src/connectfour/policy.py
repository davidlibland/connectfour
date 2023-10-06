"""The policy network"""
import math

import torch
import torch.nn as nn

from connectfour.nn import ConnectFourFeatures, ResidualLayer
from connectfour.play_state import PlayState, play_state_embedding_ix


class PolicyNet(nn.Module):
    """This is a simple resnet"""

    def __init__(self, embedding: nn.Module, kernel_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert kernel_size % 2 == 1
        self.embedding = embedding
        self.collapser = nn.Conv2d(self.embedding.out_channels, 1, 1)

        # Initializatio. code
        nn.init.normal_(
            self.collapser.weight,
            mean=0,
            std=0.02 / math.sqrt(self.embedding.out_channels),
        )
        nn.init.zeros_(self.collapser.bias)

    @property
    def device(self):
        return self.layers[-1].weight.device

    def forward(self, x):
        """
        x: A (batch_size, 3, rows, cols) shaped tensor
        """
        all_vals = self.collapser(self.embedding(x))
        # Drop the (trivial) channel index:
        all_vals = all_vals[:, 0, :, :]

        # Determine the number of plays in each column:
        num_plays = torch.sum(
            x[:, play_state_embedding_ix(PlayState.BLANK) + 1 :, :, :],
            dim=(1, 2),
        ).to(dtype=torch.int64)
        # Use that to determine the locations one could play at:
        num_rows = x.shape[2]
        row = num_rows - 1 - num_plays
        # Add a (trivial) row index:
        row = row[:, None, :]
        # Clip to possible vals:
        row = row.clip(min=0, max=num_rows - 1)
        focused_vals = torch.gather(all_vals, 1, row)
        # Drop the (trivial) channel index:
        focused_vals = focused_vals[:, 0, :]
        return focused_vals
