"""The policy network"""
import torch
import torch.nn as nn

from connectfour.nn import ResidualLayer, ConnectFourFeatures
from connectfour.play_state import play_state_embedding_ix, PlayState


class PolicyNet(nn.Module):
    """This is a simple resnet"""

    def __init__(self, run_lengths, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n_deep_channels = ConnectFourFeatures.compute_out_channels(
            3, *run_lengths
        )
        self.layers = nn.Sequential(
            ConnectFourFeatures(*run_lengths),
            ResidualLayer(in_channels=n_deep_channels, filter_size=3),
            ResidualLayer(in_channels=n_deep_channels, filter_size=3),
            ResidualLayer(in_channels=n_deep_channels, filter_size=3),
            ResidualLayer(in_channels=n_deep_channels, filter_size=3),
            nn.Conv2d(n_deep_channels, 1, 1),
        )

    @property
    def device(self):
        return self.layers[-1].weight.device

    def forward(self, x):
        """
        x: A (batch_size, 3, rows, cols) shaped tensor
        """
        all_vals = self.layers(x)
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
