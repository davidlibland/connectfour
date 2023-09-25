"""The policy network"""
import torch
import torch.nn as nn

from connectfour.nn import ResidualLayer, ConnectFourFeatures


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
        return torch.amax(all_vals, dim=(1, 2))
