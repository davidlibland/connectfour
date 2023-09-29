"""The value network"""
import torch
import torch.nn as nn
from connectfour.nn import ResidualLayer, ConnectFourFeatures


class ValueNet(nn.Module):
    """This is a simple resnet"""

    def __init__(self, run_lengths, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.ll = nn.Linear(rows * cols * 3, 1)
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

    def forward(self, x):
        """
        x: A (batch_size, 3, rows, cols) shaped tensor
        """
        value = self.layers(x)
        value = torch.mean(value, dim=(1, 2, 3))

        # if the board is empty, the value should be zero:
        empty_masks = x[:, 1:, :, :].sum(dim=(1, 2, 3)) == 0
        zeros = torch.zeros_like(value)
        value = torch.where(empty_masks, zeros, value)
        return value
