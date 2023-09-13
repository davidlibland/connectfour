"""The value network"""
import torch
import torch.nn as nn
from connectfour.nn import ResidualLayer


class ValueNet(nn.Module):
    """This is a simple resnet"""

    def __init__(self, rows, cols, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rows = rows
        self.cols = cols
        # self.ll = nn.Linear(rows * cols * 3, 1)
        self.layers = nn.Sequential(
            ResidualLayer(
                height=rows, width=cols, n_channels=3, filter_size=3
            ),
            ResidualLayer(
                height=rows, width=cols, n_channels=3, filter_size=3
            ),
            ResidualLayer(
                height=rows, width=cols, n_channels=3, filter_size=3
            ),
            ResidualLayer(
                height=rows, width=cols, n_channels=3, filter_size=3
            ),
            nn.Conv2d(3, 1, 1),
        )

    def forward(self, x):
        """
        x: A (batch_size, 3, rows, cols) shaped tensor
        """
        value = self.layers(x)
        return torch.mean(value, dim=(1, 2, 3))
