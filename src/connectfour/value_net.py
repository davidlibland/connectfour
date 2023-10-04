"""The value network"""
import math

import torch
import torch.nn as nn
from connectfour.nn import ResidualLayer, ConnectFourFeatures, ConvAttention


class ValueNet(nn.Module):
    """This is a simple resnet"""

    def __init__(
        self, embedding: nn.Module, kernel_size, n_rows, n_cols, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert kernel_size % 2 == 1
        self.embedding = embedding
        attention_dim = self.embedding.out_channels
        self.dot_dim = self.embedding.out_channels * n_rows * n_cols
        self.fully_connected_key_query = nn.Linear(
            self.embedding.out_channels * n_rows * n_cols, attention_dim
        )
        self.fully_connected_value = nn.Linear(
            self.embedding.out_channels * n_rows * n_cols, attention_dim
        )

    def forward(self, x):
        """
        x: A (batch_size, 3, rows, cols) shaped tensor
        """
        out = self.embedding(x)
        attention = nn.functional.softmax(
            self.fully_connected_key_query(out.reshape(x.shape[0], -1))
            / math.sqrt(self.dot_dim),
            dim=1,
        )
        value = (
            attention * self.fully_connected_value(out.reshape(x.shape[0], -1))
        ).sum(dim=1)

        # value = torch.mean(value, dim=(1, 2, 3))

        # if the board is empty, the value should be zero:
        empty_masks = x[:, 1:, :, :].sum(dim=(1, 2, 3)) == 0
        zeros = torch.zeros_like(value)
        value = torch.where(empty_masks, zeros, value)
        # pass the reward through a tanh - it should never exceed the true reward in [-1, 1]
        out = value / (1 + torch.maximum(value, -value))
        return out
