"""The policy network"""
import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    """This is a simple resnet"""

    def __init__(self, rows, cols, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rows = rows
        self.cols = cols
        self.ll = nn.Linear(rows * cols * 3, cols)

    def forward(self, x):
        """
        x: A (batch_size, 3, rows, cols) shaped tensor
        """
        logits = self.ll(x.reshape(-1, self.rows * self.cols * 3))
        return logits
