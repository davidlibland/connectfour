import torch
import torch.nn as nn
import torch.nn.functional as F

from src.game import BatchGameState


class Policy(nn.Module):
    def __init__(self,
                 num_policy_2d_layers: int=4,
                 num_policy_1d_layers: int=3,
                 num_policy_filters: int=64,
                 ):
        super().__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv_2d = nn.ModuleList()
        self.batch_2d = nn.ModuleList()
        for i in range(num_policy_2d_layers):
            num_in_filters = 3 if i == 0 else num_policy_filters
            self.conv_2d.append(nn.Conv2d(num_in_filters, num_policy_filters, 3, padding=1))
            self.batch_2d.append(nn.BatchNorm2d(num_policy_filters))
        self.conv_1d = nn.ModuleList()
        self.batch_1d = nn.ModuleList()
        for i in range(num_policy_1d_layers-1):
            self.conv_1d.append(nn.Conv1d(num_policy_filters, num_policy_filters, 3, padding=1))
            self.batch_1d.append(nn.BatchNorm1d(num_policy_filters))
        self.final = nn.Conv1d(num_policy_filters, 1, 3, padding=1)
        self.options = {
            "num_policy_2d_layers": num_policy_2d_layers,
            "num_policy_1d_layers": num_policy_1d_layers,
            "num_policy_filters": num_policy_filters
        }

    def forward(self, x):
        """returns the logits"""
        for c, b in zip(self.conv_2d, self.batch_2d):
            x = F.relu(b(c(x)))
        # reduce over spatial dimensions
        x = torch.max(x, dim=2)[0]
        for c, b in zip(self.conv_1d, self.batch_1d):
            x = F.relu(b(c(x)))
        x = self.final(x)
        width = x.shape[-1]
        return x.view([-1, width])
