import torch
import torch.nn as nn
import torch.nn.functional as F


class Value(nn.Module):
    def __init__(self,
        num_reward_layers: int=4,
        num_reward_filters: int=64
    ):
        super().__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv = nn.ModuleList()
        self.batch = nn.ModuleList()
        for i in range(num_reward_layers):
            num_in_filters = 3 if i == 0 else num_reward_filters
            self.conv.append(nn.Conv2d(num_in_filters, num_reward_filters, 3, padding=1))
            self.batch.append(nn.BatchNorm2d(num_reward_filters))
        # an affine operation: y = Wx + b
        self.fc = nn.Linear(num_reward_filters, 1)
        self.options = {
            "num_reward_layers": num_reward_layers,
            "num_reward_filters": num_reward_filters
        }

    def forward(self, x):
        """returns the logits"""
        for c, b in zip(self.conv, self.batch):
            x = F.relu(b(c(x)))
        # reduce over spatial dimensions
        x = torch.max(x, dim=-1)[0]
        x = torch.max(x, dim=-1)[0]
        x = self.fc(x)
        return x

