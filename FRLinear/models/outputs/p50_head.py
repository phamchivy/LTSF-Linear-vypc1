import torch.nn as nn
import torch


class P50Head(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.gate = nn.Parameter(
            torch.zeros(1,1,channels)
        )

    def forward(self, base, residual):

        g = self.gate.sigmoid()

        return base + g * residual