import torch
import torch.nn as nn

from models.Linear import Model as Linear


class ResidualPredictor(nn.Module):

    def __init__(self, configs):
        super().__init__()

        self.model = Linear(configs)

    def forward(self, residual):

        return self.model(residual)