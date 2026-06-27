import torch
import torch.nn as nn


class MLPResidual(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(configs.seq_len, 128),
            nn.GELU(),
            nn.Linear(128, configs.pred_len)
        )

    def forward(self, x):
        # [B,L,C]

        return self.net(
            x.permute(0,2,1)
        ).permute(0,2,1)