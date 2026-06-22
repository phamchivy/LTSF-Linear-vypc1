import torch
import torch.nn as nn
from .base_decomposition import BaseDecomposition


class WaveletDecomposition(BaseDecomposition):

    def __init__(self):
        super().__init__()

        self.ma_long = nn.AvgPool1d(
            25,
            stride=1,
            padding=12
        )

        self.ma_short = nn.AvgPool1d(
            7,
            stride=1,
            padding=3
        )

    def forward(self, x):

        xt = x.permute(0,2,1)

        trend = self.ma_long(xt).permute(0,2,1)

        smooth_short = self.ma_short(xt).permute(0,2,1)

        seasonal = smooth_short - trend

        residual = x - trend - seasonal

        return trend, seasonal, residual