import torch
import torch.nn as nn
from .base_decomposition import BaseDecomposition
from .ma_decomposition import MovingAverage


class HierarchicalMADecomposition(BaseDecomposition):

    def __init__(
            self,
            short_kernel=7,
            long_kernel=51):

        super().__init__()

        assert short_kernel < long_kernel

        self.short_ma = MovingAverage(short_kernel)
        self.long_ma = MovingAverage(long_kernel)

    def forward(self, x):
        # x : [B,L,C]

        trend_small = self.short_ma(x)
        trend_large = self.long_ma(x)

        trend = trend_large

        seasonal = trend_small - trend_large

        residual = x - trend_small

        return trend, seasonal, residual