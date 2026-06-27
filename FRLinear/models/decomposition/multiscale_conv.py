import torch
import torch.nn as nn


class MultiScaleConvDecomposition(nn.Module):
    def __init__(self):
        super().__init__()

        self.trend_conv = nn.Conv1d(
            1, 1, kernel_size=51,
            padding=25,
            bias=False
        )

        self.seasonal_conv = nn.Conv1d(
            1, 1, kernel_size=25,
            padding=12,
            bias=False
        )

    def forward(self, x):
        # x : [B,L,C]

        B, L, C = x.shape

        x_perm = x.permute(0, 2, 1)          # [B,C,L]
        x_flat = x_perm.reshape(B*C, 1, L)

        trend = self.trend_conv(x_flat)

        detrended = x_flat - trend

        seasonal = self.seasonal_conv(detrended)

        residual = x_flat - trend - seasonal

        trend = trend.reshape(B, C, L).permute(0, 2, 1)
        seasonal = seasonal.reshape(B, C, L).permute(0, 2, 1)
        residual = residual.reshape(B, C, L).permute(0, 2, 1)

        return trend, seasonal, residual