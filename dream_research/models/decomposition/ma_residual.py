import torch
import torch.nn as nn


class moving_avg(nn.Module):
    """
    Moving average block giống DLinear
    """
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=stride,
            padding=0
        )

    def forward(self, x):
        # x: [B,L,C]

        front = x[:, 0:1, :].repeat(
            1, (self.kernel_size - 1) // 2, 1)

        end = x[:, -1:, :].repeat(
            1, (self.kernel_size - 1) // 2, 1)

        x = torch.cat([front, x, end], dim=1)

        x = self.avg(
            x.permute(0, 2, 1)
        ).permute(0, 2, 1)

        return x


class MAResidualDecomposition(nn.Module):
    """
    X = Trend + Seasonal
    Residual = Seasonal
    """

    def __init__(self, kernel_size=25):
        super().__init__()

        self.moving_avg = moving_avg(kernel_size)

    def forward(self, x):
        # x : [B,L,C]

        trend = self.moving_avg(x)

        seasonal = x - trend

        residual = seasonal

        return trend, seasonal, residual