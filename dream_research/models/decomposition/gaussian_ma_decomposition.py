import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_decomposition import BaseDecomposition


class GaussianMovingAverage(nn.Module):

    def __init__(self, max_kernel_size=63, init_sigma=8.0):
        super().__init__()

        assert max_kernel_size % 2 == 1

        self.max_kernel_size = max_kernel_size

        # học log_sigma để đảm bảo sigma > 0
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(init_sigma)))

    def forward(self, x):
        # x: [B, L, C]

        B, L, C = x.shape

        pad = (self.max_kernel_size - 1) // 2

        # replicate padding giống MA
        front = x[:, 0:1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)

        x_pad = torch.cat([front, x, end], dim=1)

        # [B,C,L+2pad]
        x_pad = x_pad.permute(0, 2, 1)

        # sigma > 0
        sigma = F.softplus(self.log_sigma) + 1e-6

        # [-pad, ..., 0, ..., +pad]
        pos = torch.arange(
            -pad,
            pad + 1,
            device=x.device,
            dtype=x.dtype
        )

        # Gaussian weights
        weights = torch.exp(-(pos ** 2) / (2 * sigma ** 2))

        # normalize
        weights = weights / weights.sum()

        # depthwise conv
        kernel = weights.view(1, 1, -1).repeat(C, 1, 1)

        trend = F.conv1d(
            x_pad,
            kernel,
            groups=C
        )

        trend = trend.permute(0, 2, 1)

        return trend


class GaussianDecomposition(BaseDecomposition):

    def __init__(self, max_kernel_size=63, init_sigma=8.0):
        super().__init__()

        self.gaussian_ma = GaussianMovingAverage(
            max_kernel_size=max_kernel_size,
            init_sigma=init_sigma
        )

    def forward(self, x):
        # x: [B, L, C]

        trend = self.gaussian_ma(x)

        seasonal = x - trend

        residual = None

        return trend, seasonal, residual