import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAverage(nn.Module):

    def __init__(self, kernel_size):
        super().__init__()

        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2
        )

    def forward(self, x):

        x = x.permute(0,2,1)

        trend = self.avg(x)

        trend = trend.permute(0,2,1)

        return trend


class AdaptiveMADecomposition(nn.Module):

    def __init__(self,
                 kernel_sizes=[9,21,41]):

        super().__init__()

        self.mas = nn.ModuleList([
            MovingAverage(k)
            for k in kernel_sizes
        ])

        self.kernel_weights = nn.Parameter(
            torch.ones(len(kernel_sizes))
        )

    def forward(self,x):

        weights = F.softmax(
            self.kernel_weights,
            dim=0
        )

        trend = 0

        for w,ma in zip(weights,self.mas):

            trend = trend + w*ma(x)

        seasonal = x-trend

        return trend,seasonal,None