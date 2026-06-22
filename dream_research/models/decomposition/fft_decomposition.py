import torch
from .base_decomposition import BaseDecomposition


class FFTDecomposition(BaseDecomposition):

    def __init__(self, keep_ratio=0.1):
        super().__init__()

        self.keep_ratio = keep_ratio

    def forward(self, x):

        B,L,C = x.shape

        freq = torch.fft.rfft(x, dim=1)

        n_freq = freq.shape[1]

        k = int(n_freq*self.keep_ratio)

        mask = torch.zeros_like(freq)

        mask[:, :k, :] = 1

        low_freq = freq * mask

        trend = torch.fft.irfft(
            low_freq,
            n=L,
            dim=1
        )

        seasonal = x - trend

        residual = torch.zeros_like(x)

        return trend, seasonal, residual