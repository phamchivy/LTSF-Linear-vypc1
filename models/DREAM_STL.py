import torch
import torch.nn as nn
import torch.nn.functional as F
from statsmodels.tsa.seasonal import STL


class ResidualMLP(nn.Module):
    """
    Simple residual expert.
    Có thể thay bằng Attention/CNN/Mamba sau này.
    """
    def __init__(self, seq_len, pred_len, channels):
        super().__init__()
        self.channels = channels
        self.fc = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x : [B,L,C]
        x = self.fc(x.permute(0,2,1)).permute(0,2,1)
        return x


class ResidualGate(nn.Module):
    """
    Forecast = base + gate * residual
    """
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(1,1,channels))

    def forward(self, base, residual):
        g = torch.sigmoid(self.gate)
        return base + g * residual


class STLDecomposition:
    def __init__(self, period=24):
        self.period = period

    def __call__(self, x):
        # x : [B,L,C]
        B,L,C = x.shape

        trend = torch.zeros_like(x)
        seasonal = torch.zeros_like(x)
        residual = torch.zeros_like(x)

        x_np = x.detach().cpu().numpy()

        for b in range(B):
            for c in range(C):
                result = STL(
                    x_np[b,:,c],
                    period=self.period,
                    robust=True
                ).fit()

                trend[b,:,c] = torch.tensor(result.trend)
                seasonal[b,:,c] = torch.tensor(result.seasonal)
                residual[b,:,c] = torch.tensor(result.resid)

        return (
            trend.to(x.device),
            seasonal.to(x.device),
            residual.to(x.device)
        )


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in

        self.stl = STLDecomposition(period=24)

        # DLinear backbone
        self.Linear = nn.Linear(self.seq_len, self.pred_len)

        # residual branch
        self.residual_expert = ResidualMLP(
            self.seq_len,
            self.pred_len,
            self.channels
        )

        self.residual_gate = ResidualGate(self.channels)

    def forward(self, x):
        # x : [B,L,C]

        trend, seasonal, residual = self.stl(x)

        base_input = trend + seasonal

        # P50 backbone
        p50 = self.Linear(
            base_input.permute(0,2,1)
        ).permute(0,2,1)

        # residual correction
        residual_pred = self.residual_expert(residual)

        # final forecast
        out = self.residual_gate(
            p50,
            residual_pred
        )

        return out