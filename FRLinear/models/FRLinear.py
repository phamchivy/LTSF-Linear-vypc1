import torch.nn as nn
import torch

from FRLinear.models.backbones.dlinear_backbone import DLinearBackbone
from FRLinear.models.decomposition.ma_decomposition import MADecomposition

class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()

        # backbone DLinear gốc
        self.trend_backbone = DLinearBackbone(configs, channels=1)

        self.seasonal_backbone = DLinearBackbone(configs, channels=2)

        self.mixed_backbone = DLinearBackbone(configs, channels=4)

        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        if configs.decomposition == "ma":
            self.decomposition = MADecomposition()
        else:
            self.decomposition = MADecomposition()

        self.regime_gate = nn.Sequential(
            nn.Linear(configs.enc_in * 2, 64),
            nn.GELU(),
            nn.Linear(64, 3)
        )


    def forward(self, x):

        trend, seasonal, residual = self.decomposition(x)

        TREND_IDX = [6]           # OT

        SEASONAL_IDX = [0,2]      # HUFL MUFL

        MIXED_IDX = [1,3,4,5]     # HULL MULL LUFL LULL

        trend_trend = trend[:,:,TREND_IDX]
        seasonal_trend = seasonal[:,:,TREND_IDX]

        trend_seasonal = trend[:,:,SEASONAL_IDX]
        seasonal_seasonal = seasonal[:,:,SEASONAL_IDX]

        trend_mixed = trend[:,:,MIXED_IDX]
        seasonal_mixed = seasonal[:,:,MIXED_IDX]

        out_trend = self.trend_backbone(
            trend_trend,
            seasonal_trend
        )

        out_seasonal = self.seasonal_backbone(
            trend_seasonal,
            seasonal_seasonal
        )

        out_mixed = self.mixed_backbone(
            trend_mixed,
            seasonal_mixed
        )

        B = x.size(0)

        out = torch.zeros(
            B,
            self.pred_len,
            self.enc_in,
            device=x.device
        )

        out[:,:,TREND_IDX] = out_trend

        out[:,:,SEASONAL_IDX] = out_seasonal

        out[:,:,MIXED_IDX] = out_mixed

        return out