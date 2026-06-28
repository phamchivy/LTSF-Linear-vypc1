import torch.nn as nn
import torch

from FRLinear.models.backbones.dlinear_backbone import DLinearBackbone
from FRLinear.models.decomposition.ma_decomposition import MADecomposition
from FRLinear.models.decomposition.adaptive_ma import AdaptiveMADecomposition
from FRLinear.models.regime.feature_regime import FeatureRegime

class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()

        dataset = configs.data

        self.regime = FeatureRegime(
            f"FRLinear/regime_cache/{dataset}.json"
        )

        if self.regime is not None:
            n_trend = max(1, len(self.regime.trend_idx))
            n_seasonal = max(1, len(self.regime.seasonal_idx))
            n_mixed = max(1, len(self.regime.mixed_idx))
        else:
            n_trend = 1
            n_seasonal = 2
            n_mixed = 4

        # backbone DLinear gốc
        self.trend_backbone = DLinearBackbone(configs, channels=n_trend)

        self.seasonal_backbone = DLinearBackbone(configs, channels=n_seasonal)

        self.mixed_backbone = DLinearBackbone(configs, channels=n_mixed)

        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        self.trend_decomposition = AdaptiveMADecomposition(
            kernel_sizes=[21,31,41]
        )

        self.seasonal_decomposition = AdaptiveMADecomposition(
            kernel_sizes=[5,9,13]
        )

        self.mixed_decomposition = AdaptiveMADecomposition(
            kernel_sizes=[11,21,31]
        )

        self.decomposition = MADecomposition()

        self.regime_gate = nn.Sequential(
            nn.Linear(configs.enc_in * 2, 64),
            nn.GELU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):

        if self.regime is not None:
            trend_idx = self.regime.trend_idx
            seasonal_idx = self.regime.seasonal_idx
            mixed_idx = self.regime.mixed_idx
        else:
            trend_idx = [6]
            seasonal_idx = [0, 2]
            mixed_idx = [1, 3, 4, 5]

        # ===================================
        # Split features
        # ===================================

        trend_feature = x[:, :, trend_idx]
        seasonal_feature = x[:, :, seasonal_idx]
        mixed_feature = x[:, :, mixed_idx]

        # ===================================
        # Trend expert
        # ===================================

        trend_t, seasonal_t, _ = \
            self.trend_decomposition(
                trend_feature
            )

        trend_out = self.trend_backbone(
            trend_t,
            seasonal_t
        )

        # ===================================
        # Seasonal expert
        # ===================================

        trend_s, seasonal_s, _ = \
            self.seasonal_decomposition(
                seasonal_feature
            )

        seasonal_out = self.seasonal_backbone(
            trend_s,
            seasonal_s
        )

        # ===================================
        # Mixed expert
        # ===================================

        trend_m, seasonal_m, _ = \
            self.mixed_decomposition(
                mixed_feature
            )

        mixed_out = self.mixed_backbone(
            trend_m,
            seasonal_m
        )

        # ===================================
        # Residual regime gate
        # ===================================

        feat = torch.cat(
            [
                x.mean(dim=1),
                x.std(dim=1)
            ],
            dim=-1
        )

        gate = self.regime_gate(feat)
        gate = torch.tanh(gate)

        delta = 0.005

        g1 = 1 + delta * gate[:, 0].view(-1, 1, 1)
        g2 = 1 + delta * gate[:, 1].view(-1, 1, 1)
        g3 = 1 + delta * gate[:, 2].view(-1, 1, 1)

        # THỰC SỰ DÙNG GATE
        trend_out = g1 * trend_out
        seasonal_out = g2 * seasonal_out
        mixed_out = g3 * mixed_out

        # ===================================
        # Merge output
        # ===================================

        B = x.size(0)

        out = torch.zeros(
            B,
            self.pred_len,
            self.enc_in,
            device=x.device
        )

        out[:, :, trend_idx] = trend_out
        out[:, :, seasonal_idx] = seasonal_out
        out[:, :, mixed_idx] = mixed_out

        return out