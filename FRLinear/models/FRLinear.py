import torch.nn as nn
import torch

from FRLinear.models.backbones.dlinear_backbone import DLinearBackbone
from FRLinear.models.decomposition.ma_decomposition import MADecomposition
from FRLinear.models.decomposition.adaptive_ma import AdaptiveMADecomposition
from FRLinear.models.decomposition.fft_decomposition import FFTDecomposition
from FRLinear.models.decomposition.wavelet_decomposition import WaveletDecomposition
from FRLinear.models.decomposition.gaussian_ma_decomposition import GaussianDecomposition
from FRLinear.models.decomposition.hierarchical_ma import HierarchicalMADecomposition
from FRLinear.models.decomposition.multiscale_conv import MultiScaleConvDecomposition
from FRLinear.models.spatial.svd_lowrank import SVDLowRank
from FRLinear.models.residuals.mlp_residual import MLPResidual
from FRLinear.models.outputs.p50_head import P50Head

class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()

        # backbone DLinear gốc
        self.trend_backbone = DLinearBackbone(configs, channels=1)

        self.seasonal_backbone = DLinearBackbone(configs, channels=2)

        self.mixed_backbone = DLinearBackbone(configs, channels=4)

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

        if configs.decomposition == "ma":
            self.decomposition = MADecomposition()

        elif configs.decomposition == "fft":
            self.decomposition = FFTDecomposition()

        elif configs.decomposition == "wavelet":
            self.decomposition = WaveletDecomposition()

        elif configs.decomposition == "gaussian":
            self.decomposition = GaussianDecomposition()

        elif configs.decomposition == "hierarchical":
            self.decomposition = HierarchicalMADecomposition()

        else:
            self.decomposition = MultiScaleConvDecomposition()

        self.svd = SVDLowRank(2)

        self.residual_model = MLPResidual(configs)

        self.output_head = P50Head(configs.enc_in)

        self.alpha = 0.1

        self.regime_gate = nn.Sequential(
            nn.Linear(configs.enc_in * 2, 64),
            nn.GELU(),
            nn.Linear(64, 3)
        )


    def forward(self, x):

        # trend, seasonal, residual = self.decomposition(x)

        TREND_IDX = [6]           # OT

        SEASONAL_IDX = [0,2]      # HUFL MUFL

        MIXED_IDX = [1,3,4,5]     # HULL MULL LUFL LULL

        trend_feature = x[:, :, TREND_IDX]
        seasonal_feature = x[:, :, SEASONAL_IDX]
        mixed_feature = x[:, :, MIXED_IDX]

        # ===============================
        # Trend group
        # ===============================

        trend_t, seasonal_t, _ = self.trend_decomposition(
            trend_feature
        )

        out_trend = self.trend_backbone(
            trend_t,
            seasonal_t
        )


        # ===============================
        # Seasonal group
        # ===============================

        trend_s, seasonal_s, _ = self.seasonal_decomposition(
            seasonal_feature
        )

        out_seasonal = self.seasonal_backbone(
            trend_s,
            seasonal_s
        )


        # ===============================
        # Mixed group
        # ===============================

        trend_m, seasonal_m, _ = self.mixed_decomposition(
            mixed_feature
        )

        out_mixed = self.mixed_backbone(
            trend_m,
            seasonal_m
        )

        B = x.size(0)

        out = torch.zeros(
            B,
            self.pred_len,
            self.enc_in,
            device=x.device
        )

        out[:, :, TREND_IDX] = out_trend
        out[:, :, SEASONAL_IDX] = out_seasonal
        out[:, :, MIXED_IDX] = out_mixed

        return out