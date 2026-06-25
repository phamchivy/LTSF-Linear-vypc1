import torch.nn as nn
import torch

from dream_research.models.backbones.dlinear_backbone import DLinearBackbone

from dream_research.models.decomposition.multiscale_conv import MultiScaleConvDecomposition
from dream_research.models.residuals.mlp_residual import MLPResidual
from dream_research.models.outputs.p50_head import P50Head

from dream_research.models.decomposition.ma_decomposition import MADecomposition
from dream_research.models.decomposition.fft_decomposition import FFTDecomposition
from dream_research.models.decomposition.wavelet_decomposition import WaveletDecomposition
from dream_research.models.decomposition.gaussian_ma_decomposition import GaussianDecomposition
from dream_research.models.spatial.svd_lowrank import SVDLowRank
from dream_research.models.decomposition.hierarchical_ma import HierarchicalMADecomposition

class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()

        # backbone DLinear gốc
        self.trend_expert = DLinearBackbone(configs)
        self.seasonal_expert = DLinearBackbone(configs)
        self.mixed_expert = DLinearBackbone(configs)

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

        self.svd = SVDLowRank(rank=configs.svd_rank)

        self.residual_model = MLPResidual(configs)

        self.output_head = P50Head(configs.enc_in)

        self.alpha = 0.1

        self.regime_gate = nn.Sequential(
            nn.Linear(configs.enc_in * 2, 64),
            nn.GELU(),
            nn.Linear(64, 3)
        )


    def forward(self, x):

        trend, seasonal, residual = self.decomposition(x)

        # DLinear dự báo Trend+Seasonal

        trend_only = trend
        seasonal_zero = torch.zeros_like(seasonal)

        out_trend = self.trend_expert(
            trend_only,
            seasonal_zero
        )

        trend_zero = torch.zeros_like(trend)

        out_seasonal = self.seasonal_expert(
            trend_zero,
            seasonal
        )

        out_mixed = self.mixed_expert(
            trend,
            seasonal
        )

        trend_feat = trend.mean(dim=1)
        seasonal_feat = seasonal.mean(dim=1)

        gate_input = torch.cat(
            [trend_feat, seasonal_feat],
            dim=-1
        )

        gate = torch.softmax(
            self.regime_gate(gate_input),
            dim=-1
        )

        g_trend = gate[:,0].view(-1,1,1)
        g_seasonal = gate[:,1].view(-1,1,1)
        g_mixed = gate[:,2].view(-1,1,1)

        base_forecast = (
            g_trend * out_trend
            + g_seasonal * out_seasonal
            + g_mixed * out_mixed
        )

        if self.training:
            print(
                gate[:,0].mean().item(),
                gate[:,1].mean().item()
            )

        # residual_forecast = self.residual_model(residual)

        # if residual is None:
        #     out = base_forecast
        # else:
        #     residual_lowrank = self.svd(
        #         residual
        #     )
        #     residual_forecast = self.residual_model(residual_lowrank)
        #     out = base_forecast + self.alpha * residual_forecast

        # out = base_forecast + self.alpha * residual_forecast

        return base_forecast