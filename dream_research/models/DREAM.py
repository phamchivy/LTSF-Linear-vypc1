import torch.nn as nn

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
        self.backbone = DLinearBackbone(configs)

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


    def forward(self, x):

        trend, seasonal, residual = self.decomposition(x)

        # DLinear dự báo Trend+Seasonal

        base_forecast = self.backbone(
            trend,
            seasonal
        )

        # residual_forecast = self.residual_model(residual)

        if residual is None:
            out = base_forecast
        else:
            residual_lowrank = self.svd(
                residual
            )
            residual_forecast = self.residual_model(residual_lowrank)
            out = base_forecast + self.alpha * residual_forecast

        # out = base_forecast + self.alpha * residual_forecast

        return out