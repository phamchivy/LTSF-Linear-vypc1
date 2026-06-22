import torch
import torch.nn as nn

from dream_research.decomposition_analysis.residual_energy import residual_energy
from dream_research.decomposition_analysis.residual_predictability import ResidualPredictor


class DecompositionBenchmark:

    def __init__(
            self,
            decomposition,
            configs,
            criterion=nn.L1Loss()
    ):

        self.decomposition = decomposition

        self.predictor = ResidualPredictor(configs)

        self.criterion = criterion

    def evaluate(self, batch_x, batch_y):

        """
        batch_x : [B,seq_len,C]
        batch_y : [B,pred_len,C]
        """

        # trend, seasonal, residual = self.decomposition(batch_x)

        # residual_y = batch_y - batch_y.mean(
        #     dim=1,
        #     keepdim=True
        # )

        trend_x, seasonal_x, residual_x = self.decomposition(batch_x)

        trend_y, seasonal_y, residual_y = self.decomposition(batch_y)

        residual_pred = self.predictor(residual_x)

        mae = self.criterion(
            residual_pred,
            residual_y
        ).item()

        energy = residual_energy(
            batch_x,
            residual
        )

        return {
            "residual_mae": mae,
            "residual_energy": energy
        }