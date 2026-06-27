import torch
import torch.nn as nn


class DLinearBackbone(nn.Module):

    def __init__(self, configs, channels):
        super().__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = channels
        self.individual = configs.individual

        if self.individual:

            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for _ in range(self.channels):
                self.Linear_Seasonal.append(
                    nn.Linear(self.seq_len, self.pred_len)
                )
                self.Linear_Trend.append(
                    nn.Linear(self.seq_len, self.pred_len)
                )

        else:

            self.Linear_Seasonal = nn.Linear(
                self.seq_len,
                self.pred_len
            )

            self.Linear_Trend = nn.Linear(
                self.seq_len,
                self.pred_len
            )
        
        print("Trend:")
        print(self.Linear_Trend.weight[0,:5])

        print("Seasonal:")
        print(self.Linear_Seasonal.weight[0,:5])

    def forward(self, trend, seasonal):

        # [B,L,C] -> [B,C,L]
        trend = trend.permute(0, 2, 1)
        seasonal = seasonal.permute(0, 2, 1)

        if self.individual:

            seasonal_output = torch.zeros(
                seasonal.size(0),
                seasonal.size(1),
                self.pred_len,
                device=seasonal.device
            )

            trend_output = torch.zeros(
                trend.size(0),
                trend.size(1),
                self.pred_len,
                device=trend.device
            )

            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend[:, i, :])

        else:

            seasonal_output = self.Linear_Seasonal(seasonal)
            trend_output = self.Linear_Trend(trend)

        out = seasonal_output + trend_output

        return out.permute(0, 2, 1)