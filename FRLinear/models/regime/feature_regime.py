import json


class FeatureRegime:

    def __init__(self, path):

        with open(path, "r") as f:
            regime = json.load(f)

        self.trend_idx = regime["trend"]
        self.seasonal_idx = regime["seasonal"]
        self.mixed_idx = regime["mixed"]

    def __repr__(self):

        return (
            f"Trend: {self.trend_idx}\n"
            f"Seasonal: {self.seasonal_idx}\n"
            f"Mixed: {self.mixed_idx}"
        )