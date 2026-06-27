import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy.signal import periodogram
from statsmodels.tsa.stattools import acf


def compute_scores(x):

    x = np.asarray(x)

    # ===== Trend score =====
    acf_values = acf(
        x,
        nlags=min(500, len(x) // 2),
        fft=True
    )

    trend_score = np.mean(acf_values[1:50])

    # ===== Seasonal score =====
    freq, power = periodogram(x)

    if len(power) < 2:
        seasonal_score = 0
    else:
        idx = np.argmax(power[1:]) + 1
        seasonal_score = power[idx] / (
            power.sum() + 1e-8
        )

    return trend_score, seasonal_score


def assign_regime(
        trend_score,
        seasonal_score,
        trend_th=0.8,
        seasonal_th=0.3
):

    if trend_score > trend_th \
            and seasonal_score < seasonal_th:
        return "trend"

    elif seasonal_score > seasonal_th \
            and trend_score < trend_th:
        return "seasonal"

    else:
        return "mixed"


def main(args):

    data = pd.read_csv(args.path)

    feature_names = list(data.columns)[1:]

    regimes = {
        "trend": [],
        "seasonal": [],
        "mixed": []
    }

    detail = {}

    for idx, col in enumerate(feature_names):

        ts = data[col].values

        trend_score, seasonal_score = \
            compute_scores(ts)

        regime = assign_regime(
            trend_score,
            seasonal_score
        )

        regimes[regime].append(idx)

        detail[col] = {
            "index": idx,
            "trend_score": float(trend_score),
            "seasonal_score": float(seasonal_score),
            "regime": regime
        }

        print(
            f"{col:5s}"
            f" trend={trend_score:.3f}"
            f" seasonal={seasonal_score:.3f}"
            f" -> {regime}"
        )

    os.makedirs(
        args.save_dir,
        exist_ok=True
    )

    save_file = os.path.join(
        args.save_dir,
        f"{args.name}.json"
    )

    with open(save_file, "w") as f:
        json.dump(
            {
                "trend": regimes["trend"],
                "seasonal": regimes["seasonal"],
                "mixed": regimes["mixed"],
                "detail": detail
            },
            f,
            indent=4
        )

    print()
    print("Saved to:")
    print(save_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--name",
        type=str,
        required=True
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="../regime_cache"
    )

    args = parser.parse_args()

    main(args)