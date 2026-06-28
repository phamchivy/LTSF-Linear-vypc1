import os
import json
import numpy as np
import pandas as pd

from pathlib import Path
from statsmodels.tsa.stattools import acf
from scipy.fft import fft, fftfreq

from src.config import DATASET_DIR, DATASET

SAVE_DIR = "../regime_cache"
os.makedirs(SAVE_DIR, exist_ok=True)


def extract_feature_vector(x):
    x = np.asarray(x)

    ############################
    # ACF
    ############################

    acf_vals = acf(
        x,
        nlags=336,
        fft=True
    )

    trend_score = np.mean(
        np.abs(acf_vals[:100])
    )

    acf_24 = abs(acf_vals[24])
    acf_48 = abs(acf_vals[48])
    acf_168 = abs(acf_vals[168])

    ############################
    # FFT
    ############################

    N = len(x)

    yf = np.abs(
        fft(x)
    )[:N // 2]

    xf = fftfreq(
        N,
        d=1
    )[:N // 2]

    power = yf ** 2
    total_power = power.sum() + 1e-8

    idx24 = np.argmin(
        np.abs(xf - 1 / 24)
    )

    idx12 = np.argmin(
        np.abs(xf - 1 / 12)
    )

    idx8 = np.argmin(
        np.abs(xf - 1 / 8)
    )

    season_score = (
        power[idx24]
        + power[idx12]
        + power[idx8]
    ) / total_power

    return np.array(
        [
            trend_score,
            acf_24,
            acf_48,
            acf_168,
            season_score
        ]
    )


def main():

    DATA_PATH = DATASET_DIR / DATASET
    dataset_name = Path(DATASET).stem

    df = pd.read_csv(DATA_PATH)

    feature_cols = list(
        df.columns[1:]
    )

    feature_vectors = []

    for col in feature_cols:
        vec = extract_feature_vector(
            df[col].values
        )
        feature_vectors.append(vec)

    feature_vectors = np.stack(
        feature_vectors
    )

    trend_scores = feature_vectors[:, 0]
    season_scores = feature_vectors[:, 4]

    ############################
    # Trend
    ############################

    trend_idx = int(
        np.argmax(trend_scores)
    )

    ############################
    # Seasonal
    ############################

    q = np.quantile(
        season_scores,
        0.70
    )

    seasonal_idx = []

    for i in range(len(feature_cols)):

        if i == trend_idx:
            continue

        if season_scores[i] >= q:
            seasonal_idx.append(i)

    if len(seasonal_idx) == 0:

        tmp = season_scores.copy()
        tmp[trend_idx] = -1

        seasonal_idx.append(
            int(np.argmax(tmp))
        )

    ############################
    # Mixed
    ############################

    mixed_idx = []

    for i in range(len(feature_cols)):

        if i == trend_idx:
            continue

        if i in seasonal_idx:
            continue

        mixed_idx.append(i)

    result = {
        "trend": [trend_idx],
        "seasonal": seasonal_idx,
        "mixed": mixed_idx,
        "detail": {}
    }

    for i, col in enumerate(feature_cols):

        if i in result["trend"]:
            regime = "trend"
        elif i in result["seasonal"]:
            regime = "seasonal"
        else:
            regime = "mixed"

        result["detail"][col] = {
            "index": i,
            "trend_score":
                float(trend_scores[i]),
            "seasonal_score":
                float(season_scores[i]),
            "regime":
                regime
        }

        print(
            f"{col:5s}"
            f" trend={trend_scores[i]:.3f}"
            f" seasonal={season_scores[i]:.3f}"
            f" -> {regime}"
        )

    save_path = (
        Path(SAVE_DIR)
        / f"{dataset_name}.json"
    )

    with open(
        save_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            result,
            f,
            indent=4
        )

    print()
    print("Saved to:")
    print(save_path)


if __name__ == "__main__":
    main()