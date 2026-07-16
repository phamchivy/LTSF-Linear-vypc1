import os
import json
import numpy as np
import pandas as pd

from pathlib import Path
from statsmodels.tsa.stattools import acf
from scipy.fft import fft, fftfreq
from sklearn.cluster import KMeans

from src.config import DATASET_DIR, DATASET

SAVE_DIR = "../regime_cache"
os.makedirs(SAVE_DIR, exist_ok=True)

MIN_GROUP_SIZE = 2
RANDOM_STATE = 42


# ============================================================
# FIX #1 (giữ nguyên từ lần trước): Sampling-rate-aware period
# ============================================================

def infer_samples_per_hour(dataset_filename: str) -> int:
    name = Path(dataset_filename).stem.lower()

    if "ettm" in name:
        return 4
    elif "etth" in name:
        return 1
    else:
        print(
            f"[WARNING] Không nhận diện được tần suất lấy mẫu từ "
            f"tên file '{dataset_filename}'. Mặc định dùng 1 sample/giờ."
        )
        return 1


def extract_feature_vector(x, samples_per_hour: int):
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

    period_24h_samples = 24 * samples_per_hour
    period_12h_samples = 12 * samples_per_hour
    period_8h_samples = 8 * samples_per_hour

    idx24 = np.argmin(np.abs(xf - 1 / period_24h_samples))
    idx12 = np.argmin(np.abs(xf - 1 / period_12h_samples))
    idx8 = np.argmin(np.abs(xf - 1 / period_8h_samples))

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


# ============================================================
# FIX #2 (mới): Phân cụm 2D thay cho ngưỡng cứng (argmax + quantile)
# ============================================================
# Bug/giới hạn gốc:
#   - trend: argmax 1 chiều trên trend_score -> luôn ép ra đúng 1 kênh
#     bất kể có "elbow" (khoảng cách rõ) hay không.
#   - seasonal: quantile 0.70 cố định trên season_score -> luôn cắt ra
#     "top 30%" bất kể phân bố có cụm tự nhiên hay không.
#
# Sửa: chuẩn hoá (trend_score, season_score) rồi chạy KMeans (k=3)
# trên không gian 2 chiều. Cụm được gán nhãn "trend"/"seasonal"/"mixed"
# dựa trên đặc điểm centroid (không hardcode index kênh nào thuộc
# nhóm nào), nên ranh giới hoàn toàn theo cấu trúc thực của dữ liệu
# thay vì một ngưỡng số cố định áp cho mọi dataset.

def cluster_regimes(trend_scores, season_scores, min_size=MIN_GROUP_SIZE):
    C = len(trend_scores)

    # Chuẩn hoá về [0, 1] theo từng chiều để KMeans không bị chi phối
    # bởi thang đo khác nhau giữa trend_score (~0-1) và season_score
    # (thường rất nhỏ, ~1e-4 đến 1e-1).
    def normalize(v):
        v = np.asarray(v, dtype=float)
        vmin, vmax = v.min(), v.max()
        if vmax - vmin < 1e-12:
            return np.zeros_like(v)
        return (v - vmin) / (vmax - vmin)

    trend_norm = normalize(trend_scores)
    season_norm = normalize(season_scores)

    X = np.stack([trend_norm, season_norm], axis=1)

    kmeans = KMeans(
        n_clusters=3,
        n_init=10,
        random_state=RANDOM_STATE
    )
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_  # shape [3, 2] -> (trend_norm, season_norm)

    # Gán nhãn regime cho từng cụm dựa trên đặc điểm centroid:
    # - "seasonal": centroid có season_norm cao nhất
    # - trong 2 cụm còn lại, "trend": centroid có trend_norm cao hơn
    # - cụm còn lại: "mixed"
    cluster_ids = [0, 1, 2]

    seasonal_cluster = max(cluster_ids, key=lambda c: centroids[c][1])
    remaining = [c for c in cluster_ids if c != seasonal_cluster]

    trend_cluster = max(remaining, key=lambda c: centroids[c][0])
    mixed_cluster = [c for c in remaining if c != trend_cluster][0]

    trend_idx = [i for i in range(C) if labels[i] == trend_cluster]
    seasonal_idx = [i for i in range(C) if labels[i] == seasonal_cluster]
    mixed_idx = [i for i in range(C) if labels[i] == mixed_cluster]

    # --- Đảm bảo trend không rỗng ---
    # KMeans luôn gán mỗi cụm >=1 điểm nếu dữ liệu không suy biến hoàn
    # toàn, nhưng vẫn phòng hờ trường hợp cạnh biên.
    if len(trend_idx) == 0:
        best = int(np.argmax(trend_scores))
        trend_idx = [best]
        if best in seasonal_idx:
            seasonal_idx.remove(best)
        if best in mixed_idx:
            mixed_idx.remove(best)

    # --- Ràng buộc kích thước tối thiểu cho seasonal/mixed ---
    # (áp dụng lại logic mượn kênh biên giới như bản trước, nhưng dựa
    # trên season_score gốc để chọn kênh "gần ranh giới nhất")
    seasonal_idx, mixed_idx = enforce_min_group_size(
        seasonal_idx, mixed_idx, season_scores, min_size=min_size
    )

    debug_info = {
        "centroids": centroids.tolist(),
        "cluster_label_map": {
            "trend_cluster": int(trend_cluster),
            "seasonal_cluster": int(seasonal_cluster),
            "mixed_cluster": int(mixed_cluster),
        }
    }

    return trend_idx, seasonal_idx, mixed_idx, debug_info


def enforce_min_group_size(seasonal_idx, mixed_idx, season_scores, min_size=MIN_GROUP_SIZE):
    seasonal_idx = list(seasonal_idx)
    mixed_idx = list(mixed_idx)

    while len(seasonal_idx) < min_size and len(mixed_idx) > 0:
        best = max(mixed_idx, key=lambda i: season_scores[i])
        mixed_idx.remove(best)
        seasonal_idx.append(best)

    while len(mixed_idx) < min_size and len(seasonal_idx) > min_size:
        worst = min(seasonal_idx, key=lambda i: season_scores[i])
        seasonal_idx.remove(worst)
        mixed_idx.append(worst)

    return seasonal_idx, mixed_idx


def main():

    DATA_PATH = DATASET_DIR / DATASET
    dataset_name = Path(DATASET).stem

    samples_per_hour = infer_samples_per_hour(DATASET)
    print(f"Dataset: {DATASET} -> samples_per_hour = {samples_per_hour}")

    df = pd.read_csv(DATA_PATH)

    feature_cols = list(
        df.columns[1:]
    )

    feature_vectors = []

    for col in feature_cols:
        vec = extract_feature_vector(
            df[col].values,
            samples_per_hour=samples_per_hour
        )
        feature_vectors.append(vec)

    feature_vectors = np.stack(
        feature_vectors
    )

    trend_scores = feature_vectors[:, 0]
    season_scores = feature_vectors[:, 4]

    ############################
    # Regime clustering (2D, không ngưỡng cứng)
    ############################

    trend_idx, seasonal_idx, mixed_idx, debug_info = cluster_regimes(
        trend_scores,
        season_scores,
        min_size=MIN_GROUP_SIZE
    )

    result = {
        "trend": sorted(trend_idx),
        "seasonal": sorted(seasonal_idx),
        "mixed": sorted(mixed_idx),
        "detail": {},
        "_debug_clustering": debug_info
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
            f" trend={trend_scores[i]:.4f}"
            f" seasonal={season_scores[i]:.6f}"
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
    print("Cluster centroids (trend_norm, season_norm):")
    print(debug_info["centroids"])
    print()
    print("Saved to:")
    print(save_path)


if __name__ == "__main__":
    main()