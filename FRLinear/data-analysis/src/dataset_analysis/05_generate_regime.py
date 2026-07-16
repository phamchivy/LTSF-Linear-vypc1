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

# ============================================================
# FIX #1: Sampling-rate-aware period detection
# ============================================================
# Bug gốc: idx24/idx12/idx8 dùng fftfreq(N, d=1) rồi tìm tần số
# gần 1/24, 1/12, 1/8 -- điều này ngầm giả định 1 sample = 1 giờ.
# Với ETTh1/ETTh2 (sample = 1 giờ) thì đúng, nhưng với ETTm1/ETTm2
# (sample = 15 phút, tức 4 sample/giờ) thì "period 24 sample" chỉ
# là 6 giờ, không phải 1 ngày -> season_score đo sai chu kỳ.
#
# Sửa: quy đổi chu kỳ giờ mong muốn (24h/12h/8h) sang số sample
# tương ứng theo tần suất lấy mẫu thực tế của dataset.

def infer_samples_per_hour(dataset_filename: str) -> int:
    """
    Suy ra số sample/giờ dựa trên tên file dataset.
    ETTh* -> lấy mẫu theo giờ -> 1 sample/giờ
    ETTm* -> lấy mẫu 15 phút -> 4 sample/giờ

    Nếu dùng dataset khác không theo quy ước ETT, hãy chỉnh lại
    hàm này hoặc truyền samples_per_hour thủ công.
    """
    name = Path(dataset_filename).stem.lower()

    if "ettm" in name:
        return 4
    elif "etth" in name:
        return 1
    else:
        print(
            f"[WARNING] Không nhận diện được tần suất lấy mẫu từ "
            f"tên file '{dataset_filename}'. Mặc định dùng 1 sample/giờ. "
            f"Hãy kiểm tra lại season_score có ý nghĩa đúng không."
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

    # --- FIX #1 áp dụng ở đây ---
    # Quy đổi 24h / 12h / 8h sang số sample thực tế, thay vì hardcode
    # số sample = số giờ.
    period_24h_samples = 24 * samples_per_hour
    period_12h_samples = 12 * samples_per_hour
    period_8h_samples = 8 * samples_per_hour

    idx24 = np.argmin(
        np.abs(xf - 1 / period_24h_samples)
    )

    idx12 = np.argmin(
        np.abs(xf - 1 / period_12h_samples)
    )

    idx8 = np.argmin(
        np.abs(xf - 1 / period_8h_samples)
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


# ============================================================
# FIX #2: Minimum group size constraint
# ============================================================
# Bug gốc: seasonal/mixed có thể co lại còn 1 kênh (thấy ở
# ETTh2.json: mixed=[2], ETTm2.json: seasonal=[6]). Khi đó
# individual=False khiến 1 nn.Linear "share" cho đúng 1 kênh,
# nghĩa là mất hoàn toàn lợi ích regularization của weight-sharing
# -> tương đương individual=True cho riêng kênh đó, dễ overfit.
#
# Sửa: sau khi gán nhãn theo quantile, nếu 1 nhóm nhỏ hơn
# MIN_GROUP_SIZE, "mượn" kênh biên giới (gần ngưỡng nhất) từ
# nhóm còn lại để đạt kích thước tối thiểu.

MIN_GROUP_SIZE = 2


def enforce_min_group_size(seasonal_idx, mixed_idx, season_scores, min_size=MIN_GROUP_SIZE):
    seasonal_idx = list(seasonal_idx)
    mixed_idx = list(mixed_idx)

    # Case 1: seasonal quá nhỏ -> kéo từ mixed kênh có season_score
    # cao nhất (tức "gần seasonal nhất" trong số các kênh mixed).
    while len(seasonal_idx) < min_size and len(mixed_idx) > 0:
        best = max(mixed_idx, key=lambda i: season_scores[i])
        mixed_idx.remove(best)
        seasonal_idx.append(best)

    # Case 2: mixed quá nhỏ -> trả lại từ seasonal kênh có
    # season_score thấp nhất (tức "yếu seasonal nhất" trong nhóm đó),
    # miễn là không làm seasonal tụt xuống dưới min_size.
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
    # Trend
    # (Giữ nguyên logic winner-take-all argmax như bản gốc.
    #  Lưu ý: đây vẫn là giả định cấu trúc chưa được kiểm định --
    #  xem ghi chú "elbow gap" đã bàn riêng, không nằm trong phạm vi
    #  2 fix lần này.)
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

    # --- FIX #2 áp dụng ở đây ---
    seasonal_idx, mixed_idx = enforce_min_group_size(
        seasonal_idx,
        mixed_idx,
        season_scores,
        min_size=MIN_GROUP_SIZE
    )

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
    print("Saved to:")
    print(save_path)


if __name__ == "__main__":
    main()