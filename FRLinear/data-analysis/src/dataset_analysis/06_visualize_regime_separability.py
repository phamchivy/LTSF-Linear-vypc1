"""
06_visualize_regime_separability.py

Đọc regime_cache/{ETTh1,ETTh2,ETTm1,ETTm2}.json (đầu ra của 05_generate_regime.py)
và vẽ scatter 2D (trend_score đã normalize, seasonal_score đã normalize) cho từng
dataset, tô màu theo regime được gán (trend / seasonal / mixed), kèm centroid cluster.

Mục đích: minh hoạ trực quan vì sao FRLinear (KMeans k=3 cố định) hoạt động tốt
trên dataset có cấu trúc tách biệt rõ (VD: ETTm1) nhưng kém hiệu quả trên dataset
mà các kênh co cụm gần nhau (VD: ETTh2, ETTm2).

Output: regime_separability.pdf (và .png để xem nhanh), sẵn sàng chèn vào
IEEE two-column paper dưới dạng \begin{figure*}.

Cách chạy:
    python 06_visualize_regime_separability.py

Yêu cầu:
    pip install matplotlib numpy --break-system-packages
    (4 file JSON đặt tại REGIME_CACHE_DIR, hoặc sửa đường dẫn bên dưới)
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# =============================================================
# CẤU HÌNH — chỉnh đường dẫn cho khớp máy của bạn
# =============================================================
REGIME_CACHE_DIR = Path(__file__).resolve().parents[3] / "regime_cache"   # nơi chứa ETTh1.json, ETTh2.json, ...
OUTPUT_DIR = Path("./output_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]

REGIME_COLORS = {
    "trend":    "#1f77b4",   # xanh dương
    "seasonal": "#d62728",   # đỏ
    "mixed":    "#2ca02c",   # xanh lá
}
REGIME_MARKERS = {
    "trend":    "o",
    "seasonal": "^",
    "mixed":    "s",
}
REGIME_LABELS = {
    "trend":    "Trend",
    "seasonal": "Seasonal",
    "mixed":    "Mixed",
}

# =============================================================
# STYLE — chuẩn IEEE (Times-like serif, cỡ chữ vừa đủ khi thu nhỏ vào cột)
# =============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.35,
    "pdf.fonttype": 42,   # embed font dạng Type 42, tránh lỗi font khi nộp IEEE
    "ps.fonttype": 42,
})


def min_max_normalize(values: np.ndarray) -> np.ndarray:
    """Khớp đúng công thức normalize trong 05_generate_regime.py (cluster_regimes)."""
    vmin, vmax = values.min(), values.max()
    if vmax - vmin < 1e-12:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def load_regime_json(dataset_name: str) -> dict:
    path = REGIME_CACHE_DIR / f"{dataset_name}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_arrays(regime_json: dict):
    """Trả về channel names, raw trend/season scores, regime label mỗi kênh."""
    names, trend_raw, season_raw, labels = [], [], [], []
    for col, info in regime_json["detail"].items():
        names.append(col)
        trend_raw.append(info["trend_score"])
        season_raw.append(info["seasonal_score"])
        labels.append(info["regime"])
    return (
        names,
        np.array(trend_raw, dtype=float),
        np.array(season_raw, dtype=float),
        labels,
    )


def plot_dataset(ax, dataset_name: str):
    regime_json = load_regime_json(dataset_name)
    names, trend_raw, season_raw, labels = extract_arrays(regime_json)

    trend_norm = min_max_normalize(trend_raw)
    season_norm = min_max_normalize(season_raw)

    # --- scatter từng điểm theo regime ---
    for regime in ["trend", "seasonal", "mixed"]:
        mask = [l == regime for l in labels]
        if not any(mask):
            continue
        ax.scatter(
            trend_norm[mask],
            season_norm[mask],
            c=REGIME_COLORS[regime],
            marker=REGIME_MARKERS[regime],
            s=55,
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
            label=REGIME_LABELS[regime],
        )

    # --- centroid (không gian normalize, lấy từ _debug_clustering) ---
    centroids = np.array(regime_json["_debug_clustering"]["centroids"])
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="black",
        marker="x",
        s=70,
        linewidths=1.6,
        zorder=4,
        label="Centroid",
    )

    # --- annotate tên kênh, tránh đè lên điểm ---
    for i, name in enumerate(names):
        ax.annotate(
            name,
            (trend_norm[i], season_norm[i]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=9,        # tăng từ 6.5 -> 9
            color="#333333",
            zorder=5,
        )

    ax.set_title(dataset_name, fontweight="bold", pad=4)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.grid(True, linestyle="--", zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def main():
    fig, axes = plt.subplots(
        4, 1,
        figsize=(3.45, 10.5),   # ~ chiều rộng 2 cột IEEE (7.16in), tỉ lệ vuông vức
        dpi=300,
        sharex=True,
        sharey=True,
    )

    for ax, dataset_name in zip(axes.flat, DATASETS):
        plot_dataset(ax, dataset_name)

    # trục chung
    fig.text(0.5, 0.008, r"Normalized trend score $\tilde{s}^{\mathrm{trend}}$", ha="center")
    fig.text(0.0, 0.5, r"Normalized seasonal score $\tilde{s}^{\mathrm{season}}$",
              va="center", rotation="vertical")

    # legend chung, đặt phía trên, tránh lặp lại 4 lần
    handles, labels_ = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels_,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
    )

    fig.tight_layout(rect=[0.09, 0.02, 1, 0.94])

    out_pdf = OUTPUT_DIR / "regime_separability.pdf"
    out_png = OUTPUT_DIR / "regime_separability.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()