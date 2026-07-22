# FRLinear: Feature-Regime-Aware Linear Forecasting with Adaptive Decomposition for Long-Term Time Series Prediction

> **Anonymous repository for double-blind peer review.**
> This repository is submitted as supplementary code for a paper currently under double-blind review. All author-identifying information has been removed. Please do not attempt to identify the authors from this repository.

---

## Overview

**FRLinear** is a lightweight linear forecasting framework for multivariate long-term time series forecasting (LTSF). Instead of applying a single decomposition operator and a shared forecasting backbone to all variables, FRLinear explicitly models channel-wise temporal heterogeneity by:

1. **Automated feature regime discovery** — categorizing each variable into a *trend-dominant*, *seasonal-dominant*, or *mixed* regime using autocorrelation (ACF) and frequency-domain (FFT) descriptors, clustered via $k$-means.
2. **Regime-specific linear experts** — each regime is decomposed with an adaptive multi-kernel moving-average module and forecast independently, following a DLinear-style trend/seasonal linear projection.
3. **Residual regime gate** — a lightweight calibration module that rescales each regime's prediction using global mean/std statistics, bounded by a scaled $\tanh$ function to preserve stability.

FRLinear is evaluated on the four ETT (Electricity Transformer Temperature) benchmarks against strong linear and Transformer-based baselines (NLinear, DLinear, Informer, Autoformer, FEDformer). Results are reported honestly as **dataset-dependent**: FRLinear achieves the best average performance on ETTm1, but trails baselines on ETTh1 (at longer horizons), ETTh2, and ETTm2 — a pattern we analyze in detail via regime separability and ablation studies.

---

## Repository Structure

```
FRLinear
├── data-analysis/
│   └── src/
│       └── dataset_analysis/
│           ├── 05_generate_regime.py          # Offline feature regime discovery 
│           └── 06_visualize_regime_separability.py  # Scatter plot of channel descriptors 
├── models/
│   └── FRLinear.py                            # Model definition 
├── regime_cache/                              # Cached regime assignments per dataset (JSON)
│   ├── ETTh1.json
│   ├── ETTh2.json
│   ├── ETTm1.json
│   └── ETTm2.json
└── README.md

FRLinear_output/
FRLinear_scripts/
```

---

## Requirements

```bash
pip install -r requirements
```

---

## 1. Generate Feature Regimes

Regime assignment is computed **offline**, once per dataset, and cached to disk:

```bash
cd data-analysis
python -m src.dataset_analysis.05_generate_regime
```

This produces `regime_cache/{ETTh1,ETTh2,ETTm1,ETTm2}.json`, containing:
- `trend`, `seasonal`, `mixed`: channel indices per regime
- `detail`: per-channel `trend_score`, `seasonal_score`, assigned `regime`
- `_debug_clustering`: $k$-means cluster centroids (normalized space)

---

## 2. Visualize Regime Separability (optional)

```bash
python -m src.dataset_analysis.06_visualize_regime_separability
```

Produces a scatter plot of normalized $(\tilde{s}^{\text{trend}}, \tilde{s}^{\text{season}})$ descriptors per channel, colored by discovered regime, with cluster centroids — used to illustrate why regime-aware routing helps on some datasets (e.g., ETTm1) but not others (e.g., ETTh2).

---

## 3. Train / Evaluate FRLinear

Running scripts in FRLinear_scripts/

---

## Default Hyperparameters

| Hyperparameter | Value |
|---|---|
| Look-back length $L$ | 336 |
| Prediction horizons $H$ | {96, 192, 336, 720} |
| Optimizer | Adam |
| Learning rate | $5 \times 10^{-3}$ |
| Batch size | 32 |
| Early stopping patience | 3 |
| Max training epochs | 20 |
| Number of runs (`itr`) | 5 |
| Trend kernels | {21, 31, 41} |
| Seasonal kernels | {5, 9, 13} |
| Mixed kernels | {11, 21, 31} |
| Gate scaling coefficient $\delta$ | 0.005 |

---

## Datasets

The four ETT benchmarks are publicly available from the original DLinear / Informer repositories. Place the raw `.csv` files (`ETTh1.csv`, `ETTh2.csv`, `ETTm1.csv`, `ETTm2.csv`) under a local `dataset/` directory referenced by `--data_path`.

---

## Reproducibility Notes

- All reported numbers are averaged over 5 independent runs (`itr=5`) with different random seeds, except ablation variants (`itr=3`) due to time constraints.
- No test-set feedback was used to select hyperparameters or clustering configurations — all design choices (kernel sizes, gate scaling, $k=3$) were fixed a priori based on validation performance and domain reasoning, not test-set performance.
- Mixed results across datasets (FRLinear outperforms on ETTm1, underperforms on ETTh1/ETTh2/ETTm2) are reported as-is; see the paper's Discussion and Limitations sections for analysis.

---

## Citation

Citation details withheld during double-blind review. A full BibTeX entry will be added upon acceptance.

---

## License

License to be determined upon acceptance.