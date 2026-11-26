# RQ2 final verification (makeup-RQ2)

Generator: `makeup-RQ2/run_makeup_rq2.py`.

## Holiday mask and S9/S13

- Calendar: `shared/data/holiday_calendar.parquet` (`role ∈ {mainland, macau}`)
- Mask rows: **34,776** = 24,948 (reopening) + 9,828 (stress)
- Holiday day counts: reopening **1053**; stress **405** (do not use the 702-day lightweight calendar)
- S9 holiday `n_days`/`MAE` replay max abs diff = `4.547e-13`
- S13 holiday 24-row `R_origin_mean_vs_ETS` max `|ΔR|` = `1.110e-16`
- Dual aggregation: `pooled_day` (S9) and `task_then_market_equal` (score holidays inside each task, then equal-weight markets; missing holiday tasks are not zero-filled)
- Bootstrap: block length 3, B = 4000, seed `20261116`; indices saved under `raw/`

Full holiday R table: `tables/S9_S13_holiday_reconciliation.csv`.

## S10 WIS (50 rows)

- Formula: `WIS₂ = (0.5|y−q50| + 0.10·IS80 + 0.25·IS50) / 2.5`; `aggregation=pooled_day`
- Rows: **50** = 2 periods × 5 models × {overall, official_holiday, non_holiday, Week1, Week4}
- Matches original S10 on `n_days` / `coverage80` / `IS80` (max abs 0)
- Pinball cross-check: `max|WIS₂ − 0.4∑Pinball| = 2.910e-11`

Table: `tables/table_S10_wis_complete_50rows.csv`.

## Stress-period quantile crossings

| model | n_days | adjacent crossings | share |
|---|---:|---:|---:|
| chronos_2 | 9828 | 13 | 0.00132 |
| moirai | 9828 | 307 | 0.03124 |
| granite_patchtst_fm | 9828 | 0 | 0 |
| timesfm_3 | 9828 | 0 | 0 |
| lag_llama | 9828 | 0 | 0 |

Detail: `logs/quantile_crossing_audit.md`.

## S12 extended inputs

| model | MAE univariate | MAE extended | pct_change_MAE | MASE extended |
|---|---:|---:|---:|---:|
| timesfm_3 | 565.434 | 495.819 | −12.31% | 1.077 |
| chronos_2 | 568.037 | 508.863 | −10.42% | 1.129 |
| moirai | 671.034 | 833.430 | +24.20% | 1.562 |
| LL01 vs univariate LightGBM | 885.648 | 631.233 | −28.73% | 1.260 |

## Explicitly not run

| Item | Status |
|---|---|
| S11 Hong Kong zero_days / positive_days split | `not_run` |
| Macau H-2021 truncation control | `not_run` |
| Foundation calendar-only ablation | `not_run` (only calendar+related / cross-learning modes exist) |

Figures 5–10: `makeup-RQ2/figures/` (DejaVu Sans, despine, 300 DPI PNG + PDF).
