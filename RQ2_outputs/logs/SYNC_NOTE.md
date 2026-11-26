# RQ2_outputs sync note

**When:** 2026-11-17 (post `rq2-fill` `[fill done]` + stress EXIT:0)
**Inference ETA:** 0 — all related jobs finished.

## Synced into this folder
- `raw/five_quantile_runs/` — mirrored from `RQ1/raw/five_quantile_runs/` (outer_backtest O01–O33 + pandemic_stress A01–A13 × 5 foundation models; predictions + run_manifest + timings/node.log)
- `logs/stress_moirai.log`, `logs/stress_lag_llama.log` — copied from `RQ1/raw/extra_logs/`
- Tables S1–S13, figures 5–10, summary MD, panels, nan_* reports — already written in-place by `rq2-fill` / `fill_and_recompute` (no invent)

## S10 stress foundation models (verified)
chronos_2, granite_patchtst_fm, lag_llama, moirai, timesfm_3 — both splits × 5 subsets; S10 NaNs = 0
