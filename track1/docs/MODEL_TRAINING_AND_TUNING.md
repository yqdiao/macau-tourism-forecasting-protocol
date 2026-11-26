# Track 1: training, tuning, and frozen candidates

This note covers models that **fit or train on target history** or that require **B-group selection**. The five zero-shot foundation models (Granite, Chronos-2, TimesFM 3.0, Moirai, Lag-Llama) do not train on the Macau tasks; they use registered source and checkpoint revisions only. For RQ tables, see the repository root README and `RQ1/`–`RQ3/`.

Implementation:

- Baselines and tabular models: `track1/code/models.py`
- PatchTST / Informer: `track1/code/foundation_models/adapters/{patchtst,informer,_trainable}.py`
- Lightweight families: `track1/code/light_models.py`
- Search and freeze files: `track1/configs/{baseline_search_spaces,foundation_search_spaces,light_model_plan,light_model_round2,frozen_*}.yaml`
- Quantile levels for foundation adapters: `FoundationAdapter.quantile_levels()` in `contracts.py` (used by RQ1 five-quantile runs)

## Coverage

| Family | Model | Frozen candidate | Training | Seeds |
|---|---|---|---|---|
| Statistical | Seasonal naive | SN01 | None | 42 |
| Statistical | ETS | ETS03 | Per-task Holt–Winters | 42 |
| Statistical | SARIMA | SAR03 | Per-task SARIMAX | 42 |
| Tabular direct | Ridge / OLS / Lasso / Elastic Net / SVR / LightGBM / GBDT / RF / RF-800 | RID02, LIN01, LAS03, EN03, SVR02, LGB01, GB01, RF03, PRF01 | 28 direct regressors per task | 42 |
| Small network | Transformer | TR03 | Per-task encoder from scratch | B: 11/29/42; outer: 42 |
| Trainable foundation | PatchTST | PT06 | Per-task supervised training | 11, 29, 42 |
| Trainable foundation | Informer | INF07 | Per-task supervised training | 11, 29, 42 |
| Lightweight global | Global LightGBM | GL03 | One model per origin, 27 markets pooled | 42 |
| Lightweight global | NLinear | NL01 | Shared linear map per origin | 11, 29, 42 |
| Ablations (no retune) | GL-noCal / GL-local | GL03 params | Drop calendar / per-market | 42 |
| Post-hoc round 2 | Per-market LightGBM + calendar | LL01 | One model per market; exploratory | 42 |

DLinear entered B-group as NL03 only; it was not frozen and is not run on the outer set.

## Shared protocol

Each task forecasts 28 days from origin \(o\) using history available by \(o\) (`history=all` by default). Predictors never read `targets.parquet`.

| Split | Role | Macau tasks |
|---|---|---|
| A01–A13 | Pandemic stress; report only | \(27\times 13=351\) |
| B01–B12 | **Only** selection window | \(27\times 12=324\) |
| O01–O33 | Outer backtest | \(27\times 33=891\) (baselines/trainable foundation also score 2 Hong Kong series → 957) |

Primary metric: task MASE(7) with seasonal-naive scale of period 7 computed on history up to the origin. Aggregation: average seeds → average folds within market → equal weight over 27 markets. Ties: lower mean fit seconds, then smaller candidate id. **Outer, A-group, history-180/365, and Hong Kong scores must not rewrite a frozen candidate.**

## Selected frozen results (Macau equal-weight MASE(7))

| Model | Candidate | B | Outer (`history=all`) | Notes |
|---|---|---:|---:|---|
| ETS | ETS03 | 1.047 | 1.639 | Best baseline on B; cost contrast often Granite vs ETS |
| NLinear | NL01 | 1.073 | 1.627 | Lightweight champion under B rules |
| Global LightGBM | GL03 | 1.115 | 1.338 | Calendar features; pooled markets |
| SVR | SVR02 | 1.154 | 1.446 | Best frozen baseline on outer among original tabular set |
| PatchTST | PT06 | 1.364 | 1.590 | Expensive relative to zero-shot foundation |
| Informer | INF07 | 2.031 | 1.672 | Expensive; weak B score |
| Per-market LGB + calendar | LL01 | 1.187 | 1.260 | Exploratory; does not replace NL01 |

Zero-shot outer scores with market-only history (for contrast): Granite 1.094, TimesFM 3.0 1.117, Chronos-2 1.152. After calendar + cross-market inputs (2026-10-11): Chronos-2 1.129, TimesFM 1.077, Moirai 1.562. Granite and Lag-Llama checkpoints cannot take those extra channels.

## Lightweight runners

```bash
# Round 1 plan (GL / NL search and frozen outer)
python track1/scripts/run_light_models.py --split outer_backtest --families global_lightgbm,global_linear

# Round 2 exploratory per-market LightGBM
python track1/scripts/run_light_models.py \
  --split outer_backtest \
  --families local_lightgbm \
  --plan track1/configs/light_model_round2.yaml

# History slices used by RQ2
python track1/scripts/run_light_models.py --split outer_backtest --history 180 ...
```

Output layout: `track1/results/light_models/<split>[/<history>]/<model>/<candidate>/seed<seed>/<fold>_<origin>/` with `predictions.parquet`, `timings.jsonl`, and `run_manifest.json` (`target_artifact_accessed=false`).

## Five-quantile foundation runs (RQ1 / RQ2 intervals)

Registry: `RQ1/raw/registry_five_quantiles.yaml` sets `quantile_levels: [0.1, 0.25, 0.5, 0.75, 0.9]`. Runner: `RQ1/run_five_quantiles.py` → `run_foundation_backtest.py`. Adapters read levels through `quantile_levels()`. TimesFM interpolates non-decile levels between decile heads and records them under `detail.interpolated_quantiles`. Lag-Llama seeds torch before sampling.

## Code map

| Topic | Location |
|---|---|
| Naive / ETS / SARIMA / tabular / Transformer | `track1/code/models.py` |
| Trainable windows and validation | `foundation_models/adapters/_trainable.py` |
| PatchTST / Informer | `adapters/patchtst.py`, `adapters/informer.py` |
| Calendar and GL features | `light_models.Calendar`, `light_models._gl_features` |
| NLinear / DLinear | `light_models._linear_module` |
| Baseline search (33 candidates) | `configs/baseline_search_spaces.yaml` |
| PT / INF search (8 each) | `configs/foundation_search_spaces.yaml` |
| Lightweight plans | `light_model_plan.yaml`, `light_model_round2.yaml` |

Document date of the underlying experiment archive: 2026-10-11. RQ1–RQ3 sealed tables are authoritative for manuscript numbers when they differ from earlier working notes.
