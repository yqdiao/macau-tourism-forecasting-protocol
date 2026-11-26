# Macau tourism forecasting protocol

Companion repository for **Are large models more cost-effective for tourism demand forecasting? Evidence from different demand periods in Macau**.

The paper answers three research questions. Engineering directories `track1/`, `track2/`, and `shared/` hold the shared calendar, isolated inputs, and single-model backtests. Directories `RQ1/`–`RQ3/` and `makeup-RQ2/` rebuild the sealed manuscript tables and figures from those upstream predictions.

## Research questions

| RQ | Question | Computation layer | Main sealed outputs |
|---|---|---|---|
| **RQ1** | Which forecasting configurations offer useful accuracy–resource trade-offs? | `RQ1/`, `RQ1_outputs/` | Tables 3–5, Pareto / cost figures, five-quantile raw runs |
| **RQ2** | How do comparisons change across demand regimes, markets, horizons, and history length? | `RQ2/`, `RQ2_outputs/`, `makeup-RQ2/` | Tables S1–S13, Figures 5–10 |
| **RQ3** | Do combination, LLM judgment, and fallback justify their extra cost? | `RQ3/`, `RQ3_outputs/` | Tables S14–S18, Figures 11–12, DeepSeek decision caches |

Reference designs: [Wu et al. (2025)](https://doi.org/10.1016/j.tourman.2024.105119) and [Wu et al. (2027)](https://doi.org/10.1016/j.tourman.2026.105479). This repository does not claim to reproduce every design in either paper.

Narrative map: [docs/manuscript_structure.md](docs/manuscript_structure.md). Release boundaries: [docs/reproducibility_status.md](docs/reproducibility_status.md). Audited companion inventory: [paper/README.md](paper/README.md). Redundancy notes: [docs/repo_hygiene.md](docs/repo_hygiene.md).

## Start here by RQ

1. **RQ1 accuracy and cost** — builders in `RQ1/`; cite sealed CSVs in `RQ1_outputs/tables/`. Figures live in `RQ1/figures/` (not copied into `RQ1_outputs/`).
2. **RQ2 mechanisms and boundaries** — builders in `RQ2/`; panels in `RQ2_outputs/`. Holiday dual aggregation, complete S10 WIS, S12, and final Figures 5–10 are in `makeup-RQ2/`. Do not overwrite `RQ2_outputs/` with makeup tables; RQ3 still reads `RQ2_outputs/`.
3. **RQ3 combination and LLM judgment** — builders in `RQ3/`. Cite `RQ3_outputs/cache/*_deepseek.*` only. Earlier proxy files `llm_judgments.jsonl` / `all_decisions.parquet` are not for the manuscript text. Writing notes: [RQ3_outputs/README.md](RQ3_outputs/README.md).

```bash
python -m pip install -r shared/requirements.txt
python RQ1/build_rq1.py
python RQ2/run_all.sh
python makeup-RQ2/run_makeup_rq2.py
python RQ3/run_all.py
python RQ3/fill_gaps.py
# DeepSeek requires RQ3/.env (not committed)
python RQ3/run_deepseek_rq3.py
```

Rebuilding RQ tables needs local prediction stores that Git ignores: `track1/results/foundation_outer_default/`, `outer_server_all/`, `light_models/`, `sensitivity/pandemic_stress/`, `zeroshot_matched_inputs/`, `compute_benchmark/benchmark.jsonl`, plus `track1/evaluation/outer_main_table/` and `volume_breakeven/`. The Qwen3-4B delivery log used by RQ1 is `RQ1/raw/qwen3_4b_D-S1_deliveries.jsonl`.

## Model families and training

Training, B-group selection, and frozen candidates are documented in [track1/docs/MODEL_TRAINING_AND_TUNING.md](track1/docs/MODEL_TRAINING_AND_TUNING.md). Short map:

| Family | Models | What is trained / tuned | Freeze / selection |
|---|---|---|---|
| Zero-shot foundation | Granite, Chronos-2, TimesFM 3.0, Moirai, Lag-Llama | No target-task training; registered revisions only | Default / smoke-accepted configs; five-quantile levels via `runtime.extra.quantile_levels` |
| Supervised foundation | PatchTST (PT06), Informer (INF07) | Per-task training from scratch; B-group grid | `track1/configs/frozen_foundation_configs.yaml` |
| Statistical / tabular baselines | Seasonal naive, ETS, SARIMA, Ridge, OLS, Lasso, Elastic Net, SVR, LightGBM, GBDT, RF, RF-800, small Transformer | Per-task fit or train; B01–B12 search | `track1/configs/frozen_baseline_configs.yaml` (2026-09-26) |
| Lightweight global | Global LightGBM (GL03), NLinear (NL01) | One model per origin across 27 markets (or shared linear map) | `track1/configs/frozen_light_configs.yaml`; plan in `light_model_plan.yaml` |
| Lightweight exploratory | Per-market LightGBM + calendar (LL01) | Registered after outer results; does not replace NL01 | `light_model_round2.yaml` (`registered_after_outer_results`) |

Selection rule (all trainable families): task MASE(7) → average seeds → average folds within market → equal-weight over 27 Macau markets. Outer, stress, history-180/365, and Hong Kong scores must never rewrite a frozen candidate.

Adapters live under `track1/code/foundation_models/adapters/`. Baseline and tabular code: `track1/code/models.py`. Lightweight models: `track1/code/light_models.py`. Runners: `track1/scripts/run_light_models.py`, `track1/code/run_foundation_backtest.py`, `track1/code/run_backtest.py`.

Track 2 selected members (Granite, Chronos-2, TimesFM 3.0) are frozen in `track2/configs/selected_track2_models.yaml`. Protocol: [track2/docs/ENSEMBLE_RESIDUAL_AGENT_PROTOCOL.md](track2/docs/ENSEMBLE_RESIDUAL_AGENT_PROTOCOL.md).

## Evaluation unit

One task = one destination × source market × forecast origin × 28-day path. Primary Macau outer set: **27 markets × 33 origins = 891 tasks**. Pandemic stress: 351 Macau tasks. B-group selection: 324 Macau tasks. Hong Kong is a separate extension and is not pooled into the Macau primary score.

Predictors may read only history available by the origin (`shared/data/task_inputs.parquet` and origin shards). The evaluator alone reads `shared/data/targets.parquet` after forecasts are sealed. Common calendar: `shared/configs/protocol.yaml`.

## Repository layout

```text
.
├── docs/                 # Manuscript map and release boundaries (English)
├── paper/                # Audited companion tables, figures, inventory
├── shared/               # Protocol YAML, holiday calendar, inputs, targets
├── track1/               # Single-model code, configs, adapters, GPU guides
├── track2/               # Ensemble / agent protocol and stage code
├── RQ1/ , RQ1_outputs/   # Section 4.1 builders and sealed tables
├── RQ2/ , RQ2_outputs/   # Section 4.2 builders and sealed panels
├── makeup-RQ2/           # Final RQ2 holiday, WIS, S12, Figures 5–10
└── RQ3/ , RQ3_outputs/   # Section 4.3 builders, tables, DeepSeek caches
```

`extensions/lightweight_legacy/` archives an earlier packaging of lightweight code; prefer `track1/code/light_models.py` and the RQ builders for new work.

## CPU smoke (does not rerun the paper)

```bash
python track1/code/audit_protocol_artifacts.py
python track1/code/run_backtest.py \
  --inputs shared/data/task_inputs.parquet \
  --destinations Macau \
  --markets CN_Chinese_mainland \
  --origin-date 2023-12-31 \
  --history all \
  --splits outer_backtest \
  --models seasonal_naive,ets,sarima \
  --output track1/results/smoke_isolated
python track1/code/evaluate_predictions.py \
  --predictions track1/results/smoke_isolated/predictions.parquet \
  --targets shared/data/targets.parquet \
  --output track1/evaluation/smoke_isolated
```

GPU guides: [track1/docs/FOUNDATION_MODELS_GPU.md](track1/docs/FOUNDATION_MODELS_GPU.md), [track1/docs/GPU_SEQUENTIAL_RUN_CHECKLIST.md](track1/docs/GPU_SEQUENTIAL_RUN_CHECKLIST.md).

## Data, credentials, and licenses

Do not commit API keys, weights, container layers, or bulk `track1/results/`. `RQ3/.env` and `track2/.env` are gitignored. Respect Macau/Hong Kong data terms and each model’s code and weight license. Cite the manuscript and the exact repository commit when using this material.
