# Macau tourism forecasting: accuracy, resource use, and adoption

Research companion repository for **Are large models more cost-effective for tourism demand forecasting? Evidence from different demand periods in Macau**.

The manuscript evaluates forecasting tools and additional system components through three questions:

1. Which forecasting configurations offer useful combinations of predictive accuracy and recorded computational resources?
2. How do these comparisons change with demand periods, source markets, forecast horizons, and available history?
3. Do forecast combination, language-model judgment, and fallback mechanisms deliver improvements that justify their additional resources?

The principal reference studies are [Wu et al. (2025), *How well can ChatGPT forecast tourism demand?*](https://doi.org/10.1016/j.tourman.2024.105119) and [Wu et al. (2027), *Multi-agent collaboration strategies for tourism demand forecasting*](https://doi.org/10.1016/j.tourman.2026.105479). Their forecasting and collaboration designs motivate the comparison; this repository does not claim to reproduce every design in either paper.

## Start here

- [Paper companion materials and evidence](paper/README.md)
- [Six-section manuscript structure](docs/manuscript_structure.md)
- [Release scope and reproducibility status](docs/reproducibility_status.md)
- [Track 1 implementation](track1/README.md)
- [Shared data and task boundaries](shared/README.md)
- [Track 2 protocol and relationship to later evidence](track2/README.md)

## Research-question computation layer

`RQ1/`, `RQ2/`, `RQ3/`, and `makeup-RQ2/` rebuild the manuscript tables and figures from sealed upstream predictions. `track1/`, `track2/`, and `shared/` stay the protocol: time splits, isolated inputs, and single-model backtests. The RQ scripts only read those upstream results.

| Path | Role |
|---|---|
| `RQ1/` | Section 4.1 accuracy, cost, and asymmetric loss. Build scripts write tables here. Figures live in `RQ1/figures/`. |
| `RQ1_outputs/` | Sealed RQ1 tables and raw metrics. RQ3 reads this copy. The CSVs match `RQ1/` today. |
| `RQ2/`, `RQ2_outputs/` | Section 4.2 mechanisms and boundaries. RQ3 reads `RQ2_outputs/`. |
| `makeup-RQ2/` | Final RQ2 holiday dual aggregation, S10 WIS, S12, and figures 5–10. Do not overwrite `RQ2_outputs/` with these tables. |
| `RQ3/`, `RQ3_outputs/` | Section 4.3 combination, selection, and cost. Cite `RQ3_outputs/cache/*_deepseek.*`. The earlier proxy files `llm_judgments.jsonl` and `all_decisions.parquet` are not for the manuscript text. |

Rebuilding the tables needs local prediction stores that Git does not carry: `track1/results/foundation_outer_default/`, `outer_server_all/`, `light_models/`, `sensitivity/pandemic_stress/`, `zeroshot_matched_inputs/`, `compute_benchmark/benchmark.jsonl`, plus `track1/evaluation/outer_main_table/` and `volume_breakeven/`. The Qwen3-4B delivery log used by RQ1 is `RQ1/raw/qwen3_4b_D-S1_deliveries.jsonl`.

```bash
python RQ1/build_rq1.py
python RQ2/run_all.sh
python makeup-RQ2/run_makeup_rq2.py
python RQ3/run_all.py
python RQ3/fill_gaps.py
# DeepSeek calls need RQ3/.env, which is not committed
python RQ3/run_deepseek_rq3.py
```

## Current release and historical implementation

This release updates the research narrative and documentation in English, and adds the audited paper companion materials. The existing `track1/` implementation and frozen baseline audit are retained. The `track1/` and `track2/` directory names are engineering identifiers; the paper is organized by research questions rather than by the order in which experiments were run.

**Code availability and evidence availability are separate.** The historical repository contains the Track 1 data pipeline, 13 baseline implementations, seven time-series adapters, container definitions, and baseline selection audit. The presence of a later result in `paper/` does not imply that its complete server execution code, checkpoints, raw predictions, or resource logs are included. Consult the companion inventory before attempting a full rerun.

The September protocol documents are retained as **archival design records**. Their pending checklists, proposed 20-model scope, and proposed operational extensions are not a completion statement for the current paper. They are not retrospective evidence that every later experiment was specified before its outcomes were inspected.

## Evaluation unit and information boundaries

A task is one destination/source-market series at one forecast origin with a 28-day forecast path. The main Macau evaluation comprises **27 markets, 33 origins, and 891 tasks**. The pandemic stress set contains 351 Macau tasks; the selection set contains 324 Macau tasks. The two Hong Kong series form a separate extension and are not pooled into the Macau primary score.

The 33 main windows score 2024-01-01 through 2026-07-12. The source observations extend through 2026-07-31, but the final 19 days are outside those main windows. The stress set consists of selected windows, not every day of the pandemic.

Predictors may read only history available by each origin:

- `shared/data/task_inputs.parquet`: historical inputs without future targets;
- `shared/data/splits/*/task_inputs.parquet`: inputs separated by evaluation phase;
- `shared/data/origin_shards/`: generated, physically isolated origin-level inputs for GPU containers.

The independent evaluator reads `shared/data/targets.parquet` after forecasts are sealed. Model containers must not mount the whole shared data directory. The common task calendar is in `shared/configs/protocol.yaml`.

Frozen target-task weights do not establish the absence of overlap with upstream pretraining data. Model revisions and retrospective evaluation boundaries remain part of the interpretation.

## Models and system roles

| Component | Role in the paper | Interpretation |
|---|---|---|
| Pretrained time-series models | Produce numerical forecast paths from historical observations | Frozen-weight inference is evaluated within each recorded information setting |
| Statistical, machine-learning, and lightweight configurations | Fit or train on available target history | Original frozen selections and later exploratory revisions are distinguished |
| Local Qwen3-4B-thinking | Generate numerical forecasts directly | Output quality, extreme predictions, and shared-sample sensitivity are retained |
| Local Qwen3-8B | Assign weights to existing candidate forecasts | Programmatic aggregation produces the final numerical forecasts |
| Deterministic and LLM coordination | Validate, route, and invoke frozen fallback rules | Reliability evidence is conditional on the documented fault construction |

Time-series models can be forecasting tools inside an agent workflow. Tool inclusion alone does not make each model a language-based agent. Independent role branches also do not imply concurrent GPU execution.

## Accuracy and resource interpretation

The current manuscript reports task-level **MASE(7), MAE, and mean task RMSE**, with task coverage and paired uncertainty where available. When the historical MASE scale is zero, its MASE is undefined; the task remains in applicable unscaled metrics. Native scales and common-history scales must be identified explicitly.

Monetary costs are represented by a resource-price vector and a cost function, `C_j(p)`. Prices vary across providers, regions, devices, billing arrangements, and dates. Recorded call time, device occupancy, and complete service cost are different quantities. Missing labor, maintenance, or deployment costs are not zero, and predictive error reductions are not measured operating profits.

## Repository layout

```text
.
├── docs/                   # Current English narrative and release boundaries
├── paper/                  # Audited companion materials; see its own inventory
├── shared/
│   ├── configs/            # Common A/B/O task calendar
│   ├── data/               # Existing versioned input/target artifacts
│   └── requirements.txt
├── track1/
│   ├── audit/              # Historical baseline selection evidence
│   ├── code/               # Pipeline, baselines, adapters, and evaluators
│   ├── configs/            # Search spaces, frozen choices, upstream registry
│   ├── compose/            # CPU and GPU service definitions
│   ├── docker/             # Separate model environments
│   ├── docs/               # English archival protocols and execution guides
│   ├── notebooks/          # English data and backtesting notebooks
│   ├── scripts/
│   └── tests/
├── track2/
│   └── docs/               # Archival ensemble and coordination protocol
├── RQ1/ , RQ1_outputs/     # Section 4.1 builders and sealed tables
├── RQ2/ , RQ2_outputs/     # Section 4.2 builders and sealed panels
├── makeup-RQ2/             # Final holiday, interval, and figure 5–10 package
└── RQ3/ , RQ3_outputs/     # Section 4.3 builders, tables, and DeepSeek judgments
```

Generated results, model weights, caches, and credentials are excluded from Git. The repository does not introduce a new license grant for existing data or third-party models.

## Historical baseline selection audit

On 2026-09-26, baseline B01–B12 validation completed **12,636 of 12,636 candidate forecast paths**, with zero terminal failures, across 33 candidate configurations. The final state was `frozen_after_B_group_tuning`.

- [Frozen configurations](track1/configs/frozen_baseline_configs.yaml)
- [Audit records](track1/audit/b_validation/README.md)

These are historical selection records, not the number of tasks in the current manuscript. Outer-test results must not be used to replace the original frozen choice. Later lightweight revisions are reported under their exploratory identity.

## Quick start: existing CPU implementation

Run commands from the repository root:

```bash
python -m pip install -r shared/requirements.txt
python track1/code/audit_protocol_artifacts.py
python track1/code/baseline_config.py --task-manifest shared/data/task_manifest.json
```

Example baseline smoke run:

```bash
python track1/code/run_backtest.py \
  --inputs shared/data/task_inputs.parquet \
  --destinations Macau \
  --markets CN_Chinese_mainland \
  --origin-date 2023-12-31 \
  --history all \
  --splits outer_backtest \
  --models seasonal_naive,ets,sarima \
  --output track1/results/smoke_isolated
```

Evaluate in the separate scoring environment:

```bash
python track1/code/evaluate_predictions.py \
  --predictions track1/results/smoke_isolated/predictions.parquet \
  --targets shared/data/targets.parquet \
  --output track1/evaluation/smoke_isolated
```

Docker entry points:

```bash
docker compose -f track1/compose/baseline.yaml run --rm predictor
docker compose -f track1/compose/baseline.yaml run --rm evaluator
```

For the historical Windows B-group driver:

```powershell
powershell -ExecutionPolicy Bypass -File track1/scripts/run_b_local.ps1 -Workers 4
```

These examples exercise the existing implementation. They do not rerun every later server experiment in the paper.

## GPU implementation and version control

Consult the [GPU guide](track1/docs/FOUNDATION_MODELS_GPU.md) and [sequential execution checklist](track1/docs/GPU_SEQUENTIAL_RUN_CHECKLIST.md). Each model uses its own container environment. Inspect the registered revisions and configuration before running; old smoke defaults are not automatically the final paper configuration.

For a new run, version resolution is an explicit setup step:

```bash
python track1/scripts/pin_upstream_revisions.py
git diff -- track1/configs/upstream_revisions.yaml
docker compose --env-file track1/.env.pinned \
  -f track1/compose/foundation-gpu.yaml build chronos_2
```

Do not run the pinning script during a frozen replication if it would replace the revisions under study. Restore the recorded run-specific versions instead.

## Data, credentials, and third-party rights

Existing tracked artifacts are preserved by this documentation update. Their presence is not a new assertion of unrestricted redistribution rights. Respect the original Macau and Hong Kong data terms and each model's code and weight license. Store credentials outside the repository. Do not commit API keys, access tokens, model weights, container layers, or private local paths.

For academic use, cite the manuscript when publication details are available and record the exact repository commit used. Repository updates do not assign a publication DOI or establish journal acceptance.
