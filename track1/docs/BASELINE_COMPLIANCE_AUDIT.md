# Baseline protocol compliance audit

**Archival audit, dated 2026-09-23.** This report describes the implementation at that date, not the current completion status of every paper experiment. The subsequent [2026-09-26 baseline freezing audit](../audit/b_validation/README.md) supersedes the pending B-group status below. See the [current release inventory](../../paper/README.md) for later evidence.

Scope: `track1/code/data_pipeline.py`, `track1/code/models.py`, `track1/code/run_backtest.py`, the data snapshot, and smoke outputs. Reference: [Unified experiment protocol](UNIFIED_EXPERIMENT_PROTOCOL.md).

## Findings at the audit date

The code provided a common 28-day univariate interface for 13 baselines. Training features did not use targets after the forecast origin. Physical input/target separation, `all` history, raw/clipped outputs, task metrics, atomic writes, and A/B/O calendars were implemented. Thirty-three candidate configurations were registered, but B-group selection, freezing, full resource telemetry, and the seven time-series adapters had not yet been completed. The smoke outputs therefore did not establish the final paper ranking.

The 13 models were Seasonal Naive, ETS, SARIMA, Ridge, Linear Regression, Random Forest, Gradient Boosting, SVR, Lasso, Elastic Net, Parrot Random Forest 800, LightGBM, and a local Transformer.

## Compliant items

| Item | Finding | Evidence |
|---|---|---|
| Forecast horizon | Compliant | All adapters defaulted to 28 days |
| Univariate inputs | Substantially compliant | `forecast()` received only one market's one-dimensional history, semantically equivalent to `[1, context_length, 1]` |
| Origin boundary | Compliant | Tabular and Transformer training labels ended no later than the origin |
| Seasonal Naive | Compliant | Repeated the final seven observed values without reading future observations |
| Common outer origins | Compliant | 33 origins from 2023-12-31 through 2026-06-14; final scored date 2026-07-12 |
| Nonnegative outputs | Compliant | Raw values, clipped values, and clipping flags were retained |
| A/B/O calendar | Compliant | A01–A13, B01–B12, O01–O33 in `shared/configs/protocol.yaml`; hashes in task index/manifest |
| 180/365-day history | Runnable | Parameters and smoke outputs existed; complete common-sample evaluation was pending |

## Gaps and resolved items at that date

| Severity/status | Item | Audit observation and requirement |
|---|---|---|
| Resolved | Physical target isolation | `ForecastInputTask` excluded actuals; predictor read only inputs; evaluator read targets; predictor service had no target mount |
| Resolved | Main `all` history | All-available tasks were generated and smoked; `all` was no longer converted to zero |
| Blocking then | B-group selection and freezing | Registered folds and 33 candidates were supported, but the configuration remained `pending_B_group_tuning`; B selects, A stresses, O never retunes |
| Resolved | Raw forecasts | `ForecastResult` and Parquet retained `prediction_raw`, `prediction`, `was_clipped` |
| Resolved | Aggregation unit | Evaluator generated `task_metrics.parquet` before equal-task macro-aggregation |
| High | Run provenance | Data hashes existed; code/upstream/checkpoint/container versions and complete success/failure/retry records were missing |
| High | Resume behavior | Atomic writes existed; completed-task indices, partitioned recovery, and failure files remained incomplete |
| Medium | Hyperparameter freezing | Search spaces and seeds existed, but smoke defaults were not frozen outer settings |
| Medium | Transformer reproducibility | `torch.manual_seed` was fixed; checkpoints, full deterministic settings, multiple seeds, and hardware/training logs were incomplete |
| Medium | Resources and costs | Fit/inference/total seconds existed; loading, warm-up, p50/p95, RSS, VRAM, throughput, costs, failures, and retries were incomplete |
| Medium | Audit schema | Stable task IDs, fold/split, context, raw prediction, and clipping existed; full version/resource/failure fields were pending |
| Medium | Smoke coverage | Only some models, one market, one origin, and 180-day history had been tested |

## Model-specific interpretation

- **Seasonal Naive:** retain raw/clipped outputs and independent scoring.
- **ETS/SARIMA:** register specifications, select candidates only on B, and record failures/convergence.
- **Nine tabular baselines:** no obvious future leakage in direct multi-step training; freeze B-selected parameters and retain per-horizon settings.
- **Local Transformer:** a locally trained neural baseline, not a pretrained time-series foundation model; retain selection, checkpoints, seeds, training logs, and resource evidence.

## Historical repository-upload boundaries

Suitable artifacts: pipeline, adapters, evaluator, configurations, environment locks, Dockerfiles, tests, this report, synthetic fixtures, schemas, field dictionaries, quality summaries, checksums, and upstream URLs/revisions/licenses/download scripts.

Items requiring source-rights and privacy checks: the approximately 9 MB `normalized_daily.jsonl`, student notebooks/reports, and source data. A private repository still redistributes data to authorized collaborators. Manifests and README files should contain logical source IDs rather than local absolute paths. Remove personal identifiers and credentials before any release.

Do not commit model weights, caches, virtual environments, API keys, complete large prediction collections, full GPU log archives, or unversioned copies of third-party source. Use pinned source references or properly licensed submodules/vendor copies when necessary. Never mount targets together with predictor-readable inputs.

## Historical remediation order

1. Version A/B folds and all 33 common origins; task indices were already implemented.
2. Complete B selection and freezing after connecting candidate parameters; then run the A stress set.
3. Add task-level resume indices, partitioned outputs, failures, and retry logs.
4. Add Seasonal Naive win rates, IQR, worst-decile summaries, and declared uncertainty estimates.
5. Complete dependency locks, upstream/checkpoint revisions, licenses, resources, and costs.
6. Integrate the seven pinned model environments and adapters.
