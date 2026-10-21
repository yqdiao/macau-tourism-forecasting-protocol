# Unified GPU design for seven time-series models and 13 baselines

**Archival design, updated 2026-09-23; English edition for the 2026-10-21 release.** The chosen design comprises TimesFM 3.0, Chronos-2, Lag-Llama, Moirai, Granite TimeSeries PatchTST-FM, PatchTST, and Informer. It uses single-market daily histories, 28-day forecasts, all-available expanding histories, and separate 180/365-day sensitivities. See the [unified protocol](UNIFIED_EXPERIMENT_PROTOCOL.md) and [current evidence inventory](../../paper/README.md).

This document preserves the original design rationale and planned implementation contract. Proposed directories, resource fields, model counts, and unchecked milestones are not claims that all were implemented or measured.

## 1. Scope

The source list of open-source repositories contained examples involving minute-level stocks and order books. Only repository/interface/dependency information was relevant here; those financial tasks were not adopted.

The original study design includes 27 Macau market series and two Hong Kong category series, reported separately. Each task is a destination/market, origin, and full 28-day path. Targets are available from 2024-01-01 through 2026-07-31. There are 916 possible daily rolling origins, but the common sample selects every 28th origin from 2023-12-31, giving 33 origins and primary scored dates ending 2026-07-12.

Main inputs are only the market's observed counts through each origin. No future actuals, other markets, retrospective weather, or event labels enter the main univariate comparison. Clip final predictions at zero while retaining raw values and clipping flags. The 13 baseline configurations are Seasonal Naive, ETS, SARIMA, Ridge, Linear Regression, Random Forest, Gradient Boosting, SVR, Lasso, Elastic Net, Parrot Random Forest 800, LightGBM, and a locally trained encoder-style Transformer. The last is not a pretrained foundation model.

## 2. Engineering approach

Use one research repository, separate pinned containers, and a common execution layer. A single Python environment is not assumed to support the conflicting Python/PyTorch/TensorFlow/GluonTS/CUDA requirements of all candidate repositories.

The intended workflow is:

1. Keep pipeline, adapters, Dockerfiles, locks, configurations, tests, and small summaries in Git.
2. Build an independent image per model; a container registry can distribute immutable image digests.
3. Mount authorized data read-only, outputs read-write, and keep raw data, weights, and large predictions outside Git.
4. Run on Linux hosts with compatible drivers and NVIDIA Container Toolkit.
5. Use CI for formatting, unit/schema tests, and small CPU fixtures; run formal GPU experiments on allocated servers or appropriate runners.
6. Assign immutable run IDs and retain configurations, versions, predictions, metrics, timing, resources, failures, costs, and resumption state.

A container fixes user-space dependencies, not the host NVIDIA driver or available VRAM. GPU availability remains an actual hardware constraint.

## 3. Candidate-repository decisions

| Original repository | Nature | Suitability for this task | Treatment |
|---|---|---|---|
| TimesFM | Pretrained forecasting foundation model | High | Frozen-weight main comparison; 3.0 design with a separately identified 2.5 replication option |
| Chronos | Pretrained forecasting foundation model | High | Frozen-weight main comparison; Chronos-2 registered |
| Lag-Llama | Probabilistic forecasting foundation model | Moderate/high | Frozen-weight comparison in a separately pinned environment |
| PatchTST | Locally trainable Transformer | High | Locally trained deep-model comparison |
| Informer2020 | Locally trainable Transformer | Moderate/high | Locally trained deep-model comparison |
| Amazon Forecast Samples | Cloud-service examples, not local weights | Low | Optional service benchmark only with actual service access |
| DeepLOB | Limit-order-book direction classifier | Domain mismatch | Excluded from the tourism accuracy ranking; retain rationale |

### Exclusions and alternatives

Amazon Forecast Samples demonstrates AWS API data import, predictor training, and forecast export. It does not supply downloadable local forecasting weights. The historical source review noted restricted new-customer availability and an archived samples repository. Service access and terms must be checked for any actual run; its training/API/storage/export/retry costs would be separate.

DeepLOB expects order-book tensors and predicts price-direction classes. Replacing its inputs and head to predict 28 visitor counts would create a new regression adaptation, not a reproduction of its original task. Record `domain_mismatch`, rather than inventing comparable results.

The final planned seven-model set substitutes Moirai and Granite for these two unsuitable repository examples: TimesFM 3.0, Chronos-2, Lag-Llama, Moirai, Granite PatchTST-FM, locally trained PatchTST, and locally trained Informer. A table of the original seven repositories may use `N/A` with reasons for excluded entries; repository counts must not be confused with completed comparable models.

### TimesFM versions and licensing

The original review distinguished non-commercial TimesFM 3.0 weight terms from the Apache-2.0 2.5 weights. Review the actual pinned model cards and licenses for the intended use. Record code and weight licenses, model-card URLs, and revisions. A retained 2.5 replication option is a version sensitivity, not an artificial extra model counted to enlarge the comparison.

## 4. Proposed repository contract

The original architecture proposed a core orchestration package with separate modules for data preparation/validation, adapters, evaluation/statistics, telemetry/cost/provenance, and a runner; model/hardware/price YAMLs; separate core/model Dockerfiles; a pinned upstream registry; build/smoke/common-sample/resume scripts; and tests for leakage, adapter contracts, metrics, and schemas.

The actual repository uses `shared/`, `track1/`, and `track2/`, rather than claiming that every originally proposed `src/`, `environments/`, `Makefile`, `uv.lock`, or registry component exists. Its current layout is described in the root [README](../../README.md).

The intended run archive is:

```text
results/runs/<run_id>/
  run_manifest.json
  predictions/*.parquet
  task_metrics.parquet
  aggregate_metrics.parquet
  timings.jsonl
  resources.jsonl
  costs.parquet
  failures.jsonl
  logs/
```

Pin upstream URLs/commits instead of copying floating `main` branches. If source changes are required, record a patch hash or licensed submodule/vendor revision. Fetch weights at exact revisions into a separately managed cache.

## 5. Common adapter contract

Illustrative contract, not a claim that this exact CLI exists:

```text
forecast \
  --input /work/tasks/task_batch.parquet \
  --config /work/configs/models/chronos_2.yaml \
  --output /work/results/<run_id>/model=chronos_2
```

| Input field | Meaning |
|---|---|
| `task_id` | Stable destination, market, origin, and history identifier |
| `destination` | Macau or HongKong |
| `market` | Source market or visitor category |
| `origin_date` | Last observable date |
| `history_label` | Main all-available expanding setting, or 180/365 sensitivity |
| `history_start` | Actual history start |
| `history_values` | Daily counts through the origin |
| `forecast_horizon` | 28 |
| `frequency` | D |

| Output field | Meaning |
|---|---|
| `run_id, task_id, model_id, attempt` | Provenance and retries |
| `target_date, horizon_day` | Future date and step 1–28 |
| `actual` | Joined by the independent evaluator only after forecasting |
| `prediction_raw` | Original point forecast |
| `prediction` | Common nonnegative point forecast |
| `q_0_1, q_0_5, q_0_9` | Quantiles where supplied |
| `sample_id, sample_value` | Samples in a separate long table where supplied |
| `effective_context_days` | History actually read |
| `batch_size, device, precision` | Inference configuration |

Never include actual future targets in predictor inputs. The independent evaluator joins by task and target date after forecasts are written.

## 6. Fair-comparison rules

### Information

Use target-market univariate history in the main comparison even if a model supports multivariate context or covariates. Enriched-information capability experiments are separate. GPU batching must not introduce cross-market in-context information; infer tasks separately or use explicitly isolated groups where required.

### Point and probabilistic forecasts

Use official q0.5, otherwise sample median, otherwise official native point output. Retain raw and clipped values. Compare distributional metrics only among models with valid distributions/quantiles, without excluding point-only benchmarks from the point comparison.

### History

Distinguish all history used to construct training examples from the final forward-pass context. Record `training_history_days`, `requested_context_days`, and `effective_context_days`. If architecture limits context, use the most recent supported observations under a fixed rule and report truncation. The 180/365 settings are sensitivities.

### Local training and randomness

PatchTST/Informer train separately on each market/origin/history's available data. Tune only before outer targets start on 2024-01-01. Internal early-stopping validation must also precede the origin. Retries retain settings; a changed hyperparameter configuration receives a new model ID.

Deterministic frozen point inference can run once. Sampling uses fixed seeds and counts. The original proposal recommends three prespecified seeds for PatchTST/Informer, reporting the mean and between-seed variation. Record Python, NumPy, framework seeds and deterministic switches; disclose remaining CUDA nondeterminism.

## 7. Planned scale and staged execution

These are **design workloads**, not current completion counts:

```text
29 series × 33 origins × 1 history = 957 tasks per model
957 × 28 = 26,796 daily forecast values
20 configurations = 19,140 tasks and 535,920 daily values
20 configurations × 3 histories = 57,420 tasks and 1,607,760 daily values

29 series × 916 daily origins × 3 histories = 79,692 tasks per model
20 configurations = 1,593,840 tasks and 44,627,520 daily values
```

Stage the work:

1. Interface smoke: one market/origin/history; validate shape, dates, clipping, and isolation.
2. Engineering pilot: three preselected markets × three origins × three histories; estimate VRAM, batch size, failure rate, and task time.
3. Common sample: registered origins/histories for comparable models, with actual coverage reported.
4. Optional daily rolling: start with baselines/frozen models; extend expensive local models only under the predeclared pilot budget rule.
5. Separate sensitivities by history, destination, market scale, high demand, stress/recovery, and probability outputs.

Do not change common origins after inspecting model accuracy. Pilot results determine engineering feasibility and budget progression, not outcome-driven sample selection.

## 8. Accuracy and statistical comparison

For seasonal period 7 and each 28-day task:

```text
MASE = mean_h(abs(y_h - yhat_h)) /
       mean_{t=8..T}(abs(y_t - y_{t-7}))

RMSSE = sqrt(mean_h((y_h - yhat_h)^2) /
             mean_{t=8..T}((y_t - y_{t-7})^2))
```

Scales use only pre-origin history. Calculate task metrics before averaging; report undefined-scale counts, means, medians, and quartiles. Do not silently replace task-level macro-MASE by a pooled ratio.

Additional registered outcomes: MAE; bias `mean(prediction-actual)`; high-demand MASE; mean underprediction magnitude `mean(max(actual-prediction,0))`; underprediction frequency; quantile coverage and width; CRPS with sufficient samples/distributions, or identified pinball/WIS when only quantiles exist; horizons 1, 7, 14, 28 and full paths; market/destination summaries, IQR, worst decile, and Seasonal Naive win rates.

Genuine zeros rule out MAPE as the main metric. R-squared is not the main multi-step ranking criterion. Use paired block-bootstrap intervals preserving market/origin dependence; if DM tests are used, choose HAC variance for overlapping forecasts. Any Friedman/post-hoc comparison needs a stated multiple-comparison treatment. Report effect sizes and intervals alongside p-values. The current manuscript's main metric set is described in `paper/`; a proposed historical metric is not necessarily a completed result.

## 9. Timing and resource contract

The proposed telemetry separates:

1. First weight download.
2. Cached cold start and loading.
3. Preprocessing.
4. Fitting/training.
5. Warmed task inference.
6. Postprocessing and disk writing.
7. p50/p95 latency.
8. Complete Macau workload wall time.
9. Successful tasks per second.

Do not mix download time with regular inference. Synchronize GPU operations around timers, use fixed warm-ups, and report batch size plus actual throughput.

Record actual weight bytes, compressed/uncompressed image sizes, peak RSS, allocated/reserved/whole-device VRAM, GPU model and capacity, driver/CUDA/cuDNN, CPU/cores/RAM/OS, framework/dependency versions, and optional power/Wh. Timestamp resource samples and reset peak counters between models.

## 10. Resource-to-cost mapping

The archival normalization is:

```text
cost_per_1000 = 1000 * total_cost_including_failed_attempts / successful_tasks
```

The intended total includes API use, training/inference instance time, storage/network use, and failed/retried attempts, each counted once. Retain provider, region, instance type, currency, unit price, price date, billed seconds, success/failure counts, and separate training/inference/retry/storage components.

Local hardware may use a preselected cloud-equivalent price, but observed runtime must also be reported. Compare under consistent pricing conditions rather than selecting favorable providers per model. Separate cold-start and cached steady-state scopes. Local training belongs to each task's workload; upstream pretraining is outside the user's downstream accounting boundary, not cost-free. Report tuning separately and active researcher hours without an arbitrary monetary conversion.

The current paper expresses costs symbolically using measured resources and applicable prices. This historical list does not establish that all lifecycle components were measured. Missing fields remain unmeasured.

## 11. Implementation burden

The proposed manual `implementation_log.csv` fields are `model_id`, `activity` (installation/adaptation/tuning/debugging), `started_at`, `ended_at`, `active_minutes`, executor identifier, `issue_url`, and `outcome` (success/failure/abandonment with reason). Exclude unattended compute from active time and avoid publishing private personnel identifiers.

Automatic records should identify required/actual device, fresh-environment time to a correct forecast, tuning trial/compute counts, tasks/failures/retries/missing results, external API/service calls, model/image sizes, and peak RAM/VRAM. Report these separately rather than creating a subjective complexity score.

## 12. Run manifests and recovery

Record run ID, UTC start/end, command, full YAML, repository commit and dirty state, upstream URL/commit, model ID/revision/hash, image digest, input hash/series/date range, origin list/history/horizon/frequency, seed/precision/batch/device, hardware/software, successes/failures/retries/skips, network/API/covariate/cross-series use, and license locations.

Use `task_id + model_id + seed` as the intended completion key, with additional setting distinctions where needed. Write temporary predictions, validate dates/28 rows/finite values, then rename atomically. Skip only validated completions; successful retries must not erase earlier failure logs. The original launcher requires separate output roots until its full key semantics are audited.

## 13. Git and storage

Commit source, Dockerfiles, locks, configuration, tests, small fixtures, summaries, plotting code, manifests, and upstream/license records. Keep unauthorized raw redistribution, weights, caches, credentials, container layers, and tens of millions of daily predictions outside Git. Use authorized artifact storage with checksums for large files. Inject secrets through proper credential stores/environment mechanisms. Read-only input and read-write output mounts remain separate.

## 14. Acceptance gates

A model entering a common ranking should have: exactly 28 aligned finite values on the fixed smoke task; no future targets in inputs; verified Seasonal Naive behavior; common task coverage or explicit missingness; raw/clipped outputs; history-only scales; recorded versions/hardware/time/resources/failures; a clean-host smoke reproduction; no silent outcome-driven changes of seeds/history/origins/versions; and a prespecified completion threshold. Models failing coverage requirements receive availability reports rather than an unqualified average rank.

## 15. Historical TODO register

### A. Freeze design

- [x] Version the common protocol, 29-series planned scope, horizon, 33 origins, and three histories.
- [x] Export immutable task indices and SHA256.
- [ ] Finalize comparable model set versus original repository audit; use Moirai/Granite alternatives where justified.
- [ ] Review TimesFM 3.0 use terms and preserve any separately specified 2.5 replication setting.
- [ ] Fix high-demand/probability/statistical/missingness rules.
- [ ] Fix local-training budgets, seeds, and early stopping.

### B. Repository and artifacts

- [ ] Establish an installable core, version registry, rights inventory, citation metadata, CI, and large-artifact storage as appropriate.
- [ ] Keep existing pipeline/baseline semantics compatible during any refactoring.
- [ ] Fix access and publication scope before release; no implicit permission expansion.

### C. Inputs and outputs

- [ ] Produce Arrow/Parquet tasks and a common adapter contract.
- [ ] Integrate baseline manifests and date/length/NaN/negative/duplicate/leakage checks.
- [ ] Add probabilistic schemas with a point-only path.

### D. Independent environments

- [ ] Build core CPU and all applicable pinned model images.
- [ ] Validate Chronos batching isolation; Lag-Llama compatibility; local daily/horizon training logs.
- [ ] Treat Amazon Forecast as an optional authorized API benchmark and DeepLOB as a domain audit.
- [ ] Register immutable digests for any distributed images.

### E. Telemetry

- [ ] Implement separated timers, GPU synchronization/warm-up, RSS/VRAM and optional power sampling.
- [ ] Register hardware and dated prices; price 1,000 successful tasks with retries included once.
- [ ] Keep manual active-work records separately.

### F. Staged execution

- [ ] Smoke each adapter, then run the prespecified 3×3×3 pilot.
- [ ] Freeze engineering and study settings before outer evaluation.
- [ ] Run common origins for comparable baselines and time-series models.
- [ ] Check coverage and date alignment before ranking.
- [ ] Apply the previously recorded budget rule to any optional 916-origin run.

### G. Analysis and release

- [ ] Produce task/market/destination metrics and high-demand outcomes.
- [ ] Report available probability scores, latency, throughput, memory, size, resources, and failures.
- [ ] Apply declared paired inference and coverage reporting; explain all `N/A` entries.
- [ ] Reproduce one smoke task and common-sample shard on a clean environment.

## 16. Initial execution priorities

Finalize the comparable model list; export immutable tasks; establish adapters/manifests/resumption/telemetry; integrate TimesFM and Chronos before the remaining models; and estimate the common-sample budget from the prespecified pilot before scaling up. Installation failures and retries are implementation evidence and must remain in the record.
