# Macau daily tourism demand: unified experiment protocol and checklist

**Archival design, version 2026-09-23; translated for the 2026-10-21 release.** This record combines the original main-experiment decisions with rolling prediction and temporal validation rules. It documents the design at that time; its planned models, workloads, and unchecked items are not a claim of current completion. The [current paper inventory](../../paper/README.md) identifies the released evidence. Later exploratory changes retain their own identity.

Dates and rules below are study decisions. References support methodological principles, not every chosen Macau window or the performance of every model.

## 1. Rationale and registered decisions

### Reference support

The [CCF AI directory](https://www.ccf.org.cn/Academic_Evaluation/AI/) lists ICML, NeurIPS, and AAAI as category-A venues in the directory consulted for this protocol. The relevant sources and their actual scope are:

| Reference | Methodological support | Study-specific boundary |
|---|---|---|
| Ben Taieb & Hyndman, ICML 2014, [*Boosting multi-step autoregressive forecasts*](https://proceedings.mlr.press/v32/taieb14.html), [PDF](https://proceedings.mlr.press/v32/taieb14.pdf) | Rolling-origin time-series cross-validation for hyperparameters; direct versus recursive multi-step prediction | Supports chronological folds and multi-step losses, not particular Macau years or pandemic dates |
| Mouatadid et al., NeurIPS 2023 **Datasets and Benchmarks Track**, [*SubseasonalClimateUSA*](https://proceedings.neurips.cc/paper_files/paper/2023/hash/18ef499ee57c4822e1e3ea9b9948af18-Abstract-Datasets_and_Benchmarks.html), [PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/18ef499ee57c4822e1e3ea9b9948af18-Paper-Datasets_and_Benchmarks.pdf) | Section 3 recommends progressive training/validation with data available at issue time and common forecast dates | Supports origin-time availability and expanding updates; identify the conference track accurately |
| Zhou et al., AAAI 2021 main conference, [*Informer*](https://ojs.aaai.org/index.php/AAAI/article/view/17325) | Generative decoding produces a long forecast sequence in one forward pass | Supports adapting Informer to 28 outputs; does not require identical internal architectures for other models |
| Das et al., ICML 2024, [*A decoder-only foundation model for time-series forecasting*](https://proceedings.mlr.press/v235/das24c.html) | TimesFM foundation-model and cross-dataset zero-shot setting | Supports frozen-weight comparison; predates TimesFM 3.0 and is not its version-specific technical validation |
| Woo et al., ICML 2024, [*Unified Training of Universal Time Series Forecasting Transformers*](https://proceedings.mlr.press/v235/woo24a.html) | Pretrained universal forecasting without training on each target dataset | Supports a frozen Moirai comparison; checkpoint identity and training overlap still require separate review |
| Liu et al., NeurIPS 2022 main conference, [*Non-stationary Transformers*](https://proceedings.neurips.cc/paper_files/paper/2022/hash/4054556fcaa934b0bf76da52cf4f92cb-Abstract-Conference.html) | Effects of changing real-series distributions on forecasting | Supports reporting shock/recovery conditions separately; supplies no Macau policy cutoffs |

These papers support chronological validation, rolling origins, multi-step forecasts, and nonstationary contexts. Auditable dates, input isolation, raw predictions, versions, and sensitivity analyses provide the study-specific evidence. No claim is made that all model papers used this identical backtest.

### Data, inputs, outputs, and candidate scope

The original data snapshot contains 27 Macau source-market daily series, 2017-01-01 through 2026-07-31, and two Hong Kong arrival-category series, 2021-01-01 through 2026-07-31. Macau is the primary destination; Hong Kong is a separately reported extension. Absolute person-count errors from the two destinations are not pooled as the main ranking.

A task is one destination/market at one origin and one complete 28-day future path. Logical input shape is `[batch, context_length, 1]`, with only that market's observed history through the origin; output shape is `[batch, 28]`. Native APIs may differ, but adapters must preserve these semantics. Other markets, weather, holidays, events, and future covariates are excluded from this historical univariate main design; any enriched-information experiment is separate.

The planned comparison contains 13 baselines—Seasonal Naive, ETS, SARIMA, Ridge, Linear Regression, Random Forest, Gradient Boosting, SVR, Lasso, Elastic Net, Parrot Random Forest 800, LightGBM, and a small local Transformer—and seven time-series models: **TimesFM 3.0, Chronos-2, Lag-Llama, Moirai, Granite TimeSeries PatchTST-FM, PatchTST, and Informer**. This is the planned September scope, not the completed-model count in the current paper.

Use an official q0.5 forecast when available, otherwise a sample median or the official native point forecast. Save `prediction_raw`, `prediction=max(0,prediction_raw)`, and `was_clipped` for every step. Seasonal Naive repeats the corresponding weekday values from the final observed week across all 28 forecast days.

### Context and conditions requiring verification

- The 2020-01-22 stress boundary follows the [Macau first imported-case announcement](https://www.gov.mo/en/news/122165/). The 2023-01-08 reopening boundary follows the [entry/transit relaxation announcement](https://www.gov.mo/zh-hans/news/664390/). Policy dates fix evaluation windows; demand responses can occur earlier or later.
- In the historical snapshot, mean daily sums of the 27 Macau markets were 98,190.6 before the stress boundary; 15,766.9 during 2020-01-22–2023-01-07; 78,187.1 during 2023-01-08–2023-12-31; and 105,059.5 during 2024-01-01–2026-07-31. These are the sample's sums, **not all official Macau arrivals**.
- Corresponding day counts were 1,116, 1,082, 358, and 943; medians were 94,409, 17,026, 75,514, and 97,815. Frozen data hashes and registered windows govern final analysis.
- Audit model cards, checkpoint versions, and pretraining sources for the five pretrained models. If Macau/Hong Kong overlap cannot be excluded, report frozen weights without target-data fine-tuning and disclose unknown overlap; do not equate input isolation with proof of uncontaminated pretraining.
- Record code and weight licenses separately. The original design required reviewing the TimesFM 3.0 non-commercial weight terms. Record model release date, pretraining cutoff if available, and checkpoint revision; rely on actual pinned files rather than a current-version assumption.

## 2. Splits, validation, and rolling prediction

### Purpose-specific boundaries

| Purpose | Dates/origins | Target use |
|---|---|---|
| Initial Macau history | 2017-01-01–2020-01-21 | History for the first A fold |
| A: pandemic stress | Selected 28-day windows within 2020-01-22–2023-01-07 | Separate stress evaluation, no main-configuration selection |
| B: recovery selection | Twelve consecutive 28-day windows from 2023-01-08–2023-12-09 | Select hyperparameters and training epochs |
| Outer retrospective evaluation | Available targets from 2024-01-01–2026-07-31; primary windows end 2026-07-12 | Main evaluation after sealing predictions |
| Truly future forecasts | Example: origin 2026-07-31, targets 2026-08-01–2026-08-28 | Store without scoring until observations are available |

Keep validation, historical backtesting, and future forecasts separate. Existing historical targets do not become predictor inputs. Predictor and evaluator mounts enforce this separation.

### Registered expanding folds

Random K-fold validation breaks time order. Each training history runs from the market's first available day through the origin; validation is the following 28 days. A applies only to Macau because Hong Kong observations do not cover the early A folds.

| Fold | Origin/training cutoff | Target dates | Role |
|---|---|---|---|
| A01 | 2020-01-21 | 2020-01-22–2020-02-18 | Initial shock |
| A02 | 2020-04-21 | 2020-04-22–2020-05-19 | Stress |
| A03 | 2020-07-21 | 2020-07-22–2020-08-18 | Stress |
| A04 | 2020-10-20 | 2020-10-21–2020-11-17 | Stress |
| A05 | 2021-01-19 | 2021-01-20–2021-02-16 | Stress |
| A06 | 2021-04-20 | 2021-04-21–2021-05-18 | Stress |
| A07 | 2021-07-20 | 2021-07-21–2021-08-17 | Stress |
| A08 | 2021-10-19 | 2021-10-20–2021-11-16 | Stress |
| A09 | 2022-01-18 | 2022-01-19–2022-02-15 | Stress |
| A10 | 2022-04-19 | 2022-04-20–2022-05-17 | Stress |
| A11 | 2022-07-19 | 2022-07-20–2022-08-16 | Stress |
| A12 | 2022-10-18 | 2022-10-19–2022-11-15 | Stress |
| A13 | 2022-12-10 | 2022-12-11–2023-01-07 | Pre-relaxation transition |
| B01 | 2023-01-07 | 2023-01-08–2023-02-04 | Reopening boundary |
| B02 | 2023-02-04 | 2023-02-05–2023-03-04 | Recovery |
| B03 | 2023-03-04 | 2023-03-05–2023-04-01 | Recovery |
| B04 | 2023-04-01 | 2023-04-02–2023-04-29 | Recovery |
| B05 | 2023-04-29 | 2023-04-30–2023-05-27 | Recovery |
| B06 | 2023-05-27 | 2023-05-28–2023-06-24 | Recovery |
| B07 | 2023-06-24 | 2023-06-25–2023-07-22 | Recovery |
| B08 | 2023-07-22 | 2023-07-23–2023-08-19 | Recovery |
| B09 | 2023-08-19 | 2023-08-20–2023-09-16 | Recovery |
| B10 | 2023-09-16 | 2023-09-17–2023-10-14 | Recovery |
| B11 | 2023-10-14 | 2023-10-15–2023-11-11 | Recovery |
| B12 | 2023-11-11 | 2023-11-12–2023-12-09 | Recovery |

A01 uses only pre-pandemic history. B01 includes pandemic history but no post-reopening observations. Later B folds add newly observed data. The 13 A windows are sparse and do not score every pandemic day.

Each candidate is evaluated on `27 × 12 = 324` Macau B tasks. Calculate MASE for each path, then give markets and folds equal weight. Register candidate settings, search spaces, seeds, epoch decisions, maximum trials, and tie-breaking rules before outer evaluation. A prespecified near-tie rule can prefer lower resources. Apply the frozen choice unchanged to `27 × 13 = 351` A tasks. Do not combine A and B into an undeclared score; any cross-regime objective requires prior weights and a separate model ID.

B targets end on 2023-12-09; the first outer target is 2024-01-01. Once a configuration is frozen, observed validation data may enter the subsequent fitting history. This is not outer-target tuning. Retain per-market/fold losses, time, failures, and seeds for every trial. For PatchTST/Informer, determine the epoch rule on B—for example a prespecified median of fold-optimal epochs—then use that fixed epoch count when retraining within each outer origin.

**Hong Kong extension:** Macau B determines main hyperparameters. At each Hong Kong origin, fit only on that series' own past observations. Hong Kong does not participate in Macau parameter selection. Separate Hong Kong tuning would be a separately labeled experiment.

### Outer evaluation as an origin-time decision

For origin `o`, only observations with `date ≤ o` are available; targets are `o+1` through `o+28`. The first origin is 2023-12-31, followed every 28 days for 33 non-overlapping target windows. The final regular origin is 2026-06-14 and the final primary target is **2026-07-12**.

A separately registered terminal-window check could use origin 2026-07-03 and end at 2026-07-31. Its targets overlap the previous window by nine days, so it does not belong to the 33-window primary ranking. This was a proposed extension, not a completion claim.

| Origin | Available history | Forecast path |
|---|---|---|
| 2023-12-31 | First observation through 2023-12-31 | 2024-01-01–2024-01-28 |
| 2024-01-28 | First observation through 2024-01-28 | 2024-01-29–2024-02-25 |
| 2024-02-25 | First observation through 2024-02-25 | 2024-02-26–2024-03-24 |

A later origin may use earlier targets after they have been observed. An earlier task may not. Autoregressive generation may feed back its own forecasts/samples, never unobserved future actuals.

```python
for market in markets:
    for origin in fixed_common_origins:
        history = observations[market].loc[:origin]
        assert history.index.max() == origin
        if adapter.requires_fit:
            adapter.fit(history, frozen_hyperparameters)
        forecast = adapter.predict(history, horizon=28)
        write_predictions_without_actuals(forecast)

# A separate evaluator loads and joins targets by task_id and target_date.
```

The main history is all available history through each origin. Trainable models generate windows `X_t=[y_(t-L+1),...,y_t]`, `Y_t=[y_(t+1),...,y_(t+28)]` with `t+28≤o`; the final forward pass can still use only the latest finite lookback `L`. Frozen pretrained models update context, not weights, and use a preregistered recent-history truncation when context limits bind. Record `training_history_days`, `lookback_days`, `requested_context_days`, and `effective_context_days` separately.

ETS/SARIMA and regression baselines refit their frozen specifications at each origin. PatchTST/Informer retrain with B-selected settings and epochs. A cheaper monthly retraining schedule is a distinct model ID and comparison.

The original workload plan was `29 series × 33 origins × 20 models = 19,140` paths and `535,920` daily forecast values. Adding all/180/365 histories yields `57,420` paths. Daily rolling over 916 origins is a separate operational scenario with overlapping errors; it requires its own workload and uncertainty analysis. The current primary Macau paper count is 891 tasks per configuration, not this planned 29-series aggregate.

## 3. Evaluation, resource, and provenance rules

### Main accuracy metric

For each 28-day task, MASE(7) divides mean future absolute error by `mean(|y_t-y_(t-7)|)` from pre-origin history. A zero denominator yields `undefined_scale`; omit that task only from the defined-scale MASE summary under the declared rule, retain MAE, and report the count. Summarize equal-weight markets/origins using mean, median, IQR, and worst decile. Do not pool raw person-count errors across markets as the main ranking.

The scale changes as history expands, especially across the pandemic. Also report MAE and Seasonal Naive win rates, and prespecify alternative seasonal/fixed-history scale checks. Do not switch the primary metric after seeing outer rankings.

### Auxiliary outcomes

MAE, RMSSE, bias, Seasonal Naive win rates; high-demand thresholds based on each origin's historical 90th percentile; high-demand MASE, mean underprediction magnitude, and underprediction frequency; horizons 1, 7, 14, 28 and complete paths. Use CRPS when sufficient samples/distributions exist; otherwise identify quantile-based pinball loss/WIS, 80% coverage, and mean interval width. Point-only baselines remain eligible for point-forecast comparisons. Genuine zeros make MAPE unsuitable as the primary metric.

### Time, resources, and accounting

Requested telemetry separates first download, cached load, preprocessing, fitting, warmed inference, postprocessing, writing, and the complete Macau workload. Synchronize GPU timers; report p50/p95, batch size, and throughput. Record weights/image sizes, peak RSS, allocated/reserved/device VRAM, GPU/driver/CUDA, CPU, and RAM.

The archival accounting proposal normalizes training, inference, API, storage, and retry resources to 1,000 successful 28-day tasks using a common provider/region/instance/currency/date. Local hardware can be reported using observed time plus an explicitly chosen cloud-equivalent price; researcher active hours remain separate. The current paper only prices resources actually recorded and uses symbolic platform-dependent prices; missing telemetry is not zero and the proposal does not establish complete lifecycle cost measurement.

### Inference and task coverage

Use common task keys for paired loss comparisons, with total coverage, failure, and missing-task reports alongside them. Block bootstrap should preserve adjacent-origin dependence within markets. A DM test must use a variance estimate appropriate to any overlapping multi-step errors. Report effect sizes and intervals, not just p-values.

### Output contract

The intended run archive includes `run_manifest.json`, partitioned `predictions.parquet`, task/aggregate metrics, `timings.jsonl`, `resources.jsonl`, `costs.parquet`, `failures.jsonl`, and manual `implementation_log.csv`. Record input SHA256, task list, local/upstream commits, checkpoint revision, image digest, licenses, hardware, seeds, batch size, effective context, and success/failure/retry counts.

Use stable task IDs based on destination, market, origin, history setting, and horizon. Record origin, training start/end, forecast start/end, fit strategy, hyperparameter source, validation-fold IDs, `actuals_visible_to_model=false`, and forecast/evaluation times. Results are keyed by `task_id + model_id + seed`. Optional quantiles/samples remain missing when unavailable. Write to temporary files, validate 28-day alignment and finite values, then rename atomically; preserve unsuccessful attempts during resumption.

### Pretraining audit

For each pretrained model, record known sources, cutoffs, and overlap information from its model card. Where the source does not enumerate all pretraining series, report overlap as unknown. Input-isolation tests cover the present run, not upstream pretraining contamination.

## 4. Historical execution checklist

The checkmarks below reflect the original document, not a new audit of all later runs.

### Freeze method and data

- [x] Version A01–A13, B01–B12, and 33 outer origins with task indices and hashes.
- [ ] Freeze seven time-series and 13 baseline IDs, 28-day horizon, seasonality 7, main expanding history, and 180/365 sensitivities.
- [ ] Register B search spaces, trial budgets, seeds, epochs, metric, zero-scale handling, and failures; reserve A for stress testing.
- [x] Generate separate task inputs and targets; ensure origin-aligned histories and 28 consecutive target days; exclude Hong Kong from Macau A/B selection.
- [ ] Verify source-data rights, revision dates, weight licenses, and input hashes.

### Build the common runtime

- [ ] Establish versioned code/configuration, ignore rules, license inventory, upstream registry, CI, and container distribution; keep weights and large results outside Git.
- [ ] Implement adapter load/fit/forecast/close semantics, point/quantile/sample outputs, clipping, and context metadata.
- [ ] Migrate and compare the 13 baselines on a common smoke task; verify Seasonal Naive's repeated final week.
- [ ] Integrate separate pinned environments for all seven time-series models.
- [ ] Add time, memory, cost, active labor, failure, and retry recording before formal GPU runs.

| Model | Registered adaptation checks |
|---|---|
| TimesFM 3.0 | Pin `google/timesfm-3.0-pytorch`, review weight terms, univariate input, official q0.5, context limit/truncation |
| Chronos-2 | Pin `amazon/chronos-2`, univariate target without covariates, q0.1/q0.5/q0.9, no cross-market information through batching |
| Lag-Llama | Pin source/checkpoint/Python/PyTorch/GluonTS/Lightning, RoPE/context, sample count and seed; retain samples and median |
| Moirai | Pin Uni2TS/checkpoint, daily frequency, `prediction_length=28`, patch rule, samples/quantiles; no outer-based per-task patch selection |
| Granite | Pin source/checkpoint, verify frozen-weight mode, record padding/context/truncation; any fine-tuning gets a separate model ID |
| PatchTST | Freeze lookback, patch/stride, layers, learning rate, epochs, seeds; train on origin-available windows; retain losses/checkpoint size |
| Informer | Freeze `features=S`, `freq=d`, `pred_len=28`, seq/label lengths, layers/heads, learning rate, epochs, seeds; retain training/validation logs |

Check every API against the pinned revision rather than assuming current interfaces match this table.

### Run and report

- [ ] Smoke each model on one market/origin with aligned finite outputs and no future inputs.
- [ ] Use a preselected 3-market × 3-origin pilot to estimate memory, time, and failures; adjust engineering settings without choosing on outer accuracy.
- [ ] Complete B selection and freeze; apply unchanged settings to A.
- [ ] Run the original planned 33-origin common sample (957 tasks per model including the two separate Hong Kong series); identify actual completion and compatible comparison sets.
- [ ] Separately report 180/365 histories, the optional terminal window, and any budget-approved 916-origin operational run.
- [ ] Report accuracy, high demand, probability, timing, memory, resource costs, failures, active labor where measured, and coverage.
- [ ] Reproduce a smoke task and one shard on a clean GPU host using recorded versions.

Minimum defensibility requires fixed temporal boundaries, B-only selection, origin-time inputs, independent target joins, comparable tasks, traceable failures, separate stress/selection/outer results, and honest disclosure of unknown pretraining overlap. These are methodological controls, not a guarantee against all criticism.
