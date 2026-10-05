# Track 2 protocol: selected-model ensembles, residual correction, and agent coordination

**Archival design, version 2026-09-23; English edition for the 2026-10-21 release.** At the original document date, the three members awaited Track 1 B01–B12 selection. This protocol is preserved as a design record, not a current completion statement or proof that all later experiments were preregistered. See [paper/](../../paper/README.md) for the released evidence and [reproducibility status](../../docs/reproducibility_status.md) for availability boundaries.

The original scope is Macau daily demand, with Hong Kong as an external extension after freezing. Track 1 planned 13 baselines and seven time-series models; this Track 2 design uses three B-selected time-series models to test combination, residual correction, and coordination.

**Naming note:** the original document used M1/M2/M3 for ranked ensemble members. Here they are called **member 1/member 2/member 3**, to avoid confusion with the later Qwen experiment labels M1–M5. The original deterministic coordinator design is also distinct from later LLM weight-judgment experiments.

## 1. Scope and questions

Each task uses only one destination/market's daily history through its origin:

```text
Input:  [batch, context_length, 1]
Output: [batch, 28]
```

The original main setting adds no other markets, holidays, weather, flights, searches, or future actuals. Origins, horizon, task keys, and target isolation are shared with Track 1.

The design asks:

1. Does combining the three B-selected models improve on the best single member?
2. Do validation-error or constrained weights improve on equal weights?
3. Does optional correction of out-of-sample residuals yield stable improvements?
4. Does coordination improve task success, error detection, fallback, or execution efficiency?

In this historical coordinator design, numerical aggregation, weights, metric calculations, constraints, and fallback are deterministic. An LLM may invoke authorized tools and explain logs; it may not rewrite forecasts in natural language. This restriction does not describe the distinct later 8B weight-judgment design.

## 2. Shared temporal boundaries

Reuse `shared/configs/protocol.yaml`:

| Set | Folds | Track 2 purpose | Selection permitted? |
|---|---|---|---|
| Pandemic stress | A01–A13 | Stress evaluation after freezing | No |
| Recovery selection | B01–B12 | Select members, weights, and residual activation | Yes |
| Outer evaluation | O01–O33 | Final Macau evaluation and separate Hong Kong extension | No |
| Future deployment | Example targets after 2026-08-01 | Forecast without scoring until actuals arrive | No |

Track 1 hyperparameters are frozen first. O01–O33 may not select members, combination methods, weights, residual models, shrinkage, or fallback thresholds.

Selecting the three models after viewing the seven-model outer ranking and then scoring on the same outer data would reuse the test set for development. The original rule is B-only member selection and outer-target release to the evaluator only after freezing. If prior outer results have been inspected during development, label the resulting analysis exploratory or evaluate on a new, unused future period. A written protocol alone does not establish the chronological history of an experiment.

## 3. Selecting three members

### 3.1 Candidate set

The registered candidates are TimesFM 3.0, Chronos-2, Lag-Llama, Moirai, Granite TimeSeries PatchTST-FM, PatchTST, and Informer. Each must pass source/version registration, origin-level smoke, target isolation, 28 finite outputs, and provenance acceptance.

### 3.2 Eligibility

A candidate must:

1. Use only the history available at each B origin.
2. Use the frozen Track 1 configuration and recorded source/checkpoint/image versions.
3. Cover the prescribed Macau B tasks, with failures explicitly recorded.
4. Produce exactly 28 finite values per task.
5. Retain raw/clipped forecasts, context, and failure records.
6. Avoid any predictor access or mount of `targets.parquet`.
7. Have no unexplained omissions, manual replacements, or outer-target tuning.

The main design requires 100% B coverage. A lower threshold, no less than 95%, could only be registered before execution for unrecoverable failures. Ranking would then use the jointly completed tasks and prioritize failure/coverage before speed in tie breaking. This option is not permission to lower a threshold after observing outcomes.

### 3.3 Ranking and freezing

Calculate MASE(7) for every 28-day path; average folds equally within each market and markets equally overall. The proposed ordering is:

1. Lowest B macro task MASE(7).
2. RMSSE(7), when MASE differences are below a previously registered tolerance.
3. High-demand mean underprediction magnitude.
4. Failure rate and coverage.
5. GPU-equivalent resource cost per 1,000 successful tasks and p95 latency.

Register the tolerance, coverage threshold, and complete tie rules before reading B scores. Do not skip an eligible higher-ranked candidate to obtain a preferred model narrative.

Illustrative freezing schema:

```yaml
schema_version: 1
selection_source: B01-B12
primary_metric: task_MASE7_market_fold_macro
selected_models:
  - rank: 1
    model_id: PENDING
    frozen_config_sha256: PENDING
  - rank: 2
    model_id: PENDING
    frozen_config_sha256: PENDING
  - rank: 3
    model_id: PENDING
    frozen_config_sha256: PENDING
selection_table_sha256: PENDING
track1_code_commit: PENDING
frozen_at_utc: PENDING
outer_targets_seen: false
```

`PENDING` values are archival schema placeholders. They are not current selected members or a completed configuration. Set `outer_targets_seen` from the actual development history, not by copying the example.

### 3.6 Recorded freeze (2026-10-05)

The B-group top three are frozen in `track2/configs/selected_track2_models.yaml`: M1 Granite TimeSeries PatchTST-FM (GRA01, B MASE(7) 0.927, 324/324), M2 Chronos-2 (CH01, 0.982), M3 TimesFM 3.0 (TFM01, 0.989). Moirai is rank 4 at 1.054 and was not selected. No tie tolerance was registered before scoring. Membership does not depend on one: rank 3 and rank 4 differ by 0.065. Only the M2/M3 order (difference 0.007) could, and the MASE(7) order is kept.

The B ranking file was written at 2026-10-03T09:03:48Z, about 2.5 minutes before the O01–O33 targets were first read for the Track 1 main table. This freeze file was written afterwards, with `outer_targets_seen: true`. The outer top three are the same set. Track 2 results on O01–O33 are exploratory. The confirmatory window is the prospective period O34–O36, which is not used to choose members, weights, residual models, or fallback thresholds. Dates after that window are deployment forecasts and are not scored before their actuals exist.

### 3.4 Why three members?

The proposed rationale is local B evidence rather than model size/reputation; bounded compute for combination/residual/calibration/coordination comparisons; enough members to compare one-, two-, and three-model systems and survive one failure; and fewer meta-parameters with only 12 B origins. B-only ablations of Top-1/2/3/4/7 could assess marginal membership gains if actually run and declared.

Report error correlations, prediction disagreement, and overlapping failures to interpret combination. These diagnostics do not override the accuracy-based member-selection rule.

### 3.5 Conditional examples

The original attachment illustrated TimesFM 3.0, Chronos-2, and Lag-Llama. They were examples, not a verified top-three result. Each becomes a selected member only if it meets B ranking and acceptance rules. Their potential roles are long-context/quantile forecasting, an independent probabilistic member or residual candidate, and sample-based forecasting, respectively. Architecture can explain complementarity but cannot substitute for B evidence. If other members qualify, use those actually selected; probability outputs can come from their distributions or a separately specified calibration method.

## 4. Forecast-production layers

### 4.1 Independent numerical candidates

```text
Origin-isolated history ─┬─> member 1 ─┐
                         ├─> member 2 ─┼─> output checks ─> deterministic ensemble
                         └─> member 3 ─┘
```

All members use the same origin boundary. Each may retain its Track 1 frozen context, with actual `effective_context_days` recorded.

```yaml
task_id: string
model_id: string
origin_date: YYYY-MM-DD
horizon_days: 28
prediction_raw: [28 values]
prediction_nonnegative: [28 values]
point_source: q0.5_or_sample_median_or_native_point
quantiles:
  q_0_1: optional
  q_0_5: optional
  q_0_9: optional
effective_context_days: integer
input_sha256: string
model_revision: string
warnings: []
```

### 4.2 Deterministic combination

| ID | Method | Purpose |
|---|---|---|
| T2-E0 | Best B-ranked single member | Minimum comparison |
| T2-E1 | Equal weights on members 1–2 | Increment from a second member |
| T2-E2 | Equal weights on members 1–3 | Simple three-member comparison |
| T2-E3 | Inverse B-MASE weights | Greater weight for better B performance |
| T2-E4 | Nonnegative least-squares weights summing to one | Interpretable fitted combination |

Equal weights:

\[
\hat y^{EW}_{i,h}=\frac{1}{3}\sum_{m=1}^{3}\hat y^{(m)}_{i,h}.
\]

B-only inverse-error weights:

\[
w_{m,h}=\frac{1/(E_{m,h}+\epsilon)}{\sum_{j=1}^{3}1/(E_{j,h}+\epsilon)},
\qquad
\hat y^{ENS}_{i,h}=\sum_{m=1}^{3}w_{m,h}\hat y^{(m)}_{i,h}.
\]

The first design prefers one fixed weight vector across horizons. Activate horizon-specific weights only if B cross-fitting supports a stable improvement under declared selection rules. A complex neural gating model is outside this main design.

### 4.3 Optional out-of-sample residual correction

Construct residuals from B cross-fitted ensemble predictions:

\[
e_{i,h}=y_{i,h}-\hat y^{ENS}_{i,h}.
\]

The correction model may read only residual history available at the issue time:

\[
\hat y^{RES}_{i,h}=\hat y^{ENS}_{i,h}+\lambda_h\hat e_{i,h},
\qquad 0\leq\lambda_h\leq1.
\]

Residual-model candidates are restricted to selected members capable of processing the residual series through the common univariate interface. B cross-fitting selects model/context/shrinkage. If no stable gain is established, freeze `lambda=0` and retain the original ensemble.

Requirements:

- Use out-of-sample prediction residuals, not in-sample fitting residuals.
- Prespecify a correction cap, for example 25% of the absolute uncorrected forecast; an example cap is not an inferred final setting.
- Save the uncorrected forecast, raw correction, shrunken correction, and clipped final result.
- Inspect high- and low-demand correction behavior separately.
- Use `lambda=0` when residual history is insufficient.

### 4.4 Probabilistic forecasts and calibration

If members supply matching quantiles, the design permits frozen-weight quantile aggregation followed by monotonicity correction. Otherwise, use B out-of-sample residuals for a declared split-conformal or horizon-specific empirical calibration. These approaches require their own coverage interpretation under temporal dependence.

Retain q10/q50/q90 and report pinball loss, WIS, 80% coverage, and average width where implemented. Availability of probabilistic outputs does not remove otherwise eligible point-only models from member selection.

## 5. B-group cross-fitting and final freeze

Avoid fitting and evaluating combination weights on the same forecast path:

1. At B origin `Bk`, learn meta-parameters only from `B01...B(k-1)`.
2. Generate an out-of-sample meta-forecast for `Bk`.
3. Use equal weights and `lambda=0` while earlier B history is insufficient.
4. Compare evaluable cross-fitted outcomes to select T2-E2/E3/E4.
5. Refit the chosen method on all B01–B12.
6. Freeze member, ensemble, and fallback configurations.
7. Only then evaluate O01–O33 under the original confirmatory design.

Do not randomly split markets to break temporal order. Main weights are shared across Macau markets to constrain degrees of freedom; any grouped-market weights belong to a separately labeled sensitivity.

The final record includes all member config/source/checkpoint/image identifiers; B forecast hashes; method/weights/epsilon/horizon scope; residual model/context/shrinkage/cap; quantile/calibration settings; fallback order; cross-fit/selection hashes; and the truthful outer-inspection status.

## 6. Coordinator design

### 6.1 Components

```text
Coordinator
  ├── data-audit tool
  ├── member-1 forecasting tool
  ├── member-2 forecasting tool
  ├── member-3 forecasting tool
  ├── deterministic ensemble/residual tool
  ├── output-quality tool
  └── audit/explanation tool
```

The original protocol called a common controller plus a frozen time-series adapter and structured messages a forecasting agent. Numerical adapters remain tools; this engineering label does not imply independent language reasoning. Members run independently and do not consume another member's unvalidated output.

### 6.2 Allowed and prohibited operations

A coordinator may validate task/origin/history/hash/version, schedule model tools, check length/dates/finite values/nonnegativity/quantile order, invoke frozen combination/residual functions, retry or downgrade under the frozen policy, and explain results using actual logs.

It may not read outer targets; change models or weights based on outer actuals; invent covariates; manually rewrite predictions/quantiles/metrics; select an apparently reasonable model outside the policy; or suppress failures, retries, fallback, or clipping.

### 6.3 Fixed fallback

```text
Frozen three-member ensemble
  -> if a member fails: renormalize frozen weights over available members
  -> if valid weights are unavailable: equal-weight available members
  -> if all time-series members fail: Seasonal Naive
```

A residual-module failure disables that module without discarding a valid ensemble. Record original statuses, fault class, attempts, final route, and effect on the delivered 28-day path.

### 6.4 Coordinator comparisons

| ID | System | Numerical policy | Purpose |
|---|---|---|---|
| T2-A0 | Direct script | Frozen ensemble and fallback | Engineering reference |
| T2-A1 | Deterministic workflow coordinator | Same as T2-A0 | Scheduling/logging/recovery |
| T2-A2 | General-purpose LLM coordinator | Same as T2-A0 | Task interpretation, fault explanation, and tool coordination |

On identical valid inputs with no failures, numerical predictions should be identical. Evaluate any coordination value through task success, fault detection/recovery, measured active labor, call counts, latency, and costs. A history-feature router that changes member choice is a separately declared dynamic-routing experiment. Later Qwen weight judgment likewise requires its own comparison definition.

### 6.5 Two controls

| ID | System | Composition | Question |
|---|---|---|---|
| T2-C1 | Baseline ensemble | Equal weight of the top three Track 1 baselines by B MASE(7): ETS (ETS03), Ridge (RID02), ElasticNet (EN03), using frozen Track 1 parameters. CPU only. B MASE(7) 1.047, 1.093, 1.102. | Does a Track 2 gain come from ensembling itself, or from the foundation models? |
| T2-C2 | Single model plus deterministic fallback | M1 alone. If the section 6.6 quality checks fail, fall back to Seasonal Naive. No agent. | Does a reliability gain come from the fallback rule, or from multi-agent coordination? |

T2-C2 and T2-A1 share the same quality checks. The only difference is whether more than one model and a coordinator are present.

### 6.6 Lightweight fault injection (secondary)

In normal runs, T2-A0, T2-A1, and T2-A2 share the same numerical forecast; the agent layer does not change accuracy. The completed Track 1 runs (744 origin runs, 576 B-search tasks, 198 outer reruns, and 13×957 baseline tasks) recorded no hard failures. Fault injection measures whether the system can still deliver an acceptable forecast when something goes wrong. It is an evaluation, not an accuracy improvement, and is reported as a secondary experiment. It runs only on Macau O01–O33 and is exploratory. The prospective window is not fault-injected.

Replay archived failures first, then add constructed faults:

| ID | Type | Source | Injection |
|---|---|---|---|
| R1 | Level collapse (silent) | Informer default INF01: about 58% of B01–B09 forecasts fell below 10% of the recent 28-day level | Replace that member's forecast with the archived INF01 output |
| R2 | Near-zero forecast (silent) | Lag-Llama smoke: 20 of 27 markets had a near-zero mean point forecast | Replace with the archived smoke output |
| R3 | Import clash (error) | `track1/code/models.py` shadowed the upstream `models` package before commit `e5f49d4` | Restore the conflicting import path at container start |
| R4 | Checkpoint argument mismatch (error) | Same pre-`e5f49d4` checkpoint loading failure | Load weights with the pre-fix arguments |
| S1 | Timeout | Constructed | Member inference exceeds the frozen timeout |
| S2 | Non-finite value | Constructed | Write NaN or inf on 1–3 horizons |
| S3 | Missing day | Constructed | Emit 27 values |
| S4 | Wrong origin | Constructed | History ends after the forecast origin |
| S5 | Out of memory | Constructed | Member raises CUDA OOM |

Quality checks are frozen before any injection, using clean B output only: exactly 28 consecutive finite dates, origin aligned with the history end, and versions matching the frozen config; the ratio of the 28-day forecast mean to the previous 28-day actual mean must lie between the 0.5% and 99.5% percentiles of that ratio on the three members' clean B output (skip tasks whose previous 28-day mean is below 5 visitors); timeout is 3× that member's p95 single-origin wall time from the Track 1 timing test.

Compare T2-A0′ (M1 script, no checks), T2-C2, T2-A1, and T2-A2 on the same fault schedule at 1%, 5%, and 10% task-level fault rates. Draw fault types uniformly, with R1–R4 together half of the draws. Seed 20261005. Save the schedule SHA256 before the run. Tasks without a fault reuse sealed forecasts.

Auxiliary ensemble metrics, not used for selection: win rate versus M1, mean MASE(7) of the worst 10% of tasks and the maximum task MASE(7), standard deviation of the 33 origin-level macro MASE(7) values, fraction of the gap from M1 to the per-task best member that the ensemble captures, and the member error-correlation matrix.

Cost uses the Track 1 volume-breakeven price basis (`track1/configs/volume_breakeven_plan.yaml`): RunPod Secure Cloud A6000, priced from the isolated timing test, summing the three members plus coordination. T2-A2 also records LLM call counts, input and output tokens, and API cost on the same price date. Report the incremental cost per 1,000 tasks against the outer MASE(7) change along Seasonal Naive → ETS → T2-C1 → M1 → T2-C2 → T2-E2/E3/E4 → T2-05 → T2-A1 → T2-A2.

## 7. Outer evaluation

### Execution

For each Macau O01–O33 task, read its isolated pre-origin history, generate and seal the three candidate outputs, apply frozen combination/residual configurations, close the prediction environment, and only then join targets in the evaluator. Hong Kong uses the frozen Macau choices and thresholds without reselection; history sensitivities and A stress tasks are separate. Report actual context limits when `all` exceeds a model's supported context.

### Metrics

Primary: task MASE(7), macro-averaged over markets and origins.

Proposed auxiliary outcomes: MAE, RMSSE(7), WAPE, bias, high-demand MASE/underprediction, q10/q50/q90 pinball loss, WIS, 80% coverage, and interval width. Genuine zeros rule out MAPE as the main metric. WAPE also requires a nonzero realized-total denominator; do not treat it as a universal repair for undefined MASE. Use common task keys and report coverage/failure/fallback rates. Current completed metrics are identified in the paper inventory, not inferred from this proposal.

### Resources and implementation burden

The intended records include loading/fitting, p50/p95 inference, full batch time, peak RAM/VRAM, model/image size, per-1,000-successful-task training/inference/API/storage/retry resources, agent/tool calls, failures/retries, measured active setup work, fallback counts, and human-intervention counts. A field absent from actual logs remains unmeasured.

### Statistical comparisons

Compare best single, equal-weight, learned-weight, and residual configurations through paired task losses. Preserve adjacent-origin dependence in block bootstrap, use appropriate robust variance for any DM test with overlapping paths, and report effect sizes, intervals, and coverage. Macau is the primary evaluation; Hong Kong is separate transfer evidence rather than a pooled mean.

## 8. Planned ablations and claim boundaries

| ID | Method | Question |
|---|---|---|
| T2-01 | Best selected member | Single-tool reference |
| T2-02 | Top-2 equal weights | Second-member increment |
| T2-03 | Top-3 equal weights | Third-member increment |
| T2-04 | B-learned weights | Increment beyond equal weights |
| T2-05 | T2-04 plus residual correction | Correction increment |
| T2-06 | T2-05 plus probability calibration | Interval performance |
| T2-07 | Deterministic coordination/fallback | Delivery and recovery |
| T2-08 | LLM coordination | System outcome changes with frozen numerical rules |

Learned-weight value requires a supported increment over equal weights; residual value requires an increment over the uncorrected combination; LLM-coordinator value requires an improvement in the actually measured system outcomes while preserving its allowed numerical rules. Fixed B-learned weights should not be called dynamically changing weights. Additional complexity does not itself establish superiority. Retain results favoring simple averaging or a single model.

## 9. Intended artifacts

The original proposal envisioned `track2/configs/`, selection/ensemble/fallback YAMLs, code for selection, cross-fitting, residual correction, calibration, coordination, and schemas; result directories for B cross-fit, Macau outer, Hong Kong extension, and pandemic stress; and independent evaluation outputs. This directory tree is a design contract, not a statement that the later server implementation is included here.

Each run should retain three candidate predictions/hashes; combined forecasts, weights, residual adjustments; manifest/time/resource logs; failures/fallbacks/tool calls; selections/frozen configs/commit; and truthful `actuals_visible_to_models` and `actuals_visible_to_coordinator` fields.

Weights, caches, credentials, and bulk prediction collections are not committed by default. Release authorized code/configuration, build instructions, checksums, small examples, and summaries with a precise inventory.

## 10. Historical implementation checklist

The following unchecked items preserve the original proposal; consult later evidence for actual completion.

### Stage 1: Track 1 freezing

- [ ] Complete and freeze the 13-baseline B validation.
- [ ] Smoke all seven models on the GPU server.
- [ ] Evaluate B candidates and freeze each configuration.
- [ ] Produce common-task B rankings, coverage, and resources.
- [ ] Apply the registered top-three rule and record selection.

### Stage 2: Deterministic Track 2

- [ ] Verify identical task/origin/date/horizon keys.
- [ ] Implement Top-1/2/3 and B temporal cross-fitting for inverse/constrained weights.
- [ ] Implement out-of-sample residuals, shrinkage, and `lambda=0`.
- [ ] Implement declared quantile monotonicity/calibration.
- [ ] Freeze combination/residual/probability/fallback settings and hashes.

### Stage 3: Coordination and fault injection

- [ ] Register the numerical adapters as forecasting tools.
- [ ] Implement structured messages, timeouts, retries, and idempotent task IDs.
- [ ] Verify the coordinator cannot access targets.
- [ ] Test declared faults such as timeout, NaN, missing day, wrong origin, or memory failure.
- [ ] Verify fixed-policy handling and audit logs.
- [ ] Verify identical normal-case numerical outputs across script, deterministic, and LLM coordination.

### Stage 4: Outer assessment

- [ ] Run Macau O01–O33 without changing frozen settings.
- [ ] Run separately labeled Hong Kong, history, and stress evaluations.
- [ ] Report measured accuracy/probability/resources/costs/failures/labor with coverage.
- [ ] Complete declared paired inference and ablations.
- [ ] Reproduce one shard and its audit chain on a clean GPU environment.

## 11. Interpretation of the original pending state

At the September design date, the framework could be specified but the three members had not yet been verified by the planned seven-model B selection. It was therefore not justified to call TimesFM 3.0, Chronos-2, and Lag-Llama the best three.

For a completed run, the selection statement must be grounded in actual evidence: eligible versions and target isolation, documented B task coverage, recorded B ranking and tie rules, freeze time, and truthful outer-inspection history. Correlation, probabilistic output, and resources explain combination behavior and fallback; they do not retroactively change the selection rule. The current release describes completed findings separately, without converting this archival pending schema into a fabricated final configuration.
