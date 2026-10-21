# Data dictionary

## Identifiers and scope

- `model` / `model_id`: configuration label; a label is not a claim of a novel architecture.
- `experiment`: audit module. M1 is the 180-day Macao O configuration; M2 is the all-history numeric-candidate bridge; M3 is the pressure-window evaluation; M4 is the Hong Kong extension; M5 is the existing 4B-thinking direct-forecast evaluation.
- `strategy`: S1 is one generic weight decision; S3-I averages three independent generic weight decisions; S3-R uses a generic chain's last decision; P3-R averages independent role decisions; C3-R uses the role chain's last decision; C3-off removes upstream messages and uses the last decision. Strategy labels do not imply that roles were physically executed simultaneously.
- `market`, `market_id`: public geographic market code, preserved across audit tables.
- `task_id`: deterministic task identifier; no credential or user identifier.
- `origin`: last observation date available at the forecast origin.
- `split`: `recovery_tuning` is the 324-task B selection set; `outer_backtest` is the 891-task O set; `pandemic_stress` is the 351-task A set.
- `scope`: `full` retains all delivered tasks. `matched_890_excluding_extreme_task_sensitivity` removes one extreme 4B task from both 4B and its comparator, only as sensitivity analysis. Zero/positive-scale scopes partition the stress tasks.
- `selection_identity`: distinguishes original selected configurations from post-outer exploratory configurations. Original selected configurations do not all share the same information set or training procedure.

## Accuracy fields

- `mae`: 28-day mean absolute error, in arrivals/day.
- `mse`: 28-day mean squared error, averaged over repeats before summary-level pooling if used.
- `rmse`: square root of a delivered task's mean squared error; the released task field averages these errors across repeats. Aggregate `rmse_market_equal` averages task RMSE within each market and then across markets.
- `mase` / `mase_evaluation`: task MAE divided by the historical mean absolute 7-day seasonal difference for the explicitly named evaluation history.
- `mase_all_reference`: the same delivered errors evaluated against the all-history scale. This is not another model run.
- `q_evaluation`, `q_all`, `q180`: historical scaling denominators. A zero denominator leaves the relevant MASE undefined.
- `wape`: task absolute-error sum divided by task actual sum; undefined when that sum is zero.
- `actual_sum`, `abs_error_sum`: task-level sums across forecast days, averaged across repeated deliveries where applicable.
- `max_abs_error`, `max_forecast`, `max_actual`: source audit task diagnostics, already averaged over repeats in the task table. They are not necessarily the maximum of all raw repeated-run records.
- `*_market_equal`: mean of within-market means, excluding undefined values for the relevant metric.
- `*_task_equal`: arithmetic mean over defined task values. It may differ from market-equal aggregation after unequal task exclusions.
- `rmse_pooled_task_equal`: square root of mean task MSE; distinct from mean task RMSE.
- `zero_scale_tasks`, `valid_mase_tasks`, `zero_actual_sum_tasks`, `valid_task_wape_tasks`: explicit metric coverage counts.

## Paired comparisons

`difference_first_minus_second` is candidate minus comparator error. Negative values favour the first entry. `ci95_lower` and `ci95_upper` retain the existing audit's 4000-repetition circular-origin-block bootstrap with block length 3, keeping markets together and averaging repeated-run errors first. The release script validates point differences for the headline M2 comparison; it does not regenerate those interval draws. An interval spanning zero does not establish statistical equivalence.

## Resource fields

- `all_sent_seconds`: sum of recorded durations for requests sent, including applicable failed requests.
- `deliveries`: delivered repeated runs, not the count of unique tasks.
- `additional_call_hours_per_1000_deliveries`: duration coefficient before applying a local price.
- `relative_registered_charge_chronos2_1`: within-group registered charge divided by the Chronos-2 reference charge; based on rounded historical registration entries.
- `conditional_nondominated`: point-estimate nondominance among the five TSFMs in that panel and under that panel's common accounting assumptions; not a significance test.
- R3 `fit_seconds_sum` and `inference_seconds_sum`: partial recorded CPU times. They do not cover a complete deployment lifecycle.

## Display selection and ranks

The top-ten market subset uses cumulative observed arrivals during 2017-01-01–2026-07-31. It is a retrospective descriptive display choice. Main inference continues to use all 27 markets. Ranks are ascending point-estimate ranks among the listed configurations; ties use average ranks at stored precision. Neither bolding nor a rank is a significance claim.
