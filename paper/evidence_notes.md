# Evidence notes and interpretation boundaries

## Main comparison

The main market table includes 15 configurations and 891 tasks per configuration. Granite has MASE 1.0943995801290252 and MAE 553.7993589730643; TimesFM 3.0 has mean task RMSE 811.2965639168455. These are different objectives. Other TSFMs and conventional baselines remain in the comparison. A stronger result for selected TSFMs is not a category-wide dominance claim.

LL01 is retained for descriptive comparison with an exploratory label. R3 revisions follow the prior inspection of outer results. Their later internal freeze does not turn them into a new independent confirmatory test of that same outer set.

## Same-candidate M2 comparison

The audited equal-weight comparator has all-history MASE 1.098528607427494, MAE 542.2917039196876 and mean task RMSE 803.8740200863232. P3-R has 1.0983099936612446, 542.5960343202288 and 805.7134831867562, respectively. The MASE difference is about -0.000218614, a relative reduction of about 0.01990%; its paired 95% interval spans zero. The comparison preserves the additional request workload.

M2 candidate forecasts were matched to the original TSFM outputs during the audit. Equivalent complete candidate-value matching is not claimed here for all M1, M3 and M4 inputs. M2's same-candidate incremental finding should not be assigned to every chain strategy in M1.

M1 and M2 differ in candidate history while their LLM recent-history prompts both contain 180 days. Differences between the modules are not an isolated effect of adding older observations to the LLM prompt.

## Pressure evaluation

Under the 180-day seasonal scale, 29 of 351 tasks have undefined MASE; MAE and RMSE still cover all 351. The shared zero-scale tasks are retained as explicit subgroups. WAPE is also undefined for 123 tasks with zero future totals in this pressure evaluation. Subgroup coverage is part of the result, not a missing row to discard silently.

## Direct 4B and abnormal output

The complete 4B-thinking evaluation includes one extreme generated output in the Hong Kong source-market task with origin 2024-03-24 (`a66eb1bfd3810b8dd562c918`). One raw repeated delivery reached approximately 1.672e15 despite being returned as a successful call. The full-sample error includes that output.

The matched-890 sensitivity table removes that entire task from both 4B and seasonal naive, across repeats. It does not replace the full-sample result. The task-level diagnostic columns in this release average repeated-run diagnostics and should not be presented as the raw single-call maximum.

The 4B direct and 8B weighting settings change the backbone size, thinking configuration and output permission together. Their difference does not isolate a model-size, thinking, or collaboration effect.

## Reliability and resource use

The original fault experiment concerns explicitly constructed faults. Its delivery gains do not estimate real production availability. Logical parallel-role designs were executed serially by the local 8B service, so a strategy name is not a latency claim.

The resource figure preserves conditional accounting. Request duration is not synonymous with invoice cost, energy use, GPU kernel occupancy or the entire lifecycle cost. This release does not report realised business profit or monetise forecast errors without an observed operational loss function.

## Selected figures

- `td_accuracy_cost_joint.pdf`: three metric-specific TSFM accuracy-resource panels and the M2 incremental workload panel; its numerical inputs are released.
- `td_model_workflow_8b.pdf`: predictor fitting, frozen numerical inference and LLM usage roles.
- `td_m14_scope_coverage.pdf`: stress metric coverage and the Hong Kong extension.
- `td_m14_history_increment.pdf`: candidate-history bridge and same-candidate incremental comparisons.
- `td_group_original_track2.pdf`: original combination and constructed-fault findings. This export is retained as published evidence; the full original fault simulator and all raw fault outputs are not added by this release.

The figures use English text. Except for the joint resource plot, the release preserves the audited rendered exports rather than claiming to provide a self-contained figure-generation pipeline for every panel.
