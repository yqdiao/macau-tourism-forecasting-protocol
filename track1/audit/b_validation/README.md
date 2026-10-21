# Baseline B-group validation and freezing audit

## Historical completion record

- Completion: 2026-09-26 11:38 (Asia/Shanghai)
- Validation folds: B01–B12
- Candidate configurations: 33
- Completed prediction paths: 12,636 / 12,636
- Terminal failed paths: 0
- Unresolved paths: 0
- Selection metric: 28-day task MASE(7), macro-averaged over markets, folds, and seeds
- Configuration status: `frozen_after_B_group_tuning`
- Model-code commit: `256ff6652c68bc418ac4c4d43c5d66610ebc991b`

## Frozen selections

| Model | Selected candidate | B-group MASE(7) | Mean fit seconds |
|---|---:|---:|---:|
| Seasonal Naive | SN01 | Fixed benchmark | 0.0000 |
| ETS | ETS03 | 1.047266 | 0.5076 |
| SARIMA | SAR03 | 1.200664 | 1.5655 |
| Ridge | RID02 | 1.093038 | 0.2325 |
| Linear Regression | LIN01 | 1.311860 | 0.2865 |
| Random Forest | RF03 | 1.258608 | 229.6868 |
| Gradient Boosting | GB01 | 1.262905 | 55.5233 |
| SVR | SVR02 | 1.154077 | 14.2630 |
| Lasso | LAS03 | 1.106368 | 1.3168 |
| Elastic Net | EN03 | 1.101505 | 1.8070 |
| Parrot Random Forest 800 | PRF01 | 1.259145 | 871.4802 |
| LightGBM | LGB01 | 1.254014 | 37.1744 |
| Transformer baseline | TR03 | 1.288447 | 56.1705 |

Seasonal Naive was fixed in advance and did not compete in candidate selection; no B-group candidate-selection score is reported for it here.

## Committed audit artifacts

- `validation_status.json`: completeness, task coverage, source hashes, and freezing status.
- `driver_status.json`: final local driver status and exit code.
- `candidate_summary.parquet`: summary of all 33 candidates.
- `frozen_baseline_configs.generated.bin`: original CRLF bytes written at freezing, retained for byte-level hash verification.
- `../../configs/frozen_baseline_configs.yaml`: final selections for the 13 baselines.

This directory does not contain all 12,636 forecast shards, runtime logs, or task-level metric files. `validation_status.json` records completion-time hashes for the candidate summary, task-level metrics, protected targets, and frozen configuration so that separately retained archives can be checked.

## Completion-time SHA256 values

| Artifact | SHA256 |
|---|---|
| B-group task list | `1074e3be580a1a2cea63aaa502ad0c2267a15ff2dc01e5c5f78e04f8fcce928d` |
| Candidate summary | `e51cab19c98f26257bd6db0bdd37ba02a01cedaa35e7a18fd6e244c0e8204cdf` |
| Task-level metrics | `e7eb223db9f4e2e95df3dc2b2c4a0f1da08b4a1a8513599411f9812b1d7898a9` |
| Protected targets | `1ec5dee35f3846c8a51fbe4d5faab9ac536a853944c84d43bbb56dffdc092715` |
| Original frozen bytes, `frozen_baseline_configs.generated.bin` | `1771fcb4d4605a139db8e79c0a6533679afa002bce0aa48289de7eaa65889595` |

The working YAML is normalized to LF under repository rules and is semantically equivalent to the original archive. Verify the completion-time byte hash against the `.bin` file; outer backtests use `track1/configs/frozen_baseline_configs.yaml`.

Outer predictors read the frozen configuration and historical inputs. Only the independent evaluator reads `targets.parquet`.
