# Set B preselection audit

## Source

- NL01 and GL03 Set B daily forecasts: **recovered on this server** (no Phase-2 rerun).
- NL01: `track1/results/light_models/recovery_tuning/global_linear/NL01` — 9072 rows, 324 tasks, targets 2023-01-08–2023-12-09.
- GL03: `track1/results/light_models/recovery_tuning/global_lightgbm/GL03` — 9072 rows, 324 tasks, targets 2023-01-08–2023-12-09.
- Foundation Set B: read from `track1/artifacts/predictions/foundation_b_default/{granite_patchtst_fm,timesfm_3,chronos_2}.tar.gz` (local unpack under `extracted_foundation_b/`, gitignored).
- Combinations use NL01 **seed42** (same as RQ outer panels).

### Reproduce

```bash
mkdir -p outputs/set_b_preselection_audit/extracted_foundation_b
for m in granite_patchtst_fm timesfm_3 chronos_2; do
  tar -xzf track1/artifacts/predictions/foundation_b_default/${m}.tar.gz \
    -C outputs/set_b_preselection_audit/extracted_foundation_b
done
python outputs/set_b_preselection_audit/run_set_b_preselection_audit.py
```

Requires local `track1/results/light_models/recovery_tuning/` predictions (not carried in Git).

## Single-model repro vs archived Set B scores

```
model scoring                     recomputed_MASE  archived_MASE  delta_MASE  recomputed_MAE  note
NL01  seed42_only                        1.071485       1.071485    0.000000      655.385274  Exact match to light_models_b task_metrics seed=42
NL01  seed_average_task_metrics          1.072660       1.072660    0.000000      650.344452  Exact match to candidate_summary (3 seeds)
GL03  seed42                             1.115448       1.115448    0.000000      765.582798  Exact match to archived Set B score
```

Phase 2 was **not** executed (predictions recovered on server).

## Set B fixed combinations

```
          scheme_id        kind                                 members     MASE        MAE       RMSE  n_tasks
granite_patchtst_fm      single                     granite_patchtst_fm 0.927490 545.574405 721.734739      324
          timesfm_3      single                               timesfm_3 0.989163 576.399860 756.117252      324
          chronos_2      single                               chronos_2 0.981828 566.531205 747.494017      324
               NL01      single                                    NL01 1.071485 655.385274 844.614388      324
               GL03      single                                    GL03 1.115448 765.582798 938.043661      324
                 C1 combo_equal      granite_patchtst_fm,timesfm_3,NL01 0.960076 570.841744 752.820210      324
                 C2 combo_equal      granite_patchtst_fm,timesfm_3,GL03 0.955629 564.346904 743.912197      324
                 C3 combo_equal      granite_patchtst_fm,chronos_2,NL01 0.958511 570.087190 751.047053      324
                 C4 combo_equal      granite_patchtst_fm,chronos_2,GL03 0.950516 563.233481 742.395504      324
                 C5 combo_equal                timesfm_3,chronos_2,NL01 0.984128 580.134782 762.840171      324
                 C6 combo_equal                timesfm_3,chronos_2,GL03 0.977763 575.885356 757.471919      324
                 C7 combo_equal           granite_patchtst_fm,NL01,GL03 0.973945 585.170895 769.899097      324
                 C8 combo_equal                     timesfm_3,NL01,GL03 1.000620 596.121374 783.775087      324
                 C9 combo_equal                     chronos_2,NL01,GL03 0.998541 597.976238 783.848186      324
              B_3FM combo_equal granite_patchtst_fm,timesfm_3,chronos_2 0.942893 554.771192 733.650761      324
```

## Preselection winners on Set B

```json
{
  "best_C_by_MASE": {
    "scheme_id": "C4",
    "members": "granite_patchtst_fm,chronos_2,GL03",
    "MASE": 0.950516237710012,
    "MAE": 563.233481209494,
    "RMSE": 742.3955040355942
  },
  "best_C_by_MAE": {
    "scheme_id": "C4",
    "members": "granite_patchtst_fm,chronos_2,GL03",
    "MASE": 0.950516237710012,
    "MAE": 563.233481209494,
    "RMSE": 742.3955040355942
  },
  "best_all_fixed_by_MASE": {
    "scheme_id": "B_3FM",
    "members": "granite_patchtst_fm,timesfm_3,chronos_2",
    "MASE": 0.9428929112785126,
    "MAE": 554.7711918566383,
    "RMSE": 733.6507608103856
  },
  "best_all_fixed_by_MAE": {
    "scheme_id": "B_3FM",
    "members": "granite_patchtst_fm,timesfm_3,chronos_2",
    "MASE": 0.9428929112785126,
    "MAE": 554.7711918566383,
    "RMSE": 733.6507608103856
  }
}
```

## Mapped Set O performance (sealed S14/S15; not rerun)

```
         scheme_id     MASE        MAE       RMSE  n_tasks               role               period                source
                C4 1.136127 555.999104 813.707757    891.0        best_C_MASE Set_O_outer_backtest RQ3_outputs/table_S14
                C4 1.136127 555.999104 813.707757    891.0         best_C_MAE Set_O_outer_backtest RQ3_outputs/table_S14
             B_3FM 1.098529 542.291704 803.874020    891.0      best_all_MASE Set_O_outer_backtest RQ3_outputs/table_S14
             B_3FM 1.098529 542.291704 803.874020    891.0       best_all_MAE Set_O_outer_backtest RQ3_outputs/table_S14
             B_3FM 1.098529 542.291704 803.874020    891.0              B_3FM Set_O_outer_backtest RQ3_outputs/table_S14
                C2 1.126881 553.853747 800.978190    891.0                 C2 Set_O_outer_backtest RQ3_outputs/table_S14
      G1_Stat_Stat 1.151930 581.390815 829.116302      NaN       G1_Stat_Stat Set_O_outer_backtest RQ3_outputs/table_S15
G2_LLM_Stat_eta0.3 1.141050 566.287753 815.068596      NaN G2_LLM_Stat_eta0.3 Set_O_outer_backtest RQ3_outputs/table_S15
```
