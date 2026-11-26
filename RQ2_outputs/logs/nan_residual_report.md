# RQ2 NaN / gap residual report
After fill_and_recompute

## S10 coverage
```
model            chronos_2  granite_patchtst_fm  lag_llama  moirai  timesfm_3
split_name                                                                   
outer_backtest           5                    5          5       5          5
pandemic_stress          5                    5          5       5          5
```

## S5 models
GL03, LL01, NL01, chronos_2, ets, granite_patchtst_fm, lag_llama, lightgbm, moirai, sarima, seasonal_naive, svr, timesfm_3, transformer

## Actionable scan

table_S10_prediction_interval_80_coverage_and_wis.csv: OK
table_S11_hongkong_validation_and_macau_control.csv: OK
table_S12_exogenous_information_ablation_4metrics.csv: null_cells=4 {'seconds_per_vector': 1, 'MAE_extended': 3}
table_S13_rq2_boundary_summary_matrix.csv: OK
table_S1_regime_comparison.csv: OK
table_S2_stress_stage_market_decomposition.csv: OK
table_S3_metric_concordance_matrix.csv: OK
table_S4_market_decomposition_and_spectrum.csv: structural multi-section binder
table_S5_history_length_and_denominator_control.csv: OK
table_S6_market_size_groups_4metrics.csv: OK
table_S7_top10_markets_bootstrap_winner_stability.csv: OK
table_S8_horizon_weekly_blocks_and_same_weekday.csv: metric_nulls=0; SameWeekday R_* nulls expected=56
table_S9_high_demand_holiday_underprediction_risk.csv: OK
