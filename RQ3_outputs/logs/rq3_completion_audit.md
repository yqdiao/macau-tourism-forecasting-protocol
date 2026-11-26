# RQ3 完成度核对报告

- 自动检查：**74/75** 通过
- 失败项：**1**

## 失败明细

| ok    | item        | detail                                   |
|:------|:------------|:-----------------------------------------|
| False | S18:schemes | missing=['B_3FM_equal', 'B_5Pool_equal'] |

## 相对原需求的缺口/部分完成项

- S16缺少平日/法定假日分层选模比例（仅有大中小市场与时期）
- S15未单列相对ETS的DM（有相对Granite/G1/B3FM）
- 冷启动warm_start（内层选择集先验）未作为对照导出（主回放为等权C2）
- LLM为信息集对齐proxy（非真实Qwen3-4B推理；因GPU驱动不可用）
- S14未报告相对G1_Stat_Stat的DM（该对比主要在S15）

## 全量检查明细

| ok    | item                                                           | detail                                                            |
|:------|:---------------------------------------------------------------|:------------------------------------------------------------------|
| True  | file:table_S14_fixed_combinations_benchmark.csv                | rows=38                                                           |
| True  | file:table_S15_dynamic_selection_and_weighting_ablation.csv    | rows=30                                                           |
| True  | file:table_S16_subset_switching_and_market_heterogeneity.csv   | rows=176                                                          |
| True  | file:table_S17_hyperparameter_sensitivity_grid.csv             | rows=17                                                           |
| True  | file:table_S18_mode1_vs_mode2_compute_cost_and_reliability.csv | rows=8                                                            |
| True  | fig:figure11_hybrid_structures_and_2x2_ablation.pdf            | 28728B                                                            |
| True  | fig:figure11_hybrid_structures_and_2x2_ablation.png            | 336337B                                                           |
| True  | fig:figure12_mode1_mode2_cost_accuracy_frontier.pdf            | 23264B                                                            |
| True  | fig:figure12_mode1_mode2_cost_accuracy_frontier.png            | 172292B                                                           |
| True  | summary_md                                                     |                                                                   |
| True  | llm_cache_jsonl                                                | lines=15444                                                       |
| True  | decisions_parquet                                              |                                                                   |
| True  | baseline_log                                                   |                                                                   |
| True  | S14:outer_backtest:5_singles                                   | ['GL03', 'NL01', 'chronos_2', 'granite_patchtst_fm', 'timesfm_3'] |
| True  | S14:outer_backtest:C1-9+B                                      | missing=[]                                                        |
| True  | S14:outer_backtest:B3FM_StatWeight                             |                                                                   |
| True  | S14:outer_backtest:group_means                                 |                                                                   |
| True  | S14:pandemic_stress:5_singles                                  | ['GL03', 'NL01', 'chronos_2', 'granite_patchtst_fm', 'timesfm_3'] |
| True  | S14:pandemic_stress:C1-9+B                                     | missing=[]                                                        |
| True  | S14:pandemic_stress:B3FM_StatWeight                            |                                                                   |
| True  | S14:pandemic_stress:group_means                                |                                                                   |
| True  | S14:error_corr_dispersion                                      |                                                                   |
| True  | baseline_vs_table3                                             | max_abs=2.274e-13                                                 |
| True  | S14_align_granite_patchtst_fm_MASE                             | diff=0.000e+00                                                    |
| True  | S14_align_granite_patchtst_fm_MAPE                             | diff=0.000e+00                                                    |
| True  | S14_align_granite_patchtst_fm_MAE                              | diff=0.000e+00                                                    |
| True  | S14_align_granite_patchtst_fm_RMSE                             | diff=2.274e-13                                                    |
| True  | S14_align_timesfm_3_MASE                                       | diff=0.000e+00                                                    |
| True  | S14_align_timesfm_3_MAPE                                       | diff=0.000e+00                                                    |
| True  | S14_align_timesfm_3_MAE                                        | diff=0.000e+00                                                    |
| True  | S14_align_timesfm_3_RMSE                                       | diff=1.137e-13                                                    |
| True  | S14_align_chronos_2_MASE                                       | diff=0.000e+00                                                    |
| True  | S14_align_chronos_2_MAPE                                       | diff=0.000e+00                                                    |
| True  | S14_align_chronos_2_MAE                                        | diff=0.000e+00                                                    |
| True  | S14_align_chronos_2_RMSE                                       | diff=0.000e+00                                                    |
| True  | S14_align_NL01_MASE                                            | diff=0.000e+00                                                    |
| True  | S14_align_NL01_MAPE                                            | diff=1.421e-14                                                    |
| True  | S14_align_NL01_MAE                                             | diff=2.274e-13                                                    |
| True  | S14_align_NL01_RMSE                                            | diff=0.000e+00                                                    |
| True  | S14_align_GL03_MASE                                            | diff=4.441e-16                                                    |
| True  | S14_align_GL03_MAPE                                            | diff=0.000e+00                                                    |
| True  | S14_align_GL03_MAE                                             | diff=0.000e+00                                                    |
| True  | S14_align_GL03_RMSE                                            | diff=0.000e+00                                                    |
| True  | S15:reopen_schemes                                             | missing=[]                                                        |
| True  | S15:stress_G_schemes                                           | missing=[]                                                        |
| True  | S15:col:DM_MASE_vs_Granite                                     |                                                                   |
| True  | S15:col:p_MASE_vs_Granite                                      |                                                                   |
| True  | S15:col:DM_MASE_vs_G1                                          |                                                                   |
| True  | S15:col:R_MASE_vs_Granite                                      |                                                                   |
| True  | S16:cols                                                       |                                                                   |
| True  | S16:G1_activation                                              |                                                                   |
| True  | S16:market_groups                                              | ['ALL', 'Large', 'Medium', 'Small']                               |
| True  | S17:grids                                                      | ['delta', 'driver', 'lambda_alpha', 'refresh_K']                  |
| True  | S17:lambda_alpha_full                                          | n=9                                                               |
| True  | S17:delta                                                      |                                                                   |
| True  | S17:driver                                                     |                                                                   |
| True  | S17:K                                                          |                                                                   |
| False | S18:schemes                                                    | missing=['B_3FM_equal', 'B_5Pool_equal']                          |
| True  | S18:col:gpu_seconds_eval_reopen                                |                                                                   |
| True  | S18:col:cpu_seconds_eval_reopen                                |                                                                   |
| True  | S18:col:cost_ratio_vs_ETS_S1                                   |                                                                   |
| True  | S18:col:cost_ratio_vs_ETS_S2                                   |                                                                   |
| True  | S18:col:cost_ratio_vs_ETS_S3                                   |                                                                   |
| True  | S18:col:annual_usd_S1_N27                                      |                                                                   |
| True  | S18:col:annual_usd_S1_N1000                                    |                                                                   |
| True  | S18:col:annual_usd_S1_N10000                                   |                                                                   |
| True  | weights_sum_to_1                                               | max_abs=2.220e-16                                                 |
| True  | weights_nonneg                                                 |                                                                   |
| True  | cold_start_counts                                              | {'outer_backtest': 297, 'pandemic_stress': 297}                   |
| True  | cold_only_first_origin:outer_backtest                          | n_cold=27                                                         |
| True  | cold_only_first_origin:pandemic_stress                         | n_cold=27                                                         |
| True  | mode1_mode2_same_accuracy_G1                                   | 1.1519301032107954 vs 1.1519301032107954                          |
| True  | LL01_appendix                                                  |                                                                   |
| True  | log_RQ3_DONE                                                   |                                                                   |
| True  | log_no_traceback                                               |                                                                   |
