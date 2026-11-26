# RQ2 history post-processing rerun note

**When:** 2026-11-17T05:48:56+00:00

**Failure:** `fill_and_recompute.log` ended with wrong path  
`/root/DYQ/macau-tourism-forecasting-protocol/m423_history.py` (ENOENT).  
Correct script is `RQ2/m423_history.py`.

**State before rerun:**
- `RQ2_outputs/raw/history_task_metrics.parquet` already expanded (37,422 rows, 14 models × 3 history settings) — no rebuild needed.
- `table_S5_…csv` stale (24 rows / 8 models from earlier 21,384-row panel).
- figure7 + summary still reflected stale S5.

**Commands run:**
```bash
cd /root/DYQ/macau-tourism-forecasting-protocol
PYTHONPATH=RQ2 /root/DYQ/macau-tourism-world-model/.venv-server/bin/python RQ2/m423_history.py
# fig7 only via plot_figures.fig7
PYTHONPATH=RQ2 …/python -c '… fig7(s5) …'
PYTHONPATH=RQ2 …/python RQ2/write_summary.py
```

**Checks:**
- Seasonal Naive MAE range across histories = 0 (identity holds); MAPE/RMSE/MASE_fixed also identical; `pct_change_vs_Full_MAE` = 0.
- S5: 42 rows, 14 models, all `status=ok`; models match history parquet.
- figure7 PNG/PDF and `4.2_RQ2_实验数据核对汇总稿.md` refreshed (summary S5 section: 行数=42).

**Outputs:**
- `RQ2_outputs/tables/table_S5_history_length_and_denominator_control.csv`
- `RQ2_outputs/figures/figure7_history_length_and_covid_contamination.{png,pdf}`
- `RQ2_outputs/4.2_RQ2_实验数据核对汇总稿.md`
