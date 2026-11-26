# RQ2 NaN inventory (post `rq2-fill`)

**As of:** 2026-11-17T06:06:57+00:00 (`[fill done]` in `fill_and_recompute.log`)  
**Pipeline:** tmux `rq2-fill` relaunch 06:02 → history expand → m423 (S5) → m424 (S6–S10) → m425 → plots → summary.  
**Do not invent values** for remaining NaNs; they are structural / definitional.

---

## Formerly missing → now fixed

### S10 (`table_S10_prediction_interval_80_coverage_and_wis.csv`)
- **Class:** formerly-missing-now-fixed (inference gap closed).
- **Cause:** `pandemic_stress` five-quantile runs were never scheduled for `moirai` / `lag_llama` (`STRESS_MODELS` was STRESS3-only). Rows were absent, not NaN cells.
- **Fix:** five-quantile 13/13 for all five foundation models; S10 rebuilt at 06:06 with **0 NaNs**.
- **Coverage check:** both `outer_backtest` and `pandemic_stress` have 5 subsets × `{granite_patchtst_fm, timesfm_3, chronos_2, moirai, lag_llama}` = 50 rows. Stress preds on disk: **13/13** each.

---

## Structural / definitional NaNs (leave labeled; do not fill)

| Table | Class | Why NaNs remain |
|-------|--------|-----------------|
| **S1** | structural | `DM_*` / `p_*` undefined for ETS vs itself; `sig_*` empty when \(p \ge 0.05\). |
| **S4** | structural | Multi-section stacked binder CSV → cross-section column holes (detail in `RQ2_outputs/raw/s4_*.csv`). |
| **S5** | structural | `context_length_limit` NaN when unknown/unlimited; `DM_*`/`p_*`/`sig_*` vs Full undefined on Full self-row and Seasonal Naive (identical forecasts). Metrics themselves complete (42 rows / 14 models). |
| **S8** | structural | SameWeekday rows store W4/W1 degradation in MASE/MAPE/MAE/RMSE; `R_*_vs_ETS` intentionally NaN (schema). `note` annotates those rows. |
| **S11** | structural | Same DM/p/sig pattern as S1 (self-compare / non-significant). |
| **S12** | structural | `MAE_extended` NaN: source `table3_calendar_related_extension.csv` has no MAE column; `seconds_per_vector` NaN on synthetic `LL01_vs_lightgbm` row. |

---

## Clean (no actionable / no structural metric NaNs of concern)

- **S2, S3, S6, S7, S9, S13** — OK (0 metric NaNs).
- **S10** — OK after fill (see above).

---

## History / figure refresh (this run)

- `history_task_metrics.parquet` expanded (37,422 rows) — 06:05.
- **S5** rewritten (14 models × 3 history settings) — 06:05; structural DM/context NaNs retained.
- **Fig7** (`figure7_history_length_and_covid_contamination.{png,pdf}`) refreshed — 06:06.
- Summary `4.2_RQ2_实验数据核对汇总稿.md` rewritten — 06:06.
- Machine scan also in `logs/nan_residual_report.md` (filters DM/p/sig/context from “actionable” nulls).

---

## Related notes

- Pre-fill snapshot: `logs/nan_inventory_pre_fill.md`
- 4.2 audit note: `logs/nan_audit_note_for_4_2.md`
- Earlier S5 path fix: `logs/m423_history_rerun_note.md`
