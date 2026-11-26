# RQ2 NaN inventory (pre-fill snapshot)

## Why Moirai/Lag-Llama stress five-quantiles were missing
`RQ1/run_five_quantiles.py` originally set `STRESS_MODELS = {granite, timesfm, chronos}` only — matching RQ1 Table5 STRESS3. Outer five-quantile ran for all 5; pandemic_stress was never scheduled for moirai/lag_llama (**never run**, not a join bug).

## Classification of NaNs / gaps

### Needs inference (never run) — IN PROGRESS in tmux `rq2-stress-q` / `rq2-fill`
- S10 `pandemic_stress` × `{moirai, lag_llama}`: rows absent (not NaN). moirai now 13/13; lag_llama was mid-run.

### Run but not joined into S5/S13 history dims
- History caches exist for GL03/LL01/NL01/lag_llama/transformer/sarima at 180/365, but `build_history_panels()` only scored 8 representative models → absent from S5 and S13 dims 6/7 for LL01 etc.

### Intentionally absent / non-metric
- S1/S11: `DM_*`/`p_*` NaN for ETS vs itself; `sig_*` empty string when p≥0.05 (stars()).
- S5: `context_length_limit` NaN when unknown/unlimited; `DM_*_vs_Full` NaN on Full-with-COVID self-row and Seasonal Naive (identical forecasts → DM undefined).
- S4: multi-section stacked CSV → structural NaNs across sections (detail in `RQ2_outputs/raw/s4_*.csv`).
- S8 SameWeekday rows: degradation ratios stored in MASE/MAPE/MAE/RMSE; `R_*_vs_ETS` left NaN (schema), not missing preds.
- S12: `MAE_extended` NaN because source `table3_calendar_related_extension.csv` has no MAE column; `seconds_per_vector` NaN for synthetic LL01_vs_lightgbm row.

### Clean
- S2, S3, S6, S7, S9, S13 R_vs_ETS: no actionable metric NaNs.
