# RQ2 NaN audit + fill-in final report

## 1. Why Moirai / Lag-Llama stress five-quantiles were missing
`RQ1/run_five_quantiles.py` originally hard-coded:
`STRESS_MODELS = {granite_patchtst_fm, timesfm_3, chronos_2}`
so pandemic_stress jobs were **never scheduled** for moirai/lag_llama (not a join failure). Outer five-quantile already existed for all five models.

## 2. Gap classification (before fill)

| Gap | Class | Action |
|---|---|---|
| S10 pandemic_stress × moirai/lag_llama rows absent | **never run** | Ran in tmux `rq2-stress-q` then synced |
| S5 missing GL03/LL01/NL01/lag_llama/transformer/sarima | **run but not joined** (180/365 caches existed) | Expanded `build_history_panels` on CPU |
| S1/S11 DM/p NaN for ETS; empty `sig_*` when p≥0.05 | **intentionally absent** | keep |
| S5 `context_length_limit` / DM vs Full on Full self-row | **intentionally absent** | keep |
| S4 stacked multi-section CSV NaNs | **structural** | detail in `raw/s4_*.csv` |
| S8 SameWeekday `R_*_vs_ETS` NaN | **schema** (degradation stored in MASE/MAPE/MAE/RMSE) | annotated `note` column |
| S12 `MAE_extended` / one timing cell | **source schema** (`table3_calendar_related_extension.csv` has no MAE) | residual; needs MAE from day preds if required |

## 3. tmux jobs
- `rq2-stress-q`: moirai (GPU0) + lag_llama (GPU1) pandemic_stress five-quantile → **13/13 each, EXIT:0**
- `rq2-fill`: wait → sync to `RQ1_outputs` → expand history panel → recompute S5/S6–S10/S13 → figures 5–10 → summary MD → **`[fill done]`**

## 4. Recomputed outputs
- S10: both splits × all 5 foundation models × 5 subsets (no missing rows)
- S5: 14 configs × Clean-180 / Clean-365 / Full-with-COVID
- S13 boundary matrix refreshed
- Figures 5–10 + `makeup-RQ2/RQ2_verification_summary.md` regenerated
- Residual scan: `RQ2_outputs/logs/nan_residual_report.md`

## 5. Residual NaNs (no new fake numbers; need new source work only if desired)
1. **S12** `MAE_extended` for timesfm/chronos/moirai extension rows; `seconds_per_vector` for synthetic LL01_vs_lightgbm — source table lacks MAE / no timing id.
2. **S8** SameWeekday `R_*` — not defined by design (degradation ratios only).
3. **S4** binder structural NaNs.
4. **DM/sig** empty cells for non-significant or ETS-vs-self — statistical, not missing preds.
