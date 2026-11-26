# makeup-RQ2

Final RQ2 package on top of sealed `RQ2_outputs/`: holiday masks, S9/S13 dual aggregation, complete S10 WIS₂ rows, quantile-crossing audit, S12 `MAE_extended`, and redrawn Figures 5–10.

```bash
python makeup-RQ2/run_makeup_rq2.py
```

## Deliverables

| Path | Role |
|---|---|
| `tables/S9_S13_holiday_reconciliation.csv` | Holiday R vs ETS, dual aggregation |
| `tables/S9_holiday_dual_aggregation.csv` | `pooled_day` vs `task_then_market_equal` |
| `tables/holiday_target_membership.parquet` | Holiday day membership |
| `tables/table_S10_wis_complete_50rows.csv` | 50-row WIS₂ table |
| `tables/table_S12_complete.csv` | Extended-input MAE and DM |
| `figures/figure5`–`figure10` | 300 DPI PNG + PDF |
| `logs/` | Provenance, crossing audit, missing-item manifest |
| `RQ2_verification_summary.md` | English verification note |

Do not copy these tables over `RQ2_outputs/`; RQ3 still reads the original panels there.
