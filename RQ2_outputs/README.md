# RQ2_outputs

Sealed RQ2 panels, tables S1–S13, and Figures 5–10 as first produced by `RQ2/`. RQ3 reads this directory.

| Path | Contents |
|---|---|
| `tables/` | S1–S13 CSV tables |
| `figures/` | Figures 5–10 (PNG/PDF) |
| `raw/` | Task/day metric panels used by later stages |
| `logs/` | Coverage and NaN-fill audits |

Final holiday dual aggregation, complete S10 WIS rows, S12 extensions, and redrawn Figures 5–10 live in `makeup-RQ2/`. Keep both directories: makeup is the manuscript figure package; this directory remains the RQ3 input panel.

```bash
python RQ2/run_all.sh
python makeup-RQ2/run_makeup_rq2.py
```
