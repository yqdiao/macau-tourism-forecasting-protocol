# RQ1_outputs

Sealed RQ1 tables and intermediate metrics for Section 4.1. RQ3 reads this directory only (not the working copies under `RQ1/`).

| Path | Contents |
|---|---|
| `tables/` | Final CSV tables (Table 3–5, appendices, Pareto coordinates) |
| `raw/` | Task/market metrics, light timing log, five-quantile predictions |

Build scripts and PNG figures remain under `RQ1/` (`RQ1/figures/`). The CSV set in `tables/` matches the CSV files written next to the builders in `RQ1/` at seal time.

```bash
python RQ1/build_rq1.py
python RQ1/build_rq1_remaining.py
python RQ1/build_rq1_five_tau.py
```
