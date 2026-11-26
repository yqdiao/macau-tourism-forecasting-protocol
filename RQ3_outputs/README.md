# RQ3_outputs

## Authoritative files

- Tables: `tables/table_S14`–`table_S18`
- Figures: `figures/figure11_*`, `figures/figure12_*`
- DeepSeek judgments: `cache/llm_judgments_deepseek.jsonl`
- DeepSeek decisions: `cache/all_decisions_deepseek.parquet`
- Verification summary (writing notes W1–W7): `RQ3_verification_summary.md`
- Gap audit: `logs/rq3_output_gap_audit_final.md`
- Quality checks: `logs/rq3_output_quality_checks.csv`

## Do not cite in the manuscript (W7)

`cache/llm_judgments.jsonl`, `*_fill.jsonl`, and `all_decisions.parquet` are early llm_proxy caches. Text, footnotes, and reproduction notes must cite `*_deepseek.*` only.

## Writing notes (not missing runs)

| Tag | One-line rule |
|---|---|
| **W1** | G3 ≈ G1 is an empirical result, not a missing table |
| **W2** | Mode2 changes cost accounting only; accuracy equals Mode1 |
| **W3** | S18 LLM times are cloud API wall time (CPU-equivalent), not local GPU seconds |
| **W4** | S14 group means have no bootstrap CI |
| **W5** | S15 DM tests emphasize MASE/MAE |
| **W6** | Figure 12c is simplified; compute detail is in S17 `models_run` |
| **W7** | Do not cite the old proxy caches |

## Reproduce

```bash
python RQ3/run_all.py
python RQ3/fill_gaps.py
python RQ3/run_deepseek_rq3.py   # needs RQ3/.env
```
