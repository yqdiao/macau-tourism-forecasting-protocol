# RQ3 output gap audit (final)

Based on DeepSeek pipeline completion (2376 API calls, all successful).

## Authoritative outputs

| Path | Role |
|---|---|
| `tables/table_S14_*.csv` | Fixed combination benchmarks |
| `tables/table_S15_*.csv` | 2×2 dynamic ablation; LLM = DeepSeek API |
| `tables/table_S16_*.csv` | Selection behavior (market / holiday / weekday) |
| `tables/table_S17_*.csv` | λ/α/δ/K / driver-metric sensitivity (G1) |
| `tables/table_S18_*.csv` | Mode1/2 cost and reliability (DeepSeek wall times) |
| `figures/figure11_*`, `figure12_*` | Manuscript figures |
| `cache/llm_judgments_deepseek.jsonl` | Authoritative LLM judgments |
| `cache/all_decisions_deepseek.parquet` | Authoritative dynamic decisions |

Do not cite early proxy caches: `llm_judgments.jsonl`, `*_fill.jsonl`, `all_decisions.parquet`.

## Quality checks

12/12 passed (`logs/rq3_output_quality_checks.csv`).

Headline (reopening): Granite MASE 1.0944; B_3FM Equal MAE 542.29; G1 MASE 1.1519; G2 1.1411; G4 ≈ 1.1408; G3 ≈ G1. DeepSeek ok=2376 fail=0; mean latency ≈ 1.522 s. Weight sum deviation < 1e-15; dynamic fallback = 0.

## Complete / needs wording / still missing

**Complete for drafting:** M5 pool and C1–C9 / B_3FM / B_5Pool metrics; no-leakage replay; Stat-Select / Stat-Weight 2×2 (G1–G4); DeepSeek API (not local Qwen3-4B — rename in text); S16–S18; Figures 11–12; Table 3 alignment to 1e-13.

**Wording only (W1–W7):** see `RQ3_outputs/README.md` and `RQ3_verification_summary.md`. LLM model name was intentionally switched from planned Qwen3-4B to DeepSeek.

**Still missing (do not fabricate):**

1. Warm-start inner prior (`not_run` — no M5 B-fold error library).
2. Local Qwen3-4B GGUF inference (replaced by DeepSeek API).
3. Literal Qwen3-4B judgments would need a matching API or GPU re-run.

Suggested citation sentence: Section 4.3 language judgments use DeepSeek API (`deepseek-chat`, temperature 0); 2,376 structured JSON calls validated with zero fallback. Relative to statistical baseline G1, DeepSeek dynamic selection (G2/G4) lowers reopening MASE from 1.152 to about 1.141; replacing weighting alone (G3) adds almost no gain.
