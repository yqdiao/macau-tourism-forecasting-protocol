# RQ3 verification summary (DeepSeek final)

## Scope and inputs

- Predictions and single-model metrics: read-only `RQ1_outputs/` and `RQ2_outputs/`
- LLM: DeepSeek API (`deepseek-chat`, `RQ3/.env`); judgments in `cache/llm_judgments_deepseek.jsonl` (2376/2376 ok)
- Authoritative dynamic decisions: `cache/all_decisions_deepseek.parquet`
- Single-model vs Table 3 max abs diff = `2.274e-13`
- History rule: `target_end(r) ≤ origin(t)`; cold start = equal-weight C2
- Measured DeepSeek latency: mean 1.522 s, p50 1.422 s

## Headline results (reopening)

| scheme | MASE | MAE | MAPE | RMSE | R_MASE_vs_Granite | fallback |
|---|---:|---:|---:|---:|---:|---:|
| Granite | 1.0944 | 553.799 | 37.2641 | 820.828 | — | — |
| B_3FM Equal | 1.09853 | 542.292 | 38.3375 | 803.874 | — | — |
| C2 Equal | 1.12688 | 553.854 | 38.8033 | 800.978 | — | — |
| G1_Stat_Stat | 1.15193 | 581.391 | 39.8849 | 829.116 | 1.05257 | 0 |
| G2_LLM_Stat_eta0.3 | 1.14105 | 566.288 | 39.4847 | 815.069 | 1.04263 | 0 |
| G3_Stat_LLM_eta0.3 | 1.15193 | 581.391 | 39.8849 | 829.116 | 1.05257 | 0 |
| G4_LLM_LLM_eta0.3 | 1.14094 | 566.299 | 39.4838 | 815.143 | 1.04253 | 0 |
| C2_StatWeight | 1.12323 | 551.465 | 38.7553 | 798.603 | 1.02634 | — |
| B_3FM_Equal | 1.09853 | 542.292 | 38.3375 | 803.874 | 1.00377 | — |

**W1 — G3 ≈ G1:** G3 matches G1 on all four metrics (MASE = 1.15193). DeepSeek weighting on a fixed statistical subset is essentially inverse-error weighting. Report “replacing the weighting step alone adds no gain,” not “G3 was not run.”

## S18 cost snapshot (scenario S1)

Mode2 changes which models are billed; under sealed offline forecasts, **point predictions equal Mode1 (W2)**. LLM wall times in S18 are **cloud API** seconds folded into a CPU-equivalent cost, not local GPU seconds (**W3**). Full rows: `tables/table_S18_mode1_vs_mode2_compute_cost_and_reliability.csv`.

## Writing checklist (W1–W7)

| Tag | Topic | Required wording |
|---|---|---|
| W1 | G3 ≈ G1 | Empirical near-zero gain from LLM weights on a fixed subset |
| W2 | Mode2 accuracy | Cost mechanism only; same point forecasts as Mode1 |
| W3 | S18 units | Cloud API wall time as CPU-equivalent |
| W4 | S14 CIs | Group means are point estimates only |
| W5 | S15 DM | Emphasize MASE/MAE vs ETS/G1 |
| W6 | Figure 12c | Main panel is K–MASE; see S17 `models_run` for compute detail |
| W7 | Proxy caches | Cite `*_deepseek.*` only |

Open `tables/table_S14`…`S18` for full width tables. Gap detail: `logs/rq3_output_gap_audit_final.md`.
