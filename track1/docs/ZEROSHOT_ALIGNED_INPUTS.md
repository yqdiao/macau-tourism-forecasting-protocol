# Zero-shot models with calendar and cross-market inputs

On 2026-10-11, Chronos-2, TimesFM 3.0, and Moirai were re-run on the Macau outer set (891 tasks) with statutory calendar features and other markets’ history where the checkpoint accepts them. Granite and Lag-Llama checkpoints have no such input channels and were not re-run.

| Model | Outer MASE(7), market-only history | Outer MASE(7), aligned inputs |
|---|---:|---:|
| Chronos-2 | 1.152 | 1.129 |
| TimesFM 3.0 | 1.117 | 1.077 |
| Moirai 2.0 | (see main table) | 1.562 |

Predictions: `track1/results/zeroshot_matched_inputs/`. Runner: `track1/scripts/run_zeroshot_matched_inputs.py`. Exploratory per-market LightGBM + calendar (LL01) remains at outer MASE 1.260, still 0.131 above aligned Chronos-2.
