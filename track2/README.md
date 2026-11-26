# Track 2: forecast combination and coordination protocol

This directory preserves the September design for combining three models selected from the Track 1 B01–B12 results, with optional residual correction, probability calibration, and workflow coordination.

- [Archival ensemble, residual, and agent protocol](docs/ENSEMBLE_RESIDUAL_AGENT_PROTOCOL.md) (includes the 2026-10-05 member freeze, controls, and fault injection)
- Frozen members: [configs/selected_track2_models.yaml](configs/selected_track2_models.yaml) (Granite, Chronos-2, TimesFM 3.0)
- Stage code under `code/` (inventory, cross-fit, agents, faults, scoring)
- Paper combination / LLM evidence for RQ3: `../RQ3_outputs/` and [paper/README.md](../paper/README.md)
- Release boundaries: [docs/reproducibility_status.md](../docs/reproducibility_status.md)

The archival protocol proposes a frozen deterministic numerical pipeline, with an LLM coordinator permitted to invoke tools and explain recorded outcomes. It is distinct from RQ3 DeepSeek / Qwen weighting experiments and from Qwen3-4B-thinking direct forecasts. Those designs and their evidence must not be conflated.

`track2/code/` holds stage scripts for the historical Track 2 design. RQ3 sealed tables are the manuscript authority for combination and LLM-judgment results.

O01–O33 must not be used to retrospectively change the original selected members, weights, residual rules, or fallback thresholds. Any development after inspecting previous outer results retains its exploratory identity.
