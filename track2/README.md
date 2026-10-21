# Track 2: forecast combination and coordination protocol

This directory preserves the September design for combining three models selected from the Track 1 B01–B12 results, with optional residual correction, probability calibration, and workflow coordination.

- [Archival ensemble, residual, and agent protocol](docs/ENSEMBLE_RESIDUAL_AGENT_PROTOCOL.md)
- [Current paper evidence](../paper/README.md)
- [Current reproducibility boundaries](../docs/reproducibility_status.md)

The archival protocol proposes a frozen deterministic numerical pipeline, with an LLM coordinator permitted to invoke tools and explain recorded outcomes. It is distinct from the later Qwen3-8B experiment in which language judgments assign candidate weights, and from Qwen3-4B-thinking directly generating forecasts. Those designs and their evidence must not be conflated.

The current `track2/` directory does not itself contain the full later server execution implementation. A protocol or exported result is not a substitute for source code, final configurations, model revisions, and complete logs. Consult the `paper/` inventory for the materials actually released.

O01–O33 must not be used to retrospectively change the original selected members, weights, residual rules, or fallback thresholds. Any development after inspecting previous outer results retains its exploratory identity.
