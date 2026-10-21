# Track 1: common single-model backtesting implementation

The historical design registers 13 baselines and seven time-series models under daily univariate inputs, a 28-day horizon, and common rolling origins. This planned implementation scope is not a claim that all 20 configurations appear in the final paper with complete, comparable evidence. See the [paper companion](../paper/README.md) for the actual release inventory.

## Directories

- `code/`: Python pipeline, baseline implementation, seven adapters, and evaluators;
- `configs/`: baseline search spaces, frozen selections, and upstream revision registry;
- `compose/` and `docker/`: CPU and separate CUDA environments;
- `docs/`: English archival protocols, audits, and server guides;
- `notebooks/01_data_pipeline.ipynb`: data preparation notebook;
- `notebooks/02_models_and_backtesting.ipynb`: model and backtesting notebook;
- `scripts/`: baseline selection driver and GPU batch launchers;
- `tests/`: interface, 28-day output, and target-isolation tests;
- `audit/b_validation/`: the completed September baseline selection audit.

All commands assume the repository root. Shared inputs are in `shared/data/`; generated results normally go to ignored `track1/results/` and `track1/evaluation/` directories.

## Documentation

- [Unified historical protocol](docs/UNIFIED_EXPERIMENT_PROTOCOL.md)
- [Seven-model historical design](docs/SEVEN_MODEL_GPU_PROTOCOL.md)
- [GPU execution guide](docs/FOUNDATION_MODELS_GPU.md)
- [Sequential GPU checklist](docs/GPU_SEQUENTIAL_RUN_CHECKLIST.md)
- [Server architecture](docs/SERVER_MODEL_EXECUTION.md)
- [Historical compliance audit](docs/BASELINE_COMPLIANCE_AUDIT.md)

A document's archival status is explicit at its top. Do not infer present execution status from an old pending checkbox or infer final checkpoint versions from a smoke configuration.
