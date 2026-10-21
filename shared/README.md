# Shared task artifacts

`shared/` contains the artifacts that the historical Track 1 and Track 2 protocols share:

- `configs/protocol.yaml`: the 28-day forecasting protocol and fixed origins;
- `data/task_inputs.parquet`: model-readable histories without future targets;
- `data/targets.parquet`: realized targets for the independent evaluator;
- `data/task_index.parquet` and `data/task_manifest.json`: task indices, counts, and SHA256 values;
- `data/splits/`: inputs separated into stress, selection, and outer-evaluation phases;
- `requirements.txt`: dependencies for the CPU pipeline and baseline implementation.

Predictor environments must not mount `data/targets.parquet`. GPU predictors should use one origin shard at a time, rather than the complete data directory. Downstream experiments reuse task IDs and temporal boundaries; a new experiment must document any intentional difference.

The current paper uses 891 Macau outer-evaluation tasks, while the historical implementation also supports two separate Hong Kong series and additional history settings. Dataset availability and scored-window coverage are different. See the [current release boundaries](../docs/reproducibility_status.md).

This update retains existing data bytes. It does not change data licensing or authorize new redistribution.
