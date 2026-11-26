# Quantile crossing audit (pandemic_stress)

## Counts (adjacent-order violations on saved `q0.1 < q0.25 < q0.5 < q0.75 < q0.9`)

| model | n_days | n_adjacent_crossings | share |
|-------|--------|----------------------|-------|
| Chronos-2 | 9,828 | **13** | 0.132% |
| Moirai 2.0 | 9,828 | **307** | 3.124% |
| Granite / TimesFM 3.0 / Lag-Llama | 9,828 | 0 | 0 |

These match the previously reported stress-period pattern (13 / 9,828 and 307 / 9,828).

## Code / config sources (no post-hoc monotone sort)

1. **Five-quantile registry** — `RQ1/raw/registry_five_quantiles.yaml`  
   `quantile_levels: [0.1, 0.25, 0.5, 0.75, 0.9]` for each foundation model.
2. **Runner** — `RQ1/run_five_quantiles.py` → `track1/code/run_foundation_backtest.py`  
   Writes adapter quantile arrays into `predictions.parquet` columns `q0.1`…`q0.9` without rearrangement.
3. **Chronos-2 adapter** — `track1/code/foundation_models/adapters/chronos2.py`

```python
levels = self.quantile_levels()
forecast = self.model.predict_df(
    ...,
    quantile_levels=levels,
    ...
)
quantiles={f"q{q:g}": rows[f"{q:g}"].to_numpy(float) for q in levels}
```

4. **Moirai 2.0 adapter** — `track1/code/foundation_models/adapters/moirai.py`

```python
forecast = next(iter(predictor.predict(dataset)))
quantiles = {f"q{q:g}": np.asarray(forecast.quantile(q), dtype=float)
             for q in self.quantile_levels()}
```

Neither adapter applies isotonic / sorted rearrangement before save.

## Interpretation

- Crossings are **native quantile-head outputs** (or library `forecast.quantile` evaluations), not column-map bugs: column names match requested levels, and Granite/TimesFM/Lag-Llama show zero crossings under the same pipeline.
- **Main metrics keep the original saved values** (no overwrite).
- Sensitivity: `makeup-RQ2/raw/stress_{chronos_2,moirai}_quantile_monotone_sensitivity_sample.parquet` adds `q*_monotone_rearranged` by row-wise `np.sort` for optional re-scoring; mean absolute shift if sorted is ~0.002–0.003 visitors on stress samples.
