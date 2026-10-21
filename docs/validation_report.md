# Validation of the 2026.10.21 documentation and evidence release

The checks below validate interfaces and released calculations. They do not
constitute a new forecasting experiment or a rerun of the GPU workloads.

## Code and notebooks

- All four existing tests in `track1/tests/test_foundation_contract.py` passed:
  rejection of negative histories, the 28-day forecast contract, origin-shard
  target isolation, and the mock runner's raw/clipped output behavior.
- Seven additional temporary checks passed: A/B/O fold counts, the actual
  frozen baseline configuration and task manifest, rejection of stress folds
  for tuning, input-history isolation from future targets, known-path metric
  values, retention of unscaled errors when the MASE scale is zero, and rejection
  of incomplete forecast paths.
- The frozen baseline configuration contains 13 models and 33 candidates.
- All ten generated notebook code cells passed Python syntax compilation.
  Their executable structure and numerical parameters were preserved; changes
  concern explanatory and display strings.
- Unicode escaping of source-workbook column labels preserves the exact parsed
  Python syntax tree and the values accepted by the data reader.

No models were trained and no weights were downloaded for these checks.

## Environment scope

The smoke checks used the existing local Python 3.13.4 environment, with NumPy
2.4.6, pandas 2.3.2, and PyArrow 23.0.1. An already-installed pure-Python OpenPyXL
3.1.5 was temporarily added to the import path. PyArrow 23.0.1 is outside the
historical requirement of `<23`; these results are therefore current-environment
smoke checks, not certification of the pinned historical environment.

## Evidence reconstruction

Run:

```bash
python3 paper/scripts/rebuild_tables.py
```

This standard-library script verifies the evidence-file checksums and rebuilds
the supplied main, lightweight, and LLM summary calculations from their released
derived scores. Read `paper/README.md` for the score provenance, the complete
sample versus sensitivity sample, resource-accounting boundaries, and intervals
that are retained from the source audit rather than recomputed by this script.

## Release boundaries

No repository license file was present in the source snapshot, and this update
does not assign a new license. Historical data, third-party model, and upstream
code terms must be checked for the intended use. Later experiment runners that
are absent from the supplied archives are not represented as executable code.
