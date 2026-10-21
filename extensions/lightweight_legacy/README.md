# Original lightweight-model implementation snapshot

This directory contains the supplied implementation and configuration snapshots for the original **NL01, GL03 and LL01** analyses. It is distinct from the later R3 V1–V3 revisions. The recorded R3 configuration is in [`paper/configs/r3/`](../../paper/configs/r3/); its implementation source was not present in the uploaded result bundle.

## Contents

- `code/light_models.py`: global NLinear and global/local LightGBM model implementation.
- `code/run_light_models.py`: split/origin runner and recorded resource/status manifests.
- `code/evaluate_light_models.py`: B-group scoring and the original freeze procedure.
- `code/score_light_slices.py`, `score_light_extra.py`, `score_light_round2.py`: supplemental scoring scripts.
- `configs/light_model_plan.yaml`: original registered plan (JSON-compatible text with its original suffix).
- `configs/frozen_light_configs.yaml`: original B-selected configurations; NL01 remains the lightweight champion.
- `configs/light_model_round2.yaml`: post-outer local-tree exploration; it does not replace the original selection.
- `source_manifest.json`: SHA256 values for the supplied and published versions, with all portability changes recorded.

## Portability and preservation

The original model implementation is retained. Configuration values are retained; the plan's Chinese calendar-name filter literals are written with Unicode escapes, preserving their exact JSON values. The runner and scoring scripts have small path-resolution edits for this repository layout. The runner's host-specific shard/journal defaults were replaced by repository-relative defaults. Training logic, model parameters, selection rules, recorded decisions and numerical results were not changed. The source manifest distinguishes representation and portability edits from unmodified historical source files.

The model-code SHA256 still matches the original frozen configuration. The original plan bytes matched its recorded hash before the Unicode-escape conversion; the published plan now has a different byte hash with identical parsed values. An edited runner also has a different hash. Both original and released hashes are recorded. Do not substitute a published file's new hash for a historical manifest without recording the representation change or a new execution, as applicable.

## Dependencies and data prerequisites

The scripts depend on the repository's `track1/code/data_pipeline.py`, NumPy, pandas, a Parquet engine, LightGBM, PyTorch and the historical input contracts. The recorded plan also identifies an environment using scikit-learn and `holidays` version 0.106. It does not supply a complete environment lockfile for these runs.

The expected data include:

1. `shared/data/origin_shards/origin_shards_manifest.json` and its isolated origin input files;
2. `shared/data/targets.parquet` and `shared/data/task_manifest.json` for independent scoring;
3. `shared/data/holiday_calendar.parquet`, with registered SHA256 `9060f62e542a884cf6f56d2feda8e527bb0b37eaebdedfbb5441a4357cf6cadc`;
4. the original prediction directories for scripts that compare already completed runs.

The supplied upload did not contain the holiday-calendar builder or the registered calendar file. They are not generated or fabricated in this release. The runner retains the calendar hash check. This source snapshot is therefore inspectable, but a full historical rerun requires those additional matching artifacts.

## Validation performed for this release

All six Python files passed Python syntax compilation. The three configuration files parse as JSON. The supplied model and plan hashes matched the recorded freeze; the released plan's parsed values remain unchanged after Unicode escaping. No training, model selection or server inference was rerun, and the imported dependency environment was not independently validated. Paper scores can be checked without these dependencies using:

```bash
python3 paper/scripts/rebuild_tables.py
```

The historical scoring script can write a new freeze if invoked for that purpose. Preserve the supplied frozen record when preparing any new experiment, and give new runs separate output/configuration identities. No new execution command is presented as a verified one-command reproduction.
