"""Create Track 1 data and baseline demonstration notebooks using JSON v4."""
from __future__ import annotations

import json
from pathlib import Path

TRACK1_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = TRACK1_ROOT / "notebooks"


def md(value: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": value.splitlines(keepends=True)}


def code(value: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": value.splitlines(keepends=True)}


def save(name: str, cells: list[dict]) -> None:
    notebook = {"cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": "3.12"}},
                "nbformat": 4, "nbformat_minor": 5}
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    (NOTEBOOK_DIR / name).write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


save("01_data_pipeline.ipynb", [
    md("""# Daily visitor arrivals in Macau and Hong Kong: data pipeline

The task includes 27 Macau source-market series and two Hong Kong visitor-category series. At each origin, only the available history is used to forecast the next 28 days. Source data are read-only, and zero observations are retained. Set the working directory to the repository root before running this notebook. This notebook demonstrates data preparation and boundary checks; it does not reproduce every experiment reported from the server."""),
    code("""from pathlib import Path
import sys
import pandas as pd
import numpy as np
try:
    from IPython.display import display
except ImportError:
    def display(value): print(value.to_string() if hasattr(value, 'to_string') else value)
PROJECT = Path.cwd()
CODE = PROJECT / 'track1' / 'code'
assert CODE.is_dir(), 'Set the working directory to the repository root.'
sys.path.insert(0, str(CODE))
from data_pipeline import (
    MACAU_DIR, HONG_KONG_WORKBOOK, TEST_START, TEST_END,
    load_all, quality_summary, make_input_task,
    write_data_snapshot, write_task_artifacts,
)
from protocol_config import load_protocol
print('Macau:', MACAU_DIR)
print('Hong Kong workbook:', HONG_KONG_WORKBOOK)
print('Target-date test period:', TEST_START, 'to', TEST_END)
"""),
    md("""## 1. Load and validate the data

Macau data are read from the `data,Tourism` CSV sources. Hong Kong data are read from the workbook's daily-data worksheet: mainland visitors and other visitors are separate target series. The workbook's total visitor count is used only for reconciliation. The loader retains the original source worksheet and column identifiers."""),
    code("""series = load_all()
qa = pd.DataFrame(quality_summary(series))
display(qa.groupby('destination').agg(series=('market','count'),
                                      first_date=('first_date','min'),
                                      last_date=('last_date','max'),
                                      total_days=('days','sum'),
                                      zero_days=('zero_days','sum')))
display(qa.head(4))
assert len(series) == 29
assert (qa['days'] > 0).all()
"""),
    md("""## 2. Generate normalized snapshots, model inputs and isolated targets

`task_inputs.parquet` contains no target observations; `targets.parquet` is reserved for the independent evaluator. The manifest retains logical source identifiers and SHA256 hashes rather than machine-specific absolute paths."""),
    code("""manifest = write_data_snapshot(PROJECT / 'shared' / 'data')
task_manifest = write_task_artifacts(series, PROJECT / 'shared' / 'data')
print('series:', manifest['series_count'])
print('normalized observations:', sum(item['days'] for item in manifest['quality']))
print('source files hashed:', len(manifest['source_sha256']))
print('isolated tasks:', task_manifest['task_count'])
print('split task counts:', task_manifest['split_task_counts'])
"""),
    md("""## 3. Inspect the three configured sets of forecast origins

`shared/configs/protocol.yaml` explicitly lists A01–A13 (pandemic stress), B01–B12 (tuning) and O01–O33 (outer evaluation). Sets A and B include Macau only; Hong Kong participates in outer evaluation only. Listing these origins does not by itself establish when a configuration was frozen."""),
    code("""protocol = load_protocol()
for split in ('pandemic_stress', 'recovery_tuning', 'outer_backtest'):
    folds = [fold for fold in protocol.folds if fold.split_name == split]
    print(split, len(folds), folds[0].fold_id, folds[0].origin,
          'to', folds[-1].fold_id, folds[-1].origin, '| destinations:', folds[0].destinations)
"""),
    md("""## 4. Check the model input boundary

In this O33 example, `task.history` ends on 2026-06-14, and the task object has no `actual` attribute. Target observations are stored separately in `targets.parquet`."""),
    code("""from datetime import date
item = series['Macau:CN_Chinese_mainland']
task = make_input_task(item, date(2026, 6, 14), history_length=180)
print(task.key, '| history:', task.history_start, 'to', task.origin, len(task.history),
      '| forecast dates:', task.target_dates[0], 'to', task.target_dates[-1])
assert len(task.history) == 180 and len(task.target_dates) == 28
assert not hasattr(task, 'actual')
assert task.target_dates[0] > task.origin
assert task.target_dates[-1] <= TEST_END
assert np.array_equal(task.history, item.values.loc[str(task.history_start):str(task.origin)].to_numpy())
"""),
    md("""Next, run `02_models_and_backtesting.ipynb`. This notebook checks data structure and temporal boundaries; it does not train models."""),
])


save("02_models_and_backtesting.ipynb", [
    md("""# Baseline models and rolling 28-day backtesting

Models receive `ForecastInputTask.history`; task objects do not contain target observations. Forecast outputs retain both raw and nonnegative-clipped values. Evaluation is performed by `track1/code/evaluate_predictions.py` after forecasts have been generated. Set the working directory to the repository root. This notebook demonstrates the baseline implementation included in this repository, not every later server experiment."""),
    code("""from pathlib import Path
from datetime import date
import sys
import pandas as pd
try:
    from IPython.display import display
except ImportError:
    def display(value): print(value.to_string() if hasattr(value, 'to_string') else value)
PROJECT = Path.cwd()
CODE = PROJECT / 'track1' / 'code'
assert CODE.is_dir(), 'Set the working directory to the repository root.'
sys.path.insert(0, str(CODE))
from data_pipeline import load_all, make_input_task
from models import MODEL_NAMES, forecast
from run_backtest import run_one, run
print('Available models:', ', '.join(MODEL_NAMES))
"""),
    md("""## Model provenance and scope

Seasonal Naive is the required baseline; ETS is an additional statistical baseline. SARIMA is inspired by the P1 `Fighting BI Team` approach. LightGBM is adapted from `6104_CJB`, and conventional regression models are adapted from `Machine Not Learning` / `Parrot Nothing`. The Transformer is a small, locally trained model written for numerical series; it is not a pretrained foundation model."""),
    code("""families = pd.DataFrame([
    ('seasonal_naive','Statistical baseline','Repeat the last week'),
    ('ets','Additional statistical baseline','Additive 7-day seasonality'),
    ('sarima','Adapted from Fighting BI Team','Weekly seasonal statistical model'),
    ('lightgbm','Adapted from 6104_CJB','One model per horizon'),
    ('random_forest','Adapted from Machine Not Learning','One model per horizon'),
    ('parrot_random_forest_800','Adapted from Parrot Nothing','800 trees; one model per horizon'),
    ('ridge / linear_regression / gradient_boosting / svr / lasso / elasticnet','Conventional regression comparison','One model per horizon'),
    ('transformer','Written for numerical series','56-day input; 28-day output'),
], columns=['Model','Provenance','Input/output design'])
display(families)
"""),
    md("""## Single-task demonstration

The market and origin can be changed. This example uses the 180 days available at the origin to forecast the next 28 days; future observations are not passed to the model. The default demonstration uses Seasonal Naive, ETS and SARIMA. Add model names to `demo_models` to run LightGBM, Random Forest or the local Transformer, with their dependencies installed."""),
    code("""series = load_all()
task = make_input_task(series['Macau:CN_Chinese_mainland'], date(2026, 6, 14), history_length=180)
demo_models = ['seasonal_naive', 'ets', 'sarima']
forecast_rows, time_rows = [], []
for name in demo_models:
    rows, timing = run_one(task, name)
    forecast_rows.extend(rows)
    time_rows.append(timing)
predictions = pd.DataFrame(forecast_rows)
display(predictions[['model','target_date','horizon_day','prediction_raw','prediction','was_clipped']].head(12))
display(pd.DataFrame(time_rows)[['model','fit_seconds','inference_seconds','total_seconds']])
assert predictions.groupby('model').size().eq(28).all()
assert 'actual' not in predictions.columns
"""),
    md("""## Inspect the four forecast horizons used in the paper

At this stage, inspect predictions only. MASE, RMSSE, absolute errors and high-demand-day metrics are calculated by the independent evaluator after joining the isolated targets."""),
    code("""key_horizons = predictions[predictions['horizon_day'].isin([1,7,14,28])]
display(key_horizons[['model','horizon_day','target_date','prediction_raw','prediction','was_clipped']])
"""),
    md("""## Baseline rolling backtest (disabled by default)

Outer evaluation requires completion of B-group tuning and a frozen baseline configuration. When enabled, the cell below calls `baseline_config.validate(...)` to check the supplied configuration and task manifest, then requires the returned freeze status to be `frozen_after_B_group_tuning`. The notebook does not assume the current freeze status. The switch is disabled by default, and batch execution is rejected if validation fails or the required status is absent. This run covers only the selected baseline models below."""),
    code("""RUN_PAPER_BACKTEST = False
if RUN_PAPER_BACKTEST:
    from baseline_config import validate
    audit = validate(task_manifest_path=PROJECT / 'shared' / 'data' / 'task_manifest.json')
    if audit['freeze_status'] != 'frozen_after_B_group_tuning':
        raise RuntimeError('Complete B-group tuning and freeze the baseline configuration before running outer evaluation.')
    selected = ('seasonal_naive','ets','sarima','random_forest','lightgbm','transformer')
    for destination in ('Macau','HongKong'):
        result = run(inputs=PROJECT / 'shared' / 'data' / 'task_inputs.parquet',
                     model_names=selected, destinations=(destination,),
                     history_lengths=('all',), splits=('outer_backtest',),
                     output=PROJECT / 'track1' / 'results' / destination.lower())
        print(destination, result['forecast_tasks'], result['prediction_rows'])
"""),
    md("""**Scope:** This notebook does not invoke TimesFM, Chronos, Moirai or multi-agent services, and it does not generate holiday/concert labels or cloud-instance billing records. The small Transformer used here is a locally trained model. Reproducing later server experiments requires their corresponding implementations, configurations, checkpoints and archived inputs/outputs; this notebook alone is not a reproduction entry point for all manuscript results."""),
])

print("Created 01_data_pipeline.ipynb and 02_models_and_backtesting.ipynb")
