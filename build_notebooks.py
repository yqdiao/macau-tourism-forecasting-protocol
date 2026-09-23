"""Create the two reproducible paper notebooks using notebook JSON v4."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


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
    (HERE / name).write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


save("01_数据管道.ipynb", [
    md("""# 澳门与香港日度访客量：数据管道

论文任务：27 条澳门客源地序列 + 2 条香港访客类别序列；每个起点只使用已知历史，预测未来 28 天。源数据只读，零值保留。请从本 notebook 所在文件夹运行。"""),
    code("""from pathlib import Path
import pandas as pd
import numpy as np
try:
    from IPython.display import display
except ImportError:
    def display(value): print(value.to_string() if hasattr(value, 'to_string') else value)
from data_pipeline import (
    MACAU_DIR, HONG_KONG_WORKBOOK, TEST_START, TEST_END,
    load_all, quality_summary, rolling_origins, make_input_task,
    write_data_snapshot, write_task_artifacts,
)
PROJECT = Path.cwd()
assert (PROJECT / 'data_pipeline.py').exists(), '请将工作目录设为 paper_pipeline 文件夹'
print('Macau:', MACAU_DIR)
print('Hong Kong workbook:', HONG_KONG_WORKBOOK)
print('Target-date test period:', TEST_START, 'to', TEST_END)
"""),
    md("""## 1. 读取与核对

澳门只读 `data,Tourism` CSV。香港只读工作簿的 `每日数据`：内地访客和其他访客分别为两条目标序列；工作簿中的访客总量仅用于对账。"""),
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
    md("""## 2. 生成标准化快照、模型输入与隔离目标

`task_inputs.parquet` 不含真实目标；`targets.parquet` 只允许独立评估器读取。manifest 仅保留逻辑来源标识和 SHA256，不保存本机绝对路径。"""),
    code("""manifest = write_data_snapshot(PROJECT / 'data')
task_manifest = write_task_artifacts(series, PROJECT / 'data')
print('series:', manifest['series_count'])
print('normalized observations:', sum(item['days'] for item in manifest['quality']))
print('source files hashed:', len(manifest['source_sha256']))
print('isolated tasks:', task_manifest['task_count'])
"""),
    md("""## 3. 事先固定滚动起点

完整样本每日滚动（stride=1）；共同计算样本从 2023-12-31 起每隔 28 天滚动（stride=28）。每条未来 28 天路径都完整落在 2024-01-01—2026-07-31。"""),
    code("""for key in ['Macau:CN_Chinese_mainland', 'HongKong:Mainland visitors', 'HongKong:Other visitors']:
    item = series[key]
    full = rolling_origins(item, stride_days=1)
    common = rolling_origins(item, stride_days=28)
    print(key, '| full:', len(full), full[0], 'to', full[-1],
          '| common:', len(common), common[0], 'to', common[-1])
"""),
    md("""## 4. 检查模型输入的信息边界

`task.history` 截止 2026-07-03，任务对象没有 `actual` 属性。真实目标只存在于单独的 `targets.parquet`。"""),
    code("""from datetime import date
item = series['Macau:CN_Chinese_mainland']
task = make_input_task(item, date(2026, 7, 3), history_length=180)
print(task.key, '| history:', task.history_start, 'to', task.origin, len(task.history),
      '| forecast dates:', task.target_dates[0], 'to', task.target_dates[-1])
assert len(task.history) == 180 and len(task.target_dates) == 28
assert not hasattr(task, 'actual')
assert task.target_dates[0] > task.origin
assert task.target_dates[-1] <= TEST_END
assert np.array_equal(task.history, item.values.loc[str(task.history_start):str(task.origin)].to_numpy())
"""),
    md("""下一步运行 `02_统一模型与回测.ipynb`。本 notebook 只做数据结构与时间边界检查，不训练模型。"""),
])


save("02_统一模型与回测.ipynb", [
    md("""# 统一 28 天模型与滚动回测

所有模型只接收 `ForecastInputTask.history`，任务对象不含真实目标。预测输出同时保留 raw 和非负裁剪值；评价由 `evaluate_predictions.py` 在预测完成后进行。"""),
    code("""from pathlib import Path
from datetime import date
import pandas as pd
try:
    from IPython.display import display
except ImportError:
    def display(value): print(value.to_string() if hasattr(value, 'to_string') else value)
from data_pipeline import load_all, make_input_task
from models import MODEL_NAMES, forecast
from run_backtest import run_one, run
PROJECT = Path.cwd()
assert (PROJECT / 'models.py').exists(), '请将工作目录设为 paper_pipeline 文件夹'
print('Available models:', ', '.join(MODEL_NAMES))
"""),
    md("""## 模型来源与性质

Seasonal Naïve 是必备基线；ETS 为本次新增；SARIMA 受 P1 的 `Fighting BI Team` 方法启发。LightGBM 迁移自 `6104_CJB`；传统回归迁移自 `Machine Not Learning` / `Parrot Nothing`；Transformer 是为数值序列另写的本地小模型，不是预训练基础模型。"""),
    code("""families = pd.DataFrame([
    ('seasonal_naive','统计基线','最后一周重复'),
    ('ets','新增统计基线','7天加性季节'),
    ('sarima','Fighting BI Team 方法迁移','周季节统计模型'),
    ('lightgbm','6104_CJB 迁移','每步长一模型'),
    ('random_forest','Machine Not Learning 迁移','每步长一模型'),
    ('parrot_random_forest_800','Parrot Nothing 迁移','800树，每步长一模型'),
    ('ridge / linear_regression / gradient_boosting / svr / lasso / elasticnet','传统回归对照','每步长一模型'),
    ('transformer','数值序列新改写','56天输入，28天输出'),
], columns=['模型','来源','输入与输出方式'])
display(families)
"""),
    md("""## 单任务演示

可修改市场和起点。使用该起点的前 180 天历史，输出未来 28 天路径；不将未来实际值传入模型。默认演示 Seasonal Naïve、ETS、SARIMA，运行较快。把模型名添加到 `demo_models` 即可运行 LightGBM、Random Forest 或 Transformer。"""),
    code("""series = load_all()
task = make_input_task(series['Macau:CN_Chinese_mainland'], date(2026, 7, 3), history_length=180)
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
    md("""## 对齐论文的四个提前期

本阶段只查看预测。MASE、RMSSE、绝对误差和高需求日指标由独立评估器在合并 targets 后计算。"""),
    code("""key_horizons = predictions[predictions['horizon_day'].isin([1,7,14,28])]
display(key_horizons[['model','horizon_day','target_date','prediction_raw','prediction','was_clipped']])
"""),
    md("""## 完整滚动实验（默认关闭）

要比较论文所有共同样本起点，先估算算力与费用；这里的开关默认为 `False`。若运行，澳门和香港分别输出，不合并训练。28 个步长的树模型和每起点重训的 Transformer 会耗时较长。"""),
    code("""RUN_PAPER_BACKTEST = False
if RUN_PAPER_BACKTEST:
    selected = ('seasonal_naive','ets','sarima','random_forest','lightgbm','transformer')
    for destination in ('Macau','HongKong'):
        result = run(inputs=PROJECT / 'data' / 'task_inputs.parquet',
                     model_names=selected, destinations=(destination,),
                     history_lengths=('180','365','all'),
                     output=PROJECT / 'results' / destination.lower())
        print(destination, result['forecast_tasks'], result['prediction_rows'])
"""),
    md("""**当前边界：**该文件夹尚未接入 TimesFM、Chronos、Moirai 或多智能体 API；也未生成节假日/演唱会标签或云实例费用。现有的小型 Transformer 应归为本地训练模型。"""),
])

print("Created 01_数据管道.ipynb and 02_统一模型与回测.ipynb")
