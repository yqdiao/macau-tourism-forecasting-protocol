#!/usr/bin/env bash
set -uo pipefail
REPO=/root/DYQ/macau-tourism-forecasting-protocol
LOG=$REPO/RQ2_outputs/logs/fill_and_recompute.log
exec > >(tee -a "$LOG") 2>&1
echo "[fill start] $(date -Is)"

echo "Waiting for lag_llama pandemic_stress five-quantile (need 13/13)..."
while true; do
  n=$(find "$REPO/RQ1/raw/five_quantile_runs/pandemic_stress/lag_llama" -name predictions.parquet 2>/dev/null | wc -l)
  echo "  lag_llama stress preds: $n/13  $(date +%H:%M:%S)"
  if [ "$n" -ge 13 ]; then break; fi
  if grep -q '^EXIT:[^0]' "$REPO/RQ1/raw/extra_logs/stress_lag_llama.log" 2>/dev/null; then
    echo "lag_llama FAILED"; tail -40 "$REPO/RQ1/raw/extra_logs/stress_lag_llama.log"; exit 1
  fi
  sleep 20
done
echo "moirai stress: $(find "$REPO/RQ1/raw/five_quantile_runs/pandemic_stress/moirai" -name predictions.parquet | wc -l)/13"
echo "lag_llama stress: $(find "$REPO/RQ1/raw/five_quantile_runs/pandemic_stress/lag_llama" -name predictions.parquet | wc -l)/13"

python3 - <<'PY'
from pathlib import Path
import shutil
src=Path('/root/DYQ/macau-tourism-forecasting-protocol/RQ1/raw/five_quantile_runs/pandemic_stress')
dst=Path('/root/DYQ/macau-tourism-forecasting-protocol/RQ1_outputs/raw/five_quantile_runs/pandemic_stress')
n=0
for model in ['moirai','lag_llama']:
    for pred in (src/model).glob('*/predictions.parquet'):
        out=dst/model/pred.parent.name/'predictions.parquet'
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pred, out)
        man=pred.parent/'run_manifest.json'
        if man.is_file():
            shutil.copy2(man, out.parent/'run_manifest.json')
        n+=1
print(f'synced {n} prediction files to RQ1_outputs')
# also sync into RQ2-visible path used by m424 (RQ1/raw is canonical; m424 uses RQ1)
print('canonical RQ1/raw already has preds')
PY

cd "$REPO/RQ2"
export PYTHONPATH=$REPO/RQ2
PY=/root/DYQ/macau-tourism-world-model/.venv-server/bin/python

echo "[history expand]"
$PY -u - <<'PY'
from build_panels import build_history_panels
build_history_panels([
    "granite_patchtst_fm", "timesfm_3", "chronos_2", "moirai", "lag_llama",
    "ets", "sarima", "seasonal_naive", "svr", "lightgbm", "transformer",
    "GL03", "LL01", "NL01",
])
print('history panel expanded')
PY

echo "[m423]"
$PY -u m423_history.py

echo "[m424 rebuild S6-S10]"
$PY -u m424_conditions.py

$PY -u - <<'PY'
import numpy as np, pandas as pd
from common import TABLES
s8=pd.read_csv(TABLES/'table_S8_horizon_weekly_blocks_and_same_weekday.csv')
mask=s8.week_block=='SameWeekday_W4_over_W1'
s8['note']=np.where(mask, 'SameWeekday row stores W4/W1 degradation in MASE/MAPE/MAE/RMSE; R_* vs ETS not defined', '')
s8.to_csv(TABLES/'table_S8_horizon_weekly_blocks_and_same_weekday.csv', index=False)
import pandas as pd
from pathlib import Path
s10=pd.read_csv(TABLES/'table_S10_prediction_interval_80_coverage_and_wis.csv')
print(s10.groupby(['split_name','model']).size().unstack(fill_value=0))
PY

echo "[m425]"
$PY -u m425_boundary.py
echo "[plots]"
$PY -u plot_figures.py
echo "[summary]"
$PY -u write_summary.py

$PY -u - <<'PY'
import pandas as pd, re
from pathlib import Path
OUT=Path('/root/DYQ/macau-tourism-forecasting-protocol/RQ2_outputs')
lines=['# RQ2 NaN / gap residual report\n', f'After fill_and_recompute\n\n']
s10=pd.read_csv(OUT/'tables/table_S10_prediction_interval_80_coverage_and_wis.csv')
lines.append('## S10 coverage\n```\n')
lines.append(str(s10.groupby(['split_name','model']).size().unstack(fill_value=0)))
lines.append('\n```\n\n')
s5=pd.read_csv(OUT/'tables/table_S5_history_length_and_denominator_control.csv')
lines.append('## S5 models\n')
lines.append(', '.join(sorted(s5.config_id.unique()))+'\n\n')
report=[]
for p in sorted((OUT/'tables').glob('table_S*.csv')):
    df=pd.read_csv(p)
    if 'S4_market' in p.name:
        report.append(f'{p.name}: structural multi-section binder')
        continue
    drop=re.compile(r'(^sig_|DM_|p_|context_length_limit|^note$)')
    cols=[c for c in df.columns if not drop.search(c)]
    # S8: R_* nan on SameWeekday is annotated
    if 'S8_' in p.name and 'note' in df.columns:
        metric_cols=[c for c in cols if not c.startswith('R_')]
        n=int(df[metric_cols].isna().sum().sum())
        rnan=int(df.loc[df.week_block=='SameWeekday_W4_over_W1',[c for c in df.columns if c.startswith('R_')]].isna().sum().sum()) if 'week_block' in df.columns else 0
        report.append(f'{p.name}: metric_nulls={n}; SameWeekday R_* nulls expected={rnan}')
        continue
    sub=df[cols]
    n=int(sub.isna().sum().sum())
    if n==0:
        report.append(f'{p.name}: OK')
    else:
        nulls=sub.isna().sum(); nulls=nulls[nulls>0]
        report.append(f'{p.name}: null_cells={n} {nulls.to_dict()}')
lines.append('## Actionable scan\n\n'+'\n'.join(report)+'\n')
Path(OUT/'logs/nan_residual_report.md').write_text(''.join(lines), encoding='utf-8')
print('\n'.join(report))
PY

echo "[fill done] $(date -Is)"
