#!/bin/bash
set -euo pipefail
REPO=/root/DYQ/macau-tourism-forecasting-protocol
PY=/root/miniforge3/bin/python3
OUT=$REPO/track1/results/outer_server_all
LOG=/root/DYQ/run-logs
mkdir -p "$OUT" "$LOG"

if tmux has-session -t macau 2>/dev/null; then
  echo "tmux session macau already exists"
  exit 1
fi

# 1. Score the sealed B reproduction. No --freeze.
tmux new-session -d -s macau -n score "bash -lc '
echo SCORE_START \$(date -u +%Y-%m-%dT%H:%M:%SZ)
cd $REPO
$PY track1/code/evaluate_b_validation.py \
  --predictions track1/results/b_validation_server_repro \
  --targets shared/data/targets.parquet \
  --task-manifest shared/data/task_manifest.json \
  --output track1/evaluation/b_validation_server_repro
echo SCORE_DONE \$(date -u +%Y-%m-%dT%H:%M:%SZ)
$PY - << \"PY\"
import json
from pathlib import Path
import pandas as pd
root = Path(\"$REPO\")
summary = pd.read_parquet(root/\"track1/evaluation/b_validation_server_repro/candidate_summary.parquet\")
frozen = json.loads((root/\"track1/configs/frozen_baseline_configs.yaml\").read_text())
status = json.loads((root/\"track1/evaluation/b_validation_server_repro/validation_status.json\").read_text())
print(json.dumps({\"complete\": status[\"complete\"], \"completed_paths\": status[\"completed_paths\"], \"expected_paths\": status[\"expected_paths\"], \"terminal_failure_paths\": status[\"terminal_failure_paths\"]}, ensure_ascii=False))
rows = []
for model, entry in frozen[\"models\"].items():
    cid = entry[\"candidate_id\"]
    hit = summary[(summary.model==model) & (summary.candidate_id==cid)].iloc[0]
    official = entry.get(\"validation_score\")
    reproduced = None if pd.isna(hit.mean_MASE7) else float(hit.mean_MASE7)
    delta = None if official is None or reproduced is None else reproduced - official
    rows.append({\"model\": model, \"candidate_id\": cid, \"frozen_MASE7\": official, \"repro_MASE7\": reproduced, \"delta\": delta, \"eligible\": bool(hit.eligible)})
out = root/\"track1/evaluation/b_validation_server_repro/frozen_comparison.json\"
out.write_text(json.dumps(rows, indent=2) + \"\n\")
print(json.dumps(rows, ensure_ascii=False, indent=2))
PY
echo SCORE_COMPARE_DONE \$(date -u +%Y-%m-%dT%H:%M:%SZ)
exec bash
'"

# 2. Outer O01-O33, history=all, frozen candidates, parallel groups.
launch_outer() {
  local name="$1" models="$2" workers="$3"
  tmux new-window -t macau -n "$name" "bash -lc '
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
cd $REPO
echo OUTER_START $name \$(date -u +%Y-%m-%dT%H:%M:%SZ)
$PY $LOG/run_outer_frozen.py --models $models --workers $workers --output $OUT
echo OUTER_DONE $name \$(date -u +%Y-%m-%dT%H:%M:%SZ)
exec bash
'"
}
launch_outer fast "seasonal_naive,ets,sarima,ridge,linear_regression,lasso,elasticnet" 8
launch_outer svr svr 4
launch_outer lgb lightgbm 3
launch_outer gb gradient_boosting 6
launch_outer rf random_forest 8
launch_outer transformer transformer 8
launch_outer prf parrot_random_forest_800 10

# 3. Foundation B01-B12 default runtime. Two processes per GPU.
tmux new-window -t macau -n gpu0a "bash -lc '
echo GPU0A_START \$(date -u +%Y-%m-%dT%H:%M:%SZ)
$PY $LOG/run_foundation_b.py --model timesfm_3 --python /root/DYQ/timesfm-cu128/bin/python --gpu 0 --batch-size 8
$PY $LOG/run_foundation_b.py --model chronos_2 --python /root/DYQ/chronos-cu128/bin/python --gpu 0 --batch-size 8
$PY $LOG/run_foundation_b.py --model moirai --python /root/DYQ/moirai-cu128/bin/python --gpu 0 --batch-size 1
echo GPU0A_DONE \$(date -u +%Y-%m-%dT%H:%M:%SZ)
exec bash
'"
tmux new-window -t macau -n gpu0b "bash -lc '
echo GPU0B_START \$(date -u +%Y-%m-%dT%H:%M:%SZ)
$PY $LOG/run_foundation_b.py --model patchtst --python /root/DYQ/timesfm-cu128/bin/python --gpu 0 --batch-size 32
echo GPU0B_DONE \$(date -u +%Y-%m-%dT%H:%M:%SZ)
exec bash
'"
tmux new-window -t macau -n gpu1a "bash -lc '
echo GPU1A_START \$(date -u +%Y-%m-%dT%H:%M:%SZ)
$PY $LOG/run_foundation_b.py --model granite_patchtst_fm --python /root/DYQ/granite-cu128/bin/python --gpu 1 --batch-size 1
$PY $LOG/run_foundation_b.py --model lag_llama --python /root/DYQ/cu121-py311/bin/python --gpu 1 --batch-size 1
echo GPU1A_DONE \$(date -u +%Y-%m-%dT%H:%M:%SZ)
exec bash
'"
tmux new-window -t macau -n gpu1b "bash -lc '
echo GPU1B_START \$(date -u +%Y-%m-%dT%H:%M:%SZ)
$PY $LOG/run_foundation_b.py --model informer --python /root/DYQ/cu121-py311/bin/python --gpu 1 --batch-size 16
echo GPU1B_DONE \$(date -u +%Y-%m-%dT%H:%M:%SZ)
exec bash
'"

tmux ls
echo launched
