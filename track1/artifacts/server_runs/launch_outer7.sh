#!/bin/bash
set -euo pipefail
REPO=/root/DYQ/macau-tourism-forecasting-protocol
PY=/root/miniforge3/bin/python3
LOG=/root/DYQ/run-logs
SESSION=macau-outer

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session $SESSION already exists"
  exit 1
fi

launch() {
  local name="$1"
  local body="$2"
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux new-session -d -s "$SESSION" -n "$name" "bash -lc $(printf '%q' "$body")"
  else
    tmux new-window -t "$SESSION" -n "$name" "bash -lc $(printf '%q' "$body")"
  fi
}

# Fast zero-shot models: one fold at a time, split across the two GPUs.
launch timesfm "$PY $LOG/run_foundation_outer.py --model timesfm_3 --python /root/DYQ/timesfm-cu128/bin/python --gpu 0 --batch-size 8 --jobs 2; echo TIMESFM_EXIT \$?; exec bash"
launch chronos "$PY $LOG/run_foundation_outer.py --model chronos_2 --python /root/DYQ/chronos-cu128/bin/python --gpu 0 --batch-size 8 --jobs 2; echo CHRONOS_EXIT \$?; exec bash"
launch moirai "$PY $LOG/run_foundation_outer.py --model moirai --python /root/DYQ/moirai-cu128/bin/python --gpu 1 --batch-size 1 --jobs 2; echo MOIRAI_EXIT \$?; exec bash"
launch granite "$PY $LOG/run_foundation_outer.py --model granite_patchtst_fm --python /root/DYQ/granite-cu128/bin/python --gpu 1 --batch-size 1 --jobs 2; echo GRANITE_EXIT \$?; exec bash"
launch lagllama "$PY $LOG/run_foundation_outer.py --model lag_llama --python /root/DYQ/cu121-py311/bin/python --gpu 1 --batch-size 1 --jobs 2; echo LAG_EXIT \$?; exec bash"

# PatchTST and Informer are the long per-origin training jobs.
# Each is split across both GPUs, with several origins in parallel, so both cards stay busy.
launch patchtst "set -m
$PY $LOG/run_foundation_outer.py --model patchtst --python /root/DYQ/timesfm-cu128/bin/python --gpu 0 --batch-size 32 --jobs 4 --fold-mod 2 --fold-rem 1 &
p0=\$!
$PY $LOG/run_foundation_outer.py --model patchtst --python /root/DYQ/timesfm-cu128/bin/python --gpu 1 --batch-size 32 --jobs 4 --fold-mod 2 --fold-rem 0 &
p1=\$!
wait \$p0; e0=\$?
wait \$p1; e1=\$?
echo PATCHTST_EXIT \$e0 \$e1
exec bash"

launch informer "set -m
$PY $LOG/run_foundation_outer.py --model informer --python /root/DYQ/cu121-py311/bin/python --gpu 0 --batch-size 16 --jobs 6 --fold-mod 2 --fold-rem 1 &
p0=\$!
$PY $LOG/run_foundation_outer.py --model informer --python /root/DYQ/cu121-py311/bin/python --gpu 1 --batch-size 16 --jobs 6 --fold-mod 2 --fold-rem 0 &
p1=\$!
wait \$p0; e0=\$?
wait \$p1; e1=\$?
echo INFORMER_EXIT \$e0 \$e1
exec bash"

tmux ls
echo launched
