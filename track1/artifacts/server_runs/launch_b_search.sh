#!/bin/bash
# B01-B12 candidate search for PatchTST and Informer, split across both GPUs.
set -euo pipefail
REPO=/root/DYQ/macau-tourism-forecasting-protocol
PY=/root/miniforge3/bin/python3
RUNNER=$REPO/track1/scripts/run_foundation_b_search.py
SESSION=macau-bsearch

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session $SESSION already exists"
  exit 1
fi

launch() {
  local name="$1" body="$2"
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux new-session -d -s "$SESSION" -n "$name" "bash -lc $(printf '%q' "$body")"
  else
    tmux new-window -t "$SESSION" -n "$name" "bash -lc $(printf '%q' "$body")"
  fi
}

for gpu in 0 1; do
  launch "informer-g$gpu" "cd $REPO; echo START \$(date -u +%FT%TZ)
$PY $RUNNER --model informer --python /root/DYQ/cu121-py311/bin/python --gpu $gpu --batch-size 16 --jobs 6 --job-mod 2 --job-rem $gpu
echo INFORMER_G${gpu}_EXIT \$? \$(date -u +%FT%TZ); exec bash"
  launch "patchtst-g$gpu" "cd $REPO; echo START \$(date -u +%FT%TZ)
$PY $RUNNER --model patchtst --python /root/DYQ/timesfm-cu128/bin/python --gpu $gpu --batch-size 32 --jobs 4 --job-mod 2 --job-rem $gpu
echo PATCHTST_G${gpu}_EXIT \$? \$(date -u +%FT%TZ); exec bash"
done

tmux ls
echo launched
