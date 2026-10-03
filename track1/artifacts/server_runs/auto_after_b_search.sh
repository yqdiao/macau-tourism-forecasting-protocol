#!/bin/bash
# Wait for the B search, score B only, freeze, commit the freeze, then launch the 3-seed outer rerun.
set -uo pipefail
REPO=/root/DYQ/macau-tourism-forecasting-protocol
PY=/root/miniforge3/bin/python3
JOURNAL=/root/DYQ/run-logs/b_search_nodes.jsonl
SESSION=macau-bsearch
OUTER_SESSION=macau-outer-frozen
RUNNER=$REPO/track1/scripts/run_foundation_outer_frozen.py
stamp() { date -u +%FT%TZ; }

echo "[$(stamp)] waiting for 4 B-search runners to finish"
while true; do
  done_count=$(grep -c '"event": "runner_done"' "$JOURNAL" 2>/dev/null || true)
  live=$(pgrep -fc 'run_foundation_b_search.py' || true)
  if [ "${done_count:-0}" -ge 4 ] && [ "${live:-0}" -eq 0 ]; then break; fi
  if [ "${live:-0}" -eq 0 ] && [ "${done_count:-0}" -lt 4 ]; then
    echo "[$(stamp)] runners exited without runner_done (${done_count}/4); stopping, check tmux windows"
    exit 1
  fi
  jobs_done=$(grep -c '"event": "job_done"' "$JOURNAL" 2>/dev/null || true)
  echo "[$(stamp)] runners_done=${done_count}/4 jobs_done=${jobs_done}/576"
  sleep 600
done
grep '"event": "runner_done"' "$JOURNAL"

echo "[$(stamp)] scoring B candidates and freezing"
cd "$REPO"
if ! $PY track1/code/evaluate_foundation_b_search.py --freeze; then
  echo "[$(stamp)] evaluation or freeze failed; outer rerun NOT launched"
  exit 1
fi
$PY - <<'PY'
import pandas as pd
s = pd.read_parquet("track1/evaluation/foundation_b_search/candidate_summary.parquet")
cols = [c for c in ["model", "candidate_id", "eligible", "mean_MASE7", "seed_sd_MASE7", "mean_fit_seconds"] if c in s]
print(s.sort_values(["model", "mean_MASE7"])[cols].to_string(index=False))
PY

git add track1/configs/frozen_foundation_configs.yaml
git commit -q -m "按登记规则在 B 组冻结 PatchTST 与 Informer 参数，零样本模型登记为默认配置。" \
  && git log -1 --format='[freeze commit] %H %s'

if tmux has-session -t "$OUTER_SESSION" 2>/dev/null; then
  echo "tmux session $OUTER_SESSION already exists; not launching"
  exit 1
fi
launch() {
  local name="$1" body="$2"
  if ! tmux has-session -t "$OUTER_SESSION" 2>/dev/null; then
    tmux new-session -d -s "$OUTER_SESSION" -n "$name" "bash -lc $(printf '%q' "$body")"
  else
    tmux new-window -t "$OUTER_SESSION" -n "$name" "bash -lc $(printf '%q' "$body")"
  fi
}
for gpu in 0 1; do
  launch "informer-g$gpu" "cd $REPO; echo START \$(date -u +%FT%TZ)
$PY $RUNNER --model informer --python /root/DYQ/cu121-py311/bin/python --gpu $gpu --batch-size 16 --jobs 6 --job-mod 2 --job-rem $gpu
echo INFORMER_OUTER_G${gpu}_EXIT \$? \$(date -u +%FT%TZ); exec bash"
  launch "patchtst-g$gpu" "cd $REPO; echo START \$(date -u +%FT%TZ)
$PY $RUNNER --model patchtst --python /root/DYQ/timesfm-cu128/bin/python --gpu $gpu --batch-size 32 --jobs 4 --job-mod 2 --job-rem $gpu
echo PATCHTST_OUTER_G${gpu}_EXIT \$? \$(date -u +%FT%TZ); exec bash"
done
echo "[$(stamp)] outer rerun launched in tmux session $OUTER_SESSION"
tmux list-windows -t "$OUTER_SESSION"
