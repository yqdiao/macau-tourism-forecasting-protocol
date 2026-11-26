#!/usr/bin/env bash
# CPU timing waits for both GPU jobs, as in the registered benchmark, then the tables are rebuilt.
set -u
cd /root/DYQ/macau-tourism-forecasting-protocol
LOG=RQ1/raw/extra_logs
while ! { [ -f "$LOG/gpu0.done" ] && [ -f "$LOG/gpu1.done" ]; }; do
  sleep 30
done
if grep -q FAILED "$LOG"/gpu*.done; then
  echo "a GPU job failed; see $LOG" >&2
  exit 1
fi
/root/miniforge3/bin/python -u RQ1/benchmark_light.py > "$LOG/light_benchmark.log" 2>&1 || { echo "timing failed" >&2; exit 1; }
/root/DYQ/macau-tourism-world-model/.venv-server/bin/python -u RQ1/build_rq1_five_tau.py > "$LOG/build_five_tau.log" 2>&1 \
  && echo "RQ1 extra outputs complete" || echo "assembly failed; see $LOG/build_five_tau.log"
