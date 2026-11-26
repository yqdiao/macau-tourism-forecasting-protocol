#!/usr/bin/env bash
set -euo pipefail
cd /root/DYQ/macau-tourism-forecasting-protocol/RQ2
PY=/root/DYQ/macau-tourism-world-model/.venv-server/bin/python
# fallback
if [ ! -x "$PY" ]; then PY=/root/miniforge3/bin/python; fi
export PYTHONPATH=/root/DYQ/macau-tourism-forecasting-protocol/RQ2
LOG=/root/DYQ/macau-tourism-forecasting-protocol/RQ2_outputs/logs
mkdir -p "$LOG"
{
  echo "[start] $(date -Is)"
  $PY -u build_panels.py
  $PY -u m421_regime.py
  $PY -u m422_scales.py
  $PY -u m423_history.py
  $PY -u m424_conditions.py
  $PY -u m425_boundary.py
  $PY -u plot_figures.py
  $PY -u write_summary.py
  echo "[done] $(date -Is)"
} 2>&1 | tee "$LOG/run_all.log"
