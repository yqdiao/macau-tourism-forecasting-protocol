#!/usr/bin/env bash
LOG=/root/DYQ/macau-tourism-forecasting-protocol/RQ1/raw/extra_logs
RQ1=/root/DYQ/macau-tourism-forecasting-protocol/RQ1
echo "=== tmux rq1-extra ==="
tmux list-windows -t rq1-extra -F '  #{window_index}:#{window_name}  #{pane_current_command}' 2>/dev/null || echo '  (session missing)'
echo
echo "=== GPU five-quantile progress ==="
python3 - <<'PY'
import json
from pathlib import Path
log=Path('/root/DYQ/macau-tourism-forecasting-protocol/RQ1/raw/extra_logs')
for name in ['gpu0','gpu1']:
    p=log/f'{name}.log'
    rows=[json.loads(l) for l in p.read_text().splitlines() if l.startswith('{')] if p.exists() else []
    done=log/f'{name}.done'
    mark=done.read_text().strip() if done.exists() else 'RUNNING'
    if rows:
        print(f"  {name}: {rows[-1]['job']}  last={rows[-1]['model']}/{rows[-1]['fold']}  {rows[-1]['seconds']}s  [{mark}]")
    else:
        print(f"  {name}: (no log yet) [{mark}]")
PY
echo
echo "=== light timing ==="
if [ -f "$RQ1/raw/light_benchmark.jsonl" ]; then
  wc -l "$RQ1/raw/light_benchmark.jsonl" | awk '{print "  lines="$1" (expect 30 = 3 models x 10 runs)"}'
  tail -n 1 "$RQ1/raw/light_benchmark.jsonl"
else
  echo "  not started (waits for both GPUs)"
fi
echo
echo "=== chain / build ==="
if [ -s "$LOG/chain.log" ]; then cat "$LOG/chain.log"; else echo "  waiting for GPUs..."; fi
[ -f "$LOG/build_five_tau.log" ] && { echo "--- build_five_tau.log ---"; tail -n 10 "$LOG/build_five_tau.log"; }
echo
echo "=== RQ1 five-tau outputs present? ==="
for f in appendix_five_tau_point_forecast_as_decision.csv appendix_five_tau_quantile_loss.csv \
         table4_timing_sd_cv.csv figure4_pareto_coordinates.csv \
         figures/five_tau_point_forecast_as_decision.png \
         figures/five_tau_quantile_reopening_891.png \
         figures/five_tau_quantile_pandemic_351.png; do
  if [ -e "$RQ1/$f" ]; then echo "  OK  $f"; else echo "  ..  $f"; fi
done
