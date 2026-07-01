#!/usr/bin/env bash
# Real main_HCP.py run — SMOKE=0, CAB-NP 381 SC + tract-length delays +
# per-subject FC + correct (0-3) BASIS_BOUNDS. Sizes overridable via env.
# Clears the stale feature cache (not invalidated on a bounds/config change)
# and refuses to double-launch.
#
#   bash run_real.sh                                  # full 70/10/20, N_SIM=2000
#   N_TRAIN=10 N_VAL=5 N_TEST=10 N_SIM=1000 bash run_real.sh   # smaller real run
#   USE_DELAYS=0 bash run_real.sh                     # delays off (faster)
#   bash watch_run.sh "$LOG"                          # monitor
set -euo pipefail

cd /scratch/home/wog3597/vbi
LOG="${LOG:-run_real_cabnp_delay.log}"

# one run at a time
if pgrep -af "python .*main_HCP.py" >/dev/null; then
  echo "ABORT: a main_HCP.py run is already alive —"
  pgrep -af "python .*main_HCP.py"
  echo "kill it first or wait, then rerun."
  exit 1
fi

# stale feature cache: meta json does NOT capture BASIS_BOUNDS, so a bounds/config
# change is not auto-invalidated. Force fresh simulation.
rm -f output_hcp/features_stage1.npz output_hcp/features_stage1_meta.json
echo "cleared stale feature cache."

SMOKE=0 \
N_TRAIN="${N_TRAIN:-70}" N_VAL="${N_VAL:-10}" N_TEST="${N_TEST:-20}" \
N_SIM="${N_SIM:-2000}" GPU_BATCH="${GPU_BATCH:-2000}" \
USE_DELAYS="${USE_DELAYS:-1}" \
nohup python main_HCP.py > "$LOG" 2>&1 &

PID=$!
echo "started  PID $PID  ->  log: $LOG"
echo "sizes    N_TRAIN=${N_TRAIN:-70} N_SIM=${N_SIM:-2000} USE_DELAYS=${USE_DELAYS:-1}"
echo "monitor  :  bash watch_run.sh $LOG"
echo "WARN: delays ON + 70x2000 ~ many hours. Use smaller N_TRAIN/N_SIM to test."
