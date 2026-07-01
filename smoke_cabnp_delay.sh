#!/usr/bin/env bash
# Smoke run of main_HCP.py — CAB-NP 381 SC + tract-length delays, tiny sizes.
# Just runs main_HCP with small params (env overrides). No file edits, no long
# command line. Refuses to launch if a main_HCP run is already alive (prevents
# the double-launch that interleaves the log).
#
#   bash smoke_cabnp_delay.sh
#   N_TRAIN=2 bash watch_run.sh run_smoke_cabnp_delay.log   # monitor
set -euo pipefail

cd /scratch/home/wog3597/vbi
LOG=run_smoke_cabnp_delay.log

# guard: one run at a time
if pgrep -af "python .*main_HCP.py" >/dev/null; then
  echo "ABORT: a main_HCP.py run is already alive —"
  pgrep -af "python .*main_HCP.py"
  echo "kill it first (kill <pid>) or wait for it to finish, then rerun."
  exit 1
fi

PARAMETER_MODE=basis_regionwise REQUIRE_BASIS=1 \
SC_DATASET=cabnp381 SC_FILE=HCP_CABNP381_SC_first100.mat \
USE_DELAYS=1 GROUP_AVG_FC=0 \
N_SUBJECTS=4 N_TRAIN=2 N_VAL=1 N_TEST=1 N_SIM=64 GPU_BATCH=64 \
nohup python main_HCP.py > "$LOG" 2>&1 &

PID=$!
echo "started  PID $PID  ->  log: $LOG"
echo "monitor  :  N_TRAIN=2 bash watch_run.sh $LOG"
echo "key check:  grep -nE 'cabnp381|CAB-NP|conduction delays|SUBJECT-SPECIFIC|Traceback|Error' $LOG"
