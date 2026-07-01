#!/usr/bin/env bash
# All-in-one status dashboard for a main_HCP.py basis_regionwise run.
# Parses the run log: config checks, target-FC mode, progress, errors, results.
#
# Prints the status dashboard once, then live-streams the log (tail -f) until Ctrl-C.
#
# Usage:
#   bash watch_run.sh                          # defaults to run_basis_t20_s2000.log
#   bash watch_run.sh run_basis_t20_s2000.log
set -u

LOG="${1:-run_basis_t20_s2000.log}"
N_TRAIN="${N_TRAIN:-20}"      # expected train subjects (for progress bar)

c_g=$'\033[32m'; c_r=$'\033[31m'; c_y=$'\033[33m'; c_b=$'\033[1m'; c_0=$'\033[0m'
ok(){ printf "  ${c_g}PASS${c_0}  %s\n" "$1"; }
no(){ printf "  ${c_r}FAIL${c_0}  %s\n" "$1"; }
wa(){ printf "  ${c_y}WARN${c_0}  %s\n" "$1"; }

if [[ ! -f "$LOG" ]]; then echo "no log: $LOG"; exit 1; fi

# helper: grep -q on the log
has(){ grep -qE "$1" "$LOG"; }
cnt(){ grep -cE "$1" "$LOG"; }
last(){ grep -E "$1" "$LOG" | tail -n "${2:-1}"; }

echo "============================================================"
printf "  ${c_b}main_HCP run status${c_0}   log=%s\n" "$LOG"
echo "============================================================"

# ── 1. process ───────────────────────────────────────────────
PIDS=$(pgrep -af "python .*main_HCP.py" | awk '{print $1}' | tr '\n' ' ')
AGE=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
SIZE=$(stat -c %s "$LOG")
if [[ -n "$PIDS" ]]; then
  printf "  ${c_g}RUNNING${c_0}  pid: %s  | log last-write %ss ago | %s bytes\n" "$PIDS" "$AGE" "$SIZE"
else
  printf "  ${c_y}NOT RUNNING${c_0}  (finished/crashed) | log last-write %ss ago | %s bytes\n" "$AGE" "$SIZE"
fi

# ── 2. config checks ─────────────────────────────────────────
echo "------------------------------------------------------------"
echo "  CONFIG"
has "PARAMETER_MODE  : basis_regionwise" && ok "PARAMETER_MODE = basis_regionwise" || no "PARAMETER_MODE not basis_regionwise"
has "INFERENCE_MODEL : rwweib2"          && ok "INFERENCE_MODEL = rwweib2"          || no "INFERENCE_MODEL not rwweib2"
has "theta_dim       : 12"               && ok "theta_dim = 12"                     || no "theta_dim != 12"
has "N_REGIONS       : 360"              && ok "N_REGIONS = 360"                    || no "N_REGIONS != 360"
has "main_HCP expects the basis"         && no "GUARD RAISED (wrong mode)"          || ok "startup guard OK (no raise)"
# target FC mode
if   has "mode=SUBJECT-SPECIFIC"; then ok "target FC = SUBJECT-SPECIFIC (per-subject)"
elif has "mode=GROUP-AVG";        then wa "target FC = GROUP-AVG (set GROUP_AVG_FC=0 for per-subject)"
else wa "target FC mode line not printed yet"; fi
# basis dispatch + delays
has "basis_regionwise \(myelin/gradient\)" && ok "basis dispatch block reached" || wa "basis dispatch not printed yet"
if has "conduction delays ENABLED"; then wa "delays ENABLED (~9x cost)"
elif has "delays present but config.USE_DELAYS is False"; then ok "delays OFF (gated)"
else echo "        (delays: no sim line yet)"; fi

# ── 3. progress ──────────────────────────────────────────────
echo "------------------------------------------------------------"
echo "  PROGRESS"
SIMS=$(cnt "running .* sims")
printf "  sim batches started : %s  (each '<sid>: running N sims')\n" "$SIMS"
# crude train bar (first N_TRAIN batches = training set)
DONE=$(( SIMS<N_TRAIN ? SIMS : N_TRAIN ))
BAR=""; for ((i=0;i<N_TRAIN;i++)); do if ((i<DONE)); then BAR+="#"; else BAR+="."; fi; done
printf "  train sims          : [%s] %s/%s\n" "$BAR" "$DONE" "$N_TRAIN"
# stage markers
has "SNPE"           && echo "  stage: SNPE-C training reached"
has "posterior"      && echo "  stage: posterior built/sampling"
has "SBC"            && echo "  stage: SBC running"
has "baseline"       && echo "  stage: baseline eval"
last "running .* sims" 1 | sed 's/^/  last sim line: /'

# ── 4. errors ────────────────────────────────────────────────
echo "------------------------------------------------------------"
echo "  ERRORS"
ERR=$(grep -nE "Traceback|Error|Exception|CUDA|out of memory|OOM|assert|RuntimeError" "$LOG" \
      | grep -vE "no error|0 error" | tail -n 8)
if [[ -n "$ERR" ]]; then printf "${c_r}%s${c_0}\n" "$ERR"; else ok "no error/traceback found"; fi

# ── 5. results (when eval runs) ──────────────────────────────
echo "------------------------------------------------------------"
echo "  RESULTS (FC corr / metrics)"
RES=$(grep -nE "FC corr|fc_corr|FC[_ ]?r=|corr mean|RMSE|SBC|coverage|saved|DONE|END" "$LOG" | tail -n 12)
if [[ -n "$RES" ]]; then echo "$RES"; else echo "  (none yet — appears after training+eval)"; fi

# ── 6. live follow (continuous stream until Ctrl-C) ──────────
echo "------------------------------------------------------------"
printf "  ${c_b}LIVE${c_0}  (last 12 lines, then streaming — Ctrl-C to stop)\n"
echo "============================================================"
exec tail -n 12 -f "$LOG"
