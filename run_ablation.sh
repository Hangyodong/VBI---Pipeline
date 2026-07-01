#!/usr/bin/env bash
# Stage 7 — SC-conditioning ablation A/B/C/D (run on a GPU node).
#
# Decision-9 arms (delay channel dropped per the delay-sanity result; sc_channels
# still supports "delay" if you want to add it back):
#   A  q(theta | FC)                      SC_CONDITION=0           (FC-only baseline)
#   B  q(theta | FC, SC_weight)           SC_CHANNELS=sc_weight
#   C  q(theta | FC, SC_mask)             SC_CHANNELS=sc_mask
#   D  q(theta | FC, SC_weight, SC_mask)  SC_CHANNELS=sc_weight,sc_mask
#
# Parity: all arms share ONE simulation cache. Arm A runs first and populates
# output_hcp/features_stage1.npz (theta, fc_raw, subj_ids); B/C/D reuse it (the
# Step-2 sim is identical regardless of conditioning -> sc_condition/sc_channels
# are intentionally NOT in the cache key), so only the NPE re-trains. theta/SC/
# split stay identical across arms -> the ONLY difference is the SC channel.
#
# Decisive comparison (eval plan): corr(D) - corr(A) on held-out TEST subjects,
# plus the Stage-6 SC-permutation control delta(post-perm).
set -u

export VBI_SC_SCALE=maxnorm
# Shared config (same split/seed/N_SIM across arms == parity).
export N_SUBJECTS=100 N_TRAIN=80 N_VAL=10 N_TEST=10
export N_SIM=1000 GPU_BATCH=1000 SMOKE=0
export USE_DELAYS=0 PARAMETER_MODE=basis_regionwise INFERENCE_MODEL=rwweib2
export SC_DATASET=cabnp381 SC_FILE=HCP_CABNP381_SC_first100.mat
export RUN_SC_DIAG=1          # Stage-6 diagnostics (prior-predictive + SC-permutation)
# export DETERMINISTIC=1      # uncomment for bit-reproducible NPE training (slower)

mkdir -p logs

run_arm () {
  local name="$1"
  echo "================ ARM ${name} (SC_CONDITION=${SC_CONDITION:-0} SC_CHANNELS=${SC_CHANNELS:-none}) ================"
  python main_HCP.py 2>&1 | tee "logs/ablation_${name}.log"
  echo "  [arm ${name}] done -> logs/ablation_${name}.log"
}

# A first: simulates + caches. B/C/D reuse the cache (sim skipped, fast).
SC_CONDITION=0                               run_arm A
SC_CONDITION=1 SC_CHANNELS=sc_weight          run_arm B
SC_CONDITION=1 SC_CHANNELS=sc_mask            run_arm C
SC_CONDITION=1 SC_CHANNELS=sc_weight,sc_mask  run_arm D

echo
echo "================ ABLATION SUMMARY ================"
python tools/ablation_summary.py \
  A:logs/ablation_A.log B:logs/ablation_B.log \
  C:logs/ablation_C.log D:logs/ablation_D.log
