#!/usr/bin/env python
# coding: utf-8
"""CPU assembly smoke for the SC-conditioned amortized NPE wiring.

No GPU / cupy / SNPE / simulator. Exercises ONLY the data-assembly path that
main_HCP.py Step 1.5 + Step 4 + the eval x_obs builder use:

  1. build_sc_table over ALL fake subjects (train+val+test).
  2. ScChannelScaler fit on TRAIN rows only, transform the whole table.
  3. Fabricate subj_ids + fc_raw for a few train sims; build the training
     x=[idx|fc] exactly as Step 4 does.
  4. Construct MultiChannelMatrixEmbedding(SC_TABLE); forward(torch.tensor(x))
     -> assert (B, EMBED_DIM) finite.
  5. Build eval x_obs=[idmap[val_sid] | fc_emp_uppertri]; forward -> finite.

Run:
    PARAMETER_MODE=basis_regionwise INFERENCE_MODEL=rwweib2 \
      python tests/smoke/sc_condition_assembly.py
"""
import os
import sys

# Repo-root on sys.path so `inference`, `config`, etc. import like main_HCP.
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch

import config
from inference.sc_channels import build_sc_table, ScChannelScaler, CHANNELS
from inference.embedding import MultiChannelMatrixEmbedding
from features.fc import fc_to_upper_tri


def main():
    rng = np.random.default_rng(0)

    # ── Fake cohort: ~10 subjects, N=360 (so FC_DIM matches config) ──────────
    N = int(config.N_REGIONS)                 # 360 under the active HCP config
    FC_DIM = int(config.FC_DIM)               # N*(N-1)//2
    EMBED_DIM = int(config.EMBED_DIM)
    n_subj = 10
    channels = ("sc_weight", "sc_mask")       # match main_HCP default SC_CHANNELS

    sids = [1000 + i for i in range(n_subj)]
    subject_data = {}
    for sid in sids:
        raw = rng.random((N, N)).astype(np.float64)
        keep = (rng.random((N, N)) < 0.10)
        keep = np.triu(keep, k=1)
        keep = keep | keep.T
        sc = raw * keep
        sc = np.triu(sc, k=1)
        sc = sc + sc.T
        if sc.max() > 0:
            sc = sc / sc.max()                # maxnorm-like
        mask = (sc > 0).astype(np.float64)
        delays = (rng.random((N, N)) * mask)
        delays = np.triu(delays, k=1)
        delays = delays + delays.T
        # symmetric empirical FC in [-1, 1] with unit diagonal
        a = rng.standard_normal((N, N)).astype(np.float64)
        fc = np.tanh((a + a.T) / 2.0)
        np.fill_diagonal(fc, 1.0)
        subject_data[sid] = {"sc": sc, "delays": delays, "fc": fc}

    # Split: 6 train / 2 val / 2 test (mirrors the leakage-safe contract).
    train = sids[:6]
    val = sids[6:8]
    test = sids[8:]

    # ── Step 1.5 replica: full table over ALL subjects, scaler on TRAIN only ─
    _all = list(train) + list(val) + list(test)
    sctab_raw, scidmap = build_sc_table(subject_data, _all, channels)
    trows = [scidmap[int(s)] for s in train]
    scsc = ScChannelScaler().fit(sctab_raw[trows], channels)
    SC_TABLE = scsc.transform(sctab_raw)
    config.SC_IDMAP = scidmap

    C = len(channels)
    assert SC_TABLE.shape == (n_subj, C, N, N), f"bad SC_TABLE {SC_TABLE.shape}"
    assert np.all(np.isfinite(SC_TABLE)), "SC_TABLE not finite"
    print(f"  SC_TABLE shape={SC_TABLE.shape}  channels={list(channels)}  "
          f"#train_rows={len(trows)}")

    # ── Step 4 replica: training x=[idx|fc] for a few train sims ─────────────
    n_sims = 5
    sim_subj = rng.choice(train, size=n_sims).astype(np.int64)   # subj_ids
    fc_raw = rng.uniform(-1.0, 1.0, size=(n_sims, FC_DIM)).astype(np.float32)
    idx_col = np.array(
        [config.SC_IDMAP[int(s)] for s in sim_subj], dtype=np.float32)[:, None]
    x_input = np.concatenate([idx_col, fc_raw.astype(np.float32)], axis=1)
    assert x_input.shape == (n_sims, 1 + FC_DIM), f"bad x_input {x_input.shape}"
    print(f"  train x=[idx|fc] shape={x_input.shape}  (1 + FC_DIM={1 + FC_DIM})")

    # ── Encoder: MultiChannelMatrixEmbedding over the scaled SC table ────────
    emb = MultiChannelMatrixEmbedding(
        fc_input_dim=FC_DIM, sc_table=SC_TABLE,
        out_dim=EMBED_DIM, n_regions=N, use_fc_mask=True,
    )
    out = emb(torch.tensor(x_input))
    assert out.shape == (n_sims, EMBED_DIM), f"bad embed out {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "training embedding output not finite"
    print(f"  train forward -> {tuple(out.shape)}  finite=True")

    # ── Eval replica: x_obs=[idmap[val_sid] | fc_emp_uppertri] ───────────────
    val_sid = val[0]
    fc_emp = subject_data[val_sid]["fc"]
    fc_vec = np.asarray(fc_to_upper_tri(fc_emp), dtype=np.float32).reshape(-1)
    assert fc_vec.shape[0] == FC_DIM, f"fc upper-tri {fc_vec.shape[0]} != {FC_DIM}"
    x_obs = np.concatenate(
        [np.array([[config.SC_IDMAP[int(val_sid)]]], dtype=np.float32),
         fc_vec[None, :]], axis=1)
    assert x_obs.shape == (1, 1 + FC_DIM), f"bad x_obs {x_obs.shape}"
    emb.eval()
    with torch.no_grad():
        out_obs = emb(torch.tensor(x_obs))
    assert out_obs.shape == (1, EMBED_DIM), f"bad eval out {tuple(out_obs.shape)}"
    assert torch.isfinite(out_obs).all(), "eval embedding output not finite"
    print(f"  eval  x_obs[idx={config.SC_IDMAP[int(val_sid)]}] forward -> "
          f"{tuple(out_obs.shape)}  finite=True")

    print("sc_condition_assembly smoke: ALL PASS")


if __name__ == "__main__":
    main()
