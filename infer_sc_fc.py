#!/usr/bin/env python
"""infer_sc_fc.py — sbi-native amortized q(theta | SC, FC), iterating on the CACHE
(no re-simulation). Implements two inference-logic changes:

  1. MAXIMAL sbi: the flow + embedding + trainer are all sbi built-ins
     (sbi.neural_nets.posterior_nn(model='nsf', embedding_net=FCEmbedding), SNPE_C).
     We only supply (theta, x) pairs + the SC/FC feature engineering (data prep).
  2. SC-conditioning: x = concat(FC features, SC summary) so the network learns
     q(theta | SC, FC) instead of q(theta | FC). The goal is amortization to an
     ARBITRARY new subject's (SC, FC).

Why cache-only: SC is a per-subject KNOWN covariate; the simulated FCs already
exist in output_hcp/features_stage1.npz. We just attach each sim's subject SC
summary to its x. => zero new simulation; fast inference-logic iteration.

Metric (cheap, on cached sims): held-out-SUBJECT theta-recovery
  recovery = corr(theta_true, theta_posterior_mean) on sims from subjects NOT in
  the training split. Compares q(theta|FC) vs q(theta|SC,FC), plus an
  SC-PERMUTATION control (shuffle SC across held-out subjects -> recovery must
  DROP if the net genuinely uses SC; if unchanged, SC is dead weight).
(Empirical FC-corr eval needs GPU re-sim and is the pipeline's job, not here.)

Usage
-----
  python infer_sc_fc.py                      # full (GPU if available)
  python infer_sc_fc.py --smoke              # CPU: subsample, few epochs
  python infer_sc_fc.py --fc-dim 256 --emb-dim 64 --flow nsf
"""
import argparse
import os
import numpy as np

os.environ.setdefault("VBI_SC_SCALE", "maxnorm")

# config setup so data_loader_hcp resolves FC/SC paths (mirror node_ceiling.py)
from pipeline_setup import PipelineConfig, setup_pipeline
_cfg = PipelineConfig(
    DATA_DIR="/scratch/home/wog3597/vbi", OUTPUT_DIR="./output_hcp",
    FC_FILE="HCP_FC.mat", SC_FILE="HCP_CABNP381_SC_first100.mat",
    N_REGIONS=360, N_SUBJECTS=100, N_TRAIN=70, N_VAL=10, N_TEST=20, SEED=42,
    N_SIM=1, GPU_BATCH=1, T_END_MS=630_000.0, T_CUT_MS=60_000.0, DT=1.0,
    DECIMATE=720, TR_SEC=0.72)
setup_pipeline(_cfg, print_summary=False)
import config
config.SC_DATASET = "cabnp381"; config.N_REGIONS = 360


# ── SC summary features (support-robust, present in EVERY subject) ────────────
def sc_summary(sc):
    """360x360 SC -> per-subject summary vector. Node strength (360) + global
    scalars. No raw edges (avoids the 1.1%-common-support OOD trap)."""
    sc = np.asarray(sc, dtype=np.float64)
    np.fill_diagonal(sc, 0.0)
    strength = sc.sum(1)                                  # (R,) node strength
    pos = sc[sc > 0]
    glob = np.array([
        np.log1p(sc.sum()),                              # log total weight
        float((sc > 0).mean()),                          # density
        strength.mean(), strength.std(),
        float(np.median(pos)) if pos.size else 0.0,
        float(pos.mean()) if pos.size else 0.0,
    ])
    return np.concatenate([strength, glob])              # (R+6,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="output_hcp/features_stage1.npz")
    ap.add_argument("--fc-dim", type=int, default=256, help="FC PCA comps (no whiten)")
    ap.add_argument("--emb-dim", type=int, default=64, help="sbi FCEmbedding output dim")
    ap.add_argument("--flow", default="nsf", choices=["nsf", "maf"])
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--transforms", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.2, help="held-out SUBJECT fraction")
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--eval-sims", type=int, default=800, help="held-out sims scored")
    ap.add_argument("--post-samples", type=int, default=50)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.fc_dim, args.max_epochs, args.eval_sims, args.post_samples = 32, 25, 200, 20

    import torch
    from sklearn.decomposition import PCA
    from sbi.inference import SNPE_C
    from sbi.neural_nets import posterior_nn
    from sbi.neural_nets.embedding_nets import FCEmbedding
    from sbi.utils import BoxUniform

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.RandomState(42)
    print(f"  device={dev}  flow={args.flow}  fc_dim={args.fc_dim}  emb_dim={args.emb_dim}")

    # ── load cache (theta already scaled to [-1,1]) ──────────────────────────
    z = np.load(args.cache, mmap_mode="r")
    theta = np.asarray(z["theta_scaled"], dtype=np.float32)       # (N,12) in [-1,1]
    subj = np.asarray(z["subj_ids"]).astype(int)                  # (N,)
    N, P = theta.shape
    subjects = np.unique(subj)
    print(f"  cache: {N} sims, {len(subjects)} subjects, theta_dim={P}")
    row_map = np.arange(N)                                        # local idx -> fcmm row
    if args.smoke:
        keep = set(rng.choice(subjects, min(12, len(subjects)), replace=False).tolist())
        sel = np.array([s in keep for s in subj])
        row_map = np.where(sel)[0]; theta = theta[row_map]; subj = subj[row_map]
        N = len(row_map); subjects = np.unique(subj)
        print(f"  [smoke] subsampled -> {N} sims / {len(subjects)} subjects")

    # ── subject-disjoint split (held-out SUBJECTS test amortization) ─────────
    perm = rng.permutation(subjects)
    n_val = max(1, int(len(subjects) * args.val_frac))
    val_subj = set(perm[:n_val].tolist()); train_subj = set(perm[n_val:].tolist())
    tr = np.array([s in train_subj for s in subj]); va = ~tr
    print(f"  split: train {len(train_subj)} subj / {tr.sum()} sims | "
          f"held-out {len(val_subj)} subj / {va.sum()} sims")

    # ── SC summary per subject (z-scored on TRAIN subjects only) ─────────────
    import data_loader_hcp as dl
    df, fc_mat, sc_mat, fc_ids, sc_ids, bm, bi = dl.load_raw_data()
    sdata = dl.load_all_subjects(sorted(subjects.tolist()), fc_mat, sc_mat,
                                 fc_ids, sc_ids, bm, bi)
    sc_feat = {int(s): sc_summary(sdata[int(s)]["sc"]) for s in subjects}
    SC = np.stack([sc_feat[int(s)] for s in subj]).astype(np.float32)   # (N, Dsc)
    mu = SC[tr].mean(0); sd = SC[tr].std(0) + 1e-6
    SC = (SC - mu) / sd
    print(f"  SC summary dim = {SC.shape[1]}  (node strength {SC.shape[1]-6} + 6 global)")

    # ── FC features: PCA (no whiten), fit on TRAIN sims only (memory-safe) ───
    fcmm = z["fc_raw"]                                            # memmap (N_full, 64620)
    n_fit = min(15000, int(tr.sum()))                            # RAM cap for PCA fit
    fit_local = np.sort(rng.choice(np.where(tr)[0], n_fit, replace=False))
    pca = PCA(n_components=min(args.fc_dim, n_fit - 1), whiten=False,
              svd_solver="randomized", random_state=0).fit(
                  np.asarray(fcmm[row_map[fit_local]], dtype=np.float32))
    FC = np.empty((N, pca.n_components_), dtype=np.float32)       # transform all in chunks
    for i in range(0, N, 5000):
        FC[i:i + 5000] = pca.transform(
            np.asarray(fcmm[row_map[i:i + 5000]], dtype=np.float32))
    print(f"  FC PCA: {fcmm.shape[1]} -> {FC.shape[1]}  (explained var {pca.explained_variance_ratio_.sum():.3f})")

    # ── sbi trainer (MAXIMAL sbi: built-in flow + FCEmbedding + SNPE_C) ───────
    prior = BoxUniform(low=-torch.ones(P), high=torch.ones(P), device=dev)

    def train_q(X, tag):
        emb = FCEmbedding(input_dim=X.shape[1], output_dim=args.emb_dim,
                          num_layers=3, num_hiddens=args.hidden)
        de = posterior_nn(model=args.flow, embedding_net=emb,
                          hidden_features=args.hidden, num_transforms=args.transforms,
                          z_score_x="independent", z_score_theta="independent")
        inf = SNPE_C(prior=prior, density_estimator=de, device=dev)
        inf.append_simulations(torch.as_tensor(theta[tr]), torch.as_tensor(X[tr]))
        est = inf.train(training_batch_size=512, stop_after_epochs=args.patience,
                        max_num_epochs=args.max_epochs, show_train_summary=False)
        post = inf.build_posterior(est)
        print(f"  [{tag}] trained.")
        return post

    def recovery(post, X_eval, theta_eval):
        """corr(theta_true, posterior-mean theta) over a subsample of held-out sims."""
        idx = rng.choice(X_eval.shape[0], min(args.eval_sims, X_eval.shape[0]), replace=False)
        Xt = torch.as_tensor(X_eval[idx], device=dev)
        means = np.empty((len(idx), P), dtype=np.float32)
        with torch.no_grad():
            for i in range(len(idx)):
                s = post.sample((args.post_samples,), x=Xt[i], show_progress_bars=False)
                means[i] = s.mean(0).cpu().numpy()
        tt = theta_eval[idx]
        overall = float(np.corrcoef(tt.ravel(), means.ravel())[0, 1])
        perdim = [float(np.corrcoef(tt[:, k], means[:, k])[0, 1]) for k in range(P)]
        return overall, perdim, idx

    x_FC = FC
    x_SCFC = np.concatenate([FC, SC], axis=1)

    print("\n  ── train q(theta | FC) ──");   postA = train_q(x_FC, "FC-only")
    print("\n  ── train q(theta | SC,FC) ──"); postB = train_q(x_SCFC, "SC+FC")

    print("\n  ── held-out-subject theta recovery (corr theta_true vs posterior-mean) ──")
    rA, pdA, _ = recovery(postA, x_FC[va], theta[va])
    rB, pdB, idxB = recovery(postB, x_SCFC[va], theta[va])
    # SC-permutation control: shuffle SC block across held-out sims, re-infer with B
    SC_va = SC[va].copy()
    perm_rows = rng.permutation(SC_va.shape[0])
    x_SCFC_perm = np.concatenate([FC[va], SC_va[perm_rows]], axis=1)
    rBp, _, _ = recovery(postB, x_SCFC_perm, theta[va])

    print("\n" + "=" * 64)
    print("  RESULT — held-out-subject theta recovery (higher = better)")
    print(f"    q(theta | FC)            overall corr = {rA:+.4f}")
    print(f"    q(theta | SC, FC)        overall corr = {rB:+.4f}   (delta {rB-rA:+.4f})")
    print(f"    q(theta | SC, FC) SC-shuf overall corr = {rBp:+.4f}   (drop {rB-rBp:+.4f})")
    print("  per-dim corr (SC+FC):")
    names = ["g_LRE","g_FFI","I_o","sigma"]; cols=["const","myelin","grad"]
    for k in range(P):
        print(f"    {names[k//3]+'_'+cols[k%3]:16} FC={pdA[k]:+.3f}  SC+FC={pdB[k]:+.3f}")
    print("=" * 64)
    print("  READ: SC+FC > FC by a margin => SC-conditioning helps identifiability.")
    print("        SC-shuf << SC+FC       => the net genuinely USES SC (not memorized).")
    print("        SC-shuf ~= SC+FC       => SC ignored / redundant given FC -> drop it.")


if __name__ == "__main__":
    main()
