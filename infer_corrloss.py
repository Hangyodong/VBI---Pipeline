#!/usr/bin/env python
"""infer_corrloss.py — train the inference net with a CORRELATION loss instead of
the SNPE density loss. Objective change requested by the user:

  OLD (SNPE):  loss = -log q(theta_true | FC)        # recover the sim-generating theta
  NEW (here):  loss = 1 - corr(Sim(theta_hat), FC)   # output theta that REPRODUCES FC

cuBNM is non-differentiable, so we backprop the corr term through a differentiable
SURROGATE of the simulator:

  surrogate_psi : [theta, SC_feat] -> FC_pca      (regression, trained on the cache)
  inference_phi : [FC_pca, SC_feat] -> theta
  loss(phi)     = 1 - corr( recon(psi(phi(x))) , recon(FC_pca_in) ) + lambda*mse

Both nets are plain MLPs trained on the CACHED (theta, simFC) pairs — zero new
simulation. We compare the corr-loss inference vs an MSE-on-theta inference (=
the recovery objective) on HELD-OUT SUBJECTS (surrogate-corr proxy; the real
cuBNM corr is a later GPU step).

Usage:  python infer_corrloss.py            (GPU if avail)
        python infer_corrloss.py --smoke    (CPU subsample)
"""
import argparse
import os
import numpy as np

os.environ.setdefault("VBI_SC_SCALE", "maxnorm")
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


def sc_summary(sc):
    sc = np.asarray(sc, dtype=np.float64); np.fill_diagonal(sc, 0.0)
    strength = sc.sum(1); pos = sc[sc > 0]
    glob = np.array([np.log1p(sc.sum()), float((sc > 0).mean()),
                     strength.mean(), strength.std(),
                     float(np.median(pos)) if pos.size else 0.0,
                     float(pos.mean()) if pos.size else 0.0])
    return np.concatenate([strength, glob])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fc-dim", type=int, default=256)
    ap.add_argument("--surr-epochs", type=int, default=60)
    ap.add_argument("--inf-epochs", type=int, default=80)
    ap.add_argument("--lam-rmse", type=float, default=0.3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.fc_dim, args.surr_epochs, args.inf_epochs = 32, 15, 20

    import torch, torch.nn as nn
    from sklearn.decomposition import PCA
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.RandomState(42); torch.manual_seed(42)
    print(f"  device={dev}  fc_dim={args.fc_dim}  lam_rmse={args.lam_rmse}")

    # ── cache ────────────────────────────────────────────────────────────────
    z = np.load("output_hcp/features_stage1.npz", mmap_mode="r")
    theta = np.asarray(z["theta_scaled"], dtype=np.float32)        # (N,12) in [-1,1]
    subj = np.asarray(z["subj_ids"]).astype(int); fcmm = z["fc_raw"]
    N, P = theta.shape; subjects = np.unique(subj)
    row_map = np.arange(N)
    if args.smoke:
        keep = set(rng.choice(subjects, 14, replace=False).tolist())
        sel = np.array([s in keep for s in subj]); row_map = np.where(sel)[0]
        theta = theta[row_map]; subj = subj[row_map]; N = len(row_map); subjects = np.unique(subj)
    print(f"  {N} sims, {len(subjects)} subjects")

    perm = rng.permutation(subjects); nval = max(1, int(len(subjects) * args.val_frac))
    val_s = set(perm[:nval].tolist())
    tr = np.array([s not in val_s for s in subj]); va = ~tr
    print(f"  train {int(tr.sum())} sims / held-out {int(va.sum())} sims ({nval} subj)")

    # ── SC features (z on train subj) ─────────────────────────────────────────
    import data_loader_hcp as dl
    d_, fc_mat, sc_mat, fc_ids, sc_ids, bm, bi = dl.load_raw_data()
    sd = dl.load_all_subjects(sorted(subjects.tolist()), fc_mat, sc_mat, fc_ids, sc_ids, bm, bi)
    scf = {int(s): sc_summary(sd[int(s)]["sc"]) for s in subjects}
    SC = np.stack([scf[int(s)] for s in subj]).astype(np.float32)
    mu, sg = SC[tr].mean(0), SC[tr].std(0) + 1e-6; SC = (SC - mu) / sg

    # ── FC PCA (fit train, no whiten) ─────────────────────────────────────────
    nfit = min(15000, int(tr.sum()))
    fidx = np.sort(rng.choice(np.where(tr)[0], nfit, replace=False))
    pca = PCA(n_components=min(args.fc_dim, nfit - 1), whiten=False,
              svd_solver="randomized", random_state=0).fit(np.asarray(fcmm[row_map[fidx]], np.float32))
    FC = np.empty((N, pca.n_components_), np.float32)
    for i in range(0, N, 5000):
        FC[i:i+5000] = pca.transform(np.asarray(fcmm[row_map[i:i+5000]], np.float32))
    print(f"  FC PCA -> {FC.shape[1]} (explained {pca.explained_variance_ratio_.sum():.3f})")
    comps = torch.tensor(pca.components_, dtype=torch.float32, device=dev)   # (k, 64620)
    pmean = torch.tensor(pca.mean_, dtype=torch.float32, device=dev)         # (64620,)

    # tensors
    T = torch.tensor(theta, device=dev); Xsc = torch.tensor(SC, device=dev); Xfc = torch.tensor(FC, device=dev)
    tr_t = torch.tensor(np.where(tr)[0], device=dev); va_t = torch.tensor(np.where(va)[0], device=dev)

    def recon(coords):                                  # (B,k) -> (B,64620)
        return coords @ comps + pmean

    def pearson(a, b):                                  # (B,D),(B,D) -> (B,)
        a = a - a.mean(1, keepdim=True); b = b - b.mean(1, keepdim=True)
        return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-8)

    def mlp(din, dout, h=(512, 512)):
        L = []; d = din
        for w in h:
            L += [nn.Linear(d, w), nn.ReLU()]; d = w
        L += [nn.Linear(d, dout)]; return nn.Sequential(*L).to(dev)

    def train(net, step_fn, idx, epochs, lr=1e-3, tag=""):
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        n = len(idx)
        for ep in range(epochs):
            perm = idx[torch.randperm(n, device=dev)]
            tot = 0.0
            for i in range(0, n, args.batch):
                b = perm[i:i+args.batch]; opt.zero_grad()
                loss = step_fn(b); loss.backward(); opt.step(); tot += loss.item() * len(b)
            if ep % max(1, epochs // 5) == 0 or ep == epochs - 1:
                print(f"    [{tag}] epoch {ep+1}/{epochs}  loss={tot/n:.4f}", flush=True)

    # ── 1. surrogate  [theta, SC] -> FC_pca  (MSE) ────────────────────────────
    surr = mlp(P + SC.shape[1], FC.shape[1])
    def surr_step(b):
        pred = surr(torch.cat([T[b], Xsc[b]], 1)); return ((pred - Xfc[b]) ** 2).mean()
    print("\n  ── train surrogate (theta,SC -> FC_pca) ──")
    train(surr, surr_step, tr_t, args.surr_epochs, tag="surrogate")
    for p in surr.parameters(): p.requires_grad_(False)
    with torch.no_grad():
        vp = surr(torch.cat([T[va_t], Xsc[va_t]], 1))
        sr_r2 = 1 - ((vp - Xfc[va_t]) ** 2).sum() / ((Xfc[va_t] - Xfc[va_t].mean(0)) ** 2).sum()
    print(f"    surrogate held-out R^2 (FC_pca) = {sr_r2.item():.3f}")

    # ── 2. inference nets: corr-loss vs theta-MSE ─────────────────────────────
    def make_inf(): return mlp(FC.shape[1] + SC.shape[1], P, h=(256, 128))

    inf_corr = make_inf()
    def corr_step(b):
        th = inf_corr(torch.cat([Xfc[b], Xsc[b]], 1))          # theta_hat
        fc_hat = recon(surr(torch.cat([th, Xsc[b]], 1)))       # surrogate FC
        fc_tgt = recon(Xfc[b])                                  # input FC (reconstructed)
        c = pearson(fc_hat, fc_tgt)
        rmse = (fc_hat - fc_tgt).pow(2).mean(1).sqrt().mean()
        return (1 - c).mean() + args.lam_rmse * rmse
    print("\n  ── train inference: CORR loss ──")
    train(inf_corr, corr_step, tr_t, args.inf_epochs, tag="corr-loss")

    inf_mse = make_inf()
    def mse_step(b):
        th = inf_mse(torch.cat([Xfc[b], Xsc[b]], 1)); return ((th - T[b]) ** 2).mean()
    print("\n  ── train inference: theta-MSE (recovery baseline) ──")
    train(inf_mse, mse_step, tr_t, args.inf_epochs, tag="theta-mse")

    # ── 3. eval held-out: surrogate-corr(Sim(theta_hat), FC) ──────────────────
    def eval_corr(net):
        with torch.no_grad():
            th = net(torch.cat([Xfc[va_t], Xsc[va_t]], 1))
            fc_hat = recon(surr(torch.cat([th, Xsc[va_t]], 1)))
            fc_tgt = recon(Xfc[va_t])
            return pearson(fc_hat, fc_tgt).mean().item()
    def eval_theta(net):
        with torch.no_grad():
            th = net(torch.cat([Xfc[va_t], Xsc[va_t]], 1))
            return pearson(th, T[va_t]).mean().item()  # rough theta recovery

    cc, cm = eval_corr(inf_corr), eval_corr(inf_mse)
    tc, tm = eval_theta(inf_corr), eval_theta(inf_mse)
    print("\n" + "=" * 60)
    print("  HELD-OUT (surrogate-corr = how well Sim(theta_hat) reproduces FC):")
    print(f"    inference CORR-loss  : FC-corr = {cc:+.4f}   theta-recovery = {tc:+.4f}")
    print(f"    inference theta-MSE  : FC-corr = {cm:+.4f}   theta-recovery = {tm:+.4f}")
    print(f"    delta (corr-loss - mse) FC-corr = {cc-cm:+.4f}")
    print("=" * 60)
    print("  READ: CORR-loss FC-corr > MSE FC-corr => the new objective makes the")
    print("        inference output BETTER-FITTING theta (the goal). theta-recovery")
    print("        may be LOWER for corr-loss (it doesn't care about exact theta,")
    print("        only FC match) — that's expected and fine.")
    print("  NOTE: surrogate-corr is a proxy; real cuBNM corr = pipeline/GPU step.")


if __name__ == "__main__":
    main()
