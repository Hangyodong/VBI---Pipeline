"""Cortex-only principal functional gradient extraction (Margulies 2016).

Recomputes the per-subject functional gradient from the CORTICAL 360x360 FC
submatrix only (drops the 21 subcortical rows/cols BEFORE embedding), so the
cortical gradient is not contaminated by subcortex in either the diffusion
embedding or the per-subject z-score. Mirrors the cortex-only myelin basis.

Method: cosine affinity (top-10% row sparsity) -> diffusion-map embedding
-> Procrustes alignment of every subject to a group-average-FC reference
(fixes eigenvector sign/order flips across subjects).

Inputs
  HCP_FC.mat            'C' (n,2): col0 = subject id, col1 = 381x381 FC
  <SC file>             'sub_num' (S,1): target subject ids + row order

Outputs
  gradient_subjects_cortex.npy  (S, 360)  per-subject z-scored G1, sub_num order
  basis_cortex.npy              (360, 3)  [const, myelin_z, gradient_z] group basis
"""
import argparse
import numpy as np
import scipy.io as sio
from brainspace.gradient import GradientMaps

N_CORT = 360          # cortical regions (Glasser), first 360 of 381
N_COMP = 10           # embedding components (keep G1)
SPARSITY = 0.9        # keep top 10% per row
KERNEL = "cosine"
APPROACH = "dm"       # diffusion map
SEED = 0


def load_fc_index(fc_path):
    """Return dict id -> 381x381 FC from HCP_FC.mat 'C' (n,2)."""
    C = sio.loadmat(fc_path)["C"]                       # (n, 2) object
    idx = {}
    for r in range(C.shape[0]):
        sid = int(np.asarray(C[r, 0]).ravel()[0])
        fc = np.asarray(C[r, 1], dtype=np.float64)      # 381x381
        idx[sid] = fc
    return idx


def prep_cortical(fc):
    """381x381 -> 360x360 cortical, clip negatives, zero diagonal."""
    c = np.array(fc[:N_CORT, :N_CORT], dtype=np.float64)
    c[c < 0] = 0.0
    np.fill_diagonal(c, 0.0)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fc", default="HCP_Data/HCP_FC.mat")
    ap.add_argument("--sc", default="HCP_Data/HCP_CABNP381_SC_first100.mat")
    ap.add_argument("--myelin", default="HCP_Data/myelin_subjects.npy")
    ap.add_argument("--out", default="HCP_Data/gradient_subjects_cortex.npy")
    ap.add_argument("--basis_out", default="HCP_Data/basis_cortex.npy")
    args = ap.parse_args()

    sub_num = np.asarray(sio.loadmat(args.sc)["sub_num"]).ravel().astype(int)
    print(f"[load] {len(sub_num)} target subjects (sub_num order)")

    fc_idx = load_fc_index(args.fc)
    print(f"[load] HCP_FC: {len(fc_idx)} subjects available")

    cort, missing = [], []
    for sid in sub_num:
        if sid not in fc_idx:
            missing.append(sid); cort.append(None); continue
        cort.append(prep_cortical(fc_idx[sid]))
    if missing:
        print(f"[warn] {len(missing)} subjects missing FC: {missing[:8]}")
    ok = [i for i, c in enumerate(cort) if c is not None]
    mats = [cort[i] for i in ok]

    # 1) reference gradients from group-average cortical FC
    group_fc = np.mean(np.stack(mats, 0), 0)
    ref = GradientMaps(n_components=N_COMP, approach=APPROACH, kernel=KERNEL,
                       random_state=SEED).fit(group_fc, sparsity=SPARSITY)
    print(f"[ref] group gradient fitted. lambdas[:4]={np.round(ref.lambdas_[:4],3)}")

    # 2) per-subject gradients, Procrustes-aligned to reference
    gm = GradientMaps(n_components=N_COMP, approach=APPROACH, kernel=KERNEL,
                      alignment="procrustes", random_state=SEED)
    gm.fit(mats, sparsity=SPARSITY, reference=ref.gradients_)
    aligned = np.stack(gm.aligned_, 0)                  # (n_ok, 360, N_COMP)
    g1 = aligned[:, :, 0]                                # principal gradient

    # 3) per-subject z-score (match myelin_subjects.npy convention)
    g1 = (g1 - g1.mean(1, keepdims=True)) / g1.std(1, keepdims=True)

    # place into full (S,360), NaN rows for missing
    out = np.full((len(sub_num), N_CORT), np.nan)
    for k, i in enumerate(ok):
        out[i] = g1[k]
    np.save(args.out, out)
    print(f"[save] {args.out} {out.shape}")

    # 4) group basis_cortex.npy = [const, myelin_z, gradient_z]
    myel = np.load(args.myelin)[:, :N_CORT]             # (S,360)
    myel_mu = np.nanmean(myel, 0)
    grad_mu = np.nanmean(out, 0)
    zc = lambda v: (v - v.mean()) / v.std()
    basis = np.stack([np.ones(N_CORT), zc(myel_mu), zc(grad_mu)], 1)  # (360,3)
    np.save(args.basis_out, basis)
    print(f"[save] {args.basis_out} {basis.shape}")

    # 5) quick validation vs old (381-based) gradient sliced to 360
    try:
        old = np.load("HCP_Data/gradient_subjects.npy")[:, :N_CORT]
        valid = ~np.isnan(out).any(1)
        o, n = old[valid], out[valid]
        pc_old = np.corrcoef(o); pc_new = np.corrcoef(n)
        iu = np.triu_indices(valid.sum(), 1)
        sn = lambda X: X.mean(0).var() / (X.std(0) ** 2).mean()
        print("\n=== VALIDATION (cortical 360, %d subj) ===" % valid.sum())
        print("pairwise subject corr : old %.3f  ->  new %.3f"
              % (pc_old[iu].mean(), pc_new[iu].mean()))
        print("min pairwise corr     : old %.3f  ->  new %.3f"
              % (pc_old[iu].min(), pc_new[iu].min()))
        print("spatial S/N           : old %.2f  ->  new %.2f" % (sn(o), sn(n)))
        rr = np.array([np.corrcoef(n[i], n.mean(0))[0, 1] for i in range(len(n))])
        print("subj vs group-mean corr: mean %.3f  min %.3f  outliers<0.5: %d"
              % (rr.mean(), rr.min(), (rr < 0.5).sum()))
        print("corr(new group G1, old group G1) = %.3f (sign-agnostic %.3f)"
              % (np.corrcoef(n.mean(0), o.mean(0))[0, 1],
                 abs(np.corrcoef(n.mean(0), o.mean(0))[0, 1])))
    except Exception as e:
        print("[valid] skipped:", e)


if __name__ == "__main__":
    main()
