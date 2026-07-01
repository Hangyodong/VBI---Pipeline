#!/usr/bin/env python
"""
flow_overfit_ablation.py
========================

CPU-only ablation that EMPIRICALLY ATTRIBUTES the per-epoch validation-NLL
divergence (classic overfit) seen in the real SNPE-C amortized-posterior run.

Real-run symptom (GPU, N_TRAIN=70x2000=140k):
    train NLL  -0.68 -> -12.8   (keeps dropping)
    val   NLL  bottoms  -3.4 @ epoch ~21  then DIVERGES UP to +18.9 @ epoch 200
    config: x = PCA-256 WHITENED, MAF hidden=128 x 8 transforms, Identity embed,
            early stopping DISABLED.

Question: which knob CONTROLS the divergence -- feature dim, PCA whitening,
flow capacity, or early-stop -- and does early-stop + smaller features fix it?

Everything runs on CPU from cached features (NO simulation):
    output_hcp/features_stage1.npz
        theta_scaled (80000,12) float32 in [-1,1]   -> theta (used as-is)
        fc_raw       (80000,64620) float32          -> PCA features x
        subj_ids     (80000,) int64                 -> 80 subjects, 1000 sims each

Method
------
* SUBJECT-DISJOINT split (the headline design): 60 train subjects / 20 val
  subjects, chosen by subj_ids. The held-out val subjects test AMORTIZATION
  (cross-subject generalization), not just cross-sim.
* For each config: fit sklearn PCA(n_components=dim, whiten=whiten) on TRAIN x
  ONLY, transform train + val. Train sbi SNPE_C (MAF, Identity embedding,
  single round, proposal = BoxUniform(-1,1)^12) on CPU.
* HEADLINE signal = sbi's INTERNAL random-split validation_loss curve
  (inferer._summary['validation_loss']); its min-vs-final shape is exactly the
  "val diverges up" signal from the real run.
* BONUS = held-out-SUBJECT NLL at the FINAL model and at the BEST (min internal
  val) model, computed with the SAME per-sample NLL sbi uses
  (estimator.loss == -log_prob with z-scoring baked in). This tests whether the
  overfit FINAL model also generalizes worse across subjects, and whether
  picking the best (= what early stop does) recovers it.

Note on whitening vs sbi z-scoring (important, see CAVEATS in output): we keep
sbi's DEFAULT z_score_x="independent" to faithfully replicate the real pipeline
(SC_CONDITION off). Because PCA components are already orthogonal, sbi's
per-dim z-scoring rescales each component to unit variance REGARDLESS of the
PCA `whiten` flag -- so whiten is largely a no-op given sbi's z-scoring. The
ablation reports this empirically (compare C0 vs C5).

Usage
-----
    cd /scratch/home/wog3597/vbi
    python tools/flow_overfit_ablation.py            # full run (~30 min CPU)
Env overrides (optional):
    N_TRAIN=16000 N_VAL=4000 MAX_EPOCHS=120 THREADS=32 SEED=0 \
        ABLATION_CONFIGS=C0,C1   python tools/flow_overfit_ablation.py
"""
import os
import sys
import time
import warnings
import traceback
from copy import deepcopy

import numpy as np

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# knobs (env-overridable)                                                      #
# --------------------------------------------------------------------------- #
SEED        = int(os.environ.get("SEED", "0"))
N_TRAIN     = int(os.environ.get("N_TRAIN", "16000"))   # rows from 60 train subj
N_VAL       = int(os.environ.get("N_VAL", "4000"))      # rows from 20 held-out subj
MAX_EPOCHS  = int(os.environ.get("MAX_EPOCHS", "120"))
THREADS     = int(os.environ.get("THREADS", "32"))
BATCH       = int(os.environ.get("BATCH", "512"))
N_TRAIN_SUBJ = int(os.environ.get("N_TRAIN_SUBJ", "60"))
DIVERGE_THRESH = 2.0
NPZ = os.environ.get("FEATURES_NPZ", "output_hcp/features_stage1.npz")

import torch
torch.set_num_threads(THREADS)
import torch.nn as nn
from sklearn.decomposition import PCA
from sbi.inference import SNPE_C
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform


# --------------------------------------------------------------------------- #
# config table (C0..C5)                                                        #
# --------------------------------------------------------------------------- #
# each: name, dim, whiten, transforms, hidden, early_stop(bool), stop_after
CONFIGS = [
    dict(name="C0", dim=256, whiten=True,  transforms=8, hidden=128, early=False, stop_after=MAX_EPOCHS,
         note="baseline (reproduce divergence)"),
    dict(name="C1", dim=64,  whiten=False, transforms=8, hidden=128, early=False, stop_after=MAX_EPOCHS,
         note="small dim, no whiten"),
    dict(name="C2", dim=32,  whiten=False, transforms=8, hidden=128, early=False, stop_after=MAX_EPOCHS,
         note="tiny dim, no whiten"),
    dict(name="C3", dim=256, whiten=True,  transforms=4, hidden=64,  early=False, stop_after=MAX_EPOCHS,
         note="smaller flow, bad (big) features"),
    dict(name="C4", dim=64,  whiten=False, transforms=4, hidden=64,  early=True,  stop_after=15,
         note="small dim + small flow + EARLY STOP"),
    dict(name="C5", dim=256, whiten=False, transforms=8, hidden=128, early=False, stop_after=MAX_EPOCHS,
         note="isolate whiten alone vs C0"),
]

_only = os.environ.get("ABLATION_CONFIGS", "").strip()
if _only:
    keep = {s.strip() for s in _only.split(",")}
    CONFIGS = [c for c in CONFIGS if c["name"] in keep]


def log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# data: load once, build subject-disjoint split, extract train/val rows        #
# --------------------------------------------------------------------------- #
def load_data():
    log(f"[data] loading {NPZ} (fc_raw is compressed ~20GB -> decompress to RAM, ~100s)")
    t0 = time.time()
    npz = np.load(NPZ, mmap_mode="r")  # mmap is ignored for compressed npz arrays
    theta = np.asarray(npz["theta_scaled"], dtype=np.float32)   # (80000,12)
    subj  = np.asarray(npz["subj_ids"])                          # (80000,)
    log(f"[data] theta {theta.shape} range [{theta.min():.3f},{theta.max():.3f}]  "
        f"finite={bool(np.isfinite(theta).all())}")

    subjects = np.unique(subj)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(subjects))
    train_subj = set(subjects[perm[:N_TRAIN_SUBJ]].tolist())
    val_subj   = set(subjects[perm[N_TRAIN_SUBJ:]].tolist())
    log(f"[data] {len(subjects)} subjects -> {len(train_subj)} train / {len(val_subj)} held-out val")

    train_mask = np.array([s in train_subj for s in subj])
    val_mask   = ~train_mask
    train_pool = np.where(train_mask)[0]
    val_pool   = np.where(val_mask)[0]

    n_tr = min(N_TRAIN, len(train_pool))
    n_va = min(N_VAL,  len(val_pool))
    train_rows = np.sort(rng.choice(train_pool, size=n_tr, replace=False))
    val_rows   = np.sort(rng.choice(val_pool,   size=n_va, replace=False))
    if n_tr < N_TRAIN:
        log(f"[data] NOTE: train pool only {len(train_pool)} rows; using {n_tr}")
    log(f"[data] subsample: N_TRAIN={n_tr}  N_VAL={n_va}  (subject-disjoint)")

    # decompress fc_raw once, gather the rows we need, then free the 20GB blob.
    fc_full = npz["fc_raw"]                      # triggers full decompress (~100s)
    fc_train = np.ascontiguousarray(fc_full[train_rows])
    fc_val   = np.ascontiguousarray(fc_full[val_rows])
    del fc_full
    log(f"[data] fc_train {fc_train.shape} {fc_train.nbytes/1e9:.1f}GB  "
        f"fc_val {fc_val.shape}  (decompress+gather {time.time()-t0:.0f}s)")

    theta_train = theta[train_rows]
    theta_val   = theta[val_rows]
    return fc_train, fc_val, theta_train, theta_val


# --------------------------------------------------------------------------- #
# held-out-subject NLL (same per-sample NLL as sbi's validation_loss)          #
# --------------------------------------------------------------------------- #
def held_out_nll(estimator, theta_t, x_t, batch=4096):
    estimator.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, x_t.shape[0], batch):
            th = theta_t[i:i + batch]
            xx = x_t[i:i + batch]
            # estimator.loss(input, condition) == -log_prob with z-scoring baked in,
            # exactly the per-sample quantity sbi averages for validation_loss.
            losses = estimator.loss(th, xx)
            tot += float(losses.sum())
            n += losses.shape[0]
    return tot / max(n, 1)


# --------------------------------------------------------------------------- #
# one config                                                                   #
# --------------------------------------------------------------------------- #
def run_config(cfg, fc_train, fc_val, theta_train, theta_val):
    name = cfg["name"]
    log(f"\n========== {name}: dim={cfg['dim']} whiten={cfg['whiten']} "
        f"transforms={cfg['transforms']} hidden={cfg['hidden']} "
        f"early_stop={'ON(' + str(cfg['stop_after']) + ')' if cfg['early'] else 'OFF'} "
        f"-- {cfg['note']} ==========")
    t0 = time.time()

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # --- PCA fit on TRAIN only ------------------------------------------------
    pca = PCA(n_components=cfg["dim"], whiten=cfg["whiten"],
              random_state=SEED, svd_solver="randomized")
    x_train = pca.fit_transform(fc_train).astype(np.float32)
    x_val   = pca.transform(fc_val).astype(np.float32)
    evr = float(pca.explained_variance_ratio_.sum())
    log(f"[{name}] PCA dim={cfg['dim']} whiten={cfg['whiten']} EVR_sum={evr:.3f}  "
        f"x_train_std~{x_train.std():.3g}  ({time.time()-t0:.0f}s)")

    theta_tr_t = torch.tensor(theta_train, dtype=torch.float32)
    x_tr_t     = torch.tensor(x_train,     dtype=torch.float32)
    theta_va_t = torch.tensor(theta_val,   dtype=torch.float32)
    x_va_t     = torch.tensor(x_val,       dtype=torch.float32)

    # --- SNPE-C MAF (Identity embedding, sbi-default z-scoring) ---------------
    prior = BoxUniform(low=-torch.ones(12), high=torch.ones(12))
    de = posterior_nn(model="maf", embedding_net=nn.Identity(),
                      hidden_features=cfg["hidden"], num_transforms=cfg["transforms"])
    inferer = SNPE_C(prior=prior, density_estimator=de, device="cpu")
    inferer.append_simulations(theta_tr_t, x_tr_t, data_device="cpu")

    estimator = inferer.train(
        training_batch_size=BATCH,
        stop_after_epochs=cfg["stop_after"],
        max_num_epochs=MAX_EPOCHS,
        show_train_summary=False,
    )

    tr = list(inferer._summary["training_loss"])
    va = list(inferer._summary["validation_loss"])
    n_ep = len(va)
    best_idx = int(np.argmin(va))
    min_val   = float(va[best_idx])
    final_val = float(va[-1])
    final_tr  = float(tr[-1])
    gap = final_val - min_val
    diverges = gap > DIVERGE_THRESH

    # held-out-subject NLL: FINAL model (what early-stop-OFF returns)
    held_final = held_out_nll(estimator, theta_va_t, x_va_t)
    # held-out-subject NLL: BEST model (what early stop would pick / restore)
    final_sd = deepcopy(estimator.state_dict())
    try:
        estimator.load_state_dict(inferer._best_model_state_dict)
        held_best = held_out_nll(estimator, theta_va_t, x_va_t)
    finally:
        estimator.load_state_dict(final_sd)

    res = dict(
        name=name, dim=cfg["dim"], whiten=cfg["whiten"], transforms=cfg["transforms"],
        hidden=cfg["hidden"], early=cfg["early"], stop_after=cfg["stop_after"],
        evr=evr, epochs_run=n_ep, min_val=min_val, min_epoch=best_idx + 1,
        final_val=final_val, gap=gap, diverges=diverges, final_train=final_tr,
        held_final=held_final, held_best=held_best, secs=time.time() - t0,
    )
    log(f"[{name}] epochs_run={n_ep}  min_val={min_val:.3f}@{best_idx+1}  "
        f"final_val={final_val:.3f}  gap={gap:.3f}  diverges={diverges}  "
        f"final_train={final_tr:.3f}")
    log(f"[{name}] held-out-SUBJECT NLL  final={held_final:.3f}  best={held_best:.3f}  "
        f"({res['secs']:.0f}s)")
    # compact internal val curve trace (min, every ~20 ep, final)
    keep = sorted(set([0, best_idx, n_ep - 1] + list(range(0, n_ep, max(1, n_ep // 6)))))
    trace = "  ".join(f"e{i+1}:{va[i]:.2f}" for i in keep)
    log(f"[{name}] val curve: {trace}")
    return res


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    log("=" * 78)
    log("FLOW OVERFIT ABLATION  (CPU, cached features, no simulation)")
    log(f"SEED={SEED} THREADS={THREADS} N_TRAIN={N_TRAIN} N_VAL={N_VAL} "
        f"MAX_EPOCHS={MAX_EPOCHS} BATCH={BATCH}")
    log(f"sbi={__import__('sbi').__version__} torch={torch.__version__}")
    log("=" * 78)

    fc_train, fc_val, theta_train, theta_val = load_data()

    results = []
    for cfg in CONFIGS:
        try:
            results.append(run_config(cfg, fc_train, fc_val, theta_train, theta_val))
        except Exception as e:
            log(f"[{cfg['name']}] FAILED: {e}")
            traceback.print_exc()
            results.append(dict(name=cfg["name"], failed=str(e)))

    # ---- summary table -------------------------------------------------------
    log("\n" + "=" * 100)
    log("SUMMARY TABLE  (headline = sbi internal random-split validation_loss)")
    log("=" * 100)
    hdr = (f"{'cfg':<4} {'dim':>4} {'whit':>5} {'tf':>3} {'hid':>4} {'early':>6} "
           f"{'EVR':>5} {'ep':>4} {'min_val':>8} {'@ep':>4} {'final_val':>10} "
           f"{'gap':>7} {'div?':>5} {'fin_trn':>8} {'held_fin':>9} {'held_best':>9}")
    log(hdr)
    log("-" * len(hdr))
    for r in results:
        if r.get("failed"):
            log(f"{r['name']:<4} FAILED: {r['failed']}")
            continue
        early = (f"ON/{r['stop_after']}" if r["early"] else "OFF")
        log(f"{r['name']:<4} {r['dim']:>4} {str(r['whiten']):>5} {r['transforms']:>3} "
            f"{r['hidden']:>4} {early:>6} {r['evr']:>5.2f} {r['epochs_run']:>4} "
            f"{r['min_val']:>8.3f} {r['min_epoch']:>4} {r['final_val']:>10.3f} "
            f"{r['gap']:>7.3f} {str(r['diverges']):>5} {r['final_train']:>8.3f} "
            f"{r['held_final']:>9.3f} {r['held_best']:>9.3f}")

    # ---- attribution verdict -------------------------------------------------
    by = {r["name"]: r for r in results if not r.get("failed")}

    def g(n):  # internal-val divergence gap
        return by[n]["gap"] if n in by else float("nan")

    def d(n):
        return by[n]["diverges"] if n in by else None

    log("\n" + "=" * 100)
    log("ATTRIBUTION  (divergence_gap = final_internal_val - min_internal_val; diverges if > 2.0)")
    log("=" * 100)
    log(f"  feature dim + whiten  : C0(dim256,whit) gap={g('C0'):.2f} div={d('C0')}  vs  "
        f"C1(dim64) gap={g('C1'):.2f} div={d('C1')}  vs  C2(dim32) gap={g('C2'):.2f} div={d('C2')}")
    log(f"  whiten ALONE          : C0(dim256,whit=T) gap={g('C0'):.2f}  vs  "
        f"C5(dim256,whit=F) gap={g('C5'):.2f}   -> delta={g('C0')-g('C5'):+.2f}")
    log(f"  flow capacity         : C0(tf8,h128) gap={g('C0'):.2f}  vs  "
        f"C3(tf4,h64,same dim256) gap={g('C3'):.2f}   -> delta={g('C0')-g('C3'):+.2f}")
    log(f"  early stop            : C1(dim64,noES) gap={g('C1'):.2f} div={d('C1')}  vs  "
        f"C4(dim64,ES15) gap={g('C4'):.2f} div={d('C4')}")
    log("=" * 100)
    log("Interpretation rule: the knob whose change collapses `gap` below 2.0 (and "
        "lowers held-out NLL) is the controller. Whitening is partly absorbed by "
        "sbi's default z_score_x='independent' (kept ON to match the real pipeline).")
    return results


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
