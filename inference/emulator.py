"""Method A — differentiable FC emulator + amortized corr/rmse fitter.

The cuBNM simulator is non-differentiable, so FC-correlation / FC-RMSE cannot be
back-propagated into an inference network directly. Method A gets around that in
two stages:

  1. FCEmulator  E: theta(12) -> FC feature (PCA-256 whitened)  [differentiable
     surrogate of the whole cuBNM forward + PCA pipeline], trained by MSE on the
     existing (theta_scaled, fc_raw) simulation pairs.

  2. AmortizedFitNet  Q: empFC feature(256) -> theta(12), trained with
        L = (1 - corr(FC_hat, empFC)) + lambda * rmse(FC_hat, empFC)
     where FC_hat = decode(E(Q(empFC))). E is frozen; the gradient of the *FC*
     objective flows Q <- E <- (linear PCA decode), so Q learns to emit theta
     whose emulated FC has high correlation / low RMSE with the target FC.

corr/rmse are evaluated in *FC space* (via the differentiable linear PCA decode),
i.e. the metric the user actually cares about — not the whitened feature space.

CPU self-test: ``python inference/emulator.py`` (tiny synthetic, no GPU/data).
"""
import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# Differentiable linear PCA projector (feature <-> FC), from an sklearn PCA
# ----------------------------------------------------------------------------
class PCAProjector(nn.Module):
    """Torch wrapper around a fitted whitened PCA. encode/decode are linear and
    differentiable so FC-space losses flow through the 256-d feature space."""

    def __init__(self, components, mean, explained_variance, whiten=True):
        super().__init__()
        Vt = torch.as_tensor(components, dtype=torch.float32)        # (k, D)
        mu = torch.as_tensor(mean, dtype=torch.float32)              # (D,)
        ev = torch.as_tensor(explained_variance, dtype=torch.float32)  # (k,)
        sev = torch.sqrt(torch.clamp(ev, min=1e-12)) if whiten else torch.ones_like(ev)
        self.register_buffer("Vt", Vt)
        self.register_buffer("mean", mu)
        self.register_buffer("sqrt_ev", sev)
        self.k, self.D = Vt.shape

    @classmethod
    def from_sklearn(cls, pca, whiten=True):
        return cls(pca.components_, pca.mean_, pca.explained_variance_, whiten)

    @classmethod
    def fit(cls, fc_raw, n_components=256, whiten=True, seed=0):
        from sklearn.decomposition import PCA
        p = PCA(n_components=n_components, whiten=whiten,
                svd_solver="auto", random_state=seed).fit(fc_raw)
        return cls.from_sklearn(p, whiten), p

    def encode(self, fc):                       # (B,D) -> (B,k) whitened feature
        return ((fc - self.mean) @ self.Vt.t()) / self.sqrt_ev

    def decode(self, feat):                     # (B,k) whitened -> (B,D) FC
        return (feat * self.sqrt_ev) @ self.Vt + self.mean


# ----------------------------------------------------------------------------
# Networks
# ----------------------------------------------------------------------------
def _mlp(sizes, act=nn.GELU, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers += [nn.LayerNorm(sizes[i + 1]), act()]
    if out_act is not None:
        layers.append(out_act())
    return nn.Sequential(*layers)


class FCEmulator(nn.Module):
    """theta_scaled (theta_dim) -> FC feature (feat_dim). Surrogate of cuBNM+PCA."""

    def __init__(self, theta_dim=12, feat_dim=256, hidden=(256, 512, 512)):
        super().__init__()
        self.net = _mlp([theta_dim, *hidden, feat_dim])

    def forward(self, theta):
        return self.net(theta)


class AmortizedFitNet(nn.Module):
    """empFC feature (feat_dim) -> theta_scaled (theta_dim), bounded to [-1,1]
    (the scaled BoxUniform prior support, so E stays in-distribution)."""

    def __init__(self, feat_dim=256, theta_dim=12, hidden=(512, 512, 256)):
        super().__init__()
        self.net = _mlp([feat_dim, *hidden, theta_dim], out_act=nn.Tanh)

    def forward(self, feat):
        return self.net(feat)


# ----------------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------------
def batched_pearson(a, b, eps=1e-8):
    """Row-wise Pearson r for (B,D) tensors -> (B,)."""
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    num = (a * b).sum(dim=1)
    den = torch.sqrt((a * a).sum(dim=1) * (b * b).sum(dim=1) + eps)
    return num / (den + eps)


def batched_rmse(a, b):
    return torch.sqrt(((a - b) ** 2).mean(dim=1) + 1e-12)


def fc_corr_rmse_loss(fc_pred, fc_tgt, lam=0.5):
    """L = mean(1 - corr) + lam * mean(rmse). Returns (loss, corr_mean, rmse_mean)."""
    corr = batched_pearson(fc_pred, fc_tgt)
    rmse = batched_rmse(fc_pred, fc_tgt)
    loss = (1.0 - corr).mean() + lam * rmse.mean()
    return loss, corr.mean(), rmse.mean()


# ----------------------------------------------------------------------------
# Training loops
# ----------------------------------------------------------------------------
def train_emulator(emu, theta, feat, *, epochs=200, batch=1024, lr=1e-3,
                   val_frac=0.1, device="cpu", verbose=True):
    """MSE( E(theta), feat ). theta (N,theta_dim), feat (N,feat_dim) tensors."""
    emu = emu.to(device)
    theta = theta.to(device); feat = feat.to(device)
    n = theta.shape[0]
    n_val = max(1, int(n * val_frac))
    perm = torch.randperm(n)
    vi, ti = perm[:n_val], perm[n_val:]
    opt = torch.optim.Adam(emu.parameters(), lr=lr)
    best = float("inf"); best_state = None; hist = []
    for ep in range(epochs):
        emu.train()
        for idx in torch.randperm(ti.numel()).split(batch):
            b = ti[idx]
            opt.zero_grad()
            loss = ((emu(theta[b]) - feat[b]) ** 2).mean()
            loss.backward(); opt.step()
        emu.eval()
        with torch.no_grad():
            vloss = ((emu(theta[vi]) - feat[vi]) ** 2).mean().item()
        hist.append(vloss)
        if vloss < best:
            best = vloss; best_state = {k: v.detach().clone()
                                        for k, v in emu.state_dict().items()}
        if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
            print(f"  [emu] ep {ep:4d}  val_mse {vloss:.5f}  best {best:.5f}")
    if best_state:
        emu.load_state_dict(best_state)
    return emu, hist


def train_amortized_fit(net, emu, pca, emp_feat, *, lam=0.5, epochs=300,
                        batch=64, lr=1e-3, val_frac=0.15, device="cpu",
                        verbose=True):
    """Train Q with FC corr/rmse loss through frozen E and linear PCA decode.

    emp_feat (M, feat_dim) = PCA features of the empirical FCs to fit."""
    net = net.to(device); emu = emu.to(device).eval(); pca = pca.to(device)
    for p in emu.parameters():
        p.requires_grad_(False)
    emp_feat = emp_feat.to(device)
    m = emp_feat.shape[0]
    n_val = max(1, int(m * val_frac))
    perm = torch.randperm(m)
    vi, ti = perm[:n_val], perm[n_val:]
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    best = float("inf"); best_state = None; hist = []
    fc_emp_all = pca.decode(emp_feat)                     # target FCs (M,D)
    for ep in range(epochs):
        net.train()
        for idx in torch.randperm(ti.numel()).split(batch):
            b = ti[idx]
            opt.zero_grad()
            theta = net(emp_feat[b])                      # (B,theta_dim)
            fc_hat = pca.decode(emu(theta))               # emulated FC (B,D)
            loss, _, _ = fc_corr_rmse_loss(fc_hat, fc_emp_all[b], lam)
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            theta_v = net(emp_feat[vi])
            fc_hat_v = pca.decode(emu(theta_v))
            vloss, vcorr, vrmse = fc_corr_rmse_loss(fc_hat_v, fc_emp_all[vi], lam)
            vloss = vloss.item()
        hist.append({"loss": vloss, "corr": float(vcorr), "rmse": float(vrmse)})
        if vloss < best:
            best = vloss; best_state = {k: v.detach().clone()
                                        for k, v in net.state_dict().items()}
        if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
            print(f"  [fit] ep {ep:4d}  val_loss {vloss:.4f}  "
                  f"corr {float(vcorr):.4f}  rmse {float(vrmse):.4f}")
    if best_state:
        net.load_state_dict(best_state)
    return net, hist


# ----------------------------------------------------------------------------
# CPU self-test (synthetic; verifies shapes, gradient flow, loss decreases)
# ----------------------------------------------------------------------------
def _selftest():
    torch.manual_seed(0); np.random.seed(0)
    N, D, K, TH = 2000, 400, 32, 12
    # synthetic ground-truth linear-ish forward: theta -> FC
    W = np.random.randn(TH, D).astype(np.float32) * 0.3
    theta = np.random.uniform(-1, 1, (N, TH)).astype(np.float32)
    fc = np.tanh(theta @ W) + 0.02 * np.random.randn(N, D).astype(np.float32)

    pca, skp = PCAProjector.fit(fc, n_components=K, whiten=True)
    feat = skp.transform(fc).astype(np.float32)          # (N,K)

    emu = FCEmulator(theta_dim=TH, feat_dim=K, hidden=(128, 128))
    emu, eh = train_emulator(emu, torch.tensor(theta), torch.tensor(feat),
                             epochs=40, batch=256, verbose=True)
    print(f"[selftest] emulator val_mse {eh[0]:.4f} -> {min(eh):.4f}")

    # amortized fit on the same synthetic FCs (as stand-in for empirical)
    emp_feat = torch.tensor(feat[:300])
    net = AmortizedFitNet(feat_dim=K, theta_dim=TH, hidden=(128, 128))
    net, fh = train_amortized_fit(net, emu, pca, emp_feat, lam=0.5,
                                  epochs=60, batch=32, verbose=True)
    c0, c1 = fh[0]["corr"], fh[-1]["corr"]
    print(f"[selftest] fit corr {c0:.4f} -> {c1:.4f}  "
          f"rmse {fh[0]['rmse']:.4f} -> {fh[-1]['rmse']:.4f}")
    assert min(eh) < eh[0], "emulator did not learn"
    assert c1 > c0, "amortized fit corr did not improve"
    print("[selftest] PASS")


if __name__ == "__main__":
    _selftest()
