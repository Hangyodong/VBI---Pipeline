"""Stage 1 end-to-end driver (RegionTransformer + attention selection).

Public API
----------
- run_stage1_snpe(train_subjects, subject_data, ...) -> dict
- run_phase1(train_subjects, subject_data, ...)
- run_phase2(phase1_artifacts, fc_obs, ...)
- run_phase3(phase1, fc_selected_indices, sc_matrix=None, ...)

Phases
------
Phase 1: simulate with global scalars P, Q, g_e, g_i, c_ei, train SNPE-C
on raw upper-tri FC (6555-dim).

Phase 2: attention × gradient feature selection on top of the Phase-1
posterior + embedding network → ``fc_selected_indices`` (length k).

Phase 3 (a.k.a. Phase 4): re-use the Phase-1 simulations sliced to the
selected indices (``fc_raw[:, indices]``) and train a second SNPE-C using a
fresh RegionTransformer on the reduced (k-dim) FC. No re-simulation.

Returned dict keys
------------------
- posterior_1, posterior_2, embedding_1, embedding_2
- theta_scaled, theta_raw, fc_raw, fcd_raw
- param_scaler, feature_pipeline, prior_scaled
- fc_selected_indices, pca_diagnostic
"""
import os

import numpy as np

import config
from inference.snpe import (
    step4_fit_feature_scalers,
    step5_fit_feature_pipeline,
    step6_pca_diagnostic,
    step7_fit_param_scaler,
    train_snpe,
)
from inference.training_data import (
    load_extracted_features,
    save_extracted_features,
    step2_simulate_train,
    step3_summary_features,
)


# ---------------------------------------------------------------------------
# Feature cache helpers
# ---------------------------------------------------------------------------

def _load_cached_features(tag, n_train_subjects, n_sim, expected_fc_dim=None,
                          verbose=True):
    """Try to load a cached features_{tag}.npz. Returns dict or None.

    Cache hit requires:
      - file exists
      - theta_scaled.shape[0] == n_train_subjects * n_sim
      - fc_raw.shape[1] == expected_fc_dim (if provided)
    """
    path = os.path.join(config.OUTPUT_DIR, f"features_{tag}.npz")
    if not os.path.exists(path):
        return None
    try:
        loaded = load_extracted_features(
            save_dir=config.OUTPUT_DIR, tag=tag,
        )
    except Exception as e:
        if verbose:
            print(f"  [cache] failed to load {path}: {e} — re-simulating")
        return None
    theta_s = loaded["theta_scaled"]
    fc_raw = loaded["fc_raw"]
    n_expected = n_train_subjects * n_sim
    if theta_s.shape[0] != n_expected:
        if verbose:
            print(
                f"  [cache] {path}: sample count {theta_s.shape[0]} != "
                f"expected {n_expected} (n_subjects={n_train_subjects} x "
                f"n_sim={n_sim}) — re-simulating"
            )
        return None
    if expected_fc_dim is not None and fc_raw.shape[1] != expected_fc_dim:
        if verbose:
            print(
                f"  [cache] {path}: fc_dim {fc_raw.shape[1]} != expected "
                f"{expected_fc_dim} — re-simulating"
            )
        return None
    if verbose:
        print(f"  [cache] loaded {path}  theta={theta_s.shape}  fc={fc_raw.shape}")
    return loaded


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_sc_matrix_for_embedding(subject_data):
    """Pick a representative SC matrix for use as the embedding's positional encoding."""
    for sid in subject_data:
        sc = subject_data[sid].get("sc")
        if sc is not None:
            return np.asarray(sc, dtype=np.float32)
    return None


def _make_embedding(input_dim, sc_matrix, triu_indices=None, use_mask=False):
    from inference.embedding import make_embedding_net
    return make_embedding_net(
        "region_transformer",
        input_dim=input_dim,
        out_dim=config.EMBED_DIM,
        sc_matrix=sc_matrix,
        triu_indices=triu_indices,
        use_mask=use_mask,
        n_regions=config.N_REGIONS,
        d_model=config.REGION_TRANSFORMER_D_MODEL,
        n_heads=config.REGION_TRANSFORMER_HEADS,
        n_layers=config.REGION_TRANSFORMER_LAYERS,
    )


# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------

def run_phase1(train_subjects, subject_data, n_sim=None,
               apply_bw=True, verbose=True):
    """Phase 1: simulation + first SNPE-C round on raw 6555-dim FC."""
    n_sim = n_sim or config.N_SIM
    if verbose:
        print("\n" + "=" * 65)
        print("  Stage 1 / Phase 1: simulation + SNPE-C #1")
        print(
            f"  Params: {len(config.STAGE1_PARAMS)} global scalars "
            f"({', '.join(config.STAGE1_PARAMS)})"
        )
        print("=" * 65)

    sc_for_pe = _resolve_sc_matrix_for_embedding(subject_data)

    param_scaler, prior_scaled = step7_fit_param_scaler(verbose=verbose)

    cached = _load_cached_features(
        tag="stage1",
        n_train_subjects=len(train_subjects),
        n_sim=n_sim,
        verbose=verbose,
    )
    if cached is not None:
        theta_s = cached["theta_scaled"]
        theta_r = cached["theta_raw"]
        fc_raw = cached["fc_raw"]
        fcd_raw = cached["fcd_raw"]
    else:
        theta_s, theta_r, fc_raw, fcd_raw = step2_simulate_train(
            train_subjects, subject_data, prior_scaled, param_scaler,
            n_sim=n_sim, apply_bw=apply_bw, verbose=verbose,
        )
        save_extracted_features(
            theta_s, theta_r, fc_raw, fcd_raw,
            param_names=config.STAGE1_PARAMS,
            save_dir=config.OUTPUT_DIR, tag="stage1", verbose=verbose,
        )
    step3_summary_features(fc_raw, fcd_raw, verbose=verbose)

    step4_fit_feature_scalers(fc_raw, fcd_raw, verbose=verbose)
    pipeline, x_input = step5_fit_feature_pipeline(
        fc_raw, fcd_raw, verbose=verbose,
    )
    pca_diag = step6_pca_diagnostic(
        pipeline, fc_raw, fcd_raw, verbose=verbose,
    )

    posterior, embedding_net = train_snpe(
        theta_s, x_input, prior_scaled,
        embedding_net=None,
        use_embedding=False,
        sc_matrix=None,
        verbose=verbose,
    )

    return {
        "posterior": posterior,
        "embedding_net": embedding_net,
        "theta_scaled": theta_s,
        "theta_raw": theta_r,
        "fc_raw": fc_raw,
        "fcd_raw": fcd_raw,
        "x_input": x_input,
        "param_scaler": param_scaler,
        "feature_pipeline": pipeline,
        "prior_scaled": prior_scaled,
        "sc_matrix": sc_for_pe,
        "pca_diagnostic": pca_diag,
    }


# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------

def run_phase2(phase1, fc_obs, k=None, n_samples=None, save_path=None,
               verbose=True):
    """Phase 2: attention × gradient feature selection."""
    from inference.attention_selection import AttentionSelector
    from inference.feature_pipeline import save_selected_indices

    k = int(k or config.SELECTION_K)
    n_samples = int(n_samples or 200)

    selector = AttentionSelector(
        posterior=phase1["posterior"],
        embedding_net=phase1["embedding_net"],
        param_names=config.STAGE1_PARAMS,
        n_regions=config.N_REGIONS,
    )
    if verbose:
        print("\n" + "=" * 65)
        print(
            f"  Stage 1 / Phase 2: attention × gradient selection (k={k})"
        )
        print("=" * 65)
    importance = selector.get_importance(fc_obs, n_samples=n_samples)
    indices = selector.select_top_k(importance, k=k)
    if verbose:
        print(
            f"  Selected {len(indices)} / {selector.fc_dim} FC entries"
        )
    save_path = save_path or os.path.join(
        config.OUTPUT_DIR, "fc_selected_indices.npy",
    )
    save_selected_indices(indices, save_path)
    return {
        "fc_selected_indices": indices,
        "importance": importance,
        "save_path": save_path,
    }


# ---------------------------------------------------------------------------
# Phase 3
# ---------------------------------------------------------------------------

def run_phase3(phase1, fc_selected_indices,
               sc_matrix=None, verbose=True):
    """Phase 4: SNPE-C re-train on selected FC entries.

    Re-uses the Phase-1 ``(theta_scaled, fc_raw)`` pairs — no re-simulation.
    Slices ``fc_raw[:, indices]`` down to the k selected edges, then trains a
    second SNPE-C round with a fresh RegionTransformer (value + mask).
    """
    indices = np.asarray(fc_selected_indices, dtype=np.int64)

    # Re-use Phase 1 data (no re-simulation).
    theta_s = phase1["theta_scaled"]
    theta_r = phase1["theta_raw"]
    fc_raw_all = phase1["fc_raw"]          # (N_total, 6555)
    prior_scaled = phase1["prior_scaled"]
    param_scaler = phase1["param_scaler"]

    if verbose:
        print("\n" + "=" * 65)
        print(
            f"  Phase 4: SNPE-C #2 on {len(indices)} selected FC entries"
        )
        print(f"  Phase 1 데이터 재사용 — 시뮬 재실행 없음")
        print(f"  theta: {theta_s.shape}  fc_raw: {fc_raw_all.shape}")
        print("=" * 65)

    # Slice the cached FC to the selected upper-tri edges.
    fc_selected = fc_raw_all[:, indices]   # (N_total, k)

    if verbose:
        print(
            f"  FC 슬라이싱: {fc_raw_all.shape} → {fc_selected.shape}"
        )

    if sc_matrix is None:
        sc_matrix = phase1.get("sc_matrix")

    # Resolve canonical (row, col) for the selected upper-tri entries.
    rr, cc = np.triu_indices(config.N_REGIONS, k=1)
    sel_rows = rr[indices]
    sel_cols = cc[indices]

    # Phase 4 (2nd SBI) runs on a sparse, selected FC subset, so we feed
    # the value+mask channels to distinguish absent edges from true zeros.
    embedding_net = _make_embedding(
        input_dim=int(indices.shape[0]),
        sc_matrix=sc_matrix,
        triu_indices=(sel_rows, sel_cols),
        use_mask=True,
    )
    posterior, embedding_net = train_snpe(
        theta_s, fc_selected, prior_scaled,
        embedding_net=embedding_net,
        use_embedding=True,
        sc_matrix=sc_matrix,
        verbose=verbose,
    )
    return {
        "posterior":           posterior,
        "embedding_net":       embedding_net,
        "theta_scaled":        theta_s,
        "theta_raw":           theta_r,
        "fc_raw":              fc_selected,   # sliced (k)
        "fc_raw_full":         fc_raw_all,    # original (6555)
        "fc_selected_indices": indices,
        "prior_scaled":        prior_scaled,
        "param_scaler":        param_scaler,
        "sc_matrix":           sc_matrix,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_stage1_snpe(train_subjects, subject_data, n_sim=None,
                    apply_bw=True, verbose=True, fc_obs=None,
                    run_phase2_phase3=True, selection_k=None):
    """End-to-end Stage 1 inference (Phase 1, optional Phase 2 + 3)."""
    p1 = run_phase1(
        train_subjects, subject_data,
        n_sim=n_sim, apply_bw=apply_bw, verbose=verbose,
    )
    out = {
        "posterior_1": p1["posterior"],
        "embedding_1": p1["embedding_net"],
        "theta_scaled": p1["theta_scaled"],
        "theta_raw": p1["theta_raw"],
        "fc_raw": p1["fc_raw"],
        "fcd_raw": p1["fcd_raw"],
        "x_input": p1["x_input"],
        "param_scaler": p1["param_scaler"],
        "feature_pipeline": p1["feature_pipeline"],
        "prior_scaled": p1["prior_scaled"],
        "sc_matrix": p1["sc_matrix"],
        "pca_diagnostic": p1["pca_diagnostic"],
        # Back-compat aliases:
        "posterior": p1["posterior"],
        "embedding_net": p1["embedding_net"],
    }

    if not run_phase2_phase3:
        return out

    if fc_obs is None:
        fc_obs = p1["fc_raw"][0]

    p2 = run_phase2(
        p1, fc_obs, k=selection_k, verbose=verbose,
    )
    out["fc_selected_indices"] = p2["fc_selected_indices"]
    out["importance"] = p2["importance"]

    p3 = run_phase3(
        phase1=p1,
        fc_selected_indices=p2["fc_selected_indices"],
        sc_matrix=p1.get("sc_matrix"),
        verbose=verbose,
    )
    out["posterior_2"] = p3["posterior"]
    out["embedding_2"] = p3["embedding_net"]
    out["theta_scaled_phase3"] = p3["theta_scaled"]
    out["fc_raw_phase3"] = p3["fc_raw"]
    return out
