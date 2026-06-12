"""Save per-subject region-wise parameter maps (latent_regionwise mode).

For each subject: empirical FC -> posterior q(z|FC) -> posterior-mean z ->
decode (subject SC basis) -> per-region maps for the active HETERO_PARAMS.
Output: param_maps array of shape (n_subjects, n_regions, n_active_params).
"""
import numpy as np


def save_param_maps(posterior, feature_pipeline, param_scaler, subject_data,
                    subjects, out_path, n_post=None, verbose=True):
    import config
    from simulator import extract_observed_features
    from inference.posterior import infer_subject_raw
    from region_basis import build_region_basis
    from param_decoder import decode_latent_to_param_maps
    from engine_select import load_network_labels

    labels = load_network_labels()
    hp = list(config.HETERO_PARAMS)
    n_post = n_post or getattr(config, "N_POSTERIOR", 1000)
    R = int(config.N_REGIONS)

    maps_all, z_all, sids = [], [], []
    for sid in subjects:
        d = subject_data[sid]
        fc_obs_raw, fcd_obs_raw = extract_observed_features(d)
        x = feature_pipeline.transform(fc_obs_raw, fcd_obs_raw)
        _, zmean, _, _ = infer_subject_raw(
            posterior, x, param_scaler, n_samples=n_post, verbose=False)
        basis = build_region_basis(d["sc"], labels, config)
        m = decode_latent_to_param_maps(np.asarray(zmean)[None, :], basis, config)
        stacked = np.stack([m[p][0] for p in hp], axis=-1)   # (R, n_active)
        assert stacked.shape == (R, len(hp))
        maps_all.append(stacked)
        z_all.append(np.asarray(zmean))
        sids.append(int(sid))

    arr = np.stack(maps_all)                                  # (n_subj, R, n_active)
    np.savez(
        out_path,
        param_maps=arr.astype(np.float32),
        param_order=np.array(hp),
        subjects=np.array(sids),
        z_mean=np.stack(z_all).astype(np.float32),
    )
    if verbose:
        print(f"  [param maps] saved {out_path}")
        print(f"    shape = {arr.shape}  (n_subjects, n_regions, {len(hp)})  "
              f"params = {hp}")
        for k, p in enumerate(hp):
            v = arr[:, :, k]
            print(f"    {p:7s}: mean={v.mean():.4f} std={v.std():.4f} "
                  f"min={v.min():.4f} max={v.max():.4f}")
    return arr
