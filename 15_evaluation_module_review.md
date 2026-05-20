# 15 — Evaluation Module Review

**Date:** 2026-05-18
**Branch:** refactor/02-simulation
**Reviewer:** Claude (re-created out-of-order after `16_opus_refactor_decision_plan.md`)
**Files inspected:**
`evaluation/metrics.py`, `evaluation/validation.py`,
`evaluation/model_selection.py`, `evaluation/final_test.py`,
`evaluation/plots.py`, `evaluation/reports.py`, `evaluation/__init__.py`,
`evaluate.py`
**Cross-referenced:** `11_import_audit.md`, `14_inference_module_review.md`,
`03_module_index.md`, `04_data_flow.md`, `07_refactor_rules.md`,
`config.py` (constants), `inference/posterior.py` and
`inference/__init__.py` (consumer surface)

---

## 1. Validation Metrics

The evaluation layer measures how well a trained posterior reconstructs
observed FC (and FCD, when enabled), and it does so the same way at every
stage of the pipeline — Stage 1 validation, Stage 2 validation, baseline,
and final test all flow through one shared per-subject function:

```
evaluate_subject(sid, ..., posterior, param_scaler, feature_pipeline)
  │
  ├── extract_observed_features(subject_data[sid])      → fc_obs_raw, fcd_obs_raw
  ├── feature_pipeline.transform(fc_obs_raw, fcd_obs_raw) → x_obs_input (300-dim)
  ├── infer_subject_raw(posterior, x_obs_input, param_scaler, n_samples=N_TEST_RESIM)
  │     → samples_raw, means_raw, stds_raw, samples_scaled  (50 samples by default)
  ├── compute_shrinkage_scaled(samples_scaled)          → per-parameter shrinkage
  └── _resimulate_and_score(samples_raw, sc, dly, fc_obs_full, fcd_obs_raw, ...)
        │   for each of N_TEST_RESIM posterior samples:
        ├── simulate_single(sc, params, n_repeat=1, delays=dly, apply_bw=True)
        ├── fc_pred = compute_fc(bold)
        ├── m = fc_metrics(fc_obs_full, fc_pred)        → corr, rmse, mae
        └── (if USE_FCD)
              fcd_pred_vec = fcd_to_upper_tri(compute_sim_fcd_matrix(bold))
              fcd_vec_rmse(fcd_obs_raw, fcd_pred_vec)
        → returns fc_corrs[], fc_rmses[], fcd_rmses[], fc_preds[]
```

The per-subject result dict carries enough state for plotting (`fc_obs`,
`fc_preds`, `samples_raw`) and aggregation (`fc_corr_mean`, `fc_rmse_mean`,
`fcd_rmse_mean`, `shrinkage_scaled`, etc.).

**N_TEST_RESIM = 50** (`config.py:261`) per subject for both validation and
final test. The same constant is reused at every layer; no separate
`N_VAL_RESIM` knob exists.

---

## 2. FC Correlation Metric

**Function:** `evaluation/metrics.py:fc_metrics` (lines 46–64)

```python
n  = fc_obs.shape[0]
iu = np.triu_indices(n, k=1)        # upper triangle, no diagonal
a  = fc_obs[iu]                      # 6555 values when N=115
b  = fc_pred[iu]
mask = np.isfinite(a) & np.isfinite(b)
if mask.sum() < 2:
    return {"corr": 0.0, "rmse": 1.0, "mae": 1.0}
a, b = a[mask], b[mask]
r = np.corrcoef(a, b)[0, 1] if (a.std() > 0 and b.std() > 0) else 0.0
return {"corr": r, "rmse": sqrt(mean((a-b)**2)), "mae": mean(|a-b|)}
```

Key properties:

| Aspect | Detail |
|---|---|
| Vectorisation | Full upper triangle, `k=1` (diagonal excluded) |
| NaN handling | Element-wise mask on `a` and `b`; both NaN-safe |
| Constant-signal guard | Returns `corr=0` if either vector has zero variance |
| Degenerate input | If <2 finite pairs survive masking: `corr=0, rmse=1, mae=1` |
| Output type | Plain Python floats (not numpy) |
| Returns 3 metrics in one call | Allows `_resimulate_and_score` to grab `corr` and `rmse` without recomputing the mask |

**Important:** `fc_metrics` does NOT consult `config.NAN_MASK`. It uses the
runtime `np.isfinite` mask only. The NAN_MASK protocol is applied earlier
(inside `features.fc.fc_to_upper_tri` during feature extraction), not here.
Observed FC in `subject_data[sid]["fc"]` has already been NaN→0 normalised
by `data_loader`, so the `isfinite` check is effectively redundant for the
observed side — but it remains a robust guard for the simulated side.

---

## 3. FC RMSE Metric

Computed in the same call as `fc_metrics`:

```python
rmse = sqrt(((a - b) ** 2).mean())
```

- Computed on the same masked upper-tri vector pair `(a, b)` used for
  correlation.
- Returns `1.0` if degenerate (paired with `corr=0`).
- No normalisation — raw RMSE on Pearson r values in `[-1, 1]`.
- Smaller is better. Selection score flips its sign (Section 6).

The "`rmse=1.0`" fallback is used both when fewer than 2 finite values
survive AND when the empty-list aggregation hits `r["fc_rmse_mean"] = 1.0`
(line 160). This is a non-zero failure floor — useful as a soft penalty
that keeps a degenerate subject from quietly tilting the validation mean
toward zero.

---

## 4. FCD Summary RMSE / Distance Metric

**Function:** `evaluation/metrics.py:fcd_vec_rmse` (lines 67–74)

```python
a = fcd_obs_vec.astype(float64)
b = fcd_pred_vec.astype(float64)
mask = np.isfinite(a) & np.isfinite(b)
if mask.sum() < 2:
    return 1.0
return sqrt(((a[mask] - b[mask]) ** 2).mean())
```

The function name `fcd_vec_rmse` and the deprecated alias `fcd_summary_rmse`
both refer to the same function. The "summary" naming is historical
(predates the move from 5-stat summary to upper-tri).

### Current call sites — all assume 6555-dim FCD upper-tri

`_resimulate_and_score` (line 211–214) and `baseline_eval` (line 277–280):

```python
fcd_pred_vec = fcd_to_upper_tri(compute_sim_fcd_matrix(bold))   # (6555,)
fcd_rmses.append(fcd_vec_rmse(fcd_obs_raw, fcd_pred_vec))
```

This requires `fcd_obs_raw` to also be **6555-dim** for the RMSE to be
meaningful. But `fcd_obs_raw` comes from
`features.extraction.extract_observed_features(d)`, which:
- Returns `(fc_vec, np.zeros(0))` when `FEATURE_SET == "fc_only"`.
- (Per doc 13, would return a non-empty FCD vector when USE_FCD=True.)

### Latent dimension-mismatch bug (USE_FCD=True only)

The same `fcd_obs_raw` is passed to **two consumers with conflicting
dimensional requirements:**

| Consumer | Expected `fcd_obs_raw` shape |
|---|---|
| `feature_pipeline.transform(fc_obs_raw, fcd_obs_raw)` | `(5,)` summary stats when `FeaturePipeline.use_fcd=True` |
| `fcd_vec_rmse(fcd_obs_raw, fcd_pred_vec)` paired with `fcd_to_upper_tri(...)` (6555,) | `(6555,)` upper triangle |

When `USE_FCD=False` (current production):
- Pipeline ignores the FCD branch (`use_fcd=False` inside `FeaturePipeline`).
- RMSE compare is gated by `if use_fcd:` and is skipped.
- Bug is dormant.

When `USE_FCD=True` (not currently the case):
- One of the two consumers will raise `ValueError`. There is no shape that
  satisfies both.

This bug exists in **four places**, matching the inference-side bugs in
RISK-I / RISK-II of doc 14:

| File | Line | Function |
|---|---|---|
| `evaluation/metrics.py` | 123 + 211 | `evaluate_subject` + `_resimulate_and_score` |
| `evaluation/metrics.py` | 245 + 277 | `baseline_eval` |
| `evaluation/validation.py` | 135 | `evaluate_validation_stage2` (s1_pipeline.transform) |
| `evaluation/final_test.py` | 133 | `_test_stage2` (s1_pipeline.transform) |

Doc 14 already flagged the same pattern in `inference/posterior.py` and
`inference/diagnostics.py`. Bundled with the inference-side fix, the
correct change is to pass `fcd_to_summary_stats(fcd_mat)` (5-dim) to the
pipeline and keep `fcd_to_upper_tri(fcd_mat)` (6555-dim) only for the
RMSE comparison. **Do not touch this until `USE_FCD` is enabled** — the
fix is scope-bundled with FCD enablement, not part of structural cleanup.

---

## 5. Posterior Predictive Validation Logic

There are **two distinct posterior-predictive code paths** in the repo:

| Function | Module | Role |
|---|---|---|
| `inference.posterior.posterior_predictive_check` | inference/ | Returns mean/std FC corr + FCD RMSE; **not called by validation** in current production |
| `evaluation.metrics.evaluate_subject` + `_resimulate_and_score` | evaluation/ | Production validation/test path |

Stage 1 and Stage 2 validation both call `evaluate_subject` (via
`evaluation/validation.py`), not `posterior_predictive_check`. The two
functions overlap in intent but differ in:

- Output shape: `evaluate_subject` returns per-subject `fc_corr_all` list,
  `fc_preds` list, shrinkage vector, posterior samples — richer dict for
  downstream plotting.
- Sampling: `evaluate_subject` draws `N_TEST_RESIM=50` samples;
  `posterior_predictive_check` defaults to `N_PPC=50` (same).
- `compute_shrinkage_scaled` is invoked inside `evaluate_subject`;
  `posterior_predictive_check` does not compute shrinkage.

The duplication is historical. New evaluation code should call
`evaluate_subject` (the production path); `posterior_predictive_check`
remains for direct inference-side diagnostic use and for SBC.

### Per-subject sampling flow

```
samples_scaled = posterior.sample((N_TEST_RESIM,), x=x_obs_input)
                                                    ↑
                                  fitted FeaturePipeline-transformed observed FC

samples_raw = param_scaler.inverse_transform(samples_scaled)
shrinkage   = compute_shrinkage_scaled(samples_scaled)

For each of N_TEST_RESIM samples:
   bolds = simulate_single(sc, params, n_repeat=1, delays=dly, apply_bw=True)
   fc_pred = compute_fc(bolds[0])
   collect (fc_corr, fc_rmse, fcd_rmse)
```

### Resimulation cost

Per validation subject: 50 simulations × `simulate_single` (each is a full
WC + BW run on GPU, ~T_END_MS / DT = 600,000 steps). For 2 validation
subjects × 50 = 100 sims per stage. Combined with 2 stages, plus 2 test
subjects × 50 = 100 sims, the resimulation budget is ~300 GPU sims per
pipeline run — substantial but tractable on H100.

### Robustness to per-sim failures

`_resimulate_and_score` wraps each per-sample simulation in
`try / except Exception` (line 215). A failing simulation prints a line
and is dropped from the metric averages. If ALL sims fail for a subject,
the result dict carries `fc_corr_mean=0.0, fc_rmse_mean=1.0` (the
degenerate fallbacks), which guarantees the subject contributes a
worst-case score rather than a NaN.

---

## 6. Model Selection Logic

**Function:** `evaluation/model_selection.py:select_best_model` (lines 87–126)

Inputs:
- `stage1_agg` — aggregate from Stage 1 validation
- `stage2_agg` — aggregate from Stage 2 validation (or `None`)
- `baseline_agg` — aggregate from `baseline_eval_subjects` (or `None`)

```
score_1 = compute_selection_score(stage1_agg, baseline_agg)
score_2 = compute_selection_score(stage2_agg, baseline_agg) if stage2_agg else -inf
best    = 2 if score_2 > score_1 else 1
```

### Selection score formulas

`compute_selection_score(val_agg, baseline_agg=None)` has two branches:

**With baseline (recommended):**

```
fc_corr_norm  = val.fc_corr - baseline.fc_corr                                  # delta (positive = better)
fc_rmse_norm  = (baseline.fc_rmse - val.fc_rmse) / max(baseline.fc_rmse, 1e-8)  # fractional improvement
fcd_rmse_norm = (baseline.fcd_rmse - val.fcd_rmse) / max(baseline.fcd_rmse, 1e-8)   [USE_FCD only]
score         = W_FC_CORR * fc_corr_norm + W_FC_RMSE * fc_rmse_norm + W_FCD_RMSE * fcd_rmse_norm
```

**Without baseline:**

```
fc_corr_norm  = val.fc_corr
fc_rmse_norm  = -val.fc_rmse                              [sign flipped]
fcd_rmse_norm = -val.fcd_rmse if USE_FCD else 0
score         = W_FC_CORR * fc_corr_norm + W_FC_RMSE * fc_rmse_norm + W_FCD_RMSE * fcd_rmse_norm
```

Both branches preserve **higher = better** for every term. RMSE
contributions are sign-flipped (or expressed as fractional improvement
over baseline) so the additive score is consistent.

### Weights (config.py:269–271)

```
SELECT_W_FC_CORR  = 1.0
SELECT_W_FC_RMSE  = 0.5
SELECT_W_FCD_RMSE = 0.5
```

FC correlation dominates; RMSE acts as a regulariser. FCD weight is set
but the term is silenced when `USE_FCD=False` (line 78 of
`model_selection.py`):

```python
if use_fcd:
    score += config.SELECT_W_FCD_RMSE * fcd_rmse_norm
```

The model_selection docstring explicitly justifies this: "Mixing
zero-valued FCD RMSEs from both stages would still tie the metric to
noise, so we drop the term entirely."

### Stage 1-only branch

If `stage2_agg is None` (Stage 2 was skipped via `RUN_STAGE2=0`), the
function unconditionally returns `best=1`, regardless of how Stage 1
compares to baseline. The score table is printed in a single column
(`_print_selection_table` lines 158–161 handle the N/A formatting).

### Printed output

`_print_selection_table` shows a 3-column table (Stage 1, Stage 2, delta).
FCD row is omitted when `USE_FCD=False`. Delta is computed as
`stage2 - stage1` for FC corr (positive = Stage 2 better) and
`stage2 - stage1` for RMSEs (negative = Stage 2 better — note the sign
convention is NOT flipped in the printed delta, only the selection score
flips it; this is a UX inconsistency, not a correctness bug).

---

## 7. Final Test Logic

**Function:** `evaluation/final_test.py:final_test` (lines 32–78)

Inputs:
- `test_subjects` — the held-out test set (2 subjects in default split)
- `best_stage` — `1` or `2`, output of `select_best_model`
- `stage1_result`, `stage2_result` — artifact dicts

Flow:

```
if best_stage == 1:
    results = _test_stage1(...)        # uses Stage 1 posterior + pipeline
else:
    results = _test_stage2(...)        # uses Stage 2 posterior, with Stage 1 nuisance fix

all_fc_corrs = flatten(r["fc_corr_all"] for r in results)
fc_corr_boot = bootstrap_ci(all_fc_corrs)            # mean + 95% CI
fc_rmse_boot = bootstrap_ci([r["fc_rmse_mean"] ...])
fcd_rmse_boot = bootstrap_ci([r["fcd_rmse_mean"] ...])  # all zeros when USE_FCD=False

return {
    "best_stage": int,
    "per_subject": list[result_dict],
    "fc_corr_boot_ci":  (mean, lo, hi),
    "fc_rmse_boot_ci":  (mean, lo, hi),
    "fcd_rmse_boot_ci": (mean, lo, hi),
}
```

### Stage-aware test driver

- `_test_stage1`: calls `evaluate_subject` with Stage 1 posterior/scaler/
  pipeline and `param_names=config.STAGE1_PARAMS`. No nuisance handling
  needed.
- `_test_stage2`: for each test subject, first runs Stage 1 posterior
  forward (2000 samples — hardcoded, NOT `config.N_POSTERIOR`) to get the
  posterior mean of nuisance parameters, then calls `evaluate_subject`
  with Stage 2 posterior/scaler/pipeline and `param_names=stage2_params`,
  passing `fixed_overrides=fixed_for_s2` to inject the nuisance means.

This **Stage 2 nuisance handling at test time is `"fix_mean"`** — NOT
`"posterior_sample"` (`config.NUISANCE_METHOD`). It does not consult the
config knob; it is hardcoded as fix-mean. This is a deliberate choice
(more reproducible test evaluation) but it means `config.NUISANCE_METHOD`
controls training-data collection only, not test-time scoring.

### Bootstrap CI

```python
n        = config.BOOTSTRAP_N (1000)
rng      = RandomState(42)            # fixed seed for reproducibility
boots    = [rng.choice(values, size=len(values), replace=True).mean()
            for _ in range(1000)]
return (np.mean(boots), percentile(boots, 2.5), percentile(boots, 97.5))
```

- Seeded with `42` (not `config.SEED`). This is intentional so that the
  bootstrap is reproducible independently of the global pipeline seed,
  but it does mean changing `config.SEED` won't affect CI numbers.
- `values < 2` finite → returns `(mean, 0.0, 0.0)` (zero-width CI). Catches
  the degenerate single-subject case.

### Test rule enforcement

The test set is invoked **once** in `final_test`, after `select_best_model`.
R10 (train-only fitting) is respected because `final_test` only calls
`evaluate_subject`, which only calls `feature_pipeline.transform` (never
`.fit`). The pipeline was fitted in Stage 1 training and frozen.

---

## 8. Plot / Report Generation

### `evaluation/plots.py` — 6 figure functions + 2 aliases

| Function | Output | Notes |
|---|---|---|
| `plot_posteriors(results, param_names, prior_low, prior_high, title, save_path)` | `(n_subj × n_p)` histogram grid of posterior samples per subject and parameter, with red vline at posterior mean | x-range fixed to prior bounds; figure saved to `OUTPUT_DIR/posterior_<title>.png` |
| `plot_fc_comparison(results, save_path, title)` | `(n_subj × 2)` heatmap grid: observed FC vs mean predicted FC | Uses `RdBu_r` colormap, fixed range `[-1, 1]`; predicted is the elementwise mean of `fc_preds` |
| `plot_posteriors_two_stage(...)` | Alias of `plot_posteriors` with `title="Stage 1+2"` | Same figure logic |
| `plot_fc_comparison_two_stage(...)` | Alias of `plot_fc_comparison` | Same figure logic |
| `plot_sbc_rank_histogram(ranks, param_names, save_path)` | `(1 × n_p)` histogram of SBC ranks per parameter, with red dashed uniform reference | `bins=SBC_BINS=20`; saved as `sbc_ranks.png`. Returns early if `ranks is None` or empty. |
| `plot_pca_diagnostic(pca_diag, save_path)` | Bar plot of top-5 FC PC explained variance ratios | Reads `pca_diag["fc_pca"]["explained_variance_top5"]`; tolerates flat dict via `fc_diag = pca_diag.get("fc_pca", pca_diag)`. Returns early if EVR data missing. |
| `plot_one_simulation(sid, subject_data, theta_raw, param_names, sim_idx, save_name)` | 2x2 figure: BOLD timeseries (5 regions), full BOLD heatmap, sim FC, obs FC | Re-simulates `theta_raw[sim_idx]` via `simulator.simulate_single` for sanity check after Step 2 |

**Matplotlib backend forced to `Agg`** at module load (`plots.py:20` and
`reports.py:17`) — figures save without a display server. `plot_one_simulation`
calls `plt.show()` (line 266) which is a no-op under `Agg` but a regression
risk if backend ever changes.

### `evaluation/reports.py` — 16 console reporters

Two layers of report functions:

1. **End-of-run summaries:** `print_summary_two_stage`, `print_final_summary`
2. **Notebook-style step reports:** `report_step1` through `report_step14` —
   one per pipeline step, designed to be called from `main.ipynb` cells

`report_step1` and `report_step2` produce matplotlib figures
(SC sparsity + FC NaN bar plots; Stage-1 theta histograms). All other
report functions print only.

**`evaluate_all_two_stage`** (line 66) is a stub that raises
`NotImplementedError` — intentional placeholder for a legacy notebook
hook that has been replaced by the `main.py` orchestration. Do not call.

### Output directory

All `save_path = None` defaults resolve to
`os.path.join(config.OUTPUT_DIR, "<step>_<name>.png")`. Default
`OUTPUT_DIR = "./output_mouse_mptp"`. The directory is created by
`save_extracted_features` and other inference-side calls; plots assume
it exists.

---

## 9. Whether Root-Level `evaluate.py` Is Still Used

**Status: ACTIVE COMPAT WRAPPER. Cannot be deleted yet.**

Verified via grep:

```
grep -n "^import evaluate\b\|^from evaluate import" *.py inference/ evaluation/ pipelines/
```

Results:
- **`pipelines/stage1_stage2.py:38`** — `import evaluate`  *(top-level, production)*
- **`main.ipynb` cell ~163** — `import evaluate`
- **`debug_notebook.py`** — 2 sites referencing `evaluate`

The single top-level production import is the critical blocker. Every
pipeline run executes:

```
pipelines/stage1_stage2.py:38   import evaluate     ←  evaluate.py:24  from evaluation import *
```

…then uses names like `evaluate.evaluate_validation_stage1`,
`evaluate.select_best_model`, `evaluate.final_test`,
`evaluate.print_final_summary`. All of these resolve through the wildcard
re-export in `evaluate.py`:

```python
# evaluate.py — entire content (35 lines, no logic)
from evaluation import *                          # noqa: F401, F403
from evaluation import (                          # noqa: F401
    _aggregate_validation, _print_selection_table, _print_test_summary,
    _print_validation_summary, _progress, _resimulate_and_score,
    _test_stage1, _test_stage2,
)
```

The second `from evaluation import (...)` block explicitly re-exports
**private helpers** that the package's `__init__.py` also exports — this
covers any legacy caller that reaches into `evaluate._foo`. Verified by
reading `evaluation/__init__.py`: all 8 underscored names appear in its
public imports.

**Conclusion:** `evaluate.py` is purely a re-export shim. Zero logic.
It is the smallest file in the legacy layer (35 lines). It will be
deletable as soon as the single line in `pipelines/stage1_stage2.py:38`
is migrated.

---

## 10. Stage 1 vs Stage 1 + Stage 2 Comparison Support

The evaluation module supports comparing Stage 1 alone vs Stage 1+2 at
three layers:

### 10.1 Per-stage validation

- `evaluate_validation_stage1(val_subjects, subject_data, stage1_result)`
  → `(results, agg)` for Stage 1.
- `evaluate_validation_stage2(val_subjects, subject_data, stage2_result,
  stage1_result)` → `(results, agg)` for Stage 2.

Both return the same agg dict schema (`fc_corr_mean`, `fc_rmse_mean`,
`fcd_rmse_mean`, `shrinkage_mean`, `shrinkage_per_param`, `per_subject`,
`param_names`). Stage 2 agg additionally has `nuisance_params`.

**Stage 2 validation uses Stage 1 to fix nuisances** (lines 134–141):
```python
fc_obs_raw, fcd_obs_raw = extract_observed_features(d)
x_s1 = s1_pipeline.transform(fc_obs_raw, fcd_obs_raw)
_, s1_means_raw, _, _ = infer_subject_raw(
    s1_posterior, x_s1, s1_param_scaler,
    n_samples=2000, verbose=False,
)
fixed_for_s2 = {n: float(s1_lookup[n]) for n in nuisance}
```

This is `fix_mean` nuisance handling (NOT `posterior_sample` from
`config.NUISANCE_METHOD`) — same convention as `_test_stage2`.

### 10.2 Selection comparison

`select_best_model(stage1_agg, stage2_agg, baseline_agg)` returns
`(best_stage, scores)`. The selection score formula treats both stages
symmetrically (same weighted combination). Stage 2's `score_2` is set to
`-inf` if no Stage 2 was run; otherwise Stage 2 wins only when
`score_2 > score_1` (strict inequality — ties go to Stage 1).

### 10.3 Reporting

- `_print_selection_table` (lines 129–162 of `model_selection.py`) —
  side-by-side 3-column table with delta.
- `print_summary_two_stage` (reports.py lines 28–63) — compact two-line
  summary with `Δ corr` and `Δ rmse`. Reports `Δ rmse = s1 - s2`
  (positive = Stage 2 better) — note this inverts the sign convention
  used in `_print_selection_table` (which prints `s2 - s1`). Two different
  display conventions in two different printers.
- `print_final_summary` (reports.py lines 78–122) — 4-column table:
  metric × (Val S1, Val S2, Test). FCD row hidden when `USE_FCD=False`.
- `plot_posteriors_two_stage`, `plot_fc_comparison_two_stage` — figure
  aliases for two-stage results.

### 10.4 Two-stage rule enforcement

The two-stage comparison strictly obeys R10:
- Stage 2 training uses validation-derived `theta_bad`
  (sensitivity + shrinkage from validation subjects).
- Final test re-uses the **selected** stage's posterior on the held-out
  test set.
- Test subjects never enter validation, selection, or training. Verified
  by reading the entire `final_test.py` and `validation.py` — neither
  module references `test_subjects` outside `final_test`.

---

## 11. Import Conflict Risks

### IRISK-1 — All evaluation deferred `from simulator import` calls

Verified count: **8 deferred imports** in `evaluation/` (matching the
inventory in `11_import_audit.md` Section 2.2):

| File | Line | Symbols |
|---|---|---|
| `evaluation/metrics.py` | 110 | `extract_observed_features` (inside `evaluate_subject`) |
| `evaluation/metrics.py` | 184 | `compute_fc, compute_sim_fcd_matrix, fcd_to_upper_tri, simulate_single` (inside `_resimulate_and_score`) |
| `evaluation/metrics.py` | 245 | `compute_fc, compute_sim_fcd_matrix, fcd_to_upper_tri, extract_observed_features, simulate_single` (inside `baseline_eval`) |
| `evaluation/validation.py` | 123 | `extract_observed_features` (inside `evaluate_validation_stage2`) |
| `evaluation/final_test.py` | 112 | `extract_observed_features` (inside `_test_stage2`) |
| `evaluation/plots.py` | 203 | `simulate_single, compute_fc` (inside `plot_one_simulation`) |

All are inside function bodies. Module-load time is unaffected. Each
fires only when the enclosing function is called; in production every
validation and test step hits at least one. Removing `simulator.py`
before these are migrated will cause `ModuleNotFoundError` at the first
call site.

### IRISK-2 — `from inference import infer_subject_raw` (correct, but indirect)

`evaluation/metrics.py:111`, `evaluation/validation.py:122`,
`evaluation/final_test.py:111` all use `from inference import infer_subject_raw`
(deferred). This resolves to the `inference/` package (correct) — but
relies on Python's package-over-module resolution. If `inference.py`
monolith ever shadowed the package (it currently does not — verified at
runtime), these calls would silently use the monolith's `infer_subject_raw`
which has its own `from simulator import` chain plus the `n_subj` NameError
documented in `14_inference_module_review.md`. Mitigated by R8 and runtime
verification.

### IRISK-3 — `evaluation/__init__.py` re-exports private helpers

The `evaluate.py` compat wrapper accesses private helpers
(`_aggregate_validation`, `_print_selection_table`, `_print_test_summary`,
`_print_validation_summary`, `_progress`, `_resimulate_and_score`,
`_test_stage1`, `_test_stage2`) via the explicit second import block.
This works because `evaluation/__init__.py` explicitly re-imports each
private name from its submodule. If anyone tightens `evaluation/__init__.py`
(e.g., adds an `__all__` that excludes underscored names), the compat
wrapper breaks silently. Keep `evaluation/__init__.py` as a permissive
re-export hub for now.

### IRISK-4 — FCD dimension mismatch (latent — see Section 4)

`fcd_obs_raw` from `extract_observed_features` is dual-purposed: it is
passed to `feature_pipeline.transform` (needs 5-dim summary stats when
USE_FCD=True) and to `fcd_vec_rmse` paired with 6555-dim
`fcd_to_upper_tri`. Bug is dormant when `USE_FCD=False`. Same root cause
as RISK-I/II in doc 14. **Must be fixed in evaluation AT THE SAME TIME
as the inference-side fix**, or USE_FCD enablement will partly succeed
(inference passes) and partly fail (evaluation crashes).

### IRISK-5 — Redundant `shrinkage_mean` / `shrinkage_per_param` in agg dict

`evaluation/validation.py:_aggregate_validation` (lines 184–189) stores
**the same computation twice**:

```python
"shrinkage_mean":     np.mean([r["shrinkage_scaled"] for r in results], axis=0),
"shrinkage_per_param": np.mean([r["shrinkage_scaled"] for r in results], axis=0),
```

Both are identical. Downstream consumers reference both — `report_step9`
uses `shrinkage_mean`, while `pipelines/stage1_stage2.py` may use
`shrinkage_per_param` for θ_bad selection (per doc 10/14). Not a bug
today, but a code smell: changing one without the other will desync.
**Do not unify yet** — wait for an explicit refactor pass with the user.

### IRISK-6 — `2000` hardcoded in Stage 2 nuisance sampling

`evaluation/validation.py:138` and `evaluation/final_test.py:136` both
sample 2000 from the Stage 1 posterior to compute nuisance means. This
is independent of `config.N_POSTERIOR` (which equals 2000 anyway). If a
future user reduces `N_POSTERIOR` for speed, the nuisance estimation
will retain its 2000-sample budget. Latent inconsistency, not a bug.

### IRISK-7 — Two-stage delta sign convention divergence

`_print_selection_table` (model_selection.py:135–146) prints
`delta = stage2 - stage1` for both corr and RMSE. The reader must mentally
flip the sign for RMSE (negative = Stage 2 better).
`print_summary_two_stage` (reports.py:57–62) prints `Δ rmse = s1 - s2`
(positive = Stage 2 better). Two displays disagree on sign convention.
Cosmetic/UX issue, not numeric.

---

## 12. Minimal Future Refactor Plan for `evaluation/`

### Step E1 — Migrate the deferred `from simulator import` calls (Priority: HIGH)

Bundled with Tier 4 of `11_import_audit.md` (and Tier 4 / Patch P-4 series
of `16_opus_refactor_decision_plan.md`). Six mechanical replacements:

| File | Line | New imports |
|---|---|---|
| `evaluation/metrics.py` | 110 | `from features.extraction import extract_observed_features` |
| `evaluation/metrics.py` | 184 | `from features.fc import compute_fc`<br>`from features.fcd import compute_sim_fcd_matrix, fcd_to_upper_tri`<br>`from simulation.wc_runner import simulate_single` |
| `evaluation/metrics.py` | 245 | All of the above + `from features.extraction import extract_observed_features` |
| `evaluation/validation.py` | 123 | `from features.extraction import extract_observed_features` |
| `evaluation/final_test.py` | 112 | `from features.extraction import extract_observed_features` |
| `evaluation/plots.py` | 203 | `from features.fc import compute_fc`<br>`from simulation.wc_runner import simulate_single` |

Each replacement is a deferred import (inside a function body); no other
code in the function changes. After all six land, `simulator.py` can be
deleted (jointly with the inference-side migrations from doc 14 / doc 16).

### Step E2 — Pair `feature_pipeline` FCD input with `fcd_to_summary_stats` (Priority: MEDIUM, before enabling FCD)

**Do not apply yet.** Bundled with USE_FCD enablement. Once enabled:

- `evaluate_subject` (line 123) and `baseline_eval` (line 252) must split
  `fcd_obs_raw` into two: `fcd_obs_pipeline = fcd_to_summary_stats(fcd_mat)` (5-dim)
  for the pipeline call, and `fcd_obs_rmse = fcd_to_upper_tri(fcd_mat)` (6555-dim)
  for the RMSE compare.
- `_resimulate_and_score` should accept both shapes via a single
  `fcd_obs_*` argument (or, cleaner, take the raw FCD matrix and
  compute both internally).
- `_test_stage2` and `evaluate_validation_stage2` need analogous splits
  on lines 133/135 respectively.

The minimal API change: `extract_observed_features` should return the
FCD matrix (`(N, N)`) instead of a single vector. Downstream consumers
then choose summary vs upper-tri.

### Step E3 — Replace `evaluate.evaluate_all_two_stage` stub or document it (Priority: LOW)

`reports.py:66` raises `NotImplementedError`. It is exported by
`evaluation/__init__.py` and re-exported by `evaluate.py`. Confirm no
notebook cell calls it; if confirmed dead, remove. If a notebook cell
does call it, replace with a working two-stage runner or update the
docstring to point to `main.py`'s orchestration.

### Step E4 — Reconcile shrinkage_mean / shrinkage_per_param (Priority: LOW)

After Step E1 lands, choose one key and update all consumers. Either:
- Keep `shrinkage_per_param` and remove `shrinkage_mean` from the agg
  dict (more explicit name).
- Or keep `shrinkage_mean` and remove `shrinkage_per_param` (shorter).

Pick one; grep all callers in `pipelines/`, `reports.py`, and any
notebook. Make the substitution atomically.

### Step E5 — Unify two-stage delta sign convention (Priority: LOW, UX)

Pick one convention (recommended: `s1 - s2` everywhere, with "positive =
Stage 2 better" annotation) and apply to `_print_selection_table` and
`print_summary_two_stage` together.

### Step E6 — Replace hardcoded `n_samples=2000` with `config.N_POSTERIOR` (Priority: LOW)

In `evaluation/validation.py:138` and `evaluation/final_test.py:136`.
Strictly cosmetic today (both equal 2000), but removes a latent drift
risk.

### Step E7 — Move `bootstrap_ci` seed to `config.SEED` or document (Priority: LOW)

`bootstrap_ci` uses `RandomState(42)` (line 89). If the user expects
`config.SEED` to control all randomness, this is surprising. Either
read from `config.SEED` or update the docstring to clarify the
intentional independence.

---

## 13. Test Commands for `evaluation/`

### 13.1 Compile check (no GPU)

```bash
python -m py_compile evaluation/__init__.py evaluation/metrics.py \
    evaluation/validation.py evaluation/model_selection.py \
    evaluation/final_test.py evaluation/plots.py evaluation/reports.py
python -m py_compile evaluate.py
echo "evaluation/ + evaluate.py compile OK"
```

### 13.2 Import resolution (no GPU)

```bash
python -c "
import evaluation
assert 'evaluation/__init__' in evaluation.__file__, evaluation.__file__
from evaluation import (
    fc_metrics, fcd_vec_rmse, bootstrap_ci, evaluate_subject,
    baseline_eval, baseline_eval_subjects,
    evaluate_validation_stage1, evaluate_validation_stage2,
    compute_selection_score, select_best_model,
    final_test,
    plot_posteriors, plot_fc_comparison, plot_sbc_rank_histogram,
    plot_pca_diagnostic, plot_one_simulation,
    print_summary_two_stage, print_final_summary,
)
print('evaluation imports OK')

# compat wrapper still works
import evaluate
assert hasattr(evaluate, 'fc_metrics')
assert hasattr(evaluate, 'evaluate_validation_stage1')
assert hasattr(evaluate, 'select_best_model')
assert hasattr(evaluate, 'final_test')
assert hasattr(evaluate, 'print_final_summary')
# private helpers
assert hasattr(evaluate, '_aggregate_validation')
assert hasattr(evaluate, '_print_selection_table')
assert hasattr(evaluate, '_progress')
print('evaluate.py compat OK')
"
```

### 13.3 Dependency rule (R3) — evaluation/ may import inference/, simulation/, features/, but not pipelines/

```bash
grep -rn "from pipelines\|import pipelines" evaluation/*.py
# Expected: zero output

# These are allowed:
grep -n "from inference\|from simulation\|from features" evaluation/*.py
# Expected: present (deferred) — currently via 'from simulator import' compat wrapper
```

### 13.4 fc_metrics smoke test (no GPU)

```bash
python -c "
import numpy as np
from evaluation.metrics import fc_metrics, fcd_vec_rmse, bootstrap_ci
rng = np.random.RandomState(0)
N = 115
fc_obs = (np.eye(N) + 0.1 * rng.randn(N, N)).astype(np.float32)
fc_obs = (fc_obs + fc_obs.T) / 2  # symmetrize
fc_pred = fc_obs + 0.05 * rng.randn(N, N).astype(np.float32)
m = fc_metrics(fc_obs, fc_pred)
assert 0.5 < m['corr'] < 1.0, f'corr {m[\"corr\"]}'
assert m['rmse'] < 0.2, f'rmse {m[\"rmse\"]}'
assert m['mae']  < 0.2, f'mae  {m[\"mae\"]}'
print('fc_metrics OK:', m)

vec_obs = rng.randn(6555).astype(np.float32)
vec_pred = vec_obs + 0.1 * rng.randn(6555).astype(np.float32)
print('fcd_vec_rmse OK:', fcd_vec_rmse(vec_obs, vec_pred))

mean, lo, hi = bootstrap_ci([0.7, 0.75, 0.8, 0.72, 0.78])
print(f'bootstrap_ci OK: mean={mean:.3f}, CI=[{lo:.3f}, {hi:.3f}]')
"
```

### 13.5 Model selection smoke test (no GPU)

```bash
python -c "
from evaluation.model_selection import compute_selection_score, select_best_model
import config

s1 = {'fc_corr_mean': 0.7, 'fc_rmse_mean': 0.2, 'fcd_rmse_mean': 0.0}
s2 = {'fc_corr_mean': 0.8, 'fc_rmse_mean': 0.15, 'fcd_rmse_mean': 0.0}
base = {'fc_corr_mean': 0.4, 'fc_rmse_mean': 0.35, 'fcd_rmse_mean': 0.0}

# without baseline
sc1 = compute_selection_score(s1)
sc2 = compute_selection_score(s2)
assert sc2 > sc1, f's2 should beat s1: sc1={sc1}, sc2={sc2}'

# with baseline
sc1b = compute_selection_score(s1, base)
sc2b = compute_selection_score(s2, base)
assert sc2b > sc1b, f'with baseline: sc1b={sc1b}, sc2b={sc2b}'

best, scores = select_best_model(s1, s2, base, verbose=False)
assert best == 2
print(f'select_best_model OK: best={best}, scores={scores}')

best1, _ = select_best_model(s1, None, base, verbose=False)
assert best1 == 1
print('select_best_model (no stage2) OK: returns 1')
"
```

### 13.6 No top-level legacy imports (R7)

```bash
grep -n "^import \|^from " evaluation/*.py | grep -v "^evaluation/" | head -20
# (Expected: top-level imports are all numpy/matplotlib/time/os/config and
# internal evaluation.* — no 'from simulator', no 'from evaluate')
```

### 13.7 Confirm evaluate.py is still a thin wrapper

```bash
python -c "
import os
with open('evaluate.py') as f:
    src = f.read()
# Should contain no function/class definitions
assert 'def ' not in src, 'evaluate.py should not define functions'
assert 'class ' not in src, 'evaluate.py should not define classes'
# Should only contain imports + comments
assert 'from evaluation' in src
print('evaluate.py is a pure re-export wrapper:', len(src), 'bytes')
"
```

---

## 14. Constants Used by `evaluation/`

| Constant | Value | Used in |
|---|---|---|
| `N_TEST_RESIM` | 50 | `evaluate_subject` (and validation+final_test default) |
| `N_PPC` | 50 | `inference/posterior.py:posterior_predictive_check` (not the evaluation path) |
| `N_SBC` | 200 | `inference/diagnostics.py:simulation_based_calibration` |
| `SBC_BINS` | 20 | `plot_sbc_rank_histogram` |
| `BOOTSTRAP_N` | 1000 | `bootstrap_ci` |
| `DIFFICULT_SHRINKAGE` | 0.3 | `evaluate_subject` (low-shrinkage tag); `report_step9/10` |
| `SELECT_W_FC_CORR` | 1.0 | `compute_selection_score` |
| `SELECT_W_FC_RMSE` | 0.5 | `compute_selection_score` |
| `SELECT_W_FCD_RMSE` | 0.5 | `compute_selection_score` (silenced if USE_FCD=False) |
| `USE_FCD` | False | every USE_FCD-gated branch in evaluation |
| `OUTPUT_DIR` | `./output_mouse_mptp` | every plot save path |
| `STAGE1_PARAMS` | `["P", "Q", "g_e", "g_i"]` | `evaluate_validation_stage1`, `_test_stage1`, baseline params |
| `STAGE1_PRIOR_LOW/HIGH` | tuples | `baseline_eval` (prior midpoint), plot x-ranges |
| `C_PARAM_PRIOR` | dict | `report_step10` |
| `SEED` | 42 | NOT used by `bootstrap_ci` (which uses its own `RandomState(42)`) |
| `TR_SEC` | 1.0 | `plot_one_simulation` time axis |

---

## Final Assessment

### Core evaluation files

| File | Status | Notes |
|---|---|---|
| `evaluation/__init__.py` | **Core.** Re-export hub. Includes private helpers (intentional for `evaluate.py` parity). | Do not tighten with an `__all__` until the compat wrapper is removed. |
| `evaluation/metrics.py` | **Core.** Single-subject evaluation, resimulation, baseline, FC/FCD metric primitives, bootstrap CI. | Contains 3 deferred `from simulator import` sites. Houses the FCD dual-role pattern (IRISK-4). |
| `evaluation/validation.py` | **Core.** Stage 1 and Stage 2 validation drivers; aggregation. | Contains 1 deferred simulator import; uses Stage 1 posterior to fix Stage 2 nuisances (fix_mean). |
| `evaluation/model_selection.py` | **Core.** Selection-score formula, best-stage picker, table printer. | Pure logic, no simulator imports. Cleanest file in the package. |
| `evaluation/final_test.py` | **Core.** Held-out test driver with bootstrap CI. | Contains 1 deferred simulator import. Enforces R10 by calling only `transform`, never `.fit`. |
| `evaluation/plots.py` | **Core.** 6 figure functions + 2 aliases. `Agg` backend forced. | Contains 1 deferred simulator import (in `plot_one_simulation`). |
| `evaluation/reports.py` | **Core.** 16 console reporters + 2 figure reporters (step1/step2). | No simulator imports. `evaluate_all_two_stage` is a `NotImplementedError` stub. |

### Whether `evaluate.py` is legacy or active

**Active compat wrapper. Not deletable today.**

- 0 lines of logic (35 lines total — all `from evaluation import …` lines).
- 1 live top-level caller: `pipelines/stage1_stage2.py:38`.
- 2 notebook cell callers (`main.ipynb` cells ~163/165, `debug_notebook.py`).
- Re-exports 8 private helpers (`_aggregate_validation`, etc.) for legacy
  callers that reach into private namespaces.

It becomes deletable after Patch P-1 from
`16_opus_refactor_decision_plan.md`:

```diff
-import evaluate
+import evaluation as evaluate
```

…plus the two notebook cell updates (Tier 4b of doc 16). After both
land, `evaluate.py` is unused at module-load time and can be removed in
Tier 5.

### Files that must NOT be deleted yet

| File | Reason |
|---|---|
| `evaluate.py` | Live top-level call site (`pipelines/stage1_stage2.py:38`). Single-line patch unblocks removal. |
| `simulator.py` | 6 of the 8 evaluation deferred imports depend on it. Cannot be removed until Patches P-4 series of doc 16 land. |
| `evaluation/metrics.py` | Houses `_resimulate_and_score`, which is the shared engine used by validation and final test. Do not rewrite. |
| `evaluation/validation.py` | Contains the only validation entrypoints. Stage 2 nuisance handling here is `fix_mean` (hardcoded) — touching this changes scientific behavior at validation time. |
| `evaluation/final_test.py` | Single-shot test path. R10-critical. |
| `evaluation/model_selection.py` | Selection score formula is the **only** thing that picks Stage 1 vs Stage 2. Touching weights or the FCD branch is a scientific change (R1). |

### What should be done before real code modifications

In order:

1. **Confirm working-tree state of `config.py` and `simulation/wc_runner.py`**
   (both have status `M`). Run `git diff config.py` and
   `git diff simulation/wc_runner.py` to check whether any scientific
   constant or per-sim parameter contract has unsigned changes pending.
   If yes, surface them to the user before any structural patch begins.

2. **Run the test sequence from `16_opus_refactor_decision_plan.md` Section 11**
   (T-A through T-D) once on the current tree. Establish a baseline
   "everything compiles / resolves" before any patch lands.

3. **Read `pipelines/stage1_stage2.py` in full** — every `evaluate.X(...)`
   call in this file must be in `evaluation/__init__.py`. Spot-check
   shows they are; a full pass closes the loop before Patch P-1.

4. **Apply Patch P-1** (`pipelines/stage1_stage2.py:38` →
   `import evaluation as evaluate`). Re-run T-A through T-E. This is the
   smallest change that unblocks compat-wrapper removal.

5. **Apply Patch P-2** (`data_loader.py:278` →
   `from simulation.delays import compute_delay_matrix`). Re-run tests.

6. **Apply Patches P-3a..e** (5 inference files) and **P-4 series**
   (4 evaluation files + plots) in the Tier order, one patch per commit.

7. **Do NOT touch `USE_FCD`, `FEATURE_SET`, or any scientific constant in
   `config.py`.** The FCD dimension fixes (Step E2) are bundled with
   FCD enablement and require explicit user direction; they are not
   part of structural cleanup.

8. **Do NOT modify `inference.py`** (R8). It will be deleted in Tier 7
   only after `inference/__init__.py` is verified to export a strict
   superset.

9. **Do NOT edit `main.ipynb`** mid-task. Notebooks produce noisy diffs
   and may carry the user's WIP. Save it for last (Tier 4b).

10. **Stop and ask the user** before any deletion (Tiers 5–7), any
    notebook edit, or any change that crosses the structural-vs-scientific
    boundary. The stop conditions in
    `16_opus_refactor_decision_plan.md` Section 13 apply uniformly.

The recommended first concrete action remains Patch P-1 from doc 16.
The evaluation package is structurally clean, fully consistent with R3
(no `from pipelines` anywhere; deferred simulator imports only), and
needs no Tier-X scientific edits as long as `USE_FCD` stays False.
