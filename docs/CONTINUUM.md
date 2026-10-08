# The continuum in 0.22: joint correction, corner diagnostic, continuum study

The 0.20 validation on 17 disks found the continuum to be the single largest systematic: the model-aware
refinement pass cut χ² by a third on GW Lup, water temperatures differ by 100–300 K from Romero-Mirza+2024 (GP
continuum) even where the fit is otherwise fine, and in 7 blind / 8 known disks the hot water component sits at
both upper bounds (T ≈ 1500 K, log N ≈ 21, R ≈ 0.01–0.02 au) — a tiny, hot, optically thick slab whose forest
of weak lines mimics a pseudo-continuum, after which the warm and cold components move to compensate.  0.22
adds three tools.  None of them changes an existing config's results unless switched on.

## 1. Joint continuum correction — `fit.continuum_fit` (experimental)

```yaml
fit:
  continuum_fit: none          # none (default) | offset | spline
  continuum_correction:
    knot_spacing_um: 1.0       # spline: knots every 1 µm per sub-band
    prior: continuum           # continuum | noise: what prior_width is relative to
    prior_width: 0.02          # Gaussian prior σ of each coefficient: 2 % of the sub-band's median continuum
    mode: marginalise          # marginalise | profile (in the MCMC; the optimiser always profiles)
```

The continuum is still estimated and subtracted before the fit (`pipeline.prepare`, the IRSQR / median_sg /
aspls … estimators).  With `continuum_fit` a correction c(λ) = Σ β_j B_j(λ) is added *inside* the likelihood:
`offset` is one constant per MRS sub-band, `spline` a clamped cubic B-spline per sub-band with knots every
`knot_spacing_um`.  The β_j are linear parameters with a zero-mean Gaussian prior of width τ_j and are handled
with the same machinery as the emitting areas (`jalebi.linear`): marginalised analytically (default) or
profiled (penalised least squares).  So

* `fit.mcmc.linear: marginalise` / `profile`: the β_j are extra columns of the linear design matrix next to the
  areas — exact, one Cholesky per likelihood call;
* `fit.mcmc.linear: sample`: the areas stay sampled and the β_j are profiled / marginalised at every call;
* the DE optimiser alternates NNLS areas and the penalised continuum solve (`FitProblem.solve_areas`).

Products: `model.csv` gains `continuum_correction`, `continuum_fit.csv` lists the coefficients with their prior
σ, `diagnostics.json["continuum_fit"]` has the largest |correction| / continuum and the rms correction / noise.
`chi2` and `best_fit.json` are computed with the correction profiled at the best fit.

What to expect and what to watch: the prior width and the knot spacing set how much line flux can leak into the
correction.  2 % of the continuum and 1 µm knots cannot follow a Q branch (0.02–0.1 µm) but can absorb the few
per cent, sub-band-wide errors that produce the corner; the synthetic test (`tests/test_continuum_fit.py`)
injects a 1.5 % continuum error and recovers it with `offset`, bringing the fitted T, N and R back to the truth.
On real disks the right setting is an empirical question — see §3.  Not implemented: a correction that depends
on the sampled nonlinear parameters, per-component continua, or priors other than zero-mean Gaussian.

## 2. Corner / pseudo-continuum diagnostic — `fit.corner_check`

```yaml
fit:
  corner_check: {enabled: true, bound_frac: 0.02, min_pinned: 2, smooth_window_um: 0.3, smooth_threshold: 0.5,
                 smooth_molecules: [H2O]}
```

After the fit (on the MCMC median or the optimum) every component is checked for

* **pinned**: ≥ `min_pinned` of its parameters within `bound_frac` of their prior range of a bound (`T@hi`,
  `logN@hi`, `logR@lo` is the hot-water corner);
* **smooth fraction**: the share of the component's convolved flux that survives a running median of
  `smooth_window_um` per sub-band — ~0 for isolated lines, → 1 for a hump or a forest of blended weak lines;
  above `smooth_threshold` the component is mostly pseudo-continuum.  The flag is raised only for the
  molecules in `smooth_molecules` (default `[H2O]`, where the corner occurs); Q-branch molecules (CO2 at 15 µm
  has 0.50 on FZ Tau, C2H2 0.77) put most of their flux into a 0.05–0.1 µm band head that a 0.3 µm running
  median keeps by nature, so their smooth fraction is reported, not flagged.

Flagged components are reported in the log (`warning: H2O_hot: pinned at logN@hi, T@hi, logR@lo; smooth fraction
0.71 …`), in `diagnostics.json["corner"]` (per unit: `pinned`, `n_pinned`, `smooth_fraction`, `tau_max`, `flags`)
and in the population table (`<unit>_smooth_frac`, `<unit>_pinned`, `<unit>_corner_flags`).  The diagnostic does
not change the fit.  A flagged water component means: do not quote its T and N; look at `fit_components.png`
(0.22: one panel per component, the data with the other components subtracted) and try `continuum_fit`.

## 3. Continuum study — `runs/continuum_study.py`

Which continuum treatment reproduces the published values best is decided empirically on the known-parameter
validation set (the 19 runs of `validation_known.yaml`):

```
python runs/continuum_study.py list                           # the built-in settings
python runs/continuum_study.py run validation_known.yaml --workers 17      # all settings, all sources (hours)
python runs/continuum_study.py run validation_known.yaml --settings irsqr_q0.1_k25,irsqr_q0.1_k25+spline1um --only DR_Tau,GW_Lup
python runs/continuum_study.py rank validation_known.yaml     # ranking table from the scores
```

Built-in settings: IRSQR with quantile 0.05 / 0.1 / 0.2 and knot spacing 25 / 50 / 75 pixels, without the
refinement pass, median_sg (two windows), aspls (two stiffnesses), asls, and the 0.22 correction on top of the
survey continuum (`offset`, `spline` with 0.5 and 1 µm knots, a 5 % prior).  Every setting gets its own derived
manifest and output folder under `results/continuum_study/<setting>/`; the runs go through `jalebi_runs.py run`
(so they resume), are scored with `jalebi_runs.py compare` (strict / 0.20-rule / lenient verdicts, N·A scoring)
and ranked by the 0.20-rule pass count, with the tier-A count, the median χ²_red and the number of
corner-flagged components alongside.  `RUN_ME_continuum_study.sh` runs it.  The same tool with
`--settings-file runs/co_bound_settings.yaml` compares the CO temperature prior (1500 / 2000 / 3000 K;
`RUN_ME_co_bound.sh`).

Decision rule suggested for the survey config: take the setting with the highest 0.20-rule pass count whose
corner-flag count is not above the current survey continuum's, and only then set `fit.continuum_fit` in
`runs/survey_0.22.yaml` (it is `none` until this study has run).
