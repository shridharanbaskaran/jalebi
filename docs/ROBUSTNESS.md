# Continuum robustness (0.24, `fit.continuum_robustness`, `jalebi robustness`)

After every fit, jalebi refits the disk under about 20 continua and scores every molecule with the shift-null S
(docs/SHIFT_NULL.md). A molecule that disappears in some variants, or whose parameters move by more than its
posterior error, is flagged. Idea and flagging rule: B. Shridharan (2026-10-10).

## Variants

| # | kind | continuum |
|---|---|---|
| 0 | nominal | the continuum the fit used (the refined one if refinement was accepted) |
| 1–6 | estimator | the spectrum re-estimated with IRSQR q 0.05, q 0.2, knots 15, knots 40, `median_sg` and `sgclip`, each followed by one model-aware refinement pass when the config refines |
| 7–19 | perturbation | variant 0 × (1 + a_b + g_b x) per sub-band b, with x ∈ [−1, 1] across the band, a_b ~ N(0, σ_b), g_b ~ N(0, σ_b/2) |

σ_b is the rms, over the estimator variants, of their median relative offset from variant 0 in band b, clipped
to 0.5–5 %. The test is therefore as large as the real disagreement between continuum methods for that disk and
band.

## Refit and score

- A variant changes only the continuum-subtracted data, y_v = y + c0 − c_v. `FitProblem.with_data` reuses the
  model, the emulator tables, the LSF operator, the noise and the windows.
- **Refit:** a local refit from the best fit (Nelder–Mead on T, log N, log R/NA, ratios and noise scale; areas by
  NNLS; ≤ 3000 evaluations). `refit: de` runs a short DE (40 × 8) per variant instead.
- **Score:** S per unit, with the template bank of the main fit built once (`shift_null.null_cache`), using the
  exact model as for `shift_null.csv`.

## Classes and flags (per unit)

f = share of variants with S ≥ 5.

| class | rule |
|---|---|
| robust | detected at variant 0 and f ≥ 0.9 |
| not detected | not detected at variant 0 and f ≤ 0.1 |
| continuum-dependent | otherwise |

For T, log N and log NA:

- σ_cont = half the 16–84 % range over the variants where the unit is detected;
- σ_post = the MCMC half-width;
- the parameter is flagged `continuum-limited` when σ_cont > σ_post;
- the catalogue gives `<unit>_<p>_syst` = σ_cont, and `continuum_robustness.csv` also gives the total
  √(σ_post² + σ_cont²).

## Files

- `continuum_robustness.csv`: per unit, the class, f, S at variant 0, minimum and median S, and per parameter
  the nominal value, syst, post, total and flag;
- `continuum_variants.csv`: per variant and unit, S, T, log N, log NA and χ²;
- `continuum_variants.npz`: the continua;
- `continuum_robustness.png`;
- `robustness.json`: the checkpoint, written per variant. It is keyed by the stage key, the best-fit θ, the
  fit continuum and the settings, so a rerun continues where it stopped.

## Running it

The settings are run-only, i.e. not part of the resume key. On finished fits:

```bash
jalebi robustness RESULTS/DR_Tau
jalebi robustness RESULTS --survey --workers 30 --backend emulator     # -> RESULTS/continuum_robustness_survey.csv
```

Cost: 13 variants of a 2-molecule synthetic fit take 22 s with the exact model. A survey disk with the emulator
should take a few minutes.

## Checks (tests/test_0240_robustness_accretion.py)

- Synthetic CO2 + HCN, both present: both robust.
- HCN fitted but absent: CO2 robust, HCN not detected.
- A rerun is served from the checkpoint and gives the same numbers.

## The `sgclip` continuum (`continuum.method: sgclip`)

The approach of the JDISC `ctool` (Pontoppidan et al. 2026; Zhang et al. 2026):

- Savitzky–Golay smoothing (`sg_window` 51, `sg_order` 2), then clip pixels more than `clip_sigma` (3) robust
  σ above the baseline (emission mode) or below it (absorption mode); replace them and repeat, up to `clip_iter`
  (10) times.
- `continuum.intervals: [{range: [lo, hi], mode: emission | absorption | auto}]` sets the mode per interval.
  Elsewhere `clip_mode` applies; `auto` picks the mode from the sign of the residual skewness.
- Zhang+2026 do not publish their window or threshold, so these defaults are ours.

## Refinement passes (`continuum.refine_tol`)

`refine_iterations` is the maximum number of model-aware passes. With `refine_tol` > 0 (default 0 = off) the
passes stop once χ² improves by less than that fraction; the last accepted pass is still followed by its DE pass.
New continuum keys left at their defaults stay out of the resume key, so 0.22/0.23 checkpoints remain valid.
