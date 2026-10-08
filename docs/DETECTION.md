# Detection probability with the continuum varied (0.22, experimental)

`jalebi detect-prob CONFIG` — module `jalebi.detection_prob`, options `fit.detection_prob`.

A ΔBIC > 10 at one continuum says a molecule improves the fit *given that continuum*.  Weak species — HCN and
C2H2 Q branches, isotopologues, cold water — sit on the pseudo-continuum, so whether they are "detected" can
turn on continuum choices that are all defensible (the known-set validation: 13CO2 detected in 1/6, HCN with
N and R off in opposite directions).  Two ways to put a number on that.

## Ensemble mode (default)

```
jalebi detect-prob disk.yaml                    # 30 continua, DE 40 x 8 per variant, optimum only
jalebi detect-prob disk.yaml --n 50 --mcmc-steps 300
jalebi detect-prob --survey results/survey_0.22       # merge <disk>/detection_probability.csv into one table
```

1. N plausible continua are drawn (`n_variants`, seed `seed`): the configured estimator with its quantile /
   knot spacing (IRSQR), window / percentile (median_sg) or stiffness (asls, aspls) drawn within the ranges of
   `fit.detection_prob.*_range`, optionally other `methods` mixed in, and a global multiplicative offset
   ~ N(0, `offset_sigma`) (default 1 %).  Variant 0 is the nominal continuum.
2. Each variant is refitted from the disk's best-fit checkpoint (`de_pass2.json` / `de_pass1.json` /
   `best_fit.json`, else the config start): a short DE pass (`de_maxiter` × `de_popsize`) and, with
   `mcmc_nsteps` > 0, a short MCMC (DE moves, scaled init, areas profiled, vectorised) for the parameter spread.
3. Per unit (component / opacity group): the ΔBIC of removing it in every variant (`FitProblem.component_significance`).

`detection_probability.csv`, one row per unit: `n_variants`, `n_detected`, `detection_fraction` (ΔBIC >
`threshold`), `nominal_detected`, `dBIC_nominal / min / median / max`, the 16–84 % half-spread and min / max of
T, log N, log R and log N·A across the variants (a systematic error to quote next to the statistical one), and

| class | rule |
| --- | --- |
| robust | detected in ≥ `robust_frac` (95 %) of the variants |
| continuum-dependent | detected at the nominal continuum or in some variants, but in < 95 % |
| not detected | detected in ≤ `absent_frac` (5 %) of the variants and not at the nominal continuum |

Variants are cached in `detection_prob_variants.json` (per disk; `cache_variants`), so an interrupted or
extended run (`--n` larger) only computes the missing ones.  The synthetic test (`tests/test_detection_prob.py`)
has a strong CO2 slab plus a 1.5 % continuum hump at the HCN Q branch: HCN is "detected" (ΔBIC +62) at the
nominal continuum, CO2 comes out robust (30/30) and HCN continuum-dependent (6/8 variants, T spread 250 K).

Cost: N × (DE pass + MCMC) per disk — minutes per disk with the emulator and no MCMC, an hour with 300 steps.
The ranges of the variations are a judgement call; the probability is only as good as the ensemble.

## Bayesian mode

```yaml
fit:
  continuum_fit: spline                 # the continuum freedom lives in the likelihood
  detection_prob: {mode: bayesian, prior_odds: 1.0}
  dynesty: {nlive: 500, ...}
```

With the continuum correction marginalised (`fit.continuum_fit: spline`), the 0.20 molecule evidence applies:
dynesty gives ln Z of the full model and of the model without each component (`jalebi.nested.evidence_without`,
same pixels and noise), and

    P(present) = 1 / (1 + exp(−(Δ ln Z + ln prior_odds))),   Δ ln Z = ln Z(full) − ln Z(without)

Classes by P: robust ≥ 0.95, not detected ≤ 0.05, otherwise continuum-dependent / uncertain.  Δ ln Z includes
the Occam factor of the removed parameters' prior ranges, so it depends on the bounds (as in 0.20).  Cost: one
nested run per component plus the full model.

## Survey level

`jalebi detect-prob --survey ROOT` concatenates every `ROOT/<disk>/detection_probability.csv` into
`ROOT/detection_probability_survey.csv` (one row per disk × unit) for population statistics: which molecules are
robust in how many disks, which are only ever continuum-dependent.  `RUN_ME_detection_prob.sh` runs the ensemble
mode on the validation disks and builds the survey table.
