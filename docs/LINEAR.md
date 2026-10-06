# Linear parameters: sampling, profiling or marginalising the emitting areas (jalebi 0.17)

`fit.mcmc.linear: sample | profile | marginalise` (default `sample`, the behaviour of every earlier version).
Code: `src/jalebi/linear.py`; hook: `FitProblem.mcmc(..., linear=...)`; tests: `tests/test_linear.py`.

## Why

The 0.15 blind validation had τ ≈ 330 steps for every parameter in 13–26 dimensions. Every slab component
carries an emitting area that the model depends on *linearly*, and that area is strongly correlated with
log N and T (for optically thick gas F ≈ πR² B(T)/d²; for thin gas F ∝ N·πR²). DuCKLinG (Kaeufer et al.
2024, A&A 687, A209) removes such linear parameters from the Bayesian run and reports ≈ 80× faster fits with
no significant change in the molecular parameters. jalebi already did this for the grid and the optimiser
(NNLS); 0.17 extends it to the MCMC.

## Audit: what is linear in jalebi's model

The model is F(pixel) = Σᵤ aᵤ fᵤ(θ; pixel) + F_screens(θ) with aᵤ = Rᵤ² (au²) and fᵤ the flux of unit u
for R = 1 au (`SlabModel.unit_fluxes`). `jalebi.linear.linear_audit(problem)` prints the table below for any
problem; the fit log lists the removed parameters and every area that stays in the sampler, with the reason.

| Structure | Linear? | Detail |
| --- | --- | --- |
| plain slab, area `logR` | **yes** | F = a · f(T, N, rv, Δv) |
| plain slab, area `logNA` | **yes** | `logNA = log N + log π + 2 log R` mixes N and the area. In the linear modes `logNA` is removed; what is sampled is log N (and T …). The chain column `logNA` is rebuilt per sample from log N and the solved area |
| opacity group (`group:`) | **yes, one area per group** | members add their τ *before* 1 − e^−τ and share T and the area: the group's flux is still a × f(…); the members' log N stay nonlinear |
| tied isotopologue (`tie_to:`) | parent's area only | the child has no area parameter; its flux is added to the parent's column (`SlabModel.fold_tied`), so the parent's area stays linear; `ratio` is nonlinear |
| two-temperature slab (`Tvib`) | **yes** | the source function changes, not the area |
| component `windows:` | **yes** | a pixel mask on f |
| absorber with `covers: all` in front of slabs | slabs: **yes** | the transmission multiplies f, the area still scales it |
| `kind: annuli` | **no** | `logR` is the outer radius of the T(r), N(r) power law (the annuli edges move with it); `logRin`, `q`, `p` are nonlinear too |
| `kind: absorption`, covering fraction `fc` | **no** | transmissions multiply (several screens), and fc ∈ [0, 1]; jalebi's detection can solve fc linearly for one screen, the fit does not |
| area in `fixed:` | — | not a parameter: its flux is subtracted at the fixed value |
| area with a Gaussian `priors:` entry | kept in the sampler | the prior would be ignored by the linear solve |
| noise scale `log_s` | **no** (but see below) | multiplies every σ; sampled |
| per-sub-band continuum offsets | **none exist** | the continuum is estimated and subtracted before the fit (`pipeline.prepare`); the likelihood has no offset terms, so there is nothing to remove. (If offsets are added later, they are linear and would be extra columns of the same solve.) |

## profile (DuCKLinG-style)

At every likelihood call, with the nonlinear parameters θ fixed: whiten by √wᵢ/σᵢ (σ and the pixel weights
`window_weights`), subtract every nonlinear unit's flux, and solve the k areas by NNLS (Lawson & Hanson).
If the solution leaves the area range that the prior bounds of the removed parameter allow
(logR ∈ [−2.5, 1.5] → a ∈ [10⁻⁵, 10³] au²; for `logNA` the range depends on log N), it is redone by
bounded least squares (BVLS, `scipy.optimize.lsq_linear`). So the areas are positive and inside the old bounds.

ln L(θ) = max_a ln L(θ, a) = −½ χ²_min(θ)/s² − ½ Σ wᵢ ln(2π σᵢ² s²).

The NNLS solution does not depend on the noise scale s, which stays a sampled parameter. Note that a profile
likelihood is not a marginal: it drops the volume factor of the area posterior (DuCKLinG accepts this; the
tests show it does not move T or log N beyond 30 K / 0.1 dex here).

Per posterior sample the NNLS areas are stored (emcee blobs) and written into the chain.

## marginalise

With a broad Gaussian prior a ~ N(0, Λ), Λ = diag(λᵤ²) and λᵤ = `linear_prior_scale` (default: the largest
area the prior bounds allow, 10^(2 logR_max) = 10³ au²), the integral over the areas is analytic. With
W = diag(wᵢ/σᵢ²), the columns M = [fᵤ], and the noise scale s:

    G     = Mᵀ W M / s² + Λ⁻¹                 (posterior precision of the areas)
    b     = Mᵀ W y / s²,      â = G⁻¹ b       (conditional mean)
    ln L  = −½ [ (y − M â)ᵀ W (y − M â) / s² + âᵀ Λ⁻¹ â ]
            − ½ ln det G − ½ ln det Λ − ½ Σᵢ wᵢ ln(2π σᵢ² s²)

* The **log-determinant** term −½ ln det G is the Occam factor: it depends on θ (through M) and on s.
* The **noise scale** enters both the quadratic form and G, so it is sampled consistently with the areas
  integrated out (for data-dominated areas ln det G ≈ const − k ln s², i.e. k fewer effective data points).
* `tests/test_linear.py::test_marginal_matches_a_numerical_integral` checks the formula against brute-force
  quadrature (pixel weights ≠ 1, s ≠ 1) to 10⁻⁵ in ln L.
* `linear_prior: log` (**the default**) adds −Σᵤ ln âᵤ: the Laplace approximation of a prior uniform in
  log R, which is what `sample` uses. With it, marginalise and sample ask the same question whenever âᵤ is
  many σ from zero. With `linear_prior: gaussian` (the bare Gaussian marginal) the implicit prior on log N
  of a component whose N is not measured (only N·A) is not flat: the marginal favours small N and large
  areas. On AS 209 it moved CO from log N 17.5 / R 0.11 au to log N 13.3 / R 11 au, and HCN T from 840 K
  to 730 K. That result is why `log` is the default.

**Positivity.** It is *not* imposed in the marginal likelihood: the Gaussian integral runs over all real a.
It is imposed on the area draws: per posterior sample one draw from N(â, G⁻¹) is stored, and a draw with any
aᵤ ≤ 0 is redrawn (rejection, i.e. the conditional Gaussian truncated to a > 0). Every sample whose
conditional mean âᵤ ≤ 0 is counted; the fraction per unit is written to `diagnostics.json`
(`linear_neg_frac`) and a warning is logged above 1 %. Such a component is not detected and its marginal
likelihood includes unphysical negative areas: use `profile` or `sample` for it. The fraction of drawn areas
outside the old prior bounds is reported too (`linear_outside_bounds_frac`).

## What is written

`chain.npz` keeps its layout: `chain` (steps × walkers × *every* free parameter, same `names` as `sample`),
`log_prob`, … The area columns are filled in per sample (profile: NNLS; marginalise: the draw), so
`summary.csv` (with `R_au`, `logNA`, `logNmol`), the corner plots and the QA notebook work unchanged. New keys:
`linear` (mode), `sampled` (False for the filled-in columns), `area_mean` (NNLS areas or conditional means
â per sample, in au²). Old files lack them: read as `sample` / all True. `log_prob` is the profile or
marginal ln P. `diagnostics.json` adds `linear`, `linear_prior`, `linear_units`, `linear_params`, `sampled`,
`tau_sampled` (τ over the sampled parameters only), `linear_neg_frac`, `linear_outside_bounds_frac`.

## With the 0.16 sampler options

* `moves: de`, `init: scaled` act on the reduced vector (the local widths are measured with the areas profiled
  or marginalised).
* `blocks: auto` splits the reduced vector. Every likelihood call still solves *all* areas jointly (the
  other blocks' nonlinear parameters held at the optimum); after merging, each area is taken from the block
  that samples its component. A component with no nonlinear parameter left (e.g. log N and T `fixed`) has an
  empty block, which is dropped: its area is still solved at every call and taken from block 0.
* With no nonlinear parameter at all, the conditional of the areas is drawn directly (no sampler).

## Vectorised likelihood

Not used. A batched evaluation of all walkers (one sparse × dense product for the opacities and the LSF,
Planck functions for all T at once) was prototyped on AS 209: 23.7 → 19.1 ms per walker (1.24×), for
≈ 60 MB of temporaries per unit and a second code path for T_vib, annuli, screens and velocity shifts. The
linear solve itself is < 0.2 ms. emcee's `vectorize=True` also excludes the process pool, which gives more.
