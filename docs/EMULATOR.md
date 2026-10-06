# Emulator backend (jalebi 0.18): `fit.model_backend: emulator`

The exact model costs 10–80 ms per evaluation: a sparse opacity basis on a fine ln λ grid (up to ~470 000
points for 4.9–27.5 µm), then the MIRI LSF + pixel integration of the 12 sub-bands. With the emulator, each
slab's flux is read from a precomputed table instead. A whole ln P costs about 0.1–0.5 ms.

```yaml
fit:
  model_backend: emulator          # exact (default) | emulator
  emulator:
    target_sigma: 0.1              # max |emulator − exact| / σ
    target_flux: 0.001             # max integrated-flux error
    method: cubic                  # cubic (4-point Lagrange per axis) | linear
    cache_dir: null                # null = $JALEBI_EMULATOR_DIR, else ~/.jalebi/emulator
  mcmc:
    vectorize: false               # one ln P call for all walkers (emcee vectorize=True; no process pool)
```

```bash
jalebi emulator build config.yaml          # build / verify the tables a fit of this config will use
jalebi emulator list                       # what is in the cache
jalebi fit config.yaml --backend emulator  # or set fit.model_backend
```

The pipeline builds (or loads) the tables itself when the backend is `emulator`. The fit's products
(`best_fit.json`, `model.csv`, the ΔBIC test, every figure) are always written with the exact model; only the
grid, the optimiser and the MCMC use the tables. The web app has a toggle in the sidebar's Display panel
("fast model sliders"). It builds tables for the components on screen in the background, and the sliders use them.

## What is tabulated

For each emulable unit, the pixel flux of a 1-au-radius slab, F(pixel; T, log N), is tabulated on **the
data's own pixels**: after the LSF and pixel integration of every sub-band, at the component's fixed
v_shift and line width, with the component's own line list (`linelist_key`, e.g. `H2O:hitemp` for hot water,
`H2O:hitran` for cold water) and with its `windows` mask. The area multiplies the table (it is linear), so
`linear: profile / marginalise` work unchanged.

* Interpolated quantity: ln(F + ε) with a per-pixel floor ε = 10⁻⁴ σ / a_max. For thin gas F ∝ N, so ln F is
  linear in log N, and the Boltzmann and partition-function dependence is smooth in ln F.
* Axes: ln T and log N, with 4-point Lagrange (cubic) weights per axis on a non-uniform tensor grid (`linear`
  = bilinear).
* Support: only pixels where some node's flux exceeds 10⁻⁴ σ at the largest allowed area are stored. The rest
  are 0, at an error below 10⁻⁴ σ.
* Peak optical depth: tabulated too, as log τ_max, for the τ flags.

### Adaptive grid

Building starts with 9 × 9 nodes over the reachable box: the prior bounds of the free T and log N, or the fixed
value. A tied isotopologue's box is its parent's log N box shifted by the ratio range. The build then repeats:

1. exact rows at every T-interval midpoint, for each log N node and each cell centre; exact values at every
   log N midpoint, for each T node (one opacity product per T: all log N at a given T cost one sparse product);
2. every interval whose check exceeds the tolerance gets its midpoint inserted;

until no check fails (tolerances = `safety` × targets, default 0.05 σ and 0.05 %). `n_validate` random points
are then compared and recorded in the table (`meta.validation`).

### Error measure

At a check point, the emulated and exact fluxes are compared at **the largest area the data allow**:
a = min(a_max, f_ref / max F_exact), with f_ref the brightest continuum-subtracted pixel of the fit and
a_max the largest area of the prior.

* e_σ = max over pixels of |F_emu − F_exact| · a / σ_pixel. Here σ is the noise of the fit (σ/√weight)
  with noise scale 1.
* e_flux = |Σ(F_emu − F_exact)| / max(Σ F_exact, σ_int / a), with σ_int = √(Σ σ²). That is, the error is
  relative to the integrated flux, or to the 1σ noise of the integrated flux when the flux is weaker than
  that noise. Without this floor, a 0.1 % requirement on an undetectable flux could not be met: 100 K CO at
  5 µm has a peak flux of 10⁻²¹ Jy, and its relative flux error stayed at 10⁸ however fine the grid.
  The report gives the fraction of points where the floor applies.

## What falls back to the exact model

The model mixes emulated and exact units freely; `EmulatorSet.exact_units` and the fit log say why each
exact unit stays exact.

| unit | why exact |
| --- | --- |
| opacity group with several members | the opacities add before 1 − e^−τ: one table dimension per member's log N |
| `kind: annuli` | T(r), N(r), R_in, R_out: 5 dimensions |
| `kind: absorption`, or any screen with `covers: all` | the transmission multiplies the emission on the fine grid |
| `Tvib` set | a third dimension |
| free `rv` or `fwhm` | the table is built at one velocity and width |
| a value outside the table (e.g. the sampler steps past a bound, or a slider in the app) | that call only |

Tied isotopologues are emulated with their own table: the same T as the parent and their own log N.

## Cache

Each table is one `.npz` file (float32 ln F on the support pixels), named after the line list and a hash.
The hash covers everything that changes the answer, so a stale table is never loaded:

- the line list: the content of the lines that reach the opacity basis plus a SHA-256 of the source file;
- the pixel wavelengths, the LSF matrix and the fine grid;
- `LSF_VERSION`, R_model / R_scale / R_constant, the distance, v_shift, the line width and thermal width, the
  component's windows and `eup_max`;
- the (T, log N) box and a_max, the noise vector and f_ref used for the certification;
- the targets, method and node limits, the table format and **the jalebi version** (every release rebuilds).

Identical tables are shared, for example two water components with the same release and bounds. A different
continuum (and so a different σ) gives new tables. With `continuum.refine_iterations`, the tables of the first
continuum serve the grid and the optimiser. The candidate continuum is judged with the exact model, and tables
are built again only when the refined continuum is kept.

## Measured (FZ Tau, one core)

Accuracy: `runs/emulator_report.py accuracy`. The FZ Tau pixel grid covers 4.9–5.35 and 9–27.5 µm, 4430 pixels
in 12 sub-bands, with the FZ_Tau_quick continuum. Each component was checked at 2000 random (T, log N) points
drawn uniformly in ln T and log N inside its bounds. Errors are at the largest area the data allow.

| component | line list | T [K] | log N | nodes T × N | pixels | max / p99 err [σ] | max / p99 flux err | flux < noise | build | MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H2O hot | HITEMP | 600–1500 | 14–21 | 33 × 30 | 4422 | 0.026 / 0.021 | 0.023 / 0.020 % | 0 % | 51 s | 17.6 |
| H2O warm | HITEMP | 250–900 | 14–21 | 33 × 30 | 4422 | 0.031 / 0.018 | 0.023 / 0.020 % | 0.2 % | 50 s | 17.6 |
| H2O cold | HITRAN | 100–400 | 14–21 | 33 × 30 | 4324 | 0.025 / 0.018 | 0.041 / 0.029 % | 15 % | 43 s | 17.2 |
| CO | HITEMP | 100–3000 | 14–21 | 35 × 39 | 561 | 0.036 / 0.022 | 0.055 / 0.024 % | 36 % | 92 s | 3.1 |
| CO2 | HITRAN | 100–1500 | 13–21 | 40 × 58 | 1817 | 0.041 / 0.021 | 0.075 / 0.041 % | 5 % | 181 s | 16.9 |
| ¹³CO2 (tied, ratio 70) | HITRAN | 100–1500 | 11.2–19.2 | 34 × 28 | 1817 | 0.044 / 0.025 | 0.049 / 0.038 % | 20 % | 76 s | 7.0 |
| C2H2 | HITRAN | 100–1500 | 13–21 | 40 × 63 | 1602 | 0.036 / 0.024 | 0.091 / 0.039 % | 4 % | 221 s | 16.2 |
| HCN | HITRAN | 100–1500 | 13–21 | 42 × 62 | 1593 | 0.045 / 0.025 | 0.080 / 0.050 % | 6 % | 247 s | 16.6 |
| OH | HITRAN | 100–3000 | 13–21 | 67 × 47 | 2229 | 0.044 / 0.031 | 0.044 / 0.031 % | 14 % | 255 s | 28.1 |

All 9 pass the 0.1σ and 0.1 % targets. The total is 140 MB, built in 20 min on one core. "flux < noise" is the
fraction of points whose integrated flux at that area is below its own 1σ noise; for those, the flux error is
relative to that noise (cold CO, the coldest OH and water). C2H2 (0.091 %) is the closest to the flux target.

Posterior: `runs/emulator_report.py posterior`. The FZ_Tau_quick fit covers 13.45–17.5 µm, 1674 pixels:
H2O hot and warm, CO2 + ¹³CO2, C2H2 and HCN. It used `linear: profile`, `moves: de`, `init: scaled`, 8000 steps
× 44 walkers, from the same optimum. Exact model: 1652 s (τ_max 231); emulator: 112 s (τ_max 200), with its
tables built in 68 s. Every median of T, log N, log R and log N·A agrees within 0.018 dex and 1.0 K (targets
0.05 dex and 20 K). The posterior widths are, for example, 16 K for hot water T and 0.8 dex for C2H2 log N.
Neither chain reaches 50 τ (the C2H2/HCN corners are slow), so this compares two equally long runs.

Time per ln P (`runs/emulator_report.py timing`, the same FZ_Tau_quick problem, one core, after warm-up):

| | single call | `linear: profile`, single | vectorised, 64 walkers (per walker) | profile, vectorised (per walker) |
| --- | --- | --- | --- | --- |
| exact | 7.8 ms | 8.6 ms | 10.7 ms | 7.8 ms |
| emulator | 0.30 ms | 0.34 ms | 0.18 ms | 0.23 ms |

Speed-ups: 26× single, 34–58× vectorised. On the 4.9–27.5 µm grid (H2O hot + CO2 + ¹³CO2, 467 000 fine-grid
points) the exact model costs 78.7 ms against 0.17 ms with the emulator (460×). The interpolation of one unit
costs ~30 µs, and loading 60 MB of tables takes 0.12 s. Vectorising saves the per-walker Python overhead; the
table gather itself is memory-bound (~38 µs per unit and walker).
