# Changelog

All notable changes to JALEBI. The format follows [Keep a Changelog](https://keepachangelog.com/) and the
version numbers follow [Semantic Versioning](https://semver.org/).

## [0.20.0] — 2026-10-06 — dynesty nested sampling, molecule evidences, sampler benchmark

### Added
- `fit.sampler: emcee | dynesty` (default emcee) and `fit.dynesty` (`nlive` 500, `sample` rslice, `bound` multi,
  `dynamic`, `dlogz_init`, `pfrac`, `n_effective`, `maxcall`, `processes`, `seed`, `slices`, `walks`,
  `evidence_without`); `jalebi fit --sampler`. New module `jalebi.nested` (`run_dynesty`, `PriorTransform`,
  `NestedLikelihood`, `NestedResult`, `problem_without`, `evidence_without`).
- Optional extra `nested` (`pip install "jalebi[nested]"` → dynesty ≥ 2.1, tested with 3.1.0); included in `all`.
  - **Likelihood:** the same as emcee's, including `linear: profile` (the NNLS areas come back as dynesty blobs)
    and the emulator backend.
  - **Prior transform:** uniform, or truncated-normal inverse CDF for Gaussian priors. T ordering (hot > warm >
    cold) uses a **sorted transform**: k sorted uniforms are exactly uniform on the ordered region, the prior
    emcee uses. Different bounds within a chain, non-chain orderings and T_vib ≤ T are handled by rejection.
    Stick-breaking was not used because it is not uniform on that region.
  - **Sampling:** dynamic nested sampling with random slice sampling (rslice, 3 + d slices), chosen for 6–15
    correlated, curved dimensions; rwalk is the alternative. dynesty's process pool when `processes > 1`.
  - **Output:** equal-weight posterior samples in the usual chain.npz layout (rows × 32 pseudo-walkers × every
    free parameter; extra keys `sampler`, `logz`, `logzerr`), so the summary, corner plots and QA notebook are
    unchanged. `diagnostics.json` gets `logz`, `logzerr`, `likelihood_calls`, `n_effective`, the method and the
    prior transform used.
- Molecule evidence: `fit.dynesty.evidence_without: [names]` refits without each component on **the same pixels,
  noise and weights** (`problem_without`; with `line_regions` a config rebuild would change the pixels) and writes
  `evidence.csv` with Δln Z ± error next to ΔBIC (Kaeufer et al. 2024). These refits always sample the areas (a
  profiled area has no prior volume). AS 209: Δln Z(HCN) = 73.2 ± 0.9 (ΔBIC/2 = 257 unscaled, 110 rescaled by s²).
- `runs/sampler_benchmark.py` (`list` / `run` / `report` / `evidence`), `runs/RUN_ME_sampler_benchmark.sh`,
  `runs/slurm_sampler_benchmark.sh`, and `docs/SAMPLER_BENCHMARK.md` with the sandbox results and the
  survey recommendation.
- `tests/test_nested.py` (6): the sorted transform is uniform on the ordered simplex and the truncated normal is
  right; rejection; recovery and chain layout with sample and profile; evidence prefers the true model;
  `problem_without` keeps the data; the pipeline option.

### Benchmark (emulator, one core; exact-model cells via RUN_ME)
| | AS 209 emcee | AS 209 dynesty | FZ Tau emcee | FZ Tau dynesty |
| --- | --- | --- | --- | --- |
| `linear: sample` ESS / CPU s | 6.9 (R̂ 1.19, 17 τ) | 18.8 (2.8 M calls, 775 s) | 5.5 (R̂ 1.14) | > 45 min (RUN_ME) |
| `linear: profile` ESS / CPU s | 17.3 (R̂ 1.02, 47 τ, 143 s) | 28.7 (1.3 M calls, 449 s) | 21.4 (R̂ 1.05, 54 τ, 103 s) | > 7 min (RUN_ME) |

Posteriors agree between the samplers, including AS 209's CO thin-gas ridge (27–31 % of the mass in every run).
No second modes. Recommended survey default: emcee (de, scaled, blocks auto) + `linear: profile` +
`model_backend: emulator`, 8000 steps: ~10–15 min per disk, 50–75 core-hours for 300 disks (the exact-model
emcee setup: 350–450). Use dynesty where the evidence matters.

## [0.19.0] — 2026-10-06 — quick errors at the optimum (Laplace approximation)

After the optimiser, Gaussian uncertainties in seconds, without an MCMC. On the synthetic two-molecule test the
Laplace σ agree with a converged emcee run to 1–3 % for every parameter; the whole Hessian takes 146–678 ln P calls
(0.1 s with the emulator).

### Added
- `jalebi.fit.laplace(problem, theta)` (module `jalebi.laplace`):
  - the Hessian of −ln P by central finite differences, with steps from `FitProblem.local_widths` and one
    Richardson extrapolation (or numdifftools with `method="numdifftools"`);
  - near a prior bound the stencil moves inside the prior (flag `one-sided`);
  - eigen-analysis in prior-span units: directions the data constrain less than the uniform prior (`flat`) or
    with negative curvature (`saddle`) are named by the parameters they load on, and regularised to the prior
    width;
  - per-parameter flags `flat` / `saddle` (with the direction, e.g. CO T on a ridge), `unconstrained`
    (σ > 25 % of the prior span), `edge` (within 2σ of a bound) and `one-sided`;
  - it returns the covariance, correlation matrix, 1σ errors and condition number, plus derived R, log N·A and
    N_mol from 4000 Gaussian draws truncated to the prior;
  - `linear="profile"` (0.17) takes the Hessian over the nonlinear parameters only, with the areas from the
    conditional Gaussian of the linear solve per draw. Whatever model backend is attached (0.18) is used.
- Pipeline: `fit.laplace: true` / `jalebi fit --laplace` write `laplace.json`, `laplace_summary.csv`,
  `laplace_corner.png` (1/2σ error ellipses, with the MCMC contours drawn on top when a chain exists) and
  `laplace_correlation.png`. `run_laplace_stage`, `save_laplace`.
- Web app:
  - Fit tab: a **Quick errors (Laplace)** button, enabled once an optimiser result exists. It runs in a
    background thread, which also draws the figures; the page is updated with a next-tick document callback,
    and progress goes to the activity log.
  - Results tab: a new "Quick errors" section with a value ± σ table and flags, the corner-style ellipse plot
    (MCMC contours on top when a chain exists) and the correlation heatmap.
- `plots.plot_laplace_corner`, `plots.plot_laplace_correlation`.
- `tests/test_laplace.py` (5): σ within 30 % of a converged emcee run (measured 1–3 %); flags for a component
  at its log N bound; JSON and plots; `jalebi fit --laplace`; the app button running in a thread.
  `runs/ui_check_laplace.py`: headless Playwright + Chromium check of the button, table and figures, with and
  without a chain.

## [0.18.0] — 2026-10-06 — a precomputed emulator of the slab model

The exact model costs 8–80 ms per evaluation: an opacity basis on a fine ln λ grid, then the LSF and pixel
integration of 12 sub-bands. `fit.model_backend: emulator` reads each slab's flux from a table built once on the
data's own pixels. A ln P then costs 0.2–0.3 ms, and the posterior is unchanged.

FZ Tau, one core (`runs/emulator_report.py`; details in docs/EMULATOR.md):

| | exact | emulator |
| --- | --- | --- |
| ln P, FZ_Tau_quick fit (1674 px, 6 units) | 7.8 ms | 0.30 ms (0.18 ms per walker vectorised over 64) |
| ln P, 4.9–27.5 µm (H2O hot + CO2 + ¹³CO2) | 78.7 ms | 0.17 ms |
| MCMC, `linear: profile`, 8000 steps × 44 walkers | 1652 s | 112 s (+ 68 s to build the 60 MB of tables) |
| posterior medians | — | within 0.018 dex and 1.0 K of exact (targets 0.05 dex, 20 K) |
| accuracy, 9 components × 2000 random (T, log N), 4.9–27.5 µm | — | max 0.045σ (p99 ≤ 0.031σ), max 0.091 % in integrated flux |

### Added
- `fit.model_backend: exact | emulator` (default `exact`) and `fit.emulator` (`target_sigma` 0.1, `target_flux`
  1e-3, `safety` 0.5, `method` cubic | linear, `n_start`, `max_nodes`, `n_validate`, `cache_dir`, `rebuild`).
  New module `jalebi.emulator`.
- Tables: ln(F + ε) of a 1-au slab per pixel, after the LSF and pixel integration of every sub-band, at the
  component's fixed v_shift and width, with its own line list (e.g. `H2O:hitemp` for hot water, `H2O:hitran` for
  cold water). They use 4-point Lagrange (cubic) interpolation in (ln T, log N) on an adaptive tensor grid:
  interval midpoints and cell centres are checked against the exact model, and nodes are added until every check
  is below half the target. The error is measured at the largest area the data allow, in σ of each pixel, and in
  integrated flux (relative to the flux, or to its 1σ noise when the flux is weaker than that).
- Emulated: plain slabs and tied isotopologues (each tied unit has its own table over its shifted log N range).
  Exact: opacity groups with several members, annuli, absorption screens, any `covers: all` screen, T_vib
  components, free rv / fwhm, and any call outside a table. One model mixes both; the reasons are logged.
- Cache: `.npz` files in `$JALEBI_EMULATOR_DIR` or `~/.jalebi/emulator`. The key covers the line-list content and
  file SHA-256, the pixel grid, the LSF matrix and version, the fine grid, R(λ), the distance, v_shift, width,
  windows, the (T, log N) box, the certification noise and reference flux, the targets and the jalebi version.
  Identical tables are shared.
- `jalebi emulator build CONFIG` (prepares the spectrum exactly as `jalebi fit` does) and `jalebi emulator list`;
  `jalebi fit --backend`. The pipeline builds or loads the tables when the backend is `emulator`. All written
  products (best fit, model.csv, ΔBIC test, plots) use the exact model (`SlabModel.exact()`).
- `fit.mcmc.vectorize`: one ln P call for all walkers (`FitProblem.log_prob_many`,
  `LinearProblem.log_prob_blob_many`, emcee `vectorize=True`), also with `blocks: auto`.
- Web app: the "fast model sliders (emulator tables)" checkbox in the sidebar's Display panel builds tables for
  the components on screen in the background; the live model then uses them. The layout is otherwise unchanged.
- `tests/test_emulator.py` (10 tests, among them 2000 random points per unit for CO2, ¹³CO2 and HCN on FZ Tau; the full
  9-component FZ Tau campaign runs with `JALEBI_EMULATOR_FULL=1`) and
  `runs/emulator_report.py` (accuracy | posterior | timing).

## [0.17.0] — 2026-10-06 — the emitting areas can leave the MCMC

The slab flux is linear in the emitting area R², and that area is strongly correlated with log N and T.
DuCKLinG (Kaeufer et al. 2024, A&A 687, A209) removes such linear parameters from the Bayesian run. jalebi already
profiled them in the grid and the optimiser; now the MCMC can do the same.

| AS 209, 2000 steps, `de` / `scaled` / `blocks: auto` | dims | max τ | median τ | max R̂ | wall (s) | ms / independent sample |
| --- | --- | --- | --- | --- | --- | --- |
| `linear: sample` (0.16) | 13 | 125 | 72 | 1.31 | 1268 | 2476 |
| `linear: profile` | 9 | 94 | 35 | 1.06 | 1184 | 1745 |
| `linear: marginalise` (prior `log`) | 9 | 83 | 33 | 1.06 | 1210 | 1570 |
| `linear: marginalise` (prior `gaussian`) | 9 | 90 | 28 | 1.09 | 1067 | 1507 |
| stretch / ball / joint: `sample` → `profile` | 13 → 9 | 192 → 142 | 138 → 107 | 1.75 → 1.46 | 1469 → 946 | 2716 → 1860 |

Water and HCN mix 2.5–4× faster; CO (its own log N–T degeneracy) stays the slowest parameter. The medians agree
with `sample` for every parameter except CO (≤ 0.02 dex, ≤ 9 K; CO, the least converged, 0.12–0.14 dex and 31–38 K),
except for the bare Gaussian prior (see Changed).

### Added
- `fit.mcmc.linear`: `sample` (default, unchanged), `profile` (areas solved by NNLS at every likelihood call, bounded
  to the old logR / logNA range by BVLS when needed) or `marginalise` (analytic Gaussian marginal over the areas:
  log-determinant Occam term, noise scale s inside it). `fit.mcmc.linear_prior` (`log` | `gaussian`) and
  `linear_prior_scale` (σ of the Gaussian on R², default R_max²). New module `jalebi.linear`
  (`LinearProblem`, `linear_audit`, `run_linear_mcmc`); hook in `FitProblem.mcmc(linear=...)`.
- Linear audit (docs/LINEAR.md, and in the fit log): slab areas are linear, with one area per opacity group;
  tied isotopologues are folded into the parent's column. Annuli radii, covering fractions, ratios, and areas
  that are fixed or carry a Gaussian prior stay in the sampler. There are no continuum offsets in the likelihood.
  With `area_param: logNA`, `logNA` is removed and log N is sampled; `logNA` is rebuilt per sample.
- Area posteriors: profile stores the NNLS areas per sample, and marginalise stores one draw from the conditional
  Gaussian truncated to a > 0 (emcee blobs). Both are written into the chain, so `summary.csv` (R_au, logNA,
  logNmol), corner plots and the QA notebook work unchanged.
- `chain.npz` keeps its layout and gains `linear`, `sampled` and `area_mean`; old files read as `sample`.
  `diagnostics.json` adds `linear`, `linear_prior`, `linear_units`, `linear_params`, `sampled`, `tau_sampled`,
  `linear_neg_frac` (samples with a conditional mean area ≤ 0, warned above 1 %) and `linear_outside_bounds_frac`.
- Works with `moves: de`, `init: scaled`, `blocks: auto`, process pools and HDF5 checkpoints. A block left with no
  nonlinear parameter is dropped, and its area is still solved and taken from block 0.
- `runs/bench_linear.py`: rebuilds a finished disk from config.yaml + model.csv + best_fit.json and times the modes.
- `tests/test_linear.py` (13 tests), including: the analytic marginal against quadrature; ln L(profile) equals the
  full ln L with ties, groups, annuli and screens; two-molecule recovery within 1σ; agreement with `sample` to
  ≤ 30 K and ≤ 0.1 dex on converged runs.

### Changed
- `linear_prior` defaults to `log`, which adds −ln â, the Jacobian of the uniform-in-log-R prior that `sample`
  uses. The bare Gaussian marginal (`gaussian`) pulled AS 209's CO, whose N is barely measured, to log N 13.3 and
  R 11 au.
- `runs/validation_blind.yaml`, `runs/survey_300_autodetect.yaml` (still in sync) and `validation_known.yaml`:
  the option is present, commented out. The configs keep `sample`, so the planned validation_blind_v2 run is
  unchanged.
- QA notebook: the convergence table shows `linear` and `n_sampled`.

### Fixed
- Continuum with pybaselines 1.1 under NumPy 2 (`irsqr`/`asls`/`aspls` failed with "Unable to avoid copy while
  creating an array as requested": pybaselines 1.1 calls `np.array(x, dtype=float, copy=False)` on the integer
  `x_data` jalebi passed). jalebi now passes native float64 arrays (also in `jalebi.cube`), and the minimum is
  pybaselines ≥ 1.2 (pyproject, requirements.txt, environment.yml, `jalebi doctor`), because 1.1 also lacks
  `aspls(asymmetric_coef=...)`.

### Not done
- A vectorised likelihood (`emcee vectorize=True`): a batched model evaluation gave only 1.24× (23.7 → 19.1 ms per
  walker) on AS 209, at ≈ 60 MB per unit and with a second code path. The process pool gains more.

## [0.16.0] — 2026-10-06 — a sampler that converges on multi-molecule fits

The blind validation run with 0.15 (17 disks, 5000 steps) converged nowhere: the integrated autocorrelation
time was τ ≈ 330 steps for *every* parameter (noise scale included), so the chains were only 10–15 τ long and
split-R̂ was 1.45–2.0. A τ that is the same for all parameters is the signature of emcee's stretch move in
13–26 dimensions, not of one degeneracy. On AS 209 (13 parameters, 2000 steps) the new options give:

| sampler | max τ | median τ | acceptance |
| --- | --- | --- | --- |
| 0.15: stretch move, 1 % ball (5000-step run) | 427 | 290 | 0.21 |
| `moves: de`, `init: scaled` | 151 | 113 | 0.08 |
| `moves: de`, `de_gamma: 0.5` | 143 | 101 | 0.26 |
| `moves: de+stretch` | 142 | 105 | 0.12 |
| `moves: de`, `init: scaled`, `blocks: auto` | 131 | 61 | 0.10 |

Posterior medians agree with the 0.15 run to ≤ 0.1 dex and ≤ 30 K. The slowest parameters left are CO's, whose
temperature sat at the 1500 K prior bound; hence `bounds_by_molecule`.

### Added
- `fit.mcmc.moves`: `stretch` (default, unchanged), `de` (80 % `DEMove` + 20 % `DESnookerMove`, ter Braak &
  Vrugt 2008) or `de+stretch` (60/20/20); `fit.mcmc.de_gamma` scales the DE step (2.38/√(2 ndim) × this).
- `fit.mcmc.init`: `ball` (default, unchanged: 1 % of each prior range) or `scaled`: walkers start at the local
  posterior width of each parameter, measured from the curvature of ln P around the optimum
  (`FitProblem.local_widths`, ~2 × 12 × ndim likelihood calls).
- `fit.mcmc.blocks`: `joint` (default, unchanged) or `auto`: components that share no fitted pixel (flux above
  0.02 σ at the optimum; ties, opacity groups and `ordering` also link components) are sampled by separate
  samplers with the other blocks held at the optimum, then merged into one chain (`FitProblem.independent_blocks`).
  The likelihood is a sum over blocks, so this is exact except for the shared noise scale, which is sampled with
  the block that has the most pixels. Typically CO + ro-vibrational water (4.9–8 µm) and the 12–27.5 µm
  components split; C₂H₂ (7.5 µm ν4+ν5 band) links them again. 9 of the 17 validation disks split.
- `fit.bounds_by_molecule`, e.g. `{CO: {T: [100, 3000]}}`: prior bounds for every component of a molecule,
  including those written by the auto-detection (a component's own `bounds` still win). CO HITEMP pruned at
  1500 K keeps 99.7 % of the 3000 K opacity.
- `diagnostics.json` records `moves`, `de_gamma`, `init`, the parameter `blocks`, and per-block acceptance and
  walker numbers. The fit log names the blocks.
- `tests/test_sampler.py` (6 tests).

### Changed
- `runs/validation_blind.yaml` and `runs/survey_300_autodetect.yaml` (in sync): `moves: de`, `init: scaled`,
  `blocks: auto`, `nsteps: 6000`, CO / ¹³CO T up to 3000 K. The blind validation now writes to
  `results/validation_blind_v2/`, so the 0.15 run stays for comparison.

## [0.15.0] — 2026-10-05 — the terminal shows what the web app is doing

### Added
- **Activity log** (`jalebi.activity`): `jalebi serve` prints everything the app does in its terminal — what the user
  clicked, typed, chose or uploaded in the browser (widgets, tabs, buttons, plot taps, draw tools, table edits; amber
  `👤` lines), every callback of the app modules with its duration, the steps of opening a source (headers, x1d, each
  cube read with its size and time, cache hits, the source position), spectra loaded and their pixel counts, continuum
  and masks, model rebuilds and χ² (≤ 1 line/s while dragging), line lists loaded, grid / optimiser / MCMC stages, molecule
  detection, batch targets, cube maps, PV cuts, ratio maps, rotation-diagram measurement and fits, results written
  (and the files in the folder), the browser notifications, the status texts of each module, warnings and errors with
  their traceback (also from worker threads), and browser tabs connecting and closing.
- Progress bars in the terminal for the long jobs: reading the cubes, each grid, the optimiser, MCMC (acceptance, ⟨lnP⟩),
  molecule detection, batch, rotation-diagram MCMC (tqdm bars on a terminal, log lines kept above them; a line every
  10 % when the output is redirected); jobs without a progress report print "still running" every 10 s.
- Each line carries the browser session (`s1`, `s2` …; `s1·bg` for a background job started by that tab) and is
  indented under the call it belongs to.
- `jalebi serve --log-level trace|debug|info|warning|error` (default `debug`), `--log-file FILE`, `--quiet`; the server's
  start-up facts (version, folders, caches, threads) are printed first. `JALEBI_LOG_LEVEL` / `JALEBI_LOG_FILE` or
  `jalebi.activity.configure()` switch it on for `pn.serve(make_app)` and notebooks.
- `tests/test_activity.py` (13 tests).

### Changed
- In the served app, the fit's messages (`RunResult.say`) and the grid / MCMC progress of batch runs go through the
  activity log instead of `print` / separate tqdm bars. CLI and library use are unchanged (nothing is printed by the
  new code unless the log is switched on).

## [0.14.0] — 2026-10-05 — open a source once: the Source page and `jalebi.source`

### Added
- **Source page** (web app, new first module `source`): scan a data root, open a target folder once (headers, x1d, all
  s3d cubes read in parallel threads, source position), see the sub-band coverage, header facts (programme,
  observation, jwst `CAL_VER`, `CRDS_CTX`) and previews, then launch **LTE slab fit**, **Cube maps** or **Rotation
  diagram**. The app now starts here (`--module lte|cube|rotdiag`, an LTE tab other than Data, or a config with a
  target still open the old way). `jalebi serve --source DIR` opens a source on start; the header shows it in every module.
- **`jalebi.source`**: `Source` (x1d, `aperture_spectrum`, `region_spectrum`, `spectrum("x1d"|"s3d"|"auto")`, `cubes`,
  `position`, `image`, `coverage`, `summary`), all memoised and returned as copies; `open_source` keeps the last
  `JALEBI_SOURCE_CACHE` (3) sources of the process, so the web app's sessions (browser refreshes, tabs) share them;
  `scan_sources`, `get_cubeset`. Per-source settings (distance, RV, RA/Dec, name) in `jalebi_source.yaml` next to the
  data or `~/.jalebi/sources/`.
- Modules receive the open source (`use_source(src, distance_pc, rv_kms)`): the Cube module uses its cubes, the
  Rotation diagram a 1″ circle on them (or the x1d), the LTE fit its x1d or an aperture on the cubes in memory; the
  LTE *Data* tab moves the old target picker into an *Another spectrum* card.
- `CubeSet.preload(workers, progress, stop)`, thread-safe `CubeSet.load` (concurrent requests read a cube once),
  `cache_size`, `estimated_bytes`, `loaded`; cubes above `JALEBI_CUBE_CACHE_MB` (6000) are read on demand.
- `jalebi.data.extract_cube` / `aperture_photometry` / `peak_position`: the photometry of `extract_s3d` on a cube in
  memory (`extract_s3d` now calls the same code; outputs identical to 0.13.0 on the bundled cubes).
- CLI `jalebi source list | info [--preload] | set | spectrum`. `rotdiag.pipeline.load_cube_spectrum(cfg, source=)`
  and `load_input_spectrum(cfg, source=)`.
- `docs/SOURCE.md`; `tests/test_source.py` (13 tests).

### Changed
- The LTE Data tab's cube preview, *header target position*, *brightest pixel* and click-to-set RA/Dec use the cached
  cube instead of re-reading the FITS file on every change; the Cube module opens cubes through the source cache.

## [0.13.0] — 2026-10-01 — the 5–8 µm water band: T_vib, per-component windows, curated line regions

### Added
- **Vibrational temperature** `Tvib` per component (`Component.Tvib`, YAML `Tvib:`, app checkbox *vibrational
  temperature*): two-temperature populations (Boltzmann at T within a vibrational state, at T_vib between states),
  per-line source function, opacity-weighted source function on the fine grid; vibrational energies read from the
  HITRAN/HITEMP global quanta (`LineList.vib_energies`, `vib_arrays`; water band origins recovered to < 0.1 cm⁻¹).
  `Tvib` is a free parameter when set (bounds 100–1500 K, prior `fit.tvib_below_trot: true`); annuli keep T_vib/T
  constant; tied isotopologues inherit it. `Tvib = T` reproduces LTE to 1e-5. Motivation: LTE fits of the rotational
  lines over-predict the 5–8 µm ν₂ band by 3–6× (Banzatti+2025 Fig. 7, Pontoppidan+2024 §4.3) — `docs/ROVIB_WATER.md`.
- **Default water split** `fit.water_split_um: 9.5`: when the fit windows reach below 9.5 µm, H₂O slabs without their
  own `windows` or `Tvib` emit only beyond it and "*rovib*" components only below it (tied isotopologues follow their
  parent), in the pipeline and in the app; the log reports the windows applied and warns when nothing covers the band.
- **Per-component wavelength windows** `windows: [[lo, hi], ...]`: a component contributes only there, so the
  separate-region fits of Gasman+2023 / Temmink+2024 (a ro-vibrational H₂O slab at 5–9 µm, rotational slabs at
  12–27 µm, CO at 4.9–5.7 µm) run as one simultaneous fit; the grid stage grids such a component on its own windows.
- **Curated line regions** `fit.line_regions: [H2O_v0-0 | H2O_v1-0 | H2O_v1-1 | H2O_general | general | file.csv]`,
  `fit.region_pad_um`, `fit.region_weight_beyond: [20, 5]` (new module `jalebi.regions`): fit only narrow regions
  around the isolated lines of the Banzatti+2025 lists, Temmink+2025-style weights beyond 20 µm, instead of every pixel.
  On CI Tau this recovers the published two-temperature solution (827/386 K vs Banzatti+2023 840 K, Romero-Mirza+2024
  903/477 K) where the every-pixel fit gave 1281/605 K.
- `fit.region_other_molecules: features | default | none` (+ `region_feature_pad_um`): with `line_regions`, the
  non-water components keep their Q-branch / band-head ranges (or default windows); `ComponentConfig.priors`
  (Gaussian `{T: [mu, sigma]}` on any free parameter). Detection judges hot/warm water on 12–27.5 µm so the
  17.5–27.5 µm lines are fitted even without a cold slab. `runs/validation_blind.yaml` and
  `runs/survey_300_autodetect.yaml` fit the isolated-line regions (blind CI Tau: 934/525 K instead of 1500/295 K).
- Example configs `examples/configs/CI_Tau_rovib_tvib.yaml`, `CI_Tau_rovib_separate.yaml`, `CI_Tau_lines_B23.yaml`;
  validation manifest runs `CI_Tau_B23` (line regions) added, CI Tau / AS 209 published values corrected (RM24 Table).
- Continuum method `aspls` for the 1D slab fit (pybaselines adaptive smoothness-penalised LS, `aspls_lam` 5e6,
  `aspls_alpha`), the same estimator the cube module and the group's `cube_maps.py` use; in the app's method list.
- Tests `tests/test_tvib_and_regions.py` (8 tests) + `aspls` in the continuum-method test.

### Fixed
- App: slider titles, checkbox text and the component-card subtitle rendered black on the dark panel. The theme's CSS
  variables were defined per component `:host` only and did not reach the Bokeh widgets' shadow roots; they are now set on
  `:root`, every widget-label rule has a literal fallback, and labels use a lighter grey (`PAL.label`).

## [0.12.1] — 2026-10-01 — rotation diagrams from the cubes

### Added / fixed (jalebi.rotdiag)
- `spectrum.source: s3d` (default for a cube folder): the cubes summed over a region (circle / annulus / ellipse / polygon /
  whole field, offsets from the source), the region's solid angle as the aperture; `jalebi rotdiag fit --source s3d
  --region --dx --dy --radius-arcsec`; the app's *s3d cubes (region)* input; example config `rotdiag_H2_HV_Tau_C.yaml`.
  The pipeline x1d (background annulus) turns extended H₂ lines negative — kept as an option only.
- Measurement: pixels near modelled lines are never clipped (bright lines lost their cores); strong lines get their own
  velocity and width (sub-band offsets, lines narrower than the R(λ) law); unresolvable blends with other species flagged
  and left out of the fit; H₂ preset selects v=0–0 and v=1–1 with a per-molecule strength floor; OPR bound 0.1–6.
- Extinction curves KP5 (Pontoppidan et al. 2024; now the default), KP5_benchmark, HD23 (Hensley & Draine 2023) and
  McClure (2009; A_K < 1 and > 1) bundled next to G23/G21/CT06/F11.
- Verified on HV Tau C (full MINDS cubes, 1″): S(1)–S(8) and v=1–1 S(3)–S(9) detected; T 723/2150 K, OPR 3.3, A_V 6.2 (KP5),
  10.7 (G23), 16.3 (McClure09).

## [0.12.0] — 2026-09-30 — app modules and rotation diagrams

### Added
- **Web-app modules** (`jalebi.modules`): the app is now a shell around independent modules, switched from the
  header — **LTE slab fit** (Data · Continuum · Model · Fit · Results · Batch), **Cube maps** (was the *Cube* tab)
  and **Rotation diagram**. Modules other than the LTE fit are built the first time they are opened. New modules
  plug in with `register_module(ModuleSpec(key, label, icon, description, "package.module:Class"))`.
  `jalebi serve --module lte|cube|rotdiag` (`--tab cube` still opens the Cube module), `--rotdiag-config FILE`;
  `JalebiApp.switch_module`, `JalebiApp.send_spectrum`, `make_app(start_module=...)`. The Cube module can send a
  region spectrum to the rotation diagram (*Send to rotation diagram*; the region's solid angle becomes the aperture).
- **`jalebi.rotdiag`**: rotation (population) diagrams. See `docs/ROTDIAG.md`.
  - Molecules H₂, CO, ¹³CO, OH, H₂O (+ any molecule with a line list) with per-molecule presets (default bands,
    ranking temperature, spin species, curated lines).
  - Line selection inside the spectrum coverage, thin-LTE ranking, blends as single features (flux = sum of the
    members), neighbours, other-band blends and contaminants flagged and fitted jointly.
  - Flux measurement: pixel-integrated Gaussians of the MRS resolution at one velocity + polynomial baseline by
    linear least squares (exact errors, iterative clipping of unmodelled lines); velocity and width scale (per MRS
    channel) from the profile likelihood; Gaussian (free) and window-integration methods.
  - Physics: one temperature, two temperatures (T₁ < T₂) or a power law dN ∝ T^−b dT (Neufeld & Yuan 2008);
    ortho-to-para ratio thermal / free with each spin species in LTE (exact) / the ln(OPR/3) offset; extinction A_V
    with G23 (R_V 3.1, 5.5), G21, CT06, F11 or a user CSV (e.g. KP5); curve-of-growth optical depth of a Gaussian
    slab; normalisation to the number of molecules, an emitting radius, an aperture or intensity.
  - Fit in flux space (χ² with a relative systematic, 10 % by default): multi-start bounded least squares with the
    columns solved linearly at each start, then emcee (vectorised); derived column, number of molecules, mass,
    equivalent radius, A_K, LTE OPR, hot fraction, mean T, τ_max, line luminosity; BIC/AIC comparison of the models.
  - Inputs: the s3d cubes summed over a region (`spectrum.source: s3d`, the default for a cube folder; circle /
    annulus / ellipse / polygon / whole field as offsets from the source; the region's solid angle sets the aperture),
    x1d, CSV incl. `.csv.gz`, FITS, a cube region from the Cube module, or a flux table (label or wavelength;
    `W m-2`, `erg s-1 cm-2`, with a scale factor). Verified on HV Tau C: from the cubes all S(1)–S(8) and v=1–1
    S(3)–S(9) lines are detected (S/N 90–380), while the x1d gives S(2) and S(4) in absorption (background annulus).
  - Measurement robustness: pixels near modelled lines are never clipped; strong lines get their own velocity and
    width (sub-band offsets); unresolvable blends with other species are flagged and left out of the fit.
  - Paper-style figures (`plots.plot_excitation`, `plot_model_panels`): log₁₀/ln axes, observed and de-reddened
    points per vibrational band, labelled lines, warm/hot components, parameter box; `fit.also: [powerlaw]` fits the
    power law next to the main model (+ MCMC) and draws the two side by side (also in the app's *Paper figure* tab).
  - Outputs: `results/{target}/rotdiag/{molecule}/` with lines, members, diagram, fit, parameters, derived quantities,
    model comparison, chain, figures (diagram, corner, line fits) and the config.
  - CLI `jalebi rotdiag molecules|curves|lines|fit|run|init|demo`; YAML config (`RotDiagConfig`, examples h2,
    h2_fluxes, co, oh, h2o); web-app module with spectrum/line/diagram/posterior views and the equivalent command.
- **H₂ line data**: the full line list of Roueff et al. (2019) as the bundled release `roueff2019` (S(0) at 28.2 µm
  included; the HITRAN list stopped at 26 µm) and its 302 levels (`data_files/H2_levels_Roueff2019.csv`), giving
  exact ortho/para partition sums.
- `data_files/extinction_curves.csv` (from `dust_extinction`; not a dependency).
- Example data `synthetic/rotdiag_H2_synthetic.csv.gz` (+ truth) and `synthetic/rotdiag_H2_fluxes.csv`;
  `examples/12_rotation_diagram.py`, `examples/configs/rotdiag_*.yaml`.
- `load_csv` reads gzipped CSV files.
- `tests/test_rotdiag.py` (17 tests).

### Changed
- `JalebiApp.TAB_NAMES` lists the six LTE-fit tabs only; `JalebiApp.cube_ws` is built on first use; the header
  status chips are shown in the LTE module only; the sidebar's configuration panel is titled *LTE-fit configuration*.

### Validation
- H₂ level sums = HITRAN Q(T) to < 0.2 % (100–1000 K); injection–recovery of H₂ diagrams (T, N, A_V, OPR; two
  temperatures; power law) within 1.6σ; LTE slab spectra of jalebi.model: thin H₂ (log N 22.01 ± 0.007, T 798 ± 3.5 for
  22.0, 800 K) and thick CO (18.504 ± 0.011, 1098 ± 3 K for 18.5, 1100 K; a thin fit gives 1740 K); against pdrtpy
  3.0.1: identical temperatures and OPR, columns within 3 %, and pdrtpy's A_V is ln 10 too large (its extinction term).

## [0.11.0] — 2026-09-30 — LTE gas absorption (released on main together with 0.12.0)

### Added
- **Absorption screens** (`kind: absorption`): a foreground slab seen against the continuum,
  $F = F_c\,[1 - f_c\,(1 - e^{-\tau})]$ (Li, Boogert & Tielens 2024; the `spec_abs` of the group's
  `slabby.py`). Same opacity, fine grid, line lists and LSF as the emission; free parameters log N, T, v
  (always free), the covering fraction `fc`, and Δv with `fit_fwhm`. Several screens multiply; `covers: all`
  also attenuates the emission components (the `two_slabs_spec` geometry). The model stays linear in the
  emitting areas, so NNLS, the (log N, T) grid, DE and emcee work unchanged. See README § The physics 6 and
  `docs/ABSORPTION.md`.
- **Detection in absorption**: `detect_molecules(mode="both"|"emission"|"absorption")`,
  `fit.detect.mode` (default `both`), `jalebi detect --mode`. One screen template per candidate joins
  the linear solve with the covering fraction as a bounded (0–1) coefficient
  (`SlabModel.solve_linear(solve_fc=True)`, BVLS; `SlabModel.screen_columns`); candidates `<mol>_abs` are
  judged like emission candidates (4-parameter BIC penalty) but only on pixels below the continuum, in
  three passes (emission alone; all templates with the screens judged; all emission with the detected
  screens), and the screen temperature is picked afterwards among 50–500 K. Detected screens are
  suggested as `kind: absorption` components. The detection table has `kind` and `fc` columns (CLI table
  and app show `log R / f_c`). FZ Tau: no false screens, emission unchanged; the bundled synthetic
  absorber: the CO₂ screen and the CO₂ emission behind it.
- `Component.fwhm_thermal`: add the thermal width at T in quadrature to `fwhm` (any component kind).
- `SlabModel(continuum=...)` / `build_model(continuum=...)` and `SlabModel.set_continuum`; the fit
  problem, the display model of the app and the full-spectrum plot pass the continuum through.
- Continuum method `given`: keep the continuum loaded with the spectrum. `load_csv` reads a `continuum`,
  `baseline`, `cont` or `base_fluxes` column as the continuum.
- `jalebi model --kind absorption --fc --rv --continuum`: writes the absorbed continuum and the transmission.
- Web app: `kind: absorption` in the component card, with an `f_c` slider, an *absorbs* selector
  (continuum only / continuum + emission) and the thermal-width checkbox; log R is hidden for a screen;
  the display model follows continuum changes.
- `examples/11_absorption_fit.py` + `configs/absorption_synthetic.yaml`: a CO₂ screen (v = −40 km/s,
  f_c = 0.6) in front of hot CO₂ emission, recovered within 1σ by the normal pipeline.
- `tests/test_absorption.py` (13 tests): transmission formula, f_c scaling, velocity direction, screens
  multiplying, `covers: all`, exact area solve with a screen, thermal width, free parameters and an
  injection–recovery with DE, config validation, CSV baseline + `given`, synthetic spectra, CLI.

### Changed
- `FitProblem.area_units()` no longer lists screens; `all_units()` lists every independent unit and is what
  `component_significance` and the grid use. `ABSORPTION_BOUNDS` (T 20–1500 K, v ±200 km/s, log N 13–22)
  are the default bounds of a screen; per-component `bounds:` still override them.
- `plots.plot_fit` draws absorption units dashed.

## [0.10.1] — 2026-09-29 — line maps and velocity maps from IFU cubes

### Added
- **`jalebi.cube`**: line maps and velocity maps from JWST `s3d` cubes, for the extended line emission
  (jets, winds, H₂, [Ne II], [Fe II], CO) of disks that are unresolved in the continuum. See `docs/CUBE.md`.
  - `CubeSet` reads every sub-band cube of a target (headers only until data are needed; `WAVE-TAB` and
    `.fits.gz` supported) and picks the cube that covers a line.
  - Per-spaxel local continuum: a polynomial over the line-free channels either side of each line, all
    spaxels solved at once with sigma clipping (or IRSQR/ASLS per spaxel).
  - Point-source removal: the continuum image next to the line, background-subtracted and scaled in the
    PSF core, removed plane by plane, so the template follows the PSF's change across a sub-band.
    Companions (e.g. HV Tau AB next to HV Tau C) are found and removed too.
  - Moment 0/1/2 with S/N masks for the continuum-subtracted and the extended emission; Gaussian centroid
    velocities per spaxel with Monte Carlo errors (a vectorised, bounded Levenberg–Marquardt: ~10⁵ fits/s).
  - Velocity zero points: systemic RV, per-sub-band offsets, or `zero_point: star`.
  - Stacking of several lines of one species (e.g. H₂ S(1)–S(3)), aligned on the source, on a common
    velocity axis and the coarsest spaxel grid.
  - Channel maps (FITS with a `VRAD` axis) and position–velocity cuts along any PA.
  - Regions (circle, ellipse, annulus, polygon; DS9 fk5 in and out) → spectrum over all sub-bands →
    the normal slab fit, so region-by-region LTE fits need no extra code.
  - FITS maps with the celestial WCS (open in CARTA/DS9), a multi-extension file, CSV spectra, JSON
    summaries and PNG figures (including the notebook-style moment-0 panel).
  - Synthetic cubes with a point source, a Keplerian ring and a jet, PSF-convolved plane by plane, with
    optional resampling wiggles and sub-band offsets (`jalebi.cube.synthetic`, `jalebi cube synth`).
  - YAML config (`jalebi cube init/run`) and a config-driven pipeline, parallel over lines (`n_jobs`).
- **`jalebi.lines`**: the emission-line catalogue (H₂ S(0)–S(8) with E_u/A/g_u, fine-structure lines,
  H I) with forgiving names (`"[NeII]"`, `"h2 s1"`, `12.8135`, `"CO P(10)=4.9876"`), groups for stacking,
  unit conversions, the batched Gaussian fitter and `fit_line` for one line in a 1-D spectrum (the packaged
  `find_and_measure_line` of the old `line_fit_essentials.py`, with Monte Carlo errors).
- CLI: `jalebi cube info | lines | init | run | maps | stack | channels | pv | region | cutout | synth | demo`.
- Web app: a **Cube** workspace (`jalebi serve --tab cube [--cube DIR]`): map view with continuum, moments,
  extended emission, velocity, errors, S/N and channel maps; click a spaxel for its spectrum and Gaussian
  fit; circle/ellipse/annulus regions or a drawn polygon → region spectrum → *Send to slab fit*; PV cuts;
  FITS/PNG export; the equivalent terminal command and Python code, and the cube config YAML.
- Bundled example: five MIRI-MRS cutouts of **HV Tau C** (MINDS, JWST PID 1282; 3.3 MB, `example:HV_Tau_C_cube`),
  `examples/08_cube_maps.py`, `examples/09_cube_region_fit.py`, `examples/configs/HV_Tau_C_cube.yaml`.
- `jalebi doctor` checks the cube example and makes one map.
- 24 tests (`tests/test_cube.py`): calibrated velocity errors, flux recovery of point source and extended
  emission, rotation of the synthetic ring, sub-band offsets, resampling wiggles, stacking, channel maps,
  PV, regions and DS9, FITS WCS, config round trip, CLI, the web-app workspace (north-up and rotated cubes),
  the shared velocity frame after `zero_point: star`, stacks with missing channels, and the HV Tau C jet.
- `docs/CUBE.md` (method, pitfalls, validation) and `docs/VERSION_CONTROL.md` (branches, tags, releases).
- **The old `cube_maps.py` recipe**, checked against the old code on the HV Tau C cutouts (same NaN
  patterns, maps equal to ≤ 6 × 10⁻⁷ of the peak; `docs/CUBE.md` §6):
  - `jalebi.cube.cube_maps`: `get_channel`, `get_continuum`, `make_moment0`, `get_moment0`,
    `get_moment0_all`, `make_ratio_plot`, `plot_moment0_map`, `plot_moment0_map_with_bkgd_mask`,
    `add_au_box`, `get_channel_maps` with the old arguments and file names (no spectral_cube needed).
  - Continuum `method: aspls` (pybaselines, lam 5e6) and `nan_policy`; `window_um` (the planes
    `spectral_slab` selects); `band: nominal` (the `get_channel` boundaries) or a fixed sub-band;
    `dq_mask: false`.
  - Moment-0 windows of native channels, `component: full | slow | fast` (centre ± 4, ± 2, and the wings).
  - Masks from an empty-sky circle (`rms_region`, nanstd / mean / median → `mom0_masked`) and from a
    Background2D background (photutils, or a transcription of photutils 3.0 that gives the same numbers).
  - Line-ratio maps (`cube.ratio_map`, config `ratios:`, `jalebi cube ratio`): line 2 reprojected onto
    line 1 (bilinear, as `reproject_interp`), masks, FITS + three-panel PNG.
  - The native channels of the moment window as single 2-D FITS files named by wavelength
    (`channel_slices`, `channels.slices: true`, `jalebi cube channels --slices`).
  - `jalebi cube moment0` (the old `make_moment0` from the terminal); `--recipe cube_maps` and the recipe
    options for `maps`, `stack`, `channels` and `ratio` (`--continuum`, `--component`, `--window-um`,
    `--band`, `--rms-region`, `--no-dq`, `--zeros-valid`, `--nan-policy`, `--min-valid`);
    `config.apply_recipe`; config fields `zero_is_nan`, `moments.min_valid`.
  - Web app: continuum method, moment window, window in µm, sub-band choice, DQ mask, the old NaN / zero
    rules, a *cube_maps.py recipe* button, an RMS-circle mask ("masked") and a ratio map ("ratio"); the
    ratio's second line is made with the settings of the first; everything in the code panel and the YAML.
  - `jalebi cube init --example cube_maps`; `examples/10_cube_maps_recipe.py`; 18 more tests (103 in all),
    including one that parses every command the web app's code panel shows.

### Changed
- `JalebiApp` gained `use_spectrum()` (load a Spectrum made elsewhere as the target) and `start_tab`;
  `jalebi serve` gained `--tab` and `--cube`.
- `list_examples()` hides `.fits.gz` files like `.fits` files.
- `get_line("5.3402=FeII")` works as well as `"FeII=5.3402"` (an exact wavelength with a name).
- Channel maps, PV cuts and ratios of lines with custom names ("12.2786=myH2") are found and named by
  their own tag (they were looked up in the catalogue by name).

## [0.9.1] — 2026-09-29

### Fixed
- Results now go to one folder per source. The default output is `results/{target}`, and `{target}` is
  replaced by the source name (`V* FZ Tau` → `results/FZ_Tau`). Before, the web app and configs made
  with `jalebi init --example blank` wrote every source straight into `results/`.
  - An `output: results` left over from 0.9.0 is treated the same way.
  - Any other path without `{target}` is used as written.
- `jalebi batch` and the app's Batch tab put each target in its own folder and `population.csv` in the
  common parent (`results/`).
- When `target.name` is not set, the name comes from the FITS `TARGNAME`, or else the file name without
  its extension.

### Added
- `--name` for `jalebi prep`, `detect` and `fit`. With `--target` and no `--name`, the name is taken from
  the data, so another source's results never land in the config's original folder.
- The terminal and the web app print the output folder when a run starts, and the app shows it under
  the output box.
- The example configs use `results/{target}` (or `results/{target}/<model>`, e.g. `water_hot_cold`).

## [0.9.0] — 2026-09-28 — first public release

JALEBI grew out of the in-house `slabfit` code, which was renamed and packaged for distribution.

### Added
- Installable package (`pip install .`, hatchling build) with the `jalebi` command and `python -m jalebi`.
- Interactive installer `install.py` (plus `install.sh` and `install.bat`). It checks the system, creates a
  conda or virtual environment or uses the current one, lets you pick optional extras, installs only
  what is missing, runs `jalebi doctor` and offers a taste-test fit.
- `jalebi doctor`: checks the packages, line lists, partition functions, example data and parallelism,
  and runs a model-speed benchmark. It also works when dependencies are missing (`python -m jalebi.doctor`).
- Bundled line lists (HITRAN 2020 for 16 molecules, HITEMP H₂O and CO; 27 MB) and a two-level cache:
  your downloads in `~/.jalebi/linedata` (or `$JALEBI_DATA`) shadow the bundled files.
- Bundled example data: the FZ Tau MIRI-MRS x1d spectrum (GO 1549) and a synthetic channel-3 spectrum
  with known parameters. The `example:<name>` path prefix works in configs and on the CLI.
- `jalebi.synthetic`: MIRI-like synthetic spectra for injection–recovery tests (`jalebi synth`).
- New commands: `jalebi demo`, `jalebi examples`, `jalebi synth`, `jalebi about`, `jalebi --version`, and
  `jalebi init --example {fz_tau,synthetic,water_hot_cold,blank}`.
- Example scripts 01–07, a quick-start notebook, example configs, and a batch target table.
- Tests for packaging consistency, bundled data, synthetic recovery, MCMC summaries and the CLI (54 tests).
- GitHub Actions for tests on Linux/macOS with Python 3.10–3.14, a wheel build check, and PyPI publishing.

### Changed
- The package, module and command are renamed `slabfit` → `jalebi`, and `$SLABFIT_DATA` → `$JALEBI_DATA`
  (the old variable is still read).
- The MIRI-MRS resolving-power relation R = 4603 − 128 λ is now credited to Argyriou et al. (2023). The
  config key is `argyriou2023`; `pontoppidan2024` is still accepted.
- The web app opens on the bundled example data when no `--data-root` is given.
- MCMC checkpointing switches itself off with a message when h5py is missing.

### Fixed
- The linear area solve (NNLS, used by the grid, the optimiser, detection and the app's *Auto areas*) now
  keeps a tied isotopologue on its parent's area: its flux joins the parent's column. Before, the
  optimiser's objective gave it an independent area and the grid left it out. The final model and the
  MCMC always applied the tie correctly.
- The outer radius of an `annuli` component is no longer rescaled by NNLS, because the flux is not linear
  in R² there. It is fitted as an ordinary parameter.
- Detection windows added for ¹³CO, ¹³CCH₂ and H¹³CN, so isotopologue candidates are no longer skipped.
- Figures show inline in Jupyter: `jalebi.plots` switches to the Agg backend only outside IPython.
- `jalebi model` accepts `--logN/--T/--R` (and lower-case forms) with every Typer version.
- New config key `R_constant` for a constant resolving power.

### Carried over from slabfit
- LTE slab forward model on a fine ln λ grid with the MIRI LSF and pixel-integration operator; opacity
  groups; isotopologue ties; radial-gradient (annuli) components; per-component line-list releases.
- Continuum methods (IRSQR, median + Savitzky–Golay, ASLS, convex hull, rolling minimum, manual spline,
  Banzatti+2025 windows), Q-branch protection, default masks, model-aware refinement.
- Staged fitting (grid → differential evolution with NNLS areas → emcee), convergence diagnostics,
  corner/correlation/trace/posterior-predictive plots, ΔBIC detection test, automatic molecule detection.
- Typer CLI and Panel web app.
