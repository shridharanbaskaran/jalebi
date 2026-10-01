# Changelog

All notable changes to JALEBI. The format follows [Keep a Changelog](https://keepachangelog.com/) and the
version numbers follow [Semantic Versioning](https://semver.org/).

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
