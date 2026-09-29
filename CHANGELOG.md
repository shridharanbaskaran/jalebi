# Changelog

All notable changes to JALEBI. The format follows [Keep a Changelog](https://keepachangelog.com/) and the
version numbers follow [Semantic Versioning](https://semver.org/).

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
