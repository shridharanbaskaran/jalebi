# Changelog

All notable changes to JALEBI. The format follows [Keep a Changelog](https://keepachangelog.com/) and the
version numbers follow [Semantic Versioning](https://semver.org/).

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
