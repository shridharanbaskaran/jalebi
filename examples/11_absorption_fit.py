"""Example 11 — LTE gas *absorption*: a cold CO2 screen in front of a continuum source, with a hot
CO2 emission component behind it (embedded protostar / edge-on disk geometry).

The screen follows the group's slabby.spec_abs:

    F = F_c (1 - f_c (1 - e^-tau)),     tau from (log N, T, Δv) as for an emitting slab,
                                        shifted by v (negative = blueshifted, e.g. an outflow)

and with `covers: all` it also attenuates the emission behind it (two_slabs_spec geometry).

The script makes a synthetic spectrum (continuum + emission, absorbed, noise), stores it as a CSV
with the continuum in a `continuum` column, and fits it with the ordinary pipeline
(grid -> differential evolution -> emcee) using examples/configs/absorption_synthetic.yaml.

Run:  python 11_absorption_fit.py                 (about 2-4 min on 4 cores)
      python 11_absorption_fit.py --nsteps 3000 --processes 8
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jalebi import plots
from jalebi.config import ProjectConfig
from jalebi.model import Component, build_model
from jalebi.pipeline import RunResult, build_problem, prepare, run_grid_stage, run_mcmc_stage, run_optimise_stage
from jalebi.synthetic import band_pixels

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--processes", type=int, default=4)
ap.add_argument("--nsteps", type=int, default=800)
ap.add_argument("--snr", type=float, default=300.0, help="continuum S/N per pixel")
ap.add_argument("--out", default=str(HERE / "results" / "11_absorption"))
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

# 1. synthetic spectrum: MRS band 3B (13.3-15.6 um) around the CO2 nu2 band at 15 um --------------
truth = {"CO2_hot": dict(logN=17.5, T=600.0, logR=-0.3),
         "CO2_abs": dict(logN=18.3, T=120.0, rv=-40.0, fc=0.6)}
wave = band_pixels("3B")
cont = 0.8 * (wave / 15.0) ** 2.0                      # rising continuum, Jy
comps = [Component("CO2_hot", "CO2", **truth["CO2_hot"]),
         Component("CO2_abs", "CO2", kind="absorption", covers="all", **truth["CO2_abs"])]
m = build_model(comps, wave, 400.0, [(float(wave[0]), float(wave[-1]))], continuum=cont)
rng = np.random.default_rng(0)
sigma = np.full(len(wave), float(np.median(cont)) / args.snr)
flux = cont + m.evaluate() + rng.normal(0.0, sigma)
csv = out / "absorption_synthetic.csv"      # the same spectrum is bundled: example:synthetic/absorption_synthetic.csv
pd.DataFrame({"wavelength": wave, "flux": flux, "flux_err": sigma, "continuum": cont, "band": "3B"}).to_csv(csv, index=False)
print(f"wrote {csv}: {len(wave)} pixels, tau_max of the screen = {m.tau_flags()['CO2_abs']:.1f}")

# 2. the fit, exactly as `jalebi fit configs/absorption_synthetic.yaml` would run it -------------------
cfg = ProjectConfig.load(HERE / "configs" / "absorption_synthetic.yaml")
cfg.target.path = str(csv)
cfg.fit.optimise.workers = args.processes
cfg.fit.mcmc.processes = args.processes
cfg.fit.mcmc.nsteps = args.nsteps
spec = prepare(cfg)
problem = build_problem(cfg, spec)
print(f"{problem.ndim} free parameters: {[p.key for p in problem.free]}")
run = RunResult(cfg, spec, problem)
run_grid_stage(run)
run_optimise_stage(run)
run_mcmc_stage(run, outdir=str(out))

# 3. truth vs posterior -----------------------------------------------------------------------------
summ = run.mcmc.summary().set_index("parameter")
print("\nparameter        truth   median   -err   +err")
for cname, pars in truth.items():
    for k, tv in pars.items():
        key = f"{cname}.{k}"
        if key in summ.index:
            r = summ.loc[key]
            print(f"{key:15s} {tv:7.2f}  {r['median']:7.2f}  {r['minus']:5.2f}  {r['plus']:5.2f}")
d = run.mcmc.diagnostics()
print(f"\nacceptance {d['acceptance']:.2f}, chain = {d['steps_over_tau']:.0f} autocorrelation times, "
      f"max R-hat {np.nanmax(d['rhat']):.2f}")
print(problem.component_significance(run.theta).to_string(index=False))

for name, fig in (("fit_windows.png", plots.plot_fit(problem, run.theta, title="CO2 screen (v = -40 km/s, f_c = 0.6) + hot CO2 emission")),
                  ("corner.png", plots.plot_corner(run.mcmc)),
                  ("posterior_predictive.png", plots.plot_posterior_predictive(problem, run.mcmc))):
    fig.savefig(out / name, dpi=100); plt.close(fig)
for name, g in run.grids.items():
    fig = plots.plot_grid(g); fig.savefig(out / f"grid_{name}.png", dpi=100); plt.close(fig)
print(f"\nfigures in {out}")
