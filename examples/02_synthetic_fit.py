"""Example 2 — injection–recovery on the bundled synthetic spectrum, step by step with the Python API.

The spectrum (MIRI channel 3, S/N 150) was made from H2O + CO2/13CO2 + C2H2 + HCN slabs with the
parameters in synthetic_truth.yaml.  This script walks through every stage the pipeline runs:

    prepare (continuum, masks, noise) -> grid over (log N, T) -> differential evolution -> emcee

and compares the result with the truth.

Run:  python 02_synthetic_fit.py                       (about 2–5 min on 4 cores)
      python 02_synthetic_fit.py --nsteps 3000 --processes 8
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from jalebi import plots
from jalebi.config import ProjectConfig
from jalebi.examples import example_path
from jalebi.pipeline import RunResult, build_problem, prepare, run_grid_stage, run_mcmc_stage, run_optimise_stage

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--processes", type=int, default=4, help="worker processes for DE and emcee")
ap.add_argument("--nsteps", type=int, default=600, help="MCMC steps (real fits: >= 50 autocorrelation times)")
ap.add_argument("--out", default=str(HERE / "results" / "02_synthetic"))
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

# 1. configuration: the same YAML the CLI and the web app use --------------------------------------
cfg = ProjectConfig.load(HERE / "configs" / "synthetic.yaml")
cfg.fit.optimise.workers = args.processes
cfg.fit.mcmc.processes = args.processes
cfg.fit.mcmc.nsteps = args.nsteps
truth = yaml.safe_load(open(example_path("synthetic/synthetic_truth.yaml")))

# 2. data preparation: load, continuum, masks, noise ------------------------------------------------
spec = prepare(cfg)
print(f"{spec.name}: {len(spec.wave)} pixels in bands {spec.bands}; continuum = {cfg.continuum.method}")

# 3. the fit problem: model + free parameters + likelihood -----------------------------------------
problem = build_problem(cfg, spec)
print(f"{problem.ndim} free parameters: {[p.key for p in problem.free]}")
run = RunResult(cfg, spec, problem)

# 4. stages ----------------------------------------------------------------------------------------
run_grid_stage(run)          # per component chi2 over (log N, T), area by NNLS
run_optimise_stage(run)      # differential evolution over all components jointly, areas by NNLS
run_mcmc_stage(run, outdir=str(out))

# 5. results ---------------------------------------------------------------------------------------
summ = run.mcmc.summary().set_index("parameter")
print("\ncomponent  param   truth    median   -err   +err")
for c in truth["components"]:
    if c.get("tie_to"):
        continue
    for k in ("T", "logN", "logR", "logNA"):
        tv = c[k] if k != "logNA" else c["logN"] + np.log10(np.pi) + 2 * c["logR"]
        r = summ.loc[f"{c['name']}.{k}"]
        print(f"{c['name']:9s}  {k:6s} {tv:7.2f}  {r['median']:8.2f}  {r['minus']:5.2f}  {r['plus']:5.2f}")
d = run.mcmc.diagnostics()
print(f"\nacceptance {d['acceptance']:.2f}, chain = {d['steps_over_tau']:.0f} autocorrelation times, max R-hat "
      f"{np.nanmax(d['rhat']):.2f}  (converged: >= 50 tau and R-hat < 1.05)")
print("tau_max < 1 fraction per component (1 = optically thin: only N x area constrained):", run.mcmc.tau_flag())

for name, fig in (("fit_windows.png", plots.plot_fit(problem, run.theta, title="synthetic disk: median posterior model")),
                  ("corner.png", plots.plot_corner(run.mcmc)),
                  ("correlation.png", plots.plot_correlation(run.mcmc)),
                  ("posterior_predictive.png", plots.plot_posterior_predictive(problem, run.mcmc))):
    fig.savefig(out / name, dpi=100); plt.close(fig)
for name, g in run.grids.items():
    fig = plots.plot_grid(g); fig.savefig(out / f"grid_{name}.png", dpi=100); plt.close(fig)
print(f"\nfigures in {out}")
