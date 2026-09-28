"""Example 5 — simultaneous LTE fit of FZ Tau (13.45–17.5 µm): hot + warm water, CO2 (+13CO2 tied),
C2H2 and HCN, with posterior uncertainties and degeneracy plots.

    python 05_fit_fz_tau.py                              # grid + optimiser (a few minutes)
    python 05_fit_fz_tau.py --mcmc --processes 8         # + emcee: corner, correlations, predictive
    python 05_fit_fz_tau.py --mcmc --nsteps 5000         # a posterior you can quote (check tau and R-hat)

The same run from the terminal:  jalebi fit configs/FZ_Tau_quick.yaml --stages grid,optimise,mcmc
Everything is written to results/FZ_Tau/ (config.yaml, best_fit.json, summary.csv, *.png, chain.npz).
"""
import argparse
import os
from pathlib import Path

from jalebi.config import ProjectConfig
from jalebi.pipeline import run_pipeline

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--mcmc", action="store_true", help="also run emcee")
ap.add_argument("--processes", type=int, default=min(os.cpu_count() or 1, 8))
ap.add_argument("--nsteps", type=int, default=1500)
args = ap.parse_args()

cfg = ProjectConfig.load(HERE / "configs" / "FZ_Tau_quick.yaml")
cfg.output = str(HERE / "results" / "FZ_Tau")
cfg.fit.optimise.workers = args.processes
cfg.fit.mcmc.processes = args.processes
cfg.fit.mcmc.nsteps = args.nsteps
stages = ["grid", "optimise"] + (["mcmc"] if args.mcmc else [])

run = run_pipeline(cfg, stages=stages)

P, _ = run.problem.params_from_theta(run.theta)
print(f"\nbest fit (chi2_red = {run.opt.chi2_red:.2f}):")
for c in cfg.components:
    p = run.problem.model.resolve_params(P)[c.name]
    print(f"  {c.name:9s} T = {p['T']:6.0f} K   log N = {p['logN']:5.2f}   R = {10 ** p['logR']:.3f} au")
print("tau_max per component:", {k: round(v, 2) for k, v in run.problem.model.tau_flags(P).items()})
if run.mcmc is not None:
    print(run.mcmc.summary()[["parameter", "median", "minus", "plus", "at_edge"]].round(3).to_string(index=False))
print(f"\nresults in {cfg.output}")
