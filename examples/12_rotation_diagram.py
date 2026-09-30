"""Example 12 — rotation diagrams (jalebi.rotdiag): H2, a flux table, CO with optical depth.

1. The bundled synthetic MIRI-MRS spectrum of H2 (two temperatures, A_V = 10, OPR = 2.3, lines at +8 km/s):
   find the S(J) lines, measure them, fit one temperature, two temperatures and a power law (BIC), and sample the
   two-temperature posterior (T1, T2, N1, N2, A_V, OPR) with emcee. Recovered vs true values are printed.
2. The same H2 lines as a table of fluxes (as copied from a paper): fitted without a spectrum.
3. CO v=1-0 in FZ Tau (MIRI ch1A): one temperature, with the curve-of-growth optical depth of a slab of
   R = 0.3 au and intrinsic Δv = 4.7 km/s, next to the same fit assuming thin lines.
4. Why the opacity correction matters: an optically thick CO slab spectrum made with jalebi's LTE slab model
   (log N 18.5, 1100 K), fitted with and without it.

Run:  python 12_rotation_diagram.py            (about 1 min)
Results: results/12_rotation_diagram/ (one folder per part; figures rotation_diagram.png, corner.png, line_fits.png)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from jalebi.examples import example_path
from jalebi.rotdiag import (FitConfig, Geometry, MCMCConfig, Selection, example_config, find_features, fit_rotation,
                            measure_features, run_rotdiag)
from jalebi.rotdiag.plots import plot_rotation_diagram
from jalebi.synthetic import make_synthetic_spectrum

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=2500)
ap.add_argument("--out", default=str(HERE / "results" / "12_rotation_diagram"))
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

# ---- 1. synthetic H2 spectrum -------------------------------------------------------------------
cfg = example_config("h2")                       # two temperatures, free OPR (spin species) and A_V, compare models
cfg.mcmc.steps = args.steps; cfg.mcmc.burn = args.steps // 3
res = run_rotdiag(cfg, outdir=str(out / "1_H2_synthetic"))
print("== 1. synthetic H2\n" + res.fit.summary())
print(res.comparison[["model", "n_free", "chi2_red", "bic", "dBIC"]].to_string(index=False))
truth = yaml.safe_load(open(example_path("synthetic/rotdiag_H2_synthetic_truth.yaml")))["params"]
t = res.fit.table().set_index("param")
print("   parameter      fit                  truth")
for k, v in truth.items():
    print(f"   {k:6s}  {t.loc[k, 'median']:9.4g} -{t.loc[k, 'err_lo']:.3g} +{t.loc[k, 'err_hi']:.3g}   {v:g}")

# ---- 2. a flux table -----------------------------------------------------------------------------
cfg = example_config("h2_fluxes")                # label, wave, flux, err in W m^-2; two temperatures + A_V
cfg.mcmc.steps = args.steps; cfg.mcmc.burn = args.steps // 3
res2 = run_rotdiag(cfg, outdir=str(out / "2_H2_table"))
print("\n== 2. H2 flux table\n" + res2.fit.summary())

# ---- 3. CO in FZ Tau -----------------------------------------------------------------------------
cfg = example_config("co")                       # v=1-0, 30 lines, opacity on, R = 0.3 au, Δv = 4.7 km/s
cfg.mcmc.enabled = False
res3 = run_rotdiag(cfg, outdir=str(out / "3_CO_FZ_Tau"))
thin = fit_rotation(res3.features, res3.members, FitConfig(model="single", geometry=res3.fit.model.geometry))
print(f"\n== 3. CO in FZ Tau: {int(res3.features['detected'].sum())} lines detected")
print(f"   optically thick slab: T = {res3.fit.best['T']:.0f} K, log N = {res3.fit.best['logN']:.2f}, χ²_red {res3.fit.chi2_red:.2f}")
print(f"   thin lines assumed:   T = {thin.best['T']:.0f} K, log N = {thin.best['logN']:.2f}, χ²_red {thin.chi2_red:.2f}")

# ---- 4. thin vs thick on a slab spectrum with known answers --------------------------------------
spec, _ = make_synthetic_spectrum([dict(name="CO", molecule="CO", logN=18.5, T=1100.0, logR=-1.0, fwhm=4.7, linelist_release="hitemp")],
                                  bands=("1A",), snr=300, seed=4, continuum=dict(level=0.2, slope=1.0, wiggle=0.0))
geo = Geometry(mode="radius", distance_pc=140.0, R_au=0.1)
F, M = find_features("CO", spec, Selection(max_features=40))
Fm, stamps = measure_features(spec, F, M)
thick = fit_rotation(Fm, M, FitConfig(model="single", geometry=geo, opacity=True, fwhm_kms=4.7, sys_frac=0.0),
                     mcmc=MCMCConfig(steps=1500, burn=500))
thin4 = fit_rotation(Fm, M, FitConfig(model="single", geometry=geo, sys_frac=0.0))
print("\n== 4. slab spectrum, truth log N 18.5, T 1100 K")
print(f"   with the opacity correction: log N = {thick.best['logN']:.3f}, T = {thick.best['T']:.0f} K (χ²_red {thick.chi2_red:.2f})")
print(f"   assuming thin lines:        log N = {thin4.best['logN']:.3f}, T = {thin4.best['T']:.0f} K (χ²_red {thin4.chi2_red:.1f})")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
plot_rotation_diagram(Fm, geo, thick, ax=axes[0], title="CO slab: optical depth fitted", label_points=False)
plot_rotation_diagram(Fm, geo, thin4, ax=axes[1], title="CO slab: thin lines assumed", label_points=False)
fig.tight_layout(); fig.savefig(out / "4_CO_thin_vs_thick.png", dpi=130)
print(f"\nresults in {out}")
