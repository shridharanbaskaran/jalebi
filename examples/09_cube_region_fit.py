"""Example 9 — region spectra from a cube, and the slab fit on them (jalebi.cube + jalebi.pipeline).

A circle, ellipse or polygon on a map becomes a spectrum over all sub-bands (each cube through its own
WCS), which is an ordinary jalebi Spectrum: the whole slab-fitting machinery runs on it unchanged, so
region-by-region LTE fits need no extra code.

    python 09_cube_region_fit.py                                        # regions of the bundled HV Tau C cutouts
    python 09_cube_region_fit.py --cube /path/to/full_s3d_cubes --fit   # + H2O/CO2 slab fit of the on-source region

The bundled cutouts only hold five line windows; point --cube at a folder of full s3d cubes for fits.
The same from the terminal:
    jalebi cube region /path/to/cubes --circle "0 0 0.6" --offsets --out on_source.csv --fit configs/FZ_Tau_quick.yaml
"""
import argparse
from pathlib import Path

import numpy as np

from jalebi import cube
from jalebi.cube.plots import plot_regions

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--cube", default="example:HV_Tau_C_cube")
ap.add_argument("--fit", action="store_true", help="slab-fit the on-source region (needs full cubes)")
ap.add_argument("--distance", type=float, default=140.0)
args = ap.parse_args()
out = HERE / "results" / "HV_Tau_C_regions"
out.mkdir(parents=True, exist_ok=True)

cs = cube.CubeSet(args.cube)
star = cube.resolve_center(cs, "auto")
regions = {
    "on_source": cube.offset_region("circle", star, 0.0, 0.0, 0.6),
    "jet_north": cube.offset_region("circle", star, 0.42, 0.91, 0.35),
    "jet_south": cube.offset_region("circle", star, -0.42, -0.91, 0.35),
    "h2_east": cube.offset_region("ellipse", star, 1.93, -0.52, 1.0, 0.6, 105.0),
    "halo": cube.offset_region("annulus", star, 0.0, 0.0, 1.5, 2.5),
}
spectra = {}
for name, reg in regions.items():
    s = cube.region_spectrum(cs, reg, name=f"HV Tau C {name}", distance_pc=args.distance)
    s.save(str(out / f"{name}.csv"))
    spectra[name] = s
    print(f"{name:10s} {reg.to_ds9():60s} {len(s.wave):5d} pixels  {', '.join(s.bands)}")
cube.to_ds9_file(list(regions.values()), str(out / "regions.reg"))

# line fluxes of the jet lines in each region with the 1-D fitter of jalebi.lines
from jalebi.lines import fit_line  # noqa: E402
for name, s in spectra.items():
    row = []
    for line in ("[Fe II] 5.34", "[Ne II] 12.81", "H2 S(1)"):
        try:
            r = fit_line(s, line=line, n_mc=100)
            row.append(f"{line} {r.flux_W_m2:.2e}±{r.flux_W_m2_err:.1e} ({r.v_kms:+.0f} km/s)")
        except ValueError:
            row.append(f"{line} —")
    print(f"{name:10s} " + " | ".join(row))

lm = cube.line_maps(cube.prepare_line(cs, "[Fe II] 5.34"), kinematics=False)
fig = plot_regions(np.where(lm["snr_ext"] >= 2, lm["mom0_ext"], np.nan), lm.lc.cube, lm.lc.center_radec,
                   list(regions.values()), labels=list(regions))
fig.savefig(out / "regions.png", dpi=140, bbox_inches="tight")

if args.fit:
    from jalebi.config import ProjectConfig
    from jalebi.pipeline import run_pipeline
    cfg = ProjectConfig.load(HERE / "configs" / "FZ_Tau_quick.yaml")      # a starting model; edit the components
    cfg.target.name = "HV Tau C on-source"; cfg.target.distance_pc = args.distance; cfg.target.rv_kms = 0.0
    cfg.output = str(out / "fit_on_source")
    run = run_pipeline(cfg, spec=spectra["on_source"], stages=["grid", "optimise"])
    print(f"on-source slab fit: chi2_red = {run.opt.chi2_red:.2f} -> {cfg.output}/")
print(f"\nregion spectra (CSV), regions.reg (DS9) and regions.png in {out}/")
