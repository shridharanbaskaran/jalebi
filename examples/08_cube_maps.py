"""Example 8 — line maps and velocity maps of an IFU cube (jalebi.cube), on the bundled HV Tau C cubes.

HV Tau C is an edge-on Class II disk: the continuum is a point source, but [Fe II] and [Ne II] trace a
bipolar jet along PA ~ 25 deg and H2 is extended along the disk plane.  This script

  1. removes a per-spaxel continuum and the point source (continuum image as the PSF) for five lines,
  2. makes moment and Gaussian-centroid velocity maps with Monte Carlo errors,
  3. stacks H2 S(1)-S(3) for a higher-S/N velocity map,
  4. makes channel maps and a position-velocity cut along the jet,
  5. writes FITS maps (with WCS: open them in CARTA or DS9) and PNG figures to results/HV_Tau_C_cube/.

    python 08_cube_maps.py                 # ~30 s
    python 08_cube_maps.py --cube /path/to/target_folder_with_s3d_files --jet-pa 25

The same from the terminal:
    jalebi cube maps example:HV_Tau_C_cube -l "[Fe II] 5.34" -l "H2 S(1)" --zero-point star --pa 25
    jalebi cube run configs/HV_Tau_C_cube.yaml
"""
import argparse
from pathlib import Path

import numpy as np

from jalebi import cube
from jalebi.cube.plots import plot_channel_maps, plot_fancy_moment0, plot_pv

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--cube", default="example:HV_Tau_C_cube", help="folder of *_s3d.fits cubes")
ap.add_argument("--distance", type=float, default=140.0)
ap.add_argument("--jet-pa", type=float, default=25.0)
ap.add_argument("--n-mc", type=int, default=100)
args = ap.parse_args()
out = HERE / "results" / "HV_Tau_C_cube"

cs = cube.CubeSet(args.cube)
print(cs.table().to_string(index=False))
print("lines covered:", ", ".join(ln.name for ln in cs.lines_covered()))

# 1-2. one line at a time: prepare (slab, continuum, PSF removal) -> maps -> files
maps = {}
for line in ["[Fe II] 5.34", "[Ne II] 12.81", "H2 S(3)", "H2 S(2)", "H2 S(1)"]:
    if not cs.covering(cube.io.get_line(line).wave):
        continue
    lc = cube.prepare_line(cs, line, continuum={"order": 1}, psf={"enabled": True})
    m = cube.line_maps(lc, n_mc=args.n_mc, zero_point="star")
    m.write(str(out), distance_pc=args.distance, pa_deg=args.jet_pa, extent_arcsec=3.5)
    s = m.summary
    print(f"{line:14s} total {s['total_flux_W_m2_snr_masked']:.2e}  point source {s['point_source_line_flux_W_m2']:.2e}  "
          f"extended {s['extended_flux_W_m2']:.2e} W/m2   v(source) {s['source_velocity_kms']:+.1f} km/s")
    maps[line] = m

# the jet: median velocity of the two lobes of [Fe II]
m = maps.get("[Fe II] 5.34")
if m is not None:
    dx, dy = m.lc.cube.offsets(m.lc.center_radec)
    along = dx * np.sin(np.radians(args.jet_pa)) + dy * np.cos(np.radians(args.jet_pa))
    good = np.isfinite(m["vcen"]) & (m["snr_ext"] > 8)
    print(f"[Fe II] jet: v(north lobe) = {np.nanmedian(m['vcen'][good & (along > 0.3)]):+.0f} km/s, "
          f"v(south lobe) = {np.nanmedian(m['vcen'][good & (along < -0.3)]):+.0f} km/s (relative to the source)")
    fig = plot_fancy_moment0(m, distance_pc=args.distance, pa_deg=args.jet_pa, extent_arcsec=3.0)
    fig.savefig(out / "FeII_5.34_fancy_mom0.png", dpi=200, bbox_inches="tight")

# 3. stack H2 S(1)-S(3): common velocity axis and the coarsest spaxel grid
h2 = [ln for ln in ("H2 S(1)", "H2 S(2)", "H2 S(3)") if ln in maps]
if len(h2) >= 2:
    st = cube.line_maps(cube.stack_lines(cs, h2, name="H2 stack"), n_mc=args.n_mc, zero_point="star")
    st.write(str(out), distance_pc=args.distance, pa_deg=args.jet_pa, extent_arcsec=3.5)
    one = maps["H2 S(1)"]
    both = np.isfinite(one["vcen_err"]) & np.isfinite(st["vcen_err"])
    print(f"H2 stack: median velocity error {np.nanmedian(st['vcen_err'][both]):.1f} km/s "
          f"(H2 S(1) alone {np.nanmedian(one['vcen_err'][both]):.1f} km/s)")

# 4. channel maps and a PV cut along the jet
if m is not None:
    cm = cube.channel_maps(m.lc, vmin=-180, vmax=180, dv=45)
    cm.write_fits(str(out / "FeII_5.34" / "FeII_5.34_channels.fits"))
    plot_channel_maps(cm, distance_pc=args.distance, extent_arcsec=3.0).savefig(out / "FeII_5.34_channels.png", dpi=130, bbox_inches="tight")
    pv = cube.pv_diagram(m.lc, pa_deg=args.jet_pa, length_arcsec=5.0)
    pv.write_fits(str(out / "FeII_5.34" / "FeII_5.34_pv.fits"))
    plot_pv(pv, vlim=400).savefig(out / "FeII_5.34_pv.png", dpi=140, bbox_inches="tight")
print(f"\nmaps, spectra and figures in {out}/")
