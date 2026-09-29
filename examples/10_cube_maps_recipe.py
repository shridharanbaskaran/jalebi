"""Example 10 — the old cube_maps.py recipe in jalebi, on the bundled HV Tau C cubes.

The functions of cube_maps.py live on in `jalebi.cube.cube_maps` with the same arguments and file names
(and the same numbers: see docs/CUBE.md §6).  This script

  1. makes [Fe II] 5.34 moment-0 maps with the old recipe (+-0.1 um slab, aspls continuum per spaxel,
     full / slow / fast channel windows) through the old API,
  2. makes the same map with the jalebi API, which adds errors, S/N and velocities, and checks that the
     numbers agree,
  3. makes the old three-panel [Fe II] / [Ne II] ratio figure, and the ratio as a line-flux ratio,
  4. plots a masked map with a 250 au box, and writes the native channels as single FITS files.

    python 10_cube_maps_recipe.py            # ~40 s
    python 10_cube_maps_recipe.py --data /path/to/DATA --source HV_Tau_C     # your own Level3_ch*_s3d.fits

The same from the terminal:
    jalebi cube moment0 /path/to/DATA/HV_Tau_C -l 5.3402=FeII --component full --png
    jalebi cube ratio  /path/to/DATA/HV_Tau_C -l "[Fe II] 5.34" -l "[Ne II] 12.81" --rms-region "12 8 3" \\
        --recipe cube_maps --unit "MJy/sr m"
    jalebi cube init cube.yaml --example cube_maps && jalebi cube run cube.yaml
"""
import argparse
from pathlib import Path

import numpy as np

from jalebi import cube
from jalebi.cube import cube_maps as cm

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--data", default=None, help="folder that holds SOURCE/ (default: the bundled example)")
ap.add_argument("--source", default="example:HV_Tau_C_cube")
ap.add_argument("--distance", type=float, default=140.0)
args = ap.parse_args()
out = HERE / "results" / "HV_Tau_C_cube_maps"
out.mkdir(parents=True, exist_ok=True)

# 1. the old API: moment 0 of [Fe II] 5.3402 um in the three windows (files as cube_maps.py named them)
for comp in ("full", "slow", "fast"):
    m, fname = cm.make_moment0([5.3402], ["FeII"], args.source, 0.1, component=comp, save=True, folder_path=args.data,
                               output_path=str(out), verbose=False)
    print(f"{comp:4s}: channels {m.channels}  peak {np.nanmax(m.value):.3e} {m.unit}  -> {fname}")

# 2. the jalebi API with the same recipe: the same numbers, plus errors, S/N and velocities
cs = cube.CubeSet(cm._source_folder(args.data, args.source)[0], dq_mask=False, zero_is_nan=False)
lc = cube.prepare_line(cs, "5.3402=FeII", window_um=0.1, band="nominal", psf={"enabled": False},
                       continuum={"method": "aspls", "nan_policy": "propagate"})
lm = cube.line_maps(lc, component="full", min_valid=1, n_mc=50, rms_region=(12, 8, 3), rms_sigma=3)
old = cm.make_moment0([5.3402], ["FeII"], args.source, 0.1, folder_path=args.data, output_path=str(out), verbose=False)[0]
diff = np.nanmax(np.abs(lm.native("mom0", "MJy/sr m") - old.value)) / np.nanmax(np.abs(old.value))
print(f"jalebi line_maps(component='full') vs make_moment0: max |difference| = {diff:.1e} of the peak")
print(f"  and jalebi adds: S/N (median {np.nanmedian(lm['snr']):.1f}), velocities in {np.isfinite(lm['vcen']).sum()} spaxels, "
      f"v(source) = {lm.summary['source_velocity_kms']:+.1f} km/s")
lm.write(str(out / "jalebi_maps"), distance_pc=args.distance, pa_deg=25)

# 3. the old ratio figure (numbers of the old maps), and the line-flux ratio
fig, rm = cm.make_ratio_plot([5.3402, 12.8135], ["FeII", "NeII"], args.source, sigma_thresh=[5, 5], folder_path=args.data,
                             output_path=str(out), showfig=False, savefig=str(out / "FeII_NeII_ratio_cube_maps.png"), verbose=False)
flux_ratio = rm.ratio * (12.8135 / 5.3402) ** 2
print(f"[Fe II]/[Ne II]: median {np.nanmedian(rm.ratio):.3f} in the old units -> line-flux ratio {np.nanmedian(flux_ratio):.3f} "
      f"({rm.summary()['n_valid']} spaxels)")

# 4. a masked map with a 250 au box, and the native channels as single files
fig, ax = cm.plot_moment0_map(str(out / "moment0_maps" / Path(cm._source_folder(args.data, args.source)[1]) /
                                  "5.3402_FeII_full_moment0.fits"),
                              title="[Fe II] 5.34 µm (full)", sigma_clip=True, sigma_thresh=3, showfig=False)
x0, y0 = lc.center_pix
cm.add_au_box(ax, args.distance, au_size=250, pixel_scale=lc.cube.pixscale, color="black", center_x=x0, center_y=y0)
fig.savefig(out / "FeII_full_masked_au_box.png", dpi=150, bbox_inches="tight")
files = cube.channel_slices(lc, "full").write(str(out / "channel_slices" / "5.3402"))
print(f"wrote {len(files)} channel slices ({Path(files[0]).name} ... {Path(files[-1]).name}) and the figures to {out}/")
