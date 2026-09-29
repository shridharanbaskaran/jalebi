# JALEBI examples

Every example runs on the data bundled with the package (the FZ Tau MIRI-MRS spectrum, cutouts of the
HV Tau C MIRI-MRS cubes and a synthetic spectrum with known answers), so nothing needs downloading. Copy the examples anywhere with

```bash
jalebi examples ./my_examples && cd my_examples
```

| File | What it shows | Time on 2 cores |
| --- | --- | --- |
| `01_quick_model.py` | LTE slab models without data: CO₂ thin → thick, water at three temperatures, the curve of growth | 5 s |
| `02_synthetic_fit.py` | injection–recovery, stage by stage with the Python API; the posterior compared with the true parameters | 3 min |
| `03_continuum_methods.py` | every continuum method on FZ Tau, and against the known continuum of the synthetic spectrum | 5 s |
| `04_detect_molecules.py` | automatic molecule detection on FZ Tau; writes `results/FZ_Tau_detected.yaml` | 10 s |
| `05_fit_fz_tau.py` | simultaneous fit of FZ Tau at 13.45–17.5 µm (`--mcmc` for posteriors) | 2 min (+ MCMC) |
| `06_batch.sh` | several targets with one config → `population.csv` | 2–3 min |
| `07_line_lists.py` | bundled line lists, partition functions, HITEMP versus HITRAN, fetching more | 5 s |
| `08_cube_maps.py` | line and velocity maps of the bundled HV Tau C cubes: point-source removal, five lines, the H₂ stack, channel maps, a PV cut along the jet | 30 s |
| `09_cube_region_fit.py` | region spectra of HV Tau C (jet lobes, H₂, halo) over all sub-bands, line fluxes per region; `--cube DIR --fit` slab-fits a region of full cubes | 15 s |
| `10_cube_maps_recipe.py` | the old `cube_maps.py` functions (`make_moment0`, `make_ratio_plot`, `plot_moment0_map`, `add_au_box`, channel slices) and the same recipe through the jalebi API, with the numbers compared | 35 s |
| `notebooks/jalebi_quickstart.ipynb` | the tour in a notebook | 5 min |

All results go to `results/` next to the scripts.

## Configs

| Config | Target and model |
| --- | --- |
| `configs/synthetic.yaml` | the synthetic spectrum: H₂O, CO₂ + ¹³CO₂ (tied), C₂H₂, HCN |
| `configs/FZ_Tau_quick.yaml` | FZ Tau 13.45–17.5 µm: hot + warm H₂O (HITEMP), CO₂ + ¹³CO₂, C₂H₂, HCN |
| `configs/FZ_Tau_water_hot_cold.yaml` | three water temperatures (hot and warm on HITEMP, cold on HITRAN), 13.45–17.5 + 21–27 µm |
| `configs/FZ_Tau_water_CO.yaml` | CO fundamental and hot water at 4.95–7.5 µm |
| `configs/FZ_Tau_autodetect.yaml` | no components listed: the detection decides |
| `configs/FZ_Tau_annuli.yaml` | a radial temperature gradient for water instead of discrete slabs |
| `configs/targets.csv` | a target table for `jalebi batch` |
| `configs/HV_Tau_C_cube.yaml` | a cube config (`jalebi cube run`): five lines, the H₂ stack, channel maps, a PV cut along the jet, four regions |
| `configs/HV_Tau_C_cube_maps.yaml` | the old `cube_maps.py` recipe as a cube config: ±0.1 µm, aspls, 9-channel moment 0, RMS-circle masks, the [Fe II]/[Ne II] ratio, channel slices |

Each one runs as it is: `jalebi fit configs/FZ_Tau_quick.yaml --stages grid,optimise`. For your own
disk, copy one and change `target.path` to the folder that holds your `Level3_ch*_x1d.fits` files.
`example:FZ_Tau` means the bundled data.

## Things to notice

- **Optically thin components** (C₂H₂ and HCN in the synthetic spectrum) recover `log N·A`, but not N and
  R separately: the posterior lies along the thin-limit ridge (02, the correlation plot).
- **The continuum matters most for blended bands.** 03 shows the estimated minus the true continuum for
  each method. A few mJy under the C₂H₂/HCN pseudo-continuum moves their temperatures by ~100 K.
- **Weak organics in real data run to the prior edges.** In FZ Tau the C₂H₂ and HCN Q-branches are weak
  on top of the water forest, and in 05 they end at the bounds of `FZ_Tau_quick.yaml` (T = 250 K,
  log N = 19). Water and CO₂ are well determined. Use the ΔBIC table (`detections.csv`), the `at_edge`
  column of `summary.csv`, and narrower windows around the Q-branches before you interpret organics.
- **Cube maps: the point source hides extended emission in the PSF core.** In 08 the extended [Fe II]
  and [Ne II] maps are ~0 within 0.5 FWHM of HV Tau C by construction; the velocity map there comes from
  the full line cube. The source velocities of different lines differ by a few km/s (sub-band
  calibration plus real differences), which is why the example puts v = 0 at the source for each line.
