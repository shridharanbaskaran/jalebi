# HV Tau C — MIRI-MRS cube cutouts (example for `jalebi.cube`)

Five small spectral cutouts of the Level-3 `s3d` cubes of **HV Tau C**, an edge-on Class II disk with a
bipolar [Fe II]/[Ne II] jet (PA ≈ 25°) and extended H₂ emission elongated along PA ≈ 105°. Each cutout keeps
the full field of view (so HV Tau AB, 4″ to the south-west, is inside) and ±1600 km/s around its line:

| file | band | lines |
| --- | --- | --- |
| `Level3_ch1-short_s3d_FeII_5.34.fits.gz` | 1A | [Fe II] 5.340 µm |
| `Level3_ch2-medium_s3d_H2_S3.fits.gz` | 2B | H₂ S(3) 9.665 µm |
| `Level3_ch3-short_s3d_H2_S2.fits.gz` | 3A | H₂ S(2) 12.279 µm |
| `Level3_ch3-short_s3d_NeII_12.81.fits.gz` | 3A | [Ne II] 12.814 µm |
| `Level3_ch3-long_s3d_H2_S1.fits.gz` | 3C | H₂ S(1) 17.035 µm |

Provenance: JWST program 1282 (MIRI European Consortium GTO, *MIRI EC Protoplanetary and Debris Disks
Survey*, MINDS; PI T. Henning), observation 9, observed 2023-09-27, 4-point dither; `jwst` 1.18.0, CRDS
context `jwst_1364.pmap`. The data are public in MAST. The cutouts were written with
`jalebi.cube.write_cutouts(..., window_kms=1600, compress=True)`: SCI and ERR are unchanged, DQ marks the
undefined spaxels, the primary header is the original one.

    jalebi cube info example:HV_Tau_C_cube
    jalebi cube demo                      # maps, stack, channel maps, PV cut and region spectra
    jalebi cube maps example:HV_Tau_C_cube -l "[Fe II] 5.34" -l "H2 S(1)" --zero-point star

A region spectrum from these cutouts only covers the five line windows. For region-by-region slab fits,
point `jalebi cube region` at the folder with the full cubes.
