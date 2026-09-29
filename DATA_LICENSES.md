# Data bundled with JALEBI: sources, credits and terms

JALEBI's code is released under the BSD 3-Clause license (see `LICENSE`). The data files listed
below come from other sources and keep their original terms. If you use them, please cite the
original works.

## Molecular line lists (`src/jalebi/linedata/`)

Each molecule is stored as a Parquet table (wavelength, A<sub>ul</sub>, g<sub>u</sub>, g<sub>l</sub>,
E<sub>u</sub>, E<sub>l</sub>, quantum labels). Where it has one, the table has a partition-function
table `*_Q.npz` next to it.

| File(s) | Source | Reference |
| --- | --- | --- |
| `CO2`, `13CO2`, `HCN`, `H13CN`, `C2H2`, `13CCH2`, `13CO`, `CH4`, `NH3`, `C2H4`, `C2H6`, `C4H2`, `HC3N`, `OH`, `H2`, `H2O` (`*_hitran.parquet`) | HITRAN2020, downloaded from hitran.org through astroquery/HAPI over the MIRI range. The C4H2 and HC3N lists are pruned to lines with S > 3×10⁻³ S<sub>max</sub>. | Gordon, I. E. et al. 2022, JQSRT 277, 107949 |
| `H2O_hitemp.parquet` | HITEMP 2010 water, in the processed form distributed with iSLAT (Apache-2.0) | Rothman, L. S. et al. 2010, JQSRT 111, 2139 |
| `CO_hitemp.parquet` | HITEMP CO (2019 update), in the processed form distributed with iSLAT (Apache-2.0) | Li, G. et al. 2015, ApJS 216, 15; Rothman et al. 2010 |
| `*_Q.npz` | Partition sums: TIPS-2021 via HAPI, or the table shipped with the list | Gamache, R. R. et al. 2021, JQSRT 271, 107713; Kochanov, R. V. et al. 2016, JQSRT 177, 15 |

The HITRAN and HITEMP databases are freely available from https://hitran.org. Their maintainers ask
users to cite the database papers, and to get the most recent versions from the source. JALEBI can
refresh any list with `jalebi linedata fetch <molecule>`. If you redistribute the line data separately
from JALEBI, check the current HITRAN terms of use first.

## FZ Tau MIRI-MRS spectrum (`src/jalebi/example_data/FZ_Tau/`)

These are the twelve Level-3 `x1d` sub-band spectra of FZ Tau from JWST GO program 1549
(PI K. Pontoppidan, *"The deepest search for rare molecules and isotopologues in planet-forming
disks"*), observed 2023-02-28. They were reduced with the `jwst` pipeline 3.0.0 (CRDS context
`jwst_1584.pmap`). The raw data are public in the Mikulski Archive for Space Telescopes (MAST).
The spectrum was presented by Pontoppidan, K. M. et al. 2024, ApJ 963, 158 (JDISC Survey).

Suggested acknowledgement: *"This work is based on observations made with the NASA/ESA/CSA James
Webb Space Telescope. The data were obtained from the Mikulski Archive for Space Telescopes at the
Space Telescope Science Institute, which is operated by the Association of Universities for Research
in Astronomy, Inc., under NASA contract NAS 5-03127 for JWST. These observations are associated with
program #1549."*

## HV Tau C MIRI-MRS cube cutouts (`src/jalebi/example_data/HV_Tau_C_cube/`)

Five spectral cutouts (±1600 km/s around [Fe II] 5.34, H₂ S(3), S(2), S(1) and [Ne II] 12.81 µm, full field
of view) of the Level-3 `s3d` cubes of HV Tau C from JWST program 1282 (MIRI European Consortium GTO,
*MIRI EC Protoplanetary and Debris Disks Survey* (MINDS), PI T. Henning), observation 9, observed
2023-09-27; `jwst` 1.18.0, CRDS context `jwst_1364.pmap`. The data are public in MAST. SCI and ERR are
unchanged; the cutouts were made with `jalebi.cube.write_cutouts`.

Suggested acknowledgement: *"This work is based on observations made with the NASA/ESA/CSA James Webb
Space Telescope. The data were obtained from the Mikulski Archive for Space Telescopes at the Space
Telescope Science Institute, which is operated by the Association of Universities for Research in
Astronomy, Inc., under NASA contract NAS 5-03127 for JWST. These observations are associated with
program #1282."*

## Synthetic spectrum (`src/jalebi/example_data/synthetic/`)

JALEBI generated this spectrum (`jalebi.synthetic`) from the bundled line lists. It is released with
the code under BSD-3-Clause.

## Line-identification and continuum tables (`src/jalebi/data_files/`)

| File | Source |
| --- | --- |
| `cont_ranges_Banzatti+2025.csv`, `MIRI_general_Banzatti+2025.csv`, `MIRI_H2O_*.csv` | Line-free continuum windows and water line lists of Banzatti, A. et al. 2025, AJ 169, 165 (as distributed with iSLAT) |
| `Atomic_lines.csv` | Standard H I and fine-structure line wavelengths |
