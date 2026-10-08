# Local line lists (0.22): `jalebi linelist import`

HITRAN has no line-by-line list for benzene (C6H6) or propyne (C3H4); it carries benzene only as absorption
cross-sections.  jalebi's `molecules.py` knows both (HITRAN id 0), so a slab model needs a local list in the
cache layout jalebi reads: `<cache>/<MOL>_<release>.parquet` (columns `wave, nu, a, gu, gl, eu, el, vup, vlow,
qup, qlow, iso`) plus `<MOL>_<release>_Q.npz` (the partition table `T, Q`).  The cache is
`linedata.data_dir`, `$JALEBI_DATA` or `~/.jalebi/linedata`.

```
jalebi linelist import --molecule C6H6 --file c6h6_lines.csv --format csv --release geisa2020 \
        --partition c6h6_Q.csv
jalebi linelist import --molecule C6H6 --file c6h6.par --format hitran160 --partition levels
jalebi linedata list
```

then in a config: `{name: C6H6, molecule: C6H6, linelist_release: geisa2020, ...}` or
`linedata: {releases: {C6H6: geisa2020}}`.

Formats (`--format auto` picks by extension / header):

* `hitran160` — the 160-character `.par` records of HITRAN / HITEMP (positions, intensities, Einstein A, lower
  energies, degeneracies); the molecule id is not checked for molecules with HITRAN id 0.
* `islat` — the processed lists distributed with iSLAT (header with `Number of Partition` block and
  `Number of lines`; the partition table in the file is used).
* `csv` — a CSV or whitespace table with `wave` [µm] or `nu` [cm⁻¹]; `eu` or `el` [K] (or `eu_cm1` / `el_cm1`
  in cm⁻¹); `gu` (and optionally `gl`); and either the Einstein A `a` [s⁻¹] or the HITRAN-style 296 K intensity
  `sw` [cm⁻¹/(molecule cm⁻²)], which is converted with
  A = S · 8πcν² · Q(296) · exp(c₂E_l/296) / (g_u (1 − exp(−c₂ν/296))) (Šimečková et al. 2006; `--q296` or a
  partition file covering 296 K supplies Q(296)).

Partition function (`--partition`):

* a file with two columns `T, Q` (CSV or whitespace, `#` comments) — the recommended path;
* `levels` — Q(T) = Σ g_i exp(−E_i/T) over the distinct levels of the list.  Only as complete as the list: it
  underestimates Q at temperatures where levels that no line in the list touches are populated; it is written
  with a warning in the import message.  Use it when nothing better exists, and do not compare column densities
  obtained with it to ones from a published Q;
* nothing — the list's own table (iSLAT), TIPS through HAPI for molecules with a HITRAN id, else `levels`.

## Where to get a benzene list for the MIRI range

What the JWST papers did — the source I could verify, not a recommendation of a particular file:

* Tabone et al. (2023, Nature Astronomy 7, 805; J160532) state in their Methods that the slab-model parameters
  came from HITRAN 2020 "except for C6H6, for which the molecular parameters were provided based on the GEISA
  database" (GEISA 2020, Delahaye et al. 2021), that the GEISA entry lacks Einstein A coefficients and
  statistical weights, which they generated, that the partition function came from the empirical equations of
  Dang-Nhu & Plíva (1989; checked to 0.5 % for 50–500 K), and that empirical hot-band lists were built from
  high-resolution cross-section measurements.  The Data Availability statement says the spectroscopic data for
  all species except benzene are on HITRAN.
* Kanwar et al. (2024, A&A 689, A231; Sz 28) take their spectroscopic data from HITRAN and, for C3H4 and C6H6,
  "the partition functions, Einstein coefficients and degeneracies from" Arabhavi et al. (2024), citing
  Delahaye et al. (2021) for the C3H4 and C6H6 lines.

So the line positions and 296 K intensities are in the GEISA 2020 database (public, registration at the AERIS
data portal), and a usable list is what Arabhavi et al. (2024) derived from them (Einstein A, degeneracies,
Dang-Nhu & Plíva partition function, hot bands) — ask the MINDS authors for that processed list, or rebuild it:
GEISA positions and intensities → `csv` with `sw`, `gu` and `el`, a Q(T) table from the Dang-Nhu & Plíva
equations (I have not reproduced those equations here; take them from the paper), `--q296` from the same table.
Without the hot bands the ν4 Q branch at 14.85 µm is reproduced but the band's pseudo-continuum at higher T is
not.  I did not find a ready-made, publicly downloadable C6H6 line list in jalebi's layout; if one exists I was
not able to confirm it.

Sources: [Tabone et al. 2023 (arXiv:2304.05954)](https://arxiv.org/pdf/2304.05954),
[Kanwar et al. 2024, A&A](https://www.aanda.org/articles/aa/full_html/2024/09/aa50078-24/aa50078-24.html).
