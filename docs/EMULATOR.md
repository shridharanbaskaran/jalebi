# Emulator backend: `fit.model_backend: emulator` (0.18 per-disk tables, 0.21 shared tables)

Since 0.21 the default `fit.emulator.cache: shared` builds the tables **once for a whole survey** and resamples them
onto each disk ([Shared tables (0.21)](#shared-tables-021) below). `cache: per_disk` is the 0.18 behaviour described
first: tables on the data's own pixels, rebuilt per disk and per release.

# Per-disk tables (0.18): `fit.emulator.cache: per_disk`

The exact model costs 10–80 ms per evaluation: a sparse opacity basis on a fine ln λ grid (up to ~470 000
points for 4.9–27.5 µm), then the MIRI LSF + pixel integration of the 12 sub-bands. With the emulator, each
slab's flux is read from a precomputed table instead. A whole ln P costs about 0.1–0.5 ms.

```yaml
fit:
  model_backend: emulator          # exact (default) | emulator
  emulator:
    target_sigma: 0.1              # max |emulator − exact| / σ
    target_flux: 0.001             # max integrated-flux error
    method: cubic                  # cubic (4-point Lagrange per axis) | linear
    cache_dir: null                # null = $JALEBI_EMULATOR_DIR, else ~/.jalebi/emulator
  mcmc:
    vectorize: false               # one ln P call for all walkers (emcee vectorize=True; no process pool)
```

```bash
jalebi emulator build config.yaml          # build / verify the tables a fit of this config will use
jalebi emulator list                       # what is in the cache
jalebi fit config.yaml --backend emulator  # or set fit.model_backend
```

The pipeline builds (or loads) the tables itself when the backend is `emulator`. The fit's products
(`best_fit.json`, `model.csv`, the ΔBIC test, every figure) are always written with the exact model; only the
grid, the optimiser and the MCMC use the tables. The web app has a toggle in the sidebar's Display panel
("fast model sliders"). It builds tables for the components on screen in the background, and the sliders use them.

## What is tabulated

For each emulable unit, the pixel flux of a 1-au-radius slab, F(pixel; T, log N), is tabulated on **the
data's own pixels**: after the LSF and pixel integration of every sub-band, at the component's fixed
v_shift and line width, with the component's own line list (`linelist_key`, e.g. `H2O:hitemp` for hot water,
`H2O:hitran` for cold water) and with its `windows` mask. The area multiplies the table (it is linear), so
`linear: profile / marginalise` work unchanged.

* Interpolated quantity: ln(F + ε) with a per-pixel floor ε = 10⁻⁴ σ / a_max. For thin gas F ∝ N, so ln F is
  linear in log N, and the Boltzmann and partition-function dependence is smooth in ln F.
* Axes: ln T and log N, with 4-point Lagrange (cubic) weights per axis on a non-uniform tensor grid (`linear`
  = bilinear).
* Support: only pixels where some node's flux exceeds 10⁻⁴ σ at the largest allowed area are stored. The rest
  are 0, at an error below 10⁻⁴ σ.
* Peak optical depth: tabulated too, as log τ_max, for the τ flags.

### Adaptive grid

Building starts with 9 × 9 nodes over the reachable box: the prior bounds of the free T and log N, or the fixed
value. A tied isotopologue's box is its parent's log N box shifted by the ratio range. The build then repeats:

1. exact rows at every T-interval midpoint, for each log N node and each cell centre; exact values at every
   log N midpoint, for each T node (one opacity product per T: all log N at a given T cost one sparse product);
2. every interval whose check exceeds the tolerance gets its midpoint inserted;

until no check fails (tolerances = `safety` × targets, default 0.05 σ and 0.05 %). `n_validate` random points
are then compared and recorded in the table (`meta.validation`).

### Error measure

At a check point, the emulated and exact fluxes are compared at **the largest area the data allow**:
a = min(a_max, f_ref / max F_exact), with f_ref the brightest continuum-subtracted pixel of the fit and
a_max the largest area of the prior.

* e_σ = max over pixels of |F_emu − F_exact| · a / σ_pixel. Here σ is the noise of the fit (σ/√weight)
  with noise scale 1.
* e_flux = |Σ(F_emu − F_exact)| / max(Σ F_exact, σ_int / a), with σ_int = √(Σ σ²). That is, the error is
  relative to the integrated flux, or to the 1σ noise of the integrated flux when the flux is weaker than
  that noise. Without this floor, a 0.1 % requirement on an undetectable flux could not be met: 100 K CO at
  5 µm has a peak flux of 10⁻²¹ Jy, and its relative flux error stayed at 10⁸ however fine the grid.
  The report gives the fraction of points where the floor applies.

## What falls back to the exact model

The model mixes emulated and exact units freely; `EmulatorSet.exact_units` and the fit log say why each
exact unit stays exact.

| unit | why exact |
| --- | --- |
| opacity group with several members | the opacities add before 1 − e^−τ: one table dimension per member's log N |
| `kind: annuli` | T(r), N(r), R_in, R_out: 5 dimensions |
| `kind: absorption`, or any screen with `covers: all` | the transmission multiplies the emission on the fine grid |
| `Tvib` set | a third dimension |
| free `rv` or `fwhm` | the table is built at one velocity and width |
| a value outside the table (e.g. the sampler steps past a bound, or a slider in the app) | that call only |

Tied isotopologues are emulated with their own table: the same T as the parent and their own log N.

## Cache

Each table is one `.npz` file (float32 ln F on the support pixels), named after the line list and a hash.
The hash covers everything that changes the answer, so a stale table is never loaded:

- the line list: the content of the lines that reach the opacity basis plus a SHA-256 of the source file;
- the pixel wavelengths, the LSF matrix and the fine grid;
- `LSF_VERSION`, R_model / R_scale / R_constant, the distance, v_shift, the line width and thermal width, the
  component's windows and `eup_max`;
- the (T, log N) box and a_max, the noise vector and f_ref used for the certification;
- the targets, method and node limits, the table format and **the jalebi version** (every release rebuilds).

Identical tables are shared, for example two water components with the same release and bounds. A different
continuum (and so a different σ) gives new tables. With `continuum.refine_iterations`, the tables of the first
continuum serve the grid and the optimiser. The candidate continuum is judged with the exact model, and tables
are built again only when the refined continuum is kept.

## Measured (FZ Tau, one core)

Accuracy: `runs/emulator_report.py accuracy`. The FZ Tau pixel grid covers 4.9–5.35 and 9–27.5 µm, 4430 pixels
in 12 sub-bands, with the FZ_Tau_quick continuum. Each component was checked at 2000 random (T, log N) points
drawn uniformly in ln T and log N inside its bounds. Errors are at the largest area the data allow.

| component | line list | T [K] | log N | nodes T × N | pixels | max / p99 err [σ] | max / p99 flux err | flux < noise | build | MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H2O hot | HITEMP | 600–1500 | 14–21 | 33 × 30 | 4422 | 0.026 / 0.021 | 0.023 / 0.020 % | 0 % | 51 s | 17.6 |
| H2O warm | HITEMP | 250–900 | 14–21 | 33 × 30 | 4422 | 0.031 / 0.018 | 0.023 / 0.020 % | 0.2 % | 50 s | 17.6 |
| H2O cold | HITRAN | 100–400 | 14–21 | 33 × 30 | 4324 | 0.025 / 0.018 | 0.041 / 0.029 % | 15 % | 43 s | 17.2 |
| CO | HITEMP | 100–3000 | 14–21 | 35 × 39 | 561 | 0.036 / 0.022 | 0.055 / 0.024 % | 36 % | 92 s | 3.1 |
| CO2 | HITRAN | 100–1500 | 13–21 | 40 × 58 | 1817 | 0.041 / 0.021 | 0.075 / 0.041 % | 5 % | 181 s | 16.9 |
| ¹³CO2 (tied, ratio 70) | HITRAN | 100–1500 | 11.2–19.2 | 34 × 28 | 1817 | 0.044 / 0.025 | 0.049 / 0.038 % | 20 % | 76 s | 7.0 |
| C2H2 | HITRAN | 100–1500 | 13–21 | 40 × 63 | 1602 | 0.036 / 0.024 | 0.091 / 0.039 % | 4 % | 221 s | 16.2 |
| HCN | HITRAN | 100–1500 | 13–21 | 42 × 62 | 1593 | 0.045 / 0.025 | 0.080 / 0.050 % | 6 % | 247 s | 16.6 |
| OH | HITRAN | 100–3000 | 13–21 | 67 × 47 | 2229 | 0.044 / 0.031 | 0.044 / 0.031 % | 14 % | 255 s | 28.1 |

All 9 pass the 0.1σ and 0.1 % targets. The total is 140 MB, built in 20 min on one core. "flux < noise" is the
fraction of points whose integrated flux at that area is below its own 1σ noise; for those, the flux error is
relative to that noise (cold CO, the coldest OH and water). C2H2 (0.091 %) is the closest to the flux target.

Posterior: `runs/emulator_report.py posterior`. The FZ_Tau_quick fit covers 13.45–17.5 µm, 1674 pixels:
H2O hot and warm, CO2 + ¹³CO2, C2H2 and HCN. It used `linear: profile`, `moves: de`, `init: scaled`, 8000 steps
× 44 walkers, from the same optimum. Exact model: 1652 s (τ_max 231); emulator: 112 s (τ_max 200), with its
tables built in 68 s. Every median of T, log N, log R and log N·A agrees within 0.018 dex and 1.0 K (targets
0.05 dex and 20 K). The posterior widths are, for example, 16 K for hot water T and 0.8 dex for C2H2 log N.
Neither chain reaches 50 τ (the C2H2/HCN corners are slow), so this compares two equally long runs.

Time per ln P (`runs/emulator_report.py timing`, the same FZ_Tau_quick problem, one core, after warm-up):

| | single call | `linear: profile`, single | vectorised, 64 walkers (per walker) | profile, vectorised (per walker) |
| --- | --- | --- | --- | --- |
| exact | 7.8 ms | 8.6 ms | 10.7 ms | 7.8 ms |
| emulator | 0.30 ms | 0.34 ms | 0.18 ms | 0.23 ms |

Speed-ups: 26× single, 34–58× vectorised. On the 4.9–27.5 µm grid (H2O hot + CO2 + ¹³CO2, 467 000 fine-grid
points) the exact model costs 78.7 ms against 0.17 ms with the emulator (460×). The interpolation of one unit
costs ~30 µs, and loading 60 MB of tables takes 0.12 s. Vectorising saves the per-walker Python overhead; the
table gather itself is memory-bound (~38 µs per unit and walker).

# Shared tables (0.21): `fit.emulator.cache: shared`

With per-disk tables, building dominates the survey cost: about 20 min per disk for full-range fits against 2–3 min
of sampling, and every release rebuilds every table (the jalebi version is in the key). The shared tables are built
once, for the whole survey, and every disk resamples them.

```yaml
fit:
  model_backend: emulator
  emulator:
    cache: shared                  # shared (default) | per_disk (0.18)
    ref_snr: 1000                  # certification S/N at build time
    points_per_fwhm: 8             # dense-grid points per LSF FWHM
    table_oversample: 6            # fine-grid points per line FWHM of the build (not fit.oversample, see below)
    spot_check: 200                # random (T, log N) per unit checked on the disk at load (0 = off)
    read_only: false               # true on compute nodes: never build; also JALEBI_EMULATOR_READONLY=1
    cache_dir: null                # $JALEBI_EMULATOR_DIR, else ~/.jalebi/emulator; shared tables in its shared/ subfolder
```

```bash
jalebi emulator build --survey runs/survey_300_autodetect.yaml -j 16    # once: every table the config could need
jalebi emulator build cfg_a.yaml cfg_b.yaml                             # the union of several configs
jalebi emulator list runs/survey_300_autodetect.yaml                    # the cache; what each component gets
bash runs/RUN_ME_build_emulator.sh    /    sbatch runs/slurm_build_emulator.sh
```

## What is tabulated

Not the pixel flux but **H = G_R ⊗ I**, the LSF-convolved spectrum of a 1-au slab at the reference distance
d = 1 pc, on a survey-wide dense rest-frame ln λ grid:

* one uniform segment per MRS sub-band (`instrument._BANDS`; 4C extended to 28.8 µm, where the x1d products end),
  padded on each side by the LSF wings (6σ), |v| ≤ 150 km/s and a 0.3 % margin for the band-edge slop of real
  data (2C starts at 10.01 µm, 4C ends at 28.70);
* step 1/(`points_per_fwhm` × max R) per segment: 8 points per LSF FWHM by default, 48 881 points for the 12 bands
  with R = argyriou2023 (against ~4400 pixels of a full-range FZ Tau fit);
* `build_dense_lsf_operator` (next to `instrument.build_lsf_operator`): a point-sampled Gaussian of
  σ = FWHM_TO_SIGMA / R(λ) at each dense point, **no pixel box**, truncated at 6σ (a point-sampled kernel cut at 4σ
  leaves a 6e-5 step that the pixel integration of the exact model smooths over). For `jones2023` each band's own
  linear R relation is used across its padded segment, so a segment has no jump in R.
* The build is the 0.18 adaptive build (`build_unit_table`) on the dense model, with the node rows kept in float32,
  the exact rows computed in column chunks, and a failing cell centre refining only the axis whose own 1-D error is
  larger (`refine: axis`; the 0.18 rule refines both and doubled the log N nodes while the T interpolation was the
  culprit). The fine grid of the build has `table_oversample` points per line FWHM (default 6) over the padded
  segments: 684 594 points for the full range. Nodes, support points, ε and log τ_max are stored as in 0.18.

## Per-disk operator

`DenseGrid.pixel_operator` builds a sparse P_disk (npix × ndense) that maps the dense grid to the disk's pixels:

* the average over each pixel's edges — the same edges `build_lsf_operator` uses (`instrument.pixel_edges` of the
  pixel array, or the data's own) — shifted by −rv/c in ln λ, which reproduces `SlabModel._shift` (shift the
  spectrum = shift the pixel);
* of the cubic (4-point Lagrange) interpolant of H across the dense points, integrated with 3-point Gauss–Legendre
  on every sub-segment (exact for a cubic), so P_disk is linear in H and every row sums to 1;
* using the dense segment of the pixel's sub-band, the first band whose nominal range contains the wavelength —
  the rule `instrument.resolving_power` uses for the exact model, so overlap pixels get the same R;
* clipped to the fit's own fine grid: the exact model integrates a pixel only over the fine points that exist and
  divides by the full pixel width, so a pixel next to a gap between fit windows (its edges reach the gap's midpoint)
  is diluted. P_disk reproduces that with the clipping; see "uncovered pixels" below.

Pixel flux = P_disk @ H(T, log N) × (1 pc/d)² × the component's window mask, applied per disk. At load time
`project_table` computes the projected table on the disk's pixels once (P_disk restricted to the table's support,
one sparse product per T node) and wraps it in a 0.18 `UnitTable` with the disk's own ε and support, so the cost
per ln P is the 0.18 cost. The projection takes 4 s per disk for the 6 units of the FZ_Tau_quick fit and 21 s for the 8 full-range units of FZ Tau (the 1.7 GB water table: 4–5 s each time it is used).

## The approximations, measured

`runs/emulator_report.py approximations`: a smooth spectrum of 300 Gaussian lines (FWHM 4.7 km/s) in band 3B
on a 1.7e-4 ln λ pixel grid, P_disk @ H against K @ shift(I) of the exact model (the same fine grid), as a fraction
of the brightest pixel. (a) is the shift applied after the convolution: the kernel is R at the rest wavelength
instead of the observed one (ΔR = 128 λ v/c for argyriou2023, 3e-4 of σ at 150 km/s); (b) is R at each dense
point instead of at the pixel centre; (c) the cubic resampling and integration.

| points per FWHM | (c) resampling, max | (c) flux | (b) + R at the dense points | (a)+(c) shift 150 km/s, R const | all, shift 150 km/s, argyriou | e_σ at S/N 100 (all) |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 3.2e-03 | 7.4e-06 | 2.2e-03 | 2.9e-03 | 2.0e-03 | 0.201 |
| 6 | 5.8e-04 | 1.2e-05 | 5.1e-04 | 5.3e-04 | 3.0e-04 | 0.030 |
| 8 | 2.0e-04 | 1.2e-05 | 1.7e-04 | 1.5e-04 | 1.7e-04 | 0.017 |
| 10 | 7.6e-05 | 1.3e-05 | 6.6e-05 | 6.1e-05 | 2.5e-04 | 0.025 |
| 12 | 3.9e-05 | 1.3e-05 | 3.1e-05 | 3.3e-05 | 2.8e-04 | 0.028 |
| 16 | 2.6e-05 | 1.3e-05 | 2.8e-05 | 3.7e-05 | 3.0e-04 | 0.030 |

| v_shift [km/s] (8 points per FWHM) | all, argyriou2023 | R constant |
| --- | --- | --- |
| +0 | 1.7e-04 | 2.0e-04 |
| +37 | 1.1e-04 | 1.6e-04 |
| +75 | 7.5e-05 | 1.7e-04 |
| +150 | 1.7e-04 | 1.5e-04 |
| -150 | 3.9e-04 | 1.8e-04 |

Kernel truncation at 8 points per FWHM: 4σ gives 2.0e-04, 6σ 2.0e-04 (the floor at 12–16 points per FWHM moves from 6e-5 to 1e-5 with 6σ, which is why the dense kernel is cut at 6σ).

(c) scales as h⁴; (a) is a floor of ~3e-4 at |v| = 150 km/s that `points_per_fwhm` does not change, and it is zero
for a constant R (and in the survey, where the heliocentric velocity is removed before the fit and the components'
rv is 0). At 8 points per FWHM the resampling error is 2.0e-4 of the peak, i.e. 0.02σ for a disk whose brightest
pixel has S/N 100 (FZ Tau, the brightest disk of the validation set: S/N 98 over 4.9–27.5 µm) and 0.0012 % in
flux — the default was chosen for that. With |v| = 150 km/s the total is 0.04σ at S/N 100. For data with a
brighter line peak raise `points_per_fwhm` to 10–12 (the table grows in proportion); the spot check tells.

## Certification without a disk's noise

At build time every node is scaled so that its brightest dense point equals the reference peak f_ref = 1 (the 0.18
rule a = min(a_max, f_ref / peak F_exact) with a_max effectively infinite), and the emulator is compared with the
exact model against σ_ref = f_ref / `ref_snr` at every dense point, with the 0.18 error measure (`errors`): e_σ is
the pointwise error in units of σ_ref, e_flux the integrated-flux error relative to max(flux, its 1σ noise). With
`ref_snr: 1000` the tables hold the pointwise error below 1e-4 of each node's peak (build tolerance 0.5 × 0.1σ), which
meets the 0.1σ target for any disk whose brightest pixel has S/N ≤ 1000 (then e_σ ≤ 1e-4 × f_ref / σ_pixel ≤ 0.1).
This is 10–20× stricter than a 0.18 build on FZ Tau (S/N 45–100), and the node count grew accordingly only once
two sources of 1e-4-level noise in the "exact" model were removed: the Planck cache rounded T to 0.01 K (a 3e-5
jitter of B_ν; now 1e-4 K) and the cell-centre rule refined both axes.

At fit time `spot_check` draws ~200 random (T, log N) per unit (20 temperatures × 10 columns, one opacity product
per T) and compares the projected table with the exact model on the disk's pixels and σ, at the disk's f_ref and
a_max. A unit above 0.1σ or 0.1 % falls back to the exact model, with a warning, a log line and
`diagnostics.json["emulator"]["fallback"]`; the check's numbers and time go to `["spot_check"]`. The spot check
costs 4 s for the FZ_Tau_quick set and 27 s for the 8 full-range units (the exact model at the table's oversample is most of it).

**Oversample.** The tables are built at `table_oversample: 6`, not at `fit.oversample`: at 3 points per line FWHM
the exact model's saturated line profiles depend on the phase of its fine grid at the 1e-3 level (0.4–0.6 % in
flux on the synthetic 3B test), which no shared table can reproduce. The spot check is therefore made against the
exact model at the table's oversample (built once per attach when it differs from the fit's); when
`fit.oversample` differs, the discrepancy against the fit's own exact model is measured too and reported as
`spot_check[unit]["vs_fit_oversample"]` without pass/fail. The survey config uses `fit.oversample: 3` for the
products; with the emulator the exact model only writes the products, so 6 costs little there.

**Saturated lines.** The first spot checks on synthetic grids failed by 1–2σ at log N ≈ 21 while the tables
were certified to 1e-4: the exact model itself was not reproducible there. Its Gaussian line profiles were cut at
4σ, where τ is still ≫ 1 for log N ≳ 20, so the flux of a saturated line depended on where the cut landed on the
fine grid (0.5 % between two grid phases, 0.7 % between oversample 6 and 12 for CO2 at 254 K, log N 21). 0.21
evaluates the profiles to 6σ (`OpacityBasis.LINE_TRUNCATE`): the phase sensitivity drops to 2e-5, the oversample
sensitivity to 1e-4, thick-line fluxes grow by 1–3 % at log N 21 and < 0.03 % at log N 19, and thin gas is
unchanged. Both cache keys carry the value.

**Sub-band junctions.** The first and last pixel of every sub-band had no model flux at all until 0.20: the
midpoint pixel edges were computed on the concatenated pixel array, so across a junction (3B ends at 15.568 µm,
3C starts at 15.410) they were inverted and the exact model's K row was empty. The shared tables gave those
pixels their proper average, which disagreed with the "exact" model by 2.7σ (data 0.13 Jy, model 0) and shifted
the flat C2H2 log N posterior of the FZ_Tau_quick fit by 0.1 dex. 0.21 takes the edges within each run of
increasing wavelength (`instrument.pixel_edges`), so both backends agree there; `DenseGrid.pixel_operator` still
reproduces the old behaviour for inverted edges handed to it explicitly.

**Masked regions and line lists.** Two more exact-model inconsistencies surfaced the same way (a shared table
can only match a model that is a function of the physics, not of the fit's bookkeeping): `build_lsf_operator`
took the fine points within x_pix ± (4σ + width/2), so a pixel next to a masked region lost the part of its
box beyond that window (K row sums 0.5–0.6, the model 40 % low, in the OH-prompt-masked 9–13 µm region of FZ
Tau; now the whole box [lo − 4σ, hi + 4σ]); and the line-list strength cut was relative to the strongest line
inside the fit window and skipped for short lists, so the model of a molecule depended on the window at the
1e-7 opacity level — 10 % of a saturated line at log N 21, 0.3σ at S/N 250 (now `model.prune_linelist`: one
rule, relative to the strongest line in the MRS range, at 100 K, 1500 K and the molecule's upper T bound).

**Uncovered pixels.** A pixel next to a gap between fit windows has K-row sum < 0.5 in the exact model (its
model flux is diluted by the gap-spanning width; `SlabModel.covered` is False), and its value also depends on where
the exact model's fine grid stops — not a property of the convolved spectrum. These pixels are left out of the spot
check and of the accuracy protocol (`pixels_excluded`); their model value is an artefact in both backends (open
issue: cap the edge-pixel width, or give those pixels weight 0).

## Cache key, storage, concurrency

The file name is `<linelist>_<physics key>_<full key>` with 16-hex keys of the JSON written next to each table:
`EMULATOR_MODEL_VERSION` (bumped by hand only when the physics or the format changes), `LSF_VERSION`, molecule +
`linelist_key`, the line-list content hash (the arrays that reach the opacity, selected over the whole dense range
and pruned as `build_model` does) + the file's SHA-256, `eup_max`, the intrinsic `fwhm` and `fwhm_thermal`,
R model / scale / constant, `table_oversample`, the dense-grid spec (bands and ranges, padding, points per FWHM,
truncation), the (T, log N) box, `ref_snr`, the targets, method and node limits. **Not** in the key: the jalebi
version, the disk's pixels or K, v_shift, distance, windows, σ or f_ref — so one file serves every disk and every
release. The physics key leaves the box out: a disk whose bounds lie inside a wider table's box uses that table
(the survey build may use the union of several configs).

Boxes: `molecule_boxes(cfgs)` gives one survey-wide (T, log N) box per molecule: the union over configs of each
component's bounds (its own, else `fit.bounds_by_molecule`, else the default prior box) and, with
`fit.auto_detect`, of `fit.bounds_by_molecule` (else the default box) for every candidate; a tied isotopologue's log N
box is the parent's shifted by its ratio (fixed value, or the ratio bounds, default 10–300); candidate isotopologues
get the parent's box shifted by the default ratio range. `EmulatorConfig.settings(cfg)` passes the boxes to the fit,
so a fit of the survey config looks up exactly the tables `jalebi emulator build` of that config wrote. A unit
whose bounds fall outside its box stays exact, with the reason in the log and in `exact_units`.

Storage: `<cache_dir>/shared/<name>.npy` (float32 ln(H + ε) on the support points, uncompressed, loaded with
`mmap_mode="r"` so that processes on one node share the pages), `<name>.npz` (nodes, support, ε, log τ_max) and
`<name>.json`. Writes go to `<name>.tmp<pid>.*` and `os.replace`. `TableLock` (`fcntl.flock` on `<name>.lock`,
with a polled O_EXCL fallback) makes exactly one process build a table while the others wait and then load it
(`tests/test_emulator.py::test_lock_two_processes_build_once`). `read_only` (or `JALEBI_EMULATOR_READONLY=1`) on
compute nodes: a missing table is an exact fallback plus a warning, never a build.

What stays exact is as in 0.18 (groups, annuli, screens, covers=all, T_vib, free rv / fwhm, points outside a
table), plus: bounds outside the survey box, a missing table in read-only mode, a failed spot check, and pixels
outside the dense grid.

## Measured

All numbers from the sandbox (two slow cores; its exact model costs 15 ms per call where the 0.18 report's
machine took 7.8 ms, so wall times here are about 2× the user's). `runs/emulator_report.py` subcommands in brackets.

### One-time build (`jalebi emulator build --survey examples/configs/FZ_Tau_quick.yaml` + CO 100–3000 K)

| table | box T [K] | box log N | lines | nodes T × N | support points | MB | build (wall, one core) | certification max e_σ (ref S/N 1000) | max e_flux |
|---|---|---|---|---|---|---|---|---|---|
| H2O:hitemp | 100-1500 | 13.00-21.00 | 95323 | 117 × 76 | 48621 | 1729 | 112 min | 0.045 | 0.0007 % |
| CO:hitemp | 100-3000 | 13.00-21.00 | 3056 | 129 × 74 | 20418 | 780 | 99 min | 0.022 | 0.0017 % |
| CO2:hitran | 100-1500 | 13.00-21.00 | 20281 | 107 × 71 | 8267 | 251 | 7 min | 0.045 | 0.0025 % |
| 13CO2:hitran | 100-1500 | 11.15-19.15 | 12113 | 56 × 48 | 8260 | 89 | 3 min | 0.039 | 0.0025 % |
| C2H2:hitran | 100-1500 | 13.00-21.00 | 38297 | 106 × 70 | 13259 | 394 | 22 min | 0.041 | 0.0014 % |
| HCN:hitran | 100-1500 | 13.00-21.00 | 10040 | 119 × 65 | 7551 | 234 | 7 min | 0.036 | 0.0025 % |
| **total** | | | | | | **3476** | **4.2 h core time** (wall 1.9 h in parallel) | | |

Memory-mapped `.npy` files; a process touches only the pages it projects. The 15 tables of
`runs/survey_300_autodetect.yaml` (the six above plus 13CO, 13CCH2, H13CN, CH4, NH3, C2H4, C2H6, C4H2, HC3N) are an
estimated 6–8 core-hours here, i.e. 3–4 on a modern core, with the H2O table as the wall-clock critical path
(`-j 16`: about 2 h). The node counts are 2–4× those of the 0.18 per-disk tables because the certification is 10–20×
stricter (ref S/N 1000 against FZ Tau's 45–100); `ref_snr: 300` would give tables 3–4× smaller and faster to build.

### Accuracy, FZ Tau, 4.9–5.35 + 9–27.5 µm, 4430 pixels, 2000 random points per component (`accuracy --cache shared`)

Errors at the largest area the data allow, against the exact model on FZ Tau's pixels and noise (S/N of the
brightest pixel 98). The cold water component is on HITEMP (the survey's release for every H2O component: one table
serves hot, warm and cold); OH is not a survey candidate and was left out (its 100–3000 K table takes ~2 h).

| component | line list | T [K] | log N | pixels | max / p99 e_σ | max / p99 e_flux | flux < noise | spot check (σ / flux) | project + spot check |
|---|---|---|---|---|---|---|---|---|---|
| H2O_hot | H2O:hitemp | 600-1500 | 14.00-21.00 | 4430 | 0.012 / 0.010 | 0.0022 / 0.0021 % | 0 % | 0.011 / 0.0022 % | 5.6 + 3.3 s |
| H2O_warm | H2O:hitemp | 250-900 | 14.00-21.00 | 4430 | 0.009 / 0.009 | 0.0022 / 0.0021 % | 0 % | 0.010 / 0.0021 % | 4.1 + 3.0 s |
| H2O_cold | H2O:hitemp | 100-400 | 14.00-21.00 | 4430 | 0.015 / 0.014 | 0.0021 / 0.0019 % | 15 % | 0.014 / 0.0019 % | 5.1 + 3.0 s |
| CO | CO:hitemp | 100-3000 | 14.00-21.00 | 1020 | 0.005 / 0.004 | 0.0020 / 0.0016 % | 36 % | 0.005 / 0.0016 % | 2.5 + 3.4 s |
| CO2 | CO2:hitran | 100-1500 | 13.00-21.00 | 1822 | 0.005 / 0.004 | 0.0026 / 0.0023 % | 5 % | 0.005 / 0.0024 % | 1.0 + 3.2 s |
| 13CO2 | 13CO2:hitran | 100-1500 | 11.15-19.15 | 1821 | 0.004 / 0.003 | 0.0070 / 0.0053 % | 20 % | 0.004 / 0.0068 % | 0.4 + 3.3 s |
| C2H2 | C2H2:hitran | 100-1500 | 13.00-21.00 | 1606 | 0.004 / 0.003 | 0.0023 / 0.0019 % | 4 % | 0.004 / 0.0021 % | 1.8 + 4.2 s |
| HCN | HCN:hitran | 100-1500 | 13.00-21.00 | 1597 | 0.005 / 0.004 | 0.0068 / 0.0023 % | 6 % | 0.004 / 0.0021 % | 0.9 + 3.3 s |

All 8 pass with a margin of 7–25 in e_σ and 15–50 in e_flux: max 0.015σ (p99 0.014σ) and 0.007 % in flux, against
0.045σ / 0.091 % for the 0.18 per-disk tables. "flux < noise" is the fraction of points whose integrated flux is below
its 1σ noise at that area (cold CO, 13CO2, cold water), where e_flux is relative to that noise. `pixels_excluded` = 2
on every row: the two pixels next to the 5.35 / 9.0 µm gap between the fit windows (see "uncovered pixels").

### Accuracy on other pixel grids (`grids`)

Synthetic spectra (`jalebi.synthetic`) with other sub-bands, velocity, distance and S/N, 500 random points per
component, the same shared files:

| grid (bands, v_shift, distance, S/N_max) | component | T [K] | max / p99 e_σ | max / p99 e_flux | flux < noise | spot check σ |
|---|---|---|---|---|---|---|
| synthetic_1A3B (1A+3B, +15 km/s, 140 pc, 234) | H2O | 100-1500 | 0.045 / 0.041 | 0.0016 / 0.0014 % | 24 % | 0.045 |
| synthetic_1A3B (1A+3B, +15 km/s, 140 pc, 234) | CO2 | 100-1500 | 0.020 / 0.014 | 0.0040 / 0.0024 % | 1 % | 0.016 |
| synthetic_1A3B (1A+3B, +15 km/s, 140 pc, 234) | HCN | 100-1500 | 0.027 / 0.024 | 0.0046 / 0.0028 % | 2 % | 0.023 |
| synthetic_1A3B (1A+3B, +15 km/s, 140 pc, 234) | CO | 100-3000 | 0.045 / 0.041 | 0.0017 / 0.0016 % | 34 % | 0.045 |
| synthetic_2C3C4A (2C+3C+4A, -60 km/s, 90 pc, 276) | H2O | 100-1500 | 0.075 / 0.072 | 0.0018 / 0.0017 % | 12 % | 0.072 |
| synthetic_2C3C4A (2C+3C+4A, -60 km/s, 90 pc, 276) | CO2 | 100-1500 | 0.064 / 0.060 | 0.0028 / 0.0024 % | 7 % | 0.065 |
| synthetic_2C3C4A (2C+3C+4A, -60 km/s, 90 pc, 276) | HCN | 100-1500 | 0.069 / 0.068 | 0.0049 / 0.0022 % | 20 % | 0.068 |

CO on the 2C–4A grid stays exact by the spot check (0.06σ, but 0.55 % in flux): CO has two lines there, 1e-7 of
its 5 µm band, which sit at the support threshold of the table; at the largest allowed area (R = 31 au) they are
scaled to the brightest pixel. A molecule with no band inside the fit windows should not be in the fit; when it is,
the exact model takes it at no cost.

### Invariance (`invariance`)

FZ Tau (4430 pixels, rv 0, 130 pc) and a synthetic disk on 1A + 3B + 4A (1575 pixels, rv −45 km/s, 200 pc) read
**the same file** for H2O, CO, CO2 and HCN (equal keys), with spot checks of 0.004–0.035σ and ≤ 0.007 % in flux.

### Posterior, FZ_Tau_quick (`posterior --mode all`, 13.45–17.5 µm, 1674 px, `linear: profile`, `moves: de`,
`init: scaled`, 8000 steps × 40 walkers from the same optimum; 3A–3C-band tables)

| backend | MCMC wall | tables | τ_max | vs exact: pass (0.05 dex / 20 K) | max Δ dex | max Δ K |
|---|---|---|---|---|---|---|
| exact | 2826 s | — | 244 | — | — | — |
| per-disk (0.18) | 100 s | 62 s build | 247 | 20/20 | 0.011 | 1.5 |
| shared (0.21) | 93 s | 490 s (one-time build of the 3A–3C set) | 186 | 20/20 | 0.027 | 4.9 |

Shared vs per-disk: 15/20 within 0.01 dex / 5 K, max 0.027 dex and 3.3 K — the misses
are CO2 / C2H2 / HCN log N, whose posteriors are 0.3–0.8 dex wide and whose medians move by 0.01–0.03 dex from seed
to seed. Three seeds per emulated backend show every backend inside the seed scatter of the others:

| parameter | exact | per-disk (3 seeds) | shared (3 seeds) | σ_post (exact) |
|---|---|---|---|---|
| H2O_hot.logN | 18.774 | 18.772–18.773 | 18.773–18.773 | 0.048 |
| H2O_hot.T | 924 | 924–925 | 924–924 | 17 |
| H2O_warm.logN | 18.558 | 18.559–18.560 | 18.559–18.560 | 0.044 |
| H2O_warm.T | 506 | 506–506 | 506–506 | 5 |
| CO2.logN | 17.261 | 17.261–17.275 | 17.262–17.288 | 0.314 |
| CO2.T | 506 | 502–505 | 501–505 | 47 |
| C2H2.logN | 18.580 | 18.585–18.591 | 18.579–18.586 | 0.786 |
| C2H2.T | 316 | 316–317 | 317–320 | 61 |
| HCN.logN | 18.271 | 18.263–18.283 | 18.254–18.267 | 0.343 |
| HCN.T | 286 | 286–286 | 286–287 | 40 |
| H2O_hot.logNA | 18.095 | 18.092–18.095 | 18.093–18.094 | 0.058 |
| H2O_warm.logNA | 18.838 | 18.839–18.840 | 18.839–18.839 | 0.038 |
| CO2.logNA | 16.495 | 16.497–16.505 | 16.496–16.515 | 0.167 |
| C2H2.logNA | 17.193 | 17.190–17.201 | 17.169–17.196 | 0.673 |
| HCN.logNA | 17.676 | 17.669–17.684 | 17.662–17.671 | 0.458 |

(The first run of this comparison, before the three exact-model fixes above, shifted C2H2 log N by 0.1 dex: the
junction pixels with zero model flux.)

### Timings (`timing --cache both`, FZ_Tau_quick problem, one core, after warm-up)

| | single call | profile, single | vectorised, 64 walkers (per walker) | profile, vectorised (per walker) | attach per disk |
|---|---|---|---|---|---|
| exact | 15.14 ms | 16.67 ms | 17.86 ms | 19.59 ms | — |
| per-disk tables (0.18) | 0.40 ms | 0.49 ms | 0.31 ms | 0.39 ms | 72 s (build 71 s) |
| shared tables (0.21) | 0.44 ms | 0.56 ms | 0.34 ms | 0.46 ms | 10.7 s (load 0.3, project 4.1, spot check 4.2) |

The shared tables cost 5–15 % more per call than the per-disk ones (more nodes per axis, a larger gather) and
save the 71 s per-disk build; attach (load + project + spot check) is 10.7 s for the 6 units of the quick fit and
49 s for the 8 full-range units of FZ Tau (projection 21 s, spot check 27 s, of which the exact model at
the table's oversample is most). Projection of the 1.7 GB water table onto 4430 pixels: 4–5 s.

### Projected cost of the 300-disk survey (`survey`)

With the measured per-disk numbers (load + projection 22 s, spot check 27 s, emcee profile 8000 steps × 48
walkers at 0.46 ms per walker-call vectorised, prep 1 min, optimiser 1 min, products 1.5 min) in this sandbox's units:

| setup | one-time | per disk | 300 disks |
|---|---|---|---|
| shared tables (0.21): one build + 300 x (load + spot check + emcee profile 8000 steps) | 3.4 core-h | 7.3 min | **40 core-h** |
| per-disk tables (0.18): 300 x (build ~20 min + ...) | 0.0 core-h | 26.5 min | **132 core-h** |
| exact model: 300 x (optimiser ~6 min + 384 000 calls x exact ln P) | 0.0 core-h | 133.9 min | **669 core-h** |

In the units of docs/SAMPLER_BENCHMARK.md (that machine is ~2× faster) the shared setup is **~20 core-hours for 300
disks plus a 2–4 core-hour build**, against the 50–75 core-hours of the 0.18 per-disk setup and 350–450 for the exact
model. The per-disk emulator cost is no longer the build but the sampling itself.

