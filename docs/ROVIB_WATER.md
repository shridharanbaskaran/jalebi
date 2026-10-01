# The 5–8 µm ro-vibrational water band, T_vib, and how the published fits are made

## The problem

A two-component LTE slab fit of the pure-rotational water lines (12–27 µm) that reproduces the
published temperatures over-predicts the ν₂ ro-vibrational band at 5–8 µm by a factor 3–6 when the same
parameters are extrapolated there (CI Tau: data/model ≈ 0.34 at 5–6.5 µm, 0.17 at 6.5–8 µm; the
`H2O_hot` component has τ ≈ 300 in those lines, so it radiates at B_ν(T) over the whole band).
Forcing one LTE model to fit 5–8 µm *and* 12–27 µm together produces the familiar pathological
solution: a hot component at the T upper bound with log N ≈ 17 (optically thin, to weaken the band)
and a "cold" component at ≈ 280 K with log N ≈ 21 — exactly what the blind validation run of CI Tau
returned.

This is not a bug. It is well documented:

* Banzatti et al. 2025 (AJ 169, 165, water atlas, Sect. 3.1.4, Fig. 7): the ro-vibrational lines have
  critical densities ~10¹³ cm⁻³ (10⁸–10¹¹ for the v = 0–0 lines) and are sub-thermally excited;
  for CI Tau the 5–8 µm band needs a factor ≈ 4 reduction with respect to the rotational-line model,
  ≈ 1/6 for v = 1–0 lines and ≈ 1/3 for v = 1–1 hot-band lines (which fall at 10–17 µm, *inside* the
  rotational fit range); even then v = 2–1 lines stay over-predicted.
* Pontoppidan et al. 2024 (ApJ 963, 158, FZ Tau, Sect. 4.3): "the ro-vibrational bending-mode band of
  water around 6 µm is excluded from the fit because it is known to be affected by non-LTE effects … the
  line strengths of the rovibrational water lines are ×4 weaker than those predicted by fits to the
  rotational spectrum".
* Romero-Mirza et al. 2024 (ApJ 975, 78, Sect. 3): only 12.0–27.0 µm is modelled "since emission at
  shorter wavelengths is dominated by high-energy ro-vibrational H₂O lines likely emitting from hot and
  diffuse disk regions where water vapor is not thermalized".
* Banzatti et al. 2023 (ApJL 957, L22): 12–27 µm only; the 5–9 µm band "will be reported in a future
  paper" in terms of non-LTE excitation.
* Gasman et al. 2023 (A&A 679, A117, Sz 98) and Temmink et al. 2024 (A&A 689, A330, DR Tau II) fit the
  band **separately**: Sz 98 5–6.5 µm → T = 950 K, N = 3.7e18, R = 0.07 au versus 13.6–16.3 µm →
  650 K, 7.9e18, 0.28 au; DR Tau 5.0–6.5 µm → 1000 K, log N 18.4, R 0.21 au and 7.0–9.5 µm → 1000 K,
  log N 18.3, 0.21 au, versus the ≥ 10 µm three-component fit 800/470/180 K.
* Meijerink et al. 2009 (ApJ 704, 1471) predicted the effect: vibrational levels are not thermalised
  at the densities of the water-emitting layer.

So the published practice is one of: (a) exclude 5–9 µm, (b) fit 5–9 µm with its own slab, or (c) scale
the LTE rovib band by an empirical factor.  jalebi 0.13 offers (b) and a physical version of (c).

## What jalebi offers

### 1. A vibrational temperature per component (`Tvib`)

```yaml
components:
  - {name: H2O_hot, molecule: H2O, T: 900, logN: 18.3, logR: -0.6, Tvib: 600}   # Tvib set -> fitted
fit:
  windows: [[5.3, 8.0], [12.0, 27.0]]
  tvib_below_trot: true          # prior T_vib <= T (default)
```

Level populations are Boltzmann at `T` within each vibrational state and at `Tvib` between states,

    n_i / N = g_i exp(−E_vib,i / T_vib − E_rot,i / T) / Z(T, T_vib),   Z = Q(T) Z_vib(T_vib) / Z_vib(T),

the standard two-temperature approximation used for CO and CO₂ fundamentals.  The vibrational energies
come from the HITRAN/HITEMP global quanta labels (`LineList.vib_energies()`: the lowest level found with
each label; for water this recovers the band origins 010 = 1594.7, 020 = 3151.6, 100 = 3657.0,
001 = 3755.9 cm⁻¹ to < 0.1 cm⁻¹; unassigned HITEMP labels stay in LTE).  Each line then has its own
opacity ∝ (x_l − x_u) and source function S_l = (2hν³/c²)/(x_l/x_u − 1); on the fine grid overlapping
lines are combined with the opacity-weighted source function, I = (Σ τ_l S_l / Σ τ_l)(1 − e^−τ).
Consequences:

* pure-rotational lines are unchanged (apart from the renormalised partition function);
* v = 1–0 lines are weakened mainly through the source function (optically thick case) and through the
  (010) population (thin case): ∝ exp(−E_010 (1/T_vib − 1/T)) — 0.135 for T = 915, T_vib = 500 K;
* v = 1–1 hot-band lines at 10–17 µm are weakened through their population, by the same factor in the
  thin limit — which is why Banzatti et al. find different suppression for v = 1–0 and v = 1–1 lines;
* `Tvib = T` reproduces LTE to 1e-5; weak cross-band lines that would invert are dropped.

`Tvib` is a free parameter when set (bounds `Tvib: [100, 1500]`, prior T_vib ≤ T), appears in the
corner plots, the app's component card (checkbox *vibrational temperature*) and the YAML.  Annuli
components keep T_vib/T_rot constant along the radius; tied isotopologues inherit the parent's T_vib.

### 2. Per-component wavelength windows (`windows`)

```yaml
components:
  - {name: H2O_rovib, molecule: H2O, T: 950, logN: 18.3, logR: -0.8, windows: [[4.9, 9.5]]}
  - {name: H2O_hot,   molecule: H2O, T: 900, logN: 18.3, logR: -0.6, windows: [[9.5, 28]]}
  - {name: H2O_warm,  molecule: H2O, T: 450, logN: 18.2, logR: -0.2, windows: [[9.5, 28]]}
  - {name: CO,        molecule: CO,  T: 1300, logN: 18.0, logR: -0.9, windows: [[4.9, 5.7]]}
fit:
  windows: [[5.3, 8.0], [12.0, 27.0]]
```

A component with `windows` contributes only there, so the separate-region fits of Gasman+2023 /
Temmink+2024 become one simultaneous run (one continuum, one noise model, one MCMC with the
cross-correlations in the corner plot).  The grid stage grids such a component on its own windows.

**This split is the default.** `fit.water_split_um: 9.5` makes every H₂O slab that has neither its own
`windows` nor a `Tvib` emit only beyond 9.5 µm whenever the fit windows reach below it, and a component
whose name contains "rovib" only below it; tied isotopologues follow their parent. So the plain config

```yaml
components:
  - {name: H2O_rovib, molecule: H2O, T: 950, logN: 18.3, logR: -0.8}
  - {name: H2O_hot,   molecule: H2O, T: 900, logN: 18.3, logR: -0.6}
fit:
  windows: [[5.3, 8.0], [12.0, 27.0]]
```

already fits the band and the rotational lines with different slabs; the run log says which windows were
applied, and warns when the fit reaches below 9.5 µm with no water component (and no `Tvib`) there.
`water_split_um: null` switches the default off (the old behaviour: one LTE slab everywhere).

### 3. Curated line regions instead of every pixel (`fit.line_regions`)

```yaml
fit:
  windows: [[12.0, 27.0]]
  line_regions: [H2O_v0-0]          # the 56 isolated pure-rotational lines of Banzatti+2025
  region_weight_beyond: [20.0, 5.0] # Temmink+2025: weight 5 for regions beyond 20 um
```

Banzatti et al. 2023 fit the fluxes of ≈ 100 isolated rotational lines (12–16 µm, E_u = 6000–9000 K for
the hot template first, then the residuals), and Temmink et al. 2024/2025 fit narrow regions around
selected lines (their Table C.1) with weights 5–15 beyond 20 µm so that the noisier channel 4 still
constrains the cold reservoir.  `line_regions` builds the fit windows from the bundled lists
(`H2O_v0-0`, `H2O_v1-0`, `H2O_v1-1`, `H2O_general`, `general`, or a CSV with `xmin`/`xmax` columns),
clipped to `fit.windows`.  See `jalebi.regions`.

### 4. Automated (blind / survey) runs

`runs/validation_blind.yaml` and `runs/survey_300_autodetect.yaml` now carry

```yaml
fit:
  line_regions: [H2O_v1-0, H2O_v0-0]   # isolated rovib lines for the rovib slab, isolated rotational lines for the rest
  region_weight_beyond: [20.0, 5.0]
  region_other_molecules: features     # the other detected molecules keep their Q-branch / band-head ranges (+-0.15 um)
  region_feature_pad_um: 0.15
```

so an automated run fits the water the way Banzatti+2023/2025 and Temmink+2025 do, while CO₂, C₂H₂, HCN, CO …
keep their own pixels (`region_other_molecules: default` re-admits their full default windows, `none` fits the
line regions only). The detection now judges hot/warm water on 12–27.5 µm, so the long-wavelength lines
(E_up 1500–4000 K, the ones that separate warm from hot) enter the fit windows even when no cold slab is
detected. Gaussian priors are available per component (`priors: {T: [800, 150]}`, Romero-Mirza+2024 style)
for cases where a bound alone is not enough. Fully automated on CI Tau (detection → regions → DE, 60
generations): H2O_rovib 1332 K / 18.63 / 0.07 au, **H2O_hot 934 K / 18.51 / 0.22 au, H2O_warm 525 K / 17.80 /
0.48 au**, CO 1490 K, HCN 534 K, CO₂ 326 K — versus 1500 K (edge) / 295 K before.

## CI Tau (jw01640, 160 pc), what each approach gives

Optimiser stage only (no MCMC), IRSQR continuum from the user's run, HITEMP water, Δv = 4.7 km/s:

| run | what | H₂O hot | H₂O warm/cold | χ²_red 5.3–8 | χ²_red 12–25 |
| --- | --- | --- | --- | --- | --- |
| published | Banzatti+2023 (≈100 line fluxes, 12–27) | 840 K, 1.0e18, R 0.65 au | — | | |
| published | Romero-Mirza+2024 (full 12–27, GP continuum, dynesty) | 903 K, 0.10e18 | 477 K, 3.8e18, A 4.85 au² | | |
| R6 | every pixel 12–27, two LTE slabs (= `validation_known`) | 1281 K, 18.40, 0.12 au | 605 K, 18.49, 0.33 au | 164 | 3.4 |
| R5 | `line_regions: [H2O_v0-0]`, weight 5 beyond 20 µm | **827 K, 18.33, 0.31 au** | **386 K, 17.47, 0.64 au** | 218 | 4.3 |
| R2 | every pixel 5.3–8 + 12–25, LTE (= blind validation) | 1500 K (edge), 17.08 | 281 K, 20.9 (edge) | 3.0 | 4.0 |
| R3 | every pixel 5.3–8 + 12–25, `Tvib` on both water slabs | 1211 K, **T_vib 787 K**, 18.60, 0.12 au | 677 K, 18.29, 0.30 au | **2.4** | 2.3 |
| R7 | as R3 but `Tvib` on the hot slab only, differential evolution | **916 K, T_vib 534 K, 18.93, 0.20 au** | warm slab → nothing (log N 13) | 2.6 | 2.5 |
| R4 | 5.3–8 + 12–25, separate `H2O_rovib` slab (`windows`) | rovib: **1002 K, 18.54, 0.09 au**; rot: 1071 K, 18.39, 0.16 au | 574 K, 18.32, 0.34 au | **2.5** | 2.4 |

R4's ro-vibrational slab (1000 K, log N 18.5, R 0.09 au) is the same kind of solution as Sz 98's
5–6.5 µm fit (950 K, 18.57, 0.07 au; Gasman+2023) and DR Tau's (1000 K, 18.4, 0.21 au; Temmink+2024).
R7 (a global search) ends with a *single* hot slab, T_rot = 916 K and T_vib = 534 K, that fits 5–8 µm
(data/model 1.04) and 12–25 µm at once — Banzatti+2025 describe CI Tau as "a spectrum that can be described
almost entirely by a single hot component", and 916 K sits between Banzatti+2023 (840 K) and Romero-Mirza+2024
(903 K). R3 (local search) keeps two slabs with T_vib ≈ 0.65 T_rot and also fits both bands (χ²_red 2.4 at 5–8 µm, where
the LTE extrapolation gave 160–220). Runs: `results/CI_Tau/rovib_experiments/` (report.json + overview.png
per run; `runs/citau_rovib_experiments.py` reproduces them from `results/CI_Tau/prep.csv`). Optimiser only,
Nelder–Mead from the published starting values for R2–R4 — run the example configs with DE + MCMC for the
final numbers.

The temperatures of the published two-temperature fits are recovered only when the fit is restricted
to isolated rotational lines (R5); fitting every pixel lets the blends, the v = 1–1 hot bands (which
LTE over-predicts by ≈ 3), OH and the organics pull the hot component to > 1200 K and the cold one to
600 K.  The emitting radii differ from Banzatti+2023 by ≈ 2 (0.31 vs 0.65 au at the same T) because
N and R are degenerate for τ ≫ 1 lines and the two methods use different line widths and continua
(see below).

## Why a validation run can differ from a paper even with the "same" windows

1. **Pixels vs line fluxes.** Banzatti+2023 / Banzatti+2025 use isolated-line fluxes; MINDS uses
   narrow regions with weights; Romero-Mirza+2024 and Pontoppidan+2024 fit every pixel. Use
   `line_regions` for the first two.
2. **Line width.** jalebi's default Δv = 4.7 km/s (MINDS, Gasman+2023: σ = 2 km/s). Romero-Mirza+2024
   use turbulent 1 km/s ⊕ thermal (0.4–0.8 km/s), i.e. Δv ≈ 2–3 km/s: narrower lines saturate at lower
   N, so N (and R through N·A) shift by factors of a few. Use `fwhm: 2.35, fwhm_thermal: true` for that
   paper. Temmink+2025 compare both choices and most disks prefer 4.71 km/s.
3. **Continuum.** median + Savitzky–Golay (Banzatti, JDISCS), GP (Romero-Mirza), IRSQR knots 25–75 and
   quantile 0.1 (0.5 under the silicate feature) with Q-branches interpolated (MINDS), cubic spline
   through line-free points (Gasman). The cold component at 20–27 µm is the most sensitive to this.
4. **Noise.** ETC continuum S/N per sub-band (Temmink), GP residuals + a 10 mJy floor (Romero-Mirza),
   line-subtracted residuals (Gasman). jalebi's MAD per sub-band is closest to Gasman's.
5. **Priors.** Romero-Mirza+2024 put normal priors on T around 400 K and 800 K; Temmink+2025 fit a
   T(R) power law with 50 slabs (`kind: annuli`).
6. **Other species.** CI Tau's OH and organics are inside 12–17 µm; a water-only fit of every pixel
   assigns them to hot water. Add OH / C₂H₂ / HCN / CO₂ components or mask them.
7. **Hot bands.** v = 1–1 water lines (E_u ≳ 6000 K) at 10–17 µm are over-predicted in LTE by ≈ 3;
   with every-pixel fits add `Tvib` to the hot component.
8. **Distance.** Banzatti+2023 use 160 pc for CI Tau; R scales with d.
9. **Transcription.** The RM24 numbers in `runs/validation_known.yaml` for CI Tau (and the cold
   AS 209 column) did not match the paper's table (T_cold = 477 K, N_hot = 1.0e17, N_cold = 3.8e18,
   A_cold = 4.85 au²) — corrected in this release; re-check the others against the paper.
