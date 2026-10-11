# Accretion from the mid-IR H I lines (0.24, `fit.accretion`, `jalebi accretion`, app tab "Accretion")

Method: Shridharan et al. 2026, A&A 708, A22 ("Improving accretion diagnostics for young stellar objects with
mid-infrared hydrogen lines from JWST/MIRI"). A local water fit is subtracted around every line.

## Per line

Every H I transition with n_up ≤ 14 inside the spectrum is measured, about 15–25 lines in 4.9–28 µm. Vacuum
wavelengths come from the Rydberg formula.

1. **Is there water under the line?** (`water_overlap_check`, on by default.) Water is fitted only when a
   HITRAN H2O line with E_up < `water_overlap_eup` (5000 K) and A_ul > `water_overlap_aul` (1 s⁻¹) lies within
   ±`water_overlap_dlam` (1 Δλ) of the H I line centre. In practice:
   - water is fitted for (6–5), (7–6) and (12–7), as in the paper;
   - water is not fitted for (9–6), (8–6), (10–7) or (13–8).

   Where no water line overlaps, nothing is subtracted. Pixels within ±1 Δλ of the qualifying water lines in the
   cut-out are masked from the baseline and from the Gaussian fit, so nearby water emission does not leak into
   the continuum or the line wings. The `water_overlap` column records the decision.
2. **Local water** (`water: local`, overlapping lines only):
   - fit an LTE H2O slab to the continuum-subtracted spectrum within ±`water_window_um` (0.5 µm) of the line,
     with every H I line masked by ±3 Δλ (Δλ = λ/R);
   - fit the slab together with a linear baseline (bounded least squares, water area ≥ 0). This stops the water
     area from absorbing a continuum offset;
   - grid T 200–2000 K, log N 12–22 (coarse, then full resolution around the best cell, then a Nelder–Mead
     polish), intrinsic width 4.7 km/s, the config's H2O line list (HITEMP in the survey);
   - subtract the best-fit water (the slab only, not its baseline). `water: global` subtracts the disk's own
     jalebi fit (all molecules) instead; `none` skips the subtraction.
3. **Cut-out** of ±15 Δλ. Local continuum `baseline: linear`: a straight line through the pixels outside the line
   core, outside other H I lines and outside masked water lines, 3σ clipped. `baseline: als` is the paper's
   asymmetric least squares, which sits on the lower noise envelope; on synthetic tests it biases faint lines high
   by +0.1–0.15 dex in L_acc.
4. **Multi-peak fit** (`blend_fit: peaks`, 0.24.1). The H I line is fitted together with every significant peak
   around it (within ±15 Δλ) and a small linear baseline correction (bounded to ±2σ). Each component has its own
   Gaussian, and only the H I component's area is reported.
   - **H I Gaussian:** centre within ±1 Δλ, σ between 0.8 and 2 Δλ, inverse-variance weights.
   - **Neighbouring H I lines** within 4 Δλ, e.g. (8–6) + (11–7) or (7–6) + (11–8): one Gaussian each, sharing
     this line's velocity shift and fractional width.
   - **Peak detection** (two steps; components never closer than `blend_sep_dlam` = 1.5 Δλ to each other):
     1. real peaks of the lightly smoothed data are found with `scipy.signal.find_peaks` (prominence ≥ 5× noise,
        distance ≥ 1.5 Δλ, as in Shridharan+2026). All of them outside the H I core enter the fit together;
     2. shoulders and blends without their own maximum are added one at a time at the highest residual peak (≥ 5σ),
        while BIC improves by ≥ 10.

     Extra components have σ 0.3–0.75 Δλ (molecular lines are unresolved); at most `blend_max` (20) per line.
   - **H I core.** No extra component may sit within max(1.5 Δλ, 0.75× the expected H I FWHM) of the expected H I
     position (`blend_min_sep`, `blend_excl_fwhm`), and its centre bounds keep it out during the fit. Without this,
     a neighbour could slide into the core and take the H I peak (CX Tau (9–6) in an earlier build). In the first
     (free) pass, a peak within 2 FWHM of the line also needs a dip below `blend_dip` (0.9) of the lower peak. In
     the tied pass (H I centre and width pinned), every residual peak outside the core is fitted, shoulders
     included (FZ Tau (8–6), (11–7)).
   - **Known water lines.** At the positions of screened HITRAN water lines, peaks get a component without the
     dip test (centre within ±0.3 Δλ), where the water is not modelled. Example: the water emission at about
     −180 km/s from (9–6), which the 0.24.0 single Gaussian partly absorbed.
   - **Water scale.** Where water is subtracted, a local water scale (0–2) is fitted jointly in the cut-out, so a
     slab that is too strong or too weak at this line is corrected. Example: Hu α in FZ Tau, ×0.3.
   - **Flux error:** from the covariance of the linear parameters (all amplitudes, water scale, baseline) at the
     fitted centres and widths, plus the width error. This stays well-defined when components overlap.
   - **Tied kinematics** (`tie_kinematics`, second pass). All H I lines share one velocity centroid and intrinsic
     width (LSF removed). These are the S/N-weighted median over the clean strong lines: S/N ≥ 10, < 30 % blend
     flux in the core, little water, centre and width not at their bounds or the FWHM cap. Tolerances: ±25 km/s
     and ±50 % width; the reference velocity is clipped to ±40 km/s.
     - The reference is re-derived once from the tied, fully decomposed fits, and the pass is redone if it moved
       (> 10 km/s or > 10 %). First-pass widths carry unfitted blends.
     - With fewer than two clean lines, the free fit is kept.
   - **Width cap** (`fwhm_max_kms`, 300). The H I Gaussian's observed FWHM is at most 300 km/s, floored at 1.05×
     the instrumental profile. Beyond ~20 µm the MRS resolution element alone is wider than 300 km/s, so there
     the line is fitted as unresolved.
   - **No H I in absorption** (`water_floor_sigma`, 2). Water emission cannot exceed the observed flux.
     - In the window fit, the slab area is capped so that water + baseline ≤ data + 2× noise on every H I-core
       pixel where the water is significant.
     - In the line fit, the water scale is capped the same way: ws ≤ 1 + (data after water + 2× noise)/water.
     - The noise is the larger of the pipeline error and the empirical rms.

     The subtracted spectrum therefore never dips below the noise under the H I line, which the 0.24.0 single slab
     did at Hu α (FZ Tau). Dips elsewhere in a cut-out come from the global continuum, not from the water.
   - `blend_fit: hi` restores the 0.24.0 fit (one Gaussian, plus neighbouring H I only).
5. **Detection:** peak > 10 × RMS (`detect_rms`, the paper's rule; RMS = median pipeline error) and S/N ≥ 3.
   Otherwise an upper limit = max(F, 0) + 3σ_F.

## Calibrations (A&A Table 1)

log(L_acc/L☉) = a [log(L_line/L☉) − offset] + b

| line | λ (µm) | a | b | offset | note |
|---|---|---|---|---|---|
| H I (9–6) | 5.908 | 1.10 ± 0.17 | −0.80 ± 0.13 | −5.17 | water-free |
| H I (6–5) | 7.459 | 1.08 ± 0.14 | −0.70 ± 0.08 | −4.99 | water blend |
| H I (8–6) | 7.502 | 1.10 ± 0.12 | −0.84 ± 0.08 | −5.22 | water-free |
| H I (10–7) | 8.760 | 0.99 ± 0.13 | −0.73 ± 0.09 | −5.48 | water-free |
| H I (13–8) | 9.390 | 1.12 ± 0.17 | −0.80 ± 0.10 | −5.40 | OH blend |
| H I (7–6) | 12.371 | 1.06 ± 0.15 | −0.64 ± 0.08 | −5.32 | water blend; H I (11–8) at 12.387 |

(arXiv v1 had a = 1.00 for (10–7) and 1.07 for (7–6); jalebi uses the A&A values.)

## Per disk

- **L_line** = 4π d² F × 10^(0.4 A_λ). A_λ = A_V × curve: McClure (2009) when `extinction_table` gives the
  curve, otherwise Gordon et al. (2023, R_V 3.1, `dust_extinction`); the curve used is reported.
- **Per calibrated line:** log L_acc with a Monte Carlo over the flux and over a and b (16/50/84 %).
- **Combined:** weighted mean over the detected lines in `combine`, by default the water-free (9–6), (8–6) and
  (10–7). The spread between those lines is also reported. Without a detection, the combined value is the
  lowest upper limit.
- **Ṁ** = (1 − R★/R_in)⁻¹ L_acc R★ / (G M★) with R_in = 5 R★, i.e. 1.25 L_acc R★/(G M★). M★ and R★ come from
  `target.mstar`, `target.rstar` (or R★ from `target.lstar` and `target.teff`).

- **Continuum variants** (`continuum_variants: N`, default 0 = off). Remeasure every line with N perturbed
  continua: a per-band offset and tilt with an rms of `continuum_variant_sigma` (1 %) of the continuum. The
  scatter of log L_acc is reported as `logLacc_cont_syst` per line and combined. For the strong H I lines it is
  usually < 0.05 dex.

## Outputs and interfaces

- `accretion.csv` per line, `accretion.json` (combined values), `accretion.png` (every cut-out plus log L_acc per
  line).
- **Catalogue / Batch columns:** `logLacc`, `logLacc_err`, `logLacc_spread`, `logMdot`, `logMdot_err`,
  `n_HI_lines`, `HI_lines_used`; per calibrated line `logLacc_<line>` (± `_err`, `_ul`) and `logMdot_<line>`;
  `F_<line>` per measured line.
- **App:** the Accretion tab (current spectrum; A_V, M★, R★, L★, T_eff; water mode and window; choice of
  combined lines). The Batch tab estimates every scanned target in the background (spectrum plus continuum, no
  slab fit) and fills one column per calibrated line plus log L_acc and log Ṁ. Editing M★ or R★ in the table
  recomputes log Ṁ.
- **CLI:**
  - `jalebi accretion RESULTS/<disk>`
  - `jalebi accretion RESULTS --survey`
  - `jalebi accretion --targets targets.csv -c config.yaml`, where the targets columns name, path,
    distance_pc, av, mstar, rstar, lstar and teff are read.

## Checks (tests/test_0240_robustness_accretion.py)

Synthetic MIRI spectrum (S/N 300) with H I lines injected at log L_acc = −1.5 and warm water (650 K, log N 18.3)
carrying up to ~70 % of the flux under Pf α:

- per-line log L_acc within 0.15 dex;
- the combined value within 0.1 dex;
- water fitted only where a water line overlaps;
- upper limits above the injected value;
- Ṁ matches 1.25 L_acc R★/(G M★);
- the continuum-variant scatter is finite and small;
- a narrow line 2.2 Δλ from an H I line gets its own Gaussian, and the H I flux comes back within 10 % (the
  single-Gaussian fit is > 15 % high); a plain H I profile is not split.

## Validation on 12 published MIRI disks (validation_known_0.22, d from config, A_V = 0)

Offsets are relative to the mean of (10–7) and (13–8), median ± scatter over the 12 disks, in dex:

| line | 0.24.0 (single Gaussian) | 0.24.1 (multi-peak, tied kinematics) |
|---|---|---|
| (9–6) | −0.03 ± 0.30 | −0.14 ± 0.43 (Sz 114 upper limit) |
| (8–6) | +0.09 ± 0.17 | +0.10 ± 0.24 |
| Pf α (6–5), after water | +0.15 ± 0.18 | +0.19 ± 0.10 |
| Hu α (7–6), after water | +0.11 ± 0.30 | +0.18 ± 0.14 |

Combined log L_acc (0.24.1):

| DR Tau | FZ Tau | GQ Lup | CI Tau | AS 209 | DF Tau |
|---|---|---|---|---|---|
| 0.32 | 0.11 | −0.57 | −0.11 | −0.39 | −0.34 |

| GW Lup | Sz 114 | CX Tau | BP Tau | GK Tau | IQ Tau |
|---|---|---|---|---|---|
| −1.56 | −1.62 | −1.36 | −0.81 | −1.04 | −0.85 |

The joint water scale halves the Hu α scatter. (9–6) drops in water-rich disks (DF Tau, GW Lup, Sz 114),
because the water emission at 5.904 µm, about −180 km/s from the line, is now its own component. If the A&A
Table 1 calibration was built from (9–6) fluxes that included this water line, deblended (9–6) fluxes sit
below it. Use `blend_fit: hi` to compare.

## Validation against Shridharan+2026 (`runs/accretion_validate.py`)

```bash
python runs/accretion_validate.py published_79.csv --survey results/survey_0.22 --workers 30 --out accretion_validation
```

`published_79.csv` = CDS J/A+A/708/A22 Tables A.1 + A.2 merged on the star name (Source, Dist, A_V, log L_acc,
Lline(6-5) ... Lacc(10-7); "-5.57(0.03)" and "<-6.67" are parsed). jalebi is run with the paper's distance and
A_V. Outputs: comparison.csv, summary.md, validation.png.

**First check, 2026-10-11.** The 10 rows printed in the arXiv appendix were matched to 6 of the validation disks
(CX, CY, BP, FT, DF, IQ Tau; s3d spectra, 1.5 FWHM aperture, MRS aperture correction):

| line | N | median Δ log L_line (jalebi − paper) | scatter |
|---|---|---|---|
| (6–5) | 6 | +0.22 | 0.18 |
| (7–6) | 6 | +0.17 | 0.04 |
| (8–6) | 6 | +0.01 | 0.10 |
| (10–7) | 5 | +0.23 | 0.10 |

With all peaks fitted, (8–6) agrees (+0.01). (10–7), which has no water, is +0.23 dex, so part of the offset
comes from the spectra (extraction / aperture correction), not from the line fit. The full 79-star run needs the CDS tables.

Next: rerun the paper's 79-star sample (programmes 1282, 1549, 1584, 1640, 1751, 2025) and compare per line.
