# Accretion from the mid-IR H I lines (0.24, `fit.accretion`, `jalebi accretion`, app tab "Accretion")

Method: Shridharan et al. 2026, A&A 708, A22 ("Improving accretion diagnostics for young stellar objects with
mid-infrared hydrogen lines from JWST/MIRI"). A local water fit is subtracted around every line.

## Per line

Every H I transition with n_up ≤ 14 inside the spectrum is measured, about 15–25 lines in 4.9–28 µm. Vacuum
wavelengths come from the Rydberg formula.

1. **Local water** (`water: local`):
   - fit an LTE H2O slab to the continuum-subtracted spectrum within ±`water_window_um` (0.5 µm) of the line,
     with every H I line masked by ±3 Δλ (Δλ = λ/R);
   - grid T 200–2000 K, log N 12–22 (coarse, then full resolution around the best cell, then a Nelder–Mead
     polish), area by linear least squares ≥ 0, intrinsic width 4.7 km/s, the config's H2O line list (HITEMP
     in the survey);
   - subtract the best-fit water. `water: global` subtracts the disk's own jalebi fit (all molecules) instead;
     `none` skips the subtraction.
2. **Cut-out** of ±15 Δλ. Local continuum `baseline: linear`: a straight line through the pixels outside the line
   core and outside other H I lines, 3σ clipped. `baseline: als` is the paper's asymmetric least squares, which
   sits on the lower noise envelope; on synthetic tests it biases faint lines high by +0.1–0.15 dex in L_acc.
3. **Gaussian:** centre within ±1 Δλ, σ between 0.8 and 2 Δλ, inverse-variance weights; flux = area.
4. **Detection:** peak > 10 × RMS (`detect_rms`, the paper's rule; RMS = median pipeline error) and S/N ≥ 3.
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
carrying up to ~90 % of the flux under Pf α:

- per-line log L_acc within 0.15 dex;
- combined −1.52 ± 0.09;
- the water fit recovers T = 650 K;
- upper limits above the injected value;
- Ṁ matches 1.25 L_acc R★/(G M★).

Next: rerun the paper's 79-star sample (programmes 1282, 1549, 1584, 1640, 1751, 2025) and compare per line.
