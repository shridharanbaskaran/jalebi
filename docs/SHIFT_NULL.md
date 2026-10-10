# Shifted-template null test (0.23, `fit.shift_null`, `jalebi shift-null`)

## Why ΔBIC > 10 is not a detection test for MIRI disk fits

ΔBIC assumes white Gaussian noise with the right σ and a correct model. MIRI disk fits have neither:

- the residuals are correlated (median lag-1 autocorrelation ≈ 0.5 in the 0.21 survey) and over-dispersed (noise
  scale s ≈ 1.7);
- the LTE slab is an approximate model of a line forest, so the residual contains misfit structure;
- auto-detect tries ~15 flexible (T, N, R) candidates per disk, so some win by chance;
- an optically thick slab pinned at its prior corner acts as a smooth pseudo-continuum and gains χ² without any
  line structure (the hot-water corner of the 0.21 survey).

Calibrating a threshold on objects without disk gas (debris disks, white dwarfs, background stars) does not
solve this: their residuals are photospheres, silicate and ice bands rather than a line forest, a handful of them
gives an unstable maximum (one object moved the CO2 rate from 3 % to 32 %), and ΔBIC scales with S/N.

## The test: each spectrum is its own null

For every independent unit u (molecule / component) of a finished fit:

```
r       = data − continuum correction − all units          best-fit residual (no unit-u signal)
t_b     = template bank of unit u: T × {0.5, 0.7, 1, 1.4, 2}, log N + {−1, 0, +1} (inside the prior bounds),
          each high-pass filtered (0.3 µm running median per contiguous segment): smooth flux does not count
z0      = max_b ⟨t_b, r + t_u⟩_w / |t_b|_w                  match at the true wavelengths
z(v)    = max_b ⟨t_b(v), r⟩_w / |t_b(v)|_w                  1500 ≤ |v| ≤ 9000 km/s, step 400 km/s (38 shifts)
S       = (z0 − median z(v)) / (1.4826 MAD z(v))             significance against this spectrum's own null
```

with ⟨a, b⟩_w = Σ w a b, w = fit weights / (s σ)². A shifted template has the line density and band shapes of
the real one but its lines fall in the wrong places; the spread of its matches is what chance alignments with
noise, misfit structure and other molecules' lines give *in this spectrum, at this S/N*. The bank maximum is
taken at v = 0 and at every shift, so the null has the same (T, N) freedom as the fit. The null uses the residual
without unit u, so the molecule's own lines do not inflate it at shifts equal to its line spacing.

`null_sigma` (the robust spread of z(v)) is 1 for white noise with the right σ; values above 1 measure the
excess from correlated residuals and line confusion that ΔBIC ignores.

## Checks

- Synthetic, AR(1) correlated noise (ρ = 0.8, 3σ) with CO2 only, CO2 + HCN + C2H2 fitted, 20 noise seeds:
  ΔBIC > 10 for the absent molecules in 14 / 40 cases, naive matched-filter S/N > 5 in 22 / 40, **S > 5 in 0 / 40**.
- The 18 published validation disks (0.22 fits, continuum_fit: none): every hot-water component at the prior
  corner (T ≈ 1490 K, log N ≈ 21: AS 209, BP, CY, DN, DR, HP, IQ Tau) has S = −0.2 to 2.5; every physical hot water
  has S ≥ 14. The cold 13CO2 pseudo-continuum sinks (GW Lup 113 K, CX Tau 205 K) have S = 2.6–3.8. HCN, C2H2 and
  CO2 in GW Lup, Sz 114, DF Tau, CX Tau and FZ Tau have S = 6.7–47, except FZ Tau C2H2, which that fit put at
  log N 20.7 with almost no line structure left after the high-pass (line fraction 0.06): S = −0.1. Pooled null of those fits: 0 of 2007 shifted
  matches above S = 5 (FAP < 5 × 10⁻⁴).

## Use

The pipeline runs the test after every fit when `fit.shift_null.enabled` (default true) and writes
`shift_null.csv` (one row per unit: z0, null_median, null_sigma, S, fap_empirical, n_null, line_fraction,
detected_shift, null_ok) and `shift_null_curves.csv` (z per shift, for plots). The settings are post-processing
only and not part of the resume key. On finished fits:

```bash
jalebi shift-null RESULTS/DR_Tau                          # one folder
jalebi shift-null RESULTS --survey --workers 16 --backend emulator   # every folder; merged shift_null_survey.csv
python runs/shift_null_calibrate.py RESULTS --classes census.csv     # pooled null -> FAP of a threshold, rates
```

`detected_shift` uses `fit.shift_null.threshold` (default S ≥ 5). `runs/shift_null_calibrate.py` (output in `RESULTS/_shift_null_calibration/`) pools the
leave-one-out S of all shifted matches of a survey (≈ 60 000 for 270 disks) and reports the false-alarm
probability of each threshold per molecule; quote that, not a Gaussian tail.

## Limits

- 38 shifts per unit: the empirical FAP of a single disk has a floor of 1/39; S and the pooled calibration are
  the quantities to use.
- Fits restricted to narrow line windows (`fit.regions` / `line_regions`) lose a shifted template off the fitted
  pixels; shifts that keep < 50 % of the template energy are dropped, and with < 10 valid shifts S is NaN
  (`null_ok` False).
- The test measures whether the *line structure* of a template is in the data. It does not test the LTE
  parameters; T and N still come from the posterior, and N·A upper limits for non-detections come from
  injection–recovery.
