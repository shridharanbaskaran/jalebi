# LTE gas absorption in JALEBI (0.11)

Cold gas in front of a warm continuum — the envelope, an outflow or the outer disk of an embedded
protostar, or an edge-on disk — shows the ro-vibrational bands of CO, CO₂, H₂O, C₂H₂, HCN, … in
absorption. JALEBI 0.11 fits such screens with the same machinery as the emission: one fine
ln(λ) grid, the same line lists and partition functions, the same MIRI LSF operator, the same
grid → differential-evolution → emcee stages.

## The model

An absorption component (`kind: absorption`) has a column density N, a temperature T, a line
width Δv, a velocity v and a **covering fraction** f_c of the continuum source. Its optical depth
τ(x) is the slab opacity of § 2 of the README (N × Σ_l κ_l(T) φ(x − x_l − v/c)). The observed flux is

    F_ν = F_c · [ 1 − f_c · (1 − e^(−τ)) ]                          (Li, Boogert & Tielens 2024)

which is `spec_abs` in the group's `slabby.py` (there: θ = (log N, T, v, f_c), thermal width). JALEBI
works on continuum-subtracted data, so the unit flux of a screen is

    F_i − F_c,i = Σ_j K_ij · F_c(x_j) · [Tr(x_j) − 1],     Tr = 1 − f_c (1 − e^(−τ)),

with F_c interpolated from the data-preparation continuum onto the fine grid and K the LSF/pixel
operator. Because F_c is smooth over the LSF this is the same as convolving Tr and multiplying by the
pixel continuum afterwards (the old code's order): the parity script gives identical equivalent widths
and a 0.8 % peak difference for σ = 0.2 km/s lines (grid discretisation of the old code).

### Geometry

* `covers: continuum` (default): the screen sits in front of the continuum only. Emission components
  are side by side with it (their emitting areas are separate from the absorbed continuum source).
* `covers: all`: the screen sits in front of everything. Every emitting slab is multiplied by Tr on the
  fine grid before the LSF, i.e. F = (F_c + Σ F_em) · Tr — the `two_slabs_spec` geometry of the old
  code, generalised with f_c. The model is still linear in the emitting areas, so the exact NNLS area
  solve, the (log N, T) grid and the optimiser are unchanged.
* Several screens: their transmissions multiply. For the per-component curves the k-th screen is
  attributed what the first k−1 let through, so the unit fluxes still add to the total.
* Group of screens (`group:`): opacities add, shared T, Δv, v and f_c — the usual opacity group.

### Line width

`fwhm` is the intrinsic FWHM in km/s as for the emission (default 4.7). `fwhm_thermal: true` adds the
thermal FWHM at T, 2√(2 ln 2)·√(kT/m), in quadrature. Note that the old `spec_abs` used the Doppler
parameter √(2kT/m) *as* the FWHM (σ = √(2kT/m)/2.355), a profile 1.66× narrower than thermal; with
`fwhm_thermal` JALEBI uses the correct thermal FWHM. At 100 K that is 0.5 km/s for H₂O and 0.34 km/s
for CO₂: far below the MRS resolution (~100 km/s), so unless the lines are strongly saturated the
turbulent width dominates and the data cannot tell the difference. Keep `fwhm` fixed at a physically
motivated value unless the curve of growth of saturated lines constrains it (then `fit_fwhm: true`).
The fine grid step is set from the smallest `fwhm` of the components (÷ oversample), not from the
thermal width.

## Free parameters, bounds and what constrains them

| parameter | default bounds (screen) | constrained by |
| --- | --- | --- |
| log N | 13 – 22 | depth of unsaturated lines, wings of saturated ones |
| T | 20 – 1500 K | line ratios along the band (rotational ladder) |
| v | −200 – +200 km/s | position of the band (negative = blueshift: outflow) — **always free** |
| f_c | 0 – 1 | depth of saturated lines (a saturated line goes down to F_c (1 − f_c)) |
| fwhm | 1 – 60 km/s | only with `fit_fwhm`; degenerate with N in the curve of growth |

Per-component `bounds:` override these; `fixed:` removes a parameter from the fit. There is no
`logR` for a screen (it is 0 internally so that the area factor is 1). The τ flag reported for a screen
is its peak optical depth: τ_max ≫ 1 means the depth is set by f_c and the lines are saturated, so N is
constrained only through the wings and the weak lines.

## The continuum is the crux

The screen multiplies the continuum, so the continuum must be the *unabsorbed* one. The default
lower-envelope estimators (`irsqr` at quantile 0.1, `median_sg`, `asls`, …) follow the absorption
troughs and remove them. Two ways to do it properly:

1. **Give the continuum.** A CSV with a `continuum` (or `baseline`, `cont`, `base_fluxes`) column, and
   `continuum.method: given` in the config: JALEBI keeps it as is (`jalebi.data.load_csv` reads the
   column; the `Spectrum.continuum` attribute holds it).
2. **Upper envelope.** `continuum.method: irsqr` with `quantile: 0.9` (and a large `knot_spacing`)
   traces the top of the spectrum. Check it in the *Continuum* tab of the app before fitting.

With emission lines *and* absorption in the same range neither envelope is right; give the continuum
from a model (dust SED) or from line-free anchor points (`method: spline`, `anchors: [...]`).

## Running it

```yaml
components:
  - name: CO2_abs
    molecule: CO2
    kind: absorption
    logN: 18.0
    T: 150
    rv: -20          # km/s
    fc: 0.5
    covers: all      # also absorbs the emission components below
    bounds: {T: [30, 600], rv: [-120, 60]}
  - name: CO2_hot
    molecule: CO2
    logN: 17.5
    T: 600
    logR: -0.3
continuum:
  method: given
fit:
  windows: [[14.6, 15.4]]
  grid: {logN: [15, 19.5, 19], T: [40, 900, 18]}
```

```bash
jalebi fit config.yaml                            # grid → DE → emcee, results/{target}/
jalebi model --molecule H2O --kind absorption --logN 18.5 --T 300 --fc 0.6 --rv -30 --wmin 6 --wmax 7
python examples/11_absorption_fit.py              # injection–recovery, 1–3 min
```

Python:

```python
from jalebi.model import Component, build_model
screen = Component("H2O_abs", "H2O", kind="absorption", logN=18.3, T=250, rv=-35, fc=0.7)
m = build_model([screen], wave, distance_pc, [(6.0, 7.0)], continuum=cont)   # cont: Jy on wave
F_line = m.evaluate()          # = F_c (Tr − 1): negative
tr, tau_max = m.transmission(screen, m.resolve_params()["H2O_abs"])          # on the fine grid
```

In the web app choose `kind: absorption` in the component card: the log R slider is replaced by f_c,
the *absorbs* selector chooses `continuum` / `continuum + emission`, and the model tab shows the screen
as a negative unit curve. The display model follows the continuum whenever it is re-estimated.

Example 11 recovers a CO₂ screen (log N 18.3, T 120 K, v −40 km/s, f_c 0.6, `covers: all`) in front of a
hot CO₂ emission slab from a S/N 300 synthetic spectrum: every parameter within 1σ (v = −39.8 ± 0.8,
f_c = 0.60 ± 0.02, T = 124 ± 6 K, log N = 18.26 ± 0.09), ΔBIC of the screen 5.4 × 10⁴.

## Detecting absorbers automatically

`jalebi detect config.yaml` (and `fit --auto-detect`, the app's *Detect molecules* button) tests
absorption too: `fit.detect.mode: both` (default), `emission` or `absorption`. Each candidate molecule
gets one screen template (fixed log N and T, f_c as the linear coefficient, bounded 0–1) next to its
emission templates. Because a screen and an emitting slab of the same molecule are nearly anti-collinear
(one is roughly minus the other), `both` runs three passes — emission alone; every template with only the
screens judged, on the pixels *below the continuum* (a screen has to explain real troughs); and, when
screens are found, every emission candidate with the detected screens for the final verdict on the
emission, which is how emission hidden under an absorption band is recovered. The screen temperature is
chosen afterwards (50–500 K) by refitting each temperature alone to the residual. Detected screens are
written as `kind: absorption` components with `fc` from the solve and v = 0; the fit then finds the shift.
The detection needs the unabsorbed continuum like everything else here.

Checks: FZ Tau (emission only) gives the same six emission components as `mode: emission` and no screen;
the bundled synthetic absorber gives `CO2_abs` (150 K, f_c 0.64; truth 120 K, 0.6) and the hot CO₂
emission behind it (`tests/test_absorption.py::test_detect_finds_absorber_and_emission_behind_it` is the
toy version).

## Validation against the old code

`slabby.spec_abs` was transcribed (same HITRAN lines and partition function, same constant R = 3500,
σ = √(2kT/m)/2.355) and compared with a JALEBI screen of the same σ on H₂O 6.00–6.25 µm, log N 18.3,
T 250 K, v −35 km/s, f_c 0.7: equivalent width 3.8 × 10⁻⁴ µm in both, peak depth 0.0164 vs 0.0158,
τ_max 236 vs 234 (the old velocity grid is 10 points per σ with integer index truncation).

## Not yet

- Ice bands (Sujoy's τ_ice code) as a further multiplicative screen — next in the roadmap.
- A screen with its own emission (a warm absorber re-emitting: F_c e^{−τ} + B_ν(T)(1 − e^{−τ}) Ω) can
  be approximated now by an absorption component plus an emission component with the same N and T, but
  they are not tied; a `emits: true` option would tie them.
- Pixel-level partial covering of an extended source (f_c varying with wavelength).
