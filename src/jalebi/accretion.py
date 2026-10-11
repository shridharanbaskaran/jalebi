"""Accretion luminosity and rate from the mid-infrared H I lines (0.24; `fit.accretion`, `jalebi accretion`).

Method: Shridharan et al. 2026, A&A 708, A22 ("Improving accretion diagnostics for young stellar objects with
mid-infrared hydrogen lines from JWST/MIRI"), with the local water subtraction done for every line:

per H I line (every transition with n_up <= max_nup inside the spectrum)
    1. local water: an LTE H2O slab is fitted to the continuum-subtracted spectrum within +-water_window_um of the
       line (all H I lines masked by +-water_mask_dlam resolution elements), on a (T, log N) grid with the area
       solved by linear least squares (>= 0), coarse grid then a refinement around the best cell; the best-fit
       water is subtracted (water: global subtracts the jalebi fit of the disk instead; none skips it)
    2. cut-out of +-cutout_dlam/2 resolution elements (dlam = lambda / R(lambda)), continuum by asymmetric least
       squares (pybaselines asls) on the water-subtracted flux
    3. Gaussian in +-fit_dlam: centre within +-centre_dlam, sigma in sigma_range (resolution elements), inverse-
       variance weights; flux = Gaussian area; detected when the peak > detect_rms x RMS and S/N >= min_snr,
       otherwise a 3 sigma upper limit
per disk
    L_line = 4 pi d^2 F (dereddened when A_V is known), log L_acc = a [log L_line - offset] + b per calibrated line
    (Monte Carlo over the flux and a, b), the combined log L_acc = weighted mean over the detected lines of
    `combine` (default the water-free H I (9-6), (8-6), (10-7)); Mdot = (1 - R*/R_in)^-1 L_acc R* / (G M*).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

R_H = 1.0967758341e7          # hydrogen Rydberg constant (m^-1)
C_M = 2.99792458e8
C_KMS = 299792.458
L_SUN = 3.828e26              # W
M_SUN = 1.98847e30            # kg
R_SUN = 6.957e8               # m
G = 6.67430e-11
PC = 3.0856775814913673e16    # m
YEAR = 3.15576e7              # s
SIGMA_SB = 5.670374419e-8

# Shridharan et al. 2026, A&A 708, A22, Table 1: log(Lacc/Lsun) = a [log(Lline/Lsun) - offset] + b
CALIBRATIONS = {
    "HI_9-6": {"a": 1.10, "a_err": 0.17, "b": -0.80, "b_err": 0.13, "offset": -5.17},
    "HI_6-5": {"a": 1.08, "a_err": 0.14, "b": -0.70, "b_err": 0.08, "offset": -4.99},
    "HI_8-6": {"a": 1.10, "a_err": 0.12, "b": -0.84, "b_err": 0.08, "offset": -5.22},
    "HI_10-7": {"a": 0.99, "a_err": 0.13, "b": -0.73, "b_err": 0.09, "offset": -5.48},
    "HI_13-8": {"a": 1.12, "a_err": 0.17, "b": -0.80, "b_err": 0.10, "offset": -5.40},
    "HI_7-6": {"a": 1.06, "a_err": 0.15, "b": -0.64, "b_err": 0.08, "offset": -5.32},
}
# overlaps noted in the paper (water modelled for (12-7), (6-5), (7-6); (13-8) has OH; (7-6) also has H I (11-8))
NOTES = {"HI_6-5": "water blend", "HI_7-6": "water blend; H I (11-8) at 12.387", "HI_12-7": "water blend (~75%)",
         "HI_13-8": "OH blend", "HI_11-8": "blended with H I (7-6)"}


@dataclass
class AccretionSettings:
    enabled: bool = True
    combine: list = field(default_factory=lambda: ["HI_9-6", "HI_8-6", "HI_10-7"])
    water: str = "local"
    water_window_um: float = 0.5
    water_T: list = field(default_factory=lambda: [200.0, 2000.0, 37])
    water_logN: list = field(default_factory=lambda: [12.0, 22.0, 41])
    water_mask_dlam: float = 3.0
    cutout_dlam: float = 30.0
    fit_dlam: float = 15.0
    sigma_range: list = field(default_factory=lambda: [0.8, 2.0])
    centre_dlam: float = 1.0
    detect_rms: float = 10.0
    min_snr: float = 3.0
    rin_over_rstar: float = 5.0
    extinction_law: str = "mcclure09"
    extinction_table: str | None = None
    n_mc: int = 2000
    max_nup: int = 14
    save_plot: bool = True
    baseline: str = "linear"            # linear (core-masked, clipped) | als (as Shridharan+2026)
    water_overlap_check: bool = True    # fit/subtract water only where a water line overlaps the H I line
    water_overlap_dlam: float = 1.0     # overlap = a water line within +- this many resolution elements ...
    water_overlap_eup: float = 5000.0   # ... with E_up below this (K) ...
    water_overlap_aul: float = 1.0      # ... and A_ul above this (s^-1)  (the Shridharan+2026 screening)
    water_overlap_release: str = "hitran"
    continuum_variants: int = 0         # > 0: re-measure under this many continuum / baseline variants -> *_cont_syst
    continuum_variant_sigma: float = 0.01   # per-band relative continuum offset + tilt of the variants
    blend_fit: str = "peaks"            # peaks: every significant peak in the fit window gets its own Gaussian,
                                        #   fitted jointly with the H I line(s) and a linear baseline term;
                                        # hi: only neighbouring H I lines (0.24.0 behaviour)
    blend_snr: float = 5.0              # residual peak significance (3-pixel matched sum / noise) to add a component
    blend_max: int = 20                 # at most this many extra Gaussians per line
    blend_sigma_range: list = field(default_factory=lambda: [0.3, 0.75])  # extra (non-H I) components, in resolution
                                        # elements: molecular lines are unresolved (LSF sigma = 0.42 dlam), so a
                                        # broad feature under the H I line cannot be given to a nuisance component
    blend_min_sep: float = 1.5          # no extra component closer than this (dlam) to the expected H I centre ...
    blend_excl_fwhm: float = 0.75       # ... nor closer than this x the expected H I FWHM
    blend_sep_dlam: float = 1.5         # any two components at least this far apart (dlam): MRS cannot separate
                                        # closer lines (find_peaks distance, as in Shridharan+2026)
    blend_dbic: float = 10.0            # an extra component must lower BIC by this much
    blend_dip: float = 0.9              # within 2 FWHM of the H I line, an extra peak needs a dip below this fraction
    fwhm_max_kms: float = 300.0         # H I Gaussian: observed FWHM at most this (km/s; floor 1.05 x LSF; 0 = off)
    water_scale_max: float = 2.0        # joint local water scale in the line fit (0 = off)
    water_floor_sigma: float = 2.0      # water may not exceed the data by more than this many sigma on any pixel
                                        # (incl. the H I core): the subtraction never puts a line in absorption
    tie_kinematics: bool = True         # 2nd pass: every H I line shares one velocity centroid and intrinsic width,
                                        # measured on the clean strong lines (S/N >= tie_min_snr, no water overlap,
                                        # < 20 % blend flux in the core)
    tie_min_snr: float = 10.0
    tie_dv_kms: float = 25.0            # centroid tolerance around the reference (km/s; at least 0.25 dlam)
    tie_vmax_kms: float = 40.0          # the reference velocity is clipped to +- this (rest-frame spectra)
    tie_width_frac: float = 0.5         # intrinsic-width tolerance (fraction of the reference)
    water_release: str = "hitemp"


def settings_from_config(cfg) -> AccretionSettings:
    import dataclasses
    c = getattr(cfg.fit, "accretion", None)
    s = AccretionSettings()
    if c is not None:
        d = c.model_dump() if hasattr(c, "model_dump") else dict(c)
        keys = {f.name for f in dataclasses.fields(AccretionSettings)}
        s = AccretionSettings(**{k: v for k, v in d.items() if k in keys})
    rel = getattr(getattr(cfg, "linedata", None), "releases", {}) or {}
    s.water_release = rel.get("H2O", s.water_release)
    return s


# ------------------------------------------------------------------------------------------------ lines
def hi_wavelength_um(nu: int, nl: int) -> float:
    """Vacuum wavelength (micron) of the H I n_up -> n_low transition (Rydberg formula, hydrogen mass)."""
    return 1e6 / (R_H * (1.0 / nl ** 2 - 1.0 / nu ** 2))


def hi_lines(wmin: float = 4.9, wmax: float = 28.1, max_nup: int = 14) -> pd.DataFrame:
    rows = []
    for nl in range(4, 12):
        for nu in range(nl + 1, max_nup + 1):
            w = hi_wavelength_um(nu, nl)
            if wmin <= w <= wmax:
                name = f"HI_{nu}-{nl}"
                rows.append({"line": name, "nu": nu, "nl": nl, "wave": w, "calibrated": name in CALIBRATIONS,
                             "note": NOTES.get(name, "")})
    return pd.DataFrame(rows).sort_values("wave").reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ extinction
def extinction_alambda_over_av(wave_um, law: str = "mcclure09", table: str | None = None, av: float = 0.0):
    """A_lambda / A_V; (curve, law actually used).  mcclure09 needs a table (columns wave_um, alambda_over_ak for
    the A_K range of the source, or alambda_over_av); without one the Gordon et al. 2023 average (R_V 3.1,
    dust_extinction) is used and reported."""
    w = np.atleast_1d(np.asarray(wave_um, float))
    if law == "none" or not av:
        return np.zeros_like(w), "none"
    if law == "mcclure09" and table and os.path.exists(table):
        t = pd.read_csv(table)
        if "alambda_over_av" in t:
            return np.interp(w, t.wave_um, t.alambda_over_av), "mcclure09"
        ak = av / 7.75                              # McClure 2009: A_K = A_V / 7.75
        col = "alambda_over_ak_high" if ak >= 1 and "alambda_over_ak_high" in t else (
            "alambda_over_ak_low" if "alambda_over_ak_low" in t else "alambda_over_ak")
        return np.interp(w, t.wave_um, t[col]) / 7.75, "mcclure09"
    try:
        import astropy.units as u
        from dust_extinction.parameter_averages import G23
        ext = G23(Rv=3.1)
        ww = np.clip(w, 1.0 / ext.x_range[1] + 1e-6, 1.0 / ext.x_range[0] - 1e-6)
        return np.asarray(ext(ww * u.micron), float), "g23"
    except Exception:
        return np.zeros_like(w), "none (no extinction curve available)"


# ------------------------------------------------------------------------------------------------ water
def _water_model(wave_win, distance_pc, release, R_model="argyriou2023", R_scale=1.0):
    from .model import Component, build_model
    comp = Component(name="H2O_local", molecule="H2O", linelist_release=release)
    win = [(float(np.min(wave_win)), float(np.max(wave_win)))]
    return build_model([comp], wave_win, distance_pc, windows=win, releases={"H2O": release}, oversample=3,
                       T_max={"H2O": 2000.0}, R_model=R_model, R_scale=R_scale)


def _area_fit(f, y, w, xb=None, ub=None, thr=0.0):
    """Water area a >= 0 by weighted least squares; with `xb` (centred wavelengths) a local straight-line
    baseline is fitted at the same time, so that the water slab cannot absorb a continuum offset (an optically
    thick hot slab otherwise turns into a pseudo-continuum).
    ub: per-pixel upper bound on water (+ baseline), = data + k sigma on every pixel with data, including the
    masked H I pixels: water emission cannot exceed the observed flux, so the subtraction never leaves the
    spectrum (or an H I line) in absorption.  The area is lowered to the largest value that respects it and the
    baseline re-solved."""
    sw = np.sqrt(w)
    if xb is None:
        den = float(np.sum(w * f * f))
        if den <= 0:
            return 0.0, float(np.sum(w * y * y))
        a = max(float(np.sum(w * f * y)) / den, 0.0)
        if ub is not None:
            a = min(a, _area_cap(f, ub, 0.0, a, thr))
        return a, float(np.sum(w * (y - a * f) ** 2))
    from scipy.optimize import lsq_linear
    A = np.column_stack([f, np.ones_like(f), xb]) * sw[:, None]
    r = lsq_linear(A, y * sw, bounds=([0.0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]), method="bvls")
    a, c0, c1 = (float(v) for v in r.x)
    if ub is not None:
        for _ in range(3):
            cap = _area_cap(f, ub, c0 + c1 * xb, a, thr)
            if a <= cap * (1 + 1e-6):
                break
            a = cap
            # re-solve the baseline at this area
            B = np.column_stack([np.ones_like(f), xb]) * sw[:, None]
            c0, c1 = np.linalg.lstsq(B, (y - a * f) * sw, rcond=None)[0]
    model = a * f + c0 + c1 * xb
    return a, float(np.sum(w * (y - model) ** 2))


def _area_cap(f, ub, base, a_now, thr):
    """Largest area with a f + base <= ub on every pixel where the water (at the current area a_now) is
    significant, a_now f > thr (the noise): the faint wings of the model do not set the cap."""
    ok = np.isfinite(ub) & (a_now * f > max(thr, 0.0)) & (f > 0)
    if not ok.any():
        return np.inf
    return float(max(np.min((ub[ok] - np.broadcast_to(base, f.shape)[ok]) / f[ok]), 0.0))


def fit_local_water(wave, y, sigma, use, distance_pc, s: AccretionSettings, R_model="argyriou2023", R_scale=1.0,
                    core=None):
    """Best LTE water slab for continuum-subtracted data y on `use` pixels; returns (model on all `wave`, info).
    core: pixels of the H I line (masked from the fit); there the water may not exceed the data (+ k x noise), so
    the subtraction cannot put the H I line in absorption."""
    m = _water_model(wave, distance_pc, s.water_release, R_model, R_scale)
    w = np.where(use & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0), 1.0 / np.square(sigma), 0.0)
    yy = np.nan_to_num(y)
    xb = np.asarray(wave, float) - float(np.mean(wave))
    ub, thr = None, 0.0
    if core is not None and s.water_floor_sigma is not None and s.water_floor_sigma >= 0 and np.any(core):
        ok = np.isfinite(y) & np.isfinite(sigma) & (sigma > 0)
        r_ = yy[use & ok]
        noise = max(float(np.nanmedian(sigma[ok])) if ok.any() else 0.0,
                    1.4826 * float(np.median(np.abs(np.diff(r_)))) / math.sqrt(2) if len(r_) > 3 else 0.0)
        ub = np.where(ok & core, yy + s.water_floor_sigma * noise, np.nan)
        thr = noise
    best = (np.inf, None, None, 0.0)

    def flux(T, logN):
        uf, _, _ = m.unit_fluxes({"H2O_local": {"T": float(T), "logN": float(logN), "logR": 0.0}})
        return np.asarray(next(iter(uf.values())), float)

    Tlo, Thi, nT = s.water_T
    Nlo, Nhi, nN = s.water_logN
    # coarse grid (about 1/3 of the configured resolution), then the full resolution around the best cell
    Tc = np.linspace(Tlo, Thi, max(int(nT) // 3, 7))
    Nc = np.linspace(Nlo, Nhi, max(int(nN) // 3, 7))
    for T in Tc:
        for N in Nc:
            f = flux(T, N)
            a, chi2 = _area_fit(f, yy, w, xb, ub, thr)
            if chi2 < best[0]:
                best = (chi2, T, N, a)
    dT = (Thi - Tlo) / max(nT - 1, 1); dN = (Nhi - Nlo) / max(nN - 1, 1)
    T0, N0 = best[1], best[2]
    for T in np.clip(T0 + dT * np.arange(-3, 4), Tlo, Thi):
        for N in np.clip(N0 + dN * np.arange(-3, 4), Nlo, Nhi):
            f = flux(T, N)
            a, chi2 = _area_fit(f, yy, w, xb, ub, thr)
            if chi2 < best[0]:
                best = (chi2, T, N, a)
    # continuous polish of (T, log N) from the best grid cell (Nelder-Mead, area re-solved at every step)
    try:
        from scipy.optimize import minimize

        def obj(v):
            T_, N_ = float(np.clip(v[0], Tlo, Thi)), float(np.clip(v[1], Nlo, Nhi))
            return _area_fit(flux(T_, N_), yy, w, xb, ub, thr)[1]
        r = minimize(obj, [best[1], best[2]], method="Nelder-Mead",
                     options={"xatol": 1.0, "fatol": 1e-3, "maxiter": 120, "initial_simplex":
                              [[best[1], best[2]], [best[1] + dT, best[2]], [best[1], best[2] + dN]]})
        if np.isfinite(r.fun) and r.fun < best[0]:
            T_, N_ = float(np.clip(r.x[0], Tlo, Thi)), float(np.clip(r.x[1], Nlo, Nhi))
            a_, c_ = _area_fit(flux(T_, N_), yy, w, xb, ub, thr)
            best = (c_, T_, N_, a_)
    except Exception:
        pass
    chi2, T, N, a = best
    model = a * flux(T, N)
    ndof = max(int((w > 0).sum()) - 3, 1)
    from .fit import logNA_from
    logR = 0.5 * math.log10(a) if a > 0 else -np.inf
    info = {"water_T": float(T), "water_logN": float(N), "water_logR": float(logR),
            "water_logNA": float(logNA_from(N, logR)) if np.isfinite(logR) else np.nan,
            "water_chi2_red": float(chi2 / ndof)}
    return model, info


def water_overlaps(lines: pd.DataFrame, s: AccretionSettings) -> dict:
    """{H I line: [overlapping water wavelengths]} from the water line list (E_up < water_overlap_eup K,
    A_ul > water_overlap_aul s^-1, within +- water_overlap_dlam resolution elements); None if no list."""
    try:
        from .linedata import load_linelist
        t = load_linelist("H2O", release=s.water_overlap_release, fetch=False).table
    except Exception:
        return None
    sel = t[(t.eu < s.water_overlap_eup) & (t.a > s.water_overlap_aul)]
    w = np.sort(sel.wave.to_numpy(float))
    out = {"_all": w}
    for r in lines.itertuples():
        lo, hi = np.searchsorted(w, [r.wave - s.water_overlap_dlam * r.dlam, r.wave + s.water_overlap_dlam * r.dlam])
        out[r.line] = [float(x) for x in w[lo:hi]]
    return out


# ------------------------------------------------------------------------------------------------ line fit
def _trapz(y, x):
    f = getattr(np, "trapezoid", None) or np.trapz
    return f(y, x)


def _gauss(x, A, mu, s):
    return A * np.exp(-0.5 * ((x - mu) / s) ** 2)


def _jyum_to_wm2(F, lam_um):
    """Integrated F_nu d(lambda) [Jy um] -> W m^-2."""
    lam = lam_um * 1e-6
    return F * 1e-26 * 1e-6 * C_M / lam ** 2


def local_baseline(x, f, core, method: str = "linear", others=None):
    """Local continuum under a line on the cut-out.  linear (default): straight line through the pixels outside
    the line core (and outside other H I lines), 3-sigma clipped -- unbiased by the noise; als: pybaselines
    asymmetric least squares on the whole cut-out (the Shridharan+2026 choice; sits on the lower noise envelope,
    so faint lines come out slightly high)."""
    if method == "als":
        try:
            from pybaselines import Baseline
            return Baseline(x_data=np.arange(len(f))).asls(f, lam=1e4, p=0.01)[0]
        except Exception:
            pass
    keep = ~core
    if others is not None:
        keep &= ~others
    if keep.sum() < 4:
        keep = ~core if (~core).sum() >= 4 else np.ones(len(f), bool)
    xx, ff = x[keep] - x.mean(), f[keep]
    for _ in range(5):
        c = np.polyfit(xx, ff, 1)
        r = ff - np.polyval(c, xx)
        sig = 1.4826 * np.median(np.abs(r - np.median(r)))
        ok = np.abs(r) <= 3 * sig if sig > 0 else np.ones(len(r), bool)
        if ok.all() or ok.sum() < 4:
            break
        xx, ff = xx[ok], ff[ok]
    return np.polyval(c, x - x.mean())


def _multi_gauss(x, *p):
    out = np.zeros_like(x, dtype=float)
    for k in range(0, len(p), 3):
        out += _gauss(x, p[k], p[k + 1], p[k + 2])
    return out


def _distinct_peak(x, ds, j, mu, sg, dlam, dip):
    """True if pixel j is (next to) a local maximum of the smoothed data that is a separate peak from the H I line
    at mu: beyond 2 FWHM of the H I line any local maximum counts; closer in, the data must dip between the two
    peaks to below `dip` times the lower of them (a shoulder on the H I profile is left to the H I Gaussian)."""
    n = len(x)
    w = max(1, int(round(0.5 * dlam / max(np.median(np.diff(x)), 1e-12))))
    lo, hi = max(0, j - w), min(n, j + w + 1)
    k = lo + int(np.argmax(ds[lo:hi]))
    if k in (0, n - 1):
        return abs(x[k] - mu) > 2 * 2.3548 * sg
    if not (ds[k] >= ds[k - 1] and ds[k] >= ds[k + 1]):
        return False
    if abs(x[k] - mu) > 2 * 2.3548 * sg:
        return True
    m = int(np.argmin(np.abs(x - mu)))
    a, b = sorted((k, m))
    if b - a < 2:
        return False
    valley = float(np.min(ds[a:b + 1]))
    return valley <= dip * min(ds[k], ds[m])


def measure_line(wave, flux, err, lam0, dlam, s: AccretionSettings, others=None, blends=None,
                 centre_bounds=None, sigma_bounds=None, water=None, known_peaks=None, dip_test=True):
    """Gaussian fit of one line on (water-subtracted) flux; returns a dict (flux in Jy um and W m^-2).

    `blends`: rest wavelengths of other H I lines close enough to overlap (e.g. (8-6) 7.502 and (11-7) 7.508,
    (7-6) 12.372 and (11-8) 12.387): fitted together, one Gaussian each, and only this line's area is reported.
    blend_fit = "peaks" (default): after the H I fit, every significant residual peak in the fit window (other
    molecules, water residuals, unidentified lines) gets its own Gaussian, added one at a time while BIC improves
    and refitted jointly with the H I line(s) and a linear baseline correction, so the H I flux is the flux of its
    own component only.  H I blend partners share this line's velocity shift and fractional width.
    centre_bounds / sigma_bounds (um): override the free ranges (used by the tied-kinematics pass).
    water: the subtracted water model on the same grid; if given, a local water scale (0..water_scale_max, starting
    at 1) is fitted jointly, so the cut-out itself corrects a slab that is too strong or too weak at this line.
    known_peaks: wavelengths of known lines (screened HITRAN water) near the H I line; a residual peak at one of
    them gets its own component without the dip test, its centre held within +-0.3 dlam of the known line.
    dip_test=False (the tied-kinematics pass, where the H I centre and width are pinned): every residual peak,
    shoulders included, gets its own component."""
    from scipy.optimize import curve_fit
    half = 0.5 * s.cutout_dlam * dlam
    cut = np.abs(wave - lam0) <= max(half, s.fit_dlam * dlam)
    out = {"n_pix": int(cut.sum())}
    if cut.sum() < 8:
        out.update(status="no data"); return out
    x, f, e = wave[cut], flux[cut], err[cut]
    oth = others[cut] if others is not None else None
    wat = np.asarray(water, float)[cut] if water is not None else None
    ok = np.isfinite(f)
    x, f, e = x[ok], f[ok], e[ok]
    others = oth[ok] if oth is not None else None
    wat = wat[ok] if wat is not None else None
    use_w = wat is not None and np.any(np.abs(wat) > 0) and s.water_scale_max > 0
    if len(x) < 8:
        out.update(status="no data"); return out
    core = np.abs(x - lam0) <= 2.5 * dlam
    base = local_baseline(x, f, core, s.baseline, others)
    d = f - base
    resid = d[~core] if (~core).sum() >= 5 else d
    rms_emp = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    e_ok = np.isfinite(e) & (e > 0)
    rms = float(np.median(e[e_ok])) if e_ok.sum() > len(e) // 2 else float(rms_emp)
    sig = np.where(e_ok, e, rms)
    peaks = s.blend_fit == "peaks"
    fitw = np.abs(x - lam0) <= s.fit_dlam * dlam
    if others is not None and not peaks:     # pixels of other lines (water, distant H I) stay out of the fit
        keep_fit = fitw & ~(others & ~core)
        if keep_fit.sum() >= 8:
            fitw = keep_fit
    lo_s, hi_s = s.sigma_range[0] * dlam, s.sigma_range[1] * dlam
    lo_c, hi_c = lam0 - s.centre_dlam * dlam, lam0 + s.centre_dlam * dlam
    if sigma_bounds is not None:
        lo_s, hi_s = float(sigma_bounds[0]), float(max(sigma_bounds[1], sigma_bounds[0] * 1.001))
    if centre_bounds is not None:
        lo_c, hi_c = float(centre_bounds[0]), float(max(centre_bounds[1], centre_bounds[0] + 1e-9))
    if s.fwhm_max_kms:
        # observed FWHM <= fwhm_max_kms, but never narrower than 1.05 x the instrumental profile (beyond ~20 um
        # the MRS resolution element alone is wider than 300 km/s)
        cap = max(s.fwhm_max_kms / C_KMS * lam0 / 2.3548, 1.05 * dlam / 2.3548)
        hi_s = min(hi_s, cap)
        lo_s = min(lo_s, 0.9 * hi_s)
    c_core = np.abs(x - 0.5 * (lo_c + hi_c)) <= 2.5 * dlam
    p0 = [max(float(np.max(d[c_core])) if c_core.any() else float(np.max(d)), 0.0), 0.5 * (lo_c + hi_c),
          0.5 * (lo_s + hi_s)]
    out.update(rms=rms, rms_empirical=float(rms_emp), x=x, data=d, baseline=base)
    dx = float(np.median(np.diff(x)))
    sg_lsf = dlam / 2.3548
    sigF_ul = rms * math.sqrt(2 * math.sqrt(math.pi) * max(lo_s, sg_lsf) * dx)          # Jy um, matched filter
    blends = [b for b in (blends or []) if abs(b - lam0) > 0.5 * dlam]
    # parameters: [A, mu, sigma] of this line, one amplitude per H I blend partner (centre and width follow this
    # line: mu_b = b mu/lam0, sigma_b = sigma b/lam0), the extra Gaussians (A, mu, sigma), then c0, c1
    lo_b = [0.0, lo_c, lo_s]; hi_b = [np.inf, hi_c, hi_s]
    p0_all = list(p0)
    for b in blends:
        nb = np.abs(x - b) <= 1.0 * dlam
        p0_all += [max(float(np.max(d[nb])) if nb.any() else 0.0, 0.0)]
        lo_b += [0.0]; hi_b += [np.inf]
    n_hi = 1 + len(blends)
    nb_ = len(blends)
    bl_arr = np.asarray(blends, float)

    def model(xx, p):
        A, mu, sg = p[0], p[1], p[2]
        out_ = _gauss(xx, A, mu, sg)
        for j in range(nb_):
            out_ = out_ + _gauss(xx, p[3 + j], bl_arr[j] * mu / lam0, sg * bl_arr[j] / lam0)
        ex = p[3 + nb_:-2 - nw_]
        for k in range(0, len(ex), 3):
            out_ = out_ + _gauss(xx, ex[k], ex[k + 1], ex[k + 2])
        if nw_:
            out_ = out_ + (p[-3] - 1.0) * wat_f(xx)          # data = flux - water, true water = ws * water
        return out_ + p[-2] + p[-1] * (xx - lam0)

    def extras(p):
        ex = p[3 + nb_:-2 - nw_]
        return [(float(ex[k]), float(ex[k + 1]), float(ex[k + 2])) for k in range(0, len(ex), 3)]

    def blend_comps(p):
        return [(float(p[3 + j]), float(bl_arr[j] * p[1] / lam0), float(p[2] * bl_arr[j] / lam0)) for j in range(nb_)]
    nw_ = 1 if use_w else 0
    if use_w:
        xw_all = x.copy()

        def wat_f(xx):
            return np.interp(xx, xw_all, wat)
        # non-negativity: flux - ws * water - baseline >= -k sigma on every fit pixel with significant water,
        # i.e. ws <= 1 + (d + k sigma) / water  (d = flux - water - baseline)
        ws_hi = float(s.water_scale_max)
        if s.water_floor_sigma is not None and s.water_floor_sigma >= 0:
            # in the H I core only; noise = the larger of the pipeline error and the empirical rms
            nz = max(rms, float(rms_emp))
            sig_w = (np.abs(x - lam0) <= 2.5 * dlam) & (wat > nz)
            if sig_w.any():
                ws_hi = min(ws_hi, float(np.min(1.0 + (d[sig_w] + s.water_floor_sigma * nz) / wat[sig_w])))
        ws_hi = max(ws_hi, 0.0)
        p0_all += [min(1.0, ws_hi)]; lo_b += [0.0]; hi_b += [max(ws_hi, 1e-6)]
    # linear baseline correction (c0, c1), bounded to a few rms so it cannot swallow a line
    cb = 2.0 * rms
    p0_all += [0.0, 0.0]; lo_b += [-cb, -cb / (s.fit_dlam * dlam)]; hi_b += [cb, cb / (s.fit_dlam * dlam)]
    xf, yf, sf = x[fitw], d[fitw], sig[fitw]

    def fit(p0_, lo_, hi_):
        p0c = np.clip(p0_, np.asarray(lo_) + 1e-12 * np.abs(lo_), hi_)
        popt, pcov = curve_fit(lambda xx, *p: model(xx, p), xf, yf, p0=p0c, sigma=sf,
                               absolute_sigma=True, bounds=(lo_, hi_), maxfev=20000)
        chi2 = float(np.sum(((yf - model(xf, popt)) / sf) ** 2))
        return popt, pcov, chi2

    try:
        popt, pcov, chi2 = fit(p0_all, lo_b, hi_b)
        n_extra = 0
        if peaks:
            lo_x, hi_x = s.blend_sigma_range[0] * dlam, s.blend_sigma_range[1] * dlam
            nfit = len(xf)
            ds = np.convolve(yf, np.ones(3) / 3.0, "same")              # lightly smoothed data for peak shape
            kp = np.asarray(known_peaks if known_peaks is not None else [], float)
            kp = kp[(kp > xf.min()) & (kp < xf.max())] if kp.size else kp
            rej = np.zeros(len(xf), bool)
            mu_ref = 0.5 * (lo_c + hi_c)
            fwhm_exp = 2.3548 * 0.5 * (lo_s + hi_s)
            excl = max(s.blend_min_sep * dlam, s.blend_excl_fwhm * fwhm_exp)    # H I core: no extra component
            sep = s.blend_sep_dlam * dlam                                        # between any two components
            nz = max(rms, float(rms_emp))
            # (a) real peaks of the data: scipy find_peaks on the lightly smoothed spectrum, prominence >= blend_snr
            #     x noise, at least `sep` apart; all of them outside the H I core enter the fit together
            try:
                from scipy.signal import find_peaks
                dpx = max(1, int(round(sep / max(float(np.median(np.diff(xf))), 1e-12))))
                pk, _ = find_peaks(ds, prominence=s.blend_snr * nz, distance=dpx)
            except Exception:
                pk = np.array([], int)
            seeds = [int(j) for j in pk if abs(xf[j] - mu_ref) >= excl
                     and all(abs(xf[j] - c[1]) >= sep for c in blend_comps(popt))]
            seeds = sorted(seeds, key=lambda j: -ds[j])[:int(s.blend_max)]
            if seeds:
                kk = len(popt) - 2 - nw_
                tp0 = list(popt[:kk]); tlo = list(lo_b[:kk]); thi = list(hi_b[:kk])
                for j in seeds:
                    xs = float(xf[j])
                    clo, chi_ = xs - 0.4 * dlam, xs + 0.4 * dlam
                    if xs < mu_ref:
                        chi_ = min(chi_, mu_ref - excl)
                    else:
                        clo = max(clo, mu_ref + excl)
                    if chi_ - clo < 0.05 * dlam:
                        continue
                    tp0 += [max(float(ds[j]), 0.0), float(np.clip(xs, clo, chi_)), max(lo_x, min(hi_x, 0.5 * dlam))]
                    tlo += [0.0, clo, lo_x]; thi += [np.inf, chi_, hi_x]
                tp0 += list(popt[kk:]); tlo += list(lo_b[kk:]); thi += list(hi_b[kk:])
                try:
                    p2, c2, chi2b = fit(tp0, tlo, thi)
                    popt, pcov, chi2, lo_b, hi_b = p2, c2, chi2b, tlo, thi
                    n_extra = len(extras(popt))
                except Exception:
                    pass
            # (b) shoulders and blends without their own maximum: added one at a time from the residual
            n_try = 0
            while n_extra < int(s.blend_max) and n_try < 3 * int(s.blend_max):
                n_try += 1
                r = (yf - model(xf, popt))
                # 3-pixel matched sum of the residual, in units of its noise
                num = np.convolve(r, np.ones(3), "same"); den = np.sqrt(np.convolve(sf ** 2, np.ones(3), "same"))
                S = num / den
                # the tests use the expected H I position (rest wavelength or tied velocity), not the current fit,
                # which can drift towards a strong neighbour before that neighbour has its own component
                mu_hi = mu_ref
                sg_hi = min(float(popt[2]), 0.5 * (lo_s + hi_s))
                bad = (np.abs(xf - mu_hi) < excl) | rej
                for (_, t, _) in extras(popt) + blend_comps(popt):
                    bad |= np.abs(xf - t) < sep
                S = np.where(bad, -np.inf, S)
                i = -1
                kn_at = None
                for j in np.argsort(S)[::-1]:
                    if not np.isfinite(S[j]) or S[j] < s.blend_snr:
                        break
                    if kp.size:
                        dk = np.abs(kp - xf[j])
                        if dk.min() <= 0.5 * dlam and abs(kp[np.argmin(dk)] - mu_hi) >= excl:
                            i = int(j); kn_at = float(kp[np.argmin(dk)]); break
                    if not dip_test or _distinct_peak(xf, ds, j, mu_hi, sg_hi, dlam, s.blend_dip):
                        i = int(j); break
                    S[j] = -np.inf
                if i < 0:
                    break
                xs = float(xf[i]) if kn_at is None else kn_at
                hw = 0.4 * dlam if kn_at is None else 0.3 * dlam
                A0 = max(float(r[i]), 0.0)
                kk = len(popt) - 2 - nw_
                tp0 = list(popt[:kk]) + [A0, xs, max(lo_x, min(hi_x, 0.6 * dlam))] + list(popt[kk:])
                # the extra component's centre may never move closer than blend_min_sep to the expected H I
                # position (it would otherwise slide into the core and take the H I flux)
                clo, chi = xs - hw, xs + hw
                gap = excl
                if xs < mu_hi:
                    chi = min(chi, mu_hi - gap)
                else:
                    clo = max(clo, mu_hi + gap)
                if chi - clo < 0.05 * dlam:
                    rej[i] = True                     # remembered across iterations
                    continue
                xs = float(np.clip(xs, clo, chi))
                tp0[kk + 1] = xs
                tlo = lo_b[:kk] + [0.0, clo, lo_x] + lo_b[kk:]
                thi = hi_b[:kk] + [np.inf, chi, hi_x] + hi_b[kk:]
                try:
                    p2, c2, chi2b = fit(tp0, tlo, thi)
                except Exception:
                    break
                if (chi2 - chi2b) - 3 * math.log(nfit) < s.blend_dbic:
                    break
                popt, pcov, chi2, lo_b, hi_b = p2, c2, chi2b, tlo, thi
                n_extra += 1
        A, mu, sg = popt[:3]
        F = A * sg * math.sqrt(2 * math.pi)
        # flux error: covariance of the linear parameters (all amplitudes, water scale, baseline) at fixed
        # centres and widths -- well conditioned even when extra components overlap -- plus the width error
        # of this line in quadrature (from the full covariance when finite, capped at 50 %)
        cols = [_gauss(xf, 1.0, mu, sg)] + [_gauss(xf, 1.0, c[1], c[2]) for c in blend_comps(popt)] \
            + [_gauss(xf, 1.0, c[1], c[2]) for c in extras(popt)]
        if nw_:
            cols.append(wat_f(xf))
        cols += [np.ones_like(xf), xf - lam0]
        M = np.vstack(cols).T / sf[:, None]
        try:
            cov_lin = np.linalg.pinv(M.T @ M)
            sA = math.sqrt(max(cov_lin[0, 0], 0.0))
        except Exception:
            sA = np.nan
        ssg = math.sqrt(pcov[2, 2]) if np.isfinite(pcov[2, 2]) and pcov[2, 2] > 0 else 0.0
        ssg = min(ssg, 0.5 * sg)
        varF = (sA * sg * math.sqrt(2 * math.pi)) ** 2 + (A * ssg * math.sqrt(2 * math.pi)) ** 2 if np.isfinite(sA) else np.nan
        sF = math.sqrt(varF) if np.isfinite(varF) and varF > 0 else sigF_ul
        sF = max(sF, sigF_ul)
        snr = F / sF if sF > 0 else 0.0
        det = bool(A > s.detect_rms * rms and snr >= s.min_snr)
        comps = [(float(A), float(mu), float(sg))] + blend_comps(popt) + extras(popt)
        lin = popt[-2] + popt[-1] * (x - lam0)
        # flux of the other components inside the H I core (+-1.5 FWHM): how much the decomposition removed
        corew = np.abs(x - mu) <= 1.5 * 2.3548 * sg
        oth_model = sum((_gauss(x, *c) for c in comps[1:]), np.zeros_like(x))
        F_oth_core = float(_trapz(oth_model[corew], x[corew])) if corew.sum() >= 2 else 0.0
        out.update(status="fit", amp=float(A), centre=float(mu), sigma_um=float(sg), flux_jyum=float(F),
                   flux_err_jyum=float(sF), snr=float(snr), peak_snr=float(A / rms) if rms > 0 else np.nan,
                   detected=det, model=_gauss(x, A, mu, sg) + lin, total_model=model(x, popt),
                   components=comps[1:], n_hi_comps=n_hi, n_blend=len(blends), n_extra_peaks=n_extra,
                   blend_flux_core_jyum=F_oth_core, blend_frac_core=F_oth_core / (F_oth_core + F) if (F_oth_core + F) > 0 else np.nan,
                   chi2_red=chi2 / max(len(xf) - len(popt), 1), baseline_corr=lin,
                   water_scale=float(popt[-3]) if nw_ else np.nan,
                   water_resid=(1.0 - popt[-3]) * wat if nw_ else None)
    except Exception as ex:
        out.update(status=f"fit failed: {ex}", detected=False, flux_jyum=np.nan, flux_err_jyum=float(sigF_ul))
    sF = out.get("flux_err_jyum", np.nan)
    sF = float(sF) if np.isfinite(sF) else sigF_ul
    F = out.get("flux_jyum", np.nan)
    # upper limit = measured flux (if any) + 3 sigma, so that a faint line below the detection rule is not
    # given a limit below its own flux
    out["ul_jyum"] = (max(float(F), 0.0) if np.isfinite(F) else 0.0) + 3.0 * sF
    return out


# ------------------------------------------------------------------------------------------------ L_acc, Mdot
def _rstar(rstar, lstar, teff):
    if rstar is not None and np.isfinite(rstar) and rstar > 0:
        return float(rstar)
    if lstar and teff and lstar > 0 and teff > 0:
        return math.sqrt(lstar * L_SUN / (4 * math.pi * SIGMA_SB * teff ** 4)) / R_SUN
    return None


def log_mdot(logLacc, mstar, rstar, rin_over_rstar=5.0):
    """log10 Mdot (Msun/yr) = log10[(1 - R*/R_in)^-1 L_acc R* / (G M*)]."""
    if mstar is None or rstar is None or not np.isfinite(mstar) or mstar <= 0:
        return np.nan
    fac = 1.0 / (1.0 - 1.0 / rin_over_rstar)
    md = fac * (10.0 ** np.asarray(logLacc) * L_SUN) * (rstar * R_SUN) / (G * mstar * M_SUN)   # kg/s
    return np.log10(md * YEAR / M_SUN)


def lacc_mc(flux_wm2, flux_err, d_pc, cal: dict, ext_mag: float, n: int, rng):
    """Monte Carlo log L_acc samples for one line (flux in W m^-2)."""
    Fs = rng.normal(flux_wm2, flux_err, n)
    Fs = np.where(Fs > 0, Fs, flux_wm2 * 1e-3 if flux_wm2 > 0 else np.nan)
    Fs = Fs * 10.0 ** (0.4 * ext_mag)
    logL = np.log10(4 * math.pi * (d_pc * PC) ** 2 * Fs / L_SUN)
    a = rng.normal(cal["a"], cal["a_err"], n); b = rng.normal(cal["b"], cal["b_err"], n)
    return a * (logL - cal["offset"]) + b


def _reference_kinematics(prep, s: AccretionSettings):
    """S/N-weighted median velocity centroid and intrinsic width (LSF removed) of the clean strong H I lines."""
    v, w, wt = [], [], []
    for (_, r, row, _, _, _, _, _, _, m, _) in prep:
        if m.get("status") != "fit" or not m.get("detected") or m.get("snr", 0) < s.tie_min_snr:
            continue
        if (m.get("blend_frac_core") or 0) > 0.3:
            continue
        if row.get("water_overlap", False) and not (row.get("water_frac", 1.0) < 0.5):
            continue
        sg_lsf = r.dlam / 2.3548
        if abs(m["centre"] - r.wave) > 0.9 * s.centre_dlam * r.dlam:      # centre at its bound: pulled by a blend
            continue
        if not (s.sigma_range[0] * r.dlam * 1.02 < m["sigma_um"] < s.sigma_range[1] * r.dlam * 0.98):
            continue
        if s.fwhm_max_kms and 2.3548 * m["sigma_um"] / r.wave * C_KMS > 0.98 * s.fwhm_max_kms:   # at the width cap
            continue
        v.append((m["centre"] / r.wave - 1.0) * C_KMS)
        w.append(math.sqrt(max(m["sigma_um"] ** 2 - sg_lsf ** 2, 0.0)) / r.wave * C_KMS)
        wt.append(m["snr"])
    if len(v) < 2 and not (len(v) == 1 and wt[0] >= 2 * s.tie_min_snr):
        return None

    def wmed(a):
        a, ww = np.asarray(a), np.asarray(wt)
        o = np.argsort(a); c = np.cumsum(ww[o])
        return float(a[o][np.searchsorted(c, 0.5 * c[-1])])
    vr = float(np.clip(wmed(v), -s.tie_vmax_kms, s.tie_vmax_kms))       # spectra are in the rest frame
    return {"v_kms": vr, "sigma_int_kms": wmed(w), "n_lines": len(v)}


def _tied_bounds(lam0, dlam, kin, s: AccretionSettings):
    dv = max(s.tie_dv_kms, 0.25 * dlam / lam0 * C_KMS)
    mu = lam0 * (1 + kin["v_kms"] / C_KMS)
    cb = (mu - lam0 * dv / C_KMS, mu + lam0 * dv / C_KMS)
    sg_lsf = dlam / 2.3548
    si = kin["sigma_int_kms"] / C_KMS * lam0
    lo = math.sqrt(sg_lsf ** 2 + (si * (1 - s.tie_width_frac)) ** 2)
    hi = math.sqrt(sg_lsf ** 2 + (si * (1 + s.tie_width_frac)) ** 2)
    return cb, (lo, hi)


# ------------------------------------------------------------------------------------------------ main
def measure(spec, distance_pc: float | None = None, av: float | None = None, mstar: float | None = None,
            rstar: float | None = None, lstar: float | None = None, teff: float | None = None,
            settings: AccretionSettings | None = None, R_model: str = "argyriou2023", R_scale: float = 1.0,
            global_model=None, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """(per-line table, combined result) for a prepared Spectrum (flux, err, continuum, mask, rest frame).

    global_model: callable(wave) -> continuum-subtracted model flux of the disk's own jalebi fit, used when
    settings.water == "global"."""
    from .instrument import resolving_power
    s = settings or AccretionSettings()
    d = float(distance_pc if distance_pc is not None else spec.distance_pc)
    wave = np.asarray(spec.wave, float); flux = np.asarray(spec.flux, float)
    err = np.asarray(spec.err, float) if spec.err is not None else np.full(len(wave), np.nan)
    cont = np.asarray(spec.continuum, float) if spec.continuum is not None else np.zeros(len(wave))
    mask = np.asarray(spec.mask, bool) if spec.mask is not None else np.isfinite(flux)
    o = np.argsort(wave)
    wave, flux, err, cont, mask = wave[o], flux[o], err[o], cont[o], mask[o]
    lines = hi_lines(float(np.nanmin(wave)), float(np.nanmax(wave)), s.max_nup)
    R = resolving_power(lines.wave.to_numpy(), R_model, R_scale)
    lines["R"] = R; lines["dlam"] = lines.wave / R
    Rall = resolving_power(wave, R_model, R_scale)
    dl_all = wave / Rall
    hi_mask = np.zeros(len(wave), bool)
    for r in lines.itertuples():
        hi_mask |= np.abs(wave - r.wave) <= s.water_mask_dlam * r.dlam
    noise = np.where(np.isfinite(err) & (err > 0), err, np.nan)
    if not np.any(np.isfinite(noise)):
        noise = np.full(len(wave), np.nanstd(flux - cont))
    rng = np.random.default_rng(seed)
    alam, law_used = extinction_alambda_over_av(lines.wave.to_numpy(), s.extinction_law, s.extinction_table, av or 0.0)
    ext_mag = (av or 0.0) * alam
    Rs = _rstar(rstar, lstar, teff)
    overlaps = water_overlaps(lines, s) if (s.water_overlap_check and s.water != "none") else None
    rows, plots, prep = [], {}, []
    for k, r in enumerate(lines.itertuples()):
        row = {"line": r.line, "nu": r.nu, "nl": r.nl, "wave": r.wave, "R": float(r.R), "dlam": float(r.dlam),
               "calibrated": bool(r.calibrated), "note": r.note, "in_combined": r.line in s.combine,
               "A_lambda": float(ext_mag[k])}
        win = np.abs(wave - r.wave) <= s.water_window_um
        if win.sum() < 10 or np.abs(wave[win] - r.wave).min() > 2 * r.dlam:
            row["status"] = "not covered"; rows.append(row); continue
        ww = wave[win]
        water = np.zeros(win.sum())
        ov = overlaps.get(r.line, []) if overlaps is not None else None
        row["water_overlap"] = (len(ov) > 0) if ov is not None else True
        row["water_overlap_lines"] = ",".join(f"{x:.4f}" for x in ov[:6]) if ov else ""
        wmask = None
        if ov is not None and len(ov) == 0:
            row["water_status"] = f"no water line within +-{s.water_overlap_dlam:g} dlam: not fitted"
            # water lines elsewhere in the cut-out are kept out of the local baseline instead of subtracted
            wl = overlaps["_all"]
            wl = wl[(wl > ww.min()) & (wl < ww.max())]
            if len(wl):
                wmask = np.zeros(len(ww), bool)
                for x in wl:
                    wmask |= np.abs(ww - x) <= r.dlam
        elif s.water == "local":
            use = mask[win] & ~hi_mask[win]
            try:
                water, info = fit_local_water(ww, flux[win] - cont[win], noise[win], use, d, s, R_model, R_scale,
                                              core=np.abs(ww - r.wave) <= s.water_mask_dlam * r.dlam)
                row.update(info)
            except Exception as ex:
                row["water_status"] = f"water fit failed: {ex}"
        elif s.water == "global" and global_model is not None:
            try:
                water = np.nan_to_num(np.asarray(global_model(ww), float))
            except Exception as ex:
                row["water_status"] = f"global model failed: {ex}"
        sub = flux[win] - water
        oth = np.zeros(len(ww), bool) if wmask is None else wmask.copy()
        for q in lines.itertuples():
            if q.line != r.line and abs(q.wave - r.wave) > 2.5 * r.dlam:
                oth |= np.abs(ww - q.wave) <= 2.5 * q.dlam
        bl = [q.wave for q in lines.itertuples() if q.line != r.line and abs(q.wave - r.wave) <= 4.0 * r.dlam]
        wat_arg = water if np.any(water != 0) else None
        kp = None                  # known water lines as extra-peak seeds, only where the water is not modelled
        if wat_arg is None and overlaps is not None and len(overlaps.get("_all", [])):
            ka = np.asarray(overlaps["_all"], float)
            kp = ka[np.abs(ka - r.wave) <= s.fit_dlam * r.dlam]
        meas = measure_line(ww, sub, err[win], r.wave, r.dlam, s, others=oth, blends=bl, water=wat_arg, known_peaks=kp)
        if np.any(water != 0) and meas.get("status") == "fit":
            corew = np.abs(ww - r.wave) <= 1.5 * 2.3548 * meas["sigma_um"]
            Fw = float(_trapz(water[corew], ww[corew])) if corew.sum() >= 2 else 0.0
            row["water_frac"] = Fw / (Fw + meas["flux_jyum"]) if (Fw + meas["flux_jyum"]) > 0 else np.nan
        prep.append((k, r, row, win, ww, water, sub, oth, bl, meas, kp))
    # tied kinematics: one H I velocity centroid and intrinsic width for the disk, from the clean strong lines
    kin = _reference_kinematics(prep, s) if s.tie_kinematics else None

    def tied_pass(kin_):
        out_ = []
        for (k, r, row, win, ww, water, sub, oth, bl, meas, kp) in prep:
            cb, sb = _tied_bounds(r.wave, r.dlam, kin_, s)
            m2 = measure_line(ww, sub, err[win], r.wave, r.dlam, s, others=oth, blends=bl, centre_bounds=cb,
                              sigma_bounds=sb, water=water if np.any(water != 0) else None, known_peaks=kp,
                              dip_test=False)
            out_.append(m2 if (m2.get("status") == "fit" or meas.get("status") != "fit") else meas)
        return out_
    tied = None
    if kin is not None:
        tied = tied_pass(kin)
        # the first-pass widths carry unfitted blends (wider than the line): re-derive the reference once from the
        # tied, fully decomposed fits and redo the pass if it moved
        prep2 = [p_[:9] + (m_,) + p_[10:] for p_, m_ in zip(prep, tied)]
        kin2 = _reference_kinematics(prep2, s)
        if kin2 is not None and (abs(kin2["v_kms"] - kin["v_kms"]) > 10
                                 or abs(kin2["sigma_int_kms"] / max(kin["sigma_int_kms"], 1e-6) - 1) > 0.1):
            kin = kin2
            tied = tied_pass(kin)
    for i_p, (k, r, row, win, ww, water, sub, oth, bl, meas, kp) in enumerate(prep):
        if tied is not None:
            meas = tied[i_p]
            row["kinematics"] = "tied"
        else:
            row["kinematics"] = "free"
        x_c = meas.pop("x", None); d_c = meas.pop("data", None); b_c = meas.pop("baseline", None); m_c = meas.pop("model", None)
        t_c = meas.pop("total_model", None); c_c = meas.pop("components", []); l_c = meas.pop("baseline_corr", None)
        wr = meas.pop("water_resid", None)
        ws = meas.get("water_scale", np.nan)
        row["water_scale"] = ws
        if np.isfinite(ws):                                   # report the water as finally fitted in the cut-out
            water = water * ws
            if d_c is not None and wr is not None:
                d_c = d_c + wr; t_c = t_c + wr if t_c is not None else None
        for key in ("status", "amp", "centre", "sigma_um", "flux_jyum", "flux_err_jyum", "snr", "peak_snr", "rms", "detected",
                    "ul_jyum", "n_blend", "n_extra_peaks", "blend_flux_core_jyum", "blend_frac_core", "chi2_red"):
            if key in meas:
                row[key] = meas[key]
        row["detected"] = bool(row.get("detected", False))
        F = _jyum_to_wm2(row.get("flux_jyum", np.nan), r.wave)
        sF = _jyum_to_wm2(row.get("flux_err_jyum", np.nan), r.wave)
        UL = _jyum_to_wm2(row.get("ul_jyum", np.nan), r.wave)
        row.update(flux_Wm2=F, flux_err_Wm2=sF, ul_Wm2=UL)
        # water flux under the line (+-1.5 FWHM of the fitted Gaussian, or +-3 resolution elements)
        sg = row.get("sigma_um", r.dlam / 2.3548)
        core = np.abs(ww - r.wave) <= 1.5 * 2.3548 * sg
        if core.sum() >= 2:
            Fw = float(_trapz(water[core], ww[core]))
            Fh = row.get("flux_jyum", np.nan)
            row["water_flux_jyum"] = Fw
            row["water_frac"] = Fw / (Fw + Fh) if np.isfinite(Fh) and (Fw + Fh) > 0 else np.nan
        Fd = F * 10.0 ** (0.4 * ext_mag[k]) if np.isfinite(F) else np.nan
        row["logL_line"] = math.log10(4 * math.pi * (d * PC) ** 2 * Fd / L_SUN) if np.isfinite(Fd) and Fd > 0 else np.nan
        if r.calibrated:
            cal = CALIBRATIONS[r.line]
            if row["detected"] and np.isfinite(F) and F > 0:
                smp = lacc_mc(F, sF, d, cal, ext_mag[k], s.n_mc, rng)
                q = np.nanpercentile(smp, [16, 50, 84])
                row.update(logLacc=float(q[1]), logLacc_lo=float(q[0]), logLacc_hi=float(q[2]),
                           logLacc_err=float(0.5 * (q[2] - q[0])), upper_limit=False)
            elif np.isfinite(UL) and UL > 0:
                logL = math.log10(4 * math.pi * (d * PC) ** 2 * UL * 10 ** (0.4 * ext_mag[k]) / L_SUN)
                row.update(logLacc=float(cal["a"] * (logL - cal["offset"]) + cal["b"]), logLacc_err=np.nan, upper_limit=True)
            if np.isfinite(row.get("logLacc", np.nan)):
                row["logMdot"] = float(log_mdot(row["logLacc"], mstar, Rs, s.rin_over_rstar))
                row["logMdot_err"] = row.get("logLacc_err", np.nan)
        rows.append(row)
        if x_c is not None:
            plots[r.line] = {"x": x_c, "data": d_c, "baseline": b_c, "model": m_c, "wave_win": ww, "water": water,
                             "flux_win": flux[win] - cont[win], "lam0": r.wave, "total": t_c, "components": c_c,
                             "baseline_corr": l_c}
    table = pd.DataFrame(rows)
    if len(table):
        order = {ln: i for i, ln in enumerate(lines.line)}
        table = table.iloc[np.argsort([order[ln] for ln in table.line], kind="stable")].reset_index(drop=True)
    res = combine(table, s, mstar, Rs, d, av, law_used)
    if kin is not None:
        res.update(HI_v_kms=kin["v_kms"], HI_sigma_int_kms=kin["sigma_int_kms"], HI_kin_lines=kin["n_lines"])
    if s.continuum_variants and s.continuum_variants > 0 and len(table):
        _continuum_variants(spec, table, res, s, dict(distance_pc=d, av=av, mstar=mstar, rstar=rstar, lstar=lstar,
                                                      teff=teff, R_model=R_model, R_scale=R_scale,
                                                      global_model=global_model), seed)
    res["_plots"] = plots
    return table, res


def _continuum_variants(spec, table, res, s: AccretionSettings, kw: dict, seed: int):
    """Optional: re-measure under continuum variants (a per-band offset + tilt of the jalebi continuum, each with
    rms continuum_variant_sigma; moves the water fit and the local baseline); adds logLacc_cont_syst per line and
    combined.  The configured local baseline is kept (als vs linear is a method choice, not a continuum error)."""
    import dataclasses
    rng = np.random.default_rng(seed + 1000)
    vals, comb = {}, []
    for k in range(int(s.continuum_variants)):
        sp = spec.copy()
        c = np.asarray(sp.continuum, float).copy()
        for b in sp.bands:
            i = sp.band_slice(b)
            w = np.asarray(sp.wave, float)[i]
            x = (w - w.mean()) / max(np.ptp(w) / 2, 1e-9) if len(w) > 1 else np.zeros_like(w)
            c[i] = c[i] * (1.0 + rng.normal(0.0, s.continuum_variant_sigma) + rng.normal(0.0, s.continuum_variant_sigma) * x)
        sp.continuum = c
        sv = dataclasses.replace(s, continuum_variants=0, n_mc=200, save_plot=False)
        t, r = measure(sp, settings=sv, seed=seed + k + 1, **kw)
        for rr in t.to_dict("records"):
            if rr.get("calibrated") and np.isfinite(rr.get("logLacc", np.nan)) and not rr.get("upper_limit", True):
                vals.setdefault(rr["line"], []).append(rr["logLacc"])
        if np.isfinite(r.get("logLacc", np.nan)) and not r.get("upper_limit"):
            comb.append(r["logLacc"])
    table["logLacc_cont_syst"] = [float(np.std(vals[l], ddof=1)) if len(vals.get(l, [])) > 1 else np.nan for l in table.line]
    res["logLacc_cont_syst"] = float(np.std(comb, ddof=1)) if len(comb) > 1 else np.nan
    res["n_continuum_variants"] = int(s.continuum_variants)


def combine(table: pd.DataFrame, s: AccretionSettings, mstar, rstar, d, av, law_used) -> dict:
    t = table[table.line.isin(s.combine)] if len(table) else table
    det = t[(t.get("detected", False) == True) & np.isfinite(t.get("logLacc", np.nan)) & np.isfinite(t.get("logLacc_err", np.nan))] if len(t) else t
    out = {"distance_pc": d, "av": av, "extinction_law": law_used, "mstar": mstar, "rstar": rstar,
           "combine": list(s.combine), "rin_over_rstar": s.rin_over_rstar}
    if len(det):
        w = 1.0 / np.square(np.maximum(det.logLacc_err.to_numpy(float), 0.02))
        m = float(np.sum(w * det.logLacc) / np.sum(w))
        out.update(logLacc=m, logLacc_err=float(1.0 / math.sqrt(np.sum(w))),
                   logLacc_spread=float(np.std(det.logLacc, ddof=1)) if len(det) > 1 else np.nan,
                   n_HI_lines=int(len(det)), HI_lines_used=",".join(det.line), upper_limit=False)
    else:
        ul = t[np.isfinite(t.get("logLacc", np.nan))] if len(t) else t
        if len(ul):
            out.update(logLacc=float(ul.logLacc.min()), logLacc_err=np.nan, logLacc_spread=np.nan, n_HI_lines=0,
                       HI_lines_used="", upper_limit=True)
        else:
            out.update(logLacc=np.nan, logLacc_err=np.nan, logLacc_spread=np.nan, n_HI_lines=0, HI_lines_used="",
                       upper_limit=False)
    out["logMdot"] = float(log_mdot(out["logLacc"], mstar, rstar, s.rin_over_rstar)) if np.isfinite(out["logLacc"]) else np.nan
    out["logMdot_err"] = out["logLacc_err"]
    return out


def catalogue_columns(table: pd.DataFrame, res: dict) -> dict:
    """Flat columns for the population table / the app's Batch tab."""
    row = {k: res.get(k) for k in ("logLacc", "logLacc_err", "logLacc_spread", "logMdot", "logMdot_err",
                                   "n_HI_lines", "HI_lines_used", "logLacc_cont_syst")}
    row["logLacc_upper_limit"] = res.get("upper_limit", False)
    for r in table.to_dict("records") if len(table) else []:
        ln = r["line"]
        if r.get("calibrated"):
            row[f"logLacc_{ln}"] = r.get("logLacc", np.nan)
            row[f"logLacc_{ln}_err"] = r.get("logLacc_err", np.nan)
            row[f"logMdot_{ln}"] = r.get("logMdot", np.nan)
            row[f"logLacc_{ln}_ul"] = bool(r.get("upper_limit", False))
        row[f"F_{ln}"] = r.get("flux_Wm2", np.nan)
        row[f"F_{ln}_err"] = r.get("flux_err_Wm2", np.nan)
    for ln in CALIBRATIONS:                    # stable columns: every calibrated line, NaN when not covered
        for k in (f"logLacc_{ln}", f"logLacc_{ln}_err", f"logMdot_{ln}", f"F_{ln}", f"F_{ln}_err"):
            row.setdefault(k, np.nan)
        row.setdefault(f"logLacc_{ln}_ul", False)
    return row


def save(outdir: str, table: pd.DataFrame, res: dict, title: str = "", save_plot: bool = True):
    os.makedirs(outdir, exist_ok=True)
    table.to_csv(os.path.join(outdir, "accretion.csv"), index=False)
    with open(os.path.join(outdir, "accretion.json"), "w") as fh:
        json.dump({k: v for k, v in res.items() if not k.startswith("_")}, fh, indent=1, default=float)
    if save_plot:
        try:
            fig = plot(table, res, title=title)
            fig.savefig(os.path.join(outdir, "accretion.png"), dpi=100)
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:  # pragma: no cover
            pass


def describe(res: dict) -> str:
    if not np.isfinite(res.get("logLacc", np.nan)):
        return "no H I line measured"
    s = f"log L_acc = {res['logLacc']:.2f}"
    if res.get("upper_limit"):
        s = f"log L_acc < {res['logLacc']:.2f} (upper limit)"
    elif np.isfinite(res.get("logLacc_err", np.nan)):
        s += f" +- {res['logLacc_err']:.2f} from {res.get('HI_lines_used', '')}"
    if np.isfinite(res.get("logMdot", np.nan)):
        s += f"; log Mdot = {res['logMdot']:.2f} Msun/yr"
    return s


def plot(table: pd.DataFrame, res: dict, title: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots = res.get("_plots", {})
    names = [n for n in table.line if n in plots]
    n = len(names)
    ncol = 6
    nrow = max(1, math.ceil(n / ncol)) + 1
    fig = plt.figure(figsize=(2.6 * ncol, 2.2 * nrow))
    rec = table.set_index("line")
    for k, ln in enumerate(names):
        ax = fig.add_subplot(nrow, ncol, k + 1)
        p = plots[ln]
        x0 = p["lam0"]
        xv = (p["x"] - x0) * 1e3
        wat = np.interp(p["x"], p["wave_win"], p["water"])
        if np.any(wat != 0):
            ax.step(xv, p["data"] + wat, where="mid", lw=0.7, color="#c3c2b7")          # before water subtraction
            ax.plot(xv, wat, lw=0.9, color="#eb6834")                                     # local water model
        ax.step(xv, p["data"], where="mid", lw=0.8, color="#52514e")                     # after
        lin = p.get("baseline_corr")
        lin = lin if lin is not None else 0.0
        for c in p.get("components") or []:                                               # other peaks
            ax.plot(xv, _gauss(p["x"], *c) + lin, lw=0.8, ls="--", color="#8b8a83")
        if p.get("total") is not None and p.get("components"):
            ax.plot(xv, p["total"], lw=0.8, color="#b3266f", alpha=0.8)                   # total model
        if p["model"] is not None:
            ax.plot(xv, p["model"], lw=1.2, color="#2a78d6")
        ax.axhline(0, lw=0.5, color="#c3c2b7")
        r = rec.loc[ln]
        det = bool(r.get("detected", False))
        lab = f"{ln.replace('HI_', 'H I ')} {x0:.3f}"
        sub = f"S/N {r.get('snr', np.nan):.1f}" + ("" if det else " (UL)")
        if np.isfinite(r.get("logLacc", np.nan)):
            sub += f"  logLacc {r['logLacc']:.2f}" + ("" if det else " UL")
        ax.set_title(lab + "\n" + sub, fontsize=7.5, color="#0b0b0b" if det else "#52514e")
        ax.tick_params(labelsize=6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    ax = fig.add_subplot(nrow, 1, nrow)
    cal = table[table.get("calibrated", False) == True] if len(table) else table
    for j, r in enumerate(cal.to_dict("records")):
        v = r.get("logLacc", np.nan)
        if not np.isfinite(v):
            continue
        if r.get("upper_limit"):
            ax.errorbar([j], [v], yerr=[[0.25], [0]], uplims=True, fmt="v", color="#52514e")
        else:
            ax.errorbar([j], [v], yerr=[[v - r.get("logLacc_lo", v)], [r.get("logLacc_hi", v) - v]], fmt="o",
                        color="#2a78d6" if r["line"] in res.get("combine", []) else "#86b6ef")
    if np.isfinite(res.get("logLacc", np.nan)) and not res.get("upper_limit"):
        m, e = res["logLacc"], res.get("logLacc_err", 0) or 0
        ax.axhspan(m - e, m + e, color="#2a78d6", alpha=0.15, lw=0)
        ax.axhline(m, color="#2a78d6", lw=1)
    ax.set_xticks(range(len(cal))); ax.set_xticklabels([n.replace("HI_", "H I ") for n in cal.line], fontsize=8)
    ax.set_ylabel("log L_acc (L$_\\odot$)")
    ax.set_title(f"{title}  " + describe(res), loc="left", fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    return fig


def run_spectrum(cfg, spec, outdir: str | None = None, global_model=None, say=print) -> tuple[pd.DataFrame, dict]:
    """Accretion of a prepared spectrum with the config's target parameters; written to `outdir` when given."""
    s = settings_from_config(cfg)
    t = cfg.target
    table, res = measure(spec, distance_pc=t.distance_pc, av=t.av, mstar=t.mstar, rstar=t.rstar, lstar=t.lstar,
                         teff=t.teff, settings=s, R_model=cfg.R_model, R_scale=cfg.R_scale, global_model=global_model)
    if outdir:
        save(outdir, table, res, title=spec.name, save_plot=s.save_plot)
    say("accretion (H I lines, Shridharan+2026): " + describe(res))
    return table, res


def run_outdir(outdir: str, config: str | None = None, overrides: dict | None = None, say=print):
    from .config import ProjectConfig
    from .data import load_csv
    cfg = ProjectConfig.load(config or os.path.join(outdir, "config.yaml"))
    for k, v in (overrides or {}).items():
        if v is not None and hasattr(cfg.target, k):
            setattr(cfg.target, k, v)
    spec = load_csv(os.path.join(outdir, "prep.csv"))
    spec.distance_pc = cfg.target.distance_pc
    return run_spectrum(cfg, spec, outdir=outdir, say=say)


def survey_table(root: str) -> pd.DataFrame:
    import glob
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "*", "accretion.json"))):
        try:
            d = json.load(open(f))
            t = pd.read_csv(os.path.join(os.path.dirname(f), "accretion.csv"))
        except Exception:
            continue
        row = {"folder": os.path.basename(os.path.dirname(f))}
        row.update(catalogue_columns(t, d))
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out.to_csv(os.path.join(root, "accretion_survey.csv"), index=False)
    return out
