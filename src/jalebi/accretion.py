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


def _area_fit(f, y, w):
    den = float(np.sum(w * f * f))
    if den <= 0:
        return 0.0, float(np.sum(w * y * y))
    a = max(float(np.sum(w * f * y)) / den, 0.0)
    return a, float(np.sum(w * (y - a * f) ** 2))


def fit_local_water(wave, y, sigma, use, distance_pc, s: AccretionSettings, R_model="argyriou2023", R_scale=1.0):
    """Best LTE water slab for continuum-subtracted data y on `use` pixels; returns (model on all `wave`, info)."""
    m = _water_model(wave, distance_pc, s.water_release, R_model, R_scale)
    w = np.where(use & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0), 1.0 / np.square(sigma), 0.0)
    yy = np.nan_to_num(y)
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
            a, chi2 = _area_fit(f, yy, w)
            if chi2 < best[0]:
                best = (chi2, T, N, a)
    dT = (Thi - Tlo) / max(nT - 1, 1); dN = (Nhi - Nlo) / max(nN - 1, 1)
    T0, N0 = best[1], best[2]
    for T in np.clip(T0 + dT * np.arange(-3, 4), Tlo, Thi):
        for N in np.clip(N0 + dN * np.arange(-3, 4), Nlo, Nhi):
            f = flux(T, N)
            a, chi2 = _area_fit(f, yy, w)
            if chi2 < best[0]:
                best = (chi2, T, N, a)
    # continuous polish of (T, log N) from the best grid cell (Nelder-Mead, area re-solved at every step)
    try:
        from scipy.optimize import minimize

        def obj(v):
            T_, N_ = float(np.clip(v[0], Tlo, Thi)), float(np.clip(v[1], Nlo, Nhi))
            return _area_fit(flux(T_, N_), yy, w)[1]
        r = minimize(obj, [best[1], best[2]], method="Nelder-Mead",
                     options={"xatol": 1.0, "fatol": 1e-3, "maxiter": 120, "initial_simplex":
                              [[best[1], best[2]], [best[1] + dT, best[2]], [best[1], best[2] + dN]]})
        if np.isfinite(r.fun) and r.fun < best[0]:
            T_, N_ = float(np.clip(r.x[0], Tlo, Thi)), float(np.clip(r.x[1], Nlo, Nhi))
            a_, c_ = _area_fit(flux(T_, N_), yy, w)
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


# ------------------------------------------------------------------------------------------------ line fit
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


def measure_line(wave, flux, err, lam0, dlam, s: AccretionSettings, others=None, blends=None):
    """Gaussian fit of one line on (water-subtracted) flux; returns a dict (flux in Jy um and W m^-2).
    `blends`: rest wavelengths of other H I lines close enough to overlap (e.g. (8-6) 7.502 and (11-7) 7.508,
    (7-6) 12.372 and (11-8) 12.387): fitted together, one Gaussian each, and only this line's area is reported."""
    from scipy.optimize import curve_fit
    half = 0.5 * s.cutout_dlam * dlam
    cut = np.abs(wave - lam0) <= max(half, s.fit_dlam * dlam)
    out = {"n_pix": int(cut.sum())}
    if cut.sum() < 8:
        out.update(status="no data"); return out
    x, f, e = wave[cut], flux[cut], err[cut]
    oth = others[cut] if others is not None else None
    ok = np.isfinite(f)
    x, f, e = x[ok], f[ok], e[ok]
    others = oth[ok] if oth is not None else None
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
    fitw = np.abs(x - lam0) <= s.fit_dlam * dlam
    lo_s, hi_s = s.sigma_range[0] * dlam, s.sigma_range[1] * dlam
    p0 = [max(float(np.max(d[core])) if core.any() else float(np.max(d)), 0.0), lam0, 0.5 * (lo_s + hi_s)]
    out.update(rms=rms, rms_empirical=float(rms_emp), x=x, data=d, baseline=base)
    dx = float(np.median(np.diff(x)))
    sg_lsf = dlam / 2.3548
    sigF_ul = rms * math.sqrt(2 * math.sqrt(math.pi) * max(lo_s, sg_lsf) * dx)          # Jy um, matched filter
    blends = [b for b in (blends or []) if abs(b - lam0) > 0.5 * dlam]
    lo_b = [0.0, lam0 - s.centre_dlam * dlam, lo_s]; hi_b = [np.inf, lam0 + s.centre_dlam * dlam, hi_s]
    p0_all = list(p0)
    for b in blends:
        nb = np.abs(x - b) <= 1.0 * dlam
        p0_all += [max(float(np.max(d[nb])) if nb.any() else 0.0, 0.0), b, 0.5 * (lo_s + hi_s)]
        lo_b += [0.0, b - s.centre_dlam * dlam, lo_s]; hi_b += [np.inf, b + s.centre_dlam * dlam, hi_s]
    try:
        popt, pcov = curve_fit(_multi_gauss, x[fitw], d[fitw], p0=p0_all, sigma=sig[fitw], absolute_sigma=True,
                               bounds=(lo_b, hi_b), maxfev=6000)
        A, mu, sg = popt[:3]
        F = A * sg * math.sqrt(2 * math.pi)
        J = np.zeros(len(popt)); J[0] = sg * math.sqrt(2 * math.pi); J[2] = A * math.sqrt(2 * math.pi)
        varF = float(J @ pcov @ J) if np.all(np.isfinite(pcov)) else np.nan
        out["n_blend"] = len(blends)
        sF = math.sqrt(varF) if np.isfinite(varF) and varF > 0 else sigF_ul
        sF = max(sF, sigF_ul)
        snr = F / sF if sF > 0 else 0.0
        det = bool(A > s.detect_rms * rms and snr >= s.min_snr)
        out.update(status="fit", amp=float(A), centre=float(mu), sigma_um=float(sg), flux_jyum=float(F),
                   flux_err_jyum=float(sF), snr=float(snr), peak_snr=float(A / rms) if rms > 0 else np.nan,
                   detected=det, model=_gauss(x, A, mu, sg))
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
    rows, plots = [], {}
    for k, r in enumerate(lines.itertuples()):
        row = {"line": r.line, "nu": r.nu, "nl": r.nl, "wave": r.wave, "R": float(r.R), "dlam": float(r.dlam),
               "calibrated": bool(r.calibrated), "note": r.note, "in_combined": r.line in s.combine,
               "A_lambda": float(ext_mag[k])}
        win = np.abs(wave - r.wave) <= s.water_window_um
        if win.sum() < 10 or np.abs(wave[win] - r.wave).min() > 2 * r.dlam:
            row["status"] = "not covered"; rows.append(row); continue
        ww = wave[win]
        water = np.zeros(win.sum())
        if s.water == "local":
            use = mask[win] & ~hi_mask[win]
            try:
                water, info = fit_local_water(ww, flux[win] - cont[win], noise[win], use, d, s, R_model, R_scale)
                row.update(info)
            except Exception as ex:
                row["water_status"] = f"water fit failed: {ex}"
        elif s.water == "global" and global_model is not None:
            try:
                water = np.nan_to_num(np.asarray(global_model(ww), float))
            except Exception as ex:
                row["water_status"] = f"global model failed: {ex}"
        sub = flux[win] - water
        oth = np.zeros(len(ww), bool)
        for q in lines.itertuples():
            if q.line != r.line and abs(q.wave - r.wave) > 2.5 * r.dlam:
                oth |= np.abs(ww - q.wave) <= 2.5 * q.dlam
        bl = [q.wave for q in lines.itertuples() if q.line != r.line and abs(q.wave - r.wave) <= 4.0 * r.dlam]
        meas = measure_line(ww, sub, err[win], r.wave, r.dlam, s, others=oth, blends=bl)
        x_c = meas.pop("x", None); d_c = meas.pop("data", None); b_c = meas.pop("baseline", None); m_c = meas.pop("model", None)
        for key in ("status", "amp", "centre", "sigma_um", "flux_jyum", "flux_err_jyum", "snr", "peak_snr", "rms", "detected",
                    "ul_jyum", "n_blend"):
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
            Fw = float(np.trapezoid(water[core], ww[core])) if hasattr(np, "trapezoid") else float(np.trapz(water[core], ww[core]))
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
                             "flux_win": flux[win] - cont[win], "lam0": r.wave}
    table = pd.DataFrame(rows)
    res = combine(table, s, mstar, Rs, d, av, law_used)
    res["_plots"] = plots
    return table, res


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
                                   "n_HI_lines", "HI_lines_used")}
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
        ax.step((p["x"] - x0) * 1e3, p["data"], where="mid", lw=0.8, color="#52514e")
        win = np.abs(p["wave_win"] - x0) <= (p["x"].max() - p["x"].min()) / 2
        ax.plot((p["wave_win"][win] - x0) * 1e3, p["water"][win], lw=0.8, color="#eb6834")
        if p["model"] is not None:
            ax.plot((p["x"] - x0) * 1e3, p["model"], lw=1.2, color="#2a78d6")
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
