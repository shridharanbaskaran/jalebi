"""Emission-line catalogue and the Gaussian line-fitting core.

The line maps of :mod:`jalebi.cube` and any single-line measurement on a 1-D spectrum use the same
pieces, so a flux or a centroid means the same thing everywhere in JALEBI:

* ``LINES`` / ``get_line("H2 S(1)")`` — rest vacuum wavelengths (micron) of the H2 pure-rotational
  lines, the fine-structure lines seen in jets and winds ([Ne II], [Fe II], [Ar II], ...) and the H I
  recombination lines in the MIRI range.  Names are forgiving: ``"[NeII]"``, ``"NeII 12.81"``,
  ``"h2 s1"`` and a plain wavelength (``12.8135``) all work.
* ``fit_gaussian_batch`` — a vectorised, bounded Levenberg–Marquardt fit of a Gaussian (plus an
  optional constant) to many spectra at once (every spaxel of a cube, or Monte Carlo copies of one
  spectrum).  Used for centroid velocity maps; ~10^5 spectra per second on one core.
* ``fit_line`` — one line in a 1-D spectrum (JALEBI ``Spectrum`` or plain arrays): local polynomial
  continuum, Gaussian with the width tied to the instrumental resolution, Monte Carlo errors.  It is
  the packaged form of ``find_and_measure_line`` in the old ``line_fit_essentials.py``.

Unit conversions (the integrated Gaussian area A is in [flux unit x micron]):
  Jy um      -> W m^-2             : A * c / lambda^2 * 1e-26   = A * 2.998e-12 / lambda[um]^2
  MJy/sr um  -> erg s^-1 cm^-2 sr^-1: A * c / lambda^2 * 1e-17   = A * 2.998e-3  / lambda[um]^2
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

C_KMS = 299792.458
C_UM_S = 2.99792458e14
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


# ----------------------------------------------------------------------------------------------
# Catalogue
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Line:
    """One emission line: a rest vacuum wavelength and, for H2, the upper-level data."""
    name: str
    wave: float                    # micron (vacuum, rest frame)
    species: str                   # "H2", "[Ne II]", "H I", ...
    kind: str = "atomic"           # "molecular", "atomic", "recombination"
    eu_K: float | None = None      # upper-level energy (K)
    a_ul: float | None = None      # Einstein A (s^-1)
    gu: float | None = None        # upper-level degeneracy
    note: str = ""

    @property
    def tag(self) -> str:
        """File-name-safe label, e.g. 'H2_S1', 'NeII_12.81', 'HI_7-6'."""
        s = self.name
        sp = self.species.strip()
        if sp and s.startswith(sp):
            s = sp.replace("[", "").replace("]", "").replace(" ", "") + s[len(sp):]
        s = s.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
        return re.sub(r"[^A-Za-z0-9_.+-]+", "_", s).strip("_")


def _h2(J: int, wave: float, eu: float, a: float, gu: float) -> Line:
    return Line(f"H2 S({J})", wave, "H2", "molecular", eu, a, gu, "H2 0-0 pure rotational")


# H2 0-0 S(J): wavelengths, E_u, A_ul and g_u from the bundled HITRAN 2020 list (S(0) from Roueff et al. 2019)
_H2 = [
    _h2(0, 28.218836, 509.85, 2.943e-11, 5.0),
    _h2(1, 17.034846, 1015.08, 4.758e-10, 21.0),
    _h2(2, 12.278612, 1681.64, 2.753e-09, 9.0),
    _h2(3, 9.664911, 2503.74, 9.830e-09, 33.0),
    _h2(4, 8.025041, 3474.50, 2.641e-08, 13.0),
    _h2(5, 6.909509, 4586.06, 5.876e-08, 45.0),
    _h2(6, 6.108564, 5829.84, 1.141e-07, 17.0),
    _h2(7, 5.511183, 7196.71, 2.000e-07, 57.0),
    _h2(8, 5.053115, 8677.15, 3.234e-07, 21.0),
]

# Fine-structure lines (vacuum wavelengths, NIST / ISO line lists)
_FS = [
    Line("[Fe II] 5.34", 5.340169, "[Fe II]", note="a4F9/2 - a6D9/2; jets"),
    Line("[Fe II] 5.67", 5.673904, "[Fe II]"),
    Line("[Fe II] 6.72", 6.721283, "[Fe II]"),
    Line("[Fe II] 17.94", 17.935950, "[Fe II]", note="a4F7/2 - a4F9/2"),
    Line("[Fe II] 24.52", 24.519250, "[Fe II]"),
    Line("[Fe II] 25.99", 25.988290, "[Fe II]", note="a6D7/2 - a6D9/2"),
    Line("[Ni II] 6.64", 6.636000, "[Ni II]"),
    Line("[Ni II] 10.68", 10.682200, "[Ni II]"),
    Line("[Ar II] 6.99", 6.985274, "[Ar II]", note="jets, photoevaporative winds"),
    Line("[Ar III] 8.99", 8.991380, "[Ar III]"),
    Line("[S IV] 10.51", 10.510500, "[S IV]"),
    Line("[Ne II] 12.81", 12.813548, "[Ne II]", note="jets, photoevaporative winds"),
    Line("[Ne V] 14.32", 14.321700, "[Ne V]"),
    Line("[Ne III] 15.56", 15.555100, "[Ne III]"),
    Line("[S III] 18.71", 18.713000, "[S III]"),
    Line("[Ne V] 24.32", 24.317500, "[Ne V]"),
    Line("[S I] 25.25", 25.249000, "[S I]", note="shocks"),
]


def _hi_lines() -> list[Line]:
    """H I lines of the bundled Atomic_lines.csv that fall in the MIRI range."""
    import os
    path = os.path.join(os.path.dirname(__file__), "data_files", "Atomic_lines.csv")
    out = []
    try:
        with open(path, encoding="utf-8-sig") as fh:
            next(fh)
            for row in fh:
                parts = [p.strip() for p in row.split(",")]
                if len(parts) >= 3 and parts[1] == "HI":
                    w = float(parts[0])
                    if 4.8 < w < 28.5:
                        up_lo = parts[2].strip("()")
                        out.append(Line(f"H I {up_lo}", w, "H I", "recombination"))
    except OSError:
        pass
    return out


LINES: dict[str, Line] = {ln.name: ln for ln in _H2 + _FS + _hi_lines()}

# groups for stacking and for "all H2 lines in this cube"
GROUPS: dict[str, list[str]] = {
    "H2": [ln.name for ln in _H2],
    "H2 low-J": ["H2 S(1)", "H2 S(2)", "H2 S(3)"],
    "H2 high-J": ["H2 S(4)", "H2 S(5)", "H2 S(6)", "H2 S(7)", "H2 S(8)"],
    "[Fe II]": [ln.name for ln in _FS if ln.species == "[Fe II]"],
    "jet": ["[Fe II] 5.34", "[Ni II] 6.64", "[Ar II] 6.99", "[Ne II] 12.81", "[Fe II] 17.94", "[Fe II] 25.99"],
}


def _norm(s: str) -> str:
    return re.sub(r"[\s\[\]()_\-]+", "", s.lower())


def get_line(name: str | float | Line, tol_um: float = 0.01) -> Line:
    """Look up a line by name, alias or wavelength (micron).

    Accepts the catalogue names ("H2 S(1)", "[Ne II] 12.81"), relaxed spellings ("h2 s1", "NeII",
    "[FeII] 5.34", "H I 7-6", "Hu alpha" is not supported), a number in micron (the nearest catalogue
    line within `tol_um`, otherwise an ad-hoc line of that wavelength), or "name=wavelength" (or
    "wavelength=name") for an exact wavelength or a line that is not in the catalogue ("CO P(10)=4.9876")."""
    if isinstance(name, Line):
        return name
    if isinstance(name, (int, float, np.floating)):
        w = float(name)
        best = min(LINES.values(), key=lambda ln: abs(ln.wave - w))
        return best if abs(best.wave - w) <= tol_um else Line(f"{w:.4f} um", w, "?")
    s = str(name).strip()
    if "=" in s:
        nm, w = s.split("=", 1)
        try:
            float(w)
        except ValueError:
            nm, w = w, nm                     # "5.3402=FeII" works as well as "FeII=5.3402"
        return Line(nm.strip(), float(w), nm.strip().split(" ")[0])
    try:
        return get_line(float(s), tol_um)
    except ValueError:
        pass
    if s in LINES:
        return LINES[s]
    key = _norm(s)
    # H2 S(J) spelled loosely: "h2s1", "h2 s(1)", "H2_S1", "S(1)"
    m = re.fullmatch(r"(?:h2)?(?:00)?s(\d)", key)
    if m:
        return LINES[f"H2 S({m.group(1)})"]
    exact = [ln for ln in LINES.values() if _norm(ln.name) == key]
    if exact:
        return exact[0]
    # species + optional wavelength ("neii", "feii5.34", "neii12.8")
    m = re.fullmatch(r"([a-z]+)([0-9.]*)", key)
    if m:
        sp, wv = m.group(1), m.group(2)
        cands = [ln for ln in LINES.values() if _norm(ln.species) == sp]
        if cands:
            if wv:
                w = float(wv)
                return min(cands, key=lambda ln: abs(ln.wave - w))
            if len(cands) == 1 or sp != "h2":
                return cands[0]          # e.g. "[Fe II]" -> [Fe II] 5.34, the first catalogue entry
            raise KeyError(f"{name!r} is ambiguous: give the transition, e.g. 'H2 S(1)', or use the group 'H2'")
    raise KeyError(f"unknown line {name!r}. Use a catalogue name (jalebi cube lines), a wavelength in micron, "
                   f"or 'name=wavelength'.")


def expand_lines(items) -> list[Line]:
    """Names, wavelengths or group names (GROUPS) -> list of Line (duplicates removed, order kept)."""
    out: list[Line] = []
    for it in items if isinstance(items, (list, tuple)) else [items]:
        if isinstance(it, str) and it in GROUPS:
            lines = [LINES[n] for n in GROUPS[it]]
        else:
            lines = [get_line(it)]
        for ln in lines:
            if ln not in out:
                out.append(ln)
    return out


def lines_in_range(wmin: float, wmax: float, species: str | None = None) -> list[Line]:
    out = [ln for ln in LINES.values() if wmin <= ln.wave <= wmax]
    if species:
        out = [ln for ln in out if _norm(ln.species) == _norm(species)]
    return sorted(out, key=lambda ln: ln.wave)


def catalogue_table():
    """The catalogue as a pandas DataFrame (name, wavelength, species, E_u, A_ul, g_u, note)."""
    import pandas as pd
    return pd.DataFrame([{"name": ln.name, "wave_um": ln.wave, "species": ln.species, "kind": ln.kind,
                          "eu_K": ln.eu_K, "a_ul": ln.a_ul, "gu": ln.gu, "note": ln.note} for ln in LINES.values()]
                        ).sort_values("wave_um").reset_index(drop=True)


# ----------------------------------------------------------------------------------------------
# Resolution and unit helpers
# ----------------------------------------------------------------------------------------------

def resolving_power(wave_um, model: str = "argyriou2023"):
    from .instrument import resolving_power as _rp
    return _rp(wave_um, model)


def lsf_fwhm_kms(wave_um, model: str = "argyriou2023"):
    """Instrumental FWHM in km/s (c / R)."""
    return C_KMS / np.asarray(resolving_power(wave_um, model), float)


def lsf_sigma_um(wave_um, model: str = "argyriou2023"):
    w = np.asarray(wave_um, float)
    return w / np.asarray(resolving_power(w, model), float) * FWHM_TO_SIGMA


def area_to_W_m2(area_jy_um, wave_um):
    """Integrated area of a line in a Jy spectrum [Jy um] -> line flux [W m^-2]."""
    return np.asarray(area_jy_um) * C_UM_S * 1e-26 / np.asarray(wave_um) ** 2


def area_to_cgs_sr(area_mjysr_um, wave_um):
    """Integrated area in a surface-brightness cube [MJy sr^-1 um] -> [erg s^-1 cm^-2 sr^-1]."""
    return np.asarray(area_mjysr_um) * C_UM_S * 1e-17 / np.asarray(wave_um) ** 2


def velocity_kms(wave_um, rest_um):
    return C_KMS * (np.asarray(wave_um) - rest_um) / rest_um


# ----------------------------------------------------------------------------------------------
# Batched Gaussian fitting
# ----------------------------------------------------------------------------------------------

@dataclass
class BatchFit:
    """Result of fit_gaussian_batch; arrays have the batch shape (n,)."""
    amp: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    offset: np.ndarray
    cov: np.ndarray               # (n, k, k) parameter covariance (k = 3 or 4)
    chi2: np.ndarray
    dof: np.ndarray
    ok: np.ndarray                # converged, finite, better than no line, centre not pinned to a bound
    sigma_at_bound: np.ndarray    # width pinned to its lower/upper bound (the centroid is still valid)

    @property
    def area(self) -> np.ndarray:
        return self.amp * self.sigma * np.sqrt(2 * np.pi)

    @property
    def area_err(self) -> np.ndarray:
        # d(area) = sqrt(2pi) (sigma dA + A dsigma)
        g = np.sqrt(2 * np.pi) * np.stack([self.sigma, np.zeros_like(self.mu), self.amp], axis=-1)
        c = self.cov[:, :3, :3]
        return np.sqrt(np.clip(np.einsum("ni,nij,nj->n", g, c, g), 0, None))

    @property
    def mu_err(self) -> np.ndarray:
        return np.sqrt(np.clip(self.cov[:, 1, 1], 0, None))


def _gauss_model(x, P, with_offset, jac=True):
    a, m, s = P[:, 0:1], P[:, 1:2], P[:, 2:3]
    z = (x[None, :] - m) / s
    e = np.exp(-0.5 * z * z)
    f = a * e
    if not jac:
        return (f + P[:, 3:4] if with_offset else f), None
    J = [e, a * e * z / s, a * e * z * z / s]
    if with_offset:
        f = f + P[:, 3:4]
        J.append(np.ones_like(e))
    return f, np.stack(J, axis=-1)


def fit_gaussian_batch(x, Y, W, p0, lo, hi, with_offset: bool = False, maxiter: int = 60,
                       chunk: int = 20000) -> BatchFit:
    """Bounded Levenberg–Marquardt fit of  A exp(-(x-mu)^2 / 2 sigma^2) [+ c]  to every row of Y.

    x      : (m,) common abscissa (e.g. wavelength in micron)
    Y, W   : (n, m) data and weights 1/sigma (0 = ignore the pixel)
    p0     : (n, k) start values [A, mu, sigma(, c)];  lo, hi : (k,) or (n, k) bounds
    Returns a BatchFit; `ok` is False where the fit hit a bound on mu or sigma, did not reduce chi2
    below the no-line value, or has fewer than k+1 usable pixels."""
    x = np.asarray(x, float)
    Y = np.atleast_2d(np.asarray(Y, float)); W = np.atleast_2d(np.asarray(W, float))
    n, m = Y.shape
    k = 4 if with_offset else 3
    p0 = np.atleast_2d(np.asarray(p0, float))[:, :k]
    lo = np.broadcast_to(np.asarray(lo, float)[..., :k], (n, k)); hi = np.broadcast_to(np.asarray(hi, float)[..., :k], (n, k))
    W = np.where(np.isfinite(Y) & np.isfinite(W), W, 0.0)
    Y = np.where(W > 0, Y, 0.0)
    out_P = np.empty((n, k)); out_cov = np.full((n, k, k), np.nan); out_chi2 = np.empty(n); out_ok = np.zeros(n, bool)
    out_sb = np.zeros(n, bool)
    dof = (W > 0).sum(axis=1) - k
    eye = np.eye(k)[None]
    for s0 in range(0, n, chunk):
        sl = slice(s0, min(n, s0 + chunk))
        y, w = Y[sl], W[sl]
        L, U = lo[sl], hi[sl]
        P = np.clip(p0[sl].copy(), L, U)
        nn = len(y)
        lam = np.full(nn, 1e-2)
        f, _ = _gauss_model(x, P, with_offset, jac=False)
        chi2 = np.sum(((y - f) * w) ** 2, axis=1)
        act = np.arange(nn)                          # rows still iterating (converged rows drop out)
        for _ in range(maxiter):
            if act.size == 0:
                break
            ya, wa, Pa = y[act], w[act], P[act]
            fa, Ja = _gauss_model(x, Pa, with_offset)
            Jw = Ja * wa[..., None]
            JwT = np.swapaxes(Jw, 1, 2)
            A = JwT @ Jw                                              # (na, k, k)
            g = (JwT @ ((ya - fa) * wa)[..., None])[..., 0]           # (na, k)
            D = A * eye
            D = np.where(D > 0, D, 1e-30 * eye)
            M = A + lam[act, None, None] * D
            try:
                dP = np.linalg.solve(M, g[..., None])[..., 0]
            except np.linalg.LinAlgError:
                dP = (np.linalg.pinv(M) @ g[..., None])[..., 0]
            Pn = np.clip(Pa + dP, L[act], U[act])
            fn, _ = _gauss_model(x, Pn, with_offset, jac=False)
            chi2n = np.sum(((ya - fn) * wa) ** 2, axis=1)
            better = chi2n < chi2[act]
            rel = (chi2[act] - chi2n) / np.maximum(chi2[act], 1e-300)
            P[act[better]] = Pn[better]
            chi2[act[better]] = chi2n[better]
            lam[act] = np.where(better, np.maximum(lam[act] / 3.0, 1e-7), np.minimum(lam[act] * 4.0, 1e7))
            done = (better & (rel < 1e-6)) | (~better & (lam[act] > 1e3))
            act = act[~done]
        f, J = _gauss_model(x, P, with_offset)
        Jw = J * w[..., None]
        A = np.swapaxes(Jw, 1, 2) @ Jw
        with np.errstate(all="ignore"):
            try:
                cov = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                cov = np.linalg.pinv(A)
        if with_offset:
            c0 = np.sum(y * w * w, 1, keepdims=True) / np.maximum(np.sum(w * w, 1, keepdims=True), 1e-300)
            chi2_null = np.sum(((y - c0) * w) ** 2, axis=1)
        else:
            chi2_null = np.sum((y * w) ** 2, axis=1)
        span = U - L
        pinned = (np.abs(P[:, 1:3] - L[:, 1:3]) < 1e-6 * span[:, 1:3]) | (np.abs(P[:, 1:3] - U[:, 1:3]) < 1e-6 * span[:, 1:3])
        ok = np.isfinite(chi2) & (chi2 < chi2_null) & ~pinned[:, 0] & (dof[sl] > 0) & np.all(np.isfinite(P), axis=1)
        out_P[sl] = P; out_cov[sl] = cov; out_chi2[sl] = chi2; out_ok[sl] = ok; out_sb[sl] = pinned[:, 1]
    offset = out_P[:, 3] if with_offset else np.zeros(n)
    return BatchFit(out_P[:, 0], out_P[:, 1], out_P[:, 2], offset, out_cov, out_chi2, dof, out_ok, out_sb)


def gaussian_start(x, Y, rest_um, sigma0):
    """Start values [A, mu, sigma, 0] from the data: peak amplitude, flux-weighted centroid of the
    positive part near the line, instrumental sigma."""
    Yp = np.clip(np.nan_to_num(Y), 0, None)
    near = np.abs(x - rest_um) < 3 * np.max(sigma0)
    Ypn = Yp[:, near]
    tot = Ypn.sum(axis=1)
    mu = np.where(tot > 0, (Ypn * x[near][None]).sum(axis=1) / np.maximum(tot, 1e-300), rest_um)
    amp = np.nanmax(np.where(near[None], np.nan_to_num(Y), -np.inf), axis=1)
    amp = np.where(np.isfinite(amp) & (amp > 0), amp, np.nanmax(np.abs(np.nan_to_num(Y)), axis=1) * 0.1 + 1e-30)
    sig = np.broadcast_to(np.asarray(sigma0, float), mu.shape)
    return np.stack([amp, mu, sig, np.zeros_like(mu)], axis=1)


# ----------------------------------------------------------------------------------------------
# One line in a 1-D spectrum
# ----------------------------------------------------------------------------------------------

@dataclass
class LineFit:
    line: Line
    area: float                   # [flux unit x um]
    area_err: float
    flux_W_m2: float | None       # when the spectrum is in Jy
    flux_W_m2_err: float | None
    center_um: float
    v_kms: float
    v_err_kms: float
    fwhm_kms: float
    fwhm_err_kms: float
    snr: float
    detected: bool
    wave: np.ndarray = field(repr=False)
    data: np.ndarray = field(repr=False)          # continuum-subtracted
    err: np.ndarray = field(repr=False)
    continuum: np.ndarray = field(repr=False)
    model: np.ndarray = field(repr=False)

    def as_dict(self) -> dict:
        return {"line": self.line.name, "rest_um": self.line.wave, "area": self.area, "area_err": self.area_err,
                "flux_W_m2": self.flux_W_m2, "flux_W_m2_err": self.flux_W_m2_err, "center_um": self.center_um,
                "v_kms": self.v_kms, "v_err_kms": self.v_err_kms, "fwhm_kms": self.fwhm_kms,
                "fwhm_err_kms": self.fwhm_err_kms, "snr": self.snr, "detected": self.detected}


def local_continuum_1d(wave, flux, err, rest_um, inner_kms, outer_kms, order=1, clip=3.0, exclude=()):
    """Polynomial continuum through the channels with inner <= |v| <= outer (other lines in `exclude`
    — (lo, hi) micron ranges — are left out), with two rounds of sigma clipping."""
    v = velocity_kms(wave, rest_um)
    use = (np.abs(v) >= inner_kms) & (np.abs(v) <= outer_kms) & np.isfinite(flux)
    for lo, hi in exclude:
        use &= ~((wave >= lo) & (wave <= hi))
    xs = v / max(outer_kms, 1.0)
    coef = np.zeros(order + 1)
    for _ in range(3):
        if use.sum() < order + 2:
            break
        coef = np.polyfit(xs[use], flux[use], order)
        r = flux - np.polyval(coef, xs)
        s = 1.4826 * np.nanmedian(np.abs(r[use] - np.nanmedian(r[use])))
        new = use & (np.abs(r) < clip * max(s, 1e-300))
        if new.sum() == use.sum():
            break
        use = new
    return np.polyval(coef, xs), use


def fit_line(spec_or_wave, flux=None, err=None, line: str | float | Line = "[Ne II] 12.81", rv_kms: float = 0.0,
             window_kms: float = 1500.0, inner_kms: float | None = None, cont_order: int = 1,
             center_shift_kms: float = 200.0, sigma_range=(0.8, 2.5), R_model: str = "argyriou2023",
             n_mc: int = 200, snr_detect: float = 3.0, seed: int | None = 0, flux_unit: str = "Jy") -> LineFit:
    """Fit one emission line: local polynomial continuum + Gaussian with sigma in
    [sigma_range] x the instrumental sigma and the centre within +-center_shift_kms of the rest
    wavelength (after removing `rv_kms`).  Errors from `n_mc` Monte Carlo realisations of the noise
    (0 = covariance errors).  Works on a JALEBI Spectrum (the pixels of the sub-band that contains
    the line) or on plain arrays."""
    ln = get_line(line)
    if flux is None:
        s = spec_or_wave
        w_all, f_all, e_all = s.wave, s.flux, s.err
        lam_obs = ln.wave * (1 + rv_kms / C_KMS)
        best = None
        for b in s.bands:
            i = s.band_slice(b)
            if len(i) and s.wave[i].min() < lam_obs < s.wave[i].max():
                margin = min(lam_obs - s.wave[i].min(), s.wave[i].max() - lam_obs)
                if best is None or margin > best[0]:
                    best = (margin, i)
        if best is None:
            raise ValueError(f"{ln.name} ({ln.wave} um) is outside the spectrum")
        i = best[1]
        wave, flux, err = w_all[i], f_all[i], e_all[i]
        if getattr(s, "rest_frame", False):
            rv_kms = 0.0
    else:
        wave = np.asarray(spec_or_wave, float); flux = np.asarray(flux, float)
        err = np.full_like(flux, np.nan) if err is None else np.asarray(err, float)
    wave_r = np.asarray(wave, float) / (1 + rv_kms / C_KMS)
    v = velocity_kms(wave_r, ln.wave)
    win = np.abs(v) <= window_kms
    wave_r, fl, er = wave_r[win], np.asarray(flux, float)[win], np.asarray(err, float)[win]
    sig_lsf = float(lsf_sigma_um(ln.wave, R_model))
    fwhm_lsf = float(lsf_fwhm_kms(ln.wave, R_model))
    inner = inner_kms if inner_kms is not None else max(250.0, 2.0 * fwhm_lsf)
    others = [(o.wave * (1 - inner / C_KMS), o.wave * (1 + inner / C_KMS)) for o in LINES.values()
              if o is not ln and abs(o.wave - ln.wave) / ln.wave * C_KMS < window_kms]
    cont, lf = local_continuum_1d(wave_r, fl, er, ln.wave, inner, window_kms, cont_order, exclude=others)
    d = fl - cont
    resid_sd = 1.4826 * np.nanmedian(np.abs(d[lf] - np.nanmedian(d[lf]))) if lf.sum() > 5 else np.nanstd(d)
    # per-pixel error: the larger of the given error and the empirical scatter of the line-free residuals
    # (region sums of cube ERR arrays ignore the spaxel-to-spaxel correlations of cube building)
    e = np.where(np.isfinite(er) & (er > 0), np.maximum(er, resid_sd), resid_sd)
    near = np.abs(velocity_kms(wave_r, ln.wave)) <= inner
    x, y, ee = wave_r[near], d[near], e[near]
    dmu = ln.wave * center_shift_kms / C_KMS
    lo = np.array([0.0, ln.wave - dmu, sigma_range[0] * sig_lsf]); hi = np.array([np.inf, ln.wave + dmu, sigma_range[1] * sig_lsf])
    p0 = gaussian_start(x, y[None], ln.wave, sig_lsf)
    p0[:, 0] = max(p0[0, 0], 1e-30)
    hi[0] = max(10 * np.nanmax(np.abs(y)), 1e-20)
    W = np.where(np.isfinite(y), 1.0 / ee, 0.0)[None]
    bf = fit_gaussian_batch(x, y[None], W, p0, lo, hi)
    A, mu, sg = bf.amp[0], bf.mu[0], bf.sigma[0]
    area = float(bf.area[0])
    if n_mc and n_mc > 0:
        rng = np.random.default_rng(seed)
        Ym = y[None] + rng.normal(size=(n_mc, len(y))) * ee[None]
        pm = np.repeat(np.array([[A, mu, sg, 0.0]]), n_mc, axis=0)
        mc = fit_gaussian_batch(x, Ym, np.repeat(W, n_mc, axis=0), pm, lo, hi)
        area_err = float(np.nanstd(mc.area)); mu_err = float(np.nanstd(mc.mu)); sg_err = float(np.nanstd(mc.sigma))
    else:
        area_err = float(bf.area_err[0]); mu_err = float(bf.mu_err[0]); sg_err = float(np.sqrt(max(bf.cov[0, 2, 2], 0)))
    snr = area / area_err if area_err > 0 else np.nan
    f_W = area_to_W_m2(area, mu) if flux_unit.lower() == "jy" else None
    f_We = area_to_W_m2(area_err, mu) if flux_unit.lower() == "jy" else None
    model = A * np.exp(-0.5 * ((wave_r - mu) / sg) ** 2)
    fwhm = sg / mu * C_KMS / FWHM_TO_SIGMA
    return LineFit(ln, area, area_err, None if f_W is None else float(f_W), None if f_We is None else float(f_We),
                   float(mu), float(velocity_kms(mu, ln.wave)), float(mu_err / ln.wave * C_KMS), float(fwhm),
                   float(sg_err / mu * C_KMS / FWHM_TO_SIGMA), float(snr), bool(bf.ok[0] and snr >= snr_detect),
                   wave_r, d, e, cont, model)
