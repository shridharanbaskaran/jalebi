"""Per-spaxel local continuum around a line.

A full-spectrum continuum per spaxel is too slow for whole cubes, and a line map only needs the
continuum under one line.  The default is therefore a low-order polynomial through the line-free
channels on both sides of the line (``inner_kms <= |v| <= outer_kms``, other catalogue lines left out),
solved for every spaxel at once with sigma clipping.  The spectra of the individual spaxels are
fitted independently, so the continuum follows the spaxel-to-spaxel "resampling wiggles" of the cube
building as far as the polynomial order allows (order 2-3 helps for bright point sources).

``method="aspls"`` / ``"irsqr"`` / ``"asls"`` run a pybaselines baseline through *all* channels of every
spaxel instead: ``aspls`` with ``lam=5e6`` is the continuum of the old ``cube_maps.py``
(``Baseline(wavelengths).aspls(spectrum, lam=5e6)``), ``irsqr`` with ``lam=1e3`` the one of the
``channel_maps`` notebook.  They are slower (one fit per spaxel; parallel with ``n_jobs``) and depend on
the width of the window they see (cube_maps used +-0.1 um: ``prepare_line(..., window_um=0.1)``).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from ..lines import C_KMS, LINES, Line


@dataclass
class CubeContinuumSettings:
    method: str = "poly"          # poly | median | aspls | irsqr | asls
    order: int = 1                # polynomial order (poly)
    inner_kms: float | None = None    # line-free channels start here (default: max(250, 2 x instrumental FWHM))
    outer_kms: float = 1200.0     # ... and end here
    clip_sigma: float = 3.0
    n_iter: int = 3
    lam: float | None = None      # aspls / irsqr / asls smoothness (default: aspls 5e6, as cube_maps.py; irsqr, asls 1e3)
    quantile: float = 0.05        # irsqr quantile
    p: float = 0.02               # asls asymmetry
    n_jobs: int = 1
    nan_policy: str = "omit"      # baselines: "omit" NaN channels (interpolate over them) or "propagate"
                                  # (a spaxel with any NaN channel gets no continuum, as in cube_maps.py)

    @property
    def lam_value(self) -> float:
        if self.lam is not None:
            return float(self.lam)
        return 5e6 if self.method == "aspls" else 1e3


BASELINE_METHODS = ("aspls", "irsqr", "asls")


def line_free_channels(wave: np.ndarray, line: Line, inner_kms: float, outer_kms: float, rv_kms: float = 0.0,
                       exclude_lines=True, extra_ranges=()) -> np.ndarray:
    """Boolean mask of the continuum channels around a line (observed-frame wavelengths)."""
    lam = line.wave * (1 + rv_kms / C_KMS)
    v = C_KMS * (wave - lam) / lam
    use = (np.abs(v) >= inner_kms) & (np.abs(v) <= outer_kms)
    if exclude_lines:
        for o in LINES.values():
            if o.name == line.name:
                continue
            lo = o.wave * (1 + rv_kms / C_KMS)
            if abs(lo - lam) / lam * C_KMS < outer_kms + inner_kms:
                use &= np.abs(C_KMS * (wave - lo) / lo) > inner_kms
    for a, b in extra_ranges:
        use &= ~((wave >= a) & (wave <= b))
    return use


def _poly_continuum(x: np.ndarray, D: np.ndarray, use: np.ndarray, order: int, clip: float, n_iter: int):
    """D (nz, npix) -> continuum (nz, npix): weighted least squares per column with per-column clipping."""
    V = np.vander(x, order + 1, increasing=True)                     # (nz, k)
    good = np.isfinite(D) & use[:, None]
    Dz = np.where(good, D, 0.0)
    coef = np.zeros((V.shape[1], D.shape[1]))
    for _ in range(max(n_iter, 1)):
        w = good.astype(float)
        A = np.einsum("zi,zj,zp->pij", V, V, w)                     # (npix, k, k)
        b = np.einsum("zi,zp->pi", V, Dz * w)                       # (npix, k)
        ok = w.sum(axis=0) >= order + 2
        A[~ok] = np.eye(V.shape[1])
        b[~ok] = 0.0
        coef = np.linalg.solve(A, b[..., None])[..., 0].T           # (k, npix)
        model = V @ coef
        r = np.where(good, D - model, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            s = 1.4826 * np.nanmedian(np.abs(r - np.nanmedian(r, axis=0)), axis=0)
        new = good & (np.abs(np.nan_to_num(r)) <= clip * np.where(np.isfinite(s) & (s > 0), s, np.inf))
        if new.sum() == good.sum():
            break
        good = new
        Dz = np.where(good, D, 0.0)
    model = V @ coef
    model[:, (good.sum(axis=0) < order + 2)] = np.nan
    return model, good


def _baseline_column(x, y, method, s: CubeContinuumSettings):
    from pybaselines import Baseline
    ok = np.isfinite(y)
    if ok.sum() < 5 or (s.nan_policy == "propagate" and not ok.all()):
        return np.full_like(y, np.nan)
    b = Baseline(x_data=x[ok], check_finite=False)
    if method == "aspls":
        base = b.aspls(y[ok], lam=s.lam_value)[0]
    elif method == "irsqr":
        base = b.irsqr(y[ok], lam=s.lam_value, quantile=s.quantile)[0]
    else:
        base = b.asls(y[ok], lam=s.lam_value, p=s.p)[0]
    out = np.full_like(y, np.nan)
    out[ok] = base
    return np.interp(x, x[ok], base) if not ok.all() else out


def spaxel_continuum(wave: np.ndarray, sci: np.ndarray, line: Line, settings: CubeContinuumSettings | None = None,
                     rv_kms: float = 0.0, fwhm_kms: float = 100.0, extra_ranges=()) -> tuple[np.ndarray, np.ndarray]:
    """Continuum cube (same shape as `sci`) and the boolean line-free channel mask used."""
    s = settings or CubeContinuumSettings()
    inner = s.inner_kms if s.inner_kms is not None else max(250.0, 2.0 * fwhm_kms)
    use = line_free_channels(wave, line, inner, s.outer_kms, rv_kms, extra_ranges=extra_ranges)
    if use.sum() < s.order + 2:
        # window too narrow for this line: fall back to everything outside the core
        use = line_free_channels(wave, line, inner, np.inf, rv_kms, extra_ranges=extra_ranges)
    nz, ny, nx = sci.shape
    D = sci.reshape(nz, -1).astype(float)
    lam = line.wave * (1 + rv_kms / C_KMS)
    x = (wave - lam) / lam * C_KMS / max(s.outer_kms, 1.0)
    if s.method == "poly":
        cont, _ = _poly_continuum(x, D, use, s.order, s.clip_sigma, s.n_iter)
    elif s.method == "median":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cont = np.repeat(np.nanmedian(np.where(use[:, None], D, np.nan), axis=0)[None], nz, axis=0)
    elif s.method in BASELINE_METHODS:
        cols = range(D.shape[1])
        if s.n_jobs and s.n_jobs != 1:
            from joblib import Parallel, delayed
            res = Parallel(n_jobs=s.n_jobs)(delayed(_baseline_column)(wave, D[:, p], s.method, s) for p in cols)
        else:
            res = [_baseline_column(wave, D[:, p], s.method, s) for p in cols]
        cont = np.stack(res, axis=1)
    else:
        raise ValueError(f"unknown cube continuum method {s.method!r}: poly, median, aspls, irsqr, asls")
    return cont.reshape(nz, ny, nx), use
