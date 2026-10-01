"""Continuum estimation, protected ranges and default masks.

Methods (all return a continuum array on the spectrum's pixels; run per sub-band):
  irsqr        Iteratively reweighted spline quantile regression (pybaselines) — MINDS default
  median_sg    Iterative running median + Savitzky–Golay smoothing — JDISCS style
  asls         Asymmetric least squares (pybaselines) — the legacy setting from the notebooks
  aspls        Adaptive smoothness-penalised least squares (pybaselines; Zhang et al. 2020), the
               continuum of the group's cube_maps.py (lam ~ 5e6); good under dense line forests (5-8 um)
  convex_hull  Lower convex hull of the spectrum in overlapping segments, smoothed
  rolling_min  Rolling minimum (percentile) envelope, smoothed
  spline       Cubic spline through user-supplied anchor wavelengths (manual mode)
  banzatti     Fixed line-free windows from Banzatti et al. (2025) interpolated by a spline

"Protected" ranges (Q-branches by default) are removed before estimating the baseline and
bridged by cubic interpolation, so the pseudo-continuum under C2H2/HCN/CO2 Q-branches is
not eaten.  A model-aware refinement re-estimates the continuum on (data - gas model).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import median_filter, minimum_filter, uniform_filter1d
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull

from .data import Spectrum

DATA_FILES = os.path.join(os.path.dirname(__file__), "data_files")

# Q-branches and other regions where the lower envelope is not the continuum (micron)
DEFAULT_PROTECTED = {
    "C2H2 Q": (13.66, 13.75),
    "HCN Q": (13.98, 14.06),
    "CO2 Q": (14.93, 15.02),
    "13CO2 Q": (15.40, 15.44),
    "C6H6": (14.80, 14.90),
    "CH4 Q": (7.62, 7.70),
    "C4H2": (15.90, 15.94),
}

# Lines no slab model contains: H I, H2, fine-structure; and non-LTE OH prompt emission 9-13 um
DEFAULT_MASKS = {
    "H I 5-4 (Pfβ)": (4.048, 4.057),
    "H I 8-5 (Pfβ)": (4.650, 4.658),
    "H I 6-5 (Pfβ)": (7.455, 7.465),
    "H I 7-6 (Huα)": (12.366, 12.378),
    "H I 9-6": (7.503, 7.512),
    "[Ar II] 6.99": (6.983, 6.990),
    "H2 S(5) 6.91": (6.906, 6.913),
    "H2 S(4) 8.03": (8.023, 8.030),
    "H2 S(3) 9.66": (9.660, 9.670),
    "H2 S(2) 12.28": (12.273, 12.284),
    "[Ne II] 12.81": (12.807, 12.820),
    "[Ne III] 15.55": (15.548, 15.562),
    "H2 S(1) 17.03": (17.028, 17.042),
    "[Fe II] 17.94": (17.928, 17.944),
    "[S III] 18.71": (18.705, 18.722),
    "[Fe II] 25.99": (25.975, 26.005),
    "[S I] 25.25": (25.240, 25.260),
    "H2 S(0) 28.22": (28.21, 28.23),
}
OH_PROMPT_RANGE = (9.0, 13.0)
# Approximate centres of the OH pure-rotational quadruplets (N' ~ 20-30) that are non-LTE
# prompt emission from water photodissociation (Tabone et al. 2021, 2024).  Used only when the
# OH line list is not cached; each is masked +/- OH_PROMPT_HALFWIDTH micron.
OH_PROMPT_LINES = [9.09, 9.13, 9.21, 9.30, 9.38, 9.47, 9.57, 9.67, 9.79, 9.90, 10.03, 10.23, 10.37, 10.60,
                   10.80, 11.05, 11.30, 11.55, 11.85, 12.15, 12.50, 12.85]
OH_PROMPT_HALFWIDTH = 0.02


@dataclass
class ContinuumSettings:
    method: str = "irsqr"
    # generic
    protect: bool = True
    protected: dict = field(default_factory=lambda: dict(DEFAULT_PROTECTED))
    smooth: int = 0                 # extra Savitzky–Golay smoothing window (pixels, 0 = none)
    # irsqr
    quantile: float = 0.1
    knot_spacing: int = 25          # pixels between spline knots (MINDS: 25–75 per sub-band)
    # median_sg
    median_window: int = 101        # pixels (JDISCS: 45–200 channels)
    median_percentile: float = 25.0 # running percentile (50 = plain median)
    sg_window: int = 51
    sg_order: int = 2
    n_iter: int = 5
    # asls
    lam: float = 1e3
    p: float = 0.01
    # aspls (adaptive smoothness penalty; no asymmetry parameter -- the weights adapt)
    aspls_lam: float = 5e6
    aspls_alpha: float = 0.5       # asymmetric_coef of pybaselines (0.5 = default)
    # convex hull / rolling minimum
    segment: int = 300              # pixels per hull segment
    overlap: int = 100
    percentile: float = 5.0
    min_window: int = 61
    # spline (manual anchors)
    anchors: list = field(default_factory=list)   # wavelengths (micron)
    anchor_width: int = 3           # pixels averaged around each anchor
    # banzatti
    banzatti_file: str = os.path.join(DATA_FILES, "cont_ranges_Banzatti+2025.csv")

    def to_dict(self):
        d = dict(self.__dict__)
        d["protected"] = {k: list(v) for k, v in self.protected.items()}
        return d


METHODS = ["irsqr", "median_sg", "asls", "aspls", "convex_hull", "rolling_min", "spline", "banzatti", "none", "given"]


# ------------------------------------------------------------------------------------
# Individual estimators on one (wave, flux) array with a boolean `use` mask
# ------------------------------------------------------------------------------------

def _pchip_clamped(x, y):
    """PCHIP interpolant that holds the end values constant outside [x[0], x[-1]]
    (PCHIP extrapolation can run away at band edges)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    o = np.argsort(x); x, y = x[o], y[o]
    if len(x) < 2:
        return lambda xn: np.full(np.shape(xn), y[0] if len(y) else np.nan)
    p = PchipInterpolator(x, y, extrapolate=False)

    def f(xn):
        xn = np.asarray(xn, float)
        out = p(np.clip(xn, x[0], x[-1]))
        return out
    return f


def _fill(wave, flux, use):
    """Interpolate over unused pixels (protected ranges, masked pixels) with a clamped PCHIP spline."""
    if use.all():
        return flux.copy()
    if use.sum() < 4:
        return np.full_like(flux, np.nanmedian(flux[use]) if use.any() else np.nan)
    f = flux.copy()
    f[~use] = _pchip_clamped(wave[use], flux[use])(wave[~use])
    return f


def cont_irsqr(wave, flux, use, s: ContinuumSettings):
    from pybaselines import Baseline
    f = _fill(wave, flux, use)
    n = len(f)
    nk = max(int(n / max(s.knot_spacing, 2)), 3)
    bl = Baseline(x_data=np.arange(n))
    base, _ = bl.irsqr(f, quantile=s.quantile, num_knots=nk, spline_degree=3, diff_order=3, max_iter=100)
    return base


def cont_median_sg(wave, flux, use, s: ContinuumSettings):
    """Iterative running-median (percentile) filtering with emission clipping, then Savitzky–Golay
    smoothing (JDISCS style; Banzatti et al. 2025).  `median_percentile` < 50 makes the running
    estimate follow the lower envelope in dense line forests."""
    from scipy.ndimage import percentile_filter
    f = _fill(wave, flux, use)
    base = f.copy()
    mw = s.median_window if s.median_window % 2 else s.median_window + 1
    mw = max(min(mw, len(f) - 1 if (len(f) - 1) % 2 else len(f) - 2), 5)
    sw = s.sg_window if s.sg_window % 2 else s.sg_window + 1
    sw = max(sw, s.sg_order + 2)
    for _ in range(max(s.n_iter, 1)):
        env = percentile_filter(base, s.median_percentile, size=mw, mode="nearest")
        resid = base - env
        neg = resid[resid < 0]
        mad = 1.4826 * np.median(np.abs(neg)) if len(neg) > 10 else 1.4826 * np.median(np.abs(resid - np.median(resid)))
        base = np.where(resid > 2.0 * mad, env, base)
    base = median_filter(base, size=mw, mode="nearest")
    if len(base) > sw:
        base = savgol_filter(base, sw, s.sg_order, mode="nearest")
    return base


def cont_asls(wave, flux, use, s: ContinuumSettings):
    from pybaselines import Baseline
    f = _fill(wave, flux, use)
    bl = Baseline(x_data=np.arange(len(f)))
    base, _ = bl.asls(f, lam=s.lam, p=s.p)
    return base


def cont_aspls(wave, flux, use, s: ContinuumSettings):
    """Adaptive smoothness-penalised least squares (pybaselines `aspls`), as in the cube module and the
    group's cube_maps.py.  `aspls_lam` sets the stiffness (5e6 for MRS sub-bands), `aspls_alpha` the
    asymmetry coefficient."""
    from pybaselines import Baseline
    f = _fill(wave, flux, use)
    bl = Baseline(x_data=np.arange(len(f)))
    base, _ = bl.aspls(f, lam=s.aspls_lam, asymmetric_coef=s.aspls_alpha)
    return base


def cont_convex_hull(wave, flux, use, s: ContinuumSettings):
    """Lower convex hull of (wave, flux) computed in overlapping segments, then averaged in the
    overlaps and lightly smoothed.  Segmenting lets the hull follow a curved continuum."""
    f = _fill(wave, flux, use)
    n = len(f)
    seg = max(int(s.segment), 20)
    ov = min(max(int(s.overlap), 0), seg // 2)
    acc = np.zeros(n); cnt = np.zeros(n)
    x = (wave - wave[0]) / (wave[-1] - wave[0] + 1e-30)
    y = f / (np.nanmax(np.abs(f)) + 1e-30)
    start = 0
    while start < n:
        stop = min(start + seg, n)
        i = np.arange(start, stop)
        if len(i) >= 4:
            pts = np.column_stack([x[i], y[i]])
            try:
                hull = ConvexHull(pts)
                v = hull.vertices
                # lower hull: vertices from the leftmost to the rightmost going through the minimum
                order = np.argsort(pts[v, 0])
                vs = v[order]
                lower = [vs[0]]
                for k in vs[1:]:
                    while len(lower) >= 2:
                        (x1, y1), (x2, y2), (x3, y3) = pts[lower[-2]], pts[lower[-1]], pts[k]
                        if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) <= 0:   # not convex from below -> pop
                            lower.pop()
                        else:
                            break
                    lower.append(k)
                lower = np.array(lower)
                yl = np.interp(x[i], pts[lower, 0], pts[lower, 1])
            except Exception:
                yl = np.full(len(i), y[i].min())
            acc[i] += yl; cnt[i] += 1
        if stop == n:
            break
        start = stop - ov
    base = acc / np.maximum(cnt, 1) * (np.nanmax(np.abs(f)) + 1e-30)
    if s.smooth == 0 and n > 31:
        base = savgol_filter(base, 31, 2, mode="nearest")
    return base


def cont_rolling_min(wave, flux, use, s: ContinuumSettings):
    f = _fill(wave, flux, use)
    w = s.min_window if s.min_window % 2 else s.min_window + 1
    if s.percentile <= 0:
        env = minimum_filter(f, size=w, mode="nearest")
    else:
        from scipy.ndimage import percentile_filter
        env = percentile_filter(f, s.percentile, size=w, mode="nearest")
    base = uniform_filter1d(env, size=w, mode="nearest")
    return base


def cont_spline(wave, flux, use, s: ContinuumSettings):
    anchors = np.asarray([a for a in s.anchors if wave.min() <= a <= wave.max()], float)
    if len(anchors) < 2:
        return np.full_like(flux, np.nan)
    ya = []
    hw = max(int(s.anchor_width) // 2, 0)
    for a in anchors:
        j = int(np.argmin(np.abs(wave - a)))
        sl = slice(max(j - hw, 0), j + hw + 1)
        ya.append(np.nanmedian(flux[sl]))
    o = np.argsort(anchors)
    return _pchip_clamped(anchors[o], np.asarray(ya)[o])(wave)


def cont_banzatti(wave, flux, use, s: ContinuumSettings):
    """Spline through the medians of the line-free windows of Banzatti et al. (2025)."""
    tab = pd.read_csv(s.banzatti_file)
    xs, ys = [], []
    for _, r in tab.iterrows():
        m = (wave >= r["xmin"]) & (wave <= r["xmax"]) & use
        if m.sum() >= 2:
            xs.append(np.mean(wave[m])); ys.append(np.median(flux[m]))
    if len(xs) < 2:
        return np.full_like(flux, np.nan)
    o = np.argsort(xs)
    return _pchip_clamped(np.asarray(xs)[o], np.asarray(ys)[o])(wave)


ESTIMATORS = {"irsqr": cont_irsqr, "median_sg": cont_median_sg, "asls": cont_asls, "aspls": cont_aspls, "convex_hull": cont_convex_hull,
              "rolling_min": cont_rolling_min, "spline": cont_spline, "banzatti": cont_banzatti}


# ------------------------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------------------------

def in_ranges(wave: np.ndarray, ranges) -> np.ndarray:
    m = np.zeros(len(wave), bool)
    for r in (ranges.values() if isinstance(ranges, dict) else ranges):
        m |= (wave >= r[0]) & (wave <= r[1])
    return m


def cont_central(wave, flux, use, s: ContinuumSettings):
    """Running median + Savitzky–Golay through the *centre* of the data: used for the model-aware
    refinement, where the gas model has been subtracted and the residual scatters symmetrically
    about the continuum (a lower-envelope method would bias it low)."""
    f = _fill(wave, flux, use)
    mw = s.median_window if s.median_window % 2 else s.median_window + 1
    mw = max(min(mw, len(f) - 1 if (len(f) - 1) % 2 else len(f) - 2), 5)
    base = median_filter(f, size=mw, mode="nearest")
    # one clipping pass against remaining (unmodelled) emission
    resid = f - base
    mad = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    base = median_filter(np.where(resid > 3 * mad, base, f), size=mw, mode="nearest")
    sw = s.sg_window if s.sg_window % 2 else s.sg_window + 1
    sw = max(sw, s.sg_order + 2)
    if len(base) > sw:
        base = savgol_filter(base, sw, s.sg_order, mode="nearest")
    return base


def estimate_continuum(spec: Spectrum, settings: ContinuumSettings, gas_model: np.ndarray | None = None) -> np.ndarray:
    """Continuum on every pixel of `spec`, estimated per sub-band.

    gas_model : optional model line flux (Jy) subtracted before the estimate (model-aware
                refinement); the estimate then uses the central (median) estimator on the
                residual, whatever `settings.method` is, and falls back to the original
                continuum where the gas model is not defined (NaN or outside the fit windows).
    """
    if settings.method == "none":
        return np.zeros(len(spec.wave))
    if settings.method == "given":
        if spec.continuum is None or not np.any(np.isfinite(spec.continuum) & (spec.continuum != 0)):
            raise ValueError("continuum method 'given' needs a continuum loaded with the spectrum "
                             "(a 'continuum' or 'baseline' column in the CSV)")
        return np.asarray(spec.continuum, float).copy()
    est = ESTIMATORS[settings.method] if gas_model is None else cont_central
    cont = np.full(len(spec.wave), np.nan)
    flux_all = spec.flux - (gas_model if gas_model is not None else 0.0)
    for b in spec.bands:
        i = spec.band_slice(b)
        w, f = spec.wave[i], flux_all[i]
        use = np.isfinite(f) & spec.mask[i]
        if gas_model is not None:
            # only refine where the gas model exists; elsewhere keep the previous continuum
            has_model = np.isfinite(gas_model[i]) & (gas_model[i] != 0)
            if has_model.sum() < 10:
                cont[i] = spec.continuum[i] if spec.continuum is not None else np.nan
                continue
            use &= has_model
        if settings.protect and settings.protected:
            use &= ~in_ranges(w, settings.protected)
        if use.sum() < 10:
            continue
        # work on a finite copy
        ff = f.copy(); ff[~np.isfinite(ff)] = np.nanmedian(f[use])
        try:
            c = est(w, ff, use, settings)
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"continuum method {settings.method} failed on band {b}: {e}")
        if settings.smooth and settings.smooth > 3 and len(c) > settings.smooth:
            sw = settings.smooth if settings.smooth % 2 else settings.smooth + 1
            c = savgol_filter(c, sw, 2, mode="nearest")
        if gas_model is not None and spec.continuum is not None:
            has_model = np.isfinite(gas_model[i]) & (gas_model[i] != 0)
            c = np.where(has_model, c, spec.continuum[i])
        cont[i] = c
    return cont


def oh_prompt_ranges(eup_min: float = 12000.0, halfwidth: float = 0.012) -> list[tuple[float, float]]:
    """Wavelength ranges of high-excitation OH lines at 9-13 um (prompt emission), from the cached
    OH line list when available, else from the approximate quadruplet table."""
    try:
        from .linedata import load_linelist
        oh = load_linelist("OH", fetch=False).select(*OH_PROMPT_RANGE, aul_min=1e-2)
        w = oh.wave[oh.eu >= eup_min]
        if len(w):
            return [(x - halfwidth, x + halfwidth) for x in w]
    except Exception:
        pass
    return [(x - OH_PROMPT_HALFWIDTH, x + OH_PROMPT_HALFWIDTH) for x in OH_PROMPT_LINES]


def default_mask(spec: Spectrum, mask_oh_prompt: bool = True, extra: dict | None = None,
                 oh_whole_range: bool = False) -> np.ndarray:
    """Boolean pixel mask (True = fit) removing atomic/H2 lines and, optionally, the OH prompt lines
    (or the whole 9-13 um range with `oh_whole_range`)."""
    m = np.ones(len(spec.wave), bool)
    ranges = dict(DEFAULT_MASKS)
    if extra:
        ranges.update(extra)
    m &= ~in_ranges(spec.wave, ranges)
    if mask_oh_prompt:
        m &= ~in_ranges(spec.wave, [OH_PROMPT_RANGE] if oh_whole_range else oh_prompt_ranges())
    return m
