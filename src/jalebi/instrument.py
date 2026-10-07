"""Instrument model: MIRI-MRS resolving power and the sparse LSF + pixel-integration operator.

The observed model on pixel i is  F_i = sum_j K_ij I_j  where I is the model on the fine
ln(lambda) grid and K applies a Gaussian LSF of FWHM lambda/R(lambda) and integrates over
the pixel [x_i^-, x_i^+] (x = ln lambda).  K is built once per data set and stored as a CSR
matrix; a model evaluation is then a sparse matrix-vector product.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.special import erf

from .constants import FWHM_TO_SIGMA


# --- resolving power ----------------------------------------------------------

def resolving_power_argyriou(wave_um):
    """R(lambda) = 4603 - 128 lambda[micron]: the empirical in-flight MRS relation of Argyriou et al.
    (2023, A&A 675, A111), as used by JDISCS-style slab fitting."""
    w = np.asarray(wave_um, float)
    return 4603.0 - 128.0 * w


_JONES2023 = {  # (channel, band): (R at band start, R at band end) approx, Jones et al. 2023
    "1A": (3320, 3710), "1B": (3190, 3750), "1C": (3100, 3610),
    "2A": (2990, 3110), "2B": (2750, 3170), "2C": (2860, 3300),
    "3A": (2530, 2880), "3B": (1790, 2640), "3C": (1980, 2790),
    "4A": (1460, 1930), "4B": (1680, 1770), "4C": (1630, 1330),
}
_BANDS = {"1A": (4.90, 5.74), "1B": (5.66, 6.63), "1C": (6.53, 7.65), "2A": (7.51, 8.77), "2B": (8.67, 10.13),
          "2C": (10.02, 11.70), "3A": (11.55, 13.47), "3B": (13.34, 15.57), "3C": (15.41, 17.98),
          "4A": (17.70, 20.95), "4B": (20.69, 24.48), "4C": (24.19, 27.90)}


def resolving_power_jones(wave_um):
    """Piecewise-linear R(lambda) per MRS band from Jones et al. (2023)."""
    w = np.atleast_1d(np.asarray(wave_um, float))
    R = resolving_power_argyriou(w)
    for band, (lo, hi) in _BANDS.items():
        m = (w >= lo) & (w <= hi)
        if m.any():
            r0, r1 = _JONES2023[band]
            R[m] = r0 + (r1 - r0) * (w[m] - lo) / (hi - lo)
    return R


resolving_power_pontoppidan = resolving_power_argyriou     # legacy name
# "pontoppidan2024" is kept as an alias so configs written by earlier versions still load
RESOLVING_POWER = {"argyriou2023": resolving_power_argyriou, "pontoppidan2024": resolving_power_argyriou,
                   "jones2023": resolving_power_jones}


def resolving_power(wave_um, model: str = "argyriou2023", scale: float = 1.0, constant: float | None = None):
    if constant is not None:
        return np.full_like(np.asarray(wave_um, float), constant)
    return RESOLVING_POWER[model](wave_um) * scale


def band_of(wave_um) -> np.ndarray:
    """MRS sub-band label for each wavelength (first match wins in overlaps)."""
    w = np.atleast_1d(np.asarray(wave_um, float))
    out = np.full(w.shape, "", dtype=object)
    for band, (lo, hi) in _BANDS.items():
        m = (out == "") & (w >= lo) & (w <= hi)
        out[m] = band
    return out


# --- LSF + pixel operator -------------------------------------------------------

def pixel_edges(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lower/upper edges of pixels centred at x (midpoints; end pixels mirrored).

    The edges are taken within each run of increasing x: the pixel array of a MIRI spectrum is the concatenation
    of its sub-bands, and across a junction (e.g. the last 3B pixel at 15.568 um followed by the first 3C pixel at
    15.410) the midpoint rule gave inverted edges, so the first and last pixel of every sub-band got no model
    flux at all (until 0.20; 22 pixels of a 12-band fit)."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return x.copy(), x.copy()
    lo = np.empty(len(x)); hi = np.empty(len(x))
    cuts = np.concatenate([[0], np.flatnonzero(np.diff(x) < 0) + 1, [len(x)]])
    for a, b in zip(cuts[:-1], cuts[1:]):
        xs = x[a:b]
        if len(xs) == 1:                                   # a run of one pixel: use the neighbours' typical width
            w = np.median(np.abs(np.diff(x))) if len(x) > 1 else 0.0
            lo[a] = xs[0] - 0.5 * w; hi[a] = xs[0] + 0.5 * w
            continue
        mid = 0.5 * (xs[1:] + xs[:-1])
        lo[a:b] = np.concatenate([[xs[0] - (mid[0] - xs[0])], mid])
        hi[a:b] = np.concatenate([mid, [xs[-1] + (xs[-1] - mid[-1])]])
    return lo, hi


def build_lsf_operator(x_fine: np.ndarray, dx: float, wave_pix: np.ndarray, R_pix: np.ndarray,
                       truncate: float = 4.0, edges: tuple[np.ndarray, np.ndarray] | None = None) -> sparse.csr_matrix:
    """Sparse operator mapping the fine ln(lambda) grid to pixel-integrated, LSF-convolved fluxes.

    x_fine   : fine grid in ln(lambda) (uniform step dx, may consist of several segments)
    wave_pix : pixel centre wavelengths (micron), sorted within each sub-band
    R_pix    : resolving power at each pixel
    Each row sums to ~1 for pixels fully inside the fine grid (flux conservation).
    """
    x_pix = np.log(np.asarray(wave_pix, float))
    if len(x_pix) == 0:
        return sparse.csr_matrix((0, len(x_fine)))
    if edges is None:
        lo, hi = pixel_edges(x_pix)
    else:
        lo, hi = edges
    sig = FWHM_TO_SIGMA / R_pix                # LSF sigma in ln(lambda)
    n_fine = len(x_fine)
    rows, cols, vals = [], [], []
    # the fine points that reach the pixel: [lo - truncate sigma, hi + truncate sigma].  (Until 0.20 the window was
    # x_pix +- (truncate sigma + width / 2): the same for a pixel centred between its edges, but a pixel next to a
    # masked region, whose far edge is the midpoint to the next unmasked pixel, lost the part of its box beyond the
    # window -- row sums of 0.5-0.6 in the OH-prompt-masked 9-13 um region.)
    wlo = np.minimum(lo, hi) - truncate * sig
    whi = np.maximum(lo, hi) + truncate * sig
    # fine grid may have gaps between segments; use searchsorted on the sorted x_fine
    for i in range(len(x_pix)):
        if hi[i] <= lo[i]:                     # inverted edges (only when handed in explicitly): no model flux
            continue
        j0 = np.searchsorted(x_fine, wlo[i])
        j1 = np.searchsorted(x_fine, whi[i])
        if j1 <= j0:
            continue
        xj = x_fine[j0:j1]
        s2 = np.sqrt(2.0) * sig[i]
        w = 0.5 * (erf((hi[i] - xj) / s2) - erf((lo[i] - xj) / s2)) * dx / (hi[i] - lo[i])
        rows.append(np.full(j1 - j0, i)); cols.append(np.arange(j0, j1)); vals.append(w)
    if not rows:
        return sparse.csr_matrix((len(x_pix), n_fine))
    K = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(len(x_pix), n_fine))
    return K
