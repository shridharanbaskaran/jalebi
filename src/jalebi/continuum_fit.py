"""Joint continuum correction inside the fit (0.22, fit.continuum_fit: none | offset | spline).

The continuum is still estimated and subtracted before the fit (`pipeline.prepare`); this module adds a
small correction c(lambda) = sum_j beta_j B_j(lambda) to it *inside* the likelihood:

    y - c(lambda) = sum_u a_u f_u(theta) + screens,      i.e. the model gains the linear columns B_j.

offset : one constant per MRS sub-band (B_j = 1 on the band's fitted pixels)
spline : a cubic B-spline per sub-band, knots every `knot_spacing_um` over the band's fitted range (clamped
         ends), so broad continuum errors are absorbed while line-sized structure (R ~ 3000, ~0.005 um) is not

The coefficients are linear parameters with a zero-mean Gaussian prior of width tau_j: `prior_width` x the
median continuum of the band (prior: continuum, default 2 %) or x the median noise (prior: noise).  They are
handled with the same machinery as the emitting areas (jalebi.linear): marginalised analytically (the default,
fit.continuum_correction.mode: marginalise) or profiled (penalised least squares, mode: profile).  The
optimiser (DE) always profiles them, alternating with the NNLS areas.  In fit.mcmc.linear: sample runs the
coefficients are profiled / marginalised at every likelihood call as well (the areas stay sampled).

What it is for: the hot-water corner and the pseudo-continuum of blended line forests both trade flux with the
continuum; a continuum that is slightly too low makes a broad, weak-line component (hot, dense, tiny water)
look necessary.  Letting the continuum move by a few per cent within a prior takes that lever away.  It is
EXPERIMENTAL: the prior width and knot spacing decide how much line flux can leak into the correction; use
runs/continuum_study.py to compare settings against the published values before switching it on in a survey.
"""
from __future__ import annotations

import numpy as np

MODES = ("none", "offset", "spline")


def bspline_basis(x: np.ndarray, knot_spacing: float, degree: int = 3) -> np.ndarray:
    """Clamped cubic B-spline design matrix on `x` with interior knots every `knot_spacing` (at least one
    polynomial piece); columns are the basis functions."""
    from scipy.interpolate import BSpline
    x = np.asarray(x, float)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo <= 0:
        return np.ones((len(x), 1))
    n_int = max(int(np.floor((hi - lo) / knot_spacing)), 0)
    inner = np.linspace(lo, hi, n_int + 2)[1:-1]
    t = np.concatenate([np.full(degree + 1, lo), inner, np.full(degree + 1, hi)])
    B = BSpline.design_matrix(x, t, degree).toarray()
    return B


def continuum_basis(wave: np.ndarray, band: np.ndarray, mode: str, knot_spacing_um: float = 1.0) -> tuple[np.ndarray, list[str]]:
    """(B, labels): the correction basis on the fitted pixels, one block of columns per sub-band."""
    if mode not in MODES:
        raise ValueError(f"unknown fit.continuum_fit {mode!r}: none | offset | spline")
    wave = np.asarray(wave, float); band = np.asarray(band)
    if mode == "none" or len(wave) == 0:
        return np.zeros((len(wave), 0)), []
    cols, labels = [], []
    bands = list(dict.fromkeys(band.tolist()))
    for b in bands:
        sel = band == b
        if sel.sum() < 2:
            continue
        if mode == "offset":
            c = np.zeros(len(wave)); c[sel] = 1.0
            cols.append(c); labels.append(f"{b}:offset")
        else:
            Bb = bspline_basis(wave[sel], knot_spacing_um)
            for j in range(Bb.shape[1]):
                c = np.zeros(len(wave)); c[sel] = Bb[:, j]
                cols.append(c); labels.append(f"{b}:s{j}")
    if not cols:
        return np.zeros((len(wave), 0)), []
    return np.column_stack(cols), labels


def prior_widths(B: np.ndarray, continuum: np.ndarray | None, sigma: np.ndarray, prior: str = "continuum",
                 prior_width: float = 0.02) -> np.ndarray:
    """tau_j per column: prior_width x the median continuum (or noise) over the column's support."""
    if prior not in ("continuum", "noise"):
        raise ValueError(f"unknown fit.continuum_correction.prior {prior!r}: continuum | noise")
    tau = np.empty(B.shape[1])
    for j in range(B.shape[1]):
        sup = B[:, j] > 0.1 * np.max(B[:, j]) if np.max(B[:, j]) > 0 else np.ones(len(B), bool)
        if prior == "continuum" and continuum is not None and np.any(np.isfinite(continuum[sup])):
            ref = float(np.nanmedian(np.abs(continuum[sup])))
        else:
            ref = float(np.nanmedian(sigma[sup]))
        tau[j] = max(prior_width * ref, 1e-30)
    return tau


class ContinuumCorrection:
    """The basis, the prior and the solvers used by FitProblem / LinearProblem."""

    def __init__(self, wave, band, continuum, sigma, mode: str, knot_spacing_um: float = 1.0, prior: str = "continuum",
                 prior_width: float = 0.02, solve: str = "marginalise"):
        self.mode = mode
        self.B, self.labels = continuum_basis(wave, band, mode, knot_spacing_um)
        self.n = self.B.shape[1]
        self.tau = prior_widths(self.B, continuum, sigma, prior, prior_width) if self.n else np.zeros(0)
        self.solve_mode = "profile" if solve == "profile" else "marginalise"
        self.knot_spacing_um = knot_spacing_um
        self.prior = prior
        self.prior_width = prior_width

    def describe(self) -> str:
        return (f"{self.mode}: {self.n} coefficient(s), prior sigma {self.prior_width:g} x {self.prior} "
                f"({self.solve_mode} in the sampler" + (f", knots every {self.knot_spacing_um:g} um" if self.mode == "spline" else "") + ")")

    def solve(self, resid: np.ndarray, sw: np.ndarray, s2: float = 1.0) -> tuple[np.ndarray, float]:
        """MAP coefficients for a residual r = y - gas model (whitening sw = sqrt(w)/sigma, noise scale s^2) and
        the ln-likelihood change: -1/2 [chi2 drop] - 1/2 beta^T Lambda^-1 beta (profile) plus the Occam term
        -1/2 ln det G - 1/2 ln det Lambda (marginalise)."""
        if self.n == 0:
            return np.zeros(0), 0.0
        Bw = self.B * sw[:, None]
        rw = resid * sw
        lam2 = self.tau ** 2
        G = (Bw.T @ Bw) / s2 + np.diag(1.0 / lam2)
        b = (Bw.T @ rw) / s2
        try:
            L = np.linalg.cholesky(G)
        except np.linalg.LinAlgError:
            return np.zeros(self.n), 0.0
        beta = np.linalg.solve(L.T, np.linalg.solve(L, b))
        r2 = rw - Bw @ beta
        dlnL = 0.5 * (float(rw @ rw) - float(r2 @ r2)) / s2 - 0.5 * float(np.sum(beta ** 2 / lam2))
        if self.solve_mode == "marginalise":
            dlnL += -float(np.sum(np.log(np.diag(L)))) - 0.5 * float(np.sum(np.log(lam2)))
        return beta, dlnL

    def correction(self, beta: np.ndarray) -> np.ndarray:
        return self.B @ np.asarray(beta, float) if self.n else np.zeros(len(self.B))
