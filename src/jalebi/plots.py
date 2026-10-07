"""Matplotlib figures shared by the CLI report and the web app."""
from __future__ import annotations

import sys

import numpy as np
import matplotlib
# Headless by default (CLI, batch jobs, the web app server), but leave the backend alone inside
# Jupyter/IPython so figures show inline in notebooks.
if "ipykernel" not in sys.modules and "IPython" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .molecules import get_molecule


def _colour(molecule: str) -> str:
    try:
        return get_molecule(molecule).colour
    except KeyError:
        return "#444444"


def _band_segments(problem, sel):
    """Index arrays of pixels in `sel`, split per sub-band so lines are not drawn across overlaps."""
    band = getattr(problem, "band", None)
    idx = np.flatnonzero(sel)
    if band is None:
        return [idx]
    return [idx[band[idx] == b] for b in dict.fromkeys(band[idx])]


def plot_fit(problem, theta, windows=None, per_component=True, title="", figsize=(14, 7)):
    """Data, total model and per-component models with residuals, one panel per window."""
    P, _ = problem.params_from_theta(theta)
    total, units, tmax = problem.model.evaluate(P, per_unit=True)
    windows = windows or problem.windows
    n = len(windows)
    fig, axes = plt.subplots(2 * n, 1, figsize=(figsize[0], figsize[1] * n / 2 + 1), sharex=False,
                             gridspec_kw={"height_ratios": [3, 1] * n})
    axes = np.atleast_1d(axes)
    for k, (lo, hi) in enumerate(windows):
        ax, axr = axes[2 * k], axes[2 * k + 1]
        sel = (problem.wave >= lo) & (problem.wave <= hi)
        for j, i in enumerate(_band_segments(problem, sel)):
            w = problem.wave[i]
            lab = (lambda s: s if j == 0 else None)
            ax.step(w, problem.y[i], where="mid", color="k", lw=0.7, label=lab("data − continuum"))
            ax.fill_between(w, -problem.sigma[i], problem.sigma[i], color="0.85", step="mid", label=lab("±1σ noise"))
            if per_component:
                for key, f in units.items():
                    lead = next((c for c in problem.components if c.name == key or c.group == key), None)
                    mol = lead.molecule if lead is not None else key
                    ls = "--" if lead is not None and lead.kind == "absorption" else "-"
                    ax.plot(w, f[i], lw=0.8, alpha=0.9, color=_colour(mol), ls=ls, label=lab(f"{key} (τmax={tmax.get(key, np.nan):.1f})"))
            ax.plot(w, total[i], color="crimson", lw=1.0, label=lab("total model"))
            r = (problem.y[i] - total[i]) / problem.sigma[i]
            axr.step(w, r, where="mid", color="k", lw=0.6)
        ax.set_xlim(lo, hi)
        ax.set_ylabel("F_ν [Jy]")
        if k == 0:
            ax.legend(fontsize=8, ncol=4, loc="upper right")
            ax.set_title(title)
        r_all = (problem.y[sel] - total[sel]) / problem.sigma[sel]
        axr.axhline(0, color="crimson", lw=0.8)
        axr.set_ylim(-6, 6); axr.set_ylabel("(d−m)/σ"); axr.set_xlim(lo, hi)
        axr.text(0.01, 0.8, f"χ²_red = {np.mean(r_all**2):.2f}", transform=axr.transAxes, fontsize=8)
    axes[-1].set_xlabel("wavelength [µm]")
    fig.tight_layout()
    return fig


def plot_grid(grid_result, figsize=(6, 4.5)):
    """Δχ² map over (log N, T) with the best point and the 1/2/3σ contours."""
    g = grid_result
    fig, ax = plt.subplots(figsize=figsize)
    d = g.delta_chi2()
    im = ax.pcolormesh(g.logN, g.T, np.log10(d + 1), shading="nearest", cmap="viridis_r")
    try:
        ax.contour(g.logN, g.T, d, levels=[2.3, 6.17, 11.8], colors=["w", "w", "w"], linewidths=[1.2, 0.9, 0.6])
    except Exception:
        pass
    b = g.best
    ax.plot(b["logN"], b["T"], "r+", ms=12, mew=2)
    ax.contour(g.logN, g.T, g.tau_max, levels=[1.0], colors="orange", linestyles="--", linewidths=1)
    ax.set_xlabel("log N [cm⁻²]"); ax.set_ylabel("T [K]")
    ax.set_title(f"{g.comp}: log N={b['logN']:.2f}, T={b['T']:.0f} K, log R={b['logR']:.2f}" + (" (edge!)" if b["at_edge"] else ""))
    fig.colorbar(im, ax=ax, label="log10(Δχ² + 1)")
    fig.tight_layout()
    return fig


def plot_corner(result, burn=None, comps: list[str] | None = None, **kw):
    """Corner plot of the free parameters (optionally restricted to some components)."""
    import corner
    flat = result.flat(burn)
    idx = [i for i, p in enumerate(result.problem.free) if comps is None or p.comp in comps]
    labels = [result.labels[i] for i in idx]
    fig = corner.corner(flat[:, idx], labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True,
                        title_kwargs={"fontsize": 9}, label_kwargs={"fontsize": 9}, **kw)
    return fig


def plot_correlation(result, burn=None, figsize=(7, 6)):
    """Cross-parameter (and cross-molecule) posterior correlation matrix."""
    C = result.correlation(burn)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(C.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(C))); ax.set_yticks(range(len(C)))
    ax.set_xticklabels(C.columns, rotation=90, fontsize=8); ax.set_yticklabels(C.index, fontsize=8)
    for i in range(len(C)):
        for j in range(len(C)):
            if abs(C.values[i, j]) > 0.5 and i != j:
                ax.text(j, i, f"{C.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="posterior correlation")
    ax.set_title("Parameter correlations (degeneracies)")
    fig.tight_layout()
    return fig


def plot_traces(result, figsize=(10, 8)):
    fig, axes = plt.subplots(result.chain.shape[2], 1, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes)
    for i, ax in enumerate(axes):
        ax.plot(result.chain[:, :, i], color="k", alpha=0.15, lw=0.5)
        ax.set_ylabel(result.labels[i], fontsize=7)
        ax.axvline(result.burn(), color="r", lw=0.8)
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    return fig


def plot_posterior_predictive(problem, result, n=100, windows=None, figsize=(14, 4)):
    med, lo, hi = result.posterior_predictive(n)
    windows = windows or problem.windows
    fig, axes = plt.subplots(len(windows), 1, figsize=(figsize[0], figsize[1] * len(windows)))
    axes = np.atleast_1d(axes)
    for ax, (a, b) in zip(axes, windows):
        sel = (problem.wave >= a) & (problem.wave <= b)
        for j, i in enumerate(_band_segments(problem, sel)):
            w = problem.wave[i]
            ax.step(w, problem.y[i], where="mid", color="k", lw=0.7)
            ax.fill_between(w, lo[i], hi[i], color="crimson", alpha=0.3, step="mid", label="16–84 % posterior" if j == 0 else None)
            ax.plot(w, med[i], color="crimson", lw=0.8, label="median" if j == 0 else None)
        ax.set_xlim(a, b); ax.set_ylabel("F_ν [Jy]")
    axes[0].legend(fontsize=8); axes[-1].set_xlabel("wavelength [µm]")
    fig.tight_layout()
    return fig


def full_spectrum_model(spec, problem, theta, oversample: int = 3):
    """Evaluate the fitted model on every pixel of the spectrum (not only the fit windows).
    Builds a coarser display model over the whole range; returns (total, per_unit) in Jy."""
    from .model import build_model
    P, _ = problem.params_from_theta(theta)
    wmin, wmax = float(np.nanmin(spec.wave)), float(np.nanmax(spec.wave))
    m = build_model(problem.components, spec.wave, spec.distance_pc, [(wmin, wmax)],
                    linelists=None, releases=None, oversample=oversample,
                    R_model=problem.model.R_model, R_scale=problem.model.R_scale, continuum=spec.continuum)
    total, units, _ = m.evaluate(P, per_unit=True)
    return total, units


def plot_full_spectrum(spec, problem, theta, title="", figsize=(18, 10), oversample: int = 3, model=None):
    """Whole-spectrum figure: data + continuum, continuum-subtracted data with the model,
    residuals.  Fit windows are shaded; masked pixels are left out of the residual."""
    total, units = full_spectrum_model(spec, problem, theta, oversample) if model is None else model
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [2, 3, 1]})
    noise = spec.err.copy()
    try:
        from .data import estimate_noise
        n = estimate_noise(spec)
        noise = np.where(np.isfinite(n) & (n > 0), n, noise)
    except Exception:
        pass
    bands = spec.bands
    for j, b in enumerate(bands):
        i = spec.band_slice(b)
        w = spec.wave[i]
        ax0.plot(w, spec.flux[i], color="k", lw=0.5, label="data" if j == 0 else None)
        ax0.plot(w, spec.continuum[i], color="tab:orange", lw=1.0, label="continuum" if j == 0 else None)
        y = spec.line_flux[i]
        ax1.step(w, np.where(spec.mask[i], y, np.nan), where="mid", color="k", lw=0.6, label="data − continuum" if j == 0 else None)
        for key, f in units.items():
            mol = next((c.molecule for c in problem.components if c.name == key or c.group == key), key)
            ax1.plot(w, f[i], lw=0.7, alpha=0.9, color=_colour(mol), label=key if j == 0 else None)
        ax1.plot(w, total[i], color="crimson", lw=0.8, label="total model" if j == 0 else None)
        r = (y - total[i]) / noise[i]
        ax2.step(w, np.where(spec.mask[i], r, np.nan), where="mid", color="k", lw=0.5)
    for a, b in problem.windows:
        for ax in (ax0, ax1, ax2):
            ax.axvspan(a, b, color="tab:blue", alpha=0.07, lw=0)
    ax2.axhline(0, color="crimson", lw=0.8)
    ax0.set_ylabel("F_ν [Jy]"); ax1.set_ylabel("F_ν − continuum [Jy]"); ax2.set_ylabel("(d−m)/σ")
    ax2.set_xlabel("wavelength [µm]")
    ax0.set_title(title or f"{spec.name}: full spectrum (shaded = fit windows)")
    ax0.legend(fontsize=8, loc="upper right"); ax1.legend(fontsize=8, ncol=6, loc="upper right")
    lo = np.nanpercentile(spec.line_flux[spec.mask], 0.5); hi = np.nanpercentile(spec.line_flux[spec.mask], 99.8)
    ax1.set_ylim(min(lo, -0.05 * hi), 1.3 * hi)
    ax2.set_ylim(-6, 6)
    ax0.set_xlim(np.nanmin(spec.wave), np.nanmax(spec.wave))
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------------------------
# Laplace approximation (0.19)
# ------------------------------------------------------------------------------------------------

def _laplace_params(lap, params=None, comps=None, max_params: int = 12):
    names = list(lap.names)
    if params:
        idx = [names.index(p) for p in params if p in names]
    elif comps:
        idx = [i for i, n in enumerate(names) if n.split(".")[0] in comps]
    else:
        idx = list(range(len(names)))
    return idx[:max_params]


def _hist2d_levels(x, y, bins=40, levels=(0.393, 0.865)):
    """Smoothed 2-D histogram and the density thresholds enclosing `levels` of the mass (1 and 2 sigma in 2-D)."""
    from scipy.ndimage import gaussian_filter
    H, xe, ye = np.histogram2d(x, y, bins=bins)
    H = gaussian_filter(H, 1.0)
    s = np.sort(H.ravel())[::-1]
    c = np.cumsum(s) / s.sum()
    thr = [s[min(np.searchsorted(c, lv), len(s) - 1)] for lv in levels]
    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    return xc, yc, H.T, sorted(set(thr))


def plot_laplace_corner(lap, mcmc=None, params=None, comps=None, max_params: int = 12, burn=None, size: float = 1.6):
    """Corner-style plot of the Laplace approximation: Gaussian marginals on the diagonal and 1-/2-sigma error
    ellipses below it.  With an MCMC result, its histograms and 1-/2-sigma contours are drawn on top, so the
    places where the Gaussian approximation fails (curved ridges, skewed or bounded posteriors) stand out."""
    idx = _laplace_params(lap, params, comps, max_params)
    n = len(idx)
    fig, axes = plt.subplots(n, n, figsize=(size * n + 1, size * n + 1), squeeze=False)
    names = [lap.names[i] for i in idx]
    flat = None
    if mcmc is not None:
        mnames = list(mcmc.names)
        if all(nm in mnames for nm in names):
            flat = mcmc.flat(burn)[:, [mnames.index(nm) for nm in names]]
    mu, sd = lap.theta[idx], lap.sigma[idx]
    C = lap.cov[np.ix_(idx, idx)]
    t = np.linspace(0, 2 * np.pi, 200)
    circ = np.vstack([np.cos(t), np.sin(t)])
    lap_col, mc_col = "#d97706", "#2563eb"
    for r in range(n):
        for c in range(n):
            ax = axes[r, c]
            if c > r:
                ax.axis("off"); continue
            lo_x = mu[c] - 3.5 * sd[c]; hi_x = mu[c] + 3.5 * sd[c]
            if flat is not None:
                lo_x = min(lo_x, np.percentile(flat[:, c], 0.5)); hi_x = max(hi_x, np.percentile(flat[:, c], 99.5))
            lo_x, hi_x = max(lo_x, lap.lo[idx[c]]), min(hi_x, lap.hi[idx[c]])
            if r == c:
                x = np.linspace(lo_x, hi_x, 200)
                ax.plot(x, np.exp(-0.5 * ((x - mu[c]) / sd[c]) ** 2), color=lap_col, lw=1.4)
                if flat is not None:
                    hh, e = np.histogram(flat[:, c], bins=40, range=(lo_x, hi_x))
                    ax.step(0.5 * (e[1:] + e[:-1]), hh / max(hh.max(), 1), where="mid", color=mc_col, lw=1.0)
                ax.set_yticks([])
                fl = lap.flags.get(names[c], [])
                ax.set_title(f"{names[c]}\n{mu[c]:.3g} ± {sd[c]:.2g}" + (" ⚑" if fl else ""), fontsize=7,
                             color="#b91c1c" if fl else "k")
            else:
                lo_y = mu[r] - 3.5 * sd[r]; hi_y = mu[r] + 3.5 * sd[r]
                if flat is not None:
                    lo_y = min(lo_y, np.percentile(flat[:, r], 0.5)); hi_y = max(hi_y, np.percentile(flat[:, r], 99.5))
                lo_y, hi_y = max(lo_y, lap.lo[idx[r]]), min(hi_y, lap.hi[idx[r]])
                if flat is not None:
                    try:
                        xc, yc, H, lev = _hist2d_levels(flat[:, c], flat[:, r])
                        ax.contour(xc, yc, H, levels=lev, colors=mc_col, linewidths=0.9)
                    except Exception:
                        pass
                S = C[np.ix_([c, r], [c, r])]
                w, v = np.linalg.eigh(S)
                for k, ls in ((np.sqrt(2.30), "-"), (np.sqrt(6.17), "--")):       # 68 % and 95 % in 2-D
                    e = v @ (np.sqrt(np.maximum(w, 0))[:, None] * circ) * k
                    ax.plot(mu[c] + e[0], mu[r] + e[1], color=lap_col, lw=1.1, ls=ls)
                ax.plot(mu[c], mu[r], "+", color=lap_col, ms=5)
                ax.set_ylim(lo_y, hi_y)
                if c == 0:
                    ax.set_ylabel(names[r], fontsize=7)
                else:
                    ax.set_yticklabels([])
            ax.set_xlim(lo_x, hi_x)
            if r == n - 1:
                ax.set_xlabel(names[c], fontsize=7)
            else:
                ax.set_xticklabels([])
            ax.tick_params(labelsize=6)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=lap_col, lw=1.4, label="Laplace (Gaussian) 68 / 95 %")]
    if flat is not None:
        handles.append(Line2D([], [], color=mc_col, lw=1.2, label="MCMC 68 / 95 %"))
    fig.legend(handles=handles, loc="upper right", fontsize=9, frameon=False)
    fig.tight_layout()
    return fig


def plot_laplace_correlation(lap, figsize=(7, 6)):
    """Correlation matrix of the Laplace approximation."""
    n = len(lap.names)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(lap.corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(lap.names, rotation=90, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(lap.names, fontsize=7)
    for i in range(n):
        for j in range(n):
            if i != j and abs(lap.corr[i, j]) >= 0.5:
                ax.text(j, i, f"{lap.corr[i, j]:.1f}", ha="center", va="center", fontsize=5,
                        color="w" if abs(lap.corr[i, j]) > 0.75 else "k")
    fig.colorbar(im, ax=ax, shrink=0.8, label="correlation")
    ax.set_title(f"Laplace correlations (condition number {lap.condition:.2g})", fontsize=9)
    fig.tight_layout()
    return fig
