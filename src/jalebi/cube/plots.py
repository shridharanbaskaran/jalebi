"""Figures for line maps, channel maps, PV diagrams and regions (matplotlib).

All maps are drawn in offsets from the source (arcsec, east to the left, north up) computed through
the cube's WCS, so rotated ("ifualign") cubes are shown correctly too.  The moment-0 style — asinh
stretch, faint white contours, star marker, scale bar in au, PSF beam — follows the old
channel_maps notebook (`draw_fancy_moment0_on_ax`).
"""
from __future__ import annotations

import numpy as np

from .. import plots as _backend  # noqa: F401  (non-interactive backend outside IPython, inline in Jupyter)
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.ticker import ScalarFormatter


def _grids(cube, center_radec):
    """Offsets (east, north) of pixel corners and centres."""
    ny, nx = cube.shape[1:]
    yc, xc = np.mgrid[-0.5:ny, -0.5:nx]
    ex, ey = cube.offsets(center_radec, xc, yc)
    y, x = np.mgrid[0:ny, 0:nx].astype(float)
    cx, cy = cube.offsets(center_radec, x, y)
    return ex, ey, cx, cy


def _limits(img, interval="percentile", percentile=99.5, vmin=None, vmax=None, symmetric=False):
    v = img[np.isfinite(img)]
    if v.size == 0:
        return 0.0, 1.0
    if symmetric:
        m = vmax if vmax is not None else float(np.nanpercentile(np.abs(v), 95))
        m = max(m, 1e-3)
        return -m, m
    if interval == "zscale":
        from astropy.visualization import ZScaleInterval
        lo, hi = ZScaleInterval().get_limits(v)
    elif interval == "minmax":
        lo, hi = float(v.min()), float(v.max())
    elif interval == "manual":
        lo, hi = vmin, vmax
    else:
        lo, hi = float(np.nanpercentile(v, 100 - percentile)), float(np.nanpercentile(v, percentile))
    lo = vmin if vmin is not None else lo
    hi = vmax if vmax is not None else hi
    if hi <= lo:
        hi = lo + 1e-30 + abs(lo) * 1e-3
    return lo, hi


def _norm(stretch, lo, hi, asinh_a=0.1):
    from astropy.visualization import AsinhStretch, ImageNormalize, LinearStretch, LogStretch
    st = {"asinh": AsinhStretch(a=asinh_a), "log": LogStretch(), "linear": LinearStretch()}[stretch]
    return ImageNormalize(vmin=lo, vmax=hi, stretch=st, clip=False)


def _sci_colorbar(fig, im, ax, label="", location="right"):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, location=location)
    fmt = ScalarFormatter(useMathText=True); fmt.set_powerlimits((-2, 3))
    cb.formatter = fmt; cb.update_ticks()
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    return cb


def draw_map(ax, img, cube, center_radec, cmap="inferno", stretch="asinh", interval="percentile", percentile=99.5,
             vmin=None, vmax=None, symmetric=False, asinh_a=0.1, contours=None, n_contours=8, contour_img=None,
             star=True, beam_fwhm=None, scale_bar_au=None, distance_pc=None, pa_deg=None, pa_length=1.0,
             label=None, core_radius=None, extent_arcsec=None):
    """One map in source offsets.  Returns the QuadMesh (for a colour bar)."""
    ex, ey, cx, cy = _grids(cube, center_radec)
    lo, hi = _limits(img, interval, percentile, vmin, vmax, symmetric)
    norm = None if symmetric else _norm(stretch, lo, hi, asinh_a)
    im = ax.pcolormesh(ex, ey, np.ma.masked_invalid(img), cmap=cmap, norm=norm, vmin=lo if symmetric else None,
                       vmax=hi if symmetric else None, shading="flat", rasterized=True)
    cimg = img if contour_img is None else contour_img
    if contours is None:
        contours = contour_img is not None or stretch == "asinh"
    if contours and np.isfinite(cimg).sum() > 10:
        clo, chi = _limits(cimg, "percentile", 99.5)
        if chi > max(clo, 0):
            lev = np.linspace(max(clo, 0) + 0.1 * (chi - max(clo, 0)), chi, n_contours)
            ax.contour(cx, cy, np.nan_to_num(cimg, nan=0.0), levels=lev, colors="white", linewidths=0.5, alpha=0.45)
    ax.set_facecolor("black" if cmap in ("inferno", "magma", "viridis", "cividis") else "0.85")
    if extent_arcsec:
        ax.set_xlim(extent_arcsec, -extent_arcsec); ax.set_ylim(-extent_arcsec, extent_arcsec)
    else:
        ax.set_xlim(np.nanmax(ex), np.nanmin(ex)); ax.set_ylim(np.nanmin(ey), np.nanmax(ey))
    ax.set_aspect("equal")
    if star:
        ax.scatter([0], [0], marker="*", s=70, facecolors="yellow", edgecolors="black", linewidths=0.8, zorder=10)
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    if beam_fwhm:
        ax.add_patch(Circle((x0 - 0.12 * (x0 - x1), y0 + 0.12 * (y1 - y0)), beam_fwhm / 2, fc="none", ec="white", lw=1.0, zorder=9))
    if scale_bar_au and distance_pc:
        L = scale_bar_au / distance_pc                    # arcsec
        xa = x1 + 0.08 * (x0 - x1); ya = y0 + 0.08 * (y1 - y0)
        ax.plot([xa + L, xa], [ya, ya], color="white", lw=2, zorder=9)
        ax.text(xa + L / 2, ya + 0.03 * (y1 - y0), f"{scale_bar_au:g} au", color="white", ha="center", va="bottom", fontsize=7)
    if pa_deg is not None:
        t = np.deg2rad(pa_deg)
        ax.plot([-pa_length * np.sin(t), pa_length * np.sin(t)], [-pa_length * np.cos(t), pa_length * np.cos(t)],
                color="cyan", lw=1.0, ls="--", zorder=8)
    if core_radius:
        ax.add_patch(Circle((0, 0), core_radius, fc="none", ec="0.7", lw=0.8, ls=":", zorder=8))
    if label:
        ax.text(0.04, 0.95, label, transform=ax.transAxes, color="white", fontsize=8, fontweight="bold", va="top",
                bbox=dict(fc="black", ec="none", alpha=0.35, pad=2))
    ax.tick_params(direction="in", color="white", labelsize=7)
    return im


def velocity_limit(v, verr=None, pct: float = 95.0) -> float | None:
    """Symmetric colour limit for a velocity map from its better-measured spaxels (errors in the best 75 %),
    so a few noisy edge spaxels do not wash out the map."""
    good = np.isfinite(v)
    if verr is not None:
        e = np.where(np.isfinite(verr), verr, np.inf)
        if np.isfinite(e).any():
            good &= e <= np.nanpercentile(e[np.isfinite(e)], 75)
    if good.sum() < 5:
        return None
    return float(max(np.nanpercentile(np.abs(v[good]), pct), 1.0))


def _scale_bar_au(distance_pc, fov_arcsec):
    if not distance_pc:
        return None
    target = 0.25 * fov_arcsec * distance_pc
    for v in (10, 20, 25, 50, 100, 200, 250, 500, 1000):
        if v >= target * 0.6:
            return v
    return 1000


def plot_line_maps(lm, distance_pc: float | None = None, pa_deg: float | None = None, figsize=(15, 9.2), extent_arcsec=None):
    """Six panels: continuum, line moment 0, extended moment 0, centroid velocity, velocity error, and the
    integrated line spectra (total / point source / extended)."""
    lc = lm.lc
    cube = lc.cube
    c0 = lc.center_radec
    ps = lc.fwhm_arcsec
    fov = cube.shape[2] * cube.pixscale
    sb = _scale_bar_au(distance_pc, extent_arcsec * 2 if extent_arcsec else fov)
    fig, axs = plt.subplots(2, 3, figsize=figsize)
    kw = dict(beam_fwhm=ps, distance_pc=distance_pc, scale_bar_au=sb, pa_deg=pa_deg, extent_arcsec=extent_arcsec)
    im = draw_map(axs[0, 0], lm["continuum"], cube, c0, cmap="viridis", stretch="log", percentile=99.9, contours=False,
                  label=f"continuum @ {lc.line.wave:.3f} µm", **kw)
    _sci_colorbar(fig, im, axs[0, 0], "MJy sr$^{-1}$")
    m0 = np.where(lm["snr"] >= 2, lm["mom0"], np.nan)
    im = draw_map(axs[0, 1], m0, cube, c0, label=f"{lc.line.name}  mom 0 (S/N≥2)", **kw)
    stack = lm.units.get("mom0") == "normalised"
    _sci_colorbar(fig, im, axs[0, 1], "normalised (stack)" if stack else "erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$")
    if "mom0_ext" in lm.maps:
        me = np.where(lm["snr_ext"] >= 2, lm["mom0_ext"], np.nan)
        core = lc.settings.get("psf", {}).get("core_radius_fwhm", 0.5) * ps
        im = draw_map(axs[0, 2], me, cube, c0, label="extended (point source removed)", core_radius=core, **kw)
        _sci_colorbar(fig, im, axs[0, 2], "erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$")
    else:
        im = draw_map(axs[0, 2], lm["snr"], cube, c0, cmap="magma", stretch="linear", contours=False, label="S/N (mom 0)", **kw)
        _sci_colorbar(fig, im, axs[0, 2], "S/N")
    if "vcen" in lm.maps:
        v = lm["vcen"]
        im = draw_map(axs[1, 0], v, cube, c0, cmap="RdBu_r", symmetric=True, vmax=velocity_limit(v, lm["vcen_err"]),
                      contour_img=lm.maps.get("mom0_ext", lm["mom0"]),
                      label="centroid velocity (Gaussian)" + (" − v$_\\star$" if lm.summary.get("zero_point") == "star" else ""), **kw)
        _sci_colorbar(fig, im, axs[1, 0], "km s$^{-1}$")
        im = draw_map(axs[1, 1], lm["vcen_err"], cube, c0, cmap="cividis", stretch="linear", contours=False,
                      label="velocity error (Monte Carlo)" if lm.summary.get("n_mc") else "velocity error", **kw)
        _sci_colorbar(fig, im, axs[1, 1], "km s$^{-1}$")
    else:
        im = draw_map(axs[1, 0], lm["mom1"], cube, c0, cmap="RdBu_r", symmetric=True, contour_img=lm["mom0"], label="moment 1", **kw)
        _sci_colorbar(fig, im, axs[1, 0], "km s$^{-1}$")
        im = draw_map(axs[1, 1], lm["mom2"], cube, c0, cmap="cividis", stretch="linear", contours=False, label="moment 2", **kw)
        _sci_colorbar(fig, im, axs[1, 1], "km s$^{-1}$")
    for a in axs.flat[:5]:
        a.set_xlabel("ΔRA [″]", fontsize=8); a.set_ylabel("ΔDec [″]", fontsize=8)
    ax = axs[1, 2]
    sp = lm.spectra["integrated"]
    ax.step(sp["v_kms"], sp["line_Jy"] * 1e3, where="mid", color="k", lw=1.2, label="all spaxels")
    if "point_source_Jy" in sp:
        ax.step(sp["v_kms"], sp["point_source_Jy"] * 1e3, where="mid", color="tab:orange", lw=1.0, label="point source")
        ax.step(sp["v_kms"], sp["extended_Jy"] * 1e3, where="mid", color="tab:blue", lw=1.0, label="extended")
    ax.axhline(0, color="0.6", lw=0.6); ax.axvline(0, color="0.6", lw=0.6, ls=":")
    lk = lm.summary.get("line_kms", 300)
    ax.axvspan(-lk, lk, color="0.9", zorder=-1)
    ax.set_xlabel("velocity [km s$^{-1}$]", fontsize=8)
    ax.set_ylabel("stacked profile (normalised)" if stack else "line flux density [mJy]", fontsize=8)
    ax.legend(fontsize=7, frameon=False); ax.tick_params(labelsize=7)
    ax.set_title("integrated spectra (continuum removed)", fontsize=8)
    s = lm.summary
    title = (f"{cube.name} — {lc.line.name} ({lc.line.wave:.4f} µm, {cube.band}) · PSF {ps:.2f}″ · instrumental FWHM "
             f"{lc.lsf_fwhm_kms:.0f} km/s")
    if s.get("total_flux_W_m2_snr_masked") is not None:
        title += f" · total {s['total_flux_W_m2_snr_masked']:.2e} W m$^{{-2}}$"
    if s.get("point_source_line_flux_W_m2") is not None:
        title += f" · point source {s['point_source_line_flux_W_m2']:.2e}"
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def plot_fancy_moment0(lm, which: str = "mom0_ext", distance_pc=None, pa_deg=None, extent_arcsec=None, asinh_a=0.25,
                       cmap="inferno", label=None, figsize=(3.6, 4.2), snr_min=2.0):
    """A single publication-style moment-0 panel (the notebook's `draw_fancy_moment0_on_ax`)."""
    lc = lm.lc
    img = lm.maps.get(which, lm["mom0"])
    snr = lm.maps.get("snr_ext" if which == "mom0_ext" else "snr", lm["snr"])
    img = np.where(snr >= snr_min, img, np.nan)
    fig, ax = plt.subplots(figsize=figsize)
    fov = lc.cube.shape[2] * lc.cube.pixscale
    im = draw_map(ax, img, lc.cube, lc.center_radec, cmap=cmap, asinh_a=asinh_a, n_contours=10, beam_fwhm=lc.fwhm_arcsec,
                  distance_pc=distance_pc, scale_bar_au=_scale_bar_au(distance_pc, extent_arcsec * 2 if extent_arcsec else fov),
                  pa_deg=pa_deg, extent_arcsec=extent_arcsec, label=label or f"{lc.cube.name}\n{lc.line.name}")
    _sci_colorbar(fig, im, ax, "erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$", location="top")
    ax.set_xlabel("ΔRA [″]", fontsize=8); ax.set_ylabel("ΔDec [″]", fontsize=8)
    fig.tight_layout()
    return fig


def plot_channel_maps(cm, ncols: int = 6, distance_pc=None, extent_arcsec=None, cmap="inferno", asinh_a=0.2, panel=2.3):
    nv = len(cm.velocity)
    nrows = int(np.ceil(nv / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(panel * ncols, panel * nrows + 0.6), squeeze=False)
    lo, hi = _limits(cm.data, "percentile", 99.7)
    lo = max(lo, 0.0)
    lc = cm.lc
    fov = lc.cube.shape[2] * lc.cube.pixscale
    sb = _scale_bar_au(distance_pc, extent_arcsec * 2 if extent_arcsec else fov)
    im = None
    for i, ax in enumerate(axs.flat):
        if i >= nv:
            ax.axis("off"); continue
        im = draw_map(ax, cm.data[i], lc.cube, lc.center_radec, cmap=cmap, vmin=lo, vmax=hi, interval="manual", asinh_a=asinh_a,
                      contours=False, beam_fwhm=lc.fwhm_arcsec if i == 0 else None, distance_pc=distance_pc,
                      scale_bar_au=sb if i == 0 else None, extent_arcsec=extent_arcsec, label=f"{cm.velocity[i]:+.0f} km/s")
        if i % ncols:
            ax.set_yticklabels([])
        if i < nv - ncols:
            ax.set_xticklabels([])
    fig.suptitle(f"{lc.cube.name} — {lc.line.name} channel maps ({cm.source}, Δv = {cm.dv:.0f} km/s)", fontsize=10)
    if im is not None:
        _sci_colorbar(fig, im, list(axs.flat), "erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$ per channel")
    return fig


def plot_pv(pv, figsize=(6.5, 4.5), cmap="inferno", vlim=None):
    fig, ax = plt.subplots(figsize=figsize)
    lo, hi = _limits(pv.data, "percentile", 99.5)
    lo = max(lo, 0.0) if hi > 0 else lo
    ds = np.diff(pv.offset).mean() if len(pv.offset) > 1 else 1.0
    dv = np.diff(pv.velocity).mean() if len(pv.velocity) > 1 else 1.0
    ext = [pv.offset[0] - ds / 2, pv.offset[-1] + ds / 2, pv.velocity[0] - dv / 2, pv.velocity[-1] + dv / 2]
    im = ax.imshow(pv.data, origin="lower", aspect="auto", extent=ext, cmap=cmap, norm=_norm("asinh", lo, hi, 0.2))
    if np.isfinite(pv.data).sum() > 10:
        lev = np.linspace(lo + 0.15 * (hi - lo), hi, 6)
        ax.contour(pv.offset, pv.velocity, np.nan_to_num(pv.data), levels=lev, colors="white", linewidths=0.5, alpha=0.5)
    ax.axhline(0, color="w", lw=0.5, ls=":"); ax.axvline(0, color="w", lw=0.5, ls=":")
    if vlim:
        ax.set_ylim(-vlim, vlim)
    ax.set_xlabel(f"offset along PA {pv.pa_deg:g}° [″]"); ax.set_ylabel("velocity [km s$^{-1}$]")
    ax.set_title(f"{pv.lc.cube.name} — {pv.lc.line.name} PV ({pv.source}, width {pv.width_arcsec:.2f}″)", fontsize=9)
    _sci_colorbar(fig, im, ax, "MJy sr$^{-1}$")
    fig.tight_layout()
    return fig


def plot_regions(image, cube, center_radec, regions, spectrum=None, labels=None, figsize=(13, 4.6)):
    """A map with region outlines, and (optionally) the extracted region spectrum."""
    ncols = 2 if spectrum is not None else 1
    fig = plt.figure(figsize=figsize if spectrum is not None else (5, 4.6))
    ax = fig.add_subplot(1, ncols, 1)
    im = draw_map(ax, image, cube, center_radec, label=cube.name)
    _sci_colorbar(fig, im, ax, "")
    from .regions import _tangent
    cols = ["cyan", "lime", "magenta", "orange", "white"]
    for i, r in enumerate(regions if isinstance(regions, (list, tuple)) else [regions]):
        ra, dec = r.outline()
        dx, dy = _tangent(ra, dec, *center_radec)
        ax.plot(dx, dy, color=cols[i % len(cols)], lw=1.2)
        if labels:
            ax.text(np.nanmean(dx), np.nanmax(dy), labels[i], color=cols[i % len(cols)], fontsize=7, ha="center", va="bottom")
    ax.set_xlabel("ΔRA [″]"); ax.set_ylabel("ΔDec [″]")
    if spectrum is not None:
        ax2 = fig.add_subplot(1, 2, 2)
        for b in spectrum.bands:
            i = spectrum.band_slice(b)
            ax2.plot(spectrum.wave[i], spectrum.flux[i] * 1e3, lw=0.7)
        ax2.set_xlabel("wavelength [µm]"); ax2.set_ylabel("flux density [mJy]")
        ax2.set_title(spectrum.name, fontsize=9)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------------------------
# WCS-axes figures in the style of cube_maps.py (pixel grid, RA/Dec axes, gist_rainbow)
# ----------------------------------------------------------------------------------------------

def _stretch_limits(data, stretch="percentile", percentile=95, vmin=None, vmax=None):
    fin = data[np.isfinite(data)]
    if stretch == "percentile":
        from astropy.visualization import PercentileInterval
        return PercentileInterval(percentile).get_limits(fin) if fin.size else (0.0, 1.0)
    if stretch == "zscale":
        from astropy.visualization import ZScaleInterval
        return ZScaleInterval().get_limits(fin) if fin.size else (0.0, 1.0)
    if stretch == "linear":
        if vmin is None or vmax is None:
            raise ValueError("For linear stretch, both vmin and vmax must be provided.")
        return vmin, vmax
    raise ValueError("stretch must be 'percentile', 'zscale', or 'linear'")


def _rms_patch(ax, center_x, center_y, radius, fontsize=9):
    ax.add_patch(Circle((center_x, center_y), radius, edgecolor="cyan", facecolor="none", lw=2, linestyle="--"))
    ax.text(center_x, center_y - radius - 2, "RMS region", color="cyan", fontsize=fontsize)


def plot_map_wcs(data, wcs=None, fig=None, title=None, cmap="gist_rainbow", stretch="percentile", percentile=95, vmin=None,
                 vmax=None, center_x=12, center_y=8, radius=3, sigma_clip=False, sigma_mode="rms", sigma_thresh=3,
                 show_outline=False, show_colorbar=True, figsize=(8, 6), fontsize=11, subplot_index=111, label="Flux",
                 verbose=False):
    """A 2-D map on RA/Dec axes (pixel grid, `imshow(origin="lower")`): the engine of cube_maps.py's
    `plot_moment0_map`.  sigma_clip=True keeps pixels >= sigma_thresh x (nanstd | nanmean | nanmedian
    of the RMS circle) and draws the circle.  Returns (fig, ax, im, threshold)."""
    from matplotlib.patches import Rectangle
    data = np.array(data, float)
    ny, nx = data.shape
    thr = None
    if sigma_clip:
        from .masks import rms_mask
        data, thr, _ = rms_mask(data, center_x, center_y, radius, sigma_thresh, sigma_mode, check=False)
        if verbose:
            print(f"Applied {sigma_thresh}σ threshold: {thr:.3g}")
    lo, hi = _stretch_limits(data, stretch, percentile, vmin, vmax)
    if fig is None:
        fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(subplot_index, projection=wcs) if wcs is not None else fig.add_subplot(subplot_index)
    im = ax.imshow(data, origin="lower", cmap=cmap, vmin=lo, vmax=hi)
    ax.set_xlabel("RA" if wcs is not None else "x [pixel]", fontsize=fontsize)
    ax.set_ylabel("Dec" if wcs is not None else "y [pixel]", fontsize=fontsize)
    ax.set_title(title if title else "Moment-0 Map", fontsize=fontsize)
    if show_colorbar:
        cb = fig.colorbar(im, ax=ax, orientation="vertical", label=label)
        cb.ax.tick_params(labelsize=fontsize - 1)
    if sigma_clip:
        _rms_patch(ax, center_x, center_y, radius, fontsize - 2)
    if show_outline:
        ax.add_patch(Rectangle((0, 0), nx, ny, edgecolor="white", facecolor="none", lw=2))
    return fig, ax, im, thr


def add_au_box(ax, distance_pc, au_size=250, pixel_scale=0.2, color="black", center_x=None, center_y=None, ra=None, dec=None,
               marker_style="*", marker_size=50, label=False):
    """A square of `au_size` au (side = au_size / distance_pc / pixel_scale pixels) and a marker at its centre,
    centred on (center_x, center_y) pixels or (ra, dec) degrees on a WCS axis — cube_maps.py's `add_au_box`.
    pixel_scale: arcsec per spaxel (MRS: 0.13 ch1, 0.17 ch2, 0.20 ch3, 0.35 ch4 for the pipeline's cubes;
    `Cube.pixscale` gives it)."""
    from matplotlib.patches import Rectangle
    if ra is not None and dec is not None:
        if not hasattr(ax, "wcs"):
            raise ValueError("RA/Dec given but axis does not have WCS info.")
        from astropy.coordinates import SkyCoord
        center_x, center_y = ax.wcs.world_to_pixel(SkyCoord(ra, dec, unit="deg"))
    if center_x is None or center_y is None:
        raise ValueError("You must provide either (center_x, center_y) or (ra, dec).")
    side = au_size / distance_pc / pixel_scale
    ax.scatter(center_x, center_y, marker=marker_style, s=marker_size, color=color, zorder=3)
    rect = Rectangle((center_x - side / 2, center_y - side / 2), side, side, edgecolor=color, facecolor="none", lw=2, linestyle="-")
    ax.add_patch(rect)
    if label:
        ax.text(center_x, center_y + side / 2 + 2, f"{au_size:g} au", color=color, ha="center", fontsize=9)
    return rect


def plot_ratio_map(rm, percentile_intervals=(80, 80, 80), cmap="gist_rainbow", figsize=(18, 6), label_unit=True):
    """cube_maps.py's `make_ratio_plot` figure: [line 1], [line 2 on line 1's grid], line 1 / line 2, each on
    RA/Dec axes with percentile limits, the RMS circle and a white outline around the ratio panel."""
    from astropy.visualization import PercentileInterval
    from matplotlib.patches import Rectangle
    n1, n2 = rm.names
    if np.isfinite(rm.masked1).sum() == 0:
        raise ValueError(f"No valid data in {n1} map after masking. Check the RMS threshold or the line strength.")
    if np.isfinite(rm.masked2).sum() == 0:
        raise ValueError(f"No valid data in {n2} map after masking. Check the RMS threshold or the line strength.")
    lims = [PercentileInterval(percentile_intervals[0]).get_limits(rm.masked1[np.isfinite(rm.masked1)]),
            PercentileInterval(percentile_intervals[1]).get_limits(rm.masked2[np.isfinite(rm.masked2)]),
            PercentileInterval(percentile_intervals[2]).get_limits(rm.ratio[rm.valid]) if rm.valid.any() else (0.0, 1.0)]
    fig = plt.figure(figsize=figsize)
    ny, nx = rm.ratio.shape
    how = "σ Masked" if rm.rms_region is not None else "σ (S/N) Masked"
    unit = f" [{rm.unit}]" if (label_unit and rm.unit) else ""
    panels = [(rm.masked1, f"{n1} Moment-0 ({rm.sigma_thresh[0]:g}{how})", "Flux" + unit),
              (rm.masked2, f"{n2} Moment-0 ({rm.sigma_thresh[1]:g}{how})", "Flux" + unit),
              (rm.ratio, f"{n1}/{n2} Flux Ratio Map", "Flux Ratio")]
    axes = []
    for i, (img, title, lab) in enumerate(panels):
        ax = fig.add_subplot(1, 3, i + 1, projection=rm.wcs) if rm.wcs is not None else fig.add_subplot(1, 3, i + 1)
        im = ax.imshow(img, origin="lower", cmap=cmap, vmin=lims[i][0], vmax=lims[i][1])
        ax.set_title(title); ax.set_xlabel("RA"); ax.set_ylabel("Dec")
        if rm.rms_region is not None:
            _rms_patch(ax, *rm.rms_region)
        if i == 2:
            ax.add_patch(Rectangle((0, 0), nx, ny, edgecolor="white", facecolor="none", lw=2))
        plt.colorbar(im, ax=ax, orientation="vertical", label=lab)
        axes.append(ax)
    fig.tight_layout()
    return fig, axes
