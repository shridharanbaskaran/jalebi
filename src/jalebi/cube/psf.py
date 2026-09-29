"""Point-source removal: what is left after the unresolved disk emission is taken out.

Most Class II disks are unresolved in the continuum, so the continuum image next to a line *is* the
PSF at that wavelength (including the MRS PSF's wavelength dependence, its asymmetries and the
cube-building artefacts of that sub-band).  For every plane k of the continuum-subtracted cube L:

    P_k(x, y) = C_k(x, y)            continuum model of that spaxel at that wavelength, within
                                     `psf_radius_fwhm` x FWHM(lambda_k) of the source (0 outside)
    s_k       = sum_core(w L_k P_k) / sum_core(w P_k^2)   least squares in the PSF core
    E_k       = L_k - s_k P_k        extended emission

`s_k` is the line-to-continuum ratio of the unresolved component, so `s_k * sum(P_k)` is the
point-source line spectrum.  Because the PSF template is the continuum at the same wavelength, the
change of the PSF across a sub-band is followed plane by plane.

The price: any extended line emission inside the core is absorbed into the point source, so E is
~0 in the core by construction (the maps mark it).  Several sources (binaries) can be removed at once;
every spaxel belongs to its nearest source.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..data import mrs_psf_fwhm


@dataclass
class PSFSettings:
    enabled: bool = True
    scale: str = "core"                # core (least squares in the core) | peak (zero residual at the peak spaxel)
    core_radius_fwhm: float = 0.5      # core used to scale the template
    psf_radius_fwhm: float = 8.0       # template support (cosine taper over its outer 20 %)
    template: str = "continuum"        # continuum | a FITS image/cube path (external PSF model)
    extra_sources: list = field(default_factory=list)    # more point sources [(ra, dec), ...]
    auto_sources: bool = True          # also remove other compact continuum peaks (e.g. a companion)
    auto_threshold: float = 0.1        # ... brighter than this fraction of the main peak
    template_background: bool = True   # subtract the median continuum far from all sources from the template


@dataclass
class PSFResult:
    model: np.ndarray            # (nz, ny, nx) point-source model of the line emission (MJy/sr)
    extended: np.ndarray         # L - model
    scale: np.ndarray            # (nsrc, nz) s_k per source
    point_flux_jy: np.ndarray    # (nsrc, nz) point-source line spectrum in Jy
    core_mask: np.ndarray        # (ny, nx) spaxels inside the core of any source
    centers_pix: list


def _external_template(path: str, shape, wave):
    from astropy.io import fits
    d = np.asarray(fits.getdata(path), float)
    if d.ndim == 2:
        return np.repeat(d[None], shape[0], axis=0)
    if d.shape == shape:
        return d
    raise ValueError(f"PSF template {path} has shape {d.shape}, expected (ny, nx) or {shape}")


def find_companions(image: np.ndarray, main_xy, fwhm_pix: float, threshold: float = 0.1, min_sep_fwhm: float = 3.0,
                    max_sources: int = 4) -> list[tuple[float, float]]:
    """Other compact peaks of a continuum image: local maxima (in a 2 FWHM box) brighter than `threshold` x the
    main source, at least `min_sep_fwhm` FWHM away from it and from each other, and at least 3x brighter than
    their surrounding annulus (so extended scattered light is not taken for a star)."""
    from scipy.ndimage import maximum_filter
    from .io import _refine_peak
    img = np.where(np.isfinite(image), image, -np.inf)
    ny, nx = img.shape
    mx, my = main_xy
    iy, ix = int(round(min(max(my, 0), ny - 1))), int(round(min(max(mx, 0), nx - 1)))
    main = np.nanmax(image[max(iy - 1, 0):iy + 2, max(ix - 1, 0):ix + 2])
    if not np.isfinite(main) or main <= 0:
        return []
    size = max(3, int(2 * fwhm_pix) | 1)
    peaks = (img == maximum_filter(img, size=size)) & (img > threshold * main)
    yy, xx = np.mgrid[0:ny, 0:nx]
    found = []
    for y, x in sorted(zip(*np.nonzero(peaks)), key=lambda t: -img[t]):
        if np.hypot(x - mx, y - my) < min_sep_fwhm * fwhm_pix:
            continue
        if any(np.hypot(x - a, y - b) < min_sep_fwhm * fwhm_pix for a, b in found):
            continue
        r = np.hypot(xx - x, yy - y)
        ann = image[(r > 1.5 * fwhm_pix) & (r < 2.5 * fwhm_pix) & np.isfinite(image)]
        if ann.size < 4 or image[y, x] < 3.0 * max(np.nanmedian(ann), 0.0) + 1e-30:
            continue
        # skip peaks on the field edge (partially covered, often artefacts)
        if x < 1 or y < 1 or x > nx - 2 or y > ny - 2 or not np.all(np.isfinite(image[y - 1:y + 2, x - 1:x + 2])):
            continue
        found.append(_refine_peak(image, int(x), int(y), 2))
        if len(found) >= max_sources:
            break
    return found


def subtract_point_sources(line_cube: np.ndarray, cont: np.ndarray, wave: np.ndarray, centers_pix, pixscale: float,
                           pixar_sr: float, err: np.ndarray | None = None, settings: PSFSettings | None = None) -> PSFResult:
    """Remove unresolved emission centred at each (x, y) of `centers_pix` (0-based pixel coordinates)."""
    s = settings or PSFSettings()
    nz, ny, nx = line_cube.shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    centers = [tuple(map(float, c)) for c in centers_pix]
    # each spaxel belongs to its nearest source
    d2 = np.stack([(xx - cx) ** 2 + (yy - cy) ** 2 for cx, cy in centers])
    owner = np.argmin(d2, axis=0)
    fwhm_pix = mrs_psf_fwhm(wave) / pixscale                                     # (nz,)
    if s.template == "continuum":
        T = np.where(np.isfinite(cont), cont, 0.0)
        if s.template_background:
            # a diffuse continuum level (background residuals, a companion's halo) is not PSF: remove the
            # median continuum of the spaxels far from every source, plane by plane
            rmin = np.sqrt(d2.min(axis=0))
            fin = np.isfinite(cont[len(cont) // 2])
            r75 = float(np.percentile(rmin[fin], 75)) if fin.any() else 0.0
            rfar = np.minimum(s.psf_radius_fwhm * fwhm_pix, r75)                     # (nz,)
            far = (rmin[None] > rfar[:, None, None]) & (rfar[:, None, None] >= 3.0 * fwhm_pix[:, None, None])
            if (far & np.isfinite(cont)).sum() > 20 * nz:
                with np.errstate(all="ignore"):
                    b = np.nanmedian(np.where(far & np.isfinite(cont), cont, np.nan), axis=(1, 2))
                T = T - np.nan_to_num(b)[:, None, None]
    else:
        T = np.nan_to_num(_external_template(s.template, line_cube.shape, wave))
    L = np.where(np.isfinite(line_cube), line_cube, 0.0)
    good = np.isfinite(line_cube)
    w = np.ones_like(L) if err is None else np.where(np.isfinite(err) & (err > 0), 1.0 / np.maximum(err, 1e-30) ** 2, 0.0)
    model = np.zeros_like(L)
    scale = np.zeros((len(centers), nz)); pflux = np.zeros((len(centers), nz))
    core_any = np.zeros((ny, nx), bool)
    for j, (cx, cy) in enumerate(centers):
        r = np.sqrt(d2[j])                                                       # (ny, nx)
        mine = owner == j
        rmax = s.psf_radius_fwhm * fwhm_pix[:, None, None]
        # cosine taper over the outer 20 % of the support, so a finite template leaves no ring
        taper = np.clip((rmax - r[None]) / (0.2 * rmax), 0.0, 1.0)
        taper = 0.5 - 0.5 * np.cos(np.pi * taper)
        support = mine[None] & (r[None] <= rmax)
        core = mine[None] & (r[None] <= np.maximum(s.core_radius_fwhm * fwhm_pix[:, None, None], 1.0)) & good
        core_any |= (mine & (r <= max(s.core_radius_fwhm * float(np.median(fwhm_pix)), 1.0)))
        P = np.where(support, T * taper, 0.0)
        P = np.clip(P, 0.0, None)
        if s.scale == "peak":
            iy, ix = int(round(cy)), int(round(cx))
            iy = min(max(iy, 0), ny - 1); ix = min(max(ix, 0), nx - 1)
            den = P[:, iy, ix]
            sk = np.where(den > 0, L[:, iy, ix] / np.where(den > 0, den, 1.0), 0.0)
        else:
            num = np.sum(np.where(core, w * L * P, 0.0), axis=(1, 2))
            den = np.sum(np.where(core, w * P * P, 0.0), axis=(1, 2))
            sk = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
        model += sk[:, None, None] * P
        scale[j] = sk
        pflux[j] = sk * P.sum(axis=(1, 2)) * pixar_sr * 1e6
    model = np.where(good, model, np.nan)
    return PSFResult(model, line_cube - model, scale, pflux, core_any, centers)
