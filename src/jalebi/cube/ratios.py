"""Line-ratio maps (cube_maps.py's `make_ratio_plot`, on top of jalebi's line maps).

    m1 = cube.line_maps(cube.prepare_line(cs, "[Fe II] 5.34"))
    m2 = cube.line_maps(cube.prepare_line(cs, "[Ne II] 12.81"))
    rm = cube.ratio_map(m1, m2, rms_region=(12, 8, 3), sigma_thresh=(5, 5))     # line 2 onto line 1's grid
    rm.write_fits("FeII_NeII_ratio.fits"); cube.plots.plot_ratio_map(rm)

Line 2 is reprojected onto the spaxel grid of line 1 (bilinear, `reproject_interp` when the reproject
package is installed, as in cube_maps.py), each map is masked at sigma_thresh x its noise, and the ratio
is taken where both survive and line 2 > 0.  The noise is either the empty-sky circle of cube_maps.py
(`rms_region=(x, y, r)` in line-1 pixels, nanstd / mean / median) or, without it, the propagated moment-0
errors of jalebi (S/N >= sigma_thresh).

Units.  From LineMaps the maps are in erg s-1 cm-2 sr-1, so the ratio is a ratio of line fluxes.  The
moment-0 files of cube_maps.py are integrals of I_nu over wavelength (MJy/sr x m); their ratio equals the
flux ratio times (lambda_1 / lambda_2)^2 — `unit="native"` reproduces those numbers.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

import numpy as np

from .masks import check_circle, region_level


@dataclass
class RatioMap:
    """Masked line maps on line 1's grid and their ratio."""
    names: tuple                     # (line 1, line 2)
    waves: tuple                     # rest wavelengths [um]
    map1: np.ndarray                 # line 1, unmasked
    map2: np.ndarray                 # line 2 reprojected onto line 1's grid, unmasked
    masked1: np.ndarray
    masked2: np.ndarray
    ratio: np.ndarray
    wcs: object                      # celestial WCS of line 1 (None for plain arrays)
    unit: str                        # unit of map1 / map2
    thresholds: tuple
    noise: tuple                     # the level each threshold is sigma_thresh times
    sigma_thresh: tuple
    rms_region: tuple | None         # (x, y, r) pixels, or None (propagated errors)
    rms_mode: str = "rms"
    reprojected: bool = False
    header: object = None
    meta: dict = field(default_factory=dict)

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.ratio)

    def summary(self) -> dict:
        r = self.ratio[self.valid]
        return {"lines": list(self.names), "unit": self.unit, "n_valid": int(r.size),
                "median_ratio": float(np.median(r)) if r.size else None,
                "p16_p84": [float(np.percentile(r, 16)), float(np.percentile(r, 84))] if r.size else None,
                "thresholds": list(self.thresholds), "noise": list(self.noise), "sigma_thresh": list(self.sigma_thresh),
                "rms_region": list(self.rms_region) if self.rms_region is not None else None, "rms_mode": self.rms_mode,
                "reprojected": self.reprojected}

    def write_fits(self, path: str) -> str:
        """RATIO (primary), LINE1, LINE2 (masked, on line 1's grid) with line 1's celestial WCS."""
        from astropy.io import fits
        h = fits.Header() if self.header is None else self.header.copy()
        if self.header is None and self.wcs is not None:
            h.update(self.wcs.to_header())
        h["LINE1"] = str(self.names[0]); h["LINE2"] = str(self.names[1])
        h["WAVE1"] = (float(self.waves[0]), "rest wavelength of line 1 [um]")
        h["WAVE2"] = (float(self.waves[1]), "rest wavelength of line 2 [um]")
        h["MAPUNIT"] = (self.unit, "unit of the line maps")
        fin = lambda x: float(x) if np.isfinite(x) else None      # noqa: E731  (FITS headers cannot hold NaN)
        h["SIGMA1"] = float(self.sigma_thresh[0]); h["SIGMA2"] = float(self.sigma_thresh[1])
        h["THRESH1"] = fin(self.thresholds[0]); h["THRESH2"] = fin(self.thresholds[1])
        if self.rms_region is not None:
            h["RMS_X"], h["RMS_Y"], h["RMS_R"] = (float(v) for v in self.rms_region)
            h["RMS_MODE"] = self.rms_mode
        hr = h.copy(); hr["BUNIT"] = ""; hr["EXTNAME"] = "RATIO"
        h1 = h.copy(); h1["BUNIT"] = self.unit; h1["EXTNAME"] = "LINE1"
        h2 = h.copy(); h2["BUNIT"] = self.unit; h2["EXTNAME"] = "LINE2"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fits.HDUList([fits.PrimaryHDU(self.ratio.astype(np.float32), header=hr),
                      fits.ImageHDU(self.masked1.astype(np.float32), header=h1, name="LINE1"),
                      fits.ImageHDU(self.masked2.astype(np.float32), header=h2, name="LINE2")]).writeto(path, overwrite=True)
        return path


def reproject_image(img: np.ndarray, wcs_in, wcs_out, shape_out, exact: bool | None = None) -> np.ndarray:
    """Bilinear resampling of a surface-brightness image onto another celestial grid.  Uses
    reproject.reproject_interp when installed (as cube_maps.py; exact=False skips it), otherwise
    scipy map_coordinates through the two WCS."""
    img = np.asarray(img, float)
    if exact is not False:
        try:
            from reproject import reproject_interp
        except ImportError:
            if exact:
                raise
        else:
            out, _ = reproject_interp((img, wcs_in), wcs_out, shape_out=tuple(shape_out))
            return np.asarray(out, float)
    from scipy.ndimage import map_coordinates
    ny, nx = shape_out
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    sky = wcs_out.pixel_to_world_values(xx, yy)
    xs, ys = wcs_in.world_to_pixel_values(*sky)
    # as reproject's bilinear sampling: edge-padded by one pixel so the outer half of the edge pixels is
    # defined, NaN outside [-0.5, n - 0.5], and a NaN neighbour makes the sample NaN
    pad = np.pad(img, 1, mode="edge")
    out = map_coordinates(pad, [ys + 1, xs + 1], order=1, mode="constant", cval=np.nan)
    out[(xs < -0.5) | (xs > img.shape[1] - 0.5) | (ys < -0.5) | (ys > img.shape[0] - 0.5)] = np.nan
    return out


def _same_grid(w1, w2, s1, s2) -> bool:
    if tuple(s1) != tuple(s2):
        return False
    try:
        return bool(w1.wcs.compare(w2.wcs))
    except Exception:
        return False


def _as_map(m, key: str, unit: str, wcs=None):
    """(image, wcs, unit, name, wave, err, header) from LineMaps, a FITS file/HDU or an array."""
    from .maps import UNIT_LABELS, LineMaps, native_factor, unit_kind
    if isinstance(m, LineMaps):
        if key not in m.maps:
            why = (" (point-source removal was off)" if key.endswith("_ext") else
                   " (velocity fits were off)" if key.startswith("g") else "")
            raise KeyError(f"no '{key}' map for {m.line.name}{why}; available: {', '.join(sorted(m.maps))}")
        img = np.asarray(m.maps[key], float)
        err = m.maps.get(key + "_err" if key + "_err" in m.maps else "mom0_err")
        k = unit_kind(unit)
        f = native_factor(m.line.wave, unit)
        return (img * f, m.lc.cube.wcs, UNIT_LABELS[k], m.line.name, m.line.wave,
                None if err is None else np.asarray(err, float) * f, m.lc.cube.celestial_header())
    if isinstance(m, (str, os.PathLike)):
        from astropy.io import fits
        with fits.open(m) as h:
            named = [x for x in h if x.name.upper() == key.upper() and x.data is not None]
            hdu = named[0] if named else next((x for x in h if x.data is not None and np.ndim(x.data) >= 2), None)
            if hdu is None:
                raise ValueError(f"{m}: no 2-D image in the file")
            return _as_map(hdu, key, unit, wcs)
    if hasattr(m, "data") and hasattr(m, "header"):
        from astropy.wcs import WCS, FITSFixedWarning
        hdr = m.header
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            w = WCS(hdr).celestial if wcs is None else wcs
        return (np.asarray(m.data, float).squeeze(), w, str(hdr.get("BUNIT", "")), str(hdr.get("LINE", "")),
                float(hdr.get("RESTWAV", np.nan)), None, hdr)
    if hasattr(m, "value") and hasattr(m, "wcs"):          # spectral_cube Projection / Moment0Map
        return np.asarray(m.value, float), m.wcs if wcs is None else wcs, str(getattr(m, "unit", "")), "", np.nan, None, \
            getattr(m, "header", None)
    return np.asarray(m, float), wcs, "", "", np.nan, None, None


def ratio_map(line1, line2, rms_region=None, sigma_thresh=(5.0, 5.0), rms_mode: str = "rms", key: str = "mom0",
              unit: str = "cgs", wcs1=None, wcs2=None, names=None, waves=None, exact_reproject: bool | None = None) -> RatioMap:
    """Ratio map line1 / line2 (see the module docstring).

    line1, line2 : LineMaps, FITS moment-0 files / HDUs, or 2-D arrays (then give wcs1 / wcs2)
    rms_region   : (x, y, r) pixel circle on line 1's grid (the cube_maps.py RMS region), or None to use
                   the propagated errors (LineMaps only)
    sigma_thresh : (sigma for line 1, sigma for line 2)
    key          : which map of LineMaps ("mom0", "mom0_ext", "gflux")
    unit         : for LineMaps: "cgs" (flux ratio) or "MJy/sr um" / "MJy/sr m" (cube_maps.py's numbers)
    """
    a1, w1, u1, n1, l1, e1, h1 = _as_map(line1, key, unit, wcs1)
    a2, w2, u2, n2, l2, e2, _ = _as_map(line2, key, unit, wcs2)
    if names is not None:
        n1, n2 = names
    if waves is not None:
        l1, l2 = waves
    s1, s2 = (float(sigma_thresh), float(sigma_thresh)) if np.isscalar(sigma_thresh) else (float(sigma_thresh[0]), float(sigma_thresh[1]))
    reproj = False
    if w1 is not None and w2 is not None and not _same_grid(w1, w2, a1.shape, a2.shape):
        a2 = reproject_image(a2, w2, w1, a1.shape, exact_reproject)
        if e2 is not None:
            e2 = reproject_image(e2, w2, w1, a1.shape, exact_reproject)
        reproj = True
    elif a1.shape != a2.shape:
        raise ValueError(f"maps of different shapes {a1.shape} / {a2.shape} and no WCS to reproject with")
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        if rms_region is not None:
            cx, cy, rr = (float(v) for v in rms_region)
            check_circle(a1.shape, cx, cy, rr)
            n_1 = region_level(a1, cx, cy, rr, rms_mode); n_2 = region_level(a2, cx, cy, rr, rms_mode)
            t1, t2 = s1 * n_1, s2 * n_2
            m1 = np.where(a1 >= t1, a1, np.nan); m2 = np.where(a2 >= t2, a2, np.nan)
            region = (cx, cy, rr)
        else:
            if e1 is None or e2 is None:
                raise ValueError("give rms_region=(x, y, r) for maps without errors (FITS files or arrays)")
            m1 = np.where(a1 >= s1 * e1, a1, np.nan); m2 = np.where(a2 >= s2 * e2, a2, np.nan)
            n_1 = float(np.nanmedian(e1)); n_2 = float(np.nanmedian(e2)); t1, t2 = s1 * n_1, s2 * n_2
            region = None
        ok = np.isfinite(m1) & np.isfinite(m2) & (m2 > 0)
        r = np.full(a1.shape, np.nan)
        r[ok] = m1[ok] / m2[ok]
    from .maps import LineMaps
    meta = {"unit_arg": unit, "key": key}
    if isinstance(line1, LineMaps) and isinstance(line2, LineMaps):
        meta["tags"] = (line1.line.tag, line2.line.tag)
    return RatioMap((n1, n2), (l1, l2), a1, a2, m1, m2, r, w1, u1 or u2, (float(t1), float(t2)), (float(n_1), float(n_2)),
                    (s1, s2), region, rms_mode, reproj, h1, meta)


def ratio_from_cubes(cubes, line1, line2, prepare_kw: dict | None = None, maps_kw: dict | None = None, **ratio_kw) -> RatioMap:
    """Prepare both lines, make their moment-0 maps (no velocity fits) and return ratio_map(...)."""
    from .maps import line_maps, prepare_line
    pk = dict(prepare_kw or {}); mk = dict(maps_kw or {}); mk["kinematics"] = False
    m1 = line_maps(prepare_line(cubes, line1, **pk), **mk)
    m2 = line_maps(prepare_line(cubes, line2, **pk), **mk)
    rm = ratio_map(m1, m2, **ratio_kw)
    rm.meta["maps"] = (m1, m2)
    return rm
