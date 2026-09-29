"""Reading JWST MIRI-MRS / NIRSpec-IFU `_s3d.fits` cubes, and sets of them.

`Cube`     one data cube (SCI, ERR, DQ) with its celestial WCS and wavelength axis.
`CubeSet`  all cubes of a target (a folder, a list of files or `example:HV_Tau_C_cube`); picks the cube
           that covers a line, reading headers only until data are needed.

Surface brightness stays in the pipeline unit (MJy/sr); `Cube.pixar_sr` converts to flux per spaxel.
"""
from __future__ import annotations

import glob
import os
import re
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from ..data import mrs_psf_fwhm, parse_radec
from ..lines import C_KMS, Line, get_line

_BAND_RE = re.compile(r"ch(\d)-(short|medium|long)", re.I)
_LETTER = {"short": "A", "medium": "B", "long": "C"}
CUBE_PATTERNS = ("*s3d*.fits", "*s3d*.fits.gz", "*_cube*.fits")


def _wave_axis(h, hd) -> np.ndarray:
    """Wavelengths (micron) of the planes: linear header (CTYPE3 = WAVE) or the jwst WAVE-TAB table."""
    nz = hd["NAXIS3"]
    ctype = str(hd.get("CTYPE3", "WAVE")).upper()
    if "TAB" in ctype:
        for ext in h:
            if ext.name.upper().startswith("WCS-TABLE"):
                col = ext.columns.names[0]
                return np.asarray(ext.data[col], float).ravel()[:nz]
    w = hd["CRVAL3"] + hd["CDELT3"] * (np.arange(nz) + 1 - hd.get("CRPIX3", 1.0))
    unit = str(hd.get("CUNIT3", "um")).strip().lower()
    if unit in ("m",):
        w = w * 1e6
    elif unit in ("nm",):
        w = w * 1e-3
    elif unit in ("angstrom", "a"):
        w = w * 1e-4
    return np.asarray(w, float)


def _band_label(hdr0, path: str) -> str:
    ch = str(hdr0.get("CHANNEL", "")).strip()
    bd = str(hdr0.get("BAND", "")).strip().lower()
    if ch.isdigit() and bd in _LETTER:
        return f"{ch}{_LETTER[bd]}"
    m = _BAND_RE.search(os.path.basename(path))
    if m:
        return f"{m.group(1)}{_LETTER[m.group(2).lower()]}"
    g = str(hdr0.get("GRATING", "")).strip()
    return g or "cube"


# MRS sub-band chosen for a line by fixed wavelength boundaries (the `get_channel` rule of the old
# cube_maps.py): each boundary is the short-wavelength edge of the next sub-band, so a line in an overlap
# goes to the longer-wavelength sub-band.  `CubeSet.for_line(band="nominal")` uses it.
MRS_CHANNEL_EDGES_UM = (5.66, 6.53, 7.51, 8.67, 10.01, 11.55, 13.34, 15.41, 17.70, 20.69, 24.40)
MRS_CHANNELS = tuple(f"ch{c}-{b}" for c in (1, 2, 3, 4) for b in ("short", "medium", "long"))


def get_channel(line_wave: float) -> str:
    """MRS sub-band of a line, as in cube_maps.py: 'ch1-short' (< 5.66 um) ... 'ch4-long' (> 24.40 um).

    Half-open intervals: a wavelength exactly on a boundary belongs to the next sub-band (the old code's
    strict inequalities sent it to 'ch4-long')."""
    i = int(np.searchsorted(MRS_CHANNEL_EDGES_UM, float(line_wave), side="right"))
    return MRS_CHANNELS[min(i, len(MRS_CHANNELS) - 1)]


def channel_band(name: str) -> str:
    """'ch1-short' / '1SHORT' / '1A' -> jalebi's band label '1A'."""
    s = str(name).strip().lower().replace("_", "-")
    m = _BAND_RE.search(s)
    if m:
        return f"{m.group(1)}{_LETTER[m.group(2)]}"
    m = re.fullmatch(r"(\d)\s*-?\s*(short|medium|long|a|b|c)", s)
    if m:
        b = m.group(2)
        return f"{m.group(1)}{_LETTER.get(b, b.upper())}"
    return str(name).strip()


def band_channel(band: str) -> str:
    """'1A' -> 'ch1-short' (the naming of the jwst pipeline's Level3_ch1-short_s3d.fits)."""
    inv = {v: k for k, v in _LETTER.items()}
    b = channel_band(band)
    if len(b) == 2 and b[0].isdigit() and b[1] in inv:
        return f"ch{b[0]}-{inv[b[1]]}"
    return band


@dataclass
class Cube:
    """One IFU cube.  `sci` and `err` are (nz, ny, nx) float32 arrays in MJy/sr, NaN where undefined."""
    sci: np.ndarray
    err: np.ndarray
    wave: np.ndarray
    wcs: object                      # astropy.wcs.WCS, celestial (2-D)
    header: object = None            # SCI header
    primary: object = None           # primary header
    band: str = ""
    path: str = ""
    name: str = "target"
    meta: dict = field(default_factory=dict)

    # ---- geometry ---------------------------------------------------------------------------
    @property
    def shape(self):
        return self.sci.shape

    @property
    def pixscale(self) -> float:
        """Spaxel size (arcsec)."""
        from astropy.wcs.utils import proj_plane_pixel_scales
        return float(np.sqrt(np.prod(proj_plane_pixel_scales(self.wcs)))) * 3600.0

    @property
    def pixar_sr(self) -> float:
        v = None if self.header is None else self.header.get("PIXAR_SR")
        return float(v) if v else (self.pixscale / 206264.806) ** 2

    @property
    def target_radec(self) -> tuple[float, float] | None:
        p = self.primary or {}
        try:
            return float(p["TARG_RA"]), float(p["TARG_DEC"])
        except (KeyError, TypeError, ValueError):
            return None

    @property
    def north_up(self) -> bool:
        """True for the default 'skyalign' cubes (north up, east left, no rotation)."""
        pc = self.wcs.wcs.get_pc()
        return abs(pc[0, 1]) < 1e-6 and abs(pc[1, 0]) < 1e-6 and self.wcs.wcs.cdelt[0] * pc[0, 0] < 0

    def world_to_pix(self, ra, dec):
        x, y = self.wcs.all_world2pix(np.asarray(ra, float), np.asarray(dec, float), 0)
        return x, y

    def pix_to_world(self, x, y):
        ra, dec = self.wcs.all_pix2world(np.asarray(x, float), np.asarray(y, float), 0)
        return ra, dec

    def offsets(self, center_radec, x=None, y=None):
        """East and north offsets (arcsec) of pixel positions (default: every spaxel centre) from a sky
        position, in the tangent plane at that position."""
        ny, nx = self.shape[1:]
        if x is None:
            y, x = np.mgrid[0:ny, 0:nx].astype(float)
        ra, dec = self.pix_to_world(x, y)
        ra0, dec0 = center_radec
        dra = ((ra - ra0 + 180.0) % 360.0) - 180.0
        return dra * np.cos(np.deg2rad(dec0)) * 3600.0, (dec - dec0) * 3600.0

    def psf_fwhm_arcsec(self, wave=None):
        return mrs_psf_fwhm(self.wave if wave is None else wave)

    # ---- data access ------------------------------------------------------------------------
    def index_range(self, wmin: float, wmax: float, rule: str = "inside") -> tuple[int, int]:
        """Plane indices [i0, i1) of a wavelength range.  rule="inside": planes with wmin <= wave <= wmax;
        rule="closest": from the plane nearest to wmin to the plane nearest to wmax, both included (what
        spectral_cube's `spectral_slab` selects, used by the cube_maps recipe)."""
        if rule == "closest":
            i0 = int(np.argmin(np.abs(self.wave - wmin)))
            i1 = int(np.argmin(np.abs(self.wave - wmax))) + 1
            return min(i0, i1 - 1), i1
        i0 = int(np.searchsorted(self.wave, wmin, side="left"))
        i1 = int(np.searchsorted(self.wave, wmax, side="right"))
        return max(i0, 0), min(i1, len(self.wave))

    def slab(self, wmin: float, wmax: float, rule: str = "inside") -> "Cube":
        """Copy of the planes with wmin <= wave <= wmax (rule="closest": see `index_range`)."""
        i0, i1 = self.index_range(wmin, wmax, rule)
        return Cube(self.sci[i0:i1].copy(), self.err[i0:i1].copy(), self.wave[i0:i1].copy(), self.wcs,
                    self.header, self.primary, self.band, self.path, self.name, dict(self.meta, slab=(float(wmin), float(wmax))))

    def image(self, wmin: float | None = None, wmax: float | None = None, stat: str = "median") -> np.ndarray:
        """Collapsed image (MJy/sr) over a wavelength range (default: the whole cube)."""
        i0, i1 = self.index_range(self.wave[0] if wmin is None else wmin, self.wave[-1] if wmax is None else wmax)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sub = self.sci[i0:i1]
            return np.nanmean(sub, axis=0) if stat == "mean" else np.nanmedian(sub, axis=0)

    def covers(self, wave_um: float, margin_um: float = 0.0) -> bool:
        return self.wave[0] + margin_um <= wave_um <= self.wave[-1] - margin_um

    def find_source(self, near=None, search_arcsec: float = 1.5, wmin=None, wmax=None, box: int = 2):
        """Continuum peak closest to `near` (RA, Dec; default the header target position): the brightest
        spaxel of the median image within `search_arcsec`, refined by a background-subtracted centroid
        in a (2 box + 1)^2 window.  Keeps a brighter neighbour (e.g. HV Tau AB next to HV Tau C) from
        being taken for the target.  Returns (ra, dec, x, y)."""
        img = self.image(wmin, wmax)
        ny, nx = img.shape
        near = near if near is not None else self.target_radec
        good = np.isfinite(img)
        if near is not None:
            dx, dy = self.offsets(near)
            sel = good & (np.hypot(dx, dy) <= search_arcsec)
            if sel.sum() >= 1:
                good = sel
        if not good.any():
            raise ValueError("no finite pixels to locate the source")
        work = np.where(good, img, -np.inf)
        iy, ix = np.unravel_index(int(np.argmax(work)), img.shape)
        xc, yc = _refine_peak(img, ix, iy, max(box, 3))
        ra, dec = self.pix_to_world(xc, yc)
        return float(ra), float(dec), xc, yc

    # ---- output -----------------------------------------------------------------------------
    def celestial_header(self):
        """A 2-D FITS header with the celestial WCS (for maps written from this cube)."""
        hdr = self.wcs.to_header()
        for k in ("TARGNAME", "TARG_RA", "TARG_DEC", "PROGRAM", "OBSERVTN", "INSTRUME", "CHANNEL", "BAND", "DATE-OBS", "CAL_VER"):
            if self.primary is not None and k in self.primary:
                hdr[k] = self.primary[k]
        hdr["PIXAR_SR"] = (self.pixar_sr, "spaxel solid angle [sr]")
        return hdr

    def write(self, path: str, overwrite: bool = True):
        """Write as a jwst-like `_s3d.fits` (PRIMARY + SCI + ERR + DQ), linear wavelength axis."""
        from astropy.io import fits
        prim = fits.PrimaryHDU(header=self.primary.copy() if self.primary is not None else None)
        hdr = self.wcs.to_header()
        dw = np.diff(self.wave)
        if len(dw) and not np.allclose(dw, dw[0], rtol=1e-5):
            warnings.warn("non-uniform wavelength axis written with its mean step")
        hdr["WCSAXES"] = 3
        hdr["CTYPE3"] = "WAVE"; hdr["CUNIT3"] = "um"; hdr["CRPIX3"] = 1.0
        hdr["CRVAL3"] = float(self.wave[0]); hdr["CDELT3"] = float(np.mean(dw)) if len(dw) else 1.0
        hdr["PC3_3"] = 1.0
        hdr["BUNIT"] = "MJy/sr"
        hdr["PIXAR_SR"] = self.pixar_sr
        if self.header is not None:
            for k in ("PIXAR_A2", "SRCTYPE", "RADESYS"):
                if k in self.header:
                    hdr[k] = self.header[k]
        sci = fits.ImageHDU(self.sci.astype(np.float32), header=hdr, name="SCI")
        err = fits.ImageHDU(self.err.astype(np.float32), header=hdr, name="ERR")
        dq = fits.ImageHDU((~np.isfinite(self.sci)).astype(np.int32), header=hdr, name="DQ")
        fits.HDUList([prim, sci, err, dq]).writeto(path, overwrite=overwrite)
        return path


def _refine_peak(img: np.ndarray, ix: int, iy: int, box: int = 3) -> tuple[float, float]:
    """Sub-pixel position of a peak: 2-D Gaussian + constant fitted in a (2 box + 1)^2 window
    (falls back to the background-subtracted centroid)."""
    ny, nx = img.shape
    y0, y1, x0, x1 = max(iy - box, 0), min(iy + box + 1, ny), max(ix - box, 0), min(ix + box + 1, nx)
    sub = img[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(float)
    ok = np.isfinite(sub)
    bg = float(np.nanmin(sub)) if ok.any() else 0.0
    s = np.clip(np.nan_to_num(sub - bg), 0.0, None)
    xc, yc = (float((s * xx).sum() / s.sum()), float((s * yy).sum() / s.sum())) if s.sum() > 0 else (float(ix), float(iy))
    if ok.sum() < 8:
        return xc, yc
    from scipy.optimize import least_squares

    def resid(p):
        a, x, y, sg, c = p
        return (a * np.exp(-0.5 * ((xx[ok] - x) ** 2 + (yy[ok] - y) ** 2) / sg ** 2) + c - sub[ok]) / max(np.nanmax(sub[ok]), 1e-30)
    try:
        r = least_squares(resid, [float(np.nanmax(sub)) - bg, xc, yc, 1.5, bg],
                          bounds=([0, x0 - 0.5, y0 - 0.5, 0.3, -np.inf], [np.inf, x1 - 0.5, y1 - 0.5, 3.0 * box, np.inf]))
        if r.success and abs(r.x[1] - ix) < box and abs(r.x[2] - iy) < box:
            return float(r.x[1]), float(r.x[2])
    except Exception:
        pass
    return xc, yc


def read_cube(path: str, wmin: float | None = None, wmax: float | None = None, dq_mask: bool = True,
              zero_is_nan: bool = True) -> Cube:
    """Read one `_s3d.fits` cube (optionally only the planes in [wmin, wmax] micron).

    dq_mask     : blank DO_NOT_USE pixels of the DQ extension (spectral_cube, and so cube_maps.py, does not)
    zero_is_nan : treat SCI == 0 as undefined (some jwst versions fill uncovered spaxels with 0)"""
    from astropy.io import fits
    from astropy.wcs import WCS
    from ..examples import resolve_path
    path = resolve_path(path)
    # memmap=False: the jwst DQ extension is uint32 stored with BZERO, which cannot be memory-mapped
    with fits.open(path, memmap=False) as h, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hd = h["SCI"].header
        hdr0 = h[0].header
        wave = _wave_axis(h, hd)
        try:
            wcs = WCS(hd, fobj=h).celestial
        except Exception:
            wcs = WCS(hd, naxis=2)
        i0, i1 = 0, len(wave)
        if wmin is not None or wmax is not None:
            i0 = int(np.searchsorted(wave, -np.inf if wmin is None else wmin, side="left"))
            i1 = int(np.searchsorted(wave, np.inf if wmax is None else wmax, side="right"))
        sci = np.array(h["SCI"].data[i0:i1], dtype=np.float32)
        err = (np.array(h["ERR"].data[i0:i1], dtype=np.float32) if "ERR" in h else np.full_like(sci, np.nan))
        if dq_mask and "DQ" in h:
            dq = np.asarray(h["DQ"].data[i0:i1]).astype(np.int64)
            sci[(dq & 1) != 0] = np.nan              # DO_NOT_USE
        if zero_is_nan:
            sci[sci == 0] = np.nan                   # jwst fills uncovered spaxels with 0 in some versions
        err[~np.isfinite(sci)] = np.nan
        name = str(hdr0.get("TARGNAME") or hdr0.get("TARGPROP") or os.path.basename(os.path.dirname(os.path.abspath(path))))
        meta = {k: hdr0.get(k) for k in ("TARGNAME", "TARGPROP", "PROGRAM", "OBSERVTN", "DATE-OBS", "CAL_VER", "CRDS_CTX",
                                          "INSTRUME", "PATTTYPE") if hdr0.get(k) is not None}
        return Cube(sci, err, wave[i0:i1], wcs, hd.copy(), hdr0.copy(), _band_label(hdr0, path), path, name, meta)


@dataclass
class CubeInfo:
    path: str
    band: str
    wmin: float
    wmax: float
    nz: int
    ny: int
    nx: int


def _cube_info(path: str) -> CubeInfo:
    from astropy.io import fits
    with fits.open(path, memmap=True) as h:
        hd = h["SCI"].header
        w = _wave_axis(h, hd)
        return CubeInfo(path, _band_label(h[0].header, path), float(w[0]), float(w[-1]), hd["NAXIS3"], hd["NAXIS2"], hd["NAXIS1"])


class CubeSet:
    """The cubes of one target.  `for_line(line)` returns the cube (sliced to the line window) of the
    sub-band that covers the line with the widest margin; data are read on demand and the last few
    cubes are kept in memory."""

    def __init__(self, paths, name: str | None = None, cache: int = 3, dq_mask: bool = True, zero_is_nan: bool = True):
        from ..examples import resolve_path
        self._read_kw = dict(dq_mask=dq_mask, zero_is_nan=zero_is_nan)
        if isinstance(paths, (str, os.PathLike)):
            p = resolve_path(paths)
            if os.path.isdir(p):
                files = sorted({f for pat in CUBE_PATTERNS for f in glob.glob(os.path.join(p, pat))})
            elif any(ch in p for ch in "*?["):
                files = sorted(glob.glob(p))
            else:
                files = [p]
            self.root = p
        else:
            files = [resolve_path(f) for f in paths]
            self.root = os.path.dirname(files[0]) if files else ""
        if not files:
            raise FileNotFoundError(f"no s3d cubes found in {paths}")
        self.info = [_cube_info(f) for f in files]
        self.info.sort(key=lambda c: c.wmin)
        self._cache: OrderedDict[str, Cube] = OrderedDict()
        self._ncache = cache
        self._name = name

    # ---- basic facts ------------------------------------------------------------------------
    @property
    def files(self) -> list[str]:
        return [c.path for c in self.info]

    @property
    def bands(self) -> list[str]:
        return [c.band for c in self.info]

    @property
    def wave_range(self) -> tuple[float, float]:
        return min(c.wmin for c in self.info), max(c.wmax for c in self.info)

    @property
    def name(self) -> str:
        if self._name:
            return self._name
        from astropy.io import fits
        h = fits.getheader(self.info[0].path, 0)
        return str(h.get("TARGNAME") or h.get("TARGPROP") or os.path.basename(self.root.rstrip("/")))

    def table(self):
        import pandas as pd
        return pd.DataFrame([{"band": c.band, "wmin_um": round(c.wmin, 4), "wmax_um": round(c.wmax, 4), "planes": c.nz,
                              "ny": c.ny, "nx": c.nx, "file": os.path.basename(c.path)} for c in self.info])

    def covering(self, wave_um: float, margin_kms: float = 0.0) -> list[CubeInfo]:
        m = wave_um * margin_kms / C_KMS
        return [c for c in self.info if c.wmin + m <= wave_um <= c.wmax - m]

    def best_for(self, wave_um: float, margin_kms: float = 0.0) -> CubeInfo | None:
        cands = self.covering(wave_um, margin_kms) or self.covering(wave_um, 0.0)
        if not cands:
            return None
        return max(cands, key=lambda c: min(wave_um - c.wmin, c.wmax - wave_um) / (c.wmax - c.wmin))

    def lines_covered(self, lines=None, margin_kms: float = 300.0) -> list[Line]:
        from ..lines import LINES
        cands = lines if lines is not None else LINES.values()
        return [ln for ln in (get_line(x) for x in cands) if self.covering(ln.wave, margin_kms)]

    # ---- data ---------------------------------------------------------------------------------
    def load(self, path: str) -> Cube:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        c = read_cube(path, **self._read_kw)
        if self._name:
            c.name = self._name
        self._cache[path] = c
        while len(self._cache) > self._ncache:
            self._cache.popitem(last=False)
        return c

    def cube(self, band: str) -> Cube:
        for c in self.info:
            if c.band == band:
                return self.load(c.path)
        raise KeyError(f"no cube for band {band}; have {self.bands}")

    def choose(self, wave_um: float, band: str | None = None, margin_kms: float = 0.0) -> CubeInfo | None:
        """The cube for a wavelength.  band=None: the sub-band with the widest margin; "nominal": the fixed
        boundaries of `get_channel` (the cube_maps.py rule; falls back to the widest margin when that
        sub-band is not in the set); or a sub-band name ("3A", "ch3-short")."""
        if band is None or str(band).lower() in ("auto", "margin", ""):
            return self.best_for(wave_um, margin_kms)
        want = channel_band(get_channel(wave_um) if str(band).lower() == "nominal" else band)
        cands = [c for c in self.info if c.band == want and c.wmin <= wave_um <= c.wmax]
        if cands:                              # several cubes of one sub-band (cutouts): the widest margin
            return max(cands, key=lambda c: min(wave_um - c.wmin, c.wmax - wave_um))
        if str(band).lower() == "nominal":
            got = self.best_for(wave_um, margin_kms)
            if got is not None:
                warnings.warn(f"no {band_channel(want)} cube for {wave_um:.4f} um; using {got.band}")
            return got
        raise ValueError(f"no cube of band {band} covering {wave_um:.4f} um; have {', '.join(self.bands)}")

    def for_line(self, line, rv_kms: float = 0.0, window_kms: float = 1500.0, band: str | None = None,
                 window_um: float | None = None) -> Cube:
        """Cube slab around a line (observed wavelength = rest x (1 + rv/c)), +-window_kms wide, or
        +-window_um micron with the planes chosen like spectral_cube's `spectral_slab` (the cube_maps
        `dlambda`).  `band`: see `choose`."""
        ln = get_line(line)
        lam = ln.wave * (1 + rv_kms / C_KMS)
        ci = self.choose(lam, band, window_kms if window_um is None else 0.0)
        if ci is None:
            raise ValueError(f"{ln.name} at {lam:.4f} um is not covered by these cubes ({', '.join(self.bands)})")
        c = self.load(ci.path)
        if window_um is not None:
            return c.slab(lam - window_um, lam + window_um, rule="closest")
        dl = lam * window_kms / C_KMS
        return c.slab(lam - dl, lam + dl)

    def source_position(self, near=None, search_arcsec: float = 1.5, band: str | None = None):
        """(RA, Dec) of the continuum peak nearest to the header target (see Cube.find_source), measured
        in `band` (default: the shortest-wavelength cube, sharpest PSF)."""
        c = self.cube(band) if band else self.load(self.info[0].path)
        ra, dec, _, _ = c.find_source(near=near, search_arcsec=search_arcsec)
        return ra, dec


def resolve_center(cubes: CubeSet | Cube, center=None, search_arcsec: float = 1.5):
    """Centre (RA, Dec) in degrees from a config value: None/"auto" (continuum peak near the header target),
    "header" (TARG_RA/TARG_DEC), "peak" (brightest spaxel anywhere), or (ra, dec) in degrees or sexagesimal."""
    c0 = cubes.load(cubes.info[0].path) if isinstance(cubes, CubeSet) else cubes
    if center is None or (isinstance(center, str) and center.lower() == "auto"):
        ra, dec, _, _ = c0.find_source(search_arcsec=search_arcsec)
        return ra, dec
    if isinstance(center, str) and center.lower() == "header":
        t = c0.target_radec
        if t is None:
            raise ValueError("no TARG_RA/TARG_DEC in the header")
        return t
    if isinstance(center, str) and center.lower() == "peak":
        ra, dec, _, _ = c0.find_source(near=None, search_arcsec=1e9) if c0.target_radec is None else \
            c0.find_source(near=c0.target_radec, search_arcsec=1e9)
        return ra, dec
    if isinstance(center, dict):
        return parse_radec(center.get("ra"), center.get("dec"))
    ra, dec = center
    return parse_radec(ra, dec)


def write_cutouts(cubes: CubeSet, lines, outdir: str, window_kms: float = 1500.0, rv_kms: float = 0.0,
                  compress: bool = False) -> list[str]:
    """Small cubes around each line (to share, or for the bundled example), named after the lines they hold,
    e.g. `Level3_ch3-short_s3d_NeII_12.81.fits`.  Lines in one sub-band closer than 2 x window_kms share a
    cutout.  `compress=True` writes `.fits.gz` (read transparently by JALEBI, astropy, DS9 and CARTA)."""
    os.makedirs(outdir, exist_ok=True)
    groups: dict[str, list[tuple[float, str]]] = {}
    for ln in (get_line(x) for x in lines):
        lam = ln.wave * (1 + rv_kms / C_KMS)
        ci = cubes.best_for(lam, window_kms)
        if ci is not None:
            groups.setdefault(ci.path, []).append((lam, ln.tag))
    out = []
    for path, items in groups.items():
        c = cubes.load(path)
        items = sorted(items)
        segs = [[items[0][0], items[0][0], [items[0][1]]]]
        for lam, tag in items[1:]:
            if (lam - segs[-1][1]) / lam * C_KMS < 2 * window_kms:
                segs[-1][1] = lam; segs[-1][2].append(tag)
            else:
                segs.append([lam, lam, [tag]])
        for a, b, tags in segs:
            dl = a * window_kms / C_KMS
            s = c.slab(a - dl, b + dl)
            base = re.sub(r"_s3d.*$", "", os.path.basename(path)) + "_s3d_" + "+".join(tags) + ".fits"
            p = s.write(os.path.join(outdir, base))
            if compress:
                import gzip
                import shutil
                with open(p, "rb") as fi, gzip.open(p + ".gz", "wb", compresslevel=9) as fo:
                    shutil.copyfileobj(fi, fo)
                os.remove(p); p = p + ".gz"
            out.append(p)
    return out
