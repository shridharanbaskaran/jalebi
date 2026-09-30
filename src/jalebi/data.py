"""Spectra: ingest, rest-frame shift, spike filtering, masks, noise.

A Spectrum keeps every pixel of every sub-band in flat arrays with a `band` label, so the
model can use one instrument operator per sub-band and overlaps between sub-bands are kept
(not stitched) for fitting; `stitched()` gives a display-friendly merged spectrum.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .constants import C
from .instrument import _BANDS, band_of

BAND_ORDER = ["1A", "1B", "1C", "2A", "2B", "2C", "3A", "3B", "3C", "4A", "4B", "4C"]
_X1D_RE = re.compile(r"ch(\d)-(short|medium|long)", re.I)
_BAND_LETTER = {"short": "A", "medium": "B", "long": "C"}


@dataclass
class Spectrum:
    wave: np.ndarray            # micron, observed frame unless rest_frame=True
    flux: np.ndarray            # Jy
    err: np.ndarray             # Jy
    band: np.ndarray            # sub-band label per pixel ("1A" ... "4C" or "")
    name: str = "target"
    distance_pc: float = 140.0
    rv_kms: float = 0.0         # radial velocity applied (positive = redshift removed)
    rest_frame: bool = False
    continuum: np.ndarray | None = None       # Jy, same shape as flux
    mask: np.ndarray | None = None            # True = use pixel
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.wave)
        if self.mask is None:
            self.mask = np.isfinite(self.flux) & np.isfinite(self.err) & (self.err > 0)
        if self.continuum is None:
            self.continuum = np.zeros(n)

    # ---- derived -----------------------------------------------------------------
    @property
    def line_flux(self) -> np.ndarray:
        """Continuum-subtracted flux (Jy)."""
        return self.flux - self.continuum

    @property
    def bands(self) -> list[str]:
        return [b for b in BAND_ORDER if (self.band == b).any()] + sorted(set(self.band) - set(BAND_ORDER) - {""})

    def band_slice(self, b: str) -> np.ndarray:
        return np.flatnonzero(self.band == b)

    def to_rest_frame(self, rv_kms: float) -> "Spectrum":
        """Shift wavelengths to the rest frame for a source moving at rv (heliocentric km/s)."""
        s = self.copy()
        if self.rest_frame:
            s.wave = self.wave * (1 + self.rv_kms * 1e3 / C)   # undo previous shift
        s.wave = s.wave / (1 + rv_kms * 1e3 / C)
        s.rv_kms = rv_kms
        s.rest_frame = True
        return s

    def copy(self) -> "Spectrum":
        return Spectrum(self.wave.copy(), self.flux.copy(), self.err.copy(), self.band.copy(), self.name,
                        self.distance_pc, self.rv_kms, self.rest_frame,
                        None if self.continuum is None else self.continuum.copy(),
                        None if self.mask is None else self.mask.copy(), dict(self.meta))

    def select(self, wmin: float, wmax: float) -> np.ndarray:
        return (self.wave >= wmin) & (self.wave <= wmax)

    def stitched(self, overlap: str = "mean") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Merged spectrum for display: in sub-band overlaps keep the bluer band's pixels
        below the overlap midpoint and the redder band's above it."""
        keep = np.ones(len(self.wave), bool)
        bands = self.bands
        for b1, b2 in zip(bands[:-1], bands[1:]):
            i1, i2 = self.band_slice(b1), self.band_slice(b2)
            if len(i1) == 0 or len(i2) == 0:
                continue
            lo, hi = self.wave[i2].min(), self.wave[i1].max()
            if hi > lo:
                mid = 0.5 * (lo + hi)
                keep[i1[self.wave[i1] > mid]] = False
                keep[i2[self.wave[i2] <= mid]] = False
        o = np.argsort(self.wave[keep])
        return self.wave[keep][o], self.flux[keep][o], self.err[keep][o]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({"wave": self.wave, "flux": self.flux, "err": self.err, "band": self.band,
                             "continuum": self.continuum, "mask": self.mask})

    def save(self, path: str):
        """Write to CSV (with the continuum and mask) or HDF5 (`.h5`)."""
        df = self.to_dataframe()
        if path.endswith(".h5") or path.endswith(".hdf5"):
            df.to_hdf(path, key="spectrum", mode="w")
            pd.Series({**self.meta, "name": self.name, "distance_pc": self.distance_pc, "rv_kms": self.rv_kms,
                       "rest_frame": self.rest_frame}).to_hdf(path, key="meta")
        else:
            with open(path, "w") as fh:
                fh.write(f"# name={self.name} distance_pc={self.distance_pc} rv_kms={self.rv_kms} rest_frame={self.rest_frame}\n")
                df.to_csv(fh, index=False)


# ----------------------------------------------------------------------------------
# Readers
# ----------------------------------------------------------------------------------

def read_x1d(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, dict]:
    """Read one jwst pipeline Level-3 `_x1d.fits` product."""
    from astropy.io import fits
    with fits.open(path) as h:
        d = h[1].data
        hdr0 = h[0].header
        wave = np.array(d["WAVELENGTH"], float)
        flux = np.array(d["FLUX"], float)
        err = np.array(d["FLUX_ERROR"], float)
        unit = h[1].header.get("TUNIT2", "Jy")
        if unit.lower() in ("mjy",):
            flux *= 1e-3; err *= 1e-3
        ch = str(hdr0.get("CHANNEL", "")).strip()
        bd = str(hdr0.get("BAND", "")).strip().lower()
        m = _X1D_RE.search(os.path.basename(path))
        if ch and bd in _BAND_LETTER:
            band = f"{ch}{_BAND_LETTER[bd]}"
        elif m:
            band = f"{m.group(1)}{_BAND_LETTER[m.group(2).lower()]}"
        else:
            band = ""
        meta = {k: hdr0.get(k) for k in ("TARGNAME", "TARG_RA", "TARG_DEC", "PROGRAM", "CAL_VER", "CRDS_CTX", "DATE-OBS")}
    return wave, flux, err, band, meta


def load_x1d_folder(folder: str, name: str | None = None, distance_pc: float = 140.0,
                    pattern: str = "*x1d.fits") -> Spectrum:
    """Load all sub-band x1d files of one target folder into one Spectrum."""
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if not files:
        raise FileNotFoundError(f"No {pattern} files in {folder}")
    W, F, E, B = [], [], [], []
    meta = {}
    for f in files:
        w, fl, er, b, mt = read_x1d(f)
        W.append(w); F.append(fl); E.append(er); B.append(np.full(len(w), b, dtype=object))
        meta.update({k: v for k, v in mt.items() if v is not None})
    wave = np.concatenate(W); flux = np.concatenate(F); err = np.concatenate(E); band = np.concatenate(B)
    # order by band then wavelength (keeps overlaps)
    order = np.lexsort((wave, np.array([BAND_ORDER.index(b) if b in BAND_ORDER else 99 for b in band])))
    tname = name or str(meta.get("TARGNAME") or os.path.basename(folder.rstrip("/")))
    s = Spectrum(wave[order], flux[order], err[order], band[order].astype(str), tname, distance_pc, meta=meta)
    # pixels beyond the nominal sub-band ranges are unreliable (no calibration): mask them
    for b, (lo, hi) in _BANDS.items():
        i = s.band_slice(b)
        if len(i):
            s.mask[i[(s.wave[i] < lo - 0.02) | (s.wave[i] > hi + 0.02)]] = False
    return s


# ----------------------------------------------------------------------------------
# s3d cube extraction
# ----------------------------------------------------------------------------------

def parse_radec(ra, dec) -> tuple[float, float]:
    """RA/Dec given as degrees (float) or sexagesimal strings ("04:32:31.76", "+24:20:03.0",
    "04h32m31.76s") -> degrees."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    try:
        return float(ra), float(dec)
    except (TypeError, ValueError):
        pass
    ra_s, dec_s = str(ra).strip(), str(dec).strip()
    unit = (u.hourangle, u.deg) if (":" in ra_s or "h" in ra_s.lower()) else (u.deg, u.deg)
    c = SkyCoord(ra_s, dec_s, unit=unit, frame="icrs")
    return float(c.ra.deg), float(c.dec.deg)


def mrs_psf_fwhm(wave_um):
    """MIRI-MRS PSF FWHM in arcsec (Law et al. 2023): 0.033 lambda + 0.106."""
    return 0.033 * np.asarray(wave_um, float) + 0.106


# Encircled energy of the MRS PSF inside a circular aperture of radius r/FWHM, measured on FZ Tau
# (jwst 1.x, CRDS jwst_1584) by comparing cube apertures with the pipeline's aperture-corrected x1d
# in ch2 and ch4 (agree to 1-2 %).  Used by apcorr="mrs".
MRS_EE_R = np.array([0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0])
MRS_EE = np.array([0.0, 0.35, 0.565, 0.70, 0.78, 0.84, 0.908, 0.94, 0.965, 0.985, 1.0])


def _circular_weights(ny, nx, xc, yc, r, oversample=5):
    """Fractional-pixel weights of a circle of radius r (pixels) centred at (xc, yc), 0-based."""
    o = oversample
    yy, xx = np.mgrid[0:ny * o, 0:nx * o]
    xs = (xx + 0.5) / o - 0.5; ys = (yy + 0.5) / o - 0.5
    inside = ((xs - xc) ** 2 + (ys - yc) ** 2) <= r * r
    return inside.reshape(ny, o, nx, o).mean(axis=(1, 3))


def extract_s3d(path: str, center="wcs", center_radec=None, aperture_fwhm_scale: float = 1.5,
                aperture_arcsec: float | None = None, annulus_arcsec: tuple | None = None,
                apcorr: str = "mrs") -> dict:
    """Aperture photometry of one jwst `_s3d.fits` cube, channel by channel.

    center : "wcs" (default: the target RA/Dec from the header, TARG_RA/TARG_DEC), "peak"
             (brightest pixel of the median-collapsed cube, centroid-refined), or (x, y) 0-based
             pixel coordinates *in this cube*; `center_radec=(ra, dec)` in degrees overrides all of
             these (use it to give your own coordinates, e.g. for one component of a binary).
    aperture : radius = aperture_fwhm_scale * FWHM(lambda) unless aperture_arcsec is given.
    annulus_arcsec : (r_in, r_out) background annulus (median per plane), None = no background.
    apcorr : "mrs" (default) uses the empirical MRS encircled-energy curve (MRS_EE), "gaussian" a
             Gaussian PSF (underestimates the wings), "none" no correction; "x1d" is handled by
             load_s3d_folder (per-band rescaling to the pipeline x1d level).
    Returns dict(wave, flux, err, band, meta, center_pix, center_radec).
    """
    from astropy.io import fits
    from astropy.wcs import WCS
    import warnings
    with fits.open(path) as h, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sci = h["SCI"].data.astype(float); err = h["ERR"].data.astype(float)
        hd = h["SCI"].header; hdr0 = h[0].header
        wcs = WCS(hd)
        nz, ny, nx = sci.shape
        wave = hd["CRVAL3"] + hd["CDELT3"] * (np.arange(nz) + 1 - hd["CRPIX3"])
        pixar_sr = hd.get("PIXAR_SR"); pixscale = abs(hd["CDELT2"]) * 3600.0    # arcsec/pixel
        if pixar_sr is None:
            pixar_sr = (pixscale / 206265.0) ** 2
        # centre
        if center_radec is not None:
            xc, yc, _ = wcs.all_world2pix(center_radec[0], center_radec[1], hd["CRVAL3"], 0)
        elif isinstance(center, str) and center == "wcs":
            xc, yc, _ = wcs.all_world2pix(hdr0["TARG_RA"], hdr0["TARG_DEC"], hd["CRVAL3"], 0)
        elif isinstance(center, str) and center == "peak":
            img = np.nanmedian(sci, axis=0)
            iy, ix = np.unravel_index(np.nanargmax(np.where(np.isfinite(img), img, -np.inf)), img.shape)
            # refine with the flux-weighted centroid in a 5x5 box
            y0, y1, x0, x1 = max(iy - 2, 0), min(iy + 3, ny), max(ix - 2, 0), min(ix + 3, nx)
            sub = np.nan_to_num(img[y0:y1, x0:x1]); sub = np.clip(sub - np.nanmedian(img), 0, None)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            xc, yc = (float((sub * xx).sum() / sub.sum()), float((sub * yy).sum() / sub.sum())) if sub.sum() > 0 else (float(ix), float(iy))
        else:
            xc, yc = float(center[0]), float(center[1])
        ra, dec, _ = wcs.all_pix2world(xc, yc, hd["CRVAL3"], 0)
        # per-plane aperture
        fwhm = mrs_psf_fwhm(wave)
        r_as = np.full(nz, aperture_arcsec) if aperture_arcsec else aperture_fwhm_scale * fwhm
        r_pix = r_as / pixscale
        flux = np.full(nz, np.nan); ferr = np.full(nz, np.nan)
        # cache weights per distinct radius (rounded) to save time
        cache = {}
        for k in range(nz):
            key = round(float(r_pix[k]), 2)
            if key not in cache:
                cache[key] = _circular_weights(ny, nx, xc, yc, key)
            w = cache[key]
            plane = sci[k]; eplane = err[k]
            good = np.isfinite(plane)
            if good.sum() < 5:
                continue
            bg = 0.0
            if annulus_arcsec:
                wi = _circular_weights(ny, nx, xc, yc, annulus_arcsec[0] / pixscale) if round(annulus_arcsec[0] / pixscale, 2) not in cache else cache[round(annulus_arcsec[0] / pixscale, 2)]
                wo = _circular_weights(ny, nx, xc, yc, annulus_arcsec[1] / pixscale)
                ann = (wo - wi) > 0.5
                if (ann & good).sum() >= 5:
                    bg = float(np.nanmedian(plane[ann & good]))
            f = np.nansum(w * np.where(good, plane - bg, 0.0))
            e = np.sqrt(np.nansum((w * np.where(np.isfinite(eplane), eplane, 0.0)) ** 2))
            # undefined (NaN) pixels inside the aperture: scale up by the covered fraction
            cov = np.sum(w * good) / max(np.sum(w), 1e-9)
            if cov < 0.5:
                continue
            flux[k] = f / cov * pixar_sr * 1e6          # MJy/sr * sr -> Jy
            ferr[k] = e / cov * pixar_sr * 1e6
        if apcorr == "gaussian":
            sig = fwhm / 2.3548
            frac = 1.0 - np.exp(-0.5 * (r_as / sig) ** 2)
            flux /= frac; ferr /= frac
        elif apcorr == "mrs":
            frac = np.interp(r_as / fwhm, MRS_EE_R, MRS_EE)
            flux /= frac; ferr /= frac
        ch = str(hdr0.get("CHANNEL", "")).strip(); bd = str(hdr0.get("BAND", "")).strip().lower()
        band = f"{ch}{_BAND_LETTER[bd]}" if ch and bd in _BAND_LETTER else ""
        meta = {k: hdr0.get(k) for k in ("TARGNAME", "TARG_RA", "TARG_DEC", "PROGRAM", "CAL_VER", "CRDS_CTX", "DATE-OBS")}
    return {"wave": wave, "flux": flux, "err": ferr, "band": band, "meta": meta,
            "center_pix": (float(xc), float(yc)), "center_radec": (float(ra), float(dec)), "pixscale": pixscale}


def load_s3d_folder(folder: str, ra, dec, name: str | None = None, distance_pc: float = 140.0,
                    aperture_fwhm_scale: float = 1.5, aperture_arcsec: float | None = None,
                    annulus_arcsec: tuple | None = None, apcorr: str = "mrs", pattern: str = "*s3d.fits",
                    verbose: bool = True) -> Spectrum:
    """Extract a spectrum from all `_s3d.fits` cubes of a target folder with an aperture centred at
    the sky position (ra, dec) — degrees or sexagesimal — in every sub-band (through each cube's WCS).
    Use `cube_positions()` to find the source's coordinates from a cube if you do not know them."""
    if ra is None or dec is None:
        raise ValueError("s3d extraction needs the aperture centre: give ra and dec (degrees or sexagesimal); "
                         "cube_positions(folder) lists the header target position and the brightest pixel")
    radec = parse_radec(ra, dec)
    center = "wcs"
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if not files:
        raise FileNotFoundError(f"No {pattern} files in {folder}")
    if verbose:
        print(f"s3d: aperture centre RA, Dec = {radec[0]:.6f}, {radec[1]:.6f} deg")
    W, F, E, B = [], [], [], []
    meta = {}; centres = {}; scales = {}
    for f in files:
        r = extract_s3d(f, center=center, center_radec=radec, aperture_fwhm_scale=aperture_fwhm_scale,
                        aperture_arcsec=aperture_arcsec, annulus_arcsec=annulus_arcsec,
                        apcorr="none" if apcorr == "x1d" else apcorr)
        if apcorr == "x1d":
            # rescale this band to the pipeline point-source extraction (single sources only)
            x1d = f.replace("_s3d.fits", "_x1d.fits")
            if os.path.exists(x1d):
                w1, f1, _, _, _ = read_x1d(x1d)
                f2 = np.interp(w1, r["wave"], r["flux"])
                ok = np.isfinite(f1) & np.isfinite(f2) & (f2 > 0)
                sc = float(np.nanmedian(f1[ok] / f2[ok])) if ok.sum() > 10 else 1.0
                r["flux"] = r["flux"] * sc; r["err"] = r["err"] * sc; scales[r["band"]] = sc
        W.append(r["wave"]); F.append(r["flux"]); E.append(r["err"]); B.append(np.full(len(r["wave"]), r["band"], dtype=object))
        meta.update({k: v for k, v in r["meta"].items() if v is not None}); centres[r["band"]] = r["center_pix"]
    wave = np.concatenate(W); flux = np.concatenate(F); err = np.concatenate(E); band = np.concatenate(B)
    order = np.lexsort((wave, np.array([BAND_ORDER.index(b) if b in BAND_ORDER else 99 for b in band])))
    tname = name or str(meta.get("TARGNAME") or os.path.basename(folder.rstrip("/")))
    meta["extraction"] = {"source": "s3d", "ra": radec[0], "dec": radec[1], "center_radec": radec,
                          "aperture_fwhm_scale": aperture_fwhm_scale, "aperture_arcsec": aperture_arcsec,
                          "annulus_arcsec": annulus_arcsec, "apcorr": apcorr, "center_pix_per_band": centres,
                          "x1d_scale_per_band": scales}
    s = Spectrum(wave[order], flux[order], err[order], band[order].astype(str), tname, distance_pc, meta=meta)
    for b, (lo, hi) in _BANDS.items():
        i = s.band_slice(b)
        if len(i):
            s.mask[i[(s.wave[i] < lo - 0.02) | (s.wave[i] > hi + 0.02)]] = False
    return s


def cube_positions(folder_or_file: str) -> dict:
    """Header target position and the brightest-pixel position (RA/Dec, degrees) of a cube, to help
    choose the aperture centre.  Takes a cube file or a target folder (first cube found)."""
    f = folder_or_file
    if os.path.isdir(f):
        files = sorted(glob.glob(os.path.join(f, "*s3d.fits")))
        if not files:
            raise FileNotFoundError(f"No s3d cubes in {f}")
        f = next((x for x in files if "ch2-short" in x), files[0])
    hdr = extract_s3d(f, center="wcs", apcorr="none")
    pk = extract_s3d(f, center="peak", apcorr="none")
    return {"file": f, "header_radec": hdr["center_radec"], "header_pix": hdr["center_pix"],
            "peak_radec": pk["center_radec"], "peak_pix": pk["center_pix"], "pixscale_arcsec": hdr["pixscale"]}


def load_csv(path: str, name: str | None = None, distance_pc: float = 140.0, wave_col=None, flux_col=None,
             err_col=None) -> Spectrum:
    """Load a CSV / whitespace table.  Recognises the column names used in your notebooks
    (`wavelength`, `flux`, `Flux_err`, `flux_lines`), iSLAT (`wave`, `flux`, `err`) and JDISCS.
    A `continuum` / `baseline` column is loaded as the continuum (kept by continuum method "given")."""
    meta = {}
    if str(path).endswith(".gz"):                 # gzipped CSV (pandas reads it directly)
        import gzip
        with gzip.open(path, "rt") as fh:
            first = fh.readline()
    else:
        with open(path) as fh:
            first = fh.readline()
    if first.startswith("#"):
        for kv in first[1:].split():
            if "=" in kv:
                k, v = kv.split("=", 1); meta[k] = v
    df = pd.read_csv(path, comment="#", sep=None, engine="python")
    cols = {c.lower(): c for c in df.columns}
    def pick(cands, given):
        if given:
            return given
        for c in cands:
            if c in cols:
                return cols[c]
        return None
    wc = pick(["wave", "wavelength", "wl", "lambda", "wave_um"], wave_col)
    fc = pick(["flux", "flux_jy", "fnu"], flux_col)
    ec = pick(["err", "flux_err", "error", "flux_error", "sigma", "unc"], err_col)
    if wc is None or fc is None:
        raise ValueError(f"Cannot identify wavelength/flux columns in {path}: {list(df.columns)}")
    wave = df[wc].to_numpy(float); flux = df[fc].to_numpy(float)
    err = df[ec].to_numpy(float) if ec else np.full_like(flux, np.nan)
    band = df["band"].astype(str).to_numpy() if "band" in df else band_of(wave)
    cc = pick(["continuum", "baseline", "cont", "base_fluxes"], None)
    cont = df[cc].to_numpy(float) if cc else None
    mask = df["mask"].to_numpy(bool) if "mask" in df else None
    s = Spectrum(wave, flux, err, np.asarray(band, dtype=str), name or meta.get("name") or os.path.splitext(os.path.basename(path))[0],
                 float(meta.get("distance_pc", distance_pc)), float(meta.get("rv_kms", 0.0)),
                 meta.get("rest_frame", "False") == "True", cont, mask, meta)
    if not ec:
        s.err = estimate_noise(s)
    return s


def load_spectrum(path: str, source: str = "x1d", extraction: dict | None = None, **kw) -> Spectrum:
    """Load a target: folder of x1d files (default), folder of s3d cubes (`source="s3d"` with the
    extraction options of `load_s3d_folder`), a single FITS file, or a CSV.

    `path` may be written as ``example:<name>`` to use a bundled example (e.g. ``example:FZ_Tau``)."""
    from .examples import resolve_path
    path = resolve_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"spectrum not found: {path}")
    if os.path.isdir(path):
        if source == "s3d":
            return load_s3d_folder(path, **{**(extraction or {}), **kw})
        return load_x1d_folder(path, **kw)
    if path.endswith(".fits"):
        w, f, e, b, meta = read_x1d(path)
        return Spectrum(w, f, e, np.full(len(w), b, dtype=str),
                        kw.get("name") or meta.get("TARGNAME") or os.path.splitext(os.path.basename(path))[0],
                        kw.get("distance_pc", 140.0), meta=meta)
    return load_csv(path, **kw)


# ----------------------------------------------------------------------------------
# Cleaning and noise
# ----------------------------------------------------------------------------------

def spike_filter(spec: Spectrum, nsigma: float = 8.0, width: int = 9, neighbour_frac: float = 0.3) -> Spectrum:
    """Flag single-pixel spikes (cosmic rays, bad pixels) before the continuum step.

    A pixel is a spike when it deviates from a running median by more than `nsigma` times the
    local MAD scatter *and* both neighbours deviate by less than `neighbour_frac` of that
    excursion.  Real lines at MIRI resolution (FWHM >= 1.8 pixels) keep neighbours above ~40 %
    of the peak, so they are not flagged."""
    from scipy.ndimage import median_filter
    s = spec.copy()
    for b in s.bands:
        i = s.band_slice(b)
        f = s.flux[i]
        good = np.isfinite(f)
        if good.sum() < width + 2:
            continue
        med = median_filter(np.where(good, f, np.nanmedian(f)), size=width, mode="nearest")
        resid = np.where(good, f - med, 0.0)
        mad = 1.4826 * np.nanmedian(np.abs(resid[good] - np.nanmedian(resid[good])))
        thr = nsigma * max(mad, 1e-12)
        big = np.abs(resid) > thr
        left = np.roll(resid, 1); right = np.roll(resid, -1)
        isolated = (np.abs(left) < neighbour_frac * np.abs(resid)) & (np.abs(right) < neighbour_frac * np.abs(resid))
        bad = big & isolated
        s.mask[i[bad]] = False
        s.mask[i[~good]] = False
    return s


def estimate_noise(spec: Spectrum, window: int = 25, use_line_free: bool = True) -> np.ndarray:
    """Robust per-pixel noise from the high-pass residual scatter in each sub-band (MAD),
    using line-free pixels when a continuum is set.  Returns sigma per pixel (Jy)."""
    from scipy.ndimage import median_filter
    sig = np.full(len(spec.wave), np.nan)
    for b in spec.bands:
        i = spec.band_slice(b)
        f = spec.flux[i]
        good = np.isfinite(f) & spec.mask[i]
        if good.sum() < 10:
            continue
        smooth = median_filter(np.where(good, f, np.nanmedian(f[good])), size=window, mode="nearest")
        resid = f - smooth
        r = resid[good]
        if use_line_free and spec.continuum is not None and np.any(spec.continuum[i] != 0):
            lf = (spec.line_flux[i] < 2.0 * 1.4826 * np.nanmedian(np.abs(r - np.nanmedian(r))))[good]
            if lf.sum() > 10:
                r = r[lf]
        mad = 1.4826 * np.nanmedian(np.abs(r - np.nanmedian(r)))
        sig[i] = mad
    return sig
