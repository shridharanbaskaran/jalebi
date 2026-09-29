"""Channel maps and position–velocity (PV) cuts of a prepared line.

    cm = channel_maps(lc, vmin=-300, vmax=300, dv=None)      # (nv, ny, nx) + velocity centres
    pv = pv_diagram(lc, pa_deg=20, length_arcsec=4, width_arcsec=0.4)
    cm.write_fits("ch.fits"); pv.write_fits("pv.fits")        # spectral axis VRAD in km/s
    sl = channel_slices(lc, component="full")                # the native channels, one 2-D FITS each
    sl.write("channel_slices/")                             # 5.3368.fits, 5.3377.fits, ... (cube_maps.py)

Position angles are degrees east of north; for a jet, use the jet axis (for HV Tau C about -20 deg).
The PV offset runs from -length/2 to +length/2, positive towards the PA.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..lines import area_to_cgs_sr
from .maps import LineCube


@dataclass
class ChannelMaps:
    lc: LineCube
    velocity: np.ndarray             # (nv,) km/s, bin centres
    dv: float
    data: np.ndarray                 # (nv, ny, nx) erg s^-1 cm^-2 sr^-1 per channel bin
    err: np.ndarray
    source: str

    def write_fits(self, path: str):
        from astropy.io import fits
        h = self.lc.cube.celestial_header()
        h["WCSAXES"] = 3
        h["CTYPE3"] = "VRAD"; h["CUNIT3"] = "km/s"; h["CRPIX3"] = 1.0
        h["CRVAL3"] = float(self.velocity[0]); h["CDELT3"] = float(self.dv); h["PC3_3"] = 1.0
        h["SPECSYS"] = "SOURCE"
        h["RESTWAV"] = (self.lc.line.wave * 1e-6, "rest wavelength [m]")
        h["LINE"] = self.lc.line.name
        h["BUNIT"] = "erg s-1 cm-2 sr-1"
        h["CHSRC"] = (self.source, "line, or extended (PSF removed)")
        hdus = [fits.PrimaryHDU(self.data.astype(np.float32), header=h), fits.ImageHDU(self.err.astype(np.float32), header=h, name="ERR")]
        fits.HDUList(hdus).writeto(path, overwrite=True)
        return path


def channel_maps(lc: LineCube, vmin: float = -300.0, vmax: float = 300.0, dv: float | None = None,
                 source: str = "extended") -> ChannelMaps:
    """Integrated surface brightness in velocity bins of width dv (default: the native channel width)."""
    D = lc.extended if (source == "extended" and lc.psf is not None) else lc.line_data
    src = "extended" if (source == "extended" and lc.psf is not None) else "line"
    v = lc.vel
    step = float(np.median(np.abs(np.diff(v))))
    dv = float(dv or step)
    edges = np.arange(vmin, vmax + 0.5 * dv, dv)
    if len(edges) < 2:
        edges = np.array([vmin, vmax])
    centres = 0.5 * (edges[1:] + edges[:-1])
    conv = area_to_cgs_sr(1.0, lc.line.wave)
    dl = lc.dlam
    ny, nx = D.shape[1:]
    out = np.full((len(centres), ny, nx), np.nan); err = np.full_like(out, np.nan)
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        # fractional overlap of each native channel with the bin
        lo = v - 0.5 * step; hi = v + 0.5 * step
        frac = np.clip((np.minimum(hi, b) - np.maximum(lo, a)) / step, 0.0, 1.0)
        if frac.sum() <= 0:
            continue
        wts = (frac * dl)[:, None, None]
        with np.errstate(invalid="ignore"):
            out[i] = np.nansum(D * wts, axis=0) * conv
            err[i] = np.sqrt(np.nansum((lc.err * wts) ** 2, axis=0)) * conv
            out[i][np.all(~np.isfinite(D[frac > 0]), axis=0)] = np.nan
    return ChannelMaps(lc, centres, dv, out, err, src)


@dataclass
class ChannelSlices:
    """Native channels of the moment window (continuum-subtracted, MJy/sr), as cube_maps.py's
    `get_channel_maps` writes them: one 2-D FITS per channel, named by its wavelength (4 decimals)."""
    lc: LineCube
    index: np.ndarray                # plane indices in the line cube
    wave_um: np.ndarray              # observed wavelength of each plane
    velocity: np.ndarray             # km/s
    data: np.ndarray                 # (n, ny, nx) MJy/sr
    err: np.ndarray
    source: str
    component: str

    def filenames(self) -> list[str]:
        return [f"{float(np.around(w, 4))}.fits" for w in self.wave_um]

    def write(self, outdir: str) -> list[str]:
        import os

        from astropy.io import fits
        os.makedirs(outdir, exist_ok=True)
        out = []
        for k, name in enumerate(self.filenames()):
            h = self.lc.cube.celestial_header()
            h["BUNIT"] = "MJy/sr"
            h["LINE"] = self.lc.line.name
            h["RESTWAV"] = (self.lc.line.wave, "rest wavelength [um]")
            h["WAVELEN"] = (float(self.wave_um[k]), "wavelength of this channel [um]")
            h["VELOCITY"] = (float(self.velocity[k]), "velocity of this channel [km/s]")
            h["CHANNEL"] = (int(self.index[k]), "plane in the line window")
            h["CHSRC"] = (self.source, "line, or extended (PSF removed)")
            h["WINDOW"] = (self.component, "moment window (full | slow | fast)")
            p = os.path.join(outdir, name)
            fits.PrimaryHDU(self.data[k].astype(np.float32), header=h).writeto(p, overwrite=True)
            out.append(p)
        return out


def channel_slices(lc: LineCube, component: str = "full", half_full: int = 4, half_slow: int = 2,
                   source: str = "line") -> ChannelSlices:
    """The individual native channels of a moment window (`maps.moment_window`), continuum-subtracted
    ("line", as cube_maps.py) or point-source-subtracted ("extended")."""
    from .maps import moment_window
    D = lc.extended if (source == "extended" and lc.psf is not None) else lc.line_data
    src = "extended" if (source == "extended" and lc.psf is not None) else "line"
    sel, _ = moment_window(lc, None, component, half_full, half_slow)
    idx = np.flatnonzero(sel)
    return ChannelSlices(lc, idx, lc.cube.wave[idx], lc.vel[idx], D[idx], lc.err[idx], src, component)


@dataclass
class PVDiagram:
    lc: LineCube
    offset: np.ndarray               # (ns,) arcsec along the cut
    velocity: np.ndarray             # (nz,) km/s
    data: np.ndarray                 # (nz, ns) MJy/sr (mean across the slit width)
    err: np.ndarray
    pa_deg: float
    width_arcsec: float
    center_radec: tuple
    source: str

    def write_fits(self, path: str):
        from astropy.io import fits
        h = fits.Header()
        h["CTYPE1"] = "OFFSET"; h["CUNIT1"] = "arcsec"; h["CRPIX1"] = 1.0
        h["CRVAL1"] = float(self.offset[0]); h["CDELT1"] = float(np.diff(self.offset).mean()) if len(self.offset) > 1 else 1.0
        h["CTYPE2"] = "VRAD"; h["CUNIT2"] = "km/s"; h["CRPIX2"] = 1.0
        h["CRVAL2"] = float(self.velocity[0]); h["CDELT2"] = float(np.diff(self.velocity).mean()) if len(self.velocity) > 1 else 1.0
        h["BUNIT"] = "MJy/sr"
        h["LINE"] = self.lc.line.name; h["RESTWAV"] = self.lc.line.wave
        h["PV_PA"] = (self.pa_deg, "cut position angle [deg E of N]")
        h["PV_WIDTH"] = (self.width_arcsec, "slit width [arcsec]")
        h["PV_RA"] = self.center_radec[0]; h["PV_DEC"] = self.center_radec[1]
        h["PVSRC"] = self.source
        fits.HDUList([fits.PrimaryHDU(self.data.astype(np.float32), header=h),
                      fits.ImageHDU(self.err.astype(np.float32), header=h, name="ERR")]).writeto(path, overwrite=True)
        return path


def pv_diagram(lc: LineCube, pa_deg: float, length_arcsec: float = 4.0, width_arcsec: float | None = None,
               center=None, step_arcsec: float | None = None, source: str = "extended", vmax_kms: float | None = None) -> PVDiagram:
    """Position–velocity cut through `center` (default: the source) along `pa_deg`."""
    from scipy.ndimage import map_coordinates
    D = lc.extended if (source == "extended" and lc.psf is not None) else lc.line_data
    src = "extended" if (source == "extended" and lc.psf is not None) else "line"
    cube = lc.cube
    ra0, dec0 = center if center is not None else lc.center_radec
    step = step_arcsec or cube.pixscale
    width = width_arcsec if width_arcsec is not None else max(cube.pixscale, 0.5 * lc.fwhm_arcsec)
    s = np.arange(-0.5 * length_arcsec, 0.5 * length_arcsec + 1e-9, step)
    nw = max(1, int(round(width / cube.pixscale)))
    wv = (np.arange(nw) - 0.5 * (nw - 1)) * (width / nw)
    t = np.deg2rad(pa_deg)
    S, Wd = np.meshgrid(s, wv)                                   # (nw, ns)
    dx = S * np.sin(t) + Wd * np.cos(t)                          # east
    dy = S * np.cos(t) - Wd * np.sin(t)                          # north
    ra = ra0 + dx / 3600.0 / np.cos(np.deg2rad(dec0)); dec = dec0 + dy / 3600.0
    xs, ys = cube.world_to_pix(ra, dec)
    sel = slice(None) if vmax_kms is None else (np.abs(lc.vel) <= vmax_kms)
    Dk = D[sel]; Ek = lc.err[sel]
    out = np.full((Dk.shape[0], len(s)), np.nan); err = np.full_like(out, np.nan)
    for k in range(Dk.shape[0]):
        p = Dk[k]; good = np.isfinite(p)
        a = map_coordinates(np.where(good, p, 0.0), [ys, xs], order=1, mode="constant", cval=0.0)
        g = map_coordinates(good.astype(float), [ys, xs], order=1, mode="constant", cval=0.0)
        e = map_coordinates(np.where(np.isfinite(Ek[k]), Ek[k], 0.0), [ys, xs], order=1, mode="constant", cval=0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            val = np.where(g > 0.5, a / np.where(g > 0, g, 1), np.nan)
            out[k] = np.nanmean(val, axis=0)
            err[k] = np.sqrt(np.nanmean(np.where(g > 0.5, e, np.nan) ** 2, axis=0) / nw)
    return PVDiagram(lc, s, lc.vel[sel], out, err, float(pa_deg), float(width), (float(ra0), float(dec0)), src)
