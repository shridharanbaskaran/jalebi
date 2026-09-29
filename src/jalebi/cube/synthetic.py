"""Synthetic MRS-like cubes with known answers, for testing the line maps.

`make_synthetic_cube` builds one sub-band cube (jwst `_s3d.fits` layout when written) containing

* a point source: continuum (Jy, power law) + unresolved line emission,
* a Keplerian ring of line emission (inclination, PA, radius, width, Keplerian speed),
* jet knots (bipolar, +-v along a PA),

each convolved plane by plane with a Gaussian PSF of FWHM 0.033 lambda + 0.106 arcsec (so the PSF
changes across the cube), spectrally with the MRS LSF, plus Gaussian noise.  Optional extras mimic the
cube-building "resampling wiggles" (a spaxel-dependent sinusoidal modulation of the point-source
spectrum) and a sub-band wavelength-calibration offset.

`make_synthetic_cube_set(outdir)` writes three cubes (2B: H2 S(3); 3A: H2 S(2) + [Ne II]; 3C: H2 S(1))
sharing one geometry, for stacking and region tests.  The truth is returned (and written as YAML).
"""
from __future__ import annotations

import os

import numpy as np

from ..data import mrs_psf_fwhm
from ..instrument import _BANDS
from ..lines import C_KMS, FWHM_TO_SIGMA, area_to_cgs_sr, area_to_W_m2, get_line, lsf_sigma_um
from .io import Cube

PIXSCALE = {"1": 0.13, "2": 0.17, "3": 0.20, "4": 0.35}          # arcsec
DLAM = {"1": 0.0008, "2": 0.0013, "3": 0.0025, "4": 0.006}       # micron per channel
BAND_NAME = {"A": "SHORT", "B": "MEDIUM", "C": "LONG"}

DEFAULT_COMPONENTS = [
    {"kind": "point", "line": "H2 S(1)", "flux_W_m2": 1.0e-17},
    {"kind": "ring", "line": "H2 S(1)", "flux_W_m2": 3.0e-17, "r_arcsec": 1.0, "width_arcsec": 0.25,
     "incl_deg": 50.0, "pa_deg": 30.0, "v_kep_kms": 60.0},
    {"kind": "knots", "line": "[Ne II] 12.81", "flux_W_m2": 1.2e-17, "pa_deg": -60.0, "v_kms": 150.0,
     "offsets_arcsec": [0.8, 1.6], "size_arcsec": 0.15},
    {"kind": "point", "line": "[Ne II] 12.81", "flux_W_m2": 2.0e-17},
]


def _wcs(nx, ny, pixscale, ra0, dec0):
    from astropy.wcs import WCS
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra0, dec0]
    w.wcs.crpix = [(nx + 1) / 2.0, (ny + 1) / 2.0]
    w.wcs.cdelt = [-pixscale / 3600.0, pixscale / 3600.0]
    w.wcs.cunit = ["deg", "deg"]
    w.wcs.radesys = "ICRS"
    return w


def _gauss_psf_kernel(fwhm_pix, size):
    s = fwhm_pix * FWHM_TO_SIGMA
    r = np.arange(size) - size // 2
    g = np.exp(-0.5 * (r / s) ** 2)
    k = np.outer(g, g)
    return k / k.sum()


def make_synthetic_cube(band: str = "3C", components=None, wmin: float | None = None, wmax: float | None = None,
                        nx: int = 41, ny: int = 41, pixscale: float | None = None, ra0: float = 60.0, dec0: float = 25.0,
                        star_offset_pix=(0.3, -0.2), cont_jy: float = 0.3, cont_slope: float = 1.0,
                        noise_mjysr: float = 1.0, err_underestimate: float = 1.0, wiggle_amp: float = 0.0,
                        wiggle_period_ch: float = 35.0, v_offset_kms: float = 0.0, rv_kms: float = 0.0,
                        oversample: int = 3, seed: int = 1, name: str = "SYNTH-DISK"):
    """Returns (Cube, truth).  Fluxes in W m^-2, sizes in arcsec, velocities in km/s, PA east of north.

    The line centroid of every component is shifted by rv_kms + v_offset_kms (v_offset mimics a
    sub-band wavelength-calibration error).  truth["maps"] holds noiseless maps of the extended line
    emission (erg s^-1 cm^-2 sr^-1) and its flux-weighted velocity per line."""
    comps = [dict(c) for c in (components or DEFAULT_COMPONENTS)]
    ch = band[0]
    pixscale = pixscale or PIXSCALE[ch]
    lines = sorted({get_line(c["line"]).name for c in comps}, key=lambda n: get_line(n).wave)
    lo_b, hi_b = _BANDS[band]
    inb = [get_line(n) for n in lines if lo_b <= get_line(n).wave <= hi_b]
    if wmin is None or wmax is None:
        if not inb:
            raise ValueError(f"no component line inside band {band}")
        pad = 1600.0 / C_KMS
        wmin = min(ln.wave for ln in inb) * (1 - pad) if wmin is None else wmin
        wmax = max(ln.wave for ln in inb) * (1 + pad) if wmax is None else wmax
    dl = DLAM[ch]
    wave = np.arange(wmin, wmax + 0.5 * dl, dl)
    nz = len(wave)
    rng = np.random.default_rng(seed)
    o = oversample
    NX, NY = nx * o, ny * o
    ps_o = pixscale / o
    xc = (nx - 1) / 2.0 + star_offset_pix[0]; yc = (ny - 1) / 2.0 + star_offset_pix[1]
    # oversampled pixel-centre offsets from the star (arcsec): east is -x on a north-up grid
    yy, xx = np.mgrid[0:NY, 0:NX].astype(float)
    xs = (xx + 0.5) / o - 0.5; ys = (yy + 0.5) / o - 0.5
    east = -(xs - xc) * pixscale; north = (ys - yc) * pixscale
    wcs = _wcs(nx, ny, pixscale, ra0, dec0)
    pixar = (pixscale / 206264.806) ** 2
    ext_cube = np.zeros((nz, NY, NX))                # intrinsic extended line emission, MJy/sr
    point_spec = np.zeros(nz)                         # Jy, unresolved line emission
    truth = {"band": band, "pixscale": pixscale, "star_pix": (xc, yc), "components": comps, "rv_kms": rv_kms,
             "v_offset_kms": v_offset_kms, "noise_mjysr": noise_mjysr, "lines": {}}
    for c in comps:
        ln = get_line(c["line"])
        if not (wave[0] < ln.wave < wave[-1]):
            continue
        sig = float(lsf_sigma_um(ln.wave))
        area_jyum = c["flux_W_m2"] / float(area_to_W_m2(1.0, ln.wave))          # Jy um
        vshift = rv_kms + v_offset_kms
        rec = truth["lines"].setdefault(ln.name, {"point_W_m2": 0.0, "extended_W_m2": 0.0})
        if c["kind"] == "point":
            mu = ln.wave * (1 + (vshift + c.get("v_kms", 0.0)) / C_KMS)
            point_spec += area_jyum / (np.sqrt(2 * np.pi) * sig) * np.exp(-0.5 * ((wave - mu) / sig) ** 2)
            rec["point_W_m2"] += c["flux_W_m2"]
            continue
        if c["kind"] == "ring":
            t = np.deg2rad(c.get("pa_deg", 0.0)); inc = np.deg2rad(c.get("incl_deg", 45.0))
            u = east * np.sin(t) + north * np.cos(t)                  # along the major axis
            w = (east * np.cos(t) - north * np.sin(t)) / np.cos(inc)   # deprojected minor axis
            r = np.hypot(u, w)
            sb = np.exp(-0.5 * ((r - c["r_arcsec"]) / (c.get("width_arcsec", 0.2) * FWHM_TO_SIGMA)) ** 2)
            phi = np.arctan2(w, u)
            vk = c.get("v_kep_kms", 50.0) * np.sqrt(c["r_arcsec"] / np.maximum(r, 0.05))
            vlos = vk * np.sin(inc) * np.cos(phi)
        elif c["kind"] == "knots":
            t = np.deg2rad(c.get("pa_deg", 0.0))
            sb = np.zeros_like(east); vlos = np.zeros_like(east)
            s = c.get("size_arcsec", 0.15) * FWHM_TO_SIGMA
            for d in c.get("offsets_arcsec", [1.0]):
                for sign in (+1, -1):
                    g = np.exp(-0.5 * ((east - sign * d * np.sin(t)) ** 2 + (north - sign * d * np.cos(t)) ** 2) / s ** 2)
                    sb += g; vlos = np.where(g > 1e-3 * g.max(), sign * c.get("v_kms", 100.0), vlos)
        else:
            raise ValueError(f"unknown component kind {c['kind']!r}")
        # normalise the surface brightness to the requested total flux: sum(SB * pixar/o^2) * 1e6 = area [Jy um]
        tot = sb.sum() * pixar / o ** 2 * 1e6
        sb = sb * area_jyum / tot                               # MJy/sr um per oversampled pixel
        mu = ln.wave * (1 + (vshift + vlos) / C_KMS)
        prof = np.exp(-0.5 * ((wave[:, None, None] - mu[None]) / sig) ** 2) / (np.sqrt(2 * np.pi) * sig)
        ext_cube += sb[None] * prof
        rec["extended_W_m2"] += c["flux_W_m2"]
    # PSF convolution per plane (FFT), then block-average to the output grid
    from scipy.signal import fftconvolve
    fwhm_pix_o = mrs_psf_fwhm(wave) / ps_o
    ksize = int(2 * np.ceil(3 * fwhm_pix_o.max()) + 1)
    ext_obs = np.empty((nz, ny, nx))
    for k in range(nz):
        kern = _gauss_psf_kernel(fwhm_pix_o[k], ksize)
        conv = fftconvolve(ext_cube[k], kern, mode="same") if ext_cube[k].any() else ext_cube[k]
        ext_obs[k] = conv.reshape(ny, o, nx, o).mean(axis=(1, 3))
    # point source: pixel-integrated Gaussian PSF per plane (erf), unit total flux
    from scipy.special import erf
    xe = np.arange(nx + 1) - 0.5; ye = np.arange(ny + 1) - 0.5
    sig_pix = mrs_psf_fwhm(wave) / pixscale * FWHM_TO_SIGMA
    fx = np.diff(0.5 * (1 + erf((xe[None, :] - xc) / (np.sqrt(2) * sig_pix[:, None]))), axis=1)
    fy = np.diff(0.5 * (1 + erf((ye[None, :] - yc) / (np.sqrt(2) * sig_pix[:, None]))), axis=1)
    psf = fy[:, :, None] * fx[:, None, :]                    # (nz, ny, nx), sums to ~1
    cont_spec = cont_jy * (wave / wave.mean()) ** cont_slope
    star = (cont_spec + point_spec)[:, None, None] * psf / pixar / 1e6        # MJy/sr
    if wiggle_amp > 0:
        phase = rng.uniform(0, 2 * np.pi, size=(ny, nx))
        star = star * (1 + wiggle_amp * np.sin(2 * np.pi * np.arange(nz)[:, None, None] / wiggle_period_ch + phase[None]))
    model = star + ext_obs
    noise = rng.normal(size=model.shape) * noise_mjysr
    sci = (model + noise).astype(np.float32)
    err = np.full(sci.shape, noise_mjysr * err_underestimate, np.float32)
    # the edge spaxels of real cubes are undefined
    sci[:, 0, :] = np.nan; sci[:, :, -1] = np.nan
    ra_s, dec_s = wcs.all_pix2world(xc, yc, 0)
    from astropy.io import fits
    prim = fits.Header()
    prim["TARGNAME"] = name; prim["TARG_RA"] = float(ra_s); prim["TARG_DEC"] = float(dec_s)
    prim["INSTRUME"] = "MIRI"; prim["CHANNEL"] = ch; prim["BAND"] = BAND_NAME[band[1]]
    prim["ORIGIN"] = "jalebi.cube.synthetic"
    hdr = wcs.to_header(); hdr["PIXAR_SR"] = pixar; hdr["BUNIT"] = "MJy/sr"
    cube = Cube(sci, err, wave, wcs, hdr, prim, band, "", name, {"synthetic": True})
    truth["star_radec"] = (float(ra_s), float(dec_s))
    truth["continuum_Jy_at_center"] = float(np.interp(np.mean(wave), wave, cont_spec))
    # noiseless truth maps of the extended emission per line
    maps = {}
    for nm in truth["lines"]:
        ln = get_line(nm)
        v = C_KMS * (wave - ln.wave * (1 + (rv_kms + v_offset_kms) / C_KMS)) / ln.wave
        sel = np.abs(v) <= 400.0
        dlam = np.gradient(wave)[sel][:, None, None]
        m0 = np.sum(ext_obs[sel] * dlam, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            m1 = np.sum(ext_obs[sel] * dlam * v[sel][:, None, None], axis=0) / m0
        maps[nm] = {"mom0_cgs": area_to_cgs_sr(m0, ln.wave), "mom1_kms": m1}
    truth["maps"] = maps
    return cube, truth


def make_synthetic_cube_set(outdir: str, components=None, seed: int = 1, noise_mjysr: float = 15.0, **kw) -> dict:
    """Three cubes (2B, 3A, 3C) with one geometry: H2 S(3), S(2) and S(1) rings (flux ratio 1 : 1.5 : 2),
    a [Ne II] jet and point sources.  Written as jwst-style `Level3_chN-<band>_s3d.fits` files; returns the truth."""
    os.makedirs(outdir, exist_ok=True)
    geom = {"r_arcsec": 1.0, "width_arcsec": 0.3, "incl_deg": 50.0, "pa_deg": 30.0, "v_kep_kms": 60.0}
    comps = components or [
        {"kind": "point", "line": "H2 S(3)", "flux_W_m2": 0.6e-17},
        {"kind": "ring", "line": "H2 S(3)", "flux_W_m2": 1.5e-17, **geom},
        {"kind": "point", "line": "H2 S(2)", "flux_W_m2": 0.8e-17},
        {"kind": "ring", "line": "H2 S(2)", "flux_W_m2": 2.2e-17, **geom},
        {"kind": "point", "line": "H2 S(1)", "flux_W_m2": 1.0e-17},
        {"kind": "ring", "line": "H2 S(1)", "flux_W_m2": 3.0e-17, **geom},
        {"kind": "point", "line": "[Ne II] 12.81", "flux_W_m2": 2.0e-17},
        {"kind": "knots", "line": "[Ne II] 12.81", "flux_W_m2": 1.2e-17, "pa_deg": -60.0, "v_kms": 150.0,
         "offsets_arcsec": [0.8, 1.6], "size_arcsec": 0.15},
    ]
    truth_all = {}
    names = {"2B": "ch2-medium", "3A": "ch3-short", "3C": "ch3-long"}
    for i, band in enumerate(("2B", "3A", "3C")):
        lo, hi = _BANDS[band]
        sub = [c for c in comps if lo <= get_line(c["line"]).wave <= hi]
        if not sub:
            continue
        cube, truth = make_synthetic_cube(band, sub, seed=seed + i, noise_mjysr=noise_mjysr, **kw)
        cube.write(os.path.join(outdir, f"Level3_{names[band]}_s3d.fits"))
        truth_all[band] = truth
    # YAML summary without the arrays
    import yaml
    slim = {b: {k: v for k, v in t.items() if k not in ("maps",)} for b, t in truth_all.items()}
    with open(os.path.join(outdir, "truth.yaml"), "w") as fh:
        yaml.safe_dump(_plain(slim), fh, sort_keys=False)
    return truth_all


def _plain(o):
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o
