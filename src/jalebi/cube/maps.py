"""Line maps: continuum-subtracted line cubes, moment maps, centroid velocity maps, line stacking.

    lc = prepare_line(cubes, "H2 S(1)")            # slab, per-spaxel continuum, point-source removal
    m  = line_maps(lc)                             # moments 0/1/2 + Gaussian centroid velocities (MC errors)
    m.write("out/")                                # FITS (celestial WCS: open in CARTA / DS9) + PNG

Velocities.  The MRS resolves 85-200 km/s (R ~ 1500-3700), so the kinematic information is in the
*centroid shifts*, which a Gaussian fit measures to a few km/s at S/N >~ 20, not in the widths (mom2 and
the fitted FWHM are dominated by the instrument; the maps give them for completeness).  Velocities are
relative to the rest wavelength after removing `rv_kms` (systemic) and a per-sub-band zero-point
offset `band_offsets_kms` (the MRS wavelength calibration differs by a few km/s between sub-bands);
`zero_point="star"` instead puts v = 0 at the source position for each line, which removes both.
"""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field

import numpy as np

from ..data import mrs_psf_fwhm
from ..lines import (C_KMS, FWHM_TO_SIGMA, Line, area_to_cgs_sr, fit_gaussian_batch, gaussian_start, get_line,
                     lsf_fwhm_kms, lsf_sigma_um)
from .continuum import CubeContinuumSettings, spaxel_continuum
from .io import Cube, CubeSet, resolve_center
from .psf import PSFSettings, subtract_point_sources


# ----------------------------------------------------------------------------------------------
# Line cube
# ----------------------------------------------------------------------------------------------

@dataclass
class LineCube:
    """Everything about one line in one cube, on the cube's pixel grid."""
    line: Line
    cube: Cube                        # the slab (observed wavelengths)
    vel: np.ndarray                   # (nz,) km/s, rest frame of the line, systemic and band offset removed
    data: np.ndarray                  # (nz, ny, nx) MJy/sr, observed
    err: np.ndarray                   # (nz, ny, nx) per-channel uncertainty used (MJy/sr)
    cont: np.ndarray                  # per-spaxel continuum model
    line_data: np.ndarray             # data - cont
    psf: object | None                # PSFResult, or None when point-source removal is off
    line_free: np.ndarray             # (nz,) channels used for the continuum
    noise: np.ndarray                 # (ny, nx) empirical rms of the line-free residuals
    center_radec: tuple
    center_pix: tuple
    rv_kms: float = 0.0
    band_offset_kms: float = 0.0
    smooth_fwhm_pix: float = 0.0
    settings: dict = field(default_factory=dict)
    zero_point_kms: float = 0.0       # velocity subtracted from `vel` by line_maps(zero_point="star")

    @property
    def extended(self) -> np.ndarray:
        return self.line_data if self.psf is None else self.psf.extended

    @property
    def lsf_fwhm_kms(self) -> float:
        return float(lsf_fwhm_kms(self.line.wave))

    @property
    def dlam(self) -> np.ndarray:
        """Channel widths (micron)."""
        w = self.cube.wave
        return np.gradient(w) if len(w) > 1 else np.ones(1)

    @property
    def fwhm_arcsec(self) -> float:
        return float(mrs_psf_fwhm(self.line.wave))

    def continuum_image(self) -> np.ndarray:
        """Continuum surface brightness at the line (MJy/sr)."""
        k = int(np.argmin(np.abs(self.vel)))
        return self.cont[k]


def _smooth_planes(cube: np.ndarray, fwhm_pix: float) -> np.ndarray:
    """NaN-aware 2-D Gaussian smoothing of every plane (normalised convolution)."""
    if not fwhm_pix or fwhm_pix <= 0:
        return cube
    from scipy.ndimage import gaussian_filter
    sig = fwhm_pix * FWHM_TO_SIGMA
    good = np.isfinite(cube)
    num = gaussian_filter(np.where(good, cube, 0.0), sigma=(0, sig, sig), mode="constant")
    den = gaussian_filter(good.astype(float), sigma=(0, sig, sig), mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[~good] = np.nan
    return out


def prepare_line(cubes: CubeSet | Cube | str, line, rv_kms: float = 0.0, window_kms: float = 1500.0,
                 continuum: CubeContinuumSettings | dict | None = None, psf: PSFSettings | dict | None = None,
                 center=None, band_offsets_kms: dict | None = None, smooth_fwhm_pix: float = 0.0,
                 noise: str = "max", search_arcsec: float = 1.5, window_um: float | None = None,
                 band: str | None = None) -> LineCube:
    """Cut the line window out of the right cube, subtract a per-spaxel continuum and (optionally) the
    point source.

    window_um : half-width of the window in micron instead of window_kms, with the planes chosen as
                spectral_cube's `spectral_slab` does (cube_maps.py: `dlambda`, 0.1 um)
    band      : which cube: None (the sub-band with the widest margin), "nominal" (the fixed boundaries
                of cube_maps.py's `get_channel`) or a sub-band ("3A", "ch3-short")

    noise : "max" (default) per-channel uncertainty = max(pipeline ERR, empirical rms of the line-free
            residuals of that spaxel) — the empirical term carries the cube-building resampling noise and
            the correlated noise the ERR array misses; "err" or "empirical" use one of them.
    """
    ln = get_line(line)
    if window_um is not None and not window_um > 0:
        raise ValueError(f"window_um must be > 0 (got {window_um}); use None for the km/s window")
    if isinstance(cubes, str):
        cubes = CubeSet(cubes)
    cont_s = continuum if isinstance(continuum, CubeContinuumSettings) else CubeContinuumSettings(**(continuum or {}))
    psf_s = psf if isinstance(psf, PSFSettings) else PSFSettings(**(psf or {}))
    if isinstance(cubes, CubeSet):
        cube = cubes.for_line(ln, rv_kms, window_kms, band=band, window_um=window_um)
        radec = resolve_center(cubes, center, search_arcsec)
    else:
        lam = ln.wave * (1 + rv_kms / C_KMS)
        if window_um is not None:
            cube = cubes.slab(lam - window_um, lam + window_um, rule="closest")
        else:
            dl = lam * window_kms / C_KMS
            cube = cubes.slab(lam - dl, lam + dl)
        radec = resolve_center(cubes, center, search_arcsec)
    if len(cube.wave) < 8:
        raise ValueError(f"only {len(cube.wave)} channels around {ln.name}: widen {'window_um' if window_um else 'window_kms'}"
                         " (or the line is outside this cube)")
    offs = (band_offsets_kms or {}).get(cube.band, 0.0)
    # refine the centre on this cube's own continuum (the MRS channels are not perfectly co-aligned)
    try:
        ra, dec, xc, yc = cube.find_source(near=radec, search_arcsec=max(0.6, 1.2 * float(mrs_psf_fwhm(ln.wave))))
    except ValueError:
        xc, yc = cube.world_to_pix(*radec); ra, dec = radec
    sci = _smooth_planes(cube.sci.astype(float), smooth_fwhm_pix)
    err_pipe = cube.err.astype(float)
    if smooth_fwhm_pix and smooth_fwhm_pix > 0:
        # white noise after a normalised Gaussian kernel: sigma / (2 sqrt(pi) s) = sigma / (1.505 FWHM)
        err_pipe = err_pipe / max(1.0, 1.505 * smooth_fwhm_pix)
    fwhm_kms = float(lsf_fwhm_kms(ln.wave))
    cont, use = spaxel_continuum(cube.wave, sci, ln, cont_s, rv_kms + offs, fwhm_kms)
    L = sci - cont
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r = np.where(use[:, None, None], L, np.nan)
        emp = 1.4826 * np.nanmedian(np.abs(r - np.nanmedian(r, axis=0)), axis=0)
    emp = np.where(np.isfinite(emp), emp, np.nan)
    if noise == "err":
        E = err_pipe
    elif noise == "empirical":
        E = np.broadcast_to(emp, L.shape).copy()
    else:
        E = np.fmax(np.where(np.isfinite(err_pipe), err_pipe, 0.0), np.broadcast_to(np.nan_to_num(emp), L.shape))
        E = np.where(E > 0, E, np.nan)
    ps = None
    companions = []
    if psf_s.enabled:
        centers = [(xc, yc)]
        for src in psf_s.extra_sources or []:
            from ..data import parse_radec
            x2, y2 = cube.world_to_pix(*parse_radec(*src))
            centers.append((float(x2), float(y2)))
        if psf_s.auto_sources:
            from .psf import find_companions
            k0 = int(np.argmin(np.abs(cube.wave - ln.wave * (1 + rv_kms / C_KMS))))
            fw = float(mrs_psf_fwhm(ln.wave)) / cube.pixscale
            for cx, cy in find_companions(cont[k0], (xc, yc), fw, psf_s.auto_threshold):
                if all(np.hypot(cx - a, cy - b) > 2 * fw for a, b in centers):
                    centers.append((cx, cy)); companions.append(tuple(map(float, cube.pix_to_world(cx, cy))))
        ps = subtract_point_sources(L, cont, cube.wave, centers, cube.pixscale, cube.pixar_sr, err=E, settings=psf_s)
    lam0 = ln.wave * (1 + (rv_kms + offs) / C_KMS)
    vel = C_KMS * (cube.wave - lam0) / lam0
    return LineCube(ln, cube, vel, sci, E, cont, L, ps, use, emp, (float(ra), float(dec)), (float(xc), float(yc)),
                    rv_kms, offs, smooth_fwhm_pix,
                    {"continuum": dict(cont_s.__dict__), "psf": dict(psf_s.__dict__), "noise": noise,
                     "window_kms": window_kms, "window_um": window_um, "band": band, "companions_radec": companions})


# ----------------------------------------------------------------------------------------------
# Maps
# ----------------------------------------------------------------------------------------------

@dataclass
class LineMaps:
    """2-D products of one line (all on the cube's spaxel grid)."""
    lc: LineCube
    maps: dict                         # name -> (ny, nx) array
    units: dict                        # name -> unit string
    spectra: dict                      # name -> dict of 1-D arrays (integrated spectra)
    summary: dict

    @property
    def line(self) -> Line:
        return self.lc.line

    def __getitem__(self, k):
        return self.maps[k]

    def native(self, key: str = "mom0", unit: str = "MJy/sr um") -> np.ndarray:
        """A moment-0 type map in the cube's own units: "MJy/sr um" (integral of I_nu over lambda in micron)
        or "MJy/sr m" (the numbers in the FITS files of cube_maps.py, whose spectral_cube axis is in metres).
        The maps themselves are in erg s-1 cm-2 sr-1 (= MJy/sr um x 2.998e-3 / lambda[um]^2)."""
        f = native_factor(self.line.wave, unit)
        return np.asarray(self.maps[key]) * f

    def write(self, outdir: str, prefix: str | None = None, formats=("fits", "png"), distance_pc: float | None = None,
              pa_deg: float | None = None, subfolder: bool = True, extent_arcsec: float | None = None) -> list[str]:
        """Into `outdir/<line tag>/`: FITS maps with the celestial WCS (one file per product, so CARTA and
        DS9 open them directly), `*_all_maps.fits` with every product as an extension, the integrated
        spectra (CSV), a JSON summary and the PNG figure."""
        from astropy.io import fits
        outdir = os.path.join(outdir, self.line.tag) if subfolder else outdir
        os.makedirs(outdir, exist_ok=True)
        prefix = prefix or f"{_safe(self.lc.cube.name)}_{self.line.tag}"
        written = []
        hdr0 = self.lc.cube.celestial_header()
        hdr0["LINE"] = (self.line.name, "line")
        hdr0["RESTWAV"] = (self.line.wave, "rest wavelength [um]")
        hdr0["RV_SYS"] = (self.lc.rv_kms, "systemic velocity removed [km/s]")
        hdr0["VOFFSET"] = (self.lc.band_offset_kms, "sub-band velocity offset removed [km/s]")
        hdr0["PSFSUB"] = (self.lc.psf is not None, "point source removed")
        hdr0["RA_SRC"] = (self.lc.center_radec[0], "source RA [deg]")
        hdr0["DEC_SRC"] = (self.lc.center_radec[1], "source Dec [deg]")
        from .. import __version__
        hdr0["CREATOR"] = f"jalebi {__version__} (jalebi.cube)"
        if "fits" in formats:
            hdus = [fits.PrimaryHDU(header=hdr0)]
            for k, a in self.maps.items():
                h = hdr0.copy(); h["BUNIT"] = self.units.get(k, ""); h["EXTNAME"] = k.upper()[:68]
                p = os.path.join(outdir, f"{prefix}_{k}.fits")
                fits.PrimaryHDU(np.asarray(a, np.float32), header=h).writeto(p, overwrite=True); written.append(p)
                hdus.append(fits.ImageHDU(np.asarray(a, np.float32), header=h, name=k.upper()[:68]))
            p = os.path.join(outdir, f"{prefix}_all_maps.fits")
            fits.HDUList(hdus).writeto(p, overwrite=True); written.append(p)
        import pandas as pd
        for k, d in self.spectra.items():
            p = os.path.join(outdir, f"{prefix}_{k}.csv")
            pd.DataFrame(d).to_csv(p, index=False); written.append(p)
        p = os.path.join(outdir, f"{prefix}_summary.json")
        with open(p, "w") as fh:
            json.dump(_jsonable(self.summary), fh, indent=2)
        written.append(p)
        if "png" in formats:
            from .plots import plot_line_maps
            fig = plot_line_maps(self, distance_pc=distance_pc, pa_deg=pa_deg, extent_arcsec=extent_arcsec)
            p = os.path.join(outdir, f"{prefix}_maps.png")
            fig.savefig(p, dpi=150, bbox_inches="tight"); written.append(p)
            import matplotlib.pyplot as plt
            plt.close(fig)
        return written


def _safe(s: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(s)).strip("_") or "target"


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


COMPONENTS = ("full", "slow", "fast")


def moment_window(lc: LineCube, line_kms: float | None = None, component: str | None = None, half_full: int = 4,
                  half_slow: int = 2) -> tuple[np.ndarray, dict]:
    """Channels integrated for moment 0 (boolean mask over the planes) and a description.

    component=None : a velocity window |v| <= line_kms (default max(200 km/s, 1.5 x instrumental FWHM));
    component="full" | "slow" | "fast" : a window of native channels around the channel nearest to the line
        (the cube_maps.py recipe): "full" = the central channel +-half_full (9 channels), "slow" = +-half_slow
        (5 channels, the line core), "fast" = full minus slow (the 2 + 2 wing channels).  The central channel
        is found in the systemic frame (`zero_point="star"` does not move it)."""
    v = lc.vel + lc.zero_point_kms
    nz = len(v)
    if component is None or component in ("", "velocity", "none"):
        if line_kms is None:
            line_kms = max(1.5 * lc.lsf_fwhm_kms, 200.0)
        sel = np.abs(lc.vel) <= line_kms
        return sel, {"window": "velocity", "line_kms": float(line_kms)}
    if component not in COMPONENTS:
        raise ValueError(f"component must be one of {COMPONENTS} (or None for a velocity window), not {component!r}")
    ci = int(np.argmin(np.abs(v)))
    idx = np.arange(nz)
    full = (idx >= max(0, ci - half_full)) & (idx < min(nz, ci + half_full + 1))
    slow = (idx >= max(0, ci - half_slow)) & (idx < min(nz, ci + half_slow + 1))
    sel = full if component == "full" else slow if component == "slow" else (full & ~slow)
    want = {"full": 2 * half_full + 1, "slow": 2 * half_slow + 1, "fast": 2 * (half_full - half_slow)}[component]
    if sel.sum() < want:
        warnings.warn(f"{lc.line.name}: only {int(sel.sum())} of the {want} channels of the '{component}' window are inside "
                      f"the slab (channel {ci} of {nz}); the line is at the edge of the cube"
                      + (" (cube_maps.py stopped with an error here)" if component == "fast" and
                         (ci - half_full < 0 or ci + half_full + 1 > nz) else ""))
    vs = lc.vel[sel]
    return sel, {"window": "channels", "component": component, "central_channel": ci, "half_full": half_full,
                 "half_slow": half_slow, "channels": [int(i) for i in idx[sel]],
                 "line_kms": float(np.max(np.abs(vs))) if len(vs) else np.nan,
                 "v_range_kms": [float(vs.min()), float(vs.max())] if len(vs) else [np.nan, np.nan]}


def moments(lc: LineCube, source: str = "line", line_kms: float | None = None, snr_min: float = 3.0,
            channel_clip: float = 0.0, component: str | None = None, half_full: int = 4, half_slow: int = 2,
            min_valid: int | None = None) -> dict:
    """Moment 0 (erg s^-1 cm^-2 sr^-1), its error and S/N, moment 1 and 2 (km/s).

    The window is |v| <= line_kms, or the native channels of `component` ("full" | "slow" | "fast", the
    cube_maps.py windows, see `moment_window`).  source: "line" (continuum-subtracted) or "extended"
    (point source removed).  Moments 1 and 2 use spaxels with S/N >= snr_min, and optionally only
    channels above channel_clip x sigma.  A spaxel needs `min_valid` finite channels in the window
    (default: half of them, at least 3; 1 = spectral_cube's rule, NaN only when all are NaN)."""
    D = lc.line_data if source == "line" else lc.extended
    v = lc.vel
    sel, win = moment_window(lc, line_kms, component, half_full, half_slow)
    dl = lc.dlam[sel][:, None, None]
    Ds = D[sel]; Es = lc.err[sel]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m0 = np.nansum(Ds * dl, axis=0)
        e0 = np.sqrt(np.nansum((Es * dl) ** 2, axis=0))
        nvalid = np.isfinite(Ds).sum(axis=0)
        need = max(3, sel.sum() // 2) if min_valid is None else max(1, int(min_valid))
        m0[nvalid < min(need, max(int(sel.sum()), 1))] = np.nan
        conv = area_to_cgs_sr(1.0, lc.line.wave)
        M0 = m0 * conv; E0 = e0 * conv
        snr = M0 / E0
        Dc = Ds if channel_clip <= 0 else np.where(Ds > channel_clip * Es, Ds, 0.0)
        Dc = np.nan_to_num(Dc)
        wsum = np.sum(Dc * dl, axis=0)
        vv = v[sel][:, None, None]
        m1 = np.sum(Dc * dl * vv, axis=0) / wsum
        m2 = np.sqrt(np.clip(np.sum(Dc * dl * (vv - m1) ** 2, axis=0) / wsum, 0, None))
        ok = (snr >= snr_min) & (wsum > 0)
        m1 = np.where(ok, m1, np.nan); m2 = np.where(ok, m2, np.nan)
        peak = np.nanmax(np.where(np.isfinite(Ds), Ds, -np.inf), axis=0)
        peak[~np.isfinite(peak)] = np.nan
    return {"mom0": M0, "mom0_err": E0, "snr": snr, "mom1": m1, "mom2": m2, "peak": peak, "line_kms": win["line_kms"],
            "window": win, "sel": sel}


def gaussian_velocity(lc: LineCube, source: str = "line", snr_min: float = 5.0, n_mc: int = 100,
                      center_shift_kms: float = 200.0, sigma_range=(0.8, 2.5), fit_kms: float | None = None,
                      with_offset: bool = False, seed: int | None = 0, snr_map: np.ndarray | None = None,
                      where: np.ndarray | None = None) -> dict:
    """Gaussian fit per spaxel: centroid velocity, its Monte Carlo error, FWHM, line flux.

    The width is bounded to [sigma_range] x the instrumental sigma and the centre to
    +-center_shift_kms (as in the old channel_maps notebook).  Errors: `n_mc` noise realisations per
    spaxel refitted in one batch (0 = covariance errors)."""
    D = lc.line_data if source == "line" else lc.extended
    ln = lc.line
    lam0 = ln.wave * (1 + (lc.rv_kms + lc.band_offset_kms) / C_KMS)
    if fit_kms is None:
        fit_kms = max(2.5 * lc.lsf_fwhm_kms, center_shift_kms + 1.5 * lc.lsf_fwhm_kms)
    sel = np.abs(lc.vel) <= fit_kms
    x = lc.cube.wave[sel]
    nz, ny, nx = D.shape
    if snr_map is None:
        snr_map = moments(lc, source)["snr"]
    todo = np.isfinite(snr_map) & (snr_map >= snr_min)
    if where is not None:
        todo &= where
    idx = np.flatnonzero(todo.ravel())
    out = {k: np.full((ny, nx), np.nan) for k in ("vcen", "vcen_err", "fwhm", "fwhm_err", "gflux", "gflux_err", "gchi2r", "width_at_bound")}
    if len(idx) == 0:
        return out
    Y = D[sel].reshape(sel.sum(), -1)[:, idx].T
    E = lc.err[sel].reshape(sel.sum(), -1)[:, idx].T
    W = np.where(np.isfinite(E) & (E > 0), 1.0 / np.where(E > 0, E, 1.0), 0.0)
    sig0 = float(lsf_sigma_um(lam0))
    dmu = lam0 * center_shift_kms / C_KMS
    p0 = gaussian_start(x, Y, lam0, sig0)
    amax = np.nanmax(np.abs(np.nan_to_num(Y)), axis=1) * 5 + 1e-30
    lo = np.array([0.0, lam0 - dmu, sigma_range[0] * sig0, -np.inf])
    hi = np.column_stack([amax, np.full(len(idx), lam0 + dmu), np.full(len(idx), sigma_range[1] * sig0), np.full(len(idx), np.inf)])
    lo = np.broadcast_to(lo, hi.shape)
    p0 = np.clip(p0, lo, hi)
    bf = fit_gaussian_batch(x, Y, W, p0, lo, hi, with_offset=with_offset)
    v = C_KMS * (bf.mu - lam0) / lam0
    fw = bf.sigma / FWHM_TO_SIGMA / bf.mu * C_KMS
    area = bf.area
    if n_mc and n_mc > 0:
        # all realisations of all spaxels in one batch: (n_mc * n) spectra, refitted from the best fit
        rng = np.random.default_rng(seed)
        n = len(idx)
        Pbest = np.column_stack([bf.amp, bf.mu, bf.sigma, bf.offset])
        Emc = np.where(W > 0, 1.0 / np.where(W > 0, W, 1.0), 0.0)
        best_model = bf.amp[:, None] * np.exp(-0.5 * ((x[None] - bf.mu[:, None]) / bf.sigma[:, None]) ** 2) + bf.offset[:, None]
        vs = np.empty((n_mc, n)); fs = np.empty((n_mc, n)); ws = np.empty((n_mc, n))
        per = max(1, 60000 // max(n, 1))                  # realisations per batch (memory)
        for k0 in range(0, n_mc, per):
            kk = min(per, n_mc - k0)
            Ym = (best_model[None] + rng.normal(size=(kk,) + Y.shape) * Emc[None]).reshape(kk * n, -1)
            rep = lambda a: np.broadcast_to(a, (kk,) + a.shape).reshape((kk * n,) + a.shape[1:])  # noqa: E731
            m = fit_gaussian_batch(x, Ym, rep(W), rep(Pbest), rep(lo), rep(hi), with_offset=with_offset, maxiter=30)
            vs[k0:k0 + kk] = (C_KMS * (m.mu - lam0) / lam0).reshape(kk, n)
            fs[k0:k0 + kk] = m.area.reshape(kk, n)
            ws[k0:k0 + kk] = (m.sigma / FWHM_TO_SIGMA / m.mu * C_KMS).reshape(kk, n)
        ve = np.nanstd(vs, axis=0); ae = np.nanstd(fs, axis=0); we = np.nanstd(ws, axis=0)
    else:
        ve = bf.mu_err / lam0 * C_KMS; ae = bf.area_err; we = np.sqrt(np.clip(bf.cov[:, 2, 2], 0, None)) / FWHM_TO_SIGMA / bf.mu * C_KMS
    good = bf.ok
    conv = area_to_cgs_sr(1.0, ln.wave)
    for k, val in (("vcen", v), ("vcen_err", ve), ("fwhm", fw), ("fwhm_err", we), ("gflux", area * conv),
                   ("gflux_err", ae * conv), ("gchi2r", bf.chi2 / np.maximum(bf.dof, 1)),
                   ("width_at_bound", bf.sigma_at_bound.astype(float))):
        a = np.full(ny * nx, np.nan)
        a[idx] = np.where(good, val, np.nan)
        out[k] = a.reshape(ny, nx)
    return out


def _aperture_flux(img: np.ndarray, x0: float, y0: float, r_in: float, r_out: float) -> float:
    from ..data import _circular_weights
    ny, nx = img.shape
    wo = _circular_weights(ny, nx, x0, y0, r_out)
    wi = _circular_weights(ny, nx, x0, y0, r_in) if r_in > 0 else 0.0
    w = wo - wi
    return float(np.nansum(np.where(np.isfinite(img), img, 0.0) * w))


def line_maps(lc: LineCube, snr_min: float = 3.0, line_kms: float | None = None, kinematics: bool = True,
              kin_source: str | None = None, kin_snr_min: float = 5.0, n_mc: int = 100, center_shift_kms: float = 200.0,
              sigma_range=(0.8, 2.5), zero_point: str = "none", channel_clip: float = 0.0, seed: int | None = 0,
              component: str | None = None, half_full: int = 4, half_slow: int = 2, min_valid: int | None = None,
              rms_region=None, rms_sigma: float = 3.0, rms_mode: str = "rms") -> LineMaps:
    """All 2-D products of a prepared line.

    kin_source : spectra used for the velocity fit, "line" (default when there is no point-source
                 removal) or "extended" (default when there is).  In the PSF core the extended cube
                 is ~0 by construction, so the core velocities come from the "line" cube.
    zero_point : "none" or "star" (subtract the centroid velocity measured on the source-integrated,
                 continuum-subtracted spectrum, per line).
    component  : moment-0 window: None (|v| <= line_kms) or "full" | "slow" | "fast" (native channels
                 around the line, the cube_maps.py windows; see `moment_window`).
    rms_region : (x, y, r) pixel circle of empty sky: adds "mom0_masked", moment 0 where it is at least
                 rms_sigma x the level of that circle (rms_mode "rms" = nanstd, "mean", "median"), as
                 cube_maps.py's `plot_moment0_map(sigma_clip=True)` does.
    """
    maps: dict = {}; units: dict = {}
    if lc.zero_point_kms:
        # a previous zero_point="star" call shifted the velocity axis: start again from the systemic frame
        lc.vel = lc.vel + lc.zero_point_kms
        lc.zero_point_kms = 0.0
    wkw = dict(component=component, half_full=half_full, half_slow=half_slow, min_valid=min_valid)
    mo = moments(lc, "line", line_kms, snr_min, channel_clip, **wkw)
    sel = mo["sel"]
    for k in ("mom0", "mom0_err", "snr", "mom1", "mom2", "peak"):
        maps[k] = mo[k]
    units.update(mom0="erg s-1 cm-2 sr-1", mom0_err="erg s-1 cm-2 sr-1", snr="", mom1="km s-1", mom2="km s-1", peak="MJy sr-1")
    maps["continuum"] = lc.continuum_image(); units["continuum"] = "MJy sr-1"
    if lc.psf is not None:
        me = moments(lc, "extended", mo["line_kms"], snr_min, channel_clip, **wkw)
        maps["mom0_ext"] = me["mom0"]; maps["snr_ext"] = me["snr"]; maps["mom1_ext"] = me["mom1"]
        units.update(mom0_ext="erg s-1 cm-2 sr-1", snr_ext="", mom1_ext="km s-1")
        maps["psf_model_mom0"] = moments_of(lc, lc.psf.model, mo["line_kms"], sel=sel); units["psf_model_mom0"] = "erg s-1 cm-2 sr-1"
    rms_info = None
    if rms_region is not None:
        from .masks import rms_mask
        cx, cy, rr = (float(x) for x in rms_region)
        try:
            masked, thr, _ = rms_mask(maps["mom0"], cx, cy, rr, rms_sigma, rms_mode)
        except ValueError as ex:          # the circle (pixels) is not inside this cube's grid
            warnings.warn(f"{lc.line.name}: {ex} No mom0_masked for this line.")
            rms_info = {"center_x": cx, "center_y": cy, "radius": rr, "error": str(ex)}
        else:
            maps["mom0_masked"] = masked; units["mom0_masked"] = "erg s-1 cm-2 sr-1"
            rms_info = {"center_x": cx, "center_y": cy, "radius": rr, "sigma": rms_sigma, "mode": rms_mode, "threshold": thr}
    # spaxel flux in W m^-2 (for aperture sums)
    maps["flux_W_m2"] = maps["mom0"] * 1e-3 * lc.cube.pixar_sr; units["flux_W_m2"] = "W m-2 per spaxel"
    zp = 0.0
    if kinematics:
        src = kin_source or ("extended" if lc.psf is not None else "line")
        kin = gaussian_velocity(lc, src, kin_snr_min, n_mc, center_shift_kms, sigma_range, seed=seed,
                                snr_map=maps["snr_ext"] if src == "extended" else maps["snr"])
        if src == "extended":
            # fill the PSF core, where the extended cube is ~0 by construction, from the full line cube
            core = lc.psf.core_mask
            kin_c = gaussian_velocity(lc, "line", kin_snr_min, n_mc, center_shift_kms, sigma_range, seed=seed, snr_map=maps["snr"],
                                      where=core)
            for k in kin:
                kin[k] = np.where(core, kin_c[k], kin[k])
        # velocity of the unresolved emission from the source-integrated spectrum
        vstar = _source_velocity(lc, center_shift_kms, sigma_range)
        if zero_point == "star" and np.isfinite(vstar[0]):
            zp = vstar[0]
            kin["vcen"] = kin["vcen"] - zp
            maps["mom1"] = maps["mom1"] - zp
            if "mom1_ext" in maps:
                maps["mom1_ext"] = maps["mom1_ext"] - zp
        maps.update(kin)
        units.update(vcen="km s-1", vcen_err="km s-1", fwhm="km s-1", fwhm_err="km s-1", gflux="erg s-1 cm-2 sr-1",
                     gflux_err="erg s-1 cm-2 sr-1", gchi2r="", width_at_bound="")
    else:
        vstar = (np.nan, np.nan)
    if zp:
        # put the whole line cube in the source frame, so channel maps, PV cuts, the integrated spectra and
        # the web app's spectra use the same velocities as the maps
        lc.vel = lc.vel - zp
        lc.zero_point_kms = float(zp)
    spectra = {"integrated": _integrated_spectra(lc)}
    # extendedness (the notebook's "relative ratio"): flux within 2 FWHM vs the 2-4 FWHM annulus
    fw_pix = lc.fwhm_arcsec / lc.cube.pixscale
    x0, y0 = lc.center_pix
    ext = {}
    for k in ("mom0", "mom0_ext"):
        if k in maps:
            fin = _aperture_flux(maps[k], x0, y0, 0.0, 2 * fw_pix); fan = _aperture_flux(maps[k], x0, y0, 2 * fw_pix, 4 * fw_pix)
            ext[k] = {"flux_inside_2fwhm": fin * 1e-3 * lc.cube.pixar_sr, "flux_annulus_2_4fwhm": fan * 1e-3 * lc.cube.pixar_sr,
                      "ratio": fin / fan if fan != 0 else np.nan}
    tot = np.nansum(np.where(maps["snr"] >= snr_min, maps["flux_W_m2"], 0.0))
    summary = {
        "target": lc.cube.name, "line": lc.line.name, "rest_um": lc.line.wave, "band": lc.cube.band,
        "cube": os.path.basename(lc.cube.path), "channels": int(len(lc.vel)), "pixscale_arcsec": lc.cube.pixscale,
        "psf_fwhm_arcsec": lc.fwhm_arcsec, "lsf_fwhm_kms": lc.lsf_fwhm_kms, "line_kms": mo["line_kms"],
        "center_radec": lc.center_radec, "center_pix": lc.center_pix, "rv_kms": lc.rv_kms,
        "band_offset_kms": lc.band_offset_kms, "zero_point": zero_point, "zero_point_kms": zp,
        "moment_window": {k: v for k, v in mo["window"].items()}, "rms_region": rms_info,
        "source_velocity_kms": vstar[0], "source_velocity_err_kms": vstar[1],
        "total_flux_W_m2_snr_masked": tot, "point_source_line_flux_W_m2": _point_flux(lc, sel),
        "extended_flux_W_m2": float(np.nansum(np.where(maps.get("snr_ext", maps["snr"]) >= snr_min,
                                                       maps.get("mom0_ext", maps["mom0"]), 0.0)) * 1e-3 * lc.cube.pixar_sr),
        "n_spaxels_snr": int(np.sum(maps["snr"] >= snr_min)),
        "n_spaxels_velocity": int(np.isfinite(maps.get("vcen", np.full(1, np.nan))).sum()),
        "extendedness": ext, "settings": lc.settings, "snr_min": snr_min, "kin_snr_min": kin_snr_min, "n_mc": n_mc,
    }
    if "stack" in lc.settings:
        # a stack is in normalised units: its maps show shapes and velocities, not fluxes
        for k in ("total_flux_W_m2_snr_masked", "point_source_line_flux_W_m2", "extended_flux_W_m2"):
            summary[k] = None
        for k in ("mom0", "mom0_err", "flux_W_m2", "gflux", "gflux_err", "peak", "mom0_masked"):
            if k in units:
                units[k] = "normalised"
        summary["stack"] = lc.settings["stack"]
    return LineMaps(lc, maps, units, spectra, summary)


def unit_kind(unit: str) -> str:
    """'cgs' (erg s-1 cm-2 sr-1), 'um' (MJy/sr x micron) or 'm' (MJy/sr x metre, the cube_maps.py files)."""
    u = str(unit).lower().replace("µ", "u").replace("micron", "um")
    for ch in " *.^()":
        u = u.replace(ch, "")
    u = u.replace("sr-1", "/sr").replace("s-1", "/s").replace("cm-2", "/cm2")
    if u == "cgs" or u.startswith("erg"):
        return "cgs"
    if u in ("native", "m"):                  # "m": the BUNIT of the old files
        return "m"
    rest = u.replace("mjy", "", 1).replace("/sr", "").replace("sr", "") if "mjy" in u else None
    if rest in ("um", "/um"):
        return "um"
    if rest in ("m", "/m"):
        return "m"
    raise ValueError(f"unknown unit {unit!r}: 'MJy/sr um', 'MJy/sr m' or 'erg s-1 cm-2 sr-1'")


UNIT_LABELS = {"cgs": "erg s-1 cm-2 sr-1", "um": "MJy um / sr", "m": "MJy m / sr"}


def native_factor(wave_um: float, unit: str = "MJy/sr um") -> float:
    """Multiply an erg s-1 cm-2 sr-1 map by this to get `unit` ("MJy/sr um" | "MJy/sr m" | "erg s-1 cm-2 sr-1")."""
    k = unit_kind(unit)
    if k == "cgs":
        return 1.0
    f = 1.0 / area_to_cgs_sr(1.0, wave_um)
    return f if k == "um" else f * 1e-6


def moments_of(lc: LineCube, D: np.ndarray, line_kms: float, sel: np.ndarray | None = None) -> np.ndarray:
    if sel is None:
        sel = np.abs(lc.vel) <= line_kms
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m0 = np.nansum(D[sel] * lc.dlam[sel][:, None, None], axis=0)
    return m0 * area_to_cgs_sr(1.0, lc.line.wave)


def _integrated_spectra(lc: LineCube) -> dict:
    """Spectra summed over all spaxels (Jy): data, continuum, line, point source, extended."""
    f = lc.cube.pixar_sr * 1e6
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d = {"wave_um": lc.cube.wave, "v_kms": lc.vel,
             "data_Jy": np.nansum(lc.data, axis=(1, 2)) * f, "continuum_Jy": np.nansum(lc.cont, axis=(1, 2)) * f,
             "line_Jy": np.nansum(lc.line_data, axis=(1, 2)) * f,
             "line_free": lc.line_free.astype(int)}
        if lc.psf is not None:
            d["point_source_Jy"] = lc.psf.point_flux_jy[0]
            d["extended_Jy"] = np.nansum(lc.psf.extended, axis=(1, 2)) * f
    return d


def _point_flux(lc: LineCube, sel: np.ndarray | None = None) -> float | None:
    if lc.psf is None:
        return None
    if sel is None:
        sel = np.abs(lc.vel) <= max(1.5 * lc.lsf_fwhm_kms, 200.0)
    area = float(np.sum(lc.psf.point_flux_jy[0][sel] * lc.dlam[sel]))
    from ..lines import area_to_W_m2
    return float(area_to_W_m2(area, lc.line.wave))


def _source_velocity(lc: LineCube, center_shift_kms: float, sigma_range) -> tuple[float, float]:
    """Centroid velocity of the line in a 1-FWHM aperture on the source."""
    from ..data import _circular_weights
    ny, nx = lc.data.shape[1:]
    w = _circular_weights(ny, nx, lc.center_pix[0], lc.center_pix[1], max(lc.fwhm_arcsec / lc.cube.pixscale, 1.0))
    y = np.nansum(np.nan_to_num(lc.line_data) * w[None], axis=(1, 2))
    e = np.sqrt(np.nansum((np.nan_to_num(lc.err) * w[None]) ** 2, axis=(1, 2)))
    from ..lines import fit_line
    lam0 = lc.line.wave * (1 + (lc.rv_kms + lc.band_offset_kms) / C_KMS)
    try:
        r = fit_line(lc.cube.wave / (lam0 / lc.line.wave), y + 0.0, e, lc.line, window_kms=float(np.max(np.abs(lc.vel))),
                     center_shift_kms=center_shift_kms, sigma_range=sigma_range, n_mc=100, flux_unit="MJy/sr")
        return (r.v_kms, r.v_err_kms) if r.detected else (np.nan, np.nan)
    except Exception:
        return (np.nan, np.nan)


# ----------------------------------------------------------------------------------------------
# Stacking several lines of one species
# ----------------------------------------------------------------------------------------------

def reproject_cube(D: np.ndarray, src: Cube, dst: Cube, order: int = 1) -> np.ndarray:
    """Resample every plane of D (on src's spaxel grid) onto dst's spaxel grid through the WCS
    (surface brightness, so no flux rescaling)."""
    from scipy.ndimage import map_coordinates
    ny, nx = dst.shape[1:]
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    ra, dec = dst.pix_to_world(xx, yy)
    xs, ys = src.world_to_pix(ra, dec)
    out = np.empty((D.shape[0], ny, nx))
    for k in range(D.shape[0]):
        p = D[k]
        good = np.isfinite(p)
        a = map_coordinates(np.where(good, p, 0.0), [ys, xs], order=order, mode="constant", cval=np.nan)
        g = map_coordinates(good.astype(float), [ys, xs], order=1, mode="constant", cval=0.0)
        a[g < 0.5] = np.nan
        out[k] = a
    return out


def stack_lines(cubes: CubeSet, lines, name: str | None = None, dv_kms: float | None = None, vmax_kms: float = 600.0,
                weighting: str = "snr", reference: str = "coarsest", use_extended: bool = False, align: str = "source",
                **prepare_kw) -> LineCube:
    """Stack several lines of one species in velocity space on a common spaxel grid (raises S/N for
    velocity maps, e.g. H2 S(1)-S(3)).

    Each line is continuum-subtracted (and PSF-subtracted if requested) on its own cube, normalised by
    its source-integrated flux, resampled to the reference grid (the coarsest spaxels by default) and to
    a common velocity axis (step = the median channel width of the lines), then averaged with 1/sigma^2
    weights ("snr") or equally ("equal").  align="source" first shifts every line so that its centroid on
    the source is at 0 km/s: this removes the sub-band wavelength-calibration offsets (a few km/s), which
    would otherwise blur the stack, and makes the stacked velocities relative to the source.  The result
    is a LineCube in normalised units: moment 0 of a stack is a shape, not a flux; velocities are real."""
    lcs = [prepare_line(cubes, ln, **prepare_kw) for ln in (get_line(x) for x in lines)]
    ref = max(lcs, key=lambda lc: lc.cube.pixscale) if reference == "coarsest" else min(lcs, key=lambda lc: lc.cube.pixscale)
    step = dv_kms or float(np.median([np.median(np.abs(np.diff(lc.vel))) for lc in lcs]))
    vgrid = np.arange(-vmax_kms, vmax_kms + 0.5 * step, step)
    num = 0.0; den = 0.0
    shifts = {}
    for lc in lcs:
        D = lc.extended if (use_extended and lc.psf is not None) else lc.line_data
        sel = np.abs(lc.vel) <= max(1.5 * lc.lsf_fwhm_kms, 200.0)
        norm = np.nansum(lc.line_data[sel] * lc.dlam[sel][:, None, None]) * area_to_cgs_sr(1.0, lc.line.wave)
        if not (np.isfinite(norm) and norm > 0):
            warnings.warn(f"stack: {lc.line.name} has no positive integrated flux; left out of the stack")
            continue
        v0 = 0.0
        if align == "source":
            v0, _ = _source_velocity(lc, 200.0, (0.8, 2.5))
            v0 = float(v0) if np.isfinite(v0) else 0.0
        shifts[lc.line.name] = v0
        # velocity resampling (linear) spaxel by spaxel, then spatial reprojection
        Dv = _resample_velocity(lc.vel - v0, D, vgrid) / norm
        Ev = _resample_velocity(lc.vel - v0, lc.err, vgrid) / norm
        if lc is not ref:
            Dv = reproject_cube(Dv, lc.cube, ref.cube); Ev = reproject_cube(Ev, lc.cube, ref.cube)
        w = 1.0 / np.where(np.isfinite(Ev) & (Ev > 0), Ev, np.inf) ** 2 if weighting == "snr" else np.ones_like(Dv)
        w = np.where(np.isfinite(Dv), w, 0.0)          # a missing channel must not pull the average to zero
        num = num + np.nan_to_num(Dv) * w
        den = den + w
    used = [lc for lc in lcs if lc.line.name in shifts]
    if not used:
        raise ValueError("stack: none of the lines has a positive integrated flux")
    with np.errstate(invalid="ignore", divide="ignore"):
        S = num / den
        E = 1.0 / np.sqrt(den) if weighting == "snr" else np.full_like(S, np.nan)
    # the continuum shown with a stack is the reference line's own (MJy/sr), not a normalised average
    Cs = _resample_velocity(ref.vel, ref.cont, vgrid)
    S[den == 0] = np.nan
    ref_ln = ref.line
    stack_line = Line(name or ("stack " + "+".join(lc.line.name for lc in used)), ref_ln.wave, ref_ln.species, ref_ln.kind,
                      note="stack of " + ", ".join(lc.line.name for lc in used))
    # fake a wavelength axis at the reference line so all 1-D machinery works (v -> lambda of the ref line)
    lam0 = ref_ln.wave * (1 + (ref.rv_kms + ref.band_offset_kms) / C_KMS)
    wave = lam0 * (1 + vgrid / C_KMS)
    cube = Cube(S.astype(np.float32), E.astype(np.float32), wave, ref.cube.wcs, ref.cube.header, ref.cube.primary, ref.cube.band,
                ref.cube.path, ref.cube.name, {"stack": [lc.line.name for lc in used]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        emp = np.nanmedian(np.where((np.abs(vgrid) > 2.5 * ref.lsf_fwhm_kms)[:, None, None], np.abs(S), np.nan), axis=0) * 1.4826
    lc = LineCube(stack_line, cube, vgrid, S, np.fmax(np.nan_to_num(E), np.nan_to_num(emp)[None]), Cs, S, None,
                  np.abs(vgrid) > 2.5 * ref.lsf_fwhm_kms, emp, ref.center_radec, ref.center_pix, ref.rv_kms, ref.band_offset_kms, 0.0,
                  {"stack": [x.line.name for x in used], "weighting": weighting, "reference": ref.line.name,
                   "use_extended": use_extended, "dv_kms": step, "align": align, "shifts_kms": shifts})
    lc.members = used
    return lc


def _resample_velocity(v: np.ndarray, D: np.ndarray, vgrid: np.ndarray) -> np.ndarray:
    nz, ny, nx = D.shape
    flat = D.reshape(nz, -1)
    out = np.full((len(vgrid), flat.shape[1]), np.nan)
    j = np.searchsorted(v, vgrid) - 1
    inside = (j >= 0) & (j < nz - 1)
    jj = np.clip(j, 0, nz - 2)
    t = ((vgrid - v[jj]) / (v[jj + 1] - v[jj]))[:, None]
    val = flat[jj] * (1 - t) + flat[jj + 1] * t
    out[inside] = val[inside]
    return out.reshape(len(vgrid), ny, nx)
