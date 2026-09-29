"""The functions of the old ``cube_maps.py`` (moment-0 maps, ratio maps, channel slices), on jalebi.cube.

Same names, arguments, file names and numbers, so old notebooks keep working after one import change::

    # from cube_maps import make_moment0, make_ratio_plot, plot_moment0_map, add_au_box
    from jalebi.cube.cube_maps import make_moment0, make_ratio_plot, plot_moment0_map, add_au_box

The recipe is the old one: the cube of the sub-band `get_channel` picks, a slab of +-dlambda micron
(the planes spectral_cube's ``spectral_slab`` selects), a per-spaxel ``aspls`` continuum
(pybaselines, lam = 5e6) through all channels of the slab, and moment 0 over native channels around the
channel nearest to the line: "full" = 9 channels (centre +-4), "slow" = 5 (centre +-2), anything else
("fast") = the 2 + 2 wing channels in between.  The pipeline DQ flags are not applied (spectral_cube
does not apply them either), and a spaxel with a NaN channel anywhere in the slab gets no continuum and
so no moment (pybaselines returns NaN for it); ``nan_policy="omit"`` keeps those edge spaxels by
fitting the continuum through the finite channels.  What the old code did not have (continuum polynomials, point-source removal,
velocity maps, errors, S/N) is in `jalebi.cube.prepare_line` / `line_maps` (the same windows are
``line_maps(..., component="full")``), the CLI (``jalebi cube moment0``, or ``--recipe cube_maps`` for
``maps / stack / ratio / channels``) and the web app (the *cube_maps.py recipe* button).

Units.  spectral_cube expresses the wavelength axis of an s3d cube in metres and the continuum-subtracted
cube had no unit, so the old moment-0 FITS files hold  sum(I_nu [MJy/sr] x d_lambda [m])  (BUNIT "m"),
which is 1e-6 x the MJy/sr um the comments mention.  ``unit="MJy/sr m"`` (default) reproduces those
numbers; ``"MJy/sr um"`` or ``"erg s-1 cm-2 sr-1"`` (x 2.998e-3 / lambda^2 of MJy/sr um) give physical
units.  A ratio of two such maps is the line-flux ratio times (lambda_1 / lambda_2)^2; ``ratio_map`` of
two LineMaps gives the flux ratio directly.

Default folders: the old defaults were paths on one computer; here they come from the environment
variables JALEBI_CUBE_DATA (cubes, `folder_path`) and JALEBI_CUBE_OUTPUT (`output_path`), else ".".
"""
from __future__ import annotations

import glob
import os
import warnings
from dataclasses import dataclass, field

import numpy as np

from ..lines import Line
from .io import channel_band, get_channel, read_cube  # noqa: F401  (get_channel is part of this API)
from .plots import add_au_box, plot_map_wcs, plot_ratio_map  # noqa: F401  (add_au_box is part of this API)

__all__ = ["get_channel", "get_continuum", "make_moment0", "get_moment0", "get_moment0_all", "make_ratio_plot",
           "plot_moment0_map", "plot_moment0_map_with_bkgd_mask", "add_au_box", "get_channel_maps", "moment0_map",
           "Moment0Map"]

DATA_ENV = "JALEBI_CUBE_DATA"
OUTPUT_ENV = "JALEBI_CUBE_OUTPUT"
ASPLS_LAM = 5e6


# ----------------------------------------------------------------------------------------------
# building blocks
# ----------------------------------------------------------------------------------------------

def get_continuum(cube: np.ndarray, wavelengths: np.ndarray, lam: float = ASPLS_LAM, n_jobs: int = 1,
                  nan_policy: str = "propagate") -> np.ndarray:
    """aspls baseline (pybaselines, lam=5e6) of every spaxel of a (nz, ny, nx) array.  A spaxel with a NaN
    channel gets a NaN baseline, as before; nan_policy="omit" fits through its finite channels instead."""
    from .continuum import CubeContinuumSettings, _baseline_column
    s = CubeContinuumSettings(method="aspls", lam=lam, n_jobs=n_jobs, nan_policy=nan_policy)
    nz, ny, nx = np.shape(cube)
    D = np.asarray(cube, float).reshape(nz, -1)
    x = np.asarray(wavelengths, float)
    if n_jobs and n_jobs != 1:
        from joblib import Parallel, delayed
        cols = Parallel(n_jobs=n_jobs)(delayed(_baseline_column)(x, D[:, p], "aspls", s) for p in range(D.shape[1]))
    else:
        cols = [_baseline_column(x, D[:, p], "aspls", s) for p in range(D.shape[1])]
    return np.stack(cols, axis=1).reshape(nz, ny, nx)


@dataclass
class Moment0Map:
    """A moment-0 map with its celestial header (stands in for spectral_cube's Projection:
    `.value`, `.wcs`, `.header`, `.hdu`, `.unit`, `.write()`, and numpy functions work on it)."""
    value: np.ndarray
    header: object
    unit: str
    line_wave: float
    transition: str = ""
    component: str = "full"
    channels: list = field(default_factory=list)
    wave_um: np.ndarray | None = None      # wavelengths of the channels summed
    cube_file: str = ""
    lc: object = None                      # the jalebi LineCube it came from

    @property
    def wcs(self):
        return _wcs(self.header)

    @property
    def hdu(self):
        from astropy.io import fits
        return fits.PrimaryHDU(np.asarray(self.value, float), header=self.header)

    @property
    def shape(self):
        return np.shape(self.value)

    def write(self, path: str, overwrite: bool = False) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.hdu.writeto(path, overwrite=overwrite)
        return path

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.value, dtype=dtype)


def _unit_label(unit: str) -> str:
    from .maps import UNIT_LABELS, unit_kind
    return UNIT_LABELS[unit_kind(unit)]


def _center(cube):
    from .io import resolve_center
    try:
        return resolve_center(cube, None)
    except Exception:
        ny, nx = cube.sci.shape[1:]
        ra, dec = cube.pix_to_world((nx - 1) / 2, (ny - 1) / 2)
        return float(ra), float(dec)


def prepare_classic(cube_file: str, line_wave: float, dlambda: float = 0.1, transition: str = "", n_jobs: int = 1,
                    continuum: dict | None = None, nan_policy: str = "propagate"):
    """The LineCube of the old recipe: slab of +-dlambda, aspls continuum, no DQ mask, no PSF removal."""
    from .maps import prepare_line
    cube = read_cube(cube_file, dq_mask=False, zero_is_nan=False)
    ln = Line(str(transition) or f"{line_wave} um", float(line_wave), "?")
    cont = {"method": "aspls", "lam": ASPLS_LAM, "n_jobs": n_jobs, "nan_policy": nan_policy, **(continuum or {})}
    return prepare_line(cube, ln, window_um=float(dlambda), continuum=cont, psf={"enabled": False}, center=_center(cube))


def moment0_map(cube_file: str, line_wave: float, dlambda: float = 0.1, component: str = "full", transition: str = "",
                unit: str = "MJy/sr m", n_jobs: int = 1, lc=None, nan_policy: str = "propagate") -> Moment0Map:
    """Moment 0 of one line from one cube with the old recipe (see the module docstring)."""
    from .. import __version__
    from .maps import moments, native_factor
    comp = component if component in ("full", "slow") else "fast"
    lc = lc if lc is not None else prepare_classic(cube_file, line_wave, dlambda, transition, n_jobs, nan_policy=nan_policy)
    mo = moments(lc, "line", component=comp, min_valid=1, snr_min=-np.inf)
    val = mo["mom0"] * native_factor(float(line_wave), unit)
    h = lc.cube.celestial_header()
    h["BUNIT"] = _unit_label(unit)
    h["LINE"] = (str(transition), "transition")
    h["RESTWAV"] = (float(line_wave), "line wavelength [um]")
    h["DLAMBDA"] = (float(dlambda), "half-width of the slab [um]")
    h["WINDOW"] = (comp, "full: centre +-4 ch, slow: +-2, fast: full-slow")
    ch = mo["window"]["channels"]
    h["NCHAN"] = (len(ch), "channels summed")
    h["CHANNELS"] = (",".join(str(c) for c in ch), "planes of the slab summed")
    h["CONTINUU"] = (f"aspls lam={lc.settings['continuum'].get('lam') or ASPLS_LAM:g}", "per-spaxel continuum")
    h["CREATOR"] = f"jalebi {__version__} (jalebi.cube.cube_maps)"
    return Moment0Map(val, h, _unit_label(unit), float(line_wave), str(transition), comp, list(ch), lc.cube.wave[mo["sel"]],
                      cube_file, lc)


def _data_dir(folder_path):
    return folder_path if folder_path is not None else os.environ.get(DATA_ENV, ".")


def _out_dir(output_path):
    return output_path if output_path is not None else os.environ.get(OUTPUT_ENV, ".")


def find_cube(folder: str, channel: str, line_wave: float, pattern: str = "Level3_{channel}_s3d.fits", warn: bool = True) -> str:
    """`folder/pattern` (the old layout); if absent, any cube of that sub-band in the folder that covers the
    line (e.g. `Level3_ch1-short_s3d.fits.gz`, or a jalebi cutout), else the jalebi choice for the line."""
    from ..examples import resolve_path
    folder = resolve_path(folder)
    p = os.path.join(folder, pattern.format(channel=channel))
    if os.path.exists(p):
        return p
    from .io import CubeSet
    cands = sorted(glob.glob(os.path.join(folder, f"*{channel}*s3d*.fits*")))
    if not cands:
        cands = sorted({f for pat in ("*s3d*.fits", "*s3d*.fits.gz") for f in glob.glob(os.path.join(folder, pat))})
    if not cands:
        raise FileNotFoundError(f"{p} not found (and no s3d cubes in {folder})")
    cs = CubeSet(cands)
    ci = cs.choose(float(line_wave), band=channel_band(channel)) if any(c.band == channel_band(channel) for c in cs.info) \
        else cs.choose(float(line_wave), band="nominal")
    if ci is None:
        raise FileNotFoundError(f"no cube in {folder} covers {line_wave} um")
    if warn and os.path.basename(ci.path) != os.path.basename(p):
        warnings.warn(f"{os.path.basename(p)} not found; using {os.path.basename(ci.path)}")
    return ci.path


def _source_folder(folder_path, SOURCE):
    if str(SOURCE).startswith("example:"):
        from ..examples import resolve_path
        return resolve_path(SOURCE), os.path.basename(resolve_path(SOURCE).rstrip("/"))
    return os.path.join(_data_dir(folder_path), str(SOURCE)), str(SOURCE)


# ----------------------------------------------------------------------------------------------
# the old API
# ----------------------------------------------------------------------------------------------

def make_moment0(lines, names, SOURCE, dlambda, component="full", save=False, folder_path=None, output_path=None,
                 unit="MJy/sr m", cube_pattern="Level3_{channel}_s3d.fits", n_jobs=1, verbose=True, nan_policy="propagate"):
    """Moment-0 maps of `lines` (micron) named `names` from `{folder_path}/{SOURCE}/Level3_{channel}_s3d.fits`.

    save=True writes `{output_path}/moment0_maps/{SOURCE}/{line_wave}_{name}_{component}_moment0.fits`.
    Returns (moment0, fname) for one line, as before (the old function stopped after the first line);
    for several lines, lists (moment0s, fnames)."""
    if len(lines) != len(names):
        raise ValueError("len(names) and len(lines) does not match")
    folder, src = _source_folder(folder_path, SOURCE)
    outdir = os.path.join(_out_dir(output_path), "moment0_maps", src)
    os.makedirs(outdir, exist_ok=True)
    maps, fnames = [], []
    for line_wave, transition in zip(lines, names):
        channel = get_channel(line_wave)
        cube_file = find_cube(folder, channel, line_wave, cube_pattern, warn=not str(SOURCE).startswith("example:"))
        if verbose:
            print(cube_file)
        m = moment0_map(cube_file, line_wave, dlambda, component, transition, unit, n_jobs, nan_policy=nan_policy)
        fname = os.path.join(outdir, f"{line_wave}_{transition}_{component}_moment0.fits")
        if save:
            m.write(fname, overwrite=True)
        maps.append(m); fnames.append(fname)
    return (maps[0], fnames[0]) if len(maps) == 1 else (maps, fnames)


def _rows(df, transition_cols=("Transitions,J", "Transitions")):
    tcol = next((c for c in transition_cols if c in df.columns), None)
    if tcol is None:
        raise KeyError(f"the table needs a column {' or '.join(repr(c) for c in transition_cols)}")
    for i in range(len(df)):
        r = df.iloc[i]
        yield r.get("Species", ""), float(r["Lab wavelength(μm)"]), r[tcol]


def get_moment0(df, path, SOURCE, dlambda, component, unit="MJy/sr m", cube_pattern="Level3_{channel}_wcs1_s3d.fits", n_jobs=1,
                nan_policy="propagate"):
    """For every row of `df` (columns 'Lab wavelength(μm)', 'Transitions,J'): moment 0 from
    `{path}/{SOURCE}/cubes/Level3_{channel}_wcs1_s3d.fits` into
    `{path}/{SOURCE}/moment0_maps/{component}/{line_wave}_{transition}_moment0.fits`.  Returns the file names."""
    folder = os.path.join(path, str(SOURCE), "cubes")
    out = []
    for _, line_wave, transition in _rows(df):
        cube_file = find_cube(folder, get_channel(line_wave), line_wave, cube_pattern)
        m = moment0_map(cube_file, line_wave, dlambda, component, str(transition), unit, n_jobs, nan_policy=nan_policy)
        out.append(m.write(os.path.join(path, str(SOURCE), "moment0_maps", str(component), f"{line_wave}_{transition}_moment0.fits"),
                           overwrite=True))
    return out


def get_moment0_all(df, path, SOURCE, dlambda, component, unit="MJy/sr m", cube_pattern="Level3_{channel}_wcs1_s3d.fits", n_jobs=1,
                    nan_policy="propagate"):
    """As get_moment0, for a table with 'Species' and 'Transitions' columns, into
    `{path}/{SOURCE}/all_species_m0_maps/{component}/{species}_{line_wave}_{transition}_moment0.fits`."""
    folder = os.path.join(path, str(SOURCE), "cubes")
    out = []
    for species, line_wave, transition in _rows(df, ("Transitions", "Transitions,J")):
        cube_file = find_cube(folder, get_channel(line_wave), line_wave, cube_pattern)
        m = moment0_map(cube_file, line_wave, dlambda, component, str(transition), unit, n_jobs, nan_policy=nan_policy)
        out.append(m.write(os.path.join(path, str(SOURCE), "all_species_m0_maps", str(component),
                                        f"{species}_{line_wave}_{transition}_moment0.fits"), overwrite=True))
    return out


def get_channel_maps(df, path, SOURCE, dlambda, channel=None, cube_pattern="Level3_{channel}_wcs1_s3d.fits", n_jobs=1,
                     nan_policy="propagate"):
    """The 9 native channels (centre +-4) of every line, continuum-subtracted, one 2-D FITS each:
    `{path}/{SOURCE}/channel_maps/ch_maps_{channel}/{line_wave}/{wavelength rounded to 4}.fits` (MJy/sr).
    channel=None picks the sub-band with `get_channel` (the old code had channel = 'ch1-long' fixed)."""
    from .channels import channel_slices
    folder = os.path.join(path, str(SOURCE), "cubes")
    out = []
    for _, line_wave, transition in _rows(df):
        ch = channel or get_channel(line_wave)
        cube_file = find_cube(folder, ch, line_wave, cube_pattern)
        lc = prepare_classic(cube_file, line_wave, dlambda, str(transition), n_jobs, nan_policy=nan_policy)
        sl = channel_slices(lc, "full", source="line")
        out += sl.write(os.path.join(path, str(SOURCE), "channel_maps", f"ch_maps_{ch}", str(line_wave)))
    return out


def make_ratio_plot(line_wavelengths, line_names, source, dlambda=0.1, percentile_intervals=[80, 80, 80], center_x=12,  # noqa: B006
                    center_y=8, radius=3, sigma_thresh=[5, 5], folder_path=None, output_path=None, component="full",  # noqa: B006
                    unit="MJy/sr m", showfig=True, savefig=None, n_jobs=1, nan_policy="propagate", verbose=True):
    """Three panels: [line 1], [line 2 reprojected onto line 1], line 1 / line 2, each masked at
    sigma_thresh x the nanstd of the RMS circle (center_x, center_y, radius; line-1 pixels).  Writes both
    moment-0 maps like make_moment0(save=True).  Returns (fig, RatioMap) — the ratio is `rm.ratio`."""
    import matplotlib.pyplot as plt
    from .ratios import ratio_map
    kw = dict(component=component, save=True, folder_path=folder_path, output_path=output_path, unit=unit, n_jobs=n_jobs,
              nan_policy=nan_policy, verbose=verbose)
    m1, f1 = make_moment0([line_wavelengths[0]], [line_names[0]], source, dlambda, **kw)
    m2, f2 = make_moment0([line_wavelengths[1]], [line_names[1]], source, dlambda, **kw)
    rm = ratio_map(m1.hdu, m2.hdu, rms_region=(center_x, center_y, radius), sigma_thresh=sigma_thresh, names=tuple(line_names),
                   waves=tuple(line_wavelengths))
    if rm.reprojected and verbose:
        print("Getting reprojected")
    fig, _ = plot_ratio_map(rm, percentile_intervals)
    if savefig:
        fig.savefig(savefig, dpi=150, bbox_inches="tight")
    if showfig:
        plt.show()
    return fig, rm


def _wcs(header):
    """Celestial WCS of a header, without astropy's 'obsfix'/'datfix' chatter."""
    from astropy.wcs import WCS, FITSFixedWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        return WCS(header).celestial


def _load_2d(fitsfile):
    """(data, header) from a FITS path, an HDUList/HDU, a Moment0Map or a spectral_cube Projection."""
    from astropy.io import fits
    if isinstance(fitsfile, (str, os.PathLike)):
        with fits.open(fitsfile) as h:
            return np.array(h[0].data, float), h[0].header.copy()
    if isinstance(fitsfile, Moment0Map):
        return np.array(fitsfile.value, float), fitsfile.header
    if isinstance(fitsfile, fits.HDUList):
        return np.array(fitsfile[0].data, float), fitsfile[0].header
    if hasattr(fitsfile, "data") and hasattr(fitsfile, "header"):
        return np.array(fitsfile.data, float), fitsfile.header
    if hasattr(fitsfile, "hdu"):
        return np.array(fitsfile.hdu.data, float), fitsfile.hdu.header
    raise TypeError(f"cannot read a map from {type(fitsfile).__name__}")


def plot_moment0_map(fitsfile, fig=None, title=None, cmap="gist_rainbow", stretch="percentile", percentile=95, vmin=None,
                     vmax=None, center_x=12, center_y=8, radius=3, sigma_clip=False, sigma_mode="rms", sigma_thresh=3,
                     show_outline=False, show_colorbar=True, figsize=(8, 6), fontsize=11, savefig=None, showfig=True, dpi=150,
                     subplot_index=111):
    """A moment-0 FITS map on a WCS axis, optionally masked at sigma_thresh x the RMS circle's
    nanstd / nanmean / nanmedian (sigma_mode).  Returns (fig, ax)."""
    import matplotlib.pyplot as plt
    data, header = _load_2d(fitsfile)
    fig, ax, _, thr = plot_map_wcs(data, _wcs(header), fig=fig, title=title, cmap=cmap, stretch=stretch,
                                   percentile=percentile, vmin=vmin, vmax=vmax, center_x=center_x, center_y=center_y,
                                   radius=radius, sigma_clip=sigma_clip, sigma_mode=sigma_mode, sigma_thresh=sigma_thresh,
                                   show_outline=show_outline, show_colorbar=show_colorbar, figsize=figsize, fontsize=fontsize,
                                   subplot_index=subplot_index)
    if sigma_clip:
        print(f"Applied {sigma_thresh}σ threshold: {thr:.3g}")
    if savefig:
        fig.savefig(savefig, dpi=dpi, bbox_inches="tight")
    if showfig:
        plt.show()
    return fig, ax


def plot_moment0_map_with_bkgd_mask(fits_file, fig=None, line_name="Line", use_wcs=True, apply_mask=True, sigma_thresh=3,
                                    box_size=5, filter_size=(3, 3), stretch_percentile=90, cmap="gist_rainbow", showfig=True):
    """A moment-0 map masked at sigma_thresh x nanstd(Background2D background) (photutils when installed,
    else jalebi's numpy version of it).  Returns (fig, ax)."""
    import matplotlib.pyplot as plt
    from astropy.visualization import PercentileInterval
    from .masks import background_mask
    data, header = _load_2d(fits_file)
    masked, thr, _ = background_mask(data, sigma_thresh, box_size, filter_size)
    print(f"[{line_name}] Threshold applied = {thr}")
    img = masked if apply_mask else data.copy()
    fin = img[np.isfinite(img)]
    vmin, vmax = PercentileInterval(stretch_percentile).get_limits(fin) if fin.size else (0.0, 1.0)
    fig = fig if fig is not None else plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, projection=_wcs(header)) if use_wcs else fig.add_subplot(1, 1, 1)
    im = ax.imshow(img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f"{line_name} Moment-0 ({sigma_thresh}σ Masked)" if apply_mask else f"{line_name} Moment-0")
    ax.set_xlabel("RA" if use_wcs else "X (pixels)")
    ax.set_ylabel("Dec" if use_wcs else "Y (pixels)")
    fig.colorbar(im, ax=ax, label="Flux")
    fig.tight_layout()
    if showfig:
        plt.show()
    return fig, ax
