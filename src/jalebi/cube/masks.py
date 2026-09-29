"""Masks for 2-D maps: the empty-sky circle of cube_maps.py and a Background2D-style background.

    masked, thr, circ = rms_mask(mom0, center_x=12, center_y=8, radius=3, sigma_thresh=3)   # plot_moment0_map
    masked, thr, bkg  = background_mask(mom0, sigma_thresh=3, box_size=5, filter_size=(3, 3))

Both keep the pixels at or above a threshold and set the rest to NaN; they are a display/ratio mask and do
not replace the propagated errors of the maps (`snr`).  Pixel coordinates are 0-based (x = column,
y = row), as numpy and matplotlib's `imshow(origin="lower")` use them.
"""
from __future__ import annotations

import warnings

import numpy as np


def circle_mask(shape, center_x: float, center_y: float, radius: float) -> np.ndarray:
    """Boolean mask of the pixels whose centres are within `radius` of (center_x, center_y)."""
    y, x = np.indices(shape)
    return (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2


def check_circle(shape, center_x: float, center_y: float, radius: float, warn: bool = True):
    """Raise if the centre is outside the image, warn if the circle runs over the edge (as cube_maps.py)."""
    ny, nx = shape
    if not (0 <= center_x < nx) or not (0 <= center_y < ny):
        raise ValueError(f"Center coordinates ({center_x}, {center_y}) are out of bounds for image size ({nx}, {ny}).")
    if warn and (center_x - radius < 0 or center_x + radius >= nx or center_y - radius < 0 or center_y + radius >= ny):
        warnings.warn(f"RMS circle at ({center_x}, {center_y}) with radius {radius} extends outside the image bounds ({nx}, {ny}).")


def region_level(img: np.ndarray, center_x: float = 12, center_y: float = 8, radius: float = 3, mode: str = "rms") -> float:
    """Noise level in the circle: "rms" = nanstd, "mean" = nanmean, "median" = nanmedian of the pixels."""
    m = circle_mask(img.shape, center_x, center_y, radius)
    vals = np.asarray(img, float)[m]
    nfin = int(np.isfinite(vals).sum())
    if nfin < 5:
        warnings.warn(f"the RMS circle ({center_x}, {center_y}, r={radius}) has only {nfin} finite pixels "
                      f"of {vals.size}: move it onto empty sky inside the field")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if mode == "rms":
            return float(np.nanstd(vals))
        if mode == "mean":
            return float(np.nanmean(vals))
        if mode == "median":
            return float(np.nanmedian(vals))
    raise ValueError(f"mode must be 'rms', 'mean' or 'median', not {mode!r}")


def rms_mask(img: np.ndarray, center_x: float = 12, center_y: float = 8, radius: float = 3, sigma_thresh: float = 3.0,
             mode: str = "rms", check: bool = True):
    """Keep pixels >= sigma_thresh x region_level(...): returns (masked image, threshold, circle mask)."""
    img = np.asarray(img, float)
    if check:
        check_circle(img.shape, center_x, center_y, radius)
    thr = sigma_thresh * region_level(img, center_x, center_y, radius, mode)
    with np.errstate(invalid="ignore"):
        out = np.where(img >= thr, img, np.nan)
    return out, float(thr), circle_mask(img.shape, center_x, center_y, radius)


# ----------------------------------------------------------------------------------------------
# Background2D
# ----------------------------------------------------------------------------------------------

def background2d(img: np.ndarray, box_size=5, filter_size=(3, 3), sigma: float = 3.0, exact: bool | None = None) -> np.ndarray:
    """Smooth background map: sigma-clipped median in boxes of `box_size`, median-filtered over
    `filter_size` boxes and interpolated back to the full grid.

    With photutils installed (and exact is not False) this is photutils' Background2D with
    MedianBackground and SigmaClip(sigma), as in cube_maps.py.  Otherwise a numpy/scipy transcription
    of photutils 3.0's algorithm (`_background2d_numpy`); other photutils versions differ in details
    (older ones padded the image and filled empty boxes differently)."""
    img = np.asarray(img, float)
    if exact is not False:
        try:
            from astropy.stats import SigmaClip
            from photutils.background import Background2D, MedianBackground
        except ImportError:
            if exact:
                raise
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bkg = Background2D(img, box_size=box_size, filter_size=filter_size, sigma_clip=SigmaClip(sigma=sigma),
                                   bkg_estimator=MedianBackground())
            return np.asarray(bkg.background, float)
    return _background2d_numpy(img, box_size, filter_size, sigma)


def _background2d_numpy(img, box_size, filter_size, sigma, exclude_percentile: float = 10.0, n_neighbors: int = 10):
    """photutils 3.0's Background2D (MedianBackground, astropy SigmaClip(sigma) = 5 iterations, BkgZoomInterpolator)
    step by step: boxes (the remainder boxes count their missing pixels as masked), a box is kept when more
    than (100 - exclude_percentile) % of its pixels survive the clipping, empty boxes are filled by
    inverse-distance weighting of the 10 nearest boxes, a nanmedian filter, then a cubic zoom clipped to
    the mesh range."""
    from astropy.stats import sigma_clip
    from scipy.ndimage import generic_filter, zoom
    by, bx = (int(box_size), int(box_size)) if np.isscalar(box_size) else tuple(int(b) for b in box_size)
    fy, fx = (int(filter_size), int(filter_size)) if np.isscalar(filter_size) else tuple(int(f) for f in filter_size)
    if fy % 2 == 0 or fx % 2 == 0:
        raise ValueError("filter_size must be odd along both axes (as photutils requires)")
    ny, nx = img.shape
    py, px = (-ny) % by, (-nx) % bx
    a = np.pad(np.where(np.isfinite(img), img, np.nan), ((0, py), (0, px)), constant_values=np.nan)
    my, mx = a.shape[0] // by, a.shape[1] // bx
    blocks = a.reshape(my, by, mx, bx).transpose(0, 2, 1, 3).reshape(my, mx, by * bx)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clipped = sigma_clip(blocks, sigma=sigma, maxiters=5, axis=-1, masked=False)    # clipped -> NaN
        mesh = np.nanmedian(clipped, axis=-1)
    ngood = np.isfinite(clipped).sum(axis=-1)
    mesh[ngood <= (1 - exclude_percentile / 100.0) * by * bx] = np.nan
    if not np.isfinite(mesh).any():
        raise ValueError(f"All boxes contain <= {(1 - exclude_percentile / 100.0) * by * bx:g} unmasked or finite pixels "
                         f"(box_size={box_size}): use a smaller box_size")
    bad = ~np.isfinite(mesh)
    if bad.any():
        from scipy.spatial import cKDTree
        gy, gx = np.nonzero(~bad)
        tree = cKDTree(np.column_stack([gy, gx]), leafsize=10)      # photutils' leafsize: same neighbours on ties
        qy, qx = np.nonzero(bad)
        k = min(n_neighbors, len(gy))
        d, j = tree.query(np.column_stack([qy, qx]), k=k)
        d = np.atleast_2d(d.T).T if k == 1 else d
        j = np.atleast_2d(j.T).T if k == 1 else j
        w = 1.0 / d
        mesh = mesh.copy()
        mesh[qy, qx] = np.sum(w * mesh[gy[j], gx[j]], axis=1) / np.sum(w, axis=1)
    if (fy, fx) != (1, 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mesh = generic_filter(mesh, np.nanmedian, size=(fy, fx), mode="constant", cval=np.nan)
    if np.ptp(mesh) == 0:
        return np.full(img.shape, float(mesh.min()))
    full = zoom(mesh, (by, bx), order=3, mode="reflect", grid_mode=True)[:ny, :nx]
    return np.clip(full, mesh.min(), mesh.max())


def background_mask(img: np.ndarray, sigma_thresh: float = 3.0, box_size=5, filter_size=(3, 3), sigma: float = 3.0,
                    exact: bool | None = None):
    """cube_maps.py's `plot_moment0_map_with_bkgd_mask` mask: keep pixels >= sigma_thresh x nanstd of the
    Background2D map.  Returns (masked image, threshold, background map)."""
    img = np.asarray(img, float)
    bkg = background2d(img, box_size, filter_size, sigma, exact)
    thr = sigma_thresh * float(np.nanstd(bkg))
    with np.errstate(invalid="ignore"):
        out = np.where(img >= thr, img, np.nan)
    return out, thr, bkg
