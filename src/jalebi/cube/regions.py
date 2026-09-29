"""Sky regions (circle, ellipse, polygon, annulus) and region spectra.

A region drawn on any map becomes a spectrum over *all* sub-bands (each cube uses its own WCS), which
goes straight into the normal slab fit, so region-by-region LTE fits come for free:

    reg  = CircleRegion(ra, dec, r=0.4)            # or EllipseRegion / PolygonRegion / parse_ds9(...)
    spec = region_spectrum("example:HV_Tau_C_cube", reg)       # jalebi.Spectrum in Jy
    run  = jalebi.run_pipeline(cfg, spec=spec)

Angles: position angles are degrees east of north (counter-clockwise on the sky with north up and east
left).  Sizes are in arcsec.  Regions read and write DS9 region strings in fk5, so they move between
JALEBI, DS9 and CARTA.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import numpy as np

from ..data import BAND_ORDER, Spectrum
from .io import Cube, CubeSet


def _tangent(ra, dec, ra0, dec0):
    dra = ((np.asarray(ra) - ra0 + 180.0) % 360.0) - 180.0
    return dra * np.cos(np.deg2rad(dec0)) * 3600.0, (np.asarray(dec) - dec0) * 3600.0


def _from_tangent(dx, dy, ra0, dec0):
    return ra0 + np.asarray(dx) / 3600.0 / np.cos(np.deg2rad(dec0)), dec0 + np.asarray(dy) / 3600.0


@dataclass
class Region:
    kind = "region"

    def center(self) -> tuple[float, float]:
        raise NotImplementedError

    def contains(self, dx, dy) -> np.ndarray:
        """Points given as east/north offsets (arcsec) from center() -> inside?"""
        raise NotImplementedError

    def weights(self, cube: Cube, oversample: int = 5) -> np.ndarray:
        """Fraction of each spaxel inside the region (sub-pixel sampling)."""
        ny, nx = cube.shape[1:]
        o = oversample
        yy, xx = np.mgrid[0:ny * o, 0:nx * o].astype(float)
        xs = (xx + 0.5) / o - 0.5; ys = (yy + 0.5) / o - 0.5
        ra, dec = cube.pix_to_world(xs, ys)
        dx, dy = _tangent(ra, dec, *self.center())
        inside = self.contains(dx, dy)
        return inside.reshape(ny, o, nx, o).mean(axis=(1, 3))

    def outline(self, n: int = 120) -> tuple[np.ndarray, np.ndarray]:
        """RA, Dec of the outline (for plotting)."""
        raise NotImplementedError

    def to_dict(self) -> dict:
        return {"shape": self.kind, **asdict(self)}

    def to_ds9(self) -> str:
        raise NotImplementedError

    def area_arcsec2(self) -> float:
        raise NotImplementedError


@dataclass
class CircleRegion(Region):
    ra: float
    dec: float
    r: float                     # arcsec
    kind = "circle"

    def center(self):
        return self.ra, self.dec

    def contains(self, dx, dy):
        return dx * dx + dy * dy <= self.r * self.r

    def outline(self, n=120):
        t = np.linspace(0, 2 * np.pi, n)
        return _from_tangent(self.r * np.sin(t), self.r * np.cos(t), self.ra, self.dec)

    def to_ds9(self):
        return f'circle({self.ra:.7f},{self.dec:.7f},{self.r:.4f}")'

    def area_arcsec2(self):
        return np.pi * self.r ** 2


@dataclass
class AnnulusRegion(Region):
    ra: float
    dec: float
    r_in: float
    r_out: float
    kind = "annulus"

    def center(self):
        return self.ra, self.dec

    def contains(self, dx, dy):
        r2 = dx * dx + dy * dy
        return (r2 >= self.r_in ** 2) & (r2 <= self.r_out ** 2)

    def outline(self, n=120):
        t = np.linspace(0, 2 * np.pi, n)
        a = _from_tangent(self.r_out * np.sin(t), self.r_out * np.cos(t), self.ra, self.dec)
        b = _from_tangent(self.r_in * np.sin(t[::-1]), self.r_in * np.cos(t[::-1]), self.ra, self.dec)
        return np.concatenate([a[0], [np.nan], b[0]]), np.concatenate([a[1], [np.nan], b[1]])

    def to_ds9(self):
        return f'annulus({self.ra:.7f},{self.dec:.7f},{self.r_in:.4f}",{self.r_out:.4f}")'

    def area_arcsec2(self):
        return np.pi * (self.r_out ** 2 - self.r_in ** 2)


@dataclass
class EllipseRegion(Region):
    ra: float
    dec: float
    a: float                     # semi-major axis (arcsec)
    b: float                     # semi-minor axis (arcsec)
    pa: float = 0.0              # major axis, degrees east of north
    kind = "ellipse"

    def center(self):
        return self.ra, self.dec

    def _rot(self, dx, dy):
        t = np.deg2rad(self.pa)
        u = dx * np.sin(t) + dy * np.cos(t)          # along the major axis
        w = dx * np.cos(t) - dy * np.sin(t)          # across
        return u, w

    def contains(self, dx, dy):
        u, w = self._rot(dx, dy)
        return (u / self.a) ** 2 + (w / self.b) ** 2 <= 1.0

    def outline(self, n=120):
        s = np.linspace(0, 2 * np.pi, n)
        u, w = self.a * np.cos(s), self.b * np.sin(s)
        t = np.deg2rad(self.pa)
        dx = u * np.sin(t) + w * np.cos(t); dy = u * np.cos(t) - w * np.sin(t)
        return _from_tangent(dx, dy, self.ra, self.dec)

    def to_ds9(self):
        # DS9 ellipse angle: counter-clockwise from the +x (west->east is -x) axis, i.e. from north + 90
        return f'ellipse({self.ra:.7f},{self.dec:.7f},{self.a:.4f}",{self.b:.4f}",{(self.pa + 90.0) % 360:.3f})'

    def area_arcsec2(self):
        return np.pi * self.a * self.b


@dataclass
class PolygonRegion(Region):
    vertices: list               # [(ra, dec), ...] degrees
    kind = "polygon"

    def center(self):
        v = np.asarray(self.vertices, float)
        return float(np.mean(v[:, 0])), float(np.mean(v[:, 1]))

    def contains(self, dx, dy):
        from matplotlib.path import Path
        ra0, dec0 = self.center()
        v = np.asarray(self.vertices, float)
        px, py = _tangent(v[:, 0], v[:, 1], ra0, dec0)
        path = Path(np.column_stack([px, py]))
        pts = np.column_stack([np.ravel(dx), np.ravel(dy)])
        return path.contains_points(pts).reshape(np.shape(dx))

    def outline(self, n=0):
        v = np.asarray(self.vertices, float)
        return np.append(v[:, 0], v[0, 0]), np.append(v[:, 1], v[0, 1])

    def to_ds9(self):
        return "polygon(" + ",".join(f"{a:.7f},{b:.7f}" for a, b in self.vertices) + ")"

    def area_arcsec2(self):
        ra0, dec0 = self.center()
        v = np.asarray(self.vertices, float)
        x, y = _tangent(v[:, 0], v[:, 1], ra0, dec0)
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


# ---- construction helpers ----------------------------------------------------------------------

def offset_region(kind: str, center_radec, *params) -> Region:
    """A region given as offsets (arcsec, east/north) from a centre, e.g. the star:
    offset_region("circle", star, dx, dy, r); ("ellipse", star, dx, dy, a, b, pa);
    ("polygon", star, dx1, dy1, dx2, dy2, ...); ("annulus", star, dx, dy, r_in, r_out)."""
    ra0, dec0 = center_radec
    p = [float(x) for x in params]
    if kind == "polygon":
        xy = np.asarray(p).reshape(-1, 2)
        ra, dec = _from_tangent(xy[:, 0], xy[:, 1], ra0, dec0)
        return PolygonRegion([(float(a), float(b)) for a, b in zip(ra, dec)])
    ra, dec = _from_tangent(p[0], p[1], ra0, dec0)
    ra, dec = float(ra), float(dec)
    if kind == "circle":
        return CircleRegion(ra, dec, p[2])
    if kind == "ellipse":
        return EllipseRegion(ra, dec, p[2], p[3], p[4] if len(p) > 4 else 0.0)
    if kind == "annulus":
        return AnnulusRegion(ra, dec, p[2], p[3])
    raise ValueError(f"unknown region shape {kind!r}")


def region_from_dict(d: dict) -> Region:
    d = dict(d)
    kind = d.pop("shape", d.pop("kind", "circle"))
    if "ds9" in d:
        return parse_ds9(d["ds9"])[0]
    if "offset" in d:          # {"shape": "circle", "offset": [dx, dy], "r": 0.4, "center": [ra, dec]}
        raise ValueError("offset regions need a centre: use offset_region()")
    return {"circle": CircleRegion, "ellipse": EllipseRegion, "polygon": PolygonRegion, "annulus": AnnulusRegion}[kind](**d)


def _ds9_len(s: str) -> float:
    s = s.strip()
    if s.endswith('"'):
        return float(s[:-1])
    if s.endswith("'"):
        return float(s[:-1]) * 60.0
    if s.endswith("d"):
        return float(s[:-1]) * 3600.0
    return float(s) * 3600.0            # DS9 default for fk5 is degrees


def parse_ds9(text: str) -> list[Region]:
    """Circles, ellipses, polygons and annuli of a DS9 region file/string in fk5/icrs (degrees or sexagesimal)."""
    from ..data import parse_radec
    out = []
    for raw in re.split(r"[\n;]", text):
        line = raw.split("#")[0].strip()
        m = re.match(r"(circle|ellipse|polygon|annulus)\s*\((.*)\)", line, re.I)
        if not m:
            continue
        kind, args = m.group(1).lower(), [a.strip() for a in m.group(2).split(",")]
        if kind == "polygon":
            v = [parse_radec(args[i], args[i + 1]) for i in range(0, len(args) - 1, 2)]
            out.append(PolygonRegion(v))
            continue
        ra, dec = parse_radec(args[0], args[1])
        if kind == "circle":
            out.append(CircleRegion(ra, dec, _ds9_len(args[2])))
        elif kind == "annulus":
            out.append(AnnulusRegion(ra, dec, _ds9_len(args[2]), _ds9_len(args[3])))
        else:
            ang = float(args[4]) if len(args) > 4 else 0.0
            out.append(EllipseRegion(ra, dec, _ds9_len(args[2]), _ds9_len(args[3]), (ang - 90.0) % 360.0))
    return out


def to_ds9_file(regions, path: str):
    with open(path, "w") as fh:
        fh.write("# Region file format: DS9 version 4.1 (written by jalebi.cube)\nfk5\n")
        for r in regions if isinstance(regions, (list, tuple)) else [regions]:
            fh.write(r.to_ds9() + "\n")


# ---- region spectra ------------------------------------------------------------------------------

def region_spectrum(cubes: CubeSet | str, region: Region, name: str | None = None, distance_pc: float = 140.0,
                    bands=None, min_coverage: float = 0.8, background: Region | None = None) -> Spectrum:
    """Sum the cubes over a region in every sub-band -> Spectrum (Jy) ready for the slab fit.

    Each cube uses its own WCS, so the same patch of sky is extracted in every sub-band.  Planes where
    less than `min_coverage` of the region has data are NaN; partial coverage above that is scaled
    up.  `background`: optional region whose median surface brightness per plane is subtracted.
    No aperture correction is applied (the right choice for extended emission; for a point source use
    `jalebi.data.load_s3d_folder`)."""
    if isinstance(cubes, str):
        cubes = CubeSet(cubes)
    W, F, E, B = [], [], [], []
    per_band = {}
    for ci in cubes.info:
        if bands and ci.band not in bands:
            continue
        c = cubes.load(ci.path)
        w = region.weights(c)
        if w.sum() <= 0:
            continue
        good = np.isfinite(c.sci)
        cov = np.einsum("zyx,yx->z", good.astype(float), w) / w.sum()
        sci = np.where(good, c.sci, 0.0).astype(float)
        if background is not None:
            wb = background.weights(c) > 0.5
            with np.errstate(all="ignore"):
                bg = np.nanmedian(np.where(wb[None] & good, c.sci, np.nan), axis=(1, 2))
            sci = sci - np.nan_to_num(bg)[:, None, None] * good
        f = np.einsum("zyx,yx->z", sci, w)
        e = np.sqrt(np.einsum("zyx,yx->z", np.where(np.isfinite(c.err), c.err.astype(float), 0.0) ** 2, w ** 2))
        with np.errstate(all="ignore"):
            f = np.where(cov >= min_coverage, f / cov, np.nan) * c.pixar_sr * 1e6
            e = np.where(cov >= min_coverage, e / cov, np.nan) * c.pixar_sr * 1e6
        W.append(c.wave); F.append(f); E.append(e); B.append(np.full(len(c.wave), ci.band, dtype=object))
        per_band[ci.band] = {"n_spaxels": float(w.sum()), "median_coverage": float(np.nanmedian(cov))}
    if not W:
        raise ValueError("the region does not overlap any cube")
    wave = np.concatenate(W); flux = np.concatenate(F); err = np.concatenate(E); band = np.concatenate(B).astype(str)
    order = np.lexsort((wave, np.array([BAND_ORDER.index(b) if b in BAND_ORDER else 99 for b in band])))
    meta = {"extraction": {"source": "s3d-region", "region": region.to_dict(), "ds9": region.to_ds9(),
                           "area_arcsec2": region.area_arcsec2(), "per_band": per_band,
                           "background": None if background is None else background.to_ds9()}}
    s = Spectrum(wave[order], flux[order], err[order], band[order], name or f"{cubes.name} [{region.kind}]", distance_pc, meta=meta)
    from ..instrument import _BANDS
    for b, (lo, hi) in _BANDS.items():
        i = s.band_slice(b)
        if len(i):
            s.mask[i[(s.wave[i] < lo - 0.02) | (s.wave[i] > hi + 0.02)]] = False
    return s
