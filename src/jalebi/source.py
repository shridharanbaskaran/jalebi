"""A *source*: one target folder (x1d spectra and/or s3d cubes of one disk), opened once and shared.

Every analysis in JALEBI starts from the same data: the LTE slab fit needs a 1-D spectrum (pipeline x1d or an
aperture on the cubes), the cube maps need the cubes, the rotation diagram needs a region spectrum of the cubes
or the x1d.  `Source` reads each of these at most once and keeps them in memory, so switching between the
analyses (or re-opening the web app in another browser tab) never reads the FITS files again:

    from jalebi.source import open_source
    src = open_source("YSOs_MIRI_reduced_cube_data_Aug2026/disk_only/V-HV-TAU-C_...")
    src.preload(workers=6)                       # all 12 cubes into memory, in parallel
    spec = src.spectrum("x1d")                   # -> LTE slab fit
    spec = src.spectrum("s3d", dict(aperture_fwhm_scale=1.5, apcorr="mrs"))   # aperture at the source
    cs = src.cubes                               # -> jalebi.cube (line maps, regions ...)
    reg = src.region_spectrum(offset_region("circle", src.position()[:2], 0, 0, 1.0))  # -> rotation diagram

`open_source` keeps the last few sources of the process (`JALEBI_SOURCE_CACHE`, default 3) so that the web app's
sessions share them.  Settings you type once per source (distance, RV, aperture centre, notes) are remembered in
`jalebi_source.yaml` next to the data, or in ~/.jalebi/sources/ when the data folder is read-only.

Terminal:  jalebi source list ROOT · jalebi source info PATH [--preload] · jalebi source set PATH --distance 140 --rv 16
"""
from __future__ import annotations

import glob
import hashlib
import os
import re
import threading
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from .data import BAND_ORDER, Spectrum, extract_cube, load_csv, parse_radec, read_x1d

SOURCE_FILE = "jalebi_source.yaml"
X1D_PATTERNS = ("*x1d*.fits", "*x1d*.fits.gz")
S3D_PATTERNS = ("*s3d*.fits", "*s3d*.fits.gz", "*_cube*.fits")
TABLE_EXT = (".csv", ".txt", ".dat", ".csv.gz", ".h5", ".hdf5")
_BAND_RE = re.compile(r"ch(\d)-(short|medium|long)", re.I)
_LETTER = {"short": "A", "medium": "B", "long": "C"}


def _cache_mb() -> float:
    return float(os.environ.get("JALEBI_CUBE_CACHE_MB", 6000))


def _band_from_name(path: str) -> str:
    m = _BAND_RE.search(os.path.basename(path))
    return f"{m.group(1)}{_LETTER[m.group(2).lower()]}" if m else ""


def _glob_any(folder: str, patterns) -> list[str]:
    return sorted({f for p in patterns for f in glob.glob(os.path.join(folder, p))})


# ------------------------------------------------------------------------------------------------
# finding sources
# ------------------------------------------------------------------------------------------------

@dataclass
class SourceFiles:
    """What a source path contains."""
    path: str
    kind: str                                   # folder | cube-file | x1d-file | table
    x1d: dict = field(default_factory=dict)     # band -> x1d file
    s3d: list = field(default_factory=list)     # cube files
    tables: list = field(default_factory=list)  # CSV / HDF5 spectra

    @property
    def n_x1d(self) -> int:
        return len(self.x1d)

    @property
    def n_s3d(self) -> int:
        return len(self.s3d)

    @property
    def empty(self) -> bool:
        return not (self.x1d or self.s3d or self.tables)


def inspect_path(path: str) -> SourceFiles:
    """Classify a path: a target folder (x1d and/or s3d files), one cube, one x1d file or a table."""
    from .examples import resolve_path
    p = resolve_path(path)
    if os.path.isdir(p):
        x1d = {}
        for f in _glob_any(p, X1D_PATTERNS):
            b = _band_from_name(f) or os.path.basename(f)
            x1d.setdefault(b, f)
        tables = [f for f in sorted(glob.glob(os.path.join(p, "*"))) if f.lower().endswith(TABLE_EXT)
                  and not os.path.basename(f).startswith(("jalebi_source", "."))]
        return SourceFiles(p, "folder", x1d, _glob_any(p, S3D_PATTERNS), tables)
    if not os.path.exists(p):
        raise FileNotFoundError(f"source not found: {path}")
    low = p.lower()
    if "s3d" in os.path.basename(low) or "_cube" in os.path.basename(low):
        return SourceFiles(p, "cube-file", s3d=[p])
    if low.endswith((".fits", ".fits.gz")):
        return SourceFiles(p, "x1d-file", x1d={_band_from_name(p) or "x1d": p})
    return SourceFiles(p, "table", tables=[p])


@dataclass
class SourceEntry:
    """One source found under a data root (see `scan_sources`)."""
    name: str
    path: str
    n_x1d: int
    n_s3d: int
    n_tables: int
    loaded: bool = False

    @property
    def label(self) -> str:
        parts = []
        if self.n_x1d:
            parts.append(f"{self.n_x1d} x1d")
        if self.n_s3d:
            parts.append(f"{self.n_s3d} s3d")
        if self.n_tables and not (self.n_x1d or self.n_s3d):
            parts.append("table")
        return f"{'● ' if self.loaded else ''}{self.name}  —  {' · '.join(parts)}"


def scan_sources(root: str, depth: int = 2) -> list[SourceEntry]:
    """Target folders (with x1d or s3d files) and spectrum tables under `root`, `depth` levels deep.
    The root itself counts when it holds data.  Sources already open in this process are marked `loaded`."""
    from .examples import resolve_path
    root = resolve_path(os.path.expanduser(root))
    out: list[SourceEntry] = []
    if not os.path.isdir(root):
        return out
    open_paths = {s.path for s in _SOURCES.values()}
    dirs = [root]
    level = [root]
    for _ in range(depth):
        nxt = []
        for d in level:
            try:
                nxt += sorted(os.path.join(d, e) for e in os.listdir(d) if not e.startswith("."))
            except OSError:
                continue
        level = [d for d in nxt if os.path.isdir(d)]
        dirs += level
        for f in nxt:
            if os.path.isfile(f) and f.lower().endswith(TABLE_EXT) and not os.path.basename(f).startswith("jalebi_source"):
                rel = os.path.relpath(f, root)
                out.append(SourceEntry(rel, f, 0, 0, 1, f in open_paths))
    for d in dirs:
        nx, ns = len(_glob_any(d, X1D_PATTERNS)), len(_glob_any(d, S3D_PATTERNS))
        if nx or ns:
            rel = os.path.relpath(d, root)
            out.append(SourceEntry(os.path.basename(root.rstrip("/")) if rel == "." else rel, d, nx, ns, 0, d in open_paths))
    out.sort(key=lambda e: (e.n_x1d + e.n_s3d == 0, e.name.lower()))
    return out


# ------------------------------------------------------------------------------------------------
# the source
# ------------------------------------------------------------------------------------------------

def _settings_paths(path: str, name: str) -> list[str]:
    folder = path if os.path.isdir(path) else os.path.dirname(path)
    h = hashlib.sha1(os.path.abspath(path).encode()).hexdigest()[:8]
    safe = re.sub(r"[^A-Za-z0-9_.+-]+", "_", name).strip("_") or "source"
    return [os.path.join(folder, SOURCE_FILE), os.path.join(os.path.expanduser("~"), ".jalebi", "sources", f"{safe}_{h}.yaml")]


class Source:
    """One target, opened once.  All products are memoised; spectra are returned as copies, so a module
    can change its own (rest-frame shift, continuum, masks) without affecting the others."""

    def __init__(self, path: str, dq_mask: bool = True, zero_is_nan: bool = True, name: str | None = None):
        self.files = inspect_path(path)
        if self.files.empty:
            raise FileNotFoundError(f"no x1d files, s3d cubes or spectrum tables in {path}")
        self.path = self.files.path
        self.kind = self.files.kind
        self.read_kw = dict(dq_mask=dq_mask, zero_is_nan=zero_is_nan)
        self.opened_at = time.time()
        self._lock = threading.RLock()
        self._cubes = None
        self._memo: dict = {}
        self._meta: dict | None = None
        self._name = name
        self.settings: dict = {}
        self.settings_file: str | None = None
        self._load_settings()

    # ---- identity ---------------------------------------------------------------------------
    @property
    def key(self) -> tuple:
        return (os.path.abspath(self.path), self.read_kw["dq_mask"], self.read_kw["zero_is_nan"])

    @property
    def meta(self) -> dict:
        """Header facts (TARGNAME, TARG_RA/DEC, PROGRAM, OBSERVTN, DATE-OBS, CAL_VER, CRDS_CTX ...) from the first
        x1d or cube primary header."""
        if self._meta is None:
            from astropy.io import fits
            m = {}
            first = next(iter(self.files.x1d.values()), None) or (self.files.s3d[0] if self.files.s3d else None)
            if first is not None:
                try:
                    h = fits.getheader(first, 0)
                    m = {k: h.get(k) for k in ("TARGNAME", "TARGPROP", "TARG_RA", "TARG_DEC", "PROGRAM", "OBSERVTN", "DATE-OBS",
                                               "CAL_VER", "CRDS_CTX", "CRDS_VER", "INSTRUME", "PATTTYPE", "DETECTOR")
                         if h.get(k) not in (None, "")}
                except Exception:
                    m = {}
            self._meta = m
        return self._meta

    @property
    def name(self) -> str:
        if self._name:
            return self._name
        if self.settings.get("name"):
            return str(self.settings["name"])
        m = self.meta
        if m.get("TARGNAME") or m.get("TARGPROP"):
            return str(m.get("TARGNAME") or m.get("TARGPROP")).strip()
        base = self.path.rstrip("/")
        return os.path.basename(base if os.path.isdir(base) else os.path.splitext(base)[0])

    @property
    def has_x1d(self) -> bool:
        return bool(self.files.x1d)

    @property
    def has_cubes(self) -> bool:
        return bool(self.files.s3d)

    @property
    def has_table(self) -> bool:
        return bool(self.files.tables)

    @property
    def has_1d(self) -> bool:
        return self.has_x1d or self.has_table

    @property
    def x1d_bands(self) -> list[str]:
        return [b for b in BAND_ORDER if b in self.files.x1d] + sorted(set(self.files.x1d) - set(BAND_ORDER))

    @property
    def cube_bands(self) -> list[str]:
        return self.cubes.bands if self.has_cubes else []

    # ---- remembered settings ----------------------------------------------------------------
    def _load_settings(self):
        import yaml
        for p in _settings_paths(self.path, self.name if self._name or self.meta else os.path.basename(self.path)):
            if os.path.isfile(p):
                try:
                    with open(p) as fh:
                        self.settings = yaml.safe_load(fh) or {}
                    self.settings_file = p
                    return
                except Exception as ex:
                    warnings.warn(f"could not read {p}: {ex}")

    def save_settings(self, **kw) -> str:
        """Remember settings for this source (None values are removed).  Written next to the data, or to
        ~/.jalebi/sources/ when the data folder is read-only.  Returns the file written."""
        import yaml
        with self._lock:
            for k, v in kw.items():
                if v is None or v == "":
                    self.settings.pop(k, None)
                else:
                    self.settings[k] = float(v) if isinstance(v, (np.floating,)) else v
            text = "# JALEBI source settings (written by the app / `jalebi source set`)\n" + yaml.safe_dump(self.settings, sort_keys=True)
            for p in ([self.settings_file] if self.settings_file else []) + _settings_paths(self.path, self.name):
                try:
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    with open(p, "w") as fh:
                        fh.write(text)
                    self.settings_file = p
                    return p
                except OSError:
                    continue
        raise OSError("could not write the source settings anywhere")

    @property
    def distance_pc(self) -> float | None:
        v = self.settings.get("distance_pc")
        return float(v) if v is not None else None

    @property
    def rv_kms(self) -> float | None:
        v = self.settings.get("rv_kms")
        return float(v) if v is not None else None

    # ---- cubes ------------------------------------------------------------------------------
    @property
    def cubes(self):
        """The `jalebi.cube.CubeSet` of the source (headers read now, data on demand, kept in memory)."""
        if not self.has_cubes:
            raise FileNotFoundError(f"{self.name}: no s3d cubes in {self.path}")
        with self._lock:
            if self._cubes is None:
                from .cube.io import CubeSet
                cs = CubeSet(self.files.s3d, name=self._name or self.settings.get("name"), **self.read_kw)
                cs.root = self.path if os.path.isdir(self.path) else os.path.dirname(self.path)
                cs.cache_size = len(cs.info) if self.fits_in_memory(cs) else max(3, int(len(cs.info) * _cache_mb() * 1e6 / max(cs.estimated_bytes(), 1)))
                self._cubes = cs
            return self._cubes

    def fits_in_memory(self, cs=None) -> bool:
        cs = cs or self.cubes
        return cs.estimated_bytes() <= _cache_mb() * 1e6

    def cube_memory_mb(self) -> float:
        return self.cubes.estimated_bytes() / 1e6 if self.has_cubes else 0.0

    def preload(self, workers: int = 4, progress=None, stop=None) -> list[str]:
        """Read all cubes into memory in parallel (see CubeSet.preload).  Skipped (with a warning) when they
        would take more than JALEBI_CUBE_CACHE_MB (default 6000 MB): cubes are then read when needed."""
        if not self.has_cubes:
            return []
        if not self.fits_in_memory():
            warnings.warn(f"{self.name}: the cubes need {self.cube_memory_mb():.0f} MB > JALEBI_CUBE_CACHE_MB={_cache_mb():.0f}; "
                          "they are read when needed instead")
            return []
        return self.cubes.preload(workers=workers, progress=progress, stop=stop)

    def cube(self, band: str):
        return self.cubes.cube(band)

    def image(self, band: str | None = None):
        """Median-collapsed image of one cube (default: the first, sharpest PSF) -> (image, Cube)."""
        c = self.cube(band) if band else self.cubes.load(self.cubes.info[0].path)
        k = ("image", c.path)
        with self._lock:
            if k not in self._memo:
                self._memo[k] = c.image()
            return self._memo[k], c

    def position(self, band: str | None = None) -> tuple[float, float, str]:
        """(RA, Dec, how) of the source: the remembered position, the continuum peak near the header target in
        the cubes, or the header target position."""
        if self.settings.get("ra") is not None and self.settings.get("dec") is not None:
            ra, dec = parse_radec(self.settings["ra"], self.settings["dec"])
            return ra, dec, "remembered"
        if self.has_cubes:
            k = ("position", band)
            with self._lock:
                if k not in self._memo:
                    try:
                        ra, dec = self.cubes.source_position(band=band)
                        self._memo[k] = (float(ra), float(dec), "continuum peak in the cubes")
                    except Exception:
                        self._memo[k] = None
                if self._memo[k] is not None:
                    return self._memo[k]
        m = self.meta
        if m.get("TARG_RA") is not None and m.get("TARG_DEC") is not None:
            return float(m["TARG_RA"]), float(m["TARG_DEC"]), "header target position"
        raise ValueError(f"{self.name}: no position known (no cubes, no TARG_RA/TARG_DEC)")

    # ---- 1-D spectra --------------------------------------------------------------------------
    def _finish(self, spec: Spectrum, distance_pc: float | None) -> Spectrum:
        s = spec.copy()
        d = distance_pc or self.distance_pc
        if d:
            s.distance_pc = float(d)
        return s

    def x1d(self, distance_pc: float | None = None) -> Spectrum:
        """The pipeline x1d spectrum (all sub-bands; read once).  For a table source: the table."""
        with self._lock:
            if "x1d" not in self._memo:
                if self.has_x1d:
                    self._memo["x1d"] = self._read_x1d()
                elif self.has_table:
                    self._memo["x1d"] = load_csv(self.files.tables[0])
                else:
                    raise FileNotFoundError(f"{self.name}: no x1d files (use spectrum('s3d') for an aperture on the cubes)")
            return self._finish(self._memo["x1d"], distance_pc)

    def _read_x1d(self) -> Spectrum:
        from .instrument import _BANDS
        W, F, E, B = [], [], [], []
        meta = {}
        for b, f in self.files.x1d.items():
            w, fl, er, band, mt = read_x1d(f)
            band = band or b
            W.append(w); F.append(fl); E.append(er); B.append(np.full(len(w), band, dtype=object))
            meta.update({k: v for k, v in mt.items() if v is not None})
        wave = np.concatenate(W); flux = np.concatenate(F); err = np.concatenate(E); band = np.concatenate(B)
        order = np.lexsort((wave, np.array([BAND_ORDER.index(b) if b in BAND_ORDER else 99 for b in band])))
        s = Spectrum(wave[order], flux[order], err[order], band[order].astype(str), self.name, self.distance_pc or 140.0, meta=meta)
        for b, (lo, hi) in _BANDS.items():                 # pixels beyond the nominal sub-band ranges: masked
            i = s.band_slice(b)
            if len(i):
                s.mask[i[(s.wave[i] < lo - 0.02) | (s.wave[i] > hi + 0.02)]] = False
        return s

    def aperture_spectrum(self, ra=None, dec=None, aperture_fwhm_scale: float = 1.5, aperture_arcsec: float | None = None,
                          annulus_arcsec=None, apcorr: str = "mrs", distance_pc: float | None = None) -> Spectrum:
        """Point-source aperture on every cube at (ra, dec) (default: `position()`), the same photometry as
        `jalebi.data.load_s3d_folder`, but on the cubes in memory.  apcorr: mrs | gaussian | none | x1d."""
        from .instrument import _BANDS
        if ra is None or dec is None:
            ra, dec, _ = self.position()
        radec = parse_radec(ra, dec)
        ann = tuple(float(x) for x in annulus_arcsec) if annulus_arcsec else None
        key = ("aperture", round(radec[0], 7), round(radec[1], 7), round(float(aperture_fwhm_scale), 4),
               None if not aperture_arcsec else round(float(aperture_arcsec), 4), ann, apcorr)
        with self._lock:
            hit = self._memo.get(key)
        if hit is None:
            W, F, E, B = [], [], [], []
            meta = {}; centres = {}; scales = {}
            x1d = self.x1d() if (apcorr == "x1d" and self.has_x1d) else None
            for ci in self.cubes.info:
                c = self.cubes.load(ci.path)
                r = extract_cube(c, center_radec=radec, aperture_fwhm_scale=aperture_fwhm_scale, aperture_arcsec=aperture_arcsec,
                                 annulus_arcsec=ann, apcorr="none" if apcorr == "x1d" else apcorr)
                band = r["band"] or ci.band
                if x1d is not None:
                    i = x1d.band_slice(band)
                    if len(i) > 10:
                        f2 = np.interp(x1d.wave[i], r["wave"], r["flux"])
                        ok = np.isfinite(x1d.flux[i]) & np.isfinite(f2) & (f2 > 0)
                        sc = float(np.nanmedian(x1d.flux[i][ok] / f2[ok])) if ok.sum() > 10 else 1.0
                        r["flux"] = r["flux"] * sc; r["err"] = r["err"] * sc; scales[band] = sc
                W.append(r["wave"]); F.append(r["flux"]); E.append(r["err"]); B.append(np.full(len(r["wave"]), band, dtype=object))
                meta.update({k: v for k, v in r["meta"].items() if v is not None}); centres[band] = r["center_pix"]
            wave = np.concatenate(W); flux = np.concatenate(F); err = np.concatenate(E); band = np.concatenate(B)
            order = np.lexsort((wave, np.array([BAND_ORDER.index(b) if b in BAND_ORDER else 99 for b in band])))
            meta["extraction"] = {"source": "s3d", "ra": radec[0], "dec": radec[1], "center_radec": radec,
                                  "aperture_fwhm_scale": aperture_fwhm_scale, "aperture_arcsec": aperture_arcsec,
                                  "annulus_arcsec": ann, "apcorr": apcorr, "center_pix_per_band": centres, "x1d_scale_per_band": scales}
            hit = Spectrum(wave[order], flux[order], err[order], band[order].astype(str), self.name, self.distance_pc or 140.0, meta=meta)
            for b, (lo, hi) in _BANDS.items():
                i = hit.band_slice(b)
                if len(i):
                    hit.mask[i[(hit.wave[i] < lo - 0.02) | (hit.wave[i] > hi + 0.02)]] = False
            with self._lock:
                self._memo[key] = hit
        return self._finish(hit, distance_pc)

    def region_spectrum(self, region, background=None, distance_pc: float | None = None, name: str | None = None,
                        min_coverage: float = 0.8, bands=None) -> Spectrum:
        """Cubes summed over a `jalebi.cube.Region` (no aperture correction: extended emission), memoised."""
        from .cube.regions import region_spectrum
        key = ("region", region.to_ds9(), background.to_ds9() if background is not None else None, round(min_coverage, 3),
               tuple(bands) if bands else None)
        with self._lock:
            hit = self._memo.get(key)
        if hit is None:
            hit = region_spectrum(self.cubes, region, name=name or self.name, distance_pc=self.distance_pc or 140.0,
                                  bands=bands, min_coverage=min_coverage, background=background)
            with self._lock:
                self._memo[key] = hit
        s = self._finish(hit, distance_pc)
        if name:
            s.name = name
        return s

    def spectrum(self, source: str = "auto", extraction: dict | None = None, distance_pc: float | None = None) -> Spectrum:
        """The 1-D spectrum for the LTE slab fit: "x1d" (pipeline), "s3d" (aperture on the cubes, options of
        `aperture_spectrum`), "table", or "auto" (x1d, else table, else an aperture at the source)."""
        if source == "auto":
            source = "x1d" if self.has_x1d else ("table" if self.has_table else "s3d")
        if source in ("x1d", "table", "csv"):
            return self.x1d(distance_pc)
        if source == "s3d":
            ex = dict(extraction or {})
            ex.pop("source", None)
            return self.aperture_spectrum(distance_pc=distance_pc, **ex)
        raise ValueError(f"unknown spectrum source '{source}' (x1d | s3d | table | auto)")

    # ---- summaries ----------------------------------------------------------------------------
    def memory_mb(self) -> float:
        """Memory held now (cubes in memory + memoised spectra/images)."""
        n = 0
        if self._cubes is not None:
            with self._cubes._lock:
                n += sum(c.sci.nbytes + c.err.nbytes for c in self._cubes._cache.values())
        for v in list(self._memo.values()):
            if isinstance(v, np.ndarray):
                n += v.nbytes
            elif isinstance(v, Spectrum):
                n += v.wave.nbytes * 6
        return n / 1e6

    def coverage(self):
        """One row per MRS sub-band: x1d / cube present, wavelength range, cube shape, in memory."""
        import pandas as pd
        rows = {b: {"band": b, "x1d": b in self.files.x1d, "s3d": False, "wmin_um": np.nan, "wmax_um": np.nan,
                    "cube": "", "in_memory": False} for b in BAND_ORDER}
        if self.has_cubes:
            mem = set(self.cubes.loaded())
            for ci in self.cubes.info:
                r = rows.setdefault(ci.band, {"band": ci.band, "x1d": ci.band in self.files.x1d})
                r.update(s3d=True, wmin_um=round(ci.wmin, 4), wmax_um=round(ci.wmax, 4),
                         cube=f"{ci.nz}×{ci.ny}×{ci.nx}", in_memory=ci.band in mem)
        return pd.DataFrame(list(rows.values()))

    def summary(self) -> dict:
        d = {"name": self.name, "path": self.path, "kind": self.kind, "x1d_bands": self.x1d_bands,
             "cube_bands": self.cube_bands, "tables": [os.path.basename(t) for t in self.files.tables],
             "distance_pc": self.distance_pc, "rv_kms": self.rv_kms, "settings_file": self.settings_file,
             **{k.lower(): v for k, v in self.meta.items()}}
        if self.has_cubes:
            d["cube_memory_mb"] = round(self.cube_memory_mb(), 1)
            d["cubes_in_memory"] = self.cubes.loaded()
        try:
            ra, dec, how = self.position()
            d.update(ra=ra, dec=dec, position_from=how)
        except Exception:
            pass
        return d

    def describe(self) -> str:
        s = self.summary()
        lines = [f"{s['name']}  ({s['kind']}: {s['path']})"]
        if s["x1d_bands"]:
            lines.append(f"  x1d   : {len(s['x1d_bands'])} sub-bands  {' '.join(s['x1d_bands'])}")
        if s["cube_bands"]:
            lines.append(f"  cubes : {len(s['cube_bands'])}  {' '.join(s['cube_bands'])}  (~{s['cube_memory_mb']:.0f} MB in memory, "
                         f"{len(s['cubes_in_memory'])} loaded)")
        if s["tables"]:
            lines.append(f"  tables: {', '.join(s['tables'])}")
        if "ra" in s:
            lines.append(f"  position RA {s['ra']:.6f}  Dec {s['dec']:.6f}  ({s['position_from']})")
        facts = [f"{k}={s[k]}" for k in ("program", "observtn", "date-obs", "cal_ver", "crds_ctx") if s.get(k) is not None]
        if facts:
            lines.append("  " + "  ".join(facts))
        lines.append(f"  distance {s['distance_pc'] or '—'} pc · RV {s['rv_kms'] if s['rv_kms'] is not None else '—'} km/s"
                     + (f"  (remembered in {s['settings_file']})" if s["settings_file"] else ""))
        return "\n".join(lines)

    def __repr__(self):
        return f"Source({self.name!r}, {self.kind}, x1d={len(self.files.x1d)}, s3d={len(self.files.s3d)})"


# ------------------------------------------------------------------------------------------------
# the process-wide cache
# ------------------------------------------------------------------------------------------------

_SOURCES: "OrderedDict[tuple, Source]" = OrderedDict()
_SOURCES_LOCK = threading.RLock()


def _max_sources() -> int:
    return max(int(os.environ.get("JALEBI_SOURCE_CACHE", 3)), 1)


def open_source(path: str, dq_mask: bool = True, zero_is_nan: bool = True, refresh: bool = False) -> Source:
    """The Source at `path`, from this process's cache when it was opened before (with the same NaN rules).
    The last `JALEBI_SOURCE_CACHE` (default 3) sources are kept; `refresh=True` re-reads from disk."""
    from .examples import resolve_path
    key = (os.path.abspath(resolve_path(os.path.expanduser(path))), bool(dq_mask), bool(zero_is_nan))
    with _SOURCES_LOCK:
        if not refresh and key in _SOURCES:
            _SOURCES.move_to_end(key)
            return _SOURCES[key]
    src = Source(path, dq_mask=dq_mask, zero_is_nan=zero_is_nan)
    with _SOURCES_LOCK:
        _SOURCES[key] = src
        _SOURCES.move_to_end(key)
        while len(_SOURCES) > _max_sources():
            _SOURCES.popitem(last=False)
    return src


def loaded_sources() -> list[Source]:
    with _SOURCES_LOCK:
        return list(reversed(_SOURCES.values()))


def forget_sources():
    """Drop every cached source (frees the memory of their cubes)."""
    with _SOURCES_LOCK:
        _SOURCES.clear()


def get_cubeset(path: str, dq_mask: bool = True, zero_is_nan: bool = True):
    """The CubeSet of a folder / cube file through the source cache: the web app's modules use this, so a
    cube is read from disk once per server process however often it is needed."""
    return open_source(path, dq_mask=dq_mask, zero_is_nan=zero_is_nan).cubes
