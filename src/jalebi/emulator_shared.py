"""Shared emulator tables (jalebi 0.21): `fit.emulator.cache: shared` (the default).

The 0.18 emulator (jalebi.emulator) tabulates the pixel flux of every unit on the data's own pixels, so every
disk and every release rebuilds every table (~20 min per disk for full-range fits).  Here the tables are built
once for a whole survey and resampled onto each disk at load time:

    H(x; T, log N) = (G_R * I)(x)          the LSF-convolved 1-au slab spectrum at the reference distance of 1 pc,

tabulated on a survey-wide dense rest-frame ln(lambda) grid with one segment per MRS sub-band (instrument._BANDS,
4C extended to the 28.8 um the x1d products reach), each padded by the LSF wings, |v| <= 150 km/s and a small
margin for the band-edge slop of real data.  The dense step is 1 / (points_per_fwhm x max R) per segment
(default 8 points per LSF FWHM).  G_R is a point-sampled Gaussian of sigma = FWHM_TO_SIGMA / R(lambda) at each
dense point (no pixel box: build_dense_lsf_operator, next to instrument.build_lsf_operator).

Per disk, a sparse operator P_disk maps the dense grid to the disk's pixels: the average over each pixel's
edges (the same edges instrument.build_lsf_operator uses), shifted by -rv/c in ln(lambda) to reproduce
SlabModel._shift, of the 4-point Lagrange (cubic) interpolant of H across the dense points, integrated with
3-point Gauss-Legendre on every sub-segment (exact for a cubic), so P_disk is linear in H.  Pixel flux =
P_disk @ H(T, log N) x (1 pc / d)^2 x the component's window mask.  At load time the projected table on the
disk's pixels is computed once (< 2 s) and wrapped in a 0.18 UnitTable, so the cost per ln P call is unchanged.

Approximations relative to the exact pixel model (measured in docs/EMULATOR.md, all far below the 0.1 sigma /
0.1 % targets): the shift is applied after the convolution (the kernel is R at the rest wavelength, not at the
observed one); R is taken at each dense point instead of at the pixel centre; and the cubic resampling.

Certification without a disk's noise: at build time each node is scaled so that its brightest dense point
equals f_ref = 1, and compared with the exact model against sigma_ref = f_ref / ref_snr (default 1000), with
the 0.18 error measure (emulator.errors, including the max(flux, 1 sigma) rule for the flux error).  The
tables therefore hold the pointwise error below 1e-4 of the node's peak, which meets the 0.1 sigma target for
any disk whose brightest pixel has S/N <= ref_snr.  At fit time a spot check (spot_check random (T, log N)
points per unit, emulator vs exact model on the disk's pixels and sigma) catches the rest: a unit that fails
0.1 sigma or 0.1 % falls back to the exact model, with a warning and an entry in diagnostics.json.

Cache key (hashed into the file name, and written in readable form to the JSON next to each table):
EMULATOR_MODEL_VERSION (bumped by hand only when the physics or the table format changes), LSF_VERSION, the
molecule + linelist_key, the line-list content hash + source-file SHA-256, eup_max, the intrinsic fwhm and
fwhm_thermal, R model / scale / constant, the dense-grid spec, the (T, log N) box, ref_snr, the targets, method
and node limits.  Not in the key: the jalebi version, the disk's pixels, LSF matrix, v_shift, distance, windows,
noise or f_ref.

Storage: `<cache_dir>/shared/<linelist>_<key>.npy` (float32 ln(H + eps) on the support points, uncompressed,
memory-mapped at load so that processes on one node share pages) + `.npz` (nodes, support, eps, log tau_max)
+ `.json`.  Writes go to a temp file + os.replace under a lock file (fcntl.flock, with a portable fallback), so
one process builds a table while the others wait and then read it.  With fit.emulator.read_only (or
$JALEBI_EMULATOR_READONLY=1) a missing table means an exact fallback plus a warning, never a build.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import time
import warnings
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import sparse

from .constants import AU, C, FWHM_TO_SIGMA, PC
from .emulator import (EmulatorSettings, UnitTable, EmulatorSet, _arr_sha, _file_sha, build_unit_table, emulable, errors,
                       exact_rows, default_cache_dir, table_from_rows, LSF_VERSION)
from .instrument import _BANDS, _JONES2023, RESOLVING_POWER, pixel_edges
from .model import Component, SlabModel, linelist_key as _linelist_key

EMULATOR_MODEL_VERSION = 1          # bump by hand when the tabulated physics or the table format changes
V_MAX_KMS = 150.0                   # |v_shift| the padding of every dense segment allows
BAND_EXTENT = {**_BANDS, "4C": (24.19, 28.80)}      # the x1d products of 4C reach 28.7 um
SHARED_SUBDIR = "shared"
F_REF = 1.0                         # the reference peak every node is scaled to at build time
A_MAX_REF = 1e12                    # "no area cap" at build time (eps = 1e-4 sigma_ref / a_max stays finite)
_GL_NODES = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)])
_GL_WEIGHTS = np.array([5.0, 8.0, 5.0]) / 9.0


# ---------------------------------------------------------------------------------------------------
# Dense grid
# ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class DenseSpec:
    """The survey-wide dense grid (part of the cache key)."""
    points_per_fwhm: float = 8.0
    pad_kms: float = V_MAX_KMS
    margin: float = 0.003            # extra ln(lambda) on each side (band-edge slop of the x1d products)
    truncate: float = 6.0            # LSF kernel truncation in sigma (6: a point-sampled kernel cut at 4 sigma
                                     # leaves a 6e-5 step that the pixel integration of the exact model does not have)
    bands: tuple = tuple(BAND_EXTENT)

    def key_dict(self) -> dict:
        return {"points_per_fwhm": float(self.points_per_fwhm), "pad_kms": float(self.pad_kms), "margin": float(self.margin),
                "truncate": float(self.truncate),
                "bands": {b: [float(BAND_EXTENT[b][0]), float(BAND_EXTENT[b][1])] for b in self.bands}}


def band_resolving_power(band: str, wave_um, R_model: str = "argyriou2023", R_scale: float = 1.0, R_constant=None):
    """R(lambda) of one sub-band's own relation, valid beyond the band edges too (jones2023 is piecewise per
    band; its relation is extended into the padding so that a segment has no jump in R)."""
    w = np.atleast_1d(np.asarray(wave_um, float))
    if R_constant is not None:
        return np.full(w.shape, float(R_constant))
    if R_model == "jones2023":
        lo, hi = _BANDS[band]
        r0, r1 = _JONES2023[band]
        return (r0 + (r1 - r0) * (w - lo) / (hi - lo)) * R_scale
    return RESOLVING_POWER[R_model](w) * R_scale


class DenseGrid:
    """Dense rest-frame ln(lambda) grid, one uniform segment per MRS sub-band."""

    def __init__(self, spec: DenseSpec | None = None, R_model: str = "argyriou2023", R_scale: float = 1.0, R_constant=None):
        self.spec = spec or DenseSpec()
        self.R_model, self.R_scale, self.R_constant = R_model, R_scale, R_constant
        self.bands = list(self.spec.bands)
        self.extent = np.array([BAND_EXTENT[b] for b in self.bands])            # (nseg, 2) um, unpadded
        xs, Rs, segs = [], [], []
        off = 0
        for i, b in enumerate(self.bands):
            lo, hi = self.extent[i]
            Rb = band_resolving_power(b, np.array([lo, hi]), R_model, R_scale, R_constant)
            R_min, R_max = float(Rb.min()), float(Rb.max())
            pad = self.spec.pad_kms * 1e3 / C + self.spec.truncate * FWHM_TO_SIGMA / R_min + self.spec.margin
            x0, x1 = np.log(lo) - pad, np.log(hi) + pad
            h = 1.0 / (self.spec.points_per_fwhm * R_max)
            n = int(np.ceil((x1 - x0) / h)) + 1
            x = x0 + h * np.arange(n)
            xs.append(x); Rs.append(band_resolving_power(b, np.exp(x), R_model, R_scale, R_constant))
            segs.append((off, n, x0, h)); off += n
        self.x = np.concatenate(xs)
        self.wave = np.exp(self.x)
        self.R = np.concatenate(Rs)
        self.segments = segs                                                    # (offset, n, x0, h) per band
        self.n = len(self.x)

    def windows(self) -> list[tuple[float, float]]:
        """Wavelength windows (um) of the padded segments (for the fine grid of the build)."""
        return [(float(np.exp(x0)), float(np.exp(x0 + h * (n - 1)))) for _, n, x0, h in self.segments]

    def segment_of(self, wave_um) -> np.ndarray:
        """Index of the first sub-band whose nominal range contains each wavelength (-1: none).  The first
        match is what instrument.resolving_power uses for the exact model, so overlap pixels get the same R."""
        w = np.atleast_1d(np.asarray(wave_um, float))
        out = np.full(w.shape, -1, int)
        for i in range(len(self.bands)):
            lo, hi = self.extent[i]
            m = (out < 0) & (w >= lo) & (w <= hi)
            out[m] = i
        return out

    def pixel_operator(self, wave_pix, rv_kms: float = 0.0, edges=None, fine_ranges=None) -> tuple[sparse.csr_matrix, np.ndarray]:
        """(P_disk (npix, ndense), covered (npix,) bool).  P_disk averages the cubic interpolant of a function
        tabulated on the dense grid over every pixel's edges (ln lambda; instrument.pixel_edges of the pixel
        array unless `edges` is given), shifted by -rv/c so that P_disk @ H reproduces K @ shift(I, rv).

        fine_ranges: the (x_start, x_end) segments of the disk model's fine grid.  The exact model integrates
        a pixel only over the fine points that exist, and divides by the full pixel width: a pixel next to a
        gap between fit windows (whose edges reach the midpoint of the gap) is diluted.  With the ranges
        given, the integral is clipped the same way, so P_disk reproduces that pixel too."""
        x_pix = np.log(np.asarray(wave_pix, float))
        npix = len(x_pix)
        lo, hi = pixel_edges(x_pix) if edges is None else edges
        lo = np.asarray(lo, float); hi = np.asarray(hi, float)
        delta = rv_kms * 1e3 / C
        a_all = lo - delta
        b_all = hi - delta
        width = np.abs(hi - lo)
        # Inverted edges (hi < lo): the first / last pixels of adjacent sub-bands, whose midpoint edges are computed
        # on the concatenated pixel array.  instrument.build_lsf_operator then integrates only within
        # x_pix +- (4 sigma - |hi - lo| / 2) of the centre and divides by the full |hi - lo|: nothing at all when the
        # overlap exceeds 8 sigma (the usual case: the model is 0 there), a diluted average otherwise.  Reproduced
        # here so that a fit sees the same model as the exact backend at those pixels (an exact-model artefact).
        inv = hi < lo
        if inv.any():
            R = band_resolving_power(self.bands[0], np.exp(x_pix[inv]), self.R_model, self.R_scale, self.R_constant) \
                if self.R_model != "jones2023" else np.array([band_resolving_power(self.bands[max(s, 0)], np.exp(x), self.R_model, self.R_scale, self.R_constant)[0]
                                                               for x, s in zip(x_pix[inv], self.segment_of(wave_pix)[inv])])
            half = 4.0 * FWHM_TO_SIGMA / R - 0.5 * width[inv]
            a_all[inv] = np.maximum(hi[inv], x_pix[inv] - half) - delta
            b_all[inv] = np.minimum(lo[inv], x_pix[inv] + half) - delta
            a_all[inv] = np.where(half > 0, a_all[inv], np.inf)             # empty row
        if fine_ranges:
            fr = np.asarray(fine_ranges, float) - delta
            a_c = np.full(npix, np.inf); b_c = np.full(npix, -np.inf)
            for xa, xb in fr:                       # the fine segment that holds the (shifted) pixel centre
                inside = (x_pix - delta >= xa) & (x_pix - delta <= xb)
                a_c[inside] = np.maximum(a_all[inside], xa); b_c[inside] = np.minimum(b_all[inside], xb)
            has = np.isfinite(a_c) & (b_c > a_c)
            a_all = np.where(has, a_c, a_all); b_all = np.where(has, b_c, b_all)
        # Each pixel's range is partitioned by the first-match band ranges (band s owns [max(lo_s, hi_{s-1}), hi_s]),
        # so a pixel whose edges reach into the next band -- the last pixel before a masked region, say -- is
        # integrated piecewise, each piece in its own segment (the same rule instrument.resolving_power uses).
        covered_len = np.zeros(npix)
        rows, cols, vals = [], [], []
        prev_hi = -np.inf
        for s, (off, n, x0, h) in enumerate(self.segments):
            lo_s, hi_s = self.extent[s]
            L, U = np.log(max(lo_s, prev_hi)), np.log(hi_s)
            if s == 0:
                L = x0                                              # the first band owns its padding too
            if s == len(self.segments) - 1:
                U = x0 + h * (n - 1)                                # and the last band its upper padding
            prev_hi = max(prev_hi, hi_s)
            a = np.maximum(a_all, L); b = np.minimum(b_all, U)
            idx = np.flatnonzero((b > a) & (width > 0) & np.isfinite(a_all))
            if len(idx) == 0:
                continue
            a = a[idx]; b = b[idx]
            covered_len[idx] += b - a
            xend = x0 + h * (n - 1)
            assert np.all(a >= x0 - 1e-12) and np.all(b <= xend + 1e-12), "dense segment does not cover its band"
            ka = np.clip(np.floor((a - x0) / h).astype(int), 0, n - 2)
            kb = np.clip(np.ceil((b - x0) / h).astype(int) - 1, 0, n - 2)
            kb = np.maximum(kb, ka)
            M = int(np.max(kb - ka)) + 1
            for m in range(M):
                k = ka + m
                live = k <= kb
                if not live.any():
                    break
                i = idx[live]; k = k[live]; al = a[live]; bl = b[live]
                u = np.maximum(al, x0 + k * h); v = np.minimum(bl, x0 + (k + 1) * h)
                s0 = np.clip(k - 1, 0, n - 4)
                half = 0.5 * (v - u); mid = 0.5 * (u + v)
                for q in range(3):
                    t = mid + half * _GL_NODES[q]
                    wq = half * _GL_WEIGHTS[q] / width[i]
                    xi = (t - (x0 + s0 * h)) / h                       # 0..3 on the 4-point stencil
                    l0 = -(xi - 1) * (xi - 2) * (xi - 3) / 6.0
                    l1 = xi * (xi - 2) * (xi - 3) / 2.0
                    l2 = -xi * (xi - 1) * (xi - 3) / 2.0
                    l3 = xi * (xi - 1) * (xi - 2) / 6.0
                    for j, lj in enumerate((l0, l1, l2, l3)):
                        rows.append(i); cols.append(off + s0 + j); vals.append(wq * lj)
        full = np.where(np.isfinite(a_all), b_all - a_all, 0.0)
        covered = (full > 0) & (covered_len >= full * (1 - 1e-9))
        if not rows:
            return sparse.csr_matrix((npix, self.n)), covered
        P = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(npix, self.n))
        P.sum_duplicates()
        return P, covered


def build_dense_lsf_operator(x_fine: np.ndarray, dx: float, x_dense: np.ndarray, R_dense: np.ndarray,
                             truncate: float = 6.0, chunk: int = 4000) -> sparse.csr_matrix:
    """Sparse operator (ndense, nfine) that samples the Gaussian-LSF-convolved fine-grid spectrum at the dense
    points: row i = dx G_sigma_i(x_fine - x_dense_i) with sigma_i = FWHM_TO_SIGMA / R_dense_i, truncated at
    `truncate` sigma.  The point-sampled counterpart of instrument.build_lsf_operator (no pixel box).  Built
    straight in CSR form (float32 weights, int32 columns) in chunks of rows: for the full MRS range the
    operator has ~3e7 entries and this keeps the build under ~0.5 GB."""
    x_fine = np.asarray(x_fine, float); x_dense = np.asarray(x_dense, float)
    sig = FWHM_TO_SIGMA / np.asarray(R_dense, float)
    j0 = np.searchsorted(x_fine, x_dense - truncate * sig)
    j1 = np.searchsorted(x_fine, x_dense + truncate * sig)
    cnt = np.maximum(j1 - j0, 0)
    indptr = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)
    if indptr[-1] == 0:
        return sparse.csr_matrix((len(x_dense), len(x_fine)))
    data = np.empty(indptr[-1], np.float32)
    indices = np.empty(indptr[-1], np.int32)
    for a in range(0, len(x_dense), chunk):
        b = min(a + chunk, len(x_dense))
        c = cnt[a:b]
        tot = int(c.sum())
        if tot == 0:
            continue
        starts = np.cumsum(np.concatenate([[0], c[:-1]]))
        cols = (np.arange(tot) - np.repeat(starts, c)) + np.repeat(j0[a:b], c)
        sr = np.repeat(sig[a:b], c)
        w = dx / (sr * np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * ((x_fine[cols] - np.repeat(x_dense[a:b], c)) / sr) ** 2)
        data[indptr[a]:indptr[b]] = w
        indices[indptr[a]:indptr[b]] = cols
    return sparse.csr_matrix((data, indices, indptr), shape=(len(x_dense), len(x_fine)))


# ---------------------------------------------------------------------------------------------------
# What a table is: TableSpec, line lists, keys
# ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TableSpec:
    """Everything that defines one shared table (one molecule / line list / width / box)."""
    molecule: str
    release: str
    T: tuple                      # (lo, hi) K
    logN: tuple                   # (lo, hi)
    fwhm: float = 4.7
    fwhm_thermal: bool = False
    eup_max: float | None = None
    linelist_path: str | None = None
    R_model: str = "argyriou2023"
    R_scale: float = 1.0
    R_constant: float | None = None
    oversample: int = 6           # fine-grid points per line FWHM of the exact rows (fit.emulator.table_oversample).
                                  # Not the fit's oversample: at 3 the exact model's line profiles depend on the
                                  # phase of its fine grid at the 1e-3 level, which no shared table can reproduce

    @property
    def linelist(self) -> str:
        return _linelist_key(self.molecule, self.release, self.linelist_path)

    def component(self) -> Component:
        return Component(self.molecule, self.molecule, logN=0.5 * (self.logN[0] + self.logN[1]),
                         T=float(np.sqrt(self.T[0] * self.T[1])), logR=0.0, fwhm=self.fwhm, fwhm_thermal=self.fwhm_thermal,
                         linelist_release=self.release, eup_max=self.eup_max, linelist_path=self.linelist_path)

    def label(self) -> str:
        return f"{self.linelist} T {self.T[0]:.0f}-{self.T[1]:.0f} K, log N {self.logN[0]:.2f}-{self.logN[1]:.2f}"


def dense_spec(settings: EmulatorSettings) -> DenseSpec:
    """The DenseSpec of the settings (points_per_fwhm, bands)."""
    kw = {"points_per_fwhm": float(settings.points_per_fwhm)}
    if settings.bands:
        kw["bands"] = tuple(settings.bands)
    return DenseSpec(**kw)


def spec_from_component(model, comp, p: dict, box: dict, oversample: int = 6) -> TableSpec:
    """TableSpec of a component of a disk model over the survey box `box` ({"T": (lo, hi), "logN": (lo, hi)})."""
    return TableSpec(comp.molecule, model.release_of(comp), (float(box["T"][0]), float(box["T"][1])),
                     (float(box["logN"][0]), float(box["logN"][1])), float(p["fwhm"]), bool(comp.fwhm_thermal),
                     None if comp.eup_max is None else float(comp.eup_max), comp.linelist_path,
                     model.R_model, float(model.R_scale), None if model.R_constant is None else float(model.R_constant),
                     int(oversample))


_LL_CACHE: dict = {}


def shared_linelist(spec: TableSpec, grid: DenseGrid):
    """The line list of a shared table: selected over the whole dense grid and pruned as model.build_model does
    for a fit (model.prune_linelist: the same rule for every window)."""
    from .linedata import load_linelist
    wlo, whi = grid.wave[0] - 0.1, grid.wave[-1] + 0.1
    key = (spec.molecule, spec.release, spec.linelist_path, spec.eup_max, round(wlo, 4), round(whi, 4), float(spec.T[1]))
    ll = _LL_CACHE.get(key)
    if ll is None:
        from .model import prune_linelist
        ll = load_linelist(spec.molecule, release=spec.release, path=spec.linelist_path, fetch=False)
        ll = prune_linelist(ll, wlo, whi, eup_max=spec.eup_max, T_max=spec.T[1])
        if len(_LL_CACHE) > 32:
            _LL_CACHE.clear()
        _LL_CACHE[key] = ll
    return ll


def shared_linelist_hash(ll) -> str:
    """Content hash of a selected line list (the arrays that reach the opacity) + the source file's SHA-256."""
    Tq = np.array([50.0, 100.0, 200.0, 500.0, 1000.0, 1500.0, 3000.0])
    Q = np.array([ll.partition(t) for t in Tq])
    return _arr_sha(ll.wave, ll.a, ll.gu, ll.eu, ll.el, Q) + ":" + _file_sha(getattr(ll, "source", None))


def key_fields(spec: TableSpec, dense: DenseSpec, settings: EmulatorSettings, ll_hash: str) -> dict:
    """The readable cache key (also written to the JSON next to the table)."""
    from .model import OpacityBasis
    return {"model_version": EMULATOR_MODEL_VERSION, "lsf": LSF_VERSION, "line_truncate": float(OpacityBasis.LINE_TRUNCATE),
            "molecule": spec.molecule,
            "linelist_key": spec.linelist, "linelist_hash": ll_hash, "eup_max": spec.eup_max, "fwhm": float(spec.fwhm),
            "fwhm_thermal": bool(spec.fwhm_thermal), "R": [spec.R_model, float(spec.R_scale), spec.R_constant],
            "oversample": int(spec.oversample), "dense": dense.key_dict(), "T_box": [float(spec.T[0]), float(spec.T[1])],
            "logN_box": [float(spec.logN[0]), float(spec.logN[1])], "ref_snr": float(settings.ref_snr),
            "target": [float(settings.target_sigma), float(settings.target_flux), float(settings.safety)],
            "method": settings.method, "n_start": list(settings.n_start), "max_nodes": list(settings.max_nodes)}


def _sha(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()


def table_keys(fields: dict) -> tuple[str, str]:
    """(physics key, full key): the physics key leaves out the (T, log N) box, so that a table with a box that
    contains a disk's bounds can be found when no table with exactly the disk's box exists."""
    phys = {k: v for k, v in fields.items() if k not in ("T_box", "logN_box")}
    return _sha(phys)[:16], _sha(fields)[:16]


def shared_dir(settings: EmulatorSettings) -> str:
    return os.path.join(settings.cache_dir or default_cache_dir(), SHARED_SUBDIR)


def table_basename(spec: TableSpec, phys: str, full: str) -> str:
    ll = spec.linelist.replace(":", "-").replace("@", "_").replace("/", "_")[:40]
    return f"{ll}_{phys}_{full}"


# ---------------------------------------------------------------------------------------------------
# Locks (one builder, the others wait or read)
# ---------------------------------------------------------------------------------------------------

class TableLock:
    """Exclusive lock on `<path>.lock`: fcntl.flock where available, else an O_EXCL lock file that is polled
    (stale after `stale_s`).  Blocking."""

    def __init__(self, path: str, stale_s: float = 3600.0, poll_s: float = 0.5):
        self.path = path + ".lock"
        self.stale_s, self.poll_s = stale_s, poll_s
        self.fd = None
        self._excl = False

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        try:
            import fcntl
            self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
            fcntl.flock(self.fd, fcntl.LOCK_EX)
            return self
        except ImportError:
            pass
        while True:                                         # portable fallback
            try:
                self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
                os.write(self.fd, str(os.getpid()).encode())
                self._excl = True
                return self
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(self.path) > self.stale_s:
                        os.remove(self.path)
                        continue
                except OSError:
                    pass
                time.sleep(self.poll_s)

    def __exit__(self, *exc):
        if self.fd is not None:
            try:
                if not self._excl:
                    import fcntl
                    fcntl.flock(self.fd, fcntl.LOCK_UN)
            except Exception:
                pass
            os.close(self.fd)
            if self._excl:
                try:
                    os.remove(self.path)
                except OSError:
                    pass
        return False


# ---------------------------------------------------------------------------------------------------
# A shared table on disk
# ---------------------------------------------------------------------------------------------------

@dataclass
class SharedTable:
    """ln(H + eps) on the support points of the dense grid over a (ln T, log N) tensor grid (1 au slab, 1 pc)."""
    spec: TableSpec
    lnT: np.ndarray
    logN: np.ndarray
    sup: np.ndarray                    # indices into the dense grid
    L: np.ndarray                      # float32 (nT, nN, nsup), memory-mapped when loaded
    eps: np.ndarray                    # float32 (nsup)
    ltau: np.ndarray                   # (nT, nN) log10 peak optical depth
    rv: float
    fwhm: float
    method: str
    meta: dict = field(default_factory=dict)
    path: str = ""

    @property
    def nbytes(self) -> int:
        return int(self.L.size * 4)

    def covers(self, T_bounds, logN_bounds, tol: float = 1e-9) -> bool:
        return bool(self.lnT[0] - tol <= np.log(T_bounds[0]) and np.log(T_bounds[1]) <= self.lnT[-1] + tol
                    and self.logN[0] - tol <= logN_bounds[0] and logN_bounds[1] <= self.logN[-1] + tol)

    @staticmethod
    def files(base: str) -> tuple[str, str, str]:
        return base + ".npy", base + ".npz", base + ".json"

    def save(self, base: str):
        """Temp files + os.replace, so a reader never sees a partial table."""
        npy, npz, js = self.files(base)
        tmp = base + f".tmp{os.getpid()}"
        if isinstance(self.L, np.memmap) and getattr(self.L, "filename", None) and os.path.abspath(self.L.filename) == os.path.abspath(tmp + ".npy"):
            self.L.flush()                                   # built straight into the temp file
            del self.L
        else:
            np.save(tmp + ".npy", np.ascontiguousarray(self.L, dtype=np.float32))
        np.savez(tmp + ".npz", lnT=self.lnT, logN=self.logN, sup=self.sup, eps=self.eps, ltau=self.ltau, rv=self.rv,
                 fwhm=self.fwhm, method=self.method)
        with open(tmp + ".json", "w") as fh:
            json.dump({"spec": asdict(self.spec), "meta": self.meta}, fh, indent=1, default=float)
        os.replace(tmp + ".npy", npy); os.replace(tmp + ".npz", npz); os.replace(tmp + ".json", js)

    @classmethod
    def load(cls, base: str, mmap: bool = True) -> "SharedTable":
        npy, npz, js = cls.files(base)
        with open(js) as fh:
            info = json.load(fh)
        sp = info["spec"]
        spec = TableSpec(sp["molecule"], sp["release"], tuple(sp["T"]), tuple(sp["logN"]), sp["fwhm"], sp["fwhm_thermal"],
                         sp["eup_max"], sp["linelist_path"], sp["R_model"], sp["R_scale"], sp["R_constant"], int(sp.get("oversample", 6)))
        z = np.load(npz, allow_pickle=False)
        L = np.load(npy, mmap_mode="r" if mmap else None)
        return cls(spec, z["lnT"], z["logN"], z["sup"], L, z["eps"], z["ltau"], float(z["rv"]), float(z["fwhm"]),
                   str(z["method"]), info.get("meta", {}), base)


def dense_model(spec: TableSpec, grid: DenseGrid, ll=None) -> tuple[SlabModel, Component, np.ndarray]:
    """A SlabModel whose "pixels" are the dense grid points: the fine grid over the padded segments, the
    opacity basis of the shared line list, and the point-sampled LSF operator in place of K.  Distance 1 pc.
    Only the segments the line list reaches are built (CO has lines in 1A-1B only: the other segments are exactly
    zero and would cost 10x the fine grid for nothing); the returned index array maps the model's points to the
    full dense grid."""
    comp = spec.component()
    ll = ll if ll is not None else shared_linelist(spec, grid)
    wl = np.asarray(ll.wave, float)
    lo, hi = (float(wl.min()), float(wl.max())) if len(wl) else (grid.wave[0], grid.wave[-1])
    keep = np.zeros(grid.n, bool)
    windows = []
    for (off, n, x0, h), w in zip(grid.segments, grid.windows()):
        if w[1] >= lo * (1 - 0.01) and w[0] <= hi * (1 + 0.01):          # the segment is within 1 % of a line
            keep[off:off + n] = True
            windows.append(w)
    if not windows:
        keep[:] = True; windows = grid.windows()
    idx = np.flatnonzero(keep)
    # two placeholder "pixels" keep the constructor cheap; the dense grid and its operator replace them below
    m = SlabModel([comp], {spec.linelist: ll}, grid.wave[:2], distance_pc=1.0, windows=windows,
                  oversample=spec.oversample, R_model=spec.R_model, R_scale=spec.R_scale, R_constant=spec.R_constant,
                  releases={spec.molecule: spec.release})
    m.wave_pix = grid.wave[idx]
    m.K = build_dense_lsf_operator(m.grid.x, m.grid.dx, grid.x[idx], grid.R[idx], truncate=grid.spec.truncate)
    m.covered = np.asarray(m.K.sum(axis=1)).ravel() > 0.5
    return m, comp, idx


def build_shared_table(spec: TableSpec, settings: EmulatorSettings, dense: DenseSpec | None = None, say=None,
                       grid: DenseGrid | None = None, L_path: str | None = None) -> SharedTable:
    """Build one shared table (the 0.18 adaptive build on the dense grid, certified at the reference S/N).
    L_path: keep ln(H + eps) in a memory-mapped file there during the build (a full-range H2O table is ~0.8 GB)."""
    t0 = time.time()
    say = say or (lambda m: None)
    dense = dense or dense_spec(settings)
    grid = grid or DenseGrid(dense, spec.R_model, spec.R_scale, spec.R_constant)
    ll = shared_linelist(spec, grid)
    m, comp, idx = dense_model(spec, grid, ll)
    p = m.resolve_params()[comp.name]
    sigma = np.full(len(idx), F_REF / settings.ref_snr)
    say(f"  shared table {spec.label()}: {len(ll)} lines, fine grid {m.grid.n} points, dense grid {len(idx)} of {grid.n} points")
    import dataclasses
    st = dataclasses.replace(settings, refine="axis")
    m.planck_cache_max = 4                           # hundreds of check temperatures x 5 MB each otherwise
    tab = build_unit_table(m, comp, p, spec.T, spec.logN, sigma, F_REF, A_MAX_REF, st, unit=spec.molecule, say=say,
                           row_dtype=np.float32, L_path=L_path, row_dir=(L_path + ".rows") if L_path else None)
    fields = key_fields(spec, dense, settings, shared_linelist_hash(ll))
    phys, full = table_keys(fields)
    meta = {**tab.meta, "key": fields, "key_physics": phys, "key_full": full, "dense_points": int(grid.n),
            "support_points": int(len(tab.pix)), "lines": int(len(ll)), "fine_points": int(m.grid.n),
            "build_wall_s": time.time() - t0, "built_with_jalebi": __import__("jalebi").__version__,
            "size_MB": tab.L.size * 4 / 1e6}
    return SharedTable(spec, tab.lnT, tab.logN, idx[tab.pix], tab.L, tab.eps, tab.ltau, 0.0, float(spec.fwhm), settings.method, meta)


def find_table(spec: TableSpec, settings: EmulatorSettings, dense: DenseSpec, grid: DenseGrid, ll=None):
    """(base path or None, fields, phys, full, ll_hash): the table with exactly this key, else any table of the
    same physics key whose box contains the spec's box (the survey build may have used a wider box)."""
    ll = ll if ll is not None else shared_linelist(spec, grid)
    h = shared_linelist_hash(ll)
    fields = key_fields(spec, dense, settings, h)
    phys, full = table_keys(fields)
    d = shared_dir(settings)
    base = os.path.join(d, table_basename(spec, phys, full))
    if all(os.path.exists(f) for f in SharedTable.files(base)):
        return base, fields, phys, full, h
    best = None
    for js in sorted(glob.glob(os.path.join(d, f"*_{phys}_*.json"))):
        b = js[:-5]
        if not all(os.path.exists(f) for f in SharedTable.files(b)):
            continue
        try:
            with open(js) as fh:
                info = json.load(fh)
            T, N = info["spec"]["T"], info["spec"]["logN"]
        except Exception:
            continue
        if T[0] <= spec.T[0] * (1 + 1e-9) and spec.T[1] <= T[1] * (1 + 1e-9) and N[0] <= spec.logN[0] + 1e-9 and spec.logN[1] <= N[1] + 1e-9:
            area = (np.log(T[1]) - np.log(T[0])) * (N[1] - N[0])
            if best is None or area < best[0]:
                best = (area, b)
    return (best[1] if best else None), fields, phys, full, h


def get_or_build(spec: TableSpec, settings: EmulatorSettings, dense: DenseSpec | None = None, say=None,
                 grid: DenseGrid | None = None, build: bool | None = None) -> tuple[SharedTable | None, str]:
    """Load the shared table of `spec` (memory-mapped), building it under a lock when it is missing and
    building is allowed.  Returns (table or None, "cache" | "built" | reason it is missing)."""
    say = say or (lambda m: None)
    dense = dense or dense_spec(settings)
    grid = grid or DenseGrid(dense, spec.R_model, spec.R_scale, spec.R_constant)
    ll = shared_linelist(spec, grid)
    base, fields, phys, full, h = find_table(spec, settings, dense, grid, ll)
    if base is not None and not settings.rebuild:
        try:
            return SharedTable.load(base), "cache"
        except Exception as e:                       # unreadable: rebuild below if allowed
            say(f"  shared table {os.path.basename(base)} unreadable ({e})")
    allowed = (not settings.is_read_only()) if build is None else build
    if not allowed:
        return None, "no shared table in the cache and building is off (read_only)"
    d = shared_dir(settings)
    os.makedirs(d, exist_ok=True)
    base = os.path.join(d, table_basename(spec, phys, full))
    with TableLock(base):
        if all(os.path.exists(f) for f in SharedTable.files(base)) and not settings.rebuild:
            try:
                return SharedTable.load(base), "cache"       # another process built it while we waited
            except Exception:
                pass
        tab = build_shared_table(spec, settings, dense, say, grid, L_path=base + f".tmp{os.getpid()}.npy")
        tab.save(base)
        tab.path = base
    return SharedTable.load(base), "built"


# ---------------------------------------------------------------------------------------------------
# Per disk: projection and spot check
# ---------------------------------------------------------------------------------------------------

def project_table(shared: SharedTable, grid: DenseGrid, model, comp, p: dict, sigma, f_ref: float, a_max: float,
                  unit: str, P=None, covered=None) -> UnitTable:
    """The 0.18-style UnitTable of `comp` on the disk's pixels from a shared table: P_disk restricted to the
    table's support, the (1 pc / d)^2 scaling, the component's window mask and the disk's own eps."""
    if P is None:
        fr = [(float(model.grid.x[a]), float(model.grid.x[b - 1])) for a, b in model.grid.segments]
        P, covered = grid.pixel_operator(model.wave_pix, float(p.get("rv", 0.0)), fine_ranges=fr)
    Ps = P[:, shared.sup].tocsr()
    scale = model._omega_unit / ((AU / PC) ** 2 * np.pi)          # Omega(d) / Omega(1 pc) = (1 pc / d)^2
    nT, nN, _ = shared.L.shape
    npix = len(model.wave_pix)
    wm = model.window_mask(comp)
    Fn = np.empty((nT, nN, npix), np.float32)
    for i in range(nT):
        F = np.exp(np.asarray(shared.L[i], np.float64)) - shared.eps[None, :]        # (nN, nsup) at 1 pc
        Fp = (Ps @ F.T).T * scale                                                   # (nN, npix)
        if wm is not None:
            Fp = Fp * wm[None, :]
        Fn[i] = Fp
    fw = float(p["fwhm"])
    tab = table_from_rows(Fn, 10.0 ** shared.ltau, shared.lnT, shared.logN, sigma, f_ref, a_max, unit, comp.name,
                          comp.molecule, model.linelist_key(comp), npix, float(p.get("rv", 0.0)), fw, shared.method)
    tab.meta = {"nodes": [nT, nN], "support_pixels": int(len(tab.pix)), "shared_file": shared.path,
                "shared_validation": shared.meta.get("validation"), "from_cache": True, "cache": "shared",
                "T_bounds": [float(np.exp(shared.lnT[0])), float(np.exp(shared.lnT[-1]))],
                "logN_bounds": [float(shared.logN[0]), float(shared.logN[-1])], "f_ref": f_ref, "a_max": a_max,
                "file_bytes": shared.nbytes}
    return tab


def spot_check(model, comp, p: dict, tab: UnitTable, bounds: dict, sigma, f_ref: float, a_max: float, n: int = 200,
               seed: int = 0, mask=None) -> dict:
    """Emulator vs exact model at ~n random (T, log N) inside `bounds` on the disk's pixels and noise: n is
    laid out as nT random temperatures x nN random columns each (one opacity product per T).

    mask (default model.covered): pixels compared.  A pixel next to a gap between fit windows is "uncovered"
    (its edges reach the gap's midpoint, so the exact model's K row sums to < 0.5 and its model flux is diluted);
    the exact model's value there also depends on where its fine grid stops, which a table of the convolved
    spectrum cannot know, so those pixels are left out (their model value is an artefact in both backends)."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    mask = np.asarray(model.covered if mask is None else mask, bool)
    nN = max(1, min(10, n)); nT = max(1, int(round(n / nN)))
    T_lo, T_hi = bounds["T"]; N_lo, N_hi = bounds["logN"]
    Ts = np.exp(rng.uniform(np.log(T_lo), np.log(T_hi), nT)) if T_hi > T_lo else np.full(nT, T_lo)
    es, ef, fl = [], [], []
    for T in Ts:
        Ns = rng.uniform(N_lo, N_hi, nN) if N_hi > N_lo else np.full(nN, N_lo)
        Fx, _ = exact_rows(model, comp, p, float(T), Ns)
        Fe, _ = tab.flux_batch(np.full(nN, T), Ns)
        Fe[:, ~mask] = Fx[:, ~mask]
        a, b, c = errors(Fe, Fx, np.asarray(sigma, float), f_ref, a_max, return_floor=True)
        es.append(a); ef.append(b); fl.append(c)
    es, ef, fl = np.concatenate(es), np.concatenate(ef), np.concatenate(fl)
    return {"n": int(len(es)), "max_sigma": float(es.max()), "p99_sigma": float(np.percentile(es, 99)),
            "max_flux": float(ef.max()), "p99_flux": float(np.percentile(ef, 99)), "flux_below_noise": float(fl.mean()),
            "pixels_excluded": int((~mask).sum()), "time_s": time.time() - t0}


def fallback_decision(sc: dict, settings) -> tuple[bool, str]:
    """(fall back to the exact model?, reason) for one unit's spot check `sc` (see `spot_check`).

    sc["max_sigma"] / ["max_flux"] = emulator vs the exact model at the table's oversample (the reference).
    sc["vs_fit_oversample"] = emulator vs the exact model at fit.oversample, i.e. the model a fallback would use,
    on the same random (T, log N) points.  By the triangle inequality the fit's exact model is at least
    vs_fit - max_sigma off the reference: a fallback only helps if the emulator is worse than that.
    In the 0.21 survey 110 of the 111 fallbacks replaced a 0.1-5 sigma emulator by an exact model that was at least
    1.4-47 sigma (median 1.9 sigma, 13x the emulator's error) off the reference, at ~15x the sampling time."""
    mode = (getattr(settings, "fallback", "relative") or "relative").lower()
    if mode not in ("relative", "absolute", "never"):
        raise ValueError(f"fit.emulator.fallback must be relative | absolute | never, not {mode!r}")
    es, ef = float(sc["max_sigma"]), float(sc["max_flux"])
    miss = es > settings.target_sigma or ef > settings.target_flux
    head = f"spot check max {es:.3f} sigma / {100 * ef:.3f} % (targets {settings.target_sigma} / {100 * settings.target_flux} %)"
    if not miss:
        return False, f"ok: {head}"
    if mode == "never":
        return False, f"kept (fallback: never): {head}"
    if mode == "absolute":
        return True, f"spot check failed: {head}"
    v = sc.get("vs_fit_oversample")
    if not v:                                    # the fit's exact model IS the reference: it is the better model
        return True, f"spot check failed: {head}"
    xs = max(float(v["max_sigma"]) - es, 0.0)                    # lower bound of |exact(fit) - reference| / sigma
    xf = max(float(v.get("max_flux", 0.0)) - ef, 0.0)
    if es > xs or (ef > settings.target_flux and ef > xf):
        return True, (f"spot check failed: {head}; the exact model at fit.oversample {v['oversample']} may be "
                      f"closer (>= {xs:.3f} sigma / {100 * xf:.3f} % off the reference)")
    return False, (f"kept: {head}, but the exact model at fit.oversample {v['oversample']} is >= {xs:.3f} sigma off "
                   f"the reference ({xs / max(es, 1e-12):.0f}x the emulator's error); raise fit.oversample to "
                   f"{sc.get('oversample', 6)} if you want the exact model to be the better one")


def clone_model(model, oversample: int):
    """The same SlabModel (components, line lists, pixels, distance, windows, R, continuum) at another oversample."""
    m = SlabModel(model.components, model.linelists, model.wave_pix, model.distance_pc, model.windows, oversample=oversample,
                  R_model=model.R_model, R_scale=model.R_scale, R_constant=model.R_constant, tau_min_line=model.tau_min_line,
                  logN_max=model.logN_max, releases=model.releases, continuum=model.continuum_pix)
    return m


def default_box(bounds: dict) -> dict:
    """A unit's bounds widened to the default prior box (used when no survey boxes are given)."""
    from .fit import DEFAULT_BOUNDS
    T = (min(bounds["T"][0], DEFAULT_BOUNDS["T"][0]), max(bounds["T"][1], DEFAULT_BOUNDS["T"][1]))
    N = (min(bounds["logN"][0], DEFAULT_BOUNDS["logN"][0]), max(bounds["logN"][1], DEFAULT_BOUNDS["logN"][1]))
    return {"T": T, "logN": N}


def attach_shared_emulator(model, bounds: dict[str, dict], sigma, f_ref: float, settings: EmulatorSettings | None = None,
                           free_keys: set[str] | None = None, a_max: dict | None = None, say=None,
                           spot_seed: int = 0) -> EmulatorSet:
    """Shared-table counterpart of emulator.attach_emulator: load (or build) the shared table of every emulable
    unit, project it onto this disk's pixels, spot-check it against the exact model and attach the set."""
    settings = settings or EmulatorSettings()
    say = say or (lambda m: None)
    t_all = time.time()
    ok, why = emulable(model, free_keys or set())
    P = model.resolve_params()
    dense = dense_spec(settings)
    grid = DenseGrid(dense, model.R_model, model.R_scale, model.R_constant)
    sigma = np.asarray(sigma, float)
    boxes = settings.boxes or {}
    tables, info = {}, {"cache_dir": shared_dir(settings), "cache": "shared", "files": {}, "spot_check": {},
                        "timing": {}, "fallback": {}}
    ops: dict[float, tuple] = {}                                  # rv -> (P_disk, covered)
    ref_model = None                                              # exact model at the table's oversample (spot check)
    fine_ranges = [(float(model.grid.x[a]), float(model.grid.x[b - 1])) for a, b in model.grid.segments]
    exact_mask = model.covered
    for key, comp in ok.items():
        b = bounds.get(key)
        if b is None:
            why[key] = "no bounds given"; continue
        p = P[comp.name]
        am = float((a_max or {}).get(key, 10.0 ** 3))
        box = boxes.get(comp.molecule) or default_box(b)
        if not (box["T"][0] <= b["T"][0] * (1 + 1e-9) and b["T"][1] <= box["T"][1] * (1 + 1e-9)
                and box["logN"][0] <= b["logN"][0] + 1e-9 and b["logN"][1] <= box["logN"][1] + 1e-9):
            why[key] = (f"bounds T {b['T'][0]:.0f}-{b['T'][1]:.0f} K, log N {b['logN'][0]:.2f}-{b['logN'][1]:.2f} outside the "
                        f"survey box T {box['T'][0]:.0f}-{box['T'][1]:.0f} K, log N {box['logN'][0]:.2f}-{box['logN'][1]:.2f}")
            info["fallback"][key] = why[key]; continue
        spec = spec_from_component(model, comp, p, box, settings.table_oversample)
        t0 = time.time()
        shared, how = get_or_build(spec, settings, dense, say, grid)
        if shared is None:
            why[key] = how
            info["fallback"][key] = how
            warnings.warn(f"emulator {key}: {how}; using the exact model for this unit")
            continue
        if not shared.covers(b["T"], b["logN"]):
            why[key] = f"shared table {os.path.basename(shared.path)} does not cover the bounds"
            info["fallback"][key] = why[key]; continue
        rv = float(p.get("rv", 0.0))
        if rv not in ops:
            ops[rv] = grid.pixel_operator(model.wave_pix, rv, fine_ranges=fine_ranges)
        Pd, cov = ops[rv]
        missing = np.flatnonzero(exact_mask & ~cov)
        if len(missing):
            why[key] = f"{len(missing)} pixels outside the dense grid (e.g. {model.wave_pix[missing[0]]:.3f} um)"
            info["fallback"][key] = why[key]; continue
        t1 = time.time()
        tab = project_table(shared, grid, model, comp, p, sigma, f_ref, am, key, Pd, cov)
        t2 = time.time()
        tab.meta["load_s"] = t1 - t0; tab.meta["project_s"] = t2 - t1; tab.meta["how"] = how
        if settings.spot_check > 0:
            if ref_model is None:
                ref_model = model if model.oversample == spec.oversample else clone_model(model, spec.oversample)
            sc = spot_check(ref_model, comp, p, tab, b, sigma, f_ref, am, settings.spot_check, seed=spot_seed, mask=model.covered)
            sc["oversample"] = int(spec.oversample)
            if ref_model is not model:
                # the fit's own exact model (its oversample differs): its discretisation error vs the table, reported only
                sd = spot_check(model, comp, p, tab, b, sigma, f_ref, am, settings.spot_check, seed=spot_seed)
                sc["vs_fit_oversample"] = {"oversample": int(model.oversample), "max_sigma": sd["max_sigma"],
                                           "p99_sigma": sd["p99_sigma"], "max_flux": sd["max_flux"], "time_s": sd["time_s"]}
            tab.meta["spot_check"] = sc
            info["spot_check"][key] = sc
            fall, note = fallback_decision(sc, settings)
            sc["decision"] = note
            if note.startswith("kept"):
                info.setdefault("kept_despite_target", {})[key] = note
                say(f"  emulator {key}: {note}")
            if fall:
                why[key] = note
                info["fallback"][key] = why[key]
                warnings.warn(f"emulator {key}: {why[key]}; using the exact model for this unit")
                say(f"  emulator {key}: {why[key]}")
                continue
        say(f"  emulator {key}: {how} {os.path.basename(shared.path)} ({shared.meta.get('nodes')} nodes, "
            f"{shared.nbytes / 1e6:.0f} MB shared, {len(tab.pix)} px) load {t1 - t0:.2f} s, project {t2 - t1:.2f} s"
            + (f", spot check max {tab.meta['spot_check']['max_sigma']:.3f} sigma / {100 * tab.meta['spot_check']['max_flux']:.3f} % "
               f"in {tab.meta['spot_check']['time_s']:.1f} s" if settings.spot_check > 0 else ""))
        tables[key] = tab
        info["files"][key] = shared.path
        info["timing"][key] = {"load_s": t1 - t0, "project_s": t2 - t1,
                               "spot_check_s": tab.meta.get("spot_check", {}).get("time_s", 0.0)}
    info["timing"]["total_s"] = time.time() - t_all
    em = EmulatorSet(tables, why, info)
    model.emulator = em
    return em


# ---------------------------------------------------------------------------------------------------
# Survey: which tables a set of configs could need
# ---------------------------------------------------------------------------------------------------

def _union(a, b):
    return (min(float(a[0]), float(b[0])), max(float(a[1]), float(b[1])))


def molecule_boxes(cfgs) -> dict[str, dict]:
    """Survey-wide (T, log N) box per molecule from one or more ProjectConfigs: the union over configs of every
    component's bounds (its own `bounds`, else fit.bounds_by_molecule, else the default prior box) and, with
    fit.auto_detect, of fit.bounds_by_molecule (else the default box) for every candidate.  A tied isotopologue's
    log N box is the parent's shifted by its ratio (the fixed value, or the ratio bounds, default 10-300); a
    candidate isotopologue gets its parent's box shifted by the default ratio range."""
    from .fit import DEFAULT_BOUNDS
    from .molecules import MOLECULES
    if not isinstance(cfgs, (list, tuple)):
        cfgs = [cfgs]
    boxes: dict[str, dict] = {}

    def add(mol, T, N):
        if mol in boxes:
            boxes[mol] = {"T": _union(boxes[mol]["T"], T), "logN": _union(boxes[mol]["logN"], N)}
        else:
            boxes[mol] = {"T": (float(T[0]), float(T[1])), "logN": (float(N[0]), float(N[1]))}

    r_lo, r_hi = DEFAULT_BOUNDS["ratio"]
    for cfg in cfgs:
        bbm = cfg.fit.bounds_by_molecule or {}

        def mol_bounds(mol, par):
            v = (bbm.get(mol) or {}).get(par)
            return tuple(v) if v else DEFAULT_BOUNDS[par]
        mols = {c.molecule for c in cfg.components}
        if cfg.fit.auto_detect:        # the detection writes components with fit.bounds_by_molecule (else default) bounds
            cands = cfg.fit.detect.candidates or [m for m in MOLECULES if MOLECULES[m].hitran_id]
            mols |= set(cands)
            for mol in sorted(cands):
                add(mol, mol_bounds(mol, "T"), mol_bounds(mol, "logN"))
        for c in cfg.components:
            if c.kind != "slab":
                continue
            T = tuple(c.bounds.get("T") or mol_bounds(c.molecule, "T"))
            N = tuple(c.bounds.get("logN") or mol_bounds(c.molecule, "logN"))
            if c.tie_to:
                par = next((d for d in cfg.components if d.name == c.tie_to), None)
                if par is not None:
                    T = tuple(par.bounds.get("T") or mol_bounds(par.molecule, "T"))
                    pN = tuple(par.bounds.get("logN") or mol_bounds(par.molecule, "logN"))
                    ratio = c.ratio if c.ratio is not None else (MOLECULES[c.molecule].default_ratio or 70.0)
                    if "ratio" in (c.fixed or []):
                        N = (pN[0] - np.log10(ratio), pN[1] - np.log10(ratio))
                    else:
                        rb = c.bounds.get("ratio") or (r_lo, r_hi)
                        N = (pN[0] - np.log10(rb[1]), pN[1] - np.log10(rb[0]))
            add(c.molecule, T, N)
        # isotopologue candidates of the detection are tied to their parent at the default ratio (free)
        if cfg.fit.auto_detect:
            for mol in sorted(cands):
                par = MOLECULES[mol].parent if mol in MOLECULES else None
                if par and par in boxes:
                    pb = boxes[par]
                    add(mol, pb["T"], (pb["logN"][0] - np.log10(r_hi), pb["logN"][1] - np.log10(r_lo)))
    return boxes


def survey_specs(cfgs) -> list[TableSpec]:
    """Every shared table the given configs could need: one per (molecule, release, width, box, R)."""
    if not isinstance(cfgs, (list, tuple)):
        cfgs = [cfgs]
    boxes = molecule_boxes(cfgs)
    seen, out = set(), []
    for cfg in cfgs:
        rel = cfg.linedata.releases or {}
        comps = list(cfg.components)
        mols = {c.molecule: None for c in comps}
        if cfg.fit.auto_detect:
            from .molecules import MOLECULES
            for m in (cfg.fit.detect.candidates or [m for m in MOLECULES if MOLECULES[m].hitran_id]):
                mols.setdefault(m, None)
        items = [(m, rel.get(m) or "hitran", 4.7, False, None, None) for m in mols]
        for c in comps:
            if c.kind != "slab" or getattr(c, "Tvib", None) is not None:
                continue
            items.append((c.molecule, c.linelist_release or rel.get(c.molecule) or "hitran", float(c.fwhm),
                          bool(c.fwhm_thermal), c.eup_max, c.linelist_path))
        for mol, release, fw, fth, eup, path in items:
            b = boxes[mol]
            spec = TableSpec(mol, release, b["T"], b["logN"], fw, fth, eup, path, cfg.R_model, float(cfg.R_scale),
                             None if cfg.R_constant is None else float(cfg.R_constant), int(cfg.fit.emulator.table_oversample))
            if spec not in seen:
                seen.add(spec); out.append(spec)
    return out


def build_survey(specs: list[TableSpec], settings: EmulatorSettings, n_jobs: int = 1, say=None, build=True) -> list[dict]:
    """Build (or verify in the cache) every table in `specs`, in parallel over tables with joblib."""
    say = say or print
    dense = dense_spec(settings)

    def one(spec):
        t0 = time.time()
        tab, how = get_or_build(spec, settings, dense, say if n_jobs == 1 else None, build=build)
        if tab is None:
            return {"table": spec.label(), "status": how, "wall_s": time.time() - t0}
        v = tab.meta.get("validation", {})
        return {"table": spec.label(), "molecule": spec.molecule, "linelist": spec.linelist, "status": how,
                "nodes": "x".join(map(str, tab.meta.get("nodes", []))), "support": int(len(tab.sup)), "MB": tab.nbytes / 1e6,
                "max_sigma": v.get("max_sigma"), "max_flux": v.get("max_flux"), "build_s": tab.meta.get("build_s"),
                "wall_s": time.time() - t0, "file": os.path.basename(tab.path)}
    if n_jobs == 1 or len(specs) <= 1:
        return [one(s) for s in specs]
    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_jobs, backend="loky")(delayed(one)(s) for s in specs)


def list_tables(settings: EmulatorSettings) -> list[dict]:
    """The shared tables in the cache with their key fields and sizes."""
    out = []
    for js in sorted(glob.glob(os.path.join(shared_dir(settings), "*.json"))):
        base = js[:-5]
        try:
            with open(js) as fh:
                info = json.load(fh)
            sp, meta = info["spec"], info.get("meta", {})
            npy = base + ".npy"
            out.append({"file": os.path.basename(base), "molecule": sp["molecule"], "linelist": _linelist_key(sp["molecule"], sp["release"], sp["linelist_path"]),
                        "T": sp["T"], "logN": sp["logN"], "fwhm": sp["fwhm"], "R": meta.get("key", {}).get("R"),
                        "nodes": meta.get("nodes"), "support": meta.get("support_points"),
                        "MB": os.path.getsize(npy) / 1e6 if os.path.exists(npy) else 0.0,
                        "max_sigma": meta.get("validation", {}).get("max_sigma"), "max_flux": meta.get("validation", {}).get("max_flux"),
                        "ref_snr": meta.get("key", {}).get("ref_snr"), "ppf": meta.get("key", {}).get("dense", {}).get("points_per_fwhm"),
                        "build_s": meta.get("build_s"), "complete": all(os.path.exists(f) for f in SharedTable.files(base))})
        except Exception as e:          # pragma: no cover
            out.append({"file": os.path.basename(base), "error": str(e)})
    return out


def config_table_status(cfg) -> list[dict]:
    """Per component (and per auto-detect candidate) of a config: the shared table that serves it, or why it
    will stay exact (the structural reasons of emulator.emulable at config level, a missing table, or bounds
    outside the survey box)."""
    from .molecules import MOLECULES
    settings = cfg.fit.emulator.settings(cfg)
    settings.cache = "shared"
    dense = dense_spec(settings)
    boxes = settings.boxes or molecule_boxes(cfg)
    rel = cfg.linedata.releases or {}
    comps = list(cfg.components)
    groups: dict[str, list] = {}
    for c in comps:
        if c.group:
            groups.setdefault(c.group, []).append(c)
    covers_all = any(c.kind == "absorption" and c.covers == "all" for c in comps)
    grids: dict = {}
    out = []

    def status_of(mol, release, fw, fth, eup, path, name):
        b = boxes[mol]
        spec = TableSpec(mol, release, b["T"], b["logN"], fw, fth, eup, path, cfg.R_model, float(cfg.R_scale),
                         None if cfg.R_constant is None else float(cfg.R_constant), int(cfg.fit.emulator.table_oversample))
        g = grids.get(spec.R_model) or DenseGrid(dense, spec.R_model, spec.R_scale, spec.R_constant)
        grids[spec.R_model] = g
        base, *_ = find_table(spec, settings, dense, g)
        return {"component": name, "molecule": mol, "linelist": spec.linelist, "T": f"{b['T'][0]:.0f}-{b['T'][1]:.0f}",
                "logN": f"{b['logN'][0]:.2f}-{b['logN'][1]:.2f}", "file": os.path.basename(base) if base else "—",
                "status": "shared table" if base else "exact: no shared table (run jalebi emulator build)"}
    for c in comps:
        why = None
        if covers_all and c.kind != "absorption":
            why = "exact: an absorber with covers=all multiplies the emission"
        elif c.kind == "absorption":
            why = "exact: absorption screen"
        elif c.kind == "annuli":
            why = "exact: annuli"
        elif c.group and len(groups.get(c.group, [])) > 1:
            why = f"exact: opacity group of {len(groups[c.group])}"
        elif c.Tvib is not None:
            why = "exact: T_vib"
        elif cfg.fit.fit_rv or cfg.fit.fit_fwhm:
            why = "exact: velocity or line width is free"
        if why:
            out.append({"component": c.name, "molecule": c.molecule, "linelist": _linelist_key(c.molecule, c.linelist_release or rel.get(c.molecule) or "hitran", c.linelist_path),
                        "T": "", "logN": "", "file": "—", "status": why})
            continue
        out.append(status_of(c.molecule, c.linelist_release or rel.get(c.molecule) or "hitran", float(c.fwhm), bool(c.fwhm_thermal),
                             c.eup_max, c.linelist_path, c.name))
    if cfg.fit.auto_detect:
        have = {c.molecule for c in comps}
        for mol in (cfg.fit.detect.candidates or [m for m in MOLECULES if MOLECULES[m].hitran_id]):
            if mol in have:
                continue
            out.append(status_of(mol, rel.get(mol) or "hitran", 4.7, False, None, None, f"(candidate) {mol}"))
    return out
