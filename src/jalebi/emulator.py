"""Precomputed emulator of the slab model: `fit.model_backend: emulator` (default `exact`).

For every emulable unit the pixel flux of a 1-au-radius slab, F(pixel; T, log N), is tabulated on the data's
own pixel grid -- after the MIRI LSF and pixel integration of every sub-band, at the component's fixed v_shift
and line width, with the component's own line list (e.g. "H2O:hitemp" for hot water, "H2O:hitran" for cold)
-- and interpolated at run time:

    ln(F + eps) on a tensor grid in (ln T, log N), 4-point Lagrange (cubic) per axis (or bilinear),
    nodes placed adaptively until the interpolation error is below the target everywhere it is checked.

Interpolating ln F rather than F: for optically thin gas F is proportional to N, so ln F is linear in log N,
and the Boltzmann / partition-function dependence on T is smooth in ln F. eps is a per-pixel floor
(1e-4 sigma for the largest area the data allow) so that line-free pixels do not dominate.

Error measure (the build target and the test): for a check point (T, log N) the emulator and the exact model
are compared at the area the data would at most allow, a = min(a_max, f_ref / max F_exact), with f_ref the
brightest continuum-subtracted pixel of the fit,

    e_sigma = max_pixels |F_emu - F_exact| a / sigma_pixel,      e_flux = |sum (F_emu - F_exact)| / sum F_exact.

(e_flux is taken relative to the 1-sigma noise of the integrated flux at that area instead when the flux is
weaker than that noise: a 0.1 % requirement on an undetectable flux, e.g. 100 K CO at 5 um, cannot be met
and does not matter.)  The build refines until e_sigma < tol_sigma and e_flux < tol_flux at every interval midpoint and cell centre
(default tolerances half the targets, 0.05 sigma / 0.05 %), then checks `n_validate` random points.

Emulable units: a single-member slab (`kind: slab`, no `Tvib`) whose velocity and width are not free, and
the isotopologues tied to one (each with its own table over its own log N range). Opacity groups with several
members (opacity summed before the exponential: a table in T and every member's log N), annuli,
absorption screens, T_vib components, free rv / fwhm and screens with covers=all fall back to the exact
model for that unit only; the model mixes both.

Tables are cached as .npz files under `fit.emulator.cache_dir` ($JALEBI_EMULATOR_DIR, else
~/.jalebi/emulator), keyed on the line-list content and file hash, the pixel grid and resolving power, the
fine grid, LSF version, v_shift, line width, distance, the component's windows, the (T, log N) bounds, the
noise and reference flux used for the certification, the tolerances and the jalebi version: a stale table
is never loaded.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import time
import warnings
from dataclasses import dataclass, field

import numpy as np

from .constants import C, JY

FORMAT_VERSION = 1
LSF_VERSION = "erf-pixel-gauss-v1"      # instrument.build_lsf_operator: Gaussian LSF x pixel box, erf form
LOG10PI = np.log10(np.pi)


def default_cache_dir() -> str:
    return os.environ.get("JALEBI_EMULATOR_DIR") or os.path.join(os.path.expanduser("~"), ".jalebi", "emulator")


@dataclass
class EmulatorSettings:
    """fit.emulator in the config (see config.EmulatorConfig)."""
    target_sigma: float = 0.1          # acceptance: max |emu - exact| / sigma
    target_flux: float = 1e-3          # acceptance: relative integrated-flux error
    safety: float = 0.5                # build tolerances = safety x targets
    method: str = "cubic"              # cubic (4-point Lagrange per axis) | linear
    n_start: tuple = (9, 9)            # initial nodes (T, log N)
    max_nodes: tuple = (257, 257)
    n_validate: int = 200              # random checks at the end of a build
    cache_dir: str | None = None
    rebuild: bool = False
    seed: int = 0


# ---------------------------------------------------------------------------------------------------
# Exact unit fluxes, a whole row of log N at one T (the same arithmetic as SlabModel.unit_fluxes)
# ---------------------------------------------------------------------------------------------------

def _shift_matrix(model, I: np.ndarray, rv: float) -> np.ndarray:
    """SlabModel._shift for every column of I (same linear interpolation, zero outside)."""
    if rv == 0.0:
        return I
    x = model.grid.x
    xs = x - rv * 1e3 / C
    j = np.searchsorted(x, xs) - 1
    ok = (xs >= x[0]) & (xs <= x[-1])
    j = np.clip(j, 0, len(x) - 2)
    f = (xs - x[j]) / (x[j + 1] - x[j])
    out = I[j] * (1 - f)[:, None] + I[j + 1] * f[:, None]
    out[~ok] = 0.0
    return out


def exact_rows(model, comp, p: dict, T: float, logNs) -> tuple[np.ndarray, np.ndarray]:
    """Pixel fluxes (Jy, R = 1 au) of a plain LTE slab at temperature T for every log N in `logNs`:
    (len(logNs), npix) and the peak optical depths.  One opacity product per T."""
    logNs = np.atleast_1d(np.asarray(logNs, float))
    pp = {**p, "T": float(T)}
    basis = model.basis(comp, model.line_fwhm(comp, pp))
    if basis.n_lines == 0:
        return np.zeros((len(logNs), len(model.wave_pix))), np.zeros(len(logNs))
    base = basis.phi @ basis.kappa(float(T))                       # tau / (N [m^-2])
    N = 10.0 ** logNs * 1e4
    tau = base[:, None] * N[None, :]
    tmax = tau.max(axis=0)
    I = model.planck(float(T))[:, None] * (-np.expm1(-tau))
    I = _shift_matrix(model, I, float(pp.get("rv", 0.0)))
    F = (model.K @ I).T * model._omega_unit / JY
    wm = model.window_mask(comp)
    if wm is not None:
        F = F * wm[None, :]
    return F, tmax


# ---------------------------------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------------------------------

def _stencil(nodes: np.ndarray, v, order: int):
    """Start index and Lagrange weights of the `order`-point stencil around v (vectorised over v)."""
    v = np.atleast_1d(np.asarray(v, float))
    n = len(nodes)
    k = min(order, n)
    if k == 1:
        return np.zeros(len(v), int), np.ones((len(v), 1))
    i = np.searchsorted(nodes, v, side="right") - 1
    s = np.clip(i - (k // 2 - 1), 0, n - k)
    X = nodes[s[:, None] + np.arange(k)[None, :]]               # (nv, k)
    W = np.ones((len(v), k))
    for a in range(k):
        for b in range(k):
            if a != b:
                W[:, a] *= (v - X[:, b]) / (X[:, a] - X[:, b])
    return s, W


def _stencil1(nodes: list, v: float, order: int):
    """Scalar `_stencil` in plain Python (a model call needs two per unit; NumPy overhead dominates)."""
    n = len(nodes)
    k = min(order, n)
    if k == 1:
        return 0, [1.0]
    i = bisect.bisect_right(nodes, v) - 1
    s = min(max(i - (k // 2 - 1), 0), n - k)
    X = nodes[s:s + k]
    W = []
    for a in range(k):
        w = 1.0
        xa = X[a]
        for b in range(k):
            if a != b:
                w *= (v - X[b]) / (xa - X[b])
        W.append(w)
    return s, W


@dataclass
class UnitTable:
    """One emulated unit: ln(F + eps) on the support pixels over a (ln T, log N) tensor grid."""
    unit: str
    component: str
    molecule: str
    linelist: str
    lnT: np.ndarray                    # nodes (sorted)
    logN: np.ndarray                   # nodes (sorted)
    pix: np.ndarray                    # support pixel indices into the model's pixel array
    L: np.ndarray                      # float32 (nT, nN, npix_support): ln(F + eps)
    eps: np.ndarray                    # float32 (npix_support)
    ltau: np.ndarray                   # (nT, nN) log10 peak optical depth
    npix: int
    rv: float
    fwhm: float
    method: str = "cubic"
    meta: dict = field(default_factory=dict)
    _lists: tuple | None = field(default=None, repr=False)

    @property
    def order(self) -> int:
        return 4 if self.method == "cubic" else 2

    def covers(self, T, logN, rv=None, fwhm=None) -> bool:
        if rv is not None and abs(rv - self.rv) > 1e-9:
            return False
        if fwhm is not None and abs(fwhm - self.fwhm) > 1e-9:
            return False
        lt = np.log(T)
        tol = 1e-9
        return bool(self.lnT[0] - tol <= lt <= self.lnT[-1] + tol and self.logN[0] - tol <= logN <= self.logN[-1] + tol)

    def flux(self, T: float, logN: float) -> tuple[np.ndarray, float]:
        """(pixel flux for R = 1 au on the full pixel array, peak tau)."""
        if self._lists is None:
            self._lists = (self.lnT.tolist(), self.logN.tolist())
        a, wT = _stencil1(self._lists[0], math.log(T), self.order)
        b, wN = _stencil1(self._lists[1], float(logN), self.order)
        kT, kN = len(wT), len(wN)
        w = np.multiply.outer(wT, wN).ravel().astype(np.float32)
        lnF = w @ self.L[a:a + kT, b:b + kN].reshape(kT * kN, -1)
        out = np.zeros(self.npix)
        out[self.pix] = np.exp(lnF) - self.eps
        lt = float(np.asarray(wT) @ self.ltau[a:a + kT, b:b + kN] @ np.asarray(wN))
        return out, 10.0 ** lt

    def flux_batch(self, T, logN) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised over walkers: (nw, npix) fluxes and (nw,) peak tau."""
        T = np.asarray(T, float); logN = np.asarray(logN, float)
        sT, wT = _stencil(self.lnT, np.log(T), self.order)
        sN, wN = _stencil(self.logN, logN, self.order)
        kT, kN = wT.shape[1], wN.shape[1]
        iT = sT[:, None] + np.arange(kT)[None, :]
        iN = sN[:, None] + np.arange(kN)[None, :]
        G = self.L[iT[:, :, None], iN[:, None, :]]                       # (nw, kT, kN, nsup)
        W = (wT[:, :, None] * wN[:, None, :]).astype(np.float32)
        lnF = np.einsum("wab,wabp->wp", W, G, optimize=True)
        out = np.zeros((len(T), self.npix))
        out[:, self.pix] = np.exp(lnF.astype(float)) - self.eps
        lt = np.einsum("wa,wab,wb->w", wT, self.ltau[iT[:, :, None], iN[:, None, :]], wN)
        return out, 10.0 ** lt

    def save(self, path: str):
        np.savez(path, lnT=self.lnT, logN=self.logN, pix=self.pix, L=self.L, eps=self.eps, ltau=self.ltau,
                 npix=self.npix, rv=self.rv, fwhm=self.fwhm, method=self.method,
                 info=json.dumps({"unit": self.unit, "component": self.component, "molecule": self.molecule,
                                  "linelist": self.linelist, "meta": self.meta}, default=float))

    @classmethod
    def load(cls, path: str) -> "UnitTable":
        z = np.load(path, allow_pickle=False)
        info = json.loads(str(z["info"]))
        return cls(info["unit"], info["component"], info["molecule"], info["linelist"], z["lnT"], z["logN"], z["pix"],
                   z["L"], z["eps"], z["ltau"], int(z["npix"]), float(z["rv"]), float(z["fwhm"]), str(z["method"]),
                   info.get("meta", {}))


# ---------------------------------------------------------------------------------------------------
# Hashing (cache keys)
# ---------------------------------------------------------------------------------------------------

_FILE_HASH: dict = {}


def _file_sha(path: str | None) -> str:
    if not path or not os.path.exists(path):
        return ""
    st = os.stat(path)
    k = (path, st.st_size, st.st_mtime_ns)
    if k not in _FILE_HASH:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        _FILE_HASH[k] = h.hexdigest()
    return _FILE_HASH[k]


def _arr_sha(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a))
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def linelist_hash(model, comp, fwhm: float) -> str:
    """Content hash of the lines that reach the opacity basis (after selection and pruning), their partition
    function, and the hash of the source file."""
    b = model.basis(comp, fwhm)
    ll = model.linelist_for(comp)
    Tq = np.array([50.0, 100.0, 200.0, 500.0, 1000.0, 1500.0, 3000.0])
    Q = np.array([b.partition(t) for t in Tq])
    return _arr_sha(b.idx, b.a, b.gu, b.eu, b.el, b.lam3_8pi, Q) + ":" + _file_sha(getattr(ll, "source", None))


def table_key(model, comp, p: dict, bounds: dict, sigma, f_ref: float, a_max: float, settings: EmulatorSettings) -> str:
    from . import __version__
    fw = model.line_fwhm(comp, {**p, "T": 500.0}) if not comp.fwhm_thermal else float(p["fwhm"])
    d = {"format": FORMAT_VERSION, "jalebi": __version__, "lsf": LSF_VERSION,
         "linelist": linelist_hash(model, comp, fw), "linelist_key": model.linelist_key(comp),
         "pixels": _arr_sha(model.wave_pix), "lsf_matrix": _arr_sha(model.K.indptr, model.K.indices, model.K.data),
         "fine_grid": _arr_sha(model.grid.x), "R": [model.R_model, model.R_scale, model.R_constant],
         "distance_pc": model.distance_pc, "rv": float(p.get("rv", 0.0)), "fwhm": float(p["fwhm"]),
         "fwhm_thermal": bool(comp.fwhm_thermal), "windows": comp.windows, "eup_max": comp.eup_max,
         "bounds": {k: [float(v[0]), float(v[1])] for k, v in sorted(bounds.items())},
         "sigma": _arr_sha(np.asarray(sigma, np.float64)), "f_ref": float(f_ref), "a_max": float(a_max),
         "target": [settings.target_sigma, settings.target_flux, settings.safety], "method": settings.method,
         "n_start": list(settings.n_start), "max_nodes": list(settings.max_nodes)}
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()


# ---------------------------------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------------------------------

def errors(F_emu, F_ex, sigma, f_ref, a_max, return_floor: bool = False):
    """(e_sigma, e_flux) of emulated vs exact rows (n, npix) at the largest area the data allow.

    e_flux = |sum(F_emu - F_exact)| / max(sum F_exact, sigma_int / a): relative to the integrated flux, or to
    the 1-sigma noise of the integrated flux (sigma_int = sqrt(sum sigma^2)) at that area when the flux is
    weaker than that -- a 0.1 % error on a flux nobody can detect (cold CO at 5 um) is not a requirement.
    With return_floor, also whether the noise floor was used."""
    F_emu = np.atleast_2d(F_emu); F_ex = np.atleast_2d(F_ex)
    peak = np.max(F_ex, axis=1)
    a = np.minimum(a_max, f_ref / np.maximum(peak, 1e-300))
    es = np.max(np.abs(F_emu - F_ex) / sigma[None, :], axis=1) * a
    tot = np.sum(F_ex, axis=1)
    floor = np.sqrt(np.sum(np.asarray(sigma, float) ** 2)) / a
    ef = np.abs(np.sum(F_emu - F_ex, axis=1)) / np.maximum(np.maximum(tot, floor), 1e-300)
    return (es, ef, tot < floor) if return_floor else (es, ef)


def build_unit_table(model, comp, p: dict, T_bounds, logN_bounds, sigma, f_ref: float, a_max: float,
                     settings: EmulatorSettings, unit: str | None = None, say=None) -> UnitTable:
    """Adaptive tensor grid in (ln T, log N) for one unit (see the module docstring)."""
    t0 = time.time()
    say = say or (lambda m: None)
    sigma = np.asarray(sigma, float)
    tol_s = settings.safety * settings.target_sigma
    tol_f = settings.safety * settings.target_flux
    fw = float(p["fwhm"])
    T_lo, T_hi = map(float, T_bounds); N_lo, N_hi = map(float, logN_bounds)
    fixT, fixN = T_hi <= T_lo, N_hi <= N_lo
    lnT = np.array([np.log(T_lo)]) if fixT else np.linspace(np.log(T_lo), np.log(T_hi), settings.n_start[0])
    logN = np.array([N_lo]) if fixN else np.linspace(N_lo, N_hi, settings.n_start[1])
    rows: dict[float, tuple[np.ndarray, np.ndarray]] = {}       # ln T -> (F (nN, npix), tmax)
    n_exact = [0]

    def ex(lt, Ns):
        F, tm = exact_rows(model, comp, p, float(np.exp(lt)), Ns)
        n_exact[0] += len(np.atleast_1d(Ns))
        return F, tm

    def node_rows():
        for lt in lnT:
            have = rows.get(lt)
            if have is None or have[0].shape[0] != len(logN):
                rows[lt] = ex(lt, logN)
        return np.stack([rows[lt][0] for lt in lnT]), np.stack([rows[lt][1] for lt in lnT])

    def make_table(Fn, tm):
        peak = Fn.max(axis=2)
        a = np.minimum(a_max, f_ref / np.maximum(peak, 1e-300))
        amax = float(np.max(a))
        scaled = (Fn * a[:, :, None]) / sigma[None, None, :]
        sup = np.flatnonzero(np.max(scaled, axis=(0, 1)) > 1e-4)
        eps = 1e-4 * sigma[sup] / amax
        L = np.log(np.maximum(Fn[:, :, sup], 0.0) + eps[None, None, :]).astype(np.float32)
        ltau = np.log10(np.maximum(tm, 1e-30))
        return UnitTable(unit or comp.name, comp.name, comp.molecule, model.linelist_key(comp), lnT.copy(), logN.copy(),
                         sup, L, eps.astype(np.float32), ltau, len(model.wave_pix), float(p.get("rv", 0.0)), fw,
                         settings.method)

    def check(tab, lts, Ns):
        """errors at (each lt in lts) x Ns"""
        es, ef = [], []
        for lt in lts:
            F, _ = ex(lt, Ns)
            E = np.stack([tab.flux(float(np.exp(lt)), n)[0] for n in Ns])
            a, b = errors(E, F, sigma, f_ref, a_max)
            es.append(a); ef.append(b)
        return np.array(es), np.array(ef)

    it = 0
    while True:
        it += 1
        Fn, tm = node_rows()
        tab = make_table(Fn, tm)
        badT = np.zeros(max(len(lnT) - 1, 0), bool)
        badN = np.zeros(max(len(logN) - 1, 0), bool)
        Nm = 0.5 * (logN[1:] + logN[:-1])
        Tm = 0.5 * (lnT[1:] + lnT[:-1])
        if len(Tm):          # T midpoints at every log N node and cell centre
            es, ef = check(tab, Tm, logN)
            badT |= np.any((es > tol_s) | (ef > tol_f), axis=1)
            if len(Nm):
                es, ef = check(tab, Tm, Nm)
                c = (es > tol_s) | (ef > tol_f)
                badT |= np.any(c, axis=1); badN |= np.any(c, axis=0)
        if len(Nm):          # log N midpoints at every T node
            es, ef = check(tab, lnT, Nm)
            badN |= np.any((es > tol_s) | (ef > tol_f), axis=0)
        say(f"    {comp.name}: pass {it}: {len(lnT)} x {len(logN)} nodes, refine {badT.sum()} T and {badN.sum()} log N intervals")
        if not badT.any() and not badN.any():
            break
        if len(lnT) + badT.sum() > settings.max_nodes[0] or len(logN) + badN.sum() > settings.max_nodes[1]:
            warnings.warn(f"emulator {comp.name}: node limit reached before the tolerance; the table is less accurate")
            break
        if badT.any():
            lnT = np.sort(np.concatenate([lnT, Tm[badT]]))
        if badN.any():
            logN = np.sort(np.concatenate([logN, Nm[badN]]))
            rows.clear()
    # final random validation
    rng = np.random.default_rng(settings.seed)
    nv = settings.n_validate
    lt = rng.uniform(lnT[0], lnT[-1], nv) if not fixT else np.full(nv, lnT[0])
    Ns = rng.uniform(logN[0], logN[-1], nv) if not fixN else np.full(nv, logN[0])
    es_v, ef_v = np.zeros(nv), np.zeros(nv)
    for i in range(nv):
        F, _ = ex(lt[i], [Ns[i]])
        E, _ = tab.flux(float(np.exp(lt[i])), Ns[i])
        es_v[i], ef_v[i] = (x[0] for x in errors(E, F, sigma, f_ref, a_max))
    tab.meta = {"nodes": [len(lnT), len(logN)], "support_pixels": int(len(tab.pix)), "exact_evaluations": n_exact[0],
                "build_s": time.time() - t0, "validation": {"n": nv, "max_sigma": float(es_v.max(initial=0)),
                "p99_sigma": float(np.percentile(es_v, 99)) if nv else 0.0, "max_flux": float(ef_v.max(initial=0)),
                "p99_flux": float(np.percentile(ef_v, 99)) if nv else 0.0},
                "T_bounds": [T_lo, T_hi], "logN_bounds": [N_lo, N_hi], "f_ref": f_ref, "a_max": a_max,
                "tolerance": [tol_s, tol_f], "passes": it}
    v = tab.meta["validation"]
    if v["max_sigma"] > settings.target_sigma or v["max_flux"] > settings.target_flux:
        warnings.warn(f"emulator {comp.name}: random validation max {v['max_sigma']:.3g} sigma / {v['max_flux']:.2e} in flux "
                      f"exceeds the target ({settings.target_sigma} / {settings.target_flux})")
    return tab


# ---------------------------------------------------------------------------------------------------
# A set of tables attached to a SlabModel
# ---------------------------------------------------------------------------------------------------

def emulable(model, free_keys: set[str]) -> tuple[dict, dict]:
    """({unit: component} that can be emulated, {unit or component: reason} that stay exact)."""
    units = model._units()
    by_name = {c.name: c for c in model.components if c.enabled}
    ok, why = {}, {}
    if any(c.kind == "absorption" and c.covers == "all" for c in by_name.values()):
        for key, members in units.items():
            if members[0].kind != "absorption":
                why[key] = "an absorber with covers=all multiplies the emission on the fine grid"
    for key, members in units.items():
        if key in why:
            continue
        lead = members[0]
        if lead.kind == "absorption":
            why[key] = "absorption screen"; continue
        if lead.kind == "annuli":
            why[key] = "annuli (T(r), N(r), R_in, R_out)"; continue
        if len(members) > 1:
            why[key] = f"opacity group of {len(members)} (summed opacity: one dimension per member's log N)"; continue
        c = lead
        par = by_name.get(c.tie_to) if c.tie_to else c
        if c.tie_to and (par is None or par.kind != "slab" or (par.group and len(units.get(par.group, [])) > 1)):
            why[key] = "tied to a component that is not a plain slab"; continue
        if c.Tvib is not None or (par is not None and par.Tvib is not None):
            why[key] = "T_vib (two-temperature populations)"; continue
        src = par.name
        if f"{src}.rv" in free_keys or f"{src}.fwhm" in free_keys:
            why[key] = "velocity or line width is free"; continue
        ok[key] = c
    return ok, why


class EmulatorSet:
    """Tables for the emulable units of one SlabModel; `SlabModel.unit_fluxes` asks it first."""

    def __init__(self, tables: dict[str, UnitTable], exact_units: dict[str, str], info: dict | None = None):
        self.tables = tables
        self.exact_units = exact_units
        self.info = info or {}
        self.enabled = True
        self.misses = 0

    def unit_flux(self, key: str, p: dict):
        if not self.enabled:
            return None
        t = self.tables.get(key)
        if t is None:
            return None
        if not t.covers(p["T"], p["logN"], p.get("rv", 0.0), p.get("fwhm")):
            self.misses += 1
            return None
        return t.flux(p["T"], p["logN"])

    def unit_fluxes_many(self, model, Ps: list[dict]):
        """Per-unit 1-au fluxes for many parameter sets at once, or None when a unit is not emulated or a
        walker is outside a table (the caller then loops over the exact model).  Returns
        ({unit: (nw, npix)}, {unit: (nw,) logR}) with the ties already resolved in `Ps`."""
        if not self.enabled:
            return None
        units = model._units()
        if set(units) - set(self.tables):
            return None
        F, lR = {}, {}
        for key, t in self.tables.items():
            lead = units[key][0].name
            T = np.array([P[lead]["T"] for P in Ps]); N = np.array([P[lead]["logN"] for P in Ps])
            if np.any(T < np.exp(t.lnT[0]) * (1 - 1e-9)) or np.any(T > np.exp(t.lnT[-1]) * (1 + 1e-9)) or \
                    np.any(N < t.logN[0] - 1e-9) or np.any(N > t.logN[-1] + 1e-9) or \
                    any(abs(P[lead].get("rv", 0.0) - t.rv) > 1e-9 or abs(P[lead]["fwhm"] - t.fwhm) > 1e-9 for P in Ps):
                return None
            F[key] = t.flux_batch(T, N)[0]
            lR[key] = np.array([P[lead]["logR"] for P in Ps])
        return F, lR

    def summary(self) -> list[str]:
        out = []
        for k, t in self.tables.items():
            v = t.meta.get("validation", {})
            how = "cache" if t.meta.get("from_cache") else "built in {:.0f} s".format(t.meta.get("build_s", 0))
            out.append(f"{k} [{t.linelist}]: {t.meta.get('nodes')} nodes, {len(t.pix)} px, "
                       f"max {v.get('max_sigma', float('nan')):.3f} sigma / {100 * v.get('max_flux', float('nan')):.3f} % ({how})")
        for k, r in self.exact_units.items():
            out.append(f"{k}: exact ({r})")
        return out


def attach_emulator(model, bounds: dict[str, dict], sigma, f_ref: float, settings: EmulatorSettings | None = None,
                    free_keys: set[str] | None = None, a_max: dict | None = None, say=None) -> EmulatorSet:
    """Build (or load from the cache) the tables of every emulable unit of `model` and attach them
    (`model.emulator`).  bounds: {unit: {"T": (lo, hi), "logN": (lo, hi)}} (the tied isotopologue's log N
    range already shifted by its ratio); a_max: {unit: largest area R^2 [au^2]}."""
    settings = settings or EmulatorSettings()
    say = say or (lambda m: None)
    ok, why = emulable(model, free_keys or set())
    P = model.resolve_params()
    cache = settings.cache_dir or default_cache_dir()
    os.makedirs(cache, exist_ok=True)
    tables, info = {}, {"cache_dir": cache, "files": {}}
    for key, comp in ok.items():
        b = bounds.get(key)
        if b is None:
            why[key] = "no bounds given"; continue
        p = P[comp.name]
        am = float((a_max or {}).get(key, 10.0 ** 3))
        k = table_key(model, comp, p, b, sigma, f_ref, am, settings)
        # named by content (line list + key), not by component: identical tables (e.g. two water components with
        # the same release and bounds) are built once and shared
        fname = os.path.join(cache, f"{model.linelist_key(comp).replace(':', '-').replace('@', '_').replace('/', '_')[:40]}_{k[:24]}.npz")
        tab = None
        if os.path.exists(fname) and not settings.rebuild:
            try:
                tab = UnitTable.load(fname)
                tab.meta["from_cache"] = True
                say(f"  emulator {key}: loaded {os.path.basename(fname)}")
            except Exception as e:                      # corrupt file: rebuild
                say(f"  emulator {key}: cache unreadable ({e}); rebuilding")
                tab = None
        if tab is None:
            say(f"  emulator {key}: building ({model.linelist_key(comp)}, T {b['T'][0]:.0f}-{b['T'][1]:.0f} K, "
                f"log N {b['logN'][0]:.2f}-{b['logN'][1]:.2f})")
            tab = build_unit_table(model, comp, p, b["T"], b["logN"], sigma, f_ref, am, settings, unit=key, say=say)
            tab.meta["key"] = k
            tmp = fname + ".tmp.npz"
            tab.save(tmp)
            os.replace(tmp, fname)
            tab.meta["from_cache"] = False
        tab.meta["file"] = fname
        tab.meta["file_bytes"] = os.path.getsize(fname)
        tables[key] = tab
        info["files"][key] = fname
    em = EmulatorSet(tables, why, info)
    model.emulator = em
    return em
