"""Forward model: LTE slab components on a fine ln(lambda) grid, convolved to MIRI pixels.

For one component c with column N (cm^-2), temperature T, line width FWHM dv and radial
velocity v:

    tau_c(x)  = N_c * sum_l kappa_l(T_c) phi(x - x_l - v/c)         (Phi is sparse)
    I_c(x)    = B_nu(T_c) (1 - exp(-tau_c))
    F_c(pix)  = Omega_c * K I_c                                      (Omega = pi R^2 / d^2)

Components in the same *opacity group* sum their tau before the exponential (shared T and
Omega).  Everything that does not depend on the parameters (Phi, K, B_nu grid) is built once.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import nnls

from .constants import AU, C, FWHM_TO_SIGMA, H, JY, KB, PC
from .instrument import build_lsf_operator, resolving_power
from .linedata import LineList
from .molecules import get_molecule

PARAM_NAMES = ("logN", "T", "logR", "rv", "fwhm")
PARAM_LABELS = {"logN": "log N [cm⁻²]", "T": "T [K]", "logR": "log R [au]", "rv": "v [km/s]",
                "fwhm": "Δv [km/s]", "logNA": "log(N·A) [cm⁻² au²]", "ratio": "ratio", "q": "q", "p": "p"}


# --------------------------------------------------------------------------
# Fine grid
# --------------------------------------------------------------------------

def merge_intervals(intervals, margin=0.0):
    iv = sorted((a - margin, b + margin) for a, b in intervals)
    out = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


@dataclass
class FineGrid:
    """Uniform grid in x = ln(lambda) covering a set of wavelength windows.

    step_kms : velocity step (default: min line FWHM / oversample)
    margin_um: extra range on each side of every window (to catch LSF and line wings)
    """
    windows: list[tuple[float, float]]
    step_kms: float = 4.7 / 8
    margin_um: float = 0.03

    def __post_init__(self):
        self.dx = self.step_kms * 1e3 / C
        segs = merge_intervals(self.windows, self.margin_um)
        xs, bounds = [], []
        n = 0
        for a, b in segs:
            xa, xb = np.log(a), np.log(b)
            k = int(np.ceil((xb - xa) / self.dx)) + 1
            x = xa + self.dx * np.arange(k)
            xs.append(x); bounds.append((n, n + k)); n += k
        self.x = np.concatenate(xs)
        self.segments = bounds
        self.wave = np.exp(self.x)
        self.n = len(self.x)

    def covers(self, wave_um):
        w = np.asarray(wave_um)
        m = np.zeros(w.shape, bool)
        for a, b in self.segments:
            m |= (w >= self.wave[a]) & (w <= self.wave[b - 1])
        return m


# --------------------------------------------------------------------------
# Opacity basis for one line list on one grid
# --------------------------------------------------------------------------

class OpacityBasis:
    """Sparse profile matrix Phi (n_fine x n_lines) for a line list on a fine grid.

    Phi[:, l] is a normalised Gaussian in x (ln lambda) of sigma = FWHM/(2.355 c),
    truncated at `truncate` sigma, scaled so that N[m^-2] * Phi @ kappa gives tau.
    """

    def __init__(self, lines: LineList, grid: FineGrid, fwhm_kms: float = 4.7, truncate: float = 4.0,
                 margin_sigma: float = 6.0):
        self.lines = lines
        self.grid = grid
        self.fwhm_kms = fwhm_kms
        sig = fwhm_kms * 1e3 / C * FWHM_TO_SIGMA
        self.sigma_x = sig
        xl_all = np.log(lines.wave)
        # keep lines that fall inside a segment (with a margin of a few sigma)
        keep = np.zeros(len(xl_all), bool)
        for a, b in grid.segments:
            keep |= (xl_all >= grid.x[a] - margin_sigma * sig) & (xl_all <= grid.x[b - 1] + margin_sigma * sig)
        self.idx = np.flatnonzero(keep)
        xl = xl_all[self.idx]
        self.n_lines = len(xl)
        half = truncate * sig
        rows, cols, vals = [], [], []
        x = grid.x
        j0 = np.searchsorted(x, xl - half)
        j1 = np.searchsorted(x, xl + half)
        for l in range(self.n_lines):
            a, b = j0[l], j1[l]
            if b <= a:
                continue
            xj = x[a:b]
            w = np.exp(-0.5 * ((xj - xl[l]) / sig) ** 2)
            # normalise: integral over x of phi dx = 1, and phi_v = phi_x / c  [s/m]
            w /= (w.sum() * grid.dx) * C
            rows.append(np.arange(a, b)); cols.append(np.full(b - a, l)); vals.append(w)
        if rows:
            self.phi = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                         shape=(grid.n, self.n_lines))
        else:
            self.phi = sparse.csr_matrix((grid.n, max(self.n_lines, 1)))
        # per-line data for kappa(T)
        self.a = lines.a[self.idx]
        self.gu = lines.gu[self.idx]
        self.eu = lines.eu[self.idx]
        self.el = lines.el[self.idx]
        self.lam3_8pi = (lines.wave[self.idx] * 1e-6) ** 3 / (8.0 * np.pi)
        self.partition = lines.partition

    def kappa(self, T: float) -> np.ndarray:
        Z = self.partition(T)
        return self.a * self.gu * self.lam3_8pi * (np.exp(-self.el / T) - np.exp(-self.eu / T)) / Z

    def tau(self, N_cm2: float, T: float) -> np.ndarray:
        """Optical depth on the fine grid for column N (cm^-2)."""
        if self.n_lines == 0:
            return np.zeros(self.grid.n)
        return (N_cm2 * 1e4) * (self.phi @ self.kappa(T))

    def tau_peaks(self, N_cm2: float, T: float) -> np.ndarray:
        """Peak optical depth of every line individually (for the tau flag and line pruning)."""
        phi0 = 1.0 / (self.sigma_x * np.sqrt(2 * np.pi)) / C
        return (N_cm2 * 1e4) * self.kappa(T) * phi0


# --------------------------------------------------------------------------
# Components
# --------------------------------------------------------------------------

@dataclass
class Component:
    """One slab (or annular) emission component.

    kind: "slab" (N, T, R) or "annuli" (power-law T(r), N(r) between R_in and R_out)
    group: components with the same group name sum their opacity (shared T and area)
    tie_to: name of a parent component; this one then shares T, logR, rv, fwhm with it and
            logN = parent.logN - log10(ratio)
    """
    name: str
    molecule: str
    logN: float = 17.0
    T: float = 500.0
    logR: float = -0.5           # log10 of emitting radius in au
    rv: float = 0.0              # km/s
    fwhm: float = 4.7            # km/s
    kind: str = "slab"
    group: str | None = None
    tie_to: str | None = None
    ratio: float | None = None   # parent/child abundance ratio for a tied component
    q: float = 0.5               # T(r) power-law index (annuli)
    p: float = 1.0               # N(r) power-law index (annuli)
    logRin: float = -1.5         # inner radius (annuli), log10 au
    n_annuli: int = 20
    enabled: bool = True
    linelist_release: str | None = None   # "hitran", "hitemp", ...; None = model default for the molecule
    eup_max: float | None = None   # optional line selection
    linelist_path: str | None = None

    def params(self) -> dict:
        d = {"logN": self.logN, "T": self.T, "logR": self.logR, "rv": self.rv, "fwhm": self.fwhm}
        if self.kind == "annuli":
            d.update({"q": self.q, "p": self.p, "logRin": self.logRin})
        if self.tie_to:
            d["ratio"] = self.ratio if self.ratio is not None else get_molecule(self.molecule).default_ratio or 70.0
        return d

    def set_params(self, d: dict):
        for k, v in d.items():
            if hasattr(self, k):
                setattr(self, k, float(v))


# --------------------------------------------------------------------------
# Full model
# --------------------------------------------------------------------------

class SlabModel:
    """Sum of components evaluated on the pixels of a Spectrum (or any wavelength array).

    Parameters
    ----------
    components : list[Component]
    linelists  : dict molecule -> LineList (already restricted to the useful lines)
    wave_pix   : pixel wavelengths (micron) to evaluate on
    distance_pc: distance
    windows    : wavelength windows to model (fine grid built only there); default: pixel range
    oversample : fine-grid points per minimum line FWHM
    R_model, R_scale : resolving power model and multiplicative scale
    """

    def __init__(self, components: list[Component], linelists: dict[str, LineList], wave_pix: np.ndarray,
                 distance_pc: float = 140.0, windows: list[tuple[float, float]] | None = None,
                 oversample: int = 6, R_model: str = "argyriou2023", R_scale: float = 1.0,
                 R_constant: float | None = None, pixel_edges_x: tuple | None = None, tau_min_line: float = 1e-4,
                 logN_max: float = 21.0, releases: dict[str, str] | None = None):
        self.components = components
        # line lists keyed by "MOL:release" (or "MOL@path"); a bare "MOL" key is accepted as a fallback
        self.linelists = linelists
        self.releases = dict(releases or {})
        self.wave_pix = np.asarray(wave_pix, float)
        self.distance_pc = distance_pc
        if windows is None:
            windows = [(float(self.wave_pix.min()), float(self.wave_pix.max()))]
        self.windows = windows
        fwhm_min = min([c.fwhm for c in components] + [4.7])
        self.grid = FineGrid(windows, step_kms=fwhm_min / oversample)
        self.oversample = oversample
        self.R_model, self.R_scale, self.R_constant = R_model, R_scale, R_constant
        R = resolving_power(self.wave_pix, R_model, R_scale, R_constant)
        self.K = build_lsf_operator(self.grid.x, self.grid.dx, self.wave_pix, R, edges=pixel_edges_x)
        self.covered = np.asarray(self.K.sum(axis=1)).ravel() > 0.5
        self._bases: dict[tuple, OpacityBasis] = {}
        self.tau_min_line = tau_min_line
        self.logN_max = logN_max
        self._planck_cache: dict[float, np.ndarray] = {}
        # Optional memo of per-unit fluxes keyed on the unit's parameters (used by the interactive
        # app so that moving one slider only re-evaluates that component).  Off by default: the
        # fitter changes every parameter at once, where a cache would only add hashing overhead.
        self.unit_cache: dict | None = None
        self._omega_unit = (AU / (distance_pc * PC)) ** 2 * np.pi   # Omega for R = 1 au
        lam = self.grid.wave * 1e-6
        self._planck_c1 = 2.0 * H * C / lam**3
        self._planck_c2 = H * C / (lam * KB)

    # ---- helpers ------------------------------------------------------------
    def release_of(self, c: Component) -> str:
        """Line-list release a component uses: its own setting, else the model default for the molecule."""
        return c.linelist_release or self.releases.get(c.molecule) or "hitran"

    def linelist_key(self, c: Component) -> str:
        return linelist_key(c.molecule, self.release_of(c), c.linelist_path)

    def linelist_for(self, c: Component) -> LineList:
        key = self.linelist_key(c)
        ll = self.linelists.get(key)
        if ll is None:
            ll = self.linelists.get(c.molecule)          # legacy dict keyed by molecule only
        if ll is None:
            raise KeyError(f"no line list loaded for {key}")
        return ll

    def basis(self, molecule_or_component, fwhm: float) -> OpacityBasis:
        """Opacity basis for a component (or a bare molecule / line-list key) at a line width."""
        if isinstance(molecule_or_component, Component):
            key0 = self.linelist_key(molecule_or_component)
            ll = self.linelist_for(molecule_or_component)
        else:
            key0 = str(molecule_or_component)
            ll = self.linelists.get(key0) or self.linelists[key0.split(":")[0].split("@")[0]]
        key = (key0, round(float(fwhm), 3))
        if key not in self._bases:
            self._bases[key] = OpacityBasis(ll, self.grid, fwhm_kms=fwhm)
        return self._bases[key]

    def planck(self, T: float) -> np.ndarray:
        key = round(float(T), 2)
        if key not in self._planck_cache:
            if len(self._planck_cache) > 512:
                self._planck_cache.clear()
            self._planck_cache[key] = self._planck_c1 / np.expm1(np.minimum(self._planck_c2 / key, 700.0))
        return self._planck_cache[key]

    def omega(self, logR: float) -> float:
        return self._omega_unit * 10.0 ** (2.0 * logR)

    def _shift(self, arr: np.ndarray, rv_kms: float) -> np.ndarray:
        if rv_kms == 0.0:
            return arr
        return np.interp(self.grid.x - rv_kms * 1e3 / C, self.grid.x, arr, left=0.0, right=0.0)

    def resolve_params(self, params: dict[str, dict] | None = None) -> dict[str, dict]:
        """Full parameter dict per component with ties applied."""
        P = {c.name: dict(c.params()) for c in self.components if c.enabled}
        if params:
            for k, v in params.items():
                if k in P:
                    P[k].update(v)
        for c in self.components:
            if c.enabled and c.tie_to and c.tie_to in P:
                par = P[c.tie_to]
                P[c.name]["T"] = par["T"]; P[c.name]["logR"] = par["logR"]
                P[c.name]["rv"] = par["rv"]; P[c.name]["fwhm"] = par["fwhm"]
                P[c.name]["logN"] = par["logN"] - np.log10(P[c.name].get("ratio", 70.0))
        return P

    # ---- intensities per emitting unit ----------------------------------------
    def _slab_tau(self, c: Component, p: dict) -> np.ndarray:
        return self.basis(c, p["fwhm"]).tau(10.0 ** p["logN"], p["T"])

    def unit_fluxes(self, params: dict[str, dict] | None = None) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
        """Per emitting unit: pixel fluxes (Jy) for Omega of a 1-au-radius disk, for each
        independent unit (component or opacity group).  Returns (unit_flux, unit_tau_max, unit_logR)."""
        P = self.resolve_params(params)
        comps = [c for c in self.components if c.enabled]
        units: dict[str, list[Component]] = {}
        for c in comps:
            key = c.group if c.group else c.name
            units.setdefault(key, []).append(c)
        flux, taumax, logR = {}, {}, {}
        for key, members in units.items():
            lead = members[0]
            p0 = P[lead.name]
            ck = None
            if self.unit_cache is not None:
                # the 1-au flux does not depend on logR for slabs (only the annuli R_out does)
                skip = () if lead.kind == "annuli" else ("logR",)
                ck = (key, lead.kind, lead.n_annuli,
                      tuple((c.name, self.linelist_key(c), tuple(sorted((k, v) for k, v in P[c.name].items() if k not in skip)))
                            for c in members))
                hit = self.unit_cache.get(ck)
                if hit is not None:
                    flux[key], taumax[key] = hit
                    logR[key] = p0["logR"]
                    continue
            if lead.kind == "annuli" and len(members) == 1:
                I = self._annuli_intensity(lead, p0)
                tmax = np.nan
            else:
                tau = np.zeros(self.grid.n)
                for c in members:
                    p = P[c.name]
                    t = self._slab_tau(c, {**p, "T": p0["T"], "fwhm": p0["fwhm"]})
                    tau += t
                tmax = float(tau.max()) if tau.size else 0.0
                I = self.planck(p0["T"]) * (-np.expm1(-tau))
            I = self._shift(I, p0["rv"])
            flux[key] = (self.K @ I) * self._omega_unit / JY     # Jy for R = 1 au
            taumax[key] = tmax
            logR[key] = p0["logR"]
            if ck is not None:
                if len(self.unit_cache) > 256:
                    self.unit_cache.clear()
                self.unit_cache[ck] = (flux[key], tmax)
        return flux, taumax, logR

    def _annuli_intensity(self, c: Component, p: dict) -> np.ndarray:
        """Sum of annuli with T(r) = T0 (r/Rin)^-q, N(r) = N0 (r/Rin)^-p between Rin and R.
        Returned as intensity times (area / pi Rout^2) so that Omega(logR) scales it."""
        r_in, r_out = 10.0 ** p["logRin"], 10.0 ** p["logR"]
        if r_out <= r_in:
            r_out = r_in * 1.01
        edges = np.geomspace(r_in, r_out, c.n_annuli + 1)
        rc = np.sqrt(edges[1:] * edges[:-1])
        w = (edges[1:] ** 2 - edges[:-1] ** 2) / r_out**2
        basis = self.basis(c, p["fwhm"])
        I = np.zeros(self.grid.n)
        for r, wi in zip(rc, w):
            T = max(p["T"] * (r / r_in) ** (-p["q"]), 20.0)
            N = 10.0 ** p["logN"] * (r / r_in) ** (-p["p"])
            I += wi * self.planck(T) * (-np.expm1(-basis.tau(N, T)))
        return I

    def evaluate(self, params: dict[str, dict] | None = None, per_unit: bool = False):
        """Total model flux (Jy) on the pixels; optionally also the per-unit fluxes."""
        uf, tmax, logR = self.unit_fluxes(params)
        total = np.zeros(len(self.wave_pix))
        scaled = {}
        for key, f in uf.items():
            s = f * 10.0 ** (2.0 * logR[key])
            scaled[key] = s
            total += s
        if per_unit:
            return total, scaled, tmax
        return total

    def _unit_lead(self, key: str) -> Component | None:
        return next((c for c in self.components if c.enabled and (c.group or c.name) == key), None)

    def tied_units(self) -> dict[str, str]:
        """Unit of every tied component (isotopologue) -> unit of its parent.  They share the parent's
        emitting area, so for the linear area solve their fluxes belong in the parent's column."""
        by_name = {c.name: c for c in self.components if c.enabled}
        out = {}
        for c in by_name.values():
            if c.tie_to and c.tie_to in by_name:
                par = by_name[c.tie_to]
                ku, kp = c.group or c.name, par.group or par.name
                if ku != kp:
                    out[ku] = kp
        return out

    def fold_tied(self, unit_flux: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Per-unit 1-au fluxes with every tied unit added to its parent's unit (new arrays)."""
        tm = self.tied_units()
        if not tm:
            return unit_flux
        out = {k: v for k, v in unit_flux.items() if k not in tm}
        for ku, kp in tm.items():
            if ku in unit_flux and kp in out:
                out[kp] = out[kp] + unit_flux[ku]
        return out

    def solve_areas(self, data: np.ndarray, sigma: np.ndarray, params: dict[str, dict] | None = None,
                    mask: np.ndarray | None = None, fixed: set[str] | None = None) -> tuple[dict[str, float], float]:
        """Non-negative least squares for the emitting areas of every unit at fixed (N, T).

        The model is linear in the areas A_u = R_u^2 of slab units, so they are solved exactly.
        Tied isotopologues share their parent's area (their flux is added to the parent's column), and
        radial-gradient (annuli) units are held at their current logR, because there R sets the outer
        radius of the power law and the flux is not linear in R^2.  Returns ({unit: logR}, chi2)."""
        uf_all, _, logR = self.unit_fluxes(params)
        tm = self.tied_units()
        uf = self.fold_tied(uf_all)
        keys = list(uf)
        m = np.ones(len(data), bool) if mask is None else mask.copy()
        m &= np.isfinite(data) & np.isfinite(sigma) & (sigma > 0)
        fixed = set(fixed or set())
        fixed |= {k for k in keys if (lead := self._unit_lead(k)) is not None and lead.kind == "annuli"}
        fixed |= {tm[k] for k in fixed if k in tm}          # a fixed isotopologue fixes its parent's area too
        free = [k for k in keys if k not in fixed]
        y = data[m].copy()
        for k in keys:
            if k in fixed:
                y -= uf[k][m] * 10.0 ** (2.0 * logR[k])
        A = np.column_stack([uf[k][m] / sigma[m] for k in free]) if free else np.zeros((m.sum(), 0))
        out = dict(logR)
        if free:
            coef, rnorm = nnls(A, y / sigma[m])
            for k, cf in zip(free, coef):
                out[k] = 0.5 * np.log10(max(cf, 1e-12))
            chi2 = rnorm**2
        else:
            chi2 = float(np.sum((y / sigma[m]) ** 2))
        for ku, kp in tm.items():
            if kp in out:
                out[ku] = out[kp]
        return out, chi2

    def tau_flags(self, params=None) -> dict[str, float]:
        _, tmax, _ = self.unit_fluxes(params)
        return tmax


def linelist_key(molecule: str, release: str, path: str | None = None) -> str:
    """Dictionary key of a line list: 'H2O:hitemp', or 'H2O@/path/file.par' for a local file."""
    return f"{molecule}@{path}" if path else f"{molecule}:{release}"


def build_model(components: list[Component], wave_pix, distance_pc, windows=None, linelists=None,
                releases: dict | None = None, oversample=6, prune=True, **kw) -> SlabModel:
    """Convenience constructor: loads the line lists each component needs (from the cache),
    restricts them to the windows and prunes weak lines.

    Each component may use its own release (e.g. a hot H2O component on HITEMP and a cold one on
    HITRAN); `releases` gives the per-molecule default for components that do not set one."""
    from .linedata import load_linelist
    linelists = dict(linelists or {})
    releases = dict(releases or {})
    if windows is None:
        windows = [(float(np.min(wave_pix)), float(np.max(wave_pix)))]
    wlo = min(w[0] for w in windows) - 0.1
    whi = max(w[1] for w in windows) + 0.1
    for c in components:
        rel = c.linelist_release or releases.get(c.molecule) or "hitran"
        key = linelist_key(c.molecule, rel, c.linelist_path)
        if key in linelists or (c.molecule in linelists and not c.linelist_release):
            continue
        ll = load_linelist(c.molecule, release=rel, path=c.linelist_path, fetch=False)
        ll = ll.select(wlo, whi, eup_max=c.eup_max)
        if prune and len(ll) > 2000:
            ll = ll.strength_cut(T_hi=1500.0, T_lo=100.0, rel=1e-7)
        linelists[key] = ll
    return SlabModel(components, linelists, wave_pix, distance_pc, windows, oversample=oversample, releases=releases, **kw)
