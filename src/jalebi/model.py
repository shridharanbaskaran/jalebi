"""Forward model: LTE slab components on a fine ln(lambda) grid, convolved to MIRI pixels.

For one component c with column N (cm^-2), temperature T, line width FWHM dv and radial
velocity v:

    tau_c(x)  = N_c * sum_l kappa_l(T_c) phi(x - x_l - v/c)         (Phi is sparse)
    I_c(x)    = B_nu(T_c) (1 - exp(-tau_c))
    F_c(pix)  = Omega_c * K I_c                                      (Omega = pi R^2 / d^2)

Components in the same *opacity group* sum their tau before the exponential (shared T and
Omega).  Everything that does not depend on the parameters (Phi, K, B_nu grid) is built once.

Absorption components (kind="absorption") are foreground screens with a covering fraction f_c:

    Tr_a(x)   = 1 - f_c (1 - exp(-tau_a(x - v/c)))
    F(pix)    = K [ F_cont(x) * (Tr_a - 1) ]           (continuum-subtracted contribution)

i.e. the observed spectrum is F_c * (1 - f_c (1 - e^-tau)) (Li, Boogert & Tielens 2024; the
`spec_abs` of the group's slabby.py).  Several absorbers multiply their transmissions; with
covers="all" an absorber also attenuates the emission components behind it.

Vibrational temperature (Component.Tvib)
----------------------------------------
LTE slabs fitted to the pure-rotational water lines (12-27 um) over-predict the ro-vibrational
nu2 band at 5-8 um by factors of 3-6 (Banzatti et al. 2025, AJ 169, 165, Fig. 7; Pontoppidan et
al. 2024, ApJ 963, 158, Sect. 4.3): the vibrational levels have critical densities ~1e13 cm^-3
and are sub-thermally excited.  A component with `Tvib` set uses a two-temperature population,

    n_i / N = g_i exp(-E_vib,i / T_vib - E_rot,i / T_rot) / Z(T_rot, T_vib),

Boltzmann at T_rot inside every vibrational state and at T_vib between states (the usual
"T_vib != T_rot" approximation used for CO and CO2 fundamentals).  Per line the opacity uses
(x_l - x_u) and the source function S_l = (2 h nu^3 / c^2) / (x_l / x_u - 1); overlapping lines
on the fine grid are combined with the opacity-weighted source function S(x) = sum_l tau_l S_l /
sum_l tau_l, so that I = S (1 - e^-tau).  Pure-rotational lines (same vibrational state) are
unchanged, v=1-0 lines are weakened through S, and hot-band v=1-1 lines through their population;
Tvib = T_rot gives back LTE exactly.
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

PARAM_NAMES = ("logN", "T", "logR", "rv", "fwhm", "fc", "Tvib")
PARAM_LABELS = {"logN": "log N [cm⁻²]", "T": "T [K]", "logR": "log R [au]", "rv": "v [km/s]",
                "fwhm": "Δv [km/s]", "logNA": "log(N·A) [cm⁻² au²]", "ratio": "ratio", "q": "q", "p": "p",
                "fc": "f_c (covering)", "Tvib": "T_vib [K]"}
KINDS = ("slab", "annuli", "absorption")


def thermal_fwhm_kms(T: float, mass_amu: float) -> float:
    """FWHM of the thermal (Doppler) line profile, 2 sqrt(2 ln 2) sqrt(kT / m), in km/s."""
    m = mass_amu * 1.66053906660e-27
    return float(2.0 * np.sqrt(2.0 * np.log(2.0)) * np.sqrt(KB * T / m) / 1e3)


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
        # 2 h c / lambda^3 per line: the Planck prefactor of the line's own source function
        self.planck_c1 = 2.0 * H * C / (lines.wave[self.idx] * 1e-6) ** 3
        self._vib = None            # (E_vib_u, E_vib_l, E_rot_u, E_rot_l, state energies) lazily built

    # ---- vibrational structure ---------------------------------------------------------
    def _vib_arrays(self):
        if self._vib is None:
            vu_all, vl_all = self.lines.vib_arrays()
            vu, vl = vu_all[self.idx], vl_all[self.idx]
            known = np.isfinite(vu) & np.isfinite(vl)
            # unassigned labels: keep the line in LTE (E_vib = 0 for both levels -> only T_rot enters)
            vu = np.where(known, vu, 0.0); vl = np.where(known, vl, 0.0)
            states = np.array(sorted(set(self.lines.vib_energies().values())) or [0.0])
            self._vib = (vu, vl, self.eu - vu, self.el - vl, states)
        return self._vib

    def zvib(self, T: float) -> float:
        """Vibrational partition sum over the states present in the line list (ground state = 1)."""
        states = self._vib_arrays()[4]
        return float(np.sum(np.exp(-states / T)))

    def populations(self, T: float, Tvib: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """x_u, x_l: Boltzmann factors per unit statistical weight of the upper and lower level,
        normalised by the partition function.  Tvib=None (or == T) is LTE."""
        if Tvib is None or Tvib == T:
            Z = self.partition(T)
            return np.exp(-self.eu / T) / Z, np.exp(-self.el / T) / Z
        vu, vl, ru, rl, _ = self._vib_arrays()
        Z = self.partition(T) * self.zvib(Tvib) / self.zvib(T)
        return np.exp(-vu / Tvib - ru / T) / Z, np.exp(-vl / Tvib - rl / T) / Z

    def kappa(self, T: float, Tvib: float | None = None) -> np.ndarray:
        xu, xl = self.populations(T, Tvib)
        k = self.a * self.gu * self.lam3_8pi * (xl - xu)
        if Tvib is not None and Tvib != T:
            # cross-band lines whose upper level lies in a *lower* vibrational state can invert for
            # T_vib < T_rot (weak masers); a slab model has no use for them: drop them
            k = np.where(xl > xu * (1.0 + 1e-6), k, 0.0)
        return k

    def source(self, T: float, Tvib: float | None = None) -> np.ndarray:
        """Source function of every line, 2 h nu^3 / c^2 / (x_l / x_u - 1)  [W m^-2 Hz^-1 sr^-1].
        Equals B_nu(T) at the line frequency in LTE.  Inverted lines (dropped from kappa) get 0."""
        xu, xl = self.populations(T, Tvib)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            ratio = xl / xu
            S = self.planck_c1 / (ratio - 1.0)
        return np.where(np.isfinite(S) & (xl > xu * (1.0 + 1e-6)), S, 0.0)

    def tau(self, N_cm2: float, T: float, Tvib: float | None = None) -> np.ndarray:
        """Optical depth on the fine grid for column N (cm^-2)."""
        if self.n_lines == 0:
            return np.zeros(self.grid.n)
        return (N_cm2 * 1e4) * (self.phi @ self.kappa(T, Tvib))

    def tau_emissivity(self, N_cm2: float, T: float, Tvib: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(tau(x), j(x)) on the fine grid with j = sum_l tau_l(x) S_l, so that the opacity-weighted
        source function is j / tau and I = (j / tau) (1 - e^-tau)."""
        if self.n_lines == 0:
            z = np.zeros(self.grid.n)
            return z, z.copy()
        k = self.kappa(T, Tvib)
        n = N_cm2 * 1e4
        return n * (self.phi @ k), n * (self.phi @ (k * self.source(T, Tvib)))

    def tau_peaks(self, N_cm2: float, T: float, Tvib: float | None = None) -> np.ndarray:
        """Peak optical depth of every line individually (for the tau flag and line pruning)."""
        phi0 = 1.0 / (self.sigma_x * np.sqrt(2 * np.pi)) / C
        return (N_cm2 * 1e4) * self.kappa(T, Tvib) * phi0


# --------------------------------------------------------------------------
# Components
# --------------------------------------------------------------------------

@dataclass
class Component:
    """One slab (or annular) emission component.

    kind: "slab" (N, T, R), "annuli" (power-law T(r), N(r) between R_in and R_out) or
          "absorption" (a foreground screen: N, T, v, covering fraction fc; no emitting area)
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
    # absorption screens only
    fc: float = 1.0                # covering fraction of the continuum source (0-1)
    covers: str = "continuum"      # "continuum": screen in front of the continuum only;
                                   # "all": it also absorbs every emission component (two-slab geometry)
    fwhm_thermal: bool = False     # add the thermal width at T in quadrature to fwhm (any kind)
    # non-LTE vibrational excitation: T_vib (K) of the two-temperature population; None = LTE (T_vib = T)
    Tvib: float | None = None
    # wavelength ranges (um) outside which this component contributes nothing, e.g. [[4.9, 9.0]] for a
    # ro-vibrational water component fitted on the nu2 band only while other components fit 12-27 um
    # (the separate-region practice of Gasman+2023, Temmink+2024).  None = everywhere.
    windows: list | None = None

    @property
    def is_absorber(self) -> bool:
        return self.kind == "absorption"

    def params(self) -> dict:
        d = {"logN": self.logN, "T": self.T, "logR": self.logR, "rv": self.rv, "fwhm": self.fwhm}
        if self.kind == "annuli":
            d.update({"q": self.q, "p": self.p, "logRin": self.logRin})
        if self.kind == "absorption":
            d["fc"] = self.fc
        if self.Tvib is not None:
            d["Tvib"] = self.Tvib
        if self.tie_to:
            d["ratio"] = self.ratio if self.ratio is not None else get_molecule(self.molecule).default_ratio or 70.0
        return d

    def set_params(self, d: dict):
        for k, v in d.items():
            if hasattr(self, k):
                setattr(self, k, None if v is None else float(v))


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
    continuum  : continuum flux (Jy) on `wave_pix`; needed by absorption components (the screen
                 multiplies it).  None = no continuum (absorbers then contribute nothing).
    """

    def __init__(self, components: list[Component], linelists: dict[str, LineList], wave_pix: np.ndarray,
                 distance_pc: float = 140.0, windows: list[tuple[float, float]] | None = None,
                 oversample: int = 6, R_model: str = "argyriou2023", R_scale: float = 1.0,
                 R_constant: float | None = None, pixel_edges_x: tuple | None = None, tau_min_line: float = 1e-4,
                 logN_max: float = 21.0, releases: dict[str, str] | None = None,
                 continuum: np.ndarray | None = None):
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
        self._winmask: dict[tuple, np.ndarray] = {}
        # Optional memo of per-unit fluxes keyed on the unit's parameters (used by the interactive
        # app so that moving one slider only re-evaluates that component).  Off by default: the
        # fitter changes every parameter at once, where a cache would only add hashing overhead.
        self.unit_cache: dict | None = None
        # Optional precomputed tables (jalebi.emulator.EmulatorSet, fit.model_backend: emulator): units it
        # covers are interpolated instead of computed; every other unit stays exact.
        self.emulator = None
        self._omega_unit = (AU / (distance_pc * PC)) ** 2 * np.pi   # Omega for R = 1 au
        lam = self.grid.wave * 1e-6
        self._planck_c1 = 2.0 * H * C / lam**3
        self._planck_c2 = H * C / (lam * KB)
        self.set_continuum(continuum)

    def set_continuum(self, continuum: np.ndarray | None):
        """Continuum (Jy) on the pixels -> interpolated onto the fine grid (used by absorbers)."""
        self.continuum_pix = None
        self._cont_fine = None
        if continuum is None:
            return
        cont = np.asarray(continuum, float)
        if cont.shape != self.wave_pix.shape:
            raise ValueError("continuum must have the shape of wave_pix")
        ok = np.isfinite(cont) & np.isfinite(self.wave_pix)
        if ok.sum() < 2:
            return
        order = np.argsort(self.wave_pix[ok])
        w, f = self.wave_pix[ok][order], cont[ok][order]
        # sub-band overlaps give repeated wavelengths: keep them, np.interp copes with ties
        self.continuum_pix = cont
        self._cont_fine = np.interp(self.grid.wave, w, f)

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
            if len(self._bases) > 64:            # thermal widths change with T: keep the cache bounded
                self._bases.clear()
            self._bases[key] = OpacityBasis(ll, self.grid, fwhm_kms=fwhm)
        return self._bases[key]

    def line_fwhm(self, c: Component, p: dict) -> float:
        """Line FWHM used for a component: p['fwhm'], plus the thermal width at T when fwhm_thermal."""
        fw = float(p["fwhm"])
        if c.fwhm_thermal:
            fw = float(np.hypot(fw, thermal_fwhm_kms(p["T"], get_molecule(c.molecule).mass_amu)))
        return fw

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
                if "Tvib" in par:
                    P[c.name]["Tvib"] = par["Tvib"]
        return P

    # ---- intensities per emitting unit ----------------------------------------
    @staticmethod
    def _tvib(p: dict):
        v = p.get("Tvib")
        return None if v is None or not np.isfinite(v) else float(v)

    def _slab_tau(self, c: Component, p: dict) -> np.ndarray:
        return self.basis(c, self.line_fwhm(c, p)).tau(10.0 ** p["logN"], p["T"], self._tvib(p))

    def _slab_tau_emis(self, c: Component, p: dict) -> tuple[np.ndarray, np.ndarray]:
        return self.basis(c, self.line_fwhm(c, p)).tau_emissivity(10.0 ** p["logN"], p["T"], self._tvib(p))

    def window_mask(self, c: Component) -> np.ndarray | None:
        """Pixel mask of a component's own `windows` (None when it emits everywhere)."""
        if not c.windows:
            return None
        key = (c.name, tuple(tuple(map(float, w)) for w in c.windows))
        m = self._winmask.get(key)
        if m is None:
            m = np.zeros(len(self.wave_pix), bool)
            for a, b in c.windows:
                m |= (self.wave_pix >= float(a)) & (self.wave_pix <= float(b))
            self._winmask[key] = m
        return m

    # ---- absorption screens ------------------------------------------------------------
    def absorbers(self) -> list[Component]:
        return [c for c in self.components if c.enabled and c.kind == "absorption"]

    def transmission(self, c: Component, p: dict) -> tuple[np.ndarray, float]:
        """Tr(x) = 1 - fc (1 - exp(-tau)) of one absorbing screen on the fine grid (shifted by its
        rv) and its peak optical depth."""
        tau = self._slab_tau(c, p)
        tmax = float(tau.max()) if tau.size else 0.0
        tau = self._shift(tau, p["rv"])
        fc = float(np.clip(p.get("fc", 1.0), 0.0, 1.0))
        return 1.0 + fc * np.expm1(-tau), tmax

    def total_transmission(self, params: dict[str, dict] | None = None, covers: str | None = None) -> np.ndarray:
        """Product of the transmissions of every absorber (optionally only those with a given
        `covers`), on the fine grid."""
        P = self.resolve_params(params)
        tr = np.ones(self.grid.n)
        for c in self.absorbers():
            if covers is None or c.covers == covers:
                tr *= self.transmission(c, P[c.name])[0]
        return tr

    def _screens(self, P: dict[str, dict], units: dict[str, list[Component]]) -> list[tuple[str, np.ndarray, float, float, str]]:
        """Per absorbing unit: (key, tau on the fine grid (shifted), fc, tau_max, covers).  Members of
        a group add their opacity and share T, fwhm, rv and fc with the leader."""
        out = []
        for key, members in units.items():
            lead = members[0]
            if lead.kind != "absorption":
                continue
            p0 = P[lead.name]
            tau = np.zeros(self.grid.n)
            for c in members:
                p = P[c.name]
                tau += self._slab_tau(c, {**p, "T": p0["T"], "fwhm": p0["fwhm"]})
            tmax = float(tau.max()) if tau.size else 0.0
            tau = self._shift(tau, p0["rv"])
            fc = float(np.clip(p0.get("fc", 1.0), 0.0, 1.0))
            out.append((key, tau, fc, tmax, lead.covers))
        return out

    def _units(self) -> dict[str, list[Component]]:
        units: dict[str, list[Component]] = {}
        for c in self.components:
            if c.enabled:
                units.setdefault(c.group if c.group else c.name, []).append(c)
        return units

    def screen_columns(self, params: dict[str, dict] | None = None) -> dict[str, np.ndarray]:
        """Per absorbing unit, the pixel flux K[F_c (e^-tau - 1)] for fc = 1, each screen alone (no
        product with the others): the model is linear in fc for one screen, so these are the columns of a
        linear solve for the covering fractions (used by the detection)."""
        P = self.resolve_params(params)
        cont = self._cont_fine
        out = {}
        for key, tau, _, _, _ in self._screens(P, self._units()):
            out[key] = np.zeros(len(self.wave_pix)) if cont is None else (self.K @ (cont * np.expm1(-tau)))
        return out

    def unit_fluxes(self, params: dict[str, dict] | None = None) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
        """Per emitting unit: pixel fluxes (Jy) for Omega of a 1-au-radius disk, for each
        independent unit (component or opacity group).  Returns (unit_flux, unit_tau_max, unit_logR).

        Absorption screens are units too: their flux is the (negative) continuum-subtracted
        contribution K[F_cont (Tr - 1)] in Jy, already complete (their logR is returned as 0 so that
        the area scaling 10^(2 logR) is 1).  Screens with covers="all" also multiply the emission
        units.  When several screens are present the k-th one acts on what the first k-1 let through,
        so that the unit fluxes add up to F_cont (prod Tr - 1)."""
        P = self.resolve_params(params)
        units = self._units()
        flux, taumax, logR = {}, {}, {}
        screens = [(key, 1.0 + fc * np.expm1(-tau), tmax, cov) for key, tau, fc, tmax, cov in self._screens(P, units)]
        tr_all = None
        all_key = ()
        if any(cov == "all" for _, _, _, cov in screens):
            tr_all = np.ones(self.grid.n)
            for key, tr, _, cov in screens:
                if cov == "all":
                    tr_all = tr_all * tr
            all_key = tuple((key, tuple(sorted(P[units[key][0].name].items()))) for key, _, _, cov in screens if cov == "all")
        for key, members in units.items():
            lead = members[0]
            if lead.kind == "absorption":
                continue
            p0 = P[lead.name]
            if self.emulator is not None and len(members) == 1:
                hit = self.emulator.unit_flux(key, p0)
                if hit is not None:
                    flux[key], taumax[key] = hit
                    logR[key] = p0["logR"]
                    continue
            ck = None
            if self.unit_cache is not None:
                # the 1-au flux does not depend on logR for slabs (only the annuli R_out does)
                skip = () if lead.kind == "annuli" else ("logR",)
                ck = (key, lead.kind, lead.n_annuli, all_key, None if not lead.windows else tuple(map(tuple, lead.windows)),
                      tuple((c.name, self.linelist_key(c), c.fwhm_thermal, tuple(sorted((k, v) for k, v in P[c.name].items() if k not in skip)))
                            for c in members))
                hit = self.unit_cache.get(ck)
                if hit is not None:
                    flux[key], taumax[key] = hit
                    logR[key] = p0["logR"]
                    continue
            if lead.kind == "annuli" and len(members) == 1:
                I = self._annuli_intensity(lead, p0)
                tmax = np.nan
            elif any(self._tvib(P[c.name]) is not None for c in members):
                # two-temperature (T_rot, T_vib) populations: opacity-weighted source function
                tau = np.zeros(self.grid.n); emis = np.zeros(self.grid.n)
                for c in members:
                    p = P[c.name]
                    t, j = self._slab_tau_emis(c, {**p, "T": p0["T"], "fwhm": p0["fwhm"], "Tvib": self._tvib(p)})
                    tau += t; emis += j
                tmax = float(tau.max()) if tau.size else 0.0
                # (1 - e^-tau) / tau -> 1 for tau -> 0: I = j * (1 - e^-tau) / tau
                with np.errstate(divide="ignore", invalid="ignore"):
                    f = np.where(tau > 1e-8, -np.expm1(-tau) / np.where(tau > 1e-8, tau, 1.0), 1.0)
                I = emis * f
            else:
                tau = np.zeros(self.grid.n)
                for c in members:
                    p = P[c.name]
                    t = self._slab_tau(c, {**p, "T": p0["T"], "fwhm": p0["fwhm"]})
                    tau += t
                tmax = float(tau.max()) if tau.size else 0.0
                I = self.planck(p0["T"]) * (-np.expm1(-tau))
            I = self._shift(I, p0["rv"])
            if tr_all is not None:
                I = I * tr_all
            flux[key] = (self.K @ I) * self._omega_unit / JY     # Jy for R = 1 au
            wm = self.window_mask(lead)
            if wm is not None:
                flux[key] = flux[key] * wm
            taumax[key] = tmax
            logR[key] = p0["logR"]
            if ck is not None:
                if len(self.unit_cache) > 256:
                    self.unit_cache.clear()
                self.unit_cache[ck] = (flux[key], tmax)
        if screens:
            cont = self._cont_fine
            through = np.ones(self.grid.n)
            for key, tr, tmax, _ in screens:
                if cont is None:
                    flux[key] = np.zeros(len(self.wave_pix))
                else:
                    flux[key] = (self.K @ (cont * through * (tr - 1.0)))
                    through = through * tr
                taumax[key] = tmax
                logR[key] = 0.0
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
        tvib0 = self._tvib(p)                     # T_vib / T_rot is kept constant along the annuli
        I = np.zeros(self.grid.n)
        for r, wi in zip(rc, w):
            T = max(p["T"] * (r / r_in) ** (-p["q"]), 20.0)
            N = 10.0 ** p["logN"] * (r / r_in) ** (-p["p"])
            if tvib0 is None:
                I += wi * self.planck(T) * (-np.expm1(-basis.tau(N, T)))
            else:
                tau, j = basis.tau_emissivity(N, T, max(tvib0 * T / p["T"], 20.0))
                with np.errstate(divide="ignore", invalid="ignore"):
                    f = np.where(tau > 1e-8, -np.expm1(-tau) / np.where(tau > 1e-8, tau, 1.0), 1.0)
                I += wi * j * f
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
        radius of the power law and the flux is not linear in R^2; absorption screens have no area and
        keep their covering fraction.  Returns ({unit: logR}, chi2)."""
        logR, _, chi2 = self.solve_linear(data, sigma, params, mask, fixed, solve_fc=False)
        return logR, chi2

    def solve_linear(self, data: np.ndarray, sigma: np.ndarray, params: dict[str, dict] | None = None,
                     mask: np.ndarray | None = None, fixed: set[str] | None = None, solve_fc: bool = False
                     ) -> tuple[dict[str, float], dict[str, float], float]:
        """NNLS over every linear parameter: the areas of the slab units and, with `solve_fc`, the
        covering fractions of the absorption screens (each screen's column is its flux alone at fc = 1,
        so overlapping screens are treated independently: exact for one screen, an approximation for
        several), with 0 <= fc <= 1.
        Returns ({unit: logR}, {screen unit: fc}, chi2)."""
        P = self.resolve_params(params)
        uf_all, _, logR = self.unit_fluxes(P)
        tm = self.tied_units()
        uf = self.fold_tied(uf_all)
        keys = list(uf)
        m = np.ones(len(data), bool) if mask is None else mask.copy()
        m &= np.isfinite(data) & np.isfinite(sigma) & (sigma > 0)
        fixed = set(fixed or set())
        fixed |= {k for k in keys if (lead := self._unit_lead(k)) is not None and lead.kind == "annuli"}
        screens = {k for k in keys if (lead := self._unit_lead(k)) is not None and lead.kind == "absorption"}
        fc_out = {k: float(np.clip(P[self._unit_lead(k).name].get("fc", 1.0), 0.0, 1.0)) for k in screens}
        if not solve_fc:
            fixed |= screens
        fixed |= {tm[k] for k in fixed if k in tm}          # a fixed isotopologue fixes its parent's area too
        free = [k for k in keys if k not in fixed]
        cols = dict(uf)
        if solve_fc and (screens - fixed):
            cols.update(self.screen_columns(P))            # fc = 1, independent columns
        y = data[m].copy()
        for k in keys:
            if k in fixed:
                y -= uf[k][m] * 10.0 ** (2.0 * logR[k])
        A = np.column_stack([cols[k][m] / sigma[m] for k in free]) if free else np.zeros((m.sum(), 0))
        out = dict(logR)
        if free:
            if any(k in screens for k in free):
                # covering fractions are bounded by 1: bounded-variable least squares
                from scipy.optimize import lsq_linear
                hi = np.array([1.0 if k in screens else np.inf for k in free])
                r = lsq_linear(A, y / sigma[m], bounds=(np.zeros(len(free)), hi), method="bvls")
                coef, rnorm = r.x, float(np.sqrt(2.0 * r.cost))
            else:
                coef, rnorm = nnls(A, y / sigma[m])
            for k, cf in zip(free, coef):
                if k in screens:
                    fc_out[k] = float(cf)
                else:
                    out[k] = 0.5 * np.log10(max(cf, 1e-12))
            chi2 = rnorm**2
        else:
            chi2 = float(np.sum((y / sigma[m]) ** 2))
        for ku, kp in tm.items():
            if kp in out:
                out[ku] = out[kp]
        return out, fc_out, chi2

    def exact(self):
        """Context manager: evaluate with the exact model even when an emulator is attached."""
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            em = self.emulator
            self.emulator = None
            try:
                yield self
            finally:
                self.emulator = em
        return _cm()

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
    HITRAN); `releases` gives the per-molecule default for components that do not set one.
    Pass `continuum=` (Jy on wave_pix) when the model has absorption components."""
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
