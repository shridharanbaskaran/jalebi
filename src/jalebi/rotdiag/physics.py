"""Forward model of a rotation (population) diagram: line fluxes from level populations.

Populations (per unit column or per molecule), for a level u of spin species s:

  one temperature      N_u/g_u = N · φ_s · exp(−E_u/kT) / Q_s(T)
  two temperatures     the sum of two such terms (warm T₁ < hot T₂), sharing A_V and the OPR
  power law            N_u/g_u = N ∫ p(T) φ_s exp(−E_u/kT)/Q_s(T) dT,  p(T) ∝ T^−b on [T_min, T_max]
                       (Neufeld & Yuan 2008; a continuous distribution of shock temperatures)

  OPR "thermal":  φ_s/Q_s = 1/Q  (ortho and para in LTE at T; OPR = Q_o/Q_p)
  OPR "species":  φ_o = OPR/(1+OPR), φ_p = 1/(1+OPR), each spin species in LTE at T within itself:
                  N_o/N_p = OPR exactly (no approximation at low T).
  OPR "offset":   N_u/g_u = N exp(−E_u/kT)/Q(T) × (OPR/3) for ortho levels — the correction z(J) = ln(OPR/3)
                  used by e.g. Francis et al. (2025, JOYS); equal to "species" when T ≳ 300 K.

Line flux of a member line i (W m⁻²), optically thin, reddened by A_V with k(λ) = A_λ/A_V:

  column mode (N in cm⁻², emitting solid angle Ω):  F_i = h ν_i A_i g_i (N_u/g_u)·10⁴ · Ω/4π · 10^(−0.4 A_V k_i)
  number mode (N = number of molecules, distance d): F_i = h ν_i A_i g_i (N_u/g_u) / (4π d²) · 10^(−0.4 A_V k_i)

Optical depth (for CO, H2O, OH ... with the opacity correction on) for a slab with a Gaussian velocity
distribution of FWHM Δv (σ_v = Δv/2.3548):

  τ₀,i = A_i c³ / (8π ν_i³) · g_i (N_u/g_u) · (exp(hν_i/kT) − 1) / (σ_v √(2π))
  F_i → F_i · f(τ₀,i),   f(τ₀) = ∫(1 − exp(−τ₀ e^(−x²))) dx / (√π τ₀)     (the curve of growth; f → 1 when thin)

so the rotation diagram is fitted in flux space; the diagram shows the points 4πF/(hνΩ Σ g A) and the
model curves.  The same slab physics as jalebi's spectral fit (jalebi.model) for a single line.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..constants import AU, C, H, KB, PC
from .extinction import ExtinctionCurve
from .species import SpinPartition

ARCSEC = np.pi / 180.0 / 3600.0
M_EARTH = 5.9722e24
M_JUP = 1.89813e27
M_SUN = 1.98847e30
L_SUN = 3.828e26
AMU = 1.66053906660e-27
MODELS = ("single", "two", "powerlaw")
OPR_MODES = ("thermal", "species", "offset")
AREA_MODES = ("number", "radius", "aperture", "intensity")

# ------------------------------------------------------------------------------------------------
# curve of growth
# ------------------------------------------------------------------------------------------------
_LT = np.linspace(-4.0, 9.0, 1301)


def _cog_table():
    x = np.linspace(-12, 12, 6001)[None, :]
    tau = 10.0 ** _LT[:, None]
    trap = getattr(np, "trapezoid", None) or np.trapz          # numpy < 2 has only trapz
    integ = trap(-np.expm1(-tau * np.exp(-x * x)), x, axis=1)
    return np.log(integ / (np.sqrt(np.pi) * 10.0 ** _LT))


_LOGF = _cog_table()


def cog_factor(tau0) -> np.ndarray:
    """f(τ₀) = F / F_thin for a Gaussian line profile (1 for τ₀ → 0, ~2√(ln τ₀)/(√π τ₀) for τ₀ → ∞)."""
    t = np.asarray(tau0, float)
    lt = np.log10(np.maximum(t, 1e-30))
    out = np.exp(np.interp(lt, _LT, _LOGF))
    return np.where(lt < _LT[0], 1.0 - t / (2.0 * np.sqrt(2.0)), out)


# ------------------------------------------------------------------------------------------------
# the model
# ------------------------------------------------------------------------------------------------

@dataclass
class ParamSpec:
    name: str
    value: float
    lo: float
    hi: float
    free: bool = True
    label: str = ""
    unit: str = ""


@dataclass
class Geometry:
    """How the populations are normalised."""
    mode: str = "number"             # number | radius | aperture | intensity
    distance_pc: float = 140.0
    R_au: float = 1.0                # emitting radius (mode radius; and for the opacity in mode number)
    aperture_arcsec: float = 0.5     # radius of the extraction aperture (mode aperture)
    omega_sr: float | None = None    # explicit solid angle (overrides the aperture radius)

    def omega(self, R_au=None):
        """Solid angle [sr] of the emission (column modes)."""
        if self.mode == "intensity":
            return 1.0
        if self.mode == "aperture":
            return self.omega_sr if self.omega_sr else np.pi * (self.aperture_arcsec * ARCSEC) ** 2
        R = self.R_au if R_au is None else R_au
        return np.pi * (np.asarray(R) * AU) ** 2 / (self.distance_pc * PC) ** 2

    @property
    def column(self) -> bool:
        return self.mode != "number"


@dataclass
class RotModel:
    """Predicts the flux of every feature (sum of its member lines) for a parameter vector."""
    kind: str                         # single | two | powerlaw
    part: SpinPartition
    member_wave: np.ndarray           # µm
    member_a: np.ndarray
    member_g: np.ndarray
    member_eu: np.ndarray             # K
    member_spin: np.ndarray           # 'o' | 'p' | ''
    member_feature: np.ndarray        # index into the features
    n_features: int
    curve: ExtinctionCurve
    geometry: Geometry = field(default_factory=Geometry)
    opr_mode: str = "thermal"
    opacity: bool = False
    params: dict[str, ParamSpec] = field(default_factory=dict)
    opr_high_t: float = 3.0
    n_tgrid: int = 48

    def __post_init__(self):
        self.nu = C / (self.member_wave * 1e-6)
        self.k_ext = self.curve(self.member_wave)
        self.ortho = self.member_spin == "o"
        self.para = self.member_spin == "p"
        self.hnuAg = H * self.nu * self.member_a * self.member_g
        self.tau_coef = self.member_a * C ** 3 / (8.0 * np.pi * self.nu ** 3) * self.member_g / np.sqrt(2.0 * np.pi)
        S = np.zeros((self.n_features, len(self.member_wave)))
        S[self.member_feature, np.arange(len(self.member_wave))] = 1.0
        self.S = S
        if not self.params:
            self.params = default_params(self.kind, self)

    # ---- parameter vector bookkeeping ----------------------------------------------------------
    @property
    def names(self) -> list[str]:
        return list(self.params)

    @property
    def free(self) -> list[str]:
        return [k for k, p in self.params.items() if p.free]

    def full(self, theta_free: np.ndarray) -> dict[str, np.ndarray]:
        """dict of parameter arrays (walkers along axis 0) from the free-parameter vector(s)."""
        th = np.atleast_2d(theta_free)
        out, j = {}, 0
        for k, p in self.params.items():
            if p.free:
                out[k] = th[:, j]; j += 1
            else:
                out[k] = np.full(th.shape[0], p.value)
        return out

    def bounds(self):
        f = self.free
        return np.array([self.params[k].lo for k in f]), np.array([self.params[k].hi for k in f])

    def start(self):
        return np.array([self.params[k].value for k in self.free])

    # ---- physics ----------------------------------------------------------------------------------
    def _spin_factor(self, T, opr):
        """(walkers, members): φ_s / Q_s(T) for the chosen OPR convention (T, opr: (walkers,) or (walkers, nT))."""
        T = np.asarray(T, float)
        if self.opr_mode == "thermal" or not self.part.has_spin or not (self.ortho.any() or self.para.any()):
            q = self.part.Q(T)
            return 1.0 / q[..., None]
        if self.opr_mode == "offset":
            q = self.part.Q(T)[..., None]
            f = np.where(self.ortho, (np.asarray(opr)[..., None] / self.opr_high_t), 1.0)
            return f / q
        opr = np.asarray(opr, float)[..., None]
        qo = self.part.Qs(T, "o")[..., None]
        qp = self.part.Qs(T, "p")[..., None]
        fo = opr / (1.0 + opr) / qo
        fp = 1.0 / (1.0 + opr) / qp
        return np.where(self.ortho, fo, np.where(self.para, fp, 1.0 / self.part.Q(T)[..., None]))

    def populations(self, P: dict) -> list[tuple[np.ndarray, np.ndarray]]:
        """[(N_u/g_u per member (walkers, members), T of the component (walkers,) or None)] per component."""
        E = self.member_eu[None, :]
        opr = P.get("OPR", np.full(len(next(iter(P.values()))), self.opr_high_t))
        if self.kind == "single":
            T = P["T"]
            return [(10.0 ** P["logN"][:, None] * self._spin_factor(T, opr) * np.exp(-E / T[:, None]), T)]
        if self.kind == "two":
            out = []
            for i in ("1", "2"):
                T = P["T" + i]
                out.append((10.0 ** P["logN" + i][:, None] * self._spin_factor(T, opr) * np.exp(-E / T[:, None]), T))
            return out
        if self.kind == "powerlaw":
            tmin, tmax, b = P["Tmin"], P["Tmax"], P["b"]
            u = np.linspace(0.0, 1.0, self.n_tgrid)[None, :]
            lt = np.log(tmin)[:, None] + u * (np.log(np.maximum(tmax, tmin * 1.0001)) - np.log(tmin))[:, None]
            Tg = np.exp(lt)                                       # (walkers, nT)
            w = Tg ** (1.0 - b[:, None])                          # p(T) dT = T^-b T dlnT
            w[:, [0, -1]] *= 0.5
            w /= w.sum(axis=1, keepdims=True)
            sf = self._spin_factor(Tg, opr[:, None]) if np.ndim(opr) else self._spin_factor(Tg, opr)
            # (walkers, nT, members)
            n = (w[:, :, None] * sf * np.exp(-E[None, :, :] / Tg[:, :, None])).sum(axis=1)
            return [(10.0 ** P["logN"][:, None] * n, None)]
        raise ValueError(f"unknown model '{self.kind}'")

    def member_flux(self, P: dict, with_tau: bool = False):
        """Predicted flux of every member line (walkers, members) [W m^-2 or W m^-2 sr^-1]; with_tau also τ₀."""
        geo = self.geometry
        ext = 10.0 ** (-0.4 * P["Av"][:, None] * self.k_ext[None, :]) if "Av" in P else 1.0
        R = 10.0 ** P["logR"] if "logR" in P else np.full(len(P[next(iter(P))]), geo.R_au)
        if geo.column:
            omega = geo.omega(R)[:, None] if geo.mode == "radius" else geo.omega()
            scale = self.hnuAg[None, :] * 1e4 * omega / (4.0 * np.pi)
            col_m2 = 1e4                                      # N_u/g_u [cm^-2] -> m^-2
        else:
            d = geo.distance_pc * PC
            scale = self.hnuAg[None, :] / (4.0 * np.pi * d * d)
            col_m2 = 1.0 / (np.pi * (R * AU) ** 2)[:, None]    # molecules -> column over the emitting area
        F = 0.0
        taumax = np.zeros((len(R), len(self.member_wave)))
        for n, T in self.populations(P):
            f = 1.0
            if self.opacity and T is not None:
                sig = P["fwhm"][:, None] * 1e3 / 2.354820045 if "fwhm" in P else 1e3
                tau = self.tau_coef[None, :] * n * col_m2 * np.expm1(H * self.nu[None, :] / (KB * T[:, None])) / sig
                f = cog_factor(tau)
                taumax = np.maximum(taumax, tau)
            F = F + scale * n * f
        F = F * ext
        return (F, taumax) if with_tau else F

    def predict(self, theta_free) -> np.ndarray:
        """Feature fluxes (walkers, features)."""
        return self.member_flux(self.full(theta_free)) @ self.S.T

    def tau(self, theta_free) -> np.ndarray:
        return self.member_flux(self.full(theta_free), with_tau=True)[1]

    # ---- diagram --------------------------------------------------------------------------------
    def curve_y(self, P1: dict, E: np.ndarray, spin: str = "") -> list[np.ndarray]:
        """ln(N_u/g_u) of the intrinsic model on an energy grid (one array per component + the total),
        for spin species `spin` ('o', 'p' or '')."""
        saved = (self.member_eu, self.member_spin, self.ortho, self.para)
        try:
            self.member_eu = np.asarray(E, float)
            self.member_spin = np.array([spin] * len(E))
            self.ortho = self.member_spin == "o"; self.para = self.member_spin == "p"
            comps = [n[0] for n, _ in self.populations({k: np.atleast_1d(v) for k, v in P1.items()})]
        finally:
            self.member_eu, self.member_spin, self.ortho, self.para = saved
        tot = np.sum(comps, axis=0)
        return [np.log(c) for c in comps] + [np.log(tot)]


def default_params(kind: str, m: RotModel | None = None, t_bounds=(50.0, 5000.0)) -> dict[str, ParamSpec]:
    lo_t, hi_t = t_bounds
    col = m is None or m.geometry.column
    nlo, nhi, nval = ((10.0, 26.0, 18.0) if col else (30.0, 56.0, 44.0))
    nlab = "log N [cm⁻²]" if col else "log N_mol [molecules]"
    P: dict[str, ParamSpec] = {}
    if kind == "single":
        P["logN"] = ParamSpec("logN", nval, nlo, nhi, True, nlab)
        P["T"] = ParamSpec("T", 600.0, lo_t, hi_t, True, "T [K]", "K")
    elif kind == "two":
        P["logN1"] = ParamSpec("logN1", nval, nlo, nhi, True, nlab.replace("N", "N₁"))
        P["T1"] = ParamSpec("T1", 400.0, lo_t, min(hi_t, 1500.0), True, "T₁ [K]", "K")
        P["logN2"] = ParamSpec("logN2", nval - 1.5, nlo, nhi, True, nlab.replace("N", "N₂"))
        P["T2"] = ParamSpec("T2", 1500.0, max(lo_t, 300.0), hi_t, True, "T₂ [K]", "K")
    elif kind == "powerlaw":
        P["logN"] = ParamSpec("logN", nval, nlo, nhi, True, nlab + " (T > T_min)")
        P["Tmin"] = ParamSpec("Tmin", 200.0, lo_t, 2000.0, True, "T_min [K]", "K")
        P["Tmax"] = ParamSpec("Tmax", 4000.0, 1000.0, max(hi_t, 5000.0), False, "T_max [K]", "K")
        P["b"] = ParamSpec("b", 4.0, 1.01, 8.0, True, "b (dN ∝ T^−b dT)")
    else:
        raise ValueError(f"model must be one of {MODELS}")
    P["Av"] = ParamSpec("Av", 0.0, 0.0, 100.0, False, "A_V [mag]", "mag")
    P["OPR"] = ParamSpec("OPR", 3.0, 0.1, 6.0, False, "OPR")
    P["logR"] = ParamSpec("logR", 0.0, -3.0, 3.0, False, "log R [au]", "au")
    P["fwhm"] = ParamSpec("fwhm", 10.0, 0.5, 200.0, False, "Δv [km/s]", "km/s")
    return P


def lnlike_fn(model: RotModel, F_obs: np.ndarray, sigma: np.ndarray, use: np.ndarray, t_order: bool = True):
    """Vectorised log-likelihood (+ flat priors inside the bounds) of the free parameters."""
    lo, hi = model.bounds()
    Fo, sg = F_obs[use], sigma[use]
    names = model.free
    i1 = names.index("T1") if "T1" in names else None
    i2 = names.index("T2") if "T2" in names else None
    fixed_T1 = model.params["T1"].value if "T1" in model.params and not model.params["T1"].free else None
    fixed_T2 = model.params["T2"].value if "T2" in model.params and not model.params["T2"].free else None
    itmin = names.index("Tmin") if "Tmin" in names else None
    itmax = names.index("Tmax") if "Tmax" in names else None

    def lnp(theta):
        th = np.atleast_2d(theta)
        ok = np.all((th >= lo) & (th <= hi), axis=1)
        if t_order and model.kind == "two":
            t1 = th[:, i1] if i1 is not None else fixed_T1
            t2 = th[:, i2] if i2 is not None else fixed_T2
            ok &= np.asarray(t1 < t2)
        if model.kind == "powerlaw":
            tmn = th[:, itmin] if itmin is not None else model.params["Tmin"].value
            tmx = th[:, itmax] if itmax is not None else model.params["Tmax"].value
            ok &= np.asarray(tmn < tmx)
        out = np.full(th.shape[0], -np.inf)
        if ok.any():
            pred = model.predict(th[ok])[:, use]
            out[ok] = -0.5 * np.sum(((Fo[None, :] - pred) / sg[None, :]) ** 2, axis=1)
        return out if np.ndim(theta) == 2 else out[0]
    return lnp
