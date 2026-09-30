"""Fit a rotation diagram: least squares (multi-start) and emcee MCMC, in flux space.

    χ² = Σ_k (F_obs,k − F_mod,k)² / (σ_k² + (f_sys F_obs,k)²)

F_mod,k is the summed flux of the member lines of feature k (physics.RotModel), so blends, upper limits
(non-detections enter with their measured flux and error — no censoring needed), extinction and optical depth
are handled exactly; the classic straight line in the (E_u, ln N_u/g_u) plane is its thin, unreddened limit.
f_sys (default 0.1) is the relative spectro-photometric uncertainty of MIRI-MRS between sub-bands.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, nnls

from ..constants import AU, C, H, PC
from ..molecules import get_molecule
from .extinction import LN10_04, get_curve
from .physics import (AMU, L_SUN, M_EARTH, M_JUP, MODELS, OPR_MODES, Geometry, RotModel,
                      default_params, lnlike_fn)
from .species import SpinPartition, load_species_linelist, partition, preset


@dataclass
class FitConfig:
    model: str = "single"               # single | two | powerlaw
    opr: str = "thermal"                # thermal | species | offset   (species/offset: OPR is a parameter)
    opr_value: float = 3.0
    opr_free: bool = True
    av: float = 0.0
    av_free: bool = False
    extinction: str = "G23"
    geometry: Geometry = field(default_factory=Geometry)
    opacity: bool = False
    fwhm_kms: float = 10.0              # intrinsic line FWHM for the optical depth (not the instrumental one)
    fwhm_free: bool = False
    R_free: bool = False                # emitting radius free (only meaningful with the opacity on)
    sys_frac: float = 0.10
    use: str = "all"                    # all measured features | detected
    snr_detect: float = 3.0
    bounds: dict = field(default_factory=dict)     # {param: [lo, hi]}
    fixed: dict = field(default_factory=dict)      # {param: value}: fixes a parameter
    start: dict = field(default_factory=dict)      # {param: value}: starting value
    tmax_powerlaw: float = 4000.0
    n_starts: int = 12


@dataclass
class MCMCConfig:
    walkers: int = 48
    steps: int = 3000
    burn: int = 1000
    thin: int = 1
    seed: int = 42


# ------------------------------------------------------------------------------------------------
# building the model
# ------------------------------------------------------------------------------------------------

def make_model(features: pd.DataFrame, members: pd.DataFrame, cfg: FitConfig, molecule: str | None = None,
               part: SpinPartition | None = None) -> RotModel:
    molecule = molecule or features.attrs.get("molecule")
    sp = preset(molecule)
    if part is None:
        part = partition(sp, load_species_linelist(sp.name, features.attrs.get("release")))
    ids = features["id"].to_numpy()
    pos = {int(f): j for j, f in enumerate(ids)}
    M = members[members["feature"].isin(ids)]
    fidx = np.array([pos[int(f)] for f in M["feature"]])
    kind = cfg.model
    if kind not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, not '{kind}'")
    if cfg.opr not in OPR_MODES:
        raise ValueError(f"opr must be one of {OPR_MODES}, not '{cfg.opr}'")
    geo = cfg.geometry
    m = RotModel(kind, part, M["wave"].to_numpy(float), M["a"].to_numpy(float), M["gu"].to_numpy(float),
                 M["eu"].to_numpy(float), M["spin"].astype(str).to_numpy(), fidx, len(features), get_curve(cfg.extinction),
                 geometry=geo, opr_mode=cfg.opr, opacity=cfg.opacity, opr_high_t=sp.opr_high_t)
    P = default_params(kind, m, sp.t_bounds)
    P["Av"].value, P["Av"].free = cfg.av, cfg.av_free
    has_spin = (cfg.opr != "thermal") and part.has_spin and bool(np.isin(m.member_spin, ["o", "p"]).any())
    P["OPR"].value, P["OPR"].free = cfg.opr_value, bool(has_spin and cfg.opr_free)
    if not has_spin:
        P["OPR"].free = False
    P["fwhm"].value, P["fwhm"].free = cfg.fwhm_kms, bool(cfg.opacity and cfg.fwhm_free)
    P["logR"].value = float(np.log10(max(geo.R_au, 1e-6)))
    P["logR"].free = bool(cfg.R_free and cfg.opacity and geo.mode in ("radius", "number"))
    if kind == "powerlaw":
        P["Tmax"].value = cfg.tmax_powerlaw
    for k, (lo, hi) in (cfg.bounds or {}).items():
        if k in P:
            P[k].lo, P[k].hi = float(lo), float(hi)
    for k, v in (cfg.start or {}).items():
        if k in P:
            P[k].value = float(v)
    for k, v in (cfg.fixed or {}).items():
        if k in P:
            P[k].value, P[k].free = float(v), False
    for p in P.values():
        p.value = float(np.clip(p.value, p.lo, p.hi))
    m.params = P
    return m


def fit_data(features: pd.DataFrame, cfg: FitConfig):
    """(F_obs, σ_total, use) of the features: σ² = σ_stat² + (f_sys F)²."""
    F = features["flux"].to_numpy(float)
    e = features["flux_err"].to_numpy(float)
    use = features["use"].to_numpy(bool).copy() if "use" in features else np.isfinite(F)
    use &= np.isfinite(F) & np.isfinite(e) & (e > 0)
    if cfg.use == "detected":
        use &= (F / np.where(e > 0, e, np.inf)) >= cfg.snr_detect
    sig = np.sqrt(np.where(np.isfinite(e), e, np.inf) ** 2 + (cfg.sys_frac * np.abs(np.nan_to_num(F))) ** 2)
    return F, sig, use


# ------------------------------------------------------------------------------------------------
# results
# ------------------------------------------------------------------------------------------------

@dataclass
class RotFit:
    model: RotModel
    cfg: FitConfig
    features: pd.DataFrame
    best: dict                       # all parameters (free and fixed)
    cov: np.ndarray | None           # of the free parameters (least squares)
    chi2: float
    n_used: int
    samples: np.ndarray | None = None      # (n, n_free)
    lnprob: np.ndarray | None = None
    mcmc_info: dict = field(default_factory=dict)
    derived: pd.DataFrame | None = None
    runtime_s: float = 0.0

    @property
    def free(self) -> list[str]:
        return self.model.free

    @property
    def dof(self) -> int:
        return max(self.n_used - len(self.free), 1)

    @property
    def chi2_red(self) -> float:
        return self.chi2 / self.dof

    @property
    def bic(self) -> float:
        return self.chi2 + len(self.free) * np.log(max(self.n_used, 1))

    @property
    def aic(self) -> float:
        return self.chi2 + 2 * len(self.free)

    def theta(self) -> np.ndarray:
        return np.array([self.best[k] for k in self.free])

    def errors(self) -> dict:
        """{param: (lo_err, hi_err)} from the MCMC percentiles, else from the covariance."""
        out = {}
        if self.samples is not None and len(self.samples):
            for j, k in enumerate(self.free):
                q = np.percentile(self.samples[:, j], [16, 50, 84])
                out[k] = (q[1] - q[0], q[2] - q[1])
        elif self.cov is not None:
            for j, k in enumerate(self.free):
                s = float(np.sqrt(max(self.cov[j, j], 0.0)))
                out[k] = (s, s)
        return out

    def table(self) -> pd.DataFrame:
        rows = []
        err = self.errors()
        for k, p in self.model.params.items():
            if k in ("logR", "fwhm") and not p.free and not self.model.opacity:
                continue
            if k == "OPR" and not p.free and self.model.opr_mode == "thermal":
                continue
            if k == "Tmax" and self.model.kind != "powerlaw":
                continue
            row = dict(param=k, label=p.label, best=self.best[k], free=p.free, lo=p.lo, hi=p.hi)
            if self.samples is not None and k in self.free:
                q = np.percentile(self.samples[:, self.free.index(k)], [16, 50, 84])
                row.update(median=q[1], err_lo=q[1] - q[0], err_hi=q[2] - q[1])
            elif k in err:
                row.update(median=self.best[k], err_lo=err[k][0], err_hi=err[k][1])
            else:
                row.update(median=self.best[k], err_lo=np.nan, err_hi=np.nan)
            rows.append(row)
        return pd.DataFrame(rows)

    def predicted(self) -> np.ndarray:
        return self.model.predict(self.theta())[0]

    def summary(self) -> str:
        t = self.table()
        lines = [f"{self.model.kind} model, OPR {self.model.opr_mode}, {self.n_used} features, χ² = {self.chi2:.1f} "
                 f"(χ²_red {self.chi2_red:.2f}, BIC {self.bic:.1f})"]
        for r in t.itertuples():
            e = "" if not np.isfinite(r.err_lo) else f" -{r.err_lo:.3g} +{r.err_hi:.3g}"
            lines.append(f"  {r.param:6s} = {r.median:.4g}{e}{'' if r.free else '  (fixed)'}")
        if self.derived is not None:
            for r in self.derived.itertuples():
                e = "" if not np.isfinite(r.err_lo) else f" -{r.err_lo:.3g} +{r.err_hi:.3g}"
                lines.append(f"  {r.quantity}: {r.value:.4g}{e} {r.unit}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = dict(model=self.model.kind, opr=self.model.opr_mode, opacity=self.model.opacity,
                 extinction=self.model.curve.name, geometry=self.model.geometry.__dict__.copy(),
                 chi2=float(self.chi2), chi2_red=float(self.chi2_red), bic=float(self.bic), aic=float(self.aic),
                 n_used=int(self.n_used), free=self.free,
                 params={r.param: dict(best=float(r.best), median=float(r.median), err_lo=float(r.err_lo), err_hi=float(r.err_hi),
                                       free=bool(r.free)) for r in self.table().itertuples()},
                 mcmc=self.mcmc_info)
        if self.derived is not None:
            d["derived"] = {r.quantity: dict(value=float(r.value), err_lo=float(r.err_lo), err_hi=float(r.err_hi), unit=r.unit)
                            for r in self.derived.itertuples()}
        return d


# ------------------------------------------------------------------------------------------------
# least squares
# ------------------------------------------------------------------------------------------------

_NPARAMS = {"single": ["logN"], "two": ["logN1", "logN2"], "powerlaw": ["logN"]}


def _linear_amplitudes(model: RotModel, theta: np.ndarray, Fo, sg, use):
    """Set the free log-N parameters to their best values given the others (the fluxes are linear in 10^logN
    when the lines are thin)."""
    names = model.free
    nparams = [k for k in _NPARAMS[model.kind] if k in names]
    if not nparams:
        return theta
    cols = []
    for k in nparams:
        th = theta.copy()
        for k2 in nparams:
            th[names.index(k2)] = 0.0 if k2 == k else -300.0
        cols.append(model.predict(th)[0][use] / sg[use])
    A = np.stack(cols, axis=1)
    if not np.all(np.isfinite(A)):
        return theta
    x, _ = nnls(A, Fo[use] / sg[use])
    th = theta.copy()
    lo, hi = model.bounds()
    for k, v in zip(nparams, x):
        j = names.index(k)
        th[j] = np.clip(np.log10(v) if v > 0 else lo[j] + 1.0, lo[j], hi[j])
    return th


def _starts(model: RotModel, n: int, rng) -> list[np.ndarray]:
    names = model.free
    lo, hi = model.bounds()
    base = np.clip(model.start(), lo, hi)
    grids = {"T": [150, 300, 500, 800, 1200, 2000, 3500], "T1": [150, 300, 500, 800], "T2": [900, 1500, 2500, 4000],
             "Tmin": [100, 200, 400, 700], "b": [2.5, 3.5, 4.5, 6.0], "Av": [0.0, 3.0, 10.0, 30.0], "OPR": [3.0, 2.0, 1.0],
             "logR": [-1.0, 0.0, 1.0], "fwhm": [2.0, 10.0, 40.0]}
    out = [base]
    keys = [k for k in names if k in grids]
    combos = [dict()]
    for k in keys:
        vals = [v for v in grids[k] if lo[names.index(k)] <= v <= hi[names.index(k)]] or [base[names.index(k)]]
        combos = [dict(c, **{k: v}) for c in combos for v in vals]
    if len(combos) > n:
        idx = rng.choice(len(combos), size=n, replace=False)
        combos = [combos[i] for i in idx]
    for c in combos:
        th = base.copy()
        for k, v in c.items():
            th[names.index(k)] = v
        out.append(th)
    return out


def least_squares_fit(model: RotModel, Fo, sg, use, n_starts: int = 12, seed: int = 0):
    lo, hi = model.bounds()
    lnp = lnlike_fn(model, Fo, sg, use)
    rng = np.random.default_rng(seed)
    wt = 1.0 / sg[use]

    def resid(th):
        return (Fo[use] - model.predict(th)[0][use]) * wt

    best = None
    for th0 in _starts(model, n_starts, rng):
        th0 = _linear_amplitudes(model, th0, Fo, sg, use)
        th0 = np.clip(th0, lo + 1e-9 * (hi - lo), hi - 1e-9 * (hi - lo))
        if not np.isfinite(lnp(th0)):          # violates the ordering priors
            continue
        try:
            r = least_squares(resid, th0, bounds=(lo, hi), x_scale="jac", method="trf", max_nfev=400)
        except Exception:
            continue
        if not np.isfinite(lnp(r.x)):
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        raise RuntimeError("no valid starting point (check the bounds / the T ordering of the two components)")
    J = best.jac
    try:
        cov = np.linalg.pinv(J.T @ J)
    except np.linalg.LinAlgError:
        cov = None
    return best.x, cov, float(2 * best.cost)


def fit_rotation(features: pd.DataFrame, members: pd.DataFrame, cfg: FitConfig | None = None, molecule: str | None = None,
                 mcmc: MCMCConfig | None = None, part: SpinPartition | None = None, progress=None) -> RotFit:
    """Fit the measured features (with 'flux', 'flux_err', 'use' columns).  With `mcmc`, also sample the posterior."""
    t0 = time.time()
    cfg = cfg or FitConfig()
    model = make_model(features, members, cfg, molecule, part)
    Fo, sg, use = fit_data(features, cfg)
    if use.sum() < 2:
        raise ValueError("fewer than 2 usable features: measure the lines first (or tick more lines)")
    if use.sum() <= len(model.free):
        raise ValueError(f"{use.sum()} usable features for {len(model.free)} free parameters ({', '.join(model.free)}): "
                         "fix some parameters (A_V, OPR, ...) or add lines")
    th, cov, chi2 = least_squares_fit(model, Fo, sg, use, cfg.n_starts)
    best = {k: float(v[0]) for k, v in model.full(th).items()}
    res = RotFit(model, cfg, features, best, cov, chi2, int(use.sum()))
    if mcmc is not None and mcmc.steps > 0:
        run_mcmc(res, mcmc, progress=progress)
    res.derived = derived_quantities(res)
    res.runtime_s = time.time() - t0
    return res


def run_mcmc(res: RotFit, mc: MCMCConfig, progress=None) -> RotFit:
    """emcee (vectorised: one model call evaluates all walkers) started around the least-squares solution."""
    import emcee
    model = res.model
    Fo, sg, use = fit_data(res.features, res.cfg)
    lnp = lnlike_fn(model, Fo, sg, use)
    nd = len(model.free)
    nw = max(mc.walkers, 2 * nd + 2)
    nw += nw % 2
    rng = np.random.default_rng(mc.seed)
    lo, hi = model.bounds()
    th = res.theta()
    scale = np.sqrt(np.clip(np.diag(res.cov), 0, None)) if res.cov is not None else np.zeros(nd)
    scale = np.where(np.isfinite(scale) & (scale > 0), np.minimum(scale, 0.05 * (hi - lo)), 1e-3 * (hi - lo))
    p0 = []
    tries = 0
    while len(p0) < nw and tries < 200 * nw:
        tries += 1
        cand = th + 0.3 * scale * rng.normal(size=nd)
        cand = np.clip(cand, lo + 1e-9 * (hi - lo), hi - 1e-9 * (hi - lo))
        if np.isfinite(lnp(cand)):
            p0.append(cand)
    if len(p0) < nw:
        raise RuntimeError("could not initialise the walkers inside the priors")
    sampler = emcee.EnsembleSampler(nw, nd, lnp, vectorize=True)
    state = np.array(p0)
    done, chunk = 0, max(50, mc.steps // 20)
    t0 = time.time()
    while done < mc.steps:
        n = min(chunk, mc.steps - done)
        state = sampler.run_mcmc(state, n, progress=False)
        done += n
        if progress is not None:
            progress(done / mc.steps)
    burn = min(mc.burn, mc.steps // 2)
    chain = sampler.get_chain(discard=burn, thin=mc.thin, flat=True)
    lp = sampler.get_log_prob(discard=burn, thin=mc.thin, flat=True)
    try:
        tau = sampler.get_autocorr_time(tol=0, discard=burn)
    except Exception:
        tau = np.full(nd, np.nan)
    res.samples, res.lnprob = chain, lp
    j = int(np.argmax(lp))
    if -2 * lp[j] < res.chi2 - 1e-6:          # the sampler found a better point
        res.best.update({k: float(v) for k, v in zip(model.free, chain[j])})
        res.chi2 = float(-2 * lp[j])
    res.mcmc_info = dict(walkers=nw, steps=mc.steps, burn=burn, thin=mc.thin, n_samples=int(len(chain)),
                         acceptance=float(np.mean(sampler.acceptance_fraction)),
                         tau=[float(t) for t in np.atleast_1d(tau)], tau_max=float(np.nanmax(tau)) if np.isfinite(tau).any() else np.nan,
                         converged=bool(np.isfinite(tau).all() and mc.steps - burn > 30 * np.nanmax(tau)),
                         seconds=round(time.time() - t0, 2))
    return res


# ------------------------------------------------------------------------------------------------
# derived quantities
# ------------------------------------------------------------------------------------------------

def derived_quantities(res: RotFit, max_samples: int = 4000) -> pd.DataFrame:
    """Total column / number of molecules, mass, emitting radius, A_K, LTE OPR, τ_max, line luminosity."""
    m = res.model
    geo = m.geometry
    mol = get_molecule(res.features.attrs.get("molecule", "H2"))
    if res.samples is not None and len(res.samples):
        S = res.samples
        if len(S) > max_samples:
            S = S[np.random.default_rng(1).choice(len(S), max_samples, replace=False)]
    else:
        S = res.theta()[None, :]
    P = m.full(S)
    bestP = m.full(res.theta())
    rows = []

    def add(name, arr, best_val, unit):
        arr = np.asarray(arr, float)
        arr = arr[np.isfinite(arr)]
        if len(arr) > 1:
            q = np.percentile(arr, [16, 50, 84])
            rows.append(dict(quantity=name, value=q[1], err_lo=q[1] - q[0], err_hi=q[2] - q[1], best=float(best_val), unit=unit))
        else:
            rows.append(dict(quantity=name, value=float(best_val), err_lo=np.nan, err_hi=np.nan, best=float(best_val), unit=unit))

    def ntot(PP):
        if m.kind == "two":
            return 10 ** PP["logN1"] + 10 ** PP["logN2"]
        return 10 ** PP["logN"]

    N, Nb = ntot(P), ntot(bestP)[0]
    d = geo.distance_pc * PC
    R = 10 ** P["logR"]; Rb = 10 ** bestP["logR"][0]
    m_mol = mol.mass_amu * AMU
    mass = mass_b = None
    if geo.mode == "number":
        add("log number of molecules", np.log10(N), np.log10(Nb), "")
        mass, mass_b = N * m_mol, Nb * m_mol
        if m.opacity or m.params["logR"].free:
            add("log N (over π R²) [cm⁻²]", np.log10(N / (np.pi * (R * AU * 100) ** 2)), np.log10(Nb / (np.pi * (Rb * AU * 100) ** 2)), "")
    elif geo.mode == "intensity":
        add("log N [cm⁻²]", np.log10(N), np.log10(Nb), "")
    else:
        add("log N [cm⁻²]", np.log10(N), np.log10(Nb), "")
        om = geo.omega(R) if geo.mode == "radius" else geo.omega() * np.ones_like(R)
        omb = float(np.atleast_1d(geo.omega(Rb) if geo.mode == "radius" else geo.omega())[0])
        num, num_b = N * 1e4 * om * d * d, Nb * 1e4 * omb * d * d
        add("log number of molecules", np.log10(num), np.log10(num_b), "")
        mass, mass_b = num * m_mol, num_b * m_mol
        if geo.mode == "aperture":
            add("equivalent radius", d * np.sqrt(om / np.pi) / AU, d * np.sqrt(omb / np.pi) / AU, "au")
        elif m.params["logR"].free:
            add("emitting radius R", R, Rb, "au")
    if mass is not None:
        add("mass", mass / M_EARTH, mass_b / M_EARTH, "M⊕")
        if mol.name == "H2":
            add("mass [M_Jup]", mass / M_JUP, mass_b / M_JUP, "M_Jup")
    if m.kind == "two":
        add("hot fraction N₂/(N₁+N₂)", 10 ** P["logN2"] / N, 10 ** bestP["logN2"][0] / Nb, "")
    if m.params["Av"].free or m.params["Av"].value > 0:
        ak = m.curve.ak_av
        add("A_K", P["Av"] * ak, bestP["Av"][0] * ak, "mag")
    if m.part.has_spin and np.isin(m.member_spin, ["o", "p"]).any():
        T = P["T"] if m.kind == "single" else P["T1"] if m.kind == "two" else P["Tmin"]
        Tb = bestP["T"][0] if m.kind == "single" else bestP["T1"][0] if m.kind == "two" else bestP["Tmin"][0]
        add("LTE OPR at T" + ("" if m.kind == "single" else "₁" if m.kind == "two" else "_min"), m.part.opr_lte(T), float(m.part.opr_lte(Tb)), "")
    if m.kind == "powerlaw":
        tmin, tmax, b = P["Tmin"], P["Tmax"], P["b"]
        lt = np.linspace(0, 1, 200)[None, :]
        Tg = np.exp(np.log(tmin)[:, None] + lt * np.log(tmax / tmin)[:, None])
        w = Tg ** (1 - b[:, None])
        mt = (w * Tg).sum(1) / w.sum(1)
        Tgb = np.exp(np.log(bestP["Tmin"])[:, None] + lt * np.log(bestP["Tmax"] / bestP["Tmin"])[:, None])
        wb = Tgb ** (1 - bestP["b"][:, None])
        add("mean T (column-weighted)", mt, float(((wb * Tgb).sum(1) / wb.sum(1))[0]), "K")
    if m.opacity:
        tb = m.tau(res.theta())[0]
        add("max line-centre τ", m.tau(S).max(axis=1) if len(S) <= 2000 else np.full(1, tb.max()), tb.max(), "")
    # dereddened luminosity of the fitted features
    Pn = dict(P); Pn["Av"] = np.zeros_like(P["Av"])
    Fint = m.member_flux(Pn) @ m.S.T
    Pb = dict(bestP); Pb["Av"] = np.zeros(1)
    Fint_b = (m.member_flux(Pb) @ m.S.T)[0]
    _, _, use = fit_data(res.features, res.cfg)
    if geo.mode != "intensity":
        add("luminosity of the fitted lines", 4 * np.pi * d * d * Fint[:, use].sum(1) / L_SUN, 4 * np.pi * d * d * Fint_b[use].sum() / L_SUN, "L⊙")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------
# the diagram
# ------------------------------------------------------------------------------------------------

def diagram_points(features: pd.DataFrame, geometry: Geometry, av: float = 0.0, curve: str = "G23", R_au: float | None = None,
                   snr_detect: float = 3.0) -> pd.DataFrame:
    """x = E_u [K], y = ln(N_u/g_u) (cm⁻² or molecules) of every feature, de-reddened by `av`; upper limits
    (snr_detect σ) for the non-detections."""
    F = features["flux"].to_numpy(float)
    e = features["flux_err"].to_numpy(float)
    wf = features["wave"].to_numpy(float)
    nu = C / (wf * 1e-6)
    gA = features["gA"].to_numpy(float)
    if geometry.column:
        om = float(np.atleast_1d(geometry.omega(R_au))[0])
        conv = 4 * np.pi / (H * nu * om * gA) * 1e-4
    else:
        d = geometry.distance_pc * PC
        conv = 4 * np.pi * d * d / (H * nu * gA)
    k = get_curve(curve)(wf)
    der = np.exp(LN10_04 * av * k)
    det = np.isfinite(F) & (F > 0) & (F >= snr_detect * e)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.where(F > 0, np.log(F * conv * der), np.nan)
        yerr = np.where(F > 0, e / F, np.nan)
        yul = np.log(snr_detect * e * conv * der)
    return pd.DataFrame(dict(id=features["id"], label=features["label"], wave=wf, eu=features["eu"], y=y, yerr=yerr,
                             y_ul=yul, detected=det, use=features.get("use", pd.Series(True, index=features.index)).to_numpy(bool),
                             spin=features.get("spin", ""), ladder=features.get("ladder", ""),
                             band_v=features.get("band_v", "")))


def model_points(res: RotFit, geometry: Geometry | None = None, deredden: bool = True) -> np.ndarray:
    """The model fluxes of the features as diagram y values (same conversion as the data points)."""
    geometry = geometry or res.model.geometry
    Fm = res.predicted()
    f = res.features.copy()
    f["flux"] = Fm; f["flux_err"] = Fm * 0.1
    R = 10 ** res.best.get("logR", 0.0)
    return diagram_points(f, geometry, res.best.get("Av", 0.0) if deredden else 0.0, res.model.curve.name, R)["y"].to_numpy()


def model_curves(res: RotFit, E: np.ndarray | None = None) -> dict[str, list[np.ndarray]]:
    """Intrinsic (thin, unreddened) ln(N_u/g_u) curves of the best fit: {spin: [components..., total]}."""
    m = res.model
    if E is None:
        e = m.member_eu
        E = np.linspace(0.0, max(e.max() * 1.08, 500.0), 300)
    P1 = {k: np.array([v]) for k, v in res.best.items()}
    spins = [""]
    if m.opr_mode != "thermal" and m.part.has_spin and np.isin(m.member_spin, ["o", "p"]).any():
        spins = ["o", "p"]
    return {"E": [E], **{s: m.curve_y(P1, E, s) for s in spins}}


def compare_models(features, members, base: FitConfig, models=("single", "two", "powerlaw"), molecule=None, part=None) -> pd.DataFrame:
    """Least-squares fits of several models; χ², BIC, AIC side by side (lower BIC = preferred)."""
    rows = []
    for k in models:
        cfg = FitConfig(**{**base.__dict__, "model": k})
        try:
            r = fit_rotation(features, members, cfg, molecule, part=part)
            rows.append(dict(model=k, n_free=len(r.free), chi2=r.chi2, chi2_red=r.chi2_red, bic=r.bic, aic=r.aic,
                             params=", ".join(f"{p}={r.best[p]:.4g}" for p in r.free)))
        except Exception as ex:
            rows.append(dict(model=k, n_free=np.nan, chi2=np.nan, chi2_red=np.nan, bic=np.nan, aic=np.nan, params=f"failed: {ex}"))
    t = pd.DataFrame(rows)
    if t["bic"].notna().any():
        t["dBIC"] = t["bic"] - t["bic"].min()
    return t
