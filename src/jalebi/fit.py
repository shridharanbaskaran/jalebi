"""Fitting engine: one likelihood, one parameter vector, staged inference.

Stages
  1. grid      chi^2 over (log N, T) for one component with its area solved by NNLS
  2. optimise  joint global optimiser (differential evolution, areas by NNLS) + Nelder-Mead polish
  3. mcmc      emcee (parallel over a process pool), convergence diagnostics, summaries; the emitting
               areas can be profiled or marginalised out of it (fit.mcmc.linear, jalebi.linear)

Parameterisation
  per component: logN [cm^-2], T [K], logR [au] (or logNA = log10(N * A[au^2]) when
  area_param="logNA"), optional rv [km/s] and fwhm [km/s]; tied components: ratio.
  absorption screens: logN, T, rv, fc (covering fraction) and optional fwhm; no area.
  global: log_s (noise scale), optional.
Priors: uniform within bounds, optional Gaussian, ordering constraints T_a > T_b.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .continuum import in_ranges
from .data import Spectrum
from .model import Component, build_model, PARAM_LABELS

LOG10PI = np.log10(np.pi)
AU_CM2 = (1.495978707e13) ** 2       # cm^2 per au^2


# ------------------------------------------------------------------------------------
# Parameters and priors
# ------------------------------------------------------------------------------------

@dataclass
class Param:
    comp: str                 # component name, or "global"
    name: str                 # logN, T, logR, logNA, rv, fwhm, ratio, q, p, logRin, log_s
    lo: float
    hi: float
    init: float | None = None
    gauss: tuple[float, float] | None = None   # (mu, sigma) Gaussian prior on top of the bounds

    @property
    def key(self):
        return f"{self.comp}.{self.name}"

    @property
    def label(self):
        return f"{self.comp} {PARAM_LABELS.get(self.name, self.name)}"


DEFAULT_BOUNDS = {"logN": (13.0, 21.0), "T": (100.0, 1500.0), "logR": (-2.5, 1.5), "logNA": (12.0, 22.0),
                  "rv": (-30.0, 30.0), "fwhm": (2.0, 30.0), "ratio": (10.0, 300.0), "q": (0.0, 1.5),
                  "p": (-1.0, 3.0), "logRin": (-2.5, 0.5), "log_s": (-1.0, 1.0), "fc": (0.0, 1.0),
                  "Tvib": (100.0, 1500.0)}
# absorbing screens: colder gas, larger velocities (outflows), higher columns
ABSORPTION_BOUNDS = {"logN": (13.0, 22.0), "T": (20.0, 1500.0), "rv": (-200.0, 200.0), "fwhm": (1.0, 60.0),
                     "fc": (0.0, 1.0)}


def logNA_from(logN, logR):
    return logN + LOG10PI + 2.0 * logR


def logR_from(logNA, logN):
    return 0.5 * (logNA - logN - LOG10PI)


# ------------------------------------------------------------------------------------
# The fit problem
# ------------------------------------------------------------------------------------

class FitProblem:
    """Binds a spectrum, a model and the free parameters into log_prob(theta)."""

    def __init__(self, spec: Spectrum, components: list[Component], windows: list[tuple[float, float]],
                 free: list[Param], area_param: str = "logR", fit_noise_scale: bool = False,
                 ordering: list[tuple[str, str]] | None = None, window_weights: dict | None = None,
                 oversample: int = 6, linelists=None, releases=None, model_kwargs: dict | None = None,
                 use_pipeline_err: bool = False, tvib_below_trot: bool = True):
        self.spec = spec
        self.components = components
        self.windows = windows
        self.area_param = area_param
        self.ordering = ordering or []
        self.tvib_below_trot = tvib_below_trot
        self._tvib_comps = [c.name for c in components if c.enabled and c.Tvib is not None and not c.tie_to]
        self.fit_noise_scale = fit_noise_scale
        # pixel selection
        used = spec.mask & in_ranges(spec.wave, windows) & np.isfinite(spec.flux)
        if spec.continuum is not None:
            used &= np.isfinite(spec.continuum)
        self.used = used
        self.wave = spec.wave[used]
        self.band = np.asarray(spec.band)[used]
        self.y = spec.line_flux[used]
        sig = spec.err[used].astype(float)
        if not use_pipeline_err:
            from .data import estimate_noise
            noise = estimate_noise(spec)[used]
            good = np.isfinite(noise) & (noise > 0)
            sig = np.where(good, noise, sig)
        sig = np.where(np.isfinite(sig) & (sig > 0), sig, np.nanmedian(sig))
        self.sigma = sig
        w = np.ones(len(self.wave))
        if window_weights:
            for (a, b), ww in zip(windows, [window_weights.get(i, 1.0) for i in range(len(windows))]):
                w[(self.wave >= a) & (self.wave <= b)] = ww
        self.weights = w
        cont = spec.continuum[used] if spec.continuum is not None else None
        self.model = build_model(components, self.wave, spec.distance_pc, windows, linelists=linelists,
                                 releases=releases, oversample=oversample, continuum=cont, **(model_kwargs or {}))
        if self.model.absorbers() and (cont is None or not np.any(np.isfinite(cont) & (cont != 0))):
            import warnings
            warnings.warn("absorption components need a continuum (spec.continuum is empty): they will "
                          "contribute nothing.  Fit a continuum first, or load one with the spectrum.")
        self.free = list(free)
        if fit_noise_scale and not any(p.name == "log_s" for p in self.free):
            self.free.append(Param("global", "log_s", *DEFAULT_BOUNDS["log_s"], init=0.0))
        self.ndim = len(self.free)
        self.lo = np.array([p.lo for p in self.free]); self.hi = np.array([p.hi for p in self.free])
        self._base = {c.name: c.params() for c in components}
        self.ncall = 0

    # ---- emulator (fit.model_backend: emulator) -----------------------------------------
    def emulator_bounds(self) -> tuple[dict, dict]:
        """Per unit: the (T, log N) box the sampler can reach (prior bounds of the free parameters, the fixed
        value otherwise; a tied isotopologue's log N box shifted by its ratio range) and the largest area."""
        free = {p.key: p for p in self.free}
        comps = {c.name: c for c in self.components if c.enabled}

        def rng(cname, par):
            p = free.get(f"{cname}.{par}")
            v = float(self._base[cname][par])
            return (p.lo, p.hi) if p is not None else (v, v)
        bounds, amax = {}, {}
        for key, members in self.model._units().items():
            c = members[0]
            if c.kind != "slab":
                continue
            par = comps.get(c.tie_to) if c.tie_to else c
            if par is None:
                continue
            T = rng(par.name, "T")
            N = rng(par.name, "logN")
            if c.tie_to:
                r = free.get(f"{c.name}.ratio")
                rv = float(self._base[c.name].get("ratio", 70.0))
                rlo, rhi = (r.lo, r.hi) if r is not None else (rv, rv)
                N = (N[0] - np.log10(rhi), N[1] - np.log10(rlo))
            a = free.get(f"{par.name}.logR")
            na = free.get(f"{par.name}.logNA")
            if a is not None:
                am = 10.0 ** (2.0 * a.hi)
            elif na is not None:
                am = 10.0 ** (na.hi - rng(par.name, "logN")[0] - LOG10PI)
            else:
                am = 10.0 ** (2.0 * float(self._base[par.name]["logR"]))
            bounds[key] = {"T": T, "logN": N}
            amax[key] = min(am, 10.0 ** (2.0 * DEFAULT_BOUNDS["logR"][1]))
        return bounds, amax

    def use_emulator(self, settings=None, say=None):
        """Build or load the emulator tables for this problem (fit.model_backend: emulator) and attach them
        to the model.  The accuracy is certified against this problem's noise (sigma / sqrt(weight)) at the
        largest area the brightest pixel allows."""
        from .emulator import attach_emulator
        bounds, amax = self.emulator_bounds()
        sig = self.sigma / np.sqrt(self.weights)
        f_ref = float(np.nanmax(np.abs(self.y))) if len(self.y) else 1.0
        return attach_emulator(self.model, bounds, sig, f_ref, settings, {p.key for p in self.free}, amax, say)

    # ---- parameter mapping ----------------------------------------------------------
    def theta0(self) -> np.ndarray:
        th = []
        for p in self.free:
            if p.init is not None:
                th.append(p.init)
            elif p.comp == "global":
                th.append(0.0)
            else:
                base = self._base[p.comp]
                if p.name == "logNA":
                    th.append(logNA_from(base["logN"], base["logR"]))
                else:
                    th.append(base.get(p.name, 0.5 * (p.lo + p.hi)))
        return np.clip(np.array(th, float), self.lo + 1e-6 * (self.hi - self.lo), self.hi - 1e-6 * (self.hi - self.lo))

    def params_from_theta(self, theta) -> tuple[dict[str, dict], float]:
        """Component parameter dict (for SlabModel.evaluate) and the noise scale."""
        P = {k: dict(v) for k, v in self._base.items()}
        log_s = 0.0
        for p, v in zip(self.free, theta):
            if p.comp == "global":
                if p.name == "log_s":
                    log_s = float(v)
                continue
            P[p.comp][p.name] = float(v)
        # convert logNA -> logR where used
        for cname, d in P.items():
            if "logNA" in d:
                d["logR"] = logR_from(d["logNA"], d["logN"])
        return P, log_s

    def theta_from_params(self, P: dict, log_s: float = 0.0) -> np.ndarray:
        th = []
        for p in self.free:
            if p.comp == "global":
                th.append(log_s)
            elif p.name == "logNA":
                d = P[p.comp]
                th.append(d["logNA"] if "logNA" in d else logNA_from(d["logN"], d["logR"]))
            else:
                th.append(P[p.comp][p.name])
        return np.array(th, float)

    # ---- probability -------------------------------------------------------------------
    def log_prior(self, theta) -> float:
        if np.any(theta < self.lo) or np.any(theta > self.hi):
            return -np.inf
        lp = 0.0
        for p, v in zip(self.free, theta):
            if p.gauss is not None:
                mu, sd = p.gauss
                lp += -0.5 * ((v - mu) / sd) ** 2
        if self.ordering or (self.tvib_below_trot and self._tvib_comps):
            P, _ = self.params_from_theta(theta)
            for hot, cold in self.ordering:
                if P[hot]["T"] <= P[cold]["T"]:
                    return -np.inf
            if self.tvib_below_trot:
                for name in self._tvib_comps:
                    tv = P[name].get("Tvib")
                    if tv is not None and tv > P[name]["T"]:
                        return -np.inf
        return lp

    def model_flux(self, theta) -> np.ndarray:
        P, _ = self.params_from_theta(theta)
        return self.model.evaluate(P)

    def chi2(self, theta, model=None) -> float:
        m = self.model_flux(theta) if model is None else model
        return float(np.sum(self.weights * ((self.y - m) / self.sigma) ** 2))

    def log_like(self, theta) -> float:
        P, log_s = self.params_from_theta(theta)
        m = self.model.evaluate(P)
        s2 = (10.0 ** log_s) ** 2
        r2 = ((self.y - m) ** 2) / (self.sigma**2 * s2)
        self.ncall += 1
        return float(-0.5 * np.sum(self.weights * (r2 + np.log(2 * np.pi * self.sigma**2 * s2))))

    def log_prob(self, theta) -> float:
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_like(theta)
        return lp + ll if np.isfinite(ll) else -np.inf

    # ---- vectorised over walkers (emcee vectorize=True) ------------------------------------
    def unit_fluxes_many(self, thetas):
        """(list of resolved parameter dicts, {unit: (nw, npix) 1-au flux}, {unit: (nw,) logR}) for many
        parameter vectors: one table gather per unit when every unit is emulated, else a loop."""
        Ps, ls = [], []
        for t in thetas:
            P, log_s = self.params_from_theta(t)
            Ps.append(self.model.resolve_params(P)); ls.append(log_s)
        em = getattr(self.model, "emulator", None)
        got = em.unit_fluxes_many(self.model, Ps) if em is not None else None
        if got is None:
            per = [self.model.unit_fluxes(P) for P in Ps]
            keys = list(per[0][0]) if per else []
            F = {k: np.stack([u[0][k] for u in per]) for k in keys}
            lR = {k: np.array([u[2][k] for u in per]) for k in keys}
        else:
            F, lR = got
        return Ps, np.array(ls), F, lR

    def log_prob_many(self, thetas) -> np.ndarray:
        """ln P for an (nw, ndim) array of parameter vectors (same values as log_prob, one model pass)."""
        thetas = np.atleast_2d(np.asarray(thetas, float))
        out = np.full(len(thetas), -np.inf)
        lp = np.array([self.log_prior(t) for t in thetas])
        ok = np.flatnonzero(np.isfinite(lp))
        if len(ok) == 0:
            return out
        _, ls, F, lR = self.unit_fluxes_many(thetas[ok])
        M = np.zeros((len(ok), len(self.y)))
        for k, f in F.items():
            M += f * (10.0 ** (2.0 * lR[k]))[:, None]
        s2 = 10.0 ** (2.0 * ls)
        r2 = (self.y[None, :] - M) ** 2 / (self.sigma[None, :] ** 2 * s2[:, None])
        ll = -0.5 * np.sum(self.weights[None, :] * (r2 + np.log(2 * np.pi * self.sigma[None, :] ** 2 * s2[:, None])), axis=1)
        self.ncall += len(ok)
        out[ok] = np.where(np.isfinite(ll), lp[ok] + ll, -np.inf)
        return out

    # ---- detection test ------------------------------------------------------------------
    def component_significance(self, theta) -> pd.DataFrame:
        """For every independent unit: chi2 increase when it is removed (logN -> -inf) and the
        corresponding delta-BIC (positive = the component is supported by the data)."""
        P, log_s = self.params_from_theta(theta)
        chi2_full = self.chi2(theta)
        n = len(self.y)
        rows = []
        for unit, lead in self.all_units().items():
            members = [c.name for c in self.components if c.enabled and (c.group == unit or c.name == unit or c.tie_to in (lead, unit))]
            P2 = {k: dict(v) for k, v in P.items()}
            for m in members:
                P2[m]["logN"] = -30.0
            chi2_wo = float(np.sum(self.weights * ((self.y - self.model.evaluate(P2)) / self.sigma) ** 2))
            k = sum(1 for p in self.free if p.comp in members)
            dchi2 = chi2_wo - chi2_full
            rows.append({"component": unit, "delta_chi2": dchi2, "k": k, "delta_BIC": dchi2 - k * np.log(n),
                         "tau_max": self.model.tau_flags(P).get(unit, np.nan),
                         "detected": bool(dchi2 - k * np.log(n) > 10)})
        return pd.DataFrame(rows)

    # ---- areas by NNLS (profiling out the linear parameters) ---------------------------
    def all_units(self) -> dict[str, str]:
        """Every independent unit (emitting or absorbing) -> leading component name."""
        units = {}
        for c in self.components:
            if c.enabled and not c.tie_to:
                units.setdefault(c.group or c.name, c.name)
        return units

    def area_units(self) -> dict[str, str]:
        """unit key (component or group) -> name of the leading component (absorption screens
        have no area and are left out)."""
        units = {}
        for c in self.components:
            if c.enabled and not c.tie_to and c.kind != "absorption":
                key = c.group or c.name
                units.setdefault(key, c.name)
        return units

    def solve_areas(self, P: dict, fixed_units: set[str] | None = None) -> tuple[dict, float]:
        logR, chi2 = self.model.solve_areas(self.y, self.sigma / np.sqrt(self.weights), P, fixed=fixed_units)
        for key, lead in self.area_units().items():
            if key in logR:
                P[lead]["logR"] = logR[key]
                if "logNA" in P[lead]:
                    P[lead]["logNA"] = logNA_from(P[lead]["logN"], logR[key])
        # push the leader's logR to group members
        au = self.area_units()
        for c in self.components:
            if c.group and au.get(c.group) and c.name != au[c.group]:
                P[c.name]["logR"] = P[au[c.group]]["logR"]
        return P, chi2

    def area_free_mask(self) -> np.ndarray:
        """Free parameters that are emitting areas of slab units (profiled by NNLS in the grid and the
        optimiser).  The logR of an annuli component is its outer radius, not a linear scale, so it stays
        an ordinary parameter."""
        kind = {c.name: c.kind for c in self.components}
        return np.array([p.name in ("logR", "logNA") and kind.get(p.comp) not in ("annuli", "absorption") for p in self.free])

    # ---- stage 1: grid --------------------------------------------------------------------
    def grid(self, comp: str, logN=None, T=None, windows=None, n_jobs: int = 1, base_theta=None,
             progress=None) -> "GridResult":
        """chi^2 map over (logN, T) for component `comp`, its area by NNLS, other components at
        their current values.  Optional `windows` restricts the pixels used."""
        logN = np.linspace(14.0, 20.0, 31) if logN is None else np.asarray(logN)
        T = np.linspace(150.0, 1200.0, 22) if T is None else np.asarray(T)
        P0, _ = self.params_from_theta(self.theta0() if base_theta is None else base_theta)
        sel = np.ones(len(self.wave), bool) if windows is None else in_ranges(self.wave, windows)
        units = self.all_units()
        cobj = next(c for c in self.components if c.name == comp)
        my_unit = cobj.group or comp
        others = set(units) - {my_unit}

        annuli = cobj.kind in ("annuli", "absorption")     # no linear area to solve for

        def one(iN, iT):
            P = {k: dict(v) for k, v in P0.items()}
            P[comp]["logN"] = float(logN[iN]); P[comp]["T"] = float(T[iT])
            uf, tmax, lR = self.model.unit_fluxes(P)
            uf = self.model.fold_tied(uf)            # isotopologues ride on their parent's area
            y = self.y.copy()
            for k in others:
                if k in uf:
                    y -= uf[k] * 10.0 ** (2.0 * lR[k])
            a = uf[my_unit][sel] / self.sigma[sel]
            b = y[sel] / self.sigma[sel]
            if annuli:                                # outer radius (or a screen): no linear scale
                coef = 10.0 ** (2.0 * lR[my_unit])
            else:
                coef = max(float(a @ b) / max(float(a @ a), 1e-300), 0.0)     # 1-D NNLS
            resid = b - coef * a
            return float(resid @ resid), 0.5 * np.log10(max(coef, 1e-12)), tmax.get(my_unit, np.nan)

        chi2 = np.zeros((len(T), len(logN))); logR = np.zeros_like(chi2); tmax = np.zeros_like(chi2)
        jobs = [(iN, iT) for iT in range(len(T)) for iN in range(len(logN))]
        if n_jobs != 1:
            from joblib import Parallel, delayed
            out = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(one)(iN, iT) for iN, iT in jobs)
        else:
            out = []
            for k, (iN, iT) in enumerate(jobs):
                out.append(one(iN, iT))
                if progress and k % 50 == 0:
                    progress(k / len(jobs))
        for (iN, iT), (c2, lr, tm) in zip(jobs, out):
            chi2[iT, iN] = c2; logR[iT, iN] = lr; tmax[iT, iN] = tm
        return GridResult(comp, logN, T, chi2, logR, tmax, int(sel.sum()))

    def _de_energy(self, sub, theta0, idx, amask) -> float:
        """Objective for the global optimiser: -2 ln(prob) with the areas profiled out by NNLS."""
        th = theta0.copy(); th[idx] = sub
        lp = self.log_prior(th)
        if not np.isfinite(lp):
            return 1e30
        P, log_s = self.params_from_theta(th)
        if amask.any():
            P, chi2 = self.solve_areas(P)
        else:
            chi2 = self.chi2(th)
        s2 = (10.0 ** log_s) ** 2
        return chi2 / s2 + np.sum(self.weights * np.log(s2)) - 2 * lp

    # ---- stage 2: optimiser --------------------------------------------------------------------
    def optimise(self, theta0=None, method: str = "de", maxiter: int = 200, popsize: int = 15, seed: int = 0,
                 polish: bool = True, workers: int = 1, callback=None, profile_areas: bool = True,
                 progress=None) -> "OptResult":
        """Global optimiser over the free parameters.  With `profile_areas` the area parameters are
        removed from the search and solved by NNLS at every objective call.
        `progress(generation, maxiter, best_energy, elapsed_s)` is called after every generation;
        `callback` (returning True) stops the search early."""
        from scipy.optimize import differential_evolution, minimize
        theta0 = self.theta0() if theta0 is None else np.asarray(theta0, float)
        gen = {"n": 0}
        t_gen0 = time.time()

        def de_callback(intermediate_result):
            # the parameter must be named exactly `intermediate_result` (scipy >= 1.12) to receive the
            # OptimizeResult with the current best energy
            gen["n"] += 1
            best = float(getattr(intermediate_result, "fun", np.nan))
            if progress:
                progress(gen["n"], maxiter, best, time.time() - t_gen0)
            if callback is not None:
                try:
                    return bool(callback(intermediate_result))
                except TypeError:
                    return bool(callback())
            return False
        amask = self.area_free_mask() & profile_areas
        idx = np.flatnonzero(~amask)
        t_start = time.time()
        objective = lambda sub: self._de_energy(sub, theta0, idx, amask)

        best_sub = theta0[idx]
        if method == "de" and len(idx) > 0:
            bounds = list(zip(self.lo[idx], self.hi[idx]))
            x0 = np.clip(theta0[idx], self.lo[idx], self.hi[idx])
            pool = None
            try:
                if workers != 1:
                    ctx = mp.get_context("fork") if hasattr(os, "fork") else mp.get_context("spawn")
                    nproc = workers if workers > 0 else (os.cpu_count() or 1)
                    pool = ctx.Pool(nproc, initializer=_init_de_worker, initargs=(self, theta0, idx, amask))
                    mapper = pool.map
                else:
                    mapper = 1
                res = differential_evolution(objective if pool is None else _de_worker, bounds, maxiter=maxiter,
                                             popsize=popsize, seed=seed, tol=1e-6, polish=False, workers=mapper,
                                             updating="deferred" if pool is not None else "immediate", x0=x0,
                                             callback=de_callback)
            finally:
                if pool is not None:
                    pool.close(); pool.join()
            best_sub = res.x
        elif len(idx) > 0:
            res = minimize(objective, theta0[idx], method="Nelder-Mead", options={"maxiter": maxiter * 20, "xatol": 1e-4, "fatol": 1e-4})
            best_sub = res.x
        th = theta0.copy(); th[idx] = best_sub
        P, log_s = self.params_from_theta(th)
        if amask.any():
            P, _ = self.solve_areas(P)
            th = self.theta_from_params(P, log_s)
        th = np.clip(th, self.lo + 1e-9, self.hi - 1e-9)
        if polish:
            res2 = minimize(lambda t: -self.log_prob(t), th, method="Nelder-Mead",
                            options={"maxiter": 4000, "xatol": 1e-5, "fatol": 1e-5})
            if np.isfinite(res2.fun) and -res2.fun >= self.log_prob(th):
                th = res2.x
        chi2 = self.chi2(th)
        return OptResult(th, chi2, self.log_prob(th), len(self.y), self.ndim, time.time() - t_start,
                         self.model.tau_flags(self.params_from_theta(th)[0]))

    # ---- stage 3: MCMC ------------------------------------------------------------------------------
    def local_widths(self, theta, target: float = 0.5, max_iter: int = 12) -> np.ndarray:
        """Per-parameter width of the posterior around `theta` from the curvature of ln P along each axis:
        the step h at which ln P drops by about `target` on average over +h and -h gives
        sigma = h / sqrt(2 * drop).  These are conditional widths (the other parameters held fixed), so
        they underestimate the marginal widths along degeneracies; used to start the walkers at the
        right scale instead of in a fixed fraction of the prior range.  Costs ~2 x max_iter x ndim calls."""
        theta = np.asarray(theta, float)
        span = self.hi - self.lo
        f0 = self.log_prob(theta)
        out = np.empty(self.ndim)
        for i in range(self.ndim):
            h = 1e-3 * span[i]
            drop = np.nan
            for _ in range(max_iter):
                vals = []
                for sgn in (1.0, -1.0):
                    t = theta.copy(); t[i] = theta[i] + sgn * h
                    v = self.log_prob(t) if self.lo[i] < t[i] < self.hi[i] else -np.inf
                    vals.append(v)
                fin = [v for v in vals if np.isfinite(v)]
                drop = f0 - np.mean(fin) if fin else np.inf
                if drop < 0.2 * target and h < 0.1 * span[i]:
                    h *= 3.0
                elif drop > 4.0 * target and h > 1e-6 * span[i]:
                    h /= 2.5
                else:
                    break
            sig = h / np.sqrt(2.0 * drop) if np.isfinite(drop) and drop > 0 else 1e-3 * span[i]
            out[i] = float(np.clip(sig, 1e-5 * span[i], 0.1 * span[i]))
        return out

    def independent_blocks(self, theta=None, frac: float = 0.02) -> list[list[int]]:
        """Indices of the free parameters split into groups that share no pixel.

        A component's support is the set of fitted pixels where its flux at `theta` exceeds `frac` x the
        noise (without `theta`: pixels within ~2 resolution elements of any of its lines, inside its own
        `windows`, which links far more).  Two components are linked when their supports overlap, when
        one is tied to the other, when they share an opacity group, or when `ordering` relates them.
        Where no component of one group is above `frac` sigma, a change of its parameters cannot move
        the other group's likelihood by more than ~frac^2 per pixel, so the likelihood is (to that
        accuracy) a sum over groups and each group can be sampled on its own.  Global parameters (the
        noise scale) go to the group with the most pixels.  Absorbing screens that cover everything make a
        single group."""
        comps = [c for c in self.components if c.enabled]
        names = [c.name for c in comps]
        if any(c.kind == "absorption" and c.covers == "all" for c in comps):
            return [list(range(self.ndim))]
        parent = {n: n for n in names}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            if a in parent and b in parent:
                parent[find(a)] = find(b)
        sup = {}
        if theta is not None:
            P, log_s = self.params_from_theta(np.asarray(theta, float))
            _, per, _ = self.model.evaluate(P, per_unit=True)
            thr = frac * self.sigma * (10.0 ** log_s)
            for key, members in self.model._units().items():
                f = per.get(key)
                m = np.abs(f) > thr if f is not None else np.zeros(len(self.wave), bool)
                for c in members:
                    sup[c.name] = m
        tol = self.wave / 1500.0
        for c in comps:
            if c.name in sup:
                if c.tie_to:
                    union(c.name, c.tie_to)
                continue
            try:
                w = np.sort(self.model.linelist_for(c).wave)
            except KeyError:
                w = np.array([])
            m = np.zeros(len(self.wave), bool)
            if len(w):
                i = np.searchsorted(w, self.wave)
                left = w[np.clip(i - 1, 0, len(w) - 1)]; right = w[np.clip(i, 0, len(w) - 1)]
                m = np.minimum(np.abs(self.wave - left), np.abs(right - self.wave)) < tol
            wm = self.model.window_mask(c)
            if wm is not None:
                m &= wm
            sup[c.name] = m
            if c.tie_to:
                union(c.name, c.tie_to)
        for c in comps:
            if c.group:
                for d in comps:
                    if d.group == c.group:
                        union(c.name, d.name)
        for hot, cold in self.ordering:
            union(hot, cold)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if np.any(sup[a] & sup[b]):
                    union(a, b)
        roots = {}
        for n in names:
            roots.setdefault(find(n), []).append(n)
        groups = list(roots.values())
        npix = [int(np.any([sup[n] for n in g], axis=0).sum()) for g in groups]
        order = np.argsort(npix)[::-1]
        groups = [groups[k] for k in order]
        blocks = [[j for j, p in enumerate(self.free) if p.comp in set(g)] for g in groups]
        glob = [j for j, p in enumerate(self.free) if p.comp == "global"]
        blocks[0] = sorted(blocks[0] + glob)
        return [b for b in blocks if b]

    def initial_walkers(self, theta0, nwalkers, rng, init: str = "ball", ball: float = 1e-2, widths=None):
        """Starting positions: "ball" = theta0 + ball x (prior range) x N(0,1) (the old behaviour);
        "scaled" = theta0 + local posterior width x N(0,1) (see `local_widths`)."""
        theta0 = np.asarray(theta0, float)
        scale = ball * (self.hi - self.lo) if init == "ball" else \
            (self.local_widths(theta0) if widths is None else np.asarray(widths, float))
        eps = 1e-6 * (self.hi - self.lo)
        p0 = np.clip(theta0 + scale * rng.standard_normal((nwalkers, self.ndim)), self.lo + eps, self.hi - eps)
        for i in range(nwalkers):            # every walker inside the prior (ordering constraints) and finite
            k = 0
            while not np.isfinite(self.log_prob(p0[i])) and k < 100:
                f = 0.5 ** (k // 10)
                p0[i] = np.clip(theta0 + f * scale * rng.standard_normal(self.ndim), self.lo + eps, self.hi - eps); k += 1
            if not np.isfinite(self.log_prob(p0[i])):
                p0[i] = theta0
        return p0, scale

    def mcmc(self, theta0, nwalkers: int | None = None, nsteps: int = 2000, processes: int = 1, seed: int = 0,
             ball: float = 1e-2, progress=None, checkpoint: str | None = None, stop_event=None,
             thin_by: int = 1, chunk: int = 50, moves: str = "stretch", init: str = "ball",
             blocks: str = "joint", de_gamma: float = 1.0, linear: str = "sample", linear_prior: str = "log",
             linear_prior_scale: float | None = None, vectorize: bool = False) -> "MCMCResult":
        """emcee run around theta0.

        moves  : "stretch" (emcee default, Goodman & Weare), "de" (80 % DEMove + 20 % DESnookerMove,
                 ter Braak & Vrugt 2008: much shorter autocorrelation times on the correlated 15-30
                 parameter posteriors of multi-molecule fits), or "de+stretch" (60/20/20).
        de_gamma : scale of the DE step relative to emcee's 2.38 / sqrt(2 ndim).
        init   : "ball" (fixed fraction `ball` of the prior range) or "scaled" (local posterior widths).
        blocks : "joint" (one sampler over all parameters) or "auto": groups of components that share no
                 pixel (e.g. CO + ro-vibrational water at 4.9-8 um vs everything at 12-27.5 um) are sampled
                 by separate samplers with the other groups held at theta0. The likelihood is a sum over
                 groups, so this is exact except for the shared noise scale, which is sampled with the
                 largest group and held at theta0 by the others. The chains are merged into one result
                 (walker j of a smaller group is reused for walker j mod n).
        linear : "sample" (the areas are MCMC parameters), "profile" (areas solved by bounded NNLS inside
                 ln L) or "marginalise" (analytic Gaussian marginal over the areas); see `jalebi.linear`.
                 The returned chain always has the full parameter vector (areas filled in per sample).
        vectorize : one ln P call evaluates every walker (`log_prob_many`; emcee vectorize=True, no pool).
        `processes` > 1 uses a process pool with the problem sent once to every worker.
        `progress(fraction, sampler)` is called every `chunk` steps; `stop_event.is_set()` stops early."""
        import emcee
        if linear and linear != "sample":
            from .linear import normalise_mode, run_linear_mcmc
            if normalise_mode(linear) != "sample":
                return run_linear_mcmc(self, theta0, linear, prior=linear_prior, prior_scale=linear_prior_scale,
                                       nwalkers=nwalkers, nsteps=nsteps, processes=processes, seed=seed, ball=ball,
                                       progress=progress, checkpoint=checkpoint, stop_event=stop_event,
                                       thin_by=thin_by, chunk=chunk, moves=moves, init=init, blocks=blocks,
                                       de_gamma=de_gamma, vectorize=vectorize)
        rng = np.random.default_rng(seed)
        theta0 = np.asarray(theta0, float)
        groups = self.independent_blocks(theta0) if blocks == "auto" else [list(range(self.ndim))]
        widths = self.local_widths(theta0) if init == "scaled" else None
        nw_default = lambda n: max(4 * n, 32)
        t_start = time.time()
        lp0 = self.log_prob(theta0)
        results = []
        total_steps = nsteps * len(groups)
        done_all = 0
        for gi, idx in enumerate(groups):
            idx = np.asarray(idx)
            nd = len(idx)
            nw = nwalkers if (nwalkers and len(groups) == 1) else (nwalkers if nwalkers and nwalkers >= 2 * nd else nw_default(nd))
            p0_full, _ = self.initial_walkers(theta0, nw, rng, init=init, ball=ball, widths=widths)
            p0 = p0_full[:, idx]
            backend = None
            if checkpoint:
                backend = emcee.backends.HDFBackend(checkpoint, name="mcmc" if len(groups) == 1 else f"mcmc_block{gi}")
                backend.reset(nw, nd)
            mv = make_moves(moves, nd, de_gamma)
            pool = None
            try:
                if vectorize:
                    fn = _target_many(self) if len(groups) == 1 else _BlockLogProbMany(self, theta0, idx)
                    init_args = None
                elif len(groups) == 1:
                    fn = _logprob_worker if processes > 1 else _target(self)
                    init_args = (_init_worker, (self,))
                else:
                    fn = _block_logprob_worker if processes > 1 else _BlockLogProb(self, theta0, idx)
                    init_args = (_init_block_worker, (self, theta0, idx))
                if processes > 1 and not vectorize:
                    ctx = mp.get_context("fork") if hasattr(os, "fork") else mp.get_context("spawn")
                    pool = ctx.Pool(processes, initializer=init_args[0], initargs=init_args[1])
                sampler = emcee.EnsembleSampler(nw, nd, fn, pool=pool, backend=backend, moves=mv, vectorize=vectorize)
                state = p0
                done = 0
                while done < nsteps:
                    n = min(chunk, nsteps - done)
                    state = sampler.run_mcmc(state, n, progress=False, thin_by=thin_by, skip_initial_state_check=True)
                    done += n
                    if progress:
                        progress((done_all + done) / total_steps, sampler)
                    if stop_event is not None and stop_event.is_set():
                        break
                done_all += done
            finally:
                if pool is not None:
                    pool.close(); pool.join()
            blobs = sampler.get_blobs() if hasattr(self, "log_prob_blob") else None
            results.append((idx, sampler.get_chain(), sampler.get_log_prob(), float(np.mean(sampler.acceptance_fraction)), nw, blobs))
            if stop_event is not None and stop_event.is_set():
                break
        if len(results) == 1 and len(groups) == 1:
            idx, chain, lnp, acc, nw, blobs = results[0]
        else:                                   # merge the groups into one (steps, walkers, ndim) chain
            ns = min(r[1].shape[0] for r in results)
            nw = max(r[4] for r in results)
            chain = np.repeat(np.repeat(theta0[None, None, :], ns, axis=0), nw, axis=1)
            lnp = np.full((ns, nw), -(len(results) - 1) * lp0)
            blobs = None
            if results[0][5] is not None:      # linear modes: each blob column comes from its unit's block
                owner = self.blob_owner(groups)
                blobs = np.full((ns, nw, len(owner)), np.nan)
            accs, wts = [], []
            for gi, (idx, ch, lp_, acc, n_, bl) in enumerate(results):
                take = np.arange(nw) % n_
                chain[:, :, idx] = ch[:ns][:, take, :]
                lnp += lp_[:ns][:, take]
                if blobs is not None:
                    cols = np.flatnonzero(owner == gi)
                    blobs[:, :, cols] = bl[:ns][:, take][:, :, cols]
                accs.append(acc); wts.append(len(idx))
            acc = float(np.average(accs, weights=wts))
        res = MCMCResult(self, chain, lnp, acc, time.time() - t_start)
        res.blobs = blobs
        res.meta = {"moves": moves, "de_gamma": de_gamma, "init": init, "vectorize": bool(vectorize), "blocks": [[self.free[j].key for j in g] for g in groups],
                    "block_acceptance": [r[3] for r in results], "block_nwalkers": [r[4] for r in results]}
        return res


def _target(problem):
    """The function the sampler calls: ln P, or (ln P, blob) for problems that carry per-sample extras
    (the solved areas of `jalebi.linear.LinearProblem`)."""
    return getattr(problem, "log_prob_blob", None) or problem.log_prob


def _target_many(problem):
    """Vectorised counterpart of `_target`: ln P of every walker, or a list of (ln P, blob)."""
    return getattr(problem, "log_prob_blob_many", None) or problem.log_prob_many


class _BlockLogProbMany:
    """Vectorised `_BlockLogProb`."""

    def __init__(self, problem, base, idx):
        self.problem, self.base, self.idx = problem, np.asarray(base, float).copy(), np.asarray(idx)

    def __call__(self, X):
        T = np.repeat(self.base[None, :], len(X), axis=0)
        T[:, self.idx] = X
        return _target_many(self.problem)(T)


def make_moves(name: str, ndim: int | None = None, gamma: float = 1.0):
    """emcee move list for a name (see FitProblem.mcmc).  `gamma` scales the differential-evolution step
    (emcee default 2.38 / sqrt(2 ndim)): below 1 raises the acceptance on curved, non-Gaussian posteriors."""
    import emcee
    name = (name or "stretch").lower()
    if name == "stretch":
        return None
    g0 = None if (ndim is None or gamma == 1.0) else gamma * 2.38 / np.sqrt(2.0 * ndim)
    de = emcee.moves.DEMove(gamma0=g0)
    if name == "de":
        return [(de, 0.8), (emcee.moves.DESnookerMove(), 0.2)]
    if name in ("de+stretch", "mixed"):
        return [(de, 0.6), (emcee.moves.DESnookerMove(), 0.2), (emcee.moves.StretchMove(), 0.2)]
    raise ValueError(f"unknown MCMC moves {name!r}: stretch | de | de+stretch")


class _BlockLogProb:
    """ln P as a function of one group's parameters, the others held at `base`."""

    def __init__(self, problem, base, idx):
        self.problem, self.base, self.idx = problem, np.asarray(base, float).copy(), np.asarray(idx)

    def __call__(self, x):
        t = self.base.copy()
        t[self.idx] = x
        return _target(self.problem)(t)


# module-level hooks for the process pool -------------------------------------------------------------
_PROBLEM: FitProblem | None = None


def _init_worker(problem):
    global _PROBLEM
    _PROBLEM = problem
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)
    except Exception:
        pass


def _logprob_worker(theta):
    return _target(_PROBLEM)(theta)


_BLOCK: "_BlockLogProb | None" = None


def _init_block_worker(problem, base, idx):
    global _BLOCK
    _BLOCK = _BlockLogProb(problem, base, idx)
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)
    except Exception:
        pass


def _block_logprob_worker(x):
    return _BLOCK(x)


_DE_STATE = None


def _init_de_worker(problem, theta0, idx, amask):
    global _PROBLEM, _DE_STATE
    _PROBLEM = problem
    _DE_STATE = (theta0, idx, amask)
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)
    except Exception:
        pass


def _de_worker(sub):
    theta0, idx, amask = _DE_STATE
    return _PROBLEM._de_energy(sub, theta0, idx, amask)


# ------------------------------------------------------------------------------------
# Results
# ------------------------------------------------------------------------------------

@dataclass
class GridResult:
    comp: str
    logN: np.ndarray
    T: np.ndarray
    chi2: np.ndarray          # shape (nT, nlogN)
    logR: np.ndarray
    tau_max: np.ndarray
    npix: int

    @property
    def best(self) -> dict:
        iT, iN = np.unravel_index(np.argmin(self.chi2), self.chi2.shape)
        return {"logN": float(self.logN[iN]), "T": float(self.T[iT]), "logR": float(self.logR[iT, iN]),
                "chi2": float(self.chi2[iT, iN]), "chi2_red": float(self.chi2[iT, iN] / max(self.npix - 3, 1)),
                "tau_max": float(self.tau_max[iT, iN]), "at_edge": bool(iT in (0, len(self.T) - 1) or iN in (0, len(self.logN) - 1))}

    def delta_chi2(self):
        return self.chi2 - self.chi2.min()


@dataclass
class OptResult:
    theta: np.ndarray
    chi2: float
    log_prob: float
    npix: int
    ndim: int
    runtime_s: float
    tau_max: dict

    @property
    def chi2_red(self):
        return self.chi2 / max(self.npix - self.ndim, 1)

    @property
    def bic(self):
        return self.chi2 + self.ndim * np.log(self.npix)


class MCMCResult:
    """Chain container with convergence diagnostics and summaries."""

    def __init__(self, problem: FitProblem, chain: np.ndarray, log_prob: np.ndarray, acceptance: float, runtime_s: float):
        self.problem = problem
        self.chain = chain              # (nsteps, nwalkers, ndim)
        self.log_prob = log_prob        # (nsteps, nwalkers)
        self.acceptance = acceptance
        self.runtime_s = runtime_s
        self.names = [p.key for p in problem.free]
        self.labels = [p.label for p in problem.free]
        self.meta: dict = {}                # sampler settings (moves, init, blocks), written to diagnostics.json
        self.blobs = None                   # per-sample extras of the sampler (linear modes: solved areas)
        self.linear = "sample"              # sample | profile | marginalise (jalebi.linear)
        self.sampled = np.ones(chain.shape[2], bool)   # columns the sampler moved (areas are filled in otherwise)
        self.area_mean = None               # linear modes: NNLS areas / conditional means a = R^2 per sample

    @property
    def nsteps(self):
        return self.chain.shape[0]

    def autocorr_time(self) -> np.ndarray:
        import emcee
        try:
            return emcee.autocorr.integrated_time(self.chain, tol=0)
        except Exception:
            return np.full(self.chain.shape[2], np.nan)

    def burn(self, frac: float = 0.5) -> int:
        tau = self.autocorr_time()
        if np.all(np.isfinite(tau)):
            return int(min(max(2 * np.nanmax(tau), 1), self.nsteps * frac))
        return int(self.nsteps * frac)

    def flat(self, burn: int | None = None, thin: int = 1) -> np.ndarray:
        b = self.burn() if burn is None else burn
        return self.chain[b::thin].reshape(-1, self.chain.shape[2])

    def rhat(self, burn: int | None = None) -> np.ndarray:
        """Split-R-hat treating walkers as chains (Gelman–Rubin, split in two halves)."""
        b = self.burn() if burn is None else burn
        c = self.chain[b:]
        n = c.shape[0] // 2
        if n < 4:
            return np.full(c.shape[2], np.nan)
        halves = np.concatenate([c[:n], c[n:2 * n]], axis=1)      # (n, 2*nwalkers, ndim)
        means = halves.mean(axis=0); vars_ = halves.var(axis=0, ddof=1)
        W = vars_.mean(axis=0)
        B = n * means.var(axis=0, ddof=1)
        var_hat = (n - 1) / n * W + B / n
        return np.sqrt(var_hat / np.where(W > 0, W, np.nan))

    def diagnostics(self) -> dict:
        tau = self.autocorr_time()
        b = self.burn()
        n_eff = (self.nsteps - b) * self.chain.shape[1] / np.where(np.isfinite(tau) & (tau > 0), tau, np.nan)
        ok_len = np.all(self.nsteps > 50 * tau) if np.all(np.isfinite(tau)) else False
        rh = self.rhat(b)
        return {"acceptance": self.acceptance, "tau": tau, "steps_over_tau": self.nsteps / np.nanmax(tau) if np.isfinite(np.nanmax(tau)) else np.nan,
                "converged_length": bool(ok_len), "rhat": rh, "rhat_ok": bool(np.nanmax(rh) < 1.05) if np.any(np.isfinite(rh)) else False,
                "n_eff": n_eff, "burn": b, "acceptance_ok": 0.15 < self.acceptance < 0.6, "runtime_s": self.runtime_s,
                **self.meta}

    def summary(self, burn: int | None = None) -> pd.DataFrame:
        """Median and 16/84 % of every free parameter plus derived log(N·A), N (molecules), R."""
        flat = self.flat(burn)
        q = np.percentile(flat, [16, 50, 84], axis=0)
        rows = []
        for i, p in enumerate(self.problem.free):
            rows.append({"parameter": p.key, "label": p.label, "median": q[1, i], "minus": q[1, i] - q[0, i],
                         "plus": q[2, i] - q[1, i], "lo_bound": p.lo, "hi_bound": p.hi,
                         "at_edge": bool(q[0, i] < p.lo + 0.02 * (p.hi - p.lo) or q[2, i] > p.hi - 0.02 * (p.hi - p.lo))})
        # derived per component
        der = self.derived(flat)
        for k, v in der.items():
            qq = np.percentile(v, [16, 50, 84])
            rows.append({"parameter": k, "label": k, "median": qq[1], "minus": qq[1] - qq[0], "plus": qq[2] - qq[1],
                         "lo_bound": np.nan, "hi_bound": np.nan, "at_edge": False})
        return pd.DataFrame(rows)

    def derived(self, flat: np.ndarray) -> dict[str, np.ndarray]:
        """Derived quantities per sample: logNA [cm^-2 au^2], log N_mol (number of molecules), R [au]."""
        out = {}
        comps = {c.name: c for c in self.problem.components}
        for cname in comps:
            if comps[cname].kind == "absorption":
                continue
            idx = {p.name: i for i, p in enumerate(self.problem.free) if p.comp == cname}
            base = self.problem._base[cname]
            logN = flat[:, idx["logN"]] if "logN" in idx else np.full(len(flat), base["logN"])
            if "logNA" in idx:
                logNA = flat[:, idx["logNA"]]
                logR = logR_from(logNA, logN)
            else:
                logR = flat[:, idx["logR"]] if "logR" in idx else np.full(len(flat), base["logR"])
                logNA = logNA_from(logN, logR)
            if comps[cname].tie_to:
                continue
            if "logNA" not in idx:
                out[f"{cname}.logNA"] = logNA
            out[f"{cname}.logNmol"] = logNA + np.log10(AU_CM2)
            if "logR" not in idx:
                out[f"{cname}.logR"] = logR
            out[f"{cname}.R_au"] = 10.0 ** logR
        return out

    def tau_flag(self, n: int = 200, seed: int = 0) -> dict[str, float]:
        """Fraction of posterior samples with tau_max < 1 per emitting unit."""
        rng = np.random.default_rng(seed)
        flat = self.flat()
        pick = flat[rng.integers(0, len(flat), min(n, len(flat)))]
        counts = {}
        for th in pick:
            P, _ = self.problem.params_from_theta(th)
            for k, t in self.problem.model.tau_flags(P).items():
                counts.setdefault(k, []).append(t < 1.0)
        return {k: float(np.mean(v)) for k, v in counts.items()}

    def correlation(self, burn=None) -> pd.DataFrame:
        flat = self.flat(burn)
        return pd.DataFrame(np.corrcoef(flat.T), index=self.names, columns=self.names)

    def best(self) -> np.ndarray:
        i = np.unravel_index(np.argmax(self.log_prob), self.log_prob.shape)
        return self.chain[i[0], i[1]]

    def median_theta(self, burn=None) -> np.ndarray:
        return np.median(self.flat(burn), axis=0)

    def posterior_predictive(self, n: int = 100, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Median and 16/84 % model flux per pixel over n posterior draws."""
        rng = np.random.default_rng(seed)
        flat = self.flat()
        pick = flat[rng.integers(0, len(flat), min(n, len(flat)))]
        M = np.array([self.problem.model_flux(th) for th in pick])
        return np.percentile(M, 50, axis=0), np.percentile(M, 16, axis=0), np.percentile(M, 84, axis=0)

    def save(self, path: str):
        # layout unchanged since 0.9 (chain = every free parameter, so readers of `chain`/`names` keep working);
        # 0.17 adds `linear` (sample | profile | marginalise) and `sampled` (False for the areas that were
        # profiled / marginalised and filled in per sample) -- old files lack them: treat as sample / all True
        extra = {}
        if self.area_mean is not None:
            extra["area_mean"] = self.area_mean
        np.savez_compressed(path, chain=self.chain, log_prob=self.log_prob, acceptance=self.acceptance,
                            runtime_s=self.runtime_s, names=np.array(self.names), labels=np.array(self.labels),
                            linear=np.array(self.linear), sampled=np.asarray(self.sampled, bool), **extra)


# ------------------------------------------------------------------------------------
# Helpers to build free-parameter lists
# ------------------------------------------------------------------------------------

def default_free_params(components: list[Component], area_param: str = "logR", fit_rv: bool = False,
                        fit_fwhm: bool = False, bounds: dict | None = None) -> list[Param]:
    """logN, T and an area parameter for every enabled, untied component; ratio for tied ones;
    one logR per opacity group.  Absorption screens get logN, T, rv and fc (their velocity is
    always free: it is the point of an absorption fit), plus fwhm with fit_fwhm."""
    bounds = {**DEFAULT_BOUNDS, **(bounds or {})}
    free = []
    seen_groups = set()
    for c in components:
        if not c.enabled:
            continue
        if c.kind == "absorption":
            ab = {**bounds, **ABSORPTION_BOUNDS}
            b = lambda n: bounds.get(f"{c.name}.{n}", ab[n])
        else:
            b = lambda n: bounds.get(f"{c.name}.{n}", bounds[n])
        if c.tie_to:
            free.append(Param(c.name, "ratio", *b("ratio"), init=c.ratio))
            continue
        if c.kind == "absorption":
            lead_of_group = c.group and c.group not in seen_groups
            free.append(Param(c.name, "logN", *b("logN"), init=c.logN))
            if not c.group or lead_of_group:
                free.append(Param(c.name, "T", *b("T"), init=c.T))
                free.append(Param(c.name, "rv", *b("rv"), init=c.rv))
                free.append(Param(c.name, "fc", *b("fc"), init=c.fc))
                if fit_fwhm:
                    free.append(Param(c.name, "fwhm", *b("fwhm"), init=c.fwhm))
            if c.group:
                seen_groups.add(c.group)
            continue
        free.append(Param(c.name, "logN", *b("logN"), init=c.logN))
        if c.kind == "annuli":
            free.append(Param(c.name, "T", *b("T"), init=c.T))
            if c.Tvib is not None:
                free.append(Param(c.name, "Tvib", *b("Tvib"), init=c.Tvib))
            free.append(Param(c.name, "q", *b("q"), init=c.q))
            free.append(Param(c.name, "p", *b("p"), init=c.p))
            free.append(Param(c.name, "logRin", *b("logRin"), init=c.logRin))
            free.append(Param(c.name, "logR", *b("logR"), init=c.logR))
            continue
        lead_of_group = c.group and c.group not in seen_groups
        if not c.group or lead_of_group:
            free.append(Param(c.name, "T", *b("T"), init=c.T))
            if c.Tvib is not None:
                free.append(Param(c.name, "Tvib", *b("Tvib"), init=c.Tvib))
            if area_param == "logNA":
                free.append(Param(c.name, "logNA", *b("logNA"), init=logNA_from(c.logN, c.logR)))
            else:
                free.append(Param(c.name, "logR", *b("logR"), init=c.logR))
            if fit_rv:
                free.append(Param(c.name, "rv", *b("rv"), init=c.rv, gauss=(0.0, 5.0)))
            if fit_fwhm:
                free.append(Param(c.name, "fwhm", *b("fwhm"), init=c.fwhm))
        if c.group:
            seen_groups.add(c.group)
    return free
