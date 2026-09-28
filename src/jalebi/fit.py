"""Fitting engine: one likelihood, one parameter vector, staged inference.

Stages
  1. grid      chi^2 over (log N, T) for one component with its area solved by NNLS
  2. optimise  joint global optimiser (differential evolution, areas by NNLS) + Nelder-Mead polish
  3. mcmc      emcee (parallel over a process pool), convergence diagnostics, summaries

Parameterisation
  per component: logN [cm^-2], T [K], logR [au] (or logNA = log10(N * A[au^2]) when
  area_param="logNA"), optional rv [km/s] and fwhm [km/s]; tied components: ratio.
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
                  "p": (-1.0, 3.0), "logRin": (-2.5, 0.5), "log_s": (-1.0, 1.0)}


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
                 use_pipeline_err: bool = False):
        self.spec = spec
        self.components = components
        self.windows = windows
        self.area_param = area_param
        self.ordering = ordering or []
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
        self.model = build_model(components, self.wave, spec.distance_pc, windows, linelists=linelists,
                                 releases=releases, oversample=oversample, **(model_kwargs or {}))
        self.free = list(free)
        if fit_noise_scale and not any(p.name == "log_s" for p in self.free):
            self.free.append(Param("global", "log_s", *DEFAULT_BOUNDS["log_s"], init=0.0))
        self.ndim = len(self.free)
        self.lo = np.array([p.lo for p in self.free]); self.hi = np.array([p.hi for p in self.free])
        self._base = {c.name: c.params() for c in components}
        self.ncall = 0

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
        if self.ordering:
            P, _ = self.params_from_theta(theta)
            for hot, cold in self.ordering:
                if P[hot]["T"] <= P[cold]["T"]:
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

    # ---- detection test ------------------------------------------------------------------
    def component_significance(self, theta) -> pd.DataFrame:
        """For every independent unit: chi2 increase when it is removed (logN -> -inf) and the
        corresponding delta-BIC (positive = the component is supported by the data)."""
        P, log_s = self.params_from_theta(theta)
        chi2_full = self.chi2(theta)
        n = len(self.y)
        rows = []
        for unit, lead in self.area_units().items():
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
    def area_units(self) -> dict[str, str]:
        """unit key (component or group) -> name of the leading component."""
        units = {}
        for c in self.components:
            if c.enabled and not c.tie_to:
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
        for c in self.components:
            if c.group and c.name != self.area_units().get(c.group):
                P[c.name]["logR"] = P[self.area_units()[c.group]]["logR"]
        return P, chi2

    def area_free_mask(self) -> np.ndarray:
        """Free parameters that are emitting areas of slab units (profiled by NNLS in the grid and the
        optimiser).  The logR of an annuli component is its outer radius, not a linear scale, so it stays
        an ordinary parameter."""
        kind = {c.name: c.kind for c in self.components}
        return np.array([p.name in ("logR", "logNA") and kind.get(p.comp) != "annuli" for p in self.free])

    # ---- stage 1: grid --------------------------------------------------------------------
    def grid(self, comp: str, logN=None, T=None, windows=None, n_jobs: int = 1, base_theta=None,
             progress=None) -> "GridResult":
        """chi^2 map over (logN, T) for component `comp`, its area by NNLS, other components at
        their current values.  Optional `windows` restricts the pixels used."""
        logN = np.linspace(14.0, 20.0, 31) if logN is None else np.asarray(logN)
        T = np.linspace(150.0, 1200.0, 22) if T is None else np.asarray(T)
        P0, _ = self.params_from_theta(self.theta0() if base_theta is None else base_theta)
        sel = np.ones(len(self.wave), bool) if windows is None else in_ranges(self.wave, windows)
        units = self.area_units()
        cobj = next(c for c in self.components if c.name == comp)
        my_unit = cobj.group or comp
        others = set(units) - {my_unit}

        annuli = cobj.kind == "annuli"

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
            if annuli:                                # outer radius is not a linear scale: keep it
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
    def mcmc(self, theta0, nwalkers: int | None = None, nsteps: int = 2000, processes: int = 1, seed: int = 0,
             ball: float = 1e-2, progress=None, checkpoint: str | None = None, stop_event=None,
             thin_by: int = 1, chunk: int = 50) -> "MCMCResult":
        """emcee run started in a small ball around theta0.  `processes` > 1 uses a process pool with
        the problem sent once to every worker.  `progress(fraction, sampler)` is called every `chunk`
        steps; `stop_event.is_set()` stops early."""
        import emcee
        rng = np.random.default_rng(seed)
        nwalkers = nwalkers or max(4 * self.ndim, 32)
        theta0 = np.asarray(theta0, float)
        p0 = theta0 + ball * (self.hi - self.lo) * rng.standard_normal((nwalkers, self.ndim))
        p0 = np.clip(p0, self.lo + 1e-6 * (self.hi - self.lo), self.hi - 1e-6 * (self.hi - self.lo))
        # make sure every walker starts inside the prior (ordering constraints)
        for i in range(nwalkers):
            k = 0
            while not np.isfinite(self.log_prior(p0[i])) and k < 100:
                p0[i] = theta0 + ball * (self.hi - self.lo) * rng.standard_normal(self.ndim)
                p0[i] = np.clip(p0[i], self.lo + 1e-6, self.hi - 1e-6); k += 1
        backend = None
        if checkpoint:
            backend = emcee.backends.HDFBackend(checkpoint)
            backend.reset(nwalkers, self.ndim)
        t_start = time.time()
        pool = None
        try:
            if processes > 1:
                ctx = mp.get_context("fork") if hasattr(os, "fork") else mp.get_context("spawn")
                pool = ctx.Pool(processes, initializer=_init_worker, initargs=(self,))
                sampler = emcee.EnsembleSampler(nwalkers, self.ndim, _logprob_worker, pool=pool, backend=backend)
            else:
                sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.log_prob, backend=backend)
            state = p0
            done = 0
            while done < nsteps:
                n = min(chunk, nsteps - done)
                state = sampler.run_mcmc(state, n, progress=False, thin_by=thin_by, skip_initial_state_check=True)
                done += n
                if progress:
                    progress(done / nsteps, sampler)
                if stop_event is not None and stop_event.is_set():
                    break
        finally:
            if pool is not None:
                pool.close(); pool.join()
        chain = sampler.get_chain()
        lnp = sampler.get_log_prob()
        return MCMCResult(self, chain, lnp, float(np.mean(sampler.acceptance_fraction)), time.time() - t_start)


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
    return _PROBLEM.log_prob(theta)


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
                "n_eff": n_eff, "burn": b, "acceptance_ok": 0.15 < self.acceptance < 0.6, "runtime_s": self.runtime_s}

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
        np.savez_compressed(path, chain=self.chain, log_prob=self.log_prob, acceptance=self.acceptance,
                            runtime_s=self.runtime_s, names=np.array(self.names), labels=np.array(self.labels))


# ------------------------------------------------------------------------------------
# Helpers to build free-parameter lists
# ------------------------------------------------------------------------------------

def default_free_params(components: list[Component], area_param: str = "logR", fit_rv: bool = False,
                        fit_fwhm: bool = False, bounds: dict | None = None) -> list[Param]:
    """logN, T and an area parameter for every enabled, untied component; ratio for tied ones;
    one logR per opacity group."""
    bounds = {**DEFAULT_BOUNDS, **(bounds or {})}
    free = []
    seen_groups = set()
    for c in components:
        if not c.enabled:
            continue
        b = lambda n: bounds.get(f"{c.name}.{n}", bounds[n])
        if c.tie_to:
            free.append(Param(c.name, "ratio", *b("ratio"), init=c.ratio))
            continue
        free.append(Param(c.name, "logN", *b("logN"), init=c.logN))
        if c.kind == "annuli":
            free.append(Param(c.name, "T", *b("T"), init=c.T))
            free.append(Param(c.name, "q", *b("q"), init=c.q))
            free.append(Param(c.name, "p", *b("p"), init=c.p))
            free.append(Param(c.name, "logRin", *b("logRin"), init=c.logRin))
            free.append(Param(c.name, "logR", *b("logR"), init=c.logR))
            continue
        lead_of_group = c.group and c.group not in seen_groups
        if not c.group or lead_of_group:
            free.append(Param(c.name, "T", *b("T"), init=c.T))
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
