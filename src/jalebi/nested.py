"""Nested sampling with dynesty: `fit.sampler: dynesty` (default `emcee`).

The same likelihood as the MCMC: every free parameter (`linear: sample`) or the nonlinear ones with the emitting
areas profiled by NNLS (`linear: profile`, 0.17), on whatever model backend is attached (exact or the 0.18
emulator).  The output is an MCMCResult with equal-weight posterior samples in the usual chain.npz layout
(steps x walkers x every free parameter), so the summary, the corner plots and the QA notebook are unchanged;
ln Z and its error go to diagnostics.json.

Prior transform (unit cube -> parameters)
-----------------------------------------
* uniform priors: lo + u (hi - lo);
* Gaussian priors on top of the bounds (`priors:`): the inverse CDF of the truncated normal;
* temperature ordering (`fit.ordering`, e.g. hot > warm > cold): a **sorted transform**.  For each chain of
  ordered components, the k uniform deviates are sorted (descending) and mapped to the chain's common range
  [min lo, max hi].  Order statistics of k uniforms are uniform on the ordered simplex, so the prior is exactly
  the uniform prior restricted to T_1 > T_2 > ... > T_k that emcee uses (density k!/range^k).  When the
  components' own bounds differ, a point outside one of them gets ln L = -inf (rejection), which keeps the prior
  uniform over the allowed region at the cost of some volume.  Orderings that are not chains (e.g. hot > warm
  and hot > cold only), and T_vib <= T, are handled by rejection alone.  (Stick-breaking, T_k ~ U(lo, T_{k-1}),
  was not used: it is not uniform on the ordered region and would change the evidence.)

Sampling method
---------------
Dynamic nested sampling (Higson et al. 2019) with `bound: multi` (multi-ellipsoid) and `sample: rslice`
(random slice sampling, Handley et al. 2015) by default.  For the 6-15 dimensional, strongly correlated and
curved (log N - T - R) posteriors of slab fits, slice sampling needs no tuning of a step size and its cost
grows linearly with dimension (slices = 3 + d by default here), whereas `rwalk` needs ~20-25 steps per new point
to decorrelate and degrades with the correlations; `unif` (dynesty's choice below 10 dimensions) is inefficient
once the ellipsoids no longer fit the banana-shaped thick-slab posteriors.  `sample: rwalk` remains available.
Parallel: `processes > 1` uses dynesty's own process pool (likelihood and prior transform shipped once).
"""
from __future__ import annotations

import os
import time

import numpy as np

from .fit import MCMCResult

DYNESTY_SAMPLES = ("rslice", "rwalk", "slice", "unif", "auto")


# --------------------------------------------------------------------------------------------------
# Prior transform
# --------------------------------------------------------------------------------------------------

def _ordering_chains(problem) -> tuple[list[list[int]], list[tuple[str, str]]]:
    """Chains of T parameter indices (hottest first) from the ordering pairs; pairs that do not form a simple
    chain are returned separately (handled by rejection)."""
    idx = {p.comp: i for i, p in enumerate(problem.free) if p.name == "T"}
    pairs = [(a, b) for a, b in getattr(problem, "ordering", []) if a in idx and b in idx]
    succ, pred = {}, {}
    ok = True
    for a, b in pairs:
        if a in succ or b in pred:
            ok = False
        succ[a] = b; pred[b] = a
    if not ok or not pairs:
        return [], pairs if not ok else []
    chains = []
    for start in [a for a in succ if a not in pred]:
        c, x = [start], start
        while x in succ:
            x = succ[x]
            if x in c:                      # a cycle: impossible ordering
                return [], pairs
            c.append(x)
        chains.append([idx[n] for n in c])
    return chains, []


class PriorTransform:
    """Unit cube -> parameter vector for a FitProblem (or LinearProblem); see the module docstring."""

    def __init__(self, problem):
        self.lo = np.asarray(problem.lo, float).copy()
        self.hi = np.asarray(problem.hi, float).copy()
        self.n = len(self.lo)
        self.gauss = {}
        for i, p in enumerate(problem.free):
            if p.gauss is not None:
                self.gauss[i] = (float(p.gauss[0]), float(p.gauss[1]))
        self.chains, self.reject_pairs = _ordering_chains(problem)
        for c in self.chains:
            for i in c:
                self.gauss.pop(i, None)     # an ordered T with a Gaussian prior: the sort wins (documented)
        self._chain_rng = [(min(self.lo[c]), max(self.hi[c])) for c in self.chains]
        self._in_chain = {i for c in self.chains for i in c}
        if self.gauss:
            from scipy.stats import truncnorm
            self._tn = {i: truncnorm((self.lo[i] - m) / s, (self.hi[i] - m) / s, loc=m, scale=s)
                        for i, (m, s) in self.gauss.items()}
        else:
            self._tn = {}

    def describe(self) -> str:
        parts = []
        if self.chains:
            parts.append("sorted transform for ordered T chains (" + "; ".join(str(len(c)) for c in self.chains) + " components)")
        if self.reject_pairs:
            parts.append(f"{len(self.reject_pairs)} ordering pair(s) by rejection")
        if self._tn:
            parts.append(f"{len(self._tn)} truncated-Gaussian prior(s)")
        return ", ".join(parts) or "uniform"

    def __call__(self, u):
        u = np.asarray(u, float)
        x = self.lo + u * (self.hi - self.lo)
        for i, tn in self._tn.items():
            x[i] = float(tn.ppf(np.clip(u[i], 1e-12, 1 - 1e-12)))
        for c, (a, b) in zip(self.chains, self._chain_rng):
            v = np.sort(u[c])[::-1]
            x[c] = a + v * (b - a)
        return x


# --------------------------------------------------------------------------------------------------
# Likelihood
# --------------------------------------------------------------------------------------------------

class NestedLikelihood:
    """ln L(theta) for dynesty: bounds and constraints (ordering, T_vib <= T) by rejection, no prior density
    (the transform carries it).  With a LinearProblem (profile) the NNLS areas are returned as the blob."""
    BAD = -1e300

    def __init__(self, problem):
        self.problem = problem
        self.profile = hasattr(problem, "log_prob_blob")
        self.k = getattr(problem, "k", 0)
        self.lo, self.hi = np.asarray(problem.lo, float), np.asarray(problem.hi, float)

    def _allowed(self, theta) -> bool:
        if np.any(theta < self.lo) or np.any(theta > self.hi):
            return False
        g = {}
        for p in self.problem.free:                 # log_prior without its Gaussian terms: only the constraints
            if p.gauss is not None:
                g[p] = p.gauss; p.gauss = None
        try:
            return bool(np.isfinite(self.problem.log_prior(theta)))
        finally:
            for p, v in g.items():
                p.gauss = v

    def __call__(self, theta):
        theta = np.asarray(theta, float)
        if not self._allowed(theta):
            return (self.BAD, np.full(self.k, np.nan)) if self.profile else self.BAD
        if self.profile:
            s = self.problem.solve(theta)
            ll = s["lnL"] if np.isfinite(s["lnL"]) else self.BAD
            return ll, np.asarray(s["a"], float)
        ll = self.problem.log_like(theta)
        return ll if np.isfinite(ll) else self.BAD


# --------------------------------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------------------------------

class NestedResult(MCMCResult):
    """Equal-weight posterior samples arranged as (rows, pseudo-walkers, ndim) plus the evidence."""

    def __init__(self, problem, chain, log_prob, runtime_s, logz, logzerr, ncall, niter, n_eff, info):
        super().__init__(problem, chain, log_prob, acceptance=float("nan"), runtime_s=runtime_s)
        self.logz, self.logzerr = float(logz), float(logzerr)
        self.ncall, self.niter, self.n_effective = int(ncall), int(niter), float(n_eff)
        self.info = info

    def burn(self, frac: float = 0.5) -> int:      # independent samples: nothing to burn
        return 0

    def diagnostics(self) -> dict:
        d = super().diagnostics()
        d.update({"sampler": "dynesty", "logz": self.logz, "logzerr": self.logzerr, "likelihood_calls": self.ncall,
                  "iterations": self.niter, "n_effective": self.n_effective, "acceptance_ok": True,
                  "converged_length": True, **self.info})
        return d

    def save(self, path: str):
        super().save(path)
        z = dict(np.load(path, allow_pickle=False))
        z.update({"sampler": np.array("dynesty"), "logz": np.array(self.logz), "logzerr": np.array(self.logzerr)})
        np.savez_compressed(path, **z)


def _systematic(weights, rng):
    w = np.asarray(weights, float); w = w / w.sum()
    n = len(w)
    pos = (rng.random() + np.arange(n)) / n
    return np.minimum(np.searchsorted(np.cumsum(w), pos), n - 1)


# --------------------------------------------------------------------------------------------------
# Running it
# --------------------------------------------------------------------------------------------------

def run_dynesty(problem, theta0=None, linear: str = "sample", nlive: int = 500, sample: str = "rslice",
                bound: str = "multi", dlogz_init: float = 0.5, n_effective: int | None = None, maxcall: int | None = None,
                processes: int = 1, seed: int = 0, pfrac: float = 0.8, slices: int | None = None, walks: int | None = None,
                n_walkers_out: int = 32, progress=None, dynamic: bool = True, checkpoint: str | None = None,
                resume: bool = False, checkpoint_every: float = 60.0) -> NestedResult:
    """Dynamic nested sampling of `problem` (FitProblem).  `linear="profile"` samples the nonlinear parameters
    with the areas solved by NNLS (they come back per sample from the blobs).  Returns a NestedResult over the
    *full* parameter vector (same columns as an emcee run).

    checkpoint (0.22): dynesty's own save file (written every `checkpoint_every` seconds); with `resume` an
    existing file is restored and the run continues from it (fit.resume auto).  The sampler state pickles the
    likelihood, i.e. the whole problem, so the file can be large with emulator tables attached."""
    import dynesty
    t0 = time.time()
    lp = None
    target = problem
    if linear and linear != "sample":
        from .linear import LinearProblem
        th = problem.theta0() if theta0 is None else np.asarray(theta0, float)
        lp = LinearProblem(problem, "profile", th)
        target = lp
    ptf = PriorTransform(target)
    like = NestedLikelihood(target)
    nd = target.ndim
    if nd == 0:
        raise ValueError("nothing to sample: no free nonlinear parameter")
    if sample not in DYNESTY_SAMPLES:
        raise ValueError(f"unknown dynesty sample method {sample!r}: {', '.join(DYNESTY_SAMPLES)}")
    rng = np.random.default_rng(seed)
    kw = dict(bound=bound, sample=sample, rstate=rng, blob=lp is not None)
    if sample in ("rslice", "slice"):
        kw["slices"] = slices or (3 + nd)
    if sample == "rwalk":
        kw["walks"] = walks or 25
    pool = None
    if processes and processes > 1:
        from dynesty.pool import Pool
        pool = Pool(processes, like, ptf)
        pool.__enter__()
        kw.update(pool=pool, queue_size=processes)
        loglike, prior = pool.loglike, pool.prior_transform
    else:
        loglike, prior = like, ptf
    last = {"t": time.time()}

    def pf(results, niter, ncall, **kws):            # dynesty's print_func: a line every 10 s
        if progress is not None and time.time() - last["t"] > 10:
            last["t"] = time.time()
            progress(f"  dynesty: iteration {niter}, {ncall:,} likelihood calls")
    restored = False
    ck = {}
    if checkpoint:
        ck = {"checkpoint_file": checkpoint, "checkpoint_every": checkpoint_every}
    try:
        cls = dynesty.DynamicNestedSampler if dynamic else dynesty.NestedSampler
        s = None
        if checkpoint and resume and os.path.exists(checkpoint):
            try:
                s = cls.restore(checkpoint, pool=pool)
                restored = True
                if progress is not None:
                    progress(f"[stage] dynesty resumed from {os.path.basename(checkpoint)}")
            except Exception as e:                       # a foreign or broken save file: start over
                if progress is not None:
                    progress(f"[stage] dynesty: could not restore {os.path.basename(checkpoint)} ({e}); starting over")
                s = None
        if s is None:
            s = cls(loglike, prior, nd, nlive=nlive, **kw)
        if dynamic:
            s.run_nested(nlive_init=nlive, dlogz_init=dlogz_init, wt_kwargs={"pfrac": pfrac}, n_effective=n_effective,
                         maxcall=maxcall, print_progress=progress is not None, print_func=pf if progress else None,
                         resume=restored, **ck)
        else:
            s.run_nested(dlogz=dlogz_init, maxcall=maxcall, print_progress=progress is not None,
                         print_func=pf if progress else None, resume=restored, **ck)
        r = s.results
    finally:
        if pool is not None:
            pool.__exit__(None, None, None)
    w = np.exp(r["logwt"] - r["logz"][-1])
    n_eff = float(w.sum() ** 2 / np.sum(w ** 2))
    idx = _systematic(w, rng)
    rng.shuffle(idx)
    X = np.asarray(r["samples"])[idx]
    logl = np.asarray(r["logl"])[idx]
    nw = max(2, min(n_walkers_out, len(X) // 4))
    ns = len(X) // nw
    X, logl = X[:ns * nw], logl[:ns * nw]
    if lp is not None:
        A = np.asarray(r["blob"])[idx][:ns * nw]
        full = np.array([lp.full_theta(x, a) for x, a in zip(X, A)])
        lpri = np.array([lp.log_prior(x) for x in X])
    else:
        full = X
        lpri = np.array([problem.log_prior(x) for x in X])
    lnp = np.where(np.isfinite(lpri), logl + lpri, -np.inf)
    chain = full.reshape(ns, nw, -1)
    ncall = int(np.sum(r["ncall"]))
    info = {"sample": sample, "bound": bound, "nlive": nlive, "dynamic": dynamic, "dlogz_init": dlogz_init,
            "pfrac": pfrac, "slices": kw.get("slices"), "walks": kw.get("walks"), "processes": processes, "seed": seed,
            "linear": "profile" if lp is not None else "sample", "prior_transform": ptf.describe(),
            "sampled": [p.key for p in target.free], "efficiency": float(r["eff"]),
            "n_equal_weight": int(ns * nw), "resumed": bool(restored)}
    res = NestedResult(problem, chain, lnp.reshape(ns, nw), time.time() - t0, r["logz"][-1], r["logzerr"][-1],
                       ncall, r["niter"], n_eff, info)
    if lp is not None:
        res.linear = "profile"
        res.sampled = np.zeros(problem.ndim, bool); res.sampled[lp.keep_idx] = True
    res.meta = dict(info)
    return res


def problem_without(problem, name: str):
    """The same fit problem with one component (and the isotopologues tied to it) removed, on exactly the same
    pixels, noise and weights -- an evidence ratio is only meaningful on the same data.  (Rebuilding from the config
    is not enough: with `line_regions` the fit windows depend on which molecules are in the model.)"""
    from .fit import FitProblem
    drop = {name} | {c.name for c in problem.components if c.tie_to == name}
    grp = next((c.group for c in problem.components if c.name == name and c.group), None)
    if grp:
        drop |= {c.name for c in problem.components if c.group == grp}
    comps = [c for c in problem.components if c.name not in drop]
    if len(comps) == len(problem.components):
        raise ValueError(f"no component named {name!r}")
    free = [p for p in problem.free if p.comp not in drop]
    m = problem.model
    p2 = FitProblem(problem.spec, comps, problem.windows, free, area_param=problem.area_param,
                    fit_noise_scale=False, ordering=[o for o in problem.ordering if not (set(o) & drop)],
                    oversample=m.oversample, linelists=m.linelists, releases=m.releases, use_pipeline_err=True,
                    tvib_below_trot=problem.tvib_below_trot,
                    model_kwargs={"R_model": m.R_model, "R_scale": m.R_scale, "R_constant": m.R_constant})
    if not np.array_equal(p2.used, problem.used):
        raise RuntimeError("the reduced problem selected other pixels")
    p2.sigma = problem.sigma.copy(); p2.weights = problem.weights.copy(); p2.y = problem.y.copy()
    p2.fit_noise_scale = problem.fit_noise_scale
    return p2


def evidence_without(problem_factory, components: list[str], full_result: NestedResult, run_kwargs: dict,
                     significance=None, say=None) -> "pd.DataFrame":
    """Delta ln Z of removing each listed component (and its tied isotopologues): ln Z(full) - ln Z(without), as in
    Kaeufer et al. (2024) for Sz 28.  `problem_factory(name)` builds the reduced FitProblem on the same pixels
    (normally `lambda n: problem_without(problem, n)`);
    `significance` (FitProblem.component_significance at the optimum) supplies the Delta BIC column.
    The evidence includes the Occam factor of the removed parameters' prior ranges, so it depends on the bounds."""
    import pandas as pd
    say = say or (lambda m: None)
    rows = []
    for name in components:
        prob = problem_factory(name)
        say(f"evidence: refitting without {name} ({prob.ndim} free parameters)")
        r = run_dynesty(prob, **run_kwargs)
        d = full_result.logz - r.logz
        err = float(np.hypot(full_result.logzerr, r.logzerr))
        row = {"component": name, "lnZ_full": full_result.logz, "lnZ_without": r.logz, "delta_lnZ": d, "delta_lnZ_err": err,
               "likelihood_calls": r.ncall, "runtime_s": r.runtime_s}
        if significance is not None:
            m = significance[significance.component == name]
            row["delta_BIC"] = float(m.delta_BIC.iloc[0]) if len(m) else np.nan
            row["delta_lnZ_from_BIC"] = row["delta_BIC"] / 2.0 if len(m) else np.nan      # BIC ~ -2 ln Z
        rows.append(row)
        say(f"  Delta ln Z({name}) = {d:.1f} +- {err:.1f}")
    return pd.DataFrame(rows)
