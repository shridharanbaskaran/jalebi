"""Quick uncertainties without MCMC: the Laplace (Gaussian) approximation at the optimum.

    lap = laplace(problem, theta_best)          # also available as jalebi.fit.laplace
    lap.summary()                               # value ± sigma, flags, derived R / N·A / N_mol
    lap.save("laplace.json"); plots.plot_laplace_corner(lap, mcmc=run.mcmc)

Method
------
H = Hessian of -ln P at the optimum by central finite differences, with per-parameter steps from
`FitProblem.local_widths` (the conditional posterior width: ln P drops by ~0.5 over one step), refined by one
Richardson extrapolation (steps h and h/2: H = (4 H(h/2) - H(h)) / 3, error O(h^4)); or numdifftools when
`method="numdifftools"` and it is installed.  A parameter closer than 2 steps to a prior bound gets its stencil
moved inside the prior (flagged "one-sided").

The Hessian is diagonalised in prior-span units (u_i = theta_i / span_i, where a uniform prior has precision 12).
Eigen-directions with precision below 12 are less constrained by the data than by the prior ("flat"), and
negative ones are not a maximum ("saddle").  Both are regularised to the prior precision: the covariance is
never wider than the prior allows.  The condition number is that of H in these units, before regularisation.

Flags per parameter: `flat` (it loads > 30 % on a flat or saddle direction; the direction is named, e.g. a CO T
ridge), `unconstrained` (sigma > 25 % of the prior span), `edge` (the optimum is within 2 sigma of a bound),
`one-sided` (the stencil had to move away from a bound).

Derived quantities (log N·A, N_mol, R, and the logR / logNA counterpart) come from 4000 draws of the Gaussian,
truncated to the prior.  With `linear="profile"` (0.17) the Hessian is that of the profile likelihood over
the nonlinear parameters only (fewer dimensions, the area-degeneracy already solved), and each draw gets its
areas from the conditional Gaussian of the linear solve, so the area errors include both parts.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PRIOR_PRECISION = 12.0          # 1 / variance of a uniform prior on a unit interval


def _batch(problem):
    """Vectorised -ln P if the problem has one, else a loop."""
    many = getattr(problem, "log_prob_many", None)
    if many is not None and not hasattr(problem, "log_prob_blob"):
        return lambda X: -np.asarray(many(X), float)
    return lambda X: -np.array([problem.log_prob(x) for x in X], float)


def _hessian_fd(f, c, h):
    """Central-difference Hessian of f (vectorised over rows) at c with steps h."""
    n = len(c)
    pts = [c]
    for i in range(n):
        for s in (1, -1):
            x = c.copy(); x[i] += s * h[i]; pts.append(x)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for i, j in pairs:
        for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            x = c.copy(); x[i] += si * h[i]; x[j] += sj * h[j]; pts.append(x)
    F = f(np.array(pts))
    f0 = F[0]
    H = np.empty((n, n))
    for i in range(n):
        H[i, i] = (F[1 + 2 * i] - 2 * f0 + F[2 + 2 * i]) / h[i] ** 2
    k = 1 + 2 * n
    for i, j in pairs:
        fpp, fpm, fmp, fmm = F[k:k + 4]; k += 4
        H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4 * h[i] * h[j])
    return H, len(pts), bool(np.all(np.isfinite(F)))


@dataclass
class LaplaceResult:
    names: list
    theta: np.ndarray
    cov: np.ndarray
    sigma: np.ndarray
    corr: np.ndarray
    hessian: np.ndarray
    condition: float
    eigvals: np.ndarray                  # precision of each eigen-direction, prior-span units (prior = 12)
    eigvecs: np.ndarray
    flags: dict                          # parameter -> list of flags
    directions: list                     # flat / saddle directions: {"precision", "kind", "loads"}
    lo: np.ndarray
    hi: np.ndarray
    samples: np.ndarray                  # Gaussian draws inside the prior (full parameter vectors)
    derived: dict                        # name -> (median, minus, plus)
    n_evals: int
    runtime_s: float
    method: str
    linear: str = "sample"
    meta: dict = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        rows = []
        for i, n in enumerate(self.names):
            rows.append({"parameter": n, "value": self.theta[i], "sigma": self.sigma[i], "minus": self.sigma[i],
                         "plus": self.sigma[i], "lo_bound": self.lo[i], "hi_bound": self.hi[i],
                         "flags": ", ".join(self.flags.get(n, []))})
        for n, (m, a, b) in self.derived.items():
            rows.append({"parameter": n, "value": m, "sigma": 0.5 * (a + b), "minus": a, "plus": b,
                         "lo_bound": np.nan, "hi_bound": np.nan, "flags": "derived (Gaussian draws)"})
        return pd.DataFrame(rows)

    @property
    def ok(self) -> bool:
        return not self.directions

    def to_dict(self) -> dict:
        return {"method": self.method, "linear": self.linear, "names": self.names, "theta": self.theta.tolist(),
                "sigma": self.sigma.tolist(), "cov": self.cov.tolist(), "corr": self.corr.tolist(),
                "condition_number": self.condition, "eigen_precision_prior_units": self.eigvals.tolist(),
                "flags": self.flags, "flat_directions": self.directions,
                "derived": {k: {"median": v[0], "minus": v[1], "plus": v[2]} for k, v in self.derived.items()},
                "n_evaluations": self.n_evals, "runtime_s": self.runtime_s, **self.meta}

    def save(self, path: str):
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=1, default=float)


def laplace(problem, theta, linear: str | None = None, method: str = "richardson", rel_step: float = 1.0,
            nsamples: int = 4000, seed: int = 0, widths=None, progress=None) -> LaplaceResult:
    """Gaussian approximation of the posterior at `theta` (normally the optimiser's result); see the module
    docstring.  linear: "sample" (Hessian over every free parameter, the default) or "profile" (over the
    nonlinear ones, areas from the linear solve per draw)."""
    from .fit import MCMCResult
    t0 = time.time()
    linear = linear or "sample"
    base = problem
    theta_full = np.clip(np.asarray(theta, float), problem.lo, problem.hi)
    lp = None
    if linear in ("profile", "marginalise", "marginalize"):
        from .linear import LinearProblem
        lp = LinearProblem(problem, "profile", theta_full)
        problem = lp
        theta = lp.reduce(theta_full)
        linear = "profile"
    else:
        theta = theta_full
    n = problem.ndim
    names = [p.key for p in problem.free]
    lo, hi = np.asarray(problem.lo, float), np.asarray(problem.hi, float)
    span = hi - lo
    f = _batch(problem)
    say = progress or (lambda m: None)
    if not np.isfinite(f(theta[None, :])[0]):
        raise ValueError("ln P is not finite at the given theta")
    w = np.asarray(problem.local_widths(theta) if widths is None else widths, float)
    h = np.clip(rel_step * w, 1e-6 * span, 0.1 * span)
    # keep the stencil inside the prior: shift the centre inward where needed
    c = theta.copy()
    shifted = np.zeros(n, bool)
    for i in range(n):
        lo_ok, hi_ok = lo[i] + 1.05 * h[i], hi[i] - 1.05 * h[i]
        if lo_ok >= hi_ok:
            h[i] = 0.2 * span[i]; lo_ok, hi_ok = lo[i] + 1.05 * h[i], hi[i] - 1.05 * h[i]
        if c[i] < lo_ok or c[i] > hi_ok:
            c[i] = np.clip(c[i], lo_ok, hi_ok); shifted[i] = True
    n_evals = int(problem.ncall) if hasattr(problem, "ncall") else 0
    if method == "numdifftools":
        try:
            import numdifftools as nd
            fx = lambda x: float(f(np.asarray(x)[None, :])[0])
            H = nd.Hessian(fx, step=h)(c)
            method_used = "numdifftools"
        except ImportError:
            method = "richardson"
    if method != "numdifftools":
        say(f"Hessian of -ln P: {n} parameters, Richardson extrapolation (2 x {1 + 2 * n + 2 * n * (n - 1)} evaluations)")
        H1, _, ok1 = _hessian_fd(f, c, h)
        H2, _, ok2 = _hessian_fd(f, c, 0.5 * h)
        H = (4.0 * H2 - H1) / 3.0 if (ok1 and ok2) else (H2 if ok2 else H1)
        if not (ok1 and ok2):
            say("  some stencil points have ln P = -inf (ordering prior?): using the finite estimate")
        method_used = "richardson" if (ok1 and ok2) else "central"
    H = 0.5 * (H + H.T)
    H = np.where(np.isfinite(H), H, 0.0)
    # prior-span units
    Hs = H * np.outer(span, span)
    ev, V = np.linalg.eigh(Hs)
    pos = ev[ev > 0]
    cond = float(ev.max() / ev.min()) if ev.min() > 0 else float("inf")
    directions = []
    for k in range(n):
        if ev[k] < PRIOR_PRECISION:
            load = V[:, k] ** 2
            top = np.argsort(load)[::-1][:3]
            directions.append({"precision": float(ev[k]), "kind": "saddle" if ev[k] < 0 else "flat",
                               "loads": {names[j]: round(float(load[j]), 3) for j in top if load[j] > 0.05}})
    ev_reg = np.maximum(ev, PRIOR_PRECISION)
    cov_s = (V / ev_reg) @ V.T
    cov = cov_s * np.outer(span, span)
    sigma = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sigma, sigma)
    flags = {nm: [] for nm in names}
    for d in directions:
        for nm, ld in d["loads"].items():
            if ld > 0.3:
                flags[nm].append(f"{d['kind']} ({' + '.join(d['loads'])})")
    for i, nm in enumerate(names):
        if sigma[i] > 0.25 * span[i]:
            flags[nm].append("unconstrained")
        if theta[i] - lo[i] < 2 * sigma[i] or hi[i] - theta[i] < 2 * sigma[i]:
            flags[nm].append("edge")
        if shifted[i]:
            flags[nm].append("one-sided")
    # Gaussian draws inside the prior -> derived quantities
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(cov + 1e-300 * np.eye(n)) if n else np.zeros((0, 0))
    X = theta[None, :] + rng.standard_normal((max(nsamples, 1) * 3, n)) @ L.T
    inside = np.all((X >= lo) & (X <= hi), axis=1)
    if hasattr(problem, "ordering") and (problem.ordering or getattr(problem, "_tvib_comps", None)):
        inside &= np.array([np.isfinite(problem.log_prior(x)) if ok else False for x, ok in zip(X, inside)])
    X = X[inside][:nsamples]
    kept = float(inside.mean())
    if lp is not None:
        from .linear import LinearProblem
        lm = LinearProblem(base, "marginalise", theta_full, prior="log")
        rows = []
        for x in X[: min(len(X), 1000)]:
            s = lm.solve(x, draw_rng=rng)
            rows.append(lm.full_theta(x, s["draw"]))
        full = np.array(rows) if rows else np.zeros((0, base.ndim))
    else:
        full = X
    derived = {}
    if len(full):
        dummy = MCMCResult(base, full[None, :, :], np.zeros((1, len(full))), 0.0, 0.0)
        for k, v in dummy.derived(full).items():
            q = np.percentile(v, [16, 50, 84])
            derived[k] = (float(q[1]), float(q[1] - q[0]), float(q[2] - q[1]))
        if lp is not None:              # the removed area parameters, from the draws
            for i in lp.lin_idx:
                q = np.percentile(full[:, i], [16, 50, 84])
                derived[base.free[i].key] = (float(q[1]), float(q[1] - q[0]), float(q[2] - q[1]))
    n_evals = (int(problem.ncall) - n_evals) if hasattr(problem, "ncall") else -1
    res = LaplaceResult(names, theta, cov, sigma, corr, H, cond, ev, V, flags, directions, lo, hi, full, derived,
                        n_evals, time.time() - t0, method_used, linear,
                        meta={"steps": h.tolist(), "draws_inside_prior": kept, "ln_prob": float(-f(theta[None, :])[0]),
                              "hessian_centre_shifted": [names[i] for i in np.flatnonzero(shifted)],
                              "positive_definite": bool(ev.min() > 0) if n else True,
                              "regularised": bool(np.any(ev < PRIOR_PRECISION))})
    say(f"  Laplace: condition number {cond:.3g}, {len(directions)} flat/saddle direction(s), {res.runtime_s:.1f} s")
    return res
