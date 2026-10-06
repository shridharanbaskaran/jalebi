"""Linear parameters out of the MCMC: the emitting areas are profiled (NNLS) or marginalised (Gaussian).

`fit.mcmc.linear`
  sample       the areas are ordinary MCMC parameters (logR or logNA; the 0.16 behaviour, default)
  profile      at every likelihood call the areas are solved by bounded non-negative least squares given
               the nonlinear parameters (T, log N, ...): ln L(theta) = max_a ln L(theta, a)
               (the approach of DuCKLinG, Kaeufer et al. 2024, A&A 687, A209)
  marginalise  ln L(theta) = ln Int L(theta, a) p(a) da with a broad Gaussian prior p(a), analytically

Which parameters are linear
---------------------------
The model is F(pix) = sum_u a_u f_u(theta; pix) + F_screens(theta), with a_u = R_u^2 [au^2] and f_u the
pixel flux of unit u for R = 1 au (`SlabModel.unit_fluxes`).  So, per unit:
  * plain slab                        linear in a_u                                     -> linear
  * opacity group (shared tau)        the members sum their opacity *before* the
                                      exponential and share one area: still linear in
                                      the group's single area; members' log N nonlinear -> one linear a per group
  * tied isotopologue (tie_to)        rides on the parent's area (its flux is added to
                                      the parent's column, `fold_tied`); ratio nonlinear -> parent's a only
  * T_vib (two-temperature) slab      the source function changes, the area does not    -> linear
  * per-component `windows`           a pixel mask on f_u                               -> linear
  * screens with covers="all"         multiply f_u by the transmission: still linear in a_u
  * kind: annuli                      logR is the *outer radius* of the T(r), N(r) power
                                      law (the annuli edges move with it); R_in, q, p    -> nonlinear (sampled)
  * kind: absorption (screen)         covering fraction fc enters through products of
                                      transmissions and is bounded to [0, 1]             -> nonlinear (sampled)
  * an area that is `fixed`, or has a Gaussian prior from `priors`                       -> left as it is (sampled /
                                                                                            fixed)
  * per-sub-band continuum offsets    there are none in the likelihood: the continuum is estimated and
                                      subtracted before the fit (`pipeline.prepare`); nothing to remove.
The noise scale s multiplies every sigma: the NNLS solution does not depend on it, the marginal does (below).

Parameterisation
----------------
With `fit.area_param: logNA` the sampled `logNA = log N + log pi + 2 log R` mixes N and A.  In the linear
modes the area parameter (logR or logNA) is removed from the sampled vector; what is sampled is exactly the
remaining parameters (log N, T, ..., log s).  The chain written to `chain.npz` is expanded back to the full
parameter vector (same columns and names as `sample`): logNA = log N + log pi + log10(a) per sample.

Marginalisation
---------------
Whitened with w_i / sigma_i^2 (the pixel weights of `window_weights`) and the noise scale s:
    G = M^T W M / s^2 + Lambda^-1,   b = M^T W y / s^2,   a_hat = G^-1 b
    ln L = -1/2 [ (y - M a_hat)^T W (y - M a_hat) / s^2 + a_hat^T Lambda^-1 a_hat ]
           - 1/2 ln det G - 1/2 ln det Lambda - 1/2 sum_i w_i ln(2 pi sigma_i^2 s^2)
Lambda = diag(lambda_u^2), lambda_u = `linear_prior_scale` (default: the largest area the prior bounds allow,
10^(2 logR_max) au^2), so it is broad.  ln det G carries the s- and theta-dependent Occam factor.
`linear_prior: log` (default) adds - sum_u ln a_hat_u, the Laplace approximation of a prior uniform in log R
(what `sample` uses), so that marginalise and sample answer the same question when a_hat >> its error.
`linear_prior: gaussian` is the bare Gaussian marginal: for a component whose N is not measured (only N.A)
it favours small N / large areas (AS 209: CO went to log N 13.3, R 11 au).
Positivity: NOT imposed in the marginal likelihood (the integral runs over all real a).  It is imposed on the
area draws (a draw with any a_u <= 0 is redrawn, i.e. the conditional Gaussian truncated to a > 0), and every
sample whose conditional mean a_hat_u <= 0 is counted and reported (`linear_neg_frac`): such a component is
not detected; use profile or sample for it.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear, nnls

from .fit import LOG10PI, DEFAULT_BOUNDS, FitProblem, MCMCResult

LINEAR_MODES = ("sample", "profile", "marginalise")
_ALIASES = {"marginalize": "marginalise", "nnls": "profile", "joint": "sample", None: "sample", "": "sample"}


def normalise_mode(mode) -> str:
    m = _ALIASES.get(mode, mode)
    m = str(m).lower()
    m = _ALIASES.get(m, m)
    if m not in LINEAR_MODES:
        raise ValueError(f"unknown fit.mcmc.linear {mode!r}: sample | profile | marginalise")
    return m


# ------------------------------------------------------------------------------------------------
# Audit: which free parameters are linear
# ------------------------------------------------------------------------------------------------

def linear_audit(problem: FitProblem) -> pd.DataFrame:
    """One row per free parameter: is it a linear amplitude that the profile/marginalise modes remove?"""
    comps = {c.name: c for c in problem.components}
    units = problem.area_units()                              # unit -> leader (slab + annuli, no screens)
    lead_of = {lead: key for key, lead in units.items()}
    rows = []
    for p in problem.free:
        c = comps.get(p.comp)
        unit, linear, why = "", False, ""
        if p.comp == "global":
            why = "noise scale: enters every sigma (sampled; the NNLS solution does not depend on it)"
        elif p.name in ("logR", "logNA"):
            unit = lead_of.get(p.comp, c.group or c.name)
            if c.kind == "annuli":
                why = "annuli: logR is the outer radius of the T(r), N(r) power law -> nonlinear"
            elif c.kind == "absorption":
                why = "absorption screens have no area"
            elif p.gauss is not None:
                why = "has a Gaussian prior (priors): left in the sampler"
            else:
                linear = True
                why = ("opacity group: one area for the summed opacity -> linear" if c.group else
                       "slab area R^2: F = a f(T, N) -> linear") + \
                      ("; log(N·A) is replaced by log N + the solved area" if p.name == "logNA" else "")
        elif p.name == "fc":
            why = "covering fraction: transmissions multiply, bounded [0, 1] -> nonlinear"
        elif p.name == "ratio":
            why = "isotopologue ratio sets the child's opacity -> nonlinear (child shares the parent's area)"
        elif p.name in ("q", "p", "logRin"):
            why = "annuli power law -> nonlinear"
        else:
            why = "line opacity / excitation / kinematics -> nonlinear"
        rows.append({"parameter": p.key, "unit": unit, "linear": linear, "reason": why})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------
# The reduced problem
# ------------------------------------------------------------------------------------------------

class LinearProblem(FitProblem):
    """A FitProblem whose sampled vector excludes the linear areas; they are profiled or marginalised
    inside `log_prob`.  Shares the model, data and weights of `base` (no copy of the opacity bases)."""

    def __init__(self, base: FitProblem, mode: str, theta_full, prior: str = "log",
                 prior_scale: float | None = None):
        self.__dict__.update(base.__dict__)
        self.base = base
        self.mode = normalise_mode(mode)
        if self.mode == "sample":
            raise ValueError("LinearProblem needs mode profile or marginalise")
        if prior not in ("gaussian", "log"):
            raise ValueError(f"unknown fit.mcmc.linear_prior {prior!r}: gaussian | log")
        self.prior = prior
        theta_full = np.asarray(theta_full, float)
        audit = linear_audit(base)
        lin_idx = [i for i, ok in enumerate(audit.linear) if ok]
        self.lin_idx = np.array(lin_idx, int)
        self.keep_idx = np.array([i for i in range(base.ndim) if i not in set(lin_idx)], int)
        self.lin_units = [audit.unit[i] for i in lin_idx]
        self.lin_lead = [base.free[i].comp for i in lin_idx]
        self.lin_kind = [base.free[i].name for i in lin_idx]               # logR | logNA
        self.k = len(lin_idx)
        self.free = [base.free[i] for i in self.keep_idx]
        self.ndim = len(self.free)
        self.lo = base.lo[self.keep_idx]; self.hi = base.hi[self.keep_idx]
        self.theta_full0 = theta_full.copy()
        P0, _ = base.params_from_theta(theta_full)
        self._logR_ref = {u: P0[lead]["logR"] for u, lead in zip(self.lin_units, self.lin_lead)}
        self._logN_pos = {lead: next((j for j, p in enumerate(self.free) if p.comp == lead and p.name == "logN"), None)
                          for lead in self.lin_lead}
        # area bounds from the prior bounds of the removed parameter
        self._blo = base.lo[self.lin_idx]; self._bhi = base.hi[self.lin_idx]
        rmax = DEFAULT_BOUNDS["logR"][1]
        hi_area = []
        for j, i in enumerate(lin_idx):
            hi_area.append(10.0 ** (2.0 * base.hi[i]) if self.lin_kind[j] == "logR" else 10.0 ** (2.0 * rmax))
        self.prior_scale = np.full(self.k, float(prior_scale)) if prior_scale else np.array(hi_area, float)
        self._sw = np.sqrt(self.weights) / self.sigma                        # whitening: w / sigma^2
        self._lnorm_w = np.sum(self.weights * np.log(2 * np.pi * self.sigma ** 2))
        self._wsum = float(np.sum(self.weights))
        self.ncall = 0

    # ---- parameter mapping ------------------------------------------------------------------------
    def reduce(self, theta_full) -> np.ndarray:
        return np.asarray(theta_full, float)[self.keep_idx]

    def params_from_theta(self, theta):
        """Component parameters with the areas at their reference values (the optimum): used by the
        priors (ordering), the support of `independent_blocks` and the model columns."""
        full = self.theta_full0.copy()
        full[self.keep_idx] = theta
        P, log_s = self.base.params_from_theta(full)
        for u, lead in zip(self.lin_units, self.lin_lead):
            P[lead]["logR"] = self._logR_ref[u]
        return P, log_s

    def area_bounds(self, theta) -> tuple[np.ndarray, np.ndarray]:
        """Bounds on a = R^2 implied by the prior bounds of the removed logR / logNA parameters."""
        lo = np.empty(self.k); hi = np.empty(self.k)
        for j in range(self.k):
            if self.lin_kind[j] == "logR":
                lo[j], hi[j] = 10.0 ** (2.0 * self._blo[j]), 10.0 ** (2.0 * self._bhi[j])
            else:                                       # logNA = logN + log pi + log10(a)
                pos = self._logN_pos[self.lin_lead[j]]
                logN = theta[pos] if pos is not None else self.base._base[self.lin_lead[j]]["logN"]
                lo[j], hi[j] = 10.0 ** (self._blo[j] - logN - LOG10PI), 10.0 ** (self._bhi[j] - logN - LOG10PI)
        return lo, hi

    def full_theta(self, theta, areas) -> np.ndarray:
        """Full parameter vector (base parameterisation) from the sampled vector and the areas a = R^2."""
        full = self.theta_full0.copy()
        full[self.keep_idx] = theta
        a = np.maximum(np.asarray(areas, float), 1e-300)
        for j, i in enumerate(self.lin_idx):
            if self.lin_kind[j] == "logR":
                full[i] = 0.5 * np.log10(a[j])
            else:
                pos = self._logN_pos[self.lin_lead[j]]
                logN = theta[pos] if pos is not None else self.base._base[self.lin_lead[j]]["logN"]
                full[i] = logN + LOG10PI + np.log10(a[j])
        return full

    # ---- design matrix --------------------------------------------------------------------------------
    def design(self, theta):
        """(y_eff, M, log_s): data minus every non-linear unit's flux, the 1-au flux columns of the linear
        units (tied isotopologues folded into their parent's column) and the noise scale."""
        P, log_s = self.params_from_theta(theta)
        uf, _, lR = self.model.unit_fluxes(P)
        uf = self.model.fold_tied(uf)
        lin = set(self.lin_units)
        y = self.y.copy()
        for key, f in uf.items():
            if key not in lin:
                y -= f * 10.0 ** (2.0 * lR[key])
        M = np.column_stack([uf[u] for u in self.lin_units]) if self.k else np.zeros((len(y), 0))
        return y, M, log_s

    def solve(self, theta, draw_rng=None) -> dict:
        """Linear solve at `theta`: ln L, the areas (profile: bounded NNLS; marginalise: conditional mean),
        a draw of the areas, and whether a conditional mean was <= 0."""
        y, M, log_s = self.design(theta)
        return self._solve_design(theta, y, M, log_s, draw_rng)

    def _solve_design(self, theta, y, M, log_s, draw_rng=None) -> dict:
        s2 = 10.0 ** (2.0 * log_s)
        sw = self._sw
        yw = y * sw
        Mw = M * sw[:, None]
        norm = -0.5 * (self._lnorm_w + self._wsum * np.log(s2))
        self.ncall += 1
        if self.k == 0:
            return {"lnL": -0.5 * float(yw @ yw) / s2 + norm, "a": np.zeros(0), "draw": np.zeros(0), "neg": np.zeros(0, bool)}
        if self.mode == "profile":
            lo, hi = self.area_bounds(theta)
            a, rnorm = nnls(Mw, yw)
            if np.any(a < lo) or np.any(a > hi):
                a = lsq_linear(Mw, yw, bounds=(lo, hi), method="bvls").x
                r = yw - Mw @ a
                chi2 = float(r @ r)
            else:
                chi2 = rnorm ** 2
            return {"lnL": -0.5 * chi2 / s2 + norm, "a": a, "draw": a, "neg": np.zeros(self.k, bool)}
        # marginalise
        lam2 = self.prior_scale ** 2
        G = (Mw.T @ Mw) / s2 + np.diag(1.0 / lam2)
        b = (Mw.T @ yw) / s2
        try:
            L = np.linalg.cholesky(G)
        except np.linalg.LinAlgError:
            return {"lnL": -np.inf, "a": np.full(self.k, np.nan), "draw": np.full(self.k, np.nan), "neg": np.ones(self.k, bool)}
        a = np.linalg.solve(L.T, np.linalg.solve(L, b))
        r = yw - Mw @ a
        Q = float(r @ r) / s2 + float(np.sum(a ** 2 / lam2))
        lnL = -0.5 * Q - float(np.sum(np.log(np.diag(L)))) - 0.5 * float(np.sum(np.log(lam2))) + norm
        neg = a <= 0
        if self.prior == "log":
            lo, _ = self.area_bounds(theta)
            lnL -= float(np.sum(np.log(np.maximum(a, lo))))
        draw = a
        if draw_rng is not None:
            draw = self._draw(a, L, draw_rng)
        return {"lnL": lnL, "a": a, "draw": draw, "neg": neg, "chol": L}

    @staticmethod
    def _draw(mean, chol, rng, tries: int = 50):
        """One draw from N(mean, G^-1) truncated to a > 0 (rejection; after `tries` failures the
        non-positive entries are set to a tiny positive value)."""
        k = len(mean)
        for _ in range(tries):
            z = rng.standard_normal(k)
            a = mean + np.linalg.solve(chol.T, z)          # cov = (L L^T)^-1 = L^-T L^-1
            if np.all(a > 0):
                return a
        return np.where(a > 0, a, 1e-12)

    # ---- probability -------------------------------------------------------------------------------------
    def log_like(self, theta) -> float:
        return float(self.solve(theta)["lnL"])

    def log_prob(self, theta) -> float:
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_like(theta)
        return lp + ll if np.isfinite(ll) else -np.inf

    def blob_size(self) -> int:
        return 3 * self.k

    def log_prob_blob(self, theta):
        """(ln P, blob) for emcee: blob = [conditional mean or NNLS areas (k), area draw (k), a_hat <= 0 (k)]."""
        theta = np.asarray(theta, float)
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf, np.full(self.blob_size(), np.nan)
        rng = None
        if self.mode == "marginalise":       # deterministic per theta: a walker that stays keeps its draw
            rng = np.random.default_rng(np.frombuffer(np.ascontiguousarray(theta).tobytes(), np.uint32))
        s = self.solve(theta, draw_rng=rng)
        if not np.isfinite(s["lnL"]):
            return -np.inf, np.full(self.blob_size(), np.nan)
        return lp + s["lnL"], np.concatenate([s["a"], s["draw"], s["neg"].astype(float)])

    def log_prob_blob_many(self, thetas):
        """Vectorised `log_prob_blob`: the unit fluxes of all walkers in one pass (emulator), the small linear
        solves per walker."""
        thetas = np.atleast_2d(np.asarray(thetas, float))
        bad = (-np.inf, np.full(self.blob_size(), np.nan))
        out = [bad] * len(thetas)
        lp = np.array([self.log_prior(t) for t in thetas])
        ok = np.flatnonzero(np.isfinite(lp))
        if len(ok) == 0:
            return out
        _, ls, F, lR = self.unit_fluxes_many(thetas[ok])
        F = self.model.fold_tied(F)
        lin = set(self.lin_units)
        Y = np.repeat(self.y[None, :], len(ok), axis=0)
        for key, f in F.items():
            if key not in lin:
                Y -= f * (10.0 ** (2.0 * lR[key]))[:, None]
        for j, i in enumerate(ok):
            th = thetas[i]
            M = np.column_stack([F[u][j] for u in self.lin_units]) if self.k else np.zeros((len(self.y), 0))
            rng = np.random.default_rng(np.frombuffer(np.ascontiguousarray(th).tobytes(), np.uint32)) \
                if self.mode == "marginalise" else None
            sres = self._solve_design(th, Y[j], M, ls[j], rng)
            if np.isfinite(sres["lnL"]):
                out[i] = (lp[i] + sres["lnL"], np.concatenate([sres["a"], sres["draw"], sres["neg"].astype(float)]))
        return out

    def blob_owner(self, groups) -> np.ndarray:
        """Block (index into `groups`) whose sampler provides each blob column: the block that samples the
        unit leader's parameters (block 0 when the leader has none left, e.g. log N and T fixed)."""
        own = np.zeros(self.k, int)
        for j, lead in enumerate(self.lin_lead):
            for g, idx in enumerate(groups):
                if any(self.free[i].comp == lead for i in idx):
                    own[j] = g
                    break
        return np.concatenate([own, own, own])


# ------------------------------------------------------------------------------------------------
# Running it
# ------------------------------------------------------------------------------------------------

def run_linear_mcmc(problem: FitProblem, theta0, mode: str, prior: str = "log", prior_scale=None,
                    **kw) -> MCMCResult:
    """MCMC over the nonlinear parameters with the areas profiled or marginalised; returns an MCMCResult
    of the *full* problem (same columns as `sample`; the areas per sample are the NNLS solution
    (profile) or a draw from the conditional Gaussian (marginalise))."""
    lp = LinearProblem(problem, mode, theta0, prior=prior, prior_scale=prior_scale)
    th_red = lp.reduce(theta0)
    t0 = time.time()
    if lp.ndim == 0:                         # nothing nonlinear left: the areas' conditional only
        nsteps = kw.get("nsteps", 2000)
        nw = kw.get("nwalkers") or 32
        rng = np.random.default_rng(kw.get("seed", 0))
        s = lp.solve(th_red)
        draws = np.array([[lp._draw(s["a"], s["chol"], rng) if "chol" in s else s["a"] for _ in range(nw)]
                          for _ in range(nsteps)])
        chain_red = np.zeros((nsteps, nw, 0))
        lnp = np.full((nsteps, nw), lp.log_prior(th_red) + s["lnL"])
        blobs = np.concatenate([np.broadcast_to(s["a"], draws.shape), draws,
                                np.broadcast_to(s["neg"].astype(float), draws.shape)], axis=2)
        acc, meta = 1.0, {"blocks": [], "block_acceptance": [], "block_nwalkers": []}
    else:
        rr = FitProblem.mcmc(lp, th_red, **kw)
        chain_red, lnp, blobs, acc, meta = rr.chain, rr.log_prob, rr.blobs, rr.acceptance, rr.meta
    ns, nw = lnp.shape
    k = lp.k
    a_mean = blobs[..., :k]; a_draw = blobs[..., k:2 * k]; neg = blobs[..., 2 * k:]
    full = np.repeat(np.repeat(np.asarray(theta0, float)[None, None, :], ns, 0), nw, 1)
    full[..., lp.keep_idx] = chain_red
    ok = np.isfinite(a_draw)
    a_safe = np.where(ok & (a_draw > 0), a_draw, 1e-300)
    for j, i in enumerate(lp.lin_idx):
        if lp.lin_kind[j] == "logR":
            full[..., i] = 0.5 * np.log10(a_safe[..., j])
        else:
            pos = lp._logN_pos[lp.lin_lead[j]]
            logN = chain_red[..., pos] if pos is not None else problem._base[lp.lin_lead[j]]["logN"]
            full[..., i] = logN + LOG10PI + np.log10(a_safe[..., j])
    res = MCMCResult(problem, full, lnp, acc, time.time() - t0)
    res.sampled = np.zeros(problem.ndim, bool); res.sampled[lp.keep_idx] = True
    res.area_mean = a_mean
    res.linear = lp.mode
    burn = res.burn()
    tail = slice(burn, None)
    neg_frac = {u: float(np.nanmean(neg[tail, :, j])) for j, u in enumerate(lp.lin_units)}
    out_b = {}
    for j, (u, i) in enumerate(zip(lp.lin_units, lp.lin_idx)):
        col = full[tail, :, i]
        out_b[u] = float(np.mean((col < problem.lo[i]) | (col > problem.hi[i])))
    try:
        import emcee
        tau_s = emcee.autocorr.integrated_time(chain_red, tol=0) if lp.ndim else np.zeros(0)
    except Exception:
        tau_s = np.full(lp.ndim, np.nan)
    res.meta = {**meta, "linear": lp.mode, "linear_prior": lp.prior if lp.mode == "marginalise" else None,
                "linear_units": lp.lin_units, "linear_params": [problem.free[i].key for i in lp.lin_idx],
                "sampled": [p.key for p in lp.free], "tau_sampled": tau_s,
                "linear_neg_frac": neg_frac, "linear_outside_bounds_frac": out_b,
                "likelihood_calls": int(ns * nw)}
    return res
