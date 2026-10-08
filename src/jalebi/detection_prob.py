"""Detection probability with the continuum varied (0.22, `jalebi detect-prob`, docs/DETECTION.md).

A Delta BIC > 10 at one continuum says a molecule improves the fit *given that continuum*.  Weak species
(HCN, C2H2 Q-branches, isotopologues, cold water) sit on the pseudo-continuum, so whether they are "detected"
can turn on continuum choices that are all defensible.  Two ways to put a number on that:

ensemble (default)
    Draw N plausible continua: the configured estimator with its smoothness / quantile / knot spacing drawn
    within reasonable ranges, optionally other methods, and a global multiplicative offset drawn from
    N(0, offset_sigma) (the continuum's own uncertainty).  Refit each from the best-fit region: start at the
    disk's de_pass2 / best_fit checkpoint, a short DE pass, optionally a short MCMC.  Per unit (molecule /
    component): the detection fraction (Delta BIC > threshold over the variants), the spread of its parameters
    across the continua (a systematic error) and a class
        robust               detected in >= `robust_frac` (95 %) of the variants
        continuum-dependent  detected at the nominal continuum or in some variants, but not in >= robust_frac
        not detected         detected in <= `absent_frac` (5 %) of the variants and not at the nominal continuum

bayesian
    With fit.continuum_fit (spline, marginalised) the continuum freedom is in the likelihood; the 0.20 molecule
    evidence (dynesty, Delta ln Z of removing the molecule) then gives P(present) = 1 / (1 + exp(-(Delta ln Z +
    ln prior_odds))).  Classes by P: robust >= 0.95, not detected <= 0.05, else continuum-dependent / uncertain.

Outputs: <disk>/detection_probability.csv (one row per unit) and, with --survey, a survey-level table
(detection_probability_survey.csv: one row per disk x unit) in the results root.

EXPERIMENTAL and heavy: N refits per disk (use the emulator backend and few MCMC steps; see
runs/RUN_ME_detection_prob.sh).  The ranges of the continuum variations are a judgement call
(`fit.detection_prob.*` in config.py) -- the number is only as good as the ensemble it is computed over.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CLASSES = ("robust", "continuum-dependent", "not detected")


@dataclass
class DetProbSettings:
    n_variants: int = 30
    seed: int = 0
    offset_sigma: float = 0.01          # global multiplicative continuum offset ~ N(0, offset_sigma)
    methods: list[str] = field(default_factory=list)   # extra continuum methods to mix in (empty = the configured one)
    quantile_range: tuple[float, float] = (0.05, 0.2)            # irsqr
    knot_spacing_range: tuple[int, int] = (15, 60)               # irsqr [pixels]
    median_window_range: tuple[int, int] = (51, 201)             # median_sg [pixels]
    median_percentile_range: tuple[float, float] = (10.0, 40.0)  # median_sg
    lam_range_dex: tuple[float, float] = (-1.0, 1.0)             # asls / aspls: lam x 10^U(range)
    threshold: float = 10.0             # Delta BIC for "detected"
    robust_frac: float = 0.95
    absent_frac: float = 0.05
    de_maxiter: int = 40                # short DE pass per variant (from the best-fit start)
    de_popsize: int = 8
    mcmc_nsteps: int = 0                # 0 = no MCMC per variant (parameter spread from the optima only)
    prior_odds: float = 1.0             # bayesian mode
    cache_variants: bool = True         # write detection_prob_variants.json in the disk folder and reuse it


def settings_from_config(cfg) -> DetProbSettings:
    d = cfg.fit.detection_prob
    return DetProbSettings(n_variants=d.n_variants, seed=d.seed, offset_sigma=d.offset_sigma, methods=list(d.methods),
                           quantile_range=tuple(d.quantile_range), knot_spacing_range=tuple(d.knot_spacing_range),
                           median_window_range=tuple(d.median_window_range), median_percentile_range=tuple(d.median_percentile_range),
                           lam_range_dex=tuple(d.lam_range_dex), threshold=d.threshold, robust_frac=d.robust_frac,
                           absent_frac=d.absent_frac, de_maxiter=d.de_maxiter, de_popsize=d.de_popsize,
                           mcmc_nsteps=d.mcmc_nsteps, prior_odds=d.prior_odds, cache_variants=d.cache_variants)


def variant_configs(cfg, s: DetProbSettings) -> list[tuple[dict, float]]:
    """N (continuum override dict, offset factor) pairs drawn with `seed`.  The first variant is the nominal
    continuum (no changes, offset 1) so that the nominal detection is part of the table."""
    rng = np.random.default_rng(s.seed)
    out = [({}, 1.0)]
    methods = [cfg.continuum.method] + [m for m in s.methods if m != cfg.continuum.method]
    for i in range(1, s.n_variants):
        m = methods[rng.integers(len(methods))] if len(methods) > 1 else methods[0]
        over = {"method": m}
        if m == "irsqr":
            over["quantile"] = float(rng.uniform(*s.quantile_range))
            over["knot_spacing"] = int(rng.integers(s.knot_spacing_range[0], s.knot_spacing_range[1] + 1))
        elif m == "median_sg":
            w = int(rng.integers(s.median_window_range[0], s.median_window_range[1] + 1))
            over["median_window"] = w | 1
            over["median_percentile"] = float(rng.uniform(*s.median_percentile_range))
        elif m in ("asls", "aspls"):
            key = "aspls_lam" if m == "aspls" else "lam"
            over[key] = float(getattr(cfg.continuum, key) * 10.0 ** rng.uniform(*s.lam_range_dex))
        off = float(1.0 + rng.normal(0.0, s.offset_sigma)) if s.offset_sigma > 0 else 1.0
        out.append((over, off))
    return out


def classify(frac: float, nominal: bool, s: DetProbSettings) -> str:
    if frac >= s.robust_frac:
        return "robust"
    if frac <= s.absent_frac and not nominal:
        return "not detected"
    return "continuum-dependent"


def _start_theta(outdir: str | None, prob):
    """The best-fit start: de_pass2 / de_pass1 checkpoint, best_fit.json, else theta0."""
    if outdir:
        for name in ("de_pass2.json", "de_pass1.json"):
            p = os.path.join(outdir, name)
            if os.path.exists(p):
                try:
                    d = json.load(open(p))["payload"]
                    if d.get("free") == [q.key for q in prob.free]:
                        return np.asarray(d["theta"], float), name
                except Exception:
                    pass
        p = os.path.join(outdir, "best_fit.json")
        if os.path.exists(p):
            try:
                d = json.load(open(p))
                P = d["free_params"]
                return prob.theta_from_params(P, d.get("log_s", 0.0)), "best_fit.json"
            except Exception:
                pass
    return prob.theta0(), "theta0"


def ensemble(cfg, spec=None, settings: DetProbSettings | None = None, outdir: str | None = None, say=print,
             backend: str | None = None) -> pd.DataFrame:
    """Ensemble-mode detection probability for one disk (see the module docstring)."""
    from .pipeline import build_problem, prepare
    s = settings or DetProbSettings()
    cfg = cfg.model_copy(deep=True)
    cfg.fit.corner_check.enabled = False
    spec0 = prepare(cfg, spec)
    outdir = outdir or cfg.output_dir(spec0.name)
    variants = variant_configs(cfg, s)
    cache = os.path.join(outdir, "detection_prob_variants.json") if (outdir and s.cache_variants) else None
    done = {}
    if cache and os.path.exists(cache):
        try:
            done = {int(k): v for k, v in json.load(open(cache)).items()}
            say(f"detect-prob: {len(done)} of {len(variants)} variants from {os.path.basename(cache)}")
        except Exception:
            done = {}
    prob0 = build_problem(cfg, spec0, backend=backend, say=lambda m: None)
    theta_start, origin = _start_theta(outdir, prob0)
    say(f"detect-prob: {len(variants)} continuum variants (offset sigma {s.offset_sigma:g}, methods "
        f"{[cfg.continuum.method] + list(s.methods)}), start from {origin}, DE {s.de_maxiter} x {s.de_popsize}"
        + (f", MCMC {s.mcmc_nsteps} steps" if s.mcmc_nsteps else ""))
    units = list(prob0.all_units())
    recs = []
    t0 = time.time()
    for i, (over, off) in enumerate(variants):
        if i in done:
            recs.append(done[i]); continue
        c = cfg.model_copy(deep=True)
        for k, v in over.items():
            setattr(c.continuum, k, v)
        sp = prepare(c, spec0.copy()) if over else spec0.copy()
        if off != 1.0:
            sp.continuum = np.asarray(sp.continuum, float) * off
        try:
            prob = build_problem(c, sp, backend=backend, say=lambda m: None)
            th = np.clip(theta_start, prob.lo + 1e-6 * (prob.hi - prob.lo), prob.hi - 1e-6 * (prob.hi - prob.lo))
            opt = prob.optimise(th, maxiter=s.de_maxiter, popsize=s.de_popsize, seed=s.seed + i, polish=True)
            theta = opt.theta
            spread = None
            if s.mcmc_nsteps > 0:
                res = prob.mcmc(theta, nsteps=s.mcmc_nsteps, seed=s.seed + i, moves="de", init="scaled",
                                linear="profile", vectorize=True)
                theta = res.median_theta()
                summ = res.summary().set_index("parameter")
                spread = {k: [float(summ.at[k, "median"]), float(summ.at[k, "minus"]), float(summ.at[k, "plus"])] for k in summ.index}
            sig = prob.component_significance(theta)
            P, _ = prob.params_from_theta(theta)
            P = prob.model.resolve_params(P)
            rec = {"variant": i, "continuum": over, "offset": off, "chi2_red": float(opt.chi2_red),
                   "dBIC": {r.component: float(r.delta_BIC) for r in sig.itertuples()},
                   "params": {k: {q: float(v) for q, v in d.items() if isinstance(v, (int, float))} for k, d in P.items()},
                   "spread": spread, "ok": True}
        except Exception as e:                      # one failed variant must not sink the ensemble
            rec = {"variant": i, "continuum": over, "offset": off, "ok": False, "error": repr(e)}
        recs.append(rec)
        done[i] = rec
        if cache:
            try:
                from .resume import atomic_write_json
                atomic_write_json(cache, {str(k): v for k, v in done.items()})
            except Exception:
                pass
        say(f"  variant {i + 1}/{len(variants)}: " + (", ".join(f"{k} {v:+.0f}" for k, v in rec["dBIC"].items()) if rec.get("ok") else rec["error"])
            + f"  ({time.time() - t0:.0f}s)")
    table = summarise_ensemble(recs, units, s, prob0)
    table.insert(0, "target", spec0.name)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        table.to_csv(os.path.join(outdir, "detection_probability.csv"), index=False)
        say(f"detect-prob -> {os.path.join(outdir, 'detection_probability.csv')}")
    return table


def summarise_ensemble(recs: list[dict], units: list[str], s: DetProbSettings, prob=None) -> pd.DataFrame:
    ok = [r for r in recs if r.get("ok")]
    rows = []
    lead = {}
    if prob is not None:
        lead = prob.all_units()
    for u in units:
        d = np.array([r["dBIC"].get(u, np.nan) for r in ok], float)
        det = d > s.threshold
        n = int(np.isfinite(d).sum())
        frac = float(det[np.isfinite(d)].mean()) if n else np.nan
        nominal = bool(ok and ok[0]["variant"] == 0 and ok[0]["dBIC"].get(u, -np.inf) > s.threshold)
        row = {"unit": u, "n_variants": n, "n_detected": int(det.sum()), "detection_fraction": frac,
               "nominal_detected": nominal, "dBIC_nominal": float(ok[0]["dBIC"].get(u, np.nan)) if ok else np.nan,
               "dBIC_min": float(np.nanmin(d)) if n else np.nan, "dBIC_median": float(np.nanmedian(d)) if n else np.nan,
               "dBIC_max": float(np.nanmax(d)) if n else np.nan,
               "class": classify(frac, nominal, s) if n else "no result"}
        name = lead.get(u, u)
        for q in ("T", "logN", "logR"):
            vals = np.array([r["params"].get(name, {}).get(q, np.nan) for r in ok], float)
            vals = vals[np.isfinite(vals)]
            if len(vals):
                lo, med, hi = np.percentile(vals, [16, 50, 84])
                row[f"{q}_median"] = float(med); row[f"{q}_spread_16_84"] = float(0.5 * (hi - lo)); row[f"{q}_min"] = float(vals.min()); row[f"{q}_max"] = float(vals.max())
        logna = np.array([r["params"].get(name, {}).get("logN", np.nan) + np.log10(np.pi) + 2 * r["params"].get(name, {}).get("logR", np.nan)
                          for r in ok], float)
        logna = logna[np.isfinite(logna)]
        if len(logna):
            lo, med, hi = np.percentile(logna, [16, 50, 84])
            row["logNA_median"] = float(med); row["logNA_spread_16_84"] = float(0.5 * (hi - lo))
        rows.append(row)
    return pd.DataFrame(rows)


def bayesian(cfg, spec=None, settings: DetProbSettings | None = None, outdir: str | None = None, say=print,
             components: list[str] | None = None, backend: str | None = None, dynesty_kwargs: dict | None = None) -> pd.DataFrame:
    """Bayesian-mode detection probability: Delta ln Z per removed component with the continuum correction
    marginalised (fit.continuum_fit should be spline) -> P(present)."""
    from .nested import evidence_without, problem_without, run_dynesty
    from .pipeline import build_problem, prepare
    s = settings or DetProbSettings()
    cfg = cfg.model_copy(deep=True)
    if cfg.fit.continuum_fit == "none":
        say("detect-prob bayesian: fit.continuum_fit is none -- the continuum is NOT marginalised; set spline for the intended use")
    spec0 = prepare(cfg, spec)
    outdir = outdir or cfg.output_dir(spec0.name)
    prob = build_problem(cfg, spec0, backend=backend, say=lambda m: None)
    theta, origin = _start_theta(outdir, prob)
    d = cfg.fit.dynesty
    kw = dict(nlive=d.nlive, sample=d.sample, bound=d.bound, dlogz_init=d.dlogz_init, n_effective=d.n_effective,
              maxcall=d.maxcall, processes=d.processes, seed=d.seed, pfrac=d.pfrac, slices=d.slices, walks=d.walks,
              dynamic=d.dynamic, linear="profile", **(dynesty_kwargs or {}))
    say(f"detect-prob bayesian: dynesty full model (start {origin}), then without each component")
    full = run_dynesty(prob, theta0=theta, **kw)
    comps = components or [c for c in prob.all_units()]
    sig = prob.component_significance(theta)
    ev = evidence_without(lambda n: problem_without(prob, n), comps, full, kw, significance=sig, say=say)
    ev["ln_prior_odds"] = float(np.log(s.prior_odds))
    ev["p_present"] = 1.0 / (1.0 + np.exp(-(ev["delta_lnZ"] + ev["ln_prior_odds"])))
    ev["class"] = ["robust" if p >= s.robust_frac else ("not detected" if p <= s.absent_frac else "continuum-dependent") for p in ev["p_present"]]
    ev.insert(0, "target", spec0.name)
    ev = ev.rename(columns={"component": "unit"})
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        ev.to_csv(os.path.join(outdir, "detection_probability.csv"), index=False)
        say(f"detect-prob -> {os.path.join(outdir, 'detection_probability.csv')}")
    return ev


def survey_table(root: str) -> pd.DataFrame:
    """Concatenate <root>/*/detection_probability.csv into <root>/detection_probability_survey.csv."""
    import glob
    parts = []
    for p in sorted(glob.glob(os.path.join(root, "*", "detection_probability.csv"))):
        try:
            t = pd.read_csv(p)
            if "target" not in t:
                t.insert(0, "target", os.path.basename(os.path.dirname(p)))
            parts.append(t)
        except Exception:
            continue
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(out):
        out.to_csv(os.path.join(root, "detection_probability_survey.csv"), index=False)
    return out
