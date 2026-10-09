"""Per-disk pipeline: prep -> grid -> optimise -> mcmc -> report.  Used by the CLI, the batch
runner and the web app so that every interface runs the same code from the same config."""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import activity as act
from .config import ProjectConfig
from .continuum import OH_PROMPT_RANGE, ContinuumSettings, default_mask, estimate_continuum, in_ranges, oh_prompt_ranges
from .data import Spectrum, load_spectrum, spike_filter
from .fit import DEFAULT_BOUNDS, FitProblem, GridResult, MCMCResult, OptResult, Param, default_free_params


def continuum_settings(cfg: ProjectConfig) -> ContinuumSettings:
    c = cfg.continuum
    return ContinuumSettings(method=c.method, protect=c.protect, protected={k: tuple(v) for k, v in c.protected.items()},
                             smooth=c.smooth, quantile=c.quantile, knot_spacing=c.knot_spacing,
                             median_window=c.median_window, median_percentile=c.median_percentile, sg_window=c.sg_window,
                             sg_order=c.sg_order, n_iter=c.n_iter, lam=c.lam, p=c.p, aspls_lam=c.aspls_lam, aspls_alpha=c.aspls_alpha, segment=c.segment,
                             overlap=c.overlap, percentile=c.percentile, min_window=c.min_window,
                             anchors=list(c.anchors), anchor_width=c.anchor_width)


def prepare(cfg: ProjectConfig, spec: Spectrum | None = None, gas_model: np.ndarray | None = None) -> Spectrum:
    """Ingest (or take the given spectrum), rest-frame shift, spike filter, continuum, masks, noise."""
    if spec is None:
        ex = cfg.target.extraction
        ext = {"ra": ex.ra, "dec": ex.dec, "aperture_fwhm_scale": ex.aperture_fwhm_scale, "aperture_arcsec": ex.aperture_arcsec,
               "annulus_arcsec": tuple(ex.annulus_arcsec) if ex.annulus_arcsec else None, "apcorr": ex.apcorr}
        # an unset target name (the default "target") lets the loader use the FITS TARGNAME / file name
        name = cfg.target.name if cfg.target.name and cfg.target.name != "target" else None
        spec = load_spectrum(cfg.target.path, source=ex.source, extraction=ext, name=name,
                             distance_pc=cfg.target.distance_pc)
        spec = spec.to_rest_frame(cfg.target.rv_kms)
        if cfg.target.spike_filter:
            spec = spike_filter(spec)
    spec.distance_pc = cfg.target.distance_pc
    t0 = time.perf_counter()
    spec.continuum = estimate_continuum(spec, continuum_settings(cfg), gas_model=gas_model)
    act.debug("lte", "continuum %s estimated in %s", cfg.continuum.method, act.fmt_time(time.perf_counter() - t0))
    m = spec.mask.copy()
    extra = {k: tuple(v) for k, v in cfg.masks.extra.items()}
    if cfg.masks.default_lines:
        m &= default_mask(spec, mask_oh_prompt=cfg.masks.oh_prompt, oh_whole_range=cfg.masks.oh_whole_range, extra=extra)
    else:
        if cfg.masks.oh_prompt:
            m &= ~in_ranges(spec.wave, [OH_PROMPT_RANGE] if cfg.masks.oh_whole_range else oh_prompt_ranges())
        if extra:
            m &= ~in_ranges(spec.wave, extra)
    spec.mask = m & np.isfinite(spec.continuum)
    act.debug("lte", "masks: default lines %s, OH prompt %s, %d extra ranges → %d of %d px used", cfg.masks.default_lines,
              cfg.masks.oh_prompt, len(extra), int(spec.mask.sum()), len(spec.mask))
    return spec


def free_params(cfg: ProjectConfig) -> list[Param]:
    comps = cfg.components_list()
    bounds = dict(DEFAULT_BOUNDS)
    for c in cfg.components:                 # per-molecule bounds (fit.bounds_by_molecule), then per component
        for k, v in (cfg.fit.bounds_by_molecule.get(c.molecule) or {}).items():
            bounds[f"{c.name}.{k}"] = tuple(v)
        for k, v in c.bounds.items():
            bounds[f"{c.name}.{k}"] = tuple(v)
    free = default_free_params(comps, area_param=cfg.fit.area_param, fit_rv=cfg.fit.fit_rv,
                               fit_fwhm=cfg.fit.fit_fwhm, bounds=bounds)
    fixed = {(c.name, f) for c in cfg.components for f in c.fixed}
    priors = {(c.name, k): (float(v[0]), float(v[1])) for c in cfg.components for k, v in c.priors.items()}
    for p in free:
        if (p.comp, p.name) in priors:
            p.gauss = priors[(p.comp, p.name)]
    return [p for p in free if (p.comp, p.name) not in fixed]


def build_problem(cfg: ProjectConfig, spec: Spectrum, backend: str | None = None, say=None) -> FitProblem:
    """The fit problem of a config and a prepared spectrum.  `backend` (default `fit.model_backend`):
    "emulator" builds or loads the emulator tables (jalebi.emulator) and attaches them to the model."""
    comps = cfg.components_list()
    prob = FitProblem(spec, comps, cfg.windows(), free_params(cfg), area_param=cfg.fit.area_param,
                      fit_noise_scale=cfg.fit.fit_noise_scale, ordering=[tuple(o) for o in cfg.fit.ordering],
                      window_weights=cfg.window_weights(), oversample=cfg.fit.oversample,
                      releases=cfg.linedata.releases, use_pipeline_err=cfg.fit.use_pipeline_err,
                      tvib_below_trot=cfg.fit.tvib_below_trot,
                      model_kwargs={"R_model": cfg.R_model, "R_scale": cfg.R_scale, "R_constant": cfg.R_constant})
    cf = getattr(cfg.fit, "continuum_fit", "none") or "none"
    if cf != "none":                                   # 0.22: joint continuum correction (docs/CONTINUUM.md)
        cc = cfg.fit.continuum_correction
        cont = prob.set_continuum_fit(cf, knot_spacing_um=cc.knot_spacing_um, prior=cc.prior, prior_width=cc.prior_width, solve=cc.mode)
        (say or (lambda m: act.info("fit", "%s", m)))(f"continuum correction (fit.continuum_fit): {cont.describe()}")
    backend = backend or cfg.fit.model_backend
    if backend == "emulator":
        say = say or (lambda m: act.info("fit", "%s", m))
        t0 = time.time()
        say(f"model backend: emulator ({cfg.fit.emulator.cache} tables in " + (cfg.fit.emulator.cache_dir or "the default cache") + ")")
        em = prob.use_emulator(cfg.fit.emulator.settings(cfg), say=say)
        for line in em.summary():
            say("  " + line)
        say(f"  emulator ready in {time.time() - t0:.0f} s")
    elif backend != "exact":
        raise ValueError(f"unknown fit.model_backend {backend!r}: exact | emulator")
    return prob


@dataclass
class RunResult:
    cfg: ProjectConfig
    spec: Spectrum
    problem: FitProblem
    grids: dict[str, GridResult] = field(default_factory=dict)
    opt: OptResult | None = None
    mcmc: MCMCResult | None = None
    theta: np.ndarray | None = None
    log: list[str] = field(default_factory=list)
    detection: object = None            # DetectionResult when fit.auto_detect was on
    laplace: object = None              # LaplaceResult (fit.laplace / jalebi fit --laplace / the app's Quick errors)
    evidence: object = None             # DataFrame of Delta ln Z per removed component (fit.dynesty.evidence_without)
    outdir: str | None = None           # folder the results were written to (output with {target} filled in)
    checkpoint: object = None           # 0.22: jalebi.resume.Checkpoint of this run (stage files, key)
    corner: dict | None = None          # 0.22: jalebi.corner.corner_check of the final parameters

    def say(self, msg):
        self.log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        if act.enabled(logging.INFO):          # `jalebi serve`: into the activity log
            act.get_logger("fit").info("%s", msg)
        else:
            print(self.log[-1], flush=True)


def run_grid_stage(run: RunResult, progress=None):
    cfg, prob = run.cfg, run.problem
    g = cfg.fit.grid
    logN = np.linspace(g.logN[0], g.logN[1], int(g.logN[2]))
    T = np.linspace(g.T[0], g.T[1], int(g.T[2]))
    order = g.order or [c.name for c in cfg.components if c.enabled and not c.tie_to]
    theta = prob.theta0() if run.theta is None else run.theta.copy()
    from .molecules import DEFAULT_WINDOWS
    for name in order:
        comp = next(c for c in cfg.components if c.name == name)
        # component windows: intersection of the fit windows with the molecule's default windows, or with
        # the component's own `windows` when it has them (a component restricted to 5-9 um is gridded there)
        own = [tuple(w) for w in comp.windows] if comp.windows else None
        wins = [w for w in prob.windows if any(a <= w[1] and b >= w[0] for a, b in (own or DEFAULT_WINDOWS.get(comp.molecule, [])))] or prob.windows
        if own:
            wins = [(max(w[0], a), min(w[1], b)) for w in wins for a, b in own if a <= w[1] and b >= w[0]] or wins
        run.say(f"grid: {name} on {wins}")
        bar = abar = None
        if progress is None and act.enabled(logging.INFO):
            abar = act.Progress(f"grid {name}", total=len(logN) * len(T), unit="model", area="fit")
        elif progress is None:
            try:
                from tqdm import tqdm
                bar = tqdm(total=100, desc=f"  grid {name}", unit="%", leave=False, dynamic_ncols=True)
            except ImportError:
                pass
        def gprog(f, n=name):
            if bar is not None:
                bar.n = int(100 * f); bar.refresh()
            if abar is not None:
                abar.update(frac=f)
            if progress:
                progress(n, f)
        try:
            gr = prob.grid(name, logN=logN, T=T, windows=wins, n_jobs=g.n_jobs, base_theta=theta, progress=gprog)
        except BaseException:
            if abar is not None:
                abar.close("failed", ok=False)
            raise
        finally:
            if bar is not None:
                bar.close()
        if abar is not None:
            abar.update(frac=1.0); abar.close()
        run.grids[name] = gr
        b = gr.best
        absorber = comp.kind == "absorption"
        run.say(f"  best log N={b['logN']:.2f} T={b['T']:.0f} K" + ("" if absorber else f" log R={b['logR']:.2f}")
                + f" chi2_red={b['chi2_red']:.2f}" + (" (at grid edge)" if b["at_edge"] else ""))
        P, log_s = prob.params_from_theta(theta)
        P[name].update({"logN": b["logN"], "T": b["T"]} if absorber else {"logN": b["logN"], "T": b["T"], "logR": b["logR"]})
        if "logNA" in P[name]:
            from .fit import logNA_from
            P[name]["logNA"] = logNA_from(b["logN"], b["logR"])
        theta = np.clip(prob.theta_from_params(P, log_s), prob.lo + 1e-6, prob.hi - 1e-6)
        if not np.isfinite(prob.log_prior(theta)):
            run.say("  ordering constraint violated by grid result; keeping previous values for this component")
            theta = run.theta if run.theta is not None else prob.theta0()
    run.theta = theta
    return run


def seed_point_theta(problem, theta, seed_points: dict) -> np.ndarray:
    """`theta` with the parameters listed in fit.optimise.seed_points replaced (clipped to the bounds)."""
    P, log_s = problem.params_from_theta(np.asarray(theta, float))
    for cname, vals in (seed_points or {}).items():
        if cname not in P:
            continue
        for k, v in vals.items():
            P[cname][k] = float(v)
        if "logNA" in P[cname] and ("logN" in vals or "logR" in vals) and "logNA" not in vals:
            from .fit import logNA_from
            P[cname]["logNA"] = logNA_from(P[cname]["logN"], P[cname]["logR"])
    th = problem.theta_from_params(P, log_s)
    return np.clip(th, problem.lo + 1e-6 * (problem.hi - problem.lo), problem.hi - 1e-6 * (problem.hi - problem.lo))


def multimodal_flags(problem, starts: list[dict], best: int, dchi2: float, dT: float, dlogN: float) -> list[dict]:
    """Starts whose -2 ln P is within `dchi2` of the best but whose T or log N differ by more than `dT` / `dlogN`."""
    out = []
    tb = np.asarray(starts[best]["theta"], float)
    for st in starts:
        if st["start"] == best or not np.isfinite(st["energy"]):
            continue
        if st["energy"] - starts[best]["energy"] > dchi2:
            continue
        th = np.asarray(st["theta"], float)
        far = []
        for j, p in enumerate(problem.free):
            if p.name == "T" and abs(th[j] - tb[j]) > dT:
                far.append(f"{p.key}: {th[j]:.0f} vs {tb[j]:.0f} K")
            elif p.name in ("logN", "logNA") and abs(th[j] - tb[j]) > dlogN:
                far.append(f"{p.key}: {th[j]:.2f} vs {tb[j]:.2f}")
        if far:
            out.append({"start": st["start"], "delta_energy": float(st["energy"] - starts[best]["energy"]), "params": far})
    return out


def run_optimise_stage(run: RunResult, callback=None, progress=None):
    """Global optimiser.  0.22: fit.optimise.n_starts DE runs (seeds seed, seed+1, ...; the second from
    fit.optimise.seed_points when given); the lowest -2 ln P (chi2 when there is no noise scale and no Gaussian
    prior) is kept and every start is recorded on run.opt.starts (-> diagnostics.json "optimise")."""
    o = run.cfg.fit.optimise
    nfree = int((~run.problem.area_free_mask()).sum())
    n_starts = max(int(getattr(o, "n_starts", 1) or 1), 1)
    run.say(f"optimise: {o.method}, maxiter={o.maxiter} generations x popsize={o.popsize} x {nfree} params "
            f"= up to {o.maxiter * o.popsize * nfree} model evaluations, workers={o.workers}"
            + (f", {n_starts} starts" if n_starts > 1 else ""))

    def prog(g, n, best, dt):
        if g == 1 or g % 5 == 0 or g == n:
            per = dt / g
            run.say(f"  generation {g}/{n}: best -2lnP = {best:.1f}  ({dt:.0f}s elapsed, ~{per * (n - g):.0f}s left)")
        if progress:
            progress(g / n)
    theta0 = run.theta if run.theta is not None else run.problem.theta0()
    starts, opts = [], []
    for k in range(n_starts):
        if k == 0:
            th, origin = theta0, "config"
        elif k == 1 and o.seed_points:
            th, origin = seed_point_theta(run.problem, theta0, o.seed_points), "seed_points"
        else:
            th, origin = theta0, "random"
        if n_starts > 1:
            run.say(f"  start {k + 1}/{n_starts} (seed {o.seed + k}, {origin})")
        opt = run.problem.optimise(th, method=o.method, maxiter=o.maxiter, popsize=o.popsize, seed=o.seed + k,
                                   polish=o.polish, workers=o.workers, callback=callback, progress=prog)
        energy = float(-2.0 * opt.log_prob) if np.isfinite(opt.log_prob) else float("inf")
        starts.append({"start": k, "seed": o.seed + k, "from": origin, "chi2": float(opt.chi2), "log_prob": float(opt.log_prob),
                       "energy": energy, "chi2_red": float(opt.chi2_red), "runtime_s": float(opt.runtime_s),
                       "theta": np.asarray(opt.theta, float).tolist()})
        opts.append(opt)
        if n_starts > 1:
            run.say(f"    chi2_red={opt.chi2_red:.3f} -2lnP={energy:.1f} in {opt.runtime_s:.0f}s")
    best = int(np.argmin([st["energy"] for st in starts]))
    opt = opts[best]
    opt.starts = starts
    opt.best_start = best
    opt.multimodal = multimodal_flags(run.problem, starts, best, o.multimodal_dchi2, o.multimodal_dT, o.multimodal_dlogN) if n_starts > 1 else []
    if n_starts > 1:
        run.say(f"  best start: {best + 1} ({starts[best]['from']}); -2lnP of the starts: " + ", ".join(f"{st['energy']:.1f}" for st in starts))
    for f in opt.multimodal:
        run.say(f"  warning: possible multimodality -- start {f['start'] + 1} is within {f['delta_energy']:.1f} of the best but differs in "
                + "; ".join(f["params"]))
    run.say("  polishing (Nelder-Mead) ...") if o.polish else None
    run.opt = opt
    run.theta = opt.theta
    run.say(f"  chi2_red={opt.chi2_red:.3f} BIC={opt.bic:.1f} in {opt.runtime_s:.0f}s")
    return run


def run_laplace_stage(run: RunResult, outdir: str | None = None, progress=None):
    """Laplace (Gaussian) uncertainties at the optimiser's solution (jalebi.laplace): Hessian of -ln P by finite
    differences, flags for flat directions and prior edges.  Uses the profiled areas when fit.mcmc.linear is
    profile / marginalise, and whatever model backend is attached.  Writes laplace.json and the corner and
    correlation figures when `outdir` is given (with the MCMC contours on top when a chain exists)."""
    from .laplace import laplace
    from .linear import normalise_mode
    theta = run.opt.theta if run.opt is not None else run.theta
    if theta is None:
        raise ValueError("run the optimiser first: the Laplace approximation is taken at the optimum")
    lin = "profile" if normalise_mode(run.cfg.fit.mcmc.linear) != "sample" else "sample"
    say = progress or run.say
    say(f"laplace: Gaussian approximation at the optimum ({lin} areas)")
    lap = laplace(run.problem, theta, linear=lin, progress=say)
    run.laplace = lap
    bad = {k: v for k, v in lap.flags.items() if v}
    say(f"  condition number {lap.condition:.3g}; " + ("; ".join(f"{k}: {', '.join(v)}" for k, v in bad.items()) if bad else "no flags"))
    if outdir:
        save_laplace(run, outdir)
    return lap


def save_laplace(run: RunResult, outdir: str):
    from . import plots
    import matplotlib.pyplot as plt
    lap = run.laplace
    os.makedirs(outdir, exist_ok=True)
    lap.save(os.path.join(outdir, "laplace.json"))
    lap.summary().to_csv(os.path.join(outdir, "laplace_summary.csv"), index=False)
    fig = plots.plot_laplace_corner(lap, mcmc=run.mcmc)
    fig.savefig(os.path.join(outdir, "laplace_corner.png"), dpi=90); plt.close(fig)
    fig = plots.plot_laplace_correlation(lap)
    fig.savefig(os.path.join(outdir, "laplace_correlation.png"), dpi=90); plt.close(fig)


def run_nested_stage(run: RunResult, outdir: str | None = None, resume: bool = False):
    """Dynamic nested sampling with dynesty (fit.sampler: dynesty, jalebi.nested) in place of emcee: the same
    likelihood (fit.mcmc.linear profile or sample, any model backend), equal-weight posterior samples in the
    chain layout, ln Z +- error in diagnostics.json; optionally Delta ln Z of removing components."""
    from .linear import normalise_mode
    from .nested import evidence_without, problem_without, run_dynesty
    d = run.cfg.fit.dynesty
    lin = "profile" if normalise_mode(run.cfg.fit.mcmc.linear) != "sample" else "sample"
    theta = run.theta if run.theta is not None else run.problem.theta0()
    kw = dict(nlive=d.nlive, sample=d.sample, bound=d.bound, dlogz_init=d.dlogz_init, n_effective=d.n_effective,
              maxcall=d.maxcall, processes=d.processes, seed=d.seed, pfrac=d.pfrac, slices=d.slices, walks=d.walks,
              dynamic=d.dynamic)
    ck = os.path.join(outdir, "dynesty.save") if (outdir and getattr(d, "checkpoint", False)) else None
    run.say(f"dynesty: {'dynamic' if d.dynamic else 'static'} nested sampling, nlive={d.nlive}, sample={d.sample}, "
            f"bound={d.bound}, {lin} areas, processes={d.processes}" + (f", checkpoint {ck}" if ck else ""))
    res = run_dynesty(run.problem, theta0=theta, linear=lin, progress=run.say, checkpoint=ck, resume=resume,
                      checkpoint_every=getattr(d, "checkpoint_every", 60.0), **kw)
    run.mcmc = res
    run.theta = res.median_theta()
    run.say(f"  done: ln Z = {res.logz:.2f} +- {res.logzerr:.2f}" + (" (profile likelihood: no area prior volume)" if lin == "profile" else "")
            + f", {res.ncall:,} likelihood calls, posterior ESS {res.n_effective:.0f}, {res.runtime_s:.0f} s "
            f"(prior transform: {res.info['prior_transform']})")
    if d.evidence_without:
        full = res if lin == "sample" else run_dynesty(run.problem, theta0=theta, linear="sample", **kw)
        def factory(name):
            p = problem_without(run.problem, name)          # same pixels, noise and weights
            em = getattr(run.problem.model, "emulator", None)
            if em is not None:                              # the same tables apply (same pixels and noise)
                from .emulator import EmulatorSet
                p.model.emulator = EmulatorSet({k: t for k, t in em.tables.items() if k in p.model._units()},
                                               dict(em.exact_units), em.info)
            return p
        try:
            sig = run.problem.component_significance(run.opt.theta if run.opt is not None else theta)
        except Exception:
            sig = None
        ev = evidence_without(factory, d.evidence_without, full, {**kw, "linear": "sample"}, significance=sig, say=run.say)
        run.evidence = ev
        if outdir:
            ev.to_csv(os.path.join(outdir, "evidence.csv"), index=False)
    return run


def run_mcmc_stage(run: RunResult, progress=None, stop_event=None, outdir: str | None = None, verbose: bool = True,
                   resume: bool = False):
    """emcee stage with a progress bar (tqdm, terminal) and a log line every 10 % giving acceptance,
    mean ln-probability, the current autocorrelation-time estimate and the time left.  With fit.sampler: dynesty
    the nested sampler runs instead (run_nested_stage).  `resume` (0.22): continue chain.h5 / dynesty.save."""
    if run.cfg.fit.sampler == "dynesty":
        return run_nested_stage(run, outdir=outdir, resume=resume)
    if run.cfg.fit.sampler != "emcee":
        raise ValueError(f"unknown fit.sampler {run.cfg.fit.sampler!r}: emcee | dynesty")
    m = run.cfg.fit.mcmc
    theta = run.theta if run.theta is not None else run.problem.theta0()
    ckpt = os.path.join(outdir, "chain.h5") if (outdir and m.checkpoint) else None
    if ckpt:
        try:
            import h5py  # noqa: F401  (emcee's HDF backend needs it)
        except ImportError:
            run.say("  h5py not installed: MCMC checkpointing is off (pip install h5py)")
            ckpt = None
    nwalkers = m.nwalkers or max(4 * run.problem.ndim, 32)
    from .linear import linear_audit, normalise_mode
    linear = normalise_mode(m.linear)
    if linear != "sample":
        aud = linear_audit(run.problem)
        nlin = int(aud.linear.sum())
        nwalkers = m.nwalkers or max(4 * (run.problem.ndim - nlin), 32)
    run.say(f"mcmc: nwalkers={nwalkers}, nsteps={m.nsteps}, processes={m.processes}, moves={m.moves}{f' (gamma x{m.de_gamma:g})' if m.moves != 'stretch' else ''}, init={m.init}, "
            f"blocks={m.blocks}, linear={linear}{f' (prior {m.linear_prior})' if linear == 'marginalise' else ''} "
            f"({nwalkers * m.nsteps:,} likelihood calls for a joint run; checkpoint {ckpt or 'off'})")
    if linear != "sample":
        run.say(f"  {nlin} linear area(s) {'profiled (NNLS)' if linear == 'profile' else 'marginalised'}: "
                + ", ".join(aud.parameter[aud.linear]) + f"; sampled: {run.problem.ndim - nlin} parameters")
        kept = aud[~aud.linear & aud.parameter.str.endswith(("logR", "logNA"))]
        for r in kept.itertuples():
            run.say(f"  {r.parameter} stays in the sampler: {r.reason}")
    if m.blocks == "auto" and linear == "sample":
        groups = run.problem.independent_blocks(run.theta if run.theta is not None else run.problem.theta0())
        if len(groups) > 1:
            run.say("  independent parameter groups (sampled one after the other): " + " | ".join(
                ", ".join(run.problem.free[j].key for j in g) for g in groups))
    bar = abar = None
    if verbose and progress is None and act.enabled(logging.INFO):
        abar = act.Progress("mcmc", total=m.nsteps, unit="step", area="fit")
    elif verbose and progress is None:
        try:
            from tqdm import tqdm
            bar = tqdm(total=m.nsteps, desc="  mcmc", unit="step", dynamic_ncols=True, leave=False)
        except ImportError:
            bar = None
    t0 = time.time()
    last = {"pct": -1}

    def prog(frac, sampler):
        done = int(round(frac * m.nsteps))
        lp = sampler.get_log_prob()
        acc = float(np.mean(sampler.acceptance_fraction))
        lnp = float(np.nanmean(np.where(np.isfinite(lp[-1]), lp[-1], np.nan)))
        if bar is not None:
            bar.n = done; bar.set_postfix(acc=f"{acc:.2f}", lnp=f"{lnp:.1f}"); bar.refresh()
        if abar is not None:
            abar.update(n=done, extra=f"acc {acc:.2f} <lnP> {lnp:.1f}")
        pct = int(frac * 10)
        if verbose and pct > last["pct"] and (pct >= 1 or done == m.nsteps):
            last["pct"] = pct
            try:
                import emcee
                tau = emcee.autocorr.integrated_time(sampler.get_chain(), tol=0)
                tau_txt = f"tau~{np.nanmax(tau):.0f} ({done / np.nanmax(tau):.0f} tau so far)"
            except Exception:
                tau_txt = "tau n/a"
            dt = time.time() - t0
            eta = dt / max(done, 1) * (m.nsteps - done)
            if bar is not None:
                bar.write(f"[{time.strftime('%H:%M:%S')}]   step {done}/{m.nsteps}: acceptance {acc:.2f}, <lnP> {lnp:.1f}, {tau_txt}, {dt:.0f}s elapsed, ~{eta:.0f}s left")
                run.log.append(f"  step {done}/{m.nsteps}: acceptance {acc:.2f}, <lnP> {lnp:.1f}, {tau_txt}")
            else:
                run.say(f"  step {done}/{m.nsteps}: acceptance {acc:.2f}, <lnP> {lnp:.1f}, {tau_txt}, {dt:.0f}s elapsed, ~{eta:.0f}s left")
        if progress:
            progress(frac, sampler)

    try:
        res = run.problem.mcmc(theta, nwalkers=m.nwalkers, nsteps=m.nsteps, processes=m.processes, seed=m.seed,
                               ball=m.ball, progress=prog, checkpoint=ckpt, stop_event=stop_event, thin_by=m.thin_by,
                               moves=m.moves, init=m.init, blocks=m.blocks, de_gamma=m.de_gamma,
                               linear=linear, linear_prior=m.linear_prior, linear_prior_scale=m.linear_prior_scale,
                               vectorize=m.vectorize, resume=resume)
    finally:
        if bar is not None:
            bar.close()
        if abar is not None:
            abar.close()
    run.mcmc = res
    run.theta = res.median_theta()
    d = res.diagnostics()
    rs = [n for n in getattr(res, "meta", {}).get("resumed_steps", []) if n]
    if rs:
        run.say(f"[stage] mcmc resumed from checkpoint ({', '.join(str(n) for n in rs)} stored steps, target {m.nsteps})")
    if linear != "sample":
        if m.blocks == "auto" and len(d.get("blocks", [])) > 1:
            run.say("  independent parameter groups (sampled one after the other): " + " | ".join(", ".join(g) for g in d["blocks"]))
        neg = {u: f for u, f in d.get("linear_neg_frac", {}).items() if f > 0.01}
        if neg:
            run.say("  warning: conditional mean area <= 0 in " + ", ".join(f"{u} ({100 * f:.0f} % of samples)" for u, f in neg.items())
                    + ": not detected; positivity is only imposed on the draws (use linear: profile or sample)")
    run.say(f"  done: acceptance={d['acceptance']:.2f} max tau={np.nanmax(d['tau']):.0f} steps/tau={d['steps_over_tau']:.0f} "
            f"{'(converged length)' if d['converged_length'] else '(need >= 50 tau)'} max Rhat={np.nanmax(d['rhat']):.3f} "
            f"burn-in {d['burn']} ({res.runtime_s:.0f}s)")
    return run


def detect_and_apply(cfg: ProjectConfig, spec: Spectrum):
    """Run the molecule detection on a prepared spectrum and return (new cfg, DetectionResult)."""
    from .detect import apply_detection, detect_molecules
    d = cfg.fit.detect
    det = detect_molecules(spec, candidates=d.candidates or None, threshold=d.threshold,
                           releases=cfg.linedata.releases, oversample=d.oversample,
                           R_model=cfg.R_model, R_scale=cfg.R_scale, mode=d.mode)
    if not det.components:
        raise ValueError("auto-detect found no molecule above the threshold; nothing to fit")
    return apply_detection(cfg, det, replace_windows=d.replace_windows, keep_undetected=d.keep_undetected), det


def run_pipeline(cfg: ProjectConfig, spec: Spectrum | None = None, stages: list[str] | None = None,
                 progress=None, stop_event=None, save: bool = True) -> RunResult:
    """Run the configured stages.  Continuum refinement (model-aware) re-runs prep with the
    optimiser's gas model subtracted and refits.

    0.22: every stage writes a checkpoint into the disk folder (jalebi.resume) and, with fit.resume: auto, a
    run continues from the last valid one: detection.json -> grid.json -> de_pass1.json -> continuum_refined.npz
    -> de_pass2.json -> chain.h5 / dynesty.save.  Log lines starting with "[stage]" say what was resumed."""
    from . import resume as rs
    stages = stages or cfg.fit.stages
    spec = prepare(cfg, spec)
    outdir = cfg.output_dir(spec.name)             # e.g. results/FZ_Tau: one folder per source
    if save:
        os.makedirs(outdir, exist_ok=True)
    cfg0 = cfg                                       # the config as given (the key is taken before auto-detect)
    resume_mode = (getattr(cfg.fit, "resume", "auto") or "auto").lower()
    if resume_mode not in ("auto", "off"):
        raise ValueError(f"unknown fit.resume {resume_mode!r}: auto | off")
    key = rs.stage_key(cfg0) if save else ""
    log0: list[str] = []
    ck = rs.Checkpoint(outdir if save else None, key, enabled=save, say=log0.append)
    reading = resume_mode == "auto"
    detection = None
    if cfg.fit.auto_detect:
        d = ck.load("detection.json") if reading else None
        if d is not None:
            detection = rs.detection_from_payload(d)
            from .detect import apply_detection
            cfg = apply_detection(cfg, detection, replace_windows=cfg.fit.detect.replace_windows,
                                  keep_undetected=cfg.fit.detect.keep_undetected)
            log0.append("[stage] detection resumed from checkpoint")
        else:
            cfg, detection = detect_and_apply(cfg, spec)
            ck.save("detection.json", rs.detection_payload(detection))
            log0.append("[stage] detection done, checkpoint written")
    run = RunResult(cfg, spec, None)
    for line in log0:
        run.say(line)
    ck.say = run.say
    prob = build_problem(cfg, spec, say=run.say)
    run.problem = prob
    run.detection = detection
    run.outdir = outdir if save else None
    run.checkpoint = ck
    if save:
        run.say(f"results -> {outdir}" + (f" (fit.resume {resume_mode}, key {key})" if ck.enabled else ""))
    if detection is not None:
        run.say("auto-detect: " + ", ".join(f"{r.candidate}{'✓' if r.detected else '✗'}(ΔBIC {r.delta_BIC:+.0f})"
                                             for r in detection.table.itertuples()))
        run.say("components: " + ", ".join(c.name for c in cfg.components) + f"; windows {cfg.fit.windows}")
    run.say(f"{spec.name}: {len(prob.y)} pixels in {len(prob.windows)} windows, {prob.ndim} free parameters, "
            f"{prob.model.grid.n} fine-grid points")
    split = cfg.fit.water_split_um
    if split is not None and prob.windows and min(w[0] for w in prob.windows) < split:
        restricted = [f"{c.name} {c.windows}" for c in prob.components if c.molecule == "H2O" and c.windows]
        if restricted:
            run.say(f"water split at {split} um (fit.water_split_um): " + ", ".join(restricted))
        if not any(c.molecule == "H2O" and c.windows and c.windows[0][0] < split for c in prob.components) \
                and not any(c.molecule == "H2O" and c.Tvib is not None for c in prob.components):
            run.say(f"  note: the fit reaches below {split} um but no water component emits there -- add an H2O_rovib "
                    f"component (or Tvib) if the nu2 band should be fitted")
    free_keys = [p.key for p in prob.free]
    for it in range(cfg.continuum.refine_iterations + 1):
        if "grid" in stages and it == 0:
            g = ck.load("grid.json") if reading else None
            if g is not None and g.get("free") == free_keys and len(g.get("theta", [])) == prob.ndim:
                run.theta = np.asarray(g["theta"], float)
                run.say("[stage] grid resumed from checkpoint (grid maps are not re-drawn)")
            else:
                run_grid_stage(run, progress=(lambda n, f: progress("grid", f, n)) if progress else None)
                ck.save("grid.json", {"theta": run.theta, "free": free_keys,
                                      "best": {n: gr.best for n, gr in run.grids.items()}})
                run.say("[stage] grid done, checkpoint written")
        if "optimise" in stages:
            name = f"de_pass{it + 1}.json"
            d = ck.load(name) if reading else None
            opt = rs.opt_from_payload(d, prob) if d is not None else None
            if opt is not None:
                run.opt = opt
                run.theta = opt.theta
                run.say(f"[stage] de_pass{it + 1} resumed from checkpoint (chi2_red={opt.chi2_red:.3f})")
            else:
                run_optimise_stage(run)
                ck.save(name, rs.opt_payload(run.opt, prob, getattr(run.opt, "starts", None)))
                run.say(f"[stage] de_pass{it + 1} done, checkpoint written")
        if it < cfg.continuum.refine_iterations:
            # model-aware continuum refinement: subtract the gas model over the fit windows, re-estimate
            # the continuum with a central estimator, refit; keep it only if chi2 improves
            if run.theta is None:
                run.theta = prob.theta0()
            cname = "continuum_refined.npz" if it == 0 else f"continuum_refined{it + 1}.npz"
            c = ck.load_arrays(cname) if reading else None
            if c is not None and len(c.get("continuum", [])) == len(spec.wave):
                accepted = bool(c["accepted"])
                run.say(f"[stage] continuum refinement pass {it + 1} resumed from checkpoint ({'accepted' if accepted else 'rejected'})")
                if not accepted:
                    break
                spec_new = spec.copy()
                spec_new.continuum = np.asarray(c["continuum"], float)
                spec_new.mask = np.asarray(c["mask"], bool) if "mask" in c else spec.mask
                prob_new = build_problem(cfg, spec_new, say=run.say)
                spec, prob = spec_new, prob_new
                run.spec, run.problem = spec, prob
                continue
            gas = np.full(len(spec.wave), np.nan)
            gas[prob.used] = prob.model_flux(run.theta)
            chi2_before = prob.chi2(run.theta)
            run.say(f"continuum refinement pass {it + 1}")
            spec_new = prepare(cfg, spec.copy(), gas_model=gas)
            prob_new = build_problem(cfg, spec_new, backend="exact")
            chi2_after = prob_new.chi2(run.theta)
            run.say(f"  chi2 with old continuum {chi2_before:.0f} -> with refined continuum {chi2_after:.0f}")
            accepted = bool(chi2_after < chi2_before)
            ck.save_arrays(cname, continuum=np.asarray(spec_new.continuum, float), mask=np.asarray(spec_new.mask, bool),
                           accepted=np.array(accepted), chi2_before=np.array(chi2_before), chi2_after=np.array(chi2_after))
            run.say(f"[stage] continuum refinement pass {it + 1} done, checkpoint written")
            if accepted:
                if cfg.fit.model_backend != "exact":       # the tables are certified for this spectrum's noise
                    prob_new = build_problem(cfg, spec_new, say=run.say)
                spec, prob = spec_new, prob_new
                run.spec, run.problem = spec, prob
            else:
                run.say("  refinement did not improve chi2; keeping the previous continuum")
                break
    if cfg.fit.laplace and run.opt is not None:
        try:
            run_laplace_stage(run)
        except Exception as e:                      # quick errors must never sink a fit
            run.say(f"  laplace failed: {e}")
    if "mcmc" in stages:
        fname = "dynesty.save" if cfg.fit.sampler == "dynesty" else "chain.h5"
        writes = cfg.fit.mcmc.checkpoint if cfg.fit.sampler != "dynesty" else bool(getattr(cfg.fit.dynesty, "checkpoint", False))
        resume_chain = reading and ck.chain_resumable(fname)
        if ck.enabled and not resume_chain and os.path.exists(ck.path(fname)):
            run.say(f"[stage] {fname} belongs to another data/config/model version (or fit.resume off): starting the sampler over")
            ck.discard_chain(fname)
        if ck.enabled and writes:
            ck.mark_chain(fname)
        run_mcmc_stage(run, progress=(lambda f, s: progress("mcmc", f, None)) if progress else None,
                       stop_event=stop_event, outdir=outdir if save else None, resume=resume_chain)
    if save:
        save_results(run, outdir)
        if run.laplace is not None:
            save_laplace(run, outdir)
    return run


def save_results(run: RunResult, outdir: str):
    """Write every product.  With the emulator backend, the best-fit model, the detection test and the plots
    use the exact model (the chain was sampled with the emulator)."""
    em = getattr(run.problem.model, "emulator", None)          # stashed by exact(): the diagnostics report it
    with run.problem.model.exact():
        _save_results(run, outdir, em)


def _save_results(run: RunResult, outdir: str, em=None):
    from . import plots
    os.makedirs(outdir, exist_ok=True)
    cfg, prob = run.cfg, run.problem
    cfg.save(os.path.join(outdir, "config.yaml"))
    run.spec.save(os.path.join(outdir, "prep.csv"))
    if run.detection is not None:
        run.detection.table.to_csv(os.path.join(outdir, "detection.csv"), index=False)
    if run.theta is not None:
        P, log_s = prob.params_from_theta(run.theta)
        P_resolved = prob.model.resolve_params(P)       # ties applied (isotopologues, opacity groups)
        with open(os.path.join(outdir, "best_fit.json"), "w") as fh:
            json.dump({"params": P_resolved, "free_params": P, "log_s": log_s, "chi2": prob.chi2(run.theta),
                       "chi2_red": prob.chi2(run.theta) / max(len(prob.y) - prob.ndim, 1), "npix": len(prob.y),
                       "tau_max": prob.model.tau_flags(P)}, fh, indent=1, default=float)
        import matplotlib.pyplot as plt
        try:
            sig = prob.component_significance(run.theta)
            sig.to_csv(os.path.join(outdir, "detections.csv"), index=False)
            run.say("detection test (ΔBIC > 10 = supported by the data): " +
                    ", ".join(f"{r.component}: Δχ²={r.delta_chi2:.0f}, ΔBIC={r.delta_BIC:+.0f}" for r in sig.itertuples()))
        except Exception as e:  # pragma: no cover
            run.say(f"  detection test failed: {e}")
        fig = plots.plot_fit(prob, run.theta, title=f"{run.spec.name} best fit (fit windows)")
        fig.savefig(os.path.join(outdir, "fit_windows.png"), dpi=110); plt.close(fig)
        try:
            fig = plots.plot_full_spectrum(run.spec, prob, run.theta)
            fig.savefig(os.path.join(outdir, "fit.png"), dpi=110); plt.close(fig)
        except Exception as e:  # pragma: no cover
            run.say(f"  full-spectrum plot failed: {e}")
        m, units, _ = prob.model.evaluate(P, per_unit=True)       # 0.22: one column per component + continuum
        cols = {"wave": prob.wave, "band": prob.band, "data": prob.y, "sigma": prob.sigma, "model": m}
        if run.spec.continuum is not None:
            cols["continuum"] = np.asarray(run.spec.continuum, float)[prob.used]
        for key, f in units.items():
            cols[f"model_{key}"] = f
        if prob.cont is not None and prob.cont.n:          # 0.22: the joint continuum correction at the best fit
            beta, corr = prob.continuum_solution(run.theta, m)
            cols["continuum_correction"] = corr
            pd.DataFrame({"coefficient": prob.cont.labels, "value": beta, "prior_sigma": prob.cont.tau}).to_csv(
                os.path.join(outdir, "continuum_fit.csv"), index=False)
        pd.DataFrame(cols).to_csv(os.path.join(outdir, "model.csv"), index=False)
        try:
            fig = plots.plot_components(prob, run.theta, title=f"{run.spec.name}: components")
            fig.savefig(os.path.join(outdir, "fit_components.png"), dpi=100); plt.close(fig)
        except Exception as e:  # pragma: no cover
            run.say(f"  component plot failed: {e}")
    for name, g in run.grids.items():
        np.savez(os.path.join(outdir, f"grid_{name}.npz"), logN=g.logN, T=g.T, chi2=g.chi2, logR=g.logR, tau_max=g.tau_max)
        fig = plots.plot_grid(g); fig.savefig(os.path.join(outdir, f"grid_{name}.png"), dpi=110); plt.close(fig)
    d = {}
    if run.mcmc is not None:
        res = run.mcmc
        res.save(os.path.join(outdir, "chain.npz"))
        summ = res.summary()
        summ["reported"] = report_mask(cfg, summ, prob.components)        # 0.22 report.co
        summ.to_csv(os.path.join(outdir, "summary.csv"), index=False)
        d = res.diagnostics()
        if em is not None:                              # 0.21: which units were emulated, spot checks, fallbacks
            d["emulator"] = {"cache": em.info.get("cache", "per_disk"), "units": sorted(em.tables),
                             "exact_units": dict(em.exact_units), "spot_check": em.info.get("spot_check", {}),
                             "fallback": em.info.get("fallback", {}), "timing": em.info.get("timing", {}),
                             "kept_despite_target": em.info.get("kept_despite_target", {})}
    if run.opt is not None and getattr(run.opt, "starts", None):   # 0.22: every DE start, the winner, multimodality
        d["optimise"] = {"n_starts": len(run.opt.starts), "best_start": int(getattr(run.opt, "best_start", 0)),
                         "starts": run.opt.starts, "multimodal": getattr(run.opt, "multimodal", [])}
    cc = getattr(cfg.fit, "corner_check", None)
    if cc is not None and cc.enabled and run.theta is not None:            # 0.22: corner / pseudo-continuum flags
        from .corner import corner_check, corner_lines
        try:
            chk = corner_check(prob, run.theta, cc.bound_frac, cc.smooth_window_um, cc.smooth_threshold, cc.min_pinned,
                               smooth_molecules=tuple(getattr(cc, "smooth_molecules", ["H2O"]) or ()))
            d["corner"] = chk
            run.corner = chk
            for line in corner_lines(chk):
                run.say(line)
            if not any(v["flags"] for v in chk.values()):
                run.say("corner check: no component pinned at >= %d bounds, none mostly pseudo-continuum" % cc.min_pinned)
        except Exception as e:  # pragma: no cover
            run.say(f"  corner check failed: {e}")
    if prob.cont is not None and prob.cont.n and run.theta is not None:
        beta, corr = prob.continuum_solution(run.theta)
        cont_pix = np.asarray(run.spec.continuum, float)[prob.used] if run.spec.continuum is not None else np.full(len(corr), np.nan)
        d["continuum_fit"] = {"mode": prob.cont.mode, "n_coefficients": int(prob.cont.n), "solve": prob.cont.solve_mode,
                              "prior": prob.cont.prior, "prior_width": prob.cont.prior_width,
                              "coefficients": dict(zip(prob.cont.labels, beta.tolist())),
                              "max_abs_correction_over_continuum": float(np.nanmax(np.abs(corr) / np.maximum(np.abs(cont_pix), 1e-30))),
                              "rms_correction_over_noise": float(np.sqrt(np.mean((corr / prob.sigma) ** 2)))}
    ck = getattr(run, "checkpoint", None)
    if ck is not None and getattr(ck, "enabled", False):           # 0.22: what was resumed / written
        d["resume"] = {"key": ck.key, "resumed": list(ck.resumed), "written": list(ck.written)}
    if d:
        with open(os.path.join(outdir, "diagnostics.json"), "w") as fh:
            json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}, fh, indent=1, default=float)
    if run.mcmc is not None:
        with open(os.path.join(outdir, "tau_flags.json"), "w") as fh:
            json.dump(res.tau_flag(), fh, indent=1)
        res.correlation().to_csv(os.path.join(outdir, "correlation.csv"))
        for fn, f in (("corner.png", lambda: plots.plot_corner(res)), ("correlation.png", lambda: plots.plot_correlation(res)),
                      ("traces.png", lambda: plots.plot_traces(res)), ("posterior_predictive.png", lambda: plots.plot_posterior_predictive(prob, res))):
            try:
                fig = f(); fig.savefig(os.path.join(outdir, fn), dpi=100); plt.close(fig)
            except Exception as e:  # pragma: no cover
                run.say(f"  plot {fn} failed: {e}")
    with open(os.path.join(outdir, "log.txt"), "w") as fh:
        fh.write("\n".join(run.log) + "\n")
    if act.enabled(logging.DEBUG):
        try:
            files = sorted(os.listdir(outdir))
            act.debug("fit", "%s now holds %d files: %s", outdir, len(files), ", ".join(files))
        except OSError:
            pass


def report_mask(cfg: ProjectConfig, summary: pd.DataFrame, components) -> np.ndarray:
    """0.22 report.co: which summary rows are reported.  With report.co: NA_only the CO (and 13CO) rows other
    than log(N.A) are hidden from the printed table and the population row (they stay in summary.csv with
    reported = False): when the T prior reaches 3000 K the hot, thin CO solution makes T and N individually
    meaningless while N.A is measured."""
    keep = np.ones(len(summary), bool)
    policy = getattr(getattr(cfg, "report", None), "co", "full")
    if policy == "full":
        return keep
    if policy != "NA_only":
        raise ValueError(f"unknown report.co {policy!r}: full | NA_only")
    co = {c.name for c in components if c.molecule in ("CO", "13CO")}
    for i, name in enumerate(summary.parameter.astype(str)):
        comp, _, par = name.rpartition(".")
        if comp in co and par != "logNA":
            keep[i] = False
    return keep


def target_config(cfg: ProjectConfig, row) -> ProjectConfig:
    """Copy of `cfg` for one row of a target table (name, path, distance_pc, rv_kms), with the output
    folder of that source: `output` with '{target}' filled in, or '<output>/<name>' for a fixed output."""
    import pandas as pd
    c = cfg.model_copy(deep=True)
    get = (lambda k: row.get(k)) if hasattr(row, "get") else (lambda k: getattr(row, k, None))
    c.target.name = str(get("name")); c.target.path = str(get("path"))
    for key, attr in (("distance_pc", "distance_pc"), ("rv_kms", "rv_kms")):
        v = get(key)
        if v is not None and not (isinstance(v, float) and pd.isna(v)):
            setattr(c.target, attr, float(v))
    if not cfg.per_target_output():
        from .config import safe_name
        c.output = os.path.join(cfg.output, safe_name(c.target.name))
    return c


def catalogue_row(run: RunResult) -> dict:
    """One population-table row: per component medians/uncertainties, flags, convergence."""
    row = {"target": run.spec.name, "distance_pc": run.spec.distance_pc}
    if run.opt is not None:
        row.update({"chi2_red": run.opt.chi2_red, "bic": run.opt.bic})
    if run.mcmc is not None:
        summ = run.mcmc.summary()
        summ = summ[report_mask(run.cfg, summ, run.problem.components)]          # 0.22 report.co
        for _, r in summ.iterrows():
            row[f"{r['parameter']}"] = r["median"]; row[f"{r['parameter']}_m"] = r["minus"]; row[f"{r['parameter']}_p"] = r["plus"]
        d = run.mcmc.diagnostics()
        row.update({"acceptance": d["acceptance"], "max_rhat": float(np.nanmax(d["rhat"])), "converged": bool(d["converged_length"] and d["rhat_ok"])})
        for k, v in run.mcmc.tau_flag().items():
            row[f"{k}_thin_frac"] = v
    if run.theta is not None:
        for r in run.problem.component_significance(run.theta).itertuples():
            row[f"{r.component}_dBIC"] = r.delta_BIC
    for key, d in (getattr(run, "corner", None) or {}).items():       # 0.22
        row[f"{key}_smooth_frac"] = d["smooth_fraction"]; row[f"{key}_pinned"] = d["n_pinned"]
        row[f"{key}_corner_flags"] = ",".join(d["flags"])
    if getattr(run, "opt", None) is not None and getattr(run.opt, "multimodal", None):
        row["multimodal"] = "; ".join(f"start {f['start'] + 1}: " + ", ".join(f["params"]) for f in run.opt.multimodal)
    return row
