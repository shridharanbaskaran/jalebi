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
    backend = backend or cfg.fit.model_backend
    if backend == "emulator":
        say = say or (lambda m: act.info("fit", "%s", m))
        t0 = time.time()
        say("model backend: emulator (tables in " + (cfg.fit.emulator.cache_dir or "the default cache") + ")")
        em = prob.use_emulator(cfg.fit.emulator.settings(), say=say)
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


def run_optimise_stage(run: RunResult, callback=None, progress=None):
    o = run.cfg.fit.optimise
    nfree = int((~run.problem.area_free_mask()).sum())
    run.say(f"optimise: {o.method}, maxiter={o.maxiter} generations x popsize={o.popsize} x {nfree} params "
            f"= up to {o.maxiter * o.popsize * nfree} model evaluations, workers={o.workers}")

    def prog(g, n, best, dt):
        if g == 1 or g % 5 == 0 or g == n:
            per = dt / g
            run.say(f"  generation {g}/{n}: best -2lnP = {best:.1f}  ({dt:.0f}s elapsed, ~{per * (n - g):.0f}s left)")
        if progress:
            progress(g / n)
    opt = run.problem.optimise(run.theta, method=o.method, maxiter=o.maxiter, popsize=o.popsize, seed=o.seed,
                               polish=o.polish, workers=o.workers, callback=callback, progress=prog)
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


def run_nested_stage(run: RunResult, outdir: str | None = None):
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
    run.say(f"dynesty: {'dynamic' if d.dynamic else 'static'} nested sampling, nlive={d.nlive}, sample={d.sample}, "
            f"bound={d.bound}, {lin} areas, processes={d.processes}")
    res = run_dynesty(run.problem, theta0=theta, linear=lin, progress=run.say, **kw)
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


def run_mcmc_stage(run: RunResult, progress=None, stop_event=None, outdir: str | None = None, verbose: bool = True):
    """emcee stage with a progress bar (tqdm, terminal) and a log line every 10 % giving acceptance,
    mean ln-probability, the current autocorrelation-time estimate and the time left.  With fit.sampler: dynesty
    the nested sampler runs instead (run_nested_stage)."""
    if run.cfg.fit.sampler == "dynesty":
        return run_nested_stage(run, outdir=outdir)
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
                               vectorize=m.vectorize)
    finally:
        if bar is not None:
            bar.close()
        if abar is not None:
            abar.close()
    run.mcmc = res
    run.theta = res.median_theta()
    d = res.diagnostics()
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
    optimiser's gas model subtracted and refits."""
    stages = stages or cfg.fit.stages
    spec = prepare(cfg, spec)
    outdir = cfg.output_dir(spec.name)             # e.g. results/FZ_Tau: one folder per source
    if save:
        os.makedirs(outdir, exist_ok=True)
    detection = None
    if cfg.fit.auto_detect:
        cfg, detection = detect_and_apply(cfg, spec)
    run = RunResult(cfg, spec, None)
    prob = build_problem(cfg, spec, say=run.say)
    run.problem = prob
    run.detection = detection
    run.outdir = outdir if save else None
    if save:
        run.say(f"results -> {outdir}")
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
    for it in range(cfg.continuum.refine_iterations + 1):
        if "grid" in stages and it == 0:
            run_grid_stage(run, progress=(lambda n, f: progress("grid", f, n)) if progress else None)
        if "optimise" in stages:
            run_optimise_stage(run)
        if it < cfg.continuum.refine_iterations:
            # model-aware continuum refinement: subtract the gas model over the fit windows, re-estimate
            # the continuum with a central estimator, refit; keep it only if chi2 improves
            if run.theta is None:
                run.theta = prob.theta0()
            gas = np.full(len(spec.wave), np.nan)
            gas[prob.used] = prob.model_flux(run.theta)
            chi2_before = prob.chi2(run.theta)
            run.say(f"continuum refinement pass {it + 1}")
            spec_new = prepare(cfg, spec.copy(), gas_model=gas)
            prob_new = build_problem(cfg, spec_new, backend="exact")
            chi2_after = prob_new.chi2(run.theta)
            run.say(f"  chi2 with old continuum {chi2_before:.0f} -> with refined continuum {chi2_after:.0f}")
            if chi2_after < chi2_before:
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
        run_mcmc_stage(run, progress=(lambda f, s: progress("mcmc", f, None)) if progress else None,
                       stop_event=stop_event, outdir=outdir if save else None)
    if save:
        save_results(run, outdir)
        if run.laplace is not None:
            save_laplace(run, outdir)
    return run


def save_results(run: RunResult, outdir: str):
    """Write every product.  With the emulator backend, the best-fit model, the detection test and the plots
    use the exact model (the chain was sampled with the emulator)."""
    with run.problem.model.exact():
        _save_results(run, outdir)


def _save_results(run: RunResult, outdir: str):
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
        m = prob.model_flux(run.theta)
        pd.DataFrame({"wave": prob.wave, "data": prob.y, "sigma": prob.sigma, "model": m}).to_csv(os.path.join(outdir, "model.csv"), index=False)
    for name, g in run.grids.items():
        np.savez(os.path.join(outdir, f"grid_{name}.npz"), logN=g.logN, T=g.T, chi2=g.chi2, logR=g.logR, tau_max=g.tau_max)
        fig = plots.plot_grid(g); fig.savefig(os.path.join(outdir, f"grid_{name}.png"), dpi=110); plt.close(fig)
    if run.mcmc is not None:
        res = run.mcmc
        res.save(os.path.join(outdir, "chain.npz"))
        summ = res.summary(); summ.to_csv(os.path.join(outdir, "summary.csv"), index=False)
        d = res.diagnostics()
        with open(os.path.join(outdir, "diagnostics.json"), "w") as fh:
            json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}, fh, indent=1, default=float)
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
        for _, r in summ.iterrows():
            row[f"{r['parameter']}"] = r["median"]; row[f"{r['parameter']}_m"] = r["minus"]; row[f"{r['parameter']}_p"] = r["plus"]
        d = run.mcmc.diagnostics()
        row.update({"acceptance": d["acceptance"], "max_rhat": float(np.nanmax(d["rhat"])), "converged": bool(d["converged_length"] and d["rhat_ok"])})
        for k, v in run.mcmc.tau_flag().items():
            row[f"{k}_thin_frac"] = v
    if run.theta is not None:
        for r in run.problem.component_significance(run.theta).itertuples():
            row[f"{r.component}_dBIC"] = r.delta_BIC
    return row
