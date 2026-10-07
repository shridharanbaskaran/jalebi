"""emcee vs dynesty on AS 209 and FZ Tau (jalebi 0.20): speed, effective samples, evidence, modes, published values.

    python runs/sampler_benchmark.py list                                  # the cells
    python runs/sampler_benchmark.py run  --cells 'FZ_Tau/*/emulator/*'    # run cells (glob on disk/sampler/backend/linear)
    python runs/sampler_benchmark.py run  --all --processes 8              # everything (hours with the exact model)
    python runs/sampler_benchmark.py report                                # bench.csv, medians.csv, corner overlays, REPORT.md
    python runs/sampler_benchmark.py evidence --disks AS_209               # Delta ln Z of one molecule per disk (+ Delta BIC)

Cells = disk x sampler x backend x linear:
  disk     AS_209  rebuilt from results/validation_blind/AS_209 (config.yaml after auto-detection + model.csv + best_fit.json:
                   chi^2 at the 0.15 optimum is reproduced exactly; H2O ro-vib + hot water, CO, HCN; 13 free parameters)
           FZ_Tau  the bundled FZ Tau MIRI x1d, FZ_Tau_quick continuum; two-temperature water (hot > warm, HITEMP),
                   CO2 + 13CO2 (ratio 70), C2H2, HCN on 13.45-17.5 and 21-27 um (15 free parameters); optimiser once
           (any jalebi YAML with --config NAME=path.yaml, e.g. the expanded validation_known configs on your machine)
  sampler  emcee   moves de, init scaled, blocks auto, --nsteps (default 6000) x 4 ndim walkers, from the optimum
           dynesty dynamic, rslice, multi-ellipsoid, nlive --nlive (500); seeds 0, 1, 2 (ln Z scatter)
  backend  exact | emulator (0.18 tables, built once into --emulator-dir)
  linear   sample | profile (0.17)

Metrics: wall time, likelihood calls, ESS per CPU-second (emcee: min over the sampled parameters of steps x walkers / tau
after burn-in; dynesty: Kish ESS of the importance weights), emcee tau / split-R-hat / steps-per-tau, dynesty ln Z +- error
and its scatter over seeds, posterior medians and 68 % intervals of every parameter, second modes (histogram peaks),
and the published values (claude_validation_sources.md).
"""
from __future__ import annotations

import argparse
import fnmatch
import glob
import json
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

PUBLISHED = {
    # disk: [(source, component (our name), quantity, value)]
    "FZ_Tau": [("Pontoppidan+2024", "H2O_hot", "T", 953.0), ("Pontoppidan+2024", "H2O_warm", "T", 516.0),
               ("Pontoppidan+2024", "H2O_hot", "logN", np.log10(2.4e18)), ("Pontoppidan+2024", "H2O_warm", "logN", np.log10(1.6e18)),
               ("Romero-Mirza+2024", "H2O_hot", "T", 903.0), ("Romero-Mirza+2024", "H2O_warm", "T", 420.0),
               ("Romero-Mirza+2024", "H2O_hot", "logN", np.log10(5.6e18)), ("Romero-Mirza+2024", "H2O_warm", "logN", np.log10(7.6e18))],
    "AS_209": [("Romero-Mirza+2024", "H2O_hot", "T", 800.0), ("Romero-Mirza+2024", "H2O_cold (none in this fit)", "T", 250.0),
               ("Romero-Mirza+2024", "H2O_hot", "logN", np.log10(0.10e18)), ("Romero-Mirza+2024", "H2O_cold (none in this fit)", "logN", np.log10(3.79e18))],
}


# --------------------------------------------------------------------------------------------------
# Problems
# --------------------------------------------------------------------------------------------------

def as209_problem(results_dir: str):
    from jalebi.config import ProjectConfig
    from jalebi.data import Spectrum
    from jalebi.pipeline import build_problem
    d = os.path.join(results_dir, "AS_209")
    cfg = ProjectConfig.load(os.path.join(d, "config.yaml"))
    cfg.fit.use_pipeline_err = True
    m = pd.read_csv(os.path.join(d, "model.csv"))
    spec = Spectrum(m.wave.values, m.data.values, m.sigma.values, np.array([""] * len(m)), name="AS 209",
                    distance_pc=cfg.target.distance_pc, rest_frame=True, continuum=np.zeros(len(m)))
    prob = build_problem(cfg, spec, backend="exact")
    bf = json.load(open(os.path.join(d, "best_fit.json")))
    th = np.clip(prob.theta_from_params(bf["free_params"], bf.get("log_s", 0.0)), prob.lo + 1e-9, prob.hi - 1e-9)
    return cfg, spec, prob, th


def fz_tau_config():
    from jalebi.config import ComponentConfig, ProjectConfig
    cfg = ProjectConfig.load(os.path.join(REPO, "examples", "configs", "FZ_Tau_quick.yaml"))
    cfg.components = [
        ComponentConfig(name="H2O_hot", molecule="H2O", logN=18.5, T=900.0, logR=-0.6, linelist_release="hitemp"),
        ComponentConfig(name="H2O_warm", molecule="H2O", logN=18.3, T=450.0, logR=-0.1, linelist_release="hitemp",
                        bounds={"logR": [-1.5, 1.0]}),
        ComponentConfig(name="CO2", molecule="CO2", logN=17.5, T=500.0, logR=-0.6),
        ComponentConfig(name="13CO2", molecule="13CO2", tie_to="CO2", ratio=70.0, fixed=["ratio"]),
        ComponentConfig(name="C2H2", molecule="C2H2", logN=16.5, T=500.0, logR=-0.8,
                        bounds={"T": [250, 1200], "logN": [14, 19], "logR": [-2.0, 0.0]}),
        ComponentConfig(name="HCN", molecule="HCN", logN=16.5, T=600.0, logR=-0.8,
                        bounds={"T": [250, 1200], "logN": [14, 19], "logR": [-2.0, 0.0]})]
    cfg.fit.windows = [[13.45, 17.5], [21.0, 27.0]]
    cfg.fit.ordering = [["H2O_hot", "H2O_warm"]]
    cfg.fit.oversample = 4
    return cfg


def fz_tau_problem(out: str):
    from jalebi.pipeline import build_problem, prepare
    cfg = fz_tau_config()
    spec = prepare(cfg)
    prob = build_problem(cfg, spec, backend="exact")
    f = os.path.join(out, "FZ_Tau_start.npy")
    if os.path.exists(f):
        th = np.load(f)
    else:
        t0 = time.time()
        th = prob.optimise(method="de", maxiter=120, popsize=12, seed=0).theta
        os.makedirs(out, exist_ok=True); np.save(f, th)
        print(f"FZ Tau optimiser (exact) {time.time() - t0:.0f} s", flush=True)
    return cfg, spec, prob, th


def config_problem(path: str, out: str, name: str):
    from jalebi.config import ProjectConfig
    from jalebi.pipeline import build_problem, detect_and_apply, prepare
    cfg = ProjectConfig.load(path)
    spec = prepare(cfg)
    if cfg.fit.auto_detect:
        cfg, _ = detect_and_apply(cfg, spec)
    prob = build_problem(cfg, spec, backend="exact")
    f = os.path.join(out, f"{name}_start.npy")
    if os.path.exists(f):
        th = np.load(f)
    else:
        th = prob.optimise(method="de", maxiter=cfg.fit.optimise.maxiter, popsize=cfg.fit.optimise.popsize, seed=0).theta
        os.makedirs(out, exist_ok=True); np.save(f, th)
    return cfg, spec, prob, th


# --------------------------------------------------------------------------------------------------
# Cells
# --------------------------------------------------------------------------------------------------

def cells(disks, samplers=("emcee", "dynesty"), backends=("exact", "emulator"), linears=("sample", "profile")):
    return [f"{d}/{s}/{b}/{l}" for d in disks for s in samplers for b in backends for l in linears]


def n_modes(x, bins=40, prominence=0.25):
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    x = x[np.isfinite(x)]
    if len(x) < 100 or np.ptp(x) == 0:
        return 1
    h, _ = np.histogram(x, bins=bins)
    h = gaussian_filter1d(h.astype(float), 1.5)
    pk, _ = find_peaks(np.concatenate([[0], h, [0]]), prominence=prominence * h.max())
    return max(len(pk), 1)


def run_cell(cell: str, a, problems: dict):
    from jalebi.emulator import EmulatorSettings
    disk, sampler, backend, linear = cell.split("/")
    cfg, spec, prob, th = problems[disk]
    if backend == "emulator":
        if getattr(prob.model, "emulator", None) is None:
            t0 = time.time()
            em = prob.use_emulator(EmulatorSettings(cache_dir=a.emulator_dir), say=lambda m: None)
            print(f"  {disk}: emulator ready in {time.time() - t0:.0f} s; exact units: {em.exact_units}", flush=True)
        prob.model.emulator.enabled = True
    elif getattr(prob.model, "emulator", None) is not None:
        prob.model.emulator.enabled = False
    seeds = list(range(a.seeds)) if sampler == "dynesty" else [0]
    rows = []
    for seed in seeds:
        d = os.path.join(a.out, disk, f"{sampler}_{backend}_{linear}" + (f"_seed{seed}" if sampler == "dynesty" else ""))
        if os.path.exists(os.path.join(d, "metrics.json")) and not a.force:
            rows.append(json.load(open(os.path.join(d, "metrics.json")))); continue
        os.makedirs(d, exist_ok=True)
        t0 = time.time(); c0 = time.process_time()
        if sampler == "emcee":
            res = prob.mcmc(th, nsteps=a.nsteps, seed=seed, moves="de", init="scaled", blocks="auto", linear=linear,
                            processes=a.processes)
            # every block runs nsteps x its own walkers (the chain merges them); + the local-width probes of init scaled
            calls = int(a.nsteps * sum(res.meta.get("block_nwalkers") or [res.chain.shape[1]]))
        else:
            from jalebi.nested import run_dynesty
            res = run_dynesty(prob, theta0=th, linear=linear, nlive=a.nlive, sample=a.sample, seed=seed,
                              processes=a.processes, maxcall=a.maxcall,
                              progress=(lambda msg: print(f"  {cell} seed {seed}{msg}", flush=True)) if a.verbose else None)
            calls = res.ncall
        wall = time.time() - t0
        cpu = (time.process_time() - c0) if a.processes <= 1 else wall * a.processes
        dg = res.diagnostics()
        sampled = np.asarray(res.sampled, bool)
        if sampler == "emcee":
            tau = np.asarray(dg.get("tau_sampled", dg["tau"]), float)
            burn = res.burn()
            ess = float(np.nanmin((res.nsteps - burn) * res.chain.shape[1] / tau))
        else:
            tau = np.full(int(sampled.sum()), np.nan)
            ess = float(res.n_effective)
        flat = res.flat()
        modes = {n: n_modes(flat[:, i]) for i, n in enumerate(res.names)}
        m = {"cell": cell, "disk": disk, "sampler": sampler, "backend": backend, "linear": linear, "seed": seed,
             "wall_s": wall, "cpu_s": cpu, "likelihood_calls": calls, "ess": ess, "ess_per_cpu_s": ess / max(cpu, 1e-9),
             "dims": int(sampled.sum()), "tau_max": float(np.nanmax(tau)) if sampler == "emcee" else np.nan,
             "tau_med": float(np.nanmedian(tau)) if sampler == "emcee" else np.nan,
             "steps_over_tau": float(res.nsteps / np.nanmax(tau)) if sampler == "emcee" else np.nan,
             "rhat_max": float(np.nanmax(dg["rhat"])) if sampler == "emcee" else np.nan,
             "acceptance": float(res.acceptance) if sampler == "emcee" else np.nan,
             "logz": getattr(res, "logz", np.nan), "logzerr": getattr(res, "logzerr", np.nan),
             "second_modes": [n for n, k in modes.items() if k > 1]}
        res.save(os.path.join(d, "chain.npz"))
        res.summary().to_csv(os.path.join(d, "summary.csv"), index=False)
        with open(os.path.join(d, "diagnostics.json"), "w") as fh:
            json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in dg.items()}, fh, indent=1, default=float)
        with open(os.path.join(d, "metrics.json"), "w") as fh:
            json.dump(m, fh, indent=1, default=float)
        print(f"{cell} seed {seed}: {wall:.0f} s, {calls:,} calls, ESS {ess:.0f} ({m['ess_per_cpu_s']:.3g}/CPU s)"
              + (f", tau {m['tau_max']:.0f}/{m['tau_med']:.0f}, R-hat {m['rhat_max']:.3f}" if sampler == "emcee" else
                 f", ln Z {m['logz']:.2f} +- {m['logzerr']:.2f}") + (f"; second modes: {m['second_modes']}" if m["second_modes"] else ""),
              flush=True)
        rows.append(m)
    return rows


# --------------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------------

def report(out: str):
    import corner
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(out, "*", "*", "metrics.json")))]
    if not rows:
        print("no finished cells"); return
    B = pd.DataFrame(rows)
    B.to_csv(os.path.join(out, "bench.csv"), index=False)
    # ln Z scatter over seeds
    dyn = B[B.sampler == "dynesty"]
    lz = dyn.groupby(["disk", "backend", "linear"]).agg(logz_mean=("logz", "mean"), logz_std=("logz", "std"),
                                                        logzerr_mean=("logzerr", "mean"), n_seeds=("seed", "count"),
                                                        wall_s=("wall_s", "mean"), calls=("likelihood_calls", "mean"),
                                                        ess_per_cpu_s=("ess_per_cpu_s", "mean")).reset_index()
    lz.to_csv(os.path.join(out, "logz_seeds.csv"), index=False)
    # medians
    meds = []
    for f in sorted(glob.glob(os.path.join(out, "*", "*", "summary.csv"))):
        cell = os.path.relpath(os.path.dirname(f), out)
        s = pd.read_csv(f)
        s["run"] = cell
        meds.append(s[["run", "parameter", "median", "minus", "plus"]])
    M = pd.concat(meds)
    M.to_csv(os.path.join(out, "medians.csv"), index=False)
    # corner overlays: emcee vs dynesty (seed 0) per disk / backend / linear
    figs = []
    for (disk, backend, linear), _ in B.groupby(["disk", "backend", "linear"]):
        fe = os.path.join(out, disk, f"emcee_{backend}_{linear}", "chain.npz")
        fd = os.path.join(out, disk, f"dynesty_{backend}_{linear}_seed0", "chain.npz")
        if not (os.path.exists(fe) and os.path.exists(fd)):
            continue
        ze, zd = np.load(fe), np.load(fd)
        names = [str(x) for x in ze["names"]]
        keep = [i for i, n in enumerate(names) if n.endswith((".T", ".logN", ".logNA", ".logR"))][:12]
        ce = ze["chain"]; ce = ce[ce.shape[0] // 2:].reshape(-1, ce.shape[2])[:, keep]
        cd = zd["chain"].reshape(-1, zd["chain"].shape[2])[:, keep]
        rng = [(min(np.percentile(ce[:, j], 0.5), np.percentile(cd[:, j], 0.5)),
                max(np.percentile(ce[:, j], 99.5), np.percentile(cd[:, j], 99.5))) for j in range(len(keep))]
        fig = corner.corner(ce, labels=[names[i] for i in keep], color="#d97706", range=rng, plot_datapoints=False,
                            plot_density=False, levels=(0.393, 0.865), hist_kwargs={"density": True}, label_kwargs={"fontsize": 7})
        corner.corner(cd, fig=fig, color="#2563eb", range=rng, plot_datapoints=False, plot_density=False,
                      levels=(0.393, 0.865), hist_kwargs={"density": True})
        fig.suptitle(f"{disk} — {backend}, {linear}: emcee (amber) vs dynesty (blue)", fontsize=12)
        fn = f"corner_{disk}_{backend}_{linear}.png"
        fig.savefig(os.path.join(out, fn), dpi=60); plt.close(fig)
        figs.append(fn)
    # published comparison (dynesty seed 0 and emcee, profile + best backend available)
    pub_rows = []
    for disk, items in PUBLISHED.items():
        for src, comp, q, v in items:
            for run in sorted(M.run.unique()):
                if not run.startswith(disk + "/") or not (run.endswith("seed0") or "/emcee_" in run):
                    continue
                r = M[(M.run == run) & (M.parameter == f"{comp}.{q}")]
                if len(r):
                    r = r.iloc[0]
                    pub_rows.append({"disk": disk, "source": src, "parameter": f"{comp}.{q}", "published": v, "run": run.split("/")[1],
                                     "fit": r["median"], "minus": r["minus"], "plus": r["plus"], "diff": r["median"] - v})
                elif "none in this fit" in comp:
                    pub_rows.append({"disk": disk, "source": src, "parameter": f"{comp}.{q}", "published": v, "run": run.split("/")[1],
                                     "fit": np.nan, "minus": np.nan, "plus": np.nan, "diff": np.nan})
    P = pd.DataFrame(pub_rows)
    P.to_csv(os.path.join(out, "published_comparison.csv"), index=False)
    _write_markdown(out, B, lz, M, P, figs)
    print(f"report written to {os.path.join(out, 'REPORT.md')}")


def _fmt(df, floatfmt="{:.3g}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) and np.isfinite(v) else ("—" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(lines)


def _write_markdown(out, B, lz, M, P, figs):
    md = ["# Sampler benchmark: emcee vs dynesty (jalebi 0.20)", "",
          "Generated by `runs/sampler_benchmark.py report` from the finished cells in this folder.", "",
          "## Speed and convergence", ""]
    cols = ["disk", "sampler", "backend", "linear", "seed", "dims", "wall_s", "likelihood_calls", "ess", "ess_per_cpu_s",
            "tau_max", "tau_med", "steps_over_tau", "rhat_max", "logz", "logzerr"]
    md += [_fmt(B[cols].sort_values(["disk", "backend", "linear", "sampler", "seed"])), ""]
    md += ["## dynesty ln Z over seeds", "", _fmt(lz), "",
           "With `linear: profile` ln Z is the evidence of the profile likelihood (no prior volume for the areas): compare "
           "evidences only between runs with the same `linear`; molecule evidences use `linear: sample`.", ""]
    md += ["## Second modes", ""]
    sm = B[B.second_modes.apply(len) > 0][["cell", "seed", "second_modes"]]
    md += [_fmt(sm) if len(sm) else "No parameter has a second peak in any run (histogram peaks with ≥ 25 % prominence).", ""]
    md += ["## Published values", "", _fmt(P) if len(P) else "—", ""]
    md += ["## Posterior medians (all runs)", "", "See `medians.csv` (median, minus, plus per run and parameter).", ""]
    for f in figs:
        md += [f"![{f}]({f})", ""]
    open(os.path.join(out, "REPORT.md"), "w").write("\n".join(md))


def evidence(a, disks, extra):
    """Delta ln Z of removing one component per disk (static nested sampling, areas sampled, same settings for both
    fits) next to the Delta BIC of the optimum, as in Kaeufer+2024 (Sz 28)."""
    from jalebi.emulator import EmulatorSettings
    from jalebi.nested import evidence_without, problem_without, run_dynesty
    which = {"AS_209": "HCN", "FZ_Tau": "C2H2", **dict(x.split("=", 1) for x in (a.molecule or []))}
    for disk in disks:
        if disk == "AS_209":
            cfg, spec, prob, th = as209_problem(a.results_dir)
        elif disk == "FZ_Tau":
            cfg, spec, prob, th = fz_tau_problem(a.out)
        else:
            cfg, spec, prob, th = config_problem(extra[disk], a.out, disk)
        name = which[disk]
        st = EmulatorSettings(cache_dir=a.emulator_dir)
        if a.backend == "emulator":
            prob.use_emulator(st)
        kw = dict(nlive=a.nlive, sample=a.sample, dynamic=False, seed=0, processes=a.processes, maxcall=a.maxcall)
        t0 = time.time()
        full = run_dynesty(prob, theta0=th, linear="sample", **kw)

        def factory(n):
            p = problem_without(prob, n)                    # same pixels, noise and weights as the full fit
            if a.backend == "emulator":
                from jalebi.emulator import EmulatorSet
                em = prob.model.emulator
                p.model.emulator = EmulatorSet({k: t for k, t in em.tables.items() if k in p.model._units()},
                                               dict(em.exact_units), em.info)
            return p
        with prob.model.exact():
            sig = prob.component_significance(th)
        ev = evidence_without(factory, [name], full, {**kw, "linear": "sample"}, significance=sig,
                              say=lambda m: print(m, flush=True))
        ev.insert(0, "disk", disk); ev["backend"] = a.backend; ev["nlive"] = a.nlive; ev["total_s"] = time.time() - t0
        os.makedirs(a.out, exist_ok=True)
        ev.to_csv(os.path.join(a.out, f"evidence_{disk}.csv"), index=False)
        print(ev.to_string(index=False), flush=True)


# --------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["list", "run", "report", "evidence"])
    ap.add_argument("--molecule", nargs="*", default=None, help="evidence: DISK=COMPONENT (default AS_209=HCN, FZ_Tau=C2H2)")
    ap.add_argument("--backend", default="emulator")
    ap.add_argument("--cells", nargs="*", default=None, help="glob patterns on disk/sampler/backend/linear")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--disks", nargs="*", default=["AS_209", "FZ_Tau"])
    ap.add_argument("--config", nargs="*", default=[], help="NAME=path.yaml: extra disks from jalebi configs")
    ap.add_argument("--results-dir", default=os.path.join(HERE, "results", "validation_blind"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "sampler_benchmark"))
    ap.add_argument("--emulator-dir", default=None)
    ap.add_argument("--nsteps", type=int, default=6000)
    ap.add_argument("--nlive", type=int, default=500)
    ap.add_argument("--sample", default="rslice")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--maxcall", type=int, default=None)
    ap.add_argument("--processes", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="dynesty progress every 10 s")
    a = ap.parse_args()
    extra = dict(x.split("=", 1) for x in a.config)
    disks = a.disks + list(extra)
    allc = cells(disks)
    if a.what == "list":
        print("\n".join(allc)); return
    if a.what == "report":
        report(a.out); return
    if a.what == "evidence":
        evidence(a, disks, extra); return
    todo = allc if a.all else [c for c in allc if any(fnmatch.fnmatch(c, p) for p in (a.cells or []))]
    if not todo:
        raise SystemExit("no cell selected (--all or --cells PATTERN)")
    os.makedirs(a.out, exist_ok=True)
    problems = {}
    for d in dict.fromkeys(c.split("/")[0] for c in todo):
        if d == "AS_209":
            problems[d] = as209_problem(a.results_dir)
        elif d == "FZ_Tau":
            problems[d] = fz_tau_problem(a.out)
        else:
            problems[d] = config_problem(extra[d], a.out, d)
        print(f"{d}: {len(problems[d][2].y)} px, {problems[d][2].ndim} free parameters", flush=True)
    for c in todo:
        run_cell(c, a, problems)
    report(a.out)


if __name__ == "__main__":
    main()
