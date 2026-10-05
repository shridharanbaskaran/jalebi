"""Command-line interface (Typer).

  jalebi                                                     banner + list of commands
  jalebi doctor                                              check the installation
  jalebi demo                                                end-to-end fit of a synthetic spectrum
  jalebi examples ./my_examples                              copy the example scripts and configs
  jalebi synth spectrum.csv                                  write a synthetic spectrum with known answers
  jalebi linedata fetch H2O CO2 C2H2 HCN                     download + cache HITRAN lines
  jalebi linedata import H2O path/to/file.par --release hitemp
  jalebi linedata list
  jalebi init config.yaml                                   write an example config
  jalebi model --molecule H2O --logN 18 --T 600 --R 0.5     quick model spectrum to CSV
  jalebi prep config.yaml                                   ingest + continuum + masks -> prep.csv, prep.png
  jalebi fit config.yaml [--stages grid,optimise,mcmc]      run the fit, write results
  jalebi batch config.yaml targets.csv --workers 8          many disks in parallel
  jalebi serve --port 5006                                  start the web app
  jalebi detect config.yaml --write config_detected.yaml    find the molecules present, write components
  jalebi batch config.yaml targets.csv --auto-detect        per-target molecule detection in batch runs
  jalebi cube demo                                          line/velocity maps of the bundled HV Tau C cubes
  jalebi cube maps DIR -l "[Fe II] 5.34" -l "H2 S(1)"       moment, extended-emission and velocity maps of s3d cubes
  jalebi cube region DIR --circle "dx dy r" --offsets --fit config.yaml   region spectrum -> slab fit
  jalebi serve --module cube                                web app opened on the Cube maps module
  jalebi rotdiag fit example:FZ_Tau -m H2 --model two --opr species --av-free --mcmc   H2 rotation diagram
  jalebi serve --module rotdiag                             web app opened on the Rotation diagram module
"""
from __future__ import annotations

import os
from typing import List, Optional

# one BLAS/OpenMP thread per process: the model is sparse-matvec bound and the parallelism is
# across walkers/optimiser population members, not inside numpy
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import typer
from rich import print as rprint
from rich.table import Table

from ._banner import acronym_rich as _acronym_rich

app = typer.Typer(add_completion=False, no_args_is_help=False, rich_markup_mode="rich",
                  help=f"[bold #f5b543]JALEBI[/] — {_acronym_rich()}\n\n"
                       "Simultaneous LTE slab fitting of molecular emission in JWST/MIRI disk spectra.")
linedata_app = typer.Typer(help="Line-list cache management.")
app.add_typer(linedata_app, name="linedata")
from .cube.cli import cube_app  # noqa: E402  (light: typer + rich only; the cube code loads on use)
app.add_typer(cube_app, name="cube")
from .rotdiag.cli import rotdiag_app  # noqa: E402  (light as well)
app.add_typer(rotdiag_app, name="rotdiag")
from .source_cli import source_app  # noqa: E402  (light: the source code loads on use)
app.add_typer(source_app, name="source")

_DATA_DIR_HELP = ("line-list cache folder. Precedence: this option, then linedata.data_dir of --config, then $JALEBI_DATA, "
                  "then ~/.jalebi/linedata (the bundled lists are always searched too). Use the same folder the app is started with.")


def _version_cb(value: bool):
    if value:
        from . import __version__
        rprint(f"jalebi {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context,
          version: bool = typer.Option(False, "--version", "-V", callback=_version_cb, is_eager=True, help="print the version and exit")):
    """JALEBI — JWST Analysis of Line Emission with Bayesian Inference."""
    if ctx.invoked_subcommand is None:
        from . import __version__
        from ._banner import banner
        print(banner(__version__))
        print()
        rprint("[bold]First time here?[/bold]  [cyan]jalebi doctor[/cyan] (check the install) · [cyan]jalebi demo[/cyan] "
               "(fit a synthetic disk) · [cyan]jalebi serve[/cyan] (web app) · [cyan]jalebi --help[/cyan] (all commands)")


def _use_cache(data_dir: Optional[str], config: Optional[str]):
    """Point the line-list cache at an explicit folder or at the one a project config names, and say where."""
    if data_dir:
        os.environ["JALEBI_DATA"] = os.path.expanduser(data_dir)
    elif config:
        from .config import ProjectConfig
        ProjectConfig.load(config)          # sets JALEBI_DATA when the config has linedata.data_dir
    from .linedata import data_dir as _dd
    rprint(f"[dim]line-list cache: {_dd()}[/dim]")


@linedata_app.command("fetch")
def linedata_fetch(molecules: List[str] = typer.Argument(..., help="e.g. H2O CO2 13CO2 C2H2 HCN CO OH"),
                   wmin: float = 4.5, wmax: float = 30.0, release: str = "hitran",
                   force: bool = typer.Option(False, "--force", help="re-download even if this release is already cached"),
                   data_dir: Optional[str] = typer.Option(None, "--data-dir", help=_DATA_DIR_HELP),
                   config: Optional[str] = typer.Option(None, "--config", help="project YAML whose linedata.data_dir names the cache")):
    """Download lines from HITRAN (astroquery, falling back to HAPI) and cache them as Parquet.

    Only HITRAN can be downloaded; HITEMP and other releases are imported from a .par file with
    `jalebi linedata import`.  A cached HITEMP list never stands in for a requested HITRAN fetch."""
    _use_cache(data_dir, config)
    from .linedata import cache_path, load_linelist
    from .molecules import get_molecule
    for m in molecules:
        mol = get_molecule(m)
        cpath = cache_path(mol.name, release)
        had = os.path.exists(cpath) and not force
        try:
            ll = load_linelist(mol, release=release, fetch=True, wmin=wmin, wmax=wmax, fallback=False, force=force)
            what = "already cached" if had else "downloaded and cached"
            rprint(f"[green]{mol.name}[/green] ({ll.release}): {len(ll)} lines {what} at {cpath} "
                   f"[{ll.wave.min():.2f}–{ll.wave.max():.2f} µm]")
        except Exception as e:
            rprint(f"[red]{mol.name} failed:[/red] {e}")


@linedata_app.command("import")
def linedata_import(molecule: str, path: str, release: str = "hitemp",
                    data_dir: Optional[str] = typer.Option(None, "--data-dir", help=_DATA_DIR_HELP),
                    config: Optional[str] = typer.Option(None, "--config", help="project YAML whose linedata.data_dir names the cache")):
    """Import a local HITRAN/HITEMP .par file (or iSLAT-format list) into the cache."""
    _use_cache(data_dir, config)
    from .linedata import import_linelist
    out = import_linelist(molecule, path, release=release)
    rprint(f"[green]imported[/green] {molecule} -> {out}")


@linedata_app.command("list")
def linedata_list(data_dir: Optional[str] = typer.Option(None, "--data-dir", help=_DATA_DIR_HELP),
                  config: Optional[str] = typer.Option(None, "--config", help="project YAML whose linedata.data_dir names the cache")):
    _use_cache(data_dir, config)
    from .linedata import available_linelists, data_dir
    df = available_linelists()
    t = Table(title=f"line lists (user cache {data_dir()} + bundled)")
    for c in ("molecule", "release", "MB", "location", "path"):
        t.add_column(c)
    for _, r in df.iterrows():
        t.add_row(r["molecule"], r["release"], f"{r['MB']:.1f}", r["location"], r["path"])
    rprint(t)


@app.command()
def init(path: str = typer.Argument("config.yaml", help="where to write the config"),
         example: str = typer.Option("fz_tau", "--example", "-e",
                                     help="starting point: fz_tau (bundled FZ Tau MIRI data), synthetic (bundled synthetic "
                                          "spectrum), water_hot_cold, blank")):
    """Write a configuration file to start from (runs as-is on the bundled example data)."""
    from .examples import examples_source
    import shutil
    names = {"fz_tau": "FZ_Tau_quick.yaml", "water_hot_cold": "FZ_Tau_water_hot_cold.yaml"}
    src = examples_source()
    if example == "synthetic":
        shutil.copyfile(_demo_config_path(), path)
    elif example in names and src is not None and (src / "configs" / names[example]).exists():
        shutil.copyfile(src / "configs" / names[example], path)
    else:
        from .config import EXAMPLE_CONFIG, ProjectConfig
        (EXAMPLE_CONFIG if example != "blank" else ProjectConfig()).save(path)
    rprint(f"wrote [green]{path}[/green] — edit target.path / distance / rv / components, then: jalebi fit {path}")


@app.command()
def model(molecule: str = "H2O",
          logN: float = typer.Option(18.0, "--logN", "--logn", help="log10 column density [cm^-2]"),
          T: float = typer.Option(600.0, "--T", "--t", help="temperature [K]"),
          R: float = typer.Option(0.5, "--R", "--r", help="emitting radius [au]"),
          distance: float = 140.0,
          wmin: float = 13.0, wmax: float = 17.0, out: str = "model.csv", dv: float = 4.7, oversample: int = 6,
          pixel_step: Optional[float] = None,
          kind: str = typer.Option("slab", help="slab | absorption (a screen of covering fraction --fc in front of a flat continuum)"),
          fc: float = typer.Option(1.0, help="absorption: covering fraction of the continuum"),
          rv: float = typer.Option(0.0, help="radial velocity of the slab [km/s], + = redshift"),
          continuum: float = typer.Option(1.0, help="absorption: flat continuum level [Jy]")):
    """Quick single-slab model spectrum (Jy) on a MIRI-like pixel grid.  With --kind absorption the
    output holds the absorbed continuum F = F_c (1 - fc (1 - e^-tau)) and the transmission."""
    import numpy as np
    import pandas as pd
    from .model import Component, build_model
    from .instrument import resolving_power
    if pixel_step is None:
        w = [wmin]
        while w[-1] < wmax:
            w.append(w[-1] * (1 + 1.0 / (2.0 * resolving_power(w[-1]))))
        wave = np.array(w)
    else:
        wave = np.arange(wmin, wmax, pixel_step)
    comp = Component("c", molecule, logN=logN, T=T, logR=np.log10(R), fwhm=dv, rv=rv, kind=kind, fc=fc)
    cont = np.full(len(wave), continuum) if kind == "absorption" else None
    m = build_model([comp], wave, distance, [(wmin, wmax)], oversample=oversample, continuum=cont)
    f = m.evaluate()
    if kind == "absorption":
        pd.DataFrame({"wave": wave, "flux": cont + f, "continuum": cont, "transmission": 1.0 + f / cont}).to_csv(out, index=False)
    else:
        pd.DataFrame({"wave": wave, "flux": f}).to_csv(out, index=False)
    rprint(f"wrote {out}: {len(wave)} pixels, tau_max = {m.tau_flags()['c']:.2f}")


_TARGET_HELP = "spectrum to use instead of target.path (folder of x1d files, FITS or CSV)"
_NAME_HELP = ("source name for the results folder; default target.name, or with --target the FITS TARGNAME / "
              "folder name")
_OUT_HELP = "output folder; '{target}' is replaced by the source name (default: the config's output, results/{target})"


def _apply_target(cfg, target: Optional[str], name: Optional[str]):
    """--target / --name overrides.  A new --target without --name takes its name from the data, so its
    results do not land in the folder of the config's original source."""
    if target:
        cfg.target.path = target
        cfg.target.name = name or "target"
    elif name:
        cfg.target.name = name


@app.command()
def prep(config: str, target: Optional[str] = typer.Option(None, help=_TARGET_HELP),
         name: Optional[str] = typer.Option(None, help=_NAME_HELP),
         out: Optional[str] = typer.Option(None, help=_OUT_HELP), plot: bool = True):
    """Ingest, rest-frame, spike-filter, continuum, masks; write prep.csv (+ prep.png)."""
    from .config import ProjectConfig
    from .pipeline import prepare
    cfg = ProjectConfig.load(config)
    _apply_target(cfg, target, name)
    if out:
        cfg.output = out
    spec = prepare(cfg)
    outdir = cfg.output_dir(spec.name)
    os.makedirs(outdir, exist_ok=True)
    spec.save(os.path.join(outdir, "prep.csv"))
    rprint(f"{spec.name}: {len(spec.wave)} pixels, {(~spec.mask).sum()} masked, continuum={cfg.continuum.method}")
    if plot:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(16, 5))
        for b in spec.bands:
            i = spec.band_slice(b)
            ax.plot(spec.wave[i], spec.flux[i], lw=0.5)
            ax.plot(spec.wave[i], spec.continuum[i], color="k", lw=0.8)
        ax.set_xlabel("wavelength [µm]"); ax.set_ylabel("F_ν [Jy]"); ax.set_title(f"{spec.name}: continuum ({cfg.continuum.method})")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "prep.png"), dpi=110)
        rprint(f"wrote {outdir}/prep.png")


@app.command()
def detect(config: str, target: Optional[str] = typer.Option(None, help=_TARGET_HELP),
           name: Optional[str] = typer.Option(None, help=_NAME_HELP), threshold: Optional[float] = None,
           write: Optional[str] = typer.Option(None, help="write the updated config (components + windows) to this YAML"),
           keep_undetected: bool = False,
           mode: Optional[str] = typer.Option(None, help="emission | absorption | both (default: fit.detect.mode, both)")):
    """Find which molecules the spectrum contains, in emission and/or absorption, and suggest the
    components to fit.

    Templates of every molecule with a cached line list (emitting slabs and absorbing screens) are fitted
    simultaneously by NNLS; a candidate is detected when removing it raises chi2 by more than the BIC
    penalty (ΔBIC > threshold, default 10).  Absorption needs the unabsorbed continuum (upper envelope or
    `continuum.method: given`)."""
    from .config import ProjectConfig
    from .detect import apply_detection, detect_molecules
    from .pipeline import prepare
    cfg = ProjectConfig.load(config)
    _apply_target(cfg, target, name)
    if threshold is not None:
        cfg.fit.detect.threshold = threshold
    if mode is not None:
        cfg.fit.detect.mode = mode
    spec = prepare(cfg)
    d = cfg.fit.detect
    det = detect_molecules(spec, candidates=d.candidates or None, threshold=d.threshold, releases=cfg.linedata.releases,
                           oversample=d.oversample, R_model=cfg.R_model, R_scale=cfg.R_scale, mode=d.mode,
                           progress=lambda msg, f: rprint(f"  [dim]{msg}[/dim]"))
    t = Table(title=f"{spec.name}: molecule detection ({det.n_pixels} pixels, threshold ΔBIC > {d.threshold:g}, mode {d.mode})")
    for c in ("candidate", "T [K]", "log R / f_c", "Δχ²", "ΔBIC", "detected", "windows"):
        t.add_column(c)
    for r in det.table.itertuples():
        t.add_row(r.candidate, f"{r.T:.0f}", f"f_c {r.fc:.2f}" if r.kind == "absorption" else f"{r.logR:.2f}",
                  f"{r.delta_chi2:.0f}", f"{r.delta_BIC:+.0f}",
                  "[green]yes[/green]" if r.detected else "[dim]no[/dim]", r.windows)
    rprint(t)
    rprint("suggested components: " + ", ".join(f"{c.name}({c.T:.0f} K)" for c in det.components) +
           f"\nsuggested windows: {det.windows}")
    if write:
        new = apply_detection(cfg, det, replace_windows=d.replace_windows, keep_undetected=keep_undetected)
        new.save(write)
        rprint(f"wrote {write}")


@app.command()
def fit(config: str, target: Optional[str] = typer.Option(None, help=_TARGET_HELP),
        name: Optional[str] = typer.Option(None, help=_NAME_HELP), stages: Optional[str] = None,
        out: Optional[str] = typer.Option(None, help=_OUT_HELP),
        processes: Optional[int] = None, nsteps: Optional[int] = None,
        auto_detect: bool = typer.Option(False, "--auto-detect", help="detect the molecules first and fit only those")):
    """Run the fit stages from a config file."""
    from .config import ProjectConfig
    from .pipeline import run_pipeline
    cfg = ProjectConfig.load(config)
    _apply_target(cfg, target, name)
    if out:
        cfg.output = out
    if auto_detect:
        cfg.fit.auto_detect = True
    if processes:
        cfg.fit.mcmc.processes = processes; cfg.fit.optimise.workers = processes
    if nsteps:
        cfg.fit.mcmc.nsteps = nsteps
    st = stages.split(",") if stages else None
    run = run_pipeline(cfg, stages=st)
    if run.mcmc is not None:
        summ = run.mcmc.summary()
        t = Table(title=f"{run.spec.name} posterior summary")
        for c in ("parameter", "median", "minus", "plus", "at_edge"):
            t.add_column(c)
        for _, r in summ.iterrows():
            t.add_row(r["parameter"], f"{r['median']:.3f}", f"{r['minus']:.3f}", f"{r['plus']:.3f}", "!" if r["at_edge"] else "")
        rprint(t)
    rprint(f"results in [green]{run.outdir}[/green]")


@app.command()
def batch(config: str, targets: str, workers: int = 4, stages: Optional[str] = None, only_failed: bool = False,
          catalogue: str = "population.csv",
          auto_detect: bool = typer.Option(False, "--auto-detect", help="per target: detect the molecules first and fit only those")):
    """Run every row of a target table (CSV with columns name,path,distance_pc,rv_kms) in parallel."""
    import pandas as pd
    from joblib import Parallel, delayed
    from .config import ProjectConfig
    cfg = ProjectConfig.load(config)
    if auto_detect:
        cfg.fit.auto_detect = True
    tab = pd.read_csv(targets)
    st = stages.split(",") if stages else None

    def one(row):
        from .pipeline import run_pipeline, catalogue_row, target_config
        c = target_config(cfg, row)                # own target + own folder, e.g. results/DR_Tau
        c.fit.mcmc.processes = 1; c.fit.optimise.workers = 1
        outdir = c.output_dir()
        if only_failed and os.path.exists(os.path.join(outdir, "summary.csv")):
            return None
        try:
            run = run_pipeline(c, stages=st)
            return catalogue_row(run)
        except Exception as e:
            os.makedirs(outdir, exist_ok=True)
            with open(os.path.join(outdir, "FAILED.txt"), "w") as fh:
                fh.write(repr(e))
            return {"target": row["name"], "error": repr(e)}

    rows = Parallel(n_jobs=workers)(delayed(one)(r) for _, r in tab.iterrows())
    rows = [r for r in rows if r]
    root = cfg.output_root() if cfg.per_target_output() else cfg.output
    os.makedirs(root, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(root, catalogue), index=False)
    rprint(f"catalogue: {os.path.join(root, catalogue)} ({len(rows)} rows); one folder per target in {root}/")


@app.command()
def doctor(json_out: bool = typer.Option(False, "--json", help="machine-readable report"),
           quick: bool = typer.Option(False, "--quick", help="packages only")):
    """Check the installation: packages, line lists, example data, parallelism and model speed."""
    from .doctor import main as doctor_main
    args = (["--json"] if json_out else []) + (["--quick"] if quick else [])
    raise typer.Exit(doctor_main(args))


@app.command()
def examples(dest: str = typer.Argument("jalebi_examples", help="folder to copy the examples into"),
             overwrite: bool = typer.Option(False, "--overwrite", help="replace files that already exist")):
    """Copy the example scripts, configs and notebook into a folder of yours."""
    from .examples import copy_examples, list_examples
    files = copy_examples(dest, overwrite=overwrite)
    rprint(f"copied {len(files)} files to [green]{dest}[/green]" + ("" if files else " (all existed; use --overwrite)"))
    rprint("bundled data (use as [cyan]example:<name>[/cyan] in configs): " + ", ".join(n for n in list_examples() if "/" not in n))
    rprint(f"next: [cyan]cd {dest} && python 01_quick_model.py[/cyan]")


@app.command()
def synth(out: str = typer.Argument("synthetic_spectrum.csv", help="CSV to write (a *_truth.yaml is written next to it)"),
          snr: float = typer.Option(150.0, help="continuum S/N per pixel"),
          seed: int = typer.Option(0, help="random seed of the noise"),
          bands: str = typer.Option("3A,3B,3C", help="MRS sub-bands, e.g. 1A,1B,1C,2A,2B,2C,3A,3B,3C,4A,4B,4C")):
    """Write a synthetic MIRI-MRS spectrum with known LTE slab parameters (for injection–recovery tests)."""
    from .synthetic import make_synthetic_spectrum, save_synthetic
    spec, truth = make_synthetic_spectrum(bands=tuple(b.strip() for b in bands.split(",") if b.strip()), snr=snr, seed=seed)
    truth_path = os.path.splitext(out)[0] + "_truth.yaml"
    save_synthetic(out, spec, truth, truth_path)
    rprint(f"wrote [green]{out}[/green] ({len(spec.wave)} pixels, bands {', '.join(spec.bands)}) and {truth_path}")


@app.command()
def demo(out: str = typer.Option("jalebi_demo", help="output folder"),
         mcmc: bool = typer.Option(True, "--mcmc/--no-mcmc", help="also run a short MCMC and make corner plots"),
         nsteps: int = typer.Option(300, help="MCMC steps (a demo; real fits need thousands)"),
         processes: int = typer.Option(0, help="worker processes for DE and MCMC (0 = all cores, max 8)")):
    """Fit the bundled synthetic spectrum end to end and compare the result with the true parameters."""
    import time as _time
    import numpy as np
    from . import __version__
    from ._banner import banner
    from .config import ProjectConfig
    from .examples import example_path
    from .pipeline import run_pipeline
    print(banner(__version__)); print()
    import yaml
    truth = yaml.safe_load(open(example_path("synthetic/synthetic_truth.yaml")))
    cfg = ProjectConfig.model_validate(yaml.safe_load(open(_demo_config_path())))
    cfg.output = out
    nproc = processes or min(os.cpu_count() or 1, 8)
    cfg.fit.optimise.workers = nproc; cfg.fit.mcmc.processes = nproc; cfg.fit.mcmc.nsteps = nsteps
    stages = ["grid", "optimise"] + (["mcmc"] if mcmc else [])
    rprint(f"[bold]Demo:[/bold] synthetic disk (H₂O + CO₂/¹³CO₂ + C₂H₂ + HCN), stages {' → '.join(stages)}, {nproc} processes")
    t0 = _time.time()
    run = run_pipeline(cfg, stages=stages)
    P, _ = run.problem.params_from_theta(run.theta)
    t = Table(title=f"truth vs fit ({_time.time() - t0:.0f} s)")
    for c in ("component", "param", "truth", "fit") + (("± (MCMC 16/84 %)",) if run.mcmc is not None else ()):
        t.add_column(c)
    summ = run.mcmc.summary().set_index("parameter") if run.mcmc is not None else None
    for comp in truth["components"]:
        if comp.get("tie_to"):
            continue
        for k in ("T", "logN", "logR", "logNA"):
            err = ""
            key = f"{comp['name']}.{k}"
            if summ is not None and key in summ.index:
                r = summ.loc[key]; err = f"-{r['minus']:.2f} +{r['plus']:.2f}"
            if k == "logNA":        # log10(N * pi R^2): what an optically thin component constrains
                tv = comp["logN"] + np.log10(np.pi) + 2 * comp["logR"]
                fv = P[comp["name"]]["logN"] + np.log10(np.pi) + 2 * P[comp["name"]]["logR"]
                cells = ["", "log N·A", f"{tv:.2f}", f"{fv:.2f}"] + ([err] if run.mcmc is not None else [])
                t.add_row(*cells, end_section=True)
            else:
                cells = [comp["name"] if k == "T" else "", k, f"{comp[k]:g}", f"{P[comp['name']][k]:.2f}"]
                t.add_row(*(cells + ([err] if run.mcmc is not None else [])))
    rprint(t)
    tied = {c.name for c in run.problem.components if c.tie_to}
    if run.mcmc is not None:      # posterior fraction of optically thin samples
        thin = [k for k, f in run.mcmc.tau_flag().items() if k not in tied and f > 0.3]
    else:
        thin = [k for k, v in run.problem.model.tau_flags(P).items() if k not in tied and np.isfinite(v) and v < 1.2]
    if thin:
        rprint(f"[dim]{', '.join(thin)}: close to optically thin, so N and R trade off against each other and the "
               f"robust quantity is log N·A (compare those rows). Temperatures of blended bands also move with the "
               f"continuum placement (examples/03_continuum_methods.py).[/dim]")
    if run.mcmc is not None:
        d = run.mcmc.diagnostics()
        rprint(f"[dim]MCMC: acceptance {d['acceptance']:.2f}, {d['steps_over_tau']:.0f} autocorrelation times, max R̂ "
               f"{np.nanmax(d['rhat']):.2f}. A demo chain is short: real fits need ≥ 50 τ and R̂ < 1.05 "
               f"(e.g. --nsteps 5000). Error bars are statistical only; continuum placement adds systematics.[/dim]")
    rprint(f"figures and tables in [green]{out}/[/green] (fit.png, fit_windows.png, grid_*.png"
           + (", corner.png, correlation.png, traces.png, posterior_predictive.png" if mcmc else "") + ")")


def _demo_config_path() -> str:
    from .examples import example_path
    return example_path("synthetic/synthetic.yaml")


@app.command()
def about():
    """Banner, version, citation and where things are."""
    from . import __version__
    from ._banner import banner
    from .linedata import BUNDLED_DIR, data_dir
    from .examples import DATA_DIR
    print(banner(__version__)); print()
    rprint(f"package      {os.path.dirname(os.path.abspath(__file__))}\nline lists   {data_dir()} (user) + {BUNDLED_DIR} (bundled)\n"
           f"example data {DATA_DIR}\n")
    rprint("If you use JALEBI in a publication, please cite it (see CITATION.cff) together with HITRAN/HITEMP, emcee and "
           "the methods you used (see the README's reference list).")


@app.command()
def serve(port: int = 5006, data_root: Optional[str] = None, config: Optional[str] = None, show: bool = False,
          address: str = "localhost", allow_websocket_origin: Optional[str] = None,
          data_dir: Optional[str] = typer.Option(None, "--data-dir", help=_DATA_DIR_HELP),
          module: Optional[str] = typer.Option(None, "--module", "-m", help="module to open: source (default) | lte (LTE slab fit) | cube | rotdiag"),
          tab: str = typer.Option("data", help="tab of the LTE slab fit to open: data | continuum | model | fit | results | batch "
                                               "(or a module name: cube, rotdiag)"),
          cube_path: Optional[str] = typer.Option(None, "--cube", help="cube folder for the Cube module (default: bundled HV Tau C)"),
          rotdiag_config: Optional[str] = typer.Option(None, "--rotdiag-config", help="rotation-diagram YAML for the Rotation diagram module"),
          source: Optional[str] = typer.Option(None, "--source", "-s", help="open this target folder on start (the Source page; "
                                                                            "the analyses then share its data in memory)"),
          log_level: str = typer.Option("debug", "--log-level", "-l",
                                        help="how much of what the app does is printed in this terminal: trace (also slider drags, "
                                             "typing, pan/zoom) | debug (default: every action, callback, file read, cache hit) | "
                                             "info (actions, main steps, timings) | warning | error"),
          log_file: Optional[str] = typer.Option(None, "--log-file", help="also write the activity log to this file"),
          quiet: bool = typer.Option(False, "--quiet", "-q", help="only warnings and errors (same as --log-level warning)")):
    """Start the interactive web app (Panel): Source page, then LTE slab fit · Cube maps · Rotation diagram.

    The terminal shows what the app is doing: what you click and type in the browser, each step with its timing,
    files read, cache hits, fit/map progress bars, warnings and errors (see --log-level)."""
    from . import activity as act
    if log_level.lower() not in act.LEVELS:
        raise typer.BadParameter(f"{log_level!r}: use one of {', '.join(act.LEVELS)}", param_hint="--log-level")
    act.configure("warning" if quiet else log_level.lower(), log_file=log_file)
    _use_cache(data_dir, config)
    import panel as pn
    from . import __version__
    from .app import make_app
    from .linedata import data_dir as _ld_dir
    kwargs = dict(port=port, show=show, address=address)
    if allow_websocket_origin:
        kwargs["websocket_origin"] = allow_websocket_origin.split(",")
    rprint(f"[bold]JALEBI web app[/bold] on http://{address}:{port}  (Ctrl+C to stop)")
    act.install_web_hooks()
    act.banner_lines(**{"jalebi": f"{__version__} (panel {pn.__version__})", "url": f"http://{address}:{port}",
                        "log level": ("warning" if quiet else log_level) + (f" · also written to {log_file}" if log_file else ""),
                        "data root": data_root or "(bundled examples)", "line lists": _ld_dir(),
                        "start module": module or (tab if tab and tab.lower() != "data" else "source page"), "source": source, "config": config, "cube folder": cube_path,
                        "source cache": f"{os.environ.get('JALEBI_SOURCE_CACHE', 3)} sources · cubes preloaded up to "
                                        f"{os.environ.get('JALEBI_CUBE_CACHE_MB', 6000)} MB",
                        "threads": f"{os.cpu_count()} CPUs; BLAS threads {os.environ.get('OMP_NUM_THREADS')}"})
    if source:
        from .source import open_source
        with act.step(f"opening {source} before the first browser session", "server"):
            open_source(source)              # read the headers once, before the first browser session
    pn.serve(lambda: make_app(data_root=data_root, config_path=config, start_tab=tab, cube_path=cube_path, start_module=module,
                              rotdiag_config=rotdiag_config, source_path=source), title="JALEBI", **kwargs)


def main():
    app()


if __name__ == "__main__":
    main()
