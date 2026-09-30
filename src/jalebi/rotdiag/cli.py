"""`jalebi rotdiag ...` — rotation diagrams from the terminal.

  jalebi rotdiag molecules                                   molecules, line lists, defaults
  jalebi rotdiag lines H2 example:FZ_Tau                     the features that would be measured
  jalebi rotdiag fit example:FZ_Tau --molecule H2 --model two --opr species --av-free --mcmc
  jalebi rotdiag fit --fluxes h2.csv --flux-unit "erg s-1 cm-2" --molecule H2 --distance 140
  jalebi rotdiag init rd.yaml --example h2  ;  jalebi rotdiag run rd.yaml
  jalebi rotdiag demo                                        synthetic H2 spectrum: recovered vs true parameters
"""
from __future__ import annotations

from typing import List, Optional

import typer
from rich import print as rprint
from rich.table import Table

rotdiag_app = typer.Typer(help="Rotation (population) diagrams of H2, CO, OH, H2O ...: find, measure and fit lines "
                               "(N, T, A_V, OPR, optical depth; least squares + MCMC).", no_args_is_help=True)


def _print_fit(res):
    if res.fit is None:
        rprint("[red]no fit[/red]: " + "; ".join(res.messages)); return
    rprint(res.fit.summary())
    if res.comparison is not None:
        t = Table(title="model comparison (lower BIC is better)")
        for c in ("model", "n_free", "chi2_red", "bic", "dBIC", "params"):
            t.add_column(c)
        for r in res.comparison.itertuples():
            t.add_row(r.model, f"{r.n_free:.0f}" if r.n_free == r.n_free else "-", f"{r.chi2_red:.2f}", f"{r.bic:.1f}",
                      f"{getattr(r, 'dBIC', float('nan')):.1f}", str(r.params))
        rprint(t)
    if res.outdir:
        rprint(f"[green]results[/green] → {res.outdir}")


@rotdiag_app.command("molecules")
def molecules():
    """Molecules with a rotation-diagram preset (any molecule with a cached line list also works)."""
    from .species import PRESETS, load_species_linelist, releases_for
    t = Table(title="rotation-diagram molecules")
    for c in ("molecule", "line list", "lines", "λ range [µm]", "default bands", "spin", "ranking T"):
        t.add_column(c)
    for k, p in PRESETS.items():
        try:
            ll = load_species_linelist(k)
            n, rng = str(len(ll)), f"{ll.wave.min():.2f}–{ll.wave.max():.1f}"
            rel = ", ".join(releases_for(k))
        except Exception as ex:
            n, rng, rel = "-", f"({ex})", p.release
        t.add_row(k, rel, n, rng, ", ".join(p.bands) or "all", p.spin or "-", f"{p.t_ref:g} K")
    rprint(t)
    for k, p in PRESETS.items():
        if p.note:
            rprint(f"[bold]{k}[/bold]: {p.note}")


@rotdiag_app.command("curves")
def curves():
    """Bundled extinction curves (or pass the path of a CSV: wavelength [µm], A_λ/A_V)."""
    from .extinction import available_curves, get_curve
    t = Table(title="extinction curves A_λ/A_V")
    for c in ("name", "curve", "range [µm]", "A_K/A_V", "A(9.66)/A_V", "A(17.0)/A_V"):
        t.add_column(c)
    for k, lab in available_curves().items():
        c = get_curve(k)
        t.add_row(k, lab, f"{c.wave[0]:.2f}–{c.wave[-1]:.1f}", f"{c.ak_av:.3f}", f"{float(c(9.665)):.4f}", f"{float(c(17.035)):.4f}")
    rprint(t)


@rotdiag_app.command("lines")
def lines(molecule: str = typer.Argument(..., help="H2, CO, OH, H2O, ..."),
          spectrum: Optional[str] = typer.Argument(None, help="spectrum (x1d folder, CSV, FITS) to restrict to its coverage"),
          wmin: Optional[float] = None, wmax: Optional[float] = None,
          bands: Optional[List[str]] = typer.Option(None, "--band", help="vibrational band 'vup-vlow' (repeat); default: the preset's"),
          all_bands: bool = typer.Option(False, "--all-bands", help="every vibrational band"),
          t_ref: Optional[float] = typer.Option(None, help="ranking temperature [K]"),
          max_lines: int = typer.Option(60, "--max"), release: Optional[str] = None,
          out: Optional[str] = typer.Option(None, help="write the table as CSV")):
    """List the features (lines and blends) that a rotation diagram of MOLECULE would use."""
    from .features import Selection, find_features
    spec = None
    if spectrum:
        from ..data import load_spectrum
        spec = load_spectrum(spectrum)
    sel = Selection(wmin=wmin, wmax=wmax, bands=[] if all_bands else (list(bands) if bands else None), t_ref=t_ref, max_features=max_lines)
    F, M = find_features(molecule, spec, sel, release=release)
    t = Table(title=f"{molecule}: {len(F)} features ({F.attrs.get('release', '')} line list)")
    for c in ("label", "λ [µm]", "E_u [K]", "A [s⁻¹]", "g_u", "members", "spin", "band", "rel. strength", "blends / contaminants"):
        t.add_column(c)
    for r in F.itertuples():
        t.add_row(r.label, f"{r.wave:.5f}", f"{r.eu:.0f}", f"{r.a:.3g}", f"{r.gu:g}", str(r.n_members), r.spin, r.band,
                  f"{r.rel_strength:.3g}", ", ".join(x for x in (r.neighbours, r.contaminants) if x))
    rprint(t)
    if out:
        F.to_csv(out, index=False); rprint(f"written {out}")


@rotdiag_app.command("fit")
def fit(spectrum: Optional[str] = typer.Argument(None, help="spectrum: x1d folder, CSV, FITS (or example:...)"),
        molecule: str = typer.Option("H2", "--molecule", "-m"),
        fluxes: Optional[str] = typer.Option(None, help="CSV of measured fluxes (label or wave, flux, err) instead of a spectrum"),
        flux_unit: str = typer.Option("W m-2", help="unit of the --fluxes table, e.g. 'erg s-1 cm-2' or '1e-17 erg s-1 cm-2'"),
        model: str = typer.Option("single", help="single | two | powerlaw"),
        opr: str = typer.Option("thermal", help="thermal | species | offset"),
        opr_value: float = typer.Option(3.0, help="OPR (start value, or the fixed value with --fix-opr)"),
        fix_opr: bool = typer.Option(False, "--fix-opr"),
        av: float = typer.Option(0.0, help="A_V [mag] (start value with --av-free)"),
        av_free: bool = typer.Option(False, "--av-free"),
        extinction: str = typer.Option("G23", help="G23 | G23_Rv5.5 | G21 | CT06 | F11 | path of a CSV (µm, A_λ/A_V)"),
        opacity: bool = typer.Option(False, "--opacity", help="curve-of-growth optical depth correction"),
        fwhm: float = typer.Option(10.0, help="intrinsic line FWHM [km/s] for the optical depth"),
        geometry: str = typer.Option("number", help="number | radius | aperture | intensity"),
        radius: float = typer.Option(1.0, help="emitting radius [au] (geometry radius; and the τ area for number)"),
        aperture: float = typer.Option(0.5, help="aperture radius [arcsec] (geometry aperture)"),
        distance: Optional[float] = typer.Option(None, help="distance [pc] (default: the spectrum's, else 140)"),
        rv: float = typer.Option(0.0, help="systemic velocity [km/s] removed from the spectrum"),
        method: str = typer.Option("gauss", help="gauss | gauss_free | integrate"),
        velocity: str = typer.Option("auto", help="line velocity [km/s] or auto"),
        width: str = typer.Option("auto", help="line width / instrumental or auto"),
        wmin: Optional[float] = None, wmax: Optional[float] = None,
        bands: Optional[List[str]] = typer.Option(None, "--band"),
        skip: Optional[List[str]] = typer.Option(None, "--skip", help="labels of lines to leave out of the fit (repeat)"),
        mcmc: bool = typer.Option(False, "--mcmc/--no-mcmc"), steps: int = 3000, walkers: int = 48, burn: int = 1000,
        compare: bool = typer.Option(False, "--compare", help="also fit single, two and power law; tabulate BIC"),
        sys_frac: float = typer.Option(0.10, help="relative flux systematic added in quadrature"),
        out: Optional[str] = typer.Option(None, help="results folder (default results/{target}/rotdiag/{molecule})"),
        save_config: Optional[str] = typer.Option(None, help="also write the equivalent config YAML")):
    """Find, measure and fit the lines of one molecule in a spectrum (or fit a table of fluxes)."""
    from .config import RotDiagConfig
    from .pipeline import run_rotdiag
    if not spectrum and not fluxes:
        raise typer.BadParameter("give a spectrum or --fluxes")
    num = lambda s: s if s == "auto" else float(s)  # noqa: E731
    d = dict(molecule=molecule,
             spectrum=dict(path=spectrum or "", rv_kms=rv, distance_pc=distance),
             fluxes=dict(path=fluxes, unit=flux_unit) if fluxes else None,
             lines=dict(wmin=wmin, wmax=wmax, bands=list(bands) if bands else None),
             measure=dict(method=method, velocity=num(velocity), width=num(width), skip=list(skip or [])),
             geometry=dict(mode=geometry, distance_pc=distance, R_au=radius, aperture_arcsec=aperture),
             fit=dict(model=model, opr=opr, opr_value=opr_value, opr_free=not fix_opr, av=av, av_free=av_free, extinction=extinction,
                      opacity=opacity, fwhm_kms=fwhm, sys_frac=sys_frac, compare=compare),
             mcmc=dict(enabled=mcmc, steps=steps, walkers=walkers, burn=burn))
    cfg = RotDiagConfig.model_validate(d)
    if save_config:
        cfg.save(save_config); rprint(f"config → {save_config}")
    res = run_rotdiag(cfg, outdir=out)
    F = res.features
    rprint(f"{molecule}: {len(F)} features, {int(F['detected'].sum())} detected, {int(F['use'].sum())} used"
           + (f"; line velocity {F.attrs['velocity_kms']:.1f} km/s" if "velocity_kms" in F.attrs else ""))
    for m in res.messages:
        rprint(f"[yellow]{m}[/yellow]")
    _print_fit(res)


@rotdiag_app.command("run")
def run(config: str = typer.Argument(..., help="rotation-diagram YAML (jalebi rotdiag init)"),
        out: Optional[str] = typer.Option(None, help="results folder (overrides output:)")):
    """Run a rotation-diagram config: lines -> fluxes -> fit (+ MCMC, model comparison) -> results folder."""
    from .config import RotDiagConfig
    from .pipeline import run_rotdiag
    res = run_rotdiag(RotDiagConfig.load(config), outdir=out)
    for m in res.messages:
        rprint(f"[yellow]{m}[/yellow]")
    _print_fit(res)


@rotdiag_app.command("init")
def init(path: str = typer.Argument("rotdiag.yaml"), example: str = typer.Option("h2", help="h2 | h2_fluxes | co | oh | h2o")):
    """Write an example rotation-diagram config."""
    from .config import example_config
    example_config(example).save(path)
    rprint(f"written {path} (example '{example}'): edit it, then [cyan]jalebi rotdiag run {path}[/cyan]")


@rotdiag_app.command("demo")
def demo(out: str = typer.Option("jalebi_rotdiag_demo", help="results folder"), steps: int = 2000):
    """Fit the bundled synthetic H2 spectrum (two temperatures, A_V 10, OPR 2.3) and compare with the truth."""
    import yaml
    from ..examples import example_path
    from .config import example_config
    from .pipeline import run_rotdiag
    cfg = example_config("h2"); cfg.mcmc.steps = steps; cfg.mcmc.burn = min(cfg.mcmc.burn, steps // 3)
    res = run_rotdiag(cfg, outdir=out)
    _print_fit(res)
    with open(example_path("synthetic/rotdiag_H2_synthetic_truth.yaml")) as fh:
        truth = yaml.safe_load(fh)["params"]
    if res.fit is not None:
        t = Table(title="recovered vs true")
        for c in ("parameter", "fit", "truth", "pull"):
            t.add_column(c)
        for r in res.fit.table().itertuples():
            if r.param in truth and r.free:
                e = r.err_hi if truth[r.param] > r.median else r.err_lo
                t.add_row(r.param, f"{r.median:.4g} −{r.err_lo:.3g} +{r.err_hi:.3g}", f"{truth[r.param]:g}",
                          f"{(r.median - truth[r.param]) / e:+.2f} σ" if e > 0 else "-")
        rprint(t)
