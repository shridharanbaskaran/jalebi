"""`jalebi cube ...` — line maps and velocity maps from the terminal.

  jalebi cube info  PATH                               cubes, bands, source position, catalogue lines covered
  jalebi cube lines [PATH]                             the line catalogue (and what PATH covers)
  jalebi cube init  cube.yaml --example hv_tau_c       write an example cube config
  jalebi cube run   cube.yaml                          everything in the config
  jalebi cube maps  PATH -l "[Fe II] 5.34" -l "H2 S(1)" --zero-point star
  jalebi cube maps  PATH -l "[Fe II] 5.34" --recipe cube_maps --rms-region "12 8 3"
  jalebi cube ratio PATH -l "[Fe II] 5.34" -l "[Ne II] 12.81" --rms-region "12 8 3" --sigma 5,5
  jalebi cube moment0 FOLDER/SOURCE -l 5.3402=FeII --component full     the old cube_maps.py make_moment0
  jalebi cube stack PATH -l "H2 S(1)" -l "H2 S(2)" -l "H2 S(3)" --name H2
  jalebi cube channels PATH -l "[Fe II] 5.34" --vmin -200 --vmax 200 --dv 30
  jalebi cube pv    PATH -l "[Fe II] 5.34" --pa 25 --length 5
  jalebi cube region PATH --circle "0.42 0.91 0.35" --offsets --fit config.yaml
  jalebi cube cutout PATH -l "[Ne II] 12.81" --out small_cubes/
  jalebi cube synth synthetic_cube/                    cubes with an injected ring, jet and point source
  jalebi cube demo                                     the bundled HV Tau C example end to end
PATH is a folder of *_s3d.fits cubes, one cube, or example:HV_Tau_C_cube.
The recipe of the old cube_maps.py is `--recipe cube_maps` (= --continuum aspls --window-um 0.1 --component full
--band nominal --no-psf --no-dq --zeros-valid --nan-policy propagate --min-valid 1; options after it override
it); `jalebi cube init cube.yaml --example cube_maps` writes it as a config.
"""
from __future__ import annotations

import os
from typing import List, Optional

import typer
from rich import print as rprint
from rich.table import Table

cube_app = typer.Typer(help="Line maps, velocity maps, channel maps, PV cuts and region spectra from IFU cubes (jalebi.cube).",
                       no_args_is_help=True)

def _print_summary(df):
    """Compact table of a run (fluxes in W m^-2, velocities in km/s)."""
    if df is None or df.empty:
        return
    t = Table(title="line maps")
    for c in ("line", "band", "total", "point source", "extended", "v(source)", "velocity spaxels"):
        t.add_column(c, justify="right" if c not in ("line", "band") else "left")

    def f(x):
        return "—" if x is None or x != x else f"{x:.2e}"
    for r in df.itertuples():
        v = "—" if r.v_source_kms != r.v_source_kms else f"{r.v_source_kms:+.1f} ± {r.v_source_err_kms:.1f}"
        t.add_row(r.line, r.band, f(r.total_W_m2), f(r.point_source_W_m2), f(r.extended_W_m2), v, str(r.n_spaxels_velocity))
    rprint(t)
    rprint("[dim]fluxes in W m⁻² (total: spaxels with S/N ≥ snr_min); a stack is in normalised units; "
           "v(source): centroid of the line on the source[/dim]")


_LINE_HELP = 'line name, wavelength or group, repeatable: -l "[Fe II] 5.34" -l "H2 S(1)" -l 12.8135 (see `jalebi cube lines`)'


def _cfg_from_options(path, lines, rv, distance, no_psf, snr, kin_snr, n_mc, zero_point, order, window, smooth, center, out,
                      no_png, name=None, kinematics=True, core=0.5):
    from .config import CubeConfig
    cfg = CubeConfig(path=path, lines=list(lines), rv_kms=rv, distance_pc=distance, window_kms=window, smooth_fwhm_pix=smooth,
                     name=name)
    if out:
        cfg.output = out
    cfg.psf.enabled = not no_psf
    cfg.psf.core_radius_fwhm = core
    cfg.moments.snr_min = snr
    cfg.kinematics.snr_min = kin_snr; cfg.kinematics.n_mc = n_mc; cfg.kinematics.zero_point = zero_point
    cfg.kinematics.enabled = kinematics
    cfg.continuum.order = order
    if center:
        parts = center.replace(",", " ").split()
        cfg.center = parts[0] if len(parts) == 1 else [parts[0], parts[1]]
    if no_png:
        cfg.formats = ["fits"]
    return cfg


def _parse_floats(txt, n=None, what="value"):
    vals = [float(v) for v in str(txt).replace(",", " ").split()]
    if n is not None and len(vals) != n:
        raise typer.BadParameter(f"{what}: give {n} numbers, got {txt!r}")
    return vals


def _apply_recipe(cfg, continuum=None, lam=None, component=None, window_um=None, band=None, rms_region=None, rms_sigma=3.0,
                  rms_mode="rms", no_dq=False, nan_propagate=False, recipe=None, nan_policy=None, zeros_valid=False,
                  min_valid=None):
    """Options shared by maps / stack / ratio / channels: a recipe first (`--recipe cube_maps`), then the
    explicit options on top: continuum method, moment window, cube choice, RMS circle, NaN / zero rules."""
    if recipe:
        from .config import apply_recipe
        try:
            apply_recipe(cfg, recipe)
        except ValueError as ex:
            raise typer.BadParameter(str(ex))
    if nan_policy:
        cfg.continuum.nan_policy = nan_policy
    if zeros_valid:
        cfg.zero_is_nan = False
    if min_valid is not None:
        cfg.moments.min_valid = min_valid
    if continuum:
        cfg.continuum.method = continuum
    if lam is not None:
        cfg.continuum.lam = lam
    if nan_propagate:
        cfg.continuum.nan_policy = "propagate"
    if component:
        cfg.moments.component = None if component in ("velocity", "none") else component
    if window_um is not None:
        cfg.window_um = window_um
    if band:
        cfg.band = band
    if rms_region:
        cfg.moments.rms_region = _parse_floats(rms_region, 3, "--rms-region")
        cfg.moments.rms_sigma = rms_sigma; cfg.moments.rms_mode = rms_mode
    if no_dq:
        cfg.dq_mask = False
    return cfg


_CONT_HELP = "continuum per spaxel: poly (default) | median | aspls (cube_maps.py, lam 5e6) | irsqr | asls"
_COMP_HELP = ("moment-0 window: velocity (|v| <= max(200 km/s, 1.5 FWHM), default) or native channels around the line "
              "as cube_maps.py: full (9) | slow (5, the core) | fast (the 2+2 wings)")
_BAND_HELP = "cube per line: widest margin (default) | nominal (cube_maps.py get_channel boundaries) | a band, e.g. 3A or ch3-short"
_RECIPE_HELP = ("cube_maps = the old cube_maps.py exactly: ±0.1 µm, aspls, 9-channel moment 0, nominal sub-bands, no DQ mask, "
                "zeros kept, spaxels with a NaN channel dropped, no PSF removal (other options override it)")


@cube_app.command("info")
def info(path: str = typer.Argument(..., help="folder of s3d cubes, a cube, or example:HV_Tau_C_cube")):
    """Cubes, sub-bands, spaxel scale, source position and the catalogue lines they cover."""
    from .io import CubeSet
    cs = CubeSet(path)
    t = Table(title=f"{cs.name} — {len(cs.info)} cubes")
    for c in ("band", "λ min [µm]", "λ max [µm]", "planes", "spaxels", "file"):
        t.add_column(c)
    for ci in cs.info:
        t.add_row(ci.band, f"{ci.wmin:.4f}", f"{ci.wmax:.4f}", str(ci.nz), f"{ci.nx}×{ci.ny}", os.path.basename(ci.path))
    rprint(t)
    c0 = cs.load(cs.info[0].path)
    tr = c0.target_radec
    ra, dec, x, y = c0.find_source()
    rprint(f"header target  RA {tr[0]:.6f}  Dec {tr[1]:.6f}" if tr else "header target  —")
    rprint(f"source (continuum peak near the target, {c0.band})  RA {ra:.6f}  Dec {dec:.6f}  (pixel {x:.2f}, {y:.2f}); "
           f"spaxel {c0.pixscale:.3f}″")
    cov = cs.lines_covered()
    rprint("[bold]catalogue lines covered:[/bold] " + (", ".join(ln.name for ln in cov) or "none"))


@cube_app.command("lines")
def lines(path: Optional[str] = typer.Argument(None, help="optional: mark the lines these cubes cover"),
          species: Optional[str] = typer.Option(None, help="only this species, e.g. H2 or '[Fe II]'")):
    """The line catalogue (rest vacuum wavelengths) and the stacking groups."""
    from ..lines import GROUPS, catalogue_table
    df = catalogue_table()
    if species:
        from ..lines import _norm
        df = df[df.species.map(_norm) == _norm(species)]
    covered = set()
    if path:
        from .io import CubeSet
        covered = {ln.name for ln in CubeSet(path).lines_covered()}
    t = Table(title="jalebi line catalogue")
    for c in ("name", "λ [µm]", "species", "E_u [K]", "A_ul [s⁻¹]", "note") + (("in cubes",) if path else ()):
        t.add_column(c)
    for r in df.itertuples():
        row = [r.name, f"{r.wave_um:.5f}", r.species, "" if r.eu_K != r.eu_K or r.eu_K is None else f"{r.eu_K:.0f}",
               "" if r.a_ul != r.a_ul or r.a_ul is None else f"{r.a_ul:.2e}", r.note]
        if path:
            row.append("✓" if r.name in covered else "")
        t.add_row(*row)
    rprint(t)
    rprint("groups: " + "; ".join(f"[cyan]{k}[/cyan] = {', '.join(v)}" for k, v in GROUPS.items()))
    rprint('lines not in the catalogue: give "name=wavelength", e.g. -l "CO P(10)=4.9876"')


@cube_app.command("init")
def init(path: str = typer.Argument("cube.yaml"),
         example: str = typer.Option("hv_tau_c", help="hv_tau_c | cube_maps (the old cube_maps.py recipe) | synthetic | blank"),
         force: bool = typer.Option(False, "--force")):
    """Write an example cube config."""
    from .config import example_config
    if os.path.exists(path) and not force:
        rprint(f"[red]{path} exists[/red] (use --force)"); raise typer.Exit(1)
    example_config(example).save(path)
    rprint(f"wrote [green]{path}[/green]  →  jalebi cube run {path}")


@cube_app.command("run")
def run(config: str, out: Optional[str] = typer.Option(None, help="output folder (default: output in the config)"),
        jobs: Optional[int] = typer.Option(None, "--jobs", "-j", help="parallel processes over lines")):
    """Run a cube config: maps of every line, stacks, channel maps, PV cuts, region spectra and region fits."""
    from .config import CubeConfig
    from .pipeline import run_cube
    cfg = CubeConfig.load(config)
    if jobs:
        cfg.n_jobs = jobs
    r = run_cube(cfg, outdir=out)
    _print_summary(r.table)


@cube_app.command("maps")
def maps(path: str, line: List[str] = typer.Option(..., "--line", "-l", help=_LINE_HELP),
         out: Optional[str] = typer.Option(None, help="output folder (default results/{target}/cube)"),
         rv: float = typer.Option(0.0, help="systemic velocity to remove [km/s]"),
         distance: float = typer.Option(140.0, help="distance [pc] (scale bars)"),
         no_psf: bool = typer.Option(False, "--no-psf", help="keep the point source (no PSF subtraction)"),
         core: float = typer.Option(0.5, help="PSF core radius used to scale the point source [x FWHM]"),
         snr: float = typer.Option(3.0, help="S/N threshold for moments 1/2"),
         kin_snr: float = typer.Option(5.0, help="S/N threshold for the Gaussian velocity fits"),
         n_mc: int = typer.Option(100, help="Monte Carlo realisations per spaxel for velocity errors (0 = covariance)"),
         zero_point: str = typer.Option("none", help="none | star (v = 0 at the source for each line)"),
         order: int = typer.Option(1, help="continuum polynomial order"),
         window: float = typer.Option(1500.0, help="half-width of the line window [km/s]"),
         smooth: float = typer.Option(0.0, help="spatial smoothing FWHM [spaxels]"),
         center: Optional[str] = typer.Option(None, help='"RA DEC", or auto | header | peak'),
         pa: Optional[float] = typer.Option(None, help="draw an axis at this PA on the maps (deg E of N)"),
         extent: Optional[float] = typer.Option(None, help="half-size of the plotted field [arcsec]"),
         continuum: Optional[str] = typer.Option(None, help=_CONT_HELP),
         lam: Optional[float] = typer.Option(None, help="smoothness of aspls / irsqr / asls"),
         component: Optional[str] = typer.Option(None, help=_COMP_HELP),
         window_um: Optional[float] = typer.Option(None, "--window-um", help="half-width of the window in micron instead of --window (cube_maps.py: 0.1)"),
         band: Optional[str] = typer.Option(None, help=_BAND_HELP),
         rms_region: Optional[str] = typer.Option(None, help='"x y r" pixel circle of empty sky: also write mom0_masked (cube_maps.py: "12 8 3")'),
         rms_sigma: float = typer.Option(3.0, help="threshold of mom0_masked in units of the RMS circle's level"),
         rms_mode: str = typer.Option("rms", help="level of the RMS circle: rms (nanstd) | mean | median"),
         no_dq: bool = typer.Option(False, "--no-dq", help="do not blank DQ = DO_NOT_USE pixels (as spectral_cube)"),
         recipe: Optional[str] = typer.Option(None, help=_RECIPE_HELP),
         nan_policy: Optional[str] = typer.Option(None, help="baselines: omit (fit through NaN channels) | propagate (drop the spaxel)"),
         zeros_valid: bool = typer.Option(False, "--zeros-valid", help="SCI = 0 is data, not a gap (as spectral_cube)"),
         min_valid: Optional[int] = typer.Option(None, help="finite channels a spaxel needs in the moment window (cube_maps.py: 1)"),
         no_png: bool = typer.Option(False, "--no-png")):
    """Moment 0/1/2, extended (PSF-subtracted) emission and centroid-velocity maps (FITS + PNG)."""
    from .pipeline import run_cube
    cfg = _cfg_from_options(path, line, rv, distance, no_psf, snr, kin_snr, n_mc, zero_point, order, window, smooth, center, out, no_png,
                            core=core)
    _apply_recipe(cfg, continuum, lam, component, window_um, band, rms_region, rms_sigma, rms_mode, no_dq, recipe=recipe,
                  nan_policy=nan_policy, zeros_valid=zeros_valid, min_valid=min_valid)
    if no_psf:
        cfg.psf.enabled = False
    cfg.pa_deg = pa; cfg.plot_extent_arcsec = extent
    r = run_cube(cfg)
    _print_summary(r.table)


@cube_app.command("stack")
def stack(path: str, line: List[str] = typer.Option(..., "--line", "-l", help=_LINE_HELP),
          name: str = typer.Option("stack", help="name of the stack"),
          out: Optional[str] = typer.Option(None, help="output folder (default results/{target}/cube)"), rv: float = 0.0, distance: float = 140.0,
          no_psf: bool = typer.Option(False, "--no-psf"), core: float = 0.5, snr: float = 3.0, kin_snr: float = 5.0,
          n_mc: int = 100, zero_point: str = typer.Option("none"), order: int = 1, window: float = 1500.0, smooth: float = 0.0,
          continuum: Optional[str] = typer.Option(None, help=_CONT_HELP),
          lam: Optional[float] = typer.Option(None, help="smoothness of aspls / irsqr / asls"),
          component: Optional[str] = typer.Option(None, help=_COMP_HELP),
          window_um: Optional[float] = typer.Option(None, "--window-um", help="half-width of the window in micron"),
          band: Optional[str] = typer.Option(None, help=_BAND_HELP),
          rms_region: Optional[str] = typer.Option(None, help='"x y r" pixel circle of empty sky: also write mom0_masked'),
          rms_sigma: float = 3.0, rms_mode: str = "rms",
          no_dq: bool = typer.Option(False, "--no-dq"),
          recipe: Optional[str] = typer.Option(None, help=_RECIPE_HELP),
          nan_policy: Optional[str] = typer.Option(None, help="baselines: omit | propagate"),
          zeros_valid: bool = typer.Option(False, "--zeros-valid"),
          min_valid: Optional[int] = typer.Option(None)):
    """Stack several lines of one species (e.g. H2 S(1)–S(3)) in velocity space and map the result."""
    from .pipeline import run_cube
    cfg = _cfg_from_options(path, line, rv, distance, no_psf, snr, kin_snr, n_mc, zero_point, order, window, smooth, None, out, False,
                            core=core)
    _apply_recipe(cfg, continuum, lam, component, window_um, band, rms_region, rms_sigma, rms_mode, no_dq, recipe=recipe,
                  nan_policy=nan_policy, zeros_valid=zeros_valid, min_valid=min_valid)
    if no_psf:
        cfg.psf.enabled = False
    cfg.stacks = {name: list(line)}
    cfg.lines = []
    run_cube(cfg)


@cube_app.command("channels")
def channels(path: str, line: str = typer.Option(..., "--line", "-l"), vmin: float = -300.0, vmax: float = 300.0,
             dv: Optional[float] = typer.Option(None, help="bin width [km/s] (default: native channels)"),
             source: str = typer.Option("extended", help="extended | line"), out: Optional[str] = None, rv: float = 0.0,
             distance: float = 140.0, no_psf: bool = typer.Option(False, "--no-psf"),
             slices: bool = typer.Option(False, "--slices", help="also write the native channels of the moment window, one "
                                         "2-D FITS each named by wavelength (cube_maps.py get_channel_maps)"),
             component: str = typer.Option("full", help="channels written by --slices: full (9) | slow (5) | fast (2+2)"),
             continuum: Optional[str] = typer.Option(None, help=_CONT_HELP),
             window_um: Optional[float] = typer.Option(None, "--window-um", help="half-width of the window in micron"),
             band: Optional[str] = typer.Option(None, help=_BAND_HELP),
             no_dq: bool = typer.Option(False, "--no-dq"),
             recipe: Optional[str] = typer.Option(None, help=_RECIPE_HELP),
             nan_policy: Optional[str] = typer.Option(None, help="baselines: omit | propagate"),
             zeros_valid: bool = typer.Option(False, "--zeros-valid")):
    """Channel maps (FITS cube with a VRAD axis + PNG grid), and optionally the single native channels."""
    from .pipeline import run_cube
    cfg = _cfg_from_options(path, [line], rv, distance, no_psf, 3.0, 5.0, 0, "none", 1, 1500.0, 0.0, None, out, False, kinematics=False)
    _apply_recipe(cfg, continuum, None, None, window_um, band, None, no_dq=no_dq, recipe=recipe, nan_policy=nan_policy,
                  zeros_valid=zeros_valid)
    if no_psf:
        cfg.psf.enabled = False
    cfg.channels.enabled = True; cfg.channels.vmin = vmin; cfg.channels.vmax = vmax; cfg.channels.dv_kms = dv
    cfg.channels.source = source
    cfg.channels.slices = slices; cfg.channels.slices_component = component
    run_cube(cfg)


@cube_app.command("pv")
def pv(path: str, line: str = typer.Option(..., "--line", "-l"), pa: float = typer.Option(..., help="cut PA [deg E of N]"),
       length: float = typer.Option(4.0, help="cut length [arcsec]"), width: Optional[float] = typer.Option(None, help="slit width [arcsec]"),
       source: str = "extended", out: Optional[str] = None, rv: float = 0.0, no_psf: bool = typer.Option(False, "--no-psf")):
    """Position–velocity diagram along a PA through the source."""
    from .pipeline import run_cube
    cfg = _cfg_from_options(path, [line], rv, 140.0, no_psf, 3.0, 5.0, 0, "none", 1, 1500.0, 0.0, None, out, False, kinematics=False)
    cfg.pv.enabled = True; cfg.pv.pa_deg = pa; cfg.pv.length_arcsec = length; cfg.pv.width_arcsec = width; cfg.pv.source = source
    run_cube(cfg)


@cube_app.command("ratio")
def ratio(path: str, line: List[str] = typer.Option(..., "--line", "-l", help="two lines: -l LINE1 -l LINE2 (ratio = 1 / 2)"),
          sigma: str = typer.Option("5,5", help="thresholds for line 1, line 2 [sigma]"),
          rms_region: Optional[str] = typer.Option(None, help='"x y r" pixel circle of empty sky on line 1\'s grid (cube_maps.py: '
                                                   '"12 8 3"); default: the propagated moment-0 errors'),
          rms_mode: str = typer.Option("rms", help="rms (nanstd) | mean | median of the circle"),
          key: str = typer.Option("mom0", help="map to divide: mom0 | mom0_ext (point source removed) | gflux"),
          unit: str = typer.Option("cgs", help='cgs (line-flux ratio) | "MJy/sr m" (the numbers of cube_maps.py\'s maps)'),
          percentile: str = typer.Option("80,80,80", help="display percentiles of the three panels"),
          out: Optional[str] = typer.Option(None, help="output folder (default results/{target}/cube)"),
          rv: float = 0.0, distance: float = 140.0,
          no_psf: bool = typer.Option(False, "--no-psf"),
          core: float = typer.Option(0.5, help="PSF core radius used to scale the point source [x FWHM]"),
          order: int = typer.Option(1, help="continuum polynomial order"),
          window: float = typer.Option(1500.0, help="half-width of the line window [km/s]"),
          smooth: float = typer.Option(0.0, help="spatial smoothing FWHM [spaxels]"),
          continuum: Optional[str] = typer.Option(None, help=_CONT_HELP),
          lam: Optional[float] = typer.Option(None, help="smoothness of aspls / irsqr / asls"),
          component: Optional[str] = typer.Option(None, help=_COMP_HELP),
          window_um: Optional[float] = typer.Option(None, "--window-um", help="half-width of the window in micron"),
          band: Optional[str] = typer.Option(None, help=_BAND_HELP),
          no_dq: bool = typer.Option(False, "--no-dq"),
          recipe: Optional[str] = typer.Option(None, help=_RECIPE_HELP),
          nan_policy: Optional[str] = typer.Option(None, help="baselines: omit | propagate"),
          zeros_valid: bool = typer.Option(False, "--zeros-valid"),
          min_valid: Optional[int] = typer.Option(None)):
    """Line-ratio map (line 2 reprojected onto line 1, both masked at sigma x noise) → FITS + 3-panel PNG."""
    from .config import CubeRatioConfig
    from .pipeline import run_cube
    if len(line) != 2:
        rprint("[red]give exactly two lines: -l LINE1 -l LINE2[/red]"); raise typer.Exit(1)
    cfg = _cfg_from_options(path, [], rv, distance, no_psf, 3.0, 5.0, 0, "none", order, window, smooth, None, out, False,
                            kinematics=key.startswith("g"), core=core)
    _apply_recipe(cfg, continuum, lam, component, window_um, band, None, no_dq=no_dq, recipe=recipe, nan_policy=nan_policy,
                  zeros_valid=zeros_valid, min_valid=min_valid)
    if no_psf:
        cfg.psf.enabled = False
    if key.endswith("_ext") and not cfg.psf.enabled:
        rprint(f"[red]--key {key} needs the point source removed (drop --no-psf / the recipe's PSF setting)[/red]"); raise typer.Exit(1)
    cfg.ratios = [CubeRatioConfig(lines=list(line), sigma_thresh=_parse_floats(sigma, 2, "--sigma"),
                                  rms_region=_parse_floats(rms_region, 3, "--rms-region") if rms_region else None,
                                  rms_mode=rms_mode, key=key, unit=unit, percentile=_parse_floats(percentile, 3, "--percentile"))]
    r = run_cube(cfg)
    for name, rm in r.ratios.items():
        sm = rm.summary()
        rprint(f"[bold]{name}[/bold]: {sm['n_valid']} spaxels" + (f", median {sm['median_ratio']:.3g} "
               f"(16–84 %: {sm['p16_p84'][0]:.3g}–{sm['p16_p84'][1]:.3g})" if sm["n_valid"] else ""))


@cube_app.command("moment0")
def moment0(path: str = typer.Argument(..., help="the source folder, e.g. DATA/HV_Tau_C (holds Level3_ch*_s3d.fits)"),
            line: List[str] = typer.Option(..., "--line", "-l", help='"wavelength=name", e.g. -l 5.3402=FeII (or just 5.3402)'),
            dlambda: float = typer.Option(0.1, help="half-width of the slab [um]"),
            component: str = typer.Option("full", help="full (9 channels) | slow (5) | fast (2+2 wings)"),
            out: str = typer.Option(".", help="output_path: maps go to OUT/moment0_maps/SOURCE/"),
            unit: str = typer.Option("MJy/sr m", help='"MJy/sr m" (as the old files) | "MJy/sr um" | "erg s-1 cm-2 sr-1"'),
            pattern: str = typer.Option("Level3_{channel}_s3d.fits", help="cube file name inside PATH"),
            png: bool = typer.Option(False, "--png", help="also a PNG of each map (plot_moment0_map)"),
            sigma_clip: Optional[float] = typer.Option(None, help="PNG: mask below this many x the RMS circle's level"),
            rms_region: str = typer.Option("12 8 3", help='PNG: "x y r" RMS circle'),
            jobs: int = typer.Option(1, "--jobs", "-j", help="parallel processes for the per-spaxel continuum")):
    """Moment-0 maps exactly as the old cube_maps.py make_moment0 (aspls continuum, channel windows, same file names)."""
    from .cube_maps import make_moment0, plot_moment0_map
    waves, names = [], []
    for x in line:
        w, _, n = x.partition("=")
        waves.append(float(w)); names.append(n.strip() or w.strip())
    p = os.path.abspath(path.rstrip("/"))
    maps_, files = make_moment0(waves, names, os.path.basename(p), dlambda, component=component, save=True,
                                folder_path=os.path.dirname(p), output_path=out, unit=unit, cube_pattern=pattern, n_jobs=jobs,
                                verbose=False)
    if len(waves) == 1:
        maps_, files = [maps_], [files]
    for m, f in zip(maps_, files):
        rprint(f"  {os.path.basename(m.cube_file)}  channels {m.channels}  →  [green]{f}[/green]")
        if png:
            import matplotlib.pyplot as plt
            x, y, r = _parse_floats(rms_region, 3, "--rms-region")
            fig, _ = plot_moment0_map(f, title=f"{m.transition} {m.line_wave} µm ({component})", sigma_clip=sigma_clip is not None,
                                      sigma_thresh=sigma_clip or 3, center_x=x, center_y=y, radius=r, showfig=False)
            fig.savefig(os.path.splitext(f)[0] + ".png", dpi=150, bbox_inches="tight"); plt.close(fig)


@cube_app.command("region")
def region(path: str,
           circle: Optional[str] = typer.Option(None, help='"RA DEC R" (deg deg arcsec) or, with --offsets, "dx dy R"'),
           ellipse: Optional[str] = typer.Option(None, help='"RA DEC A B PA" (semi-axes arcsec, PA deg E of N)'),
           polygon: Optional[str] = typer.Option(None, help='"RA1 DEC1 RA2 DEC2 ..."'),
           annulus: Optional[str] = typer.Option(None, help='"RA DEC R_IN R_OUT"'),
           ds9: Optional[str] = typer.Option(None, help="a DS9 region file (fk5); the first region is used"),
           offsets: bool = typer.Option(False, "--offsets", help="positions are east/north offsets [arcsec] from the source"),
           out: str = typer.Option("region_spectrum.csv", help="CSV to write"),
           name: Optional[str] = None, distance: float = 140.0,
           fit: Optional[str] = typer.Option(None, help="slab-fit config YAML: fit the region spectrum right away"),
           fit_out: Optional[str] = typer.Option(None, help="output folder of the fit"),
           png: bool = typer.Option(True, help="also write a PNG of the region and spectrum")):
    """Spectrum of a sky region over all sub-bands (→ CSV, loadable by every other command), optionally fitted."""
    from .io import CubeSet, resolve_center
    from .regions import AnnulusRegion, CircleRegion, EllipseRegion, PolygonRegion, offset_region, parse_ds9, region_spectrum
    cs = CubeSet(path)
    if ds9:
        reg = parse_ds9(open(ds9).read())[0]
    else:
        spec = [(k, v) for k, v in (("circle", circle), ("ellipse", ellipse), ("polygon", polygon), ("annulus", annulus)) if v]
        if len(spec) != 1:
            rprint("[red]give exactly one of --circle, --ellipse, --polygon, --annulus, --ds9[/red]"); raise typer.Exit(1)
        kind, txt = spec[0]
        vals = [float(v) for v in txt.replace(",", " ").split()]
        if offsets:
            reg = offset_region(kind, resolve_center(cs, None), *vals)
        elif kind == "circle":
            reg = CircleRegion(*vals[:3])
        elif kind == "ellipse":
            reg = EllipseRegion(*vals[:5])
        elif kind == "annulus":
            reg = AnnulusRegion(*vals[:4])
        else:
            reg = PolygonRegion([(vals[i], vals[i + 1]) for i in range(0, len(vals) - 1, 2)])
    s = region_spectrum(cs, reg, name=name, distance_pc=distance)
    s.save(out)
    rprint(f"{reg.to_ds9()}  →  [green]{out}[/green]  ({len(s.wave)} pixels, sub-bands {', '.join(s.bands)})")
    if png:
        import matplotlib.pyplot as plt
        from .plots import plot_regions
        c0 = cs.load(cs.best_for(float(s.wave[len(s.wave) // 2])).path)
        img = c0.image()
        fig = plot_regions(img, c0, resolve_center(cs, None), [reg], spectrum=s)
        p = os.path.splitext(out)[0] + ".png"; fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
        rprint(f"  and {p}")
    if fit:
        from ..config import ProjectConfig
        from ..pipeline import run_pipeline
        cfg = ProjectConfig.load(fit)
        cfg.target.name = s.name; cfg.target.distance_pc = distance
        cfg.output = fit_out or os.path.splitext(out)[0] + "_fit"
        rprint(f"slab fit of the region spectrum → {cfg.output}/")
        run_pipeline(cfg, spec=s.to_rest_frame(cfg.target.rv_kms))


@cube_app.command("cutout")
def cutout(path: str, line: List[str] = typer.Option(..., "--line", "-l", help=_LINE_HELP),
           out: str = typer.Option("cube_cutouts"), window: float = 1600.0,
           gzip_: bool = typer.Option(False, "--gzip", help="write .fits.gz")):
    """Write small cubes around lines (to share data or make test sets)."""
    from .io import CubeSet, write_cutouts
    fs = write_cutouts(CubeSet(path), line, out, window_kms=window, compress=gzip_)
    for f in fs:
        rprint(f"  {f}  ({os.path.getsize(f) / 1e6:.1f} MB)")


@cube_app.command("synth")
def synth(out: str = typer.Argument("synthetic_cube"), seed: int = 1, noise: float = 15.0):
    """Synthetic cubes (2B, 3A, 3C) with a point source, a Keplerian H2 ring and a [Ne II] jet, plus truth.yaml."""
    from .synthetic import make_synthetic_cube_set
    make_synthetic_cube_set(out, seed=seed, noise_mjysr=noise)
    rprint(f"wrote [green]{out}/[/green] (Level3_ch2-medium/ch3-short/ch3-long_s3d.fits, truth.yaml)\n"
           f"try:  jalebi cube init cube_synth.yaml --example synthetic && jalebi cube run cube_synth.yaml")


@cube_app.command("demo")
def demo(out: str = typer.Option("jalebi_cube_demo", help="output folder"),
         n_mc: int = typer.Option(50, help="Monte Carlo realisations per spaxel")):
    """HV Tau C (bundled cutouts): jet and H2 maps, the H2 stack, channel maps, a PV cut along the jet, regions."""
    from .config import example_config
    from .pipeline import run_cube
    cfg = example_config("hv_tau_c")
    cfg.kinematics.n_mc = n_mc
    cfg.plot_extent_arcsec = 3.5
    r = run_cube(cfg, outdir=out)
    _print_summary(r.table)
    rprint(f"\nopen [green]{out}/FeII_5.34/HV_Tau_C_FeII_5.34_maps.png[/green] (the jet) and "
           f"[green]{out}/H2_S1/HV_Tau_C_H2_S1_maps.png[/green] (the H2 emission); the *.fits maps open in CARTA or DS9.")
