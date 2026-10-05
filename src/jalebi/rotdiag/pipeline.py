"""One rotation-diagram run: spectrum (or a flux table) -> features -> fluxes -> fit (+ MCMC) -> results folder.

    from jalebi.rotdiag import RotDiagConfig, run_rotdiag
    res = run_rotdiag(RotDiagConfig.load("rd.yaml"))
    print(res.fit.summary())

Results (default ``results/{target}/rotdiag/{molecule}/``): lines.csv (features, fluxes, flags), members.csv,
diagram.csv (E_u, ln N_u/g_u), fit.yaml, params.csv, derived.csv, compare.csv, chain.npz, summary.txt,
rotation_diagram.png, corner.png, line_fits.png and the config that made them.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yaml

from .config import RotDiagConfig
from .features import Selection, find_features, match_table
from .fit import FitConfig, MCMCConfig, RotFit, compare_models, diagram_points, fit_rotation
from .measure import MeasureConfig, measure_features
from .physics import ARCSEC


@dataclass
class RotDiagResult:
    cfg: RotDiagConfig
    spec: object | None
    features: pd.DataFrame
    members: pd.DataFrame
    stamps: dict = field(default_factory=dict)
    fit: RotFit | None = None
    comparison: pd.DataFrame | None = None
    extra_fits: dict = field(default_factory=dict)      # {model: RotFit} of fit.also
    outdir: str | None = None
    messages: list[str] = field(default_factory=list)

    @property
    def target(self) -> str:
        return self.cfg.target or (getattr(self.spec, "name", None) or "target")


# ------------------------------------------------------------------------------------------------
# building blocks (also used by the web app)
# ------------------------------------------------------------------------------------------------

def _is_cube_folder(path: str) -> bool:
    import glob
    return os.path.isdir(path) and bool(glob.glob(os.path.join(path, "*s3d*.fits*")))


def region_from_config(s, center_radec):
    from ..cube import offset_region
    k = s.region
    if k == "circle":
        return offset_region("circle", center_radec, s.dx, s.dy, s.radius_arcsec)
    if k == "ellipse":
        b, pa = (s.params + [s.radius_arcsec, 0.0])[:2]
        return offset_region("ellipse", center_radec, s.dx, s.dy, s.radius_arcsec, b, pa)
    if k == "annulus":
        r_in = s.params[0] if s.params else 0.5 * s.radius_arcsec
        return offset_region("annulus", center_radec, s.dx, s.dy, r_in, s.radius_arcsec)
    if k == "polygon":
        return offset_region("polygon", center_radec, *s.params)
    raise ValueError(f"unknown region kind '{k}' (circle | ellipse | annulus | polygon | all)")


def load_cube_spectrum(cfg: RotDiagConfig, source=None):
    """Sum the s3d cubes over the configured region -> Spectrum (Jy) with meta['extraction'] (area, Ω).
    `source`: a jalebi.source.Source whose cubes are already in memory (the web app passes the open source);
    its position is used when the config gives no RA/Dec, and the region spectrum is memoised there."""
    from ..cube import CubeSet, offset_region, region_spectrum
    from ..data import parse_radec
    from ..examples import resolve_path
    s = cfg.spectrum
    cs = source.cubes if source is not None else CubeSet(resolve_path(s.path))
    if s.ra is not None and s.dec is not None:
        center = parse_radec(s.ra, s.dec)
    elif source is not None:
        center = source.position()[:2]
    else:
        center = cs.source_position()
    if s.region == "all":
        reg = offset_region("circle", center, 0.0, 0.0, 30.0)         # larger than any MRS field
    else:
        reg = region_from_config(s, center)
    bg = offset_region("annulus", center, 0.0, 0.0, s.background[0], s.background[1]) if s.background else None
    name = s.name or f"{cs.name}"
    if source is not None:
        spec = source.region_spectrum(reg, background=bg, distance_pc=s.distance_pc or 140.0, name=name)
    else:
        spec = region_spectrum(cs, reg, name=name, distance_pc=s.distance_pc or 140.0, background=bg)
    spec.meta.setdefault("extraction", {})["center_radec"] = [float(center[0]), float(center[1])]
    if s.rv_kms:
        spec = spec.to_rest_frame(s.rv_kms)
    return spec


def load_input_spectrum(cfg: RotDiagConfig, source=None):
    """The spectrum of the config.  `source` (a jalebi.source.Source at the same path) serves it from memory."""
    from ..data import load_spectrum
    from ..examples import resolve_path
    s = cfg.spectrum
    path = resolve_path(s.path)
    if source is not None and os.path.abspath(resolve_path(source.path)) != os.path.abspath(path):
        source = None
    src = s.source
    if src == "auto":
        src = "s3d" if _is_cube_folder(path) else ("csv" if path.lower().endswith((".csv", ".gz", ".txt", ".dat", ".h5", ".hdf5")) else "x1d")
    if src == "s3d":
        return load_cube_spectrum(cfg, source=source)
    if source is not None and source.has_1d:
        spec = source.x1d(distance_pc=s.distance_pc or None)
        if s.name:
            spec.name = s.name
        return spec.to_rest_frame(s.rv_kms) if s.rv_kms else spec
    kw = {}
    if s.distance_pc:
        kw["distance_pc"] = s.distance_pc
    if s.name:
        kw["name"] = s.name
    spec = load_spectrum(path, source="x1d" if src == "fits" else src, **kw)
    if s.rv_kms:
        spec = spec.to_rest_frame(s.rv_kms)
    return spec


def selection(cfg: RotDiagConfig) -> Selection:
    L = cfg.lines
    return Selection(wmin=L.wmin, wmax=L.wmax, bands=L.bands, eu_max=L.eu_max, eu_min=L.eu_min, t_ref=L.t_ref,
                     rel_min=L.rel_min, max_features=L.max_features, blend_fwhm=L.blend_fwhm, member_rel=L.member_rel,
                     edge_fwhm=L.edge_fwhm, curated=L.curated, include=list(L.include), exclude=list(L.exclude),
                     R_model=L.resolving_power)


def measure_config(cfg: RotDiagConfig) -> MeasureConfig:
    m = cfg.measure
    return MeasureConfig(method=m.method, velocity=m.velocity, width=m.width, window_fwhm=m.window_fwhm, joint_fwhm=m.joint_fwhm,
                         core_fwhm=m.core_fwhm, cont_order=m.cont_order, continuum=m.continuum, snr_detect=m.snr_detect,
                         integrate_fwhm=m.integrate_fwhm, contaminants=m.contaminants, flux_unit=m.flux_unit,
                         R_model=cfg.lines.resolving_power)


def fit_config(cfg: RotDiagConfig, distance_pc: float, spec=None) -> FitConfig:
    f = cfg.fit
    geo = cfg.geometry.to_geometry(distance_pc)
    ex = spec.meta.get("extraction") if (spec is not None and isinstance(spec.meta, dict)) else None
    area = ex.get("area_arcsec2") if isinstance(ex, dict) else None
    if geo.mode == "auto":
        geo.mode = "aperture" if area else "number"
    if geo.mode == "aperture" and not geo.omega_sr and area:
        geo.omega_sr = float(area) * ARCSEC ** 2          # the region's solid angle -> beam-averaged column density
    if cfg.measure.flux_unit.lower() != "jy" and geo.mode not in ("intensity",):
        geo.mode = "intensity"
    return FitConfig(model=f.model, opr=f.opr, opr_value=f.opr_value, opr_free=f.opr_free, av=f.av, av_free=f.av_free,
                     extinction=f.extinction, geometry=geo, opacity=f.opacity, fwhm_kms=f.fwhm_kms, fwhm_free=f.fwhm_free,
                     R_free=f.R_free, sys_frac=f.sys_frac, use=f.use, snr_detect=cfg.measure.snr_detect,
                     bounds={k: list(v) for k, v in f.bounds.items()}, fixed=dict(f.fixed), start=dict(f.start),
                     tmax_powerlaw=f.tmax_powerlaw)


_UNITS = {"wm-2": 1.0, "wm-2sr-1": 1.0, "w/m2": 1.0, "w/m^2": 1.0, "ergs-1cm-2": 1e-3, "erg/s/cm2": 1e-3,
          "erg/s/cm^2": 1e-3, "ergs-1cm-2sr-1": 1e-3, "erg/s/cm2/sr": 1e-3, "wm-2sr": 1.0}


def unit_factor(unit: str) -> float:
    """Multiplier to W m^-2 (or W m^-2 sr^-1) for a flux unit such as 'erg s-1 cm-2' or '1e-17 erg s-1 cm-2'."""
    u = unit.strip()
    scale = 1.0
    m = re.match(r"^([0-9.eE+-]+)\s*[x×*]?\s*(.*)$", u)
    if m and m.group(2):
        try:
            scale = float(m.group(1)); u = m.group(2)
        except ValueError:
            pass
    key = u.replace(" ", "").replace("^", "").replace("⁻", "-").replace("²", "2").lower()
    if key not in _UNITS:
        raise ValueError(f"unknown flux unit '{unit}': use W m-2, erg s-1 cm-2 (optionally with a factor, e.g. '1e-17 erg s-1 cm-2')")
    return scale * _UNITS[key]


def features_from_table(cfg: RotDiagConfig) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Features with the fluxes of a user table (label or wavelength, flux, err)."""
    from ..examples import resolve_path
    path = resolve_path(cfg.fluxes.path)
    T = pd.read_csv(path, comment="#", sep=None, engine="python")
    T.columns = [c.strip().lower() for c in T.columns]
    fcol = next((c for c in ("flux", "f", "line_flux", "intensity") if c in T.columns), None)
    ecol = next((c for c in ("err", "flux_err", "error", "e_flux", "sigma", "unc") if c in T.columns), None)
    if fcol is None or ecol is None:
        raise ValueError(f"{path}: needs a flux column (flux) and an error column (err); found {list(T.columns)}")
    sel = selection(cfg)
    wcol = next((c for c in ("wave", "wave_um", "lambda", "wavelength", "lam") if c in T.columns), None)
    if sel.wmin is None and sel.wmax is None and wcol is not None:
        sel.wmin, sel.wmax = float(T[wcol].min()) * 0.99, float(T[wcol].max()) * 1.01
    sel.rel_min = 0.0
    sel.max_features = 100000
    sel.edge_fwhm = 0.0
    F, M = find_features(cfg.molecule, None, sel, release=cfg.release)
    fid = match_table(T, F, cfg.fluxes.tol_fwhm, cfg.lines.resolving_power)
    msgs = []
    if (fid < 0).any():
        msgs.append(f"{int((fid < 0).sum())} rows of {os.path.basename(path)} matched no line of {cfg.molecule} and were skipped: "
                    + ", ".join(str(T.iloc[i].get('label', T.iloc[i].get(wcol, i))) for i in np.flatnonzero(fid < 0)))
    fac = unit_factor(cfg.fluxes.unit)
    keep = fid >= 0
    Fsel = F.set_index("id").loc[fid[keep]].reset_index()
    Fsel["flux"] = T[fcol].to_numpy(float)[keep] * fac
    Fsel["flux_err"] = T[ecol].to_numpy(float)[keep] * fac
    Fsel["snr"] = Fsel["flux"] / Fsel["flux_err"]
    Fsel["detected"] = Fsel["snr"] >= cfg.measure.snr_detect
    Fsel["upper_limit"] = cfg.measure.snr_detect * Fsel["flux_err"]
    Fsel["measured"] = True
    Fsel["use"] = np.isfinite(Fsel["flux"]) & (Fsel["flux_err"] > 0)
    Fsel["note"] = "from table"
    Fsel = Fsel.drop_duplicates("id").sort_values("wave").reset_index(drop=True)
    Fsel.attrs.update(F.attrs)
    M = M[M["feature"].isin(Fsel["id"])].reset_index(drop=True)
    return Fsel, M, msgs


def apply_use(cfg: RotDiagConfig, F: pd.DataFrame) -> pd.DataFrame:
    F = F.copy()
    if cfg.measure.use:
        want = set(cfg.measure.use)
        F["use"] = F["use"] & (F["top"].isin(want) | F["label"].isin(want))
    if cfg.measure.skip:
        skip = set(cfg.measure.skip)
        F["use"] = F["use"] & ~(F["top"].isin(skip) | F["label"].isin(skip))
    return F


# ------------------------------------------------------------------------------------------------
# the run
# ------------------------------------------------------------------------------------------------

def run_rotdiag(cfg: RotDiagConfig, spec=None, save: bool = True, progress=None, outdir: str | None = None) -> RotDiagResult:
    msgs: list[str] = []
    stamps: dict = {}
    if cfg.fluxes is not None and cfg.fluxes.path:
        F, M, m2 = features_from_table(cfg)
        msgs += m2
        dist = cfg.geometry.distance_pc or cfg.spectrum.distance_pc or 140.0
    else:
        if spec is None:
            if not cfg.spectrum.path:
                raise ValueError("give a spectrum (spectrum.path) or a flux table (fluxes.path)")
            spec = load_input_spectrum(cfg)
        F0, M = find_features(cfg.molecule, spec, selection(cfg), release=cfg.release)
        if len(F0) == 0:
            raise ValueError(f"no {cfg.molecule} lines inside the spectrum with this selection")
        F, stamps = measure_features(spec, F0, M, measure_config(cfg), cfg.molecule)
        dist = cfg.geometry.distance_pc or spec.distance_pc
    F = apply_use(cfg, F)
    fcfg = fit_config(cfg, dist, spec)
    mc = MCMCConfig(cfg.mcmc.walkers, cfg.mcmc.steps, cfg.mcmc.burn, cfg.mcmc.thin, cfg.mcmc.seed) if cfg.mcmc.enabled else None
    fit = None
    try:
        fit = fit_rotation(F, M, fcfg, cfg.molecule, mcmc=mc, progress=progress)
    except Exception as ex:
        msgs.append(f"fit failed: {ex}")
    extra = {}
    for k in cfg.fit.also:
        if k == cfg.fit.model:
            continue
        try:
            extra[k] = fit_rotation(F, M, FitConfig(**{**fcfg.__dict__, "model": k}), cfg.molecule, mcmc=mc)
        except Exception as ex:
            msgs.append(f"{k} fit failed: {ex}")
    comp = compare_models(F, M, fcfg, molecule=cfg.molecule) if cfg.fit.compare else None
    res = RotDiagResult(cfg, spec, F, M, stamps, fit, comp, extra_fits=extra, messages=msgs)
    if save:
        save_results(res, outdir)
    return res


def _clean(d):
    if isinstance(d, dict):
        return {k: _clean(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_clean(v) for v in d]
    if isinstance(d, (np.floating, float)):
        return None if not np.isfinite(d) else float(d)
    if isinstance(d, np.integer):
        return int(d)
    if isinstance(d, np.bool_):
        return bool(d)
    return d


def save_results(res: RotDiagResult, outdir: str | None = None) -> str:
    cfg = res.cfg
    outdir = outdir or cfg.output_dir(res.target)
    os.makedirs(outdir, exist_ok=True)
    res.outdir = outdir
    F = res.features
    F.to_csv(os.path.join(outdir, "lines.csv"), index=False)
    res.members.to_csv(os.path.join(outdir, "members.csv"), index=False)
    cfg.save(os.path.join(outdir, "rotdiag_config.yaml"))
    geo = res.fit.model.geometry if res.fit is not None else fit_config(cfg, getattr(res.spec, "distance_pc", 140.0), res.spec).geometry
    av = res.fit.best.get("Av", 0.0) if res.fit is not None else 0.0
    R = 10 ** res.fit.best.get("logR", 0.0) if res.fit is not None else geo.R_au
    pts = diagram_points(F, geo, av, cfg.fit.extinction, R, cfg.measure.snr_detect)
    if res.fit is not None:
        from .fit import model_points
        pts["y_model"] = model_points(res.fit)
    pts.to_csv(os.path.join(outdir, "diagram.csv"), index=False)
    summary = [f"JALEBI rotation diagram: {cfg.molecule} in {res.target}", ""]
    summary += [f"features: {len(F)} ({int(F['detected'].sum())} detected, {int(F['use'].sum())} used)"]
    if "velocity_kms" in F.attrs:
        summary.append(f"line velocity {F.attrs['velocity_kms']:.1f} km/s, width {F.attrs.get('width_scale', 1):.2f} x instrumental "
                       f"({F.attrs.get('n_calibration', 0)} calibration lines), method {F.attrs.get('method', '')}")
    summary += res.messages
    if res.fit is not None:
        fit = res.fit
        with open(os.path.join(outdir, "fit.yaml"), "w") as fh:
            yaml.safe_dump(_clean(fit.to_dict()), fh, sort_keys=False, allow_unicode=True)
        fit.table().to_csv(os.path.join(outdir, "params.csv"), index=False)
        if fit.derived is not None:
            fit.derived.to_csv(os.path.join(outdir, "derived.csv"), index=False)
        if fit.samples is not None:
            np.savez_compressed(os.path.join(outdir, "chain.npz"), samples=fit.samples, lnprob=fit.lnprob, names=np.array(fit.free))
        summary += ["", fit.summary()]
        for k, ef in res.extra_fits.items():
            with open(os.path.join(outdir, f"fit_{k}.yaml"), "w") as fh:
                yaml.safe_dump(_clean(ef.to_dict()), fh, sort_keys=False, allow_unicode=True)
            ef.table().to_csv(os.path.join(outdir, f"params_{k}.csv"), index=False)
            if ef.derived is not None:
                ef.derived.to_csv(os.path.join(outdir, f"derived_{k}.csv"), index=False)
            summary += ["", ef.summary()]
    if res.comparison is not None:
        res.comparison.to_csv(os.path.join(outdir, "compare.csv"), index=False)
        summary += ["", "model comparison (lower BIC is better):", res.comparison.to_string(index=False)]
    with open(os.path.join(outdir, "summary.txt"), "w") as fh:
        fh.write("\n".join(summary) + "\n")
    if cfg.plots:
        _save_plots(res, outdir, geo)
    with open(os.path.join(outdir, "run.json"), "w") as fh:
        json.dump(_clean(dict(target=res.target, molecule=cfg.molecule, n_features=len(F), attrs={k: v for k, v in F.attrs.items()
                                                                                                  if isinstance(v, (int, float, str))})), fh, indent=1)
    return outdir


def _save_plots(res: RotDiagResult, outdir: str, geo):
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from ..molecules import get_molecule
    from .plots import plot_corner, plot_excitation, plot_line_fits, plot_model_panels, plot_rotation_diagram
    lab = get_molecule(res.cfg.molecule).label
    tex = {"H2": "H_2", "H2O": "H_2O", "13CO": "^{13}CO"}.get(res.cfg.molecule, res.cfg.molecule)
    if res.fit is not None:
        fig = plot_excitation(res.features, res.fit, molecule_label=tex)
        fig.savefig(os.path.join(outdir, "rotation_diagram.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
        if res.extra_fits:
            fig = plot_model_panels(res.features, {res.fit.model.kind: res.fit, **res.extra_fits}, molecule_label=tex)
            fig.savefig(os.path.join(outdir, "rotation_diagram_models.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
            for k, ef in res.extra_fits.items():
                if ef.samples is not None:
                    fig = plot_corner(ef)
                    fig.savefig(os.path.join(outdir, f"corner_{k}.png"), dpi=120, bbox_inches="tight"); plt.close(fig)
    fig = plot_rotation_diagram(res.features, geo, res.fit, molecule_label=f"{res.target} {lab}")
    fig.savefig(os.path.join(outdir, "rotation_diagram_residuals.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
    if res.fit is not None and res.fit.samples is not None:
        fig = plot_corner(res.fit)
        if fig is not None:
            fig.savefig(os.path.join(outdir, "corner.png"), dpi=120, bbox_inches="tight"); plt.close(fig)
    if res.stamps:
        fig = plot_line_fits(res.features, res.stamps)
        if fig is not None:
            fig.savefig(os.path.join(outdir, "line_fits.png"), dpi=120, bbox_inches="tight"); plt.close(fig)
