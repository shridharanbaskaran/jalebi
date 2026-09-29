"""Config-driven cube run: line maps, stacks, channel maps, PV cuts and region spectra (+ slab fits).

    from jalebi.cube.config import CubeConfig
    from jalebi.cube.pipeline import run_cube
    run = run_cube(CubeConfig.load("cube.yaml"))
    run.maps["H2 S(1)"]["vcen"], run.region_spectra["jet_north"], run.table

Lines are independent, so `n_jobs > 1` runs them in parallel processes (joblib).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..lines import C_KMS, get_line
from .channels import channel_maps, pv_diagram
from .config import CubeConfig, CubeRegionConfig
from .io import CubeSet, resolve_center
from .maps import LineMaps, _jsonable, line_maps, prepare_line, stack_lines
from .regions import (AnnulusRegion, CircleRegion, EllipseRegion, PolygonRegion, Region, offset_region, parse_ds9,
                      region_spectrum, to_ds9_file)


@dataclass
class CubeRun:
    cfg: CubeConfig
    maps: dict = field(default_factory=dict)              # line name -> LineMaps
    stacks: dict = field(default_factory=dict)            # stack name -> LineMaps
    channels: dict = field(default_factory=dict)          # line name -> ChannelMaps
    pv: dict = field(default_factory=dict)                # line name -> PVDiagram
    slices: dict = field(default_factory=dict)            # line name -> ChannelSlices
    ratios: dict = field(default_factory=dict)            # ratio name -> RatioMap
    regions: dict = field(default_factory=dict)           # region name -> Region
    region_spectra: dict = field(default_factory=dict)    # region name -> Spectrum
    region_fits: dict = field(default_factory=dict)       # region name -> RunResult
    files: list = field(default_factory=list)
    center_radec: tuple | None = None

    @property
    def table(self) -> pd.DataFrame:
        rows = []
        for name, m in list(self.maps.items()) + list(self.stacks.items()):
            s = m.summary
            rows.append({"line": name, "band": s["band"], "rest_um": s["rest_um"],
                         "total_W_m2": s["total_flux_W_m2_snr_masked"], "point_source_W_m2": s["point_source_line_flux_W_m2"],
                         "extended_W_m2": s["extended_flux_W_m2"], "v_source_kms": s["source_velocity_kms"],
                         "v_source_err_kms": s["source_velocity_err_kms"], "n_spaxels_snr": s["n_spaxels_snr"],
                         "n_spaxels_velocity": s["n_spaxels_velocity"],
                         "inner_over_annulus": s["extendedness"].get("mom0", {}).get("ratio")})
        return pd.DataFrame(rows)


def settings_kwargs(cfg: CubeConfig) -> dict:
    """prepare_line keyword arguments from a config."""
    cont = cfg.continuum.model_dump()
    psf = cfg.psf.model_dump()
    psf["extra_sources"] = [tuple(x) for x in psf.get("extra_sources", [])]
    return dict(rv_kms=cfg.rv_kms, window_kms=cfg.window_kms, continuum=cont, psf=psf, center=cfg.center,
                band_offsets_kms=cfg.band_offsets_kms, smooth_fwhm_pix=cfg.smooth_fwhm_pix, noise=cfg.noise,
                window_um=cfg.window_um, band=cfg.band)


def maps_kwargs(cfg: CubeConfig) -> dict:
    k = cfg.kinematics
    mo = cfg.moments
    return dict(snr_min=mo.snr_min, line_kms=mo.line_kms, channel_clip=mo.channel_clip,
                kinematics=k.enabled, kin_source=k.source, kin_snr_min=k.snr_min, n_mc=k.n_mc,
                center_shift_kms=k.center_shift_kms, sigma_range=tuple(k.sigma_range), zero_point=k.zero_point,
                component=mo.component, half_full=mo.half_full, half_slow=mo.half_slow, min_valid=mo.min_valid,
                rms_region=tuple(mo.rms_region) if mo.rms_region else None, rms_sigma=mo.rms_sigma, rms_mode=mo.rms_mode)


def ratio_name(rc) -> str:
    return rc.name or f"{get_line(rc.lines[0]).tag}_over_{get_line(rc.lines[1]).tag}"   # get_line keeps typed names


def build_region(rc: CubeRegionConfig, center_radec) -> Region:
    from ..data import parse_radec
    if rc.ds9:
        return parse_ds9(rc.ds9)[0]
    if rc.shape == "polygon":
        if rc.offset_vertices:
            return offset_region("polygon", center_radec, *np.ravel(rc.offset_vertices))
        return PolygonRegion([tuple(map(float, v)) for v in rc.vertices])
    if rc.offset is not None:
        dx, dy = rc.offset
        if rc.shape == "circle":
            return offset_region("circle", center_radec, dx, dy, rc.r)
        if rc.shape == "ellipse":
            return offset_region("ellipse", center_radec, dx, dy, rc.a, rc.b, rc.pa)
        if rc.shape == "annulus":
            return offset_region("annulus", center_radec, dx, dy, rc.r_in, rc.r_out)
    ra, dec = parse_radec(rc.ra, rc.dec)
    if rc.shape == "circle":
        return CircleRegion(ra, dec, rc.r)
    if rc.shape == "ellipse":
        return EllipseRegion(ra, dec, rc.a, rc.b, rc.pa)
    if rc.shape == "annulus":
        return AnnulusRegion(ra, dec, rc.r_in, rc.r_out)
    raise ValueError(f"region {rc.name}: unknown shape {rc.shape!r}")


def _one_line(path, name, line, prep_kw, map_kw, dq_mask=True, zero_is_nan=True):
    cs = CubeSet(path, name=name, dq_mask=dq_mask, zero_is_nan=zero_is_nan)
    lc = prepare_line(cs, line, **prep_kw)
    return line_maps(lc, **map_kw)


def run_cube(cfg: CubeConfig, outdir: str | None = None, write: bool = True, progress=None, verbose: bool = True) -> CubeRun:
    """Run everything the config asks for; write FITS/PNG/CSV into cfg.output_dir() (or `outdir`)."""
    from ..examples import resolve_path
    say = print if verbose else (lambda *a, **k: None)
    cs = CubeSet(resolve_path(cfg.path), name=cfg.name, dq_mask=cfg.dq_mask, zero_is_nan=cfg.zero_is_nan)
    out = outdir or cfg.output_dir(cs.name)            # e.g. results/HV_Tau_C/cube
    run = CubeRun(cfg)
    run.center_radec = resolve_center(cs, cfg.center)
    say(f"cube: {cs.name} — {len(cs.info)} cubes ({', '.join(cs.bands)}), source at RA {run.center_radec[0]:.6f} Dec {run.center_radec[1]:.6f}")
    if write:
        say(f"results -> {out}")
    prep_kw = settings_kwargs(cfg); map_kw = maps_kwargs(cfg)
    prep_kw["center"] = run.center_radec
    lines = [get_line(x) for x in cfg.lines]
    for rc in cfg.ratios:                              # a ratio needs both lines' maps
        for x in rc.lines:
            if get_line(x).name not in [ln.name for ln in lines]:
                lines.append(get_line(x))
    missing = [ln.name for ln in lines if not cs.covering(ln.wave)]
    if missing:
        say(f"  not covered by these cubes (skipped): {', '.join(missing)}")
    lines = [ln for ln in lines if cs.covering(ln.wave)]
    if cfg.band and str(cfg.band).lower() not in ("nominal", "auto", "margin"):
        # a fixed sub-band: lines it does not cover are skipped, not an error
        def _in_band(ln):
            try:
                return cs.choose(ln.wave * (1 + cfg.rv_kms / C_KMS), cfg.band) is not None
            except ValueError:
                return False
        out_band = [ln.name for ln in lines if not _in_band(ln)]
        if out_band:
            say(f"  not in band {cfg.band} (skipped): {', '.join(out_band)}")
        lines = [ln for ln in lines if ln.name not in out_band]
    total = len(lines) + len(cfg.stacks) + len(cfg.regions) + 1
    step = 0

    def tick(what):
        nonlocal step
        step += 1
        if progress:
            progress(what, step / total)
    if cfg.n_jobs and cfg.n_jobs != 1 and len(lines) > 1:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=cfg.n_jobs)(delayed(_one_line)(resolve_path(cfg.path), cfg.name, ln, prep_kw, map_kw, cfg.dq_mask,
                                                              cfg.zero_is_nan)
                                          for ln in lines)
        for ln, m in zip(lines, res):
            run.maps[ln.name] = m; tick(ln.name)
    else:
        for ln in lines:
            lc = prepare_line(cs, ln, **prep_kw)
            run.maps[ln.name] = line_maps(lc, **map_kw)
            tick(ln.name)
    for ln in lines:
        m = run.maps[ln.name]; s = m.summary
        ps = s["point_source_line_flux_W_m2"]
        say(f"  {ln.name:16s} {s['band']}  total {s['total_flux_W_m2_snr_masked']:.2e} W/m2 (S/N-masked)"
            + (f"  point source {ps:.2e}  extended {s['extended_flux_W_m2']:.2e}" if ps is not None else "")
            + f"  v(source) {s['source_velocity_kms']:+.1f} km/s  velocity spaxels {s['n_spaxels_velocity']}")
    for sname, members in cfg.stacks.items():
        mem = [m for m in members if cs.covering(get_line(m).wave)]
        if len(mem) < 2:
            say(f"  stack {sname}: fewer than two covered lines, skipped"); continue
        lc = stack_lines(cs, mem, name=f"{sname} stack", **prep_kw)
        run.stacks[sname] = line_maps(lc, **{**map_kw, "zero_point": map_kw["zero_point"]})
        s = run.stacks[sname].summary
        say(f"  stack {sname} ({', '.join(mem)}): velocity spaxels {s['n_spaxels_velocity']}")
        tick(f"stack {sname}")
    def key(x):
        """The run.maps key of a line given by name, alias, wavelength or 'name=wavelength'."""
        x = str(x)
        return x if x in run.maps else get_line(x).name

    if cfg.channels.enabled:
        for name in map(key, cfg.channels.lines or list(run.maps)):
            if name in run.maps:
                run.channels[name] = channel_maps(run.maps[name].lc, cfg.channels.vmin, cfg.channels.vmax,
                                                  cfg.channels.dv_kms, cfg.channels.source)
    if cfg.channels.slices:                            # independent of the velocity-binned channel maps
        from .channels import channel_slices
        for name in map(key, cfg.channels.lines or list(run.maps)):
            if name in run.maps:
                mo = cfg.moments
                run.slices[name] = channel_slices(run.maps[name].lc, cfg.channels.slices_component, mo.half_full,
                                                  mo.half_slow, cfg.channels.source)
    for rc in cfg.ratios:
        from .ratios import ratio_map
        a, b = (key(x) for x in rc.lines)
        if a in run.maps and b in run.maps:
            rm = ratio_map(run.maps[a], run.maps[b], rms_region=tuple(rc.rms_region) if rc.rms_region else None,
                           sigma_thresh=tuple(rc.sigma_thresh), rms_mode=rc.rms_mode, key=rc.key, unit=rc.unit)
            rm.meta["percentile"] = list(rc.percentile)
            run.ratios[ratio_name(rc)] = rm
            sm = rm.summary()
            say(f"  ratio {a} / {b}: {sm['n_valid']} spaxels, median {sm['median_ratio']:.3g}" if sm["n_valid"] else
                f"  ratio {a} / {b}: no spaxel above the thresholds")
    if cfg.pv.enabled:
        for name in map(key, cfg.pv.lines or list(run.maps)):
            if name in run.maps:
                run.pv[name] = pv_diagram(run.maps[name].lc, cfg.pv.pa_deg, cfg.pv.length_arcsec, cfg.pv.width_arcsec,
                                             source=cfg.pv.source, vmax_kms=cfg.pv.vmax_kms)
    for rc in cfg.regions:
        reg = build_region(rc, run.center_radec)
        run.regions[rc.name] = reg
        spec = region_spectrum(cs, reg, name=f"{cs.name} {rc.name}", distance_pc=cfg.distance_pc)
        run.region_spectra[rc.name] = spec
        say(f"  region {rc.name}: {reg.to_ds9()} -> {len(spec.wave)} pixels in {len(spec.bands)} sub-bands")
        if rc.fit and cfg.fit_config:
            from ..config import ProjectConfig
            from ..pipeline import run_pipeline
            pc = ProjectConfig.load(resolve_path(cfg.fit_config))
            pc.target.name = spec.name; pc.target.distance_pc = cfg.distance_pc; pc.target.rv_kms = cfg.rv_kms
            pc.output = os.path.join(out, "region_fits", rc.name)
            run.region_fits[rc.name] = run_pipeline(pc, spec=spec.to_rest_frame(cfg.rv_kms))
        tick(f"region {rc.name}")
    if write:
        run.files = write_run(run, out)
        say(f"wrote {len(run.files)} files to {out}/")
    return run


def write_run(run: CubeRun, out: str) -> list[str]:
    import matplotlib.pyplot as plt
    from .plots import plot_channel_maps, plot_pv, plot_regions
    cfg = run.cfg
    os.makedirs(out, exist_ok=True)
    files = []
    for m in list(run.maps.values()) + list(run.stacks.values()):
        files += m.write(out, formats=tuple(cfg.formats), distance_pc=cfg.distance_pc, pa_deg=cfg.pa_deg,
                         extent_arcsec=cfg.plot_extent_arcsec)
    for name, cm in run.channels.items():
        d = os.path.join(out, cm.lc.line.tag); os.makedirs(d, exist_ok=True)
        base = os.path.join(d, f"{_safe(cm.lc.cube.name)}_{cm.lc.line.tag}_channels")
        if "fits" in cfg.formats:
            files.append(cm.write_fits(base + ".fits"))
        if "png" in cfg.formats:
            fig = plot_channel_maps(cm, distance_pc=cfg.distance_pc, extent_arcsec=cfg.plot_extent_arcsec)
            fig.savefig(base + ".png", dpi=130, bbox_inches="tight"); plt.close(fig); files.append(base + ".png")
    for name, sl in run.slices.items():
        d = os.path.join(out, sl.lc.line.tag, "channel_slices")
        files += sl.write(d)
    for name, rm in run.ratios.items():
        d = os.path.join(out, "ratios"); os.makedirs(d, exist_ok=True)
        tname = next(iter(run.maps.values())).lc.cube.name if run.maps else (cfg.name or "target")
        base = os.path.join(d, f"{_safe(tname)}_{name}")
        if "fits" in cfg.formats:
            files.append(rm.write_fits(base + ".fits"))
        with open(base + ".json", "w") as fh:
            json.dump(_jsonable(rm.summary()), fh, indent=2)
        files.append(base + ".json")
        if "png" in cfg.formats and np.isfinite(rm.masked1).any() and np.isfinite(rm.masked2).any():
            from .plots import plot_ratio_map
            fig, _ = plot_ratio_map(rm, rm.meta.get("percentile", (80, 80, 80)))
            fig.savefig(base + ".png", dpi=130, bbox_inches="tight"); plt.close(fig); files.append(base + ".png")
    for name, pv in run.pv.items():
        d = os.path.join(out, pv.lc.line.tag); os.makedirs(d, exist_ok=True)
        base = os.path.join(d, f"{_safe(pv.lc.cube.name)}_{pv.lc.line.tag}_pv_PA{pv.pa_deg:g}")
        if "fits" in cfg.formats:
            files.append(pv.write_fits(base + ".fits"))
        if "png" in cfg.formats:
            fig = plot_pv(pv); fig.savefig(base + ".png", dpi=140, bbox_inches="tight"); plt.close(fig); files.append(base + ".png")
    if run.region_spectra:
        d = os.path.join(out, "regions"); os.makedirs(d, exist_ok=True)
        to_ds9_file(list(run.regions.values()), os.path.join(d, "regions.reg")); files.append(os.path.join(d, "regions.reg"))
        for name, spec in run.region_spectra.items():
            p = os.path.join(d, f"{_safe(name)}_spectrum.csv"); spec.save(p); files.append(p)
        ref = next(iter(run.maps.values()), None)
        if ref is not None and "png" in cfg.formats:
            img = ref.maps.get("mom0_ext", ref["mom0"])
            fig = plot_regions(img, ref.lc.cube, ref.lc.center_radec, list(run.regions.values()), labels=list(run.regions))
            p = os.path.join(d, "regions.png"); fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig); files.append(p)
    tab = run.table
    if not tab.empty:
        p = os.path.join(out, "line_summary.csv"); tab.to_csv(p, index=False); files.append(p)
    p = os.path.join(out, "cube_config_used.yaml")
    cfg.save(p); files.append(p)
    with open(os.path.join(out, "run.json"), "w") as fh:
        json.dump(_jsonable({"center_radec": run.center_radec, "lines": list(run.maps), "stacks": list(run.stacks),
                             "ratios": {k: v.summary() for k, v in run.ratios.items()},
                             "regions": {k: v.to_ds9() for k, v in run.regions.items()}}), fh, indent=2)
    files.append(os.path.join(out, "run.json"))
    return files


def _safe(s):
    import re
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(s)).strip("_")


def quick_maps(path: str, lines, out: str | None = None, **kw) -> dict[str, LineMaps]:
    """One-liner for notebooks: maps of `lines` with default settings (kw override CubeConfig fields)."""
    cfg = CubeConfig(path=path, lines=list(lines) if not isinstance(lines, str) else [lines], **kw)
    run = run_cube(cfg, outdir=out, write=out is not None, verbose=False)
    return run.maps
