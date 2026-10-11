"""`jalebi showcase`: an informational slideshow of a finished fit.

Reads a results folder written by `jalebi fit` (config.yaml, prep.csv, model.csv, best_fit.json and, when the
MCMC ran, summary.csv, chain.npz, diagnostics.json, detections.csv, tau_flags.json, correlation.csv) and renders
1920x1080 panels:

  00  overview: target, settings, components, fit quality, sampler diagnostics
  01  data: the prepared spectrum, continuum, fit windows and masked pixels
  02  joint fit: data, stacked components, total model, residuals (with build-up frames for the video)
  03  model uncertainty: 16-84 % band of models drawn from the posterior, with zooms where it is widest
  04  which components are needed: Delta BIC when each is removed, peak optical depth, thin fraction
  05+ one panel per component: its share of the data (with its posterior band), its parameters (best fit and
      posterior), tau_max, Delta BIC, grid -> optimiser -> posterior in (log N, T), and the (log N, log R) posterior
  ..  1-D posteriors of every parameter, parameter table, correlation matrix, MCMC traces

and, when ffmpeg is on the PATH, an MP4 that holds each panel for a few seconds with short crossfades.
Nothing is refitted: the panels show what the run wrote; only the uncertainty bands re-evaluate the model (from
config.yaml) at a few hundred posterior samples (`--draws 0` skips that).

    jalebi showcase results/FZ_Tau
    jalebi showcase my_disk.yaml --seconds 5 --no-video
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

W, H, DPI = 1920, 1080, 100
INK, MUTED, FAINT, GRID, OK, WARN, BAD = "#1d1d1f", "#5f6368", "#9aa0a6", "#e8eaed", "#188038", "#b06000", "#c5221f"


# ---------------------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------------------
@dataclass
class Run:
    folder: Path
    config: dict
    prep: pd.DataFrame
    model: pd.DataFrame
    best: dict
    summary: pd.DataFrame | None = None
    detections: pd.DataFrame | None = None
    tau_flags: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    chain: np.ndarray | None = None
    names: list = field(default_factory=list)
    correlation: pd.DataFrame | None = None

    @property
    def target(self) -> str:
        return (self.config.get("target") or {}).get("name") or self.folder.name

    @property
    def components(self) -> list[dict]:
        return list(self.config.get("components") or [])

    def component(self, name: str) -> dict:
        return next((c for c in self.components if c.get("name") == name), {"name": name, "molecule": name})

    def posterior(self, burn: int | None = None) -> np.ndarray | None:
        if self.chain is None:
            return None
        b = self.chain.shape[0] // 5 if burn is None else burn
        return self.chain[b:].reshape(-1, self.chain.shape[2])


def resolve_folder(path: str) -> Path:
    """A results folder, or a config whose output folder already holds a finished run."""
    p = Path(path)
    if p.is_dir():
        return p
    if p.suffix.lower() in (".yaml", ".yml") and p.exists():
        from .config import ProjectConfig
        cfg = ProjectConfig.load(str(p))
        name = cfg.target.name or "target"
        out = Path(cfg.output_dir(name))
        if out.is_dir():
            return out
        raise FileNotFoundError(f"no results for {p} yet (expected {out}); run `jalebi fit {p}` first")
    raise FileNotFoundError(f"{path}: not a results folder or a config file")


def _read_prep(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#")


def load_run(path: str) -> Run:
    f = resolve_folder(path)
    need = ["config.yaml", "prep.csv", "model.csv", "best_fit.json"]
    missing = [n for n in need if not (f / n).exists()]
    if missing:
        raise FileNotFoundError(f"{f} is missing {', '.join(missing)}: is it the output folder of `jalebi fit`?")
    run = Run(folder=f, config=yaml.safe_load(open(f / "config.yaml")) or {}, prep=_read_prep(f / "prep.csv"),
              model=pd.read_csv(f / "model.csv"), best=json.load(open(f / "best_fit.json")))
    if (f / "summary.csv").exists():
        run.summary = pd.read_csv(f / "summary.csv")
    if (f / "detections.csv").exists():
        run.detections = pd.read_csv(f / "detections.csv")
    if (f / "tau_flags.json").exists():
        run.tau_flags = json.load(open(f / "tau_flags.json"))
    if (f / "diagnostics.json").exists():
        run.diagnostics = json.load(open(f / "diagnostics.json"))
    if (f / "chain.npz").exists():
        d = np.load(f / "chain.npz", allow_pickle=True)
        run.chain = np.asarray(d["chain"], float)
        run.names = [str(n) for n in d["names"]]
    if (f / "correlation.csv").exists():
        run.correlation = pd.read_csv(f / "correlation.csv", index_col=0)
    return run


def predictive(run: Run, n: int = 200, seed: int = 0, say=None):
    """16/50/84 % model flux, total and per component, over `n` posterior draws. Rebuilds the fit problem from the
    run's config.yaml (no fitting: only model evaluations). Returns None when that is not possible."""
    S = run.posterior()
    if S is None:
        return None
    try:
        from .config import ProjectConfig
        from . import pipeline
        cfg = ProjectConfig.load(str(run.folder / "config.yaml"))
        problem = pipeline.build_problem(cfg, pipeline.prepare(cfg))
        if len(problem.wave) != len(run.model) or not np.allclose(problem.wave, run.model["wave"].to_numpy(), rtol=0, atol=1e-6):
            raise ValueError("the rebuilt fit pixels differ from model.csv")
        pick = S[np.random.default_rng(seed).integers(0, len(S), min(n, len(S)))]
        tot, per = [], {}
        for th in pick:
            P, _ = problem.params_from_theta(th)
            t, units, _ = problem.model.evaluate(P, per_unit=True)
            tot.append(t)
            for k, v in units.items():
                per.setdefault(k, []).append(v)
        q = lambda a: np.percentile(np.asarray(a), [16, 50, 84], axis=0)
        return {"total": q(tot), "units": {k: q(v) for k, v in per.items()}, "n": len(pick)}
    except Exception as e:  # noqa: BLE001 - the showcase still works without it
        if say:
            say(f"posterior predictive skipped ({type(e).__name__}: {e})")
        return None


# ---------------------------------------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------------------------------------
def _figure():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor="white")
    FigureCanvasAgg(fig)
    return fig

def _rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()

def _header(fig, run: Run, title: str, subtitle: str = "", page: str = ""):
    fig.text(0.04, 0.945, title, fontsize=26, fontweight="bold", color=INK, va="top")
    if subtitle:
        fig.text(0.04, 0.895, subtitle, fontsize=15, color=MUTED, va="top")
    fig.text(0.96, 0.945, f"{run.target}", fontsize=15, color=MUTED, va="top", ha="right")
    fig.text(0.96, 0.012, f"jalebi showcase · {run.folder.name}" + (f" · {page}" if page else ""), fontsize=11, color=FAINT, ha="right")

def _style(ax, xlabel=None, ylabel=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=13, colors=MUTED)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=14, color=MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=14, color=MUTED)

def _colour(molecule: str) -> str:
    try:
        from .plots import _colour as c
        return c(molecule)
    except Exception:
        return "#5f6368"

def _unit_columns(run: Run) -> list[str]:
    return [c[len("model_"):] for c in run.model.columns if c.startswith("model_")]

def _unit_colour(run: Run, unit: str) -> str:
    """the molecule's colour; a second (third...) component of the same molecule gets a lighter shade."""
    from matplotlib.colors import to_rgb, to_hex
    mol = _molecule(run, unit)
    same = [u for u in _unit_columns(run) if _molecule(run, u) == mol]
    k = same.index(unit) if unit in same else 0
    c = np.array(to_rgb(_colour(mol)))
    return to_hex(c + (1 - c) * min(0.45 * k, 0.8))

def _molecule(run: Run, unit: str) -> str:
    return run.component(unit).get("molecule", unit)

def _steps(ax, w, y, band=None, **kw):
    """step lines drawn per sub-band, so overlaps are not joined."""
    if band is None:
        return ax.plot(w, y, drawstyle="steps-mid", **kw)
    out = None
    for b in dict.fromkeys(band):
        s = band == b
        out = ax.plot(w[s], y[s], drawstyle="steps-mid", **kw)
        kw.pop("label", None)
    return out

def _band(ax, w, lo, hi, band=None, **kw):
    """a step-shaded band drawn per sub-band (overlapping sub-bands would otherwise interleave)."""
    for b in (dict.fromkeys(band) if band is not None else [None]):
        s = (band == b) if b is not None else np.ones(len(w), bool)
        o = np.argsort(w[s])
        ax.fill_between(w[s][o], lo[s][o], hi[s][o], step="mid", lw=0, **kw)
        kw.pop("label", None)


def _windows(run: Run) -> list[list[float]]:
    win = ((run.config.get("fit") or {}).get("windows")) or []
    if not win:
        w = run.model["wave"].to_numpy()
        win = [[float(w.min()), float(w.max())]]
    return win

def _post(run: Run, key: str):
    """(median, minus, plus) from summary.csv, or None."""
    if run.summary is None:
        return None
    s = run.summary.set_index("parameter")
    if key not in s.index:
        return None
    r = s.loc[key]
    return float(r["median"]), float(r["minus"]), float(r["plus"]), bool(r.get("at_edge", False))

def _fmt(v, prec=2):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{prec}f}"


# ---------------------------------------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------------------------------------
def panel_overview(run: Run):
    fig = _figure()
    _header(fig, run, "Fit overview", "What was fitted, how well, and whether the sampler converged", "overview")
    cfg, t = run.config, run.config.get("target") or {}
    fit = cfg.get("fit") or {}
    lines = [("Target", f"{run.target}"), ("Spectrum", str(t.get("path", "—"))),
             ("Distance", f"{t.get('distance_pc', '—')} pc"), ("Radial velocity removed", f"{t.get('rv_kms', 0)} km/s"),
             ("Continuum", (cfg.get("continuum") or {}).get("method", "—")),
             ("Fit windows", ", ".join(f"{a:g}–{b:g} µm" for a, b in _windows(run))),
             ("Stages", " → ".join(fit.get("stages", []) or ["grid", "optimise", "mcmc"]))]
    y = 0.80
    fig.text(0.04, y + 0.03, "SETTINGS", fontsize=13, color=FAINT, fontweight="bold")
    for k, v in lines:
        fig.text(0.04, y, k, fontsize=16, color=MUTED); fig.text(0.21, y, v, fontsize=16, color=INK); y -= 0.045
    # fit quality
    b = run.best
    y -= 0.03
    fig.text(0.04, y + 0.03, "FIT QUALITY", fontsize=13, color=FAINT, fontweight="bold")
    q = [("χ²", _fmt(b.get("chi2"), 1)), ("χ²_red", _fmt(b.get("chi2_red"), 3)), ("pixels fitted", str(b.get("npix", "—"))),
         ("free parameters", str(len(run.names)) if run.names else "—")]
    for k, v in q:
        fig.text(0.04, y, k, fontsize=16, color=MUTED); fig.text(0.21, y, v, fontsize=16, color=INK); y -= 0.045
    # components table
    ax = fig.add_axes([0.52, 0.47, 0.44, 0.38]); ax.axis("off")
    rows = []
    for c in run.components:
        kind = c.get("kind", "slab")
        extra = f"tied to {c['tie_to']} (ratio {c.get('ratio', '—')})" if c.get("tie_to") else (c.get("group") and f"group {c['group']}") or ""
        rel = c.get("linelist_release") or ((cfg.get("linedata") or {}).get("releases") or {}).get(c.get("molecule"), "hitran")
        rows.append([c.get("name"), c.get("molecule"), kind, rel, extra])
    tab = ax.table(cellText=rows or [["—"] * 5], colLabels=["component", "molecule", "kind", "line list", "notes"], loc="upper left",
                   cellLoc="left", colLoc="left", colWidths=[0.21, 0.15, 0.11, 0.14, 0.39])
    tab.auto_set_font_size(False); tab.set_fontsize(14); tab.scale(1, 2.0)
    for (r, c_), cell in tab.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_text_props(color=MUTED, fontweight="bold")
        elif c_ == 0 and r - 1 < len(run.components):
            cell.set_text_props(color=_unit_colour(run, run.components[r - 1].get("name", "")), fontweight="bold")
    fig.text(0.52, 0.88, "COMPONENTS", fontsize=13, color=FAINT, fontweight="bold")
    # sampler diagnostics
    d = run.diagnostics
    y0 = 0.84 - 0.0335 * (len(run.components) + 1) - 0.07
    fig.text(0.52, y0, "SAMPLER", fontsize=13, color=FAINT, fontweight="bold")
    if d and run.chain is not None:
        rh = np.nanmax(d.get("rhat", [np.nan])); sot = d.get("steps_over_tau", np.nan)
        items = [("walkers × steps", f"{run.chain.shape[1]} × {run.chain.shape[0]}"),
                 ("acceptance", f"{d.get('acceptance', np.nan):.2f}", 0.15 <= d.get("acceptance", 0) <= 0.6),
                 ("chain length / τ", f"{sot:.0f}  (want ≥ 50)", sot >= 50),
                 ("max split-R̂", f"{rh:.2f}  (want < 1.05)", rh < 1.05)]
        y = y0 - 0.04
        for it in items:
            k, v = it[0], it[1]
            col = INK if len(it) == 2 else (OK if it[2] else WARN)
            fig.text(0.52, y, k, fontsize=16, color=MUTED); fig.text(0.69, y, v, fontsize=16, color=col); y -= 0.045
        if not d.get("converged_length", True) or not d.get("rhat_ok", True):
            fig.text(0.52, y - 0.01, "Not converged by these criteria: treat the error bars as indicative and run a longer chain.",
                     fontsize=14, color=WARN)
    else:
        fig.text(0.52, y0 - 0.04, "No MCMC in this run: parameters are the optimiser's best fit only.", fontsize=16, color=MUTED)
    return fig


def panel_data(run: Run):
    fig = _figure()
    _header(fig, run, "Data and continuum", "The prepared spectrum: rest frame, spikes removed, continuum per sub-band; shaded = fit windows", "data")
    p = run.prep
    ax = fig.add_axes([0.06, 0.12, 0.90, 0.72])
    w, f, c = p["wave"].to_numpy(), p["flux"].to_numpy(), p["continuum"].to_numpy()
    band = p["band"].astype(str).to_numpy() if "band" in p else None
    _steps(ax, w, f, band, color=INK, lw=0.7, label="data")
    _steps(ax, w, c, band, color="#e8710a", lw=1.8, label="continuum")
    if "mask" in p:
        m = ~p["mask"].astype(bool).to_numpy()
        ax.plot(w[m], f[m], ".", ms=2, color=BAD, label="excluded pixels", alpha=0.6)
    for a, b in _windows(run):
        ax.axvspan(a, b, color="#1a73e8", alpha=0.08, lw=0)
    lo, hi = np.nanpercentile(f, [0.5, 99.8])
    ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
    _style(ax, "wavelength [µm]", "Fν [Jy]")
    ax.legend(fontsize=13, frameon=False, loc="upper right")
    return fig


def refinement_passes(run: Run) -> list[dict]:
    """the model-aware continuum refinement passes the run wrote (continuum_refined.npz, ..._refined2.npz, ...)."""
    out, k = [], 1
    while True:
        f = run.folder / ("continuum_refined.npz" if k == 1 else f"continuum_refined{k}.npz")
        if not f.exists():
            return out
        d = np.load(f)
        out.append({"pass": k, "continuum": np.asarray(d["continuum"], float), "accepted": bool(d["accepted"]),
                    "chi2_before": float(d["chi2_before"]), "chi2_after": float(d["chi2_after"])})
        k += 1


def first_continuum(run: Run, say=None) -> np.ndarray | None:
    """the continuum before any refinement, re-estimated from config.yaml (prep only, a few seconds)."""
    try:
        from .config import ProjectConfig
        from . import pipeline
        spec = pipeline.prepare(ProjectConfig.load(str(run.folder / "config.yaml")))
        if len(spec.wave) != len(run.prep) or not np.allclose(spec.wave, run.prep["wave"].to_numpy(), rtol=0, atol=1e-6):
            raise ValueError("re-prepared pixels differ from prep.csv")
        return np.asarray(spec.continuum, float)
    except Exception as e:  # noqa: BLE001
        if say:
            say(f"first-pass continuum not re-estimated ({type(e).__name__}: {e})")
        return None


CONT_KEYS = {"irsqr": ("lam", "p", "n_iter", "smooth"), "median_sg": ("median_window", "median_percentile", "sg_window", "sg_order"),
             "asls": ("lam", "p", "n_iter"), "aspls": ("aspls_lam", "aspls_alpha"), "rolling_min": ("min_window",),
             "spline": ("knot_spacing", "quantile"), "banzatti": ("segment", "overlap", "percentile"), "convex_hull": ()}


def panel_continuum(run: Run, first: np.ndarray | None = None, passes: list | None = None):
    passes = passes or []
    cc = run.config.get("continuum") or {}
    method = cc.get("method", "—")
    fig = _figure()
    _header(fig, run, "Continuum fit", f"{method} continuum per sub-band"
            + ("; model-aware refinement: gas model subtracted, continuum re-estimated, kept only if χ² improves" if passes
               else "; protected bands (shaded) keep molecular Q-branches out of the baseline"), "continuum")
    p = run.prep
    w, f, cfin = p["wave"].to_numpy(), p["flux"].to_numpy(), p["continuum"].to_numpy()
    band = p["band"].astype(str).to_numpy() if "band" in p else None
    win = _windows(run)
    z0, z1 = min(a for a, _ in win), max(b for _, b in win)
    REF, FIRST = "#e8710a", "#5f6368"
    # whole spectrum
    ax = fig.add_axes([0.06, 0.56, 0.62, 0.29])
    _steps(ax, w, f, band, color=INK, lw=0.5, label="data")
    _steps(ax, w, cfin, band, color=REF, lw=1.8, label="continuum used in the fit")
    if first is not None and passes:
        _steps(ax, w, first, band, color=FIRST, lw=1.2, ls="--", label="first pass")
    ax.axvspan(z0, z1, color="#1a73e8", alpha=0.07, lw=0)
    lo, hi = np.nanpercentile(f, [0.5, 99.0]); ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))
    _style(ax, None, "Fν [Jy]"); ax.legend(fontsize=12, frameon=True, framealpha=0.9, edgecolor="none", loc="upper left", ncol=3)
    ax.set_title("whole spectrum (blue = fit range)", fontsize=13, color=MUTED, loc="right")
    # fit range zoom: data, continuum(s), continuum + gas model
    sel = (w >= z0) & (w <= z1)
    has_diff = first is not None and bool(passes)
    a2 = fig.add_axes([0.06, 0.24 if has_diff else 0.08, 0.62, 0.25 if has_diff else 0.40])
    bb = band[sel] if band is not None else None
    _steps(a2, w[sel], f[sel], bb, color=INK, lw=0.6, label="data")
    m = run.model
    wm = m["wave"].to_numpy()
    corr = m["continuum_correction"].to_numpy() if "continuum_correction" in m else 0.0
    bm = m["band"].astype(str).to_numpy() if "band" in m else None
    _steps(a2, wm, m["continuum"].to_numpy() + corr + m["model"].to_numpy(), bm, color=BAD, lw=0.7, alpha=0.8, label="continuum + gas model")
    if first is not None and passes:
        _steps(a2, w[sel], first[sel], bb, color=FIRST, lw=1.4, ls="--", label="first pass")
    for ps in passes:
        _steps(a2, w[sel], ps["continuum"][sel], bb, color=REF if ps["accepted"] else WARN, lw=1.0 if not ps["accepted"] else 1.8,
               ls="-" if ps["accepted"] else ":", label=f"refined pass {ps['pass']}" + ("" if ps["accepted"] else " (rejected)"))
    if not passes:
        _steps(a2, w[sel], cfin[sel], bb, color=REF, lw=1.8, label="continuum")
    if "continuum_correction" in m:
        _steps(a2, wm, m["continuum"].to_numpy() + corr, bm, color="#1a73e8", lw=1.2, label="+ joint correction")
    for name, (a, b) in (cc.get("protected") or {}).items():
        if b >= z0 and a <= z1 and cc.get("protect", True):
            a2.axvspan(a, b, color="#9334e6", alpha=0.10, lw=0)
            a2.text(0.5 * (a + b), 0.01, name, transform=a2.get_xaxis_transform(), fontsize=10, color="#9334e6", ha="center", va="bottom", rotation=90)
    a2.set_xlim(z0, z1)
    lo, hi = np.nanpercentile(f[sel], [0.5, 99.0]); a2.set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
    _style(a2, None if has_diff else "wavelength [µm]", "Fν [Jy]")
    a2.legend(fontsize=11, frameon=True, framealpha=0.9, edgecolor="none", loc="upper left", ncol=3)
    a2.set_title("fit range (purple = protected bands)", fontsize=13, color=MUTED, loc="right")
    if has_diff:
        a3 = fig.add_axes([0.06, 0.08, 0.62, 0.12], sharex=a2)
        _steps(a3, w[sel], 1e3 * (cfin - first)[sel], bb, color=REF, lw=1.3)
        a3.axhline(0, color=FAINT, lw=1)
        _style(a3, "wavelength [µm]", "Δ [mJy]"); a2.tick_params(labelbottom=False)
        a3.set_title("used − first pass", fontsize=12, color=MUTED, loc="right")
    # settings card
    X, V = 0.72, 0.85
    fig.text(X, 0.845, "SETTINGS", fontsize=13, color=FAINT, fontweight="bold")
    rows = [("method", method)] + [(k, f"{cc[k]:g}" if isinstance(cc.get(k), (int, float)) else str(cc.get(k)))
                                   for k in CONT_KEYS.get(method, ()) if k in cc]
    rows += [("protected bands", f"{len(cc.get('protected') or {})}" + ("" if cc.get("protect", True) else " (off)")),
             ("anchors", ", ".join(f"{a:g}" for a in cc.get("anchors") or []) or "none"),
             ("refinement passes", str(cc.get("refine_iterations", 0)))]
    fit_c = (run.config.get("fit") or {}).get("continuum_fit", "none")
    rows.append(("joint correction in fit", str(fit_c)))
    y = 0.805
    for k, v in rows:
        fig.text(X, y, k, fontsize=15, color=MUTED); fig.text(V, y, v, fontsize=15, color=INK); y -= 0.037
    y -= 0.02
    fig.text(X, y, "MODEL-AWARE REFINEMENT", fontsize=13, color=FAINT, fontweight="bold"); y -= 0.04
    if passes:
        for ps in passes:
            fig.text(X, y, f"pass {ps['pass']}", fontsize=15, color=MUTED)
            fig.text(V - 0.05, y, f"χ² {ps['chi2_before']:.0f} → {ps['chi2_after']:.0f}   " + ("kept" if ps["accepted"] else "rejected"),
                     fontsize=15, color=OK if ps["accepted"] else WARN); y -= 0.037
        if has_diff:
            d_ = (cfin - first)[sel]
            fig.text(X, y - 0.01, f"in the fit range the continuum moved by {1e3 * np.nanmedian(d_):+.1f} mJy (median),\n"
                     f"{1e3 * np.nanmin(d_):+.1f} to {1e3 * np.nanmax(d_):+.1f} mJy", fontsize=13, color=MUTED, va="top")
    else:
        fig.text(X, y, "not run (continuum.refine_iterations = 0):\nthe first-pass continuum was used.", fontsize=14, color=MUTED, va="top")
    return fig


def _magnify(peak: float, target: float) -> int:
    """a round factor (1, 2, 5, 10, 20, ...) that brings a weak component's peak up towards `target`."""
    if peak <= 0 or peak >= 0.35 * target:
        return 1
    best = 1
    for f in (2, 5, 10, 20, 50, 100, 200, 500, 1000):
        if peak * f <= 0.9 * target:
            best = f
    return best


def panel_molecules(run: Run, lo_w: float | None = None, hi_w: float | None = None, page: str = "molecules"):
    """total fit over the observed spectrum on top; each component's model below it, outlines only, offset."""
    m = run.model
    w = m["wave"].to_numpy()
    lo_w = float(w.min()) if lo_w is None else lo_w
    hi_w = float(w.max()) if hi_w is None else hi_w
    sel = (w >= lo_w) & (w <= hi_w)
    band = m["band"].astype(str).to_numpy()[sel] if "band" in m else None
    ws = w[sel]
    cont = m["continuum"].to_numpy()[sel] if "continuum" in m else np.zeros(sel.sum())
    if "continuum_correction" in m:
        cont = cont + m["continuum_correction"].to_numpy()[sel]
    obs = m["data"].to_numpy()[sel] + cont
    tot = m["model"].to_numpy()[sel] + cont
    units = _unit_columns(run)
    fig = _figure()
    full = (lo_w <= w.min() + 1e-9) and (hi_w >= w.max() - 1e-9)
    _header(fig, run, "Total fit and each molecule", ("whole fit range" if full else f"{lo_w:.2f}–{hi_w:.2f} µm")
            + ": observed spectrum with continuum + all components (top); every component on its own below, same flux scale "
            "unless marked ×", page)
    ax = fig.add_axes([0.06, 0.62, 0.855, 0.24])
    _steps(ax, ws, obs, band, color=INK, lw=0.7, label="observed")
    _steps(ax, ws, tot, band, color=BAD, lw=1.0, label="continuum + total model")
    _steps(ax, ws, cont, band, color="#e8710a", lw=1.2, ls="--", label="continuum")
    ax.set_xlim(lo_w, hi_w)
    l_, h_ = np.nanpercentile(obs, [0.2, 99.9]); ax.set_ylim(l_ - 0.06 * (h_ - l_), h_ + 0.12 * (h_ - l_))
    _style(ax, None, "Fν [Jy]"); ax.tick_params(labelbottom=False)
    ax.legend(fontsize=12, frameon=True, framealpha=0.9, edgecolor="none", loc="upper left", ncol=3)
    # offset stack
    a2 = fig.add_axes([0.06, 0.08, 0.855, 0.52], sharex=ax)
    peaks = {u: float(np.nanmax(m["model_" + u].to_numpy()[sel])) if sel.any() else 0.0 for u in units}
    pmax = max(max(peaks.values(), default=0.0), 1e-6)
    step = 0.9 * pmax
    top = 0.0
    for i, u in enumerate(units):
        y = m["model_" + u].to_numpy()[sel]
        off = -i * step
        k = _magnify(peaks[u], step)
        col = _unit_colour(run, u)
        a2.axhline(off, color=GRID, lw=0.8, zorder=0)
        if peaks[u] < 1e-4 * pmax:
            a2.text(lo_w + 0.005 * (hi_w - lo_w), off + 0.12 * step, f"{u}: no lines here", fontsize=12, color=FAINT, va="bottom")
            continue
        top = max(top, off + k * peaks[u])
        _steps(a2, ws, off + k * y, band, color=col, lw=1.1)
        a2.annotate(u + (f"  ×{k}" if k > 1 else ""), xy=(1.0, off), xycoords=("axes fraction", "data"), xytext=(6, 4),
                    textcoords="offset points", fontsize=13, color=col, fontweight="bold", va="bottom", ha="left", annotation_clip=False)
    # flux scale bar in the head-room above the first trace
    bar = max(v for v in (1e-3 * np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000])) if v <= 0.5 * step) if step > 2e-3 else step / 2
    y0 = top + 0.1 * step
    a2.plot([lo_w + 0.004 * (hi_w - lo_w)] * 2, [y0, y0 + bar], color=INK, lw=2.5, solid_capstyle="butt")
    a2.text(lo_w + 0.009 * (hi_w - lo_w), y0 + bar / 2, f"{bar * 1e3:g} mJy (×1)", fontsize=12, color=INK, va="center")
    a2.set_ylim(-(len(units) - 1) * step - 0.15 * step, y0 + bar + 0.1 * step)
    _style(a2, "wavelength [µm]", None); a2.set_yticks([])
    a2.spines["left"].set_visible(False)
    return fig


def _fit_axes(fig, run: Run):
    ax = fig.add_axes([0.06, 0.36, 0.90, 0.50]); axr = fig.add_axes([0.06, 0.12, 0.90, 0.20], sharex=ax)
    return ax, axr


def panel_fit(run: Run, upto: int | None = None):
    units = _unit_columns(run)
    k = len(units) if upto is None else upto
    sub = "Data (black), components stacked in colour, total model (red), residuals below"
    if upto is not None:
        sub = f"Adding components one at a time: {k} of {len(units)}  (the fit itself is simultaneous)"
    fig = _figure()
    _header(fig, run, "Joint fit", sub, "fit")
    m = run.model
    w, d, s = m["wave"].to_numpy(), m["data"].to_numpy(), m["sigma"].to_numpy()
    band = m["band"].astype(str).to_numpy() if "band" in m else None
    o = np.argsort(w, kind="stable")
    ax, axr = _fit_axes(fig, run)
    base = np.zeros(len(w))
    for u in units[:k]:
        y = m["model_" + u].to_numpy()
        top = base + y
        for b in (dict.fromkeys(band) if band is not None else [None]):
            sel = (band == b) if b is not None else np.ones(len(w), bool)
            oo = np.argsort(w[sel])
            ax.fill_between(w[sel][oo], base[sel][oo], top[sel][oo], step="mid", color=_unit_colour(run, u), alpha=0.75, lw=0,
                            label=u if b == (next(iter(dict.fromkeys(band))) if band is not None else None) else None)
        base = top
    _steps(ax, w, d, band, color=INK, lw=0.8)
    if upto is None or k == len(units):
        _steps(ax, w, m["model"].to_numpy(), band, color=BAD, lw=1.1)
    res = d - base
    _steps(axr, w, res, band, color=MUTED, lw=0.7)
    axr.fill_between(w[o], -s[o], s[o], color="#1a73e8", alpha=0.12, lw=0, step="mid")
    axr.axhline(0, color=FAINT, lw=1)
    chi = np.nansum((res / s) ** 2) / max(len(w) - len(run.names), 1)
    ax.text(0.995, 0.97, f"χ²_red = {chi:.2f}" + ("" if (upto is None or k == len(units)) else "  (partial model)"),
            transform=ax.transAxes, ha="right", va="top", fontsize=15, color=INK)
    _style(ax, None, "Fν − continuum [Jy]"); _style(axr, "wavelength [µm]", "residual")
    ax.tick_params(labelbottom=False)
    lim = np.nanpercentile(np.abs(res), 99.5) * 1.3
    axr.set_ylim(-lim, lim)
    ax.legend(fontsize=13, frameon=False, ncol=min(len(units), 6), loc="upper left")
    return fig


def _posterior_2d(ax, x, y, colour):
    h, xe, ye = np.histogram2d(x, y, bins=50)
    try:
        from scipy.ndimage import gaussian_filter
        h = gaussian_filter(h, 1.0)
    except Exception:
        pass
    hs = np.sort(h.ravel())[::-1]; cs = np.cumsum(hs) / hs.sum()
    levels = sorted({hs[np.searchsorted(cs, q)] for q in (0.865, 0.393)})
    ax.scatter(x[::max(len(x) // 3000, 1)], y[::max(len(y) // 3000, 1)], s=2, color=colour, alpha=0.15, lw=0)
    if len(levels) == 2 and levels[0] < levels[1]:
        ax.contour(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), h.T, levels=levels, colors=[colour], linewidths=[1.2, 2.0])


def _zoom_range(w, y, width=0.5):
    """the `width`-µm stretch where y carries the most flux (the component's most telling region)."""
    o = np.argsort(w); w, y = w[o], np.clip(np.nan_to_num(y[o]), 0, None)
    if not np.any(y) or w[-1] - w[0] <= width:
        return float(w[0]), float(w[-1])
    c = np.concatenate([[0], np.cumsum(y)])
    j = np.searchsorted(w, w + width)
    k = int(np.argmax(c[j] - c[np.arange(len(w))]))
    return float(w[k]), float(w[k] + width)


def _grid(run: Run, unit: str):
    f = run.folder / f"grid_{unit}.npz"
    if not f.exists():
        return None
    g = np.load(f)
    return g["logN"], g["T"], g["chi2"]


def _stage_theta(run: Run, stage: str, key: str):
    f = run.folder / stage
    try:
        pay = json.load(open(f))["payload"]
        return float(pay["theta"][pay["free"].index(key)])
    except Exception:
        return None


def panel_component(run: Run, unit: str, pred: dict | None = None):
    comp = run.component(unit)
    mol = comp.get("molecule", unit)
    col = _unit_colour(run, unit)
    fig = _figure()
    tied = comp.get("tie_to")
    _header(fig, run, f"{unit}  ·  {mol}", (f"tied to {tied}: shares its temperature and emitting area, column density / {comp.get('ratio', '—')}"
                                              if tied else f"{comp.get('kind', 'slab')} component: its model against the data with every other "
                                              "component subtracted"), unit)
    m = run.model
    w, d = m["wave"].to_numpy(), m["data"].to_numpy()
    band = m["band"].astype(str).to_numpy() if "band" in m else None
    mine = m["model_" + unit].to_numpy()
    others = m["model"].to_numpy() - mine
    z0, z1 = _zoom_range(w, mine)
    pk = np.nanmax(np.abs(mine)) if np.any(mine) else 0.1
    for k_, (rect, sel) in enumerate([([0.06, 0.55, 0.52, 0.29], np.ones(len(w), bool)),
                                       ([0.06, 0.10, 0.52, 0.34], (w >= z0) & (w <= z1))]):
        ax = fig.add_axes(rect)
        bb = band[sel] if band is not None else None
        _steps(ax, w[sel], (d - others)[sel], bb, color=INK, lw=0.8 if k_ else 0.6, label="data − other components")
        if pred is not None and unit in pred["units"]:
            lo_, _, hi_ = pred["units"][unit]
            _band(ax, w[sel], lo_[sel], hi_[sel], bb, color=col, alpha=0.35)
        _steps(ax, w[sel], mine[sel], bb, color=col, lw=1.8 if k_ else 1.3, label=f"{unit} model")
        lo, hi = np.nanpercentile((d - others)[sel], [0.5, 99.5])
        ax.set_ylim(min(lo, -0.1 * pk) - 0.05 * pk, max(hi, pk) * 1.18)
        _style(ax, "wavelength [µm]" if k_ else None, "Fν [Jy]")
        if k_ == 0:
            ax.axvspan(z0, z1, color=col, alpha=0.12, lw=0)
            ax.legend(fontsize=13, frameon=True, framealpha=0.9, edgecolor="none", loc="upper left", ncol=2)
            ax.set_title("whole fit range", fontsize=13, color=MUTED, loc="right")
        else:
            ax.set_xlim(z0, z1)
            ax.set_title(f"zoom: {z0:.2f}–{z1:.2f} µm, where {unit} contributes most", fontsize=13, color=MUTED, loc="right")
    # parameter card
    X0, XB, XP, dy, fs = 0.62, 0.745, 0.815, 0.037, 15
    fig.text(X0, 0.845, "PARAMETERS", fontsize=13, color=FAINT, fontweight="bold")
    fig.text(XB, 0.845, "best fit", fontsize=13, color=FAINT); fig.text(XP, 0.845, "posterior median  −16 % / +84 %", fontsize=13, color=FAINT)
    bp = (run.best.get("params") or {}).get(unit, {})
    rows = [("log N  [cm⁻²]", "logN", 2), ("T  [K]", "T", 0), ("log R  [au]", "logR", 3), ("v  [km/s]", "rv", 1), ("f_c", "fc", 2)]
    y = 0.805
    for lab, key, prec in rows:
        b = bp.get(key)
        po = _post(run, f"{unit}.{key}")
        if b is None and po is None:
            continue
        fig.text(X0, y, lab, fontsize=fs, color=MUTED)
        fig.text(XB, y, _fmt(b, prec), fontsize=fs, color=INK)
        if po:
            med, mi, pl, edge = po
            fig.text(XP, y, f"{med:.{prec}f}  −{mi:.{prec}f} / +{pl:.{prec}f}" + ("   ◂ at prior bound" if edge else ""), fontsize=fs,
                     color=(WARN if edge else INK))
        else:
            pt = _post(run, f"{tied}.{key}") if tied and key in ("logN", "T", "logR") else None
            if pt:
                off = -np.log10(float(comp.get("ratio") or 1.0)) if key == "logN" else 0.0
                fig.text(XP, y, f"{pt[0] + off:.{prec}f}  −{pt[1]:.{prec}f} / +{pt[2]:.{prec}f}", fontsize=fs, color=INK)
            else:
                fig.text(XP, y, "tied" if tied and key in ("logN", "T", "logR") else "fixed", fontsize=fs - 1, color=FAINT)
        y -= dy
    for lab, key, prec in [("log N·A  [cm⁻² au²]", "logNA", 2), ("R  [au]", "R_au", 3), ("log # molecules", "logNmol", 2)]:
        po = _post(run, f"{unit}.{key}")
        if po:
            fig.text(X0, y, lab, fontsize=fs, color=MUTED)
            fig.text(XB, y, "derived", fontsize=fs - 1, color=FAINT)
            fig.text(XP, y, f"{po[0]:.{prec}f}  −{po[1]:.{prec}f} / +{po[2]:.{prec}f}", fontsize=fs, color=INK); y -= dy
    y -= 0.012
    tmax = (run.best.get("tau_max") or {}).get(unit)
    thin = run.tau_flags.get(unit)
    det = None
    if run.detections is not None and "component" in run.detections:
        r = run.detections.set_index("component")
        det = r.loc[unit] if unit in r.index else None
    info = []
    if tmax is not None:
        info.append(("peak line optical depth", f"τ_max = {tmax:.2g}" + ("  (optically thick)" if tmax > 1 else "  (thin)")))
    if thin is not None:
        info.append(("posterior samples with τ_max < 1", f"{100 * thin:.0f} %"))
    if det is not None:
        info.append(("ΔBIC when removed", f"{det['delta_BIC']:+.1f}  " + ("needed (> 10)" if det["delta_BIC"] > 10 else "not needed")))
    for k, v in info:
        fig.text(X0, y, k, fontsize=fs, color=MUTED); fig.text(XP, y, v, fontsize=fs, color=INK); y -= dy
    # grid -> optimiser -> posterior in the (log N, T) plane, and the (log N, log R) posterior
    S = run.posterior()
    src, shift = unit, 0.0
    if tied and S is not None and f"{unit}.logN" not in run.names and f"{tied}.logN" in run.names:
        src = tied   # a tied component's posterior is its parent's, with log N shifted by the ratio
        try:
            shift = -np.log10(float(comp.get("ratio") or 1.0))
        except (TypeError, ValueError):
            shift = 0.0
    has_post = S is not None and f"{src}.T" in run.names and f"{src}.logN" in run.names
    g = _grid(run, unit)
    if g is None and not has_post:
        return fig
    a1 = fig.add_axes([0.655, 0.12, 0.14, 0.27]); a2 = fig.add_axes([0.84, 0.12, 0.13, 0.27])
    if g is not None:
        gN, gT, c2 = g
        dc = c2 - np.nanmin(c2)
        a1.pcolormesh(gN, gT, np.log10(dc + 1), shading="nearest", cmap="Greys", alpha=0.55)
        i, j = np.unravel_index(np.nanargmin(c2), c2.shape)
        a1.plot(gN[j], gT[i], "s", ms=7, mfc="none", mec=MUTED, mew=1.5, label="grid")
    if has_post:
        iN, iT = run.names.index(f"{src}.logN"), run.names.index(f"{src}.T")
        S = S.copy(); S[:, iN] += shift
        _posterior_2d(a1, S[:, iN], S[:, iT], col)
        a1.plot(bp.get("logN", np.nan), bp.get("T", np.nan), "+", ms=12, mew=2, color=INK, label="optimiser")
        if f"{src}.logR" in run.names:
            iR = run.names.index(f"{src}.logR")
            _posterior_2d(a2, S[:, iN], S[:, iR], col)
            a2.plot(bp.get("logN", np.nan), bp.get("logR", np.nan), "+", ms=12, mew=2, color=INK)
            _style(a2, "log N", "log R"); a2.set_title("posterior", fontsize=12, color=MUTED, loc="left")
        else:
            a2.axis("off")
    else:
        a2.axis("off")
    if has_post and g is not None:
        for arr, gv, setl in ((S[:, iN], gN, a1.set_xlim), (S[:, iT], gT, a1.set_ylim)):
            lo, hi = np.percentile(arr, [0.5, 99.5]); step = np.median(np.diff(gv)) if len(gv) > 1 else 0
            pad = max(0.6 * (hi - lo), 1.5 * step)
            setl(max(lo - pad, gv.min() - step / 2), min(hi + pad, gv.max() + step / 2))
    _style(a1, "log N", "T [K]")
    a1.set_title(("grid → " if g is not None else "") + "optimiser → posterior", fontsize=12, color=MUTED, loc="left")
    a1.legend(fontsize=10, frameon=True, framealpha=0.85, edgecolor="none", loc="upper right", handletextpad=0.3)
    for a_ in (a1, a2):
        a_.tick_params(labelsize=11); a_.locator_params(nbins=4)
    return fig


def panel_predictive(run: Run, pred: dict):
    fig = _figure()
    _header(fig, run, "Model uncertainty", f"Band = 16–84 % of {pred['n']} models drawn from the posterior; black = data. "
            "Below: the regions where it is widest, and its width against the noise", "predictive")
    m = run.model
    w, d = m["wave"].to_numpy(), m["data"].to_numpy()
    band = m["band"].astype(str).to_numpy() if "band" in m else None
    lo, med, hi = pred["total"]
    o = np.argsort(w, kind="stable")
    ax = fig.add_axes([0.06, 0.55, 0.90, 0.30])
    _steps(ax, w, d, band, color=INK, lw=0.6, label="data")
    _band(ax, w, lo, hi, band, color=BAD, alpha=0.35, label="16–84 % posterior models")
    _steps(ax, w, med, band, color=BAD, lw=0.8, label="median model")
    _style(ax, None, "Fν − continuum [Jy]"); ax.legend(fontsize=12, frameon=False, ncol=3, loc="upper left")
    # pick up to three zooms where components are least certain (relative 16-84 width of their flux)
    cand = []
    for u, (ulo, umed, uhi) in pred["units"].items():
        if np.nanmax(umed) <= 0.02 * np.nanmax(med):
            continue
        z0, z1 = _zoom_range(w, umed)
        sel = (w >= z0) & (w <= z1)
        rel = float(np.sum(uhi[sel] - ulo[sel]) / max(np.sum(umed[sel]), 1e-12))
        cand.append((rel, u, z0, z1))
    zooms = []
    for rel, u, z0, z1 in sorted(cand, reverse=True):
        if all(z1 < a or z0 > b for _, a, b in zooms):
            zooms.append((u, z0, z1))
        if len(zooms) == 3:
            break
    sig = m["sigma"].to_numpy()
    for i, (u, z0, z1) in enumerate(zooms):
        ax.axvspan(z0, z1, color=_unit_colour(run, u), alpha=0.12, lw=0)
        a = fig.add_axes([0.06 + i * 0.31, 0.215, 0.28, 0.25])
        sel = (w >= z0) & (w <= z1)
        bb = band[sel] if band is not None else None
        _steps(a, w[sel], d[sel], bb, color=INK, lw=0.7, label="data")
        _band(a, w[sel], lo[sel], hi[sel], bb, color=MUTED, alpha=0.3)
        _steps(a, w[sel], med[sel], bb, color=MUTED, lw=1.2, ls="--", label="total model")
        for v, (ulo, umed, uhi) in pred["units"].items():
            if np.nanmax(umed[sel]) > 0.05 * max(np.nanmax(med[sel]), 1e-12):
                c_ = _unit_colour(run, v)
                _band(a, w[sel], ulo[sel], uhi[sel], bb, color=c_, alpha=0.35)
                _steps(a, w[sel], umed[sel], bb, color=c_, lw=1.0, label=v)
        a.set_xlim(z0, z1)
        _style(a, None, "Fν [Jy]" if i == 0 else None); a.tick_params(labelbottom=False)
        a.set_title(f"{z0:.2f}–{z1:.2f} µm", fontsize=13, color=MUTED, loc="left")
        a.legend(fontsize=10, frameon=True, framealpha=0.9, edgecolor="none", loc="upper left", ncol=2)
        # how wide is the model band compared with the noise?
        s_ = fig.add_axes([0.06 + i * 0.31, 0.08, 0.28, 0.11], sharex=a)
        _steps(s_, w[sel], 1e3 * sig[sel], bb, color=FAINT, lw=1.0, label="noise σ")
        _steps(s_, w[sel], 1e3 * 0.5 * (hi - lo)[sel], bb, color=BAD, lw=1.0, label="model ± (half 16–84 width)")
        _style(s_, "wavelength [µm]", "mJy" if i == 0 else None)
        s_.set_ylim(0, 1.25 * max(np.nanmax(1e3 * sig[sel]), np.nanmax(1e3 * 0.5 * (hi - lo)[sel])))
        if i == 0:
            s_.legend(fontsize=9, frameon=False, loc="upper left", ncol=2)
    return fig


def panel_evidence(run: Run):
    fig = _figure()
    _header(fig, run, "Which components are needed?",
            "ΔBIC from refitting without each component (> 10 = strongly needed), and how optically thick each one is", "evidence")
    if run.detections is None or not len(run.detections):
        fig.text(0.04, 0.8, "No detections.csv in this run.", fontsize=18, color=MUTED)
        return fig
    dt = run.detections.copy()
    n = len(dt)
    ax = fig.add_axes([0.16, 0.18, 0.42, 0.66])
    yy = np.arange(n)[::-1]
    v = np.maximum(dt["delta_BIC"].to_numpy(float), 0.5)
    ax.barh(yy, v, color=[_unit_colour(run, u) for u in dt["component"]], height=0.6)
    ax.set_xscale("log"); ax.axvline(10, color=BAD, lw=1.5, ls="--")
    ax.text(10, n - 0.4, "  ΔBIC = 10", color=BAD, fontsize=13, va="bottom")
    ax.set_yticks(yy); ax.set_yticklabels(dt["component"], fontsize=15)
    for yi, val in zip(yy, dt["delta_BIC"]):
        ax.text(max(val, 0.5) * 1.15, yi, f"{val:+.0f}", va="center", fontsize=13, color=INK)
    ax.set_ylim(-0.6, n - 0.2)
    _style(ax, "ΔBIC (log scale)"); ax.grid(axis="y", visible=False)
    ax2 = fig.add_axes([0.66, 0.18, 0.26, 0.66])
    tm = np.maximum(dt["tau_max"].to_numpy(float), 1e-3)
    ax2.barh(yy, tm, color=[_unit_colour(run, u) for u in dt["component"]], height=0.6, alpha=0.75)
    ax2.set_xscale("log"); ax2.axvline(1, color=MUTED, lw=1.2, ls=":")
    ax2.text(1, n - 0.4, "  τ = 1", color=MUTED, fontsize=13, va="bottom")
    for yi, u, t_ in zip(yy, dt["component"], tm):
        thin = run.tau_flags.get(u)
        ax2.text(t_ * 1.2, yi, f"{t_:.2g}" + (f"   ({100 * thin:.0f} % thin)" if thin is not None else ""), va="center", fontsize=12, color=INK)
    ax2.set_yticks(yy); ax2.set_yticklabels([]); ax2.set_ylim(-0.6, n - 0.2); ax2.set_xlim(right=tm.max() * 30)
    _style(ax2, "peak line optical depth τ_max (log scale)"); ax2.grid(axis="y", visible=False)
    fig.text(0.04, 0.06, "Thick components constrain N·A better than N alone; components that are thin in every posterior sample "
             "trade N against area freely.", fontsize=13, color=MUTED)
    return fig


def panel_image(run: Run, file: str, title: str, subtitle: str, page: str):
    """an existing figure from the run (posterior_predictive.png, corner.png ...), fitted below the header."""
    fig = _figure()
    _header(fig, run, title, subtitle, page)
    import matplotlib.image as mpimg
    img = mpimg.imread(run.folder / file)
    rgb = img[..., :3] if img.ndim == 3 else img
    ink = np.where((rgb if rgb.ndim == 2 else rgb.min(axis=-1)) < (0.97 if img.dtype != np.uint8 else 247))
    if len(ink[0]):
        r0, r1, c0, c1 = ink[0].min(), ink[0].max(), ink[1].min(), ink[1].max()
        img = img[max(r0 - 10, 0):r1 + 10, max(c0 - 10, 0):c1 + 10]
    ax = fig.add_axes([0.04, 0.07, 0.92, 0.80]); ax.imshow(img, interpolation="antialiased"); ax.axis("off")
    return fig


def panel_marginals(run: Run):
    fig = _figure()
    _header(fig, run, "Uncertainties", "1-D posterior of every free parameter: median (solid), 16th–84th percentiles (shaded), "
            "best fit (dashed), prior bounds (grey edges)", "marginals")
    S = run.posterior()
    if S is None:
        fig.text(0.04, 0.8, "No chain in this run.", fontsize=18, color=MUTED)
        return fig
    n = len(run.names)
    nc = 3 if n <= 9 else (4 if n <= 16 else 6)
    nr = int(np.ceil(n / nc))
    gw, gh = 0.92 / nc, 0.78 / nr
    bounds = run.summary.set_index("parameter") if run.summary is not None else None
    for i, name in enumerate(run.names):
        r, c = divmod(i, nc)
        ax = fig.add_axes([0.04 + c * gw + 0.012, 0.86 - (r + 1) * gh + 0.045, gw - 0.03, gh - 0.075])
        unit, key = name.split(".", 1) if "." in name else (name, "")
        col = _unit_colour(run, unit)
        x = S[:, i]
        lo, med, hi = np.percentile(x, [16, 50, 84])
        ax.hist(x, bins=50, color=col, alpha=0.35, histtype="stepfilled")
        ax.hist(x, bins=50, color=col, histtype="step", lw=1.2)
        ax.axvspan(lo, hi, color=col, alpha=0.15, lw=0); ax.axvline(med, color=col, lw=2)
        bv = ((run.best.get("params") or {}).get(unit) or {}).get(key)
        if bv is not None:
            ax.axvline(bv, color=INK, lw=1.2, ls="--")
        if bounds is not None and name in bounds.index:
            b0, b1 = float(bounds.loc[name, "lo_bound"]), float(bounds.loc[name, "hi_bound"])
            x0, x1 = ax.get_xlim()
            for bb in (b0, b1):
                if x0 - 0.05 * (x1 - x0) <= bb <= x1 + 0.05 * (x1 - x0):
                    ax.axvspan(bb - 0.01 * (x1 - x0), bb + 0.01 * (x1 - x0), color=MUTED, alpha=0.5, lw=0)
        edge = bool(bounds.loc[name, "at_edge"]) if bounds is not None and name in bounds.index else False
        prec = 0 if key == "T" else (3 if key == "logR" else 2)
        ax.set_title(f"{name} = {med:.{prec}f} −{med - lo:.{prec}f} +{hi - med:.{prec}f}" + ("  ◂ bound" if edge else ""), fontsize=12,
                     color=(WARN if edge else INK), loc="left")
        ax.set_yticks([])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=10, colors=MUTED); ax.locator_params(axis="x", nbins=5)
    return fig


def panel_table(run: Run):
    fig = _figure()
    _header(fig, run, "All parameters", "Posterior medians with 16th/84th-percentile offsets; amber = within 2 % of a prior bound", "table")
    ax = fig.add_axes([0.04, 0.08, 0.92, 0.78]); ax.axis("off")
    if run.summary is None:
        ax.text(0, 1, "No summary.csv (the MCMC stage did not run).", fontsize=18, color=MUTED, va="top")
        return fig
    s = run.summary
    if "reported" in s:
        s = s[s["reported"].astype(bool)] if s["reported"].dtype != object else s
    s = s.head(30)
    cells = [[r.parameter, f"{r.median:.4g}", f"−{r.minus:.3g}", f"+{r.plus:.3g}", f"[{r.lo_bound:g}, {r.hi_bound:g}]" if np.isfinite(r.lo_bound) else "—",
              "yes" if bool(r.at_edge) else ""] for r in s.itertuples()]
    tab = ax.table(cellText=cells, colLabels=["parameter", "median", "−", "+", "prior bounds", "at bound"], loc="upper left",
                   cellLoc="left", colLoc="left", colWidths=[0.30, 0.14, 0.12, 0.12, 0.22, 0.10])
    tab.auto_set_font_size(False); tab.set_fontsize(13); tab.scale(1, 1.55)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_text_props(color=MUTED, fontweight="bold")
        elif cells[r - 1][5]:
            cell.set_text_props(color=WARN)
    return fig


def panel_correlation(run: Run):
    fig = _figure()
    _header(fig, run, "Correlations", "Posterior correlation coefficients between every pair of parameters, within and between molecules", "correlation")
    if run.correlation is None:
        fig.text(0.04, 0.8, "No correlation.csv in this run.", fontsize=18, color=MUTED)
        return fig
    c = run.correlation
    ax = fig.add_axes([0.24, 0.08, 0.52, 0.80])
    im = ax.imshow(c.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(c.columns))); ax.set_xticklabels(c.columns, rotation=60, ha="right", fontsize=11)
    ax.set_yticks(range(len(c.index))); ax.set_yticklabels(c.index, fontsize=11)
    for i in range(len(c.index)):
        for j in range(len(c.columns)):
            v = c.iat[i, j]
            if abs(v) >= 0.5 and i != j:
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8, color="white" if abs(v) > 0.75 else INK)
    cax = fig.add_axes([0.78, 0.2, 0.012, 0.55]); fig.colorbar(im, cax=cax).ax.tick_params(labelsize=12)
    return fig


def panel_traces(run: Run, max_params: int = 9):
    fig = _figure()
    _header(fig, run, "MCMC traces", "Every walker for the first parameters; the shaded part is discarded as burn-in", "traces")
    if run.chain is None:
        fig.text(0.04, 0.8, "No chain in this run.", fontsize=18, color=MUTED)
        return fig
    n = min(len(run.names), max_params)
    burn = run.chain.shape[0] // 5
    for i in range(n):
        r, c = divmod(i, 3)
        ax = fig.add_axes([0.06 + c * 0.31, 0.64 - r * 0.26, 0.27, 0.19])
        ax.plot(run.chain[:, :, i], color="#1a73e8", alpha=0.08, lw=0.6)
        ax.axvspan(0, burn, color=GRID, alpha=0.6, lw=0)
        ax.set_title(run.names[i], fontsize=13, color=INK, loc="left")
        _style(ax)
        ax.tick_params(labelsize=10)
    return fig


# ---------------------------------------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------------------------------------
def build_panels(run: Run, pred: dict | None = None, first=None, passes=None) -> list[tuple[str, object, float]]:
    """(name, figure, relative duration) in display order; build-up frames of the fit are named *_build_k (video only)."""
    out = [("overview", panel_overview(run), 1.4), ("data", panel_data(run), 1.0),
           ("continuum", panel_continuum(run, first, passes), 1.3)]
    units = _unit_columns(run)
    for k in range(1, len(units)):
        out.append((f"fit_build_{k}", panel_fit(run, upto=k), 0.45))
    out.append(("fit", panel_fit(run), 1.2))
    w = run.model["wave"].to_numpy()
    span = float(w.max() - w.min())
    out.append(("molecules", panel_molecules(run), 1.3))
    nchunk = int(np.ceil(span / 1.4)) if span > 1.6 else 1
    if nchunk > 1:
        edges = np.linspace(w.min(), w.max(), nchunk + 1)
        for i in range(nchunk):
            out.append((f"molecules_{i + 1}", panel_molecules(run, edges[i], edges[i + 1], page=f"molecules {i + 1}/{nchunk}"), 1.4))
    if pred is not None:
        out.append(("model_uncertainty", panel_predictive(run, pred), 1.2))
    elif (run.folder / "posterior_predictive.png").exists():
        out.append(("model_uncertainty", panel_image(run, "posterior_predictive.png", "Model uncertainty",
                    "Spread of models drawn from the posterior against the data", "posterior predictive"), 1.0))
    if run.detections is not None:
        out.append(("evidence", panel_evidence(run), 1.2))
    for u in units:
        out.append((f"component_{u}", panel_component(run, u, pred), 1.4))
    if run.chain is not None:
        out.append(("uncertainties", panel_marginals(run), 1.2))
    out.append(("table", panel_table(run), 1.2))
    out.append(("correlation", panel_correlation(run), 1.0))
    if run.chain is not None:
        out.append(("traces", panel_traces(run), 0.8))
    i = 0
    named = []
    for name, fig, rel in out:
        if "_build_" in name:
            named.append((name, fig, rel))
        else:
            named.append((f"{i:02d}_{name}", fig, rel)); i += 1
    return named


def write_video(frames: list[np.ndarray], durations: list[float], path: Path, fps: int = 30, fade: float = 0.4) -> bool:
    exe = shutil.which("ffmpeg")
    if not exe:
        return False
    cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    nf = int(round(fade * fps))
    for i, (img, dur) in enumerate(zip(frames, durations)):
        hold = max(int(round(dur * fps)) - (nf if i + 1 < len(frames) else 0), 1)
        for _ in range(hold):
            p.stdin.write(img.tobytes())
        if i + 1 < len(frames) and nf > 0:
            a, b = img.astype(np.float32), frames[i + 1].astype(np.float32)
            for k in range(nf):
                u = (k + 1) / (nf + 1)
                p.stdin.write((a * (1 - u) + b * u).astype(np.uint8).tobytes())
    p.stdin.close(); p.wait()
    return p.returncode == 0


def showcase(path: str, out: str | None = None, video: bool = True, seconds: float = 4.0, fps: int = 30,
             draws: int = 200, say=print) -> dict:
    """Render the panels (PNG) and, optionally, the MP4 for the run in `path`. Returns the written files."""
    run = load_run(path)
    outdir = Path(out) if out else run.folder / "showcase"
    outdir.mkdir(parents=True, exist_ok=True)
    pred = predictive(run, n=draws, say=say) if draws > 0 else None
    passes = refinement_passes(run)
    first = first_continuum(run, say=say) if passes else None
    panels = build_panels(run, pred, first, passes)
    frames, durs, pngs = [], [], []
    for name, fig, rel in panels:
        img = _rgb(fig)
        if "_build_" not in name:
            fn = outdir / f"{name}.png"
            fig.savefig(fn, dpi=DPI, facecolor="white")
            pngs.append(str(fn))
        frames.append(img); durs.append(rel * seconds)
    res = {"folder": str(outdir), "png": pngs, "video": None}
    if video:
        mp4 = outdir / f"{run.target.replace(' ', '_')}_showcase.mp4"
        if write_video(frames, durs, mp4, fps=fps):
            res["video"] = str(mp4)
        else:
            say("ffmpeg not found (or failed): wrote the PNG panels only")
    say(f"showcase: {len(pngs)} panels" + (f" + {Path(res['video']).name}" if res["video"] else "") + f" in {outdir}")
    return res
