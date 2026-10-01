"""Interactive web app (Panel + Bokeh) for simultaneous LTE slab fitting.

    jalebi serve --port 5006 --data-root /path/to/YSOs_MIRI_reduced_cube_data_Aug2026
or, in a notebook:
    from jalebi.app import make_app; make_app(data_root=...).servable()

Layout ("observatory" dark theme): a header with the module switcher and live status chips, a slim
session sidebar (config YAML in/out, cached line lists) and independent *modules* (jalebi.modules):

  LTE slab fit      Data · Continuum · Model · Fit · Results · Batch
  Cube maps         line / velocity / channel / PV maps of IFU cubes (jalebi.cube); region spectra can be
                    sent to the LTE slab fit or to the rotation diagram
  Rotation diagram  population diagrams of H2, CO, OH, H2O ... (jalebi.rotdiag)

Modules other than the LTE fit are built the first time they are opened.  The Continuum, Model and Fit
workspaces use an *inspector* layout: plots on the left, a sticky, independently scrolling control column on
the right, so the component sliders are always at hand while the spectrum stays in view.

Everything the user sets maps onto the same ProjectConfig used by the CLI and can be exported
as YAML from the sidebar.

Interactive-speed notes (see JalebiApp._schedule / update_model_plot):
  * slider events are coalesced — one model evaluation per event-loop tick, never a backlog;
  * the per-pixel noise is cached per spectrum instead of being recomputed on every move;
  * the display model memoises per-component fluxes, so moving one slider only re-evaluates
    that component (SlabModel.unit_cache);
  * each plotted series lives in a single ColumnDataSource (sub-bands separated by NaN gaps)
    instead of one source per sub-band, and only the changed columns are re-sent.
"""
from __future__ import annotations

import glob
import io
import os
import re
import threading
import time
import traceback

import numpy as np
import pandas as pd
import panel as pn
from bokeh.models import BoxAnnotation, ColumnDataSource, HoverTool, Label, Legend, LegendItem, Range1d, Span
from bokeh.palettes import Category20
from bokeh.plotting import figure

from .config import ComponentConfig, ContinuumConfig, ExtractionConfig, ProjectConfig
from .continuum import METHODS, in_ranges
from .data import BAND_ORDER, Spectrum, estimate_noise, load_spectrum, spike_filter, mrs_psf_fwhm
from .linedata import available_linelists, data_dir
from .model import Component, build_model
from .molecules import DEFAULT_WINDOWS, FEATURES, MOLECULES, get_molecule
from ._banner import acronym_html
from .pipeline import (RunResult, build_problem, catalogue_row, prepare, run_grid_stage,
                       run_mcmc_stage, run_optimise_stage, save_results)

_ACRONYM_HTML = acronym_html()      # "JWST Analysis of Line Emission with Bayesian Inference", initials in amber

# ----------------------------------------------------------------------------------------------
# Theme
# ----------------------------------------------------------------------------------------------

class _Palette:
    bg = "#0b1020"          # page
    panel = "#121a2e"       # cards / figure background
    panel2 = "#1a2440"      # inputs, nested cards
    border = "rgba(148,163,208,0.16)"
    text = "#e7eaf3"
    muted = "#8e97ad"
    accent = "#f5b543"      # amber
    teal = "#3ad0c4"
    pink = "#ff5c8a"        # total model
    danger = "#ff6b6b"
    data = "#c9d1e3"        # observed spectrum
    grid = "rgba(148,163,208,0.10)"


PAL = _Palette()
FONT = "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif"
MONO = "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
FONT_URL = "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap"

# CSS injected into every component (Panel applies raw_css inside each shadow root as well)
RAW_CSS = f"""
:host {{
  --sf-bg: {PAL.bg}; --sf-panel: {PAL.panel}; --sf-panel2: {PAL.panel2}; --sf-border: {PAL.border};
  --sf-text: {PAL.text}; --sf-muted: {PAL.muted}; --sf-accent: {PAL.accent}; --sf-teal: {PAL.teal};
  --sf-pink: {PAL.pink}; --sf-danger: {PAL.danger};
  --sf-font: {FONT}; --sf-mono: {MONO};
}}
:host(.sf-panel) {{
  background: var(--sf-panel); border: 1px solid var(--sf-border); border-radius: 14px;
  padding: 12px 14px 10px 14px; box-sizing: border-box;
}}
:host(.sf-toolbar) {{
  background: var(--sf-panel); border: 1px solid var(--sf-border); border-radius: 14px;
  padding: 6px 14px 4px 14px; box-sizing: border-box;
}}
:host(.sf-paper) {{ background: #fbfbfd; border-radius: 12px; padding: 8px; box-sizing: border-box; }}
:host(.sf-inspector) {{
  background: var(--sf-panel); border: 1px solid var(--sf-border); border-radius: 14px;
  padding: 10px 12px 12px 12px; box-sizing: border-box;
}}
:host(.sf-comp) {{
  background: var(--sf-panel2) !important; border: 1px solid var(--sf-border); border-radius: 12px;
  padding: 0 8px 6px 8px; box-sizing: border-box;
}}
:host(.sf-comp) .card-header {{ background: transparent !important; padding: 4px 0 !important; min-height: 34px; }}
:host(.sf-comp) .card-button {{ color: var(--sf-muted); }}
:host(.sf-sub) {{ background: rgba(255,255,255,0.03) !important; border: 1px dashed var(--sf-border); border-radius: 10px; padding: 0 6px 4px 6px; }}
:host(.sf-sub) .card-header {{ background: transparent !important; padding: 2px 0 !important; min-height: 28px; }}
:host(.sf-sub) .card-header .sf-subtitle {{ font-size: 11.5px; color: var(--sf-muted); font-weight: 600; letter-spacing: .02em; }}
:host(.card) {{ background: var(--sf-panel) !important; }}
.card-header {{ background: var(--sf-panel2) !important; border-radius: 10px; color: var(--sf-text) !important; cursor: pointer; }}
.card-button {{ flex: 0 0 12px !important; color: var(--sf-muted) !important; margin-right: 8px !important; }}
.card-button > svg {{ stroke: var(--sf-muted) !important; }}
.card-header:hover .card-button > svg {{ stroke: var(--sf-accent) !important; }}
.sf-comphead, .sf-subtitle {{ min-width: 0; }}

/* editable sliders: compact right-aligned value box */
:host(.slider-edit) {{ max-width: 132px !important; margin-left: auto !important; }}
:host(.slider-edit) .bk-input-group .bk-input {{ text-align: right; font-family: var(--sf-mono) !important; font-size: 12.5px !important;
  padding: 2px 22px 2px 8px !important; background: var(--sf-panel2) !important; border: 1px solid var(--sf-border) !important; border-radius: 8px !important; }}
input[type="file"] {{ color: var(--sf-muted); background: var(--sf-panel2); border: 1px solid var(--sf-border); border-radius: 8px; padding: 6px; font-family: var(--sf-font); font-size: 12.5px; }}
input[type="file"]::file-selector-button {{ background: var(--sf-panel); color: var(--sf-text); border: 1px solid var(--sf-border); border-radius: 6px; padding: 4px 10px; margin-right: 10px; font-weight: 600; cursor: pointer; }}
input[type="checkbox"] {{ accent-color: var(--sf-accent); width: 15px; height: 15px; }}
progress {{ -webkit-appearance: none; appearance: none; border: none; background: #263050; border-radius: 6px; height: 10px; }}
progress::-webkit-progress-bar {{ background: #263050; border-radius: 6px; }}
progress::-webkit-progress-value {{ background: var(--sf-accent); border-radius: 6px; transition: width .3s; }}
progress::-moz-progress-bar {{ background: var(--sf-accent); border-radius: 6px; }}

.sf-label {{ font-family: var(--sf-font); font-size: 11px; letter-spacing: .14em; text-transform: uppercase;
             color: var(--sf-muted); font-weight: 700; margin: 2px 0 8px 0; }}
.sf-title {{ font-family: var(--sf-font); font-size: 20px; font-weight: 700; letter-spacing: -.01em; color: var(--sf-text); margin: 0; }}
.sf-sub {{ font-family: var(--sf-font); color: var(--sf-muted); font-size: 12.5px; line-height: 1.5; }}
.sf-note {{ font-family: var(--sf-font); color: var(--sf-muted); font-size: 12px; line-height: 1.45; }}
.sf-mono {{ font-family: var(--sf-mono); font-size: 12px; }}
.sf-chip {{ display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px; font-size: 12px;
            background: rgba(255,255,255,0.05); border: 1px solid var(--sf-border); color: var(--sf-text);
            margin: 2px 6px 2px 0; font-family: var(--sf-mono); white-space: nowrap; }}
.sf-chip b {{ font-weight: 600; color: #fff; }}
.sf-chip.accent {{ background: rgba(245,181,67,0.14); border-color: rgba(245,181,67,0.55); color: #ffd58a; }}
.sf-chip.teal {{ background: rgba(58,208,196,0.12); border-color: rgba(58,208,196,0.5); color: #9ff0e8; }}
.sf-chip.pink {{ background: rgba(255,92,138,0.12); border-color: rgba(255,92,138,0.5); color: #ffb3c8; }}
.sf-chip.warn {{ background: rgba(255,107,107,0.12); border-color: rgba(255,107,107,0.5); color: #ffb0b0; }}
.sf-chip.dim {{ color: var(--sf-muted); }}
.sf-dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; vertical-align: middle;
           box-shadow: 0 0 0 3px rgba(255,255,255,0.06); }}
.sf-comphead {{ display: flex; align-items: center; gap: 8px; width: 100%; font-family: var(--sf-font); }}
.sf-comphead .name {{ font-weight: 700; font-size: 13.5px; color: var(--sf-text); }}
.sf-comphead .mol {{ color: var(--sf-muted); font-size: 12px; }}
.sf-comphead .tau {{ margin-left: auto; font-family: var(--sf-mono); font-size: 11.5px; color: var(--sf-muted); }}
.sf-comphead .tau.thick {{ color: #ffd58a; }} .sf-comphead .tau.thin {{ color: #9ff0e8; }}
.sf-comphead .off {{ opacity: .45; text-decoration: line-through; }}
.sf-stage {{ font-family: var(--sf-font); font-size: 22px; font-weight: 700; color: var(--sf-text); }}
.sf-stage small {{ font-size: 13px; color: var(--sf-muted); font-weight: 500; margin-left: 10px; }}
.sf-log {{ font-family: var(--sf-mono); font-size: 11.5px; line-height: 1.5; white-space: pre-wrap; color: #c9d1e3;
           background: #0b1020; border-radius: 10px; padding: 10px 12px; border: 1px solid var(--sf-border);
           max-height: 320px; overflow-y: auto; }}
.sf-table {{ font-family: var(--sf-font); font-size: 12.5px; border-collapse: collapse; width: 100%; color: var(--sf-text); }}
.sf-table th {{ text-align: left; color: var(--sf-muted); font-weight: 600; font-size: 11.5px; letter-spacing: .03em;
                padding: 6px 10px; border-bottom: 1px solid var(--sf-border); }}
.sf-table td {{ padding: 5px 10px; border-bottom: 1px solid rgba(148,163,208,0.08); font-family: var(--sf-mono); font-size: 12px; }}
.sf-table tr:last-child td {{ border-bottom: none; }}
.sf-kv {{ font-family: var(--sf-font); font-size: 12.5px; color: var(--sf-muted); line-height: 1.7; }}
.sf-kv b {{ color: var(--sf-text); font-weight: 600; }}

/* Bokeh / Panel widgets */
.bk-input, .bk-input:focus, select.bk-input {{
  background: var(--sf-panel2) !important; color: var(--sf-text) !important;
  border: 1px solid var(--sf-border) !important; border-radius: 8px !important; font-family: var(--sf-font);
}}
.bk-input:focus {{ border-color: rgba(245,181,67,0.6) !important; box-shadow: 0 0 0 3px rgba(245,181,67,0.12) !important; }}
.bk-input-group > label, .bk-slider-title {{ color: var(--sf-muted) !important; font-size: 12px; font-family: var(--sf-font); font-weight: 500; }}
.bk-slider-value {{ color: var(--sf-text) !important; font-family: var(--sf-mono); font-weight: 600; }}
.noUi-target {{ background: #263050 !important; border: none !important; box-shadow: none !important; }}
.noUi-connect {{ background: var(--sf-accent) !important; }}
.noUi-horizontal .noUi-handle {{ background: #fff !important; border: 2px solid var(--sf-accent) !important; box-shadow: none !important;
                                  border-radius: 50% !important; width: 16px !important; height: 16px !important; right: -8px !important; top: -6px !important; cursor: grab; }}
.noUi-handle::before, .noUi-handle::after {{ display: none !important; }}
.bk-btn {{ border-radius: 8px !important; font-weight: 600 !important; font-family: var(--sf-font); letter-spacing: .01em; }}
.bk-btn-primary {{ background: var(--sf-accent) !important; color: #1a1400 !important; border: 1px solid transparent !important; }}
.bk-btn-primary:hover {{ background: #ffc85c !important; }}
.bk-btn-default, .bk-btn-light {{ background: var(--sf-panel2) !important; color: var(--sf-text) !important; border: 1px solid var(--sf-border) !important; }}
.bk-btn-default:hover, .bk-btn-light:hover {{ border-color: rgba(245,181,67,0.6) !important; }}
.bk-btn-success {{ background: var(--sf-teal) !important; color: #03302c !important; border: 1px solid transparent !important; }}
.bk-btn-danger {{ background: rgba(255,107,107,0.14) !important; color: #ff9d9d !important; border: 1px solid rgba(255,107,107,0.5) !important; }}
.bk-btn:disabled {{ opacity: .45; }}
.bk-btn-group .bk-btn.bk-active {{ background: var(--sf-accent) !important; color: #1a1400 !important; border-color: transparent !important; }}
.bk-header {{ border-bottom: 1px solid var(--sf-border) !important; margin-bottom: 10px; }}
.bk-header .bk-tab {{ color: var(--sf-muted) !important; font-weight: 600; font-size: 13.5px; padding: 8px 18px !important;
                      border: none !important; border-bottom: 2px solid transparent !important; background: transparent !important; font-family: var(--sf-font); }}
.bk-header .bk-tab:hover {{ color: var(--sf-text) !important; }}
.bk-header .bk-tab.bk-active {{ color: var(--sf-accent) !important; border-bottom: 2px solid var(--sf-accent) !important; background: transparent !important; }}
:host(.sf-subtabs) .bk-header .bk-tab {{ font-size: 12.5px; padding: 6px 14px !important; }}
.bk-panel-models-widgets-Checkbox label, .bk-input-group label {{ font-family: var(--sf-font); }}
.choices__inner {{ background: var(--sf-panel2) !important; border: 1px solid var(--sf-border) !important; border-radius: 8px !important; color: var(--sf-text); }}
.choices__list--dropdown, .choices__list[aria-expanded] {{ background: var(--sf-panel2) !important; border: 1px solid var(--sf-border) !important; color: var(--sf-text); }}
.choices__list--multiple .choices__item {{ background: rgba(245,181,67,0.18) !important; border: 1px solid rgba(245,181,67,0.5) !important; color: #ffd58a !important; border-radius: 6px !important; }}
.bk-clearfix, progress {{ accent-color: var(--sf-accent); }}
.tabulator {{ background: var(--sf-panel) !important; border: 1px solid var(--sf-border) !important; border-radius: 10px; font-family: var(--sf-font); }}
.tabulator .tabulator-header, .tabulator .tabulator-header .tabulator-col {{ background: var(--sf-panel2) !important; color: var(--sf-muted) !important; border-color: var(--sf-border) !important; }}
.tabulator-row {{ background: var(--sf-panel) !important; color: var(--sf-text) !important; border-bottom: 1px solid rgba(148,163,208,0.08) !important; }}
.tabulator-row.tabulator-row-even {{ background: rgba(255,255,255,0.02) !important; }}
.tabulator-row .tabulator-cell {{ border-right: none !important; font-family: var(--sf-mono); font-size: 12px; }}
.tabulator-row.tabulator-selected {{ background: rgba(245,181,67,0.14) !important; }}
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-thumb {{ background: #2a3454; border-radius: 6px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
"""

# CSS for the template shell (light DOM only)
GLOBAL_CSS = f"""
body {{ font-family: {FONT}; }}
#sidebar {{ background: #0e1526 !important; border-right: 1px solid {PAL.border} !important; padding: 14px 12px !important; }}
#main {{ padding: 10px 16px 24px 16px !important; }}
#header {{ border-bottom: 1px solid {PAL.border}; }}
#header .title {{ font-family: {FONT}; font-weight: 700; letter-spacing: -.01em; }}
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-thumb {{ background: #2a3454; border-radius: 6px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
"""

pn.extension("tabulator", notifications=True, raw_css=[RAW_CSS], global_css=[GLOBAL_CSS])

# Sub-band colours.  On the dark background the classic Category20 set (dark blue, brown...) is hard
# to see with thin, dense MIRI spectra, so dark plots use a brighter, high-contrast set; light plots
# keep Category20.  Adjacent sub-bands alternate hue families so the overlaps stay readable.
BAND_COLOURS_LIGHT = {b: Category20[20][i] for i, b in enumerate(BAND_ORDER)}
BAND_COLOURS_DARK = dict(zip(BAND_ORDER, ["#5fb3ff", "#a9d8ff", "#ffb347", "#ffd98a", "#3ddc84", "#a8f0c0",
                                          "#ff6b6b", "#ffb3b3", "#c77dff", "#e3c3ff", "#f6c343", "#fff0a8"]))
BAND_COLOURS = BAND_COLOURS_DARK

# Figure themes: everything that is re-styled live by the "plots: dark / light" toggle in the sidebar.
PLOT_THEMES = {
    "dark": dict(bg=PAL.panel, grid=PAL.grid, tick=PAL.border, muted=PAL.muted, text=PAL.text, data=PAL.data,
                 legend_bg=PAL.bg, bands=BAND_COLOURS_DARK, zero=PAL.muted, accent=PAL.accent, total=PAL.pink,
                 mask="#8e97ad", mask_alpha=0.18),
    "light": dict(bg="#ffffff", grid="rgba(0,0,0,0.08)", tick="#c8ccd4", muted="#5b6472", text="#1f2937", data="#1f2937",
                  legend_bg="#ffffff", bands=BAND_COLOURS_LIGHT, zero="#6b7280", accent="#d97706", total="#e11d48",
                  mask="#9ca3af", mask_alpha=0.25),
}
DATA_FILES = os.path.join(os.path.dirname(__file__), "data_files")

def cached_releases() -> dict[str, list[str]]:
    """{molecule: [release, ...]} of the line lists in the cache (e.g. {'H2O': ['hitemp', 'hitran']})."""
    out: dict[str, list[str]] = {}
    try:
        for r in available_linelists().itertuples():
            out.setdefault(r.molecule, []).append(r.release)
    except Exception:
        pass
    return {k: sorted(v) for k, v in out.items()}


# figure heights: chosen so that toolbar + plots + residual fit a 900-px-tall viewport without scrolling
H_MODEL, H_RESID = 400, 150


# ----------------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------------

def _parse_ranges(text: str) -> list[tuple[float, float]]:
    out = []
    for part in text.replace(";", ",").split(","):
        p = part.strip()
        if not p:
            continue
        p = p.replace("–", "-")
        a, b = p.split("-")[:2] if p.count("-") == 1 else (p.split()[0], p.split()[-1])
        out.append((float(a), float(b)))
    return out


def _parse_named_ranges(text: str) -> dict[str, list[float]]:
    out = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        name, rng = line.split(":", 1)
        try:
            a, b = [float(x) for x in rng.replace("-", " ").replace(",", " ").split()[:2]]
            out[name.strip()] = [a, b]
        except ValueError:
            continue
    return out


def _fmt_named(d: dict) -> str:
    return "\n".join(f"{k}: {v[0]:.4g}, {v[1]:.4g}" for k, v in d.items())


def _bokeh_safe(fn):
    """Run a method that modifies Bokeh models inside Panel's unlocked context, so changes made
    from widget callbacks and periodic callbacks are dispatched to the browser."""
    import functools

    @functools.wraps(fn)
    def wrapper(self, *a, **k):
        try:
            ctx = pn.io.unlocked()
        except Exception:
            return fn(self, *a, **k)
        with ctx:
            return fn(self, *a, **k)
    return wrapper


def _fig(title="", height=320, y_label="F_ν [Jy]", x_range=None, x_label="wavelength [µm]", tools=None):
    kw = {"x_range": x_range} if x_range is not None else {}
    tools = tools or "xpan,xwheel_zoom,box_zoom,reset,save,tap"
    scroll = "xwheel_zoom" if "xwheel_zoom" in tools else ("wheel_zoom" if "wheel_zoom" in tools else "auto")
    f = figure(height=height, sizing_mode="stretch_width", x_axis_label=x_label, y_axis_label=y_label,
               tools=tools, active_scroll=scroll, output_backend="webgl", toolbar_location="above", **kw)
    f.toolbar.logo = None
    f.toolbar.autohide = True
    f.outline_line_color = None
    f.min_border_left = 56
    f.min_border_right = 14
    for ax in (f.xaxis, f.yaxis):
        ax.axis_line_color = None
        ax.minor_tick_line_color = None
        ax.major_label_text_font = FONT
        ax.major_label_text_font_size = "11px"
        ax.axis_label_text_font = FONT
        ax.axis_label_text_font_size = "11.5px"
        ax.axis_label_text_font_style = "normal"
    f.title.text = title
    f.title.text_font = FONT
    f.title.text_font_size = "13px"
    f.title.text_font_style = "bold"
    _theme_fig(f, PLOT_THEMES["dark"])
    return f


def _theme_fig(f, th: dict):
    """Apply the colour part of a plot theme to a figure (fonts and sizes are set once in _fig)."""
    f.background_fill_color = th["bg"]
    f.border_fill_color = th["bg"]
    f.border_fill_alpha = 1.0          # the template's dark Bokeh theme defaults this to 0
    f.xgrid.grid_line_color = th["grid"]
    f.ygrid.grid_line_color = th["grid"]
    for ax in (f.xaxis, f.yaxis):
        ax.major_tick_line_color = th["tick"]
        ax.major_label_text_color = th["muted"]
        ax.axis_label_text_color = th["muted"]
    f.title.text_color = th["text"]


def _style_legend(lg: Legend, location="top_left", orientation="vertical", th: dict | None = None):
    th = th or PLOT_THEMES["dark"]
    lg.location = location
    lg.orientation = orientation
    lg.click_policy = "hide"
    lg.background_fill_alpha = 0.6
    lg.label_text_font = FONT
    lg.label_text_font_size = "10.5px"
    lg.glyph_height = 12
    lg.glyph_width = 16
    lg.spacing = 2
    lg.padding = 6
    lg.margin = 8
    _theme_legend(lg, th)


def _theme_legend(lg, th: dict):
    lg.background_fill_color = th["legend_bg"]
    lg.border_line_color = th["tick"]
    lg.label_text_color = th["text"]


def _html(s: str, **kw) -> pn.pane.HTML:
    kw.setdefault("margin", (0, 0))
    return pn.pane.HTML(s, **kw)


def _label(text: str, **kw) -> pn.pane.HTML:
    kw.setdefault("margin", (2, 0, 0, 0))
    return pn.pane.HTML(f'<div class="sf-label">{text}</div>', **kw)


def _panel(*objs, title: str | None = None, cls="sf-panel", **kw) -> pn.Column:
    kw.setdefault("sizing_mode", "stretch_width")
    kw.setdefault("margin", (0, 0, 10, 0))
    items = ([_label(title)] if title else []) + [o for o in objs if o is not None]
    return pn.Column(*items, css_classes=[cls], **kw)


def _paper(obj, title: str | None = None) -> pn.Column:
    """White 'paper' card for matplotlib output (the report figures keep their print styling)."""
    items = ([_label(title)] if title else []) + [pn.Column(obj, css_classes=["sf-paper"], sizing_mode="stretch_width")]
    return pn.Column(*items, sizing_mode="stretch_width", margin=(0, 0, 14, 0))


def chip(text: str, tone: str = "") -> str:
    return f'<span class="sf-chip {tone}">{text}</span>'


def _df_html(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return '<div class="sf-note">—</div>'
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in r) + "</tr>" for r in df.itertuples(index=False))
    return f'<table class="sf-table"><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>'


# ----------------------------------------------------------------------------------------------
# Component card (inspector)
# ----------------------------------------------------------------------------------------------

class ComponentCard:
    """Widgets for one slab component, laid out as a compact inspector card."""

    def __init__(self, app, cfg: ComponentConfig):
        self.app = app
        W = dict(sizing_mode="stretch_width")
        self.name = pn.widgets.TextInput(name="name", value=cfg.name, **W)
        self.molecule = pn.widgets.Select(name="molecule", options=list(MOLECULES), value=cfg.molecule, **W)
        # line-list release: "auto" = project default for the molecule (config linedata.releases, else hitran,
        # falling back to whatever is cached); explicit choices are the releases cached for this molecule
        rel_opts = self._release_options(cfg.molecule)
        if cfg.linelist_release and cfg.linelist_release not in rel_opts:
            rel_opts.append(cfg.linelist_release)        # keep a configured release even if it is not cached yet
        self.release = pn.widgets.Select(name="line list", options=rel_opts, value=cfg.linelist_release or "auto", **W)
        self._ll_txt = ""
        self.enabled = pn.widgets.Checkbox(name="on", value=cfg.enabled, width=44, margin=(24, 0, 0, 6))
        self.remove = pn.widgets.Button(name="✕", button_type="light", width=34, margin=(22, 0, 0, 4))
        self.logN = pn.widgets.EditableFloatSlider(name="log N [cm⁻²]", start=12.0, end=22.0, step=0.02, value=cfg.logN, format="0.00", **W)
        self.T = pn.widgets.EditableFloatSlider(name="T [K]", start=50.0, end=2000.0, step=5.0, value=cfg.T, format="0", **W)
        self.logR = pn.widgets.EditableFloatSlider(name="log R [au]", start=-3.0, end=2.0, step=0.02, value=cfg.logR, format="0.00", **W)
        self.rv = pn.widgets.EditableFloatSlider(name="v [km/s]", start=-50.0, end=50.0, step=0.5, value=cfg.rv, format="0.0", **W)
        self.fwhm = pn.widgets.EditableFloatSlider(name="Δv [km/s]", start=1.0, end=50.0, step=0.1, value=cfg.fwhm, format="0.0", **W)
        self.group = pn.widgets.TextInput(name="opacity group", value=cfg.group or "", placeholder="shared τ", **W)
        self.tie_to = pn.widgets.TextInput(name="tie to", value=cfg.tie_to or "", placeholder="parent", **W)
        self.ratio = pn.widgets.FloatInput(name="ratio (parent/this, for ties)", value=cfg.ratio or get_molecule(cfg.molecule).default_ratio or 70.0, **W)
        self.kind = pn.widgets.Select(name="kind", options=["slab", "annuli", "absorption"], value=cfg.kind, **W)
        is_abs = cfg.kind == "absorption"
        self.fc = pn.widgets.EditableFloatSlider(name="f_c (covering fraction)", start=0.0, end=1.0, step=0.01, value=cfg.fc, format="0.00", visible=is_abs, **W)
        self.covers = pn.widgets.Select(name="absorbs", options={"continuum only": "continuum", "continuum + emission": "all"}, value=cfg.covers, visible=is_abs, **W)
        self.fwhm_thermal = pn.widgets.Checkbox(name="add thermal width at T to Δv", value=cfg.fwhm_thermal, **W)
        # non-LTE vibrational excitation (T_vib < T_rot weakens the 5-8 um nu2 band and the hot bands)
        self.use_tvib = pn.widgets.Checkbox(name="vibrational temperature T_vib ≠ T (non-LTE ν₂ band)", value=cfg.Tvib is not None, **W)
        self.Tvib = pn.widgets.EditableFloatSlider(name="T_vib [K]", start=50.0, end=2000.0, step=5.0,
                                                   value=float(cfg.Tvib) if cfg.Tvib is not None else min(cfg.T, 600.0),
                                                   format="0", visible=cfg.Tvib is not None, **W)
        self.windows = pn.widgets.TextInput(name="emits only in [µm]", value=", ".join(f"{a:g}-{b:g}" for a, b in (cfg.windows or [])),
                                            placeholder="everywhere (e.g. 4.9-9 for a ro-vibrational component)", **W)
        self.logR.visible = not is_abs
        self.q = pn.widgets.EditableFloatSlider(name="q (T∝r⁻ᑫ)", start=0.0, end=1.5, step=0.02, value=cfg.q, visible=cfg.kind == "annuli", **W)
        self.p = pn.widgets.EditableFloatSlider(name="p (N∝r⁻ᵖ)", start=-1.0, end=3.0, step=0.05, value=cfg.p, visible=cfg.kind == "annuli", **W)
        self.logRin = pn.widgets.EditableFloatSlider(name="log R_in [au]", start=-3.0, end=1.0, step=0.02, value=cfg.logRin, visible=cfg.kind == "annuli", **W)
        self.fixed = pn.widgets.MultiChoice(name="fixed in fit", options=["logN", "T", "logR", "rv", "fwhm", "ratio", "fc", "Tvib"], value=list(cfg.fixed), **W)
        self.head = _html("", sizing_mode="stretch_width")
        self._tau_txt = None
        # live parameters: coalesced through the app scheduler (one evaluation per tick)
        for w in (self.logN, self.T, self.logR, self.rv, self.fwhm, self.q, self.p, self.logRin, self.ratio, self.fc, self.Tvib):
            w.param.watch(lambda e: app._schedule("param"), "value")
        for w in (self.molecule, self.release, self.enabled, self.group, self.tie_to, self.kind, self.name,
                  self.covers, self.fwhm_thermal, self.use_tvib, self.windows):
            w.param.watch(lambda e: app._schedule("structure"), "value")
        self.use_tvib.param.watch(lambda e: setattr(self.Tvib, "visible", bool(e.new)), "value")
        self.molecule.param.watch(self._molecule_changed, "value")
        self.kind.param.watch(self._kind_changed, "value")
        self.remove.on_click(lambda e: app.remove_component(self))
        self.advanced = pn.Card(
            self.rv, self.fwhm,
            pn.Row(self.group, self.tie_to, sizing_mode="stretch_width"),
            pn.Row(self.kind, self.ratio, sizing_mode="stretch_width"),
            self.q, self.p, self.logRin, self.covers, self.fwhm_thermal, self.use_tvib, self.Tvib, self.windows, self.fixed,
            header=_html('<span class="sf-subtitle">velocity · geometry · absorption · T_vib · ties · fixed</span>'),
            collapsed=True, css_classes=["sf-sub"], sizing_mode="stretch_width", margin=(4, 0, 2, 0))
        self.panel = pn.Card(
            pn.Row(self.name, self.enabled, self.remove, sizing_mode="stretch_width"),
            pn.Row(self.molecule, self.release, sizing_mode="stretch_width"),
            self.logN, self.T, self.logR, self.fc, self.advanced,
            header=self.head, collapsible=True, css_classes=["sf-comp"], sizing_mode="stretch_width",
            margin=(0, 0, 10, 0), styles={"border-left": f"3px solid {get_molecule(cfg.molecule).colour}"})
        self.refresh_head()

    @property
    def colour(self) -> str:
        return get_molecule(self.molecule.value).colour

    @staticmethod
    def _release_options(molecule: str) -> list[str]:
        return ["auto"] + [r for r in cached_releases().get(molecule, []) if r != "auto"]

    def _molecule_changed(self, e):
        opts = self._release_options(e.new)
        self.release.options = opts
        if self.release.value not in opts:
            self.release.value = "auto"

    def refresh_head(self, tau: float | None = None):
        mol = get_molecule(self.molecule.value)
        off = "" if self.enabled.value else " off"
        if tau is None or not np.isfinite(tau):
            tau_html = ""
        else:
            cls = "thick" if tau >= 1 else "thin"
            tau_html = f'<span class="tau {cls}">τ<sub>max</sub> {tau:.2f}{" · thin" if tau < 1 else ""}</span>'
        tie = f' <span class="mol">→ {self.tie_to.value.strip()}</span>' if self.tie_to.value.strip() else ""
        grp = f' <span class="mol">[{self.group.value.strip()}]</span>' if self.group.value.strip() else ""
        ll = f' <span class="mol">· {self._ll_txt}</span>' if self._ll_txt else ""
        self.head.object = (f'<div class="sf-comphead"><span class="sf-dot" style="background:{mol.colour}"></span>'
                            f'<span class="name{off}">{self.name.value}</span><span class="mol">{mol.label}</span>{ll}{grp}{tie}{tau_html}</div>')
        self.panel.styles = {"border-left": f"3px solid {mol.colour}"}

    def _kind_changed(self, e):
        for w in (self.q, self.p, self.logRin):
            w.visible = self.kind.value == "annuli"
        is_abs = self.kind.value == "absorption"
        self.fc.visible = is_abs
        self.covers.visible = is_abs
        self.logR.visible = not is_abs

    def to_config(self) -> ComponentConfig:
        return ComponentConfig(name=self.name.value.strip() or "comp", molecule=self.molecule.value, logN=self.logN.value,
                               T=self.T.value, logR=self.logR.value, rv=self.rv.value, fwhm=self.fwhm.value,
                               kind=self.kind.value, group=self.group.value.strip() or None,
                               tie_to=self.tie_to.value.strip() or None, ratio=self.ratio.value, q=self.q.value,
                               p=self.p.value, logRin=self.logRin.value, enabled=self.enabled.value, fixed=list(self.fixed.value),
                               fc=self.fc.value, covers=self.covers.value, fwhm_thermal=self.fwhm_thermal.value,
                               Tvib=float(self.Tvib.value) if self.use_tvib.value else None,
                               windows=self._parse_windows(self.windows.value),
                               linelist_release=None if self.release.value == "auto" else self.release.value)

    @staticmethod
    def _parse_windows(text: str):
        """'4.9-9, 12-27' -> [[4.9, 9.0], [12.0, 27.0]]; empty or unparsable -> None (emits everywhere)."""
        out = []
        for part in re.split(r"[,;]+", text or ""):
            m = re.match(r"^\s*([0-9.]+)\s*[-–]\s*([0-9.]+)\s*$", part)
            if m and float(m.group(1)) < float(m.group(2)):
                out.append([float(m.group(1)), float(m.group(2))])
        return out or None

    def set_values(self, d: dict):
        for k in ("logN", "T", "logR", "rv", "fwhm", "q", "p", "logRin", "fc", "Tvib"):
            if k in d and d[k] is not None:
                getattr(self, k).value = float(np.clip(d[k], getattr(self, k).start, getattr(self, k).end))
        if "ratio" in d:
            self.ratio.value = float(d["ratio"])


# ----------------------------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------------------------

class JalebiApp:
    TAB_NAMES = ["Data", "Continuum", "Model", "Fit", "Results", "Batch"]      # the tabs of the LTE slab-fit module

    def __init__(self, data_root: str | None = None, config_path: str | None = None, start_tab: str = "data",
                 cube_path: str | None = None, start_module: str | None = None, rotdiag_config: str | None = None):
        self.cfg = ProjectConfig.load(config_path) if config_path else ProjectConfig()
        self._spec: Spectrum | None = None
        self._spec_version = 0
        self._noise_cache: tuple[int, np.ndarray] | None = None
        self.raw_spec: Spectrum | None = None
        self.disp_model = None
        self.disp_sel = None
        self.problem = None
        self.run: RunResult | None = None
        self.cards: list[ComponentCard] = []
        self._lock = threading.Lock()
        self._worker = None
        self._stop = threading.Event()
        self._progress = {"stage": "", "frac": 0.0, "log": [], "lnp": [], "done": False, "error": None}
        self._busy = False
        self._dirty: set[str] = set()
        self._scheduled = False
        self._last_chi2 = ""
        self.plot_theme = "dark"
        self.show_features = True
        self._feature_marks: dict = {}
        self._themed: dict[str, list] = {"band": [], "data": [], "total": [], "accent": [], "accent_fill": [], "zero": [], "legend": []}
        # default data root: the bundled examples (FZ Tau x1d + a synthetic spectrum) so the app has
        # something to load on first start; `jalebi serve --data-root DIR` points it at your data
        from .examples import DATA_DIR, resolve_path
        self.data_root = resolve_path(data_root) if data_root else (str(DATA_DIR) if DATA_DIR.is_dir() else os.getcwd())
        self._build_header()
        self._build_sidebar()
        self._build_data_tab()
        self._build_continuum_tab()
        self._build_model_tab()
        self._build_fit_tab()
        self._build_results_tab()
        self._build_batch_tab()
        self._module_options = {"cube": dict(cube_path=cube_path), "rotdiag": dict(config_path=rotdiag_config)}
        self.workspaces: dict[str, object] = {}
        for c in (self.cfg.components or []):
            self.add_component(c, rebuild=False)
        if not self.cfg.components:
            self.add_component(ComponentConfig(name="H2O_hot", molecule="H2O", logN=18.0, T=700.0, logR=-0.6), rebuild=False)
        self.tabs = pn.Tabs(("Data", self.data_tab), ("Continuum", self.cont_tab), ("Model", self.model_tab),
                            ("Fit", self.fit_tab), ("Results", self.results_tab), ("Batch", self.batch_tab),
                            sizing_mode="stretch_width", dynamic=False)
        low = [t.lower() for t in self.TAB_NAMES]
        from .modules import resolve_module
        start = resolve_module(start_module) or resolve_module(start_tab) or "lte"
        if start == "lte" and start_tab and start_tab.lower() in low:
            self.tabs.active = low.index(start_tab.lower())
        self._build_modules(start)
        self.template = pn.template.FastListTemplate(
            title="JALEBI", header=[self.module_nav, self.header], sidebar=[self.sidebar], main=[self.module_tabs],
            theme="dark", theme_toggle=False, main_layout=None, sidebar_width=290,
            background_color=PAL.bg, accent_base_color=PAL.accent, header_background=PAL.bg, header_color=PAL.text,
            neutral_color="#8e97ad", corner_radius=8, shadow=False, font=FONT, font_url=FONT_URL)
        if self.cfg.target.path:
            try:
                self.load_target(self.cfg.target.path)
            except Exception as e:
                self.status.object = f'<div class="sf-kv">⚠ could not load {self.cfg.target.path}: {e}</div>'

    # ------------------------------------------------------------------ modules
    def _build_modules(self, start: str):
        """The module switcher (header) and one hidden-header tab per module; modules other than the LTE
        fit are built the first time they are opened."""
        from .modules import MODULES
        self.module_keys = list(MODULES)
        self._module_holders = {k: pn.Column(sizing_mode="stretch_width") for k in self.module_keys}
        self._module_holders["lte"].objects = [self.tabs]
        self.workspaces["lte"] = self
        self.module_tabs = pn.Tabs(*[(MODULES[k].label, self._module_holders[k]) for k in self.module_keys],
                                   dynamic=False, sizing_mode="stretch_width",
                                   stylesheets=[".bk-header { display: none !important; }"])
        self.module_nav = pn.widgets.RadioButtonGroup(
            options={f"{MODULES[k].icon}  {MODULES[k].label}": k for k in self.module_keys}, value=start,
            button_type="default", margin=(6, 12, 6, 4), css_classes=["sf-modnav"],
            stylesheets=[".bk-btn { font-size: 13px !important; padding: 5px 16px !important; letter-spacing: .01em; }"])
        self.module_nav.param.watch(lambda e: self.switch_module(e.new), "value")
        self.switch_module(start)

    @property
    def module(self) -> str:
        return self.module_keys[self.module_tabs.active]

    def workspace(self, key: str):
        """The workspace object of a module (built on first use)."""
        from .modules import MODULES
        if key not in self.workspaces:
            cls = MODULES[key].load()
            ws = cls(self, **{k: v for k, v in self._module_options.get(key, {}).items() if v is not None})
            self.workspaces[key] = ws
            self._module_holders[key].objects = [ws.panel]
            if hasattr(ws, "apply_theme"):
                ws.apply_theme(PLOT_THEMES[self.plot_theme])
            for f in (ws.figures() if hasattr(ws, "figures") else []):
                _theme_fig(f, PLOT_THEMES[self.plot_theme])
        return self.workspaces[key]

    @property
    def cube_ws(self):
        return self.workspace("cube")

    @property
    def rotdiag_ws(self):
        return self.workspace("rotdiag")

    def switch_module(self, key: str, tab: str | None = None):
        """Show a module (and, for the LTE fit, one of its tabs)."""
        from .modules import resolve_module
        key = resolve_module(key) or "lte"
        ws = self.workspace(key)
        self.module_tabs.active = self.module_keys.index(key)
        if self.module_nav.value != key:
            self.module_nav.value = key
        if key == "lte" and tab and tab in self.TAB_NAMES:
            self.tabs.active = self.TAB_NAMES.index(tab)
        if ws is not self and hasattr(ws, "on_show"):
            ws.on_show()
        self._refresh_header()
        return ws

    def send_spectrum(self, spec: Spectrum, module: str, label: str | None = None):
        """Hand a spectrum built in one module (e.g. a cube region) to another one."""
        if module == "lte":
            return self.use_spectrum(spec, label=label)
        ws = self.switch_module(module)
        ws.use_spectrum(spec, label=label)

    # ------------------------------------------------------------------ plot theme
    def _reg(self, kind: str, obj, band: str | None = None):
        """Register a glyph renderer / annotation / legend whose colours follow the plot theme."""
        self._themed[kind].append((obj, band) if band is not None else obj)
        return obj

    def _all_figs(self):
        figs = [f for f in (getattr(self, n, None) for n in ("data_fig", "cont_fig", "sub_fig", "model_fig", "resid_fig", "lnp_fig", "cube_fig")) if f is not None]
        for k, ws in getattr(self, "workspaces", {}).items():
            if ws is not self and hasattr(ws, "figures"):
                figs += ws.figures()
        return figs

    @_bokeh_safe
    def set_plot_theme(self, name: str):
        """Restyle every figure live: 'dark' (navy, bright band colours) or 'light' (white, print-like)."""
        th = PLOT_THEMES[name]
        self.plot_theme = name
        for f in self._all_figs():
            _theme_fig(f, th)
        for r, b in self._themed["band"]:
            r.glyph.line_color = th["bands"][b]
        for r in self._themed["data"]:
            r.glyph.line_color = th["data"]
        for r in self._themed["total"]:
            (r.glyph if hasattr(r, "glyph") else r).line_color = th["total"]
        for r in self._themed["accent"]:
            r.glyph.line_color = th["accent"]
        for r in self._themed["accent_fill"]:
            r.glyph.fill_color = th["accent"]
        for r in self._themed["zero"]:
            r.line_color = th["zero"]
        for lg in self._themed["legend"]:
            _theme_legend(lg, th)
        for box in getattr(self, "_mask_boxes", []):
            box.fill_color = th["mask"]; box.fill_alpha = th["mask_alpha"]
        for k, ws in getattr(self, "workspaces", {}).items():
            if ws is not self and hasattr(ws, "apply_theme"):
                ws.apply_theme(th)

    # ------------------------------------------------------------------ molecular feature markers
    def _add_feature_marks(self, fig, labels_at: str = "top"):
        """Shade the band heads / Q-branches of every molecule in FEATURES (not H2O) behind the spectrum,
        with a small vertical label.  Registered so the sidebar toggle and the model can restyle them."""
        fig.extra_y_ranges = {**getattr(fig, "extra_y_ranges", {}), "lab": Range1d(0, 1)}
        marks = []
        y = 0.985 if labels_at == "top" else 0.02
        for mol, feats in FEATURES.items():
            colour = get_molecule(mol).colour
            for lo, hi, text in feats:
                wide = (hi - lo) > 0.5
                box = BoxAnnotation(left=lo, right=hi, fill_color=colour, fill_alpha=0.05 if wide else 0.12,
                                    line_color=colour, line_alpha=0.0 if wide else 0.45, line_width=0.6, level="underlay")
                lab = Label(x=lo if wide else 0.5 * (lo + hi), y=y, y_range_name="lab", text=text, angle=np.pi / 2,
                            text_font_size="9px", text_color=colour, text_alpha=0.9, text_font=FONT,
                            text_align="right" if labels_at == "top" else "left", text_baseline="middle",
                            x_offset=0, y_offset=0, level="overlay")
                fig.add_layout(box); fig.add_layout(lab)
                marks.append((mol, box, lab, wide))
        self._feature_marks[fig] = marks
        return marks

    @_bokeh_safe
    def set_feature_marks(self, visible: bool):
        self.show_features = visible
        for marks in self._feature_marks.values():
            for _, box, lab, _ in marks:
                box.visible = visible; lab.visible = visible

    @_bokeh_safe
    def _emphasise_features(self, molecules: set[str]):
        """In the Model overlay, brighten the markers of molecules that are in the model."""
        for fig, marks in self._feature_marks.items():
            if fig is not getattr(self, "model_fig", None):
                continue
            for mol, box, lab, wide in marks:
                on = mol in molecules
                box.fill_alpha = (0.09 if wide else 0.22) if on else (0.04 if wide else 0.09)
                lab.text_alpha = 1.0 if on else 0.6
                lab.text_font_style = "bold" if on else "normal"

    # ------------------------------------------------------------------ spectrum bookkeeping
    @property
    def spec(self) -> Spectrum | None:
        return self._spec

    @spec.setter
    def spec(self, s: Spectrum | None):
        self._spec = s
        self._spec_version += 1

    def noise(self) -> np.ndarray:
        """Per-pixel noise of the current spectrum, cached until the spectrum object changes."""
        if self._noise_cache is None or self._noise_cache[0] != self._spec_version:
            self._noise_cache = (self._spec_version, estimate_noise(self._spec))
        return self._noise_cache[1]

    # ------------------------------------------------------------------ event coalescing
    def _schedule(self, kind: str):
        """Request a model refresh ('param') or rebuild ('structure').  Requests arriving while one
        is pending are merged, so a fast slider drag never queues up evaluations."""
        if self._busy:
            return
        self._dirty.add(kind)
        if self._scheduled:
            return
        self._scheduled = True
        doc = pn.state.curdoc
        if doc is None:
            self._flush()
            return
        try:
            doc.add_timeout_callback(self._flush, 20)
        except Exception:
            self._flush()

    def _flush(self):
        self._scheduled = False
        dirty, self._dirty = self._dirty, set()
        if "structure" in dirty:
            self.on_structure_change()
        elif "param" in dirty:
            self.on_param_change()

    # ------------------------------------------------------------------ header
    def _build_header(self):
        self.header = _html("", sizing_mode="stretch_width", margin=(0, 10))
        self._refresh_header()

    def _refresh_header(self, extra: str = ""):
        s = self.spec
        if hasattr(self, "module_tabs") and self.module != "lte":       # the chips describe the LTE-fit target
            self.header.object = (f'<div style="display:flex;align-items:center;gap:6px;width:100%;">'
                                  f'<span class="sf-sub" style="margin-right:10px">{_ACRONYM_HTML}</span></div>')
            return
        if s is None:
            body = chip("no spectrum loaded", "dim")
        else:
            key = (self._spec_version, s.name)
            if getattr(self, "_hdr_cache", (None,))[0] != key:     # spec.bands is not free: cache per spectrum
                self._hdr_cache = (key, chip(f"<b>{s.name}</b>", "accent") + chip(f"{len(s.wave)} px · {len(s.bands)} bands") +
                                   chip(f"d {s.distance_pc:g} pc · RV {s.rv_kms:g}"))
            body = self._hdr_cache[1]
        if self._last_chi2:
            body += self._last_chi2
        if self._worker is not None and self._worker.is_alive():
            body += chip("● fitting", "pink")
        self.header.object = (f'<div style="display:flex;align-items:center;gap:6px;width:100%;">'
                              f'<span class="sf-sub" style="margin-right:10px" title="simultaneous LTE slab fitting of molecular emission in JWST/MIRI disk spectra">'
                              f'{_ACRONYM_HTML}</span>{body}{extra}</div>')

    # ------------------------------------------------------------------ sidebar
    def _build_sidebar(self):
        self.status = _html('<div class="sf-kv">No spectrum loaded.<br>Pick a target in <b>Data</b>.</div>', sizing_mode="stretch_width")
        self.cfg_download = pn.widgets.FileDownload(callback=self._config_bytes, filename="jalebi_config.yaml",
                                                    button_type="primary", label="⬇ Export config YAML", sizing_mode="stretch_width")
        self.cfg_upload = pn.widgets.FileInput(accept=".yaml,.yml", sizing_mode="stretch_width")
        self.cfg_upload.param.watch(self._config_uploaded, "value")
        ll = available_linelists()
        chips = "".join(chip(f"{r.molecule} <span class='dim'>{r.release} · {r.MB:.0f} MB</span>") for r in ll.itertuples()) or \
                '<span class="sf-note">none cached</span>'
        self.linelist_info = _html(
            f'<div>{chips}</div><div class="sf-note" style="margin-top:8px">cache: <span class="sf-mono">{data_dir()}</span><br>'
            'fetch more with <span class="sf-mono">jalebi linedata fetch H2O --release hitran --wmin 4.9 --wmax 28</span> '
            '(HITRAN only) or import HITEMP from a <span class="sf-mono">.par</span> file with <span class="sf-mono">jalebi linedata import</span>. '
            'Restart the app to see new lists.</div>',
            sizing_mode="stretch_width")
        self.plot_theme_btn = pn.widgets.RadioButtonGroup(options={"dark plots": "dark", "light plots": "light"}, value="dark",
                                                         button_type="default", sizing_mode="stretch_width")
        self.plot_theme_btn.param.watch(lambda e: self.set_plot_theme(e.new), "value")
        self.features_cb = pn.widgets.Checkbox(name="molecular feature markers", value=True, margin=(8, 10, 0, 10))
        self.features_cb.param.watch(lambda e: self.set_feature_marks(e.new), "value")
        self.sidebar = pn.Column(
            _panel(self.status, title="Session"),
            _panel(self.plot_theme_btn, _html('<div class="sf-note" style="margin-top:6px">Switch the figures to a white background '
                                              'if the spectra are hard to see on your screen.</div>'),
                   self.features_cb,
                   _html('<div class="sf-note" style="margin-top:2px">Shaded bands mark the Q-branches / band heads of each molecule '
                         '(no H₂O — its lines are everywhere), so you can see which components a spectrum needs.</div>'),
                   title="Display"),
            _panel(self.cfg_download, _html('<div class="sf-note" style="margin:8px 0 4px">Load a config YAML</div>'), self.cfg_upload, title="LTE-fit configuration"),
            _panel(self.linelist_info, title="Line lists"),
            sizing_mode="stretch_width")

    def _config_bytes(self):
        return io.BytesIO(self.current_config().to_yaml().encode())

    def _config_uploaded(self, e):
        try:
            import yaml
            d = yaml.safe_load(io.BytesIO(e.new).read().decode())
            self.cfg = ProjectConfig.model_validate(d)
            self.apply_config(self.cfg)
            pn.state.notifications.success("config loaded")
        except Exception as ex:
            pn.state.notifications.error(f"config error: {ex}")

    # ------------------------------------------------------------------ Data tab
    def _build_data_tab(self):
        self.root_input = pn.widgets.TextInput(name="data root (folder of target folders or files)", value=self.data_root, sizing_mode="stretch_width")
        self.scan_btn = pn.widgets.Button(name="Scan", button_type="default", width=90, margin=(22, 5, 5, 5))
        self.target_select = pn.widgets.Select(name="target", options=[], sizing_mode="stretch_width")
        self.distance = pn.widgets.FloatInput(name="distance [pc]", value=self.cfg.target.distance_pc, width=110)
        self.rv = pn.widgets.FloatInput(name="RV [km/s]", value=self.cfg.target.rv_kms, width=100)
        self.spike = pn.widgets.Checkbox(name="spike filter", value=True, margin=(26, 10, 5, 10))
        self.load_btn = pn.widgets.Button(name="Load target", button_type="primary", width=130, margin=(22, 5, 5, 5))
        self.upload = pn.widgets.FileInput(accept=".csv,.txt,.fits", width=260)
        self.scan_btn.on_click(lambda e: self.scan_root())
        self.load_btn.on_click(lambda e: self.load_target(self.target_select.value))
        self.upload.param.watch(self._file_uploaded, "value")
        ex = self.cfg.target.extraction
        self.ex_source = pn.widgets.Select(name="spectrum from", options={"x1d (pipeline extraction)": "x1d", "s3d cubes (aperture at RA, Dec)": "s3d"}, value=ex.source, width=240)
        self.ex_ra = pn.widgets.TextInput(name="RA (deg or hh:mm:ss)", value="" if ex.ra is None else str(ex.ra), sizing_mode="stretch_width")
        self.ex_dec = pn.widgets.TextInput(name="Dec (deg or ±dd:mm:ss)", value="" if ex.dec is None else str(ex.dec), sizing_mode="stretch_width")
        self.ex_header_btn = pn.widgets.Button(name="header target position", sizing_mode="stretch_width", margin=(22, 5, 5, 5))
        self.ex_peak_btn = pn.widgets.Button(name="brightest pixel", sizing_mode="stretch_width", margin=(22, 5, 5, 5))
        self.ex_header_btn.on_click(lambda e: self._fill_radec("header"))
        self.ex_peak_btn.on_click(lambda e: self._fill_radec("peak"))
        self.ex_scale = pn.widgets.FloatInput(name="aperture [× FWHM]", value=ex.aperture_fwhm_scale, step=0.25, sizing_mode="stretch_width")
        self.ex_arcsec = pn.widgets.FloatInput(name="or radius [arcsec] (0=off)", value=ex.aperture_arcsec or 0.0, step=0.1, sizing_mode="stretch_width")
        self.ex_ann = pn.widgets.TextInput(name="annulus r_in, r_out [arcsec]", value=", ".join(str(v) for v in ex.annulus_arcsec) if ex.annulus_arcsec else "", placeholder="blank = none", sizing_mode="stretch_width")
        self.ex_apcorr = pn.widgets.Select(name="aperture correction", options=["mrs", "gaussian", "none", "x1d"], value=ex.apcorr, sizing_mode="stretch_width")
        self.ex_band = pn.widgets.Select(name="preview band", options=BAND_ORDER, value="2A", width=100)
        self.ex_box = pn.Column(
            pn.Row(self.ex_ra, self.ex_dec, self.ex_band, sizing_mode="stretch_width"),
            pn.Row(self.ex_header_btn, self.ex_peak_btn, sizing_mode="stretch_width"),
            pn.Row(self.ex_scale, self.ex_arcsec, sizing_mode="stretch_width"),
            pn.Row(self.ex_ann, self.ex_apcorr, sizing_mode="stretch_width"),
            visible=ex.source == "s3d", sizing_mode="stretch_width")
        self.ex_source.param.watch(lambda e: setattr(self.ex_box, "visible", e.new == "s3d") or self._show_cube(), "value")
        self.ex_band.param.watch(lambda e: self._show_cube(), "value")
        for w in (self.ex_ra, self.ex_dec, self.ex_scale, self.ex_arcsec):
            w.param.watch(lambda e: self._show_cube(), "value")
        self.cube_fig = figure(height=330, width=360, title="cube (median image) — tap to set RA/Dec", tools="tap,reset,save",
                               x_axis_label="x [px]", y_axis_label="y [px]", match_aspect=True, toolbar_location="above")
        self.cube_fig.toolbar.logo = None
        self.cube_fig.outline_line_color = None
        self.cube_fig.title.text_font_size = "12px"
        for ax in (self.cube_fig.xaxis, self.cube_fig.yaxis):
            ax.axis_line_color = None
        _theme_fig(self.cube_fig, PLOT_THEMES["dark"])
        self.cube_src = ColumnDataSource(dict(image=[np.zeros((2, 2))], x=[0], y=[0], dw=[2], dh=[2]))
        self.cube_fig.image("image", "x", "y", "dw", "dh", source=self.cube_src, palette="Viridis256")
        self.cube_mark = ColumnDataSource(dict(x=[], y=[], r=[]))
        self.cube_fig.circle("x", "y", radius="r", source=self.cube_mark, fill_alpha=0.0, line_color=PAL.pink, line_width=2)
        self.cube_fig.scatter("x", "y", source=self.cube_mark, marker="cross", size=12, color=PAL.pink)
        self.cube_fig.on_event("tap", self._tap_cube)
        # (visibility is toggled on a wrapping Column: toggling `visible` on a Bokeh pane trips a Panel bug)
        self.cube_pane = pn.Column(pn.pane.Bokeh(self.cube_fig), visible=ex.source == "s3d", width=380)
        self.lineid_toggle = pn.widgets.Checkbox(name="line IDs (Banzatti+2025 lists)", value=False, margin=(26, 10, 5, 10))
        self.lineid_species = pn.widgets.MultiChoice(name="species", options=["H2O v=0", "H2O v=1-1", "H2O v=1-0", "CO", "OH", "HI", "H2"],
                                                     value=["H2O v=0"], width=420)
        self.lineid_toggle.param.watch(lambda e: self._update_lineids(), "value")
        self.lineid_species.param.watch(lambda e: self._update_lineids(), "value")
        self.data_fig = _fig("Spectrum — coloured by MRS sub-band", height=440)
        self._add_feature_marks(self.data_fig, labels_at="bottom")
        self.data_src = {b: ColumnDataSource(dict(w=[], f=[])) for b in BAND_ORDER}
        for b in BAND_ORDER:
            self._reg("band", self.data_fig.line("w", "f", source=self.data_src[b], color=BAND_COLOURS[b], line_width=1.3, legend_label=b), b)
        _style_legend(self.data_fig.legend, "top_left", "horizontal")
        self._reg("legend", self.data_fig.legend[0])
        self.lineid_src = ColumnDataSource(dict(w=[], y=[], species=[], lam=[], eu=[], a=[]))
        r = self.data_fig.scatter("w", "y", source=self.lineid_src, marker="inverted_triangle", size=7, color=PAL.pink, alpha=0.8)
        self.data_fig.add_tools(HoverTool(renderers=[r], tooltips=[("species", "@species"), ("λ", "@lam{0.0000} µm"), ("E_up", "@eu{0} K"), ("A_ul", "@a{0.000}")]))
        self.data_info = _html("", sizing_mode="stretch_width")
        source_panel = _panel(
            pn.Row(self.root_input, self.scan_btn, sizing_mode="stretch_width"),
            pn.Row(self.target_select, self.distance, self.rv, self.spike, self.load_btn, sizing_mode="stretch_width"),
            pn.Row(_html('<div class="sf-note" style="margin-top:8px">…or upload a CSV (wave, flux, err) / x1d FITS</div>'), self.upload),
            title="Target")
        extraction_panel = _panel(self.ex_source, self.ex_box, title="Extraction")
        self.data_tab = pn.Column(
            pn.Row(pn.Column(source_panel, extraction_panel, sizing_mode="stretch_width"), self.cube_pane, sizing_mode="stretch_width"),
            _panel(pn.Row(self.lineid_toggle, self.lineid_species), pn.pane.Bokeh(self.data_fig, sizing_mode="stretch_width"),
                   self.data_info, title="Spectrum"),
            sizing_mode="stretch_width")
        self.scan_root()

    @_bokeh_safe
    def scan_root(self):
        root = os.path.expanduser(self.root_input.value)
        opts = {}
        if os.path.isdir(root):
            if glob.glob(os.path.join(root, "*x1d.fits")):
                opts[os.path.basename(root)] = root
            for d in sorted(glob.glob(os.path.join(root, "*"))) + sorted(glob.glob(os.path.join(root, "*", "*"))):
                if os.path.isdir(d) and glob.glob(os.path.join(d, "*x1d.fits")):
                    opts[os.path.relpath(d, root)] = d
                elif os.path.isfile(d) and d.lower().endswith((".csv", ".txt")):
                    opts[os.path.relpath(d, root)] = d
        self.target_select.options = opts
        if opts:
            self.target_select.value = list(opts.values())[0]
        self.data_info.object = f'<div class="sf-note">{len(opts)} targets found under <span class="sf-mono">{root}</span></div>'

    def _file_uploaded(self, e):
        tmp = os.path.join(pn.state.cache.setdefault("tmpdir", os.path.join(os.path.expanduser("~"), ".jalebi", "uploads")), self.upload.filename)
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        with open(tmp, "wb") as fh:
            fh.write(e.new)
        self.load_target(tmp)

    def extraction_config(self) -> ExtractionConfig:
        ann = None
        try:
            vals = [float(v) for v in self.ex_ann.value.replace(";", ",").split(",") if v.strip()]
            ann = vals[:2] if len(vals) == 2 else None
        except ValueError:
            ann = None
        return ExtractionConfig(source=self.ex_source.value, ra=self.ex_ra.value.strip() or None, dec=self.ex_dec.value.strip() or None,
                                aperture_fwhm_scale=self.ex_scale.value, aperture_arcsec=self.ex_arcsec.value or None,
                                annulus_arcsec=ann, apcorr=self.ex_apcorr.value)

    def _cube_file(self):
        from .examples import resolve_path
        path = resolve_path(self.target_select.value or self.cfg.target.path)
        if not path or not os.path.isdir(path):
            return None
        band = self.ex_band.value
        tag = f"ch{band[0]}-{ {'A': 'short', 'B': 'medium', 'C': 'long'}[band[1]] }"
        files = [f for f in glob.glob(os.path.join(path, "*s3d.fits")) if tag in f]
        return files[0] if files else None

    def _fill_radec(self, which: str):
        """Fill the RA/Dec boxes from the cube header target position or the brightest pixel."""
        f = self._cube_file()
        if f is None:
            pn.state.notifications.warning("no s3d cube found for this target/band"); return
        from jalebi.data import extract_s3d
        r = extract_s3d(f, center="wcs" if which == "header" else "peak", apcorr="none")
        self.ex_ra.value = f"{r['center_radec'][0]:.6f}"; self.ex_dec.value = f"{r['center_radec'][1]:.6f}"

    @_bokeh_safe
    def _show_cube(self):
        """Median-collapsed image of the preview cube with the aperture at the given RA/Dec."""
        self.cube_pane.visible = self.ex_source.value == "s3d"
        f = self._cube_file()
        if self.ex_source.value != "s3d" or f is None:
            return
        try:
            from astropy.io import fits
            from jalebi.data import extract_s3d, parse_radec
            with fits.open(f) as h:
                sci = h["SCI"].data; hd = h["SCI"].header
            img = np.nanmedian(sci, axis=0)
            img = np.where(np.isfinite(img), img, np.nanmin(img))
            lo, hi = np.nanpercentile(img, 5), np.nanpercentile(img, 99.7)
            img = np.log10(np.clip(img - lo, 1e-3 * max(hi - lo, 1e-9), None))   # log stretch shows faint companions
            self.cube_src.data = dict(image=[img], x=[-0.5], y=[-0.5], dw=[img.shape[1]], dh=[img.shape[0]])
            ex = self.extraction_config()
            if ex.ra is None or ex.dec is None:
                self.cube_mark.data = dict(x=[], y=[], r=[])
                self.cube_fig.title.text = f"{os.path.basename(f)}: enter RA/Dec (or tap the source)"; return
            radec = parse_radec(ex.ra, ex.dec)
            r = extract_s3d(f, center_radec=radec, aperture_fwhm_scale=ex.aperture_fwhm_scale, aperture_arcsec=ex.aperture_arcsec, apcorr="none")
            xc, yc = r["center_pix"]
            lam = hd["CRVAL3"] + hd["CDELT3"] * (sci.shape[0] / 2)
            rad = (ex.aperture_arcsec or ex.aperture_fwhm_scale * float(mrs_psf_fwhm(lam))) / r["pixscale"]
            self.cube_mark.data = dict(x=[xc], y=[yc], r=[rad])
            self.cube_fig.title.text = f"{os.path.basename(f)}: RA {radec[0]:.5f} Dec {radec[1]:.5f} -> px ({xc:.1f}, {yc:.1f}), r = {rad:.1f} px at {lam:.1f} µm"
        except Exception as ex_:
            self.cube_fig.title.text = f"cube preview failed: {ex_}"

    def _tap_cube(self, event):
        """A tap on the image converts the pixel to RA/Dec through the cube's WCS and fills the boxes."""
        f = self._cube_file()
        if f is None:
            return
        from jalebi.data import extract_s3d
        r = extract_s3d(f, center=(float(event.x), float(event.y)), apcorr="none")
        self.ex_ra.value = f"{r['center_radec'][0]:.6f}"; self.ex_dec.value = f"{r['center_radec'][1]:.6f}"

    @_bokeh_safe
    def load_target(self, path: str):
        if not path:
            return
        t0 = time.time()
        ex = self.extraction_config()
        if ex.source == "s3d" and (ex.ra is None or ex.dec is None):
            pn.state.notifications.warning("s3d extraction needs RA and Dec: type them, use 'header target position', or tap the source in the cube image")
            self._show_cube(); return
        ext = {"ra": ex.ra, "dec": ex.dec, "aperture_fwhm_scale": ex.aperture_fwhm_scale, "aperture_arcsec": ex.aperture_arcsec,
               "annulus_arcsec": tuple(ex.annulus_arcsec) if ex.annulus_arcsec else None, "apcorr": ex.apcorr}
        try:
            spec = load_spectrum(path, source=ex.source, extraction=ext, distance_pc=self.distance.value)
        except Exception as e:
            pn.state.notifications.error(f"load failed: {e}"); return
        self.cfg.target.extraction = ex
        self._show_cube()
        self._set_spectrum(spec, path, t0)

    @_bokeh_safe
    def use_spectrum(self, spec: Spectrum, label: str | None = None, switch_to: str | None = "Continuum"):
        """Make a Spectrum built elsewhere the current target (e.g. a region spectrum from the Cube
        workspace).  It is also written to ~/.jalebi/uploads/<name>.csv, so an exported config points at
        a real file and the fit can be repeated from the terminal."""
        import re
        folder = os.path.join(os.path.expanduser("~"), ".jalebi", "uploads")
        os.makedirs(folder, exist_ok=True)
        fname = re.sub(r"[^A-Za-z0-9_.+-]+", "_", label or spec.name).strip("_") + ".csv"
        path = os.path.join(folder, fname)
        try:
            spec.save(path)
        except Exception:
            path = label or spec.name
        self.distance.value = spec.distance_pc
        if switch_to and hasattr(self, "module_tabs"):
            self.switch_module("lte", switch_to)                    # switch first: the continuum + model take a moment
        elif switch_to and switch_to in self.TAB_NAMES and hasattr(self, "tabs"):
            self.tabs.active = self.TAB_NAMES.index(switch_to)
        self._set_spectrum(spec, path, time.time())

    def _set_spectrum(self, spec: Spectrum, path: str, t0: float):
        spec = spec.to_rest_frame(self.rv.value)
        if self.spike.value:
            spec = spike_filter(spec)
        self.raw_spec = spec.copy()
        self.spec = spec
        self.cfg.target.path = path; self.cfg.target.name = spec.name
        self.cfg.target.distance_pc = self.distance.value; self.cfg.target.rv_kms = self.rv.value
        for b in BAND_ORDER:
            i = spec.band_slice(b)
            self.data_src[b].data = dict(w=spec.wave[i], f=spec.flux[i])
        self.data_fig.title.text = f"{spec.name} — {len(spec.wave)} pixels, {len(spec.bands)} sub-bands"
        exinfo = spec.meta.get("extraction")
        if isinstance(exinfo, dict) and exinfo.get("source") == "s3d-region":
            exline = f"cube region <span class=\"sf-mono\">{exinfo.get('ds9', '')}</span>"
        elif isinstance(exinfo, dict) and exinfo.get("center_radec"):
            exline = (f"s3d aperture {exinfo['aperture_fwhm_scale']}×FWHM, apcorr {exinfo['apcorr']}<br>"
                      f"RA/Dec {exinfo['center_radec'][0]:.5f}, {exinfo['center_radec'][1]:.5f}")
        else:
            exline = "x1d pipeline extraction"
        self.status.object = (f'<div class="sf-title" style="font-size:17px">{spec.name}</div>'
                              f'<div class="sf-kv"><b>{len(spec.wave)}</b> pixels · <b>{len(spec.bands)}</b> sub-bands<br>'
                              f'd = <b>{spec.distance_pc:g}</b> pc · RV <b>{spec.rv_kms:g}</b> km/s<br>'
                              f'masked (spikes, band edges): <b>{(~spec.mask).sum()}</b><br>{exline}</div>')
        hdr = ", ".join(f"{k}={v}" for k, v in spec.meta.items() if v and k != "extraction")
        self.data_info.object = (f'<div class="sf-note">Loaded <span class="sf-mono">{path}</span> in {time.time() - t0:.1f} s · bands {", ".join(spec.bands)}'
                                 + (f' · header: <span class="sf-mono">{hdr}</span>' if hdr else "") + "</div>")
        self._update_lineids()
        self.estimate_continuum()
        self.on_structure_change()
        self._refresh_header()
        if hasattr(self, "out_resolved"):
            self._show_outdir()

    @_bokeh_safe
    def _update_lineids(self):
        if not self.lineid_toggle.value or self.spec is None:
            self.lineid_src.data = dict(w=[], y=[], species=[], lam=[], eu=[], a=[]); return
        files = {"H2O v=0": "MIRI_H2O_v0-0.csv", "H2O v=1-1": "MIRI_H2O_v1-1.csv", "H2O v=1-0": "MIRI_H2O_v1-0.csv"}
        rows = []
        for sp in self.lineid_species.value:
            if sp in files:
                t = pd.read_csv(os.path.join(DATA_FILES, files[sp]))
                for _, r in t.iterrows():
                    rows.append((r["lam"], sp, r["e_up"], r["a_stein"]))
            else:
                t = pd.read_csv(os.path.join(DATA_FILES, "MIRI_general_Banzatti+2025.csv"))
                for _, r in t[t["species"] == sp].iterrows():
                    rows.append((r["lam"], sp, r["e_up"], r["a_stein"]))
                if sp == "HI":
                    t = pd.read_csv(os.path.join(DATA_FILES, "Atomic_lines.csv"), encoding="utf-8-sig")
                    for _, r in t.iterrows():
                        rows.append((r["wave"], f"{r['species']} {r['line']}", np.nan, np.nan))
        if not rows:
            self.lineid_src.data = dict(w=[], y=[], species=[], lam=[], eu=[], a=[]); return
        w = np.array([r[0] for r in rows])
        ymax = np.nanpercentile(self.spec.flux, 99.5)
        y = np.interp(w, *self.spec.stitched()[:2]) * 0 + ymax
        self.lineid_src.data = dict(w=w, y=y, species=[r[1] for r in rows], lam=w, eu=[r[2] for r in rows], a=[r[3] for r in rows])

    # ------------------------------------------------------------------ Continuum tab
    def _build_continuum_tab(self):
        c = self.cfg.continuum
        W = dict(sizing_mode="stretch_width")
        self.cont_method = pn.widgets.Select(name="method", options=METHODS, value=c.method, **W)
        self.cont_protect = pn.widgets.Checkbox(name="protect Q-branches / ranges", value=c.protect)
        self.cont_protected = pn.widgets.TextAreaInput(name="protected ranges (name: lo, hi)", value=_fmt_named(c.protected), height=120, **W)
        self.cont_smooth = pn.widgets.IntInput(name="extra smoothing [px]", value=c.smooth, **W)
        # method parameters
        self.w_quantile = pn.widgets.FloatInput(name="quantile", value=c.quantile, step=0.02, **W)
        self.w_knots = pn.widgets.IntInput(name="knot spacing [px]", value=c.knot_spacing, **W)
        self.w_medwin = pn.widgets.IntInput(name="median window [px]", value=c.median_window, **W)
        self.w_medpct = pn.widgets.FloatInput(name="percentile", value=c.median_percentile, **W)
        self.w_sgwin = pn.widgets.IntInput(name="SG window [px]", value=c.sg_window, **W)
        self.w_sgord = pn.widgets.IntInput(name="SG order", value=c.sg_order, **W)
        self.w_niter = pn.widgets.IntInput(name="iterations", value=c.n_iter, **W)
        self.w_lam = pn.widgets.FloatInput(name="λ (asls)", value=c.lam, **W)
        self.w_p = pn.widgets.FloatInput(name="p (asls)", value=c.p, step=0.005, **W)
        self.w_aspls_lam = pn.widgets.FloatInput(name="λ (aspls)", value=c.aspls_lam, **W)
        self.w_aspls_alpha = pn.widgets.FloatInput(name="asymmetry (aspls)", value=c.aspls_alpha, step=0.05, start=0.0, end=1.0, **W)
        self.w_segment = pn.widgets.IntInput(name="hull segment [px]", value=c.segment, **W)
        self.w_overlap = pn.widgets.IntInput(name="hull overlap [px]", value=c.overlap, **W)
        self.w_pct = pn.widgets.FloatInput(name="min percentile", value=c.percentile, **W)
        self.w_minwin = pn.widgets.IntInput(name="min window [px]", value=c.min_window, **W)
        self.w_anchors = pn.widgets.TextAreaInput(name="spline anchors [µm] — tap plot to add, double-tap to remove", value=", ".join(f"{a:.4f}" for a in c.anchors), height=80, **W)
        self.w_anchorw = pn.widgets.IntInput(name="anchor width [px]", value=c.anchor_width, **W)
        self.method_boxes = {
            "irsqr": pn.Row(self.w_quantile, self.w_knots, **W),
            "median_sg": pn.Column(pn.Row(self.w_medwin, self.w_medpct, **W), pn.Row(self.w_sgwin, self.w_sgord, self.w_niter, **W), **W),
            "asls": pn.Row(self.w_lam, self.w_p, **W),
            "aspls": pn.Row(self.w_aspls_lam, self.w_aspls_alpha, **W),
            "convex_hull": pn.Row(self.w_segment, self.w_overlap, **W),
            "rolling_min": pn.Row(self.w_pct, self.w_minwin, **W),
            "spline": pn.Column(self.w_anchors, self.w_anchorw, **W),
            "banzatti": pn.Row(_html('<div class="sf-note">spline through the Banzatti+2025 line-free windows</div>')),
            "none": pn.Row(_html('<div class="sf-note">no continuum subtraction</div>')),
        }
        self.method_box = pn.Column(*self.method_boxes.values(), **W)
        self.cont_method.param.watch(self._method_changed, "value")
        self._method_changed(None)
        self.cont_btn = pn.widgets.Button(name="Estimate continuum", button_type="primary", **W)
        self.cont_btn.on_click(lambda e: self.estimate_continuum())
        self.cont_auto = pn.widgets.Checkbox(name="auto-update", value=True, margin=(12, 10, 5, 10))
        for w in (self.w_quantile, self.w_knots, self.w_medwin, self.w_medpct, self.w_sgwin, self.w_sgord, self.w_niter,
                  self.w_lam, self.w_p, self.w_aspls_lam, self.w_aspls_alpha, self.w_segment, self.w_overlap, self.w_pct, self.w_minwin, self.w_anchorw, self.cont_smooth,
                  self.cont_protect, self.cont_method):
            w.param.watch(lambda e: self.cont_auto.value and self.estimate_continuum(), "value")
        self.w_anchors.param.watch(lambda e: self.cont_auto.value and self.cont_method.value == "spline" and self.estimate_continuum(), "value")
        self.cont_protected.param.watch(lambda e: self.cont_auto.value and self.estimate_continuum(), "value")
        # masks
        m = self.cfg.masks
        self.mask_default = pn.widgets.Checkbox(name="mask H I / H₂ / fine-structure lines", value=m.default_lines)
        self.mask_oh = pn.widgets.Checkbox(name="mask OH prompt lines (9–13 µm)", value=m.oh_prompt)
        self.mask_oh_all = pn.widgets.Checkbox(name="mask whole 9–13 µm", value=m.oh_whole_range)
        self.mask_extra = pn.widgets.TextAreaInput(name="extra masks (name: lo, hi)", value=_fmt_named(m.extra), height=90, **W)
        for w in (self.mask_default, self.mask_oh, self.mask_oh_all, self.mask_extra):
            w.param.watch(lambda e: self.estimate_continuum(), "value")
        # figures
        self.cont_fig = _fig("Continuum", height=360)
        self._add_feature_marks(self.cont_fig, labels_at="bottom")
        self.cont_src = {b: ColumnDataSource(dict(w=[], f=[], c=[])) for b in BAND_ORDER}
        for b in BAND_ORDER:
            self._reg("band", self.cont_fig.line("w", "f", source=self.cont_src[b], color=BAND_COLOURS[b], line_width=1.1), b)
            self._reg("accent", self.cont_fig.line("w", "c", source=self.cont_src[b], color=PAL.accent, line_width=1.8))
        self.anchor_src = ColumnDataSource(dict(w=[], f=[]))
        self.cont_fig.scatter("w", "f", source=self.anchor_src, size=9, color=PAL.pink, marker="circle")
        self.sub_fig = _fig("Continuum-subtracted (shaded = masked)", height=270, x_range=self.cont_fig.x_range)
        self.sub_src = {b: ColumnDataSource(dict(w=[], f=[])) for b in BAND_ORDER}
        for b in BAND_ORDER:
            self._reg("band", self.sub_fig.line("w", "f", source=self.sub_src[b], color=BAND_COLOURS[b], line_width=1.1), b)
        self.sub_fig.add_layout(self._reg("zero", Span(location=0, dimension="width", line_color=PAL.muted, line_width=0.6)))
        self._mask_boxes = []
        self.cont_fig.on_event("tap", self._tap_anchor)
        self.cont_fig.on_event("doubletap", self._untap_anchor)
        self.noise_table = _html("", sizing_mode="stretch_width")
        inspector = pn.Column(
            _label("Continuum"),
            self.cont_method, self.method_box,
            pn.Row(self.cont_smooth, pn.Column(self.cont_protect, self.cont_auto, margin=(14, 0, 0, 0)), **W),
            self.cont_protected, self.cont_btn,
            _label("Masks", margin=(14, 0, 0, 0)),
            self.mask_default, self.mask_oh, self.mask_oh_all, self.mask_extra,
            width=380, css_classes=["sf-inspector"],
            styles={"position": "sticky", "top": "0px", "max-height": "calc(100vh - 140px)", "overflow-y": "auto"})
        left = pn.Column(
            _panel(pn.pane.Bokeh(self.cont_fig, **W), pn.pane.Bokeh(self.sub_fig, **W)),
            _panel(self.noise_table, title="Noise per sub-band — MAD of the high-pass residual (line-free pixels) vs pipeline error"),
            **W)
        self.cont_tab = pn.Row(left, inspector, **W)

    def _method_changed(self, e):
        for k, box in self.method_boxes.items():
            box.visible = (k == self.cont_method.value)

    def _tap_anchor(self, event):
        if self.cont_method.value != "spline" or self.spec is None:
            return
        a = [float(x) for x in self.w_anchors.value.replace("\n", ",").split(",") if x.strip()]
        a.append(float(event.x))
        self.w_anchors.value = ", ".join(f"{x:.4f}" for x in sorted(a))

    def _untap_anchor(self, event):
        if self.cont_method.value != "spline":
            return
        a = [float(x) for x in self.w_anchors.value.replace("\n", ",").split(",") if x.strip()]
        if a:
            a.pop(int(np.argmin(np.abs(np.array(a) - event.x))))
            self.w_anchors.value = ", ".join(f"{x:.4f}" for x in sorted(a))

    def continuum_config(self) -> ContinuumConfig:
        return ContinuumConfig(method=self.cont_method.value, protect=self.cont_protect.value,
                               protected=_parse_named_ranges(self.cont_protected.value), smooth=self.cont_smooth.value,
                               quantile=self.w_quantile.value, knot_spacing=self.w_knots.value, median_window=self.w_medwin.value,
                               median_percentile=self.w_medpct.value, sg_window=self.w_sgwin.value, sg_order=self.w_sgord.value,
                               n_iter=self.w_niter.value, lam=self.w_lam.value, p=self.w_p.value, aspls_lam=self.w_aspls_lam.value,
                               aspls_alpha=self.w_aspls_alpha.value, segment=self.w_segment.value,
                               overlap=self.w_overlap.value, percentile=self.w_pct.value, min_window=self.w_minwin.value,
                               anchors=[float(x) for x in self.w_anchors.value.replace("\n", ",").split(",") if x.strip()],
                               anchor_width=self.w_anchorw.value, refine_iterations=self.fit_refine.value if hasattr(self, "fit_refine") else 0)

    @_bokeh_safe
    def estimate_continuum(self, gas_model=None):
        if self.raw_spec is None:
            return
        cfg = self.current_config(light=True)
        try:
            spec = prepare(cfg, self.raw_spec.copy(), gas_model=gas_model)
        except Exception as ex:
            pn.state.notifications.error(f"continuum failed: {ex}")
            return
        self.spec = spec
        for b in BAND_ORDER:
            i = spec.band_slice(b)
            self.cont_src[b].data = dict(w=spec.wave[i], f=spec.flux[i], c=spec.continuum[i])
            self.sub_src[b].data = dict(w=spec.wave[i], f=np.where(spec.mask[i], spec.line_flux[i], np.nan))
        if cfg.continuum.method == "spline":
            a = np.array(cfg.continuum.anchors)
            self.anchor_src.data = dict(w=a, f=np.interp(a, *spec.stitched()[:2]) if len(a) else [])
        else:
            self.anchor_src.data = dict(w=[], f=[])
        # masked regions as shaded boxes
        for box in self._mask_boxes:
            self.sub_fig.renderers.remove(box) if box in self.sub_fig.renderers else None
        self._mask_boxes = []
        w, m = spec.wave, spec.mask
        o = np.argsort(w); w, m = w[o], m[o]
        edges = np.flatnonzero(np.diff(m.astype(int)))
        starts = [0] if not m[0] else []
        starts += [e + 1 for e in edges if not m[e + 1]]
        ends = [e + 1 for e in edges if m[e + 1]]
        if not m[-1]:
            ends.append(len(m))
        for s0, e0 in zip(starts, ends):
            if e0 - s0 < 1 or (w[e0 - 1] - w[s0]) < 0.0005:
                continue
            th = PLOT_THEMES[self.plot_theme]
            box = BoxAnnotation(left=float(w[s0]), right=float(w[e0 - 1]), fill_color=th["mask"], fill_alpha=th["mask_alpha"], level="underlay")
            self.sub_fig.add_layout(box); self._mask_boxes.append(box)
        # noise table
        noise = self.noise()
        rows = []
        for b in spec.bands:
            i = spec.band_slice(b)
            rows.append({"band": b, "λ [µm]": f"{spec.wave[i].min():.2f}–{spec.wave[i].max():.2f}", "pixels": len(i),
                         "masked": int((~spec.mask[i]).sum()), "MAD noise [mJy]": f"{1e3 * np.nanmedian(noise[i]):.3f}",
                         "pipeline err [mJy]": f"{1e3 * np.nanmedian(spec.err[i]):.3f}", "median flux [Jy]": f"{np.nanmedian(spec.flux[i]):.3f}"})
        self.noise_table.object = _df_html(pd.DataFrame(rows))
        self.cont_fig.title.text = f"Continuum — {cfg.continuum.method}"
        self.cfg.continuum = cfg.continuum; self.cfg.masks = cfg.masks
        if self.disp_model is not None:
            # absorption screens multiply the continuum: give the display model the new one
            try:
                self.disp_model.set_continuum(spec.continuum[self.disp_sel])
                if self.disp_model.unit_cache is not None:
                    self.disp_model.unit_cache.clear()
            except Exception:
                pass
            self.update_model_plot(refresh_data=True)

    # ------------------------------------------------------------------ Model tab
    def _build_model_tab(self):
        W = dict(sizing_mode="stretch_width")
        self.windows_input = pn.widgets.TextInput(name="fit windows [µm] (comma separated lo-hi)", min_width=260,
                                                  value=", ".join(f"{a}-{b}" for a, b in self.cfg.fit.windows) or "13.5-16.5", **W)
        self.windows_input.param.watch(lambda e: self._schedule("structure"), "value")
        self.add_mol = pn.widgets.Select(name="add component", options=list(MOLECULES), value="H2O", **W)
        self.add_release = pn.widgets.Select(name="line list", options=ComponentCard._release_options("H2O"), value="auto", width=110)
        self.add_mol.param.watch(self._add_mol_changed, "value")
        self.add_btn = pn.widgets.Button(name="+ add", button_type="success", width=80, margin=(22, 5, 5, 5))
        self.add_btn.on_click(lambda e: self.add_component(ComponentConfig(
            name=self._unique_name(self.add_mol.value), molecule=self.add_mol.value, logN=17.0, T=500.0, logR=-0.5,
            linelist_release=None if self.add_release.value == "auto" else self.add_release.value)))
        self.windows_btn = pn.widgets.Button(name="default windows", width=150, margin=(22, 5, 5, 5),
                                             description="fill the fit windows from the default ranges of the molecules in the model")
        self.windows_btn.on_click(lambda e: self._default_windows())
        self.nnls_btn = pn.widgets.Button(name="Auto areas (NNLS)", button_type="primary", width=160, margin=(22, 5, 5, 5))
        self.nnls_btn.on_click(lambda e: self.auto_areas())
        self.detect_btn = pn.widgets.Button(name="🔍 Detect molecules", button_type="default", width=170, margin=(22, 5, 5, 5),
                                            description="template-match every cached molecule (NNLS) and suggest the components to fit")
        self.detect_btn.on_click(lambda e: self._detect_start())
        self.detect_thr = pn.widgets.FloatInput(name="ΔBIC threshold", value=self.cfg.fit.detect.threshold, step=5, width=110)
        self.detect_status = _html('<div class="sf-note">Press <b>Detect molecules</b>: templates of every molecule with a cached line list '
                                   '(hot / warm / cold water always included) are fitted simultaneously by NNLS; a molecule counts as '
                                   'detected when removing it raises χ² in its own windows by more than the BIC penalty.</div>', sizing_mode="stretch_width")
        self.detect_table = _html("", sizing_mode="stretch_width")
        self.detect_use_btn = pn.widgets.Button(name="Use detected components", button_type="primary", width=230, visible=False)
        self.detect_add_btn = pn.widgets.Button(name="Add missing only", width=150, visible=False)
        self.detect_use_btn.on_click(lambda e: self._detect_apply(replace=True))
        self.detect_add_btn.on_click(lambda e: self._detect_apply(replace=False))
        self._detection = None
        self.grid_comp = pn.widgets.Select(name="component", options=[], width=160)
        self.grid_btn = pn.widgets.Button(name="Compute grid (log N, T)", button_type="primary", width=200, margin=(22, 5, 5, 5))
        self.grid_btn.on_click(lambda e: self.run_grid_one())
        self.cards_col = pn.Column(**W)
        # figures: one ColumnDataSource per plotted series (sub-bands separated by NaN gaps)
        self.model_xr = Range1d(13.4, 16.6)     # explicit Range1d: a DataRange1d ignores later start/end updates
        self.model_fig = _fig("Model overlay", height=H_MODEL, x_range=self.model_xr)
        self._add_feature_marks(self.model_fig, labels_at="top")
        self.model_data_src = ColumnDataSource(dict(w=[], f=[]))
        self.model_fit_src = ColumnDataSource(dict(w=[], m=[]))
        self.model_unit_src: dict[str, ColumnDataSource] = {}
        self.model_unit_renderers = []
        self._reg("data", self.model_fig.line("w", "f", source=self.model_data_src, color=PAL.data, line_width=1.1))
        self.r_total = self._reg("total", self.model_fig.line("w", "m", source=self.model_fit_src, color=PAL.pink, line_width=1.5))
        self.resid_fig = _fig("Residual (data − model)/σ", height=H_RESID, y_label="(d−m)/σ", x_range=self.model_fig.x_range)
        self.resid_src = ColumnDataSource(dict(w=[], r=[]))
        self._reg("data", self.resid_fig.line("w", "r", source=self.resid_src, color=PAL.data, line_width=0.9))
        self.resid_fig.add_layout(self._reg("total", Span(location=0, dimension="width", line_color=PAL.pink, line_width=0.8)))
        for y in (-3, 3):
            self.resid_fig.add_layout(self._reg("zero", Span(location=y, dimension="width", line_color=PAL.muted, line_width=0.5, line_dash="dotted")))
        self.model_legend = Legend(items=[LegendItem(label="total model", renderers=[self.r_total])])
        _style_legend(self.model_legend, "top_left")
        self._reg("legend", self.model_legend)
        self.model_fig.add_layout(self.model_legend)
        self.chi2_html = _html("", sizing_mode="stretch_width")
        self.grid_pane = pn.pane.Matplotlib(None, dpi=90, tight=True, sizing_mode="scale_width", max_width=640, visible=False)
        self.grid_note = _html('<div class="sf-note" style="margin-top:6px">Δχ² over (log N, T) for one component with the areas of every unit solved by NNLS at each '
                               'grid point (ranges from the <b>Fit</b> workspace). The best point is pushed to the sliders.</div>')
        self._grid_paper = pn.Column(self.grid_pane, css_classes=["sf-paper"], visible=False, margin=(10, 0, 0, 0))
        plot_tabs = pn.Tabs(
            ("Spectrum", pn.Column(pn.pane.Bokeh(self.model_fig, **W), pn.pane.Bokeh(self.resid_fig, **W), self.chi2_html, **W)),
            ("Grid map (log N, T)", pn.Column(pn.Row(self.grid_comp, self.grid_btn), self.grid_note, self._grid_paper, **W)),
            ("Detect molecules", pn.Column(pn.Row(self.detect_btn, self.detect_thr, self.detect_use_btn, self.detect_add_btn),
                                           self.detect_status, self.detect_table, **W)),
            dynamic=False, css_classes=["sf-subtabs"], **W)
        self._plot_tabs = plot_tabs
        toolbar = pn.Column(
            pn.Row(self.windows_input, self.windows_btn, self.nnls_btn, **W),
            css_classes=["sf-toolbar"], margin=(0, 0, 10, 0), **W)
        inspector = pn.Column(
            pn.Row(_label("Components", margin=(10, 0, 0, 0)), **W),
            pn.Row(self.add_mol, self.add_release, self.add_btn, **W),
            self.cards_col,
            width=400, css_classes=["sf-inspector"],
            styles={"position": "sticky", "top": "0px", "max-height": "calc(100vh - 130px)", "overflow-y": "auto"})
        left = pn.Column(toolbar, _panel(plot_tabs), **W)
        self.model_tab = pn.Row(left, inspector, **W)

    def _add_mol_changed(self, e):
        opts = ComponentCard._release_options(e.new)
        self.add_release.options = opts
        if self.add_release.value not in opts:
            self.add_release.value = "auto"

    def _unique_name(self, mol):
        names = {c.name.value for c in self.cards}
        n = mol
        k = 2
        while n in names:
            n = f"{mol}_{k}"; k += 1
        return n

    def _default_windows(self):
        ws = []
        for c in self.cards:
            ws += DEFAULT_WINDOWS.get(c.molecule.value, [])
        from .model import merge_intervals
        ws = merge_intervals(ws)
        self.windows_input.value = ", ".join(f"{a}-{b}" for a, b in ws)

    def add_component(self, cfg: ComponentConfig, rebuild=True):
        card = ComponentCard(self, cfg)
        self.cards.append(card)
        self.cards_col.append(card.panel)
        self.grid_comp.options = [c.name.value for c in self.cards]
        if self.grid_comp.value not in self.grid_comp.options and self.grid_comp.options:
            self.grid_comp.value = self.grid_comp.options[0]
        if rebuild:
            self.on_structure_change()

    def remove_component(self, card):
        self.cards.remove(card)
        self.cards_col.remove(card.panel)
        self.grid_comp.options = [c.name.value for c in self.cards]
        self.on_structure_change()

    def windows(self) -> list[tuple[float, float]]:
        try:
            return _parse_ranges(self.windows_input.value) or [(13.5, 16.5)]
        except Exception:
            return [(13.5, 16.5)]

    def components(self) -> list[Component]:
        from .config import apply_water_split
        comps = [c.to_config().to_component() for c in self.cards]
        # the same default as the pipeline: water slabs without their own windows / Tvib stay off the nu2 band
        return apply_water_split(comps, self.windows(), self.cfg.fit.water_split_um)

    @staticmethod
    def _cat(idx_list: list[np.ndarray], arr: np.ndarray) -> np.ndarray:
        """Concatenate per-band pieces of `arr` with a NaN separator, so one line glyph draws every
        sub-band without connecting them across the overlaps."""
        if not idx_list:
            return np.array([])
        return np.concatenate([np.append(arr[i], np.nan) for i in idx_list])

    @_bokeh_safe
    def on_structure_change(self):
        """Rebuild the display model (new molecules, windows, distance...)."""
        if self.spec is None or self._busy:
            return
        try:
            wins = self.windows()
            comps = self.components()
            sel = in_ranges(self.spec.wave, wins)
            self.disp_sel = sel
            for card in self.cards:
                card._tau_txt = None
                card.refresh_head()
            self.grid_comp.options = [c.name.value for c in self.cards]
            self.disp_model = build_model(comps, self.spec.wave[sel], self.spec.distance_pc, wins, oversample=4,
                                          R_model=self.cfg.R_model, R_scale=self.cfg.R_scale,
                                          continuum=self.spec.continuum[sel] if self.spec.continuum is not None else None)
            self.disp_model.unit_cache = {}
            for card, comp in zip(self.cards, comps):
                try:
                    ll = self.disp_model.linelist_for(comp)
                    want = self.disp_model.release_of(comp)
                    card._ll_txt = ll.release + ("" if ll.release == want else f" (no {want} cached)")
                except Exception:
                    card._ll_txt = "?"
                card.refresh_head()
            band = self.spec.band[sel]
            self._band_idx = [i for i in (np.flatnonzero(band == b) for b in BAND_ORDER) if len(i)]
            self._cat_w = self._cat(self._band_idx, self.spec.wave[sel])
            # per-unit renderers
            for r in self.model_unit_renderers:
                self.model_fig.renderers.remove(r)
            self.model_unit_renderers = []
            self.model_unit_src = {}
            units = list(dict.fromkeys((c.group or c.name) for c in comps if c.enabled and not c.tie_to))
            items = [LegendItem(label="total model", renderers=[self.r_total])]
            for u in units:
                mol = next(c.molecule for c in comps if (c.group or c.name) == u)
                colour = get_molecule(mol).colour
                src = ColumnDataSource(dict(w=[], f=[]))
                self.model_unit_src[u] = src
                r = self.model_fig.line("w", "f", source=src, color=colour, line_width=1.0, alpha=0.9)
                self.model_unit_renderers.append(r)
                items.append(LegendItem(label=u, renderers=[r]))
            self.model_legend.items = items
            self._emphasise_features({c.molecule for c in comps if c.enabled})
            lo = min(w[0] for w in wins); hi = max(w[1] for w in wins)
            self.model_xr.start = lo - 0.02 * (hi - lo); self.model_xr.end = hi + 0.02 * (hi - lo)
            self.update_model_plot(refresh_data=True)
        except Exception as ex:
            self.chi2_html.object = f'<div class="sf-note">⚠ model build failed: {ex} (no pixels in the fit windows?)</div>'

    def on_param_change(self):
        if self.disp_model is None or self._busy:
            return
        self.update_model_plot()

    @_bokeh_safe
    def update_model_plot(self, refresh_data: bool = False):
        if self.disp_model is None or self.spec is None:
            return
        spec, sel = self.spec, self.disp_sel
        comps = self.components()
        self.disp_model.components = comps
        P = {c.name: c.params() for c in comps}
        try:
            total, units, tmax = self.disp_model.evaluate(P, per_unit=True)
        except Exception as ex:
            self.chi2_html.object = f'<div class="sf-note">⚠ model evaluation failed: {ex}</div>'; return
        wave = spec.wave[sel]; y = spec.line_flux[sel]; mask = spec.mask[sel]
        noise = self.noise()[sel]
        sig = np.where(np.isfinite(noise) & (noise > 0), noise, np.nanmedian(spec.err[sel]))
        idx, cw = self._band_idx, self._cat_w
        if refresh_data:
            self.model_data_src.data = dict(w=cw, f=self._cat(idx, np.where(mask, y, np.nan)))
        self.model_fit_src.data = dict(w=cw, m=self._cat(idx, total))
        self.resid_src.data = dict(w=cw, r=self._cat(idx, np.where(mask, (y - total) / sig, np.nan)))
        for u, f in units.items():
            src = self.model_unit_src.get(u)
            if src is not None:
                src.data = dict(w=cw, f=self._cat(idx, f))
        chips = []
        for a, bb in self.windows():
            w = (wave >= a) & (wave <= bb) & mask
            if w.sum() > 3:
                chi2 = np.sum(((y[w] - total[w]) / sig[w]) ** 2) / w.sum()
                tone = "teal" if chi2 < 1.5 else ("accent" if chi2 < 4 else "warn")
                chips.append(chip(f"{a:.2f}–{bb:.2f} µm · χ²<sub>red</sub> <b>{chi2:.2f}</b> · {w.sum()} px", tone))
        self.chi2_html.object = '<div style="margin-top:6px">' + "".join(chips) + "</div>"
        self._last_chi2 = "".join(chips[:3])
        self._refresh_header()
        for card in self.cards:
            u = (card.group.value.strip() or card.name.value)
            t = tmax.get(u)
            if t != card._tau_txt:
                card._tau_txt = t
                card.refresh_head(t)

    def auto_areas(self):
        if self.disp_model is None:
            return
        spec, sel = self.spec, self.disp_sel
        comps = self.components()
        P = {c.name: c.params() for c in comps}
        noise = self.noise()[sel]
        sig = np.where(np.isfinite(noise) & (noise > 0), noise, np.nanmedian(spec.err[sel]))
        logR, chi2 = self.disp_model.solve_areas(spec.line_flux[sel], sig, P, mask=spec.mask[sel])
        self._busy = True
        try:
            for card in self.cards:
                u = card.group.value.strip() or card.name.value
                if u in logR and not card.tie_to.value.strip():
                    card.logR.value = float(np.clip(logR[u], card.logR.start, card.logR.end))
        finally:
            self._busy = False
        self.update_model_plot()
        pn.state.notifications.info(f"areas solved by NNLS, χ² = {chi2:.0f}")

    @_bokeh_safe
    def run_grid_one(self):
        if self.spec is None or not self.cards:
            return
        name = self.grid_comp.value
        try:
            cfg = self.current_config()
            prob = build_problem(cfg, self.spec)
            g = cfg.fit.grid
            from .plots import plot_grid
            gr = prob.grid(name, logN=np.linspace(g.logN[0], g.logN[1], int(g.logN[2])), T=np.linspace(g.T[0], g.T[1], int(g.T[2])),
                           windows=[w for w in prob.windows if any(a <= w[1] and b >= w[0] for a, b in DEFAULT_WINDOWS.get(next(c.molecule for c in cfg.components if c.name == name), []))] or None)
            self.grid_pane.object = plot_grid(gr); self.grid_pane.visible = True; self._grid_paper.visible = True
            self._plot_tabs.active = 1
            b = gr.best
            card = next(c for c in self.cards if c.name.value == name)
            self._busy = True
            try:
                card.set_values({"logN": b["logN"], "T": b["T"], "logR": b["logR"]})
            finally:
                self._busy = False
            self.update_model_plot()
            pn.state.notifications.success(f"{name}: log N={b['logN']:.2f}, T={b['T']:.0f} K, log R={b['logR']:.2f}" + (" (edge!)" if b["at_edge"] else ""))
        except Exception as ex:
            pn.state.notifications.error(f"grid failed: {ex}"); traceback.print_exc()

    # ------------------------------------------------------------------ molecule detection
    def _detect_start(self):
        if self.spec is None:
            pn.state.notifications.warning("load a spectrum first"); return
        if getattr(self, "_det_worker", None) is not None and self._det_worker.is_alive():
            return
        self._plot_tabs.active = 2
        self.detect_btn.disabled = True
        self._det_state = {"msg": "starting", "frac": 0.0, "done": False, "error": None, "result": None}
        cfg = self.current_config()
        spec = self.spec.copy()

        def worker():
            st = self._det_state
            try:
                from .detect import detect_molecules
                d = cfg.fit.detect
                st["result"] = detect_molecules(spec, candidates=d.candidates or None, threshold=float(self.detect_thr.value),
                                                releases=cfg.linedata.releases, oversample=d.oversample, mode=d.mode,
                                                R_model=cfg.R_model, R_scale=cfg.R_scale,
                                                progress=lambda m, f: st.update(msg=m, frac=f))
            except Exception:
                st["error"] = traceback.format_exc()
            finally:
                st["done"] = True

        self._det_worker = threading.Thread(target=worker, daemon=True); self._det_worker.start()
        self._det_cb = pn.state.add_periodic_callback(self._detect_poll, period=400)

    @_bokeh_safe
    def _detect_poll(self):
        st = self._det_state
        if not st["done"]:
            self.detect_status.object = f'<div class="sf-stage" style="font-size:16px">detecting… <small>{st["msg"]} · {100 * st["frac"]:.0f} %</small></div>'
            return
        self._det_cb.stop()
        self.detect_btn.disabled = False
        if st["error"]:
            self.detect_status.object = f'<div class="sf-note">⚠ detection failed</div><div class="sf-log">{st["error"]}</div>'
            return
        det = st["result"]; self._detection = det
        t = det.table
        rows = []
        for r in t.itertuples():
            tone = "teal" if r.detected else "dim"
            rows.append({"candidate": f'<span class="sf-chip {tone}">{"✓" if r.detected else "·"} {r.candidate}</span>',
                         "T [K]": f"{r.T:.0f}", "log N": f"{r.logN:.1f}",
                         "log R / f_c": (f"f_c {r.fc:.2f}" if getattr(r, "kind", "slab") == "absorption" else f"{r.logR:.2f}"),
                         "Δχ²": f"{r.delta_chi2:.0f}", "ΔBIC": f"{r.delta_BIC:+.0f}", "χ²_red (local)": f"{r.chi2_red_local:.2f}",
                         "pixels": r.n_pixels, "windows [µm]": r.windows})
        self.detect_table.object = _df_html(pd.DataFrame(rows))
        names = ", ".join(f"{c.name} ({c.T:.0f} K)" for c in det.components) or "none"
        wins = ", ".join(f"{a:.2f}–{b:.2f}" for a, b in det.windows)
        self.detect_status.object = (f'<div class="sf-kv">Detected: <b>{names}</b><br>suggested fit windows: <b>{wins or "—"}</b> · '
                                     f'{det.n_pixels} pixels · χ²_red of the template solution {det.chi2_red:.2f}</div>')
        self.detect_use_btn.visible = bool(det.components); self.detect_add_btn.visible = bool(det.components)
        self._refresh_header()

    @_bokeh_safe
    def _detect_apply(self, replace: bool):
        """Push the detected components into the cards (replace all, or add the missing molecules)."""
        det = self._detection
        if det is None or not det.components:
            return
        from .detect import apply_detection
        cfg = self.current_config()
        if replace:
            new = apply_detection(cfg, det, replace_windows=cfg.fit.detect.replace_windows, keep_undetected=False)
        else:
            have = {c.molecule for c in cfg.components}
            new = cfg.model_copy(deep=True)
            new.components = cfg.components + [c for c in det.components if c.molecule not in have]
            if det.ordering and not new.fit.ordering:
                new.fit.ordering = det.ordering
        self._busy = True
        try:
            for card in list(self.cards):
                self.cards.remove(card); self.cards_col.remove(card.panel)
            for cc in new.components:
                self.add_component(cc, rebuild=False)
            if replace and cfg.fit.detect.replace_windows and det.windows:
                self.windows_input.value = ", ".join(f"{a}-{b}" for a, b in det.windows)
            if new.fit.ordering:
                self.fit_ordering.value = ", ".join(f"{a}>{b}" for a, b in new.fit.ordering)
        finally:
            self._busy = False; self._dirty.clear()
        self.on_structure_change()
        self._plot_tabs.active = 0
        pn.state.notifications.success(f"{len(new.components)} components set from the detection")

    # ------------------------------------------------------------------ Fit tab
    def _build_fit_tab(self):
        f = self.cfg.fit
        W = dict(sizing_mode="stretch_width")
        self.fit_area = pn.widgets.Select(name="area parameter", options=["logR", "logNA"], value=f.area_param, **W)
        self.fit_noise = pn.widgets.Checkbox(name="fit noise scale", value=f.fit_noise_scale)
        self.fit_rv = pn.widgets.Checkbox(name="fit RV per component", value=f.fit_rv)
        self.fit_fwhm = pn.widgets.Checkbox(name="fit line width", value=f.fit_fwhm)
        self.fit_pipe_err = pn.widgets.Checkbox(name="use pipeline errors (else MAD noise)", value=f.use_pipeline_err)
        self.fit_ordering = pn.widgets.TextInput(name="T ordering (hot>cold, comma separated)", value=", ".join(f"{a}>{b}" for a, b in f.ordering), placeholder="H2O_hot>H2O_warm", **W)
        self.fit_stages = pn.widgets.CheckBoxGroup(name="stages", options=["grid", "optimise", "mcmc"], value=list(f.stages), inline=True)
        self.fit_auto_detect = pn.widgets.Checkbox(name="auto-detect molecules first (batch / CLI: components are rewritten per target)", value=f.auto_detect)
        self.fit_refine = pn.widgets.IntInput(name="continuum refinement passes", value=self.cfg.continuum.refine_iterations, **W)
        g = f.grid
        self.g_logN = pn.widgets.TextInput(name="grid log N (lo, hi, n)", value=f"{g.logN[0]}, {g.logN[1]}, {int(g.logN[2])}", **W)
        self.g_T = pn.widgets.TextInput(name="grid T (lo, hi, n)", value=f"{g.T[0]}, {g.T[1]}, {int(g.T[2])}", **W)
        o = f.optimise
        self.o_method = pn.widgets.Select(name="optimiser", options=["de", "nelder"], value=o.method, **W)
        self.o_maxiter = pn.widgets.IntInput(name="maxiter", value=o.maxiter, **W)
        self.o_popsize = pn.widgets.IntInput(name="popsize", value=o.popsize, **W)
        m = f.mcmc
        self.m_walkers = pn.widgets.IntInput(name="walkers (0=auto)", value=m.nwalkers or 0, **W)
        self.m_steps = pn.widgets.IntInput(name="steps", value=m.nsteps, **W)
        self.m_proc = pn.widgets.IntInput(name="processes", value=max(m.processes, 1), **W)
        self.m_seed = pn.widgets.IntInput(name="seed", value=m.seed, **W)
        self.out_dir = pn.widgets.TextInput(name="output folder ({target} = source name)", value=self.cfg.output,
                                            placeholder="results/{target}", **W)
        self.out_resolved = _html("", **W)
        self.out_dir.param.watch(lambda e: self._show_outdir(), "value")
        self.run_btn = pn.widgets.Button(name="▶  Run fit", button_type="primary", width=150, margin=(0, 6, 0, 0))
        self.stop_btn = pn.widgets.Button(name="■  Stop", button_type="danger", width=90, margin=(0, 0, 0, 0))
        self.run_btn.on_click(lambda e: self.start_fit())
        self.stop_btn.on_click(lambda e: self._stop.set())
        self.stop_btn.disabled = True
        self.progress = pn.indicators.Progress(name="progress", value=0, max=100, sizing_mode="stretch_width", bar_color="warning", height=10)
        self.stage_html = _html('<div class="sf-stage">idle<small>configure the fit in the panel on the right, then press Run</small></div>', sizing_mode="stretch_width")
        self.log_pane = _html('<div class="sf-log">—</div>', sizing_mode="stretch_width")
        self.lnp_fig = _fig("MCMC — mean log-probability of the walkers (band: 16–84 %)", height=260, y_label="⟨ln p⟩", x_label="step", tools="pan,wheel_zoom,box_zoom,reset,save")
        self.lnp_src = ColumnDataSource(dict(step=[], lnp=[], lo=[], hi=[]))
        self._reg("accent", self.lnp_fig.line("step", "lnp", source=self.lnp_src, color=PAL.accent, line_width=1.5))
        self._reg("accent_fill", self.lnp_fig.varea("step", "lo", "hi", source=self.lnp_src, alpha=0.18, color=PAL.accent))
        inspector = pn.Column(
            _label("Parameterisation"),
            self.fit_area, self.fit_noise, self.fit_rv, self.fit_fwhm, self.fit_pipe_err, self.fit_ordering,
            _label("Stages", margin=(14, 0, 0, 0)), self.fit_stages, self.fit_auto_detect, self.fit_refine,
            _label("Grid", margin=(14, 0, 0, 0)), pn.Row(self.g_logN, self.g_T, **W),
            _label("Optimiser", margin=(14, 0, 0, 0)), pn.Row(self.o_method, self.o_maxiter, self.o_popsize, **W),
            _label("MCMC", margin=(14, 0, 0, 0)), pn.Row(self.m_walkers, self.m_steps, **W), pn.Row(self.m_proc, self.m_seed, **W),
            _label("Output", margin=(14, 0, 0, 0)), self.out_dir, self.out_resolved,
            width=380, css_classes=["sf-inspector"],
            styles={"position": "sticky", "top": "0px", "max-height": "calc(100vh - 140px)", "overflow-y": "auto"})
        self.lnp_panel = _panel(pn.pane.Bokeh(self.lnp_fig, **W), visible=False)
        left = pn.Column(
            _panel(pn.Row(self.stage_html, pn.Row(self.run_btn, self.stop_btn, width=260), **W), self.progress, title="Run"),
            self.lnp_panel,
            _panel(self.log_pane, title="Log"),
            **W)
        self.fit_tab = pn.Row(left, inspector, **W)

    def current_config(self, light: bool = False) -> ProjectConfig:
        """Collect every widget into a ProjectConfig (light: continuum/masks/target only)."""
        cfg = self.cfg.model_copy(deep=True)
        cfg.target.distance_pc = self.distance.value; cfg.target.rv_kms = self.rv.value
        cfg.target.spike_filter = self.spike.value
        cfg.target.extraction = self.extraction_config()
        cfg.continuum = self.continuum_config()
        cfg.masks.default_lines = self.mask_default.value; cfg.masks.oh_prompt = self.mask_oh.value
        cfg.masks.oh_whole_range = self.mask_oh_all.value; cfg.masks.extra = _parse_named_ranges(self.mask_extra.value)
        if light:
            return cfg
        cfg.components = [c.to_config() for c in self.cards]
        cfg.fit.windows = [list(w) for w in self.windows()]
        cfg.fit.area_param = self.fit_area.value; cfg.fit.fit_noise_scale = self.fit_noise.value
        cfg.fit.fit_rv = self.fit_rv.value; cfg.fit.fit_fwhm = self.fit_fwhm.value
        cfg.fit.use_pipeline_err = self.fit_pipe_err.value
        cfg.fit.ordering = [[a.strip(), b.strip()] for a, b in (p.split(">") for p in self.fit_ordering.value.split(",") if ">" in p)]
        cfg.fit.stages = list(self.fit_stages.value)
        cfg.fit.auto_detect = self.fit_auto_detect.value
        cfg.fit.detect.threshold = float(self.detect_thr.value) if hasattr(self, "detect_thr") else cfg.fit.detect.threshold
        cfg.continuum.refine_iterations = self.fit_refine.value
        try:
            cfg.fit.grid.logN = [float(x) for x in self.g_logN.value.split(",")]
            cfg.fit.grid.T = [float(x) for x in self.g_T.value.split(",")]
        except ValueError:
            pass
        cfg.fit.optimise.method = self.o_method.value; cfg.fit.optimise.maxiter = self.o_maxiter.value
        cfg.fit.optimise.popsize = self.o_popsize.value; cfg.fit.optimise.workers = 1
        cfg.fit.mcmc.nwalkers = self.m_walkers.value or None; cfg.fit.mcmc.nsteps = self.m_steps.value
        cfg.fit.mcmc.processes = self.m_proc.value; cfg.fit.mcmc.seed = self.m_seed.value
        cfg.output = self.out_dir.value
        return cfg

    @_bokeh_safe
    def apply_config(self, cfg: ProjectConfig):
        """Push a config into the widgets."""
        self._busy = True
        try:
            self.distance.value = cfg.target.distance_pc; self.rv.value = cfg.target.rv_kms
            ex = cfg.target.extraction
            self.ex_source.value = ex.source
            self.ex_ra.value = "" if ex.ra is None else str(ex.ra); self.ex_dec.value = "" if ex.dec is None else str(ex.dec)
            self.ex_scale.value = ex.aperture_fwhm_scale
            self.ex_arcsec.value = ex.aperture_arcsec or 0.0; self.ex_apcorr.value = ex.apcorr
            self.ex_ann.value = ", ".join(str(v) for v in ex.annulus_arcsec) if ex.annulus_arcsec else ""
            c = cfg.continuum
            self.cont_method.value = c.method; self.cont_protect.value = c.protect
            self.cont_protected.value = _fmt_named(c.protected); self.cont_smooth.value = c.smooth
            self.w_quantile.value = c.quantile; self.w_knots.value = c.knot_spacing; self.w_medwin.value = c.median_window
            self.w_medpct.value = c.median_percentile; self.w_sgwin.value = c.sg_window; self.w_sgord.value = c.sg_order
            self.w_niter.value = c.n_iter; self.w_lam.value = c.lam; self.w_p.value = c.p; self.w_segment.value = c.segment
            self.w_aspls_lam.value = c.aspls_lam; self.w_aspls_alpha.value = c.aspls_alpha
            self.w_overlap.value = c.overlap; self.w_pct.value = c.percentile; self.w_minwin.value = c.min_window
            self.w_anchors.value = ", ".join(f"{a:.4f}" for a in c.anchors); self.w_anchorw.value = c.anchor_width
            self.mask_default.value = cfg.masks.default_lines; self.mask_oh.value = cfg.masks.oh_prompt
            self.mask_oh_all.value = cfg.masks.oh_whole_range; self.mask_extra.value = _fmt_named(cfg.masks.extra)
            for card in list(self.cards):
                self.cards.remove(card); self.cards_col.remove(card.panel)
            for cc in cfg.components:
                self.add_component(cc, rebuild=False)
            self.windows_input.value = ", ".join(f"{a}-{b}" for a, b in cfg.fit.windows)
            f = cfg.fit
            self.fit_area.value = f.area_param; self.fit_noise.value = f.fit_noise_scale; self.fit_rv.value = f.fit_rv
            self.fit_fwhm.value = f.fit_fwhm; self.fit_pipe_err.value = f.use_pipeline_err
            self.fit_ordering.value = ", ".join(f"{a}>{b}" for a, b in f.ordering); self.fit_stages.value = list(f.stages)
            self.fit_auto_detect.value = f.auto_detect; self.detect_thr.value = f.detect.threshold
            self.fit_refine.value = cfg.continuum.refine_iterations
            self.g_logN.value = f"{f.grid.logN[0]}, {f.grid.logN[1]}, {int(f.grid.logN[2])}"
            self.g_T.value = f"{f.grid.T[0]}, {f.grid.T[1]}, {int(f.grid.T[2])}"
            self.o_method.value = f.optimise.method; self.o_maxiter.value = f.optimise.maxiter; self.o_popsize.value = f.optimise.popsize
            self.m_walkers.value = f.mcmc.nwalkers or 0; self.m_steps.value = f.mcmc.nsteps; self.m_proc.value = f.mcmc.processes
            self.m_seed.value = f.mcmc.seed; self.out_dir.value = cfg.output
        finally:
            self._busy = False
            self._dirty.clear()
        from .examples import resolve_path
        if cfg.target.path and cfg.target.path != self.cfg.target.path and os.path.exists(resolve_path(cfg.target.path)):
            self.load_target(cfg.target.path)
        else:
            self.estimate_continuum(); self.on_structure_change()

    def _show_outdir(self):
        """Show where the next fit will write, with {target} filled in."""
        try:
            c = self.cfg.model_copy()
            c.output = self.out_dir.value or "results/{target}"
            where = c.output_dir(self.spec.name if self.spec is not None else None)
            self.out_resolved.object = f'<div class="sf-note">→ <span class="sf-mono">{where}</span></div>'
        except Exception:
            pass

    # --- background fit --------------------------------------------------------------------------
    def start_fit(self):
        if self.spec is None:
            pn.state.notifications.warning("load a spectrum first"); return
        if self._worker is not None and self._worker.is_alive():
            pn.state.notifications.warning("a fit is already running"); return
        cfg = self.current_config()
        self.cfg = cfg
        self._stop.clear()
        self._progress = {"stage": "starting", "frac": 0.0, "log": [], "lnp": [], "done": False, "error": None, "run": None}
        self.lnp_src.data = dict(step=[], lnp=[], lo=[], hi=[])
        self.lnp_panel.visible = False
        self.run_btn.disabled = True; self.stop_btn.disabled = False
        self._t_fit0 = time.time()
        self._worker = threading.Thread(target=self._fit_worker, args=(cfg, self.spec.copy()), daemon=True)
        self._worker.start()
        self._refresh_header()
        self._cb = pn.state.add_periodic_callback(self._poll, period=1000)

    def _fit_worker(self, cfg, spec):
        pr = self._progress

        def progress(stage, frac, extra):
            pr["stage"] = stage if extra is None else f"{stage} ({extra})"
            pr["frac"] = frac

        def mcmc_progress(frac, sampler):
            pr["stage"] = "mcmc"; pr["frac"] = frac
            lp = sampler.get_log_prob()
            fin = np.where(np.isfinite(lp), lp, np.nan)
            pr["lnp"] = (np.arange(lp.shape[0]), np.nanmean(fin, axis=1), np.nanpercentile(fin, 16, axis=1), np.nanpercentile(fin, 84, axis=1))

        try:
            outdir = cfg.output_dir(spec.name if spec is not None else None)   # e.g. results/FZ_Tau
            os.makedirs(outdir, exist_ok=True)
            spec = prepare(cfg, spec)
            prob = build_problem(cfg, spec)
            run = RunResult(cfg, spec, prob)
            run.outdir = outdir
            orig_say = run.say
            def say(msg):
                orig_say(msg); pr["log"] = list(run.log)
            run.say = say
            run.say(f"{spec.name}: {len(prob.y)} pixels, {prob.ndim} free parameters, {prob.model.grid.n} fine-grid points")
            stages = cfg.fit.stages
            for it in range(cfg.continuum.refine_iterations + 1):
                if "grid" in stages and it == 0:
                    pr["stage"] = "grid"
                    run_grid_stage(run, progress=lambda n, f: progress("grid", f, n))
                if "optimise" in stages:
                    pr["stage"] = "optimise"; pr["frac"] = 0.0
                    run_optimise_stage(run, callback=lambda *a, **k: bool(self._stop.is_set()),
                                       progress=lambda f: progress("optimise", f, None))
                pr["run"] = run
                if it < cfg.continuum.refine_iterations and not self._stop.is_set():
                    gas = np.full(len(spec.wave), np.nan); gas[prob.used] = prob.model_flux(run.theta)
                    chi2_before = prob.chi2(run.theta)
                    run.say(f"continuum refinement pass {it + 1}")
                    spec_new = prepare(cfg, spec.copy(), gas_model=gas)
                    prob_new = build_problem(cfg, spec_new)
                    chi2_after = prob_new.chi2(run.theta)
                    run.say(f"  chi2 old continuum {chi2_before:.0f} -> refined {chi2_after:.0f}")
                    if chi2_after < chi2_before:
                        spec, prob = spec_new, prob_new; run.spec, run.problem = spec, prob
                    else:
                        run.say("  refinement did not improve chi2; keeping the previous continuum"); break
            if "mcmc" in stages and not self._stop.is_set():
                run_mcmc_stage(run, progress=mcmc_progress, stop_event=self._stop, outdir=outdir)
            pr["run"] = run
            save_results(run, outdir)
            run.say(f"saved results to {outdir}")
        except Exception:
            pr["error"] = traceback.format_exc()
        finally:
            pr["done"] = True

    @_bokeh_safe
    def _poll(self):
        pr = self._progress
        self.progress.value = int(100 * pr["frac"])
        el = time.time() - getattr(self, "_t_fit0", time.time())
        self.stage_html.object = (f'<div class="sf-stage">{pr["stage"] or "…"}<small>{100 * pr["frac"]:.0f} % · {el:.0f} s elapsed</small></div>')
        import html as _h
        self.log_pane.object = '<div class="sf-log">' + _h.escape("\n".join(pr["log"][-40:]) or "—") + "</div>"
        if pr["lnp"]:
            st, m, lo, hi = pr["lnp"]
            self.lnp_src.data = dict(step=st, lnp=m, lo=lo, hi=hi)
            self.lnp_panel.visible = True
        if pr["done"]:
            self._cb.stop()
            self.run_btn.disabled = False; self.stop_btn.disabled = True
            if pr["error"]:
                self.log_pane.object = '<div class="sf-log">' + _h.escape("\n".join(pr["log"][-40:]) + "\n\n" + pr["error"]) + "</div>"
                self.stage_html.object = '<div class="sf-stage" style="color:#ff9d9d">failed<small>see the log</small></div>'
                pn.state.notifications.error("fit failed — see log")
            else:
                self.stage_html.object = f'<div class="sf-stage">finished<small>{el:.0f} s · results in the Results workspace</small></div>'
                self.run = pr["run"]
                self.finish_fit()
            self._refresh_header()

    @_bokeh_safe
    def finish_fit(self):
        run = self.run
        if run is None:
            return
        self.spec = run.spec
        P, _ = run.problem.params_from_theta(run.theta)
        self._busy = True
        try:
            for card in self.cards:
                if card.name.value in P:
                    card.set_values(P[card.name.value])
        finally:
            self._busy = False
            self._dirty.clear()
        self.estimate_continuum() if run.cfg.continuum.refine_iterations else None
        self.on_structure_change()
        self.show_results(run)
        pn.state.notifications.success("fit finished")
        self.tabs.active = 4

    # ------------------------------------------------------------------ Results tab
    def _build_results_tab(self):
        W = dict(sizing_mode="stretch_width")
        self.res_md = pn.pane.Markdown("No results yet — run a fit in the **Fit** workspace.", **W)
        self.res_table = pn.widgets.Tabulator(pd.DataFrame(), show_index=False, disabled=True, height=330, **W)
        # scale_width keeps the figure's own aspect ratio, so the paper cards hug the images
        S = dict(sizing_mode="scale_width")
        self.res_fit = pn.pane.Matplotlib(None, dpi=80, tight=True, visible=False, **S)
        self.res_full = pn.pane.Matplotlib(None, dpi=80, tight=True, visible=False, **S)
        self.res_corner = pn.pane.Matplotlib(None, dpi=70, tight=True, visible=False, **S)
        self.res_corr = pn.pane.Matplotlib(None, dpi=80, tight=True, width=650, height=560, visible=False)
        self.res_pp = pn.pane.Matplotlib(None, dpi=80, tight=True, visible=False, **S)
        self.res_traces = pn.pane.Matplotlib(None, dpi=70, tight=True, visible=False, **S)
        self.corner_comps = pn.widgets.MultiChoice(name="corner plot: components", options=[], value=[], width=420)
        self.corner_btn = pn.widgets.Button(name="redraw corner", width=130, margin=(22, 5, 5, 5))
        self.corner_btn.on_click(lambda e: self._draw_corner())
        self.summary_download = pn.widgets.FileDownload(callback=lambda: io.BytesIO(self.res_table.value.to_csv(index=False).encode()),
                                                        filename="summary.csv", label="⬇ summary CSV", width=150)
        self.res_figs = pn.Column(
            _paper(self.res_full, "Best fit — full spectrum"),
            _paper(self.res_fit, "Best fit — fit windows"),
            visible=False, **W)
        self.res_mcmc_figs = pn.Column(
            _paper(self.res_pp, "Posterior predictive (16–84 %)"),
            pn.Column(_label("Corner plot"), pn.Row(self.corner_comps, self.corner_btn), pn.Column(self.res_corner, css_classes=["sf-paper"], **W), margin=(0, 0, 14, 0), **W),
            _paper(self.res_corr, "Cross-parameter correlations"),
            _paper(self.res_traces, "Walker traces"),
            visible=False, **W)
        self.results_tab = pn.Column(
            _panel(self.res_md, title="Summary"),
            _panel(self.res_table, pn.Row(self.summary_download), title="Parameters"),
            self.res_figs, self.res_mcmc_figs, **W)

    @_bokeh_safe
    def show_results(self, run: RunResult):
        from . import plots
        prob = run.problem
        md = [f"## {run.spec.name}", f"{len(prob.y)} pixels, {prob.ndim} free parameters, output `{run.outdir or run.cfg.output}`"]
        if run.opt is not None:
            md.append(f"**Optimiser:** χ²_red = {run.opt.chi2_red:.3f}, BIC = {run.opt.bic:.1f} ({run.opt.runtime_s:.0f} s)")
        try:
            sig = prob.component_significance(run.theta)
            md.append("**Detection test** (Δχ² and ΔBIC when the component is removed; ΔBIC > 10 ⇒ detected): " +
                      ", ".join(f"{r.component}: Δχ²={r.delta_chi2:.0f}, ΔBIC={r.delta_BIC:+.0f} {'✓' if r.detected else '✗'}" for r in sig.itertuples()))
        except Exception as ex:
            md.append(f"detection test failed: {ex}")
        self.res_fit.object = plots.plot_fit(prob, run.theta, title=f"{run.spec.name}"); self.res_fit.visible = True
        self.res_figs.visible = True
        self.res_mcmc_figs.visible = run.mcmc is not None
        try:
            self.res_full.object = plots.plot_full_spectrum(run.spec, prob, run.theta); self.res_full.visible = True
        except Exception as ex:
            md.append(f"full-spectrum plot failed: {ex}")
        if run.mcmc is not None:
            res = run.mcmc
            d = res.diagnostics()
            flags = res.tau_flag(100)
            md.append(f"**MCMC:** {res.nsteps} steps × {res.chain.shape[1]} walkers, acceptance {d['acceptance']:.2f} "
                      f"{'✓' if d['acceptance_ok'] else '⚠'}, max τ = {np.nanmax(d['tau']):.0f} (steps/τ = {d['steps_over_tau']:.0f} "
                      f"{'✓' if d['converged_length'] else '⚠ need ≥ 50'}), max R̂ = {np.nanmax(d['rhat']):.3f} {'✓' if d['rhat_ok'] else '⚠'}, "
                      f"burn-in {d['burn']} steps, {res.runtime_s:.0f} s")
            md.append("**Optically thin fraction** (share of posterior with τ_max < 1; > 50 % ⇒ only N·A is constrained): " +
                      ", ".join(f"{k}: {v:.0%}" for k, v in flags.items()))
            summ = res.summary()
            summ["value"] = [f"{m:.3f} −{a:.3f} +{b:.3f}" for m, a, b in zip(summ["median"], summ["minus"], summ["plus"])]
            self.res_table.value = summ[["parameter", "value", "median", "minus", "plus", "at_edge"]].round(4)
            self.corner_comps.options = [c.name for c in run.cfg.components]
            self.corner_comps.value = list(self.corner_comps.options)[:4]
            self._draw_corner()
            self.res_corr.object = plots.plot_correlation(res); self.res_corr.visible = True
            self.res_pp.object = plots.plot_posterior_predictive(prob, res, n=60); self.res_pp.visible = True
            self.res_traces.object = plots.plot_traces(res); self.res_traces.visible = True
        else:
            P, _ = prob.params_from_theta(run.theta)
            rows = [{"parameter": f"{c}.{k}", "value": f"{v:.3f}"} for c, d in P.items() for k, v in d.items()]
            self.res_table.value = pd.DataFrame(rows)
        self.res_md.object = "\n\n".join(md)

    @_bokeh_safe
    def _draw_corner(self):
        from . import plots
        if self.run is None or self.run.mcmc is None:
            return
        comps = self.corner_comps.value or None
        try:
            self.res_corner.object = plots.plot_corner(self.run.mcmc, comps=comps); self.res_corner.visible = True
        except Exception as ex:
            pn.state.notifications.error(f"corner failed: {ex}")

    # ------------------------------------------------------------------ Batch tab
    def _build_batch_tab(self):
        W = dict(sizing_mode="stretch_width")
        self.batch_root = pn.widgets.TextInput(name="root folder", value=self.data_root, **W)
        self.batch_scan = pn.widgets.Button(name="Scan", width=90, margin=(22, 5, 5, 5))
        self.batch_table = pn.widgets.Tabulator(pd.DataFrame(columns=["name", "path", "distance_pc", "rv_kms", "status"]),
                                                selectable="checkbox", height=350, editors={"distance_pc": "number", "rv_kms": "number"},
                                                show_index=False, **W)
        self.batch_run = pn.widgets.Button(name="▶  Run selected with current config", button_type="primary", width=300)
        self.batch_stop = pn.widgets.Button(name="■  Stop", button_type="danger", width=90, disabled=True)
        self.batch_md = _html("", **W)
        self.batch_scan.on_click(lambda e: self._batch_scan())
        self.batch_run.on_click(lambda e: self._batch_start())
        self.batch_stop.on_click(lambda e: self._stop.set())
        self.batch_download = pn.widgets.FileDownload(callback=lambda: io.BytesIO(self.batch_table.value.to_csv(index=False).encode()),
                                                      filename="targets.csv", label="⬇ target table", width=150)
        self.batch_tab = pn.Column(
            _panel(_html('<div class="sf-note">Run the current configuration on many targets (sequentially in the background; '
                         'for large batches use <span class="sf-mono">jalebi batch config.yaml targets.csv --workers N</span>). '
                         'With <b>auto-detect molecules first</b> ticked in the Fit workspace, each target gets its own component list '
                         'from the molecule detection (written to <span class="sf-mono">detection.csv</span> next to its results).</div>'),
                   pn.Row(self.batch_root, self.batch_scan, **W), self.batch_table,
                   pn.Row(self.batch_run, self.batch_stop, self.batch_download), title="Batch"),
            _panel(self.batch_md, title="Progress"), **W)

    @_bokeh_safe
    def _batch_scan(self):
        root = os.path.expanduser(self.batch_root.value)
        rows = []
        for d in sorted(glob.glob(os.path.join(root, "*"))) + sorted(glob.glob(os.path.join(root, "*", "*"))):
            if os.path.isdir(d) and glob.glob(os.path.join(d, "*x1d.fits")):
                rows.append({"name": os.path.basename(d), "path": d, "distance_pc": self.distance.value, "rv_kms": self.rv.value, "status": ""})
        self.batch_table.value = pd.DataFrame(rows, columns=["name", "path", "distance_pc", "rv_kms", "status"])
        self.batch_md.object = f'<div class="sf-note">{len(rows)} targets</div>'

    def _batch_start(self):
        sel = self.batch_table.selected_dataframe
        if sel.empty:
            pn.state.notifications.warning("select targets first"); return
        cfg = self.current_config()
        self._stop.clear()
        self.batch_run.disabled = True; self.batch_stop.disabled = False
        self._batch_rows = []
        self._batch_state = {"i": 0, "n": len(sel), "done": False, "msgs": []}

        def worker():
            for k, (_, row) in enumerate(sel.iterrows()):
                if self._stop.is_set():
                    break
                from .pipeline import target_config
                c = target_config(cfg, row)          # own folder per target, e.g. results/DR_Tau
                try:
                    from .pipeline import run_pipeline
                    run = run_pipeline(c, stop_event=self._stop)
                    self._batch_rows.append(catalogue_row(run)); status = "done"
                except Exception as ex:
                    self._batch_rows.append({"target": row["name"], "error": repr(ex)}); status = f"failed: {ex}"
                self._batch_state["msgs"].append(f"{row['name']}: {status}")
                self._batch_state["i"] = k + 1
            if self._batch_rows:
                root = cfg.output_root() if cfg.per_target_output() else cfg.output
                os.makedirs(root, exist_ok=True)
                pd.DataFrame(self._batch_rows).to_csv(os.path.join(root, "population.csv"), index=False)
            self._batch_state["done"] = True

        threading.Thread(target=worker, daemon=True).start()
        self._batch_cb = pn.state.add_periodic_callback(self._batch_poll, period=2000)

    @_bokeh_safe
    def _batch_poll(self):
        s = self._batch_state
        self.batch_md.object = (f'<div class="sf-stage">{s["i"]}/{s["n"]}<small>targets done</small></div>'
                                + '<div class="sf-log">' + ("\n".join(s["msgs"]) or "—") + "</div>")
        if s["done"]:
            self._batch_cb.stop()
            self.batch_run.disabled = False; self.batch_stop.disabled = True
            pn.state.notifications.success("batch finished; population.csv written")

    # ------------------------------------------------------------------ entry
    def servable(self):
        return self.template


def make_app(data_root: str | None = None, config_path: str | None = None, start_tab: str = "data", cube_path: str | None = None,
             start_module: str | None = None, rotdiag_config: str | None = None):
    return JalebiApp(data_root=data_root, config_path=config_path, start_tab=start_tab, cube_path=cube_path,
                     start_module=start_module, rotdiag_config=rotdiag_config).servable()


if __name__.startswith("bokeh"):
    make_app(data_root=os.environ.get("JALEBI_DATA_ROOT")).servable()
