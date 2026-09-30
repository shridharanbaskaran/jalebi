"""The *Rotation diagram* module of the web app.

Left: the spectrum with the selected lines marked, the line table (tick lines in or out of the fit), the fit of
the selected line, the rotation diagram with the model, and the posterior (corner plot, parameters, derived
quantities, model comparison).  Right: the line selection, measurement, physics (model, extinction, OPR,
optical depth, geometry) and MCMC settings, downloads, and the equivalent terminal command / Python code.

Workflow: pick a spectrum (the LTE-fit target, a file, a region sent from the Cube module) or a table of
fluxes → molecule → ① Find lines → ② Measure → ③ Fit → ④ MCMC.  "Run all" does the four in a row.
"""
from __future__ import annotations

import io
import threading
import traceback

import numpy as np
import pandas as pd
import panel as pn
from bokeh.models import ColumnDataSource, HoverTool, Range1d, Span, Whisker

from ..app import PAL, PLOT_THEMES, _bokeh_safe, _df_html, _fig, _html, _panel, _paper, _theme_fig
from ..molecules import get_molecule
from .config import RotDiagConfig
from .extinction import available_curves
from .features import vib_bands
from .physics import ARCSEC
from .species import load_species_linelist, preset, releases_for, rotdiag_molecules

DEFAULT_EXAMPLE = "example:synthetic/rotdiag_H2_synthetic.csv.gz"
# physics suggested when a molecule is picked (the usual choices in the JWST literature; all can be changed)
SUGGESTED = {
    "H2": dict(model="two", opr="species", av_free=True, opacity=False, geo="number", method="gauss", also_pl=True),
    "CO": dict(model="single", opr="thermal", av_free=False, opacity=True, geo="radius", method="gauss"),
    "13CO": dict(model="single", opr="thermal", av_free=False, opacity=True, geo="radius", method="gauss"),
    "OH": dict(model="two", opr="thermal", av_free=False, opacity=False, geo="number", method="gauss"),
    "H2O": dict(model="single", opr="thermal", av_free=False, opacity=True, geo="radius", method="integrate"),
}
SPIN_COL = {"o": "#ff6b6b", "p": "#5fb3ff", "op": "#c77dff", "": PAL.teal}
LADDER_COLS = ["#5fb3ff", "#ff6b6b", "#3ddc84", "#ffb347", "#c77dff", "#f6c343", "#a9d8ff", "#ffb3b3"]
TABLE_COLS = ["use", "label", "wave", "eu", "gA", "flux", "flux_err", "snr", "detected", "n_members", "spin", "ladder",
              "band", "neighbours", "contaminants", "note"]


def _fmt(v, f=".4g"):
    try:
        return "—" if v is None or not np.isfinite(v) else format(v, f)
    except (TypeError, ValueError):
        return str(v)


class RotDiagWorkspace:
    def __init__(self, app, config_path: str | None = None):
        self.app = app
        self.cfg = RotDiagConfig.load(config_path) if config_path else RotDiagConfig()
        self.spec = None
        self.sent_spec = None
        self.features = None            # measured (or table) features
        self.members = None
        self.stamps: dict = {}
        self.fit = None
        self.fit_extra = {}
        self.comparison = None
        self._job = None
        self._cb = None
        self._progress = 0.0
        self._loaded_key = None
        self._shown = False
        W = dict(sizing_mode="stretch_width")
        c = self.cfg
        # ---- input ------------------------------------------------------------------------------
        self.src = pn.widgets.RadioButtonGroup(options={"file": "file", "LTE-fit target": "lte", "flux table": "table"},
                                               value="table" if (c.fluxes and c.fluxes.path) else "file", button_type="default", **W)
        self.path = pn.widgets.TextInput(name="spectrum: x1d folder, CSV, FITS (or example:…)", value=c.spectrum.path or DEFAULT_EXAMPLE, **W)
        self.table_path = pn.widgets.TextInput(name="flux table: CSV with label (S(1) …) or wave, flux, err",
                                               value=(c.fluxes.path if c.fluxes else "example:synthetic/rotdiag_H2_fluxes.csv"), **W)
        self.table_unit = pn.widgets.TextInput(name="table flux unit", value=(c.fluxes.unit if c.fluxes else "W m-2"), width=170)
        self.load_btn = pn.widgets.Button(name="Load", width=80, margin=(22, 5, 5, 5))
        self.rv = pn.widgets.FloatInput(name="systemic RV [km/s]", value=c.spectrum.rv_kms, step=1, width=140)
        self.dist = pn.widgets.FloatInput(name="distance [pc]", value=c.geometry.distance_pc or c.spectrum.distance_pc or 140.0, step=5, width=140)
        self.mol = pn.widgets.Select(name="molecule", options=rotdiag_molecules(), value=c.molecule, width=150)
        self.release = pn.widgets.Select(name="line list", options=releases_for(c.molecule), width=150)
        if c.release and c.release in self.release.options:
            self.release.value = c.release
        self.mol_note = _html("", **W)
        self.find_btn = pn.widgets.Button(name="① Find lines", button_type="default", width=120)
        self.measure_btn = pn.widgets.Button(name="② Measure", button_type="default", width=110)
        self.fit_btn = pn.widgets.Button(name="③ Fit", button_type="default", width=90)
        self.mcmc_btn = pn.widgets.Button(name="④ MCMC", button_type="primary", width=100)
        self.run_btn = pn.widgets.Button(name="▶ Run all", button_type="success", width=110)
        self.run_mcmc_cb = pn.widgets.Checkbox(name="with MCMC", value=c.mcmc.enabled, margin=(12, 5, 5, 5))
        self.info = _html('<div class="sf-note">Pick a spectrum and a molecule, then <b>Run all</b> (or step by step ① → ④).</div>', **W)
        self.progress = pn.indicators.Progress(value=0, max=100, visible=False, **W)
        # ---- line selection ---------------------------------------------------------------------
        L = c.lines
        self.wmin = pn.widgets.FloatInput(name="λ min [µm] (0 = all)", value=L.wmin or 0.0, step=0.1, width=150)
        self.wmax = pn.widgets.FloatInput(name="λ max [µm] (0 = all)", value=L.wmax or 0.0, step=0.1, width=150)
        self.bands = pn.widgets.MultiChoice(name="vibrational bands (none = all)", options=[], value=[], **W)
        self.eu_max = pn.widgets.FloatInput(name="E_u max [K] (0 = none)", value=L.eu_max or 0.0, step=500, width=150)
        self.t_ref = pn.widgets.FloatInput(name="ranking T [K] (0 = preset)", value=L.t_ref or 0.0, step=100, width=150)
        self.rel_min = pn.widgets.FloatInput(name="min rel. strength", value=L.rel_min, step=1e-3, width=150)
        self.max_feat = pn.widgets.IntInput(name="max lines", value=L.max_features, step=5, start=1, width=150)
        self.blend = pn.widgets.FloatInput(name="blend if closer than [FWHM]", value=L.blend_fwhm, step=0.1, width=150)
        self.curated = pn.widgets.Checkbox(name="curated line list (H₂O: Banzatti et al. 2025)", value=True if L.curated is None else L.curated,
                                           margin=(8, 10, 2, 10))
        self.R_model = pn.widgets.Select(name="resolving power", options={"MIRI-MRS (Argyriou+2023)": "argyriou2023",
                                                                          "MIRI-MRS (Jones+2023)": "jones2023", "constant R": "constant"},
                                         value=L.resolving_power if isinstance(L.resolving_power, str) else "constant", width=150)
        self.R_const = pn.widgets.FloatInput(name="R (constant)", value=float(L.resolving_power) if not isinstance(L.resolving_power, str) else 2700.0,
                                             step=100, width=150)
        self.include = pn.widgets.TextInput(name="always include (labels, comma-separated)", value=", ".join(L.include), **W)
        self.exclude = pn.widgets.TextInput(name="exclude (labels)", value=", ".join(L.exclude), **W)
        # ---- measurement -----------------------------------------------------------------------
        m = c.measure
        self.method = pn.widgets.Select(name="flux", options={"Gaussian (instrumental width)": "gauss", "Gaussian (free)": "gauss_free",
                                                              "integrate": "integrate"}, value=m.method, width=150)
        self.v_auto = pn.widgets.Checkbox(name="velocity: auto", value=m.velocity == "auto", margin=(8, 10, 2, 10))
        self.v_val = pn.widgets.FloatInput(name="line velocity [km/s]", value=0.0 if m.velocity == "auto" else float(m.velocity), step=5, width=150)
        self.w_auto = pn.widgets.Checkbox(name="width: auto", value=m.width == "auto", margin=(8, 10, 2, 10))
        self.w_val = pn.widgets.FloatInput(name="width / instrumental", value=1.0 if m.width == "auto" else float(m.width), step=0.05, width=150)
        self.window = pn.widgets.FloatInput(name="window ± [FWHM]", value=m.window_fwhm, step=1, width=150)
        self.cont_order = pn.widgets.IntInput(name="baseline order", value=m.cont_order, start=0, end=3, width=150)
        self.cont_src = pn.widgets.Select(name="continuum", options={"local baseline": "local", "the spectrum's continuum": "spectrum"},
                                          value=m.continuum, width=150)
        self.snr = pn.widgets.FloatInput(name="detection S/N", value=m.snr_detect, step=0.5, width=150)
        self.contam = pn.widgets.Checkbox(name="fit known lines of other species in the window", value=m.contaminants, margin=(8, 10, 2, 10))
        self.flux_unit = pn.widgets.Select(name="spectrum unit", options=["Jy", "MJy/sr"], value=m.flux_unit, width=150)
        # ---- physics ---------------------------------------------------------------------------
        f, g = c.fit, c.geometry
        self.model = pn.widgets.Select(name="model", options={"one temperature": "single", "two temperatures": "two",
                                                              "power law dN ∝ T⁻ᵇ dT": "powerlaw"}, value=f.model, width=150)
        self.geo = pn.widgets.Select(name="normalise to", options={"number of molecules": "number", "emitting radius R": "radius",
                                                                   "aperture (column density)": "aperture", "intensity (MJy/sr)": "intensity"},
                                     value=g.mode, width=150)
        self.R_au = pn.widgets.FloatInput(name="R [au]", value=g.R_au, step=0.1, width=150)
        self.ap = pn.widgets.FloatInput(name="aperture radius [″]", value=g.aperture_arcsec, step=0.1, width=150)
        self.omega = pn.widgets.FloatInput(name="or Ω [sr] (0 = from radius)", value=g.omega_sr or 0.0, step=1e-12, width=150)
        self.av = pn.widgets.FloatInput(name="A_V [mag]", value=f.av, step=0.5, width=150)
        self.av_free = pn.widgets.Checkbox(name="fit A_V", value=f.av_free, margin=(22, 10, 2, 10))
        curves = {v: k for k, v in available_curves().items()}
        curves["custom CSV …"] = "custom"
        self.curve = pn.widgets.Select(name="extinction curve", options=curves, value=f.extinction if f.extinction in curves.values() else "custom", width=150)
        self.curve_path = pn.widgets.TextInput(name="curve CSV (λ µm, A_λ/A_V)", value="" if f.extinction in curves.values() else f.extinction, width=150)
        self.opr = pn.widgets.Select(name="ortho/para", options={"thermal (LTE)": "thermal", "free: spin species": "species",
                                                                 "free: ln(OPR/3) offset": "offset"}, value=f.opr, width=150)
        self.opr_val = pn.widgets.FloatInput(name="OPR", value=f.opr_value, step=0.1, width=150)
        self.opr_free = pn.widgets.Checkbox(name="fit OPR", value=f.opr_free, margin=(22, 10, 2, 10))
        self.opacity = pn.widgets.Checkbox(name="optical depth (curve of growth)", value=f.opacity, margin=(8, 10, 2, 10))
        self.fwhm = pn.widgets.FloatInput(name="intrinsic Δv [km/s]", value=f.fwhm_kms, step=0.5, width=150)
        self.fwhm_free = pn.widgets.Checkbox(name="fit Δv", value=f.fwhm_free, margin=(22, 10, 2, 10))
        self.R_free = pn.widgets.Checkbox(name="fit R (needs thick lines)", value=f.R_free, margin=(8, 10, 2, 10))
        self.sys = pn.widgets.FloatInput(name="flux systematic [%]", value=100 * f.sys_frac, step=1, width=150)
        self.use_sel = pn.widgets.Select(name="fit", options={"all measured lines": "all", "detected lines only": "detected"}, value=f.use, width=150)
        self.tmax = pn.widgets.FloatInput(name="power law T_max [K]", value=f.tmax_powerlaw, step=500, width=150)
        self.also_pl = pn.widgets.Checkbox(name="also fit a power law (drawn next to the main model)", value="powerlaw" in f.also,
                                           margin=(8, 10, 2, 10))
        # ---- MCMC --------------------------------------------------------------------------------
        mc = c.mcmc
        self.walkers = pn.widgets.IntInput(name="walkers", value=mc.walkers, step=8, start=8, width=95)
        self.steps = pn.widgets.IntInput(name="steps", value=mc.steps, step=500, start=100, width=95)
        self.burn = pn.widgets.IntInput(name="burn-in", value=mc.burn, step=250, start=0, width=95)
        self.compare_btn = pn.widgets.Button(name="Compare models (1T · 2T · power law)", **W)
        # ---- save ---------------------------------------------------------------------------------
        self.out = pn.widgets.TextInput(name="output folder", value=c.output, **W)
        self.write_btn = pn.widgets.Button(name="Write results folder", **W)
        self.dl_cfg = pn.widgets.FileDownload(callback=self._cfg_bytes, filename="rotdiag.yaml", label="⬇ config YAML", **W)
        self.dl_lines = pn.widgets.FileDownload(callback=self._lines_bytes, filename="rotdiag_lines.csv", label="⬇ lines + fluxes CSV", **W)
        self.dl_fit = pn.widgets.FileDownload(callback=self._fit_bytes, filename="rotdiag_fit.yaml", label="⬇ fit results YAML", **W)
        self.code = _html("", **W)
        # ---- figures + tables -------------------------------------------------------------------
        self._build_figures()
        self.table = pn.widgets.Tabulator(pd.DataFrame(columns=TABLE_COLS), show_index=False, selectable=1, height=430,
                                          layout="fit_data_stretch", editors={k: None for k in TABLE_COLS if k != "use"},
                                          formatters={"use": {"type": "tickCross"}, "detected": {"type": "tickCross"}},
                                          disabled=False, **W)
        self.table.on_edit(self._table_edited)
        self.table.param.watch(self._table_selected, "selection")
        self.params_html = _html('<div class="sf-note">no fit yet</div>', **W)
        self.derived_html = _html("", **W)
        self.compare_html = _html("", **W)
        self.corner = pn.pane.Matplotlib(None, dpi=72, tight=True, format="png", sizing_mode="stretch_width")
        self.legend_html = _html("", **W)
        self.paper = pn.pane.Matplotlib(None, dpi=110, tight=True, format="png", sizing_mode="stretch_width")
        # ---- wiring --------------------------------------------------------------------------------
        self.load_btn.on_click(lambda e: self.load())
        self.find_btn.on_click(lambda e: self.find())
        self.measure_btn.on_click(lambda e: self.measure())
        self.fit_btn.on_click(lambda e: self.run_fit(mcmc=False))
        self.mcmc_btn.on_click(lambda e: self.run_fit(mcmc=True))
        self.run_btn.on_click(lambda e: self.run_all())
        self.compare_btn.on_click(lambda e: self.compare())
        self.write_btn.on_click(lambda e: self.write())
        self.mol.param.watch(self._molecule_changed, "value")
        self.src.param.watch(lambda e: self._show_source(), "value")
        self.geo.param.watch(lambda e: self._show_physics(), "value")
        self.opacity.param.watch(lambda e: self._show_physics(), "value")
        self.opr.param.watch(lambda e: self._show_physics(), "value")
        self.model.param.watch(lambda e: self._show_physics(), "value")
        self.curve.param.watch(lambda e: self._show_physics(), "value")
        self.R_model.param.watch(lambda e: self._show_physics(), "value")
        self.v_auto.param.watch(lambda e: self._show_physics(), "value")
        self.also_pl.param.watch(lambda e: self._show_physics(), "value")
        self.w_auto.param.watch(lambda e: self._show_physics(), "value")
        for w in (self.path, self.table_path, self.table_unit, self.rv, self.dist, self.mol, self.release, self.wmin, self.wmax, self.bands,
                  self.eu_max, self.t_ref, self.rel_min, self.max_feat, self.blend, self.curated, self.R_model, self.R_const, self.include,
                  self.exclude, self.method, self.v_auto, self.v_val, self.w_auto, self.w_val, self.window, self.cont_order, self.cont_src,
                  self.snr, self.contam, self.flux_unit, self.model, self.geo, self.R_au, self.ap, self.omega, self.av, self.av_free,
                  self.curve, self.curve_path, self.opr, self.opr_val, self.opr_free, self.opacity, self.fwhm, self.fwhm_free, self.R_free,
                  self.sys, self.use_sel, self.tmax, self.walkers, self.steps, self.burn, self.out, self.src):
            w.param.watch(lambda e: self.update_code(), "value")
        # ---- layout --------------------------------------------------------------------------------
        self.plot_tabs = pn.Tabs(
            ("Spectrum", pn.Column(pn.pane.Bokeh(self.spec_fig, **W), pn.pane.Bokeh(self.zoom_fig, **W), **W)),
            ("Lines", pn.Column(_html('<div class="sf-note">Tick <b>use</b> to put a line in or out of the fit; click a row to see its '
                                      'fit. Blends are single features (their members summed); <b>neighbours</b> / '
                                      '<b>contaminants</b> are fitted simultaneously in the window.</div>'), self.table, **W)),
            ("Rotation diagram", pn.Column(pn.pane.Bokeh(self.diag_fig, **W), pn.pane.Bokeh(self.res_fig, **W), self.legend_html, **W)),
            ("Paper figure", pn.Column(_html('<div class="sf-note">log₁₀ / ln axes, observed and de-reddened points per vibrational band '
                                             '(ortho points moved onto the para ladder when the OPR is fitted), components and the '
                                             'parameters; the main model and, with <b>also fit a power law</b>, the power law beside it. '
                                             'Saved as rotation_diagram_models.png by <b>Write results folder</b>.</div>'),
                                       _paper(self.paper), **W)),
            ("Posterior", pn.Column(pn.Row(pn.Column(_panel(self.params_html, title="Parameters"), _panel(self.derived_html, title="Derived"),
                                                     _panel(self.compare_html, title="Model comparison"), width=460),
                                           _paper(self.corner, title="Posterior (MCMC)"), **W), **W)),
            dynamic=False, css_classes=["sf-subtabs"], **W)
        top = _panel(self.src, pn.Row(self.path, self.load_btn, **W), pn.Row(self.table_path, self.table_unit, **W),
                     pn.Row(self.rv, self.dist, self.mol, self.release, **W), self.mol_note,
                     pn.Row(self.find_btn, self.measure_btn, self.fit_btn, self.mcmc_btn, self.run_btn, self.run_mcmc_cb),
                     self.progress, self.info, title="Rotation diagram")
        left = pn.Column(top, _panel(self.plot_tabs), **W)
        self.sel_panel = _panel(pn.Row(self.wmin, self.wmax), self.bands, pn.Row(self.eu_max, self.t_ref), pn.Row(self.rel_min, self.max_feat),
                                pn.Row(self.blend, self.R_model), pn.Row(self.R_const), self.curated, self.include, self.exclude,
                                _html('<div class="sf-note">Lines inside the spectrum (away from sub-band edges), ranked by their thin-LTE '
                                      'intensity g A ν e^(−E_u/kT) at the ranking temperature. Lines closer than the blend limit are one '
                                      'feature whose flux is the sum of its members.</div>'),
                                title="Line selection")
        self.meas_panel = _panel(pn.Row(self.method, self.snr), pn.Row(self.v_auto, self.v_val), pn.Row(self.w_auto, self.w_val),
                                 pn.Row(self.window, self.cont_order), pn.Row(self.cont_src, self.flux_unit), self.contam,
                                 _html('<div class="sf-note">Pixel-integrated Gaussians of the instrumental width (× the width scale) at '
                                       'the rest wavelengths shifted by one velocity, fitted with a polynomial baseline and the neighbours '
                                       '(linear least squares: exact errors). "auto" measures v and the width scale on the strongest lines '
                                       '(profile likelihood of the same template; the width per MRS channel).</div>'),
                                 title="Measurement")
        self.phys_panel = _panel(pn.Row(self.model, self.use_sel), pn.Row(self.geo), pn.Row(self.R_au, self.ap),
                                 pn.Row(self.omega), pn.Row(self.av, self.av_free), pn.Row(self.curve, self.curve_path),
                                 pn.Row(self.opr, self.opr_val), self.opr_free, self.opacity, pn.Row(self.fwhm, self.fwhm_free), self.R_free,
                                 pn.Row(self.sys, self.tmax), self.also_pl,
                                 _html('<div class="sf-note">Fitted in flux space: χ² = Σ (F − F_model)² / (σ² + (f_sys F)²), so blends, '
                                       'non-detections, extinction 10^(−0.4 A_V A_λ/A_V) and the optical depth (a Gaussian slab of '
                                       'intrinsic width Δv and area πR²) are exact. OPR "spin species": N_o/N_p = OPR, each in LTE at T '
                                       '(exact partition sums); "offset": ln(OPR/3) added to the ortho levels (e.g. JOYS). Number of '
                                       'molecules needs the distance; column densities need R or the aperture.</div>'),
                                 title="Physics")
        self.mcmc_panel = _panel(pn.Row(self.walkers, self.steps, self.burn), self.compare_btn,
                                 _html('<div class="sf-note">emcee, vectorised over the walkers, started around the least-squares '
                                       'solution; flat priors inside the bounds (two temperatures: T₁ < T₂).</div>'), title="MCMC")
        self.save_panel = _panel(self.out, self.write_btn, self.dl_cfg, self.dl_lines, self.dl_fit, title="Save")
        codep = _panel(self.code, title="Same thing in the terminal / Python")
        inspector = pn.Column(self.sel_panel, self.meas_panel, self.phys_panel, self.mcmc_panel, self.save_panel, codep, width=400,
                              scroll=True, css_classes=["sf-inspector"], styles={"max-height": "calc(100vh - 150px)"})
        self.panel = pn.Row(left, inspector, **W)
        self._molecule_changed(None)
        if config_path is None:
            self.apply_suggested(self.mol.value)
        self._show_source(); self._show_physics()
        self.update_code()

    # ------------------------------------------------------------------ figures
    def figures(self):
        return [self.spec_fig, self.zoom_fig, self.diag_fig, self.res_fig]

    def _build_figures(self):
        th = PLOT_THEMES["dark"]
        self.spec_fig = _fig("spectrum — lines of the molecule marked after ① Find lines", height=300)
        self.spec_src = ColumnDataSource(dict(w=[], f=[]))
        self.spec_line = self.spec_fig.line("w", "f", source=self.spec_src, color=th["data"], line_width=1.0)
        self.mark_src = ColumnDataSource(dict(x=[], y0=[], y1=[], color=[], label=[], eu=[], snr=[], status=[], id=[]))
        self.spec_fig.segment("x", "y0", "x", "y1", source=self.mark_src, color="color", line_width=2, alpha=0.9)
        top = self.spec_fig.scatter("x", "y1", source=self.mark_src, color="color", size=7)
        self.spec_fig.add_tools(HoverTool(renderers=[top], tooltips=[("line", "@label"), ("λ", "@x{0.0000} µm"), ("E_u", "@eu{0} K"),
                                                                    ("S/N", "@snr{0.0}"), ("", "@status")]))
        self.mark_src.selected.on_change("indices", lambda a, o, n: self._marker_tapped(n))
        self.zoom_fig = _fig("click a line (marker or table row) to see its fit", height=260, y_label="F_ν − baseline [Jy]")
        self.zoom_src = ColumnDataSource(dict(x=[], d=[], m=[], own=[]))
        self.zoom_data = self.zoom_fig.step("x", "d", source=self.zoom_src, mode="center", color=th["data"], legend_label="data − baseline")
        self.zoom_fig.line("x", "m", source=self.zoom_src, color=PAL.pink, line_width=2, legend_label="fit (all profiles)")
        self.zoom_fig.line("x", "own", source=self.zoom_src, color=PAL.teal, line_width=2, line_dash="dashed", legend_label="this feature")
        self.zoom_lines = ColumnDataSource(dict(x=[], y0=[], y1=[]))
        self.zoom_fig.segment("x", "y0", "x", "y1", source=self.zoom_lines, color=PAL.accent, line_dash="dotted", line_width=1)
        self.zoom_fig.legend.location = "top_right"; self.zoom_fig.legend.label_text_font_size = "10px"
        self.zoom_fig.legend.background_fill_alpha = 0.3
        tools = "pan,wheel_zoom,box_zoom,reset,save"
        self.diag_fig = _fig("rotation diagram", height=430, x_label="E_u / k [K]", y_label="ln(N_u / g_u)", tools=tools,
                             x_range=Range1d(0, 10000))
        self.pt_src = ColumnDataSource(dict(x=[], y=[], lo=[], hi=[], color=[], label=[], wave=[], snr=[]))
        self.ul_src = ColumnDataSource(dict(x=[], y=[], color=[], label=[], wave=[], snr=[]))
        self.ex_src = ColumnDataSource(dict(x=[], y=[], label=[], wave=[], snr=[]))
        self.mp_src = ColumnDataSource(dict(x=[], y=[], label=[]))
        self.cv_src = ColumnDataSource(dict(xs=[], ys=[], color=[]))
        self.cc_src = ColumnDataSource(dict(xs=[], ys=[], color=[]))
        self.diag_fig.multi_line("xs", "ys", source=self.cv_src, color="color", line_width=2, alpha=0.9)
        self.diag_fig.multi_line("xs", "ys", source=self.cc_src, color="color", line_width=1.2, line_dash="dashed", alpha=0.7)
        self.diag_fig.scatter("x", "y", source=self.mp_src, marker="square", fill_alpha=0.0, line_color=PAL.accent, size=14, line_width=1.2)
        self.diag_fig.add_layout(Whisker(base="x", upper="hi", lower="lo", source=self.pt_src, line_color="color", line_width=1.2))
        r1 = self.diag_fig.scatter("x", "y", source=self.pt_src, color="color", size=8, line_color="white", line_width=0.5)
        r2 = self.diag_fig.scatter("x", "y", source=self.ul_src, marker="inverted_triangle", color="color", fill_alpha=0.2, size=9)
        r3 = self.diag_fig.scatter("x", "y", source=self.ex_src, marker="x", color="#8e97ad", size=8)
        self.diag_fig.add_tools(HoverTool(renderers=[r1, r2, r3], tooltips=[("line", "@label"), ("λ", "@wave{0.0000} µm"), ("E_u", "@x{0} K"),
                                                                             ("ln(N_u/g_u)", "@y{0.00}"), ("S/N", "@snr{0.0}")]))
        self.res_fig = _fig("", height=170, x_label="E_u / k [K]", y_label="data − model", tools=tools)
        self.res_fig.x_range = self.diag_fig.x_range
        self.rs_src = ColumnDataSource(dict(x=[], r=[], lo=[], hi=[], color=[]))
        self.res_fig.add_layout(Whisker(base="x", upper="hi", lower="lo", source=self.rs_src, line_color="color"))
        self.res_fig.scatter("x", "r", source=self.rs_src, color="color", size=6)
        self.res_fig.add_layout(Span(location=0, dimension="width", line_color=th["muted"], line_dash="dotted"))
        for f in self.figures():
            _theme_fig(f, th)

    @_bokeh_safe
    def apply_theme(self, th: dict):
        self.spec_line.glyph.line_color = th["data"]
        self.zoom_data.glyph.line_color = th["data"]

    def on_show(self):
        if not self._shown:
            self._shown = True
            if self.src.value == "file" and self.path.value and self.spec is None:
                try:
                    self.load()
                except Exception as ex:
                    self.info.object = f'<div class="sf-note">⚠ {ex}</div>'

    # ------------------------------------------------------------------ helpers
    def _molecule_changed(self, e):
        name = self.mol.value
        sp = preset(name)
        rel = releases_for(name)
        self.release.options = rel
        if self.release.value not in rel:
            self.release.value = rel[0]
        try:
            ll = load_species_linelist(sp.name, self.release.value)
            t = ll.table
            if self.spec is not None:
                t = t[(t["wave"] >= np.nanmin(self.spec.wave)) & (t["wave"] <= np.nanmax(self.spec.wave))]
            vb = vib_bands(sp.name, t)
            opts = list(pd.Series(vb).value_counts().index[:40])
            self.bands.options = sorted(set(opts) | set(sp.bands))
            self.bands.value = [b for b in sp.bands if b in self.bands.options]
        except Exception as ex:
            self.bands.options = []; self.bands.value = []
            self.mol_note.object = f'<div class="sf-note">⚠ no line list for {name}: {ex}</div>'
            return
        self.curated.visible = bool(sp.curated)
        spin = {"h2": "ortho/para by J parity", "g2j": "ortho/para from g/(2J+1)"}.get(sp.spin, "no spin species")
        self.mol_note.object = (f'<div class="sf-note"><b>{sp.label}</b> · line list <span class="sf-mono">{ll.release}</span> '
                                f'({len(ll)} lines, {ll.source.split("/")[-1] if ll.source else ""}) · {spin} · '
                                f'T range {sp.t_bounds[0]:g}–{sp.t_bounds[1]:g} K. {sp.note}</div>')
        self.features = None; self.fit = None; self.fit_extra = {}
        if e is not None:
            self.apply_suggested(name)

    def apply_suggested(self, name: str):
        """Set the physics usually used for this molecule (see SUGGESTED)."""
        sg = SUGGESTED.get(name)
        if not sg:
            return
        self.model.value = sg["model"]; self.opr.value = sg["opr"]; self.av_free.value = sg["av_free"]
        self.opacity.value = sg["opacity"]; self.geo.value = sg["geo"]; self.method.value = sg["method"]
        self.also_pl.value = sg.get("also_pl", False)
        if sg["opr"] != "thermal":
            self.opr_free.value = True

    def _show_source(self):
        s = self.src.value
        self.path.visible = s == "file"; self.load_btn.visible = s in ("file", "lte", "sent")
        self.table_path.visible = s == "table"; self.table_unit.visible = s == "table"
        self.measure_btn.disabled = s == "table"
        self.load_btn.name = {"file": "Load", "lte": "Use target", "sent": "Reload"}.get(s, "Load")

    def _show_physics(self):
        g = self.geo.value
        self.R_au.visible = g in ("radius", "number")
        self.ap.visible = self.omega.visible = g == "aperture"
        self.curve_path.visible = self.curve.value == "custom"
        free_opr = self.opr.value != "thermal"
        self.opr_val.visible = self.opr_free.visible = free_opr
        op = self.opacity.value
        self.fwhm.visible = self.fwhm_free.visible = self.R_free.visible = op
        self.tmax.visible = self.model.value == "powerlaw" or self.also_pl.value
        self.R_const.visible = self.R_model.value == "constant"
        self.v_val.disabled = self.v_auto.value
        self.w_val.disabled = self.w_auto.value

    def current_config(self) -> RotDiagConfig:
        c = RotDiagConfig().model_dump()
        R = self.R_const.value if self.R_model.value == "constant" else self.R_model.value
        c.update(molecule=self.mol.value, release=self.release.value or None,
                 target=(self.spec.name if self.spec is not None else None))
        c["spectrum"] = dict(path=self.path.value if self.src.value == "file" else "", rv_kms=self.rv.value, distance_pc=self.dist.value)
        c["fluxes"] = dict(path=self.table_path.value, unit=self.table_unit.value) if self.src.value == "table" else None
        split = lambda s: [x.strip() for x in s.split(",") if x.strip()]
        c["lines"].update(wmin=self.wmin.value or None, wmax=self.wmax.value or None, bands=list(self.bands.value),
                          eu_max=self.eu_max.value or None, t_ref=self.t_ref.value or None, rel_min=self.rel_min.value,
                          max_features=self.max_feat.value, blend_fwhm=self.blend.value,
                          curated=self.curated.value if self.curated.visible else None,
                          include=split(self.include.value), exclude=split(self.exclude.value), resolving_power=R)
        c["measure"].update(method=self.method.value, velocity="auto" if self.v_auto.value else self.v_val.value,
                            width="auto" if self.w_auto.value else self.w_val.value, window_fwhm=self.window.value,
                            cont_order=self.cont_order.value, continuum=self.cont_src.value, snr_detect=self.snr.value,
                            contaminants=self.contam.value, flux_unit=self.flux_unit.value)
        if self.features is not None and "use" in self.features and self.features["measured"].any():
            skip = self.features.loc[~self.features["use"].astype(bool) & self.features["measured"].astype(bool), "top"].tolist()
            c["measure"]["skip"] = skip
        c["geometry"].update(mode=self.geo.value, distance_pc=self.dist.value, R_au=self.R_au.value, aperture_arcsec=self.ap.value,
                             omega_sr=self.omega.value or None)
        c["fit"].update(model=self.model.value, opr=self.opr.value, opr_value=self.opr_val.value, opr_free=self.opr_free.value,
                        av=self.av.value, av_free=self.av_free.value,
                        extinction=self.curve_path.value if self.curve.value == "custom" else self.curve.value,
                        opacity=self.opacity.value, fwhm_kms=self.fwhm.value, fwhm_free=self.fwhm_free.value, R_free=self.R_free.value,
                        sys_frac=self.sys.value / 100.0, use=self.use_sel.value, tmax_powerlaw=self.tmax.value,
                        also=["powerlaw"] if (self.also_pl.value and self.model.value != "powerlaw") else [])
        c["mcmc"].update(enabled=self.run_mcmc_cb.value, walkers=self.walkers.value, steps=self.steps.value, burn=self.burn.value)
        c["output"] = self.out.value
        return RotDiagConfig.model_validate(c)

    def _start(self, work, done, label, progress=False):
        if self._job is not None and self._job["thread"].is_alive():
            pn.state.notifications.warning("still working on the previous request"); return
        job = {"result": None, "error": None}

        def run():
            try:
                job["result"] = work()
            except Exception as ex:
                job["error"] = f"{ex}\n{traceback.format_exc(limit=4)}"
        job["thread"] = threading.Thread(target=run, daemon=True)
        self._job = job
        self._progress = 0.0
        self.progress.visible = progress; self.progress.value = 0
        self.info.object = f'<div class="sf-kv">⏳ {label} …</div>'
        job["thread"].start()

        def finish():
            self.progress.visible = False
            if job["error"]:
                msg = job["error"].splitlines()[0]
                self.info.object = f'<div class="sf-note">⚠ {label} failed: {msg}</div>'
                if pn.state.curdoc is not None:
                    pn.state.notifications.error(f"{label} failed: {msg}")
                return False
            done(job["result"])
            return True

        if pn.state.curdoc is None:           # not served (tests, scripts): run inline
            job["thread"].join()
            if job["error"]:
                raise RuntimeError(job["error"])
            finish()
            return

        def poll():
            if job["thread"].is_alive():
                if progress:
                    self.progress.value = int(100 * self._progress)
                return
            self._cb.stop(); self._cb = None
            finish()
        from ..cube.app import _bokeh_safe_fn
        self._cb = pn.state.add_periodic_callback(_bokeh_safe_fn(poll), period=250)

    # ------------------------------------------------------------------ actions
    def load(self):
        from .pipeline import load_input_spectrum
        s = self.src.value
        if s == "lte":
            if self.app.spec is None:
                pn.state.notifications.warning("the LTE-fit module has no target loaded: load one in its Data tab") if pn.state.curdoc else None
                self.info.object = '<div class="sf-note">⚠ no LTE-fit target: load one in LTE slab fit → Data</div>'
                return
            self._set_spectrum(self.app.spec.copy(), "LTE-fit target")
        elif s == "file":
            cfg = self.current_config()
            spec = load_input_spectrum(cfg)
            self._set_spectrum(spec, self.path.value)
        elif s == "sent" and self.sent_spec is not None:
            self._set_spectrum(self.sent_spec, "sent spectrum")

    def use_spectrum(self, spec, label: str | None = None):
        """A spectrum sent from another module (e.g. a Cube region)."""
        self.sent_spec = spec
        opts = dict(self.src.options)
        opts["sent from Cube"] = "sent"
        self.src.options = opts
        self.src.value = "sent"
        ex = spec.meta.get("extraction") if isinstance(spec.meta, dict) else None
        if isinstance(ex, dict) and ex.get("area_arcsec2"):
            self.geo.value = "aperture"
            self.omega.value = float(ex["area_arcsec2"]) * ARCSEC ** 2
        self.dist.value = spec.distance_pc
        self._set_spectrum(spec, label or spec.name)

    @_bokeh_safe
    def _set_spectrum(self, spec, label):
        self.spec = spec
        self.features = None; self.fit = None; self.fit_extra = {}; self.stamps = {}
        w, f, _ = spec.stitched()
        self.spec_src.data = dict(w=w, f=f)
        self.spec_fig.title.text = f"{spec.name}: {len(spec.wave)} pixels, {', '.join(spec.bands) or '1 band'}"
        self.mark_src.data = {k: [] for k in self.mark_src.data}
        if spec.distance_pc and self.src.value != "table":
            self.dist.value = spec.distance_pc
        self._molecule_changed(None)
        self.info.object = f'<div class="sf-kv">spectrum <b>{spec.name}</b> ({label}) — now <b>① Find lines</b></div>'
        self.update_code()

    def find(self):
        from .features import find_features
        from .pipeline import features_from_table, selection
        cfg = self.current_config()
        if self.src.value == "table":
            F, M, msgs = features_from_table(cfg)
            self.features, self.members, self.stamps, self.fit = F, M, {}, None
            self.show_features()
            self.info.object = (f'<div class="sf-kv">{len(F)} lines from the table{" · " + "; ".join(msgs) if msgs else ""} — '
                                f'now <b>③ Fit</b></div>')
            self.plot_tabs.active = 2
            self.show_diagram()
            return
        if self.spec is None:
            self.load()
        if self.spec is None:
            return
        F, M = find_features(cfg.molecule, self.spec, selection(cfg), release=cfg.release)
        F["use"] = True; F["measured"] = False
        for k in ("flux", "flux_err", "snr"):
            F[k] = np.nan
        F["detected"] = False; F["note"] = ""
        self.features, self.members, self.stamps, self.fit = F, M, {}, None
        self.show_features()
        self.info.object = (f'<div class="sf-kv"><b>{len(F)}</b> {get_molecule(cfg.molecule).label} features '
                            f'({int((F["n_members"] > 1).sum())} blends) — now <b>② Measure</b></div>')

    def measure(self, then=None):
        if self.features is None:
            self.find()
        if self.features is None or self.spec is None:
            return
        from .measure import measure_features
        from .pipeline import apply_use, measure_config
        cfg = self.current_config()
        F0 = self.features[[c for c in self.features.columns if c not in ("flux", "flux_err", "snr", "detected", "note", "use", "measured")]]
        F0.attrs.update(self.features.attrs)
        spec, M = self.spec, self.members
        keep_skip = set(self.features.loc[~self.features["use"].astype(bool), "top"]) if "use" in self.features else set()

        def work():
            Fm, st = measure_features(spec, F0, M, measure_config(cfg), cfg.molecule)
            Fm = apply_use(cfg, Fm)
            if keep_skip:
                Fm["use"] = Fm["use"] & ~Fm["top"].isin(keep_skip)
            return Fm, st

        def done(r):
            self.features, self.stamps = r
            self.fit = None
            F = self.features
            self.show_features()
            ws = F.attrs.get("width_scales", {})
            self.info.object = (f'<div class="sf-kv">measured <b>{int(F["measured"].sum())}</b> features, <b>{int(F["detected"].sum())}</b> '
                                f'detected (S/N ≥ {cfg.measure.snr_detect:g}) · line velocity {F.attrs.get("velocity_kms", 0):.1f} km/s · '
                                f'width × {", ".join(f"{k}: {v:.2f}" for k, v in ws.items()) or "1"} — now <b>③ Fit</b></div>')
            self.show_diagram()
            det = F[F["measured"].astype(bool)]
            if len(det):
                self.show_line(int(det.loc[det["snr"].astype(float).idxmax(), "id"]))
            if then:
                then()
        self._start(work, done, "measuring the lines")

    def run_fit(self, mcmc: bool = False, then=None):
        if self.features is None or not self.features.get("measured", pd.Series(False)).any():
            if self.src.value == "table":
                self.find()
            else:
                self.info.object = '<div class="sf-note">measure the lines first (② Measure)</div>'
                return
        from .fit import FitConfig, MCMCConfig, fit_rotation
        from .pipeline import fit_config
        cfg = self.current_config()
        F, M = self.features, self.members
        fcfg = fit_config(cfg, self.dist.value)
        mc = MCMCConfig(cfg.mcmc.walkers, cfg.mcmc.steps, cfg.mcmc.burn, cfg.mcmc.thin, cfg.mcmc.seed) if mcmc else None

        def prog(x):
            self._progress = x

        extra = [k for k in cfg.fit.also if k != cfg.fit.model]

        def work():
            main = fit_rotation(F, M, fcfg, cfg.molecule, mcmc=mc, progress=(lambda x: prog(x / (1 + len(extra)))))
            ex = {}
            for i, k in enumerate(extra):
                ex[k] = fit_rotation(F, M, FitConfig(**{**fcfg.__dict__, "model": k}), cfg.molecule, mcmc=mc,
                                     progress=(lambda x, i=i: prog((i + 1 + x) / (1 + len(extra)))))
            return main, ex

        def done(r):
            res, self.fit_extra = r
            self.fit = res
            self.show_diagram(); self.show_results()
            self.plot_tabs.active = 4 if mcmc else 2
            self.info.object = (f'<div class="sf-kv">{"MCMC" if mcmc else "least squares"}: χ²_red <b>{res.chi2_red:.2f}</b> '
                                f'({res.n_used} lines, {len(res.free)} free) · ' +
                                " · ".join(f"{k} <b>{_fmt(res.best[k])}</b>" for k in res.free) + "</div>")
            if then:
                then()
        self._start(work, done, "MCMC" if mcmc else "fitting", progress=mcmc)

    def run_all(self):
        try:
            if self.src.value != "table" and self.spec is None:
                self.load()
            self.find()
        except Exception as ex:
            self.info.object = f'<div class="sf-note">⚠ {ex}</div>'; return
        mc = self.run_mcmc_cb.value
        if self.src.value == "table":
            self.run_fit(mcmc=mc); return
        self.measure(then=lambda: self.run_fit(mcmc=mc))

    def compare(self):
        if self.features is None or not self.features.get("measured", pd.Series(False)).any():
            self.info.object = '<div class="sf-note">measure the lines first</div>'; return
        from .fit import compare_models
        from .pipeline import fit_config
        cfg = self.current_config()
        F, M = self.features, self.members
        fcfg = fit_config(cfg, self.dist.value)

        def done(t):
            self.comparison = t
            tt = t.copy()
            for k in ("chi2", "chi2_red", "bic", "aic", "dBIC"):
                if k in tt:
                    tt[k] = tt[k].map(lambda v: _fmt(v, ".2f"))
            self.compare_html.object = _df_html(tt) + ('<div class="sf-note" style="margin-top:6px">ΔBIC &gt; 6: strong preference for the '
                                                       'lower-BIC model; &lt; 2: no real preference.</div>')
            self.plot_tabs.active = 4
            self.info.object = '<div class="sf-kv">models compared — see <b>Posterior → Model comparison</b></div>'
        self._start(lambda: compare_models(F, M, fcfg, molecule=cfg.molecule), done, "comparing models")

    def write(self):
        from .pipeline import RotDiagResult, save_results
        if self.features is None:
            pn.state.notifications.warning("nothing to write yet") if pn.state.curdoc else None
            return
        cfg = self.current_config()
        res = RotDiagResult(cfg, self.spec, self.features, self.members, self.stamps, self.fit, self.comparison,
                            extra_fits=dict(self.fit_extra))
        out = save_results(res)
        self.info.object = f'<div class="sf-kv">results written to <span class="sf-mono">{out}</span></div>'
        if pn.state.curdoc is not None:
            pn.state.notifications.success(f"written to {out}")

    # ------------------------------------------------------------------ display
    def _colours(self, F):
        if F["spin"].astype(str).str.len().gt(0).any():
            return [SPIN_COL.get(s, PAL.teal) for s in F["spin"].astype(str)], {("ortho" if k == "o" else "para" if k == "p" else "o+p"): v
                                                                                  for k, v in SPIN_COL.items() if k in set(F["spin"])}
        lads = list(dict.fromkeys(F["ladder"].astype(str)))
        cmap = {k: LADDER_COLS[i % len(LADDER_COLS)] for i, k in enumerate(lads)}
        return [cmap[k] for k in F["ladder"].astype(str)], cmap

    @_bokeh_safe
    def show_features(self):
        F = self.features
        if F is None:
            return
        cols = [c for c in TABLE_COLS if c in F.columns]
        T = F[cols].copy()
        for k in ("flux", "flux_err"):
            if k in T:
                T[k] = T[k].map(lambda v: _fmt(v, ".3e"))
        for k, f_ in (("wave", ".5f"), ("eu", ".0f"), ("gA", ".3g"), ("snr", ".1f")):
            if k in T:
                T[k] = T[k].map(lambda v, f_=f_: _fmt(v, f_))
        self.table.value = T.reset_index(drop=True)
        if self.spec is not None:
            w, f, _ = self.spec.stitched()
            top = np.interp(F["wave"], w, np.nan_to_num(f)) if len(w) else np.zeros(len(F))
            span = np.nanpercentile(f, 99) - np.nanpercentile(f, 1) if len(f) else 1.0
            det = F["detected"].astype(bool).to_numpy() if "detected" in F else np.zeros(len(F), bool)
            meas = F["measured"].astype(bool).to_numpy() if "measured" in F else np.zeros(len(F), bool)
            use = F["use"].astype(bool).to_numpy() if "use" in F else np.ones(len(F), bool)
            col = np.where(~use, "#8e97ad", np.where(det, PAL.teal, np.where(meas, "#ffb347", PAL.accent)))
            status = np.where(~use, "excluded", np.where(det, "detected", np.where(meas, "not detected", "not measured")))
            self.mark_src.data = dict(x=F["wave"].to_numpy(), y0=top + 0.03 * span, y1=top + 0.18 * span, color=list(col),
                                      label=F["label"].astype(str).tolist(), eu=F["eu"].to_numpy(),
                                      snr=F["snr"].to_numpy() if "snr" in F else np.full(len(F), np.nan), status=list(status),
                                      id=F["id"].to_numpy())
        self.update_code()

    def _marker_tapped(self, idx):
        if idx and self.features is not None:
            self.show_line(int(self.mark_src.data["id"][idx[0]]))

    def _table_selected(self, e):
        if e.new and self.features is not None:
            self.show_line(int(self.features["id"].iloc[e.new[0]]))

    def _table_edited(self, e):
        if e.column == "use" and self.features is not None:
            self.features.loc[self.features.index[e.row], "use"] = bool(e.value)
            self.info.object = '<div class="sf-kv">line selection changed — press <b>③ Fit</b> again</div>'
            self.show_features(); self.show_diagram()

    @_bokeh_safe
    def show_line(self, fid: int):
        st = self.stamps.get(fid)
        r = self.features.set_index("id").loc[fid]
        if st is None:
            self.zoom_fig.title.text = f"{r['label']}: not measured"
            self.zoom_src.data = dict(x=[], d=[], m=[], own=[]); return
        d = st["data"] - st["base"]
        self.zoom_src.data = dict(x=st["wave"], d=d, m=st["model"] - st["base"], own=st["own"])
        lo, hi = np.nanmin(d), np.nanmax(d)
        L = np.atleast_1d(st["lines"]) * (st["center"] / r["wave"])
        self.zoom_lines.data = dict(x=L, y0=[lo] * len(L), y1=[hi] * len(L))
        self.zoom_fig.x_range.start, self.zoom_fig.x_range.end = float(st["wave"].min()), float(st["wave"].max())
        self.zoom_fig.title.text = (f"{r['label']}  λ {r['wave']:.5f} µm · E_u {r['eu']:.0f} K · F = {_fmt(r['flux'], '.3e')} ± "
                                    f"{_fmt(r['flux_err'], '.2e')} W m⁻² (S/N {_fmt(r['snr'], '.1f')}) · χ²_red {_fmt(r.get('chi2_red', np.nan), '.2f')}")

    @_bokeh_safe
    def show_diagram(self):
        from .fit import diagram_points, model_curves, model_points
        F = self.features
        if F is None or "flux" not in F or not np.isfinite(F["flux"].astype(float)).any():
            return
        cfg = self.current_config()
        from .pipeline import fit_config
        fc = fit_config(cfg, self.dist.value)
        geo = fc.geometry
        res = self.fit
        av = res.best.get("Av", 0.0) if res is not None else 0.0
        R = 10 ** res.best.get("logR", 0.0) if res is not None else geo.R_au
        F = F.copy(); F["flux"] = F["flux"].astype(float); F["flux_err"] = F["flux_err"].astype(float)
        pts = diagram_points(F, res.model.geometry if res is not None else geo, av, fc.extinction, R, cfg.measure.snr_detect)
        colours, legend = self._colours(F)
        colours = np.array(colours, dtype=object)
        det = pts["detected"].to_numpy() & pts["use"].to_numpy()
        ul = ~pts["detected"].to_numpy() & pts["use"].to_numpy()
        ex = ~pts["use"].to_numpy()
        snr = (F["flux"] / F["flux_err"]).to_numpy()
        y, ye = pts["y"].to_numpy(), pts["yerr"].to_numpy()
        self.pt_src.data = dict(x=pts["eu"][det], y=y[det], lo=(y - ye)[det], hi=(y + ye)[det], color=list(colours[det]),
                                label=pts["label"][det].tolist(), wave=pts["wave"][det], snr=snr[det])
        self.ul_src.data = dict(x=pts["eu"][ul], y=pts["y_ul"][ul], color=list(colours[ul]), label=pts["label"][ul].tolist(),
                                wave=pts["wave"][ul], snr=snr[ul])
        yex = np.where(pts["detected"], pts["y"], pts["y_ul"]).astype(float)
        self.ex_src.data = dict(x=pts["eu"][ex], y=yex[ex], label=pts["label"][ex].tolist(), wave=pts["wave"][ex], snr=snr[ex])
        unit = "cm⁻²" if geo.column else "molecules"
        self.diag_fig.yaxis.axis_label = f"ln(N_u / g_u) [{unit}]" + (f" · de-reddened A_V = {av:.1f}" if av else "")
        if res is not None:
            mc = model_curves(res)
            E = mc["E"][0]
            xs, ys, cc, xs2, ys2, cc2 = [], [], [], [], [], []
            for spin, curves in mc.items():
                if spin == "E":
                    continue
                c = SPIN_COL.get(spin, "#ffffff") if spin else PAL.pink
                xs.append(E); ys.append(curves[-1]); cc.append(c)
                if len(curves) > 2:
                    for cv in curves[:-1]:
                        xs2.append(E); ys2.append(cv); cc2.append(c)
            for k, ef in self.fit_extra.items():
                emc = model_curves(ef, E)
                sp0 = "p" if "p" in emc else [q for q in emc if q != "E"][0]
                xs.append(E); ys.append(emc[sp0][-1]); cc.append("#3ddc84")
            self.cv_src.data = dict(xs=xs, ys=ys, color=cc)
            self.cc_src.data = dict(xs=xs2, ys=ys2, color=cc2)
            ym = model_points(res)
            u = pts["use"].to_numpy()
            self.mp_src.data = dict(x=pts["eu"][u], y=ym[u], label=pts["label"][u].tolist())
            r = (y - ym)
            self.rs_src.data = dict(x=pts["eu"][det], r=r[det], lo=(r - ye)[det], hi=(r + ye)[det], color=list(colours[det]))
            t = res.table().set_index("param")
            self.diag_fig.title.text = (f"{get_molecule(cfg.molecule).label} · {res.model.kind} · " +
                                        ", ".join(f"{k} = {_fmt(t.loc[k, 'median'])}" for k in res.free) + f" · χ²_red {res.chi2_red:.2f}")
        else:
            for s in (self.cv_src, self.cc_src):
                s.data = dict(xs=[], ys=[], color=[])
            self.mp_src.data = dict(x=[], y=[], label=[]); self.rs_src.data = dict(x=[], r=[], lo=[], hi=[], color=[])
            self.diag_fig.title.text = f"{get_molecule(cfg.molecule).label} rotation diagram (not fitted yet)"
        allx = pts["eu"].to_numpy(float)
        if len(allx):
            self.diag_fig.x_range.start = 0.0
            self.diag_fig.x_range.end = float(np.nanmax(allx)) * 1.08 + 100
        chips = "".join(f'<span class="sf-chip"><span class="sf-dot" style="background:{c}"></span>{k}</span>' for k, c in legend.items())
        self.legend_html.object = (f'<div>{chips}<span class="sf-chip">▼ upper limit (S/N &lt; {cfg.measure.snr_detect:g})</span>'
                                   '<span class="sf-chip">× excluded</span><span class="sf-chip" style="color:#f5b543">□ model per line</span>'
                                   '<span class="sf-chip">— model (dashed: components)</span>'
                                   + ('<span class="sf-chip" style="color:#3ddc84">— power law</span>' if self.fit_extra else "") + '</div>')

    @_bokeh_safe
    def show_results(self):
        res = self.fit
        if res is None:
            return
        t = res.table()
        rows = []
        for r in t.itertuples():
            err = "" if not np.isfinite(r.err_lo) else f" −{_fmt(r.err_lo, '.3g')} +{_fmt(r.err_hi, '.3g')}"
            rows.append(dict(parameter=r.label or r.param, value=f"{_fmt(r.median)}{err}", fit="free" if r.free else "fixed"))
        info = res.mcmc_info
        extra = (f'<div class="sf-note" style="margin-top:6px">χ² {res.chi2:.1f} for {res.n_used} lines, {len(res.free)} free '
                 f'(χ²_red {res.chi2_red:.2f}, BIC {res.bic:.1f})')
        if info:
            extra += (f' · MCMC {info["walkers"]} walkers × {info["steps"]} steps (burn {info["burn"]}), acceptance '
                      f'{info["acceptance"]:.2f}, τ_max {_fmt(info.get("tau_max"), ".0f")} → '
                      f'{"converged" if info.get("converged") else "run longer (steps − burn < 30 τ)"}')
        self.params_html.object = _df_html(pd.DataFrame(rows)) + extra + "</div>"
        if res.derived is not None:
            D = res.derived
            self.derived_html.object = _df_html(pd.DataFrame([dict(quantity=r.quantity, value=f"{_fmt(r.value)}"
                                                                   + ("" if not np.isfinite(r.err_lo) else f" −{_fmt(r.err_lo, '.3g')} +{_fmt(r.err_hi, '.3g')}"),
                                                                   unit=r.unit) for r in D.itertuples()]))
        for k, ef in self.fit_extra.items():
            t2 = ef.table()
            rows2 = [dict(parameter=r.label or r.param, value=f"{_fmt(r.median)}" + ("" if not np.isfinite(r.err_lo) else
                          f" −{_fmt(r.err_lo, '.3g')} +{_fmt(r.err_hi, '.3g')}"), fit="free" if r.free else "fixed") for r in t2.itertuples()]
            self.params_html.object += (f'<div class="sf-label" style="margin-top:10px">{k} (χ²_red {ef.chi2_red:.2f}, BIC {ef.bic:.1f})</div>'
                                        + _df_html(pd.DataFrame(rows2)))
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        from .plots import plot_model_panels
        tex = {"H2": "H_2", "H2O": "H_2O", "13CO": "^{13}CO"}.get(self.mol.value, self.mol.value)
        try:
            figp = plot_model_panels(self.features, {res.model.kind: res, **self.fit_extra}, molecule_label=tex)
            self.paper.object = figp
            plt.close(figp)
        except Exception as ex:
            self.paper.object = None
            self.info.object += f'<div class="sf-note">paper figure failed: {ex}</div>'
        if res.samples is not None:
            import matplotlib
            matplotlib.use("Agg", force=False)
            import matplotlib.pyplot as plt
            from .plots import plot_corner
            fig = plot_corner(res)
            self.corner.object = fig
            if fig is not None:
                plt.close(fig)

    # ------------------------------------------------------------------ downloads + code
    def _cfg_bytes(self):
        return io.BytesIO(self.current_config().to_yaml().encode())

    def _lines_bytes(self):
        buf = io.StringIO()
        if self.features is not None:
            self.features.to_csv(buf, index=False)
        return io.BytesIO(buf.getvalue().encode())

    def _fit_bytes(self):
        import yaml
        from .pipeline import _clean
        d = _clean(self.fit.to_dict()) if self.fit is not None else {}
        return io.BytesIO(yaml.safe_dump(d, sort_keys=False, allow_unicode=True).encode())

    def update_code(self):
        try:
            cfg = self.current_config()
        except Exception as ex:
            self.code.object = f'<div class="sf-note">⚠ {ex}</div>'; return
        esc = lambda t: t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")  # noqa: E731
        f = cfg.fit
        if self.src.value == "table":
            inp = f'--fluxes {cfg.fluxes.path} --flux-unit "{cfg.fluxes.unit}"'
        elif self.src.value == "file":
            inp = cfg.spectrum.path
        else:
            inp = "SPECTRUM.csv   # the LTE-fit target / cube region: save it as CSV first"
        cli = (f"jalebi rotdiag fit {inp} --molecule {cfg.molecule} --model {f.model} --opr {f.opr}"
               + (" --av-free" if f.av_free else f" --av {f.av:g}") + f" --extinction {f.extinction}"
               + (f" --opacity --fwhm {f.fwhm_kms:g}" if f.opacity else "") + f" --geometry {cfg.geometry.mode}"
               + (f" --radius {cfg.geometry.R_au:g}" if cfg.geometry.mode == "radius" else "")
               + f" --distance {cfg.geometry.distance_pc:g}" + (" --mcmc" if cfg.mcmc.enabled else ""))
        py = ["from jalebi.rotdiag import RotDiagConfig, run_rotdiag",
              "cfg = RotDiagConfig.load('rotdiag.yaml')     # the ⬇ config YAML above",
              "res = run_rotdiag(cfg)                       # -> " + cfg.output,
              "print(res.fit.summary())"]
        self.code.object = (f'<div class="sf-note">terminal</div><pre class="sf-mono" style="white-space:pre-wrap;font-size:11px">{esc(cli)}</pre>'
                            f'<div class="sf-note">or: <span class="sf-mono">jalebi rotdiag run rotdiag.yaml</span> with the config above</div>'
                            f'<div class="sf-note">Python</div><pre class="sf-mono" style="white-space:pre-wrap;font-size:11px">{esc(chr(10).join(py))}</pre>')
