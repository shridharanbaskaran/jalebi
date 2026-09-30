"""The *Cube* workspace of the web app: line maps, velocity maps, channel maps, PV cuts and regions.

Left: the map (click a spaxel to see its spectrum; draw a polygon with the polygon tool) and sub-tabs for
the PV diagram and the region spectrum.  Right: the settings, the region, and the equivalent terminal
command and Python code for what is on screen, so a result found by clicking can be reproduced in a
script.  "Send to slab fit" loads the region spectrum as the target of the LTE slab-fit module (Continuum/Model/Fit);
"Send to rotation diagram" opens it in the Rotation diagram module (e.g. H2 S(1)-S(8) of a jet knot).
"""
from __future__ import annotations

import io
import os
import threading
import traceback

import numpy as np
import panel as pn
from bokeh.models import ColorBar, ColumnDataSource, EqHistColorMapper, HoverTool, LinearColorMapper, LogColorMapper, PolyDrawTool, Range1d, Span
from bokeh.palettes import Cividis256, Inferno256, Magma256, RdBu11, Turbo256, Viridis256

from ..app import PAL, PLOT_THEMES, _bokeh_safe, _fig, _html, _panel, _theme_fig
from ..lines import C_KMS

# Bokeh orders RdBu11 from blue to red: low (approaching) = blue, high (receding) = red
_DIVERGING = tuple(RdBu11) if RdBu11[0].lower() in ("#053061", "#2166ac") else tuple(reversed(RdBu11))
PRODUCTS = {"continuum": "continuum", "mom 0": "mom0", "masked": "mom0_masked", "extended": "mom0_ext", "velocity": "vcen",
            "σ(v)": "vcen_err", "mom 1": "mom1", "mom 2": "mom2", "S/N": "snr", "channel": "channel", "ratio": "ratio"}
# moment-0 windows: a velocity range, or native channels around the line as the old cube_maps.py
WINDOWS = {"± velocity": None, "full (9 ch)": "full", "slow (5 ch)": "slow", "fast (2+2 ch)": "fast"}
BANDS = {"widest margin": None, "nominal (cube_maps)": "nominal"}


class CubeWorkspace:
    def __init__(self, app, cube_path: str | None = None):
        from ..examples import DATA_DIR
        self.app = app
        default = "example:HV_Tau_C_cube" if (DATA_DIR / "HV_Tau_C_cube").is_dir() else ""
        self.cs = None; self.lc = None; self.lm = None
        self.region = None; self.region_spec = None; self.pvd = None; self.ratio = None; self.ratio_of = None
        self.made = None                      # (line text, prepare_line kw, line_maps kw) of the maps on screen
        self._opened = None                   # (path, dq_mask, zero_is_nan) of self.cs
        self._job = None; self._cb = None
        W = dict(sizing_mode="stretch_width")
        # ---- data + line ----------------------------------------------------------------------
        self.path = pn.widgets.TextInput(name="cube folder / file (or example:HV_Tau_C_cube)", value=cube_path or default, **W)
        self.open_btn = pn.widgets.Button(name="Open", width=80, margin=(22, 5, 5, 5))
        self.line = pn.widgets.Select(name="line", options=[], **W)
        self.custom = pn.widgets.TextInput(name="…or any line", placeholder="[Fe II] 17.94 · 12.8135 · CO P(10)=4.9876", **W)
        self.make_btn = pn.widgets.Button(name="▶  Make maps", button_type="primary", width=150, margin=(22, 5, 5, 5))
        self.product = pn.widgets.RadioButtonGroup(options=list(PRODUCTS), value="mom 0", button_type="default", **W)
        self.vel = pn.widgets.FloatSlider(name="channel velocity [km/s]", start=-400, end=400, step=10, value=0, visible=False, **W)
        self.info = _html('<div class="sf-note">Open a cube folder, pick a line, press <b>Make maps</b>.</div>', **W)
        # ---- settings ---------------------------------------------------------------------------
        self.rv = pn.widgets.FloatInput(name="systemic RV [km/s]", value=0.0, step=1, width=150)
        self.dist = pn.widgets.FloatInput(name="distance [pc]", value=140.0, step=5, width=150)
        self.window = pn.widgets.FloatInput(name="window ± [km/s]", value=1500.0, step=100, width=150)
        self.order = pn.widgets.IntInput(name="continuum order", value=1, start=0, end=4, width=150)
        self.psf = pn.widgets.Checkbox(name="remove the point source (continuum PSF)", value=True, margin=(8, 10, 2, 10))
        self.core = pn.widgets.FloatInput(name="PSF core [× FWHM]", value=0.5, step=0.1, width=150)
        self.snr = pn.widgets.FloatInput(name="S/N (moments)", value=3.0, step=0.5, width=150)
        self.kin_snr = pn.widgets.FloatInput(name="S/N (velocity fit)", value=5.0, step=0.5, width=150)
        self.n_mc = pn.widgets.IntInput(name="MC realisations", value=100, step=25, start=0, width=150)
        self.zero = pn.widgets.Select(name="velocity zero point", options=["none", "star"], value="star", width=150)
        self.smooth = pn.widgets.FloatInput(name="smoothing FWHM [spaxels]", value=0.0, step=0.5, width=150)
        self.cont_method = pn.widgets.Select(name="continuum", options=["poly", "median", "aspls", "irsqr", "asls"], value="poly", width=150)
        self.lam = pn.widgets.FloatInput(name="baseline λ (0 = default)", value=0.0, step=1e5, start=0, width=150)
        self.window_kind = pn.widgets.Select(name="moment-0 window", options=list(WINDOWS), value="± velocity", width=150)
        self.window_um = pn.widgets.FloatInput(name="window ± [µm] (0 = km/s)", value=0.0, step=0.05, start=0, width=150)
        self.band = pn.widgets.Select(name="cube per line", options=list(BANDS), value="widest margin", width=150)
        self.dq = pn.widgets.Checkbox(name="apply the DQ mask", value=True, margin=(8, 10, 2, 10))
        self.classic = pn.widgets.Checkbox(name="cube_maps.py NaN / zero rules (keep SCI = 0, drop spaxels with a NaN channel)",
                                           value=False, margin=(2, 10, 2, 10))
        self.recipe_btn = pn.widgets.Button(name="cube_maps.py recipe", width=150, margin=(22, 5, 5, 5))
        # ---- masks + ratio ------------------------------------------------------------------------
        self.rms_on = pn.widgets.Checkbox(name="mask mom 0 with an RMS circle (pixels)", value=False, margin=(8, 10, 2, 10))
        self.rms_x = pn.widgets.FloatInput(name="x", value=12.0, step=1, width=70)
        self.rms_y = pn.widgets.FloatInput(name="y", value=8.0, step=1, width=70)
        self.rms_r = pn.widgets.FloatInput(name="r", value=3.0, step=0.5, width=70)
        self.rms_sigma = pn.widgets.FloatInput(name="σ", value=3.0, step=0.5, width=70)
        self.rms_mode = pn.widgets.Select(name="level", options=["rms", "mean", "median"], value="rms", width=90)
        self.ratio_line = pn.widgets.Select(name="ratio: this line ÷", options=[], width=190)
        self.ratio_s1 = pn.widgets.FloatInput(name="σ₁", value=5.0, step=0.5, width=70)
        self.ratio_s2 = pn.widgets.FloatInput(name="σ₂", value=5.0, step=0.5, width=70)
        self.ratio_unit = pn.widgets.Select(name="map unit", options=["cgs", "MJy/sr m"], value="cgs", width=110)
        self.ratio_btn = pn.widgets.Button(name="Ratio map", width=110, margin=(22, 5, 5, 5))
        # ---- region -----------------------------------------------------------------------------
        self.shape = pn.widgets.Select(name="region", options=["circle", "ellipse", "annulus", "polygon"], value="circle", width=150)
        self.click_mode = pn.widgets.Select(name="a click on the map…", options=["shows the spectrum", "moves the region"],
                                            value="shows the spectrum", width=190)
        self.dx = pn.widgets.FloatInput(name="Δ east [″]", value=0.42, step=0.05, width=95)
        self.dy = pn.widgets.FloatInput(name="Δ north [″]", value=0.91, step=0.05, width=95)
        self.r = pn.widgets.FloatInput(name="r / a [″]", value=0.35, step=0.05, width=95)
        self.b = pn.widgets.FloatInput(name="b / r_in [″]", value=0.2, step=0.05, width=95)
        self.pa = pn.widgets.FloatInput(name="PA [°]", value=0.0, step=5, width=95)
        self.extract_btn = pn.widgets.Button(name="Extract region spectrum", button_type="primary", **W)
        self.send_btn = pn.widgets.Button(name="→ Send to slab fit", button_type="success", disabled=True, **W)
        self.send_rd_btn = pn.widgets.Button(name="→ Send to rotation diagram", button_type="default", disabled=True, **W)
        self.dl_spec = pn.widgets.FileDownload(callback=self._spec_bytes, filename="region_spectrum.csv", label="⬇ spectrum CSV", disabled=True, width=150)
        self.dl_ds9 = pn.widgets.FileDownload(callback=self._ds9_bytes, filename="region.reg", label="⬇ DS9 region", disabled=True, width=150)
        # ---- PV + output ------------------------------------------------------------------------
        self.pv_pa = pn.widgets.FloatInput(name="PV PA [°]", value=25.0, step=5, width=95)
        self.pv_len = pn.widgets.FloatInput(name="length [″]", value=5.0, step=0.5, width=95)
        self.pv_w = pn.widgets.FloatInput(name="width [″] (0=auto)", value=0.0, step=0.05, width=120)
        self.pv_btn = pn.widgets.Button(name="PV cut", width=90, margin=(22, 5, 5, 5))
        self.out = pn.widgets.TextInput(name="output folder ({target} = source name)", value="results/{target}/cube", **W)
        self.write_btn = pn.widgets.Button(name="Write FITS + PNG", **W)
        self.cfg_dl = pn.widgets.FileDownload(callback=self._cfg_bytes, filename="cube.yaml", label="⬇ cube config YAML", **W)
        self.code = _html("", **W)
        # ---- figures ----------------------------------------------------------------------------
        self._build_figures()
        # ---- wiring -----------------------------------------------------------------------------
        self.open_btn.on_click(lambda e: self.open())
        self.make_btn.on_click(lambda e: self.make_maps())
        self.product.param.watch(lambda e: self.show_product(), "value")
        self.vel.param.watch(lambda e: self.show_product(), "value")
        self.extract_btn.on_click(lambda e: self.extract_region())
        self.send_btn.on_click(lambda e: self.send_to_fit())
        self.send_rd_btn.on_click(lambda e: self.send_to_rotdiag())
        self.pv_btn.on_click(lambda e: self.make_pv())
        self.write_btn.on_click(lambda e: self.write_products())
        self.recipe_btn.on_click(lambda e: self.use_cube_maps_recipe())
        self.ratio_btn.on_click(lambda e: self.make_ratio())
        for w in (self.shape, self.dx, self.dy, self.r, self.b, self.pa):
            w.param.watch(lambda e: self.update_region(), "value")
        for w in (self.rv, self.dist, self.window, self.order, self.psf, self.core, self.snr, self.kin_snr, self.n_mc, self.zero,
                  self.smooth, self.line, self.custom, self.path, self.out, self.cont_method, self.lam, self.window_kind,
                  self.window_um, self.band, self.dq, self.classic, self.rms_on, self.rms_x, self.rms_y, self.rms_r, self.rms_sigma,
                  self.rms_mode, self.ratio_line, self.ratio_s1, self.ratio_s2, self.ratio_unit):
            w.param.watch(lambda e: self.update_code(), "value")
        # ---- layout -----------------------------------------------------------------------------
        self.plot_tabs = pn.Tabs(("Map", pn.Column(pn.pane.Bokeh(self.map_fig), pn.pane.Bokeh(self.spec_fig, **W), **W)),
                                 ("PV diagram", pn.pane.Bokeh(self.pv_fig, **W)),
                                 ("Region spectrum", pn.pane.Bokeh(self.full_fig, **W)),
                                 dynamic=False, css_classes=["sf-subtabs"], **W)
        left = pn.Column(
            _panel(pn.Row(self.path, self.open_btn, **W), pn.Row(self.line, self.custom, self.make_btn, **W), self.info, title="Cube"),
            _panel(self.product, self.vel, self.plot_tabs, title="Maps"),
            **W)
        settings = _panel(
            pn.Row(self.rv, self.dist), pn.Row(self.window, self.order), self.psf, pn.Row(self.core, self.smooth),
            pn.Row(self.snr, self.kin_snr), pn.Row(self.n_mc, self.zero),
            pn.Row(self.cont_method, self.lam), pn.Row(self.window_kind, self.window_um), pn.Row(self.band, self.recipe_btn), self.dq,
            self.classic,
            _html('<div class="sf-note">Continuum: polynomial through the line-free channels of every spaxel (or a '
                  'pybaselines baseline through all channels: aspls = the old cube_maps.py, irsqr = the channel_maps '
                  'notebook). Point source: the continuum image next to the line, scaled in the PSF core, removed plane '
                  'by plane (companions found automatically). Moment 0 over |v| ≤ max(200 km/s, 1.5 FWHM), or native '
                  'channels around the line (full / slow / fast as cube_maps.py). Velocities: Gaussian centroids with '
                  'Monte Carlo errors; the MRS resolves 85–200 km/s, so trust centroid shifts, not widths. "star" puts '
                  'v = 0 at the source for each line. <b>cube_maps.py recipe</b>: ±0.1 µm, aspls, 9 channels, nominal '
                  'sub-bands, no DQ mask, no PSF removal.</div>'),
            title="Settings")
        masks = _panel(self.rms_on, pn.Row(self.rms_x, self.rms_y, self.rms_r, self.rms_sigma, self.rms_mode),
                       pn.Row(self.ratio_line, self.ratio_unit), pn.Row(self.ratio_s1, self.ratio_s2, self.ratio_btn),
                       _html('<div class="sf-note">RMS circle in spaxels of this line\'s cube (x = column, y = row, from 0): '
                             '"masked" shows mom 0 ≥ σ × its level. Ratio: the other line is mapped with the same '
                             'settings, reprojected onto these spaxels, both masked at σ × noise (the RMS circle when '
                             'ticked, else the propagated errors); "cgs" = line-flux ratio, "MJy/sr m" = the numbers of '
                             'cube_maps.py\'s maps (flux ratio × (λ₁/λ₂)²).</div>'),
                       title="Masks and line ratio")
        region = _panel(pn.Row(self.shape, self.click_mode), pn.Row(self.dx, self.dy, self.r), pn.Row(self.b, self.pa),
                        _html('<div class="sf-note">Offsets from the source. Ellipse: r = semi-major, b = semi-minor, PA of the '
                              'major axis (° E of N). Annulus: b = r_in, r = r_out. Polygon: pick the polygon tool on the map '
                              'and click the vertices (double-click ends).</div>'),
                        self.extract_btn, self.send_btn, self.send_rd_btn, pn.Row(self.dl_spec, self.dl_ds9),
                        title="Region → spectrum → slab fit / rotation diagram")
        pvp = _panel(pn.Row(self.pv_pa, self.pv_len, self.pv_w, self.pv_btn), title="PV cut")
        outp = _panel(self.out, self.write_btn, self.cfg_dl, title="Save")
        codep = _panel(self.code, title="Same thing in the terminal / Python")
        inspector = pn.Column(settings, masks, region, pvp, outp, codep, width=400, scroll=True, css_classes=["sf-inspector"],
                              styles={"max-height": "calc(100vh - 150px)"})
        self.panel = pn.Row(left, inspector, **W)
        if self.path.value:
            try:
                self.open()
            except Exception as ex:
                self.info.object = f'<div class="sf-note">⚠ {ex}</div>'
        self.update_code()

    # ------------------------------------------------------------------ figures
    def figures(self):
        return [self.map_fig, self.spec_fig, self.full_fig, self.pv_fig]

    def _build_figures(self):
        th = PLOT_THEMES["dark"]
        self.map_fig = _fig("", height=460, y_label="ΔDec [″]", x_label="ΔRA [″]",
                            tools="pan,wheel_zoom,box_zoom,reset,save,tap")
        # square frame + equal ranges = equal arcsec per screen pixel on both axes
        self.map_fig.sizing_mode = "fixed"
        self.map_fig.frame_width = 470; self.map_fig.frame_height = 470
        self.map_fig.x_range = Range1d(4, -4); self.map_fig.y_range = Range1d(-4, 4)
        self.img_src = ColumnDataSource(dict(image=[np.zeros((2, 2))], x=[0], y=[0], dw=[1], dh=[1]))
        # one image renderer + colour bar per mapper type; switching products toggles their visibility and
        # updates the mapper's palette / limits in place (replacing a mapper model does not reach the browser)
        self.mappers = {"linear": LinearColorMapper(palette=Inferno256, nan_color=th["bg"]),
                        "log": LogColorMapper(palette=Viridis256, nan_color=th["bg"]),
                        "eqhist": EqHistColorMapper(palette=Inferno256, nan_color=th["bg"])}
        self.imgs, self.cbars = {}, {}
        for k, mp in self.mappers.items():
            self.imgs[k] = self.map_fig.image("image", "x", "y", "dw", "dh", source=self.img_src, color_mapper=mp, visible=(k == "eqhist"))
            cb = ColorBar(color_mapper=mp, width=10, label_standoff=6, background_fill_alpha=0.0, visible=(k == "eqhist"),
                          major_label_text_color=th["muted"], major_label_text_font_size="10px")
            self.map_fig.add_layout(cb, "right")
            self.cbars[k] = cb
        self.img = self.imgs["eqhist"]
        self.map_fig.add_tools(HoverTool(renderers=list(self.imgs.values()),
                                         tooltips=[("ΔRA", "$x{0.00}″"), ("ΔDec", "$y{0.00}″"), ("value", "@image{0.000e}")]))
        self.star_src = ColumnDataSource(dict(x=[0.0], y=[0.0]))
        self.map_fig.scatter("x", "y", source=self.star_src, marker="star", size=14, color="#ffe066", line_color="black")
        self.comp_src = ColumnDataSource(dict(x=[], y=[]))
        self.map_fig.scatter("x", "y", source=self.comp_src, marker="circle_x", size=12, fill_alpha=0, line_color="#ffe066")
        self.reg_src = ColumnDataSource(dict(x=[], y=[]))
        self.map_fig.line("x", "y", source=self.reg_src, color=PAL.teal, line_width=2)
        self.poly_src = ColumnDataSource(dict(xs=[], ys=[]))
        pr = self.map_fig.patches("xs", "ys", source=self.poly_src, fill_alpha=0.1, fill_color=PAL.teal, line_color=PAL.teal, line_width=2)
        self.map_fig.add_tools(PolyDrawTool(renderers=[pr], num_objects=1))
        self.poly_src.on_change("data", lambda a, o, n: self._poly_changed())
        self.pick_src = ColumnDataSource(dict(x=[], y=[]))
        self.map_fig.scatter("x", "y", source=self.pick_src, marker="square", size=10, fill_alpha=0, line_color=PAL.pink, line_width=2)
        self.map_fig.on_event("tap", self._tap)
        # spaxel / region line spectrum (velocity axis)
        self.spec_fig = _fig("click a spaxel", height=260, y_label="MJy sr⁻¹", x_label="velocity [km/s]",
                             tools="xpan,xwheel_zoom,box_zoom,reset,save")
        self.sp_src = ColumnDataSource(dict(v=[], line=[], psf=[], ext=[], fit=[]))
        self.spec_fig.step("v", "line", source=self.sp_src, mode="center", color=th["data"], legend_label="continuum-subtracted")
        self.spec_fig.step("v", "psf", source=self.sp_src, mode="center", color=PAL.accent, legend_label="point source")
        self.spec_fig.step("v", "ext", source=self.sp_src, mode="center", color=PAL.teal, legend_label="extended")
        self.spec_fig.line("v", "fit", source=self.sp_src, color=PAL.pink, line_width=2, legend_label="Gaussian fit")
        self.spec_fig.add_layout(Span(location=0, dimension="height", line_color=th["muted"], line_dash="dotted"))
        self.spec_fig.legend.location = "top_right"; self.spec_fig.legend.label_text_font_size = "10px"
        self.spec_fig.legend.background_fill_alpha = 0.3
        # full region spectrum
        self.full_fig = _fig("region spectrum (all sub-bands)", height=420)
        self.full_src = ColumnDataSource(dict(w=[], f=[]))
        self.full_fig.line("w", "f", source=self.full_src, color=th["data"], line_width=1.2)
        # PV
        self.pv_fig = _fig("PV diagram", height=420, y_label="velocity [km/s]", x_label="offset [″]", tools="pan,wheel_zoom,box_zoom,reset,save")
        self.pv_src = ColumnDataSource(dict(image=[np.zeros((2, 2))], x=[0], y=[0], dw=[1], dh=[1]))
        self.pv_mapper = EqHistColorMapper(palette=Inferno256, nan_color=th["bg"])
        self.pv_fig.image("image", "x", "y", "dw", "dh", source=self.pv_src, color_mapper=self.pv_mapper)
        for f in self.figures():
            _theme_fig(f, th)

    # ------------------------------------------------------------------ settings -> objects
    def current_line(self):
        txt = self.custom.value.strip()
        return txt if txt else self.line.value

    def _continuum(self) -> dict:
        c = {"order": int(self.order.value), "method": self.cont_method.value}
        if self.lam.value:
            c["lam"] = float(self.lam.value)
        if self.classic.value:
            c["nan_policy"] = "propagate"
        return c

    def _read_kw(self) -> dict:
        return {"dq_mask": bool(self.dq.value), "zero_is_nan": not self.classic.value}

    def _rms_region(self):
        return (float(self.rms_x.value), float(self.rms_y.value), float(self.rms_r.value)) if self.rms_on.value else None

    def settings(self) -> dict:
        return dict(rv_kms=self.rv.value, window_kms=self.window.value, continuum=self._continuum(),
                    psf={"enabled": bool(self.psf.value), "core_radius_fwhm": float(self.core.value)},
                    smooth_fwhm_pix=float(self.smooth.value), window_um=float(self.window_um.value) or None,
                    band=BANDS[self.band.value])

    def map_settings(self) -> dict:
        return dict(snr_min=float(self.snr.value), kin_snr_min=float(self.kin_snr.value), n_mc=int(self.n_mc.value),
                    zero_point=self.zero.value, component=WINDOWS[self.window_kind.value], rms_region=self._rms_region(),
                    rms_sigma=float(self.rms_sigma.value), rms_mode=self.rms_mode.value,
                    min_valid=1 if self.classic.value else None)

    def use_cube_maps_recipe(self):
        """Settings of the old cube_maps.py: +-0.1 um, aspls (lam 5e6), 9-channel moment 0, nominal sub-bands,
        no DQ mask, no point-source removal."""
        self.cont_method.value = "aspls"; self.lam.value = 0.0
        self.window_kind.value = "full (9 ch)"; self.window_um.value = 0.1
        self.band.value = "nominal (cube_maps)"; self.dq.value = False; self.psf.value = False; self.classic.value = True
        self.update_code()

    def cube_config(self):
        from .config import CubeConfig
        line = self.made[0] if (self.made and self.lm is not None) else self.current_line()
        cfg = CubeConfig(path=self.path.value, distance_pc=self.dist.value, rv_kms=self.rv.value, window_kms=self.window.value,
                         smooth_fwhm_pix=self.smooth.value, output=self.out.value,
                         lines=[] if (line or "").startswith("stack:") else [line or "[Ne II] 12.81"])
        if (line or "").startswith("stack:"):
            cfg.stacks = {"stack": [x.strip() for x in line[6:].split("+")]}
        cfg.continuum.order = int(self.order.value); cfg.continuum.method = self.cont_method.value
        cfg.continuum.lam = float(self.lam.value) or None
        cfg.window_um = float(self.window_um.value) or None; cfg.band = BANDS[self.band.value]; cfg.dq_mask = bool(self.dq.value)
        cfg.zero_is_nan = not self.classic.value
        cfg.continuum.nan_policy = "propagate" if self.classic.value else "omit"
        cfg.moments.min_valid = 1 if self.classic.value else None
        cfg.psf.enabled = bool(self.psf.value); cfg.psf.core_radius_fwhm = float(self.core.value)
        cfg.moments.snr_min = float(self.snr.value)
        cfg.moments.component = WINDOWS[self.window_kind.value]
        rr = self._rms_region()
        cfg.moments.rms_region = list(rr) if rr else None
        cfg.moments.rms_sigma = float(self.rms_sigma.value); cfg.moments.rms_mode = self.rms_mode.value
        if self.ratio is not None and self.ratio_of is self.lm:
            from .config import CubeRatioConfig
            rm = self.ratio
            cfg.ratios = [CubeRatioConfig(lines=list(rm.meta.get("lines", rm.names)), sigma_thresh=list(rm.sigma_thresh),
                                          rms_region=list(rm.rms_region) if rm.rms_region else None, rms_mode=rm.rms_mode,
                                          unit=rm.meta.get("unit_arg", "cgs"))]
        cfg.kinematics.snr_min = float(self.kin_snr.value); cfg.kinematics.n_mc = int(self.n_mc.value)
        cfg.kinematics.zero_point = self.zero.value
        if self.region is not None:
            from .config import CubeRegionConfig
            cfg.regions = [CubeRegionConfig(name="region", ds9=self.region.to_ds9())]
        return cfg

    def _cfg_bytes(self):
        return io.BytesIO(self.cube_config().to_yaml().encode())

    # ------------------------------------------------------------------ actions
    @_bokeh_safe
    def open(self):
        from .io import CubeSet
        rk = self._read_kw()
        new_target = self._opened is None or self._opened[0] != self.path.value
        self.cs = CubeSet(self.path.value, **rk)
        self._opened = (self.path.value, rk["dq_mask"], rk["zero_is_nan"])
        if new_target:                       # maps, ratio and PV of another target must not be mixed with this one
            self.lm = None; self.lc = None; self.ratio = None; self.ratio_of = None; self.pvd = None; self.made = None
        cov = self.cs.lines_covered()
        opts = [ln.name for ln in cov]
        h2 = [ln.name for ln in cov if ln.species == "H2"]
        if len(h2) >= 2:
            opts.append("stack: " + " + ".join(h2))
        keep, keep_r = self.line.value, self.ratio_line.value
        self.line.options = opts
        if opts:
            self.line.value = keep if keep in opts else next((o for o in opts if "Fe II" in o), opts[0])
        singles = [o for o in opts if not o.startswith("stack:")]
        self.ratio_line.options = singles
        if singles:
            self.ratio_line.value = keep_r if keep_r in singles else next((o for o in singles if "Ne II" in o), singles[-1])
        c0 = self.cs.load(self.cs.info[0].path)
        ra, dec, _, _ = c0.find_source()
        self.center = (ra, dec)
        self.info.object = (f'<div class="sf-kv"><b>{self.cs.name}</b> · {len(self.cs.info)} cubes '
                            f'({", ".join(self.cs.bands)}) · source RA {ra:.6f} Dec {dec:.6f} · '
                            f'{len(cov)} catalogue lines covered</div>')
        self.update_code()

    def _start(self, work, done, label):
        if self._job is not None and self._job["thread"].is_alive():
            pn.state.notifications.warning("still working on the previous request"); return
        job = {"result": None, "error": None}

        def run():
            try:
                job["result"] = work()
            except Exception as ex:
                job["error"] = f"{ex}\n{traceback.format_exc(limit=3)}"
        job["thread"] = threading.Thread(target=run, daemon=True)
        self._job = job
        self.info.object = f'<div class="sf-kv">⏳ {label} …</div>'
        job["thread"].start()

        def poll():
            if job["thread"].is_alive():
                return
            self._cb.stop(); self._cb = None
            if job["error"]:
                self.info.object = f'<div class="sf-note">⚠ {label} failed: {job["error"].splitlines()[0]}</div>'
                pn.state.notifications.error(f"{label} failed: {job['error'].splitlines()[0]}")
            else:
                done(job["result"])
        if pn.state.curdoc is None:           # not served (tests, notebooks without a server): run inline
            job["thread"].join(); poll_inline = True
        else:
            poll_inline = False
        if poll_inline:
            if job["error"]:
                raise RuntimeError(job["error"])
            done(job["result"])
        else:
            self._cb = pn.state.add_periodic_callback(_bokeh_safe_fn(poll), period=300)

    def make_maps(self):
        line = self.current_line()                    # before a reopen, which may reset the selection
        rk = self._read_kw()
        if self.cs is None or self._opened != (self.path.value, rk["dq_mask"], rk["zero_is_nan"]):
            self.open()
        from .maps import line_maps, prepare_line, stack_lines
        kw = self.settings(); mk = self.map_settings()
        self._pending = (line, kw, dict(mk))

        def work():
            if line.startswith("stack:"):
                lc = stack_lines(self.cs, [x.strip() for x in line[6:].split("+")], name="stack", **kw)
            else:
                lc = prepare_line(self.cs, line, **kw)
            return line_maps(lc, **mk)
        self._start(work, self._maps_done, f"maps of {line}")

    @_bokeh_safe
    def _maps_done(self, lm):
        self.lm = lm; self.lc = lm.lc
        self.made = getattr(self, "_pending", None)
        self.pvd = None                      # a PV cut belongs to the maps it was made from
        s = lm.summary
        v = self.lc.vel
        step = float(np.median(np.abs(np.diff(v))))
        self.vel.start = float(max(v.min(), -600)); self.vel.end = float(min(v.max(), 600)); self.vel.step = round(step, 1)
        fl = s.get("total_flux_W_m2_snr_masked")
        txt = (f'<div class="sf-kv"><b>{s["line"]}</b> ({s["band"]}, {s["rest_um"]:.4f} µm) · PSF {s["psf_fwhm_arcsec"]:.2f}″ · '
               f'instr. FWHM {s["lsf_fwhm_kms"]:.0f} km/s')
        if fl is not None:
            txt += f' · total {fl:.2e} W m⁻²'
        if s.get("point_source_line_flux_W_m2") is not None:
            txt += f' · point source {s["point_source_line_flux_W_m2"]:.2e} · extended {s["extended_flux_W_m2"]:.2e}'
        if np.isfinite(s.get("source_velocity_kms") or np.nan):
            txt += f' · v(source) {s["source_velocity_kms"]:+.1f} ± {s["source_velocity_err_kms"]:.1f} km/s'
        comps = self.lc.settings.get("companions_radec") or []
        if comps:
            txt += f' · {len(comps)} companion(s) removed too'
        self.info.object = txt + "</div>"
        from .regions import _tangent
        c0 = self.lc.center_radec
        if comps:
            dx, dy = _tangent([c[0] for c in comps], [c[1] for c in comps], *c0)
            self.comp_src.data = dict(x=list(np.atleast_1d(dx)), y=list(np.atleast_1d(dy)))
        else:
            self.comp_src.data = dict(x=[], y=[])
        self.show_product()
        self.update_region()
        self.update_code()

    # ------------------------------------------------------------------ map display
    def _geometry(self):
        """Source pixel position, spaxel size and grid size of the current cube."""
        cube = self.lc.cube
        ny, nx = cube.shape[1:]
        xs, ys = cube.world_to_pix(*self.lc.center_radec)
        ps = cube.pixscale
        return float(xs), float(ys), ps, nx, ny

    def _north_up_image(self, img):
        """A rotated cube's map resampled (nearest spaxel) onto a north-up grid of offsets from the source, so
        the map, the markers, the regions and the clicks all share one coordinate system."""
        from scipy.ndimage import map_coordinates
        from .regions import _from_tangent
        xs, ys, ps, nx, ny = self._geometry()
        cube = self.lc.cube
        yy, xx = np.mgrid[-0.5:ny, -0.5:nx]
        ex, ey = cube.offsets(self.lc.center_radec, xx, yy)
        half = float(max(np.nanmax(np.abs(ex)), np.nanmax(np.abs(ey))))
        n = int(np.ceil(2 * half / ps))
        e = half - (np.arange(n) + 0.5) * ps                   # east offsets, decreasing (column 0 = east edge)
        nn = -half + (np.arange(n) + 0.5) * ps                 # north offsets, increasing
        E, N = np.meshgrid(e[::-1], nn)                        # display columns run west -> east
        ra, dec = _from_tangent(E, N, *self.lc.center_radec)
        px, py = cube.world_to_pix(ra, dec)
        out = map_coordinates(np.asarray(img, float), [py, px], order=0, mode="constant", cval=np.nan)
        return out, -half, -half, n * ps

    @_bokeh_safe
    def show_product(self):
        if self.lm is None:
            return
        key = PRODUCTS[self.product.value]
        self.vel.visible = key == "channel"
        th = PLOT_THEMES[self.app.plot_theme]
        m = self.lm.maps
        if key == "channel":
            k = int(np.argmin(np.abs(self.lc.vel - self.vel.value)))
            img = self.lc.extended[k]; unit = "MJy/sr"
            label = f"channel {self.lc.vel[k]:+.0f} km/s ({'extended' if self.lc.psf is not None else 'line'})"
        elif key == "ratio":
            if self.ratio is not None and self.ratio_of is self.lm:
                img = self.ratio.ratio; unit = ""; label = f"{self.ratio.names[0]} / {self.ratio.names[1]}"
            else:
                img = np.full(self.lc.data.shape[1:], np.nan); unit = ""; label = "ratio: press “Ratio map”"
        elif key == "mom0_masked" and key not in m:
            img = np.full(self.lc.data.shape[1:], np.nan); unit = ""; label = "masked: tick the RMS circle and make the maps"
        elif key not in m:
            img = np.full(self.lc.data.shape[1:], np.nan); unit = ""; label = f"{self.product.value}: not available"
        else:
            img = m[key]; unit = self.lm.units.get(key, ""); label = self.product.value
            if key in ("mom0", "mom0_ext"):
                snr = m["snr_ext" if key == "mom0_ext" else "snr"]
                img = np.where(snr >= 2, img, np.nan)
        img = np.asarray(img, float)
        fin = img[np.isfinite(img)]
        if key in ("vcen", "mom1"):
            from .plots import velocity_limit
            lim = velocity_limit(img, m.get("vcen_err") if key == "vcen" else None) or 1.0
            kind, pal, lo, hi = "linear", _DIVERGING, -max(lim, 1e-3), max(lim, 1e-3)
        elif key == "continuum":
            lo = float(np.nanpercentile(fin[fin > 0], 1)) if (fin > 0).any() else 1e-3
            kind, pal, hi = "log", Viridis256, (float(np.nanmax(fin)) if fin.size else 1.0)
        elif key in ("vcen_err", "mom2"):
            kind, pal = "linear", Cividis256
            lo = float(np.nanpercentile(fin, 2)) if fin.size else 0.0
            hi = float(np.nanpercentile(fin, 98)) if fin.size else 1.0
        elif key == "snr":
            kind, pal, lo = "linear", Magma256, 0.0
            hi = float(np.nanpercentile(fin, 99)) if fin.size else 1.0
        elif key == "ratio":
            # the limits of cube_maps.py's ratio panel: PercentileInterval(80)
            kind, pal = "linear", Turbo256
            lo = float(np.nanpercentile(fin, 10)) if fin.size else 0.0
            hi = float(np.nanpercentile(fin, 90)) if fin.size else 1.0
        else:
            kind, pal, lo, hi = "eqhist", Inferno256, None, None
        mp = self.mappers[kind]
        mp.palette = list(pal)
        if lo is not None:
            mp.low = lo; mp.high = hi if hi > lo else lo + 1e-3
        mp.nan_color = th["bg"]
        for k in self.imgs:
            self.imgs[k].visible = k == kind
            self.cbars[k].visible = k == kind
        self.img = self.imgs[kind]
        xs, ys, ps, nx, ny = self._geometry()
        if self.lc.cube.north_up:
            disp = img[:, ::-1]                          # column 0 of the display is the western-most spaxel
            x0 = -(nx - 1 - xs + 0.5) * ps; y0 = -(ys + 0.5) * ps
            self.img_src.data = dict(image=[disp], x=[x0], y=[y0], dw=[nx * ps], dh=[ny * ps])
            half = max(nx, ny) * ps / 2 + 0.2
            cx = -((nx - 1) / 2 - xs) * ps; cy = ((ny - 1) / 2 - ys) * ps
        else:
            disp, x0, y0, size = self._north_up_image(img)
            self.img_src.data = dict(image=[disp], x=[x0], y=[y0], dw=[size], dh=[size])
            half = size / 2 + 0.2; cx = cy = 0.0
        self.map_fig.x_range.start = cx + half; self.map_fig.x_range.end = cx - half
        self.map_fig.y_range.start = cy - half; self.map_fig.y_range.end = cy + half
        self.map_fig.xaxis.axis_label = "ΔRA [″]"; self.map_fig.yaxis.axis_label = "ΔDec [″]"
        self.map_fig.title.text = f"{self.lc.cube.name} — {self.lc.line.name}: {label}" + (f"  [{unit}]" if unit else "")

    def _offset_to_pix(self, ex, ny_):
        from .regions import _from_tangent
        ra, dec = _from_tangent(ex, ny_, *self.lc.center_radec)
        x, y = self.lc.cube.world_to_pix(ra, dec)
        return float(x), float(y)

    def _tap(self, event):
        if self.lc is None:
            return
        if self.click_mode.value == "moves the region":
            self.dx.value = round(float(event.x), 3); self.dy.value = round(float(event.y), 3)
            return
        x, y = self._offset_to_pix(float(event.x), float(event.y))
        self.show_spaxel(int(round(x)), int(round(y)), (float(event.x), float(event.y)))

    @_bokeh_safe
    def show_spaxel(self, ix, iy, marker=None):
        lc = self.lc
        ny, nx = lc.data.shape[1:]
        if not (0 <= ix < nx and 0 <= iy < ny):
            return
        line = lc.line_data[:, iy, ix]
        psf = lc.psf.model[:, iy, ix] if lc.psf is not None else np.full_like(line, np.nan)
        ext = lc.extended[:, iy, ix] if lc.psf is not None else np.full_like(line, np.nan)
        fit = np.full_like(line, np.nan)
        m = self.lm.maps if self.lm else {}
        if "vcen" in m and np.isfinite(m["vcen"][iy, ix]):
            v0 = m["vcen"][iy, ix]           # lc.vel is in the same (zero-point) frame as vcen
            sig = m["fwhm"][iy, ix] / 2.3548
            area_cgs = m["gflux"][iy, ix]
            from ..lines import area_to_cgs_sr
            area = area_cgs / float(area_to_cgs_sr(1.0, lc.line.wave))          # MJy/sr um
            sig_um = sig / C_KMS * lc.line.wave
            wv = lc.line.wave * (1 + lc.vel / C_KMS)
            mu = lc.line.wave * (1 + v0 / C_KMS)
            fit = area / (np.sqrt(2 * np.pi) * sig_um) * np.exp(-0.5 * ((wv - mu) / sig_um) ** 2)
        self.sp_src.data = dict(v=lc.vel, line=line, psf=psf, ext=ext, fit=fit)
        vtxt = ""
        if "vcen" in m and np.isfinite(m["vcen"][iy, ix]):
            vtxt = f" · v = {m['vcen'][iy, ix]:+.1f} ± {m['vcen_err'][iy, ix]:.1f} km/s"
        self.spec_fig.title.text = f"spaxel ({ix}, {iy}) · S/N {m.get('snr', np.full((ny, nx), np.nan))[iy, ix]:.1f}{vtxt}"
        if marker is not None:
            self.pick_src.data = dict(x=[marker[0]], y=[marker[1]])

    # ------------------------------------------------------------------ regions
    def _poly_changed(self):
        d = self.poly_src.data
        if d.get("xs") and len(d["xs"]) and len(d["xs"][-1]) >= 3:
            self.shape.value = "polygon"
            self.update_region()

    def build_region(self):
        from .regions import offset_region
        c0 = self.lc.center_radec if self.lc is not None else getattr(self, "center", None)
        if c0 is None:
            return None
        sh = self.shape.value
        if sh == "circle":
            return offset_region("circle", c0, self.dx.value, self.dy.value, self.r.value)
        if sh == "ellipse":
            return offset_region("ellipse", c0, self.dx.value, self.dy.value, self.r.value, max(self.b.value, 0.01), self.pa.value)
        if sh == "annulus":
            return offset_region("annulus", c0, self.dx.value, self.dy.value, min(self.b.value, self.r.value), max(self.b.value, self.r.value))
        d = self.poly_src.data
        if not d.get("xs") or len(d["xs"][-1]) < 3:
            return None
        xy = np.column_stack([d["xs"][-1], d["ys"][-1]]).ravel()
        return offset_region("polygon", c0, *xy)

    @_bokeh_safe
    def update_region(self):
        try:
            reg = self.build_region()
        except Exception:
            reg = None
        self.region = reg
        if reg is None:
            self.reg_src.data = dict(x=[], y=[])
        else:
            from .regions import _tangent
            ra, dec = reg.outline()
            dx, dy = _tangent(ra, dec, *(self.lc.center_radec if self.lc is not None else self.center))
            self.reg_src.data = dict(x=list(dx), y=list(dy))
        self.update_code()

    def extract_region(self):
        if self.cs is None:
            self.open()
        self.update_region()
        if self.region is None:
            pn.state.notifications.warning("define a region first (make maps to set the source position)"); return
        from .regions import region_spectrum
        reg = self.region; cs = self.cs; d = self.dist.value
        self._start(lambda: region_spectrum(cs, reg, name=f"{cs.name} {reg.kind}", distance_pc=d), self._region_done, "region spectrum")

    @_bokeh_safe
    def _region_done(self, spec):
        self.region_spec = spec
        w, f, _ = spec.stitched()
        self.full_src.data = dict(w=w, f=f)
        self.full_fig.title.text = f"{spec.name}: {self.region.to_ds9()} · {len(spec.wave)} pixels in {', '.join(spec.bands)}"
        self.send_btn.disabled = False; self.send_rd_btn.disabled = False; self.dl_spec.disabled = False; self.dl_ds9.disabled = False
        self.plot_tabs.active = 2
        self.info.object = ('<div class="sf-kv">region spectrum ready — <b>Send to slab fit</b> loads it as the LTE-fit target, '
                            '<b>Send to rotation diagram</b> opens it in the Rotation diagram module</div>')

    def send_to_fit(self):
        if self.region_spec is None:
            return
        self.app.use_spectrum(self.region_spec, label=self.region_spec.name)
        pn.state.notifications.success("region spectrum loaded as the target: set the continuum, then Model / Fit")

    def send_to_rotdiag(self):
        if self.region_spec is None:
            return
        if hasattr(self.app, "send_spectrum"):
            self.app.send_spectrum(self.region_spec, "rotdiag", label=self.region_spec.name)
            pn.state.notifications.success("region spectrum sent to the Rotation diagram module: pick the molecule, Find lines")

    def _spec_bytes(self):
        buf = io.StringIO()
        if self.region_spec is not None:
            df = self.region_spec.to_dataframe()
            buf.write(f"# name={self.region_spec.name} distance_pc={self.region_spec.distance_pc} rv_kms=0 rest_frame=False\n")
            df.to_csv(buf, index=False)
        return io.BytesIO(buf.getvalue().encode())

    def _ds9_bytes(self):
        txt = "# Region file format: DS9 version 4.1 (jalebi)\nfk5\n" + (self.region.to_ds9() + "\n" if self.region else "")
        return io.BytesIO(txt.encode())

    # ------------------------------------------------------------------ PV + output
    def make_pv(self):
        if self.lc is None:
            pn.state.notifications.warning("make maps first"); return
        from .channels import pv_diagram
        lc = self.lc
        pa, L, w = self.pv_pa.value, self.pv_len.value, (self.pv_w.value or None)
        self._start(lambda: pv_diagram(lc, pa, L, w, vmax_kms=600.0), self._pv_done, "PV cut")

    @_bokeh_safe
    def _pv_done(self, pv):
        self.pvd = pv
        ds = float(np.diff(pv.offset).mean()); dv = float(np.diff(pv.velocity).mean())
        self.pv_src.data = dict(image=[pv.data], x=[pv.offset[0] - ds / 2], y=[pv.velocity[0] - dv / 2],
                                dw=[len(pv.offset) * ds], dh=[len(pv.velocity) * dv])
        self.pv_fig.x_range.start = float(pv.offset[0]); self.pv_fig.x_range.end = float(pv.offset[-1])
        self.pv_fig.y_range.start = float(pv.velocity[0]); self.pv_fig.y_range.end = float(pv.velocity[-1])
        self.pv_fig.title.text = f"{pv.lc.line.name} PV along PA {pv.pa_deg:g}° (width {pv.width_arcsec:.2f}″, {pv.source})"
        self.plot_tabs.active = 1
        self.info.object = '<div class="sf-kv">PV diagram ready</div>'
        self.update_code()

    def make_ratio(self):
        if self.lm is None:
            pn.state.notifications.warning("make the maps of the first line first"); return
        if "stack" in self.lm.lc.settings:
            pn.state.notifications.warning("a stack is in normalised units: make the maps of a single line for a ratio"); return
        from .maps import line_maps, prepare_line
        from .ratios import ratio_map
        lm1 = self.lm; other = self.ratio_line.value
        # line 2 is made exactly as line 1 was (not with settings changed since), without velocity fits
        line1, kw, mk = self.made if self.made else (self.current_line(), self.settings(), self.map_settings())
        mk = dict(mk); mk["kinematics"] = False; mk["rms_region"] = None
        rr = self._rms_region(); s12 = (float(self.ratio_s1.value), float(self.ratio_s2.value)); unit = self.ratio_unit.value
        mode = self.rms_mode.value

        def work():
            lm2 = line_maps(prepare_line(self.cs, other, **kw), **mk)
            rm = ratio_map(lm1, lm2, rms_region=rr, sigma_thresh=s12, rms_mode=mode, unit=unit)
            rm.meta["lines"] = (line1, other)
            return rm
        self._start(work, lambda rm: self._ratio_done(rm, lm1), f"ratio {lm1.line.name} / {other}")

    @_bokeh_safe
    def _ratio_done(self, rm, lm1):
        self.ratio = rm; self.ratio_of = lm1
        sm = rm.summary()
        self.info.object = (f'<div class="sf-kv"><b>{rm.names[0]} / {rm.names[1]}</b> ({sm["unit"]}): {sm["n_valid"]} spaxels'
                            + (f', median {sm["median_ratio"]:.3g} (16–84 %: {sm["p16_p84"][0]:.3g}–{sm["p16_p84"][1]:.3g})'
                               if sm["n_valid"] else " — nothing above the thresholds") + "</div>")
        self.product.value = "ratio"
        self.show_product()
        self.update_code()

    def write_products(self):
        if self.lm is None:
            pn.state.notifications.warning("make maps first"); return
        from ..config import safe_name
        lm = self.lm; d = self.dist.value; pvd = self.pvd
        rm = self.ratio if self.ratio_of is lm else None
        out = os.path.normpath(os.path.expanduser(self.out.value.replace("{target}", safe_name(self.cs.name if self.cs else lm.lc.cube.name))))

        def work():
            files = lm.write(out, distance_pc=d)
            if pvd is not None:
                files.append(pvd.write_fits(os.path.join(out, lm.line.tag, f"{lm.line.tag}_pv_PA{pvd.pa_deg:g}.fits")))
            if rm is not None:
                t1, t2 = rm.meta.get("tags", (lm.line.tag, "line2"))
                base = os.path.join(out, "ratios", f"{t1}_over_{t2}")
                files.append(rm.write_fits(base + ".fits"))
                if np.isfinite(rm.masked1).any() and np.isfinite(rm.masked2).any():
                    import matplotlib.pyplot as plt
                    from .plots import plot_ratio_map
                    fig, _ = plot_ratio_map(rm); fig.savefig(base + ".png", dpi=130, bbox_inches="tight"); plt.close(fig)
                    files.append(base + ".png")
            if self.region is not None:
                from .regions import to_ds9_file
                p = os.path.join(out, "region.reg"); to_ds9_file(self.region, p); files.append(p)
                if self.region_spec is not None:
                    p = os.path.join(out, "region_spectrum.csv"); self.region_spec.save(p); files.append(p)
            return files
        self._start(work, lambda files: setattr(self.info, "object", f'<div class="sf-kv">wrote {len(files)} files to '
                                                                      f'<span class="sf-mono">{out}/</span></div>'), "writing")

    # ------------------------------------------------------------------ code equivalents
    @_bokeh_safe
    def update_code(self):
        line = self.current_line() or "[Ne II] 12.81"
        path = self.path.value
        stack = line.startswith("stack:")
        members = [x.strip() for x in line[6:].split("+")] if stack else []
        opts = []
        if self.core.value != 0.5:
            opts.append(f"--core {self.core.value:g}")
        if self.rv.value:
            opts.append(f"--rv {self.rv.value:g}")
        if self.dist.value != 140:
            opts.append(f"--distance {self.dist.value:g}")
        if not self.psf.value:
            opts.append("--no-psf")
        if self.snr.value != 3:
            opts.append(f"--snr {self.snr.value:g}")
        if self.kin_snr.value != 5:
            opts.append(f"--kin-snr {self.kin_snr.value:g}")
        if self.n_mc.value != 100:
            opts.append(f"--n-mc {self.n_mc.value}")
        if self.zero.value != "none":
            opts.append(f"--zero-point {self.zero.value}")
        if self.order.value != 1:
            opts.append(f"--order {self.order.value}")
        if self.window.value != 1500:
            opts.append(f"--window {self.window.value:g}")
        if self.smooth.value:
            opts.append(f"--smooth {self.smooth.value:g}")
        if self.cont_method.value != "poly":
            opts.append(f"--continuum {self.cont_method.value}")
        if self.lam.value:
            opts.append(f"--lam {self.lam.value:g}")
        comp = WINDOWS[self.window_kind.value]
        if comp:
            opts.append(f"--component {comp}")
        if self.window_um.value:
            opts.append(f"--window-um {self.window_um.value:g}")
        if BANDS[self.band.value]:
            opts.append(f"--band {BANDS[self.band.value]}")
        if not self.dq.value:
            opts.append("--no-dq")
        if self.classic.value:
            opts.append("--nan-policy propagate --zeros-valid --min-valid 1")
        rr = self._rms_region()
        if rr:
            opts.append(f'--rms-region "{rr[0]:g} {rr[1]:g} {rr[2]:g}" --rms-sigma {self.rms_sigma.value:g}'
                        + (f" --rms-mode {self.rms_mode.value}" if self.rms_mode.value != "rms" else ""))
        if stack:
            cli = f'jalebi cube stack "{path}" ' + " ".join(f'-l "{m}"' for m in members) + " --name stack"
        else:
            cli = f'jalebi cube maps "{path}" -l "{line}"'
        cli += (" " + " ".join(opts) if opts else "") + f" --out {self.out.value}"
        py = ["from jalebi import cube", f'cs = cube.CubeSet("{path}")']
        prep = (f'rv_kms={self.rv.value:g}, window_kms={self.window.value:g}, continuum={self._continuum()!r}, '
                f'psf={{"enabled": {bool(self.psf.value)}, "core_radius_fwhm": {self.core.value:g}}}')
        if self.smooth.value:
            prep += f", smooth_fwhm_pix={self.smooth.value:g}"
        if self.window_um.value:
            prep += f", window_um={self.window_um.value:g}"
        if BANDS[self.band.value]:
            prep += f', band="{BANDS[self.band.value]}"'
        rk = self._read_kw()
        if not rk["dq_mask"] or not rk["zero_is_nan"]:
            py[1] = f'cs = cube.CubeSet("{path}", dq_mask={rk["dq_mask"]}, zero_is_nan={rk["zero_is_nan"]})'

        if stack:
            py.append(f"lc = cube.stack_lines(cs, {members!r}, name='stack', {prep})")
        else:
            py.append(f'lc = cube.prepare_line(cs, "{line}", {prep})')
        extra = f', component="{comp}"' if comp else ""
        if self.classic.value:
            extra += ", min_valid=1"
        if rr:
            extra += f", rms_region=({rr[0]:g}, {rr[1]:g}, {rr[2]:g}), rms_sigma={self.rms_sigma.value:g}"
        py.append(f'm = cube.line_maps(lc, snr_min={self.snr.value:g}, kin_snr_min={self.kin_snr.value:g}, '
                  f'n_mc={self.n_mc.value}, zero_point="{self.zero.value}"{extra})')
        py.append(f'm.write("{self.out.value}", distance_pc={self.dist.value:g})')
        reg_cli = ""
        if self.region is not None:
            r = self.region
            py.append(f"reg = cube.parse_ds9('{r.to_ds9()}')[0]")
            py.append("spec = cube.region_spectrum(cs, reg)          # -> jalebi.run_pipeline(cfg, spec=spec)")
            reg_cli = f"\njalebi cube region \"{path}\" --ds9 region.reg --out region.csv --fit config.yaml"
        if self.pvd is not None:
            py.append(f"pv = cube.pv_diagram(lc, pa_deg={self.pv_pa.value:g}, length_arcsec={self.pv_len.value:g})")
        if self.ratio is not None and self.ratio_of is self.lm and not stack:
            rm = self.ratio
            l1, r2 = rm.meta.get("lines", (line, rm.names[1])); rrr = rm.rms_region
            s1, s2 = rm.sigma_thresh; unit = rm.meta.get("unit_arg", "cgs")
            reg = f", rms_region=({rrr[0]:g}, {rrr[1]:g}, {rrr[2]:g})" if rrr else ""
            reg += f', rms_mode="{rm.rms_mode}"' if rm.rms_mode != "rms" else ""
            py.append(f'm2 = cube.line_maps(cube.prepare_line(cs, "{r2}", {prep}), kinematics=False{extra})')
            py.append(f"rm = cube.ratio_map(m, m2, sigma_thresh=({s1:g}, {s2:g}){reg}, "
                      f'unit="{unit}")   # rm.ratio, rm.write_fits(...), cube.plots.plot_ratio_map(rm)')
            ropts = [o for o in opts if not o.startswith(("--snr", "--kin-snr", "--n-mc", "--zero-point", "--rms-region"))]
            reg_cli += (f'\njalebi cube ratio "{path}" -l "{l1}" -l "{r2}" --sigma {s1:g},{s2:g}'
                        + (f' --rms-region "{rrr[0]:g} {rrr[1]:g} {rrr[2]:g}"' if rrr else "")
                        + (f" --rms-mode {rm.rms_mode}" if rm.rms_mode != "rms" else "")
                        + (f' --unit "{unit}"' if unit != "cgs" else "")
                        + (" " + " ".join(ropts) if ropts else "") + f" --out {self.out.value}")
        esc = lambda t: t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")  # noqa: E731
        self.code.object = (f'<div class="sf-note">terminal</div><pre class="sf-mono" style="white-space:pre-wrap;font-size:11px">{esc(cli + reg_cli)}</pre>'
                            f'<div class="sf-note">Python</div><pre class="sf-mono" style="white-space:pre-wrap;font-size:11px">{esc(chr(10).join(py))}</pre>'
                            '<div class="sf-note">or save the config (above) and run <span class="sf-mono">jalebi cube run cube.yaml</span></div>')


def _bokeh_safe_fn(fn):
    def wrapper():
        try:
            ctx = pn.io.unlocked()
        except Exception:
            return fn()
        with ctx:
            return fn()
    return wrapper
