"""The *Source* page of the web app: open a target folder once, then choose the analysis.

    ① pick a source (scan a data root, or type a folder / file)  →  ② Open (headers; cubes into memory in
    parallel)  →  ③ LTE slab fit · Cube maps · Rotation diagram, all working on the data already in memory.

The opened `jalebi.source.Source` lives in a process-wide cache, so re-opening the page, a second browser tab, or
switching back and forth between the analyses never reads the FITS files again.  Distance, RV and the source
position can be remembered per source (jalebi_source.yaml next to the data).
"""
from __future__ import annotations

import os
import threading
import time
import traceback

import numpy as np
import panel as pn
from bokeh.models import ColumnDataSource

from . import activity as act
from .app import BAND_COLOURS, PAL, _bokeh_safe, _fig, _html, _panel
from .data import BAND_ORDER

ANALYSES = [
    ("lte", "◈", "LTE slab fit",
     "Simultaneous LTE slab models of H₂O, CO, CO₂, HCN, C₂H₂ … on the 1-D spectrum (x1d, or an aperture on the cubes): "
     "continuum, molecule detection, grid → optimiser → parallel MCMC with corner plots.", "1d"),
    ("cube", "▦", "Cube maps",
     "Line, velocity, channel and PV maps of the IFU cubes, PSF subtraction, regions; region spectra can be sent to "
     "the slab fit or the rotation diagram.", "cubes"),
    ("rotdiag", "⟋", "Rotation diagram",
     "Population diagrams of H₂, CO, OH, H₂O … from a region of the cubes or the x1d: N, T (one, two or power-law), "
     "A_V, OPR with MCMC.", "any"),
]


@act.traced("source", skip=("figures", "apply_theme", "_on_progress", "settings"))
class SourceWorkspace:
    def __init__(self, app, source_path: str | None = None):
        self.app = app
        self.src = None
        self._job = None
        self._cb = None
        self._prog = {"done": 0, "total": 0, "band": "", "stage": ""}
        self._stop = threading.Event()
        W = dict(sizing_mode="stretch_width")
        root = getattr(app, "data_root", None) or os.getcwd()
        # ---- pick ---------------------------------------------------------------------------------
        self.root = pn.widgets.TextInput(name="data root (folder of target folders)", value=root, **W)
        self.scan_btn = pn.widgets.Button(name="Scan", width=90, margin=(22, 5, 5, 5))
        self.list = pn.widgets.Select(name="sources found (● = already in memory)", options={}, size=9, **W)
        self.path = pn.widgets.TextInput(name="source folder or file", value=source_path or "", placeholder="…/V-HV-TAU-C_… or example:FZ_Tau", **W)
        self.dq = pn.widgets.Checkbox(name="blank DQ DO_NOT_USE", value=True, margin=(10, 10, 2, 10))
        self.zero = pn.widgets.Checkbox(name="SCI = 0 is undefined", value=True, margin=(10, 10, 2, 10))
        self.preload_cb = pn.widgets.Checkbox(name="read all cubes into memory now", value=True, margin=(10, 10, 2, 10))
        self.workers = pn.widgets.IntInput(name="parallel readers", value=min(6, os.cpu_count() or 2), start=1, end=32, width=130)
        self.open_btn = pn.widgets.Button(name="◉  Open source", button_type="primary", width=170, margin=(22, 5, 5, 5))
        self.refresh_cb = pn.widgets.Checkbox(name="re-read from disk", value=False, margin=(30, 10, 2, 10))
        self.progress = pn.indicators.Progress(value=0, max=100, visible=False, **W)
        self.info = _html('<div class="sf-note">Scan your data root, pick a source and press <b>Open source</b>. '
                          'Everything is read once; then choose the analysis below.</div>', **W)
        # ---- per-source settings ------------------------------------------------------------------
        self.dist = pn.widgets.FloatInput(name="distance [pc]", value=140.0, step=1, width=120)
        self.rv = pn.widgets.FloatInput(name="systemic RV [km/s]", value=0.0, step=0.5, width=140)
        self.ra = pn.widgets.TextInput(name="RA (deg / hms; blank = auto)", value="", width=190)
        self.dec = pn.widgets.TextInput(name="Dec (deg / dms; blank = auto)", value="", width=190)
        self.save_btn = pn.widgets.Button(name="Remember for this source", width=200, margin=(22, 5, 5, 5), disabled=True)
        self.apply_btn = pn.widgets.Button(name="Apply to the analyses", width=180, margin=(22, 5, 5, 5), disabled=True)
        self.summary = _html("", **W)
        self.coverage = _html("", **W)
        # ---- launch cards -------------------------------------------------------------------------
        self.launch = {}
        cards = []
        for key, icon, label, text, need in ANALYSES:
            b = pn.widgets.Button(name=f"{icon}  {label}  →", button_type="primary", disabled=True, **W)
            b.on_click(lambda e, k=key: self.go(k))
            note = _html("", **W)
            self.launch[key] = (b, need, note)
            cards.append(pn.Column(_html(f'<div class="sf-title" style="font-size:16px">{icon} {label}</div>'
                                         f'<div class="sf-note" style="min-height:58px">{text}</div>', **W), note, b,
                                   css_classes=["sf-panel"], styles={"padding": "12px", "flex": "1 1 0"}, **W))
        self.cards = pn.Row(*cards, **W)
        # ---- previews -----------------------------------------------------------------------------
        self.spec_fig = _fig("spectrum preview", height=300)
        self.spec_src = {b: ColumnDataSource(dict(w=[], f=[])) for b in BAND_ORDER}
        for b in BAND_ORDER:
            self.spec_fig.line("w", "f", source=self.spec_src[b], color=BAND_COLOURS[b], line_width=1.1)
        self.img_band = pn.widgets.Select(name="image", options=[], width=110)
        self.img_fig = _fig("continuum image", height=300, y_label="Δδ [px]", x_label="Δα [px]")
        self.img_fig.match_aspect = True
        self.img_fig.width = 330; self.img_fig.sizing_mode = "fixed"
        self.img_src = ColumnDataSource(dict(image=[np.zeros((2, 2))], x=[0], y=[0], dw=[2], dh=[2]))
        self.img_fig.image("image", "x", "y", "dw", "dh", source=self.img_src, palette="Inferno256")
        self.mark = ColumnDataSource(dict(x=[], y=[]))
        self.img_fig.scatter("x", "y", source=self.mark, marker="cross", size=16, line_width=2, color=PAL.teal)
        self.img_pane = pn.Column(self.img_band, pn.pane.Bokeh(self.img_fig), visible=False, width=350)
        # ---- events -------------------------------------------------------------------------------
        self.scan_btn.on_click(lambda e: self.scan())
        self.list.param.watch(lambda e: setattr(self.path, "value", e.new or self.path.value), "value")
        self.open_btn.on_click(lambda e: self.open())
        self.save_btn.on_click(lambda e: self.remember())
        self.apply_btn.on_click(lambda e: self.apply_settings())
        self.img_band.param.watch(lambda e: self.show_image(), "value")
        cubes_panel = pn.Column(_html('<div class="sf-label">Cubes</div>'), self.img_pane, css_classes=["sf-panel"], width=380,
                                margin=(0, 0, 10, 10))
        self.details = pn.Column(pn.Row(_panel(self.summary, self.coverage, title="Source"), cubes_panel, **W),
                                 _panel(pn.pane.Bokeh(self.spec_fig, **W), title="Spectrum"), visible=False, **W)
        self.panel = pn.Column(
            pn.Row(
                pn.Column(_panel(pn.Row(self.root, self.scan_btn, **W), self.list, title="① Find a source"), width=520),
                pn.Column(_panel(self.path, pn.Row(self.dq, self.zero, **W), pn.Row(self.preload_cb, self.workers, **W),
                                 pn.Row(self.open_btn, self.refresh_cb), self.progress, self.info, title="② Open it once"),
                          _panel(pn.Row(self.dist, self.rv, **W), pn.Row(self.ra, self.dec, **W), pn.Row(self.apply_btn, self.save_btn),
                                 _html('<div class="sf-note">Used by every analysis. <b>Remember</b> writes them to '
                                       '<span class="sf-mono">jalebi_source.yaml</span> next to the data (or ~/.jalebi/sources/), '
                                       'so they come back next time.</div>', **W), title="Source settings"), **W),
                **W),
            _panel(self.cards, title="③ Choose the analysis"),
            self.details,
            **W)
        if source_path:
            self.path.value = source_path
        self.scan()

    # ------------------------------------------------------------------ helpers
    def figures(self):
        return [self.spec_fig, self.img_fig]

    def apply_theme(self, th: dict):
        for b in BAND_ORDER:
            for r in self.spec_fig.renderers:
                if getattr(r, "data_source", None) is self.spec_src[b]:
                    r.glyph.line_color = th["bands"][b]

    @_bokeh_safe
    def scan(self):
        from .source import scan_sources
        try:
            with act.step(f"scanning {self.root.value} for sources (2 levels deep)", "source") as st:
                found = scan_sources(self.root.value)
                st.note(f"{len(found)} found" + (": " + ", ".join(e.label for e in found[:8]) + (" …" if len(found) > 8 else "") if found else ""))
        except Exception as ex:
            self.info.object = f'<div class="sf-note">⚠ scan failed: {ex}</div>'; return
        self.list.options = {e.label: e.path for e in found}
        if found and not self.path.value:
            self.path.value = found[0].path
        self.info.object = (f'<div class="sf-note">{len(found)} sources under <span class="sf-mono">{self.root.value}</span>. '
                            'Pick one and press <b>Open source</b>.</div>')

    # ------------------------------------------------------------------ open
    def open(self, path: str | None = None):
        """Open the source (headers, x1d, source position) and, optionally, read all cubes in parallel —
        in a worker thread when served, inline otherwise (tests, notebooks)."""
        from .source import open_source
        path = (path or self.path.value or "").strip()
        if not path:
            pn.state.notifications.warning("type or pick a source folder first") if pn.state.curdoc else None
            return
        if self._job is not None and self._job.is_alive():
            pn.state.notifications.warning("still opening the previous source") if pn.state.curdoc else None
            return
        dq, zero, pre, nw, refresh = self.dq.value, self.zero.value, self.preload_cb.value, self.workers.value, self.refresh_cb.value
        self._stop.clear()
        res = {"src": None, "error": None, "t": time.time()}
        self._prog.update(done=0, total=0, band="", stage="reading headers")
        act.info("source", "opening %s (DQ mask %s, SCI = 0 undefined %s, %s, %d readers%s)", path, dq, zero,
                 "read all cubes now" if pre else "cubes read when needed", nw, ", re-read from disk" if refresh else "")

        def work():
            self._bar = None
            try:
                src = open_source(path, dq_mask=dq, zero_is_nan=zero, refresh=refresh)
                _ = src.meta, src.name
                act.info("source", "%s: %s", src.name, ", ".join(f"{k} {v}" for k, v in src.meta.items()
                                                                  if k in ("PROGRAM", "OBSERVTN", "DATE-OBS", "CAL_VER", "CRDS_CTX")) or "no header facts")
                if src.has_1d:
                    self._prog["stage"] = "reading x1d"; src.x1d()
                if src.has_cubes:
                    self._prog["total"] = len(src.cubes.info)
                    if pre:
                        self._prog["stage"] = "reading cubes"
                        self._bar = act.Progress(f"reading the cubes of {src.name}", total=len(src.cubes.info), unit="cube", area="source")
                        src.preload(workers=nw, progress=self._on_progress, stop=self._stop)
                        self._bar.close(f"{len(src.cubes.loaded())}/{len(src.cubes.info)} cubes in memory ({src.memory_mb():.0f} MB)")
                    self._prog["stage"] = "locating the source"
                    src.position(); src.image()
                res["src"] = src
                act.info("source", "✓ %s ready in %s · %.0f MB in memory", src.name, act.fmt_time(time.time() - res["t"]), src.memory_mb())
            except Exception as ex:
                res["error"] = f"{ex}\n{traceback.format_exc(limit=4)}"
                if self._bar is not None:
                    self._bar.close("failed", ok=False)
                act.log("source", "could not open %s: %s", path, ex, level=act.ERROR, exc_info=True)

        self.open_btn.disabled = True
        self.progress.visible = True; self.progress.value = 0
        self.info.object = f'<div class="sf-kv">⏳ opening <span class="sf-mono">{path}</span> …</div>'
        if pn.state.curdoc is None:
            work(); self._opened(res); return
        self._job = threading.Thread(target=act.thread_target(work), daemon=True, name="jalebi-open-source")
        self._job.start()

        def poll():
            if self._job.is_alive():
                p = self._prog
                self.progress.value = int(100 * p["done"] / p["total"]) if p["total"] else 5
                self.info.object = (f'<div class="sf-kv">⏳ {p["stage"]}'
                                    + (f' — {p["done"]}/{p["total"]} cubes {p["band"]}' if p["total"] else "") + " …</div>")
                return
            self._cb.stop(); self._cb = None
            self._opened(res)
        from .cube.app import _bokeh_safe_fn
        self._cb = pn.state.add_periodic_callback(_bokeh_safe_fn(poll), period=200)

    def _on_progress(self, done, total, band):
        self._prog.update(done=done, total=total, band=band)
        bar = getattr(self, "_bar", None)
        if bar is not None:
            bar.update(n=done, total=total, extra=band)
        if band:
            act.debug("source", "cube %d/%d in memory: %s", done, total, band)

    @_bokeh_safe
    def _opened(self, res):
        self.open_btn.disabled = False
        self.progress.visible = False
        if res["error"]:
            msg = res["error"].splitlines()[0]
            self.info.object = f'<div class="sf-note">⚠ could not open: {msg}</div>'
            if pn.state.curdoc is not None:
                pn.state.notifications.error(f"could not open the source: {msg}")
            return
        src = res["src"]
        self.src = src
        self.dist.value = src.distance_pc or self.dist.value
        self.rv.value = src.rv_kms if src.rv_kms is not None else 0.0
        self.ra.value = str(src.settings.get("ra", "")) if src.settings.get("ra") is not None else ""
        self.dec.value = str(src.settings.get("dec", "")) if src.settings.get("dec") is not None else ""
        self.save_btn.disabled = False; self.apply_btn.disabled = False
        self.info.object = (f'<div class="sf-kv">✓ <b>{src.name}</b> ready in {time.time() - res["t"]:.1f} s · '
                            f'{src.memory_mb():.0f} MB in memory — choose the analysis below</div>')
        self.show_source()
        self.app.set_source(src, distance_pc=self.dist.value, rv_kms=self.rv.value)
        self.scan()                                     # marks the source as loaded (●)

    # ------------------------------------------------------------------ settings
    def settings(self) -> dict:
        return dict(distance_pc=self.dist.value, rv_kms=self.rv.value, ra=self.ra.value.strip() or None, dec=self.dec.value.strip() or None)

    def remember(self):
        if self.src is None:
            return
        try:
            f = self.src.save_settings(**self.settings())
            self.src._memo = {k: v for k, v in self.src._memo.items() if not (isinstance(k, tuple) and k[0] == "position")}
            if pn.state.curdoc is not None:
                pn.state.notifications.success(f"remembered in {f}")
            self.apply_settings()
        except Exception as ex:
            if pn.state.curdoc is not None:
                pn.state.notifications.error(f"could not save: {ex}")

    def apply_settings(self):
        if self.src is None:
            return
        s = self.settings()
        if s["ra"] and s["dec"]:
            self.src.settings["ra"], self.src.settings["dec"] = s["ra"], s["dec"]
        self.show_source()
        self.app.set_source(self.src, distance_pc=s["distance_pc"], rv_kms=s["rv_kms"], force=True)

    # ------------------------------------------------------------------ display
    @_bokeh_safe
    def show_source(self):
        src = self.src
        self.details.visible = True
        s = src.summary()
        rows = [("target", f'<b>{s["name"]}</b>'), ("path", f'<span class="sf-mono">{s["path"]}</span>')]
        if "ra" in s:
            rows.append(("position", f'RA {s["ra"]:.6f} · Dec {s["dec"]:.6f} <span class="dim">({s["position_from"]})</span>'))
        for k, lab in (("program", "programme"), ("observtn", "observation"), ("date-obs", "date"), ("instrume", "instrument"),
                       ("cal_ver", "jwst pipeline"), ("crds_ctx", "CRDS context")):
            if s.get(k) is not None:
                rows.append((lab, f'<span class="sf-mono">{s[k]}</span>'))
        if src.has_cubes:
            rows.append(("cubes", f'{len(s["cube_bands"])} · ~{s["cube_memory_mb"]:.0f} MB · {len(s["cubes_in_memory"])} in memory'))
        rows.append(("memory now", f"{src.memory_mb():.0f} MB"))
        if s["settings_file"]:
            rows.append(("settings", f'<span class="sf-mono">{s["settings_file"]}</span>'))
        self.summary.object = ('<table class="sf-kv" style="border-collapse:collapse">' +
                               "".join(f'<tr><td style="padding:2px 14px 2px 0;color:{PAL.muted}">{a}</td><td>{b}</td></tr>' for a, b in rows)
                               + "</table>")
        cov = src.coverage()
        cells = []
        for r in cov.itertuples():
            on = r.x1d or r.s3d
            tag = " · ".join(t for t, v in (("x1d", r.x1d), ("s3d", r.s3d)) if v) or "—"
            mem = " ●" if getattr(r, "in_memory", False) else ""
            cells.append(f'<div style="padding:6px 8px;border-radius:6px;border:1px solid {PAL.border};'
                         f'background:{"rgba(58,208,196,0.12)" if on else "transparent"};opacity:{1 if on else 0.45}">'
                         f'<b>{r.band}</b>{mem}<br><span class="sf-note">{tag}</span></div>')
        self.coverage.object = (f'<div class="sf-note" style="margin:10px 0 4px">MRS sub-bands (● cube in memory)</div>'
                                f'<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:6px">{"".join(cells)}</div>')
        # launch buttons
        for key, (btn, need, note) in self.launch.items():
            ok = {"1d": src.has_1d or src.has_cubes, "cubes": src.has_cubes, "any": True}[need]
            btn.disabled = not ok
            if key == "lte":
                note.object = f'<div class="sf-note">input: {"x1d" if src.has_x1d else ("table" if src.has_table else "aperture on the cubes")}</div>'
            elif key == "cube":
                note.object = '<div class="sf-note">' + (f"{len(src.cube_bands)} cubes ready" if src.has_cubes else "needs s3d cubes") + "</div>"
            else:
                note.object = ('<div class="sf-note">input: ' + ("1″ circle on the cubes" if src.has_cubes else "x1d / table") + "</div>")
        # previews
        if src.has_1d:
            sp = src.x1d()
        elif src.has_cubes:
            sp = src.aperture_spectrum()
        else:
            sp = None
        for b in BAND_ORDER:
            if sp is not None:
                i = sp.band_slice(b)
                self.spec_src[b].data = dict(w=sp.wave[i], f=sp.flux[i])
            else:
                self.spec_src[b].data = dict(w=[], f=[])
        if sp is not None:
            how = "x1d" if src.has_1d else "1.5 FWHM aperture on the cubes"
            self.spec_fig.title.text = f"{src.name}: {how}, {len(sp.wave)} pixels in {', '.join(sp.bands) or '1 band'} (observed frame)"
        self.img_pane.visible = src.has_cubes
        if src.has_cubes:
            opts = list(dict.fromkeys(src.cube_bands))
            keep = self.img_band.value
            self.img_band.options = opts
            self.img_band.value = keep if keep in opts else opts[0]
            self.show_image()

    @_bokeh_safe
    def show_image(self):
        if self.src is None or not self.src.has_cubes or not self.img_band.value:
            return
        img, c = self.src.image(self.img_band.value)
        im = np.where(np.isfinite(img), img, np.nan)
        lo, hi = np.nanpercentile(im, 5), np.nanpercentile(im, 99.7)
        im = np.log10(np.clip(np.nan_to_num(im, nan=lo) - lo, 1e-3 * max(hi - lo, 1e-9), None))
        self.img_src.data = dict(image=[im], x=[-0.5], y=[-0.5], dw=[im.shape[1]], dh=[im.shape[0]])
        try:
            ra, dec, _ = self.src.position()
            x, y = c.world_to_pix(ra, dec)
            self.mark.data = dict(x=[float(x)], y=[float(y)])
        except Exception:
            self.mark.data = dict(x=[], y=[])
        self.img_fig.title.text = f"{c.band}: median image (log), + = source"

    # ------------------------------------------------------------------ launch
    def go(self, key: str):
        if self.src is None:
            return
        act.info("source", "launching %s on %s", dict((k, l) for k, _, l, _, _ in ANALYSES).get(key, key), self.src.name)
        self.app.switch_module(key)

