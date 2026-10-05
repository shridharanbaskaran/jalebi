"""The modules of the JALEBI web app.

The app is a shell (header, sidebar, theme) around independent *modules*, each with its own workflow and tabs:

  source   Source            open a target folder once (x1d, s3d cubes; cubes read in parallel), then pick an analysis
  lte      LTE slab fit      Data · Continuum · Model · Fit · Results · Batch (simultaneous LTE slab fitting)
  cube     Cube maps         line, velocity, channel and PV maps, regions of IFU cubes (jalebi.cube)
  rotdiag  Rotation diagram  population diagrams of H2, CO, OH, H2O ...: N, T, A_V, OPR (jalebi.rotdiag)

A module is any class ``Workspace(app, **options)`` with a ``panel`` attribute (the layout) and, optionally,
``figures()`` (Bokeh figures restyled by the plot-theme toggle), ``apply_theme(theme_dict)``, ``on_show()``
(called when the module is opened), ``use_spectrum(spec, label)`` (receives a spectrum sent from another
module) and ``use_source(source, distance_pc, rv_kms)`` (receives the `jalebi.source.Source` opened on the Source
page; a module built later gets it when it is built).  Register a new one with

    from jalebi.modules import ModuleSpec, register_module
    register_module(ModuleSpec("ice", "Ice", "❄", "ice absorption bands", "mypkg.ice_app:IceWorkspace"))

before the app is built; ``jalebi serve --module ice`` then opens it.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class ModuleSpec:
    key: str
    label: str
    icon: str
    description: str
    factory: str | None            # "package.module:Class" (None = built into the app: the LTE slab fit)
    accepts_spectrum: bool = False # can receive a spectrum from another module ("Send to ...")

    def load(self):
        if self.factory is None:
            return None
        mod, _, cls = self.factory.partition(":")
        return getattr(importlib.import_module(mod), cls)


MODULES: dict[str, ModuleSpec] = {}
ALIASES = {"home": "source", "start": "source", "src": "source", "slab": "lte", "fit": "lte", "slabfit": "lte", "lte-fit": "lte", "cubes": "cube", "maps": "cube",
           "rotation": "rotdiag", "rotation-diagram": "rotdiag", "rd": "rotdiag", "excitation": "rotdiag", "population": "rotdiag"}


def register_module(spec: ModuleSpec):
    MODULES[spec.key] = spec


def resolve_module(name: str | None) -> str | None:
    if not name:
        return None
    k = name.strip().lower()
    k = ALIASES.get(k, k)
    return k if k in MODULES else None


register_module(ModuleSpec("source", "Source", "◉", "open a target folder once; the analyses share the data in memory",
                           "jalebi.source_app:SourceWorkspace"))
register_module(ModuleSpec("lte", "LTE slab fit", "◈", "simultaneous LTE slab fit of molecular emission in a 1-D spectrum", None,
                           accepts_spectrum=True))
register_module(ModuleSpec("cube", "Cube maps", "▦", "line, velocity, channel and PV maps of IFU cubes; region spectra",
                           "jalebi.cube.app:CubeWorkspace"))
register_module(ModuleSpec("rotdiag", "Rotation diagram", "⟋", "population diagrams of H₂, CO, OH, H₂O …: N, T, A_V, OPR",
                           "jalebi.rotdiag.app:RotDiagWorkspace", accepts_spectrum=True))
