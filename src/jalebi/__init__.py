"""JALEBI — JWST Analysis of Line Emission with Bayesian Inference.

Simultaneous LTE slab fitting of the molecular emission (H2O, CO2, C2H2, HCN, CO, OH, ...) in
JWST/MIRI-MRS spectra of protoplanetary (Class II) disks.

Layers
------
core      : constants, molecules, linedata, partition, instrument, model
data      : data (ingest), continuum (baselines, masks, noise)
fitting   : fit (likelihood, priors, grid, optimiser, emcee), detect, plots
interfaces: config (YAML), pipeline, cli (Typer), app (Panel web app)
helpers   : synthetic (spectra with known answers), doctor (installation checks), examples

Units: wavelengths in micron, column densities in cm^-2, temperatures in K,
emitting radius in au, fluxes in Jy.

Quick start
-----------
>>> import jalebi
>>> spec, truth = jalebi.synthetic.make_synthetic_spectrum()      # a spectrum with known answers
>>> jalebi.example_path("FZ_Tau")                                  # bundled MIRI x1d files
"""

__version__ = "0.9.0"
__all__ = ["__version__", "MOLECULES", "get_molecule", "LineList", "load_linelist", "example_path",
           "ProjectConfig", "Component", "build_model", "Spectrum", "load_spectrum", "run_pipeline"]

# Only standard-library modules are imported eagerly, so `import jalebi` is fast and
# `python -m jalebi.doctor` can report missing dependencies instead of crashing on them.
from .molecules import MOLECULES, get_molecule  # noqa: F401,E402
from .examples import example_path  # noqa: F401,E402


def __getattr__(name):
    # heavier modules are imported lazily
    if name in ("LineList", "load_linelist"):
        from . import linedata
        return getattr(linedata, name)
    if name in ("ProjectConfig",):
        from .config import ProjectConfig
        return ProjectConfig
    if name in ("Component", "build_model", "SlabModel"):
        from . import model
        return getattr(model, name)
    if name in ("Spectrum", "load_spectrum"):
        from . import data
        return getattr(data, name)
    if name == "run_pipeline":
        from .pipeline import run_pipeline
        return run_pipeline
    if name in ("synthetic", "doctor", "pipeline", "fit", "detect", "plots", "continuum", "data", "model", "config"):
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module 'jalebi' has no attribute {name!r}")
