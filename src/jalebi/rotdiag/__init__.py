"""jalebi.rotdiag — rotation (population) diagrams of H2, CO, OH, H2O ... from JWST spectra.

Pick a molecule, and the module loads its line list, finds the lines inside your spectrum (blends and
contaminants flagged), measures their fluxes, builds the diagram ln(N_u/g_u) vs E_u and fits it:

* models: one temperature, two temperatures (warm + hot), or a power-law distribution of temperatures
  dN ∝ T^−b dT (Neufeld & Yuan 2008);
* extinction A_V (free or fixed) with a choice of curves (Gordon+2023 by default, or your own, e.g. KP5);
* ortho-to-para ratio (H2, H2O): thermal, free (spin species each in LTE, exact partition sums), or the
  ln(OPR/3) offset convention;
* optical depth (CO, H2O, OH): curve-of-growth correction for a slab of given line width and area;
* least squares in flux space (blends and non-detections handled exactly) + emcee posteriors, corner plots,
  derived quantities (total column, number of molecules, warm-gas mass, emitting radius, A_K, luminosity).

    from jalebi.rotdiag import RotDiagConfig, run_rotdiag
    res = run_rotdiag(RotDiagConfig(molecule="H2", spectrum={"path": "example:FZ_Tau"}))

Terminal: ``jalebi rotdiag --help``.  Web app: ``jalebi serve --module rotdiag``.  See docs/ROTDIAG.md.
"""
from .config import EXAMPLES, RotDiagConfig, example_config
from .extinction import available_curves, get_curve
from .features import Selection, find_features, match_table
from .fit import (FitConfig, MCMCConfig, RotFit, compare_models, diagram_points, fit_rotation, model_curves,
                  model_points)
from .measure import MeasureConfig, measure_features
from .physics import Geometry, RotModel, cog_factor
from .pipeline import RotDiagResult, run_rotdiag, save_results
from .species import PRESETS, partition, preset, rotdiag_molecules

__all__ = ["EXAMPLES", "FitConfig", "Geometry", "MCMCConfig", "MeasureConfig", "PRESETS", "RotDiagConfig", "RotDiagResult",
           "RotFit", "RotModel", "Selection", "available_curves", "cog_factor", "compare_models", "diagram_points",
           "example_config", "find_features", "fit_rotation", "get_curve", "match_table", "measure_features", "model_curves",
           "model_points", "partition", "preset", "rotdiag_molecules", "run_rotdiag", "save_results"]
