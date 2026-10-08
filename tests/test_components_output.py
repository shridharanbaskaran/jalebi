"""0.22 per-component output: model.csv holds one column per component plus the continuum; fit_components.png."""
import os

import numpy as np
import pandas as pd

import jalebi.pipeline as pl

from test_resume import make_config


def test_model_csv_has_component_columns(tmp_path):
    cfg = make_config(str(tmp_path / "c"))
    cfg.fit.stages = ["optimise"]; cfg.continuum.refine_iterations = 0
    run = pl.run_pipeline(cfg)
    m = pd.read_csv(os.path.join(run.outdir, "model.csv"))
    assert {"wave", "band", "data", "sigma", "model", "continuum", "model_CO2", "model_HCN"} <= set(m.columns)
    assert np.allclose(m.model_CO2 + m.model_HCN, m.model, rtol=1e-10, atol=1e-14)
    assert np.allclose(m.continuum, np.asarray(run.spec.continuum)[run.problem.used])
    assert (m.model_CO2 >= 0).all() and m.model_CO2.max() > 10 * m.sigma.median()
    assert os.path.exists(os.path.join(run.outdir, "fit_components.png"))
