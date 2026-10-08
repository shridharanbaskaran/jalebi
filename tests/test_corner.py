"""0.22 corner / pseudo-continuum diagnostic (jalebi.corner, fit.corner_check)."""
import json
import os

import numpy as np

import jalebi.pipeline as pl
from jalebi.corner import corner_check, pinned_parameters, smooth_fraction
from jalebi.fit import FitProblem, default_free_params
from jalebi.model import Component
from jalebi.synthetic import make_synthetic_spectrum

from test_resume import make_config


def test_smooth_fraction_lines_vs_hump():
    wave = np.linspace(14.0, 15.0, 800); band = np.array(["3B"] * 800)
    lines = np.zeros(800)
    for c in np.linspace(14.05, 14.95, 25):
        lines += np.exp(-0.5 * ((wave - c) / 0.002) ** 2)
    hump = np.exp(-0.5 * ((wave - 14.5) / 0.4) ** 2)
    assert smooth_fraction(wave, band, lines, 0.3) < 0.1
    assert smooth_fraction(wave, band, hump, 0.3) > 0.9
    assert np.isnan(smooth_fraction(wave, band, np.zeros(800), 0.3))
    # a dense forest (lines every 2 pixels, wider than the window resolves) is mostly smooth
    forest = np.zeros(800)
    for c in wave[::2]:
        forest += np.exp(-0.5 * ((wave - c) / 0.002) ** 2)
    assert smooth_fraction(wave, band, forest, 0.3) > 0.6


def test_pinned_and_flags():
    truth = [{"name": "H2O", "molecule": "H2O", "logN": 18.0, "T": 700.0, "logR": -0.5}]
    truth[0]["linelist_release"] = "hitran"
    spec, t = make_synthetic_spectrum(components=truth, bands=("3B",), snr=100.0, seed=1, oversample=3)
    spec.continuum = t["continuum"]
    comps = [Component("H2O", "H2O", logN=18.0, T=700.0, logR=-0.5, linelist_release="hitran")]
    prob = FitProblem(spec, comps, [(13.8, 15.2)], default_free_params(comps), oversample=3, use_pipeline_err=True)
    th = prob.theta0()
    assert pinned_parameters(prob, th) == {}
    corner = th.copy()
    for j, p in enumerate(prob.free):
        if p.name in ("T", "logN"):
            corner[j] = prob.hi[j] - 1e-4 * (prob.hi[j] - prob.lo[j])
        if p.name == "logR":
            corner[j] = prob.lo[j] + 1e-4 * (prob.hi[j] - prob.lo[j])
    pins = pinned_parameters(prob, corner)
    assert pins == {"H2O": ["logN@hi", "T@hi", "logR@lo"]} or set(pins["H2O"]) == {"logN@hi", "T@hi", "logR@lo"}
    chk = corner_check(prob, corner)
    assert "pinned" in chk["H2O"]["flags"] and chk["H2O"]["n_pinned"] == 3
    chk0 = corner_check(prob, th)
    assert "pinned" not in chk0["H2O"]["flags"] and 0.0 <= chk0["H2O"]["smooth_fraction"] <= 1.0


def test_pipeline_writes_corner_diagnostics(tmp_path):
    cfg = make_config(str(tmp_path / "corner"))
    cfg.fit.stages = ["optimise"]; cfg.continuum.refine_iterations = 0
    run = pl.run_pipeline(cfg)
    d = json.load(open(os.path.join(run.outdir, "diagnostics.json")))
    assert set(d["corner"]) == {"CO2", "HCN"}
    assert d["corner"]["CO2"]["flags"] == [] and 0 <= d["corner"]["CO2"]["smooth_fraction"] <= 1
    assert any("corner check: no component pinned" in l for l in run.log)
    row = pl.catalogue_row(run)
    assert "CO2_smooth_frac" in row and row["CO2_pinned"] == 0


def test_report_co_na_only():
    import pandas as pd
    from jalebi.config import ComponentConfig, ProjectConfig
    from jalebi.model import Component
    cfg = ProjectConfig(components=[ComponentConfig(name="CO", molecule="CO", logN=18.0, T=1500.0, logR=-0.5),
                                    ComponentConfig(name="H2O", molecule="H2O", logN=18.0, T=700.0, logR=-0.5)])
    comps = cfg.components_list()
    summ = pd.DataFrame({"parameter": ["CO.logN", "CO.T", "CO.logR", "CO.logNA", "CO.R_au", "H2O.logN", "H2O.logNA", "global.log_s"]})
    assert pl.report_mask(cfg, summ, comps).all()
    cfg.report.co = "NA_only"
    m = pl.report_mask(cfg, summ, comps)
    assert list(summ.parameter[m]) == ["CO.logNA", "H2O.logN", "H2O.logNA", "global.log_s"]


def test_pseudo_continuum_flag_only_for_listed_molecules():
    """A CO2 Q branch is a smooth band head by nature: reported, not flagged (smooth_molecules default = water)."""
    truth = [{"name": "CO2", "molecule": "CO2", "logN": 18.5, "T": 500.0, "logR": -0.3}]
    spec, t = make_synthetic_spectrum(components=truth, bands=("3B",), snr=100.0, seed=1, oversample=3)
    spec.continuum = t["continuum"]
    comps = [Component("CO2", "CO2", logN=18.5, T=500.0, logR=-0.3)]
    prob = FitProblem(spec, comps, [(14.8, 15.2)], default_free_params(comps), oversample=3, use_pipeline_err=True)
    chk = corner_check(prob, prob.theta0(), smooth_window_um=0.3, smooth_threshold=0.3)
    assert chk["CO2"]["smooth_fraction"] > 0.3 and "pseudo-continuum" not in chk["CO2"]["flags"]
    chk_all = corner_check(prob, prob.theta0(), smooth_window_um=0.3, smooth_threshold=0.3, smooth_molecules=())
    assert "pseudo-continuum" in chk_all["CO2"]["flags"]
