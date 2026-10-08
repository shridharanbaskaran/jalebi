"""0.22 detection probability with the continuum varied (jalebi.detection_prob, jalebi detect-prob)."""
import os

import numpy as np
import pytest

import jalebi.pipeline as pl
from jalebi import detection_prob as dp
from jalebi.config import ComponentConfig, ProjectConfig
from jalebi.synthetic import make_synthetic_spectrum, save_synthetic

TRUTH = [{"name": "CO2", "molecule": "CO2", "logN": 17.3, "T": 520.0, "logR": -0.6}]


@pytest.fixture(scope="module")
def fake_hcn(tmp_path_factory):
    """Strong CO2 (real) and a broad 1.5 % continuum hump at the HCN Q branch (fake): HCN is 'detected' at the
    nominal continuum (Delta BIC > 10) only because the continuum estimate passes under the hump."""
    tmp = str(tmp_path_factory.mktemp("dp"))
    spec, t = make_synthetic_spectrum(components=TRUTH, bands=("3B",), snr=150.0, seed=3, oversample=3)
    spec.flux = spec.flux + 0.015 * t["continuum"] * np.exp(-0.5 * ((spec.wave - 14.02) / 0.08) ** 2)
    save_synthetic(os.path.join(tmp, "synth.csv"), spec, t)
    cfg = ProjectConfig(components=[
        ComponentConfig(name="CO2", molecule="CO2", logN=17.0, T=450.0, logR=-0.5, bounds={"T": [300.0, 900.0], "logN": [15.0, 19.0]}),
        ComponentConfig(name="HCN", molecule="HCN", logN=16.0, T=500.0, logR=-0.8, bounds={"T": [300.0, 900.0], "logN": [14.0, 18.0]})])
    cfg.target.name = "synth"; cfg.target.path = os.path.join(tmp, "synth.csv"); cfg.target.spike_filter = False
    cfg.continuum.method = "irsqr"; cfg.continuum.quantile = 0.1; cfg.continuum.knot_spacing = 50; cfg.continuum.refine_iterations = 0
    cfg.masks.default_lines = False; cfg.masks.oh_prompt = False
    cfg.fit.windows = [[13.8, 15.2]]; cfg.fit.oversample = 3; cfg.fit.use_pipeline_err = True
    cfg.fit.stages = ["optimise"]; cfg.fit.optimise.maxiter = 15; cfg.fit.optimise.popsize = 6; cfg.fit.optimise.n_starts = 1
    cfg.output = os.path.join(tmp, "out", "{target}")
    run = pl.run_pipeline(cfg)
    return cfg, run


def test_variants_are_reproducible_and_in_range():
    cfg = ProjectConfig()
    s = dp.DetProbSettings(n_variants=10, seed=4, offset_sigma=0.01, methods=["median_sg"])
    v1 = dp.variant_configs(cfg, s); v2 = dp.variant_configs(cfg, s)
    assert v1 == v2 and len(v1) == 10 and v1[0] == ({}, 1.0)
    for over, off in v1[1:]:
        assert over["method"] in ("irsqr", "median_sg") and abs(off - 1.0) < 0.05
        if over["method"] == "irsqr":
            assert 0.05 <= over["quantile"] <= 0.2 and 15 <= over["knot_spacing"] <= 60
        else:
            assert over["median_window"] % 2 == 1 and 51 <= over["median_window"] <= 201
    assert dp.classify(1.0, True, s) == "robust" and dp.classify(0.0, False, s) == "not detected"
    assert dp.classify(0.5, True, s) == "continuum-dependent" and dp.classify(0.0, True, s) == "continuum-dependent"


def test_ensemble_flags_the_fake_detection(fake_hcn):
    cfg, run = fake_hcn
    sig = run.problem.component_significance(run.theta).set_index("component")
    assert sig.at["HCN", "delta_BIC"] > 10                    # the fake is "detected" at the nominal continuum
    s = dp.settings_from_config(cfg)
    s.n_variants = 8; s.de_maxiter = 10; s.de_popsize = 6; s.offset_sigma = 0.01; s.cache_variants = True
    s.quantile_range = (0.05, 0.2); s.knot_spacing_range = (25, 80)
    tab = dp.ensemble(cfg, settings=s, outdir=run.outdir, say=lambda m: None).set_index("unit")
    assert tab.at["CO2", "class"] == "robust" and tab.at["CO2", "detection_fraction"] == 1.0
    assert tab.at["HCN", "class"] == "continuum-dependent" and tab.at["HCN", "nominal_detected"]
    assert 0.05 < tab.at["HCN", "detection_fraction"] < 0.95
    assert tab.at["HCN", "T_spread_16_84"] > tab.at["CO2", "T_spread_16_84"]
    assert os.path.exists(os.path.join(run.outdir, "detection_probability.csv"))
    assert os.path.exists(os.path.join(run.outdir, "detection_prob_variants.json"))
    # rerun uses the cached variants (no refits) and gives the same table
    tab2 = dp.ensemble(cfg, settings=s, outdir=run.outdir, say=lambda m: None).set_index("unit")
    assert tab2.at["HCN", "detection_fraction"] == tab.at["HCN", "detection_fraction"]
    surv = dp.survey_table(os.path.dirname(run.outdir))
    assert len(surv) == 2 and set(surv.target) == {"synth"}


def test_bayesian_mode_small(fake_hcn):
    pytest.importorskip("dynesty")
    cfg, run = fake_hcn
    s = dp.settings_from_config(cfg)
    cfg2 = cfg.model_copy(deep=True)
    cfg2.fit.continuum_fit = "offset"; cfg2.fit.dynesty.nlive = 25; cfg2.fit.dynesty.maxcall = 300; cfg2.fit.dynesty.dynamic = False
    ev = dp.bayesian(cfg2, settings=s, outdir=run.outdir, say=lambda m: None, components=["CO2"]).set_index("unit")
    assert ev.at["CO2", "delta_lnZ"] > 100 and ev.at["CO2", "p_present"] > 0.99 and ev.at["CO2", "class"] == "robust"
