"""0.23 shifted-template null test (jalebi.shift_null, fit.shift_null, jalebi shift-null)."""
import os

import numpy as np
import pandas as pd
import pytest

import jalebi.pipeline as pl
from jalebi.fit import FitProblem, default_free_params
from jalebi.model import Component
from jalebi.shift_null import (ShiftNullSettings, highpass, run_outdir, segment_ids, shift_null, shift_template,
                               survey_table)
from jalebi.synthetic import make_synthetic_spectrum

CO2 = {"name": "CO2", "molecule": "CO2", "logN": 17.3, "T": 520.0, "logR": -0.6}
HCN = {"name": "HCN", "molecule": "HCN", "logN": 16.6, "T": 750.0, "logR": -0.9}


def _fit(truth, seed=1, ar=0.0, extra=0.0):
    """CO2 + HCN fitted (DE) to a 3A-3C synthetic spectrum with `truth`; optional AR(1) correlated noise of
    `extra` x sigma (the pipeline sigma does not know about it, as in real data)."""
    spec, t = make_synthetic_spectrum(components=truth, bands=("3A", "3B", "3C"), snr=150.0, seed=seed, oversample=3)
    spec.continuum = t["continuum"]
    if ar > 0:
        rng = np.random.default_rng(seed + 100)
        g = rng.standard_normal(len(spec.wave)); e = np.zeros_like(g)
        for i in range(1, len(g)):
            e[i] = ar * e[i - 1] + np.sqrt(1 - ar ** 2) * g[i]
        spec.flux = spec.flux + extra * spec.err * e
    comps = [Component(**c) for c in (CO2, HCN)]
    prob = FitProblem(spec, comps, [(13.3, 16.2)], default_free_params(comps), oversample=3, use_pipeline_err=True)
    opt = prob.optimise(prob.theta0(), maxiter=30, popsize=8, seed=seed, polish=True)
    return prob, opt.theta


def test_helpers_segments_shift_and_highpass():
    w = np.concatenate([np.linspace(13.0, 14.0, 400), np.linspace(14.5, 15.5, 400)])     # a gap at 14-14.5
    seg = segment_ids(w)
    assert seg[0] == seg[399] and seg[400] == seg[-1] and seg[0] != seg[-1]
    t = np.exp(-0.5 * ((w - 13.9) / 0.003) ** 2)                 # a line 0.1 um below the gap
    tv, ok = shift_template(w, t, 9000.0, seg)                   # +3 %: the line would land at 14.32 um (in the gap)
    assert not ok[400:420].any() or tv[ok].max() < 0.5           # nothing is interpolated across the gap
    tv0, ok0 = shift_template(w, t, 0.0, seg)
    assert np.allclose(tv0[ok0], t[ok0])
    smooth = 1.0 + 0.1 * (w - 14.0)                              # a smooth pseudo-continuum is removed by the high-pass
    assert np.max(np.abs(highpass(w, smooth, 0.3, seg))) < 1e-3
    assert np.max(highpass(w, t, 0.3, seg)) > 0.9                # a line is kept


def test_present_molecules_are_significant_absent_ones_not():
    prob, th = _fit([CO2, HCN])
    tab, curves = shift_null(prob, th, ShiftNullSettings())
    t = tab.set_index("unit")
    assert t.loc["CO2", "S"] > 50 and t.loc["HCN", "S"] > 5 and t.loc["CO2", "detected_shift"]
    assert set(curves.unit) == {"CO2", "HCN"} and (curves.v_kms == 0).sum() == 2
    assert 0.5 < t.loc["CO2", "null_sigma"] < 2.0                # white noise with the right sigma: null spread ~1
    prob, th = _fit([CO2])
    t = shift_null(prob, th, ShiftNullSettings())[0].set_index("unit")
    assert t.loc["CO2", "S"] > 50 and t.loc["HCN", "S"] < 3 and not t.loc["HCN", "detected_shift"]


def test_correlated_noise_widens_the_null():
    """With AR(1) noise the naive matched-filter S/N (z0, white-noise units) is inflated; the shifted null
    measures the excess (null_sigma > 1) and the absent molecule stays below the threshold."""
    prob, th = _fit([CO2], ar=0.8, extra=3.0, seed=2)
    t = shift_null(prob, th, ShiftNullSettings())[0].set_index("unit")
    assert t.loc["HCN", "null_sigma"] > 1.5
    assert t.loc["HCN", "S"] < 5 and t.loc["CO2", "S"] > 10


def test_pipeline_writes_shift_null_and_resume_key_ignores_it(tmp_path):
    from jalebi.resume import stage_key
    from test_resume import make_config
    cfg = make_config(str(tmp_path / "sn"), nsteps=20)
    key_on = stage_key(cfg)
    c2 = cfg.model_copy(deep=True); c2.fit.shift_null.enabled = False; c2.fit.shift_null.step_kms = 250.0
    assert stage_key(c2) == key_on                                # post-processing: checkpoints stay valid
    run = pl.run_pipeline(cfg)
    out = run.outdir
    assert os.path.exists(os.path.join(out, "shift_null.csv")) and os.path.exists(os.path.join(out, "shift_null_curves.csv"))
    t = pd.read_csv(os.path.join(out, "shift_null.csv"))
    assert {"unit", "S", "z0", "null_sigma", "fap_empirical", "detected_shift", "null_ok"} <= set(t.columns)
    assert any("shift-null test" in l for l in run.log)
    row = pl.catalogue_row(run)
    assert any(k.endswith("_Sshift") for k in row)
    # the CLI path rebuilds the fit from config.yaml + best_fit.json and gives the same numbers
    os.rename(os.path.join(out, "shift_null.csv"), os.path.join(out, "shift_null_pipeline.csv"))
    t2 = run_outdir(out)
    assert np.allclose(t2.sort_values("unit")["S"].values, t.sort_values("unit")["S"].values, rtol=1e-6, equal_nan=True)
    merged = survey_table(os.path.dirname(out))
    assert len(merged) == len(t2)


@pytest.mark.parametrize("disabled", [True])
def test_disabled_writes_nothing(tmp_path, disabled):
    from test_resume import make_config
    cfg = make_config(str(tmp_path / "off"), nsteps=10)
    cfg.fit.shift_null.enabled = False
    run = pl.run_pipeline(cfg)
    assert not os.path.exists(os.path.join(run.outdir, "shift_null.csv"))


def test_narrow_windows_give_no_null():
    """Fit windows narrower than the shifts (line regions): the shifted templates leave the fitted pixels, so no
    fair null exists -> S is NaN and null_ok False instead of a spurious huge S."""
    spec, t = make_synthetic_spectrum(components=[CO2], bands=("3B",), snr=150.0, seed=4, oversample=3)
    spec.continuum = t["continuum"]
    comps = [Component(**CO2)]
    prob = FitProblem(spec, comps, [(14.95, 15.0)], default_free_params(comps), oversample=3, use_pipeline_err=True)
    tab = shift_null(prob, prob.theta0(), ShiftNullSettings())[0]
    assert not tab.null_ok.iloc[0] and np.isnan(tab.S.iloc[0]) and not tab.detected_shift.iloc[0]
