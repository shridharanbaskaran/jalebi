"""0.24: sgclip continuum, refine_tol, continuum robustness stage, H I accretion module, pipeline/app integration."""
import os

import numpy as np
import pandas as pd
import pytest

from jalebi import accretion as AC
from jalebi import robustness as RB
from jalebi.config import ProjectConfig
from jalebi.continuum import ContinuumSettings, cont_sgclip
from jalebi.fit import FitProblem, default_free_params
from jalebi.instrument import resolving_power
from jalebi.model import Component
from jalebi.synthetic import make_synthetic_spectrum

CO2 = {"name": "CO2", "molecule": "CO2", "logN": 17.3, "T": 520.0, "logR": -0.6}
HCN = {"name": "HCN", "molecule": "HCN", "logN": 16.6, "T": 750.0, "logR": -0.9}


def test_sgclip_follows_the_continuum_in_emission_and_absorption():
    w = np.linspace(13, 16, 2000)
    c = 1 + 0.1 * (w - 13)
    rng = np.random.default_rng(0)
    lines = np.zeros_like(w)
    for l in rng.uniform(13, 16, 60):
        lines += 0.05 * np.exp(-0.5 * ((w - l) / 0.003) ** 2)
    noise = rng.normal(0, 0.003, len(w))
    use = np.ones(len(w), bool)
    be = cont_sgclip(w, c + lines + noise, use, ContinuumSettings(method="sgclip"))
    ba = cont_sgclip(w, c - lines + noise, use, ContinuumSettings(method="sgclip"))
    assert abs(np.median((be - c) / c)) < 0.01 and abs(np.median((ba - c) / c)) < 0.01
    # explicit interval modes are honoured
    s = ContinuumSettings(method="sgclip", intervals=[{"range": [13.0, 16.0], "mode": "absorption"}])
    assert np.median(cont_sgclip(w, c - lines + noise, use, s) - c) < 0.01


def test_new_keys_keep_old_stage_keys_and_robustness_is_run_only():
    from jalebi.resume import stage_key
    cfg = ProjectConfig()
    k0 = stage_key(cfg, "x")
    c = cfg.model_copy(deep=True)
    c.fit.continuum_robustness.n_perturb = 3; c.fit.accretion.water = "none"; c.target.mstar = 0.5
    assert stage_key(c, "x") == k0                          # post-processing / stellar parameters: run-only
    c.continuum.refine_tol = 0.01
    assert stage_key(c, "x") != k0                          # a fit setting away from its default enters the key


def _fit(truth):
    spec, t = make_synthetic_spectrum(components=truth, bands=("3A", "3B", "3C"), snr=150.0, seed=1, oversample=3)
    spec.continuum = t["continuum"]
    comps = [Component(**c) for c in (CO2, HCN)]
    prob = FitProblem(spec, comps, [(13.3, 16.2)], default_free_params(comps), oversample=3, use_pipeline_err=True)
    opt = prob.optimise(prob.theta0(), maxiter=30, popsize=8, seed=1, polish=True)
    return spec, prob, opt.theta


def test_robustness_classes_and_outputs(tmp_path):
    spec, prob, th = _fit([CO2])                            # HCN fitted but absent
    s = RB.RobustnessSettings(n_perturb=4)
    tab, vt = RB.run(ProjectConfig(), spec, prob, th, outdir=str(tmp_path), settings=s, target="syn", say=lambda m: None)
    t = tab.set_index("unit")
    assert t.loc["CO2", "class"] == "robust" and t.loc["CO2", "frac"] == 1.0
    assert t.loc["HCN", "class"] == "not detected"
    assert len(vt) == 2 * (1 + len(s.estimators) + s.n_perturb)
    for f in ("continuum_robustness.csv", "continuum_variants.csv", "continuum_variants.npz", "robustness.json"):
        assert os.path.exists(os.path.join(tmp_path, f))
    # a rerun with the same fit and settings is served from the checkpoint and gives the same table
    msgs = []
    tab2, _ = RB.run(ProjectConfig(), spec, prob, th, outdir=str(tmp_path), settings=s, target="syn", say=msgs.append)
    assert any("resumed from checkpoint" in m for m in msgs)
    assert np.allclose(tab2.S_nominal, tab.S_nominal)


def test_with_data_changes_only_y():
    spec, prob, th = _fit([CO2, HCN])
    p2 = prob.with_data(prob.y + 1.0)
    assert np.allclose(p2.y, prob.y + 1.0) and p2.model is prob.model and not np.allclose(prob.y, p2.y)
    with pytest.raises(ValueError):
        prob.with_data(prob.y[:-1])


def _hi_spectrum(water_logR, truth=-1.5, d=140.0):
    W = {"name": "H2O_warm", "molecule": "H2O", "logN": 18.3, "T": 650.0, "logR": water_logR}
    spec, t = make_synthetic_spectrum(components=[W], bands=("1B", "1C", "2A"), snr=300, seed=3, oversample=3)
    spec.continuum = t["continuum"]
    for name, cal in AC.CALIBRATIONS.items():
        nu, nl = map(int, name[3:].split("-"))
        lam = AC.hi_wavelength_um(nu, nl)
        if not spec.wave.min() < lam < spec.wave.max():
            continue
        logL = (truth - cal["b"]) / cal["a"] + cal["offset"]
        F = 10 ** logL * AC.L_SUN / (4 * np.pi * (d * AC.PC) ** 2)
        Fj = F / (1e-26 * 1e-6 * AC.C_M / (lam * 1e-6) ** 2)
        sg = 1.2 * lam / resolving_power(np.array([lam]))[0]
        spec.flux = spec.flux + Fj / (sg * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((spec.wave - lam) / sg) ** 2)
    return spec


def test_hi_line_list_and_calibrations():
    lines = AC.hi_lines(4.9, 28.1)
    assert abs(lines.set_index("line").loc["HI_7-6", "wave"] - 12.3719) < 2e-3
    assert abs(lines.set_index("line").loc["HI_10-7", "wave"] - 8.7601) < 2e-3
    assert set(AC.CALIBRATIONS) <= set(lines.line)
    assert AC.CALIBRATIONS["HI_10-7"]["a"] == 0.99 and AC.CALIBRATIONS["HI_7-6"]["a"] == 1.06   # A&A Table 1


def test_accretion_recovers_lacc_through_strong_water():
    spec = _hi_spectrum(-0.6)                          # water carries ~90 % of the flux under Pf alpha
    s = AC.AccretionSettings(water_release="hitran", n_mc=400)
    tab, res = AC.measure(spec, distance_pc=140.0, mstar=0.7, rstar=2.0, settings=s)
    t = tab.set_index("line")
    assert t.loc["HI_6-5", "water_frac"] > 0.5 and t.loc["HI_6-5", "detected"]
    for ln in ("HI_9-6", "HI_6-5", "HI_8-6"):
        assert abs(t.loc[ln, "logLacc"] + 1.5) < 0.15
    assert abs(res["logLacc"] + 1.5) < 0.1 and set(res["HI_lines_used"].split(",")) <= set(s.combine)
    # non-detections are upper limits above the true value
    ul = tab[(tab.calibrated) & (~tab.detected) & np.isfinite(tab.logLacc)]
    assert (ul.logLacc > -1.5).all()
    # Mdot = 1.25 Lacc R*/(G M*)
    md = 1.25 * 10 ** res["logLacc"] * AC.L_SUN * 2.0 * AC.R_SUN / (AC.G * 0.7 * AC.M_SUN) * AC.YEAR / AC.M_SUN
    assert abs(res["logMdot"] - np.log10(md)) < 1e-6
    row = AC.catalogue_columns(tab, res)
    assert "logLacc_HI_9-6" in row and "F_HI_7-6" in row and "logMdot" in row


@pytest.mark.post_stages
def test_pipeline_writes_robustness_and_accretion(tmp_path):
    import jalebi.pipeline as pl
    from test_resume import make_config
    cfg = make_config(str(tmp_path / "pp"), nsteps=20)
    cfg.fit.continuum_robustness.n_perturb = 2
    cfg.fit.continuum_robustness.estimators = ["irsqr_q0.2", "sgclip"]
    cfg.fit.accretion.water = "none"
    run = pl.run_pipeline(cfg)
    for f in ("continuum_robustness.csv", "continuum_variants.csv", "accretion.csv", "accretion.json"):
        assert os.path.exists(os.path.join(run.outdir, f)), f
    row = pl.catalogue_row(run)
    assert any(k.endswith("_robust") for k in row) and "logLacc" in row


def test_app_has_accretion_tab_and_batch_columns():
    pytest.importorskip("panel")
    from jalebi.app import JalebiApp
    a = JalebiApp()
    assert "Accretion" in JalebiApp.TAB_NAMES and hasattr(a, "acc_tab")
    cols = a._batch_columns()
    assert all(f"logLacc_{k}" in cols for k in AC.CALIBRATIONS) and {"logLacc", "logMdot", "mstar", "rstar"} <= set(cols)
    df = pd.DataFrame([{c: np.nan for c in cols}]); df.loc[0, ["logLacc", "mstar", "rstar"]] = [-1.5, 0.7, 2.0]
    a._batch_recompute_mdot(df, 0, {})
    assert np.isfinite(df.loc[0, "logMdot"])
