"""0.22 joint continuum correction (fit.continuum_fit: offset | spline, jalebi.continuum_fit)."""
import os

import numpy as np
import pytest

import jalebi.pipeline as pl
from jalebi.continuum_fit import bspline_basis, continuum_basis, prior_widths
from jalebi.fit import FitProblem, default_free_params
from jalebi.linear import LinearProblem
from jalebi.model import Component
from jalebi.synthetic import make_synthetic_spectrum

TRUTH = [{"name": "CO2", "molecule": "CO2", "logN": 17.3, "T": 520.0, "logR": -0.6}]


def problem_with_wrong_continuum(offset_frac=0.015, mode="none", **kw):
    """CO2 on 3B with the provided continuum too LOW by offset_frac (so the line flux is too high)."""
    spec, t = make_synthetic_spectrum(components=TRUTH, bands=("3B",), snr=200.0, seed=3, oversample=3)
    spec.continuum = t["continuum"] * (1.0 - offset_frac)
    comps = [Component("CO2", "CO2", logN=17.3, T=520.0, logR=-0.6)]
    prob = FitProblem(spec, comps, [(14.3, 15.5)], default_free_params(comps), oversample=3, use_pipeline_err=True)
    prob.set_continuum_fit(mode, **kw)
    return prob, spec, t


def test_basis_shapes_and_priors():
    x = np.linspace(14.0, 15.5, 300)
    B = bspline_basis(x, 0.5)
    assert B.shape[0] == 300 and B.shape[1] >= 4 and np.allclose(B.sum(axis=1), 1.0)   # partition of unity
    band = np.array(["3B"] * 150 + ["3C"] * 150)
    Bo, lo = continuum_basis(x, band, "offset")
    assert Bo.shape == (300, 2) and lo == ["3B:offset", "3C:offset"] and Bo[:150, 0].all() and not Bo[150:, 0].any()
    Bs, ls = continuum_basis(x, band, "spline", 0.5)
    assert Bs.shape[1] >= 8 and all(l.startswith(("3B:", "3C:")) for l in ls)
    tau = prior_widths(Bo, np.full(300, 0.5), np.full(300, 0.01), "continuum", 0.02)
    assert np.allclose(tau, 0.01)
    tau = prior_widths(Bo, None, np.full(300, 0.01), "noise", 3.0)
    assert np.allclose(tau, 0.03)
    with pytest.raises(ValueError):
        continuum_basis(x, band, "polynomial")


def test_offset_is_recovered_by_profile_and_marginalise_and_de():
    prob, spec, t = problem_with_wrong_continuum(0.015, "offset", prior_width=0.05)
    th = prob.theta0()
    # the correction at the true parameters recovers the injected offset (continuum is 1.5 % low -> y too high
    # by 0.015 x continuum -> the correction is + 0.015 x continuum on average)
    beta, corr = prob.continuum_solution(th)
    cont = np.asarray(spec.continuum)[prob.used]
    assert abs(np.median(corr / cont) - 0.015 / 0.985) < 0.004
    # the likelihood with the correction beats the one without, at the true parameters
    prob0, _, _ = problem_with_wrong_continuum(0.015, "none")
    assert prob.chi2(th) < prob0.chi2(th) - 50
    # LinearProblem: profile and marginalise carry the extra column and give the same coefficient
    for mode in ("profile", "marginalise"):
        lp = LinearProblem(prob, mode, th)
        assert lp.kc == 1 and lp.k == 1
        s = lp.solve(lp.reduce(th))
        assert np.isfinite(s["lnL"]) and abs(s["beta"][0] - beta[0]) < 0.3 * abs(beta[0])
        assert s["a"].shape == (1,) and s["draw"].shape == (1,)
    # DE with the correction lands closer to the truth than without
    opt = prob.optimise(th, maxiter=30, popsize=8, seed=1, polish=True)
    opt0 = prob0.optimise(th, maxiter=30, popsize=8, seed=1, polish=True)
    P, _ = prob.params_from_theta(opt.theta); P0, _ = prob0.params_from_theta(opt0.theta)
    err = abs(P["CO2"]["logN"] - 17.3) + abs(P["CO2"]["T"] - 520.0) / 100.0 + abs(P["CO2"]["logR"] + 0.6)
    err0 = abs(P0["CO2"]["logN"] - 17.3) + abs(P0["CO2"]["T"] - 520.0) / 100.0 + abs(P0["CO2"]["logR"] + 0.6)
    assert err < err0 and opt.chi2 < opt0.chi2


def test_spline_pipeline_products_and_sampler(tmp_path):
    from test_resume import make_config
    cfg = make_config(str(tmp_path / "cf"), nsteps=30)
    cfg.continuum.refine_iterations = 0
    cfg.fit.continuum_fit = "spline"; cfg.fit.continuum_correction.knot_spacing_um = 0.5
    cfg.fit.continuum_correction.prior_width = 0.03; cfg.fit.mcmc.linear = "marginalise"
    run = pl.run_pipeline(cfg)
    assert any("continuum correction (fit.continuum_fit): spline" in l for l in run.log)
    import pandas as pd
    m = pd.read_csv(os.path.join(run.outdir, "model.csv"))
    assert "continuum_correction" in m and np.isfinite(m.continuum_correction).all()
    cf = pd.read_csv(os.path.join(run.outdir, "continuum_fit.csv"))
    assert len(cf) == run.problem.cont.n and (cf.prior_sigma > 0).all()
    import json
    d = json.load(open(os.path.join(run.outdir, "diagnostics.json")))
    assert d["continuum_fit"]["mode"] == "spline" and d["continuum_fit"]["n_coefficients"] == len(cf)
    assert d["continuum_fit"]["max_abs_correction_over_continuum"] < 0.2
    # sample mode (areas sampled) also runs with the correction profiled inside the likelihood
    cfg.fit.mcmc.linear = "sample"; cfg.fit.mcmc.vectorize = True; cfg.fit.resume = "off"
    run2 = pl.run_pipeline(cfg)
    assert np.isfinite(run2.mcmc.log_prob).all()
    # off by default: no column, no file
    cfg.fit.continuum_fit = "none"
    cfg.output = os.path.join(str(tmp_path / "cf"), "out_none", "{target}")
    run3 = pl.run_pipeline(cfg)
    assert "continuum_correction" not in pd.read_csv(os.path.join(run3.outdir, "model.csv"))
    assert run3.problem.cont is None
