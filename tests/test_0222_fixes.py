"""0.22.2 fixes: component_significance with a joint continuum correction (fit.continuum_fit: offset / spline),
and the LinearProblem block solve with cached continuum columns.

Up to 0.22.1 the reduced model (component removed) was compared with the raw data while the full model was
compared with the continuum-corrected data, so delta_chi2 contained the correction's own chi2 gain and came out
nearly the same, large number for every component -- also for a component with no flux.  The corner study
(74 targets, spline setting) showed identical delta_chi2 for all components in 24 targets and 'detections' in
28 of 36 no-disk controls.
"""
import numpy as np

from jalebi.fit import FitProblem, default_free_params
from jalebi.model import Component
from jalebi.synthetic import make_synthetic_spectrum

TRUTH = [{"name": "CO2", "molecule": "CO2", "logN": 17.3, "T": 520.0, "logR": -0.6}]


def _problem(mode, offset_frac=0.015):
    spec, t = make_synthetic_spectrum(components=TRUTH, bands=("3B",), snr=200.0, seed=3, oversample=3)
    spec.continuum = t["continuum"] * (1.0 - offset_frac)            # continuum error the correction can absorb
    comps = [Component("CO2", "CO2", logN=17.3, T=520.0, logR=-0.6),
             Component("HCN", "HCN", logN=14.0, T=500.0, logR=-2.0)]   # not in the data, ~no flux
    prob = FitProblem(spec, comps, [(13.6, 15.5)], default_free_params(comps), oversample=3, use_pipeline_err=True)
    prob.set_continuum_fit(mode, knot_spacing_um=0.5, prior_width=0.05)
    return prob


def _sig(prob):
    s = prob.component_significance(prob.theta0()).set_index("component")
    return s.loc["CO2", "delta_chi2"], s.loc["HCN", "delta_chi2"]


def test_absent_component_is_not_significant_with_spline_correction():
    co2, hcn = _sig(_problem("spline"))
    assert co2 > 100                      # the real component is strongly required
    assert abs(hcn) < 5                   # the empty one is not (0.22.1: ~ the correction's chi2 gain, same as CO2)
    assert abs(co2 - hcn) > 100           # no longer the same number for every component


def test_offset_mode_and_no_correction_agree_on_the_absent_component():
    for mode in ("offset", "none"):
        co2, hcn = _sig(_problem(mode))
        assert co2 > 100 and abs(hcn) < 5


def test_reduced_chi2_reprofiles_the_continuum():
    """delta_chi2 equals chi2(reduced model, continuum re-profiled) - chi2(full model, continuum profiled)."""
    prob = _problem("spline")
    th = prob.theta0()
    P, _ = prob.params_from_theta(th)
    P2 = {k: dict(v) for k, v in P.items()}
    P2["CO2"]["logN"] = -30.0
    expect = prob.chi2(th, model=prob.model.evaluate(P2)) - prob.chi2(th)
    got = prob.component_significance(th).set_index("component").loc["CO2", "delta_chi2"]
    assert np.isclose(got, expect, rtol=1e-9)


def test_linear_block_gram_equals_dense_solve():
    """0.22.2 LinearProblem: cached sparse continuum columns give the same ln L, areas and coefficients as the
    dense (k + kc)-column solve of 0.22.1."""
    from jalebi.linear import LinearProblem
    prob = _problem("spline")
    th = prob.theta0()
    for mode in ("marginalise", "profile"):
        lp = LinearProblem(prob, mode, th)
        tr = lp.reduce(th)
        y, M, log_s = lp.design(tr)
        assert M.shape[1] == lp.k                       # continuum columns are no longer in the design matrix
        new = lp._solve_design(tr, y, M, log_s)
        old_layout = lp._solve_design(tr, y, np.column_stack([M, lp.cont.B]), log_s)   # pre-0.22.2 callers
        assert np.isclose(new["lnL"], old_layout["lnL"], rtol=0, atol=1e-8)
        if mode == "marginalise":                       # explicit dense reference
            s2 = 10.0 ** (2 * log_s)
            Mw = np.column_stack([M, lp.cont.B]) * lp._sw[:, None]; yw = y * lp._sw
            lam2 = np.concatenate([lp.prior_scale ** 2, lp.cont.tau ** 2])
            G = Mw.T @ Mw / s2 + np.diag(1 / lam2)
            x = np.linalg.solve(G, Mw.T @ yw / s2)
            assert np.allclose(new["a"], x[:lp.k], rtol=1e-10) and np.allclose(new["beta"], x[lp.k:], rtol=1e-8, atol=1e-12)
