"""0.20 dynesty nested sampling (jalebi.nested, fit.sampler: dynesty)."""
import json

import numpy as np
import pytest

dynesty = pytest.importorskip("dynesty")

from jalebi.fit import FitProblem, Param, default_free_params          # noqa: E402
from jalebi.model import Component                                      # noqa: E402
from jalebi.nested import NestedLikelihood, PriorTransform, run_dynesty   # noqa: E402
from jalebi.synthetic import make_synthetic_spectrum                     # noqa: E402

TRUTH = [{"name": "HCN", "molecule": "HCN", "logN": 17.5, "T": 650.0, "logR": -0.7},
         {"name": "CO2", "molecule": "CO2", "logN": 17.0, "T": 500.0, "logR": -0.5}]


def _problem(truth=TRUTH, comps=None, windows=((13.8, 14.1), (14.85, 15.05)), narrow=True):
    spec, t = make_synthetic_spectrum(components=truth, bands=("3B",), snr=80.0, seed=4, oversample=3)
    spec.continuum = t["continuum"]
    comps = comps or [Component(**c) for c in truth]
    free = []
    for c in comps:
        if narrow:
            free += [Param(c.name, "logN", 16.0, 19.0), Param(c.name, "T", 300.0, 1000.0), Param(c.name, "logR", -1.5, 0.0)]
        else:
            free += default_free_params([c])
    return FitProblem(spec, comps, list(windows), free, oversample=3, use_pipeline_err=True, fit_noise_scale=False)


def test_sorted_transform_is_uniform_on_the_ordered_region():
    comps = [Component("H2O_hot", "H2O"), Component("H2O_warm", "H2O"), Component("H2O_cold", "H2O")]
    free = [Param(c.name, "T", 100.0, 1500.0) for c in comps] + [Param("H2O_hot", "logN", 14.0, 20.0)]

    class P:                                    # just what the transform needs
        pass
    p = P(); p.free = free; p.lo = np.array([x.lo for x in free]); p.hi = np.array([x.hi for x in free])
    p.ordering = [("H2O_hot", "H2O_warm"), ("H2O_warm", "H2O_cold")]
    tr = PriorTransform(p)
    assert len(tr.chains) == 1 and "sorted" in tr.describe()
    rng = np.random.default_rng(0)
    X = np.array([tr(u) for u in rng.random((40000, 4))])
    assert np.all(X[:, 0] >= X[:, 1]) and np.all(X[:, 1] >= X[:, 2])
    # uniform on the ordered simplex: the marginals are the order statistics of 3 uniforms on [100, 1500]
    u = (X[:, :3] - 100.0) / 1400.0
    assert abs(u[:, 0].mean() - 0.75) < 0.01 and abs(u[:, 1].mean() - 0.5) < 0.01 and abs(u[:, 2].mean() - 0.25) < 0.01
    assert abs(((X[:, 3] - 14) / 6).mean() - 0.5) < 0.01               # untouched parameter: plain uniform
    # a Gaussian prior on top of the bounds: truncated-normal inverse CDF
    free[3].gauss = (17.0, 0.5)
    tr = PriorTransform(p)
    Y = np.array([tr(u) for u in rng.random((20000, 4))])
    assert abs(Y[:, 3].mean() - 17.0) < 0.02 and abs(Y[:, 3].std() - 0.5) < 0.02


def test_likelihood_rejects_what_the_prior_forbids():
    prob = _problem()
    prob.ordering = [("HCN", "CO2")]                                    # T_HCN > T_CO2
    like = NestedLikelihood(prob)
    th = prob.theta0()
    assert like(th) == pytest.approx(prob.log_like(th))
    bad = th.copy(); bad[1] = 400.0; bad[4] = 600.0                      # HCN colder than CO2
    assert like(bad) == NestedLikelihood.BAD


@pytest.mark.parametrize("linear", ["sample", "profile"])
def test_dynesty_recovers_the_truth_and_writes_the_usual_chain(linear, tmp_path):
    prob = _problem()
    r = run_dynesty(prob, linear=linear, nlive=150, sample="rwalk", dynamic=False, seed=2)
    assert r.chain.shape[2] == prob.ndim and r.chain.shape[1] >= 2
    s = r.summary().set_index("parameter")
    for c in TRUTH:
        for q in ("T", "logN", "logR"):
            x = s.loc[f"{c['name']}.{q}"]
            assert abs(x["median"] - c[q]) < 3 * max(x["minus"], x["plus"]), (linear, c["name"], q)
    d = r.diagnostics()
    assert d["sampler"] == "dynesty" and np.isfinite(d["logz"]) and d["logzerr"] > 0 and d["likelihood_calls"] == r.ncall
    assert d["linear"] == linear
    r.save(str(tmp_path / "chain.npz"))
    z = np.load(tmp_path / "chain.npz", allow_pickle=False)
    assert {"chain", "log_prob", "names", "linear", "sampled", "logz", "logzerr", "sampler"} <= set(z.files)
    assert str(z["sampler"]) == "dynesty" and [str(x) for x in z["names"]] == [p.key for p in prob.free]


def test_evidence_prefers_the_true_model():
    """ln Z falls when a component the data need is removed and does not rise (Occam) for a superfluous one."""
    from jalebi.nested import evidence_without
    truth = TRUTH[1:]                                                   # CO2 only in the data
    win = ((14.85, 15.05),)
    comps = [Component(**truth[0]), Component("HCN", "HCN", logN=16.5, T=600.0, logR=-1.0)]
    full = _problem(truth, comps, windows=win)
    kw = dict(nlive=80, sample="rwalk", dynamic=False, seed=3)
    r = run_dynesty(full, **kw)

    def factory(name):
        return _problem(truth, [c for c in comps if c.name != name], windows=win)
    ev = evidence_without(factory, ["CO2", "HCN"], r, kw).set_index("component")
    # problem_without keeps the pixels, noise and weights of the full problem
    from jalebi.nested import problem_without
    p2 = problem_without(full, "HCN")
    assert np.array_equal(p2.y, full.y) and np.array_equal(p2.sigma, full.sigma) and p2.ndim == full.ndim - 3
    with pytest.raises(ValueError):
        problem_without(full, "C6H6")
    assert ev.loc["CO2", "delta_lnZ"] > 50                              # CO2 is in the data
    assert ev.loc["HCN", "delta_lnZ"] < 2 * ev.loc["HCN", "delta_lnZ_err"] + 1    # HCN is not: no support


def test_pipeline_sampler_option(tmp_path):
    from jalebi.config import ComponentConfig, ProjectConfig
    from jalebi.pipeline import RunResult, build_problem, run_mcmc_stage, save_results
    spec, t = make_synthetic_spectrum(components=TRUTH[:1], bands=("3B",), snr=80.0, seed=4, oversample=3)
    spec.continuum = t["continuum"]
    cfg = ProjectConfig(components=[ComponentConfig(**TRUTH[0], bounds={"T": [300, 1000], "logN": [16, 19], "logR": [-1.5, 0.0]})])
    cfg.fit.windows = [[13.8, 14.1]]; cfg.fit.oversample = 3; cfg.fit.use_pipeline_err = True; cfg.fit.fit_noise_scale = False
    cfg.fit.sampler = "dynesty"; cfg.fit.mcmc.linear = "profile"
    cfg.fit.dynesty.nlive = 100; cfg.fit.dynesty.dynamic = False; cfg.fit.dynesty.sample = "rwalk"
    run = RunResult(cfg, spec, build_problem(cfg, spec))
    run_mcmc_stage(run, verbose=False)
    assert any("ln Z" in line for line in run.log)
    save_results(run, str(tmp_path))
    d = json.load(open(tmp_path / "diagnostics.json"))
    assert d["sampler"] == "dynesty" and "logz" in d and "logzerr" in d
    import pandas as pd
    assert {"HCN.T", "HCN.R_au", "HCN.logNmol"} <= set(pd.read_csv(tmp_path / "summary.csv").parameter)
    cfg.fit.sampler = "ultranest"
    with pytest.raises(ValueError):
        run_mcmc_stage(RunResult(cfg, spec, build_problem(cfg, spec)), verbose=False)
