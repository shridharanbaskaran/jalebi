"""0.19 Laplace (Gaussian) errors at the optimum: jalebi.laplace / jalebi.fit.laplace."""
import json

import numpy as np

from jalebi.fit import FitProblem, default_free_params, laplace
from jalebi.model import Component
from jalebi.synthetic import make_synthetic_spectrum

TRUTH = [{"name": "HCN", "molecule": "HCN", "logN": 17.5, "T": 650.0, "logR": -0.7},
         {"name": "CO2", "molecule": "CO2", "logN": 17.0, "T": 500.0, "logR": -0.5}]
WINDOWS = [(13.7, 14.1), (14.8, 15.05)]


def _problem(extra=(), area_param="logNA"):
    spec, t = make_synthetic_spectrum(components=TRUTH, bands=("3A", "3B"), snr=100.0, seed=4, oversample=3)
    spec.continuum = t["continuum"]
    comps = [Component(**c) for c in TRUTH] + list(extra)
    return FitProblem(spec, comps, WINDOWS, default_free_params(comps, area_param=area_param), oversample=3,
                      use_pipeline_err=True, fit_noise_scale=True, area_param=area_param)


def test_laplace_sigmas_match_a_converged_mcmc():
    prob = _problem()
    opt = prob.optimise(maxiter=40, popsize=10, seed=0)
    lap = laplace(prob, opt.theta)
    assert lap.meta["positive_definite"] and not lap.directions and np.isfinite(lap.condition)
    assert not any(lap.flags.values())
    res = prob.mcmc(opt.theta, nsteps=3400, seed=1, moves="de", init="scaled")
    tau = res.autocorr_time()
    assert res.nsteps >= 50 * np.nanmax(tau)                                # converged length
    s = res.summary().set_index("parameter")
    for i, n in enumerate(lap.names):                                       # all well constrained here
        sm = 0.5 * (s.loc[n, "minus"] + s.loc[n, "plus"])
        assert abs(lap.sigma[i] / sm - 1) < 0.3, (n, lap.sigma[i], sm)
    # derived R and N_mol from the Gaussian draws agree with the chain as well
    for n in ("HCN.R_au", "CO2.logR", "HCN.logNmol"):
        m, a, b = lap.derived[n]
        sm = 0.5 * (s.loc[n, "minus"] + s.loc[n, "plus"])
        assert abs(0.5 * (a + b) / sm - 1) < 0.3 and abs(m - s.loc[n, "median"]) < sm, n
    # profiled areas (0.17): the same sigmas for the nonlinear parameters, areas as derived quantities
    lp = laplace(prob, opt.theta, linear="profile")
    for i, n in enumerate(lp.names):
        j = lap.names.index(n)
        assert abs(lp.sigma[i] / lap.sigma[j] - 1) < 0.1, n
    assert {"HCN.logNA", "CO2.logNA"} <= set(lp.derived)
    assert abs(0.5 * sum(lp.derived["CO2.logNA"][1:]) / lap.sigma[lap.names.index("CO2.logNA")] - 1) < 0.3


def test_flags_for_a_parameter_at_its_bound():
    """C2H2 at its log N lower bound contributes nothing: its parameters are flat / unconstrained and at the edge,
    the Hessian is regularised to the prior and the flat direction names them."""
    prob = _problem([Component("C2H2", "C2H2", logN=13.0, T=600.0, logR=-1.0)], area_param="logR")
    lap = laplace(prob, prob.theta0())
    f = lap.flags
    assert "edge" in f["C2H2.logN"] and "one-sided" in f["C2H2.logN"]
    assert any(x.startswith(("flat", "saddle")) for x in f["C2H2.logN"] + f["C2H2.T"])
    assert "unconstrained" in f["C2H2.T"]
    assert lap.directions and all(set(d["loads"]) <= {"C2H2.logN", "C2H2.T", "C2H2.logR"} for d in lap.directions)
    assert lap.meta["regularised"]
    span = prob.hi - prob.lo
    assert np.all(lap.sigma <= span / np.sqrt(12) * 1.0001)                # never wider than the prior
    for n in ("HCN.T", "CO2.logN"):                                         # the real components stay clean
        assert not lap.flags[n]


def test_json_and_plots(tmp_path):
    from jalebi import plots
    prob = _problem()
    lap = laplace(prob, prob.theta0())
    lap.save(str(tmp_path / "laplace.json"))
    d = json.load(open(tmp_path / "laplace.json"))
    assert d["names"] == lap.names and len(d["cov"]) == len(lap.names) and "condition_number" in d
    res = prob.mcmc(prob.theta0(), nsteps=60, seed=0)
    fig = plots.plot_laplace_corner(lap, mcmc=res)
    fig.savefig(tmp_path / "c.png")
    fig = plots.plot_laplace_correlation(lap)
    fig.savefig(tmp_path / "r.png")
    assert (tmp_path / "c.png").stat().st_size > 10000


def test_cli_fit_laplace(tmp_path, monkeypatch):
    import yaml
    from typer.testing import CliRunner
    from jalebi.cli import app
    monkeypatch.chdir(tmp_path)
    r = CliRunner()
    assert r.invoke(app, ["init", "c.yaml", "--example", "synthetic"]).exit_code == 0
    d = yaml.safe_load(open("c.yaml"))
    d["components"] = [c for c in d["components"] if c["molecule"] in ("HCN", "CO2")]
    d["fit"]["windows"] = [[13.8, 14.1], [14.85, 15.05]]
    d["fit"]["auto_detect"] = False
    d["fit"]["optimise"] = {"maxiter": 15, "popsize": 6}
    d["output"] = str(tmp_path / "out")
    yaml.safe_dump(d, open("c.yaml", "w"))
    out = r.invoke(app, ["fit", "c.yaml", "--stages", "optimise", "--laplace"])
    assert out.exit_code == 0, out.output
    assert "Laplace errors" in out.output
    for f in ("laplace.json", "laplace_corner.png", "laplace_correlation.png", "laplace_summary.csv"):
        assert (tmp_path / "out" / f).exists(), f


def test_app_button_runs_in_the_background():
    import time
    from jalebi.app import JalebiApp
    from jalebi.config import ComponentConfig
    from jalebi.pipeline import RunResult, build_problem, run_optimise_stage
    app = JalebiApp()
    assert app.laplace_btn.disabled
    spec, t = make_synthetic_spectrum(components=TRUTH, bands=("3B",), snr=100.0, seed=4, oversample=3)
    spec.continuum = t["continuum"]
    cfg = app.cfg.model_copy(deep=True)
    cfg.components = [ComponentConfig(**c) for c in TRUTH]
    cfg.fit.windows = [list(w) for w in WINDOWS]; cfg.fit.oversample = 3; cfg.fit.use_pipeline_err = True
    cfg.fit.optimise.maxiter = 10; cfg.fit.optimise.popsize = 6
    run = RunResult(cfg, spec, build_problem(cfg, spec))
    run_optimise_stage(run)
    app.run = run
    app.laplace_btn.disabled = False
    t0 = time.time()
    app.start_laplace()                       # returns at once: the work is in a thread
    assert time.time() - t0 < 0.5
    app._lap_thread.join(60)
    assert app.lap_section.visible and len(app.lap_table.value) >= 6
    assert app.lap_corner.object is not None and app.lap_corr.object is not None
    assert not app.laplace_btn.disabled
