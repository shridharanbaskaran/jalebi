"""0.22 robust start: several DE starts (fit.optimise.n_starts, seed_points), the lowest -2 ln P wins, every
start is recorded, possible multimodality is flagged."""
import json
import os

import numpy as np

import jalebi.pipeline as pl
from jalebi.fit import FitProblem, default_free_params
from jalebi.model import Component
from jalebi.synthetic import make_synthetic_spectrum

from test_resume import make_config


def test_one_start_is_the_single_optimiser_call(tmp_path):
    """n_starts 1 reproduces the 0.21 behaviour exactly (one DE run with the config seed)."""
    cfg = make_config(str(tmp_path / "a"))
    cfg.fit.stages = ["optimise"]; cfg.continuum.refine_iterations = 0
    run = pl.run_pipeline(cfg)
    o = cfg.fit.optimise
    direct = run.problem.optimise(run.problem.theta0(), method=o.method, maxiter=o.maxiter, popsize=o.popsize,
                                  seed=o.seed, polish=o.polish, workers=1)
    assert np.allclose(run.opt.theta, direct.theta) and abs(run.opt.chi2 - direct.chi2) < 1e-6
    assert len(run.opt.starts) == 1 and run.opt.starts[0]["from"] == "config"


def test_three_starts_keep_the_best_and_record_all(tmp_path):
    cfg = make_config(str(tmp_path / "b"))
    cfg.fit.stages = ["optimise"]; cfg.continuum.refine_iterations = 0
    cfg.fit.optimise.n_starts = 3
    cfg.fit.optimise.seed_points = {"CO2": {"T": 520.0, "logN": 17.3}, "HCN": {"T": 600.0}}
    run = pl.run_pipeline(cfg)
    st = run.opt.starts
    assert [s["from"] for s in st] == ["config", "seed_points", "random"]
    assert [s["seed"] for s in st] == [0, 1, 2]
    energies = [s["energy"] for s in st]
    assert run.opt.best_start == int(np.argmin(energies))
    assert abs(-2.0 * run.opt.log_prob - min(energies)) < 1e-9
    assert any("start 2/3 (seed 1, seed_points)" in l for l in run.log) and any("best start:" in l for l in run.log)
    d = json.load(open(os.path.join(run.outdir, "diagnostics.json")))
    assert d["optimise"]["n_starts"] == 3 and len(d["optimise"]["starts"]) == 3
    assert len(d["optimise"]["starts"][0]["theta"]) == run.problem.ndim
    # the seed-point start really began at the seed values
    th1 = pl.seed_point_theta(run.problem, run.problem.theta0(), cfg.fit.optimise.seed_points)
    P, _ = run.problem.params_from_theta(th1)
    assert P["CO2"]["T"] == 520.0 and P["CO2"]["logN"] == 17.3 and P["HCN"]["T"] == 600.0
    # the checkpoint carries the starts
    ck = json.load(open(os.path.join(run.outdir, "de_pass1.json")))
    assert len(ck["payload"]["starts"]) == 3


def test_multimodality_flag():
    truth = [{"name": "CO2", "molecule": "CO2", "logN": 17.3, "T": 520.0, "logR": -0.6}]
    spec, t = make_synthetic_spectrum(components=truth, bands=("3B",), snr=200.0, seed=3, oversample=3)
    spec.continuum = t["continuum"]
    comps = [Component("CO2", "CO2", logN=17.3, T=520.0, logR=-0.6)]
    prob = FitProblem(spec, comps, [(14.3, 15.5)], default_free_params(comps), oversample=3, use_pipeline_err=True)
    th = prob.theta0()
    far = th.copy(); far[[p.name for p in prob.free].index("T")] += 250.0
    near = th.copy(); near[[p.name for p in prob.free].index("T")] += 20.0
    starts = [{"start": 0, "energy": 100.0, "theta": th.tolist()}, {"start": 1, "energy": 104.0, "theta": far.tolist()},
              {"start": 2, "energy": 101.0, "theta": near.tolist()}, {"start": 3, "energy": 150.0, "theta": far.tolist()}]
    flags = pl.multimodal_flags(prob, starts, 0, dchi2=10.0, dT=100.0, dlogN=0.5)
    assert [f["start"] for f in flags] == [1]                 # start 2 is close in parameters, start 3 is far in energy
    assert "CO2.T" in flags[0]["params"][0]
