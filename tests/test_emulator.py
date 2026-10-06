"""0.18 emulator backend: (T, log N) tables of the slab fluxes on the data's pixels (jalebi.emulator).

The full FZ Tau accuracy campaign (H2O hot / warm / cold, CO, CO2 + 13CO2, C2H2, HCN, OH; 2000 random points
each on 4.9-5.35 and 9-27.5 um) builds ~0.5 GB of tables and takes about an hour on one core: it runs with
JALEBI_EMULATOR_FULL=1 (runs/emulator_report.py accuracy is the same code).  The default suite checks
CO2 + 13CO2 and HCN on FZ Tau's 13.6-15.2 um pixels with 2000 points each, plus the mechanics below.
"""
import os
import sys

import numpy as np
import pytest

from jalebi.emulator import EmulatorSettings, errors, exact_rows
from jalebi.fit import FitProblem, Param, default_free_params
from jalebi.model import Component
from jalebi.synthetic import make_synthetic_spectrum

TRUTH = [{"name": "HCN", "molecule": "HCN", "logN": 17.5, "T": 650.0, "logR": -0.7},
         {"name": "CO2", "molecule": "CO2", "logN": 17.0, "T": 500.0, "logR": -0.5}]
WINDOWS = [(13.7, 14.1), (14.8, 15.05)]


def _settings(tmp_path, **kw):
    return EmulatorSettings(cache_dir=str(tmp_path / "emu"), n_validate=60, **kw)


def _narrow(comps, T=(300.0, 900.0), N=(15.0, 19.0)):
    """Free parameters with a narrow box (fast builds)."""
    free = []
    for c in comps:
        if c.tie_to or not c.enabled:
            if c.tie_to:
                free.append(Param(c.name, "ratio", 10.0, 300.0, init=c.ratio))
            continue
        if c.kind in ("slab", "annuli") and not (c.group and c.name != next(d.name for d in comps if d.group == c.group)):
            free += [Param(c.name, "logN", *N, init=c.logN), Param(c.name, "T", *T, init=c.T),
                     Param(c.name, "logR", -2.5, 1.5, init=c.logR)]
        elif c.group:
            free.append(Param(c.name, "logN", *N, init=c.logN))
        else:
            free += [Param(c.name, "logN", 13.0, 20.0, init=c.logN), Param(c.name, "T", 50.0, 900.0, init=c.T),
                     Param(c.name, "rv", -30.0, 30.0, init=c.rv), Param(c.name, "fc", 0.0, 1.0, init=c.fc)]
    return free


def _two_molecule_problem(rv=0.0):
    spec, t = make_synthetic_spectrum(components=TRUTH, bands=("3B",), snr=100.0, seed=4, oversample=3)
    spec.continuum = t["continuum"]
    comps = [Component(**c, rv=rv) for c in TRUTH]
    return FitProblem(spec, comps, WINDOWS, _narrow(comps), oversample=3, use_pipeline_err=True, fit_noise_scale=True)


# ---------------------------------------------------------------------------------------------------
def test_defaults():
    from jalebi.config import ProjectConfig
    c = ProjectConfig()
    assert c.fit.model_backend == "exact" and c.fit.mcmc.vectorize is False
    s = c.fit.emulator.settings()
    assert (s.target_sigma, s.target_flux, s.method) == (0.1, 1e-3, "cubic")


def test_exact_rows_match_the_model_with_shift_and_windows():
    spec, t = make_synthetic_spectrum(components=TRUTH, bands=("3B",), snr=100.0, seed=4, oversample=3)
    comps = [Component("HCN", "HCN", logN=17.2, T=640.0, logR=-0.7, rv=3.5, windows=[[13.75, 14.05]]),
             Component("CO2", "CO2", logN=17.0, T=500.0, logR=-0.5, fwhm_thermal=True)]
    from jalebi.model import build_model
    m = build_model(comps, spec.wave, 140.0, WINDOWS, oversample=3)
    P = m.resolve_params()
    uf, tm, _ = m.unit_fluxes(P)
    for c in comps:
        F, tmax = exact_rows(m, c, P[c.name], P[c.name]["T"], [P[c.name]["logN"], 15.0])
        assert np.allclose(F[0], uf[c.name], rtol=1e-12, atol=1e-300)
        assert abs(tmax[0] - tm[c.name]) < 1e-9 * tm[c.name]
    F, _ = exact_rows(m, comps[0], P["HCN"], 640.0, [15.0])
    assert np.all(F[0][~m.window_mask(comps[0])] == 0) and np.any(F[0] > 0)       # the component's own windows


def test_tables_meet_the_target_and_are_cached(tmp_path):
    prob = _two_molecule_problem(rv=2.0)
    em = prob.use_emulator(_settings(tmp_path))
    assert set(em.tables) == {"HCN", "CO2"} and not em.exact_units
    sig = prob.sigma / np.sqrt(prob.weights); f_ref = float(np.max(np.abs(prob.y)))
    bounds, amax = prob.emulator_bounds()
    rng = np.random.default_rng(3)
    P = prob.model.resolve_params()
    for key, tab in em.tables.items():
        comp = next(c for c in prob.components if c.name == key)
        T = np.exp(rng.uniform(np.log(300.0), np.log(900.0), 300)); N = rng.uniform(15.0, 19.0, 300)
        es, ef = [], []
        for t_, n_ in zip(T, N):
            Fx, _ = exact_rows(prob.model, comp, P[key], t_, [n_])
            a, b = errors(tab.flux(t_, n_)[0], Fx, sig, f_ref, amax[key])
            es.append(a[0]); ef.append(b[0])
        assert max(es) < 0.1 and max(ef) < 1e-3, (key, max(es), max(ef))
        assert tab.meta["validation"]["max_sigma"] < 0.1
    # second attach: from the cache, same numbers
    prob2 = _two_molecule_problem(rv=2.0)
    em2 = prob2.use_emulator(_settings(tmp_path))
    assert all(t.meta["from_cache"] for t in em2.tables.values())
    a = prob.model.evaluate(); b = prob2.model.evaluate()
    assert np.array_equal(a, b)
    files = set(os.listdir(tmp_path / "emu"))
    # anything that changes the answer changes the key: velocity, bounds, pixels, noise
    prob3 = _two_molecule_problem(rv=3.0)
    em3 = prob3.use_emulator(_settings(tmp_path))
    assert not any(t.meta["from_cache"] for t in em3.tables.values())
    prob4 = _two_molecule_problem(rv=2.0)
    prob4.free[1].hi = 950.0                                     # HCN.T upper bound
    em4 = prob4.use_emulator(_settings(tmp_path))
    assert not em4.tables["HCN"].meta["from_cache"] and em4.tables["CO2"].meta["from_cache"]
    assert len(set(os.listdir(tmp_path / "emu"))) == len(files) + 3


def test_cache_key_covers_linelist_pixels_lsf_and_version(tmp_path, monkeypatch):
    from jalebi import emulator as E
    prob = _two_molecule_problem()
    m = prob.model
    c = prob.components[0]
    p = m.resolve_params()[c.name]
    b = {"T": (300.0, 900.0), "logN": (15.0, 19.0)}
    st = _settings(tmp_path)
    k0 = E.table_key(m, c, p, b, prob.sigma, 1.0, 10.0, st)
    assert k0 == E.table_key(m, c, p, b, prob.sigma, 1.0, 10.0, st)
    monkeypatch.setattr(E, "LSF_VERSION", "other")
    assert E.table_key(m, c, p, b, prob.sigma, 1.0, 10.0, st) != k0
    monkeypatch.undo()
    import jalebi
    monkeypatch.setattr(jalebi, "__version__", "9.9.9")
    assert E.table_key(m, c, p, b, prob.sigma, 1.0, 10.0, st) != k0
    monkeypatch.undo()
    basis = m.basis(c, p["fwhm"])
    a_old = basis.a.copy()
    basis.a[0] *= 1.01                                             # a changed line list
    assert E.table_key(m, c, p, b, prob.sigma, 1.0, 10.0, st) != k0
    basis.a[:] = a_old
    m.wave_pix = m.wave_pix * (1 + 1e-9)                            # another pixel grid
    assert E.table_key(m, c, p, b, prob.sigma, 1.0, 10.0, st) != k0


def test_mixed_exact_and_emulated_units(tmp_path):
    """Opacity group, annuli and absorber stay exact, the slab and the tied isotopologue are emulated."""
    spec, t = make_synthetic_spectrum(components=TRUTH, bands=("3B",), snr=100.0, seed=2, oversample=3)
    spec.continuum = t["continuum"]
    comps = [Component("CO2", "CO2", logN=17.0, T=500.0, logR=-0.5),
             Component("13CO2", "13CO2", tie_to="CO2", ratio=70.0),
             Component("C2H2", "C2H2", logN=16.5, T=600.0, logR=-0.9, group="acet"),
             Component("13CCH2", "13CCH2", logN=15.0, T=600.0, logR=-0.9, group="acet"),
             Component("HCN", "HCN", logN=17.0, T=700.0, logR=-0.6, kind="annuli", logRin=-1.3, n_annuli=6),
             Component("HCNabs", "HCN", logN=15.5, T=300.0, kind="absorption", fc=0.5, rv=-5.0)]
    prob = FitProblem(spec, comps, [(13.6, 15.2)], _narrow(comps), oversample=3, use_pipeline_err=True, fit_noise_scale=True)
    th = prob.theta0()
    with prob.model.exact():
        lp_exact = prob.log_prob(th)
    em = prob.use_emulator(_settings(tmp_path))
    assert set(em.tables) == {"CO2", "13CO2"}
    assert set(em.exact_units) == {"acet", "HCN", "HCNabs"}
    assert "group" in em.exact_units["acet"] and "annuli" in em.exact_units["HCN"]
    lp_emu = prob.log_prob(th)
    assert abs(lp_emu - lp_exact) < 0.05 * np.sqrt(len(prob.y))
    # the 13CO2 table covers the shifted log N range of a free ratio in [10, 300]
    tab = em.tables["13CO2"]
    assert tab.logN[0] <= 15.0 - np.log10(300.0) + 1e-9 and tab.logN[-1] >= 19.0 - np.log10(10.0) - 1e-9


def test_outside_the_table_falls_back_to_exact(tmp_path):
    prob = _two_molecule_problem()
    em = prob.use_emulator(_settings(tmp_path))
    P = prob.model.resolve_params()
    P["HCN"]["T"] = 1200.0                                        # outside 300-900 K
    with prob.model.exact():
        ref = prob.model.unit_fluxes(P)[0]
    got = prob.model.unit_fluxes(P)[0]
    assert np.array_equal(got["HCN"], ref["HCN"]) and em.misses >= 1        # HCN exact, CO2 still emulated
    assert not np.array_equal(got["CO2"], ref["CO2"])
    P["HCN"]["T"] = 650.0; P["HCN"]["rv"] = 4.0                     # another velocity: exact as well
    with prob.model.exact():
        ref = prob.model.unit_fluxes(P)[0]
    assert np.array_equal(prob.model.unit_fluxes(P)[0]["HCN"], ref["HCN"])


def test_vectorised_log_prob_and_sampler(tmp_path):
    prob = _two_molecule_problem()
    prob.use_emulator(_settings(tmp_path))
    rng = np.random.default_rng(0)
    TH = np.clip(prob.theta0() + 0.01 * (prob.hi - prob.lo) * rng.standard_normal((32, prob.ndim)), prob.lo, prob.hi)
    a = np.array([prob.log_prob(t) for t in TH]); b = prob.log_prob_many(TH)
    assert np.allclose(a, b, atol=1e-3, rtol=0)
    for linear in ("sample", "profile"):
        r = prob.mcmc(prob.theta0(), nsteps=30, seed=1, moves="de", linear=linear, vectorize=True)
        assert np.all(np.isfinite(r.chain)) and r.meta["vectorize"]


def test_pipeline_backend_and_exact_products(tmp_path):
    from jalebi.config import ComponentConfig, ProjectConfig
    from jalebi.pipeline import RunResult, build_problem, run_mcmc_stage, save_results
    spec, t = make_synthetic_spectrum(components=TRUTH, bands=("3B",), snr=100.0, seed=4, oversample=3)
    spec.continuum = t["continuum"]
    cfg = ProjectConfig(components=[ComponentConfig(**c, bounds={"T": [300.0, 900.0], "logN": [15.0, 19.0]}) for c in TRUTH])
    cfg.fit.windows = [list(w) for w in WINDOWS]
    cfg.fit.oversample = 3; cfg.fit.use_pipeline_err = True
    cfg.fit.model_backend = "emulator"; cfg.fit.emulator.cache_dir = str(tmp_path / "emu"); cfg.fit.emulator.n_validate = 40
    cfg.fit.mcmc.nsteps = 30; cfg.fit.mcmc.linear = "profile"; cfg.fit.mcmc.checkpoint = False; cfg.fit.mcmc.vectorize = True
    log = []
    prob = build_problem(cfg, spec, say=log.append)
    assert prob.model.emulator is not None and any("emulator ready" in l for l in log)
    run = RunResult(cfg, spec, prob)
    run_mcmc_stage(run, verbose=False)
    save_results(run, str(tmp_path / "out"))
    import pandas as pd
    m = pd.read_csv(tmp_path / "out" / "model.csv")
    with prob.model.exact():
        exact = prob.model_flux(run.theta)
    assert np.allclose(m.model.values, exact, rtol=1e-12, atol=1e-15)       # written with the exact model
    assert prob.model.emulator is not None                                 # and the emulator is back afterwards
    with pytest.raises(ValueError):
        cfg.fit.model_backend = "gpu"; build_problem(cfg, spec)


def test_cli_emulator_build_and_list(tmp_path, monkeypatch):
    from typer.testing import CliRunner
    from jalebi.cli import app
    monkeypatch.chdir(tmp_path)
    r = CliRunner()
    assert r.invoke(app, ["init", "c.yaml", "--example", "synthetic"]).exit_code == 0
    import yaml
    d = yaml.safe_load(open("c.yaml"))
    d["components"] = [c for c in d["components"] if c["molecule"] in ("HCN",)]
    for c in d["components"]:
        c["bounds"] = {"T": [400.0, 800.0], "logN": [15.5, 18.5]}
    d.setdefault("fit", {})["windows"] = [[13.8, 14.1]]
    d["fit"]["auto_detect"] = False
    d["fit"]["emulator"] = {"n_validate": 20}
    yaml.safe_dump(d, open("c.yaml", "w"))
    out = r.invoke(app, ["emulator", "build", "c.yaml", "--cache-dir", str(tmp_path / "emu")])
    assert out.exit_code == 0, out.output
    assert "HCN" in out.output and len(os.listdir(tmp_path / "emu")) == 1
    out = r.invoke(app, ["emulator", "list", "--cache-dir", str(tmp_path / "emu")])
    assert out.exit_code == 0 and "HCN" in out.output


# ---------------------------------------------------------------------------------------------------
# FZ Tau
# ---------------------------------------------------------------------------------------------------
def _report():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "runs"))
    import emulator_report
    return emulator_report


def test_fz_tau_co2_13co2_hcn_2000_points(tmp_path):
    """FZ Tau pixels (13.6-15.2 um), default bounds (T 100-1500 K, log N 13-21; 13CO2 tied with ratio 70):
    2000 random points per unit, max error < 0.1 sigma and < 0.1 % in integrated flux."""
    rep = _report()
    from jalebi.emulator import EmulatorSettings
    _, spec = rep.fz_tau_spectrum()
    comps = [Component("CO2", "CO2", logN=17.0, T=500.0, logR=-0.6),
             Component("13CO2", "13CO2", tie_to="CO2", ratio=70.0),
             Component("HCN", "HCN", logN=16.5, T=600.0, logR=-0.8)]
    free = [p for p in default_free_params(comps) if p.name != "ratio"]
    prob = FitProblem(spec, comps, [(13.6, 15.2)], free, oversample=4, fit_noise_scale=True)
    em = prob.use_emulator(EmulatorSettings(cache_dir=os.environ.get("JALEBI_EMULATOR_TEST_CACHE", str(tmp_path / "emu")), n_validate=50))
    sig = prob.sigma / np.sqrt(prob.weights); f_ref = float(np.max(np.abs(prob.y)))
    bounds, amax = prob.emulator_bounds()
    P = prob.model.resolve_params()
    rng = np.random.default_rng(11)
    for key in ("CO2", "13CO2", "HCN"):
        tab = em.tables[key]; comp = next(c for c in comps if c.name == key); b = bounds[key]
        T = np.exp(rng.uniform(np.log(b["T"][0]), np.log(b["T"][1]), 2000)); N = rng.uniform(*b["logN"], 2000)
        es = np.empty(2000); ef = np.empty(2000)
        for i in range(2000):
            Fx, _ = exact_rows(prob.model, comp, P[key], T[i], [N[i]])
            a, f = errors(tab.flux(T[i], N[i])[0], Fx, sig, f_ref, amax[key])
            es[i], ef[i] = a[0], f[0]
        assert es.max() < 0.1 and ef.max() < 1e-3, (key, es.max(), np.percentile(es, 99), ef.max())


@pytest.mark.skipif(os.environ.get("JALEBI_EMULATOR_FULL") != "1", reason="full FZ Tau campaign: set JALEBI_EMULATOR_FULL=1 (~1 h)")
def test_fz_tau_full_accuracy_campaign(tmp_path):
    rep = _report()
    tab = rep.accuracy(n=2000, out=str(tmp_path / "rep"), cache_dir=os.environ.get("JALEBI_EMULATOR_TEST_CACHE"))
    assert len(tab) == 9
    bad = tab[~tab["pass"]]
    assert bad.empty, bad.to_string()
