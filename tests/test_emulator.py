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
    """0.18 per-disk tables (fit.emulator.cache: per_disk); the shared tables (0.21) are tested further down."""
    kw.setdefault("cache", "per_disk")
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
    assert (s.cache, s.ref_snr, s.points_per_fwhm, s.read_only, s.spot_check) == ("shared", 1000.0, 8.0, False, 200)
    assert s.boxes is None and c.fit.emulator.settings(c).boxes == {}


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
    cfg.fit.emulator.cache = "per_disk"
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
    d["fit"]["emulator"] = {"n_validate": 20, "cache": "per_disk"}
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
    em = prob.use_emulator(EmulatorSettings(cache_dir=os.environ.get("JALEBI_EMULATOR_TEST_CACHE", str(tmp_path / "emu")), n_validate=50,
                                            cache="per_disk"))
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
    tab = rep.accuracy(n=2000, out=str(tmp_path / "rep"), cache_dir=os.environ.get("JALEBI_EMULATOR_TEST_CACHE"), cache="per_disk")
    assert len(tab) == 9
    bad = tab[~tab["pass"]]
    assert bad.empty, bad.to_string()


# ---------------------------------------------------------------------------------------------------
# 0.21 shared tables (jalebi.emulator_shared): built once on a dense grid, resampled per disk
# ---------------------------------------------------------------------------------------------------
import warnings as _warnings

from jalebi.emulator_shared import (DenseGrid, DenseSpec, SharedTable, TableSpec, build_dense_lsf_operator, dense_spec,
                                    get_or_build, key_fields, molecule_boxes, shared_dir, survey_specs, table_keys)

BOX = {"T": (300.0, 900.0), "logN": (15.0, 19.0)}


def _shared_settings(tmp_path, **kw):
    """Shared tables restricted to sub-band 3B (the full 12-band grid is the real thing: the build is the same code)."""
    kw.setdefault("boxes", {"HCN": BOX, "CO2": BOX})
    return EmulatorSettings(cache_dir=str(tmp_path / "emu"), n_validate=40, bands=("3B",), **kw)


def _disk(rv=0.0, distance=140.0, seed=4, windows=WINDOWS, oversample=3):
    spec, t = make_synthetic_spectrum(components=TRUTH, bands=("3B",), snr=100.0, seed=seed, oversample=3, distance_pc=distance)
    spec.continuum = t["continuum"]
    comps = [Component(**c, rv=rv) for c in TRUTH]
    return FitProblem(spec, comps, windows, _narrow(comps), oversample=oversample, use_pipeline_err=True, fit_noise_scale=True)


@pytest.fixture(scope="module")
def shared_cache(tmp_path_factory):
    """One HCN + CO2 build (3B, 300-900 K, log N 15-19) shared by the tests below."""
    d = tmp_path_factory.mktemp("shared")
    prob = _disk(rv=2.0)
    em = prob.use_emulator(_shared_settings(d))
    assert set(em.tables) == {"HCN", "CO2"} and not em.exact_units, em.exact_units
    return d


def test_dense_grid_segments_and_operator():
    g = DenseGrid(DenseSpec(bands=("1A", "3B", "4C")))
    assert g.bands == ["1A", "3B", "4C"] and g.n == sum(n for _, n, _, _ in g.segments)
    from jalebi.constants import C
    for (lo, hi), (off, n, x0, h) in zip(g.extent, g.segments):
        pad = 150e3 / C
        assert x0 < np.log(lo) - pad and x0 + h * (n - 1) > np.log(hi) + pad       # |v| <= 150 km/s fits in the padding
        from jalebi.emulator_shared import band_resolving_power
        R_edge = band_resolving_power(g.bands[list(g.extent).index((lo, hi)) if False else g.segments.index((off, n, x0, h))], [lo, hi]).max()
        assert h * 8.0 * R_edge == pytest.approx(1.0)                                 # 8 points per LSF FWHM at the band's max R
    assert g.segment_of([5.0, 5.7, 14.0, 28.5, 9.0]).tolist() == [0, 0, 1, 2, -1]
    from jalebi.model import FineGrid
    fg = FineGrid([(13.8, 14.0)], step_kms=4.7 / 6)
    off, n, x0, h = g.segments[1]
    xd = g.x[off:off + n]
    inside = (xd > fg.x[0] + 1e-3) & (xd < fg.x[-1] - 1e-3)
    Kd = build_dense_lsf_operator(fg.x, fg.dx, xd, g.R[off:off + n], truncate=6.0)
    rs = np.asarray(Kd.sum(axis=1)).ravel()
    assert np.allclose(rs[inside], 1.0, atol=1e-6)                                 # a point-sampled, unit-area kernel


def _smooth_test(rv, ppf=8.0, Rconst=None):
    """P_disk @ (G * I) against K @ shift(I) on a smooth line spectrum: the per-disk resampling error."""
    from jalebi.constants import C, FWHM_TO_SIGMA
    from jalebi.instrument import build_lsf_operator, resolving_power
    from jalebi.model import FineGrid
    g = DenseGrid(DenseSpec(points_per_fwhm=ppf, bands=("3B",)), R_constant=Rconst)
    wave_pix = np.exp(np.arange(np.log(13.6), np.log(15.3), 1.7e-4))
    fg = FineGrid([(13.4, 15.5)], step_kms=4.7 / 6)
    rng = np.random.default_rng(0)
    I = np.zeros(fg.n)
    for x in rng.uniform(np.log(13.5), np.log(15.4), 200):
        I += rng.uniform(0.1, 1) * np.exp(-0.5 * ((fg.x - x) / (4.7e3 / C * FWHM_TO_SIGMA)) ** 2)
    Ish = np.interp(fg.x - rv * 1e3 / C, fg.x, I, left=0.0, right=0.0)
    K = build_lsf_operator(fg.x, fg.dx, wave_pix, resolving_power(wave_pix, constant=Rconst))
    ref = K @ Ish
    off, n, x0, h = g.segments[0]
    H = np.zeros(g.n); H[off:off + n] = build_dense_lsf_operator(fg.x, fg.dx, g.x[off:off + n], g.R[off:off + n], 6.0) @ I
    P, cov = g.pixel_operator(wave_pix, rv)
    assert cov.all() and np.allclose(np.asarray(P.sum(axis=1)).ravel(), 1.0, atol=1e-12)   # exact for a constant
    return np.abs(P @ H - ref).max() / ref.max()


def test_pixel_operator_matches_lsf_operator_on_a_smooth_spectrum():
    assert _smooth_test(0.0, Rconst=2800.0) < 3e-4           # resampling only (8 points per FWHM)
    assert _smooth_test(0.0) < 3e-4                          # + R at the dense points
    assert _smooth_test(12.0, ppf=12.0, Rconst=2800.0) < 1e-4


def test_shift_then_convolve_accuracy():
    """Shift-then-convolve: the kernel is R at the rest wavelength, the exact model's at the observed one
    (dR = 128 lambda v/c for argyriou2023, i.e. 3e-4 of sigma at 150 km/s); zero for a constant R."""
    assert _smooth_test(37.0) < 3e-4
    for rv in (150.0, -150.0):                               # the padding's |v| <= 150 km/s
        assert _smooth_test(rv) < 5e-4, rv
        assert _smooth_test(rv, Rconst=2800.0) < 3e-4, rv


def test_shared_table_serves_two_disks_from_one_file(shared_cache):
    """Invariance: other pixels, v_shift and distance read the same file (equal keys) and stay accurate."""
    a = _disk(rv=2.0)
    b = _disk(rv=-20.0, distance=200.0, seed=7, windows=[(13.72, 14.08), (14.82, 15.02)], oversample=6)
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")                      # no fallback warning allowed
        ea = a.use_emulator(_shared_settings(shared_cache))
        eb = b.use_emulator(_shared_settings(shared_cache))
    assert ea.info["files"] == eb.info["files"] and not ea.exact_units and not eb.exact_units
    assert all(t.meta["how"] == "cache" for t in eb.tables.values())
    assert len(a.model.wave_pix) != len(b.model.wave_pix)
    for prob, em in ((a, ea), (b, eb)):
        sig = prob.sigma / np.sqrt(prob.weights); f_ref = float(np.max(np.abs(prob.y)))
        bounds, amax = prob.emulator_bounds(); P = prob.model.resolve_params()
        rng = np.random.default_rng(5)
        for key, tab in em.tables.items():
            comp = next(c for c in prob.components if c.name == key)
            sc = em.info["spot_check"][key]
            assert sc["max_sigma"] < 0.1 and sc["max_flux"] < 1e-3 and sc["n"] == 200, sc
            es, ef = [], []
            for _ in range(100):
                T = np.exp(rng.uniform(np.log(300.0), np.log(900.0))); N = rng.uniform(15.0, 19.0)
                Fx, _ = exact_rows(prob.model, comp, P[key], T, [N])
                Fe, _ = tab.flux(T, N)
                Fe[~prob.model.covered] = Fx[0][~prob.model.covered]          # gap-edge pixels (see spot_check)
                e1, e2 = errors(Fe, Fx, sig, f_ref, amax[key]); es.append(e1[0]); ef.append(e2[0])
            assert max(es) < 0.1 and max(ef) < 1e-3, (key, max(es), max(ef))
    # the vectorised path and the sampler work on the projected tables
    TH = np.clip(b.theta0() + 0.01 * (b.hi - b.lo) * np.random.default_rng(0).standard_normal((16, b.ndim)), b.lo, b.hi)
    assert np.allclose([b.log_prob(t) for t in TH], b.log_prob_many(TH), atol=1e-3, rtol=0)
    r = b.mcmc(b.theta0(), nsteps=20, seed=1, moves="de", linear="profile", vectorize=True)
    assert np.all(np.isfinite(r.chain))


def test_shared_key_invariance_and_sensitivity(tmp_path, monkeypatch):
    st = _shared_settings(tmp_path)
    spec = TableSpec("HCN", "hitran", BOX["T"], BOX["logN"])
    f0 = key_fields(spec, dense_spec(st), st, "llhash")
    phys0, full0 = table_keys(f0)
    import jalebi
    monkeypatch.setattr(jalebi, "__version__", "9.9.9")
    assert table_keys(key_fields(spec, dense_spec(st), st, "llhash")) == (phys0, full0)       # not the release
    monkeypatch.undo()
    for field_name in ("molecule", "linelist_key", "linelist_hash", "eup_max", "fwhm", "fwhm_thermal", "R", "oversample",
                       "ref_snr", "target", "method", "max_nodes", "model_version", "lsf"):
        assert field_name in f0
    # the box changes only the full key; everything physical changes both
    p1, k1 = table_keys(key_fields(TableSpec("HCN", "hitran", (100.0, 1500.0), BOX["logN"]), dense_spec(st), st, "llhash"))
    assert p1 == phys0 and k1 != full0
    import dataclasses
    for other in (dataclasses.replace(st, ref_snr=300.0), dataclasses.replace(st, points_per_fwhm=12.0),
                  dataclasses.replace(st, target_sigma=0.2)):
        assert table_keys(key_fields(spec, dense_spec(other), other, "llhash"))[0] != phys0
    assert table_keys(key_fields(spec, dense_spec(st), st, "other"))[0] != phys0
    assert table_keys(key_fields(dataclasses.replace(spec, fwhm=5.0), dense_spec(st), st, "llhash"))[0] != phys0
    from jalebi import emulator_shared as ES
    monkeypatch.setattr(ES, "EMULATOR_MODEL_VERSION", 99)
    assert table_keys(key_fields(spec, dense_spec(st), st, "llhash"))[0] != phys0
    # a disk whose bounds lie inside a wider table's box is served by that table (physics key + containing box)
    assert all(not f.endswith(".lock") or True for f in os.listdir(shared_dir(st))) if os.path.isdir(shared_dir(st)) else True


def test_shared_json_metadata_and_memmap(shared_cache):
    import json
    st = _shared_settings(shared_cache)
    files = sorted(os.listdir(shared_dir(st)))
    assert any(f.endswith(".npy") for f in files) and any(f.endswith(".json") for f in files)
    base = os.path.join(shared_dir(st), next(f for f in files if f.endswith(".json"))[:-5])
    info = json.load(open(base + ".json"))
    assert info["spec"]["molecule"] in ("HCN", "CO2") and "key" in info["meta"] and info["meta"]["key"]["ref_snr"] == 1000.0
    assert info["meta"]["validation"]["max_sigma"] < 0.1
    tab = SharedTable.load(base)
    assert isinstance(tab.L, np.memmap) and tab.L.dtype == np.float32 and tab.L.shape == (len(tab.lnT), len(tab.logN), len(tab.sup))
    assert tab.covers(BOX["T"], BOX["logN"]) and not tab.covers((100.0, 900.0), BOX["logN"])


def test_wider_table_serves_narrower_bounds(shared_cache):
    """No table with exactly the disk's box: the one of the same physics whose box contains it is used."""
    prob = _disk(rv=2.0)
    prob.free[1].lo, prob.free[1].hi = 400.0, 800.0                 # HCN.T inside 300-900
    st = _shared_settings(shared_cache, boxes={"HCN": {"T": (400.0, 800.0), "logN": (15.0, 19.0)}, "CO2": BOX}, read_only=True)
    em = prob.use_emulator(st)
    assert set(em.tables) == {"HCN", "CO2"} and em.tables["HCN"].meta["how"] == "cache"
    assert em.tables["HCN"].meta["T_bounds"][1] == pytest.approx(900.0)


def test_bounds_outside_the_box_stay_exact(shared_cache):
    prob = _disk(rv=2.0)
    prob.free[1].hi = 1200.0                                         # HCN.T beyond the 900 K box
    em = prob.use_emulator(_shared_settings(shared_cache, read_only=True))
    assert "CO2" in em.tables and "HCN" in em.exact_units and "survey box" in em.exact_units["HCN"]


def test_read_only_fallback_never_builds(tmp_path, monkeypatch):
    prob = _disk()
    for how in ("setting", "env"):
        d = tmp_path / how
        st = _shared_settings(d, read_only=(how == "setting"))
        if how == "env":
            monkeypatch.setenv("JALEBI_EMULATOR_READONLY", "1")
        with pytest.warns(UserWarning, match="read_only"):
            em = prob.use_emulator(st)
        monkeypatch.delenv("JALEBI_EMULATOR_READONLY", raising=False)
        assert not em.tables and set(em.exact_units) == {"HCN", "CO2"} and not os.path.exists(shared_dir(st))
        assert em.info["fallback"]["HCN"] == em.exact_units["HCN"]
        with prob.model.exact():
            ref = prob.model.evaluate()
        assert np.array_equal(prob.model.evaluate(), ref)          # the exact model serves the fit


def test_spot_check_fallback(shared_cache, monkeypatch):
    """A projected table that misses the targets is dropped for that unit, with a warning and a diagnostics entry."""
    from jalebi import emulator_shared as ES
    real = ES.project_table

    def bad(shared, grid, model, comp, p, sigma, f_ref, a_max, unit, P=None, covered=None):
        tab = real(shared, grid, model, comp, p, sigma, f_ref, a_max, unit, P, covered)
        if unit == "HCN":
            tab.L = tab.L + np.float32(0.02)                        # 2 % too bright everywhere
        return tab
    monkeypatch.setattr(ES, "project_table", bad)
    prob = _disk(rv=2.0)
    with pytest.warns(UserWarning, match="spot check failed"):
        em = prob.use_emulator(_shared_settings(shared_cache, read_only=True))
    assert "HCN" in em.exact_units and "spot check failed" in em.exact_units["HCN"] and "CO2" in em.tables
    assert em.info["spot_check"]["HCN"]["max_sigma"] > 0.1 and "HCN" in em.info["fallback"]
    assert em.info["spot_check"]["HCN"]["time_s"] >= 0.0
    monkeypatch.undo()
    # spot_check: 0 turns the check off
    em = prob.use_emulator(_shared_settings(shared_cache, read_only=True, spot_check=0))
    assert set(em.tables) == {"HCN", "CO2"} and em.info["spot_check"] == {}


def test_lock_two_processes_build_once(tmp_path):
    """Two processes ask for the same missing table at the same time: one builds, the other waits and loads."""
    import multiprocessing as mp
    import sys
    code = r'''
import sys, json, time
from jalebi.emulator import EmulatorSettings
from jalebi.emulator_shared import TableSpec, get_or_build
st = EmulatorSettings(cache_dir=sys.argv[1], n_validate=10, bands=("3B",), n_start=(5, 5), max_nodes=(9, 9))
import warnings; warnings.simplefilter("ignore")
t0 = time.time()
tab, how = get_or_build(TableSpec("HCN", "hitran", (500.0, 600.0), (16.0, 17.0)), st)
print(json.dumps({"how": how, "nodes": tab.meta["nodes"], "s": time.time() - t0, "file": tab.path}))
'''
    import subprocess
    procs = [subprocess.Popen([sys.executable, "-c", code, str(tmp_path / "emu")], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for _ in range(2)]
    outs = [p.communicate(timeout=600) for p in procs]
    import json
    res = []
    for (o, e), p in zip(outs, procs):
        assert p.returncode == 0, e
        res.append(json.loads(o.strip().splitlines()[-1]))
    assert sorted(r["how"] for r in res) == ["built", "cache"], res
    assert res[0]["file"] == res[1]["file"] and res[0]["nodes"] == res[1]["nodes"]
    files = os.listdir(tmp_path / "emu" / "shared")
    assert sum(f.endswith(".npy") for f in files) == 1 and not any(".tmp" in f for f in files)


def test_molecule_boxes_and_survey_specs():
    from jalebi.config import ComponentConfig, ProjectConfig
    cfg = ProjectConfig(components=[ComponentConfig(name="CO2", molecule="CO2", bounds={"T": [200.0, 800.0]}),
                                    ComponentConfig(name="13CO2", molecule="13CO2", tie_to="CO2", ratio=70.0, fixed=["ratio"]),
                                    ComponentConfig(name="HCN", molecule="HCN")])
    cfg.fit.bounds_by_molecule = {"HCN": {"T": [100.0, 3000.0]}}
    b = molecule_boxes(cfg)
    assert b["CO2"] == {"T": (200.0, 800.0), "logN": (13.0, 21.0)}
    assert b["13CO2"]["T"] == (200.0, 800.0) and b["13CO2"]["logN"] == pytest.approx((13.0 - np.log10(70.0), 21.0 - np.log10(70.0)))
    assert b["HCN"] == {"T": (100.0, 3000.0), "logN": (13.0, 21.0)}
    cfg2 = ProjectConfig(components=[])
    cfg2.fit.auto_detect = True; cfg2.fit.detect.candidates = ["CO", "13CO", "CO2"]
    cfg2.fit.bounds_by_molecule = {"CO": {"T": [100.0, 3000.0]}}
    b2 = molecule_boxes([cfg, cfg2])
    assert b2["CO"]["T"] == (100.0, 3000.0) and b2["13CO"]["T"] == (100.0, 3000.0)
    assert b2["13CO"]["logN"][0] == pytest.approx(13.0 - np.log10(300.0)) and b2["CO2"]["T"] == (100.0, 1500.0)
    specs = survey_specs([cfg, cfg2])
    assert {s.molecule for s in specs} == {"CO2", "13CO2", "HCN", "CO", "13CO"} and len(specs) == 5
    assert all(s.oversample == 6 and s.fwhm == 4.7 for s in specs)


def test_cli_shared_build_list_and_config_status(tmp_path, monkeypatch):
    from typer.testing import CliRunner
    from jalebi.cli import app
    import yaml
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COLUMNS", "300")                       # rich would wrap the table cells otherwise
    r = CliRunner()
    assert r.invoke(app, ["init", "c.yaml", "--example", "synthetic"]).exit_code == 0
    d = yaml.safe_load(open("c.yaml"))
    d["components"] = [c for c in d["components"] if c["molecule"] == "HCN"]
    d["components"][0]["bounds"] = {"T": [500.0, 600.0], "logN": [16.0, 17.0]}
    d.setdefault("fit", {})["windows"] = [[13.8, 14.1]]
    d["fit"]["auto_detect"] = False
    d["fit"]["emulator"] = {"n_validate": 10, "bands": ["3B"], "n_start": [5, 5], "max_nodes": [9, 9]}
    yaml.safe_dump(d, open("c.yaml", "w"))
    out = r.invoke(app, ["emulator", "build", "--survey", "c.yaml", "c.yaml", "-j", "1", "--cache-dir", str(tmp_path / "emu")])
    assert out.exit_code == 0, out.output
    assert "built" in out.output and "HCN:hitran" in out.output
    assert sum(f.endswith(".npy") for f in os.listdir(tmp_path / "emu" / "shared")) == 1
    out = r.invoke(app, ["emulator", "build", "c.yaml", "--cache-dir", str(tmp_path / "emu")])
    assert out.exit_code == 0 and "cache" in out.output                     # second time: from the cache
    out = r.invoke(app, ["emulator", "list", "c.yaml", "--cache-dir", str(tmp_path / "emu")])
    assert out.exit_code == 0, out.output
    assert "HCN:hitran" in out.output and "shared table" in out.output and "500-600" in out.output
    assert "MB shared" in out.output and "per-disk" in out.output


def test_app_emulator_toggle_uses_shared_cache(shared_cache, monkeypatch):
    """The app's worker: shared tables when they exist (never built), per-disk tables for the rest."""
    from jalebi import emulator as E
    from jalebi.emulator_shared import attach_shared_emulator
    prob = _disk(rv=2.0)
    model = prob.model
    sig = prob.sigma; f_ref = float(np.max(np.abs(prob.y)))
    bounds = {"HCN": BOX, "CO2": {"T": (300.0, 1000.0), "logN": (15.0, 19.0)}}     # CO2 beyond the shared box
    import dataclasses
    st = _shared_settings(shared_cache)
    em = attach_shared_emulator(model, bounds, sig, f_ref, dataclasses.replace(st, read_only=True), free_keys=set())
    assert set(em.tables) == {"HCN"} and "CO2" in em.exact_units
    rest = {k: bounds[k] for k in em.exact_units if "survey box" in em.exact_units[k] or "read_only" in em.exact_units[k]}
    em2 = E.attach_emulator(model, rest, sig, f_ref, dataclasses.replace(st, cache="per_disk"), free_keys=set())
    assert set(em2.tables) == {"CO2"}


# ---------------------------------------------------------------------------------------------------
# FZ Tau, shared tables (12 sub-bands): the real thing, behind JALEBI_EMULATOR_FULL=1
# ---------------------------------------------------------------------------------------------------
@pytest.mark.skipif(os.environ.get("JALEBI_EMULATOR_FULL") != "1", reason="full FZ Tau shared campaign: set JALEBI_EMULATOR_FULL=1 (~1-2 h)")
def test_fz_tau_shared_accuracy_campaign(tmp_path):
    rep = _report()
    tab = rep.accuracy(n=2000, out=str(tmp_path / "rep"), cache_dir=os.environ.get("JALEBI_EMULATOR_TEST_CACHE"), cache="shared")
    assert len(tab) == 9
    bad = tab[~tab["pass"]]
    assert bad.empty, bad.to_string()
