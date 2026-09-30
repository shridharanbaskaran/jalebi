"""Rotation diagrams (jalebi.rotdiag): line data, physics, measurement, fits, pipeline, CLI and the web-app module.

Recovery tests use two independent generators: jalebi.rotdiag.synthetic (the rotation-diagram physics with
extinction and a free OPR) and jalebi.synthetic (full LTE slab spectra from jalebi.model, a separate code path)."""
import numpy as np
import pandas as pd
import pytest

from jalebi.rotdiag import (FitConfig, Geometry, MCMCConfig, RotDiagConfig, Selection, cog_factor, example_config,
                            find_features, fit_rotation, get_curve, measure_features, partition, preset, run_rotdiag)
from jalebi.rotdiag.species import load_species_linelist


def _pull(res, truth):
    t = res.table().set_index("param")
    out = {}
    for k, v in truth.items():
        if k in res.free:
            m, lo, hi = t.loc[k, "median"], t.loc[k, "err_lo"], t.loc[k, "err_hi"]
            e = hi if v > m else lo
            out[k] = (m - v) / e if e > 0 else np.inf
    return out


# ------------------------------------------------------------------------------------------------ physics / data

def test_h2_level_sums_match_hitran_and_lte_opr():
    sp = preset("H2")
    ll = load_species_linelist("H2", "roueff2019")
    P = partition(sp, ll)
    z = np.load(ll.source.replace(".parquet", "_Q.npz")) if ll.source.endswith(".parquet") else None
    hit = load_species_linelist("H2", "hitran").partition
    for T in (100.0, 300.0, 1000.0):
        assert P.Q(T) == pytest.approx(hit(T), rel=2e-3)
    assert P.opr_lte(100.0) == pytest.approx(1.59, abs=0.02)        # the textbook LTE OPR of H2 at 100 K
    assert P.opr_lte(1000.0) == pytest.approx(3.0, abs=1e-3)
    assert np.allclose(P.Qs(500.0, "o") + P.Qs(500.0, "p"), P.Q(500.0))
    assert z is None or len(z["T"]) > 1000


def test_h2_line_data_have_all_s_lines():
    F, M = find_features("H2", None, Selection(wmin=4.5, wmax=30.0))
    tops = set(F["top"])
    for J in range(0, 10):
        assert f"S({J})" in tops
    s1 = F.set_index("top").loc["S(1)"]
    assert s1["wave"] == pytest.approx(17.0348, abs=1e-3) and s1["eu"] == pytest.approx(1015.1, abs=0.2)
    assert s1["gu"] == 21 and s1["spin"] == "o"
    assert F.set_index("top").loc["S(0)", "spin"] == "p"


def test_curve_of_growth_limits():
    assert cog_factor(1e-6) == pytest.approx(1.0, abs=1e-5)
    assert cog_factor(0.1) == pytest.approx(1 - 0.1 / (2 * np.sqrt(2)) + 0.01 / (6 * np.sqrt(3)), rel=1e-4)
    big = np.array([1e2, 1e4])
    assert np.allclose(cog_factor(big), 2 * np.sqrt(np.log(big)) / (np.sqrt(np.pi) * big), rtol=0.06)
    assert np.all(np.diff(cog_factor(np.logspace(-3, 6, 50))) < 0)


def test_extinction_curves(tmp_path):
    g = get_curve("G23")
    assert 0.09 < g.ak_av < 0.13
    assert g(9.66) > 2 * g(8.0)                      # the silicate feature at S(3)
    p = tmp_path / "kp5_like.csv"
    pd.DataFrame({"wave": [1.0, 10.0, 30.0], "AlamAK": [5.0, 1.0, 0.5]}).to_csv(p, index=False)
    c = get_curve(str(p), normalise="K")
    assert c(10.0) == pytest.approx(g.ak_av, rel=1e-6)
    with pytest.raises(ValueError):
        get_curve("no_such_curve")


def test_oh_hyperfine_components_are_one_feature():
    F, M = find_features("OH", None, Selection(wmin=20.0, wmax=28.0, max_features=20))
    assert (F["n_members"] >= 2).any()
    big = F.loc[F["n_members"].idxmax()]
    m = M[M["feature"] == big["id"]]
    assert np.ptp(m["wave"]) < 0.01                  # hyperfine / unresolved pairs only


# ------------------------------------------------------------------------------------------------ recovery

@pytest.fixture(scope="module")
def h2_single():
    from jalebi.rotdiag.synthetic import make_rotdiag_spectrum
    geo = Geometry(mode="number", distance_pc=140.0)
    truth = dict(logN=50.0, T=700.0, Av=6.0, OPR=2.0)
    spec, _ = make_rotdiag_spectrum("H2", "single", truth, opr="species", geometry=geo, snr=200, v_kms=12.0, seed=3)
    return spec, truth, geo


def test_h2_measure_and_fit_recovers_T_N_Av_OPR(h2_single):
    spec, truth, geo = h2_single
    F, M = find_features("H2", spec)
    Fm, st = measure_features(spec, F, M)
    assert Fm.attrs["velocity_kms"] == pytest.approx(12.0, abs=4.0)
    assert Fm["detected"].sum() >= 7 and len(st) == len(Fm)
    res = fit_rotation(Fm, M, FitConfig(model="single", opr="species", av_free=True, geometry=geo, sys_frac=0.0),
                       mcmc=MCMCConfig(steps=1200, burn=400, walkers=32))
    for k, p in _pull(res, truth).items():
        assert abs(p) < 3.0, (k, p, res.summary())
    d = res.derived.set_index("quantity")
    assert d.loc["mass", "unit"] == "M⊕" and d.loc["mass", "value"] > 0
    assert res.mcmc_info["acceptance"] > 0.15


def test_offset_convention_equals_species_at_high_T(h2_single):
    spec, truth, geo = h2_single
    F, M = find_features("H2", spec)
    Fm, _ = measure_features(spec, F, M)
    a = fit_rotation(Fm, M, FitConfig(model="single", opr="species", av_free=True, geometry=geo))
    b = fit_rotation(Fm, M, FitConfig(model="single", opr="offset", av_free=True, geometry=geo))
    assert a.best["T"] == pytest.approx(b.best["T"], rel=0.02)
    assert a.best["OPR"] == pytest.approx(b.best["OPR"], rel=0.05)


@pytest.mark.parametrize("kind,params,opr", [
    ("two", dict(logN1=50.3, T1=450.0, logN2=48.3, T2=1800.0, Av=8.0, OPR=2.5), "species"),
    ("powerlaw", dict(logN=50.5, Tmin=250.0, Tmax=4000.0, b=4.2, Av=3.0), "thermal"),
])
def test_two_temperature_and_power_law(kind, params, opr):
    from jalebi.rotdiag.synthetic import make_rotdiag_spectrum
    geo = Geometry(mode="number", distance_pc=140.0)
    spec, _ = make_rotdiag_spectrum("H2", kind, params, opr=opr, geometry=geo, snr=300, seed=7)
    F, M = find_features("H2", spec)
    Fm, _ = measure_features(spec, F, M)
    res = fit_rotation(Fm, M, FitConfig(model=kind, opr=opr, av_free=True, geometry=geo, sys_frac=0.0),
                       mcmc=MCMCConfig(steps=1500, burn=500, walkers=32))
    for k, p in _pull(res, {k: v for k, v in params.items() if k != "Tmax"}).items():
        assert abs(p) < 3.0, (kind, k, p)


def test_slab_model_h2_thin_is_recovered():
    """Independent physics: a jalebi.model LTE slab of H2 (Ω = π R²/d²) -> the rotation diagram gives N and T back."""
    from jalebi.synthetic import make_synthetic_spectrum
    bands = ("1A", "1B", "1C", "2A", "2B", "3A", "3C")
    spec, _ = make_synthetic_spectrum([dict(name="H2", molecule="H2", logN=22.0, T=800.0, logR=0.5, linelist_release="roueff2019")],
                                      bands=bands, snr=300, seed=2, continuum=dict(level=0.2, slope=1.0, wiggle=0.0))
    geo = Geometry(mode="radius", distance_pc=140.0, R_au=10 ** 0.5)
    F, M = find_features("H2", spec)
    Fm, _ = measure_features(spec, F, M)
    res = fit_rotation(Fm, M, FitConfig(model="single", geometry=geo, sys_frac=0.0))
    err = np.sqrt(np.diag(res.cov))
    assert abs(res.best["logN"] - 22.0) < max(4 * err[0], 0.02)
    assert abs(res.best["T"] - 800.0) < max(4 * err[1], 10.0)


def test_slab_model_thick_co_needs_the_opacity_correction():
    from jalebi.synthetic import make_synthetic_spectrum
    spec, _ = make_synthetic_spectrum([dict(name="CO", molecule="CO", logN=18.5, T=1100.0, logR=-1.0, fwhm=4.7, linelist_release="hitemp")],
                                      bands=("1A",), snr=300, seed=4, continuum=dict(level=0.2, slope=1.0, wiggle=0.0))
    geo = Geometry(mode="radius", distance_pc=140.0, R_au=0.1)
    F, M = find_features("CO", spec, Selection(max_features=40))
    Fm, _ = measure_features(spec, F, M)
    thick = fit_rotation(Fm, M, FitConfig(model="single", geometry=geo, opacity=True, fwhm_kms=4.7, sys_frac=0.0))
    thin = fit_rotation(Fm, M, FitConfig(model="single", geometry=geo, opacity=False, sys_frac=0.0))
    assert abs(thick.best["logN"] - 18.5) < 0.06 and abs(thick.best["T"] - 1100.0) < 40.0
    assert thick.chi2_red < 3.0
    assert thin.best["T"] > 1400.0 and thin.chi2_red > 10 * thick.chi2_red


# ------------------------------------------------------------------------------------------------ pipeline + CLI

def test_flux_table_units_and_matching(tmp_path):
    from jalebi.rotdiag.pipeline import unit_factor
    assert unit_factor("W m-2") == 1.0
    assert unit_factor("erg s-1 cm-2") == pytest.approx(1e-3)
    assert unit_factor("1e-17 erg s-1 cm-2") == pytest.approx(1e-20)
    with pytest.raises(ValueError):
        unit_factor("Jy")
    cfg = example_config("h2_fluxes"); cfg.mcmc.enabled = False; cfg.plots = False
    res = run_rotdiag(cfg, outdir=str(tmp_path / "t"))
    assert set(res.features["top"]) == {f"S({j})" for j in range(1, 8)}
    assert res.fit is not None and 300 < res.fit.best["T1"] < 700
    assert (tmp_path / "t" / "fit.yaml").exists() and (tmp_path / "t" / "diagram.csv").exists()


def test_config_roundtrip_and_examples(tmp_path):
    for name in ("h2", "h2_fluxes", "co", "oh", "h2o"):
        c = example_config(name)
        p = tmp_path / f"{name}.yaml"
        c.save(str(p))
        assert RotDiagConfig.load(str(p)) == c
    with pytest.raises(Exception):
        RotDiagConfig.model_validate({"molecule": "H2", "fit": {"nonsense": 1}})


def test_pipeline_writes_results(tmp_path):
    cfg = example_config("h2")
    cfg.mcmc.steps, cfg.mcmc.burn, cfg.mcmc.walkers = 400, 100, 24
    res = run_rotdiag(cfg, outdir=str(tmp_path / "out"))
    for f in ("lines.csv", "members.csv", "diagram.csv", "fit.yaml", "params.csv", "derived.csv", "compare.csv", "chain.npz",
              "summary.txt", "rotation_diagram.png", "corner.png", "line_fits.png", "rotdiag_config.yaml",
              "rotation_diagram_models.png", "fit_powerlaw.yaml", "params_powerlaw.csv"):
        assert (tmp_path / "out" / f).exists(), f
    comp = res.comparison.set_index("model")
    assert comp.loc["two", "bic"] < comp.loc["single", "bic"] - 20       # the synthetic truth has two components


def test_cli(tmp_path):
    from typer.testing import CliRunner
    from jalebi.cli import app
    r = CliRunner()
    out = r.invoke(app, ["rotdiag", "lines", "H2"])
    assert out.exit_code == 0 and "S(3)" in out.output
    out = r.invoke(app, ["rotdiag", "fit", "--fluxes", "example:synthetic/rotdiag_H2_fluxes.csv", "-m", "H2", "--model", "two",
                         "--av-free", "--out", str(tmp_path / "f")])
    assert out.exit_code == 0, out.output
    assert (tmp_path / "f" / "params.csv").exists()
    cfgp = tmp_path / "rd.yaml"
    out = r.invoke(app, ["rotdiag", "init", str(cfgp), "--example", "h2_fluxes"])
    assert out.exit_code == 0 and cfgp.exists()
    out = r.invoke(app, ["rotdiag", "curves"])
    assert out.exit_code == 0 and "G23" in out.output


# ------------------------------------------------------------------------------------------------ web app

def test_app_modules_and_rotdiag_workspace(tmp_path):
    pytest.importorskip("panel")
    from jalebi.app import JalebiApp
    a = JalebiApp(start_module="rotdiag")
    assert a.module == "rotdiag" and "cube" not in a.workspaces           # modules are built when opened
    ws = a.rotdiag_ws
    ws.on_show()
    assert ws.spec is not None and ws.mol.value == "H2" and ws.model.value == "two"
    ws.steps.value, ws.burn.value, ws.walkers.value = 300, 100, 24
    ws.run_all()                                                            # inline (no server)
    assert ws.fit is not None and ws.fit.model.kind == "two" and ws.fit.samples is not None
    assert len(ws.pt_src.data["x"]) >= 7 and len(ws.cv_src.data["xs"]) == 3    # ortho + para model curves + the power law
    assert "powerlaw" in ws.fit_extra and ws.paper.object is not None
    assert "jalebi rotdiag fit" in ws.code.object
    ws.out.value = str(tmp_path / "{target}" / "{molecule}")
    ws.write()
    assert (tmp_path / "synthetic_H2" / "H2" / "fit.yaml").exists()
    ws.src.value = "table"
    ws.run_all()
    assert ws.fit is not None and len(ws.features) == 7
    a.switch_module("cube")
    assert a.module == "cube" and "cube" in a.workspaces
    a.switch_module("lte", "Model")
    assert a.module == "lte" and a.tabs.active == JalebiApp.TAB_NAMES.index("Model")
    a.set_plot_theme("light")


def test_cube_region_to_rotation_diagram():
    pytest.importorskip("panel")
    from jalebi.app import JalebiApp
    a = JalebiApp(start_tab="cube")                 # the old --tab cube still opens the Cube module
    assert a.module == "cube"
    cw = a.cube_ws
    cw.n_mc.value = 0
    cw.dx.value, cw.dy.value, cw.r.value = 0.0, 0.0, 1.0
    cw.update_region(); cw.extract_region()
    assert cw.region_spec is not None and not cw.send_rd_btn.disabled
    cw.send_to_rotdiag()
    rd = a.rotdiag_ws
    assert a.module == "rotdiag" and rd.src.value == "sent" and rd.geo.value == "aperture" and rd.omega.value > 0
    rd.find()
    assert {"S(1)", "S(2)", "S(3)"} <= set(rd.features["top"])
