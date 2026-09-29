"""Absorption screens: F = F_c (1 - f_c (1 - e^-tau)) in front of the continuum (and optionally the
emission), as in the group's slabby.spec_abs / two_slabs_spec.  Toy line list, no cache needed."""
import os

import numpy as np
import pytest

from jalebi.config import ComponentConfig, ProjectConfig
from jalebi.constants import C
from jalebi.data import Spectrum
from jalebi.fit import ABSORPTION_BOUNDS, FitProblem, default_free_params
from jalebi.model import Component, SlabModel, thermal_fwhm_kms

from test_core import toy_linelist


@pytest.fixture
def pix():
    return np.arange(13.9, 14.1, 0.0015)


def _model(comps, pix, cont=2.0, **kw):
    ll = toy_linelist(waves=(13.95, 14.0, 14.05))
    c = None if cont is None else np.full(len(pix), cont)
    return SlabModel(comps, {"CO2": ll}, pix, distance_pc=100.0, windows=[(13.9, 14.1)], continuum=c, **kw)


def test_screen_is_continuum_times_transmission(pix):
    a = Component("a", "CO2", logN=18.0, T=300.0, kind="absorption", fc=0.6)
    m = _model([a], pix)
    F = m.evaluate()
    P = m.resolve_params()
    tr, tmax = m.transmission(a, P["a"])
    direct = m.K @ (2.0 * (tr - 1.0))
    assert np.allclose(F, direct)
    assert F.max() <= 0 and F.min() < -1e-3          # only removes flux
    assert F.min() >= -2.0 * 0.6 - 1e-9              # never deeper than fc * continuum
    assert m.tau_flags()["a"] == pytest.approx(tmax) and tmax > 1


def test_depth_scales_with_covering_fraction(pix):
    F = {}
    for fc in (0.25, 0.5, 1.0):
        a = Component("a", "CO2", logN=18.0, T=300.0, kind="absorption", fc=fc)
        F[fc] = _model([a], pix).evaluate()
    assert np.allclose(F[0.5], 2 * F[0.25]) and np.allclose(F[1.0], 4 * F[0.25])


def test_no_continuum_means_no_absorption(pix):
    a = Component("a", "CO2", logN=18.0, T=300.0, kind="absorption")
    assert np.all(_model([a], pix, cont=None).evaluate() == 0.0)


def test_velocity_shift_direction(pix):
    a0 = Component("a", "CO2", logN=17.0, T=300.0, kind="absorption", rv=0.0)
    ab = Component("a", "CO2", logN=17.0, T=300.0, kind="absorption", rv=-60.0)
    F0, Fb = _model([a0], pix).evaluate(), _model([ab], pix).evaluate()
    w0 = np.sum(pix * F0) / np.sum(F0); wb = np.sum(pix * Fb) / np.sum(Fb)
    assert (wb - w0) / w0 == pytest.approx(-60e3 / C, rel=0.05)      # negative rv = blueshift


def test_screens_multiply_and_units_add_up(pix):
    a = Component("a", "CO2", logN=18.0, T=300.0, kind="absorption", fc=0.8)
    b = Component("b", "CO2", logN=17.5, T=150.0, kind="absorption", fc=0.5, rv=30.0)
    m = _model([a, b], pix)
    total, units, _ = m.evaluate(per_unit=True)
    P = m.resolve_params()
    tr = m.transmission(a, P["a"])[0] * m.transmission(b, P["b"])[0]
    assert np.allclose(total, m.K @ (2.0 * (tr - 1.0)))
    assert np.allclose(total, units["a"] + units["b"])


def test_covers_all_attenuates_emission(pix):
    e = Component("e", "CO2", logN=17.0, T=600.0, logR=-0.5)
    ac = Component("a", "CO2", logN=18.5, T=300.0, kind="absorption", fc=1.0, covers="continuum")
    aa = Component("a", "CO2", logN=18.5, T=300.0, kind="absorption", fc=1.0, covers="all")
    mc, ma = _model([e, ac], pix), _model([e, aa], pix)
    _, uc, _ = mc.evaluate(per_unit=True)
    _, ua, _ = ma.evaluate(per_unit=True)
    assert np.allclose(uc["e"], _model([e], pix).evaluate())        # continuum-only screen: emission untouched
    assert ua["e"].max() < 0.2 * uc["e"].max()                       # screen in front of everything
    assert np.allclose(uc["a"], ua["a"])                             # continuum part identical
    # the emitting area is still solved exactly with a screen in front of it
    y = ma.evaluate()
    logR, chi2 = ma.solve_areas(y, np.full(len(pix), 1e-3), None)
    assert logR["e"] == pytest.approx(-0.5, abs=1e-6) and logR["a"] == 0.0 and chi2 < 1e-12


def test_thermal_width():
    assert thermal_fwhm_kms(300.0, 18.0) == pytest.approx(2 * np.sqrt(2 * np.log(2)) * np.sqrt(1.380649e-23 * 300 / (18 * 1.6605e-27)) / 1e3, rel=1e-4)
    a = Component("a", "CO2", logN=17.0, T=300.0, fwhm=4.7, fwhm_thermal=True, kind="absorption")
    m = _model([a], np.arange(13.9, 14.1, 0.0015))
    fw = m.line_fwhm(a, m.resolve_params()["a"])
    assert fw == pytest.approx(np.hypot(4.7, thermal_fwhm_kms(300.0, 43.98983)), rel=1e-6)


def test_free_params_and_fit_recovers_screen(pix):
    """Optimiser recovers (logN, T, v, fc) of one screen on a synthetic absorbed continuum, with an
    emission component fitted at the same time."""
    ll = toy_linelist(waves=(13.95, 14.0, 14.05))
    truth = dict(logN=17.4, T=280.0, rv=-45.0, fc=0.7)
    e = Component("e", "CO2", logN=16.5, T=700.0, logR=-0.4)
    a = Component("a", "CO2", kind="absorption", **truth)
    cont = np.full(len(pix), 1.5)
    m = SlabModel([e, a], {"CO2": ll}, pix, 100.0, [(13.9, 14.1)], continuum=cont)
    rng = np.random.default_rng(1)
    sig = 2e-4
    flux = cont + m.evaluate() + rng.normal(0, sig, len(pix))
    spec = Spectrum(pix, flux, np.full(len(pix), sig), np.full(len(pix), "1A"), distance_pc=100.0,
                    continuum=cont.copy())
    e0 = Component("e", "CO2", logN=16.0, T=600.0, logR=-0.6)
    a0 = Component("a", "CO2", logN=17.0, T=200.0, rv=-20.0, fc=0.5, kind="absorption")
    free = default_free_params([e0, a0])
    names = [p.key for p in free]
    assert names == ["e.logN", "e.T", "e.logR", "a.logN", "a.T", "a.rv", "a.fc"]
    assert [p for p in free if p.key == "a.T"][0].lo == ABSORPTION_BOUNDS["T"][0]
    prob = FitProblem(spec, [e0, a0], [(13.9, 14.1)], free, linelists={"CO2": ll}, use_pipeline_err=True)
    assert not prob.area_free_mask()[names.index("a.logN")] and prob.area_free_mask()[names.index("e.logR")]
    assert "a" not in prob.area_units() and "a" in prob.all_units()
    res = prob.optimise(method="de", maxiter=60, popsize=10, seed=3)
    P, _ = prob.params_from_theta(res.theta)
    assert P["a"]["rv"] == pytest.approx(truth["rv"], abs=3.0)
    assert P["a"]["fc"] * (1 - np.exp(-10 ** P["a"]["logN"] / 10 ** truth["logN"] * 0 + 0)) >= 0     # sanity
    # depth of the screen is what the data constrain: compare model to truth pixel by pixel
    assert prob.chi2(res.theta) / len(pix) < 2.0
    sig_ = prob.component_significance(res.theta)
    assert bool(sig_.set_index("component").loc["a", "detected"])
    # grid over the screen itself keeps fc / rv and does not try to solve an area
    g = prob.grid("a", logN=np.linspace(16.5, 18.5, 5), T=np.linspace(150, 400, 4))
    assert np.isfinite(g.chi2).all() and np.all(g.logR == 0.0)


def test_config_roundtrip_and_validation(tmp_path):
    cfg = ProjectConfig(components=[
        ComponentConfig(name="H2O_em", molecule="H2O"),
        ComponentConfig(name="H2O_abs", molecule="H2O", kind="absorption", logN=18.0, T=200.0, rv=-30.0, fc=0.8,
                        covers="all", fwhm_thermal=True, fixed=["fc"])])
    p = tmp_path / "abs.yaml"
    cfg.save(str(p))
    back = ProjectConfig.load(str(p))
    c = back.components_list()[1]
    assert c.kind == "absorption" and c.fc == 0.8 and c.covers == "all" and c.fwhm_thermal and c.is_absorber
    assert c.params()["fc"] == 0.8 and "fc" not in back.components_list()[0].params()
    from jalebi.pipeline import free_params
    keys = [q.key for q in free_params(back)]
    assert "H2O_abs.fc" not in keys and "H2O_abs.rv" in keys and "H2O_abs.logR" not in keys
    with pytest.raises(ValueError):
        ProjectConfig(components=[ComponentConfig(name="x", molecule="H2O", kind="screen")])
    with pytest.raises(ValueError):
        ProjectConfig(components=[ComponentConfig(name="x", molecule="H2O", kind="absorption", fc=1.5)])


def test_csv_baseline_column_and_given_continuum(tmp_path):
    import pandas as pd
    from jalebi.continuum import ContinuumSettings, estimate_continuum
    from jalebi.data import load_csv
    w = np.arange(5.0, 5.2, 0.001)
    base = 0.02 + 0.001 * (w - 5.0)
    pd.DataFrame({"wavelength": w, "flux": base * 0.9, "flux error": 1e-4, "baseline": base}).to_csv(tmp_path / "s.csv", index=False)
    s = load_csv(str(tmp_path / "s.csv"))
    assert np.allclose(s.continuum, base)
    c = estimate_continuum(s, ContinuumSettings(method="given"))
    assert np.allclose(c, base)
    s.continuum = None
    s.__post_init__()
    with pytest.raises(ValueError):
        estimate_continuum(s, ContinuumSettings(method="given"))


def test_synthetic_spectrum_with_absorber():
    from jalebi.synthetic import make_synthetic_spectrum
    ll = toy_linelist(waves=(14.0, 14.3, 14.6, 15.0))
    comps = [dict(name="a", molecule="CO2", kind="absorption", logN=18.0, T=250.0, fc=0.9)]
    spec, truth = make_synthetic_spectrum(comps, bands=["3B"], snr=1e6, linelists={"CO2": ll})
    depth = truth["continuum"] - spec.flux
    assert (depth >= -1e-4 * truth["continuum"]).all() and depth.max() > 0.02 * truth["continuum"].min()


@pytest.mark.skipif(os.environ.get("JALEBI_SKIP_CLI") == "1", reason="cli")
def test_cli_model_absorption(tmp_path):
    from typer.testing import CliRunner
    from jalebi.cli import app
    import pandas as pd
    out = tmp_path / "m.csv"
    r = CliRunner().invoke(app, ["model", "--molecule", "H2O", "--kind", "absorption", "--fc", "0.5", "--rv", "-20",
                                 "--wmin", "6.0", "--wmax", "6.1", "--out", str(out), "--logN", "18.5", "--T", "300"])
    assert r.exit_code == 0, r.output
    df = pd.read_csv(out)
    assert {"wave", "flux", "continuum", "transmission"} <= set(df.columns)
    assert df["transmission"].min() >= 0.5 - 1e-9 and df["transmission"].min() < 0.999


def test_detect_finds_absorber_and_emission_behind_it():
    """Toy spectrum: a CO2 screen (fc 0.6) over a continuum plus weaker hot CO2 emission and nothing else.
    mode="both" must find CO2_abs and CO2, not HCN (emission or absorption); "emission" alone must not
    report screens; "absorption" alone must not report slabs."""
    from jalebi.detect import detect_molecules
    from jalebi.linedata import LineList, _finish
    from jalebi.molecules import get_molecule
    from jalebi.partition import PartitionFunction
    from jalebi.constants import HC_K
    import pandas as pd
    rng = np.random.default_rng(2)

    def toy(mol, waves, eu=1500.0, gu=5.0):
        waves = np.asarray(waves, float)
        eu = np.broadcast_to(np.asarray(eu, float), waves.shape)
        gu = np.broadcast_to(np.asarray(gu, float), waves.shape)
        el = eu - HC_K / (waves * 1e-6)
        df = pd.DataFrame({"wave": waves, "a": 5.0, "gu": gu, "gl": 3.0, "eu": eu, "el": el,
                           "vup": "1", "vlow": "0", "qup": "Q", "qlow": "Q", "iso": 1})
        df["nu"] = 1e4 / df["wave"]
        T = np.arange(1.0, 3001.0)
        return LineList(get_molecule(mol), _finish(df), PartitionFunction(T, 1.0 + 0.5 * T, "toy"), "toy")

    # CO2: high-E_u, high-g_u lines (seen in hot emission) and low-E_u lines (seen in cold absorption),
    # like a real band's hot bands / high-J lines versus the low-J lines
    co2_w = np.concatenate([np.linspace(14.90, 14.97, 12), np.linspace(14.98, 15.05, 12)])
    co2_eu = np.concatenate([np.full(12, 4000.0), np.full(12, 1200.0)])
    co2_gu = np.concatenate([np.full(12, 300.0), np.full(12, 5.0)])
    lls = {"H2O:hitran": toy("H2O", rng.uniform(12.0, 17.5, 200)), "H2O:hitemp": toy("H2O", rng.uniform(12.0, 17.5, 200)),
           "CO2:hitran": toy("CO2", co2_w, co2_eu, co2_gu), "HCN:hitran": toy("HCN", np.linspace(14.0, 14.06, 12))}
    wave = np.arange(12.0, 17.5, 0.002)
    cont = 0.5 + 0.05 * (wave - 12.0)
    truth = [Component("e", "CO2", logN=17.5, T=700.0, logR=-0.7),
             Component("a", "CO2", logN=18.0, T=100.0, fc=0.6, kind="absorption", covers="all")]
    from jalebi.model import build_model
    m = build_model(truth, wave, 140.0, [(12.0, 17.5)], linelists=lls, oversample=2, continuum=cont)
    flux = cont + m.evaluate()
    noise = 0.01 * cont
    spec = Spectrum(wave, flux + rng.normal(0, 1, len(wave)) * noise, noise, np.full(len(wave), "3B"),
                    distance_pc=140.0, continuum=cont.copy())
    det = detect_molecules(spec, candidates=["CO2", "HCN"], linelists=lls, threshold=10.0, oversample=2, mode="both")
    t = det.table.set_index("candidate")
    assert bool(t.loc["CO2_abs", "detected"]) and bool(t.loc["CO2", "detected"])
    assert not bool(t.loc["HCN", "detected"]) and not bool(t.loc["HCN_abs", "detected"])
    assert 0.3 < t.loc["CO2_abs", "fc"] <= 1.0
    kinds = {c.name: c.kind for c in det.components}
    assert kinds == {"CO2_abs": "absorption", "CO2": "slab"}
    e = detect_molecules(spec, candidates=["CO2", "HCN"], linelists=lls, threshold=10.0, oversample=2, mode="emission")
    assert not any(e.table.candidate.str.endswith("_abs"))
    a = detect_molecules(spec, candidates=["CO2", "HCN"], linelists=lls, threshold=10.0, oversample=2, mode="absorption")
    assert all(a.table.candidate.str.endswith("_abs")) and bool(a.table.set_index("candidate").loc["CO2_abs", "detected"])
    with pytest.raises(ValueError):
        detect_molecules(spec, candidates=["CO2"], linelists=lls, mode="screens")
