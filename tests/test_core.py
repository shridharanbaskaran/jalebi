"""Fast unit tests of the physics (run: pytest -q).  They build a tiny synthetic line list, so no
line-list cache or network is needed."""
import numpy as np
import pandas as pd
import pytest

from jalebi.constants import C, H, JY, planck_nu
from jalebi.linedata import LineList, _finish
from jalebi.molecules import get_molecule
from jalebi.model import Component, SlabModel
from jalebi.partition import PartitionFunction
from jalebi.data import Spectrum
from jalebi.continuum import ContinuumSettings, estimate_continuum, METHODS

trapezoid = getattr(np, "trapezoid", None) or np.trapz      # numpy < 2 has only trapz


def toy_linelist(waves=(14.0,), a=10.0, gu=5.0, eu=2000.0):
    # lower level consistent with the transition energy: E_l = E_u - hc/(lambda k)
    from jalebi.constants import HC_K
    waves = np.asarray(list(waves), float)
    el = eu - HC_K / (waves * 1e-6)
    df = pd.DataFrame({"wave": waves, "a": a, "gu": gu, "gl": 3.0, "eu": eu, "el": el,
                       "vup": "1", "vlow": "0", "qup": "Q1", "qlow": "Q1", "iso": 1})
    df["nu"] = 1e4 / df["wave"]
    T = np.arange(1.0, 3001.0)
    pf = PartitionFunction(T, 1.0 + 0.5 * T, "toy")
    return LineList(get_molecule("CO2"), _finish(df), pf, "toy")


@pytest.fixture
def pix():
    return np.arange(13.9, 14.1, 0.0015)


def test_thin_line_matches_analytic_flux(pix):
    ll = toy_linelist()
    c = Component("c", "CO2", logN=13.0, T=500.0, logR=-0.5)
    m = SlabModel([c], {"CO2": ll}, pix, distance_pc=100.0, windows=[(13.9, 14.1)])
    F = m.evaluate()
    nu_pix = C / (pix * 1e-6)
    Fint = -trapezoid(F * JY, nu_pix)
    N = 1e13 * 1e4
    Nu = N * ll.gu[0] * np.exp(-ll.eu[0] / 500.0) / ll.partition(500.0)
    Fline = H * (C / (ll.wave[0] * 1e-6)) / (4 * np.pi) * ll.a[0] * Nu * m.omega(-0.5)
    assert abs(Fint / Fline - 1) < 2e-3
    assert m.tau_flags()["c"] < 1e-2


def test_thick_limit_flux_conserved(pix):
    ll = toy_linelist()
    c = Component("c", "CO2", logN=20.0, T=500.0, logR=-0.5)
    m = SlabModel([c], {"CO2": ll}, pix, distance_pc=100.0, windows=[(13.9, 14.1)])
    tau = m.basis("CO2", 4.7).tau(1e20, 500.0)
    I = planck_nu(m.grid.wave, 500.0) * (-np.expm1(-tau))
    direct = -trapezoid(I * m.omega(-0.5), C / (m.grid.wave * 1e-6))
    conv = -trapezoid(m.evaluate() * JY, C / (pix * 1e-6))
    assert abs(conv / direct - 1) < 1e-3
    assert m.tau_flags()["c"] > 100


def test_operator_rows_sum_to_one(pix):
    ll = toy_linelist()
    m = SlabModel([Component("c", "CO2")], {"CO2": ll}, pix, distance_pc=100.0, windows=[(13.9, 14.1)])
    rs = np.asarray(m.K.sum(axis=1)).ravel()
    assert np.allclose(rs[m.covered], 1.0, atol=1e-4)


def test_curve_of_growth_monotonic(pix):
    ll = toy_linelist()
    m = SlabModel([Component("c", "CO2", T=500.0, logR=-0.5)], {"CO2": ll}, pix, distance_pc=100.0, windows=[(13.9, 14.1)])
    fluxes = [m.evaluate({"c": {"logN": lN}}).sum() for lN in (12, 14, 16, 18, 20)]
    assert all(np.diff(fluxes) > 0)
    # linear regime doubles with N, saturated regime does not
    assert fluxes[1] / fluxes[0] > 50 and fluxes[4] / fluxes[3] < 5


def test_area_is_linear_and_nnls_recovers_it(pix):
    ll = toy_linelist(waves=(13.95, 14.02, 14.07))
    m = SlabModel([Component("c", "CO2", logN=16.0, T=500.0, logR=-0.3)], {"CO2": ll}, pix, distance_pc=100.0, windows=[(13.9, 14.1)])
    f1 = m.evaluate({"c": {"logR": -0.3}}); f2 = m.evaluate({"c": {"logR": 0.2}})
    assert np.allclose(f2, f1 * 10 ** (2 * 0.5))
    data = f1 + 1e-4 * np.random.default_rng(0).standard_normal(len(pix))
    logR, chi2 = m.solve_areas(data, np.full(len(pix), 1e-4), {"c": {"logR": 0.0}})
    assert abs(logR["c"] + 0.3) < 0.01


def test_rv_shift_moves_line(pix):
    ll = toy_linelist()
    m = SlabModel([Component("c", "CO2", logN=16.0, T=500.0)], {"CO2": ll}, pix, distance_pc=100.0, windows=[(13.9, 14.1)])
    f0 = m.evaluate({"c": {"rv": 0.0}}); f1 = m.evaluate({"c": {"rv": 100.0}})
    shift = pix[np.argmax(f1)] - pix[np.argmax(f0)]
    assert abs(shift / 14.0 * C / 1e3 - 100.0) < 30.0   # within one pixel (~32 km/s)


def test_opacity_group_shares_temperature(pix):
    ll = toy_linelist()
    comps = [Component("a", "CO2", logN=16.0, T=500.0, group="g"), Component("b", "CO2", logN=16.0, T=900.0, group="g")]
    m = SlabModel(comps, {"CO2": ll}, pix, distance_pc=100.0, windows=[(13.9, 14.1)])
    total, units, tmax = m.evaluate(per_unit=True)
    assert list(units) == ["g"]
    single = SlabModel([Component("a", "CO2", logN=np.log10(2e16), T=500.0)], {"CO2": ll}, pix, 100.0, [(13.9, 14.1)]).evaluate()
    assert np.allclose(total, single, rtol=1e-6)


def test_continuum_methods_run():
    rng = np.random.default_rng(1)
    w = np.linspace(13.0, 17.0, 3000)
    cont = 1.0 + 0.1 * np.sin(w)
    lines = np.zeros_like(w)
    for x in rng.uniform(13, 17, 80):
        lines += 0.3 * np.exp(-0.5 * ((w - x) / 0.003) ** 2)
    f = cont + lines + 0.01 * rng.standard_normal(len(w))
    s = Spectrum(w, f, np.full(len(w), 0.01), np.where(w < 15.5, "3B", "3C"))
    for meth in METHODS:
        if meth == "none":
            continue
        st = ContinuumSettings(method=meth, anchors=list(np.linspace(13.1, 16.9, 12)))
        c = estimate_continuum(s, st)
        assert np.isfinite(c).all()
        assert np.median(np.abs(c - cont)) < 0.05, meth


def test_config_roundtrip(tmp_path):
    from jalebi.config import EXAMPLE_CONFIG, ProjectConfig
    p = tmp_path / "c.yaml"
    EXAMPLE_CONFIG.save(str(p))
    cfg = ProjectConfig.load(str(p))
    assert [c.name for c in cfg.components] == [c.name for c in EXAMPLE_CONFIG.components]
    assert cfg.windows() == [(13.5, 16.5), (16.5, 17.5)]


def test_detect_molecules_finds_injected_species():
    """A toy spectrum with CO2 and H2O lines: CO2 must be detected, HCN (no lines injected) must not."""
    from jalebi.detect import detect_molecules
    from jalebi.model import build_model
    from jalebi.constants import HC_K
    rng = np.random.default_rng(0)

    def toy(mol, waves, eu=1500.0):
        waves = np.asarray(waves, float)
        el = eu - HC_K / (waves * 1e-6)
        df = pd.DataFrame({"wave": waves, "a": 5.0, "gu": 5.0, "gl": 3.0, "eu": eu, "el": el,
                           "vup": "1", "vlow": "0", "qup": "Q", "qlow": "Q", "iso": 1})
        df["nu"] = 1e4 / df["wave"]
        T = np.arange(1.0, 3001.0)
        return LineList(get_molecule(mol), _finish(df), PartitionFunction(T, 1.0 + 0.5 * T, "toy"), "toy")

    lls = {"H2O:hitran": toy("H2O", rng.uniform(12.0, 17.5, 300)), "H2O:hitemp": toy("H2O", rng.uniform(12.0, 17.5, 300)),
           "CO2:hitran": toy("CO2", np.linspace(14.9, 15.05, 25)), "HCN:hitran": toy("HCN", np.linspace(14.0, 14.06, 12))}
    wave = np.arange(12.0, 17.5, 0.002)
    truth = [Component("w", "H2O", logN=17.8, T=600.0, logR=-0.5, linelist_release="hitran"),
             Component("c", "CO2", logN=17.0, T=500.0, logR=-0.5)]
    m = build_model(truth, wave, 140.0, [(12.0, 17.5)], linelists=lls, oversample=2)
    flux = m.evaluate()
    noise = 0.02 * flux.max()
    spec = Spectrum(wave, 1.0 + flux + rng.normal(0, noise, len(wave)), np.full(len(wave), noise),
                    np.full(len(wave), "3B"), distance_pc=140.0, continuum=np.ones(len(wave)))
    det = detect_molecules(spec, candidates=["CO2", "HCN"], linelists=lls, threshold=10.0, oversample=2)
    t = det.table.set_index("candidate")
    assert bool(t.loc["CO2", "detected"])
    assert not bool(t.loc["HCN", "detected"])
    assert any(c.molecule == "CO2" for c in det.components) and not any(c.molecule == "HCN" for c in det.components)


def test_nnls_respects_isotopologue_tie(pix):
    """A tied isotopologue shares its parent's area: the NNLS must solve one area for both."""
    parent = toy_linelist(waves=(13.95, 14.02))
    iso = toy_linelist(waves=(14.06, 14.08))
    iso = LineList(get_molecule("13CO2"), iso.table, iso.partition, "toy")
    comps = [Component("p", "CO2", logN=16.5, T=500.0, logR=-0.4),
             Component("i", "13CO2", tie_to="p", ratio=10.0)]
    m = SlabModel(comps, {"CO2": parent, "13CO2": iso}, pix, distance_pc=100.0, windows=[(13.9, 14.1)])
    assert m.tied_units() == {"i": "p"}
    data = m.evaluate()                                     # truth: logR = -0.4 for both
    logR, chi2 = m.solve_areas(data, np.full(len(pix), 1e-5), {"p": {"logR": 0.3}})
    assert abs(logR["p"] + 0.4) < 1e-3 and logR["i"] == logR["p"]
    assert chi2 < 1e-3


def test_annuli_area_is_not_profiled(pix):
    from jalebi.fit import FitProblem, default_free_params
    ll = toy_linelist()
    comps = [Component("g", "CO2", kind="annuli", n_annuli=4, logN=16.0, T=600.0, logR=-0.3, logRin=-1.0, p=0.0)]
    s = Spectrum(pix, np.zeros(len(pix)), np.full(len(pix), 1e-3), np.full(len(pix), "3B"))
    prob = FitProblem(s, comps, [(13.9, 14.1)], default_free_params(comps), linelists={"CO2": ll}, use_pipeline_err=True)
    names = [p.name for p in prob.free]
    assert "logR" in names and not prob.area_free_mask().any()
    logR, _ = prob.model.solve_areas(prob.y + 1.0, prob.sigma, None)
    assert logR["g"] == -0.3                                # held, not rescaled
