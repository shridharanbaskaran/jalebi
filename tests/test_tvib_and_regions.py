"""Vibrational temperature, per-component windows and curated line regions."""
import numpy as np
import pytest

from jalebi.config import ComponentConfig, FitConfig, ProjectConfig
from jalebi.linedata import load_linelist, normalize_vlabel
from jalebi.model import Component, build_model
from jalebi.regions import line_regions, region_weights


WAVE = np.concatenate([np.arange(5.3, 7.5, 0.0012), np.arange(12.5, 17.0, 0.0025)])
WINS = [(5.3, 7.5), (12.5, 17.0)]


def test_vib_energies_recover_band_origins():
    ll = load_linelist("H2O", release="hitemp", fetch=False).select(5.0, 20.0)
    ev = ll.vib_energies()
    cm = 1.0 / 1.4387769          # K per cm^-1
    assert ev["000"] == pytest.approx(0.0, abs=1e-6)
    assert ev["010"] * cm == pytest.approx(1594.75, abs=0.5)
    assert ev["100"] * cm == pytest.approx(3657.05, abs=0.5)
    assert ev["001"] * cm == pytest.approx(3755.93, abs=0.5)
    assert "-2-2-2" not in ev                    # unassigned HITEMP labels are skipped
    assert normalize_vlabel("0 1 0") == normalize_vlabel("0_1_0") == "010"


def test_tvib_equal_trot_is_lte():
    c = Component("w", "H2O", logN=18.0, T=800.0, logR=-0.5, linelist_release="hitemp")
    m = build_model([c], WAVE, 140.0, windows=WINS, oversample=3)
    f0 = m.evaluate()
    f1 = m.evaluate({"w": {"Tvib": 800.0}})
    assert np.max(np.abs(f0 - f1)) < 1e-4 * f0.max()
    # the per-line source function equals the Planck function in LTE
    b = m.basis(c, 4.7)
    import jalebi.constants as K
    lam = b.lines.wave[b.idx] * 1e-6
    B = 2 * K.H * K.C / lam**3 / np.expm1(K.H * K.C / (lam * K.KB * 800.0))
    assert np.max(np.abs(b.source(800.0) / B - 1)) < 1e-4


def test_tvib_weakens_rovib_band_but_not_rotational_lines():
    c = Component("w", "H2O", logN=18.3, T=900.0, logR=-0.6, linelist_release="hitemp")
    m = build_model([c], WAVE, 140.0, windows=WINS, oversample=3)
    f_lte = m.evaluate()
    f = m.evaluate({"w": {"Tvib": 500.0}})
    s1 = WAVE < 8.0
    s2 = WAVE > 12.0
    r1 = np.trapezoid(f[s1], WAVE[s1]) / np.trapezoid(f_lte[s1], WAVE[s1])
    r2 = np.trapezoid(f[s2], WAVE[s2]) / np.trapezoid(f_lte[s2], WAVE[s2])
    assert r1 < 0.25                              # nu2 band suppressed by > 4
    assert 0.6 < r2 < 1.0                          # rotational region only mildly affected (v=1-1 hot bands)
    assert np.all(np.isfinite(f)) and f.min() >= 0.0
    # optically thin: v=1-0 and v=1-1 lines scale with exp(-E_vib (1/Tvib - 1/T)), v=0-0 with Zvib only
    thin = Component("t", "H2O", logN=14.0, T=900.0, logR=0.0, linelist_release="hitemp")
    mt = build_model([thin], WAVE, 140.0, windows=WINS, oversample=3)
    b = mt.basis(thin, 4.7)
    vu = b.lines.table["vup"].map(normalize_vlabel).to_numpy()[b.idx]
    vl = b.lines.table["vlow"].map(normalize_vlabel).to_numpy()[b.idx]
    k0, S0 = b.kappa(900.0), b.source(900.0)
    k1, S1 = b.kappa(900.0, 500.0), b.source(900.0, 500.0)
    strong = k0 * S0 > 1e-3 * (k0 * S0).max()
    sel = (vu == "010") & (vl == "000") & strong
    expected = np.exp(-b.lines.vib_energies()["010"] * (1 / 500.0 - 1 / 900.0)) * b.zvib(900.0) / b.zvib(500.0)
    assert np.median((k1 * S1)[sel] / (k0 * S0)[sel]) == pytest.approx(expected, rel=0.02)
    sel0 = (vu == "000") & (vl == "000") & strong
    assert np.median((k1 * S1)[sel0] / (k0 * S0)[sel0]) == pytest.approx(b.zvib(900.0) / b.zvib(500.0), rel=0.02)


def test_component_windows_restrict_emission():
    rov = Component("rovib", "H2O", logN=18.0, T=950.0, logR=-0.8, windows=[[4.9, 9.0]], linelist_release="hitemp")
    rot = Component("rot", "H2O", logN=18.0, T=600.0, logR=-0.3, windows=[[9.0, 28.0]], linelist_release="hitemp")
    m = build_model([rov, rot], WAVE, 140.0, windows=WINS, oversample=3)
    tot, per, _ = m.evaluate(per_unit=True)
    assert per["rovib"][WAVE > 9.0].max() == 0.0 and per["rovib"][WAVE < 9.0].max() > 0.0
    assert per["rot"][WAVE < 9.0].max() == 0.0 and per["rot"][WAVE > 9.0].max() > 0.0
    # the NNLS area solve still works with window-restricted units
    y = tot * 1.3
    logR, chi2 = m.solve_areas(y, np.full(len(y), 1e-3))
    assert logR["rovib"] == pytest.approx(-0.8 + 0.5 * np.log10(1.3), abs=0.02)
    assert logR["rot"] == pytest.approx(-0.3 + 0.5 * np.log10(1.3), abs=0.02)


def test_config_tvib_windows_and_free_params():
    cfg = ProjectConfig(components=[
        ComponentConfig(name="H2O_hot", molecule="H2O", T=900, logN=18.3, logR=-0.6, Tvib=600.0),
        ComponentConfig(name="H2O_rovib", molecule="H2O", T=950, logN=18.3, logR=-0.8, windows=[[4.9, 9.0]])],
        fit=FitConfig(windows=[[5.3, 8.0], [12.0, 25.0]]))
    from jalebi.pipeline import free_params
    names = [(p.comp, p.name) for p in free_params(cfg)]
    assert ("H2O_hot", "Tvib") in names and ("H2O_rovib", "Tvib") not in names
    comps = cfg.components_list()
    assert comps[0].Tvib == 600.0 and comps[1].windows == [[4.9, 9.0]]
    assert "Tvib" in comps[0].params() and "Tvib" not in comps[1].params()
    with pytest.raises(ValueError):
        ProjectConfig(components=[ComponentConfig(name="x", molecule="H2O", windows=[[9.0, 5.0]])])


def test_tvib_prior_and_round_trip(tmp_path):
    from jalebi.fit import FitProblem
    from jalebi.pipeline import build_problem
    from jalebi.synthetic import make_synthetic_spectrum
    from jalebi.pipeline import prepare
    cfg = ProjectConfig(components=[ComponentConfig(name="H2O_hot", molecule="H2O", T=900, logN=18.0, logR=-0.6, Tvib=700.0)],
                        fit=FitConfig(windows=[[13.5, 15.0]], stages=["optimise"]))
    cfg.save(tmp_path / "c.yaml")
    cfg2 = ProjectConfig.load(str(tmp_path / "c.yaml"))
    assert cfg2.components[0].Tvib == 700.0
    spec, _ = make_synthetic_spectrum([dict(name="H2O_hot", molecule="H2O", T=900.0, logN=18.0, logR=-0.6, Tvib=700.0)],
                                      bands=("3B",), seed=1, oversample=3)
    spec = prepare(cfg2, spec)
    prob = build_problem(cfg2, spec)
    assert isinstance(prob, FitProblem)
    th = prob.theta0()
    assert np.isfinite(prob.log_prob(th))
    # T_vib above T_rot is excluded by the default prior
    P, _ = prob.params_from_theta(th)
    P["H2O_hot"]["Tvib"] = P["H2O_hot"]["T"] + 50.0
    assert not np.isfinite(prob.log_prior(prob.theta_from_params(P)))


def test_line_regions():
    ws = line_regions(["H2O_v0-0"])
    assert 40 < len(ws) < 60 and ws[0][0] > 9.8 and ws[-1][1] < 27.1
    assert all(b > a for a, b in ws)
    ws2 = line_regions(["H2O_v0-0"], pad_um=0.01)
    assert len(ws2) <= len(ws)
    w = region_weights(ws, (20.0, 5.0))
    assert set(w.values()) == {5.0} and all(0.5 * (ws[i][0] + ws[i][1]) > 20.0 for i in w)
    cfg = ProjectConfig(components=[ComponentConfig(name="H2O", molecule="H2O")],
                        fit=FitConfig(windows=[[12.0, 27.0]], line_regions=["H2O_v0-0"], region_weight_beyond=[20.0, 5.0]))
    cw = cfg.windows()
    assert all(a >= 12.0 and b <= 27.0 for a, b in cw) and len(cw) > 30
    assert cfg.window_weights() == region_weights(cw, (20.0, 5.0))


def test_default_water_split():
    from jalebi.config import apply_water_split
    cfg = ProjectConfig(components=[
        ComponentConfig(name="H2O_rovib", molecule="H2O", T=950, logN=18.3, logR=-0.8),
        ComponentConfig(name="H2O_hot", molecule="H2O", T=900, logN=18.3, logR=-0.6),
        ComponentConfig(name="H2O_tv", molecule="H2O", T=900, logN=18.3, logR=-0.6, Tvib=600.0),
        ComponentConfig(name="H2O_own", molecule="H2O", T=900, logN=18.3, logR=-0.6, windows=[[12.0, 20.0]]),
        ComponentConfig(name="H2_18O", molecule="H2_18O", tie_to="H2O_hot"),
        ComponentConfig(name="CO", molecule="CO", T=1300, logN=18.0, logR=-0.9)],
        fit=FitConfig(windows=[[5.3, 8.0], [12.0, 27.0]]))
    comps = {c.name: c for c in cfg.components_list()}
    assert comps["H2O_rovib"].windows == [[4.9, 9.5]]
    assert comps["H2O_hot"].windows == [[9.5, 28.5]]
    assert comps["H2_18O"].windows == [[9.5, 28.5]]          # tied: follows its parent
    assert comps["H2O_tv"].windows is None                     # Tvib handles the band
    assert comps["H2O_own"].windows == [[12.0, 20.0]]          # explicit windows are kept
    assert comps["CO"].windows is None
    # nothing happens when the fit stays above the split, or when the split is off
    cfg.fit.windows = [[12.0, 27.0]]
    assert all(c.windows is None for c in cfg.components_list() if c.name in ("H2O_rovib", "H2O_hot"))
    cfg.fit.windows = [[5.3, 8.0], [12.0, 27.0]]; cfg.fit.water_split_um = None
    assert all(c.windows is None for c in cfg.components_list() if c.name in ("H2O_rovib", "H2O_hot"))
    assert apply_water_split([], [(5.0, 8.0)], 9.5) == []


def test_region_other_molecules_and_priors():
    from jalebi.pipeline import free_params
    cfg = ProjectConfig(components=[ComponentConfig(name="H2O_rovib", molecule="H2O"),
                                    ComponentConfig(name="H2O_hot", molecule="H2O", priors={"T": [800, 150]}),
                                    ComponentConfig(name="CO2", molecule="CO2"), ComponentConfig(name="CO", molecule="CO")],
                        fit=FitConfig(windows=[[4.9, 8.0], [12.0, 27.5]], line_regions=["H2O_v1-0", "H2O_v0-0"]))
    ws = cfg.windows()
    assert any(a <= 14.93 and b >= 15.02 for a, b in ws)          # CO2 Q branch (+-0.15 um) is in
    assert any(a <= 5.0 and b >= 5.3 for a, b in ws)              # CO fundamental feature is in
    assert all(a >= 4.9 and b <= 27.5 for a, b in ws)             # clipped to fit.windows
    cfg.fit.region_other_molecules = "none"
    ws2 = cfg.windows()
    assert not any(a <= 14.93 and b >= 15.02 for a, b in ws2) and sum(b - a for a, b in ws2) < sum(b - a for a, b in ws)
    cfg.fit.region_other_molecules = "default"
    assert any(a <= 14.6 and b >= 16.4 for a, b in cfg.windows())  # the full CO2 default window
    g = {(p.comp, p.name): p.gauss for p in free_params(cfg)}
    assert g[("H2O_hot", "T")] == (800.0, 150.0) and g[("CO2", "T")] is None
