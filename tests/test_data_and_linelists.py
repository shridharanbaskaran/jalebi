"""Bundled line lists, example data, configs and the user-cache logic."""
from pathlib import Path

import numpy as np
import pytest

from jalebi.config import ProjectConfig
from jalebi.data import load_spectrum
from jalebi.examples import copy_examples, example_path, examples_source, resolve_path
from jalebi.linedata import BUNDLED_DIR, available_linelists, cache_path, find_cached, load_linelist

SMALL = ["CO2", "13CO2", "HCN", "H13CN", "C2H2", "13CCH2", "13CO", "CH4", "NH3", "C2H4", "C2H6", "C4H2", "HC3N", "OH", "H2"]


@pytest.mark.parametrize("mol", SMALL)
def test_bundled_hitran_lists_load(mol):
    ll = load_linelist(mol, release="hitran", fetch=False)
    assert len(ll) > 50
    assert np.all(np.diff(ll.wave) >= 0)
    for T in (200.0, 800.0):
        Z = ll.partition(T)
        assert np.isfinite(Z) and Z > 0
        k = ll.kappa(T)
        assert np.all(np.isfinite(k)) and k.max() > 0


def test_water_releases_both_bundled():
    hot = load_linelist("H2O", release="hitemp", fetch=False)
    cold = load_linelist("H2O", release="hitran", fetch=False)
    assert hot.release == "hitemp" and cold.release == "hitran"
    assert len(hot) > 10 * len(cold)
    # the partition functions of the two releases agree (same molecule, TIPS vs list table)
    assert abs(hot.partition(500.0) / cold.partition(500.0) - 1) < 0.05


def test_user_cache_shadows_bundled(tmp_path, monkeypatch):
    monkeypatch.setenv("JALEBI_DATA", str(tmp_path))
    bundled = find_cached("OH", "hitran")
    assert bundled is not None and Path(bundled).parent == Path(BUNDLED_DIR)
    # writes go to the user cache, never into the package
    assert Path(cache_path("OH", "hitran", write=True)).parent == tmp_path
    ll = load_linelist("OH", release="hitran", fetch=False)
    ll.select(10.0, 20.0).to_parquet(str(tmp_path / "OH_hitran.parquet"))
    assert Path(find_cached("OH", "hitran")).parent == tmp_path
    tab = available_linelists()
    row = tab[(tab.molecule == "OH") & (tab.release == "hitran")].iloc[0]
    assert row.location == "user"
    assert (tab.location == "bundled").any()


def test_fz_tau_example_loads():
    s = load_spectrum("example:FZ_Tau", distance_pc=130.0)
    assert s.bands == ["1A", "1B", "1C", "2A", "2B", "2C", "3A", "3B", "3C", "4A", "4B", "4C"]
    assert 4.8 < np.nanmin(s.wave) < 5.0 and 27.5 < np.nanmax(s.wave) < 29.0
    assert np.nanmedian(s.flux) > 0.05


def test_synthetic_example_loads():
    s = load_spectrum(example_path("synthetic/synthetic_miri_ch3.csv"))
    assert s.bands == ["3A", "3B", "3C"] and s.distance_pc == 140.0 and s.rest_frame
    assert np.all(s.continuum == 0)          # the true continuum column is not used as the continuum


def test_resolve_path():
    assert resolve_path("example:FZ_Tau") == example_path("FZ_Tau")
    assert resolve_path("~/x").startswith(str(Path.home()))
    with pytest.raises(FileNotFoundError):
        example_path("no_such_example")


def _configs():
    src = examples_source()
    return sorted((src / "configs").glob("*.yaml")) if src else []


@pytest.mark.parametrize("path", _configs(), ids=lambda p: p.name)
def test_example_configs_validate(path):
    cfg = ProjectConfig.load(str(path))
    assert Path(resolve_path(cfg.target.path)).exists()
    for c in cfg.components:
        c.to_component()


def test_copy_examples(tmp_path):
    if examples_source() is None:
        pytest.skip("example scripts not available in this installation")
    files = copy_examples(tmp_path)
    names = {Path(f).name for f in files}
    assert {"01_quick_model.py", "02_synthetic_fit.py", "synthetic.yaml"} <= names
    assert copy_examples(tmp_path) == []          # nothing overwritten by default
