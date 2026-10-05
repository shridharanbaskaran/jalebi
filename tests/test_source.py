"""jalebi.source: open a target once, share it between the analyses (and the web app's sessions)."""
import shutil
import threading

import numpy as np
import pytest
from typer.testing import CliRunner

from jalebi import source as S
from jalebi.cli import app as cli_app
from jalebi.examples import resolve_path

runner = CliRunner()
HV = "example:HV_Tau_C_cube"
FZ = "example:FZ_Tau"


@pytest.fixture(autouse=True)
def _fresh_cache():
    S.forget_sources()
    yield
    S.forget_sources()


def test_scan_and_inspect():
    found = {e.name: e for e in S.scan_sources(resolve_path("example:"))}
    assert found["FZ_Tau"].n_x1d == 12 and found["HV_Tau_C_cube"].n_s3d == 5
    assert any(e.n_tables for e in found.values())
    assert S.inspect_path(FZ).kind == "folder" and S.inspect_path("example:synthetic/synthetic_miri_ch3.csv").kind == "table"
    S.open_source(FZ)
    assert {e.name: e for e in S.scan_sources(resolve_path("example:"))}["FZ_Tau"].loaded       # ● marker


def test_open_source_is_cached_per_process():
    a = S.open_source(HV)
    assert S.open_source(HV) is a                                   # a second session gets the same object
    assert S.open_source(HV, zero_is_nan=False) is not a            # other NaN rules = other cubes
    assert S.open_source(HV, refresh=True) is not a
    assert S.get_cubeset(HV) is S.open_source(HV).cubes


def test_x1d_and_copies():
    src = S.open_source(FZ)
    assert src.has_x1d and not src.has_cubes and src.name == "V* FZ Tau"
    s1 = src.x1d(distance_pc=130.0)
    s1.flux[:] = 0.0                                                # a module changing its copy ...
    s2 = src.x1d()
    assert np.nanmax(s2.flux) > 0 and s1.distance_pc == 130.0       # ... does not change the source
    from jalebi.data import load_x1d_folder
    ref = load_x1d_folder(resolve_path(FZ))
    assert np.array_equal(ref.wave, s2.wave) and np.allclose(ref.flux, s2.flux, equal_nan=True)
    assert np.array_equal(ref.mask, s2.mask)


def test_aperture_from_memory_matches_load_s3d_folder():
    from jalebi.data import load_s3d_folder
    src = S.open_source(HV)
    ra, dec, how = src.position()
    assert "cubes" in how
    for kw in (dict(apcorr="mrs"), dict(apcorr="gaussian", aperture_arcsec=0.8), dict(apcorr="none", annulus_arcsec=(1.5, 2.5))):
        mine = src.aperture_spectrum(ra, dec, **kw)
        ref = load_s3d_folder(resolve_path(HV), ra, dec, pattern="*s3d*.fits.gz", verbose=False, **kw)
        assert np.array_equal(np.isfinite(mine.flux), np.isfinite(ref.flux))
        assert np.allclose(np.sort(mine.flux[np.isfinite(mine.flux)]), np.sort(ref.flux[np.isfinite(ref.flux)]), rtol=1e-6)
    assert src.aperture_spectrum(ra, dec) is not src.aperture_spectrum(ra, dec)     # copies of one memoised spectrum


def test_extract_s3d_refactor_unchanged():
    """extract_s3d (file) and extract_cube (memory) give the same photometry."""
    import glob
    from jalebi.cube.io import read_cube
    from jalebi.data import extract_cube, extract_s3d
    f = sorted(glob.glob(resolve_path(HV) + "/*ch1-short*"))[0]
    a = extract_s3d(f, center="peak")
    b = extract_cube(read_cube(f), center="peak")
    assert np.allclose(a["center_pix"], b["center_pix"])
    ok = np.isfinite(a["flux"])
    assert np.allclose(a["flux"][ok], b["flux"][ok], rtol=1e-6)


def test_preload_parallel_reads_each_cube_once(monkeypatch):
    from jalebi.cube import io as cio
    calls = []
    real = cio.read_cube

    def counting(path, **kw):
        calls.append(path)
        return real(path, **kw)
    monkeypatch.setattr(cio, "read_cube", counting)
    src = S.open_source(HV)
    seen = []
    threads = [threading.Thread(target=lambda: src.cubes.load(src.cubes.info[0].path)) for _ in range(4)]
    for t in threads:
        t.start()
    src.preload(workers=4, progress=lambda d, n, b: seen.append((d, n)))
    for t in threads:
        t.join()
    assert sorted(calls) == sorted(src.cubes.files)                 # every cube read exactly once
    assert len(src.cubes.loaded()) == 5 and seen[-1] == (5, 5)
    src.region_spectrum(_circle(src)); src.aperture_spectrum()
    assert len(calls) == 5                                          # analyses after the preload read nothing


def _circle(src, r=1.0):
    from jalebi.cube import offset_region
    return offset_region("circle", src.position()[:2], 0.0, 0.0, r)


def test_memory_budget(monkeypatch):
    monkeypatch.setenv("JALEBI_CUBE_CACHE_MB", "0.001")
    src = S.open_source(HV)
    assert not src.fits_in_memory()
    with pytest.warns(UserWarning, match="read when needed"):
        assert src.preload() == []
    assert src.cubes.cache_size >= 3


def test_settings_round_trip(tmp_path):
    d = tmp_path / "FZ"
    shutil.copytree(resolve_path(FZ), d)
    src = S.open_source(str(d))
    f = src.save_settings(distance_pc=130.0, rv_kms=16.5, ra="04:32:31.76", dec="+24:20:03.0")
    assert f == str(d / S.SOURCE_FILE)
    S.forget_sources()
    again = S.open_source(str(d))
    assert again.distance_pc == 130.0 and again.rv_kms == 16.5 and again.x1d().distance_pc == 130.0
    ra, dec, how = again.position()
    assert how == "remembered" and ra == pytest.approx(68.13233, abs=1e-4)
    assert S.scan_sources(str(tmp_path))[0].n_tables == 0           # the settings file is not a spectrum


def test_rotdiag_region_spectrum_from_source_matches():
    from jalebi.rotdiag.config import RotDiagConfig
    from jalebi.rotdiag.pipeline import load_cube_spectrum
    cfg = RotDiagConfig.model_validate({"molecule": "H2", "spectrum": {"path": HV, "source": "s3d", "radius_arcsec": 1.0}})
    a = load_cube_spectrum(cfg)
    b = load_cube_spectrum(cfg, source=S.open_source(HV))
    assert np.allclose(a.flux, b.flux, equal_nan=True) and a.meta["extraction"]["area_arcsec2"] == b.meta["extraction"]["area_arcsec2"]


def test_cli_source(tmp_path):
    r = runner.invoke(cli_app, ["source", "list", resolve_path("example:")])
    assert r.exit_code == 0 and "FZ_Tau" in r.output
    r = runner.invoke(cli_app, ["source", "info", HV, "--preload"])
    assert r.exit_code == 0 and "HV-Tau-C" in r.output and "read 5 cubes" in r.output
    d = tmp_path / "FZ"
    shutil.copytree(resolve_path(FZ), d)
    r = runner.invoke(cli_app, ["source", "set", str(d), "--distance", "130", "--rv", "16"])
    assert r.exit_code == 0 and (d / S.SOURCE_FILE).exists()
    out = tmp_path / "spec.csv"
    r = runner.invoke(cli_app, ["source", "spectrum", HV, str(out), "--source", "s3d"])
    assert r.exit_code == 0 and out.exists()


# ------------------------------------------------------------------------------------------------ web app

def test_app_starts_on_source_page_and_shares_data():
    pytest.importorskip("panel")
    from jalebi.app import JalebiApp
    a = JalebiApp()
    assert a.module == "source" and a.source is None
    sw = a.workspace("source")
    sw.open(HV)                                                     # inline (no server)
    assert a.source is S.open_source(HV) and len(a.source.cubes.loaded()) == 5
    assert all(not b.disabled for b, _, _ in sw.launch.values())
    sw.go("cube")
    cw = a.cube_ws
    assert a.module == "cube" and cw.cs is a.source.cubes          # the Cube module uses the cubes in memory
    sw.go("rotdiag")
    rd = a.rotdiag_ws
    assert rd.src.value == "s3d" and rd.spec is not None and rd.geo.value == "aperture"
    sw.go("lte")
    assert a.spec is not None and a.ex_source.value == "s3d" and a.spec.meta["extraction"]["source"] == "s3d"
    n = len(a.spec.wave)
    a.switch_module("cube"); a.switch_module("lte")                 # switching back does not reload
    assert len(a.spec.wave) == n and a._lte_source_key is not None
    assert "HV-Tau-C" in a.header.object
    b = JalebiApp()                                                 # a new browser session
    b.workspace("source").open(HV)
    assert b.source is a.source


def test_app_source_settings_reach_modules(tmp_path):
    pytest.importorskip("panel")
    from jalebi.app import JalebiApp
    d = tmp_path / "FZ"
    shutil.copytree(resolve_path(FZ), d)
    a = JalebiApp()
    sw = a.workspace("source")
    sw.open(str(d))
    sw.dist.value, sw.rv.value = 130.0, 16.0
    sw.remember()
    a.switch_module("lte")
    assert a.distance.value == 130.0 and a.rv.value == 16.0 and a.spec.rest_frame
    assert S.open_source(str(d)).distance_pc == 130.0
    rd = a.rotdiag_ws
    assert rd.src.value == "file" and rd.dist.value == 130.0


def test_app_start_rules():
    pytest.importorskip("panel")
    from jalebi.app import JalebiApp
    assert JalebiApp(start_tab="continuum").module == "lte"
    assert JalebiApp(start_module="cube").module == "cube"
    assert JalebiApp(start_tab="rotdiag").module == "rotdiag"
    a = JalebiApp(source_path=FZ)                                   # jalebi serve --source
    assert a.source is not None and a.source.name == "V* FZ Tau" and a.module == "source"
