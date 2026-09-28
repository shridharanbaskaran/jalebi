"""The installation pieces agree with each other: requirements lists, version, bundled files, configs."""
import re
from pathlib import Path

import pytest

import jalebi
from jalebi._requirements import CORE, OPTIONAL, parse_version, version_ok

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
needs_repo = pytest.mark.skipif(not PYPROJECT.exists(), reason="needs the source checkout")


def _pyproject():
    try:
        import tomllib
    except ImportError:          # Python 3.10
        tomllib = pytest.importorskip("tomli")
    return tomllib.loads(PYPROJECT.read_text())


def _name_min(spec: str):
    m = re.match(r"^([A-Za-z0-9_.\-]+)\s*>=\s*([0-9][0-9.]*)", spec)
    return m.group(1), m.group(2)


@needs_repo
def test_core_requirements_match_pyproject():
    deps = dict(_name_min(d) for d in _pyproject()["project"]["dependencies"])
    assert deps == {k: v[0] for k, v in CORE.items()}


@needs_repo
def test_optional_requirements_match_pyproject():
    extras = _pyproject()["project"]["optional-dependencies"]
    for group, reqs in OPTIONAL.items():
        got = dict(_name_min(d) for d in extras[group])
        assert got == {k: v[0] for k, v in reqs.items()}, group


@needs_repo
def test_requirements_txt_matches():
    lines = [ln.split("#")[0].strip() for ln in (ROOT / "requirements.txt").read_text().splitlines()]
    got = dict(_name_min(ln) for ln in lines if ln)
    assert got == {k: v[0] for k, v in CORE.items()}


@needs_repo
def test_synthetic_config_copies_identical():
    a = (ROOT / "examples" / "configs" / "synthetic.yaml").read_text()
    b = (Path(jalebi.__file__).parent / "example_data" / "synthetic" / "synthetic.yaml").read_text()
    assert a == b


def test_version_helpers():
    assert parse_version("2.1.0rc1") == (2, 1, 0)
    assert version_ok("1.26.4", "1.24") and not version_ok("1.23.5", "1.24") and not version_ok(None, "1.0")
    assert version_ok("3.0", "3.0.0")


def test_version_string():
    assert re.match(r"^\d+\.\d+\.\d+", jalebi.__version__)


def test_bundled_files_present():
    pkg = Path(jalebi.__file__).parent
    assert len(list((pkg / "linedata").glob("*.parquet"))) >= 18
    assert len(list((pkg / "example_data" / "FZ_Tau").glob("*x1d.fits"))) == 12
    assert (pkg / "example_data" / "synthetic" / "synthetic_miri_ch3.csv").exists()
    assert len(list((pkg / "data_files").glob("*.csv"))) >= 7
