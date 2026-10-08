"""0.22 `jalebi linelist import`: local line lists in jalebi's Parquet layout with a partition function."""
import os

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from jalebi.cli import app
from jalebi.linedata import import_local_linelist, load_linelist, partition_from_levels, read_csv_lines


def fake_list(n=40, seed=0):
    rng = np.random.default_rng(seed)
    nu = np.sort(rng.uniform(660.0, 690.0, n))                    # ~14.5-15.2 um (benzene nu4 region)
    el = rng.uniform(0.0, 800.0, n)                                # cm-1
    return pd.DataFrame({"nu": nu, "el_cm1": el, "gu": rng.integers(1, 12, n).astype(float), "gl": rng.integers(1, 12, n).astype(float),
                         "a": rng.uniform(0.1, 5.0, n)})


def test_csv_import_with_partition_file(tmp_path, monkeypatch):
    monkeypatch.setenv("JALEBI_DATA", str(tmp_path / "cache"))
    df = fake_list()
    p = tmp_path / "c6h6.csv"; df.to_csv(p, index=False)
    T = np.arange(1, 1501, 1.0); Q = 100.0 * (T / 296.0) ** 2.0
    q = tmp_path / "Q.csv"; pd.DataFrame({"T": T, "Q": Q}).to_csv(q, index=False)
    info = import_local_linelist("C6H6", str(p), release="test", fmt="csv", partition=str(q))
    assert info["n_lines"] == 40 and info["partition"] == "file" and abs(info["Q296"] - 100.0) < 1e-6
    ll = load_linelist("C6H6", release="test", fetch=False)
    assert len(ll) == 40 and ll.release == "test" and abs(ll.partition(296.0) - 100.0) < 1e-6
    assert np.allclose(ll.eu - ll.el, 1e4 / ll.wave * 1.438776877, rtol=1e-6)  # eu - el = h c nu / k
    assert os.path.exists(str(tmp_path / "cache" / "C6H6_test_Q.npz"))


def test_intensity_conversion_and_levels_partition(tmp_path, monkeypatch):
    monkeypatch.setenv("JALEBI_DATA", str(tmp_path / "cache"))
    df = fake_list()
    # intensities from A by the inverse relation, then back: A must be recovered
    c2, c = 1.438776877, 2.99792458e10
    Q296 = 250.0
    sw = df["a"] * df["gu"] * (1 - np.exp(-c2 * df["nu"] / 296.0)) / (8 * np.pi * c * df["nu"] ** 2 * Q296 * np.exp(c2 * df["el_cm1"] / 296.0))
    d2 = df.drop(columns=["a"]).assign(sw=sw)
    p = tmp_path / "sw.csv"; d2.to_csv(p, index=False)
    out = read_csv_lines(str(p), Q296)
    assert np.allclose(np.sort(out["a"].to_numpy()), np.sort(df["a"].to_numpy()), rtol=1e-8)
    with pytest.raises(ValueError):
        read_csv_lines(str(p), None)
    T, Q = partition_from_levels(out, np.array([100.0, 300.0, 1000.0]))
    assert np.all(np.diff(Q) > 0) and Q[0] > 0
    info = import_local_linelist("C6H6", str(p), release="lev", fmt="csv", partition="levels", q296=Q296)
    assert info["partition"] == "levels" and info["Q296"] > 0


def test_cli_linelist_import(tmp_path, monkeypatch):
    monkeypatch.setenv("JALEBI_DATA", str(tmp_path / "cache"))
    p = tmp_path / "list.csv"; fake_list().to_csv(p, index=False)
    r = CliRunner().invoke(app, ["linelist", "import", "--molecule", "C6H6", "--file", str(p), "--format", "csv",
                                 "--release", "mine", "--partition", "levels"])
    assert r.exit_code == 0, r.output
    assert "imported" in r.output and "C6H6_mine.parquet" in r.output.replace("\n", "")
