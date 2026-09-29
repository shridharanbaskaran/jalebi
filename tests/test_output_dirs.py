"""Results go to one folder per source: output 'results/{target}' -> results/FZ_Tau, results/DR_Tau, ..."""
import os
from pathlib import Path

from typer.testing import CliRunner

from jalebi.cli import app
from jalebi.config import DEFAULT_OUTPUT, ProjectConfig, safe_name
from jalebi.examples import example_path
from jalebi.pipeline import prepare, run_pipeline, target_config


def test_safe_name():
    assert safe_name("V* FZ Tau") == "FZ_Tau"
    assert safe_name("DR Tau (A)") == "DR_Tau_A"
    assert safe_name("IRAS 04302+2247") == "IRAS_04302+2247"
    assert safe_name("") == "target"


def test_output_dir_rules():
    c = ProjectConfig()
    assert c.output == DEFAULT_OUTPUT == "results/{target}"
    c.target.name = "FZ Tau"
    assert c.output_dir() == os.path.join("results", "FZ_Tau")
    c.output = "results"                                   # old default: also one folder per source
    assert c.output_dir() == os.path.join("results", "FZ_Tau")
    c.output = "runs/{target}/water_hot_cold"
    assert c.output_dir() == os.path.join("runs", "FZ_Tau", "water_hot_cold") and c.output_root() == "runs"
    c.output = "my_fixed_folder"                           # no {target}: used as written
    assert c.output_dir() == "my_fixed_folder" and not c.per_target_output()


def test_name_comes_from_fits_header_when_unset():
    c = ProjectConfig()
    c.target.path = "example:FZ_Tau"
    spec = prepare(c)
    assert spec.name == "V* FZ Tau"
    assert c.output_dir(spec.name) == os.path.join("results", "FZ_Tau")


def test_batch_target_config():
    cfg = ProjectConfig()
    row = {"name": "DR Tau", "path": "/data/DR_Tau", "distance_pc": 195.0, "rv_kms": 27.0}
    c = target_config(cfg, row)
    assert c.output_dir() == os.path.join("results", "DR_Tau") and c.target.distance_pc == 195.0
    cfg.output = "fixed"                                   # a fixed folder gets one sub-folder per target
    assert target_config(cfg, row).output_dir() == os.path.join("fixed", "DR_Tau")


def test_run_pipeline_writes_into_source_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = ProjectConfig.load(example_path("synthetic/synthetic.yaml"))
    cfg.continuum.refine_iterations = 0
    cfg.fit.grid.logN = [15.0, 19.0, 3]; cfg.fit.grid.T = [300.0, 900.0, 3]
    run = run_pipeline(cfg, stages=["grid"])
    assert Path(run.outdir) == Path("results") / "synthetic_disk"
    assert (tmp_path / "results" / "synthetic_disk" / "best_fit.json").exists()
    assert not (tmp_path / "results" / "best_fit.json").exists()


def test_cli_prep_with_new_target_uses_its_own_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    assert runner.invoke(app, ["init", "c.yaml", "--example", "fz_tau"]).exit_code == 0
    r = runner.invoke(app, ["prep", "c.yaml", "--target", "example:synthetic/synthetic_miri_ch3.csv", "--no-plot"])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "results" / "synthetic_disk" / "prep.csv").exists()
    r = runner.invoke(app, ["prep", "c.yaml", "--no-plot"])                  # the config's own source
    assert (tmp_path / "results" / "FZ_Tau" / "prep.csv").exists()
