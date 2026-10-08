"""0.22 stage checkpoints and resume (jalebi.resume, fit.resume): a run killed after any stage continues from
that stage and ends with the same products as an uninterrupted run."""
import json
import os
import shutil

import numpy as np
import pytest

import jalebi.pipeline as pl
from jalebi import resume as rs
from jalebi.config import ComponentConfig, ProjectConfig
from jalebi.synthetic import make_synthetic_spectrum, save_synthetic

TRUTH = [{"name": "CO2", "molecule": "CO2", "logN": 17.3, "T": 520.0, "logR": -0.6},
         {"name": "HCN", "molecule": "HCN", "logN": 16.8, "T": 600.0, "logR": -0.7}]


def make_config(folder, seed=3, nsteps=40, **fit):
    """A fast two-molecule synthetic fit (optimise + 1 continuum refinement + optimise + short MCMC, ~10 s)."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "synth.csv")
    if not os.path.exists(path):
        spec, t = make_synthetic_spectrum(components=TRUTH, bands=("3B",), snr=150.0, seed=seed, oversample=3)
        save_synthetic(path, spec, t)
    cfg = ProjectConfig(components=[ComponentConfig(**{**c, "logN": c["logN"] - 0.4, "T": c["T"] - 100.0},
                                                    bounds={"T": [300.0, 900.0], "logN": [15.0, 19.0]}) for c in TRUTH])
    cfg.target.name = "synth"; cfg.target.path = path; cfg.target.spike_filter = False
    cfg.continuum.method = "irsqr"; cfg.continuum.quantile = 0.1; cfg.continuum.knot_spacing = 50
    cfg.continuum.refine_iterations = 1
    cfg.masks.default_lines = False; cfg.masks.oh_prompt = False
    cfg.fit.windows = [[13.8, 15.2]]; cfg.fit.oversample = 3; cfg.fit.use_pipeline_err = True
    cfg.fit.stages = ["optimise", "mcmc"]
    cfg.fit.optimise.maxiter = 12; cfg.fit.optimise.popsize = 6; cfg.fit.optimise.n_starts = 1
    cfg.fit.mcmc.nwalkers = 12; cfg.fit.mcmc.nsteps = nsteps; cfg.fit.mcmc.linear = "profile"
    for k, v in fit.items():
        setattr(cfg.fit.mcmc if k in ("linear", "blocks", "moves") else cfg.fit, k, v)
    cfg.output = os.path.join(folder, "out", "{target}")
    return cfg


def same_data(src, dst):
    """Copy the synthetic spectrum keeping its mtime so that both runs share one resume key."""
    os.makedirs(dst, exist_ok=True)
    shutil.copy(os.path.join(src, "synth.csv"), os.path.join(dst, "synth.csv"))
    st = os.stat(os.path.join(src, "synth.csv"))
    os.utime(os.path.join(dst, "synth.csv"), ns=(st.st_mtime_ns, st.st_mtime_ns))


def load_chain(outdir):
    return dict(np.load(os.path.join(outdir, "chain.npz")))["chain"]


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    """The uninterrupted run everything is compared with."""
    folder = str(tmp_path_factory.mktemp("ref"))
    cfg = make_config(folder)
    run = pl.run_pipeline(cfg)
    return folder, cfg, run


def crash_after(monkeypatch, func_name, n_calls):
    """Make the n-th call of a pipeline stage function raise (the checkpoints of earlier stages exist)."""
    orig = getattr(pl, func_name)
    calls = {"n": 0}

    def wrapper(*a, **k):
        calls["n"] += 1
        if calls["n"] == n_calls:
            raise RuntimeError("simulated crash")
        return orig(*a, **k)
    monkeypatch.setattr(pl, func_name, wrapper)


@pytest.mark.parametrize("stage, n", [("run_optimise_stage", 1), ("run_optimise_stage", 2), ("run_mcmc_stage", 1)])
def test_crash_then_resume_equals_uninterrupted(reference, tmp_path, monkeypatch, stage, n):
    """Killed before pass 1 (nothing to resume), during pass 2 (de_pass1 + continuum resumed) or before the
    MCMC (both passes resumed): the resumed products are bit-identical to the uninterrupted run."""
    ref_folder, _, ref = reference
    folder = str(tmp_path / "crash")
    same_data(ref_folder, folder)
    cfg = make_config(folder)
    crash_after(monkeypatch, stage, n)
    with pytest.raises(RuntimeError):
        pl.run_pipeline(cfg)
    monkeypatch.undo()
    outdir = cfg.output_dir("synth")
    expected = {("run_optimise_stage", 1): [], ("run_optimise_stage", 2): ["de_pass1.json", "continuum_refined.npz"],
                ("run_mcmc_stage", 1): ["de_pass1.json", "continuum_refined.npz", "de_pass2.json"]}[(stage, n)]
    assert all(os.path.exists(os.path.join(outdir, f)) for f in expected)
    assert not os.path.exists(os.path.join(outdir, "summary.csv"))
    run = pl.run_pipeline(cfg)
    stage_lines = [l for l in run.log if "[stage]" in l]
    for f in expected:
        assert any(f.split(".")[0].replace("continuum_refined", "continuum refinement pass 1") in l and "resumed" in l for l in stage_lines), (f, stage_lines)
    assert np.array_equal(run.theta, ref.theta)
    assert np.array_equal(load_chain(outdir), load_chain(ref.outdir))
    ca = np.load(os.path.join(outdir, "continuum_refined.npz")); cb = np.load(os.path.join(ref.outdir, "continuum_refined.npz"))
    assert np.array_equal(ca["continuum"], cb["continuum"])
    bf = json.load(open(os.path.join(outdir, "best_fit.json"))); bf_ref = json.load(open(os.path.join(ref.outdir, "best_fit.json")))
    assert bf["params"] == bf_ref["params"]
    d = json.load(open(os.path.join(outdir, "diagnostics.json")))
    assert d["resume"]["resumed"] == expected


def test_longer_mcmc_continues_the_chain(reference, tmp_path):
    """fit.mcmc.nsteps 40 -> 80 in the same folder: the chain is continued from step 40 and equals an
    uninterrupted 80-step run (emcee's random state is restored from the HDF backend)."""
    ref_folder, cfg_ref, ref = reference
    folder = str(tmp_path / "longer")
    same_data(ref_folder, folder)
    cfg = make_config(folder)
    run40 = pl.run_pipeline(cfg)
    c40 = load_chain(run40.outdir)
    cfg.fit.mcmc.nsteps = 80
    run80 = pl.run_pipeline(cfg)
    assert any("mcmc resumed from checkpoint (40 stored steps, target 80)" in l for l in run80.log)
    c80 = load_chain(run80.outdir)
    assert c80.shape[0] == 80 and np.array_equal(c80[:40], c40)
    folder2 = str(tmp_path / "straight80")
    same_data(ref_folder, folder2)
    cfg2 = make_config(folder2, nsteps=80)
    straight = pl.run_pipeline(cfg2)
    assert np.array_equal(load_chain(straight.outdir), c80)
    # a complete chain is read back without sampling
    run_again = pl.run_pipeline(cfg)
    assert run_again.mcmc.meta["resumed_steps"] == [80]
    assert np.array_equal(load_chain(run_again.outdir), c80)


def test_key_changes_and_broken_files_are_ignored(reference, tmp_path):
    ref_folder, _, ref = reference
    folder = str(tmp_path / "keys")
    same_data(ref_folder, folder)
    cfg = make_config(folder)
    key1 = rs.stage_key(cfg)
    cfg.output = "elsewhere/{target}"; cfg.fit.mcmc.processes = 4; cfg.fit.mcmc.nsteps = 500; cfg.fit.optimise.workers = 2
    assert rs.stage_key(cfg) == key1                     # run-only keys do not change the key
    cfg = make_config(folder)
    cfg.fit.optimise.maxiter = 13
    assert rs.stage_key(cfg) != key1                     # a fit setting does
    cfg = make_config(folder)
    run = pl.run_pipeline(cfg)
    outdir = run.outdir
    # same folder, different optimiser setting: every stage runs again and the chain is discarded
    cfg2 = make_config(folder); cfg2.fit.optimise.maxiter = 13
    run2 = pl.run_pipeline(cfg2)
    assert any("different data/config/model version" in l for l in run2.log)
    assert any("starting the sampler over" in l for l in run2.log)
    assert not any("resumed from checkpoint" in l for l in run2.log)
    # a partial / corrupt stage file is ignored
    with open(os.path.join(outdir, "de_pass1.json"), "w") as fh:
        fh.write('{"key": "' + rs.stage_key(cfg2) + '", "stage": "de_pass1.json", "complete": false, "payload": {}')
    run3 = pl.run_pipeline(cfg2)
    assert any("de_pass1.json: unreadable" in l or "de_pass1.json: incomplete" in l for l in run3.log)
    assert any("de_pass1 done, checkpoint written" in l for l in run3.log)
    # fit.resume off: nothing is read, everything is written again
    cfg2.fit.resume = "off"
    run4 = pl.run_pipeline(cfg2)
    assert not any("resumed from checkpoint" in l for l in run4.log)
    assert json.load(open(os.path.join(outdir, "diagnostics.json")))["resume"]["written"] == ["de_pass1.json", "continuum_refined.npz", "de_pass2.json"]
    st = rs.status(outdir, rs.stage_key(cfg2))
    assert st["de_pass1.json"] == "valid" and st["chain.h5"].startswith("valid")
    assert rs.status(outdir, "nokey")["de_pass2.json"] == "stale"


def test_blocks_and_sample_mode_resume(reference, tmp_path):
    """Independent blocks (groups mcmc_block0/1 in chain.h5) with the areas sampled: each block resumes."""
    ref_folder, _, _ = reference
    folder = str(tmp_path / "blocks")
    same_data(ref_folder, folder)
    cfg = make_config(folder, nsteps=30, linear="sample", blocks="auto")
    cfg.fit.mcmc.nwalkers = None
    run = pl.run_pipeline(cfg)
    prog = rs.chain_progress(os.path.join(run.outdir, "chain.h5"))
    assert prog and all(v == 30 for v in prog.values())
    c30 = load_chain(run.outdir)
    cfg.fit.mcmc.nsteps = 50
    run2 = pl.run_pipeline(cfg)
    assert run2.mcmc.meta["resumed_steps"] == [30] * len(prog)
    c50 = load_chain(run2.outdir)
    assert c50.shape[0] == 50 and np.array_equal(c50[:30], c30)


def test_marginalise_mode_resume(reference, tmp_path):
    ref_folder, _, _ = reference
    folder = str(tmp_path / "marg")
    same_data(ref_folder, folder)
    cfg = make_config(folder, nsteps=30, linear="marginalise")
    run = pl.run_pipeline(cfg)
    c30 = load_chain(run.outdir)
    cfg.fit.mcmc.nsteps = 45
    run2 = pl.run_pipeline(cfg)
    assert run2.mcmc.meta["resumed_steps"] == [30]
    assert np.array_equal(load_chain(run2.outdir)[:30], c30)


def test_dynesty_checkpoint_and_restore(reference, tmp_path):
    pytest.importorskip("dynesty")
    ref_folder, _, _ = reference
    folder = str(tmp_path / "dyn")
    same_data(ref_folder, folder)
    cfg = make_config(folder)
    cfg.fit.sampler = "dynesty"; cfg.fit.dynesty.nlive = 30; cfg.fit.dynesty.maxcall = 400
    cfg.fit.dynesty.dynamic = False; cfg.fit.dynesty.checkpoint_every = 0.01
    run = pl.run_pipeline(cfg)
    assert os.path.exists(os.path.join(run.outdir, "dynesty.save"))
    assert run.mcmc.meta["resumed"] is False
    run2 = pl.run_pipeline(cfg)
    assert run2.mcmc.meta["resumed"] is True
    assert any("dynesty resumed from dynesty.save" in l for l in run2.log)
    assert np.isfinite(run2.mcmc.logz)


def test_detection_checkpoint(reference, tmp_path):
    ref_folder, _, _ = reference
    folder = str(tmp_path / "det")
    same_data(ref_folder, folder)
    cfg = make_config(folder, nsteps=20)
    cfg.fit.stages = ["optimise"]; cfg.continuum.refine_iterations = 0
    cfg.fit.auto_detect = True; cfg.fit.detect.candidates = ["CO2", "HCN"]; cfg.fit.detect.replace_windows = False
    run = pl.run_pipeline(cfg)
    assert os.path.exists(os.path.join(run.outdir, "detection.json"))
    names = [c.name for c in run.cfg.components]
    run2 = pl.run_pipeline(cfg)
    assert any("detection resumed from checkpoint" in l for l in run2.log)
    assert [c.name for c in run2.cfg.components] == names
    assert run2.detection.table.equals(run.detection.table)
    assert np.array_equal(run2.theta, run.theta)
