"""`jalebi showcase`: panels from a results folder, without refitting."""
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from jalebi import showcase as S


def _fake_run(tmp_path, refined=True, chain=True):
    f = tmp_path / "res"
    f.mkdir()
    rng = np.random.default_rng(1)
    w = np.linspace(13.5, 16.5, 600)
    cont = 1.0 + 0.02 * (w - 15)
    line = lambda c, a: a * np.exp(-0.5 * ((w - c) / 0.004) ** 2)
    h2o = sum(line(c, 0.2) for c in (13.7, 14.2, 15.1, 15.9, 16.3))
    co2 = line(14.98, 0.1)
    noise = rng.normal(0, 0.005, w.size)
    cfg = {"target": {"name": "Fake Disk", "path": "nowhere.csv", "distance_pc": 140.0, "rv_kms": 0.0},
           "continuum": {"method": "irsqr", "lam": 1e3, "p": 0.01, "protect": True, "protected": {"CO2 Q": [14.93, 15.02]},
                         "refine_iterations": 1 if refined else 0},
           "fit": {"windows": [[13.5, 16.5]]},
           "components": [{"name": "H2O", "molecule": "H2O", "kind": "slab"}, {"name": "CO2", "molecule": "CO2", "kind": "slab"}]}
    yaml.safe_dump(cfg, open(f / "config.yaml", "w"))
    band = np.where(w < 15.0, "ch3A", "ch3B")
    with open(f / "prep.csv", "w") as fh:
        fh.write("# name=Fake Disk distance_pc=140.0\n")
        pd.DataFrame({"wave": w, "flux": cont + h2o + co2 + noise, "err": 0.005, "band": band, "continuum": cont,
                      "mask": True}).to_csv(fh, index=False)
    pd.DataFrame({"wave": w, "band": band, "data": h2o + co2 + noise, "sigma": 0.005, "model": h2o + co2, "continuum": cont,
                  "model_H2O": h2o, "model_CO2": co2}).to_csv(f / "model.csv", index=False)
    json.dump({"params": {"H2O": {"logN": 18.5, "T": 600.0, "logR": -0.5}, "CO2": {"logN": 17.0, "T": 500.0, "logR": -0.7}},
               "chi2": 650.0, "chi2_red": 1.1, "npix": 600, "tau_max": {"H2O": 5.0, "CO2": 0.5}}, open(f / "best_fit.json", "w"))
    pd.DataFrame({"component": ["H2O", "CO2"], "delta_chi2": [900.0, 60.0], "k": [3, 3], "delta_BIC": [880.0, 41.0],
                  "tau_max": [5.0, 0.5], "detected": [True, True]}).to_csv(f / "detections.csv", index=False)
    json.dump({"H2O": 0.0, "CO2": 0.9}, open(f / "tau_flags.json", "w"))
    if refined:
        np.savez(f / "continuum_refined.npz", continuum=cont + 0.001, mask=np.ones(w.size, bool), accepted=np.array(True),
                 chi2_before=np.array(700.0), chi2_after=np.array(650.0))
    if chain:
        names = ["H2O.logN", "H2O.T", "H2O.logR", "CO2.logN", "CO2.T", "CO2.logR"]
        mu = np.array([18.5, 600, -0.5, 17.0, 500, -0.7]); sd = np.array([0.05, 20, 0.01, 0.2, 40, 0.05])
        ch = mu + sd * rng.normal(size=(200, 16, 6))
        np.savez(f / "chain.npz", chain=ch, log_prob=np.zeros((200, 16)), acceptance=np.full(16, 0.3), names=np.array(names))
        rows = [{"parameter": n, "label": n, "median": m, "minus": s, "plus": s, "lo_bound": 0.0, "hi_bound": 2000.0,
                 "at_edge": False, "reported": True} for n, m, s in zip(names, mu, sd)]
        pd.DataFrame(rows).to_csv(f / "summary.csv", index=False)
        json.dump({"acceptance": 0.3, "steps_over_tau": 60.0, "rhat": [1.01] * 6, "converged_length": True, "rhat_ok": True},
                  open(f / "diagnostics.json", "w"))
        pd.DataFrame(np.corrcoef(ch.reshape(-1, 6).T), index=names, columns=names).to_csv(f / "correlation.csv")
        np.savez(f / "grid_H2O.npz", logN=np.linspace(16, 20, 5), T=np.linspace(200, 1000, 5), chi2=rng.uniform(1, 10, (5, 5)),
                 logR=np.zeros((5, 5)), tau_max=np.ones((5, 5)))
    return f


def test_showcase_writes_panels(tmp_path):
    f = _fake_run(tmp_path)
    res = S.showcase(str(f), out=str(tmp_path / "sc"), video=False, draws=0, say=lambda m: None)
    names = sorted(p.split("/")[-1] for p in res["png"])
    for key in ("overview", "data", "continuum", "fit", "molecules", "evidence", "component_H2O", "component_CO2",
                "uncertainties", "table", "correlation", "traces"):
        assert any(key in n for n in names), key
    assert res["video"] is None


def test_showcase_without_chain_or_refinement(tmp_path):
    f = _fake_run(tmp_path, refined=False, chain=False)
    res = S.showcase(str(f), out=str(tmp_path / "sc"), video=False, draws=0, say=lambda m: None)
    names = " ".join(res["png"])
    assert "continuum" in names and "molecules" in names and "traces" not in names


def test_refinement_passes_and_magnify(tmp_path):
    f = _fake_run(tmp_path)
    run = S.load_run(str(f))
    p = S.refinement_passes(run)
    assert len(p) == 1 and p[0]["accepted"] and p[0]["chi2_after"] < p[0]["chi2_before"]
    assert S._magnify(1.0, 1.0) == 1 and S._magnify(0.01, 1.0) == 50


def test_showcase_rejects_non_results(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        S.load_run(str(tmp_path / "empty"))
