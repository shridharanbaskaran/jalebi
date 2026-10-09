#!/usr/bin/env python
"""corner_study.py (0.22.1) -- which setting stops water (and CH4 / CO / CO2) from turning into a pseudo-continuum?

The 0.21 survey put H2O_hot at T = 1500 K, log N = 21 (the prior corner, R ~ 0.01 au) in 202 of 265 fits, and did
the same in EVERY object without disk gas (white dwarfs, debris disks, background stars behind dark cores).  This
study re-fits a small subset -- those no-disk controls + the published validation disks (+ optionally a random
sample of survey disks) -- under a few settings and scores each setting on

  * controls:   how many still get a pinned (corner) component, and how many get a non-pinned component with
                dBIC_eff > 10 (= false detections, see detection_thresholds.py);
  * validation: |T_fit - T_pub| and |log N_fit - log N_pub| of the water (and CO2 / HCN) components, matched by
                temperature rank, over the disks with published slab fits;
  * all:        median chi2_red, fraction converged, CPU time.

Settings (built in; --settings-file to replace them, same layout as `list` prints):
  base            survey_0.22.yaml as it is (0.22.1 code: NaN-safe walkers, emulator fallback: relative)
  h2o_logN20      water log N <= 20 (bounds_by_molecule.H2O)
  h2o_cap         water log N <= 20 and T <= 1300 K
  spline          joint continuum correction, cubic spline, 1 um knots, 2 % prior (fit.continuum_fit: spline)
  spline_h2o20    spline + water log N <= 20

    python runs/corner_study.py list
    python runs/corner_study.py prepare --base runs/survey_0.22.yaml --targets runs/survey_targets.csv [--random 20]
    python runs/corner_study.py run --workers 30 [--settings base,spline] [--dry-run]
    python runs/corner_study.py analyse                 # -> results/corner_study/RANKING.md + ranking.csv

`run` calls run_survey_server.py (runner_core) once per setting, so a killed study re-runs only the unfinished
disks; with fit.resume: auto (0.22) a disk continues from its last stage.  Lives in runs/ next to
run_survey_server.py, collect_results.py and detection_thresholds.py (imports them).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from detection_thresholds import DEFAULT_CONTROLS, calibrate, controls_from_targets  # noqa: E402

STUDY = os.path.join(HERE, "results", "corner_study")

SETTINGS = {
    "base": {},
    "h2o_logN20": {"fit": {"bounds_by_molecule": {"H2O": {"logN": [13.0, 20.0]}}}},
    "h2o_cap": {"fit": {"bounds_by_molecule": {"H2O": {"logN": [13.0, 20.0], "T": [100.0, 1300.0]}}}},
    "spline": {"fit": {"continuum_fit": "spline",
                       "continuum_correction": {"knot_spacing_um": 1.0, "prior_width": 0.02}}},
    "spline_h2o20": {"fit": {"continuum_fit": "spline",
                             "continuum_correction": {"knot_spacing_um": 1.0, "prior_width": 0.02},
                             "bounds_by_molecule": {"H2O": {"logN": [13.0, 20.0]}}}},
}

# survey target name -> published slab fits (validation_known; first paper per source; T [K], log N [cm^-2])
PUBLISHED = {
    "V-FZ-TAU": {"H2O": [(953, 18.38), (516, 18.20)], "CO": [(1400, 18.48)]},
    "V-DR-TAU": {"H2O": [(806, 19.2), (468, 18.5), (181, 17.9)]},
    "V-CX-TAU": {"H2O": [(550, 19.0)], "CO2": [(450, 17.9)]},
    "V-GW-LUP": {"H2O": [(625, 18.51)], "CO2": [(400, 18.34)], "HCN": [(875, 17.66)], "C2H2": [(500, 17.66)]},
    "SZ-114": {"H2O": [(750, 18.8), (450, 18.2)], "CO2": [(500, 17.6)], "HCN": [(870, 15.9)]},
    "V-DF-TAU": {"H2O": [(920, 19.26), (490, 18.27), (180, 17.53)], "CO2": [(400, 18.78)], "HCN": [(800, 16.73)]},
    "V-GK-TAU": {"H2O": [(905, 18.25), (365, 17.99)]},
    "V-HP-TAU": {"H2O": [(858, 18.52), (468, 17.62)]},
    "V-GQ-Lup": {"H2O": [(836, 18.17), (361, 17.66)]},
    "V-IQ-TAU": {"H2O": [(926, 18.57), (333, 17.87)]},
    "AS-209": {"H2O": [(830, 19.45), (400, 17.66)]},
    "V-CI-TAU": {"H2O": [(840, 18.0)]},
    "V-BP-TAU": {"H2O": [(933, 18.5), (516, 18.2), (259, 15.7)]},
    "V-CY-TAU": {"H2O": [(821, 18.7), (535, 18.1), (268, 15.6)]},
    "V-DN-TAU": {"H2O": [(486, 18.7), (239, 15.8)]},
    "V-FT-TAU": {"H2O": [(920, 18.5), (364, 18.3), (217, 16.0)]},
    "PGZ2001-J160532.1-193315": {"C2H2": [(525, 20.38)], "CO2": [(430, 18.30)]},
}


def deep_update(d: dict, u: dict) -> dict:
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = copy.deepcopy(v)
    return d


def load_settings(path: str | None) -> dict:
    if not path:
        return SETTINGS
    with open(path) as fh:
        return yaml.safe_load(fh)


def cmd_list(a):
    print(yaml.safe_dump(load_settings(a.settings_file), sort_keys=False))


def subset(targets: str, random_n: int, seed: int) -> pd.DataFrame:
    t = pd.read_csv(targets)
    ctl = {**DEFAULT_CONTROLS, **controls_from_targets(targets)}
    keep = t["name"].isin(ctl) | t["name"].isin(PUBLISHED)
    rest = t[~keep]
    if random_n:
        keep |= t["name"].isin(rest.sample(min(random_n, len(rest)), random_state=seed)["name"])
    s = t[keep].copy()
    s["study_role"] = np.where(s["name"].isin(ctl), "control", np.where(s["name"].isin(PUBLISHED), "validation", "random"))
    return s


def cmd_prepare(a):
    os.makedirs(STUDY, exist_ok=True)
    s = subset(a.targets, a.random, a.seed)
    tpath = os.path.join(STUDY, "targets_subset.csv")
    s.to_csv(tpath, index=False)
    with open(a.base) as fh:
        base = yaml.safe_load(fh)
    for name, st in load_settings(a.settings_file).items():
        d = deep_update(copy.deepcopy(base), st or {})
        d["output"] = os.path.join(STUDY, name, "{target}")
        os.makedirs(os.path.join(STUDY, name), exist_ok=True)
        with open(os.path.join(STUDY, name, "config.yaml"), "w") as fh:
            yaml.safe_dump(d, fh, sort_keys=False)
    print(f"{len(s)} targets ({(s.study_role == 'control').sum()} controls, {(s.study_role == 'validation').sum()} "
          f"validation, {(s.study_role == 'random').sum()} random) -> {tpath}")
    print(f"configs in {STUDY}/<setting>/config.yaml for: {', '.join(load_settings(a.settings_file))}")


def cmd_run(a):
    names = [x for x in (a.settings.split(",") if a.settings else load_settings(a.settings_file))]
    tpath = os.path.join(STUDY, "targets_subset.csv")
    for n in names:
        cfg = os.path.join(STUDY, n, "config.yaml")
        cmd = [sys.executable, os.path.join(HERE, "run_survey_server.py"), "run", "--config", cfg, "--targets", tpath,
               "--skip-preflight", "--timeout-h", str(a.timeout_h)] + (["--workers", str(a.workers)] if a.workers else [])
        print(" ".join(cmd))
        if not a.dry_run:
            subprocess.run(cmd, check=False)


def _validation(comp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tgt, mols in PUBLISHED.items():
        for mol, pub in mols.items():
            f = comp[(comp.target == tgt) & (comp.molecule == mol) & (comp.component != "H2O_rovib") & ~comp.pinned]
            f = f.sort_values("median_T", ascending=False)
            for k, (T, N) in enumerate(sorted(pub, reverse=True)):
                if k < len(f):
                    r = f.iloc[k]
                    rows.append({"target": tgt, "molecule": mol, "rank": k, "T_pub": T, "logN_pub": N,
                                 "T_fit": r.median_T, "logN_fit": r.median_logN})
                else:
                    rows.append({"target": tgt, "molecule": mol, "rank": k, "T_pub": T, "logN_pub": N,
                                 "T_fit": np.nan, "logN_fit": np.nan})
    return pd.DataFrame(rows)


def _md(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False, floatfmt=".3g")
    except ImportError:                          # tabulate not installed
        return "```\n" + df.to_string(index=False) + "\n```"


def cmd_analyse(a):
    import collect_results as CR
    ctl = {**DEFAULT_CONTROLS, **(controls_from_targets(a.targets) if a.targets else {})}
    rank = []
    for n in load_settings(a.settings_file):
        run_dir = os.path.join(STUDY, n)
        if not os.path.isdir(run_dir):
            continue
        tdirs = sorted(os.path.join(run_dir, x) for x in os.listdir(run_dir)
                       if os.path.isdir(os.path.join(run_dir, x)) and not x.startswith(("_", ".")))
        if not tdirs:
            continue
        res = [CR.collect_target(t, True) for t in tdirs]
        comp_dir = os.path.join(run_dir, "_compiled")
        os.makedirs(comp_dir, exist_ok=True)
        t = pd.DataFrame([r["row"] for r in res])
        t.to_csv(os.path.join(comp_dir, "targets.csv"), index=False)
        for key, fn in (("params", "params_long.csv"), ("sig", "significance.csv")):
            fr = [r[key] for r in res if r[key] is not None]
            (pd.concat(fr, ignore_index=True) if fr else pd.DataFrame()).to_csv(os.path.join(comp_dir, fn), index=False)
        try:
            comp, th = calibrate(comp_dir, ctl)
        except Exception as e:                  # nothing finished yet
            print(f"{n}: cannot score yet ({e})"); continue
        comp.to_csv(os.path.join(comp_dir, "components_calibrated.csv"), index=False)
        v = _validation(comp)
        v.to_csv(os.path.join(comp_dir, "validation_pairs.csv"), index=False)
        c = comp[comp.control]; d = comp[~comp.control]
        ok = t[t.status == "ok"]
        w = v[v.molecule == "H2O"]
        rank.append({
            "setting": n, "targets_ok": len(ok), "failed": int((t.status == "failed").sum()),
            "controls_fitted": int(c.target.nunique()),
            "controls_with_pinned": int(c[c.pinned].target.nunique()),
            "controls_false_detection": int(c[c.detected_calibrated_floor].target.nunique()),
            "disks_with_pinned_water": int(d[(d.molecule == "H2O") & d.pinned].target.nunique()),
            "disks_with_water": int(d[(d.molecule == "H2O") & d.detected_calibrated_floor].target.nunique()),
            "water_pairs_matched": int(w.T_fit.notna().sum()), "water_pairs": len(w),
            "water_median_abs_dT": float((w.T_fit - w.T_pub).abs().median()),
            "water_median_abs_dlogN": float((w.logN_fit - w.logN_pub).abs().median()),
            "all_median_abs_dT": float((v.T_fit - v.T_pub).abs().median()),
            "median_chi2_red": float(pd.to_numeric(ok.get("bf_chi2_red"), errors="coerce").median()),
            "converged_frac": float(pd.to_numeric(ok.get("converged"), errors="coerce").mean()),
            "cpu_h": float(pd.to_numeric(ok.get("runtime_min"), errors="coerce").sum() / 60),
        })
    if not rank:
        sys.exit("no finished setting yet")
    r = pd.DataFrame(rank).sort_values(["controls_false_detection", "controls_with_pinned", "water_median_abs_dT"])
    r.to_csv(os.path.join(STUDY, "ranking.csv"), index=False)
    md = ["# corner study ranking", "",
          "Sorted by false detections in the no-disk controls, then pinned controls, then water |dT| vs published.",
          "", _md(r)]
    with open(os.path.join(STUDY, "RANKING.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["list", "prepare", "run", "analyse"])
    ap.add_argument("--base", default=os.path.join(HERE, "survey_0.22.yaml"))
    ap.add_argument("--targets", default=os.path.join(HERE, "survey_targets.csv"))
    ap.add_argument("--settings-file", default=None)
    ap.add_argument("--settings", default=None, help="run: comma-separated subset")
    ap.add_argument("--random", type=int, default=0, help="prepare: add N random survey disks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--timeout-h", type=float, default=12.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    {"list": cmd_list, "prepare": cmd_prepare, "run": cmd_run, "analyse": cmd_analyse}[a.command](a)


if __name__ == "__main__":
    main()
