#!/usr/bin/env python
"""continuum_study.py (0.22) -- which continuum treatment reproduces the published values best?

Runs the known-parameter validation set (validation_known.yaml, or --only a subset) under several continuum
settings -- the estimators jalebi has (IRSQR / median_sg / aspls / asls ... with their key parameters) and the
0.22 joint continuum correction (fit.continuum_fit: offset | spline) -- scores every run against the published
values with jalebi_runs.py compare, and ranks the settings.  It PREPARES and LAUNCHES the runs; the heavy part
goes through RUN_ME_continuum_study.sh (hours on 17-32 cores).  Lives next to jalebi_runs.py (imports it).

    python continuum_study.py list                                  # the built-in settings
    python continuum_study.py prepare validation_known.yaml         # derived manifests -> results/continuum_study/<setting>/
    python continuum_study.py run validation_known.yaml --workers 17 [--only DR_Tau,GW_Lup] [--settings a,b]
    python continuum_study.py rank validation_known.yaml            # ranking table (csv + md) from the scores
    python continuum_study.py run ... --dry-run                      # only write the manifests and print the commands

A setting is {continuum: {...overrides...}, fit: {continuum_fit: ..., continuum_correction: {...}}}; add your own
with --settings-file my_settings.yaml (same layout as `list` prints; the file replaces the built-in list).  Source-level continuum overrides of the
manifest (DN Tau's knot spacing, say) are removed so that every source sees the same setting (--keep-source-continuum
keeps them).  Ranking: the strict / 0.20-rule / lenient pass counts over all scored quantities, the number of tier-A
sources, the median chi2_red and the number of corner-flagged components; sorted by the 0.20-rule pass count.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SETTINGS = {
    # the 0.20 survey continuum and its neighbours
    "irsqr_q0.1_k25": {"continuum": {"method": "irsqr", "quantile": 0.1, "knot_spacing": 25}},
    "irsqr_q0.05_k25": {"continuum": {"method": "irsqr", "quantile": 0.05, "knot_spacing": 25}},
    "irsqr_q0.2_k25": {"continuum": {"method": "irsqr", "quantile": 0.2, "knot_spacing": 25}},
    "irsqr_q0.1_k50": {"continuum": {"method": "irsqr", "quantile": 0.1, "knot_spacing": 50}},
    "irsqr_q0.1_k75": {"continuum": {"method": "irsqr", "quantile": 0.1, "knot_spacing": 75}},
    "irsqr_q0.1_k25_norefine": {"continuum": {"method": "irsqr", "quantile": 0.1, "knot_spacing": 25, "refine_iterations": 0}},
    # other estimators
    "median_sg_w101_p25": {"continuum": {"method": "median_sg", "median_window": 101, "median_percentile": 25.0}},
    "median_sg_w201_p10": {"continuum": {"method": "median_sg", "median_window": 201, "median_percentile": 10.0}},
    "aspls_5e6": {"continuum": {"method": "aspls", "aspls_lam": 5e6}},
    "aspls_5e5": {"continuum": {"method": "aspls", "aspls_lam": 5e5}},
    "asls_1e3": {"continuum": {"method": "asls", "lam": 1e3, "p": 0.01}},
    # 0.22 joint continuum correction on top of the survey continuum
    "irsqr_q0.1_k25+offset": {"continuum": {"method": "irsqr", "quantile": 0.1, "knot_spacing": 25},
                              "fit": {"continuum_fit": "offset", "continuum_correction": {"prior_width": 0.02}}},
    "irsqr_q0.1_k25+spline1um": {"continuum": {"method": "irsqr", "quantile": 0.1, "knot_spacing": 25},
                                 "fit": {"continuum_fit": "spline", "continuum_correction": {"knot_spacing_um": 1.0, "prior_width": 0.02}}},
    "irsqr_q0.1_k25+spline0.5um": {"continuum": {"method": "irsqr", "quantile": 0.1, "knot_spacing": 25},
                                   "fit": {"continuum_fit": "spline", "continuum_correction": {"knot_spacing_um": 0.5, "prior_width": 0.02}}},
    "irsqr_q0.1_k25+spline1um_w5": {"continuum": {"method": "irsqr", "quantile": 0.1, "knot_spacing": 25},
                                    "fit": {"continuum_fit": "spline", "continuum_correction": {"knot_spacing_um": 1.0, "prior_width": 0.05}}},
}


def _load(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def _root(a):
    return os.path.abspath(a.root)


def derived_manifest(man: dict, name: str, setting: dict, root: str, keep_source_continuum: bool) -> dict:
    m = copy.deepcopy(man)
    d = m.setdefault("defaults", {})
    d.setdefault("continuum", {}).update(setting.get("continuum", {}))
    if "fit" in setting:
        f = d.setdefault("fit", {})
        for k, v in setting["fit"].items():
            if isinstance(v, dict):
                f.setdefault(k, {}).update(v)
            else:
                f[k] = v
    if not keep_source_continuum:
        for src in m.get("sources", []):
            if isinstance(src.get("config"), dict):
                src["config"].pop("continuum", None)
    m["output"] = os.path.join(root, name, "{run}")
    m["_continuum_study"] = {"setting": name, "overrides": setting}
    return m


def settings_from_args(a) -> dict:
    s = dict(SETTINGS)
    if getattr(a, "settings_file", None):                 # a settings file replaces the built-in list
        s = dict(_load(a.settings_file) or {})
    if getattr(a, "settings", None):
        keep = [x for x in a.settings.split(",") if x]
        unknown = [x for x in keep if x not in s]
        if unknown:
            sys.exit(f"unknown settings {unknown}; known: {', '.join(s)}")
        s = {k: s[k] for k in keep}
    return s


def cmd_list(a):
    print(yaml.safe_dump(settings_from_args(a), sort_keys=False))


def prepare(a) -> list[tuple[str, str]]:
    man = _load(a.manifest)
    root = _root(a)
    os.makedirs(root, exist_ok=True)
    out = []
    for name, setting in settings_from_args(a).items():
        folder = os.path.join(root, name)
        os.makedirs(folder, exist_ok=True)
        m = derived_manifest(man, name, setting, root, a.keep_source_continuum)
        # base / targets paths are relative to the manifest: make them absolute in the derived copy
        for key in ("base", "targets"):
            if m.get(key) and not os.path.isabs(m[key]):
                m[key] = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(a.manifest)), m[key]))
        p = os.path.join(folder, "validation_known_" + name + ".yaml")
        with open(p, "w") as fh:
            yaml.safe_dump(m, fh, sort_keys=False, allow_unicode=True)
        out.append((name, p))
        print(f"{name:32s} -> {p}")
    return out


def cmd_prepare(a):
    prepare(a)


def cmd_run(a):
    pairs = prepare(a)
    for name, p in pairs:
        cmd = [sys.executable, os.path.join(HERE, "jalebi_runs.py"), "run", p, "--workers", str(a.workers)]
        if a.only:
            cmd += ["--only", a.only]
        if a.stages:
            cmd += ["--stages", a.stages]
        print("\n$ " + " ".join(cmd), flush=True)
        if a.dry_run:
            continue
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"! setting {name}: jalebi_runs.py run returned {r.returncode} (continuing with the next setting)")
    if not a.dry_run:
        cmd_rank(a)


def cmd_rank(a):
    import numpy as np
    import pandas as pd
    import jalebi_runs as jr
    root = _root(a)
    rows = []
    for name in settings_from_args(a):
        folder = os.path.join(root, name)
        p = os.path.join(folder, "validation_known_" + name + ".yaml")
        if not os.path.exists(p):
            continue
        sc_path = os.path.join(folder, "validation_scores.csv")
        if not os.path.exists(sc_path) or a.rescore:
            if not any(os.path.exists(os.path.join(folder, d, "summary.csv")) for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))):
                continue
            jr.cmd_compare(argparse.Namespace(results=folder, manifest=p, output=sc_path, only=a.only, mode="known"))
        sc = pd.read_csv(sc_path)
        if a.only:
            sc = sc[sc.run.isin(a.only.split(","))]
        row = {"setting": name}
        for col, lab in (("status_strict", "strict"), ("status", "rule_0.20"), ("status_lenient", "lenient")):
            c = col if col in sc else "status"
            st = sc[c].astype(str)
            row[f"pass_{lab}"] = int((st == "PASS").sum()); row[f"scored_{lab}"] = int(st.isin(["PASS", "FAIL"]).sum())
            tiers = jr_gate(sc, c)
            row[f"tierA_{lab}"] = sum(v == "A" for v in tiers.values())
        chi2, corner = [], 0
        for d in os.listdir(folder):
            bf = os.path.join(folder, d, "best_fit.json"); dg = os.path.join(folder, d, "diagnostics.json")
            if os.path.exists(bf):
                try:
                    chi2.append(float(json.load(open(bf)).get("chi2_red", np.nan)))
                except Exception:
                    pass
            if os.path.exists(dg):
                try:
                    corner += sum(1 for v in json.load(open(dg)).get("corner", {}).values() if v.get("flags"))
                except Exception:
                    pass
        row["n_results"] = len(chi2); row["median_chi2_red"] = float(np.nanmedian(chi2)) if chi2 else np.nan
        row["corner_flagged_components"] = corner
        rows.append(row)
    if not rows:
        sys.exit(f"no scored runs under {root}; run the study first")
    t = pd.DataFrame(rows).sort_values(["pass_rule_0.20", "pass_strict"], ascending=False)
    t.to_csv(os.path.join(root, "continuum_study_ranking.csv"), index=False)
    md = ["# Continuum study ranking", "", f"Manifest: {a.manifest}; root: {root}", "",
          "| setting | strict | 0.20 rule | lenient | tier A (0.20) | median chi2_red | corner flags | n |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in t.to_dict("records"):          # (itertuples renames "pass_rule_0.20" to _N: use dicts)
        md.append(f"| {r['setting']} | {r['pass_strict']}/{r['scored_strict']} | {r['pass_rule_0.20']}/{r['scored_rule_0.20']} | "
                  f"{r['pass_lenient']}/{r['scored_lenient']} | {r['tierA_rule_0.20']} | {r['median_chi2_red']:.2f} | "
                  f"{r['corner_flagged_components']} | {r['n_results']} |")
    open(os.path.join(root, "continuum_study_ranking.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nranking -> {os.path.join(root, 'continuum_study_ranking.csv')}")


def jr_gate(sc, col):
    per = {}
    for run, s in sc.groupby("run", sort=False):
        st = s[col].astype(str); st = st[st != "n/a"]
        if st.str.startswith("NO RESULT").all():
            per[run] = "no result"
        else:
            per[run] = "A" if (st == "PASS").all() else f"{(st == 'PASS').sum()}/{len(st)}"
    return per


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=os.path.join(HERE, "results", "continuum_study"))
    common.add_argument("--settings", help="comma-separated subset of the settings")
    common.add_argument("--settings-file", help="YAML with extra settings")
    common.add_argument("--only", help="comma-separated run names of the manifest")
    common.add_argument("--keep-source-continuum", action="store_true")
    p = sub.add_parser("list", parents=[common]); p.set_defaults(f=cmd_list)
    p = sub.add_parser("prepare", parents=[common]); p.add_argument("manifest"); p.set_defaults(f=cmd_prepare)
    p = sub.add_parser("run", parents=[common]); p.add_argument("manifest"); p.add_argument("--workers", type=int, default=17)
    p.add_argument("--stages"); p.add_argument("--dry-run", action="store_true"); p.add_argument("--rescore", action="store_true")
    p.set_defaults(f=cmd_run)
    p = sub.add_parser("rank", parents=[common]); p.add_argument("manifest"); p.add_argument("--rescore", action="store_true")
    p.set_defaults(f=cmd_rank)
    a = ap.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
