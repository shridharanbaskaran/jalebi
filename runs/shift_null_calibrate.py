#!/usr/bin/env python
"""shift_null_calibrate.py (0.23) -- survey-level calibration of the shifted-template null test.

Each disk's shift-null test (jalebi shift-null / fit.shift_null) gives S = (z0 - median z(v)) / robust sigma
of ~38 shifted matches.  Pooling the shifted matches of all disks gives tens of thousands of null S values, so the
tail of the null -- the false-alarm probability of a threshold -- is measured on the survey itself instead of
being assumed Gaussian:

    S_null(v) = (z(v) - median_{v' != v} z(v')) / robust sigma_{v' != v}     (leave-one-out, per disk x unit)
    FAP(S > s) = fraction of pooled S_null above s                         (per molecule and overall)

    python runs/shift_null_calibrate.py RUN_ROOT [--classes census.csv] [--out RUN_ROOT/_shift_null_calibration]

Inputs: RUN_ROOT/*/shift_null.csv and shift_null_curves.csv (written by the pipeline or `jalebi shift-null
--survey RUN_ROOT`).  --classes: a table with columns name (= target folder) and plan_class (survey_science/
census.py), for detection rates per class.
Outputs: calibration.csv (threshold -> pooled FAP, per molecule), detections.csv (one row per disk x unit with S,
FAP_pooled, detected), rates.csv (per molecule x class), REPORT.md, s_distribution.png.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

THRESHOLDS = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]


def load(root: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    tabs, curves = [], []
    for f in sorted(glob.glob(os.path.join(root, "*", "shift_null.csv"))):
        d = os.path.dirname(f)
        if os.path.basename(d).startswith(("_", ".")):          # helper folders (_compiled, _shift_null_calibration)
            continue
        t = pd.read_csv(f)
        t["folder"] = os.path.basename(d)
        tabs.append(t)
        c = os.path.join(d, "shift_null_curves.csv")
        if os.path.exists(c):
            cc = pd.read_csv(c)
            cc["folder"] = os.path.basename(d)
            curves.append(cc)
    if not tabs:
        raise SystemExit(f"no */shift_null.csv under {root}")
    return pd.concat(tabs, ignore_index=True), (pd.concat(curves, ignore_index=True) if curves else pd.DataFrame())


def pooled_null(curves: pd.DataFrame, mol: dict) -> pd.DataFrame:
    """Leave-one-out S of every shifted match (v != 0), per folder x unit."""
    rows = []
    for (fold, unit), g in curves[curves.v_kms != 0].groupby(["folder", "unit"]):
        z = g["z"].to_numpy(float)
        z = z[np.isfinite(z)]
        n = len(z)
        if n < 8:
            continue
        for i in range(n):
            rest = np.delete(z, i)
            med = np.median(rest)
            sig = 1.4826 * np.median(np.abs(rest - med))
            if sig > 0:
                rows.append((fold, unit, mol.get((fold, unit), ""), (z[i] - med) / sig))
    return pd.DataFrame(rows, columns=["folder", "unit", "molecule", "S_null"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--classes", default=None, help="csv with name, plan_class (census.csv)")
    ap.add_argument("--threshold", type=float, default=5.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = a.out or os.path.join(a.root, "_shift_null_calibration")
    os.makedirs(out, exist_ok=True)
    tab, curves = load(a.root)
    mol = {(f, u): m for f, u, m in zip(tab.folder, tab.unit, tab.molecule.fillna(""))}
    null = pooled_null(curves, mol) if len(curves) else pd.DataFrame(columns=["molecule", "S_null"])

    # threshold -> pooled false-alarm probability, overall and per molecule
    cal = []
    for m, g in [("all", null)] + list(null.groupby("molecule")):
        for s in THRESHOLDS:
            n = len(g)
            k = int((g.S_null > s).sum())
            cal.append({"molecule": m, "threshold_S": s, "n_null": n, "n_above": k,
                        "FAP": (k + 1) / (n + 1) if n else np.nan,
                        "gaussian_FAP": float(0.5 * __import__("math").erfc(s / np.sqrt(2)))})
    cal = pd.DataFrame(cal)
    cal.to_csv(os.path.join(out, "calibration.csv"), index=False)

    # per detection: pooled FAP of its S (same molecule when that pool has >= 2000 values, else all)
    pools = {m: np.sort(g.S_null.to_numpy()) for m, g in null.groupby("molecule")}
    allp = np.sort(null.S_null.to_numpy()) if len(null) else np.array([])

    def fap(s, m):
        p = pools.get(m)
        p = p if (p is not None and len(p) >= 2000) else allp
        if not len(p) or not np.isfinite(s):
            return np.nan
        return (len(p) - np.searchsorted(p, s, side="right") + 1) / (len(p) + 1)
    tab["FAP_pooled"] = [fap(s, m) for s, m in zip(tab.S, tab.molecule.fillna(""))]
    tab["detected"] = tab.S >= a.threshold
    if a.classes:
        cl = pd.read_csv(a.classes)[["name", "plan_class"]].rename(columns={"name": "folder"})
        tab = tab.merge(cl, on="folder", how="left")
    else:
        tab["plan_class"] = "all"
    tab.to_csv(os.path.join(out, "detections.csv"), index=False)
    # rates: a molecule counts once per disk (any of its units detected)
    per = tab.groupby(["folder", "plan_class", "molecule"]).detected.any().reset_index()
    rates = per.groupby(["molecule", "plan_class"]).agg(disks=("folder", "nunique"), detected=("detected", "sum")).reset_index()
    rates["percent"] = (100 * rates.detected / rates.disks).round(1)
    rates.to_csv(os.path.join(out, "rates.csv"), index=False)

    L = [f"# Shift-null calibration of {os.path.abspath(a.root)}\n",
         f"{tab.folder.nunique()} fits, {len(tab)} units; pooled null: {len(null)} leave-one-out S values.\n",
         "## Threshold -> false-alarm probability (pooled null, all molecules)\n",
         "| S > | n above | FAP pooled | FAP if Gaussian |", "|---|---|---|---|"]
    for r in cal[cal.molecule == "all"].itertuples():
        L.append(f"| {r.threshold_S:g} | {r.n_above} | {r.FAP:.2e} | {r.gaussian_FAP:.2e} |")
    piv = rates.pivot_table(index="molecule", columns="plan_class", values="percent").round(0)
    try:
        table = piv.to_markdown()                  # needs the optional `tabulate` package
    except ImportError:
        table = "```\n" + piv.to_string() + "\n```"
    L += ["", f"## Detection rates (S >= {a.threshold:g}), per molecule x class\n", table]
    with open(os.path.join(out, "REPORT.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        bins = np.linspace(-6, 30, 145)
        if len(null):
            ax.hist(np.clip(null.S_null, -6, 30), bins=bins, density=True, histtype="step", lw=2, color="#52514e",
                    label=f"pooled null ({len(null)})")
        for cls, col in (("control", "#eb6834"), (None, "#2a78d6")):
            g = tab[tab.plan_class == cls] if cls else tab[tab.plan_class != "control"]
            if len(g):
                ax.hist(np.clip(g.S, -6, 30), bins=bins, density=True, histtype="step", lw=2, color=col,
                        label=f"{cls or 'disks'} units ({len(g)})")
        ax.axvline(a.threshold, color="#0b0b0b", lw=1, ls="--")
        ax.set_yscale("log"); ax.set_xlabel("S (shift-null significance; values > 30 at 30)"); ax.set_ylabel("density")
        ax.legend(frameon=False, fontsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout(); fig.savefig(os.path.join(out, "s_distribution.png"), dpi=150); plt.close(fig)
    except Exception as e:  # pragma: no cover
        print(f"plot failed: {e}")
    print(f"\nwritten to {out}/")


if __name__ == "__main__":
    main()
