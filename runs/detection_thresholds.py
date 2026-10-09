#!/usr/bin/env python
"""detection_thresholds.py (0.22.1) -- calibrate the molecule detections of a survey run on no-disk controls.

Why: in the 0.21 survey every one of 22 objects without disk gas (white dwarfs, debris disks, background stars
behind dark cores) got hot + ro-vibrational water at the prior corner (T 1500 K, log N 21, R ~ 0.01 au) with
dBIC 3000-11000, and several got CH4 / CO / CO2 too.  dBIC > 10 is therefore not a detection test for these fits:
(1) the residuals are correlated (lag-1 ACF ~ 0.5) and over-dispersed (noise scale s ~ 1.7), which inflates every
Delta chi^2, and (2) an optically thick slab pinned at the corner acts as a smooth pseudo-continuum that soaks up
continuum errors.  This script applies three corrections and reports what survives:

  * dBIC_eff = Delta chi^2 / (s^2 f) - k ln(n / f),  f = (1 + rho) / (1 - rho)
    (rho = lag-1 residual autocorrelation: the effective number of independent pixels is n / f for AR(1) noise);
  * pinned: log N within `--pin-logN` of the prior upper bound (default 0.5 dex: > 20.5 for a log N <= 21 prior), or
    T within 2 % of the upper bound with log N > 20 -- the pseudo-continuum corner;
  * per-molecule empirical threshold: the largest dBIC_eff of a non-pinned component of that molecule among the
    controls (x `--margin`, default 1.0), never below `--floor` (default 10).

Input: the folder written by collect_results_0.21.py (targets.csv, significance.csv, params_long.csv).
Controls: --targets survey_targets.csv (flag_non_stellar_otype, flag_debris_disk_program, dark-core sight lines),
or --controls FILE (csv with columns target,class, or one target per line); default = the 37 such objects of the
0.21 survey (DEFAULT_CONTROLS below; 35 were fitted).

    python runs/detection_thresholds.py COMPILED_DIR [--controls controls.csv] [--out COMPILED_DIR/calibrated]

Writes thresholds.csv (per molecule), components_calibrated.csv (one row per target x component with
dBIC_eff, pinned, threshold, detected_calibrated), detection_rates.csv and REPORT_calibration.txt.
Caveat: 22 heterogeneous controls give a coarse threshold; ice-covered background stars carry real CO2 / CH4 ice
bands that an emission model partly fits, so their CO2 / CH4 thresholds are conservative for disks.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

DEFAULT_CONTROLS = {
    # flag_non_stellar_otype in survey_targets.csv
    "GD-362": "white dwarf", "WD0145+234": "white dwarf", "IRAS-17178-2600": "non-stellar",
    # flag_debris_disk_program
    "-BET-PIC": "debris", "BETA-PIC-disk-offset-cats-tail": "debris", "-ETA-TEL": "debris", "-49-CET": "debris",
    "EXZ-ETA-CRV": "debris", "HD-172555": "debris", "HD-32297": "debris", "HD-131835": "debris",
    "HD-23514": "debris", "HD-15407": "debris", "NGC2547-ID8": "debris", "NGC2547-ID8-dr3": "debris",
    "HD166": "debris", "J0605": "debris", "J0609": "debris", "J0611": "debris", "J0712": "debris", "J0925": "debris",
    "J1044": "debris", "J1213": "debris", "J2043": "debris", "J2301": "debris", "RZPSC": "debris",
    "V-V488-PER": "debris", "V-V488-Per-dr3": "debris",
    # background stars behind dark cores (ice sight lines; B68N03 / N04 are G0/M4 III giants)
    "B68N02": "core background star", "B68N03": "core background star", "B68N04": "core background star",
    "L694-2N01": "core background star", "L694-2N02": "core background star", "L694-2N07": "core background star",
    "B335N01": "core background star", "B335N04": "core background star", "B335N16": "core background star",
}
CORE_PATTERN = r"^(?:B68N|L694-2N|B335N)"


def controls_from_targets(path: str) -> dict[str, str]:
    """Controls from a survey targets table: flag_non_stellar_otype, flag_debris_disk_program, dark-core sight lines."""
    t = pd.read_csv(path)
    out = {}
    for _, r in t.iterrows():
        n = str(r["name"])
        if bool(r.get("flag_non_stellar_otype", False)):
            out[n] = "non-stellar"
        elif bool(r.get("flag_debris_disk_program", False)):
            out[n] = "debris"
        elif pd.Series([n]).str.contains(CORE_PATTERN).iloc[0]:
            out[n] = "core background star"
    return out


MOLECULES = ["13CCH2", "13CO2", "13CO", "H13CN", "HC3N", "C2H2", "C2H4", "C2H6", "C4H2", "C6H6", "C3H4",
             "CO2", "HCN", "H2O", "CH4", "NH3", "OH", "CO"]


def guess_molecule(comp: str) -> str:
    for m in MOLECULES:
        if comp == m or comp.startswith(m + "_") or comp.startswith(m):
            return m
    return ""


def load_controls(path: str | None, targets: str | None = None) -> dict[str, str]:
    if targets:                          # the table's flags plus the known list (some debris disks are not flagged)
        return {**DEFAULT_CONTROLS, **controls_from_targets(targets)}
    if not path:
        return dict(DEFAULT_CONTROLS)
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        cls = df["class"] if "class" in df else "control"
        return dict(zip(df["target"].astype(str), np.broadcast_to(cls, len(df))))
    with open(path) as fh:
        return {ln.strip(): "control" for ln in fh if ln.strip() and not ln.startswith("#")}


def calibrate(d: str, controls: dict[str, str], pin_logN: float = 0.5, margin: float = 1.0, floor: float = 10.0,
              exclude=None):
    t = pd.read_csv(os.path.join(d, "targets.csv")).set_index("target")
    s = pd.read_csv(os.path.join(d, "significance.csv"))
    p = pd.read_csv(os.path.join(d, "params_long.csv"))
    if "molecule" not in s or s["molecule"].isna().any():
        s["molecule"] = [m if isinstance(m, str) and m else guess_molecule(str(c)) for m, c in
                         zip(s.get("molecule", [""] * len(s)), s["component"])]
    # pinned / corner flags from the posterior medians and the prior bounds
    q = p[p["quantity"].isin(["T", "logN"]) & p["free"].astype(str).str.lower().eq("true")]
    piv = q.pivot_table(index=["target", "component"], columns="quantity", values=["median", "hi_bound"], aggfunc="first")
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.reset_index()
    s = s.merge(piv, on=["target", "component"], how="left")
    hiN = s.get("hi_bound_logN", pd.Series(21.0, index=s.index)).fillna(21.0)
    hiT = s.get("hi_bound_T", pd.Series(np.nan, index=s.index))
    s["pinned"] = (s["median_logN"] > hiN - pin_logN) | ((s["median_T"] > 0.98 * hiT) & (s["median_logN"] > hiN - 1.0))
    # correlated, over-dispersed residuals
    sc = t["noise_scale_s"].astype(float) if "noise_scale_s" in t else 10 ** t["bf_log_s"].astype(float)
    s["s"] = s["target"].map(sc).fillna(1.0)
    s["rho"] = s["target"].map(t.get("resid_lag1_acf", pd.Series(dtype=float))).fillna(0.0).clip(0.0, 0.95)
    s["npix"] = s["target"].map(t["bf_npix"]).astype(float)
    s["f_corr"] = (1 + s["rho"]) / (1 - s["rho"])
    s["dBIC_eff"] = s["delta_chi2"] / (s["s"] ** 2 * s["f_corr"]) - s["k"] * np.log(s["npix"] / s["f_corr"])
    s["control"] = s["target"].isin(controls)
    s["control_class"] = s["target"].map(controls).fillna("")
    # per-molecule thresholds from the non-pinned control components
    c = s[s["control"] & ~s["pinned"] & ~s["target"].isin(exclude or [])]
    thr = c.groupby("molecule")["dBIC_eff"].max().mul(margin).clip(lower=floor)
    src = c.loc[c.groupby("molecule")["dBIC_eff"].idxmax()].set_index("molecule") if len(c) else pd.DataFrame()
    mols = sorted(set(s["molecule"].dropna()))
    th = pd.DataFrame({"molecule": mols})
    th["threshold_dBIC_eff"] = th["molecule"].map(thr).fillna(floor)
    th["from_controls"] = th["molecule"].isin(thr.index) & (th["threshold_dBIC_eff"] > floor)
    th["set_by"] = [(f"{src.loc[m, 'target']} {src.loc[m, 'component']} T={src.loc[m, 'median_T']:.0f} "
                     f"logN={src.loc[m, 'median_logN']:.2f}") if (len(src) and m in src.index and fc) else ""
                    for m, fc in zip(th["molecule"], th["from_controls"])]
    th["n_control_components"] = th["molecule"].map(s[s["control"]].groupby("molecule").size()).fillna(0).astype(int)
    th["n_control_pinned"] = th["molecule"].map(s[s["control"] & s["pinned"]].groupby("molecule").size()).fillna(0).astype(int)
    s["threshold"] = s["molecule"].map(th.set_index("molecule")["threshold_dBIC_eff"]).fillna(floor)
    s["detected_raw"] = s["delta_BIC"] > 10
    s["detected_calibrated"] = (s["dBIC_eff"] > s["threshold"]) & ~s["pinned"]
    s["detected_calibrated_floor"] = (s["dBIC_eff"] > floor) & ~s["pinned"]
    return s, th


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("compiled_dir")
    ap.add_argument("--controls", default=None, help="csv (target,class) or one target per line")
    ap.add_argument("--targets", default=None, help="survey_targets.csv: controls = flag_non_stellar_otype | "
                    "flag_debris_disk_program | B68N* / L694-2N* / B335N* (overrides --controls)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pin-logN", type=float, default=0.5, help="pinned if log N > upper bound - this (default 0.5: "
                    "in the 0.21 survey 0.2 left four control components at log N 20.6-20.7)")
    ap.add_argument("--exclude", default="", help="comma-separated controls NOT used for the thresholds "
                    "(they are still reported), e.g. a debris disk with a silica feature that sets the CO2 threshold")
    ap.add_argument("--margin", type=float, default=1.0, help="threshold = margin x max control dBIC_eff")
    ap.add_argument("--floor", type=float, default=10.0)
    a = ap.parse_args(argv)
    controls = load_controls(a.controls, a.targets)
    s, th = calibrate(a.compiled_dir, controls, a.pin_logN, a.margin, a.floor,
                      exclude=[x.strip() for x in a.exclude.split(",") if x.strip()])
    out = a.out or os.path.join(a.compiled_dir, "calibrated")
    os.makedirs(out, exist_ok=True)
    s.to_csv(os.path.join(out, "components_calibrated.csv"), index=False)
    th.to_csv(os.path.join(out, "thresholds.csv"), index=False)
    disks = s[~s["control"]]
    n_disks = disks["target"].nunique()
    rates = disks.groupby("molecule").agg(fitted=("target", "size"), raw=("detected_raw", "sum"),
                                          not_pinned_floor=("detected_calibrated_floor", "sum"),
                                          calibrated=("detected_calibrated", "sum"))
    rates["targets_calibrated"] = disks[disks["detected_calibrated"]].groupby("molecule")["target"].nunique()
    rates["targets_calibrated"] = rates["targets_calibrated"].fillna(0).astype(int)
    rates["percent_of_disks"] = (100 * rates["targets_calibrated"] / max(n_disks, 1)).round(1)
    rates.to_csv(os.path.join(out, "detection_rates.csv"))
    ctl = s[s["control"]]
    L = [f"Detection calibration of {os.path.abspath(a.compiled_dir)}",
         f"controls found: {ctl['target'].nunique()} of {len(controls)} listed; disks: {n_disks}",
         f"dBIC_eff = dchi2 / (s^2 f) - k ln(n/f), f = (1+rho)/(1-rho); pinned: log N > bound - {a.pin_logN}",
         "", "control components (raw dBIC > 10 | pinned | non-pinned dBIC_eff > floor):"]
    for m, g in ctl.groupby("molecule"):
        L.append(f"  {m:<8s} {len(g):3d} fitted  {int(g['detected_raw'].sum()):3d} raw  {int(g['pinned'].sum()):3d} pinned  "
                 f"{int(g['detected_calibrated_floor'].sum()):3d} survive the floor")
    surv = ctl[ctl["detected_calibrated_floor"]]
    if len(surv):
        L += ["", "control components that survive pinning + the floor (they set the thresholds; --exclude to drop one):"]
        L += [f"  {r.target:<32s} {r.component:<10s} T={r.median_T:6.0f} K  log N={r.median_logN:5.2f}  "
              f"dBIC_eff={r.dBIC_eff:7.1f}  ({r.control_class})" for r in surv.itertuples()]
    L += ["", "thresholds (dBIC_eff):"] + [f"  {r.molecule:<8s} {r.threshold_dBIC_eff:9.1f}  "
                                          f"{('set by ' + r.set_by) if r.from_controls else 'floor'}" for r in th.itertuples()]
    L += ["", "disks: fitted / raw dBIC>10 / not pinned & dBIC_eff>floor / calibrated (targets, % of disks):"]
    for m, r in rates.iterrows():
        L.append(f"  {m:<8s} {int(r.fitted):4d} {int(r.raw):4d} {int(r.not_pinned_floor):4d} {int(r.calibrated):4d}  "
                 f"({int(r.targets_calibrated)} targets, {r.percent_of_disks:.0f} %)")
    rep = "\n".join(L)
    with open(os.path.join(out, "REPORT_calibration.txt"), "w") as fh:
        fh.write(rep + "\n")
    print(rep)
    print(f"\nwritten to {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
