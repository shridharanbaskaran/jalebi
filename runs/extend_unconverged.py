#!/usr/bin/env python
"""extend_unconverged.py (0.22.1) -- continue the MCMC of the targets of a survey run that did not converge.

A target counts as unconverged when its diagnostics.json says steps < 50 tau (`converged_length` false) or max
split R-hat >= 1.05 (`rhat_ok` false).  With fit.resume: auto (0.22) and the same config, re-running such a target
with a larger fit.mcmc.nsteps re-uses every finished stage (detection, grid, DE passes, continuum refinement) and
CONTINUES chain.h5 from its last step: nsteps is deliberately not part of the resume key.  This script

    1. scans RUN_DIR/<target>/diagnostics.json and lists the unconverged targets with the steps they would need
       (50 x max tau, x --safety), capped at --max-steps;
    2. writes RUN_DIR/_extend/targets_unconverged.csv (the rows of --targets for those targets);
    3. (with --launch) runs run_survey_server.py once per step bucket with --force and --set fit.mcmc.nsteps=N.

    python runs/extend_unconverged.py RUN_DIR --config runs/survey_0.22.yaml --targets runs/survey_targets.csv
    python runs/extend_unconverged.py RUN_DIR ... --launch --workers 30

Only runs written by jalebi >= 0.22 can be continued: 0.21 wrote chain.h5 but no stage files, so a 0.22 run of
a 0.21 folder starts over (and the 0.21 survey has to be re-fitted anyway, see survey_0.21_STATUS).  The chain
is only continued when the config is unchanged apart from run-only keys (resume.RUN_ONLY_KEYS); anything else
(nwalkers, moves, seed, linear, ...) restarts the sampler, which the log reports as "[stage] mcmc: key changed".
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def scan(run_dir: str, safety: float = 1.2, max_steps: int = 60000, rhat_target: float = 1.05) -> pd.DataFrame:
    rows = []
    for t in sorted(os.listdir(run_dir)):
        d = os.path.join(run_dir, t)
        p = os.path.join(d, "diagnostics.json")
        if t.startswith(("_", ".")) or not os.path.isfile(p):
            continue
        try:
            dg = json.load(open(p))
        except Exception:
            continue
        tau = np.array([float(x) if x is not None else np.nan for x in np.atleast_1d(dg.get("tau", []))], float)
        rh = np.array([float(x) if x is not None else np.nan for x in np.atleast_1d(dg.get("rhat", []))], float)
        tau_max = float(np.nanmax(tau)) if np.isfinite(tau).any() else np.nan
        sot = float(dg.get("steps_over_tau") or np.nan)
        nsteps = sot * tau_max if np.isfinite(sot) and np.isfinite(tau_max) else np.nan
        conv_len, rhat_ok = bool(dg.get("converged_length")), bool(dg.get("rhat_ok"))
        need = 50 * tau_max * safety if np.isfinite(tau_max) else np.nan
        # R-hat drops roughly as 1/sqrt(steps) once the walkers mix: at least double the run when only R-hat fails
        if not rhat_ok and np.isfinite(nsteps):
            need = np.nanmax([need, 2 * nsteps])
        rows.append({"target": t, "nsteps_done": nsteps, "tau_max": tau_max, "steps_over_tau": sot,
                     "rhat_max": float(np.nanmax(rh)) if np.isfinite(rh).any() else np.nan,
                     "converged_length": conv_len, "rhat_ok": rhat_ok, "converged": conv_len and rhat_ok,
                     "nsteps_needed": int(min(max_steps, math.ceil(need / 1000.0) * 1000)) if np.isfinite(need) else max_steps,
                     "hit_cap": bool(np.isfinite(need) and need > max_steps)})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--config", required=True, help="the config the run was made with (same file!)")
    ap.add_argument("--targets", required=True, help="the targets table the run was made with")
    ap.add_argument("--safety", type=float, default=1.2)
    ap.add_argument("--max-steps", type=int, default=60000)
    ap.add_argument("--buckets", default="20000,30000,45000,60000", help="nsteps values to group targets into")
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--timeout-h", type=float, default=48.0)
    a = ap.parse_args(argv)
    s = scan(a.run_dir, a.safety, a.max_steps)
    if s.empty:
        sys.exit(f"no diagnostics.json under {a.run_dir}")
    un = s[~s.converged].copy()
    buckets = sorted(int(x) for x in a.buckets.split(","))
    un["bucket"] = [next((b for b in buckets if b >= n), buckets[-1]) for n in un.nsteps_needed]
    out = os.path.join(a.run_dir, "_extend")
    os.makedirs(out, exist_ok=True)
    s.to_csv(os.path.join(out, "convergence_scan.csv"), index=False)
    tab = pd.read_csv(a.targets)
    print(f"{len(s)} targets scanned, {len(un)} unconverged "
          f"({int((~un.converged_length).sum())} too short, {int((un.converged_length & ~un.rhat_ok).sum())} R-hat only); "
          f"{int(un.hit_cap.sum())} would need more than {a.max_steps} steps")
    for b, g in un.groupby("bucket"):
        sub = tab[tab["name"].isin(g.target)]
        missing = sorted(set(g.target) - set(sub["name"]))
        f = os.path.join(out, f"targets_unconverged_{b}.csv")
        sub.to_csv(f, index=False)
        print(f"  nsteps {b:6d}: {len(g):4d} targets -> {f}" + (f"  (not in --targets: {', '.join(missing)})" if missing else ""))
        cmd = [sys.executable, os.path.join(HERE, "run_survey_server.py"), "run", "--config", a.config, "--targets", f,
               "--force", "--skip-preflight", "--timeout-h", str(a.timeout_h), "--set", f"fit.mcmc.nsteps={b}",
               "--set", "fit.resume=auto"] + (["--workers", str(a.workers)] if a.workers else [])
        print("    " + " ".join(cmd))
        if a.launch:
            subprocess.run(cmd, check=False)
    if not a.launch:
        print("dry run: add --launch to start (or copy the commands above)")


if __name__ == "__main__":
    main()
