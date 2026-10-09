#!/usr/bin/env python
"""rerun_targets.py -- a config + targets table for re-fitting a few named targets into their own folder.

    python runs/rerun_targets.py --config runs/survey_0.22.yaml --targets runs/survey_targets.csv \
           --name failed_0.21_rerun 808-50020 HD-142666 ...
    python runs/run_survey_server.py run --config runs/results/<name>/config.yaml \
           --targets runs/results/<name>/targets.csv --skip-preflight --workers 5

Writes runs/results/<name>/config.yaml (the config with output -> runs/results/<name>/{target}) and
runs/results/<name>/targets.csv (the named rows).  A separate output root keeps the runner's status files and
population.csv of the main survey untouched.
"""
import argparse
import os
import sys

import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+")
    ap.add_argument("--config", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--set", action="append", default=[], help="key.sub=value overrides (YAML values)")
    a = ap.parse_args(argv)
    out = os.path.join(HERE, "results", a.name)
    os.makedirs(out, exist_ok=True)
    t = pd.read_csv(a.targets)
    sub = t[t["name"].isin(a.names)]
    missing = sorted(set(a.names) - set(sub["name"]))
    sub.to_csv(os.path.join(out, "targets.csv"), index=False)
    with open(a.config) as fh:
        d = yaml.safe_load(fh)
    d["output"] = os.path.join(out, "{target}")
    for s in a.set:
        k, _, v = s.partition("=")
        cur = d
        for p in k.split(".")[:-1]:
            cur = cur.setdefault(p, {})
        cur[k.split(".")[-1]] = yaml.safe_load(v)
    with open(os.path.join(out, "config.yaml"), "w") as fh:
        yaml.safe_dump(d, fh, sort_keys=False)
    print(f"{len(sub)} targets -> {out}/targets.csv, config {out}/config.yaml"
          + (f"; not in the table: {', '.join(missing)}" if missing else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
