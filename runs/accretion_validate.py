#!/usr/bin/env python
"""accretion_validate.py (0.24.1) -- jalebi H I accretion vs the published values of Shridharan et al. 2026
(A&A 708, A22; the 79-star MIRI sample, CDS J/A+A/708/A22, Tables A.1 and A.2).

    python runs/accretion_validate.py PUBLISHED.csv  FITDIR [FITDIR ...] | --survey ROOT  [--out DIR] [--workers N]

PUBLISHED.csv: one row per star with (case-insensitive, any order; Tables A.1 and A.2 merged on the name)
    Source / Target                         star name
    Dist / distance_pc                      distance used in the paper (pc)
    A_V                                     extinction used in the paper (mag)
    logLacc_lit / "log L_acc"               literature log L_acc (Table A.1), optional
    Lline(6-5) ... Lline(10-7)              published log L_line (L_sun); "-5.57(0.03)" or "<-6.67" accepted
    Lacc(6-5)  ... Lacc(10-7)               published log L_acc per line
jalebi is run on each matched fit folder (prep.csv + config.yaml) with the PAPER's distance and A_V, so the line
luminosities compare like for like.  Folder names are matched to star names after removing spaces, '_', '-', '*'
and a trailing programme / epoch tag (DF_Tau, DF-Tau_P24, ...).

Outputs (DIR, default ./accretion_validation): comparison.csv (one row per star x line: published and jalebi
log L_line, log L_acc, difference, flags), summary.md (median offset and scatter per line), validation.png
(jalebi vs published log L_line and log L_acc per line, and combined jalebi log L_acc vs literature).
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

LINES = ["6-5", "7-6", "8-6", "10-7"]


def norm(name: str) -> str:
    n = str(name).lower().replace("*", "").strip()
    n = re.sub(r"_jw\d{5}.*$", "", n)                                       # survey folders: NAME_jw01282-o003...
    n = re.sub(r"^(v|name|nam)[-_ ]+", "", n)                                # MAST prefixes V-DF-TAU, NAME-HK-TAU-B
    n = re.sub(r"[_\-\s]+(p\d+|e\d+|epoch\d+|\d{4}|[a-z]{1,3}\d{2})$", "", n)  # trailing programme / epoch tags
    return re.sub(r"[^a-z0-9+]", "", n)


def parse_val(v):
    """'-5.57(0.03)' -> (-5.57, 0.03, False); '<-6.67' -> (-6.67, nan, True); number -> (v, nan, False)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan, np.nan, False
    s = str(v).strip().replace("−", "-")
    ul = s.startswith("<")
    s = s.lstrip("<").strip()
    m = re.match(r"^([+-]?\d*\.?\d+)(?:\s*\(\s*([\d.]+)\s*\))?", s)
    if not m:
        return np.nan, np.nan, False
    return float(m.group(1)), (float(m.group(2)) if m.group(2) else np.nan), ul


def col(df, *cands):
    low = {c.lower().replace(" ", ""): c for c in df.columns}
    for c in cands:
        k = c.lower().replace(" ", "")
        if k in low:
            return low[k]
    return None


def load_published(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame()
    out["star"] = df[col(df, "Source", "Target", "name", "star")]
    dcol = col(df, "Dist", "Dist.(pc)", "distance_pc", "distance")
    out["d_pub"] = [parse_val(v)[0] for v in df[dcol]] if dcol else np.nan
    acol = col(df, "A_V", "AV", "A_V(mag)", "av")
    out["av_pub"] = [parse_val(v)[0] for v in df[acol]] if acol else 0.0
    lcol = col(df, "logLacc_lit", "logL_acc", "logLacc(Lsun)", "log L_acc (Lsun)", "log_lacc_lit")
    out["logLacc_lit"] = [parse_val(v)[0] for v in df[lcol]] if lcol else np.nan
    for ln in LINES:
        for kind in ("Lline", "Lacc"):
            c = col(df, f"{kind}({ln})", f"{kind}_{ln}", f"log{kind}({ln})", f"log{kind}_{ln}")
            vals = [parse_val(v) for v in df[c]] if c else [(np.nan, np.nan, False)] * len(df)
            out[f"pub_{kind}_{ln}"] = [v[0] for v in vals]
            out[f"pub_{kind}_{ln}_err"] = [v[1] for v in vals]
            out[f"pub_{kind}_{ln}_ul"] = [v[2] for v in vals]
    out["key"] = out.star.map(norm)
    return out


def find_dirs(args) -> dict:
    dirs = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(args.survey, "*", "prep.csv"))) if args.survey \
        else list(args.fitdirs)
    return {d: norm(os.path.basename(os.path.normpath(d))) for d in dirs}


def run_one(d, d_pc, av):
    from jalebi import accretion as AC
    from jalebi.config import ProjectConfig
    from jalebi.data import load_csv
    cfg = ProjectConfig.load(os.path.join(d, "config.yaml"))
    spec = load_csv(os.path.join(d, "prep.csv"))
    s = AC.settings_from_config(cfg)
    s.save_plot = False
    tab, res = AC.measure(spec, distance_pc=d_pc if np.isfinite(d_pc) else cfg.target.distance_pc,
                          av=av if np.isfinite(av) else 0.0, settings=s, R_model=cfg.R_model, R_scale=cfg.R_scale)
    return tab, res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("published")
    ap.add_argument("fitdirs", nargs="*")
    ap.add_argument("--survey", default=None)
    ap.add_argument("--out", default="accretion_validation")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--map", default=None, help="csv with columns star, folder for names that do not match")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    pub = load_published(a.published)
    dirs = find_dirs(a)
    jobs, used = [], set()
    forced = {}
    if a.map:
        mp = pd.read_csv(a.map)
        forced = {str(f): norm(st) for st, f in zip(mp.star, mp.folder)}
    for d, k in dirs.items():
        k2 = forced.get(os.path.basename(os.path.normpath(d)), k)
        m = pub[pub.key == k2]
        if not len(m):
            m = pub[[(k2.startswith(pk) or pk.startswith(k2)) and min(len(k2), len(pk)) >= 5 for pk in pub.key]]
        if len(m) and m.iloc[0].star not in used:          # first folder per star (repeat epochs are skipped)
            used.add(m.iloc[0].star)
            jobs.append((d, m.iloc[0]))
    miss = sorted(set(pub.star) - used)
    if miss:
        print(f"no fit folder for {len(miss)} stars: " + ", ".join(map(str, miss)))
    print(f"{len(pub)} published stars, {len(dirs)} fit folders, {len(jobs)} matched")

    def one(d, r):
        try:
            tab, res = run_one(d, r.d_pub, r.av_pub)
            return d, r, tab, res, None
        except Exception as e:                       # pragma: no cover
            return d, r, None, None, repr(e)
    if a.workers > 1:
        from joblib import Parallel, delayed
        out = Parallel(n_jobs=a.workers)(delayed(one)(d, r) for d, r in jobs)
    else:
        out = [one(d, r) for d, r in jobs]
    rows = []
    for d, r, tab, res, err in out:
        if err:
            print(f"  {os.path.basename(d)}: failed {err}")
            continue
        t = tab.set_index("line")
        for ln in LINES:
            key = f"HI_{ln}"
            jr = t.loc[key] if key in t.index else {}
            rows.append({"star": r.star, "folder": os.path.basename(os.path.normpath(d)), "line": ln,
                         "d_pc": r.d_pub, "av": r.av_pub,
                         "pub_logL_line": r[f"pub_Lline_{ln}"], "pub_logL_line_ul": r[f"pub_Lline_{ln}_ul"],
                         "pub_logLacc": r[f"pub_Lacc_{ln}"], "pub_logLacc_ul": r[f"pub_Lacc_{ln}_ul"],
                         "jal_logL_line": jr.get("logL_line", np.nan), "jal_detected": bool(jr.get("detected", False)),
                         "jal_logLacc": jr.get("logLacc", np.nan), "jal_logLacc_err": jr.get("logLacc_err", np.nan),
                         "jal_water_frac": jr.get("water_frac", np.nan), "jal_n_extra_peaks": jr.get("n_extra_peaks", np.nan),
                         "jal_logLacc_combined": res.get("logLacc", np.nan), "logLacc_lit": r.logLacc_lit})
        print(f"  {r.star}: jalebi combined log Lacc {res.get('logLacc', np.nan):.2f}")
    c = pd.DataFrame(rows)
    if not len(c):
        raise SystemExit("nothing matched")
    both = c.jal_detected & ~c.pub_logL_line_ul.astype(bool) & np.isfinite(c.pub_logL_line) & np.isfinite(c.jal_logL_line)
    c["dlogL_line"] = np.where(both, c.jal_logL_line - c.pub_logL_line, np.nan)
    c["dlogLacc"] = np.where(both, c.jal_logLacc - c.pub_logLacc, np.nan)
    c.to_csv(os.path.join(a.out, "comparison.csv"), index=False)
    L = [f"# jalebi H I accretion vs Shridharan+2026 ({c.star.nunique()} stars)\n",
         "Both detected; jalebi run with the paper's distance and A_V.\n",
         "| line | N | median Δ log L_line | scatter (1.4826 MAD) | median Δ log L_acc |", "|---|---|---|---|---|"]
    for ln, g in c.groupby("line", sort=False):
        x = g.dlogL_line.dropna()
        if len(x):
            L.append(f"| {ln} | {len(x)} | {x.median():+.2f} | {1.4826 * np.median(np.abs(x - x.median())):.2f} | "
                     f"{g.dlogLacc.dropna().median():+.2f} |")
    lit = c.drop_duplicates("star").dropna(subset=["logLacc_lit", "jal_logLacc_combined"])
    if len(lit):
        dl = lit.jal_logLacc_combined - lit.logLacc_lit
        L += ["", f"Combined jalebi log L_acc vs literature: N = {len(lit)}, median {dl.median():+.2f}, "
                  f"scatter {1.4826 * np.median(np.abs(dl - dl.median())):.2f} dex."]
    with open(os.path.join(a.out, "summary.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.4))
        cols = {"6-5": "#eb6834", "7-6": "#b3266f", "8-6": "#2a78d6", "10-7": "#1f9e89"}
        for kind, ax in (("L_line", axs[0]), ("Lacc", axs[1])):
            pk, jk = ("pub_logL_line", "jal_logL_line") if kind == "L_line" else ("pub_logLacc", "jal_logLacc")
            for ln, g in c.groupby("line", sort=False):
                ok = g.jal_detected & ~g.pub_logL_line_ul.astype(bool)
                ax.scatter(g[pk][ok], g[jk][ok], s=22, color=cols[ln], label=f"H I {ln}", zorder=3)
                ax.scatter(g[pk][~ok], g[jk][~ok], s=22, facecolor="none", edgecolor=cols[ln], zorder=2)
            v = np.concatenate([c[pk].dropna(), c[jk].dropna()])
            if len(v):
                lo, hi = np.nanmin(v) - 0.2, np.nanmax(v) + 0.2
                ax.plot([lo, hi], [lo, hi], color="#52514e", lw=0.8)
                ax.fill_between([lo, hi], [lo - 0.2, hi - 0.2], [lo + 0.2, hi + 0.2], color="#c3c2b7", alpha=0.35, lw=0)
                ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            lab = "log L$_{line}$ / L$_\\odot$" if kind == "L_line" else "log L$_{acc}$ / L$_\\odot$ (per line)"
            ax.set_xlabel("Shridharan+2026  " + lab); ax.set_ylabel("jalebi  " + lab)
            ax.legend(frameon=False, fontsize=8)
        ax = axs[2]
        if len(lit):
            ax.scatter(lit.logLacc_lit, lit.jal_logLacc_combined, s=22, color="#0b0b0b")
            v = np.concatenate([lit.logLacc_lit, lit.jal_logLacc_combined])
            lo, hi = v.min() - 0.3, v.max() + 0.3
            ax.plot([lo, hi], [lo, hi], color="#52514e", lw=0.8); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("literature log L$_{acc}$ (Table A.1)"); ax.set_ylabel("jalebi combined log L$_{acc}$")
        for ax in axs:
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        fig.suptitle("jalebi vs Shridharan+2026 (open = non-detection in either); band = ±0.2 dex", x=0.01, ha="left")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "validation.png"), dpi=140); plt.close(fig)
    except Exception as e:  # pragma: no cover
        print(f"plot failed: {e}")
    print(f"written to {a.out}/")


if __name__ == "__main__":
    main()
