#!/usr/bin/env python3
"""
collect_results.py -- compile a jalebi survey run (0.21+) for analysis (one row per target, per parameter, ...).

Reads the files with the schemas jalebi 0.21 / 0.22 write (checked against tags v0.21.0 and v0.22.0,
pipeline._save_results / MCMCResult.diagnostics / summary / FitProblem.component_significance /
detect.detect_molecules):

  per target folder  results/<run>/<target>/
    summary.csv        long: parameter,label,median,minus,plus,lo_bound,hi_bound,at_edge
    diagnostics.json   acceptance, tau[], steps_over_tau, converged_length, rhat[], rhat_ok, n_eff[],
                       burn, acceptance_ok, runtime_s, moves, init, blocks, ..., emulator{}
    best_fit.json      params, free_params, log_s, chi2, chi2_red, npix, tau_max{}
    detections.csv     posterior-fit significance: component,delta_chi2,k,delta_BIC,tau_max,detected
    detection.csv      auto-detect stage: candidate,molecule,kind,T,logN,logR,fc,delta_chi2,delta_BIC,...
    tau_flags.json     fraction of posterior samples with tau_max < 1 per unit
    model.csv          wave,data,sigma,model   (+ band, continuum, model_<unit> in 0.22)
    config.yaml        the resolved config (component -> molecule)
    FAILED.txt         written by `jalebi batch` when a target raised
  run root
    population.csv     `jalebi batch` catalogue (wide); copied if present

Usage (on the server):
    python collect_results.py --run-dir ~/LTE_run/jalebi/runs/results/survey_0.21
    python collect_results.py --run-dir ... --with-files            # + small per-target files
    python collect_results.py --run-dir ... --with-files --with-figures   # + fit/corner PNGs
    python collect_results.py --run-dir ... --with-chains           # + chain.npz (large)

Writes <out-dir> (default: <run-dir>/../compiled_<run>) and <out-dir>.tar.gz:
    targets.csv            one row per target: status, chi2_red, log_s, convergence, edges, emulator, runtime
    params_long.csv        one row per (target, parameter) with component / molecule / quantity split,
                           posterior ΔBIC, tau_max, thin fraction, per-parameter tau / R-hat / N_eff
    params_wide.csv        one row per target, columns <component>.<quantity>[_m|_p]
    significance.csv       posterior-fit ΔBIC per (target, component)            (detections.csv)
    autodetect.csv         detection-stage table per (target, candidate)           (detection.csv)
    residuals_by_channel.csv  χ²_red, scaled χ²_red, median |resid|/σ per MRS channel per target
    failed.csv             targets with FAILED.txt / no MCMC, with the error / last log lines
    population_batch.csv   copy of the run's population.csv, if it exists
    REPORT.txt             human-readable overview
    files/<target>/...     (--with-files / --with-figures / --with-chains)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# MIRI-MRS channels (µm).  Overlaps are assigned to the lower channel.
CHANNELS = [("ch1", 4.90, 7.65), ("ch2", 7.65, 11.70), ("ch3", 11.70, 17.98), ("ch4", 17.98, 28.80)]

SMALL_FILES = ["summary.csv", "diagnostics.json", "best_fit.json", "detections.csv", "detection.csv",
               "tau_flags.json", "correlation.csv", "config.yaml", "log.txt", "model.csv", "FAILED.txt",
               "evidence.csv", "laplace_summary.csv"]
FIGURES = ["fit.png", "fit_windows.png", "corner.png", "traces.png", "posterior_predictive.png"]

MOLECULES = ["13CCH2", "13CO2", "13CO", "H13CN", "HC3N", "C2H2", "C2H4", "C2H6", "C4H2", "C6H6", "C3H4",
             "CO2", "HCN", "H2O", "CH4", "NH3", "OH", "CO", "H2"]          # longest first for prefix matching


# ----------------------------------------------------------------------------------------------------
# small readers
# ----------------------------------------------------------------------------------------------------
def _json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _csv(p: Path):
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def _arr(x):
    """JSON list (possibly with None / 'NaN' / 'Infinity') -> float array."""
    if x is None:
        return np.array([])
    if np.isscalar(x):
        x = [x]
    out = []
    for v in x:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(np.nan)
    return np.array(out, float)


def _component_molecules(cfg_path: Path) -> dict[str, str]:
    try:
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
        return {c["name"]: c.get("molecule", "") for c in cfg.get("components", []) if "name" in c}
    except Exception:
        return {}


def _guess_molecule(comp: str) -> str:
    for m in MOLECULES:
        if comp == m or comp.startswith(m + "_") or comp.startswith(m):
            return m
    return ""


def _split(param: str) -> tuple[str, str]:
    if "." in param:
        c, q = param.rsplit(".", 1)
        return c, q
    return "", param


# ----------------------------------------------------------------------------------------------------
# one target
# ----------------------------------------------------------------------------------------------------
def collect_target(tdir: str, residuals: bool = True) -> dict:
    d = Path(tdir)
    name = d.name
    out = {"target": name, "row": {"target": name}, "params": None, "sig": None, "auto": None,
           "resid": None, "failed": None}
    row = out["row"]
    files = {f.name for f in d.iterdir() if f.is_file()}
    row["has_summary"] = "summary.csv" in files
    row["has_best_fit"] = "best_fit.json" in files
    row["has_detection_stage"] = "detection.csv" in files

    # ---- failure / status ----
    if "FAILED.txt" in files:
        row["status"] = "failed"
    elif "summary.csv" in files:
        row["status"] = "ok"
    elif "best_fit.json" in files:
        row["status"] = "no_mcmc"
    else:
        row["status"] = "empty"
    if row["status"] != "ok":
        err = (d / "FAILED.txt").read_text().strip()[:2000] if "FAILED.txt" in files else ""
        tail = ""
        if "log.txt" in files:
            tail = " | ".join((d / "log.txt").read_text(errors="replace").strip().splitlines()[-5:])[:2000]
        out["failed"] = {"target": name, "status": row["status"], "error": err, "log_tail": tail,
                         "files": ",".join(sorted(files))}

    comp_mol = _component_molecules(d / "config.yaml") if "config.yaml" in files else {}

    # ---- best_fit.json ----
    bf = _json(d / "best_fit.json") if "best_fit.json" in files else None
    if bf:
        for k in ("chi2", "chi2_red", "npix", "log_s"):
            v = bf.get(k)
            row[f"bf_{k}"] = float(v) if isinstance(v, (int, float)) else np.nan
        row["noise_scale_s"] = 10 ** row["bf_log_s"] if np.isfinite(row.get("bf_log_s", np.nan)) else np.nan
        tm = bf.get("tau_max") or {}
        row["n_units_thick"] = int(sum(1 for v in tm.values() if isinstance(v, (int, float)) and v > 1))

    # ---- diagnostics.json ----
    dg = _json(d / "diagnostics.json") if "diagnostics.json" in files else None
    tau = rh = neff = np.array([])
    if dg:
        tau, rh, neff = _arr(dg.get("tau")), _arr(dg.get("rhat")), _arr(dg.get("n_eff"))
        row.update({
            "acceptance": float(dg.get("acceptance", np.nan) or np.nan),
            "acceptance_ok": dg.get("acceptance_ok"),
            "steps_over_tau": float(dg.get("steps_over_tau", np.nan) or np.nan),
            "tau_max": np.nanmax(tau) if np.isfinite(tau).any() else np.nan,
            "rhat_max": np.nanmax(rh) if np.isfinite(rh).any() else np.nan,
            "rhat_median": np.nanmedian(rh) if np.isfinite(rh).any() else np.nan,
            "n_rhat_gt_1.05": int(np.sum(rh > 1.05)), "n_rhat_gt_1.1": int(np.sum(rh > 1.1)),
            "n_eff_min": np.nanmin(neff) if np.isfinite(neff).any() else np.nan,
            "burn": dg.get("burn"),
            "converged_length": dg.get("converged_length"), "rhat_ok": dg.get("rhat_ok"),
            "converged": bool(dg.get("converged_length")) and bool(dg.get("rhat_ok")),
            "runtime_min": float(dg.get("runtime_s", np.nan) or np.nan) / 60.0,
            "moves": str(dg.get("moves", "")), "init": str(dg.get("init", "")),
            "n_blocks": len(dg["blocks"]) if isinstance(dg.get("blocks"), list) else np.nan,
            "linear": str(dg.get("linear", "")),
        })
        em = dg.get("emulator") or {}
        if em:
            row["emu_cache"] = em.get("cache", "")
            row["emu_units"] = len(em.get("units", []))
            row["emu_exact_units"] = ",".join(sorted((em.get("exact_units") or {}).keys()))
            fb = em.get("fallback") or {}
            row["emu_fallbacks"] = len(fb) if isinstance(fb, (dict, list)) else int(bool(fb))
            sc = em.get("spot_check") or {}
            worst = []
            for v in (sc.values() if isinstance(sc, dict) else []):
                if isinstance(v, dict):
                    for kk in ("max_err_sigma", "max_sigma", "err_sigma", "sigma"):
                        if isinstance(v.get(kk), (int, float)):
                            worst.append(v[kk]); break
            row["emu_spot_max_sigma"] = max(worst) if worst else np.nan

        # 0.22+: multi-start DE, corner check, joint continuum correction, resume, emulator decisions (0.22.1)
        op = dg.get("optimise") or {}
        if op:
            row["de_n_starts"] = op.get("n_starts")
            row["de_best_start"] = op.get("best_start")
            mm = op.get("multimodal") or []
            row["de_multimodal"] = len(mm) if isinstance(mm, list) else int(bool(mm))
        co = dg.get("corner") or {}
        if isinstance(co, dict) and co:
            fl = {k: v.get("flags") for k, v in co.items() if isinstance(v, dict) and v.get("flags")}
            row["corner_flagged"] = ",".join(sorted(fl))
            row["n_corner_flagged"] = len(fl)
        cf = dg.get("continuum_fit") or {}
        if cf:
            row["contfit_mode"] = cf.get("mode")
            row["contfit_max_corr_over_cont"] = cf.get("max_abs_correction_over_continuum")
            row["contfit_rms_corr_over_noise"] = cf.get("rms_correction_over_noise")
        rs = dg.get("resume") or {}
        if rs:
            row["resumed_stages"] = ",".join(rs.get("resumed", []))
        if em:
            kept = em.get("kept_despite_target") or {}
            row["emu_kept_despite_target"] = ",".join(sorted(kept))
        iw = dg.get("init_walkers") or []
        if iw:
            row["init_nonfinite_walkers"] = int(sum(int(x.get("non_finite_starts", 0)) for x in iw if isinstance(x, dict)))

    # ---- tau_flags.json ----
    tf = _json(d / "tau_flags.json") if "tau_flags.json" in files else {}
    tf = tf or {}

    # ---- detections.csv (posterior-fit significance) ----
    sig = _csv(d / "detections.csv") if "detections.csv" in files else None
    if sig is not None and len(sig):
        sig.insert(0, "target", name)
        sig["molecule"] = [comp_mol.get(c) or _guess_molecule(str(c)) for c in sig["component"]]
        out["sig"] = sig
        row["n_components"] = len(sig)
        row["n_components_dBIC_gt10"] = int((sig["delta_BIC"] > 10).sum())
        row["components_not_supported"] = ",".join(sig.loc[sig["delta_BIC"] <= 10, "component"].astype(str))

    # ---- detection.csv (auto-detect stage) ----
    auto = _csv(d / "detection.csv") if "detection.csv" in files else None
    if auto is not None and len(auto):
        auto.insert(0, "target", name)
        out["auto"] = auto
        if "detected" in auto:
            det = auto[auto["detected"].astype(str).str.lower() == "true"]
            row["autodetected"] = ",".join(det["candidate"].astype(str))

    # ---- summary.csv ----
    s = _csv(d / "summary.csv") if "summary.csv" in files else None
    if s is not None and len(s):
        s.insert(0, "target", name)
        comp, qty = zip(*[_split(str(p)) for p in s["parameter"]])
        s.insert(2, "component", comp); s.insert(3, "quantity", qty)
        s.insert(4, "molecule", [comp_mol.get(c) or _guess_molecule(c) for c in comp])
        free = s["lo_bound"].notna().to_numpy()
        s["free"] = free
        s["tau"] = np.nan; s["rhat"] = np.nan; s["n_eff"] = np.nan
        nfree = int(free.sum())
        if dg and len(tau) == nfree:                       # diagnostics arrays are in free-parameter order
            idx = np.flatnonzero(free)
            s.loc[idx, "tau"] = tau; s.loc[idx, "rhat"] = rh
            if len(neff) == nfree:
                s.loc[idx, "n_eff"] = neff
        if sig is not None and len(sig):
            m = sig.set_index("component")
            s["unit_delta_BIC"] = [m["delta_BIC"].get(c, np.nan) for c in s["component"]]
            s["unit_tau_max"] = [m["tau_max"].get(c, np.nan) if "tau_max" in m else np.nan for c in s["component"]]
        s["thin_frac"] = [tf.get(c, np.nan) for c in s["component"]]
        s["sigma_sym"] = 0.5 * (s["minus"] + s["plus"])
        out["params"] = s
        edge = s[s["at_edge"].astype(str).str.lower() == "true"]
        row["n_at_edge"] = len(edge)
        row["params_at_edge"] = ",".join(edge["parameter"].astype(str))
        # water temperatures, the most-asked numbers
        for c in sorted(set(comp)):
            if c.startswith("H2O"):
                r = s[(s["component"] == c) & (s["quantity"] == "T")]
                if len(r):
                    row[f"{c}.T"] = float(r["median"].iloc[0])

    # ---- model.csv residuals per channel ----
    if residuals and "model.csv" in files:
        mdl = _csv(d / "model.csv")
        if mdl is not None and {"wave", "data", "sigma", "model"} <= set(mdl.columns):
            sscale = row.get("noise_scale_s", np.nan)
            z = (mdl["data"] - mdl["model"]) / mdl["sigma"]
            rr = []
            for ch, lo, hi in CHANNELS + [("all", 0, 99)]:
                sel = (mdl["wave"] >= lo) & (mdl["wave"] < hi) & np.isfinite(z)
                n = int(sel.sum())
                if n == 0:
                    continue
                zz = z[sel].to_numpy()
                c2 = float(np.mean(zz ** 2))
                rr.append({"target": name, "channel": ch, "n_pix": n, "chi2_red": c2,
                           "chi2_red_scaled": c2 / sscale ** 2 if np.isfinite(sscale) else np.nan,
                           "median_abs_z": float(np.median(np.abs(zz))),
                           "mean_z": float(np.mean(zz)),           # >0: model under-predicts on average
                           "lag1_acf": float(np.corrcoef(zz[:-1], zz[1:])[0, 1]) if n > 3 else np.nan,
                           "wave_min": float(mdl.loc[sel, "wave"].min()),
                           "wave_max": float(mdl.loc[sel, "wave"].max())})
            if rr:
                out["resid"] = pd.DataFrame(rr)
                a = [r for r in rr if r["channel"] == "all"][0]
                row["chi2_red_model"] = a["chi2_red"]; row["resid_lag1_acf"] = a["lag1_acf"]
    return out


# ----------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="folder holding one sub-folder per target (e.g. runs/results/survey_0.21)")
    ap.add_argument("--results-dir", type=Path, default=Path.home() / "LTE_run/jalebi/runs/results",
                    help="used with --run-name when --run-dir is not given")
    ap.add_argument("--run-name", default="survey_0.21")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-residuals", action="store_true", help="skip model.csv")
    ap.add_argument("--with-files", action="store_true", help=f"copy {', '.join(SMALL_FILES)} per target")
    ap.add_argument("--with-figures", action="store_true", help=f"copy {', '.join(FIGURES)} per target")
    ap.add_argument("--with-chains", action="store_true", help="copy chain.npz per target (large)")
    a = ap.parse_args()

    run_dir = (a.run_dir or a.results_dir / a.run_name).expanduser().resolve()
    if not run_dir.is_dir():
        sys.exit(f"run dir not found: {run_dir}")
    def _is_target(p):                       # a fit folder has its config.yaml; helper folders are skipped
        return p.is_dir() and not p.name.startswith(("_", ".")) and any((p / f).is_file() for f in
                                                                         ("config.yaml", "run.log", "DONE", "FAILED.txt", "NO_DETECTION"))
    tdirs = sorted(p for p in run_dir.iterdir() if _is_target(p))
    skipped = sorted(p.name for p in run_dir.iterdir() if p.is_dir() and not _is_target(p))
    out_dir = (a.out_dir or run_dir.parent / f"compiled_{run_dir.name}").expanduser().resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    print(f"run   : {run_dir}\ntargets: {len(tdirs)}  (skipped helper folders: {', '.join(skipped) or '-'})\nout   : {out_dir}")

    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(collect_target, [str(t) for t in tdirs], [not a.no_residuals] * len(tdirs), chunksize=4))

    def cat(key):
        fr = [r[key] for r in res if r[key] is not None]
        return pd.concat(fr, ignore_index=True) if fr else pd.DataFrame()

    targets = pd.DataFrame([r["row"] for r in res])
    params, sig, auto, resid = cat("params"), cat("sig"), cat("auto"), cat("resid")
    failed = pd.DataFrame([r["failed"] for r in res if r["failed"]])

    # wide parameter table
    wide = pd.DataFrame()
    if len(params):
        p = params.copy()
        long = pd.concat([p.assign(col=p["parameter"], val=p["median"]),
                          p.assign(col=p["parameter"] + "_m", val=p["minus"]),
                          p.assign(col=p["parameter"] + "_p", val=p["plus"])])
        wide = long.pivot_table(index="target", columns="col", values="val", aggfunc="first")
        wide = wide[sorted(wide.columns)].reset_index()

    def w(df, fn):
        if len(df):
            df.to_csv(out_dir / fn, index=False)
            print(f"  {fn:<28s} {len(df):>7,} rows x {df.shape[1]} cols")
        else:
            print(f"  {fn:<28s} (empty)")

    print("writing:")
    w(targets, "targets.csv"); w(params, "params_long.csv"); w(wide, "params_wide.csv")
    w(sig, "significance.csv"); w(auto, "autodetect.csv"); w(resid, "residuals_by_channel.csv")
    w(failed, "failed.csv")
    pop = run_dir / "population.csv"
    if pop.exists():
        shutil.copy2(pop, out_dir / "population_batch.csv"); print("  population_batch.csv         (copied)")
    for extra in run_dir.glob("*.yaml"):
        shutil.copy2(extra, out_dir / extra.name)

    # ---- per-target files ----
    if a.with_files or a.with_figures or a.with_chains:
        want = (SMALL_FILES if a.with_files else []) + (FIGURES if a.with_figures else []) + \
               (["chain.npz"] if a.with_chains else [])
        n = 0
        for t in tdirs:
            dst = out_dir / "files" / t.name
            for f in want:
                if (t / f).exists():
                    dst.mkdir(parents=True, exist_ok=True); shutil.copy2(t / f, dst / f); n += 1
        print(f"  files/                       {n:,} files")

    # ---- report ----
    L = ["=" * 78, f"jalebi survey compilation   {datetime.now():%Y-%m-%d %H:%M}",
         f"run: {run_dir}", "=" * 78]
    st = targets["status"].value_counts().to_dict()
    L.append(f"targets {len(targets)}: " + ", ".join(f"{k} {v}" for k, v in st.items()))
    ok = targets[targets["status"] == "ok"]
    if len(ok):
        def q(col, fmt="{:.2f}"):
            v = pd.to_numeric(ok.get(col), errors="coerce").dropna() if col in ok else pd.Series(dtype=float)
            if not len(v):
                return "n/a"
            return f"median {fmt.format(v.median())}  [p10 {fmt.format(v.quantile(.1))}, p90 {fmt.format(v.quantile(.9))}]  max {fmt.format(v.max())}"
        L += ["", "-- convergence (emcee) " + "-" * 55,
              f"converged (steps > 50 tau AND R-hat < 1.05): {int(ok['converged'].sum()) if 'converged' in ok else 'n/a'} / {len(ok)}",
              f"  steps > 50 tau only : {int(ok['converged_length'].fillna(False).astype(bool).sum()) if 'converged_length' in ok else 'n/a'}",
              f"  R-hat < 1.05 only   : {int(ok['rhat_ok'].fillna(False).astype(bool).sum()) if 'rhat_ok' in ok else 'n/a'}",
              f"steps/tau   : {q('steps_over_tau', '{:.1f}')}",
              f"R-hat max   : {q('rhat_max', '{:.3f}')}",
              f"N_eff min   : {q('n_eff_min', '{:.0f}')}",
              f"acceptance  : {q('acceptance', '{:.3f}')}",
              f"runtime/min : {q('runtime_min', '{:.1f}')}   total {pd.to_numeric(ok.get('runtime_min'), errors='coerce').sum() / 60:.1f} h (sampling only)",
              "", "-- goodness of fit " + "-" * 59,
              f"chi2_red (best fit) : {q('bf_chi2_red')}",
              f"noise scale s       : {q('noise_scale_s')}",
              f"residual lag-1 ACF  : {q('resid_lag1_acf')}",
              "", "-- prior edges / emulator " + "-" * 52,
              f"targets with >=1 parameter at a prior edge: {int((ok.get('n_at_edge', pd.Series(dtype=float)) > 0).sum())}"]
        if len(params):
            e = params[params["at_edge"].astype(str).str.lower() == "true"]
            if len(e):
                top = (e["molecule"].replace("", "?") + "." + e["quantity"]).value_counts().head(12)
                L += ["  most frequent edge parameters (molecule.quantity):"] + [f"    {k:<22s}{v:5d}" for k, v in top.items()]
        if "emu_fallbacks" in ok:
            L.append(f"targets with emulator fallbacks to exact: {int((ok['emu_fallbacks'] > 0).sum())}")
    if len(sig):
        L += ["", "-- posterior-fit significance (detections.csv, ΔBIC > 10) " + "-" * 19,
              f"{'molecule':<10s}{'fitted':>8s}{'ΔBIC>10':>9s}{'%':>6s}"]
        g = sig.assign(sup=sig["delta_BIC"] > 10).groupby("molecule")["sup"].agg(["size", "sum"])
        for m, r in g.sort_values("size", ascending=False).iterrows():
            L.append(f"{m or '?':<10s}{int(r['size']):>8d}{int(r['sum']):>9d}{100 * r['sum'] / r['size']:>6.0f}")
    if len(auto) and "detected" in auto:
        dd = auto[auto["detected"].astype(str).str.lower() == "true"]
        L += ["", "-- auto-detect stage (detection.csv) " + "-" * 41,
              f"{len(dd)} detections in {dd['target'].nunique()} targets; by candidate:"]
        L += [f"    {k:<14s}{v:5d}" for k, v in dd["candidate"].value_counts().items()]
    if len(failed):
        L += ["", "-- not completed " + "-" * 61] + \
             [f"  {r.target:<22s}{r.status:<9s}{(r.error or r.log_tail)[:90]}" for r in failed.itertuples()]
    L.append("=" * 78)
    rep = "\n".join(L)
    (out_dir / "REPORT.txt").write_text(rep)
    print("\n" + rep)

    tar = out_dir.parent / (out_dir.name + ".tar.gz")
    with tarfile.open(tar, "w:gz") as tf:
        tf.add(out_dir, arcname=out_dir.name)
    print(f"\narchive: {tar}  ({tar.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
