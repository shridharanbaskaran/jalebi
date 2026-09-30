"""`jalebi doctor` — check that an installation is complete and working.

    jalebi doctor              # full report (packages, line lists, example data, parallelism, speed)
    jalebi doctor --quick      # packages only
    jalebi doctor --json       # machine-readable (used by install.py)
    python -m jalebi.doctor    # works even when dependencies are missing

Exit status: 0 = ready, 1 = something required is missing or broken.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import io
import json
import os
import platform
import sys
import time
import contextlib

from ._requirements import CORE, OPTIONAL, PYTHON_MIN, version_ok

OK, WARN, FAIL = "ok", "warn", "fail"


def _installed(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def check_packages() -> dict:
    """Status of every core and optional requirement."""
    out = {"core": [], "optional": {}}
    for dist, (vmin, mod, why) in CORE.items():
        v = _installed(dist)
        st = OK if version_ok(v, vmin) else (FAIL if v is None else WARN)
        out["core"].append({"name": dist, "min": vmin, "installed": v, "status": st, "purpose": why, "import": mod})
    for group, reqs in OPTIONAL.items():
        rows = []
        for dist, (vmin, mod, why) in reqs.items():
            v = _installed(dist)
            st = OK if version_ok(v, vmin) else (WARN if v is not None else "absent")
            rows.append({"name": dist, "min": vmin, "installed": v, "status": st, "purpose": why, "import": mod})
        out["optional"][group] = rows
    return out


def check_imports(names) -> list[dict]:
    """Actually import modules (catches broken binary wheels)."""
    rows = []
    for mod in names:
        t0 = time.time()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                importlib.import_module(mod)
            rows.append({"module": mod, "status": OK, "seconds": round(time.time() - t0, 2)})
        except Exception as e:  # pragma: no cover - depends on the environment
            rows.append({"module": mod, "status": FAIL, "error": f"{type(e).__name__}: {e}"})
    return rows


def check_linelists() -> dict:
    """Cached/bundled line lists, and that opacities and partition functions can be computed."""
    from .linedata import BUNDLED_DIR, available_linelists, data_dir, load_linelist
    tab = available_linelists()
    res = {"user_cache": data_dir(), "bundled_dir": BUNDLED_DIR, "n_lists": int(len(tab)),
           "lists": [f"{r.molecule}:{r.release}" for r in tab.itertuples()],
           "MB": round(float(tab["MB"].sum()), 1) if len(tab) else 0.0, "tests": []}
    for mol, rel in (("CO2", "hitran"), ("H2O", "hitran"), ("H2O", "hitemp")):
        t0 = time.time()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                ll = load_linelist(mol, release=rel, fetch=False)
                Z = float(ll.partition(500.0))
                k = ll.select(14.0, 16.0).kappa(500.0)
            res["tests"].append({"list": f"{mol}:{rel}", "status": OK, "lines": len(ll), "Z(500K)": round(Z, 1),
                                 "kappa_ok": bool(len(k) and float(k.max()) > 0), "seconds": round(time.time() - t0, 2),
                                 "partition": ll.partition.source})
        except Exception as e:
            res["tests"].append({"list": f"{mol}:{rel}", "status": FAIL, "error": f"{type(e).__name__}: {e}"})
    res["status"] = OK if res["n_lists"] and all(t["status"] == OK for t in res["tests"]) else FAIL
    return res


def check_examples() -> dict:
    from .examples import DATA_DIR, examples_source
    fz = DATA_DIR / "FZ_Tau"
    n_x1d = len(list(fz.glob("*x1d.fits"))) if fz.exists() else 0
    syn = DATA_DIR / "synthetic" / "synthetic_miri_ch3.csv"
    hv = DATA_DIR / "HV_Tau_C_cube"
    n_cubes = len(list(hv.glob("*s3d*.fits*"))) if hv.exists() else 0
    src = examples_source()
    return {"data_dir": str(DATA_DIR), "FZ_Tau_x1d_files": n_x1d, "synthetic": syn.exists(),
            "HV_Tau_C_cubes": n_cubes, "scripts": str(src) if src else None,
            "status": OK if (n_x1d == 12 and syn.exists() and n_cubes == 5) else WARN}


def check_cube() -> dict:
    """Line maps of the bundled HV Tau C [Ne II] cube (continuum, PSF removal, moments, velocities)."""
    t0 = time.time()
    from .cube import CubeSet, line_maps, prepare_line
    lm = line_maps(prepare_line(CubeSet("example:HV_Tau_C_cube"), "[Ne II] 12.81"), n_mc=20)
    s = lm.summary
    ok = s["n_spaxels_velocity"] > 50 and (s["extended_flux_W_m2"] or 0) > 0
    return {"status": OK if ok else WARN, "seconds": round(time.time() - t0, 2), "velocity_spaxels": s["n_spaxels_velocity"],
            "point_source_W_m2": s["point_source_line_flux_W_m2"], "extended_W_m2": s["extended_flux_W_m2"]}


def check_rotdiag() -> dict:
    """An H2 rotation diagram of the bundled flux table (two temperatures + A_V, least squares)."""
    t0 = time.time()
    from .rotdiag import example_config, run_rotdiag
    cfg = example_config("h2_fluxes")
    cfg.mcmc.enabled = False; cfg.plots = False
    res = run_rotdiag(cfg, save=False)
    f = res.fit
    ok = f is not None and 300 < f.best["T1"] < 700 and len(res.features) == 7
    return {"status": OK if ok else WARN, "seconds": round(time.time() - t0, 2), "n_lines": len(res.features),
            "T1": round(f.best["T1"], 1) if f else None, "T2": round(f.best["T2"], 1) if f else None,
            "Av": round(f.best["Av"], 2) if f else None}


def check_parallel() -> dict:
    import multiprocessing as mp
    methods = mp.get_all_start_methods()
    threads = {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    return {"cpu_count": os.cpu_count(), "start_methods": methods, "fork": "fork" in methods,
            "blas_threads_env": threads, "status": OK}


def benchmark(n: int = 20) -> dict:
    """Time the forward model: hot H2O (HITEMP) + CO2 on MRS channel-3B pixels, 14.3–15.6 µm."""
    import numpy as np
    from .model import Component, build_model
    wave = np.arange(14.3, 15.6, 0.0025)
    comps = [Component("H2O", "H2O", logN=18.0, T=600.0, logR=-0.4, linelist_release="hitemp"),
             Component("CO2", "CO2", logN=17.5, T=500.0, logR=-0.6)]
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        m = build_model(comps, wave, 140.0, [(14.3, 15.6)])
    t_build = time.time() - t0
    m.evaluate()
    t0 = time.time()
    for i in range(n):
        m.evaluate({"H2O": {"T": 600.0 + i}, "CO2": {"logN": 17.5 + 0.01 * i}})
    per = (time.time() - t0) / n
    return {"pixels": len(wave), "fine_grid": m.grid.n, "build_s": round(t_build, 2), "ms_per_model": round(1e3 * per, 1),
            "status": OK if per < 0.5 else WARN}


def collect(quick: bool = False) -> dict:
    rep = {"python": {"version": platform.python_version(), "executable": sys.executable, "platform": platform.platform(),
                      "status": OK if sys.version_info[:2] >= PYTHON_MIN else FAIL}}
    try:
        from . import __version__
        rep["jalebi"] = {"version": __version__, "location": os.path.dirname(os.path.abspath(__file__))}
    except Exception:  # pragma: no cover
        rep["jalebi"] = {"version": "?", "location": os.path.dirname(os.path.abspath(__file__))}
    rep["packages"] = check_packages()
    core_ok = all(r["status"] != FAIL for r in rep["packages"]["core"])
    if core_ok:
        rep["imports"] = check_imports([r["import"] for r in rep["packages"]["core"]])
        core_ok = all(r["status"] == OK for r in rep["imports"])
    if not quick and core_ok:
        for key, fn in (("linelists", check_linelists), ("examples", check_examples), ("parallel", check_parallel),
                        ("benchmark", benchmark), ("cube", check_cube), ("rotdiag", check_rotdiag)):
            try:
                rep[key] = fn()
            except Exception as e:
                rep[key] = {"status": FAIL, "error": f"{type(e).__name__}: {e}"}
    ready = rep["python"]["status"] == OK and core_ok and rep.get("linelists", {}).get("status", OK) == OK
    rep["ready"] = bool(ready)
    missing = [r["name"] for r in rep["packages"]["core"] if r["status"] != OK]
    rep["fix"] = None
    if missing:
        rep["fix"] = f"{sys.executable} -m pip install " + " ".join(
            f'"{m}>={CORE[m][0]}"' for m in missing)
    return rep


# ----------------------------------------------------------------------------------------------
# Printing
# ----------------------------------------------------------------------------------------------

def _plain(rep: dict):
    use_colour = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
    col = {OK: "\033[32m", WARN: "\033[33m", FAIL: "\033[31m", "absent": "\033[90m"} if use_colour else {}
    end = "\033[0m" if use_colour else ""
    mark = {OK: "✔", WARN: "!", FAIL: "✘", "absent": "·"}

    def line(st, text):
        print(f"  {col.get(st, '')}{mark.get(st, '?')}{end} {text}")

    from ._banner import compact_banner
    print()
    print(compact_banner(rep["jalebi"]["version"], colour=use_colour))
    print(f"doctor report for the installation at {rep['jalebi']['location']}\n")
    p = rep["python"]
    line(p["status"], f"Python {p['version']} ({p['executable']})")
    print("\nRequired packages")
    for r in rep["packages"]["core"]:
        line(r["status"], f"{r['name']:<14} {r['installed'] or 'missing':<12} (>= {r['min']})  {r['purpose']}")
    for r in rep.get("imports", []):
        if r["status"] != OK:
            line(FAIL, f"import {r['module']} failed: {r.get('error')}")
    print("\nOptional extras   (pip install \"jalebi[<extra>]\")")
    for g, rows in rep["packages"]["optional"].items():
        st = OK if all(r["status"] == OK for r in rows) else ("absent" if all(r["status"] == "absent" for r in rows) else WARN)
        line(st, f"[{g}] " + ", ".join(f"{r['name']} {r['installed'] or '—'}" for r in rows))
    if "linelists" in rep:
        L = rep["linelists"]
        print("\nLine lists")
        if "error" in L:
            line(FAIL, L["error"])
        else:
            line(L["status"], f"{L['n_lists']} lists, {L['MB']} MB — user cache {L['user_cache']}")
            print(f"      {', '.join(L['lists'])}")
            for t in L["tests"]:
                if t["status"] == OK:
                    line(OK, f"{t['list']:<11} {t['lines']:>7} lines, Z(500 K) = {t['Z(500K)']:<9} [{t['partition']}]")
                else:
                    line(FAIL, f"{t['list']}: {t['error']}")
    if "examples" in rep:
        E = rep["examples"]
        print("\nExample data")
        line(E.get("status", FAIL), f"FZ Tau x1d files: {E.get('FZ_Tau_x1d_files')}/12, HV Tau C cube cutouts: "
                                    f"{E.get('HV_Tau_C_cubes', 0)}/5, synthetic spectrum: "
                                    f"{'yes' if E.get('synthetic') else 'no'}, scripts: {E.get('scripts') or 'repository only'}")
    if "parallel" in rep:
        P = rep["parallel"]
        print("\nParallelism")
        line(OK, f"{P['cpu_count']} CPU cores, start methods {P['start_methods']}"
                 + ("" if P["fork"] else "  (no fork: pools use spawn, slower start-up)"))
    if "benchmark" in rep:
        B = rep["benchmark"]
        print("\nSpeed")
        if "error" in B:
            line(FAIL, B["error"])
        else:
            line(B["status"], f"{B['ms_per_model']} ms per model (HITEMP H2O + CO2, {B['pixels']} pixels, {B['fine_grid']} fine-grid points; "
                              f"built in {B['build_s']} s)")
    if "cube" in rep:
        Q = rep["cube"]
        print("\nCube maps (jalebi.cube)")
        if "error" in Q:
            line(FAIL, Q["error"])
        else:
            line(Q["status"], f"HV Tau C [Ne II]: {Q['velocity_spaxels']} velocity spaxels, point source {Q['point_source_W_m2']:.2e}, "
                              f"extended {Q['extended_W_m2']:.2e} W m-2 ({Q['seconds']} s)")
    if "rotdiag" in rep:
        Q = rep["rotdiag"]
        print("\nRotation diagrams (jalebi.rotdiag)")
        if "error" in Q:
            line(FAIL, Q["error"])
        else:
            line(Q["status"], f"H2 flux table: {Q['n_lines']} lines, T1 {Q['T1']} K, T2 {Q['T2']} K, A_V {Q['Av']} ({Q['seconds']} s)")
    print()
    if rep["ready"]:
        print(f"  {col.get(OK, '')}Your jalebi is ready.{end}  Try:  jalebi demo   ·   jalebi cube demo   ·   jalebi rotdiag demo   ·   jalebi serve   ·   jalebi examples ./my_examples\n")
    else:
        print(f"  {col.get(FAIL, '')}Not ready yet.{end}")
        if rep.get("fix"):
            print(f"  Fix: {rep['fix']}")
        print()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="jalebi doctor", description="Check a JALEBI installation.")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    ap.add_argument("--quick", action="store_true", help="packages only (skip line lists, examples, benchmark)")
    a = ap.parse_args(argv)
    rep = collect(quick=a.quick)
    if a.json:
        print(json.dumps(rep, indent=1, default=str))
    else:
        _plain(rep)
    return 0 if rep["ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
