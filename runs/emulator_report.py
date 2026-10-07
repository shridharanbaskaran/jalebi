"""Emulator report on FZ Tau (jalebi 0.18 per-disk tables and 0.21 shared tables, fit.model_backend: emulator).

    python runs/emulator_report.py accuracy  [--n 2000] [--cache shared|per_disk] [--out emulator_report]
    python runs/emulator_report.py grids     [--n 500]  [--cache-dir DIR]      # synthetic grids: other bands, v_shift, distance
    python runs/emulator_report.py approximations                             # (a) shift, (b) R, (c) resampling vs points/FWHM
    python runs/emulator_report.py invariance [--cache-dir DIR]               # two disks, one shared file per unit
    python runs/emulator_report.py posterior [--nsteps 3000] [--mode exact|per_disk|shared|all]
    python runs/emulator_report.py timing    [--cache shared|per_disk]        # ms per ln P, load, projection, spot check
    python runs/emulator_report.py survey    [--out emulator_report]          # projected cost of the 300-disk survey

accuracy   For H2O hot / warm / cold (HITEMP, HITEMP, HITRAN), CO, CO2 + 13CO2, C2H2, HCN and OH on the pixel
           grid of the bundled FZ Tau MIRI spectrum (4.9-5.35 and 9-27.5 um, continuum-subtracted with the
           FZ_Tau_quick settings): build the tables, draw n random (T, log N) inside each component's bounds and
           compare with the exact model.  Reports max and 99th percentile |emulator - exact| / sigma_pixel at
           the largest area the data allow, and the integrated-flux error (targets 0.1 sigma, 0.1 %; relative to
           the 1-sigma noise of the integrated flux where the flux is weaker than that noise).
posterior  The FZ_Tau_quick fit (13.45-17.5 um: hot + warm water, CO2 + 13CO2, C2H2, HCN) with
           linear: profile, moves de, init scaled: optimiser once (exact), then MCMC with the exact model and
           with the emulator from the same start; medians compared (targets 0.05 dex, 20 K).
timing     Exact vs emulator ln P on one core: single calls and one vectorised call for 64 walkers.

0.21 (shared tables): the same protocols with --cache shared (the default of fit.emulator.cache).  The tables are
read from --cache-dir (default $JALEBI_EMULATOR_DIR, ~/.jalebi/emulator) and built there when missing: build the
FZ Tau set first with `jalebi emulator build --survey examples/configs/FZ_Tau_quick.yaml -j 4` (or let `accuracy`
build them; H2O HITEMP over the whole MRS range takes the longest).  Pixels next to a gap between fit windows
("uncovered": the exact model dilutes them, see emulator_shared.spot_check) are left out of the shared comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# (name, molecule, release, T bounds, log N bounds); the other parameters at typical values
ACCURACY_COMPONENTS = [
    ("H2O_hot", "H2O", "hitemp", (600.0, 1500.0), (14.0, 21.0)),
    ("H2O_warm", "H2O", "hitemp", (250.0, 900.0), (14.0, 21.0)),
    ("H2O_cold", "H2O", "hitran", (100.0, 400.0), (14.0, 21.0)),
    ("CO", "CO", "hitemp", (100.0, 3000.0), (14.0, 21.0)),
    ("CO2", "CO2", "hitran", (100.0, 1500.0), (13.0, 21.0)),
    ("13CO2", "13CO2", "hitran", None, None),            # tied to CO2, ratio 70 fixed
    ("C2H2", "C2H2", "hitran", (100.0, 1500.0), (13.0, 21.0)),
    ("HCN", "HCN", "hitran", (100.0, 1500.0), (13.0, 21.0)),
    ("OH", "OH", "hitran", (100.0, 3000.0), (13.0, 21.0)),
]
WINDOWS = [(4.9, 5.35), (9.0, 27.5)]


def fz_tau_spectrum():
    from jalebi.config import ProjectConfig
    from jalebi.pipeline import prepare
    cfg = ProjectConfig.load(os.path.join(REPO, "examples", "configs", "FZ_Tau_quick.yaml"))
    return cfg, prepare(cfg)


def accuracy_problem(spec, cache_dir=None, components=None):
    from jalebi.fit import FitProblem, Param
    from jalebi.model import Component
    comps, free = [], []
    for name, mol, rel, Tb, Nb in (components or ACCURACY_COMPONENTS):
        if name == "13CO2":
            comps.append(Component(name, mol, tie_to="CO2", ratio=70.0, linelist_release=rel))
            continue
        T0, N0 = 0.5 * (Tb[0] + Tb[1]), 0.5 * (Nb[0] + Nb[1])
        comps.append(Component(name, mol, logN=N0, T=T0, logR=-0.7, linelist_release=rel))
        free += [Param(name, "logN", *Nb, init=N0), Param(name, "T", *Tb, init=T0), Param(name, "logR", -2.5, 1.5, init=-0.7)]
    return FitProblem(spec, comps, WINDOWS, free, oversample=6, fit_noise_scale=True)


BANDS = None            # --bands 3A,3B,3C restricts the shared dense grid (quick tests on one fit window; another key)


def _settings(cache: str, cache_dir=None, **kw):
    from jalebi.emulator import EmulatorSettings
    if BANDS and cache == "shared":
        kw.setdefault("bands", tuple(BANDS))
    return EmulatorSettings(cache=cache, cache_dir=cache_dir, **kw)


def accuracy(n: int = 2000, out: str = "emulator_report", cache_dir: str | None = None, seed: int = 1,
             components: list[str] | None = None, cache: str = "shared", prob=None, label: str = "FZ Tau", tag: str = ""):
    from jalebi.emulator import errors, exact_rows
    os.makedirs(out, exist_ok=True)
    if prob is None:
        cfg, spec = fz_tau_spectrum()
        # shared: the cold water component on HITEMP too (the survey's release for every H2O component: one table
        # serves hot, warm and cold), and no OH (never a survey candidate; its 100-3000 K table takes ~2 h)
        comps = [(n, m, "hitemp" if (cache == "shared" and m == "H2O") else r, T, N) for n, m, r, T, N in ACCURACY_COMPONENTS
                 if not (cache == "shared" and m == "OH" and not (components and "OH" in components))]
        prob = accuracy_problem(spec, components=comps)
    print(f"{label}: {len(prob.y)} pixels, fine grid {prob.model.grid.n} points, {cache} tables", flush=True)
    st = _settings(cache, cache_dir)
    if cache == "shared":
        # the FZ_Tau_quick config's survey boxes (so the same files serve the fit), the default box for the rest
        from jalebi.emulator_shared import default_box, molecule_boxes
        from jalebi.config import ProjectConfig
        bq = molecule_boxes(ProjectConfig.load(os.path.join(REPO, "examples", "configs", "FZ_Tau_quick.yaml")))
        bounds0, _ = prob.emulator_bounds()
        st.boxes = {c.molecule: bq.get(c.molecule) or default_box(bounds0[c.name]) for c in prob.components if c.name in bounds0}
        for c in prob.components:                              # a box must contain the unit's bounds
            b = bounds0.get(c.name); box = st.boxes.get(c.molecule)
            if b and box and not (box["T"][0] <= b["T"][0] and b["T"][1] <= box["T"][1] and box["logN"][0] <= b["logN"][0] and b["logN"][1] <= box["logN"][1]):
                st.boxes[c.molecule] = default_box(b)
    t0 = time.time()
    em = prob.use_emulator(st, say=lambda m: print(m, flush=True))
    t_build = time.time() - t0
    covered = prob.model.covered
    sig = prob.sigma / np.sqrt(prob.weights)
    f_ref = float(np.nanmax(np.abs(prob.y)))
    bounds, amax = prob.emulator_bounds()
    P = prob.model.resolve_params()
    rng = np.random.default_rng(seed)
    rows = []
    for key, tab in em.tables.items():
        if components and key not in components:
            continue
        comp = next(c for c in prob.components if c.name == key)
        b = bounds[key]
        T = np.exp(rng.uniform(np.log(b["T"][0]), np.log(b["T"][1]), n))
        N = rng.uniform(b["logN"][0], b["logN"][1], n)
        es, ef, fl = np.zeros(n), np.zeros(n), np.zeros(n, bool)
        t1 = time.time()
        for i in range(n):
            Fx, _ = exact_rows(prob.model, comp, P[key], T[i], [N[i]])
            Fe, _ = tab.flux(T[i], N[i])
            if cache == "shared":
                Fe[~covered] = Fx[0][~covered]
            es[i], ef[i], fl[i] = (x[0] for x in errors(Fe, Fx, sig, f_ref, amax[key], return_floor=True))
        sc = tab.meta.get("spot_check", {})
        rows.append({"component": key, "cache": cache, "line_list": tab.linelist, "T_bounds": f"{b['T'][0]:.0f}-{b['T'][1]:.0f}",
                     "logN_bounds": f"{b['logN'][0]:.2f}-{b['logN'][1]:.2f}", "nodes_T": len(tab.lnT),
                     "nodes_logN": len(tab.logN), "support_px": len(tab.pix), "n_points": n,
                     "max_sigma": es.max(), "p99_sigma": np.percentile(es, 99), "max_flux_pct": 100 * ef.max(),
                     "p99_flux_pct": 100 * np.percentile(ef, 99),
                     "flux_below_noise_frac": float(fl.mean()),        # points where e_flux is relative to the flux noise
                     "pass": bool(es.max() < 0.1 and ef.max() < 1e-3),
                     "build_s": tab.meta.get("build_s", np.nan) if not tab.meta.get("from_cache") else np.nan,
                     "MB": tab.meta.get("file_bytes", 0) / 1e6, "check_s": time.time() - t1,
                     "pixels_excluded": int((~covered).sum()) if cache == "shared" else 0,
                     "load_s": tab.meta.get("load_s", np.nan), "project_s": tab.meta.get("project_s", np.nan),
                     "spot_check_s": sc.get("time_s", np.nan), "spot_max_sigma": sc.get("max_sigma", np.nan),
                     "spot_max_flux_pct": 100 * sc.get("max_flux", np.nan), "spot_oversample": sc.get("oversample", np.nan),
                     "shared_file": os.path.basename(tab.meta.get("shared_file", "") or "")})
        print(pd.DataFrame([rows[-1]]).to_string(index=False), flush=True)
        np.savez(os.path.join(out, f"accuracy_{key}{tag}.npz"), T=T, logN=N, e_sigma=es, e_flux=ef, flux_below_noise=fl)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(out, (f"accuracy_{cache}{tag}.csv" if cache == "shared" else f"accuracy{tag}.csv")), index=False)
    print(f"\ntables ready in {t_build:.0f} s; exact-unit fallbacks: {em.exact_units}")
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    return tab


# ---------------------------------------------------------------------------------------------------
def posterior_setup(nsteps: int):
    from jalebi.pipeline import build_problem
    cfg, spec = fz_tau_spectrum()
    cfg.fit.mcmc.linear = "profile"; cfg.fit.mcmc.moves = "de"; cfg.fit.mcmc.init = "scaled"
    cfg.fit.mcmc.nsteps = nsteps; cfg.fit.mcmc.nwalkers = None
    prob = build_problem(cfg, spec, backend="exact")
    return cfg, spec, prob


def posterior(nsteps: int = 3000, out: str = "emulator_report", mode: str = "all", seed: int = 0, cache_dir=None):
    """mode: exact | per_disk | shared | all (the three in turn) | both (exact + per_disk, the 0.18 report)."""
    os.makedirs(out, exist_ok=True)
    cfg, spec, prob = posterior_setup(nsteps)
    start = os.path.join(out, "posterior_start.npy")
    if os.path.exists(start):
        th = np.load(start)
    else:
        o = cfg.fit.optimise
        t0 = time.time()
        opt = prob.optimise(method="de", maxiter=o.maxiter, popsize=o.popsize, seed=0, workers=1)
        th = opt.theta
        np.save(start, th)
        print(f"optimiser (exact): chi2_red {opt.chi2_red:.3f} in {time.time() - t0:.0f} s", flush=True)
    backends = {"all": ["exact", "per_disk", "shared"], "both": ["exact", "per_disk"]}.get(mode, [mode])
    res = {}
    for backend in backends:
        t_tab = 0.0
        if backend in ("per_disk", "shared"):
            t0 = time.time()
            em = prob.use_emulator(_settings(backend, cache_dir, boxes=None if backend == "per_disk" else __import__("jalebi.emulator_shared", fromlist=["x"]).molecule_boxes(cfg)),
                                   say=lambda m: print(m, flush=True))
            t_tab = time.time() - t0
            print("\n".join(em.summary()), flush=True)
        t0 = time.time()
        r = prob.mcmc(th, nsteps=nsteps, seed=seed, moves="de", init="scaled", linear="profile")
        wall = time.time() - t0
        s = r.summary().set_index("parameter")
        tau = np.asarray(r.meta.get("tau_sampled"), float)
        s[["median", "minus", "plus"]].to_csv(os.path.join(out, f"posterior_{backend}.csv"))
        json.dump({"wall_s": wall, "tables_s": t_tab, "tau_max": float(np.nanmax(tau)), "nsteps": nsteps, "nwalkers": r.chain.shape[1],
                   "acceptance": r.acceptance}, open(os.path.join(out, f"posterior_{backend}.json"), "w"), indent=1)
        print(f"{backend}: {wall:.0f} s (+ {t_tab:.0f} s tables), tau_max {np.nanmax(tau):.0f}, acceptance {r.acceptance:.2f}", flush=True)
        res[backend] = s
        prob.model.emulator = None
    out_t = None
    if all(os.path.exists(os.path.join(out, f"posterior_{b}.csv")) for b in ("exact", "per_disk")):
        out_t = compare_posteriors(out, "exact", "per_disk", 0.05, 20.0)
    if all(os.path.exists(os.path.join(out, f"posterior_{b}.csv")) for b in ("exact", "shared")):
        out_t = compare_posteriors(out, "exact", "shared", 0.05, 20.0)
    if all(os.path.exists(os.path.join(out, f"posterior_{b}.csv")) for b in ("per_disk", "shared")):
        out_t = compare_posteriors(out, "per_disk", "shared", 0.01, 5.0)
    if os.path.exists(os.path.join(out, "posterior_emulator.csv")) and os.path.exists(os.path.join(out, "posterior_exact.csv")):
        out_t = compare_posteriors(out, "exact", "emulator", 0.05, 20.0)       # a 0.18 report folder
    return out_t


def compare_posteriors(out: str = "emulator_report", a_name: str = "exact", b_name: str = "shared", tol_dex: float = 0.05,
                       tol_K: float = 20.0):
    a = pd.read_csv(os.path.join(out, f"posterior_{a_name}.csv"), index_col=0)
    b = pd.read_csv(os.path.join(out, f"posterior_{b_name}.csv"), index_col=0)
    rows = []
    for k in a.index:
        if not k.endswith((".T", ".logN", ".logR", ".logNA", ".log_s")):
            continue
        d = b.loc[k, "median"] - a.loc[k, "median"]
        tol = tol_K if k.endswith(".T") else tol_dex
        rows.append({"parameter": k, a_name: a.loc[k, "median"], b_name: b.loc[k, "median"], "diff": d,
                     f"sigma_{a_name}": 0.5 * (a.loc[k, "minus"] + a.loc[k, "plus"]), "tolerance": tol, "pass": abs(d) <= tol})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out, f"posterior_compare_{a_name}_{b_name}.csv"), index=False)
    print(f"\n{a_name} vs {b_name} (tolerance {tol_dex} dex / {tol_K} K): {int(t['pass'].sum())}/{len(t)} pass, "
          f"max |diff| {t.loc[~t.parameter.str.endswith('.T'), 'diff'].abs().max():.4f} dex, "
          f"{t.loc[t.parameter.str.endswith('.T'), 'diff'].abs().max():.2f} K")
    print(t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return t


# ---------------------------------------------------------------------------------------------------
def timing(out: str = "emulator_report", cache_dir=None, nw: int = 64, cache: str = "shared"):
    import timeit
    from jalebi.linear import LinearProblem
    cfg, spec, prob = posterior_setup(10)
    os.makedirs(out, exist_ok=True)
    start = os.path.join(out, "posterior_start.npy")
    th = np.load(start) if os.path.exists(start) else prob.theta0()
    rng = np.random.default_rng(0)
    TH = np.clip(th + 0.002 * (prob.hi - prob.lo) * rng.standard_normal((nw, prob.ndim)), prob.lo + 1e-9, prob.hi - 1e-9)
    rows = []

    def measure(label, p):
        lp = LinearProblem(p, "profile", th)
        R = TH[:, lp.keep_idx]
        r = {"backend": label}
        p.log_prob(TH[0]); lp.log_prob_blob(R[0])                  # warm-up (opacity bases, caches)
        k1 = 5 if label == "exact" else 300
        r["single_ms"] = 1e3 * timeit.timeit(lambda: p.log_prob(TH[0]), number=k1) / k1
        r["profile_single_ms"] = 1e3 * timeit.timeit(lambda: lp.log_prob_blob(R[0]), number=k1) / k1
        k = 1 if label == "exact" else 20
        r["vectorised_64_ms_per_walker"] = 1e3 * timeit.timeit(lambda: p.log_prob_many(TH), number=k) / k / nw
        r["profile_vectorised_64_ms_per_walker"] = 1e3 * timeit.timeit(lambda: lp.log_prob_blob_many(R), number=k) / k / nw
        rows.append(r)
        print(r, flush=True)
    measure("exact", prob)
    for c in (["per_disk", "shared"] if cache == "both" else [cache]):
        from jalebi.emulator_shared import molecule_boxes
        t0 = time.time()
        em = prob.use_emulator(_settings(c, cache_dir, boxes=molecule_boxes(cfg) if c == "shared" else None))
        load = time.time() - t0
        measure(c, prob)
        rows[-1]["tables_MB"] = sum(tb.meta.get("file_bytes", 0) for tb in em.tables.values()) / 1e6
        rows[-1]["tables_build_s"] = sum(tb.meta.get("build_s", 0) or 0 for tb in em.tables.values())
        rows[-1]["attach_s"] = load
        tm = em.info.get("timing", {})
        rows[-1]["load_s"] = sum(v.get("load_s", 0) for k, v in tm.items() if isinstance(v, dict))
        rows[-1]["project_s"] = sum(v.get("project_s", 0) for k, v in tm.items() if isinstance(v, dict))
        rows[-1]["spot_check_s"] = sum(v.get("spot_check_s", 0) for k, v in tm.items() if isinstance(v, dict))
        rows[-1]["units"] = len(em.tables); rows[-1]["exact_units"] = len(em.exact_units)
        prob.model.emulator = None
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out, f"timing_{cache}.csv" if cache != "per_disk" else "timing.csv"), index=False)
    print(t.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    return t


# ---------------------------------------------------------------------------------------------------
# 0.21: synthetic pixel grids, the approximations, invariance, survey projection
# ---------------------------------------------------------------------------------------------------
GRID_TRUTH = [{"name": "H2O", "molecule": "H2O", "logN": 18.0, "T": 600.0, "logR": -0.3, "linelist_release": "hitemp"},
              {"name": "CO2", "molecule": "CO2", "logN": 17.0, "T": 500.0, "logR": -0.5},
              {"name": "HCN", "molecule": "HCN", "logN": 17.0, "T": 650.0, "logR": -0.7},
              {"name": "CO", "molecule": "CO", "logN": 17.5, "T": 1200.0, "logR": -0.5, "linelist_release": "hitemp"}]
GRIDS = [  # (label, bands, v_shift, distance, windows, snr)
    ("synthetic_1A3B", ("1A", "3B"), 15.0, 140.0, [(4.95, 5.3), (13.5, 15.5)], 100.0),
    ("synthetic_2C3C4A", ("2C", "3C", "4A"), -60.0, 90.0, [(10.1, 11.6), (15.5, 17.9), (17.8, 20.8)], 200.0),
]


def grid_problem(label, bands, rv, distance, windows, snr, oversample=6):
    from jalebi.fit import FitProblem, Param
    from jalebi.model import Component
    from jalebi.synthetic import make_synthetic_spectrum
    spec, t = make_synthetic_spectrum(components=GRID_TRUTH, bands=bands, snr=snr, seed=1, oversample=3, distance_pc=distance)
    spec.continuum = t["continuum"]
    comps, free = [], []
    for c in GRID_TRUTH:
        comps.append(Component(**c, rv=rv))
        Tb = (100.0, 3000.0) if c["molecule"] == "CO" else (100.0, 1500.0)
        free += [Param(c["name"], "logN", 13.0, 21.0, init=c["logN"]), Param(c["name"], "T", *Tb, init=c["T"]),
                 Param(c["name"], "logR", -2.5, 1.5, init=c["logR"])]
    return FitProblem(spec, comps, windows, free, oversample=oversample, use_pipeline_err=True, fit_noise_scale=True,
                      releases={"H2O": "hitemp", "CO": "hitemp"})


def grids(n: int = 500, out: str = "emulator_report", cache_dir=None):
    """Accuracy of the shared tables on synthetic pixel grids with other sub-bands, v_shift, distance and S/N."""
    tabs = []
    for label, bands, rv, d, windows, snr in GRIDS:
        prob = grid_problem(label, bands, rv, d, windows, snr)
        t = accuracy(n, out, cache_dir, cache="shared", prob=prob, label=f"{label} (bands {bands}, v {rv} km/s, {d} pc, S/N {snr})", tag="_" + label)
        t.insert(0, "grid", label); t["v_shift"] = rv; t["distance_pc"] = d; t["bands"] = "+".join(bands)
        t["snr_max"] = float(np.nanmax(np.abs(prob.y)) / (prob.sigma / np.sqrt(prob.weights)).min())
        tabs.append(t)
    t = pd.concat(tabs)
    t.to_csv(os.path.join(out, "accuracy_grids.csv"), index=False)
    print(t.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    return t


def approximations(out: str = "emulator_report"):
    """(a) shift-then-convolve, (b) R at the dense points, (c) cubic resampling, on a smooth 200-line spectrum in
    band 3B: max |P_disk @ H - K @ shift(I)| / max(K @ I) per points-per-FWHM and v_shift."""
    from jalebi.constants import C, FWHM_TO_SIGMA
    from jalebi.emulator_shared import DenseGrid, DenseSpec, build_dense_lsf_operator
    from jalebi.instrument import build_lsf_operator, resolving_power
    from jalebi.model import FineGrid
    os.makedirs(out, exist_ok=True)
    wave_pix = np.exp(np.arange(np.log(13.5), np.log(15.4), 1.7e-4))
    fg = FineGrid([(13.3, 15.6)], step_kms=4.7 / 6)
    rng = np.random.default_rng(0)
    I = np.zeros(fg.n)
    for x in rng.uniform(np.log(13.4), np.log(15.5), 300):
        I += rng.uniform(0.1, 1) * np.exp(-0.5 * ((fg.x - x) / (4.7e3 / C * FWHM_TO_SIGMA)) ** 2)

    def run(ppf, rv, Rconst=None, trunc=6.0):
        g = DenseGrid(DenseSpec(points_per_fwhm=ppf, truncate=trunc, bands=("3B",)), R_constant=Rconst)
        Ish = np.interp(fg.x - rv * 1e3 / C, fg.x, I, left=0.0, right=0.0)
        K = build_lsf_operator(fg.x, fg.dx, wave_pix, resolving_power(wave_pix, constant=Rconst))
        ref = K @ Ish
        off, n, x0, h = g.segments[0]
        H = np.zeros(g.n); H[off:off + n] = build_dense_lsf_operator(fg.x, fg.dx, g.x[off:off + n], g.R[off:off + n], trunc) @ I
        P, cov = g.pixel_operator(wave_pix, rv)
        m = cov & (np.asarray(K.sum(axis=1)).ravel() > 0.999)
        d = (P @ H - ref)[m]
        return np.abs(d).max() / ref.max(), abs(d.sum() / ref[m].sum())
    rows = []
    for ppf in (4, 6, 8, 10, 12, 16):
        c_max, c_flux = run(ppf, 0.0, 2800.0)
        b_max, b_flux = run(ppf, 0.0)
        a_max, a_flux = run(ppf, 150.0, 2800.0)
        all_max, all_flux = run(ppf, 150.0)
        rows.append({"points_per_fwhm": ppf, "c_resampling_max": c_max, "c_resampling_flux": c_flux,
                     "b_plus_R_max": b_max, "a_plus_shift150_Rconst_max": a_max, "all_shift150_max": all_max, "all_flux": all_flux,
                     "sigma_at_snr100": 100 * all_max, "sigma_at_snr300": 300 * all_max})
        print(rows[-1], flush=True)
    for rv in (0.0, 37.0, 75.0, 150.0, -150.0):
        m1, f1 = run(8, rv); m2, f2 = run(8, rv, 2800.0)
        rows.append({"points_per_fwhm": 8, "v_shift": rv, "all_max": m1, "all_flux": f1, "Rconst_max": m2})
        print(rows[-1], flush=True)
    rows.append({"points_per_fwhm": 8, "truncate4_max": run(8, 0.0, 2800.0, 4.0)[0], "truncate6_max": run(8, 0.0, 2800.0, 6.0)[0]})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out, "approximations.csv"), index=False)
    print(t.to_string(index=False, float_format=lambda x: f"{x:.2e}"))
    return t


def invariance(out: str = "emulator_report", cache_dir=None):
    """The same shared file serves two disks with other pixels, rv and distance (equal cache keys)."""
    os.makedirs(out, exist_ok=True)
    cfg, spec = fz_tau_spectrum()
    comps = [(n, m, "hitemp" if m == "H2O" else r, T, N) for n, m, r, T, N in ACCURACY_COMPONENTS if m != "OH"]
    a = accuracy_problem(spec, components=comps)
    b = grid_problem("synthetic", ("1A", "3B", "4A"), -45.0, 200.0, [(4.95, 5.3), (13.6, 15.4), (18.0, 20.5)], 150.0)
    rows = []
    from jalebi.emulator_shared import default_box, molecule_boxes
    bq = molecule_boxes(cfg)
    for label, prob in (("FZ_Tau", a), ("synthetic", b)):
        bounds0, _ = prob.emulator_bounds()
        boxes = {}
        for c in prob.components:
            bb = bounds0.get(c.name)
            if bb is None:
                continue
            box = bq.get(c.molecule)
            if not (box and box["T"][0] <= bb["T"][0] and bb["T"][1] <= box["T"][1] and box["logN"][0] <= bb["logN"][0] and bb["logN"][1] <= box["logN"][1]):
                box = default_box(bb)
            boxes[c.molecule] = box
        em = prob.use_emulator(_settings("shared", cache_dir, read_only=False, boxes=boxes), say=lambda m: print(m, flush=True))
        for k, f in em.info["files"].items():
            sc = em.info["spot_check"].get(k, {})
            rows.append({"disk": label, "unit": k, "molecule": next(c.molecule for c in prob.components if c.name == k),
                         "file": os.path.basename(f), "npix": len(prob.y), "rv": prob.model.resolve_params()[k]["rv"],
                         "distance_pc": prob.model.distance_pc, "spot_max_sigma": sc.get("max_sigma"), "spot_max_flux_pct": 100 * sc.get("max_flux", np.nan)})
        for k, why in em.exact_units.items():
            rows.append({"disk": label, "unit": k, "file": f"exact: {why}", "npix": len(prob.y)})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out, "invariance.csv"), index=False)
    print(t.to_string(index=False))
    both = t.dropna(subset=["molecule"]).groupby("molecule")["file"].nunique() if "molecule" in t else pd.Series(dtype=int)
    shared = {m: n for m, n in both.items() if (t.molecule == m).sum() > 1}
    print("molecules on both disks -> distinct files:", shared)
    assert all(n == 1 for n in shared.values()), shared
    return t


def survey(out: str = "emulator_report", n_disks: int = 300, build_core_h: float | None = None, per_disk_min: float | None = None,
           exact_lnp_ms: float = 10.0, emu_lnp_ms: float = 0.25, calls: int = 384000):
    """Projected cost of the survey from the measured numbers (docs/EMULATOR.md, docs/SAMPLER_BENCHMARK.md)."""
    os.makedirs(out, exist_ok=True)
    acc = os.path.join(out, "accuracy_shared.csv")
    tim = os.path.join(out, "timing_shared.csv")
    if not os.path.exists(tim):
        tim = os.path.join(out, "timing_both.csv")
    load = spot = 0.0
    if os.path.exists(acc):
        a = pd.read_csv(acc)
        load = float(np.nansum(a["load_s"]) + np.nansum(a["project_s"])); spot = float(np.nansum(a["spot_check_s"]))
    if os.path.exists(tim):
        tt = pd.read_csv(tim).set_index("backend")
        if "shared" in tt.index:
            emu_lnp_ms = float(tt.loc["shared", "profile_vectorised_64_ms_per_walker"])
        if "exact" in tt.index:
            exact_lnp_ms = float(tt.loc["exact", "profile_vectorised_64_ms_per_walker"])
    sampling_min = calls * emu_lnp_ms / 1e3 / 60
    prep_min = 1.0; opt_min = 1.0; products_min = 1.5
    per_disk_shared = prep_min + (load + spot) / 60 + opt_min + sampling_min + products_min
    per_disk_old = prep_min + (per_disk_min or 20.0) + opt_min + sampling_min + products_min
    per_disk_exact = prep_min + 6.0 + calls * exact_lnp_ms / 1e3 / 60 + products_min
    rows = [{"setup": "shared tables (0.21): one build + 300 x (load + spot check + emcee profile 8000 steps)",
             "one_time_core_h": build_core_h or np.nan, "per_disk_min": per_disk_shared,
             "total_core_h": (build_core_h or 0.0) + n_disks * per_disk_shared / 60},
            {"setup": "per-disk tables (0.18): 300 x (build ~20 min + ...)", "one_time_core_h": 0.0, "per_disk_min": per_disk_old,
             "total_core_h": n_disks * per_disk_old / 60},
            {"setup": "exact model: 300 x (optimiser ~6 min + 384 000 calls x exact ln P)", "one_time_core_h": 0.0,
             "per_disk_min": per_disk_exact, "total_core_h": n_disks * per_disk_exact / 60}]
    t = pd.DataFrame(rows)
    t["assumptions"] = (f"load+project {load:.0f} s, spot check {spot:.0f} s per disk; ln P {emu_lnp_ms:.3f} ms (shared) / "
                        f"{exact_lnp_ms:.1f} ms (exact) per walker; prep 1, optimiser 1, products 1.5 min")
    t.to_csv(os.path.join(out, "survey_projection.csv"), index=False)
    print(t.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["accuracy", "grids", "approximations", "invariance", "posterior", "timing", "compare", "survey"])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--nsteps", type=int, default=3000)
    ap.add_argument("--mode", default="all", choices=["all", "both", "exact", "per_disk", "shared", "emulator"])
    ap.add_argument("--cache", default="shared", choices=["shared", "per_disk", "both"])
    ap.add_argument("--out", default="emulator_report")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--components", nargs="*", default=None)
    ap.add_argument("--build-core-h", type=float, default=None, help="survey: measured one-time build (core-hours)")
    ap.add_argument("--bands", default=None, help="shared: restrict the dense grid to these sub-bands, e.g. 3A,3B,3C (tests)")
    a = ap.parse_args()
    if a.bands:
        BANDS = a.bands.split(",")
    if a.what == "accuracy":
        accuracy(a.n, a.out, a.cache_dir, components=a.components, cache=a.cache)
    elif a.what == "grids":
        grids(a.n, a.out, a.cache_dir)
    elif a.what == "approximations":
        approximations(a.out)
    elif a.what == "invariance":
        invariance(a.out, a.cache_dir)
    elif a.what == "posterior":
        posterior(a.nsteps, a.out, "per_disk" if a.mode == "emulator" else a.mode, cache_dir=a.cache_dir)
    elif a.what == "compare":
        compare_posteriors(a.out)
    elif a.what == "survey":
        survey(a.out, build_core_h=a.build_core_h)
    else:
        timing(a.out, a.cache_dir, cache=a.cache)
