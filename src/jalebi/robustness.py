"""Continuum robustness (0.24, `fit.continuum_robustness`, `jalebi robustness`).

After a fit, the disk is refitted under ~20 slightly different continua and every molecule is scored with the
same shifted-template null statistic S as the detection test (jalebi.shift_null).  A molecule that disappears
(S < threshold) in some variants, or whose parameters move by more than its posterior error, is flagged.
The idea and the flagging rule: B. Shridharan (2026-10-10); docs/ROBUSTNESS.md.

Variants
    0           the continuum the fit used (the refined one if the model-aware refinement was accepted)
    estimators  the same spectrum re-estimated with other continuum settings (IRSQR quantile / knots, median_sg,
                sgclip), each followed by one model-aware refinement pass when the config uses refinement
    perturb     variant 0 x (1 + a_b + g_b x) per sub-band b, x in [-1, 1] across the band; a_b ~ N(0, sigma_b),
                g_b ~ N(0, tilt_ratio sigma_b); sigma_b = rms of the estimator variants about variant 0 in that band,
                clipped to [sigma_floor, sigma_cap]

Each variant changes only the continuum-subtracted data (y + c0 - c_v); the fit problem is reused
(FitProblem.with_data) and refitted locally from the best fit (Nelder-Mead, areas by NNLS) or, with refit: de, by
a short differential evolution.  S per variant uses the template bank of the main fit (shift_null.null_cache).

Per unit:  f = share of variants with S >= threshold
    robust               detected at variant 0 and f >= robust_frac
    not detected         not detected at variant 0 and f <= absent_frac
    continuum-dependent  otherwise
and for T, log N, log NA: sigma_cont = half the 16-84 % range over the variants where the unit is detected,
flagged continuum-limited when sigma_cont > sigma_post (the MCMC half-width).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ESTIMATOR_PRESETS = {
    "irsqr_q0.05": {"method": "irsqr", "quantile": 0.05},
    "irsqr_q0.2": {"method": "irsqr", "quantile": 0.2},
    "irsqr_k15": {"method": "irsqr", "knot_spacing": 15},
    "irsqr_k40": {"method": "irsqr", "knot_spacing": 40},
    "median_sg": {"method": "median_sg"},
    "sgclip": {"method": "sgclip"},
}
PARAMS = ("T", "logN", "logNA")


@dataclass
class RobustnessSettings:
    enabled: bool = True
    n_perturb: int = 13
    estimators: list = field(default_factory=lambda: list(ESTIMATOR_PRESETS))
    sigma_floor: float = 0.005
    sigma_cap: float = 0.05
    tilt_ratio: float = 0.5
    refit: str = "local"                # local | de
    max_evals: int = 3000
    de_maxiter: int = 40
    de_popsize: int = 8
    threshold: float = 5.0
    robust_frac: float = 0.9
    absent_frac: float = 0.1
    seed: int = 0
    save_plot: bool = True

    def to_dict(self):
        return dataclasses.asdict(self)


def settings_from_config(cfg) -> RobustnessSettings:
    c = getattr(cfg.fit, "continuum_robustness", None)
    if c is None:
        return RobustnessSettings()
    d = c.model_dump() if hasattr(c, "model_dump") else dict(c)
    keys = {f.name for f in dataclasses.fields(RobustnessSettings)}
    return RobustnessSettings(**{k: v for k, v in d.items() if k in keys})


# ------------------------------------------------------------------------------------------------ variants
def _band_x(spec) -> np.ndarray:
    """x in [-1, 1] across each sub-band (per pixel)."""
    x = np.zeros(len(spec.wave))
    for b in spec.bands:
        i = spec.band_slice(b)
        w = spec.wave[i]
        lo, hi = np.nanmin(w), np.nanmax(w)
        x[i] = 2.0 * (w - lo) / max(hi - lo, 1e-12) - 1.0
    return x


def estimator_continua(cfg, spec, prob, theta, names) -> dict[str, np.ndarray]:
    """Continuum of `spec` re-estimated with each estimator preset (+ one refinement pass with the best-fit gas
    model when the config refines).  Pixels the estimator leaves undefined keep variant 0."""
    from .continuum import estimate_continuum
    from .pipeline import continuum_settings
    base = continuum_settings(cfg)
    c0 = np.asarray(spec.continuum, float)
    gas = None
    if cfg.continuum.refine_iterations > 0:
        gas = np.full(len(spec.wave), np.nan)
        gas[prob.used] = prob.model_flux(theta)
    out = {}
    for name in names:
        preset = ESTIMATOR_PRESETS.get(name)
        if preset is None:
            raise ValueError(f"unknown continuum_robustness estimator {name!r}; known: {', '.join(ESTIMATOR_PRESETS)}")
        st = dataclasses.replace(base, **preset)
        try:
            c = estimate_continuum(spec, st)
            if gas is not None:
                s2 = spec.copy(); s2.continuum = np.where(np.isfinite(c), c, c0)
                c = estimate_continuum(s2, st, gas_model=gas)
        except Exception:                       # an estimator that fails on this spectrum: skip it
            continue
        out[name] = np.where(np.isfinite(c), c, c0)
    return out


def band_sigmas(spec, c0: np.ndarray, est: dict[str, np.ndarray], used: np.ndarray, s: RobustnessSettings) -> dict:
    """Per sub-band perturbation scale: rms over the estimator variants of their median relative offset from
    variant 0 on the fitted pixels, clipped to [sigma_floor, sigma_cap]."""
    sig = {}
    for b in spec.bands:
        i = np.zeros(len(spec.wave), bool); i[spec.band_slice(b)] = True
        m = i & used & np.isfinite(c0) & (np.abs(c0) > 0)
        if m.sum() < 5 or not est:
            sig[b] = s.sigma_floor
            continue
        rel = [float(np.nanmedian((c[m] - c0[m]) / np.abs(c0[m]))) for c in est.values()]
        rms = float(np.sqrt(np.nanmean(np.square(rel)))) if rel else 0.0
        sig[b] = float(np.clip(rms if np.isfinite(rms) else 0.0, s.sigma_floor, s.sigma_cap))
    return sig


def build_variants(cfg, spec, prob, theta, s: RobustnessSettings) -> tuple[list[dict], dict]:
    """[{index, kind, label, continuum}], band sigmas."""
    c0 = np.asarray(spec.continuum, float)
    used = np.zeros(len(spec.wave), bool); used[prob.used] = True
    est = estimator_continua(cfg, spec, prob, theta, s.estimators)
    sig = band_sigmas(spec, c0, est, used, s)
    variants = [{"index": 0, "kind": "nominal", "label": "fit continuum", "continuum": c0}]
    for name, c in est.items():
        variants.append({"index": len(variants), "kind": "estimator", "label": name, "continuum": c})
    rng = np.random.default_rng(s.seed)
    x = _band_x(spec)
    for k in range(s.n_perturb):
        fac = np.ones(len(spec.wave))
        desc = []
        for b in spec.bands:
            i = spec.band_slice(b)
            a = rng.normal(0.0, sig[b]); g = rng.normal(0.0, s.tilt_ratio * sig[b])
            fac[i] = 1.0 + a + g * x[i]
            desc.append(f"{b}:{100 * a:+.1f}%/{100 * g:+.1f}%")
        variants.append({"index": len(variants), "kind": "perturb", "label": f"perturb {k + 1}",
                         "detail": " ".join(desc), "continuum": c0 * fac})
    return variants, sig


# ------------------------------------------------------------------------------------------------ refit + score
def unit_params(prob, theta) -> dict[str, dict]:
    """T, log N, log NA of the lead component of every unit."""
    from .fit import logNA_from
    P, _ = prob.params_from_theta(theta)
    out = {}
    for unit, lead in prob.all_units().items():
        d = P.get(lead, {})
        logN, logR = d.get("logN", np.nan), d.get("logR", np.nan)
        na = d.get("logNA", logNA_from(logN, logR) if np.isfinite(logN) and np.isfinite(logR) else np.nan)
        out[unit] = {"T": float(d.get("T", np.nan)), "logN": float(logN), "logNA": float(na)}
    return out


def refit_variant(prob_v, theta0, s: RobustnessSettings, seed: int):
    if s.refit == "de":
        return prob_v.optimise(theta0, method="de", maxiter=s.de_maxiter, popsize=s.de_popsize, seed=seed, polish=True)
    return prob_v.optimise(theta0, method="local", maxiter=max(s.max_evals // 20, 10), seed=seed, polish=False)


def _key(stage_key: str, theta, s: RobustnessSettings, c0) -> str:
    h = hashlib.sha256()
    h.update(str(stage_key).encode())
    h.update(np.round(np.asarray(theta, float), 6).tobytes())
    h.update(np.round(np.nan_to_num(np.asarray(c0, float)), 9).tobytes())
    h.update(json.dumps(s.to_dict(), sort_keys=True, default=str).encode())
    return h.hexdigest()[:20]


def run(cfg, spec, prob, theta, outdir: str | None = None, posterior: pd.DataFrame | None = None,
        settings: RobustnessSettings | None = None, stage_key: str = "", target: str = "", say=print,
        exact_scoring: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Robustness table (per unit) and variants table (per variant x unit); written to `outdir` when given."""
    from . import shift_null as SN
    s = settings or settings_from_config(cfg)
    t0 = time.time()
    sn = SN.settings_from_config(cfg)
    sn = dataclasses.replace(sn, save_curves=False, threshold=s.threshold)
    variants, sig = build_variants(cfg, spec, prob, theta, s)
    c0 = variants[0]["continuum"]
    key = _key(stage_key, theta, s, c0)
    ck_path = os.path.join(outdir, "robustness.json") if outdir else None
    done = {}
    if ck_path and os.path.exists(ck_path):
        try:
            d = json.load(open(ck_path))
            if d.get("key") == key:
                done = {int(r["index"]): r for r in d.get("variants", [])}
                if done:
                    say(f"[stage] continuum robustness resumed from checkpoint ({len(done)} of {len(variants)} variants)")
        except Exception:
            done = {}

    model = prob.model
    ctx = model.exact() if (exact_scoring and getattr(model, "emulator", None) is not None) else None
    if ctx is not None:
        with ctx:
            cache = SN.null_cache(prob, theta, sn)
    else:
        cache = SN.null_cache(prob, theta, sn)
    y0 = np.asarray(prob.y, float)
    used = prob.used
    records = []
    for v in variants:
        i = v["index"]
        if i in done:
            records.append(done[i]); continue
        y_v = y0 + (c0[used] - np.asarray(v["continuum"], float)[used])
        prob_v = prob.with_data(y_v)
        if i == 0:
            th = np.asarray(theta, float); chi2 = float(prob_v.chi2(th))
        else:
            try:
                opt = refit_variant(prob_v, theta, s, seed=s.seed + i)
                th, chi2 = np.asarray(opt.theta, float), float(opt.chi2)
            except Exception as e:  # pragma: no cover
                say(f"  variant {i} ({v['label']}) refit failed: {e}")
                continue
        if ctx is not None:
            with model.exact():
                tab, _ = SN.shift_null(prob_v, th, sn, target=target, cache=cache)
        else:
            tab, _ = SN.shift_null(prob_v, th, sn, target=target, cache=cache)
        par = unit_params(prob_v, th)
        rec = {"index": i, "kind": v["kind"], "label": v["label"], "detail": v.get("detail", ""), "chi2": chi2,
               "theta": [float(x) for x in th],
               "units": {r.unit: {"S": float(r.S) if np.isfinite(r.S) else None, "molecule": r.molecule,
                                  **par.get(r.unit, {})} for r in tab.itertuples()}}
        records.append(rec)
        if ck_path:
            os.makedirs(outdir, exist_ok=True)
            tmp = ck_path + ".tmp"
            json.dump({"key": key, "settings": s.to_dict(), "band_sigma": sig, "variants": records}, open(tmp, "w"),
                      default=float)
            os.replace(tmp, ck_path)
    vt = variants_table(records, target)
    table = summarise(vt, posterior, s, target)
    if outdir:
        vt.to_csv(os.path.join(outdir, "continuum_variants.csv"), index=False)
        table.to_csv(os.path.join(outdir, "continuum_robustness.csv"), index=False)
        np.savez_compressed(os.path.join(outdir, "continuum_variants.npz"), wave=spec.wave,
                            continua=np.array([v["continuum"] for v in variants]),
                            labels=np.array([v["label"] for v in variants]), flux=spec.flux)
        if s.save_plot:
            try:
                plot(spec, variants, vt, s, os.path.join(outdir, "continuum_robustness.png"), title=target)
            except Exception as e:  # pragma: no cover
                say(f"  robustness plot failed: {e}")
    say(f"continuum robustness ({len(records)} variants, {time.time() - t0:.0f} s): " + describe(table))
    return table, vt


def variants_table(records: list[dict], target: str = "") -> pd.DataFrame:
    rows = []
    for r in records:
        for unit, u in r["units"].items():
            rows.append({"target": target, "variant": r["index"], "kind": r["kind"], "label": r["label"],
                         "chi2": r["chi2"], "unit": unit, "molecule": u.get("molecule", ""),
                         "S": np.nan if u.get("S") is None else u["S"],
                         **{p: u.get(p, np.nan) for p in PARAMS}})
    return pd.DataFrame(rows)


def _sigma_post(posterior: pd.DataFrame | None, unit: str, p: str) -> float:
    if posterior is None or not len(posterior):
        return np.nan
    m = posterior[posterior.parameter.astype(str) == f"{unit}.{p}"]
    if not len(m):
        return np.nan
    r = m.iloc[0]
    return 0.5 * (float(r["minus"]) + float(r["plus"]))


def summarise(vt: pd.DataFrame, posterior: pd.DataFrame | None, s: RobustnessSettings, target: str = "") -> pd.DataFrame:
    rows = []
    if not len(vt):
        return pd.DataFrame()
    for unit, g in vt.groupby("unit", sort=False):
        g = g.sort_values("variant")
        S = g.S.to_numpy(float)
        det = np.isfinite(S) & (S >= s.threshold)
        nominal = bool(det[0]) if g.variant.iloc[0] == 0 else False
        f = float(det.mean()) if len(det) else np.nan
        if nominal and f >= s.robust_frac:
            cls = "robust"
        elif not nominal and f <= s.absent_frac:
            cls = "not detected"
        else:
            cls = "continuum-dependent"
        row = {"target": target, "unit": unit, "molecule": g.molecule.iloc[0], "class": cls, "frac": f,
               "n_variants": int(len(g)), "S_nominal": float(S[0]) if len(S) else np.nan,
               "S_min": float(np.nanmin(S)) if np.any(np.isfinite(S)) else np.nan,
               "S_median": float(np.nanmedian(S)) if np.any(np.isfinite(S)) else np.nan}
        gd = g[det] if det.any() else g
        flags = []
        for p in PARAMS:
            v = gd[p].to_numpy(float)
            v = v[np.isfinite(v)]
            sc = 0.5 * float(np.percentile(v, 84) - np.percentile(v, 16)) if len(v) >= 3 else np.nan
            sp = _sigma_post(posterior, unit, p)
            row[f"{p}_nominal"] = float(g[p].iloc[0])
            row[f"{p}_syst"] = sc
            row[f"{p}_post"] = sp
            row[f"{p}_total"] = float(np.hypot(sc, sp)) if np.isfinite(sc) and np.isfinite(sp) else np.nan
            lim = bool(np.isfinite(sc) and np.isfinite(sp) and sc > sp)
            row[f"{p}_cont_limited"] = lim
            if lim:
                flags.append(p)
        row["cont_limited"] = ",".join(flags)
        rows.append(row)
    return pd.DataFrame(rows)


def describe(table: pd.DataFrame) -> str:
    if table is None or not len(table):
        return "no units"
    return ", ".join(f"{r['unit']}: {r['class']} (f={r['frac']:.2f})" for r in table.to_dict("records"))


def plot(spec, variants, vt: pd.DataFrame, s: RobustnessSettings, path: str, title: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [2, 1.3]})
    w = spec.wave
    a1.plot(w, spec.flux, lw=0.5, color="#52514e", label="flux")
    for v in variants[1:]:
        a1.plot(w, v["continuum"], lw=0.6, color="#2a78d6" if v["kind"] == "estimator" else "#eb6834", alpha=0.5)
    a1.plot(w, variants[0]["continuum"], lw=1.4, color="#0b0b0b", label="fit continuum")
    a1.plot([], [], color="#2a78d6", label="estimator variants"); a1.plot([], [], color="#eb6834", label="perturbations")
    a1.set_xlabel("wavelength (µm)"); a1.set_ylabel("flux (Jy)"); a1.legend(frameon=False, fontsize=8, ncol=4)
    a1.set_title(title or "continuum variants", loc="left", fontsize=11)
    units = list(dict.fromkeys(vt.unit))
    for j, u in enumerate(units):
        g = vt[vt.unit == u]
        col = np.where(g.S >= s.threshold, "#2a78d6", "#c3c2b7")
        a2.scatter(np.full(len(g), j) + np.linspace(-0.25, 0.25, len(g)), np.clip(g.S, -5, 60), c=col, s=14)
        g0 = g[g.variant == 0]
        if len(g0):
            a2.scatter([j], [np.clip(g0.S.iloc[0], -5, 60)], marker="D", s=40, facecolor="none", edgecolor="#0b0b0b")
    a2.axhline(s.threshold, ls="--", lw=1, color="#0b0b0b")
    a2.set_xticks(range(len(units))); a2.set_xticklabels(units, fontsize=8, rotation=20)
    a2.set_ylabel("S per variant (clipped -5..60)")
    for ax in (a1, a2):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# ------------------------------------------------------------------------------------------------ finished fits
def run_outdir(outdir: str, settings: RobustnessSettings | None = None, backend: str | None = None,
               config: str | None = None, say=print) -> pd.DataFrame:
    """Robustness of a finished fit folder (config.yaml + prep.csv + best_fit.json [+ summary.csv])."""
    from .data import load_csv
    from .shift_null import problem_from_outdir
    cfg, prob, theta, name = problem_from_outdir(outdir, backend=backend, config=config)
    spec = load_csv(os.path.join(outdir, "prep.csv"))
    post = None
    sp = os.path.join(outdir, "summary.csv")
    if os.path.exists(sp):
        post = pd.read_csv(sp)
    key = ""
    try:
        from .resume import stage_key
        key = stage_key(cfg)
    except Exception:
        pass
    table, _ = run(cfg, spec, prob, theta, outdir=outdir, posterior=post, settings=settings, stage_key=key,
                   target=name, say=say)
    return table


def survey_table(root: str) -> pd.DataFrame:
    import glob
    frames = []
    for f in sorted(glob.glob(os.path.join(root, "*", "continuum_robustness.csv"))):
        try:
            t = pd.read_csv(f)
        except Exception:
            continue
        t["folder"] = os.path.basename(os.path.dirname(f))
        frames.append(t)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(out):
        out.to_csv(os.path.join(root, "continuum_robustness_survey.csv"), index=False)
    return out
