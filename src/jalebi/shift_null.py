"""Shifted-template null test for molecule detections (0.23, fit.shift_null, `jalebi shift-null`).

Why
    Delta BIC > 10 assumes white Gaussian noise and a correct model.  MIRI disk fits have neither: the residuals
    are correlated (lag-1 ACF ~0.5), over-dispersed (noise scale s ~1.7), the slab model is approximate, and the
    auto-detect stage tries ~15 flexible (T, N, R) candidates per disk.  A threshold set on objects without disk
    gas does not transfer either: their residuals (photospheres, silicate and ice bands) are not a line forest, a
    handful of them gives an unstable maximum, and Delta BIC scales with S/N.  This module builds the null from
    each spectrum itself.

How (per independent unit u of one fit)
    r       = data - continuum correction - (all units) : the residual of the best fit (no unit-u signal)
    t_b     = a small bank of unit-u templates around the best fit (T x {0.5..2}, log N + {-1, 0, +1}, inside
              the prior bounds; the best-fit template is one of them), each high-pass filtered (running median of
              `highpass_um`, per contiguous segment) so that smooth pseudo-continuum flux does not count
    signal  z0      = max_b < t_b , r + t_u >_w / |t_b|_w           (the template bank on residual + unit u)
    null    z(v)    = max_b < t_b(v), r >_w / |t_b(v)|_w             for Doppler shifts v_min <= |v| <= v_max
            with < a, b >_w = sum w a b, w = fit weights / (s sigma)^2, t_b(v)(lambda) = t_b(lambda / (1 + v/c))
            (pixels whose source wavelength falls outside its own contiguous segment are masked)
    S       = (z0 - median z(v)) / (1.4826 MAD z(v))                (significance against the spectrum's own null)
    FAP     = (1 + #{z(v) >= z0}) / (1 + n_null)                    (empirical; floor 1 / (n_null + 1))
    excess  = 1.4826 MAD z(v)   (1 for white noise with the right sigma; >1 = correlated residuals / confusion)

    The shifted templates have the line density and band shapes of the real one, but their lines fall at the
    wrong places: the spread of their match against the residual is what chance alignments with noise, misfit
    structure and other molecules' lines give in *this* spectrum, at *this* S/N.  Taking the bank maximum both at
    v = 0 and at every shift gives the null the same (T, N) freedom that the fit had.  The null is computed on the
    residual without unit u, so the molecule's own lines do not inflate it at shifts equal to its line spacing.

Outputs: <disk>/shift_null.csv (one row per unit) and shift_null_curves.csv (z(v) per unit, for plots).
Caveats: n_null is ~2 x (v_max - v_min) / step (default 38), so the empirical FAP floor is ~2.6 % and S is the
quantity to threshold (runs/shift_null_calibrate.py pools the nulls of a survey for the tail); adjacent shifts
closer than ~2 resolution elements (~300 km/s) are not independent.  Fits restricted to narrow line windows
(fit.regions) lose most of a shifted template off the fitted pixels: such shifts are dropped (min_energy_frac) and
with fewer than min_null valid shifts S is NaN (null_ok False).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

C_KMS = 299792.458


@dataclass
class ShiftNullSettings:
    enabled: bool = True
    v_min_kms: float = 1500.0
    v_max_kms: float = 9000.0
    step_kms: float = 400.0
    highpass_um: float = 0.3
    t_factors: tuple = (0.5, 0.7, 1.0, 1.4, 2.0)
    logN_offsets: tuple = (-1.0, 0.0, 1.0)
    threshold: float = 5.0          # S for detected_shift
    min_energy_frac: float = 0.5    # a shift counts only if >= this share of the template energy stays on fit pixels
    min_null: int = 10              # fewer valid shifts -> S = NaN, null_ok False (fit windows too narrow)
    save_curves: bool = True


def settings_from_config(cfg) -> ShiftNullSettings:
    d = getattr(cfg.fit, "shift_null", None)
    if d is None:
        return ShiftNullSettings()
    return ShiftNullSettings(enabled=d.enabled, v_min_kms=d.v_min_kms, v_max_kms=d.v_max_kms, step_kms=d.step_kms,
                             highpass_um=d.highpass_um, t_factors=tuple(d.t_factors), logN_offsets=tuple(d.logN_offsets),
                             threshold=d.threshold, min_energy_frac=getattr(d, "min_energy_frac", 0.5),
                             min_null=getattr(d, "min_null", 10), save_curves=d.save_curves)


# ------------------------------------------------------------------------------------------------ helpers
def segment_ids(wave: np.ndarray, gap_factor: float = 5.0) -> np.ndarray:
    """Contiguous-segment label per pixel: a new segment starts where the step exceeds gap_factor x the local
    median step (sub-band edges, excluded windows)."""
    wave = np.asarray(wave, float)
    if len(wave) < 3:
        return np.zeros(len(wave), int)
    dw = np.diff(wave)
    # local median step over ~50 pixels (robust to the channel-dependent sampling)
    k = 25
    pad = np.pad(dw, k, mode="edge")
    loc = np.array([np.median(pad[i:i + 2 * k + 1]) for i in range(len(dw))])
    brk = (dw > gap_factor * loc) | (dw <= 0)
    return np.concatenate([[0], np.cumsum(brk)])


def highpass(wave: np.ndarray, y: np.ndarray, width_um: float, seg: np.ndarray) -> np.ndarray:
    """y minus its running median over width_um, per segment (window in pixels from the segment's median step)."""
    from scipy.ndimage import median_filter
    out = np.array(y, float, copy=True)
    if width_um <= 0:
        return out
    for s in np.unique(seg):
        m = seg == s
        n = int(m.sum())
        if n < 5:
            out[m] = 0.0
            continue
        step = float(np.median(np.diff(wave[m])))
        size = int(round(width_um / max(step, 1e-12)))
        size = max(3, min(size | 1, n if n % 2 else n - 1))
        out[m] = y[m] - median_filter(y[m], size=size, mode="nearest")
    return out


def shift_plan(wave: np.ndarray, v_kms: float, seg: np.ndarray):
    """Interpolation plan for t(lambda / (1 + v/c)) on the same pixels: (left index, right index, fraction, mask);
    the mask drops pixels whose source wavelength is not inside the pixel's own contiguous segment (no
    interpolation across gaps or sub-band edges)."""
    src = wave / (1.0 + v_kms / C_KMS)
    j = np.searchsorted(wave, src)
    ok = (j > 0) & (j < len(wave))
    jl = np.clip(j - 1, 0, len(wave) - 1); jr = np.clip(j, 0, len(wave) - 1)
    ok &= (seg[jl] == seg) & (seg[jr] == seg)
    dx = wave[jr] - wave[jl]
    f = np.where(dx > 0, (src - wave[jl]) / np.where(dx > 0, dx, 1.0), 0.0)
    return jl, jr, np.clip(f, 0.0, 1.0), ok


def apply_plan(t: np.ndarray, plan) -> np.ndarray:
    jl, jr, f, ok = plan
    return np.where(ok, t[jl] * (1.0 - f) + t[jr] * f, 0.0)


def shift_template(wave: np.ndarray, t: np.ndarray, v_kms: float, seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """t(lambda / (1 + v/c)) on the same pixels and the valid-pixel mask (see shift_plan)."""
    plan = shift_plan(wave, v_kms, seg)
    return apply_plan(t, plan), plan[3]


def _match(t: np.ndarray, r: np.ndarray, w: np.ndarray, mask: np.ndarray | None = None) -> float:
    """< t, r >_w / |t|_w on the masked pixels (matched-filter S/N for white noise with weights w)."""
    if mask is not None:
        t = t[mask]; r = r[mask]; w = w[mask]
    den = float(np.sum(w * t * t))
    if not np.isfinite(den) or den <= 0:
        return np.nan
    return float(np.sum(w * t * r)) / np.sqrt(den)


def unit_members(prob, unit: str, lead: str) -> list[str]:
    return [c.name for c in prob.components if c.enabled and (c.group == unit or c.name == unit or c.tie_to in (lead, unit))]


def unit_molecule(prob, lead: str) -> str:
    c = next((c for c in prob.components if c.name == lead), None)
    return getattr(c, "molecule", "") if c is not None else ""


def unit_flux_folded(model, P: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Total model and per-unit fluxes on the fit pixels, tied units (isotopologues) added to their parent."""
    total, scaled, _ = model.evaluate(P, per_unit=True)
    tied = model.tied_units()
    out = {}
    for k, f in scaled.items():
        key = tied.get(k, k)
        out[key] = out.get(key, 0.0) + f
    return total, out


def template_bank(prob, P: dict, unit: str, lead: str, s: ShiftNullSettings) -> list[np.ndarray]:
    """Unit-u fluxes for T x t_factors and log N + logN_offsets (all members of the unit scaled together; the
    prior bounds of the lead component's T / log N clip the grid)."""
    members = [m for m in unit_members(prob, unit, lead) if m in P]
    lo = {p.key: p.lo for p in prob.free}; hi = {p.key: p.hi for p in prob.free}
    T0 = float(P[lead].get("T", np.nan)); N0 = float(P[lead].get("logN", np.nan))
    tlo, thi = lo.get(f"{lead}.T", 20.0), hi.get(f"{lead}.T", 5000.0)
    nlo, nhi = lo.get(f"{lead}.logN", 10.0), hi.get(f"{lead}.logN", 23.0)
    bank, seen = [], set()
    for f in s.t_factors:
        for d in s.logN_offsets:
            if np.isfinite(T0):
                Tn = float(np.clip(T0 * f, tlo, thi))
            else:
                Tn = None
            Nn = float(np.clip(N0 + d, nlo, nhi)) if np.isfinite(N0) else None
            key = (round(Tn, 3) if Tn is not None else None, round(Nn, 4) if Nn is not None else None)
            if key in seen:
                continue
            seen.add(key)
            P2 = {k: dict(v) for k, v in P.items()}
            for m in members:
                if Tn is not None and "T" in P2[m] and np.isfinite(T0) and T0 > 0:
                    P2[m]["T"] = float(P2[m]["T"]) * Tn / T0
                if Nn is not None and "logN" in P2[m] and np.isfinite(N0):
                    P2[m]["logN"] = float(P2[m]["logN"]) + (Nn - N0)
            try:
                _, fl = unit_flux_folded(prob.model, P2)
                t = fl.get(unit)
                if t is not None and np.any(np.isfinite(t)) and np.any(t != 0):
                    bank.append(np.nan_to_num(np.asarray(t, float)))
            except Exception:       # a grid point outside an emulator box / the model's validity: skip it
                continue
    return bank


# ------------------------------------------------------------------------------------------------ main
def shift_null(prob, theta, settings: ShiftNullSettings | None = None, target: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(table, curves) for every independent unit of the problem at `theta` (see the module docstring)."""
    s = settings or ShiftNullSettings()
    P, log_s = prob.params_from_theta(theta)
    wave = np.asarray(prob.wave, float)
    seg = segment_ids(wave)
    total, flux = unit_flux_folded(prob.model, P)
    corr = prob.continuum_solution(theta, total)[1] if getattr(prob, "cont", None) is not None and prob.cont.n else 0.0
    r = np.asarray(prob.y, float) - corr - total                    # best-fit residual, no unit signal
    s2 = (10.0 ** log_s) ** 2
    w = np.asarray(prob.weights, float) / (np.asarray(prob.sigma, float) ** 2 * s2)
    w = np.where(np.isfinite(w), w, 0.0)
    r_hp = highpass(wave, np.nan_to_num(r), s.highpass_um, seg)
    vs = np.arange(s.v_min_kms, s.v_max_kms + 1e-6, s.step_kms)
    vs = np.concatenate([-vs[::-1], vs])
    plans = [shift_plan(wave, v, seg) for v in vs]
    rows, curves = [], []
    for unit, lead in prob.all_units().items():
        tu = flux.get(unit)
        if tu is None:
            continue
        bank = template_bank(prob, P, unit, lead, s) or [tu]
        bank_hp = [highpass(wave, b, s.highpass_um, seg) for b in bank]
        tu_hp = highpass(wave, tu, s.highpass_um, seg)
        sig = r_hp + tu_hp                                           # residual + the unit's own fitted signal
        z0s = [_match(b, sig, w) for b in bank_hp]
        z0 = np.nanmax(z0s) if np.any(np.isfinite(z0s)) else np.nan
        zbest = _match(tu_hp, sig, w)
        zn = []
        e0 = [float(np.sum(w * b * b)) for b in bank_hp]
        for v, plan in zip(vs, plans):
            zz = []
            for b, eb in zip(bank_hp, e0):
                bv = apply_plan(b, plan)
                # a shift that moves most of the template out of the fitted pixels (narrow fit windows, band
                # edges) is not a fair null sample: require half of the unshifted template energy
                if eb <= 0 or float(np.sum(w[plan[3]] * bv[plan[3]] ** 2)) < s.min_energy_frac * eb:
                    zz.append(np.nan)
                else:
                    zz.append(_match(bv, r_hp, w, plan[3]))
            zv = np.nanmax(zz) if np.any(np.isfinite(zz)) else np.nan
            zn.append(zv)
            if s.save_curves:
                curves.append({"target": target, "unit": unit, "v_kms": float(v), "z": zv})
        if s.save_curves:
            curves.append({"target": target, "unit": unit, "v_kms": 0.0, "z": z0})
        zn = np.asarray(zn, float)
        zn = zn[np.isfinite(zn)]
        med = float(np.median(zn)) if len(zn) else np.nan
        mad = float(1.4826 * np.median(np.abs(zn - med))) if len(zn) else np.nan
        ok_null = len(zn) >= s.min_null and np.isfinite(mad) and mad > 0
        S = (z0 - med) / mad if ok_null else np.nan
        fap = (1.0 + float(np.sum(zn >= z0))) / (1.0 + len(zn)) if len(zn) else np.nan
        hp_frac = float(np.sqrt(np.sum(w * tu_hp ** 2) / max(np.sum(w * tu ** 2), 1e-300)))
        rows.append({"target": target, "unit": unit, "molecule": unit_molecule(prob, lead), "z0": z0,
                     "z0_bestfit": zbest, "null_median": med, "null_sigma": mad, "S": S, "fap_empirical": fap,
                     "n_null": int(len(zn)), "n_templates": len(bank_hp), "line_fraction": hp_frac,
                     "detected_shift": bool(np.isfinite(S) and S >= s.threshold),
                     "null_ok": bool(ok_null)})
    return pd.DataFrame(rows), pd.DataFrame(curves)


def save(outdir: str, table: pd.DataFrame, curves: pd.DataFrame | None = None):
    os.makedirs(outdir, exist_ok=True)
    table.to_csv(os.path.join(outdir, "shift_null.csv"), index=False)
    if curves is not None and len(curves):
        curves.to_csv(os.path.join(outdir, "shift_null_curves.csv"), index=False)


def describe(table: pd.DataFrame) -> str:
    return ", ".join(f"{r.unit}: S={r.S:.1f}" + (" *" if r.detected_shift else "") for r in table.itertuples())


# ------------------------------------------------------------------------------------------------ existing fits
def problem_from_outdir(outdir: str, backend: str | None = None, config: str | None = None):
    """(cfg, prob, theta, name) of a finished fit folder, rebuilt exactly as the pipeline had it at the end: the
    saved config.yaml (components and windows after auto-detect), the saved prepared spectrum prep.csv (with the
    final, possibly model-refined, continuum and masks) and best_fit.json (the reported parameters)."""
    import json
    from .config import ProjectConfig
    from .data import load_csv
    from .pipeline import build_problem, prepare
    cfg = ProjectConfig.load(config or os.path.join(outdir, "config.yaml"))
    if backend:
        cfg.fit.model_backend = backend
    prep = os.path.join(outdir, "prep.csv")
    spec = load_csv(prep) if os.path.exists(prep) else prepare(cfg)
    prob = build_problem(cfg, spec, backend=backend, say=lambda m: None)
    bf = os.path.join(outdir, "best_fit.json")
    if os.path.exists(bf):
        d = json.load(open(bf))
        theta = prob.theta_from_params(d["free_params"], d.get("log_s", 0.0))
    else:
        from .detection_prob import _start_theta
        theta, origin = _start_theta(outdir, prob)
        if origin == "theta0":
            raise FileNotFoundError(f"{outdir}: no best_fit.json / optimiser checkpoint")
    return cfg, prob, theta, spec.name


def run_outdir(outdir: str, settings: ShiftNullSettings | None = None, backend: str | None = None,
               config: str | None = None) -> pd.DataFrame:
    cfg, prob, theta, name = problem_from_outdir(outdir, backend=backend, config=config)
    s = settings or settings_from_config(cfg)
    table, curves = shift_null(prob, theta, s, target=name)
    save(outdir, table, curves if s.save_curves else None)
    return table


def survey_table(root: str) -> pd.DataFrame:
    """Merge <root>/*/shift_null.csv into <root>/shift_null_survey.csv."""
    import glob
    frames = []
    for f in sorted(glob.glob(os.path.join(root, "*", "shift_null.csv"))):
        try:
            t = pd.read_csv(f)
        except Exception:
            continue
        if "target" not in t or t["target"].isna().all():
            t["target"] = os.path.basename(os.path.dirname(f))
        frames.append(t)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(out):
        out.to_csv(os.path.join(root, "shift_null_survey.csv"), index=False)
    return out
