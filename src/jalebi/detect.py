"""Automatic molecule detection: which slab components does a spectrum need?

Rather than asking the user to list every molecule before modelling, this module decides from the
data which species are present and writes the corresponding components (and fit windows) into a
ProjectConfig.  It is what `jalebi detect`, `fit --auto-detect`, `batch --auto-detect` and the
"Detect molecules" button of the app use.

Method — simultaneous template matching by non-negative least squares
---------------------------------------------------------------------
1. Candidates: every molecule with a cached line list (or the list given), except species that no
   LTE slab describes (OH prompt emission, H2).  For each candidate a few *templates* are built: LTE
   slab spectra at fixed (log N, T) — moderate columns so the template has the molecule's band shape
   — evaluated only in the molecule's characteristic windows (`DEFAULT_WINDOWS`).
2. Water is always included, as hot / warm / cold templates, because water lines underlie every MIRI
   window and must be absorbed before anything else is judged.  Each water temperature is treated as a
   candidate of its own, so a cold-water component is suggested only if the data ask for it.
3. One NNLS solve gives the emitting area of every template on the continuum-subtracted, masked
   spectrum over the union of the candidate windows (`SlabModel.solve_areas`).
4. Significance: for each candidate all its templates are removed and the areas of the others are
   re-solved; the χ² increase Δχ² is what the candidate explains that nothing else can.  With the three
   parameters a real component would add, ΔBIC = Δχ² − 3 ln n_pix; detected if ΔBIC > `threshold`
   (default 10, the conventional "strong evidence" level, as in the MINDS detection tests).  Δχ² is
   measured against the achieved residual level (variance inflated so the full template solution has
   χ²_red = 1), so continuum errors and template mismatch on real spectra do not flag everything.
5. Isotopologues count as detected only if their parent is; they are suggested as tied components
   with the default abundance ratio.  Starting values for detected components come from the
   best-fitting template (T, log N) and its NNLS area (log R).
6. Absorption (`mode="both"`, the default, or `"absorption"`): every candidate also gets *screen*
   templates, F_c (e^-tau - 1) at fixed (log N, T) with covering fraction 1 (`kind: absorption`).  The
   model is linear in the covering fraction, so the screens join the same NNLS solve with fc as the
   coefficient, and a candidate `<mol>_abs` is judged exactly like an emission candidate (ΔBIC with the
   four parameters a screen would add).  Detected screens are suggested as `kind: absorption`
   components with fc from the solve.  Emission and absorption of the same molecule can both be
   detected (a warm inner disk behind a cold envelope).  The continuum must then be the unabsorbed one
   (upper envelope / given), see docs/ABSORPTION.md.

The result is a table (one row per candidate) and a list of ComponentConfig suggestions;
`apply_detection` writes them, together with matching fit windows and a hot>warm>cold water prior,
into a config.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import ComponentConfig, ProjectConfig
from .data import Spectrum, estimate_noise
from .model import Component, build_model, merge_intervals
from .molecules import DEFAULT_WINDOWS, MOLECULES

# Template columns (log N [cm^-2]) and temperatures [K] per molecule.  Columns are moderate so the
# template shape is the molecule's band shape (not saturated), the NNLS area supplies the scale.
# water candidates: each is a pair of templates (thin-ish and thick column) at the same T, so the NNLS
# can reproduce the line-saturation pattern of the real column and leaves less residual for others
_WATER_TEMPLATES = {"H2O_rovib": [(17.5, 950.0), (18.5, 950.0)],
                    "H2O_hot": [(17.5, 850.0), (18.5, 850.0)], "H2O_warm": [(17.5, 400.0), (18.5, 400.0)],
                    "H2O_cold": [(17.0, 170.0), (18.0, 170.0)]}
# the ro-vibrational nu2 band (5-8 um) is sub-thermal: it gets its own component restricted to these windows
_ROVIB_WATER_WINDOWS = [(5.0, 8.0)]
_ROVIB_COMPONENT_WINDOWS = [[4.9, 9.5]]
_ROT_COMPONENT_WINDOWS = [[9.5, 28.5]]
_TEMPLATE_LOGN = {"CO": 17.5, "13CO": 16.0, "CO2": 17.0, "13CO2": 15.5, "C2H2": 16.5, "13CCH2": 15.0,
                  "HCN": 16.5, "H13CN": 15.0, "CH4": 16.5, "NH3": 16.5, "C2H4": 16.0, "C2H6": 16.5,
                  "C4H2": 15.5, "HC3N": 15.5, "C6H6": 16.0, "C3H4": 16.0, "H2_18O": 15.5}
_TEMPLATE_T = {"CO": [700.0, 1500.0], "13CO": [700.0, 1500.0]}
_DEFAULT_T = [300.0, 600.0]
# windows where the water templates are judged: hot/warm on the 5-8 and 12-17.5 um forests,
# cold on the low-E_up lines longward of 17.5 um (MRS channel 4)
# hot/warm are judged on the whole rotational range: the 17.5-27.5 um lines (E_up 1500-4000 K) are what
# separates a warm from a hot slab, and they must enter the fit windows even when no cold slab is detected
_WARM_WATER_WINDOWS = [(12.0, 17.5), (17.5, 27.5)]
_COLD_WATER_WINDOWS = [(17.5, 27.5)]
_SKIP = {"OH", "H2"}          # non-LTE prompt emission / not a slab species
# absorption screens: colder gas, higher columns (embedded protostars: Lahuis & van Dishoeck 2000)
_ABS_LOGN = {"CO": 18.0, "13CO": 16.5, "CO2": 17.5, "13CO2": 16.0, "H2O": 18.0, "C2H2": 17.0, "HCN": 17.0,
             "CH4": 17.0, "NH3": 17.0, "C2H4": 16.5, "C2H6": 17.0, "C4H2": 16.0, "HC3N": 16.0}
# screen temperatures: the first is the template that enters the joint solve (one screen per candidate:
# the model is linear in fc for one screen only, and two screens of the same molecule would double-count
# saturated lines); the others are tried afterwards, one at a time, to pick the temperature
_ABS_T = {"CO": [200.0, 50.0, 500.0], "13CO": [200.0, 50.0, 500.0], "H2O": [250.0, 100.0, 400.0]}
_ABS_DEFAULT_T = [150.0, 50.0, 300.0]
_ABS_WINDOWS = {"H2O": [(5.0, 8.0), (12.0, 17.5)]}       # otherwise the molecule's DEFAULT_WINDOWS
MODES = ("emission", "absorption", "both")


@dataclass
class DetectionResult:
    table: pd.DataFrame                       # one row per candidate
    components: list[ComponentConfig]         # suggested components (detected only)
    windows: list[tuple[float, float]]        # suggested fit windows
    ordering: list[list[str]] = field(default_factory=list)
    n_pixels: int = 0
    chi2_all: float = np.nan
    chi2_red: float = np.nan                  # of the full template solution; Δχ² are scaled by max(chi2_red, 1)

    def summary(self) -> str:
        rows = [f"{r.candidate:12s} {'✓' if r.detected else '·'} T={r.T:5.0f} K  "
                + (f"fc={r.fc:5.2f}  " if getattr(r, "kind", "slab") == "absorption" else f"logR={r.logR:6.2f}")
                + f"  Δχ²={r.delta_chi2:8.0f}  ΔBIC={r.delta_BIC:+8.0f}"
                for r in self.table.itertuples()]
        return "\n".join(rows)


def candidate_molecules(exclude: set[str] | None = None) -> list[str]:
    """Molecules with a cached line list, minus the non-slab species."""
    from .linedata import available_linelists
    mols = sorted(set(available_linelists()["molecule"]))
    ex = _SKIP | set(exclude or ())
    return [m for m in mols if m in MOLECULES and m not in ex]


def _templates(candidates: list[str], coverage: tuple[float, float], mode: str = "both") -> tuple[list[Component], dict[str, list[str]], dict[str, list[tuple[float, float]]]]:
    """Template components, {candidate: [template names]}, {candidate: windows}.  Absorption
    candidates are named '<molecule>_abs' and use kind="absorption" templates with fc = 1."""
    comps, by_cand, wins = [], {}, {}
    lo_c, hi_c = coverage
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if mode in ("absorption", "both"):
        for m in candidates:
            w = [(a, b) for a, b in _ABS_WINDOWS.get(m, DEFAULT_WINDOWS.get(m, [])) if b > lo_c and a < hi_c]
            if not w:
                continue
            names = []
            for j, T in enumerate(_ABS_T.get(m, _ABS_DEFAULT_T)):
                n = f"{m}_abs@{T:.0f}"
                # only the first template is free in the joint solve (fc = 1 column); the others start
                # at fc = 0 and are tried one at a time for the temperature choice
                comps.append(Component(n, m, logN=_ABS_LOGN.get(m, 17.0), T=T, kind="absorption", fc=1.0 if j == 0 else 0.0))
                names.append(n)
            by_cand[f"{m}_abs"] = names; wins[f"{m}_abs"] = w
    if mode == "absorption":
        return comps, by_cand, wins
    # water: one candidate per temperature
    for name, tmpls in _WATER_TEMPLATES.items():
        w = _COLD_WATER_WINDOWS if name == "H2O_cold" else _ROVIB_WATER_WINDOWS if name == "H2O_rovib" else _WARM_WATER_WINDOWS
        w = [(a, b) for a, b in w if b > lo_c and a < hi_c]
        if not w:
            continue
        names = []
        for logN, T in tmpls:
            n = f"{name}@{logN:.1f}"
            # templates emit only on their own side of 9.5 um, so the rotational candidates cannot stand in
            # for the (sub-thermal) ro-vibrational band and vice versa
            comps.append(Component(n, "H2O", logN=logN, T=T, logR=-0.5,
                                   windows=_ROVIB_COMPONENT_WINDOWS if name == "H2O_rovib" else _ROT_COMPONENT_WINDOWS))
            names.append(n)
        by_cand[name] = names; wins[name] = w
    for m in candidates:
        if m == "H2O":
            continue
        w = [(a, b) for a, b in DEFAULT_WINDOWS.get(m, []) if b > lo_c and a < hi_c]
        if not w:
            continue
        names = []
        for T in _TEMPLATE_T.get(m, _DEFAULT_T):
            n = f"{m}@{T:.0f}"
            comps.append(Component(n, m, logN=_TEMPLATE_LOGN.get(m, 16.5), T=T, logR=-0.5))
            names.append(n)
        by_cand[m] = names; wins[m] = w
    return comps, by_cand, wins


def _detect_pass(spec: Spectrum, idx: np.ndarray, comps: list[Component], by_cand: dict, cand_wins: dict,
                 windows, threshold: float, releases, oversample, R_model, R_scale, progress, linelists,
                 label: str = "") -> tuple[pd.DataFrame, float, int]:
    """One template solve + one removal test per candidate.  Returns (table, chi2_all, n_pixels)."""
    wave, y = spec.wave[idx], spec.line_flux[idx]
    noise = estimate_noise(spec)[idx]
    sig = np.where(np.isfinite(noise) & (noise > 0), noise, np.nanmedian(spec.err[idx]))
    is_abs = {c.name for c in comps if c.kind == "absorption"}
    alt_screens = {n for names in by_cand.values() for n in names[1:] if n in is_abs}   # fixed at fc = 0 in every solve
    if progress:
        progress(f"building templates{label}", 0.1)
    model = build_model(comps, wave, spec.distance_pc, windows, linelists=linelists, releases=releases,
                        oversample=oversample, R_model=R_model, R_scale=R_scale, continuum=spec.continuum[idx])
    P = {c.name: c.params() for c in comps}
    if progress:
        progress(f"solving areas{label}", 0.4)

    unit_of = {c.name: (c.group or c.name) for c in comps}

    def _apply(P0, logR, fc):
        out = {}
        for k, v in P0.items():
            u = unit_of[k]
            out[k] = {**v, "logR": logR.get(u, v.get("logR", 0.0))}
            if u in fc:
                out[k]["fc"] = float(np.clip(fc[u], 0.0, 1.0))
        return out

    logR_all, fc_all, chi2_all = model.solve_linear(y, sig, P, mask=None, fixed=alt_screens, solve_fc=True)
    P_full = _apply(P, logR_all, fc_all)
    uf, _, _ = model.unit_fluxes(P_full)
    r2_full = ((y - model.evaluate(P_full)) / sig) ** 2
    n = len(y)
    rows = []
    ncand = len(by_cand)
    for i, (cand, names) in enumerate(by_cand.items()):
        if progress:
            progress(f"testing {cand}{label}", 0.4 + 0.6 * i / ncand)
        # remove the candidate's templates, re-solve every other area, and compare the residuals
        # *inside the candidate's own windows*.  Real spectra have residuals well above the formal noise
        # (continuum errors, template mismatch), so Δχ² is judged against the achieved local residual
        # level (variance inflated so the full solution has χ²_red = 1 there) with the BIC penalty for
        # the local number of pixels.
        P2 = {k: dict(v) for k, v in P.items()}
        absorber = names[0] in is_abs
        k_free = 4.0 if absorber else 3.0
        for nme in names:
            if absorber:
                P2[nme]["fc"] = 0.0
            else:
                P2[nme]["logR"] = -30.0
        units_c = {unit_of[nme] for nme in names}
        logR_wo, fc_wo, _ = model.solve_linear(y, sig, P2, mask=None, fixed=units_c | alt_screens, solve_fc=True)
        P_wo = _apply(P2, logR_wo, fc_wo)
        r2_wo = ((y - model.evaluate(P_wo)) / sig) ** 2
        wsel = np.zeros(n, bool)
        for a, b in cand_wins[cand]:
            wsel |= (wave >= a) & (wave <= b)
        if absorber:
            # a screen is judged only where the spectrum is below the continuum: it must explain real
            # troughs, not the mismatch of the emission templates between emission lines
            wsel &= y < 0
        n_loc = int(wsel.sum())
        s2 = max(float(np.sum(r2_full[wsel])) / max(n_loc - len(comps), 1), 1.0)
        dchi2 = float(np.sum(r2_wo[wsel] - r2_full[wsel])) / s2
        dbic = dchi2 - k_free * np.log(max(n_loc, 2))
        if absorber:
            # best temperature: each screen template alone (others at fc = 0) against the residual
            # without the candidate, fc solved in one bounded 1-D fit on the candidate's pixels
            resid = y - model.evaluate(P_wo)
            best, best_chi2, best_fc = names[0], np.inf, 0.0
            for nme in names:
                P1 = {k: dict(v) for k, v in P_wo.items()}
                for other in names:
                    P1[other]["fc"] = 0.0
                P1[nme]["fc"] = 1.0
                col = model.screen_columns(P1).get(nme)
                if col is None or not wsel.any():
                    continue
                a = col[wsel] / sig[wsel]; b = resid[wsel] / sig[wsel]
                fc1 = float(np.clip((a @ b) / max(a @ a, 1e-300), 0.0, 1.0))
                c2 = float(np.sum((b - fc1 * a) ** 2))
                if c2 < best_chi2:
                    best, best_chi2, best_fc = nme, c2, fc1
            fc_best = float(np.clip(fc_all.get(names[0], best_fc) if best == names[0] else best_fc, 0.0, 1.0))
        else:
            # best template = the one carrying most of the candidate's flux in the full solution
            contrib = {nme: abs(float(np.sum(uf[nme] * 10.0 ** (2.0 * logR_all[nme])))) for nme in names}
            best = max(contrib, key=contrib.get)
            fc_best = np.nan
        c0 = next(c for c in comps if c.name == best)
        mol = c0.molecule
        rows.append({"candidate": cand, "molecule": mol, "kind": "absorption" if absorber else "slab",
                     "T": c0.T, "logN": c0.logN, "logR": float(logR_all[best]) if not absorber else np.nan,
                     "fc": fc_best,
                     "delta_chi2": dchi2, "delta_BIC": dbic, "chi2_red_local": s2, "n_pixels": n_loc,
                     "detected": bool(dbic > threshold),
                     "windows": ", ".join(f"{a:.2f}-{b:.2f}" for a, b in cand_wins[cand])})
    tab = pd.DataFrame(rows).sort_values("delta_BIC", ascending=False).reset_index(drop=True)
    return tab, float(chi2_all), n


def detect_molecules(spec: Spectrum, candidates: list[str] | None = None, threshold: float = 10.0,
                     releases: dict[str, str] | None = None, oversample: int = 2, min_pixels: int = 30,
                     R_model: str = "argyriou2023", R_scale: float = 1.0, progress=None,
                     linelists: dict | None = None, mode: str = "both") -> DetectionResult:
    """Decide which molecules a continuum-subtracted spectrum contains (see module docstring).

    spec must be prepared (continuum estimated, masks applied) — e.g. `pipeline.prepare(cfg, spec)`.
    mode: "emission", "absorption" or "both".  With "both": an emission-only pass; a joint pass in which
    only the screens are judged, and only on pixels below the continuum (an emitting slab and a screen
    of the same molecule are nearly anti-collinear, so a screen must explain real troughs); and, when
    screens are found, a combined pass of every emission candidate with the detected screens for the
    final verdict on the emission (a band hidden under absorption is only found there)."""
    if spec.continuum is None or not np.any(spec.continuum != 0):
        raise ValueError("detect_molecules needs a continuum-subtracted spectrum: run prepare() first")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    candidates = candidates or candidate_molecules()
    coverage = (float(np.nanmin(spec.wave)), float(np.nanmax(spec.wave)))

    def _pass(pmode, only: set[str] | None = None, label="", judge: set[str] | None = None):
        """One pass over the templates of `pmode`, restricted to the candidates in `only`; `judge`
        keeps every template in the solve but tests only those candidates."""
        comps, by_cand, cand_wins = _templates(candidates, coverage, pmode)
        if only is not None:
            by_cand = {k: v for k, v in by_cand.items() if k in only}
            cand_wins = {k: v for k, v in cand_wins.items() if k in only}
            keep = {n for v in by_cand.values() for n in v}
            comps = [c for c in comps if c.name in keep]
        if judge is not None:
            by_cand = {k: v for k, v in by_cand.items() if k in judge}
            cand_wins = {k: v for k, v in cand_wins.items() if k in judge}
        if not comps:
            raise ValueError("no candidate molecules with cached line lists cover this spectrum")
        windows = merge_intervals([w for ws in cand_wins.values() for w in ws])
        sel = np.zeros(len(spec.wave), bool)
        for a, b in windows:
            sel |= (spec.wave >= a) & (spec.wave <= b)
        sel &= spec.mask & np.isfinite(spec.line_flux)
        idx = np.flatnonzero(sel)
        if len(idx) < min_pixels:
            raise ValueError(f"only {len(idx)} usable pixels in the detection windows")
        tab, chi2_all, n = _detect_pass(spec, idx, comps, by_cand, cand_wins, windows, threshold, releases,
                                        oversample, R_model, R_scale, progress, linelists, label)
        return tab, chi2_all, n, len(comps), cand_wins

    if mode == "both":
        # 1. emission alone; 2. every template together, but only the screens judged (a screen must
        # explain troughs below the continuum that the emission templates leave); 3. if screens are
        # found, every emission candidate together with the detected screens gives the final verdict
        # on the emission (a band hidden under absorption is only found there).  The screens of an
        # undetected absorber never enter the solution that the emission suggestions come from.
        tab_e, chi2_all, n, ncomp, wins_e = _pass("emission", label=" (emission)")
        _, all_cands, wins_all = _templates(candidates, coverage, "both")
        abs_cands = {k for k in all_cands if k.endswith("_abs")}
        tab_a, chi2_a, n_a, ncomp_a, wins_a = _pass("both", label=" (absorption)", judge=abs_cands)
        cand_wins = {**wins_e, **wins_all}
        det_a = set(tab_a.candidate[tab_a.detected])
        tab = pd.concat([tab_e, tab_a], ignore_index=True)
        if det_a:
            tab_c, chi2_all, n, ncomp, _ = _pass("both", only=set(tab_e.candidate) | det_a, label=" (combined)")
            tab = tab[~tab.candidate.isin(set(tab_c.candidate))]
            tab = pd.concat([tab_c, tab], ignore_index=True)
        tab = tab.sort_values("delta_BIC", ascending=False).reset_index(drop=True)
    else:
        tab, chi2_all, n, ncomp, cand_wins = _pass(mode)
    comps_out = []
    # isotopologues need their parent
    det = {r.candidate: bool(r.detected) for r in tab.itertuples()}
    for r in tab.itertuples():
        par = MOLECULES[r.molecule].parent
        if par and r.detected:
            if r.kind == "absorption":
                par_ok = det.get(f"{par}_abs", False)
            else:
                par_ok = det.get(par, False) or (par == "H2O" and any(det.get(k, False) for k in _WATER_TEMPLATES))
            if not par_ok:
                tab.loc[r.Index, "detected"] = False
                det[r.candidate] = False
    # suggested components (config order: water first, then by strength)
    order = [k for k in _WATER_TEMPLATES if det.get(k)] + [r.candidate for r in tab.itertuples() if r.detected and r.candidate not in _WATER_TEMPLATES]
    rel = releases or {}
    for cand in order:
        r = tab[tab.candidate == cand].iloc[0]          # a Series: use item access (r.T would be the transpose!)
        mol, T0, logN0 = str(r["molecule"]), float(r["T"]), float(r["logN"])
        parent = MOLECULES[mol].parent
        if r["kind"] == "absorption":
            fc0 = float(np.clip(r["fc"], 0.02, 1.0)) if np.isfinite(r["fc"]) else 0.5
            if parent and det.get(f"{parent}_abs"):
                comps_out.append(ComponentConfig(name=cand, molecule=mol, kind="absorption", tie_to=f"{parent}_abs",
                                                 ratio=MOLECULES[mol].default_ratio or 70.0, logN=logN0, T=T0, fc=fc0))
            else:
                comps_out.append(ComponentConfig(name=cand, molecule=mol, kind="absorption", logN=logN0, T=T0, fc=fc0))
            continue
        logR0 = float(np.clip(r["logR"], -3, 2))
        if cand in _WATER_TEMPLATES:
            # a detected ro-vibrational component emits only at 4.9-9.5 um and the rotational ones beyond
            rovib = cand == "H2O_rovib"
            wins_c = _ROVIB_COMPONENT_WINDOWS if rovib else (_ROT_COMPONENT_WINDOWS if det.get("H2O_rovib") else None)
            comps_out.append(ComponentConfig(name=cand, molecule="H2O", logN=logN0, T=T0, logR=logR0, windows=wins_c,
                                             linelist_release=rel.get("H2O") or ("hitran" if cand == "H2O_cold" else "hitemp")))
        elif parent and (parent in det or parent == "H2O"):
            h2o_parent = next((k for k in ("H2O_hot", "H2O_warm", "H2O_cold", "H2O_rovib") if det.get(k)), "H2O_hot")
            comps_out.append(ComponentConfig(name=mol, molecule=mol, tie_to=h2o_parent if parent == "H2O" else parent,
                                             ratio=MOLECULES[mol].default_ratio or 70.0, logN=logN0, T=T0, logR=logR0))
        else:
            comps_out.append(ComponentConfig(name=mol, molecule=mol, logN=logN0, T=T0, logR=logR0))
    # windows: union of the detected molecules' windows, clipped to coverage
    ws = [w for cand in order for w in cand_wins[cand]]
    ws = [(max(a, coverage[0]), min(b, coverage[1])) for a, b in merge_intervals(ws)] if ws else []
    ordering = []
    water = [k for k in ("H2O_hot", "H2O_warm", "H2O_cold") if det.get(k)]
    for a, b in zip(water[:-1], water[1:]):
        ordering.append([a, b])
    return DetectionResult(table=tab, components=comps_out, windows=ws, ordering=ordering, n_pixels=n,
                           chi2_all=float(chi2_all), chi2_red=float(chi2_all / max(n - ncomp, 1)))


def apply_detection(cfg: ProjectConfig, det: DetectionResult, replace_windows: bool = True,
                    keep_undetected: bool = False) -> ProjectConfig:
    """Return a copy of cfg whose components (and, optionally, windows / T ordering) follow the detection.

    keep_undetected: also keep the user's existing components whose molecule was not tested."""
    out = cfg.model_copy(deep=True)
    tested = set(det.table["molecule"]) | {"H2O"}
    kinds = set(det.table["kind"]) if "kind" in det.table else {"slab"}
    keep = [c for c in cfg.components if keep_undetected and (c.molecule not in tested or
            ("absorption" if c.kind == "absorption" else "slab") not in kinds)]
    names = {c.name for c in det.components}
    out.components = det.components + [c for c in keep if c.name not in names]
    if replace_windows and det.windows:
        out.fit.windows = [list(w) for w in det.windows]
    if det.ordering:
        out.fit.ordering = det.ordering
    return out
