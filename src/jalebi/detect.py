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
_WATER_TEMPLATES = {"H2O_hot": [(17.5, 850.0), (18.5, 850.0)], "H2O_warm": [(17.5, 400.0), (18.5, 400.0)],
                    "H2O_cold": [(17.0, 170.0), (18.0, 170.0)]}
_TEMPLATE_LOGN = {"CO": 17.5, "13CO": 16.0, "CO2": 17.0, "13CO2": 15.5, "C2H2": 16.5, "13CCH2": 15.0,
                  "HCN": 16.5, "H13CN": 15.0, "CH4": 16.5, "NH3": 16.5, "C2H4": 16.0, "C2H6": 16.5,
                  "C4H2": 15.5, "HC3N": 15.5, "C6H6": 16.0, "C3H4": 16.0, "H2_18O": 15.5}
_TEMPLATE_T = {"CO": [700.0, 1500.0], "13CO": [700.0, 1500.0]}
_DEFAULT_T = [300.0, 600.0]
# windows where the water templates are judged: hot/warm on the 5-8 and 12-17.5 um forests,
# cold on the low-E_up lines longward of 17.5 um (MRS channel 4)
_WARM_WATER_WINDOWS = [(5.0, 8.0), (12.0, 17.5)]
_COLD_WATER_WINDOWS = [(17.5, 27.5)]
_SKIP = {"OH", "H2"}          # non-LTE prompt emission / not a slab species


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
        rows = [f"{r.candidate:9s} {'✓' if r.detected else '·'} T={r.T:5.0f} K  logR={r.logR:6.2f}  Δχ²={r.delta_chi2:8.0f}  ΔBIC={r.delta_BIC:+8.0f}"
                for r in self.table.itertuples()]
        return "\n".join(rows)


def candidate_molecules(exclude: set[str] | None = None) -> list[str]:
    """Molecules with a cached line list, minus the non-slab species."""
    from .linedata import available_linelists
    mols = sorted(set(available_linelists()["molecule"]))
    ex = _SKIP | set(exclude or ())
    return [m for m in mols if m in MOLECULES and m not in ex]


def _templates(candidates: list[str], coverage: tuple[float, float]) -> tuple[list[Component], dict[str, list[str]], dict[str, list[tuple[float, float]]]]:
    """Template components, {candidate: [template names]}, {candidate: windows}."""
    comps, by_cand, wins = [], {}, {}
    lo_c, hi_c = coverage
    # water: one candidate per temperature
    for name, tmpls in _WATER_TEMPLATES.items():
        w = _COLD_WATER_WINDOWS if name == "H2O_cold" else _WARM_WATER_WINDOWS
        w = [(a, b) for a, b in w if b > lo_c and a < hi_c]
        if not w:
            continue
        names = []
        for logN, T in tmpls:
            n = f"{name}@{logN:.1f}"
            comps.append(Component(n, "H2O", logN=logN, T=T, logR=-0.5)); names.append(n)
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


def detect_molecules(spec: Spectrum, candidates: list[str] | None = None, threshold: float = 10.0,
                     releases: dict[str, str] | None = None, oversample: int = 2, min_pixels: int = 30,
                     R_model: str = "argyriou2023", R_scale: float = 1.0, progress=None,
                     linelists: dict | None = None) -> DetectionResult:
    """Decide which molecules a continuum-subtracted spectrum contains (see module docstring).

    spec must be prepared (continuum estimated, masks applied) — e.g. `pipeline.prepare(cfg, spec)`."""
    if spec.continuum is None or not np.any(spec.continuum != 0):
        raise ValueError("detect_molecules needs a continuum-subtracted spectrum: run prepare() first")
    candidates = candidates or candidate_molecules()
    coverage = (float(np.nanmin(spec.wave)), float(np.nanmax(spec.wave)))
    comps, by_cand, cand_wins = _templates(candidates, coverage)
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
    wave, y = spec.wave[idx], spec.line_flux[idx]
    noise = estimate_noise(spec)[idx]
    sig = np.where(np.isfinite(noise) & (noise > 0), noise, np.nanmedian(spec.err[idx]))
    if progress:
        progress("building templates", 0.1)
    model = build_model(comps, wave, spec.distance_pc, windows, linelists=linelists, releases=releases,
                        oversample=oversample, R_model=R_model, R_scale=R_scale)
    P = {c.name: c.params() for c in comps}
    if progress:
        progress("solving areas", 0.4)
    logR_all, chi2_all = model.solve_areas(y, sig, P, mask=None)
    uf, _, _ = model.unit_fluxes(P)
    P_full = {k: {**v, "logR": logR_all[k]} for k, v in P.items()}
    r2_full = ((y - model.evaluate(P_full)) / sig) ** 2
    n = len(y)
    rows, comps_out = [], []
    k_free = 3.0
    ncand = len(by_cand)
    for i, (cand, names) in enumerate(by_cand.items()):
        if progress:
            progress(f"testing {cand}", 0.4 + 0.6 * i / ncand)
        # remove the candidate's templates, re-solve every other area, and compare the residuals
        # *inside the candidate's own windows*.  Real spectra have residuals well above the formal noise
        # (continuum errors, template mismatch), so Δχ² is judged against the achieved local residual
        # level (variance inflated so the full solution has χ²_red = 1 there) with the BIC penalty for
        # the local number of pixels.
        P2 = {k: dict(v) for k, v in P.items()}
        for nme in names:
            P2[nme]["logR"] = -30.0
        logR_wo, _ = model.solve_areas(y, sig, P2, mask=None, fixed=set(names))
        P_wo = {k: {**v, "logR": logR_wo[k]} for k, v in P2.items()}
        r2_wo = ((y - model.evaluate(P_wo)) / sig) ** 2
        wsel = np.zeros(n, bool)
        for a, b in cand_wins[cand]:
            wsel |= (wave >= a) & (wave <= b)
        n_loc = int(wsel.sum())
        s2 = max(float(np.sum(r2_full[wsel])) / max(n_loc - len(comps), 1), 1.0)
        dchi2 = float(np.sum(r2_wo[wsel] - r2_full[wsel])) / s2
        dbic = dchi2 - k_free * np.log(max(n_loc, 2))
        # best template = the one carrying most of the candidate's flux in the full solution
        contrib = {nme: float(np.sum(uf[nme] * 10.0 ** (2.0 * logR_all[nme]))) for nme in names}
        best = max(contrib, key=contrib.get)
        c0 = next(c for c in comps if c.name == best)
        mol = c0.molecule
        rows.append({"candidate": cand, "molecule": mol, "T": c0.T, "logN": c0.logN, "logR": float(logR_all[best]),
                     "delta_chi2": dchi2, "delta_BIC": dbic, "chi2_red_local": s2, "n_pixels": n_loc,
                     "detected": bool(dbic > threshold),
                     "windows": ", ".join(f"{a:.2f}-{b:.2f}" for a, b in cand_wins[cand])})
    tab = pd.DataFrame(rows).sort_values("delta_BIC", ascending=False).reset_index(drop=True)
    # isotopologues need their parent
    det = {r.candidate: bool(r.detected) for r in tab.itertuples()}
    for r in tab.itertuples():
        par = MOLECULES[r.molecule].parent
        if par and r.detected:
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
        logR0 = float(np.clip(r["logR"], -3, 2))
        parent = MOLECULES[mol].parent
        if cand in _WATER_TEMPLATES:
            comps_out.append(ComponentConfig(name=cand, molecule="H2O", logN=logN0, T=T0, logR=logR0,
                                             linelist_release=rel.get("H2O") or ("hitran" if cand == "H2O_cold" else "hitemp")))
        elif parent and (parent in det or parent == "H2O"):
            comps_out.append(ComponentConfig(name=mol, molecule=mol, tie_to="H2O_hot" if parent == "H2O" else parent,
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
                           chi2_all=float(chi2_all), chi2_red=float(chi2_all / max(n - len(comps), 1)))


def apply_detection(cfg: ProjectConfig, det: DetectionResult, replace_windows: bool = True,
                    keep_undetected: bool = False) -> ProjectConfig:
    """Return a copy of cfg whose components (and, optionally, windows / T ordering) follow the detection.

    keep_undetected: also keep the user's existing components whose molecule was not tested."""
    out = cfg.model_copy(deep=True)
    tested = set(det.table["molecule"]) | {"H2O"}
    keep = [c for c in cfg.components if keep_undetected and c.molecule not in tested]
    names = {c.name for c in det.components}
    out.components = det.components + [c for c in keep if c.name not in names]
    if replace_windows and det.windows:
        out.fit.windows = [list(w) for w in det.windows]
    if det.ordering:
        out.fit.ordering = det.ordering
    return out
