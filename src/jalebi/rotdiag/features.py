"""Which lines go into a rotation diagram: selection from the line list, blends, contaminants.

A *feature* is what one measures in the spectrum: one line, or several lines of the molecule that fall within
a fraction of a resolution element of each other (Λ-doublets and hyperfine components of OH, ortho/para water
pairs, overlapping CO v=1-0 / 2-1 lines ...).  Its flux is the sum of its member lines, so the model predicts
Σ_i F_i over the members and the diagram point uses

    N_u/g_u = 4π F / (h ν Ω Σ_i g_i A_i),   E_u = Σ g_i A_i E_i / Σ g_i A_i.

Selection: lines inside the spectral coverage (away from sub-band edges and masked pixels), in the chosen
vibrational bands, below E_u,max, ranked by the optically thin LTE intensity at t_ref,
I ∝ g A ν exp(−E_u/k t_ref); features weaker than rel_min × the strongest are dropped.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..lines import C_KMS, LINES, lsf_fwhm_kms
from .species import DATA_FILES, RotSpecies, load_species_linelist, partition, preset, spin_species, transition_label


@dataclass
class Selection:
    wmin: float | None = None
    wmax: float | None = None
    bands: list[str] | None = None          # vibrational bands 'vup-vlow' (None = preset default, [] = all)
    eu_max: float | None = None             # K
    eu_min: float | None = None
    t_ref: float | None = None              # K (None = preset)
    rel_min: float | None = None            # relative to the strongest feature at t_ref (None = the preset's)
    max_features: int = 60
    blend_fwhm: float = 0.5                 # lines closer than this x the instrumental FWHM form one feature
    member_rel: float = 0.01                # weaker lines join a feature's sum if >= this x its strongest line
    edge_fwhm: float = 3.0                  # keep features this many FWHM inside a sub-band
    curated: bool | None = None             # use the preset's curated list (H2O: Banzatti et al. 2025); None = preset
    include: list[str] = field(default_factory=list)   # labels always kept (e.g. "S(0)")
    exclude: list[str] = field(default_factory=list)   # labels dropped
    R_model: str | float = "argyriou2023"  # MIRI MRS R(λ); a number = constant R (e.g. NIRSpec G395H ~2700)


def fwhm_kms(wave_um, R_model="argyriou2023"):
    if isinstance(R_model, (int, float)) and not isinstance(R_model, bool):
        return np.full_like(np.asarray(wave_um, float), C_KMS / float(R_model))
    try:
        return lsf_fwhm_kms(wave_um, str(R_model))
    except KeyError:
        return lsf_fwhm_kms(wave_um)


def coverage(spec, edge_fwhm: float = 3.0, R_model="argyriou2023") -> list[tuple[float, float, str]]:
    """(lo, hi, band) wavelength intervals with usable pixels, trimmed by edge_fwhm resolution elements."""
    out = []
    bands = spec.bands or [""]
    for b in bands:
        i = spec.band_slice(b) if b else np.arange(len(spec.wave))
        ok = i[np.isfinite(spec.flux[i]) & spec.mask[i]]
        if len(ok) < 5:
            continue
        w = spec.wave[ok]
        lo, hi = w.min(), w.max()
        d = edge_fwhm * fwhm_kms(np.array([lo, hi]), R_model) / C_KMS * np.array([lo, hi])
        if hi - d[1] > lo + d[0]:
            out.append((lo + d[0], hi - d[1], b))
    return out


def vib_bands(molecule: str, t: pd.DataFrame) -> np.ndarray:
    """'vup-vlow' for every line (vectorised version of species.vib_band)."""
    if molecule == "OH":
        vu = t["vup"].astype(str).str.extract(r"(\d+)\s*$")[0].fillna("?")
        vl = t["vlow"].astype(str).str.extract(r"(\d+)\s*$")[0].fillna("?")
        return (vu + "-" + vl).to_numpy()
    vu = t["vup"].astype(str).str.replace(r"\s+", "", regex=True)
    vl = t["vlow"].astype(str).str.replace(r"\s+", "", regex=True)
    return (vu + "-" + vl).to_numpy()


def _in_coverage(w, cov):
    m = np.zeros(len(w), bool)
    for lo, hi, _ in cov:
        m |= (w >= lo) & (w <= hi)
    return m


def _contaminant_catalogue(molecule: str) -> pd.DataFrame:
    """Lines of *other* species that can blend: the jalebi line catalogue (fine-structure, H I, H2) and the
    Banzatti et al. (2025) MIRI list (H2O, OH, CO, H2, H I)."""
    rows = [(ln.wave, ln.name, ln.species) for ln in LINES.values()]
    try:
        t = pd.read_csv(os.path.join(DATA_FILES, "MIRI_general_Banzatti+2025.csv"))
        rows += [(r.lam, f"{r.species} {r.lam:.4f}", r.species) for r in t.itertuples()]
    except OSError:
        pass
    df = pd.DataFrame(rows, columns=["wave", "name", "species"])
    norm = {"H I": "HI"}
    df["sp"] = df["species"].replace(norm).str.replace(r"[\[\]\s]", "", regex=True)
    return df[df["sp"] != molecule.replace(" ", "")].reset_index(drop=True)


def _curated_table(sp: RotSpecies) -> pd.DataFrame | None:
    if not sp.curated:
        return None
    try:
        return pd.read_csv(os.path.join(DATA_FILES, sp.curated))
    except OSError:
        return None


def find_features(molecule: str, spec=None, sel: Selection | None = None, release: str | None = None,
                  linelist=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select the features of `molecule` for a rotation diagram.

    Returns (features, members): one row per feature (id, label, wave, E_u, gA sum, spin, ladder, band, flags ...)
    and one row per member line (feature id, wave, A, g_u, E_u, label ...).  Without a spectrum, every line
    in [wmin, wmax] is a candidate (e.g. to fit a table of published fluxes)."""
    sel = sel or Selection()
    sp = preset(molecule)
    ll = linelist if linelist is not None else load_species_linelist(sp.name, release)
    P = partition(sp, ll)
    t = ll.table
    t_ref = sel.t_ref or sp.t_ref
    bands = sp.bands if sel.bands is None else tuple(sel.bands)
    wmin = sel.wmin if sel.wmin is not None else (float(np.nanmin(spec.wave)) if spec is not None else 0.0)
    wmax = sel.wmax if sel.wmax is not None else (float(np.nanmax(spec.wave)) if spec is not None else 1e9)
    m = (t["wave"].to_numpy() >= wmin) & (t["wave"].to_numpy() <= wmax) & (t["a"].to_numpy() > 0)
    if sel.eu_max is not None:
        m &= t["eu"].to_numpy() <= sel.eu_max
    if sel.eu_min is not None:
        m &= t["eu"].to_numpy() >= sel.eu_min
    cand = t[m].reset_index(drop=True)
    if len(cand) == 0:
        return _empty()
    vb = vib_bands(sp.name, cand)
    cov = coverage(spec, sel.edge_fwhm, sel.R_model) if spec is not None else None
    if cov is not None:
        keep = _in_coverage(cand["wave"].to_numpy(), cov)
        cand, vb = cand[keep].reset_index(drop=True), vb[keep]
    # every line of the molecule in the coverage (all bands) takes part in the blends; the selected bands decide
    # which blends become features (by their strongest line)
    I = (cand["gu"] * cand["a"] / cand["wave"] * np.exp(-cand["eu"] / t_ref)).to_numpy() / P.Q(t_ref)
    insel = np.isin(vb, list(bands)) if bands else np.ones(len(cand), bool)
    if not insel.any():
        return _empty()
    rel_min = sp.rel_min if sel.rel_min is None else sel.rel_min
    keep = I >= 1e-3 * rel_min * I[insel].max()  # thin LTE photon-energy-weighted intensity per column at t_ref
    cand, vb, I, insel = cand[keep].reset_index(drop=True), vb[keep], I[keep], insel[keep]
    Isel = float(I[insel].max())
    w = cand["wave"].to_numpy(); A = cand["a"].to_numpy(); g = cand["gu"].to_numpy(); E = cand["eu"].to_numpy()
    spin = spin_species(sp, cand)
    lad = vb if sp.name != "OH" else np.where(cand["vup"].astype(str).str.contains("3/2"), "Π3/2",
                                               np.where(cand["vup"].astype(str).str.contains("1/2"), "Π1/2", vb))
    if sp.name == "H2":
        lad = np.where(spin == "o", "ortho", np.where(spin == "p", "para", lad))
    _lab_cache: dict[int, str] = {}

    def label(i):
        if i not in _lab_cache:
            _lab_cache[i] = transition_label(sp.name, cand.iloc[i])
        return _lab_cache[i]

    fw_um = fwhm_kms(w, sel.R_model) / C_KMS * w
    curated = _curated_table(sp) if (sel.curated if sel.curated is not None else bool(sp.curated)) else None

    groups: list[list[int]] = []
    windows: list[tuple[float, float] | None] = []
    if curated is not None:
        for r in curated.itertuples():
            fw0 = float(np.interp(r.lam, w, fw_um))
            lo = getattr(r, "xmin", r.lam - 0.5 * fw0)
            hi = getattr(r, "xmax", r.lam + 0.5 * fw0)
            idx = np.flatnonzero((w >= lo) & (w <= hi))          # members: every line inside the listed window
            if len(idx) == 0 or np.min(np.abs(w[idx] - r.lam)) > 0.25 * fw0:
                continue
            groups.append(list(idx)); windows.append((float(lo), float(hi)))
    else:
        strong = I >= 0.1 * rel_min * Isel
        order = np.flatnonzero(strong)[np.argsort(w[strong])]
        cur = [order[0]] if len(order) else []
        for a_, b_ in zip(order[:-1], order[1:]):
            if w[b_] - w[a_] <= sel.blend_fwhm * 0.5 * (fw_um[a_] + fw_um[b_]):
                cur.append(b_)
            else:
                groups.append(cur); cur = [b_]
        if cur:
            groups.append(cur)
        windows = [None] * len(groups)
        weak = np.flatnonzero(~strong)                          # weaker lines inside a core join its sum
        if len(weak) and groups:
            centres = np.array([w[gi][np.argmax(I[gi])] for gi in groups])
            for j in weak:
                k = int(np.argmin(np.abs(centres - w[j])))
                if abs(centres[k] - w[j]) <= 0.5 * sel.blend_fwhm * fw_um[j]:
                    groups[k].append(j)
    groups = [[i for i in gi if I[i] >= sel.member_rel * I[gi].max()] for gi in groups]
    tops = [gi[int(np.argmax(I[gi]))] for gi in groups]
    is_feat = np.array([bool(insel[t_]) for t_ in tops]) if curated is None else np.ones(len(groups), bool)
    strength = np.array([I[gi].sum() for gi in groups])
    rows, mrows = [], []
    for k, gi in enumerate(groups):
        if not is_feat[k]:
            continue
        gi = np.array(gi)
        gA = g[gi] * A[gi]
        top = tops[k]
        wc = float(np.sum(I[gi] * w[gi]) / np.sum(I[gi]))
        sp_set = set(spin[gi]) - {""}
        rows.append(dict(label=label(top) + (f" +{len(gi) - 1}" if len(gi) > 1 else ""), top=label(top), wave=wc,
                         eu=float(np.sum(gA * E[gi]) / gA.sum()), gA=float(gA.sum()), a=float(A[top]), gu=float(g[top]),
                         n_members=len(gi), strength=float(strength[k]),
                         spin="".join(sorted(sp_set)) if len(sp_set) <= 1 else "op",
                         ladder=lad[top], band_v=vb[top], eu_spread=float(np.ptp(E[gi])),
                         xmin=np.nan if windows[k] is None else windows[k][0],
                         xmax=np.nan if windows[k] is None else windows[k][1], _k=k))
    if not rows:
        return _empty()
    F = pd.DataFrame(rows)
    rel = F["strength"] / F["strength"].max()
    keep = rel >= rel_min
    if sel.include:
        keep |= F["top"].isin(sel.include) | F["label"].isin(sel.include)
    if sel.exclude:
        keep &= ~(F["top"].isin(sel.exclude) | F["label"].isin(sel.exclude))
    F = F[keep]
    if len(F) > sel.max_features:
        forced = F["top"].isin(sel.include)
        F = pd.concat([F[forced], F[~forced].nlargest(sel.max_features - int(forced.sum()), "strength")])
    F = F.sort_values("wave").reset_index(drop=True)
    F.insert(0, "id", np.arange(len(F)))
    F["rel_strength"] = F["strength"] / F["strength"].max()
    chosen = set(int(k) for k in F["_k"])
    for k_old, fid in zip(F["_k"], F["id"]):
        for i in groups[k_old]:
            mrows.append(dict(feature=int(fid), wave=float(w[i]), a=float(A[i]), gu=float(g[i]), eu=float(E[i]),
                              el=float(cand["el"].iloc[i]), label=label(i), spin=spin[i], ladder=lad[i], band_v=vb[i],
                              rel=float(I[i] / I[groups[k_old]].max())))
    F = F.drop(columns=["_k"])
    M = pd.DataFrame(mrows)
    # blends with other features of the molecule (within 1.5 FWHM) and with other species (within 1 FWHM)
    fw_f = fwhm_kms(F["wave"].to_numpy(), sel.R_model) / C_KMS * F["wave"].to_numpy()
    wf = F["wave"].to_numpy()
    near = [", ".join(F["top"].iloc[j] for j in np.flatnonzero((np.abs(wf - wf[k]) < 1.5 * fw_f[k]) & (np.arange(len(F)) != k)))
            for k in range(len(F))]
    cat = _contaminant_catalogue(sp.name)
    cw = cat["wave"].to_numpy()
    cont = [", ".join(sorted(set(cat["name"].iloc[np.flatnonzero(np.abs(cw - wf[k]) < fw_f[k])]))) for k in range(len(F))]
    # a known line of another species closer than half a resolution element cannot be separated: flagged
    close = [", ".join(sorted(set(cat["name"].iloc[np.flatnonzero(np.abs(cw - wf[k]) < 0.5 * fw_f[k])]))) for k in range(len(F))]
    F["neighbours"] = near
    F["contaminants"] = cont
    F["blended"] = [c != "" for c in close]
    if cov is not None:
        F["band"] = [next((b for lo, hi, b in sorted(cov, key=lambda c: -min(x - c[0], c[1] - x)) if lo <= x <= hi), "")
                     for x in wf]
    else:
        F["band"] = ""
    # blends of the molecule that are not features (other vibrational bands, weaker or dropped lines): the
    # measurement fits them next to a feature when they are strong enough
    oth = [k for k in range(len(groups)) if k not in chosen and strength[k] >= rel_min * Isel]
    ow = [float(np.sum(I[groups[k]] * w[groups[k]]) / np.sum(I[groups[k]])) for k in oth]
    F.attrs.update(molecule=sp.name, release=ll.release, t_ref=t_ref, source=ll.source,
                   other_wave=ow, other_rel=[float(strength[k] / Isel) for k in oth],
                   feature_rel=(F["strength"] / Isel).tolist())
    return F, M


def _empty():
    cols = ["id", "label", "top", "wave", "eu", "gA", "a", "gu", "n_members", "strength", "spin", "ladder", "band_v",
            "eu_spread", "xmin", "xmax", "rel_strength", "neighbours", "contaminants", "blended", "band"]
    return pd.DataFrame(columns=cols), pd.DataFrame(columns=["feature", "wave", "a", "gu", "eu", "el", "label", "spin",
                                                             "ladder", "band_v", "rel"])


def match_table(table: pd.DataFrame, features: pd.DataFrame, tol_fwhm: float = 0.5, R_model="argyriou2023") -> np.ndarray:
    """Feature id for each row of a user flux table (by 'label' / 'line' column, else by wavelength)."""
    out = np.full(len(table), -1)
    labcol = next((c for c in ("label", "line", "transition", "name") if c in table.columns), None)
    wcol = next((c for c in ("wave", "wave_um", "lambda", "wavelength", "lam") if c in table.columns), None)
    tops = features["top"].astype(str).str.replace(" ", "").str.lower().to_numpy()
    labs = features["label"].astype(str).str.replace(" ", "").str.lower().to_numpy()
    wf = features["wave"].to_numpy()
    fw = fwhm_kms(wf, R_model) / C_KMS * wf if len(wf) else wf
    for i, r in enumerate(table.itertuples(index=False)):
        if labcol is not None and isinstance(getattr(r, labcol), str):
            key = getattr(r, labcol).replace(" ", "").lower()
            key = key.replace("h2", "", 1) if key.startswith("h2") and "s(" in key else key
            hit = np.flatnonzero((tops == key) | (labs == key))
            if len(hit):
                out[i] = int(features["id"].iloc[hit[0]]); continue
        if wcol is not None and np.isfinite(getattr(r, wcol)) and len(wf):
            j = int(np.argmin(np.abs(wf - getattr(r, wcol))))
            if abs(wf[j] - getattr(r, wcol)) <= tol_fwhm * fw[j]:
                out[i] = int(features["id"].iloc[j])
    return out
