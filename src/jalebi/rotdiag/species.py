"""Molecules for rotation diagrams: line lists, transition labels, spin species and partition functions.

Each preset says which line list to load, which transitions a rotation (population) diagram normally uses,
which reference temperature ranks the lines when they are selected, and how the levels split into nuclear-spin
species (ortho / para) when the ortho-to-para ratio (OPR) is a fit parameter.

Partition functions
-------------------
* H2: exact level sums over the 302 bound rovibrational levels of the X state (Roueff et al. 2019), separately
  for ortho (odd J, g_I = 3) and para (even J, g_I = 1): Q = Q_o + Q_p, OPR_LTE(T) = Q_o / Q_p.  The sum agrees
  with the HITRAN/TIPS Q(T) of H2 to < 0.1 % up to 1000 K.
* Other molecules: the TIPS-2021 Q(T) of the line list.  For H2O (ortho = g / (2J+1) = 3) the spin-resolved
  sums are taken in the high-temperature limit Q_o = 3/4 Q, Q_p = 1/4 Q (exact to better than 1 % above ~50 K).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from ..linedata import LineList, available_linelists, load_linelist
from ..molecules import get_molecule

DATA_FILES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_files")


@dataclass(frozen=True)
class RotSpecies:
    """How to build a rotation diagram of one molecule."""
    name: str                          # jalebi molecule name (line-list key)
    label: str                         # for plots
    release: str                       # default line-list release
    t_ref: float                       # K: ranks the lines when selecting (thin LTE intensity at t_ref)
    bands: tuple[str, ...] = ()        # default vibrational bands "vup-vlow" (empty = all)
    spin: str = ""                     # "" | "h2" (J parity) | "g2j" (g_u / (2J+1) = 3 -> ortho)
    opr_high_t: float = 3.0            # OPR in the high-T limit (both H2 and H2O: 3)
    t_bounds: tuple[float, float] = (50.0, 5000.0)
    note: str = ""
    curated: str = ""                  # optional curated line list (data_files CSV) used as the default selection
    releases: tuple[str, ...] = field(default=())


PRESETS: dict[str, RotSpecies] = {
    "H2": RotSpecies("H2", "H₂", "roueff2019", 800.0, bands=("0-0",), spin="h2", t_bounds=(80.0, 5000.0),
                     note="pure rotational S(J) lines (0-0); add 1-0 for NIRSpec. Quadrupole lines: always optically thin. "
                          "S(3) at 9.66 µm sits in the silicate feature and pins A_V."),
    "CO": RotSpecies("CO", "CO", "hitemp", 1000.0, bands=("1-0",), t_bounds=(100.0, 5000.0),
                     note="rovibrational v=1-0 (and 2-1 ...) P/R lines at 4.4-5.3 µm; the low-J lines are often optically "
                          "thick: switch on the opacity correction."),
    "13CO": RotSpecies("13CO", "¹³CO", "hitran", 800.0, bands=("1-0",), t_bounds=(100.0, 5000.0)),
    "OH": RotSpecies("OH", "OH", "hitran", 1500.0, t_bounds=(100.0, 20000.0),
                     note="pure rotational lines of both X²Π ladders; the high-N lines (< 13 µm) are prompt emission from "
                          "H2O photodissociation and are not thermal: expect curvature (two components)."),
    "H2O": RotSpecies("H2O", "H₂O", "hitran", 600.0, spin="g2j", t_bounds=(100.0, 3000.0),
                      curated="MIRI_H2O_v0-0.csv",
                      note="default selection: the isolated lines of Banzatti et al. (2025); water lines are often "
                           "optically thick: use the opacity correction."),
    "HCN": RotSpecies("HCN", "HCN", "hitran", 700.0, t_bounds=(100.0, 3000.0)),
    "C2H2": RotSpecies("C2H2", "C₂H₂", "hitran", 700.0, t_bounds=(100.0, 3000.0)),
    "CO2": RotSpecies("CO2", "CO₂", "hitran", 600.0, t_bounds=(100.0, 3000.0)),
    "CH4": RotSpecies("CH4", "CH₄", "hitran", 500.0, t_bounds=(50.0, 3000.0)),
    "NH3": RotSpecies("NH3", "NH₃", "hitran", 500.0, t_bounds=(50.0, 3000.0)),
}


def preset(name: str) -> RotSpecies:
    """The preset of a molecule (a generic one for any other molecule with a line list)."""
    mol = get_molecule(name)
    if mol.name in PRESETS:
        return PRESETS[mol.name]
    return RotSpecies(mol.name, mol.label, "hitran", 700.0)


def rotdiag_molecules() -> list[str]:
    """Molecules that can be used: the presets plus every molecule with a cached line list."""
    names = list(PRESETS)
    try:
        for m in available_linelists()["molecule"]:
            if m not in names:
                names.append(m)
    except Exception:
        pass
    return names


def releases_for(name: str) -> list[str]:
    try:
        ll = available_linelists()
        rel = sorted(set(ll[ll["molecule"] == get_molecule(name).name]["release"]))
    except Exception:
        rel = []
    p = preset(name).release
    return ([p] if p in rel else []) + [r for r in rel if r != p] or [p]


def load_species_linelist(name: str, release: str | None = None, path: str | None = None) -> LineList:
    """The line list of a rotation-diagram molecule (no download: the cached / bundled lists only)."""
    p = preset(name)
    rel = release or p.release
    try:
        return load_linelist(p.name, release=rel, path=path, fetch=False, fallback=False)
    except Exception:
        return load_linelist(p.name, release=rel, path=path, fetch=False, fallback=True)


# ------------------------------------------------------------------------------------------------
# quantum labels
# ------------------------------------------------------------------------------------------------

def _ws(s) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def _j_upper(mol: str, qup: str, qlow: str) -> float:
    """Upper-state J from the HITRAN quantum labels (NaN if it cannot be read)."""
    q = _ws(qup)
    m = re.match(r"J=(\d+)", q)
    if m:
        return float(m.group(1))
    if mol in ("H2O", "H2_18O"):             # 'J Ka Kc'
        p = q.split()
        return float(p[0]) if p and p[0].isdigit() else np.nan
    if mol in ("CO", "13CO"):                # HITEMP 'R141' / 'P116' on the lower label: branch + J_lower
        m = re.match(r"([PRQ])[_\s]*(\d+)", _ws(qlow))
        if m:
            J = int(m.group(2))
            return float(J + 1 if m.group(1) == "R" else J - 1 if m.group(1) == "P" else J)
    if mol == "H2":                           # HITRAN H2: 'S 13q' / 'O 26q' / 'Q 3q' on the lower label: branch + J_lower
        m = re.match(r"([OQS])\s*(\d+)", _ws(qlow))
        if m:
            J = int(m.group(2))
            return float(J + 2 if m.group(1) == "S" else J - 2 if m.group(1) == "O" else J)
    return np.nan


def transition_label(mol: str, row) -> str:
    """Human label of a transition, e.g. 'S(1)', 'v=1-0 P(10)', 'Π3/2 J=12.5→11.5 ff', 'H2O 21_4_17→20_3_18'."""
    vup, vlow, qup, qlow = (_ws(row[c]) for c in ("vup", "vlow", "qup", "qlow"))
    if mol == "H2":
        m = re.match(r"(\d+)-(\d+) ([OQS])\((\d+)\)", qlow)            # Roueff: '0-0 S(1)'
        if m:
            v = f"{m.group(1)}-{m.group(2)}"
            return f"{m.group(3)}({m.group(4)})" if v == "0-0" else f"{v} {m.group(3)}({m.group(4)})"
        m = re.match(r"([OQS])\s*(\d+)", qlow)                         # HITRAN: 'S 1q'
        if m:
            v = f"{vup}-{vlow}"
            return f"{m.group(1)}({m.group(2)})" if v == "0-0" else f"{v} {m.group(1)}({m.group(2)})"
    if mol in ("CO", "13CO"):
        m = re.match(r"([PRQ])[_\s]*(\d+)", qlow)
        if m:
            return f"v={vup}-{vlow} {m.group(1)}({m.group(2)})"
    if mol == "OH":
        lad_u = "Π3/2" if "3/2" in vup else "Π1/2" if "1/2" in vup else vup.split()[0] if vup else ""
        lad_l = "Π3/2" if "3/2" in vlow else "Π1/2" if "1/2" in vlow else ""
        m = re.search(r"([A-Z]{1,2})\s*(\d+\.5)\s*([ef]{2})", qlow)
        if m:
            Jl = float(m.group(2))
            br = m.group(1)
            dJ = {"R": 1, "P": -1, "Q": 0}.get(br[-1], 0)
            ladder = lad_u if lad_u == lad_l else f"{lad_u}→{lad_l}"
            return f"{ladder} {br} J={Jl + dJ:g}→{Jl:g} {m.group(3)}"
    if mol in ("H2O", "H2_18O"):
        pu, pl = qup.split(), qlow.split()
        if len(pu) >= 3 and len(pl) >= 3:
            vib = "" if vup.replace(" ", "") == vlow.replace(" ", "") == "000" else f"({vup.replace(' ', '')}-{vlow.replace(' ', '')}) "
            return f"{vib}{pu[0]}_{pu[1]}_{pu[2]}→{pl[0]}_{pl[1]}_{pl[2]}"
    return f"{vup}-{vlow} {qup}→{qlow}".strip()


def vib_band(mol: str, row) -> str:
    """'vup-vlow' (H2O: '000-000' style collapsed) for band selection and colouring."""
    vup, vlow = _ws(row["vup"]).replace(" ", ""), _ws(row["vlow"]).replace(" ", "")
    if mol == "OH":
        # ladder + vibrational level, e.g. 'X3/2 0' -> '0-0'
        vu = re.findall(r"\d+$", _ws(row["vup"])); vl = re.findall(r"\d+$", _ws(row["vlow"]))
        return f"{vu[0] if vu else '?'}-{vl[0] if vl else '?'}"
    return f"{vup}-{vlow}"


def ladder(mol: str, row) -> str:
    """A grouping used to colour the diagram: ortho/para for H2 and H2O, Π3/2 / Π1/2 for OH, the band for CO."""
    if mol == "OH":
        v = _ws(row["vup"])
        return "Π3/2" if "3/2" in v else "Π1/2" if "1/2" in v else v
    return vib_band(mol, row)


def spin_species(sp: RotSpecies, table: pd.DataFrame) -> np.ndarray:
    """'o' / 'p' per line (or '' when the molecule has no spin species)."""
    n = len(table)
    if not sp.spin:
        return np.array([""] * n)
    J = np.array([_j_upper(sp.name, qu, ql) for qu, ql in zip(table["qup"], table["qlow"])], float)
    if sp.spin == "h2":
        return np.where(np.isfinite(J), np.where(np.mod(J, 2) == 1, "o", "p"), "")
    if sp.spin == "g2j":
        r = table["gu"].to_numpy() / (2 * J + 1)
        return np.where(np.isfinite(r), np.where(np.isclose(r, 3.0, rtol=0.05), "o", np.where(np.isclose(r, 1.0, rtol=0.05), "p", "")), "")
    return np.array([""] * n)


# ------------------------------------------------------------------------------------------------
# partition functions (total and per spin species)
# ------------------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def h2_levels() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_FILES, "H2_levels_Roueff2019.csv"), comment="#")


class SpinPartition:
    """Q(T), Q_o(T), Q_p(T) tabulated at 1 K steps and linearly interpolated (vectorised)."""

    def __init__(self, T: np.ndarray, Q: np.ndarray, Qo: np.ndarray | None = None, Qp: np.ndarray | None = None,
                 source: str = ""):
        self.T = np.asarray(T, float)
        self.Qt = np.asarray(Q, float)
        self.Qo_t = None if Qo is None else np.asarray(Qo, float)
        self.Qp_t = None if Qp is None else np.asarray(Qp, float)
        self.source = source

    @property
    def has_spin(self) -> bool:
        return self.Qo_t is not None

    def _interp(self, T, table):
        """Linear interpolation; above the table a power law with the slope of its last decade
        (Q ∝ T^s: s = 1 for a linear rotor, 1.5 for a non-linear one)."""
        T = np.asarray(T, float)
        out = np.interp(T, self.T, table)
        hi = T > self.T[-1]
        if np.any(hi):
            i0 = np.searchsorted(self.T, self.T[-1] / 10.0)
            s = np.log(table[-1] / table[i0]) / np.log(self.T[-1] / self.T[i0])
            out = np.where(hi, table[-1] * (np.maximum(T, 1.0) / self.T[-1]) ** s, out)
        return out

    def Q(self, T):
        return self._interp(T, self.Qt)

    def Qs(self, T, species: str):
        if species == "o":
            return self._interp(T, self.Qo_t)
        if species == "p":
            return self._interp(T, self.Qp_t)
        return self.Q(T)

    def opr_lte(self, T):
        """Ortho-to-para ratio in LTE at T (NaN without spin species)."""
        if not self.has_spin:
            return np.full_like(np.asarray(T, float), np.nan)
        return self._interp(T, self.Qo_t) / self._interp(T, self.Qp_t)

    @property
    def tmax(self):
        return float(self.T[-1])


def partition(sp: RotSpecies, ll: LineList) -> SpinPartition:
    if sp.name == "H2":
        L = h2_levels()
        T = np.arange(1.0, 10001.0)
        g, E, o = L["g"].to_numpy(), L["E_K"].to_numpy(), (L["spin"] == "o").to_numpy()
        B = np.exp(-E[None, :] / T[:, None]) * g[None, :]
        Qo, Qp = B[:, o].sum(1), B[:, ~o].sum(1)
        return SpinPartition(T, Qo + Qp, Qo, Qp, source="level sums (Roueff et al. 2019)")
    T, Q = ll.partition.T, ll.partition.Q
    if sp.spin == "g2j":
        f = sp.opr_high_t / (1.0 + sp.opr_high_t)
        return SpinPartition(T, Q, f * Q, (1 - f) * Q, source=f"{ll.partition.source}; spin split in the high-T limit (OPR {sp.opr_high_t:g})")
    return SpinPartition(T, Q, source=ll.partition.source)
