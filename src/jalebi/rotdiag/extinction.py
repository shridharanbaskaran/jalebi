"""Dust extinction curves k(λ) = A_λ / A_V for de-reddening line fluxes.

    F_obs = F_int · 10^(−0.4 A_V k(λ))    →    ln(N_u/g_u)_obs = ln(N_u/g_u)_int − 0.921 A_V k(λ)

Bundled curves (tabulated from the `dust_extinction` package, Gordon et al. 2024, JOSS 9, 7023):

  G23        Gordon et al. (2023, ApJ 950, 86), Milky Way average, R_V = 3.1 (0.09–32 µm)  [default]
  G23_Rv5.5  the same with R_V = 5.5 (dense clouds; flatter NIR, stronger silicate relative to A_V)
  G21        Gordon et al. (2021, ApJ 916, 33), MW average from Spitzer IRS (1–32 µm)
  CT06       Chiar & Tielens (2006, ApJ 637, 774), local ISM (1.24–27 µm)
  F11        Fritz et al. (2011, ApJ 737, 73), Galactic centre from H recombination lines (1.28–19 µm)

Any other curve — e.g. KP5 (Pontoppidan et al. 2024), the dense-cloud curve used by JOYS / JDISCS, or
McClure (2009) — is read from a CSV of two columns, wavelength [µm] and A_λ/A_V (or A_λ/A_K with
``normalise="K"``, converted with A_K/A_V of G23), by passing its path as the curve name.
Outside a curve's range the nearest tabulated value is used (and `ExtinctionCurve.extrapolated` says where).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from .species import DATA_FILES

CURVES = {"G23": "G23_Rv3.1", "G23_Rv3.1": "G23_Rv3.1", "G23_Rv5.5": "G23_Rv5.5", "G21": "G21_MWAvg", "G21_MWAvg": "G21_MWAvg",
          "CT06": "CT06_MWLoc", "CT06_MWLoc": "CT06_MWLoc", "F11": "F11_MWGC", "F11_MWGC": "F11_MWGC"}
CURVE_LABELS = {"G23": "Gordon+2023 (R_V 3.1)", "G23_Rv5.5": "Gordon+2023 (R_V 5.5)", "G21": "Gordon+2021 MW avg",
                "CT06": "Chiar & Tielens 2006 local ISM", "F11": "Fritz+2011 Galactic centre"}
LN10_04 = 0.4 * np.log(10.0)          # 0.921: A_λ [mag] -> natural-log flux decrement
K_BAND_UM = 2.159


@lru_cache(maxsize=1)
def _table() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_FILES, "extinction_curves.csv"), comment="#")


@dataclass
class ExtinctionCurve:
    name: str
    wave: np.ndarray
    k: np.ndarray              # A_λ / A_V
    source: str = ""

    def __call__(self, wave_um) -> np.ndarray:
        """A_λ/A_V at the given wavelengths (nearest tabulated value outside the range)."""
        return np.interp(np.asarray(wave_um, float), self.wave, self.k)

    def extrapolated(self, wave_um) -> np.ndarray:
        w = np.asarray(wave_um, float)
        return (w < self.wave[0]) | (w > self.wave[-1])

    @property
    def ak_av(self) -> float:
        """A_K / A_V of this curve (2.16 µm)."""
        return float(self(K_BAND_UM))

    def transmission(self, wave_um, A_V: float) -> np.ndarray:
        return 10.0 ** (-0.4 * A_V * self(wave_um))


def get_curve(name: str = "G23", normalise: str = "V") -> ExtinctionCurve:
    """A bundled curve by name, or a CSV file (wave_um, A_λ/A_V; or A_λ/A_K with normalise='K')."""
    key = CURVES.get(name)
    if key is not None:
        t = _table()
        v = t[key].to_numpy()
        ok = np.isfinite(v)
        label = CURVE_LABELS.get(name) or next((CURVE_LABELS[k] for k, c in CURVES.items() if c == key and k in CURVE_LABELS), key)
        return ExtinctionCurve(name, t["wave_um"].to_numpy()[ok], v[ok], source=label)
    from ..examples import resolve_path
    path = resolve_path(name)
    if not os.path.exists(path):
        raise ValueError(f"unknown extinction curve '{name}': use one of {', '.join(sorted(set(CURVES) - set(CURVES.values())))} "
                         "or the path of a CSV file (wavelength in µm, A_λ/A_V)")
    df = pd.read_csv(path, comment="#", sep=None, engine="python")
    w = df.iloc[:, 0].to_numpy(float); k = df.iloc[:, 1].to_numpy(float)
    o = np.argsort(w); w, k = w[o], k[o]
    if normalise.upper() == "K":
        k = k * get_curve("G23").ak_av
    return ExtinctionCurve(os.path.basename(path), w, k, source=path)


def available_curves() -> dict[str, str]:
    return {k: CURVE_LABELS[k] for k in ("G23", "G23_Rv5.5", "G21", "CT06", "F11")}
