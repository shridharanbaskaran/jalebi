"""Fit regions built from curated line lists (Banzatti et al. 2025, AJ 169, 165).

Published water fits do not use every pixel: Banzatti et al. (2023) fit the fluxes of ~100 isolated
rotational lines, Temmink et al. (2024, 2025) fit narrow regions around selected lines and ortho-para
pairs (their Table C.1) with higher weights beyond 20 um, Romero-Mirza et al. (2024) fit the full
12-27 um spectrum.  `line_regions` turns the bundled Banzatti+2025 line lists into fit windows so that
jalebi can reproduce the line-selected approach:

    fit:
      line_regions: [H2O_v0-0]          # names below, or a CSV path with xmin/xmax columns
      region_pad_um: 0.0                # widen every region by this much on each side
      region_weight_beyond: [20.0, 5.0] # weight 5 for regions beyond 20 um (Temmink+2025 style)

Names: H2O_v0-0 (56 isolated pure-rotational lines, 9.9-27 um), H2O_v1-0 (27 ro-vibrational lines,
5.2-7.5 um), H2O_v1-1 (hot-band lines, 10-17 um), H2O_general (the 81 water lines of the general MIRI
list), general (every species of that list: H2O, OH, CO, H I, H2).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

DATA_FILES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_files")

LINE_REGION_FILES = {
    "H2O_v0-0": ("MIRI_H2O_v0-0.csv", "H2O"),
    "H2O_v1-0": ("MIRI_H2O_v1-0.csv", "H2O"),
    "H2O_v1-1": ("MIRI_H2O_v1-1.csv", "H2O"),
    "H2O_general": ("MIRI_general_Banzatti+2025.csv", "H2O"),
    "general": ("MIRI_general_Banzatti+2025.csv", None),
}


def line_region_table(name: str) -> pd.DataFrame:
    """Table (species, lam, e_up, a_stein, xmin, xmax) of one named list or CSV path."""
    if name in LINE_REGION_FILES:
        fn, species = LINE_REGION_FILES[name]
        t = pd.read_csv(os.path.join(DATA_FILES, fn))
        if species is not None and "species" in t:
            t = t[t["species"] == species]
    else:
        from .examples import resolve_path
        t = pd.read_csv(resolve_path(name))
    if "xmin" not in t or "xmax" not in t:
        # lists without ranges (v1-1): +/- 1.5 resolution elements at R = 3000
        lam = t["lam"].to_numpy(float)
        t = t.assign(xmin=lam * (1 - 1.5 / 3000.0), xmax=lam * (1 + 1.5 / 3000.0))
    return t.reset_index(drop=True)


def line_regions(names: list[str], pad_um: float = 0.0, rv_kms: float = 0.0,
                 limit: tuple[float, float] | None = None) -> list[tuple[float, float]]:
    """Merged fit windows (um, rest frame) from the named lists; `rv_kms` shifts them when the spectrum was
    not shifted to the rest frame; `limit` drops regions outside (lo, hi)."""
    from .model import merge_intervals
    ws = []
    for n in names:
        t = line_region_table(n)
        for a, b in zip(t["xmin"].to_numpy(float), t["xmax"].to_numpy(float)):
            a, b = a * (1 + rv_kms / 299792.458), b * (1 + rv_kms / 299792.458)
            if limit is not None and (b < limit[0] or a > limit[1]):
                continue
            ws.append((a - pad_um, b + pad_um))
    return merge_intervals(ws) if ws else []


def region_weights(windows: list[tuple[float, float]], beyond: tuple[float, float] | None) -> dict[int, float]:
    """{window index: weight} giving `beyond[1]` to every window whose centre is past `beyond[0]` um."""
    if not beyond:
        return {}
    lam0, w = float(beyond[0]), float(beyond[1])
    return {i: w for i, (a, b) in enumerate(windows) if 0.5 * (a + b) > lam0}


def summarize(windows: list[tuple[float, float]]) -> str:
    n = len(windows)
    tot = float(np.sum([b - a for a, b in windows])) if windows else 0.0
    return f"{n} regions, {tot:.2f} um in total, {windows[0][0]:.2f}-{windows[-1][1]:.2f} um" if n else "no regions"
