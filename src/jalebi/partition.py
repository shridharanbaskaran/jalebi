"""Partition functions Z(T).

Sources, in order of preference:
  1. a Q(T) table supplied with a line list (e.g. the header of an iSLAT .par file),
  2. TIPS-2021 through HAPI's in-package tables (offline, no network needed),
  3. the TIPS_2021_PYTHON QTpy pickles (legacy, needs a folder path).

Tables are held in memory as arrays and evaluated by vectorised linear interpolation,
so Z(T) costs microseconds inside the likelihood.
"""
from __future__ import annotations

import os
import pickle
import warnings
from functools import lru_cache

import numpy as np


class PartitionFunction:
    """Tabulated Z(T) with linear interpolation. T outside the table is clipped."""

    def __init__(self, T: np.ndarray, Q: np.ndarray, source: str = ""):
        T = np.asarray(T, float)
        Q = np.asarray(Q, float)
        order = np.argsort(T)
        self.T = T[order]
        self.Q = Q[order]
        self.source = source

    def __call__(self, T):
        return np.interp(T, self.T, self.Q)

    @property
    def tmin(self):
        return float(self.T[0])

    @property
    def tmax(self):
        return float(self.T[-1])


def _hapi():
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        import hapi  # noqa: F401  (prints a banner on import)
    return hapi


@lru_cache(maxsize=None)
def tips_partition(hitran_id: int, iso: int, tmax: float = 5000.0) -> PartitionFunction:
    """TIPS-2021 partition sum from HAPI, tabulated at 1 K steps up to tmax."""
    hapi = _hapi()
    T = np.arange(1.0, tmax + 1.0, 1.0)
    Q = np.empty_like(T)
    # HAPI's tables stop at different Tmax per isotopologue; probe and clip.
    upper = tmax
    try:
        hapi.partitionSum(hitran_id, iso, [tmax])
    except Exception:
        for cand in (3000.0, 2000.0, 1500.0, 1000.0):
            try:
                hapi.partitionSum(hitran_id, iso, [cand])
                upper = cand
                break
            except Exception:
                continue
    T = T[T <= upper]
    Q = np.asarray(hapi.partitionSum(hitran_id, iso, list(T)), float)
    return PartitionFunction(T, Q, source=f"TIPS-2021 via HAPI (M={hitran_id}, I={iso})")


def qtpy_partition(folder: str, hitran_id: int, iso: int) -> PartitionFunction:
    """Legacy reader for the TIPS_2021_PYTHON QTpy pickles (one file per isotopologue)."""
    path = os.path.join(folder, "QTpy", f"{hitran_id}_{iso}.QTpy")
    if not os.path.exists(path):
        path = os.path.join(folder, f"{hitran_id}_{iso}.QTpy")
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    T = np.array([float(k) for k in d.keys()])
    Q = np.array([float(v) for v in d.values()])
    return PartitionFunction(T, Q, source=path)


def partition_for(molecule, table: tuple[np.ndarray, np.ndarray] | None = None,
                  qtpy_folder: str | None = None) -> PartitionFunction:
    """Best available partition function for a Molecule."""
    if table is not None:
        return PartitionFunction(table[0], table[1], source="line-list table")
    if qtpy_folder:
        try:
            return qtpy_partition(qtpy_folder, molecule.hitran_id, molecule.iso)
        except Exception as e:  # pragma: no cover
            warnings.warn(f"QTpy partition function failed ({e}); falling back to TIPS via HAPI")
    if molecule.hitran_id > 0:
        return tips_partition(molecule.hitran_id, molecule.iso)
    raise ValueError(f"No partition function available for {molecule.name}; supply a Q(T) table")
