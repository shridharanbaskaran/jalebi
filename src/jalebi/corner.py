"""Corner and pseudo-continuum diagnostic (0.22, fit.corner_check).

Two symptoms of a component that is fitting the continuum rather than lines:

  pinned     two or more of its parameters sit at a prior bound (within `bound_frac` of the range), typically
             T at the upper bound, log N at the upper bound and a tiny area -- the "hot-water corner" seen in
             7 blind and 8 known validation disks (0.20): a hot, optically thick, tiny slab whose forest of
             weak lines mimics a pseudo-continuum, after which the warm and cold components shift to compensate.
  smooth     the share of the component's convolved flux that survives a low-pass filter (running median over
             `smooth_window_um`, wide compared with a resolution element): lines leave ~0, a forest of blended
             lines or a broad band head leaves most of the flux.  Above `smooth_threshold` the component is
             mostly pseudo-continuum.  The flag applies to `smooth_molecules` (default water): a CO2 / C2H2 / HCN
             Q branch is a smooth band head by nature (FZ Tau: CO2 0.50, C2H2 0.77) and is only reported.

Both go to diagnostics.json ("corner"), the population table (catalogue_row) and a warning in the log.  They
do not change the fit; they tell you which fits to distrust and where fit.continuum_fit may help.
"""
from __future__ import annotations

import numpy as np


def smooth_fraction(wave: np.ndarray, band: np.ndarray, flux: np.ndarray, window_um: float = 0.3) -> float:
    """Fraction of sum(flux) left after a running median of `window_um` per sub-band (0 for isolated lines,
    -> 1 for a smooth hump).  NaN when the component has no flux."""
    from scipy.ndimage import median_filter
    flux = np.asarray(flux, float)
    tot = float(np.nansum(flux))
    if not np.isfinite(tot) or tot <= 0:
        return float("nan")
    low = np.zeros_like(flux)
    band = np.asarray(band)
    for b in dict.fromkeys(band.tolist()):
        i = np.flatnonzero(band == b)
        if len(i) < 3:
            continue
        # contiguous runs of increasing wavelength inside the band (windows may split it)
        w = wave[i]
        breaks = np.flatnonzero(np.diff(w) > 5 * np.median(np.diff(w))) + 1
        for seg in np.split(np.arange(len(i)), breaks):
            if len(seg) < 3:
                continue
            idx = i[seg]
            disp = float(np.median(np.diff(wave[idx]))) if len(idx) > 1 else window_um
            n = int(max(3, round(window_um / max(disp, 1e-9))))
            n = min(n | 1, len(idx) | 1)
            low[idx] = median_filter(flux[idx], size=n, mode="nearest")
    return float(np.clip(np.nansum(np.maximum(low, 0.0)) / tot, 0.0, 1.0))


def pinned_parameters(problem, theta, bound_frac: float = 0.02) -> dict[str, list[str]]:
    """component -> parameters within `bound_frac` of the prior range of either bound (areas included)."""
    theta = np.asarray(theta, float)
    out = {}
    for p, v, lo, hi in zip(problem.free, theta, problem.lo, problem.hi):
        if p.comp == "global":
            continue
        span = hi - lo
        if span <= 0:
            continue
        if v - lo <= bound_frac * span:
            out.setdefault(p.comp, []).append(f"{p.name}@lo")
        elif hi - v <= bound_frac * span:
            out.setdefault(p.comp, []).append(f"{p.name}@hi")
    return out


def corner_check(problem, theta, bound_frac: float = 0.02, smooth_window_um: float = 0.3,
                 smooth_threshold: float = 0.5, min_pinned: int = 2, smooth_molecules=("H2O",)) -> dict[str, dict]:
    """Per unit (component or opacity group): pinned parameters, smooth fraction and flags.  The pseudo-continuum
    flag is raised only for the molecules in `smooth_molecules` (empty = all): a Q-branch band head (CO2 15 um,
    C2H2 13.7 um, HCN 14 um) is smooth on a 0.3 um scale by nature, a water forest is not."""
    P, _ = problem.params_from_theta(theta)
    _, units, tmax = problem.model.evaluate(P, per_unit=True)
    pinned = pinned_parameters(problem, theta, bound_frac)
    out = {}
    for key, f in units.items():
        members = [c.name for c in problem.components if c.enabled and (c.name == key or c.group == key or c.tie_to == key)]
        pins = sorted(f"{m}.{p}" if len(members) > 1 else p for m in members for p in pinned.get(m, []))
        sf = smooth_fraction(problem.wave, problem.band, f, smooth_window_um)
        flags = []
        if len(pins) >= min_pinned:
            flags.append("pinned")
        if np.isfinite(sf) and sf > smooth_threshold:
            flags.append("pseudo-continuum")
        lead = next((c for c in problem.components if c.name == key or c.group == key), None)
        mol = lead.molecule if lead is not None else key
        if "pseudo-continuum" in flags and smooth_molecules and mol not in set(smooth_molecules):
            flags.remove("pseudo-continuum")
        out[key] = {"molecule": mol, "pinned": pins, "n_pinned": len(pins),
                    "smooth_fraction": sf, "tau_max": float(tmax.get(key, np.nan)), "flags": flags}
    return out


def corner_lines(check: dict) -> list[str]:
    """Log lines for the flagged units."""
    out = []
    for key, d in check.items():
        if d["flags"]:
            what = []
            if "pinned" in d["flags"]:
                what.append("pinned at " + ", ".join(d["pinned"]))
            if "pseudo-continuum" in d["flags"]:
                what.append(f"smooth fraction {d['smooth_fraction']:.2f} (mostly pseudo-continuum)")
            out.append(f"  warning: {key}: " + "; ".join(what) + " -- distrust this component; see docs/CONTINUUM.md")
    return out
