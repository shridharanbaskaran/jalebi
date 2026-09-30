"""Synthetic line spectra with a known rotation diagram (for tests, demos and injection–recovery).

    from jalebi.rotdiag.synthetic import make_rotdiag_spectrum
    spec, truth = make_rotdiag_spectrum("H2", model="two", params=dict(logN1=46.0, T1=500, logN2=44.3, T2=1800, Av=8, OPR=2.2),
                                        opr="species")

The line fluxes come from the rotation-diagram physics (jalebi.rotdiag.physics: populations, OPR, extinction,
optical depth); each line is a Gaussian of the MIRI-MRS resolution (Argyriou et al. 2023) on the pixel grid of the
MRS cubes, on top of a smooth continuum with Gaussian noise.  For an independent check of the physics use
`jalebi.synthetic.make_synthetic_spectrum` (full LTE slab spectra) instead.
"""
from __future__ import annotations

import numpy as np
from scipy.special import erf

from ..data import Spectrum
from ..lines import C_KMS
from ..synthetic import band_pixels, continuum_model
from .features import Selection, find_features, fwhm_kms
from .fit import FitConfig, make_model
from .physics import Geometry

ALL_BANDS = ("1A", "1B", "1C", "2A", "2B", "2C", "3A", "3B", "3C", "4A", "4B", "4C")


def make_rotdiag_spectrum(molecule: str = "H2", model: str = "single", params: dict | None = None, opr: str = "thermal",
                          geometry: Geometry | None = None, extinction: str = "G23", opacity: bool = False,
                          bands=ALL_BANDS, snr: float = 300.0, continuum: dict | None = None, seed: int = 0,
                          v_kms: float = 0.0, width_scale: float = 1.0, release: str | None = None,
                          selection: Selection | None = None, name: str | None = None, distance_pc: float = 140.0):
    """(Spectrum, truth) with the lines of `molecule` for the given rotation-diagram model and parameters."""
    params = dict(params or {})
    geometry = geometry or Geometry(mode="number", distance_pc=distance_pc)
    rng = np.random.default_rng(seed)
    W, B = [], []
    for b in bands:
        w = band_pixels(b)
        W.append(w); B.append(np.full(len(w), b, dtype=object))
    wave = np.concatenate(W); band = np.concatenate(B).astype(str)
    cont = continuum_model(wave, **(continuum or dict(level=0.2, slope=1.0, wiggle=0.0)))
    dummy = Spectrum(wave, cont.copy(), np.full(len(wave), 1.0), band, name="tmp", distance_pc=geometry.distance_pc, rest_frame=True)
    sel = selection or Selection(rel_min=1e-5, max_features=400, edge_fwhm=0.0, blend_fwhm=0.0)
    F, M = find_features(molecule, dummy, sel, release=release)
    cfg = FitConfig(model=model, opr=opr, extinction=extinction, geometry=geometry, opacity=opacity,
                    av=params.get("Av", 0.0), opr_value=params.get("OPR", 3.0), fwhm_kms=params.get("fwhm", 10.0))
    rm = make_model(F, M, cfg, molecule)
    for k, v in params.items():
        if k in rm.params:
            rm.params[k].value = float(v)
    P = {k: np.array([p.value]) for k, p in rm.params.items()}
    Fm = rm.member_flux(P)[0]                        # W m^-2 per member line
    lw = M["wave"].to_numpy()
    line = np.zeros(len(wave))
    for b in bands:
        i = np.flatnonzero(band == b)
        x = wave[i]
        mid = 0.5 * (x[1:] + x[:-1])
        edges = np.concatenate([[x[0] - (mid[0] - x[0])], mid, [x[-1] + (x[-1] - mid[-1])]])
        sel_l = (lw > x[0] - 0.05) & (lw < x[-1] + 0.05)
        for wl, fl in zip(lw[sel_l] * (1 + v_kms / C_KMS), Fm[sel_l]):
            sig = width_scale * wl * float(fwhm_kms(wl)) / C_KMS / 2.354820045
            area = fl * (wl ** 2) / (2.99792458e14 * 1e-26)          # W m^-2 -> Jy µm
            cdf = 0.5 * (1 + erf((edges - wl) / (np.sqrt(2) * sig)))  # pixel-integrated Gaussian
            line[i] += area * np.diff(cdf) / np.diff(edges)
    noise = np.empty(len(wave))
    for b in bands:
        i = np.flatnonzero(band == b)
        noise[i] = float(np.median(cont[i])) / snr
    flux = cont + line + rng.normal(0.0, noise)
    spec = Spectrum(wave, flux, noise, band, name=name or f"synthetic {molecule}", distance_pc=geometry.distance_pc,
                    rest_frame=True, meta={"synthetic": True, "seed": seed, "snr": snr})
    truth = dict(molecule=molecule, model=model, params={k: float(p.value) for k, p in rm.params.items()}, opr=opr,
                 extinction=extinction, geometry=geometry.__dict__.copy(), opacity=opacity, v_kms=v_kms,
                 line_wave=lw, line_flux=Fm, line_spectrum=line, continuum=cont)
    return spec, truth
