"""Synthetic MIRI-MRS spectra with known answers.

Use them to check that a fit recovers the parameters it was given (injection–recovery), to try
continuum methods, or to learn the package without any data:

>>> from jalebi.synthetic import make_synthetic_spectrum
>>> spec, truth = make_synthetic_spectrum(seed=1)
>>> spec.name, len(spec.wave), [c["name"] for c in truth["components"]]

The spectrum is built exactly like the fitted model (LTE slabs -> MIRI LSF -> pixel integration),
on the pixel grid of the MRS cubes (constant wavelength step per channel), plus a smooth continuum
and Gaussian noise.  Each sub-band is modelled on its own, so overlaps between sub-bands are kept
as in real pipeline products.
"""
from __future__ import annotations

import numpy as np

from .instrument import _BANDS

# spectral step of the MRS Level-3 cubes per channel (micron), jwst pipeline defaults
CUBE_STEP = {"1": 0.0008, "2": 0.0013, "3": 0.0025, "4": 0.006}

# default truth: a warm water-rich inner disk with the usual organics (MIRI channel 3)
DEFAULT_COMPONENTS = [
    {"name": "H2O", "molecule": "H2O", "logN": 18.0, "T": 550.0, "logR": -0.3, "linelist_release": "hitran"},
    {"name": "CO2", "molecule": "CO2", "logN": 17.5, "T": 450.0, "logR": -0.7},
    {"name": "13CO2", "molecule": "13CO2", "tie_to": "CO2", "ratio": 70.0},
    {"name": "C2H2", "molecule": "C2H2", "logN": 16.8, "T": 650.0, "logR": -0.9},
    {"name": "HCN", "molecule": "HCN", "logN": 16.6, "T": 750.0, "logR": -0.8},
]
DEFAULT_BANDS = ("3A", "3B", "3C")


def continuum_model(wave, level: float = 0.35, slope: float = 0.8, wiggle: float = 0.03, pivot: float = 14.0):
    """Smooth dust-like continuum (Jy): a power law with a gentle silicate-like wiggle."""
    w = np.asarray(wave, float)
    return level * (w / pivot) ** slope * (1.0 + wiggle * np.sin(2 * np.pi * (w - pivot) / 6.0))


def band_pixels(band: str) -> np.ndarray:
    """Pixel wavelengths of one MRS sub-band at the cube sampling."""
    lo, hi = _BANDS[band]
    step = CUBE_STEP[band[0]]
    return np.arange(lo, hi, step)


def make_synthetic_spectrum(components: list[dict] | None = None, bands=DEFAULT_BANDS, distance_pc: float = 140.0,
                            snr: float = 150.0, continuum: dict | None = None, seed: int = 0, oversample: int = 6,
                            name: str = "synthetic disk", linelists: dict | None = None):
    """Return (Spectrum, truth) for LTE slab components on MRS pixels.

    components : list of dicts with the Component fields (name, molecule, logN, T, logR, tie_to, ratio,
                 linelist_release, ...); default DEFAULT_COMPONENTS
    bands      : MRS sub-bands to simulate (default 3A, 3B, 3C = 11.55–17.98 µm)
    snr        : continuum signal-to-noise per pixel (noise sigma = continuum / snr, constant per band)
    continuum  : keyword arguments of `continuum_model`
    The truth dict holds the components, distance, noise and the noiseless line and continuum spectra.
    """
    from .data import Spectrum
    from .model import Component, build_model

    comps_d = [dict(c) for c in (components or DEFAULT_COMPONENTS)]
    comps = [Component(**c) for c in comps_d]
    rng = np.random.default_rng(seed)
    W, F, E, B, L, Cc = [], [], [], [], [], []
    for band in bands:
        wave = band_pixels(band)
        m = build_model(comps, wave, distance_pc, [(float(wave[0]), float(wave[-1]))], linelists=linelists,
                        oversample=oversample)
        line = m.evaluate()
        cont = continuum_model(wave, **(continuum or {}))
        sigma = np.full(len(wave), float(np.median(cont)) / snr)
        W.append(wave); L.append(line); Cc.append(cont); E.append(sigma)
        F.append(cont + line + rng.normal(0.0, sigma))
        B.append(np.full(len(wave), band, dtype=object))
    wave = np.concatenate(W)
    spec = Spectrum(wave, np.concatenate(F), np.concatenate(E), np.concatenate(B).astype(str), name=name,
                    distance_pc=distance_pc, rest_frame=True,
                    meta={"synthetic": True, "seed": seed, "snr": snr})
    truth = {"components": comps_d, "distance_pc": distance_pc, "snr": snr, "seed": seed, "bands": list(bands),
             "line_flux": np.concatenate(L), "continuum": np.concatenate(Cc)}
    return spec, truth


def save_synthetic(path_csv: str, spec, truth, path_truth: str | None = None):
    """Write the spectrum as a CSV the loaders read (wave, flux, err, band; true continuum and line flux
    as extra columns) and the true parameters as YAML."""
    import pandas as pd
    import yaml
    df = pd.DataFrame({"wave": spec.wave, "flux": spec.flux, "err": spec.err, "band": spec.band,
                       "continuum_true": truth["continuum"], "line_flux_true": truth["line_flux"]})
    with open(path_csv, "w") as fh:
        fh.write(f"# name={spec.name.replace(' ', '_')} distance_pc={spec.distance_pc} rv_kms=0.0 rest_frame=True\n")
        df.to_csv(fh, index=False, float_format="%.7g")
    if path_truth:
        with open(path_truth, "w") as fh:
            yaml.safe_dump({k: v for k, v in truth.items() if k not in ("line_flux", "continuum")}, fh, sort_keys=False)


def truth_table(truth: dict):
    """The true parameters as a small DataFrame (one row per component)."""
    import pandas as pd
    return pd.DataFrame(truth["components"])
