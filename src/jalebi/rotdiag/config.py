"""YAML configuration of a rotation-diagram run (the web app's *Rotation diagram* module writes the same file).

    jalebi rotdiag init rd.yaml --example h2      # write an example
    jalebi rotdiag run rd.yaml                    # find, measure, fit (+ MCMC), write results/{target}/rotdiag/H2

Smallest config:

    molecule: H2
    spectrum: {path: example:FZ_Tau}

or, to fit published / your own fluxes instead of a spectrum:

    molecule: H2
    fluxes: {path: my_h2_fluxes.csv, unit: W m-2}      # columns: label (S(1) ...) or wave, flux, err
"""
from __future__ import annotations

from typing import Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .physics import Geometry


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SpectrumSource(_M):
    """Where the spectrum comes from.

    source: s3d  — the IFU cubes (preferred): the flux summed over a region of the sky in every sub-band
                   (`jalebi.cube.region_spectrum`; no aperture correction, no background annulus, so extended
                   emission is kept), the region given as offsets from the source: circle (dx, dy, r) by default.
            x1d  — the pipeline's 1-D extraction (point-source aperture with a background annulus: extended
                   lines can come out negative — avoid for H2 / jets).
            csv | fits — a table or FITS spectrum.  auto: s3d when the folder has s3d cubes, else x1d / csv."""
    path: str = ""
    source: str = "auto"                 # auto | s3d | x1d | csv
    name: Optional[str] = None
    distance_pc: Optional[float] = None
    rv_kms: float = 0.0                  # systemic velocity removed before measuring (the lines' own shift is measured)
    # s3d: the region summed (offsets in arcsec from the source position found in the cubes, or ra/dec)
    region: str = "circle"               # circle | ellipse | annulus | polygon | all (the whole field)
    dx: float = 0.0
    dy: float = 0.0
    radius_arcsec: float = 1.0           # circle radius / ellipse semi-major / annulus outer radius
    params: list[float] = Field(default_factory=list)   # ellipse: [b, pa]; annulus: [r_in]; polygon: [dx1, dy1, dx2, dy2, ...]
    ra: Optional[Union[float, str]] = None       # centre instead of the cube's source position
    dec: Optional[Union[float, str]] = None
    background: Optional[list[float]] = None     # annulus [r_in, r_out] whose median per plane is subtracted
    aperture_fwhm_scale: float = 1.5             # legacy point-source aperture (source: x1d-like extraction from s3d)


class FluxTable(_M):
    path: str = ""
    unit: str = "W m-2"                  # W m-2 | erg s-1 cm-2 | W m-2 sr-1 | erg s-1 cm-2 sr-1 (a scale factor may prefix: "1e-17 erg s-1 cm-2")
    tol_fwhm: float = 0.5                # wavelength matching tolerance (instrumental FWHM)


class LinesConfig(_M):
    wmin: Optional[float] = None
    wmax: Optional[float] = None
    bands: Optional[list[str]] = None    # vibrational bands 'vup-vlow' (None = the molecule's default, [] = all)
    eu_min: Optional[float] = None
    eu_max: Optional[float] = None
    t_ref: Optional[float] = None
    rel_min: Optional[float] = None      # default: the molecule's preset
    max_features: int = 60
    blend_fwhm: float = 0.5
    member_rel: float = 0.01
    edge_fwhm: float = 3.0
    curated: Optional[bool] = None
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    resolving_power: Union[str, float] = "argyriou2023"


class MeasureSection(_M):
    method: str = "gauss"                # gauss | gauss_free | integrate
    velocity: Union[str, float] = "auto"
    width: Union[str, float] = "auto"
    window_fwhm: float = 8.0
    joint_fwhm: float = 2.0
    core_fwhm: float = 1.2
    cont_order: int = 1
    continuum: str = "local"             # local | spectrum
    snr_detect: float = 3.0
    integrate_fwhm: float = 1.2
    contaminants: bool = True
    flux_unit: str = "Jy"
    use: Optional[list[str]] = None      # labels of the features to fit (default: every measured one)
    skip: list[str] = Field(default_factory=list)


class GeometrySection(_M):
    mode: str = "auto"                   # auto (aperture for s3d regions, else number) | number | radius | aperture | intensity
    distance_pc: Optional[float] = None  # default: the spectrum's distance
    R_au: float = 1.0
    aperture_arcsec: float = 0.5
    omega_sr: Optional[float] = None

    def to_geometry(self, distance_pc: float) -> Geometry:
        return Geometry(mode=self.mode, distance_pc=self.distance_pc or distance_pc, R_au=self.R_au,
                        aperture_arcsec=self.aperture_arcsec, omega_sr=self.omega_sr)


class FitSection(_M):
    model: str = "single"                # single | two | powerlaw
    opr: str = "thermal"                 # thermal | species | offset
    opr_value: float = 3.0
    opr_free: bool = True
    av: float = 0.0
    av_free: bool = False
    extinction: str = "KP5"              # KP5 | G23 | HD23 | McClure09 | G23_Rv5.5 | G21 | CT06 | F11 | path/to/curve.csv
    opacity: bool = False
    fwhm_kms: float = 10.0
    fwhm_free: bool = False
    R_free: bool = False
    sys_frac: float = 0.10
    use: str = "all"                     # all | detected
    tmax_powerlaw: float = 4000.0
    bounds: dict[str, list[float]] = Field(default_factory=dict)
    fixed: dict[str, float] = Field(default_factory=dict)
    start: dict[str, float] = Field(default_factory=dict)
    compare: bool = False                # also fit single, two and power law and tabulate BIC
    also: list[str] = Field(default_factory=list)   # further models fitted (+ MCMC) and drawn next to the main one, e.g. [powerlaw]


class MCMCSection(_M):
    enabled: bool = True
    walkers: int = 48
    steps: int = 3000
    burn: int = 1000
    thin: int = 1
    seed: int = 42


class RotDiagConfig(_M):
    molecule: str = "H2"
    release: Optional[str] = None
    target: Optional[str] = None         # name used in the output folder (default: the spectrum's name)
    spectrum: SpectrumSource = Field(default_factory=SpectrumSource)
    fluxes: Optional[FluxTable] = None
    lines: LinesConfig = Field(default_factory=LinesConfig)
    measure: MeasureSection = Field(default_factory=MeasureSection)
    geometry: GeometrySection = Field(default_factory=GeometrySection)
    fit: FitSection = Field(default_factory=FitSection)
    mcmc: MCMCSection = Field(default_factory=MCMCSection)
    output: str = "results/{target}/rotdiag/{molecule}"
    plots: bool = True

    @classmethod
    def load(cls, path: str) -> "RotDiagConfig":
        with open(path) as fh:
            d = yaml.safe_load(fh) or {}
        if "rotdiag" in d and isinstance(d["rotdiag"], dict):
            d = d["rotdiag"]
        return cls.model_validate(d)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json", exclude_none=True), sort_keys=False, allow_unicode=True)

    def save(self, path: str):
        with open(path, "w") as fh:
            fh.write(self.to_yaml())

    def output_dir(self, target: str) -> str:
        import re
        safe = re.sub(r"[^A-Za-z0-9_.+-]+", "_", target or "target").strip("_")
        return self.output.replace("{target}", safe).replace("{molecule}", self.molecule)


EXAMPLES = {
    "h2": dict(molecule="H2", target="synthetic_H2", spectrum=dict(path="example:synthetic/rotdiag_H2_synthetic.csv.gz"),
               geometry=dict(mode="number"), fit=dict(model="two", opr="species", av_free=True, compare=True, also=["powerlaw"]),
               mcmc=dict(steps=3000, burn=1000)),
    "h2_fluxes": dict(molecule="H2", target="H2_table", fluxes=dict(path="example:synthetic/rotdiag_H2_fluxes.csv", unit="W m-2"),
                      geometry=dict(mode="number", distance_pc=140.0),
                      fit=dict(model="two", opr="thermal", av_free=True)),
    "hv_tau_c": dict(molecule="H2", target="HV_Tau_C", spectrum=dict(path="example:HV_Tau_C_cube", source="s3d", radius_arcsec=1.0),
                     geometry=dict(mode="aperture"), fit=dict(model="two", opr="species", av_free=True, compare=True, also=["powerlaw"])),
    "co": dict(molecule="CO", spectrum=dict(path="example:FZ_Tau"), lines=dict(bands=["1-0"], max_features=30),
               geometry=dict(mode="radius", R_au=0.3), fit=dict(model="single", opacity=True, fwhm_kms=4.7)),
    "oh": dict(molecule="OH", spectrum=dict(path="example:FZ_Tau"), lines=dict(wmin=13.0, max_features=30),
               geometry=dict(mode="number"), fit=dict(model="two")),
    "h2o": dict(molecule="H2O", spectrum=dict(path="example:FZ_Tau"), measure=dict(method="integrate"),
                geometry=dict(mode="radius", R_au=0.5), fit=dict(model="single", opacity=True, fwhm_kms=4.7)),
}


def example_config(name: str = "h2") -> RotDiagConfig:
    if name not in EXAMPLES:
        raise KeyError(f"unknown example '{name}': {', '.join(EXAMPLES)}")
    return RotDiagConfig.model_validate(EXAMPLES[name])
