"""Validated YAML configuration shared by the CLI, the batch runner and the web app."""
from __future__ import annotations

import os

import yaml
from pydantic import BaseModel, Field, field_validator

from .continuum import DEFAULT_PROTECTED
from .model import Component


class ExtractionConfig(BaseModel):
    """How to get the 1D spectrum from a target folder."""
    source: str = "x1d"                 # x1d (pipeline extraction) | s3d (aperture photometry on the cubes at ra, dec)
    ra: str | float | None = None       # aperture centre, required for s3d: degrees or sexagesimal ("04:32:31.76")
    dec: str | float | None = None      # degrees or sexagesimal ("+24:20:03.0")
    aperture_fwhm_scale: float = 1.5    # radius = scale * FWHM(lambda), FWHM = 0.033 lambda + 0.106 arcsec
    aperture_arcsec: float | None = None      # fixed radius instead
    annulus_arcsec: list[float] | None = None # [r_in, r_out] background annulus, null = none
    apcorr: str = "mrs"                 # mrs | gaussian | none | x1d

    def radec_deg(self) -> tuple[float, float] | None:
        if self.ra is None or self.dec is None:
            return None
        from .data import parse_radec
        return parse_radec(self.ra, self.dec)


class TargetConfig(BaseModel):
    name: str = "target"
    path: str = ""                      # folder of x1d/s3d files, one FITS file, or a CSV
    distance_pc: float = 140.0
    rv_kms: float = 0.0                 # heliocentric radial velocity to remove
    spike_filter: bool = True
    extraction: ExtractionConfig = ExtractionConfig()


class ContinuumConfig(BaseModel):
    method: str = "irsqr"               # irsqr | median_sg | asls | convex_hull | rolling_min | spline | banzatti | none
    protect: bool = True
    protected: dict[str, list[float]] = Field(default_factory=lambda: {k: list(v) for k, v in DEFAULT_PROTECTED.items()})
    smooth: int = 0
    quantile: float = 0.1
    knot_spacing: int = 25
    median_window: int = 101
    median_percentile: float = 25.0
    sg_window: int = 51
    sg_order: int = 2
    n_iter: int = 5
    lam: float = 1e3
    p: float = 0.01
    segment: int = 300
    overlap: int = 100
    percentile: float = 5.0
    min_window: int = 61
    anchors: list[float] = Field(default_factory=list)
    anchor_width: int = 3
    refine_iterations: int = 0          # model-aware refinement passes after the first fit


class MaskConfig(BaseModel):
    default_lines: bool = True          # H I, H2, [Ne II] ... from continuum.DEFAULT_MASKS
    oh_prompt: bool = True              # OH prompt lines at 9-13 um
    oh_whole_range: bool = False
    extra: dict[str, list[float]] = Field(default_factory=dict)


class ComponentConfig(BaseModel):
    name: str
    molecule: str
    logN: float = 17.0
    T: float = 500.0
    logR: float = -0.5
    rv: float = 0.0
    fwhm: float = 4.7
    kind: str = "slab"
    group: str | None = None
    tie_to: str | None = None
    ratio: float | None = None
    q: float = 0.5
    p: float = 1.0
    logRin: float = -1.5
    n_annuli: int = 20
    enabled: bool = True
    linelist_release: str | None = None      # None = project default (linedata.releases[molecule], else "hitran")
    eup_max: float | None = None
    linelist_path: str | None = None
    bounds: dict[str, list[float]] = Field(default_factory=dict)   # per-parameter [lo, hi]
    fixed: list[str] = Field(default_factory=list)                 # parameter names held fixed

    def to_component(self) -> Component:
        d = self.model_dump(exclude={"bounds", "fixed"})
        return Component(**d)


class GridConfig(BaseModel):
    logN: list[float] = [14.0, 20.0, 25]        # lo, hi, n
    T: list[float] = [150.0, 1200.0, 22]
    order: list[str] = Field(default_factory=list)   # component order (MINDS-style sequential); default: config order
    n_jobs: int = 1


class OptimiseConfig(BaseModel):
    method: str = "de"                  # de | nelder
    maxiter: int = 150
    popsize: int = 12
    workers: int = 1
    polish: bool = True
    seed: int = 0


class MCMCConfig(BaseModel):
    nwalkers: int | None = None         # default 4 * ndim (min 32)
    nsteps: int = 3000
    processes: int = 1
    seed: int = 0
    ball: float = 0.01
    thin_by: int = 1
    checkpoint: bool = True


class DetectConfig(BaseModel):
    """Automatic molecule detection (jalebi.detect) run before the fit when fit.auto_detect is set."""
    threshold: float = 10.0                    # ΔBIC above which a candidate counts as detected
    candidates: list[str] = Field(default_factory=list)   # empty = every molecule with a cached line list
    replace_windows: bool = True               # set fit.windows from the detected molecules
    keep_undetected: bool = False              # keep configured components whose molecule was not tested
    oversample: int = 2


class FitConfig(BaseModel):
    windows: list[list[float]] = Field(default_factory=list)
    auto_detect: bool = False                  # detect molecules and rewrite components before fitting
    detect: DetectConfig = DetectConfig()
    area_param: str = "logR"            # logR | logNA
    fit_noise_scale: bool = False
    fit_rv: bool = False
    fit_fwhm: bool = False
    ordering: list[list[str]] = Field(default_factory=list)   # [[hotter, colder], ...]
    window_weights: dict[int, float] = Field(default_factory=dict)
    use_pipeline_err: bool = False
    oversample: int = 6
    stages: list[str] = ["grid", "optimise", "mcmc"]
    grid: GridConfig = GridConfig()
    optimise: OptimiseConfig = OptimiseConfig()
    mcmc: MCMCConfig = MCMCConfig()


class LineDataConfig(BaseModel):
    releases: dict[str, str] = Field(default_factory=dict)     # molecule -> release tag
    data_dir: str | None = None


class ProjectConfig(BaseModel):
    target: TargetConfig = TargetConfig()
    continuum: ContinuumConfig = ContinuumConfig()
    masks: MaskConfig = MaskConfig()
    components: list[ComponentConfig] = Field(default_factory=list)
    fit: FitConfig = FitConfig()
    linedata: LineDataConfig = LineDataConfig()
    output: str = "results"
    R_model: str = "argyriou2023"          # argyriou2023 | jones2023 (pontoppidan2024 = legacy alias)
    R_scale: float = 1.0
    R_constant: float | None = None        # a constant resolving power instead of R_model

    @field_validator("components")
    @classmethod
    def unique_names(cls, v):
        names = [c.name for c in v]
        if len(names) != len(set(names)):
            raise ValueError(f"component names must be unique: {names}")
        return v

    # ---- io -------------------------------------------------------------------------------
    @classmethod
    def load(cls, path: str) -> "ProjectConfig":
        with open(path) as fh:
            d = yaml.safe_load(fh) or {}
        cfg = cls.model_validate(d)
        if cfg.linedata.data_dir:
            os.environ["JALEBI_DATA"] = os.path.expanduser(cfg.linedata.data_dir)
        return cfg

    def save(self, path: str):
        with open(path, "w") as fh:
            yaml.safe_dump(self.model_dump(mode="json"), fh, sort_keys=False, allow_unicode=True)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False, allow_unicode=True)

    def components_list(self) -> list[Component]:
        return [c.to_component() for c in self.components]

    def windows(self) -> list[tuple[float, float]]:
        if self.fit.windows:
            return [tuple(w) for w in self.fit.windows]
        from .molecules import DEFAULT_WINDOWS
        ws = []
        for c in self.components:
            ws += DEFAULT_WINDOWS.get(c.molecule, [])
        from .model import merge_intervals
        return merge_intervals(ws) if ws else [(4.9, 28.0)]


EXAMPLE_CONFIG = ProjectConfig(
    target=TargetConfig(name="FZ Tau", path="/path/to/V-FZ-TAU_jw01549-o001_t001_miri", distance_pc=130.0, rv_kms=20.0),
    continuum=ContinuumConfig(method="irsqr"),
    components=[
        ComponentConfig(name="H2O_hot", molecule="H2O", logN=18.0, T=750.0, logR=-0.6),
        ComponentConfig(name="H2O_warm", molecule="H2O", logN=17.5, T=400.0, logR=-0.2),
        ComponentConfig(name="CO2", molecule="CO2", logN=17.5, T=500.0, logR=-0.5),
        ComponentConfig(name="13CO2", molecule="13CO2", tie_to="CO2", ratio=70.0),
        ComponentConfig(name="C2H2", molecule="C2H2", logN=17.0, T=500.0, logR=-0.7),
        ComponentConfig(name="HCN", molecule="HCN", logN=16.5, T=500.0, logR=-0.7),
    ],
    fit=FitConfig(windows=[[13.5, 16.5], [16.5, 17.5]], ordering=[["H2O_hot", "H2O_warm"]],
                  mcmc=MCMCConfig(nwalkers=64, nsteps=3000, processes=8)),
    output="results/FZ_Tau",
)
