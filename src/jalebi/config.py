"""Validated YAML configuration shared by the CLI, the batch runner and the web app."""
from __future__ import annotations

import os
import re

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
    method: str = "irsqr"               # irsqr | median_sg | asls | aspls | convex_hull | rolling_min | spline | banzatti | none
                                        # | given (keep the continuum loaded with the spectrum, e.g. a CSV
                                        #   'continuum'/'baseline' column).  For absorption-dominated
                                        #   spectra use irsqr with quantile ~0.9 (upper envelope) or given.
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
    aspls_lam: float = 5e6              # aspls stiffness (the cube_maps.py value)
    aspls_alpha: float = 0.5            # aspls asymmetry coefficient
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
    kind: str = "slab"                  # slab | annuli | absorption
    group: str | None = None
    tie_to: str | None = None
    ratio: float | None = None
    q: float = 0.5
    p: float = 1.0
    logRin: float = -1.5
    n_annuli: int = 20
    fc: float = 1.0                     # absorption: covering fraction of the continuum (0-1)
    covers: str = "continuum"           # absorption: continuum | all (also absorbs the emission components)
    fwhm_thermal: bool = False          # add the thermal width at T in quadrature to fwhm
    Tvib: float | None = None           # vibrational temperature (K) of a two-temperature population; null = LTE.
                                        # Set it (e.g. 600) to fit T_vib: the 5-8 um H2O nu2 band is sub-thermal
    windows: list[list[float]] | None = None   # ranges (um) where this component emits; null = everywhere.
                                        # e.g. [[4.9, 9.0]] for a ro-vibrational H2O component fitted on its own band
    enabled: bool = True
    linelist_release: str | None = None      # None = project default (linedata.releases[molecule], else "hitran")
    eup_max: float | None = None
    linelist_path: str | None = None
    bounds: dict[str, list[float]] = Field(default_factory=dict)   # per-parameter [lo, hi]
    priors: dict[str, list[float]] = Field(default_factory=dict)   # Gaussian priors [mu, sigma] on top of the bounds,
                                                                   # e.g. {T: [800, 150]} (Romero-Mirza+2024 use 400 / 800 K)
    fixed: list[str] = Field(default_factory=list)                 # parameter names held fixed

    def to_component(self) -> Component:
        d = self.model_dump(exclude={"bounds", "fixed", "priors"})
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
    # sampler (0.16): "stretch" = emcee default; "de" = 80 % DEMove + 20 % DESnookerMove (shorter
    # autocorrelation times for 15-30 correlated parameters); "de+stretch" = 60/20/20
    moves: str = "stretch"
    de_gamma: float = 1.0               # DE step relative to 2.38/sqrt(2 ndim); < 1 raises the acceptance
    init: str = "ball"                  # ball: theta0 + ball x prior range; scaled: theta0 + local posterior widths
    blocks: str = "joint"               # joint | auto: sample groups of components that share no pixel separately


class DetectConfig(BaseModel):
    """Automatic molecule detection (jalebi.detect) run before the fit when fit.auto_detect is set."""
    threshold: float = 10.0                    # ΔBIC above which a candidate counts as detected
    candidates: list[str] = Field(default_factory=list)   # empty = every molecule with a cached line list
    replace_windows: bool = True               # set fit.windows from the detected molecules
    keep_undetected: bool = False              # keep configured components whose molecule was not tested
    oversample: int = 2
    mode: str = "both"                         # emission | absorption | both: test emitting slabs and/or absorbing screens


class FitConfig(BaseModel):
    windows: list[list[float]] = Field(default_factory=list)
    # fit only narrow regions around curated lines (Banzatti+2025 lists; see jalebi.regions) instead of
    # every pixel of `windows`; `windows` then only limits the range.  e.g. [H2O_v0-0]
    line_regions: list[str] = Field(default_factory=list)
    region_pad_um: float = 0.0
    region_weight_beyond: list[float] | None = None     # [lambda_um, weight]: weight regions beyond lambda (Temmink+2025: [20, 5])
    # with line_regions, how the pixels of the OTHER (non-H2O) components are chosen:
    #   features: their curated Q-branch / band-head ranges (molecules.FEATURES) widened by region_feature_pad_um
    #   default:  their full default windows (molecules.DEFAULT_WINDOWS; re-admits water pixels, e.g. CO2 14.6-16.4)
    #   none:     only the line regions
    region_other_molecules: str = "features"
    region_feature_pad_um: float = 0.15
    auto_detect: bool = False                  # detect molecules and rewrite components before fitting
    detect: DetectConfig = DetectConfig()
    area_param: str = "logR"            # logR | logNA
    fit_noise_scale: bool = False
    fit_rv: bool = False
    fit_fwhm: bool = False
    ordering: list[list[str]] = Field(default_factory=list)   # [[hotter, colder], ...]
    # prior bounds for every component of a molecule, e.g. {CO: {T: [100, 3000]}} (components written by the
    # detection included); a component's own `bounds` still win.  Default bounds: jalebi.fit.DEFAULT_BOUNDS
    bounds_by_molecule: dict[str, dict[str, list[float]]] = Field(default_factory=dict)
    window_weights: dict[int, float] = Field(default_factory=dict)
    use_pipeline_err: bool = False
    oversample: int = 6
    tvib_below_trot: bool = True        # prior T_vib <= T for components with a free Tvib (sub-thermal vibration)
    # Default split of the water components at this wavelength (um) when the fit windows reach below it: H2O
    # slabs without their own `windows` and without `Tvib` emit only beyond it (pure-rotational lines), and
    # components whose name contains "rovib" only below it (the sub-thermal nu2 band, Banzatti+2025).  null = off.
    water_split_um: float | None = 9.5
    stages: list[str] = ["grid", "optimise", "mcmc"]
    grid: GridConfig = GridConfig()
    optimise: OptimiseConfig = OptimiseConfig()
    mcmc: MCMCConfig = MCMCConfig()


class LineDataConfig(BaseModel):
    releases: dict[str, str] = Field(default_factory=dict)     # molecule -> release tag
    data_dir: str | None = None


# Results go to one folder per source: "{target}" is replaced by the (file-system safe) target name.
DEFAULT_OUTPUT = "results/{target}"
_LEGACY_OUTPUTS = {"results", "./results", "results/"}      # old default: now also one folder per source
_SIMBAD_PREFIX = re.compile(r"^(V\*|\*\*|\*|NAME)\s+", re.I)


def safe_name(name: str | None) -> str:
    """Target name -> folder name: 'V* FZ Tau' -> 'FZ_Tau', 'DR Tau (A)' -> 'DR_Tau_A'."""
    s = _SIMBAD_PREFIX.sub("", str(name or "").strip())
    s = re.sub(r"[^A-Za-z0-9+\-.]+", "_", s).strip("_.")
    return s or "target"


class ProjectConfig(BaseModel):
    target: TargetConfig = TargetConfig()
    continuum: ContinuumConfig = ContinuumConfig()
    masks: MaskConfig = MaskConfig()
    components: list[ComponentConfig] = Field(default_factory=list)
    fit: FitConfig = FitConfig()
    linedata: LineDataConfig = LineDataConfig()
    output: str = DEFAULT_OUTPUT         # "{target}" -> source name, e.g. results/FZ_Tau
    R_model: str = "argyriou2023"          # argyriou2023 | jones2023 (pontoppidan2024 = legacy alias)
    R_scale: float = 1.0
    R_constant: float | None = None        # a constant resolving power instead of R_model

    @field_validator("components")
    @classmethod
    def unique_names(cls, v):
        names = [c.name for c in v]
        if len(names) != len(set(names)):
            raise ValueError(f"component names must be unique: {names}")
        from .model import KINDS
        for c in v:
            if c.kind not in KINDS:
                raise ValueError(f"component {c.name}: kind must be one of {KINDS}, got {c.kind!r}")
            if c.covers not in ("continuum", "all"):
                raise ValueError(f"component {c.name}: covers must be 'continuum' or 'all'")
            if not 0.0 <= c.fc <= 1.0:
                raise ValueError(f"component {c.name}: fc must be between 0 and 1")
            if c.windows is not None:
                for w in c.windows:
                    if len(w) != 2 or not w[0] < w[1]:
                        raise ValueError(f"component {c.name}: windows must be [[lo, hi], ...] in micron, got {c.windows}")
            if c.Tvib is not None and c.Tvib <= 0:
                raise ValueError(f"component {c.name}: Tvib must be positive (K) or null")
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

    # ---- output folders ----------------------------------------------------------------------
    def target_name(self, fallback: str | None = None) -> str:
        """The source name used for folders: target.name, or `fallback` (e.g. the FITS TARGNAME) when
        target.name was left at its default."""
        n = (self.target.name or "").strip()
        return n if n and n != "target" else (fallback or n or "target")

    def output_dir(self, target_name: str | None = None) -> str:
        """Folder this run writes to.  '{target}' in `output` is replaced by the source name; the old
        default 'results' also gets one sub-folder per source; any other path is used as written."""
        out = (self.output or DEFAULT_OUTPUT).strip()
        if out.rstrip("/\\") in {o.rstrip("/") for o in _LEGACY_OUTPUTS}:
            out = os.path.join(out.rstrip("/\\"), "{target}")
        return os.path.normpath(os.path.expanduser(out.replace("{target}", safe_name(self.target_name(target_name)))))

    def per_target_output(self) -> bool:
        """True when `output` already makes one folder per source ('{target}' or the old default)."""
        out = (self.output or "").strip()
        return "{target}" in out or out.rstrip("/\\") in {o.rstrip("/") for o in _LEGACY_OUTPUTS}

    def output_root(self) -> str:
        """Common parent of the per-source folders (where batch runs put population.csv)."""
        out = (self.output or DEFAULT_OUTPUT).strip()
        if "{target}" in out:
            out = out.split("{target}")[0]
        return os.path.normpath(os.path.expanduser(out.rstrip("/\\") or "."))

    def window_weights(self) -> dict[int, float]:
        """Explicit fit.window_weights, plus the automatic weight beyond `region_weight_beyond[0]` um."""
        w = {int(k): float(v) for k, v in self.fit.window_weights.items()}
        if self.fit.region_weight_beyond:
            from .regions import region_weights
            for k, v in region_weights(self.windows(), tuple(self.fit.region_weight_beyond)).items():
                w.setdefault(k, v)
        return w

    def components_list(self) -> list[Component]:
        comps = [c.to_component() for c in self.components]
        return apply_water_split(comps, self.windows(), self.fit.water_split_um)

    def windows(self) -> list[tuple[float, float]]:
        if self.fit.line_regions:
            from .regions import line_regions
            lim = None
            if self.fit.windows:
                lim = (min(w[0] for w in self.fit.windows), max(w[1] for w in self.fit.windows))
            ws = line_regions(self.fit.line_regions, pad_um=self.fit.region_pad_um, limit=lim)
            # the other molecules keep their own pixels (Q branches / band heads, or their default windows)
            mode = (self.fit.region_other_molecules or "features").lower()
            if mode != "none":
                from .molecules import DEFAULT_WINDOWS, FEATURES
                pad = float(self.fit.region_feature_pad_um)
                for c in self.components:
                    if c.molecule == "H2O" or not c.enabled:
                        continue
                    if mode == "default":
                        ws += [tuple(w) for w in DEFAULT_WINDOWS.get(c.molecule, [])]
                    else:
                        ws += [(a - pad, b + pad) for a, b, _ in FEATURES.get(c.molecule, [])] or \
                              [tuple(w) for w in DEFAULT_WINDOWS.get(c.molecule, [])]
                from .model import merge_intervals
                ws = merge_intervals(ws)
            if self.fit.windows and ws:
                # keep only the parts inside the explicit windows
                from .model import merge_intervals
                ws = merge_intervals([(max(a, w[0]), min(b, w[1])) for a, b in ws for w in self.fit.windows if a <= w[1] and b >= w[0]])
            if ws:
                return ws
        if self.fit.windows:
            return [tuple(w) for w in self.fit.windows]
        from .molecules import DEFAULT_WINDOWS
        ws = []
        for c in self.components:
            ws += DEFAULT_WINDOWS.get(c.molecule, [])
        from .model import merge_intervals
        return merge_intervals(ws) if ws else [(4.9, 28.0)]


def is_rovib_name(name: str) -> bool:
    return "rovib" in name.lower() or "ro-vib" in name.lower() or name.lower().endswith("_vib")


def apply_water_split(comps: list[Component], windows: list[tuple[float, float]], split: float | None) -> list[Component]:
    """Default wavelength windows for water slabs when the fit reaches below `split` um (the nu2 band).

    LTE slabs fitted to the rotational lines over-predict the 5-8 um band by 3-6x (docs/ROVIB_WATER.md), so
    an H2O component that has no `windows` of its own and no `Tvib` is restricted to [split, 28.5] um, and a
    component whose name says "rovib" to [4.9, split].  Components with explicit `windows`, with `Tvib`
    (whose two-temperature populations handle the band), or of kind absorption are left alone.  Tied
    isotopologues follow their parent.  Returns the same list, modified in place."""
    if split is None or not windows or min(w[0] for w in windows) >= split:
        return comps
    by_name = {c.name: c for c in comps}
    for c in comps:
        if c.molecule != "H2O" or c.kind == "absorption" or c.windows is not None or c.Tvib is not None:
            continue
        c.windows = [[4.9, float(split)]] if is_rovib_name(c.name) else [[float(split), 28.5]]
    for c in comps:
        if c.tie_to and c.windows is None and c.tie_to in by_name and by_name[c.tie_to].windows is not None:
            c.windows = [list(w) for w in by_name[c.tie_to].windows]
    return comps


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
    output=DEFAULT_OUTPUT,
)
