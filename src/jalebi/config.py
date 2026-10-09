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
    workers: int = 1                    # keep 1: a parallel scipy DE updates its population in a different order
    polish: bool = True
    seed: int = 0
    # 0.22: robust start.  n_starts DE runs with seeds seed, seed+1, ...; the first starts at the config values
    # (the 0.21 behaviour), the second at `seed_points` when given, the rest from random populations.  The lowest
    # -2 ln P wins; every start's chi2 and theta go to diagnostics.json ("optimise").  n_starts: 1 = 0.21 behaviour.
    n_starts: int = 3
    # literature-typical starting values per component, e.g. {H2O_hot: {T: 900, logN: 18.5}, H2O_warm: {T: 500,
    # logN: 18.0}, H2O_cold: {T: 250, logN: 17.8}}; parameters not listed keep the config value
    seed_points: dict[str, dict[str, float]] = Field(default_factory=dict)
    # possible multimodality: two starts within `multimodal_dchi2` of the best -2 ln P whose solutions differ by
    # more than `multimodal_dT` K in a T or `multimodal_dlogN` dex in a log N are flagged (diagnostics.json, log)
    multimodal_dchi2: float = 10.0
    multimodal_dT: float = 100.0
    multimodal_dlogN: float = 0.5


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
    # linear parameters (0.17): the emitting areas R^2 of slab units enter the model linearly
    #   sample      = MCMC parameters like the others (default, the 0.16 behaviour)
    #   profile     = solved by bounded NNLS at every likelihood call (DuCKLinG, Kaeufer+2024)
    #   marginalise = integrated out analytically under a broad Gaussian prior
    # The chain still holds every parameter (areas filled in per sample); see jalebi.linear
    linear: str = "sample"
    linear_prior: str = "log"           # marginalise: log (Gaussian marginal + Jacobian of a uniform-in-log R prior, as
                                        # sample uses) | gaussian (broad Gaussian on R^2 only: pulls N-unconstrained
                                        # components to small N / large R -- CO on AS 209 went to log N 13.3)
    linear_prior_scale: float | None = None   # marginalise: sigma of the Gaussian prior on R^2 [au^2]; None = R_max^2
    # 0.18: one ln P call for all walkers (emcee vectorize=True). Pays off with model_backend: emulator (no
    # process pool then: processes is ignored)
    vectorize: bool = False


class DynestyConfig(BaseModel):
    """fit.dynesty: dynamic nested sampling (jalebi.nested), used when fit.sampler is dynesty."""
    nlive: int = 500                    # live points of the initial run (and of each batch)
    sample: str = "rslice"              # rslice (default) | rwalk | slice | unif | auto
    bound: str = "multi"
    dynamic: bool = True                # dynamic nested sampling (Higson+2019); false = static (evidence only)
    dlogz_init: float = 0.5
    pfrac: float = 0.8                  # weight of the posterior (vs the evidence) in the dynamic batches
    n_effective: int | None = None      # stop at this posterior ESS (None = dynesty default)
    maxcall: int | None = None
    processes: int = 1                  # dynesty's own process pool
    seed: int = 0
    slices: int | None = None           # rslice: default 3 + ndim
    walks: int | None = None            # rwalk: default 25
    # Delta ln Z of removing each of these components (refit without it; areas always sampled for this, since a
    # profiled area has no prior volume and would bias ln Z): evidence.csv next to detections.csv
    evidence_without: list[str] = Field(default_factory=list)
    checkpoint: bool = True             # 0.22: write dynesty.save in the disk folder and resume from it (fit.resume)
    checkpoint_every: float = 60.0      # seconds between saves


class EmulatorConfig(BaseModel):
    """fit.emulator: precomputed (T, log N) tables of the slab fluxes on the data's pixels (jalebi.emulator)."""
    target_sigma: float = 0.1           # max |emulator - exact| / sigma at the largest area the data allow
    target_flux: float = 1e-3           # max relative error of the integrated flux
    safety: float = 0.5                 # nodes are added until the checks are below safety x target
    method: str = "cubic"               # cubic (4-point Lagrange per axis) | linear
    n_start: list[int] = Field(default_factory=lambda: [9, 9])
    max_nodes: list[int] = Field(default_factory=lambda: [257, 257])
    n_validate: int = 200               # random (T, log N) checks after a build
    cache_dir: str | None = None        # None = $JALEBI_EMULATOR_DIR, else ~/.jalebi/emulator
    rebuild: bool = False               # ignore cached tables
    # 0.21: shared = one table per molecule for the whole survey (LSF-convolved spectrum on a dense grid, resampled
    # onto each disk's pixels at load, spot-checked against the exact model; jalebi.emulator_shared) | per_disk =
    # the 0.18 tables on the data's own pixels (rebuilt per disk and per release)
    cache: str = "shared"
    ref_snr: float = 1000.0             # shared: certified against sigma = (node's peak, scaled to f_ref) / ref_snr
    points_per_fwhm: float = 8.0        # shared: dense-grid points per LSF FWHM
    table_oversample: int = 6           # shared: fine-grid points per line FWHM when a table is built (independent of
                                        # fit.oversample: at 3 the exact model's line profiles depend on the phase of
                                        # its fine grid at the 1e-3 level, which no shared table can reproduce)
    read_only: bool = False             # shared: never build (compute nodes; also $JALEBI_EMULATOR_READONLY=1)
    spot_check: int = 200               # shared: random (T, log N) per unit checked on the disk at load (0 = off)
    # 0.22.1: when a unit's spot check misses the targets, fall back to the exact model only if that can be the
    # better model.  relative (default): fall back only when the emulator's error exceeds the (lower bound of the)
    # error of the fit's own exact model at fit.oversample (0.21 survey: 110 of 111 fallbacks were to a model
    # >= 13x less accurate, at ~15x the time) | absolute: the 0.21 rule (any miss of the targets) | never
    fallback: str = "relative"
    bands: list[str] | None = None      # shared: MRS sub-bands of the dense grid (null = all 12; tests and experiments)

    def settings(self, cfg=None):
        """EmulatorSettings; with the ProjectConfig, the survey-wide (T, log N) boxes per molecule
        (emulator_shared.molecule_boxes) are filled in so that a fit looks up the same shared tables a
        `jalebi emulator build` of that config wrote."""
        from .emulator import EmulatorSettings
        d = self.model_dump()
        d["n_start"] = tuple(d["n_start"]); d["max_nodes"] = tuple(d["max_nodes"])
        d["bands"] = tuple(d["bands"]) if d.get("bands") else None
        s = EmulatorSettings(**d)
        if cfg is not None and s.cache == "shared":
            from .emulator_shared import molecule_boxes
            s.boxes = molecule_boxes(cfg)
        return s


class ContinuumFitConfig(BaseModel):
    """0.22: options of the joint continuum correction (fit.continuum_fit: offset | spline; docs/CONTINUUM.md).
    The correction c(lambda) is added to the continuum inside the fit: offset = one constant per sub-band,
    spline = a cubic B-spline per sub-band with knots every `knot_spacing_um`.  Its coefficients are LINEAR
    parameters handled like the emitting areas (jalebi.linear): marginalised under a Gaussian prior of width
    `prior_width` (x the median continuum of the sub-band for prior: continuum, x the noise for prior: noise),
    or profiled (least squares at every likelihood call) with mode: profile.  The optimiser always profiles."""
    knot_spacing_um: float = 1.0        # spline: knot spacing per sub-band [micron]
    prior: str = "continuum"            # continuum | noise: what prior_width is relative to
    prior_width: float = 0.02           # Gaussian prior sigma of each coefficient (2 % of the continuum by default)
    mode: str = "marginalise"           # marginalise | profile (the MCMC); the optimiser profiles either way


class CornerCheckConfig(BaseModel):
    """0.22: corner / pseudo-continuum diagnostic per component (jalebi.corner), after the fit."""
    enabled: bool = True
    bound_frac: float = 0.02            # a parameter within this fraction of its prior range of a bound is "at the bound"
    min_pinned: int = 2                 # flag "pinned" at this many bounds or more
    smooth_window_um: float = 0.3       # running-median window of the low-pass filter [micron]
    smooth_threshold: float = 0.5       # flag "pseudo-continuum" when more than this share of the flux survives
    # molecules the pseudo-continuum flag applies to (the smooth fraction is reported for all).  Q-branch molecules
    # (CO2, C2H2, HCN, C4H2, C6H6) legitimately put most of their flux into a 0.05-0.1 um band head that a 0.3 um
    # running median keeps, so by default only water (where the hot-water corner occurs) is flagged; [] = all
    smooth_molecules: list[str] = ["H2O"]


class DetectionProbConfig(BaseModel):
    """0.22: detection probability with the continuum varied (jalebi detect-prob, jalebi.detection_prob)."""
    mode: str = "ensemble"              # ensemble | bayesian
    n_variants: int = 30                # ensemble: plausible continua (the first is the nominal one)
    seed: int = 0
    offset_sigma: float = 0.01          # ensemble: global multiplicative continuum offset ~ N(0, sigma)
    methods: list[str] = Field(default_factory=list)   # ensemble: extra continuum methods to mix in
    quantile_range: list[float] = [0.05, 0.2]          # irsqr quantile
    knot_spacing_range: list[int] = [15, 60]           # irsqr knot spacing [pixels]
    median_window_range: list[int] = [51, 201]         # median_sg window [pixels]
    median_percentile_range: list[float] = [10.0, 40.0]
    lam_range_dex: list[float] = [-1.0, 1.0]           # asls / aspls stiffness x 10^U(range)
    threshold: float = 10.0             # Delta BIC for "detected"
    robust_frac: float = 0.95           # class robust at >= this detection fraction (or P(present))
    absent_frac: float = 0.05           # class "not detected" at <= this (and not detected at the nominal continuum)
    de_maxiter: int = 40                # ensemble: short DE pass per variant from the best-fit start
    de_popsize: int = 8
    mcmc_nsteps: int = 0                # ensemble: short MCMC per variant (0 = optimum only)
    prior_odds: float = 1.0             # bayesian: prior odds of presence
    cache_variants: bool = True         # ensemble: detection_prob_variants.json in the disk folder, reused on rerun


class ReportConfig(BaseModel):
    """0.22: what the summary reports (report.*)."""
    co: str = "full"                    # full | NA_only: for CO report only log(N.A) (T and N are degenerate when
                                        # the T prior reaches 3000 K: hot, thin solutions); the fit is unchanged


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
    # 0.18: exact (the full forward model at every call) | emulator (precomputed tables for the units that allow
    # it, exact for the rest; built or loaded from the cache before the fit -- see jalebi.emulator)
    model_backend: str = "exact"
    emulator: EmulatorConfig = EmulatorConfig()
    # 0.19: Laplace (Gaussian) errors at the optimum after the optimiser: laplace.json + laplace_corner.png
    laplace: bool = False
    # 0.20: emcee (default; fit.mcmc) | dynesty (dynamic nested sampling, fit.dynesty; ln Z in diagnostics.json)
    sampler: str = "emcee"
    dynesty: DynestyConfig = DynestyConfig()
    # 0.22: stage checkpoints in the disk folder (detection.json, grid.json, de_pass{i}.json, continuum_refined.npz,
    # chain.h5 / dynesty.save), each keyed on the data, the config and the model version (jalebi.resume).
    # auto = continue from the last valid stage (a crashed or killed run restarts where it stopped; a longer
    # fit.mcmc.nsteps continues the chain) | off = ignore existing checkpoints (they are still written)
    resume: str = "auto"
    # 0.22: joint continuum correction inside the fit (docs/CONTINUUM.md): none (default; the 0.21 behaviour) |
    # offset (one additive constant per sub-band) | spline (B-spline per sub-band, fit.continuum_correction)
    continuum_fit: str = "none"
    continuum_correction: ContinuumFitConfig = ContinuumFitConfig()
    corner_check: CornerCheckConfig = CornerCheckConfig()
    detection_prob: DetectionProbConfig = DetectionProbConfig()


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
    report: ReportConfig = ReportConfig()   # 0.22
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
