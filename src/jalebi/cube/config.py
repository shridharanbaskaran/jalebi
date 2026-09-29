"""YAML configuration of a cube run (the same settings the web app's Cube tab and `jalebi cube run` use).

    jalebi cube init cube.yaml --example hv_tau_c       # write an example
    jalebi cube run cube.yaml                           # maps, stacks, channel maps, PV cuts, regions (+ fits)

Everything has a default, so the smallest config is

    cube:
      path: example:HV_Tau_C_cube
      lines: ["[Fe II] 5.34", "H2 S(1)"]
"""
from __future__ import annotations

from typing import Any, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CubeContinuumConfig(_M):
    method: str = "poly"                 # poly | median | aspls (cube_maps.py) | irsqr (channel_maps notebook) | asls
    order: int = 1
    inner_kms: Optional[float] = None    # default max(250, 2 x instrumental FWHM)
    outer_kms: float = 1200.0
    clip_sigma: float = 3.0
    lam: Optional[float] = None          # baselines: default aspls 5e6, irsqr / asls 1e3
    quantile: float = 0.05
    p: float = 0.02
    nan_policy: str = "omit"             # baselines: omit | propagate (a spaxel with a NaN channel gets none)


class CubePSFConfig(_M):
    enabled: bool = True
    scale: str = "core"                  # core | peak
    core_radius_fwhm: float = 0.5
    psf_radius_fwhm: float = 8.0
    template: str = "continuum"
    template_background: bool = True
    auto_sources: bool = True
    auto_threshold: float = 0.1
    extra_sources: list[list[Union[float, str]]] = Field(default_factory=list)


class CubeMomentsConfig(_M):
    line_kms: Optional[float] = None     # half-width of the moment window, default max(200, 1.5 x FWHM)
    component: Optional[str] = None      # instead: native channels around the line (cube_maps.py) —
                                         # full (centre +-half_full) | slow (+-half_slow) | fast (full - slow)
    half_full: int = 4
    half_slow: int = 2
    min_valid: Optional[int] = None      # finite channels a spaxel needs (default half the window, >= 3; cube_maps.py: 1)
    snr_min: float = 3.0
    channel_clip: float = 0.0
    rms_region: Optional[list[float]] = None   # [x, y, r] pixels of empty sky: also write mom0_masked
    rms_sigma: float = 3.0
    rms_mode: str = "rms"                # rms (nanstd) | mean | median of the circle


class CubeKinematicsConfig(_M):
    enabled: bool = True
    source: Optional[str] = None         # line | extended (default: extended when the point source is removed)
    snr_min: float = 5.0
    n_mc: int = 100
    center_shift_kms: float = 200.0
    sigma_range: list[float] = Field(default_factory=lambda: [0.8, 2.5])
    zero_point: str = "none"             # none | star


class CubeChannelsConfig(_M):
    enabled: bool = False
    vmin: float = -300.0
    vmax: float = 300.0
    dv_kms: Optional[float] = None
    source: str = "extended"
    lines: list[str] = Field(default_factory=list)       # default: all lines
    slices: bool = False                 # also the native channels, one 2-D FITS each (cube_maps.py get_channel_maps)
    slices_component: str = "full"


class CubeRatioConfig(_M):
    """A line-ratio map line1 / line2 (line 2 reprojected onto line 1's spaxels)."""
    lines: list[str] = Field(min_length=2, max_length=2)     # [line 1, line 2]
    name: Optional[str] = None           # default "<tag1>_over_<tag2>"
    sigma_thresh: list[float] = Field(default_factory=lambda: [5.0, 5.0])
    rms_region: Optional[list[float]] = None   # [x, y, r] line-1 pixels (cube_maps.py); None: propagated errors
    rms_mode: str = "rms"
    key: str = "mom0"                    # mom0 | mom0_ext | gflux
    unit: str = "cgs"                    # cgs (flux ratio) | "MJy/sr m" (cube_maps.py's numbers)
    percentile: list[float] = Field(default_factory=lambda: [80.0, 80.0, 80.0])


class CubePVConfig(_M):
    enabled: bool = False
    pa_deg: float = 0.0
    length_arcsec: float = 4.0
    width_arcsec: Optional[float] = None
    vmax_kms: Optional[float] = 600.0
    source: str = "extended"
    lines: list[str] = Field(default_factory=list)


class CubeRegionConfig(_M):
    """A region: `ds9` string, or shape + sky parameters, or shape + offsets from the source."""
    name: str = "region"
    ds9: Optional[str] = None
    shape: str = "circle"                # circle | ellipse | polygon | annulus
    ra: Optional[Union[float, str]] = None
    dec: Optional[Union[float, str]] = None
    offset: Optional[list[float]] = None  # [dx_east, dy_north] arcsec from the source (instead of ra/dec)
    r: Optional[float] = None
    r_in: Optional[float] = None
    r_out: Optional[float] = None
    a: Optional[float] = None
    b: Optional[float] = None
    pa: float = 0.0
    vertices: Optional[list[list[float]]] = None           # [[ra, dec], ...] or offsets with offset_vertices
    offset_vertices: Optional[list[list[float]]] = None    # [[dx, dy], ...] arcsec from the source
    fit: bool = False                    # run the slab fit on this region's spectrum


class CubeConfig(_M):
    path: str = "example:HV_Tau_C_cube"
    name: Optional[str] = None
    distance_pc: float = 140.0
    rv_kms: float = 0.0
    band_offsets_kms: dict[str, float] = Field(default_factory=dict)
    center: Any = "auto"                 # auto | header | peak | [ra, dec]
    lines: list[Union[str, float]] = Field(default_factory=lambda: ["[Ne II] 12.81"])
    stacks: dict[str, list[str]] = Field(default_factory=dict)
    window_kms: float = 1500.0
    window_um: Optional[float] = None    # instead: +-window_um micron (cube_maps.py's dlambda = 0.1)
    band: Optional[str] = None           # cube per line: None (widest margin) | nominal (cube_maps.py get_channel) | "3A"
    dq_mask: bool = True                 # blank DO_NOT_USE pixels (cube_maps.py / spectral_cube did not)
    zero_is_nan: bool = True             # SCI == 0 is "no data" (cube_maps.py / spectral_cube kept zeros)
    smooth_fwhm_pix: float = 0.0
    noise: str = "max"
    continuum: CubeContinuumConfig = Field(default_factory=CubeContinuumConfig)
    psf: CubePSFConfig = Field(default_factory=CubePSFConfig)
    moments: CubeMomentsConfig = Field(default_factory=CubeMomentsConfig)
    kinematics: CubeKinematicsConfig = Field(default_factory=CubeKinematicsConfig)
    channels: CubeChannelsConfig = Field(default_factory=CubeChannelsConfig)
    pv: CubePVConfig = Field(default_factory=CubePVConfig)
    regions: list[CubeRegionConfig] = Field(default_factory=list)
    ratios: list[CubeRatioConfig] = Field(default_factory=list)
    fit_config: Optional[str] = None     # slab-fit YAML used for regions with fit: true
    plot_extent_arcsec: Optional[float] = None
    pa_deg: Optional[float] = None       # jet/disk axis drawn on the maps
    formats: list[str] = Field(default_factory=lambda: ["fits", "png"])
    output: str = "results/{target}/cube"   # like the slab fits: one folder per source ({target} = source name)
    n_jobs: int = 1

    def output_dir(self, target_name: str | None = None) -> str:
        """Folder of this run: '{target}' in `output` replaced by the (file-system safe) source name —
        `name`, or `target_name` (e.g. the cube's TARGNAME) when `name` is not set."""
        import os
        from ..config import safe_name
        out = (self.output or "results/{target}/cube").strip()
        return os.path.normpath(os.path.expanduser(out.replace("{target}", safe_name(self.name or target_name))))

    @classmethod
    def load(cls, path: str) -> "CubeConfig":
        with open(path) as fh:
            d = yaml.safe_load(fh) or {}
        return cls.model_validate(d.get("cube", d))

    def to_yaml(self) -> str:
        return yaml.safe_dump({"cube": self.model_dump(mode="json")}, sort_keys=False, allow_unicode=True)

    def save(self, path: str):
        with open(path, "w") as fh:
            fh.write(self.to_yaml())


EXAMPLES = {
    "hv_tau_c": dict(
        path="example:HV_Tau_C_cube", name="HV Tau C", distance_pc=140.0,
        lines=["[Fe II] 5.34", "H2 S(3)", "H2 S(2)", "[Ne II] 12.81", "H2 S(1)"],
        stacks={"H2": ["H2 S(1)", "H2 S(2)", "H2 S(3)"]},
        kinematics=dict(zero_point="star"),
        channels=dict(enabled=True, lines=["[Fe II] 5.34", "H2 S(1)"], vmin=-200, vmax=200),
        # the [Fe II] / [Ne II] jet runs along PA ~ 25 deg; the H2 emission is elongated along PA ~ 105 deg
        pv=dict(enabled=True, pa_deg=25.0, length_arcsec=5.0, lines=["[Fe II] 5.34", "[Ne II] 12.81"]),
        pa_deg=25.0,
        regions=[dict(name="jet_north", shape="circle", offset=[0.42, 0.91], r=0.35),
                 dict(name="jet_south", shape="circle", offset=[-0.42, -0.91], r=0.35),
                 dict(name="h2_east", shape="ellipse", offset=[1.93, -0.52], a=1.0, b=0.6, pa=105.0),
                 dict(name="h2_west", shape="ellipse", offset=[-1.93, 0.52], a=1.0, b=0.6, pa=105.0)],
        ),
    "synthetic": dict(
        path="synthetic_cube", name="SYNTH-DISK", distance_pc=140.0,
        lines=["H2 S(1)", "H2 S(2)", "H2 S(3)", "[Ne II] 12.81"], stacks={"H2": ["H2 S(1)", "H2 S(2)", "H2 S(3)"]},
        channels=dict(enabled=True, lines=["H2 S(1)"]), pv=dict(enabled=True, pa_deg=30.0, lines=["H2 S(1)"]),
        pa_deg=30.0),
    # the recipe of the old cube_maps.py (RECIPES["cube_maps"]) + masks from its empty-sky circle, a ratio
    # map and channel slices; plus velocities and errors
    "cube_maps": dict(
        path="example:HV_Tau_C_cube", name="HV Tau C", distance_pc=140.0,
        lines=["[Fe II] 5.34", "[Ne II] 12.81", "H2 S(1)"],
        window_um=0.1, band="nominal", dq_mask=False, zero_is_nan=False,
        continuum=dict(method="aspls", lam=5e6, nan_policy="propagate"),
        psf=dict(enabled=False),
        moments=dict(component="full", min_valid=1, rms_region=[12, 8, 3], rms_sigma=3.0),   # cube_maps.py's RMS circle
        channels=dict(enabled=True, lines=["[Fe II] 5.34"], slices=True, source="line"),
        ratios=[dict(lines=["[Fe II] 5.34", "[Ne II] 12.81"], sigma_thresh=[5, 5], rms_region=[12, 8, 3], unit="MJy/sr m")],
        pa_deg=25.0),
    "blank": dict(path="/path/to/target_folder_with_s3d_cubes", lines=["[Ne II] 12.81", "H2 S(1)"]),
}


# settings that reproduce an older method exactly (`jalebi cube maps --recipe cube_maps`, the web app's button)
RECIPES = {
    "cube_maps": {"window_um": 0.1, "band": "nominal", "dq_mask": False, "zero_is_nan": False,
                  "continuum": {"method": "aspls", "lam": 5e6, "nan_policy": "propagate"},
                  "psf": {"enabled": False},
                  "moments": {"component": "full", "min_valid": 1}},
}


def apply_recipe(cfg: CubeConfig, name: str) -> CubeConfig:
    """Set the fields of RECIPES[name] on `cfg` (in place); other fields are left as they are."""
    if name not in RECIPES:
        raise ValueError(f"unknown recipe {name!r}: {', '.join(RECIPES)}")
    for k, v in RECIPES[name].items():
        if isinstance(v, dict):
            sub = getattr(cfg, k)
            for kk, vv in v.items():
                setattr(sub, kk, vv)
        else:
            setattr(cfg, k, v)
    return cfg


def example_config(name: str = "hv_tau_c") -> CubeConfig:
    return CubeConfig.model_validate(EXAMPLES[name])
