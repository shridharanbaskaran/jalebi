"""jalebi.cube — line maps and velocity maps from JWST IFU cubes.

Most Class II disks are unresolved in the continuum, so the maps that matter show *extended line
emission* (winds, jets, H2, [Ne II], [Fe II], CO) after the point source is removed.  The steps:

1. per-spaxel local continuum: a low-order polynomial over the line-free channels on both sides of
   each line (`continuum.spaxel_continuum`);
2. point-source removal: the continuum image next to the line, scaled in the PSF core, is subtracted
   plane by plane (`psf.subtract_point_sources`);
3. moment 0/1/2 with S/N masks, and a Gaussian centroid per spaxel with Monte Carlo errors for velocity
   maps (`maps.line_maps`); several lines of one species can be stacked (`maps.stack_lines`);
4. channel maps and position–velocity cuts (`channels`);
5. region spectra: a circle, ellipse or polygon -> spectrum over all sub-bands -> the normal slab fit
   (`regions.region_spectrum`);
6. line-ratio maps and masks from an empty-sky circle or a Background2D background (`ratios`, `masks`).

The recipe of the old cube_maps.py (a +-0.1 um slab, aspls continuum, 9 / 5 / 2+2-channel moment 0) is
`prepare_line(..., window_um=0.1, continuum={"method": "aspls"})` + `line_maps(..., component="full")`;
`jalebi.cube.cube_maps` keeps its functions (make_moment0, make_ratio_plot, plot_moment0_map, ...) with the
same arguments, file names and numbers.

Python::

    from jalebi import cube
    cs = cube.CubeSet("example:HV_Tau_C_cube")            # or a folder of *_s3d.fits
    lc = cube.prepare_line(cs, "[Fe II] 5.34")
    m  = cube.line_maps(lc, zero_point="star")
    m.write("maps/", distance_pc=140)
    spec = cube.region_spectrum(cs, cube.CircleRegion(ra, dec, 0.5))

Terminal: ``jalebi cube info|lines|maps|channels|pv|region|stack|run|synth|demo``; web app: the *Cube* tab.
"""
# Submodules load on first use, so `jalebi --help` and `import jalebi.cube.cli` stay fast.
_EXPORTS = {
    "io": ["Cube", "CubeSet", "read_cube", "resolve_center", "write_cutouts", "get_channel", "channel_band", "band_channel"],
    "continuum": ["CubeContinuumSettings", "spaxel_continuum"],
    "psf": ["PSFSettings", "subtract_point_sources", "find_companions"],
    "maps": ["LineCube", "LineMaps", "prepare_line", "line_maps", "moments", "moment_window", "gaussian_velocity",
             "stack_lines", "reproject_cube", "native_factor"],
    "channels": ["ChannelMaps", "ChannelSlices", "PVDiagram", "channel_maps", "channel_slices", "pv_diagram"],
    "masks": ["rms_mask", "region_level", "background2d", "background_mask", "circle_mask"],
    "ratios": ["RatioMap", "ratio_map", "ratio_from_cubes", "reproject_image"],
    "regions": ["CircleRegion", "EllipseRegion", "PolygonRegion", "AnnulusRegion", "offset_region", "parse_ds9",
                "region_from_dict", "region_spectrum", "to_ds9_file"],
    "config": ["CubeConfig", "example_config"],
    "pipeline": ["run_cube", "CubeRun", "quick_maps"],
    "synthetic": ["make_synthetic_cube", "make_synthetic_cube_set"],
}
_WHERE = {name: mod for mod, names in _EXPORTS.items() for name in names}
__all__ = sorted(_WHERE)


def __getattr__(name):
    import importlib
    if name in _WHERE:
        return getattr(importlib.import_module(f".{_WHERE[name]}", __name__), name)
    if name in _EXPORTS or name in ("plots", "cli", "app", "cube_maps"):
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module 'jalebi.cube' has no attribute {name!r}")


def __dir__():
    return __all__ + list(_EXPORTS) + ["plots", "cli", "app", "cube_maps"]
