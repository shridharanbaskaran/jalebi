"""`jalebi source …`: open a target folder once and see what is in it (the terminal side of the Source page)."""
from __future__ import annotations

from typing import Optional

import typer
from rich import print as rprint
from rich.table import Table

source_app = typer.Typer(help="Target folders (x1d + s3d cubes): list them, inspect one, remember its distance / RV / position.")


@source_app.command("list")
def list_sources(root: str = typer.Argument(..., help="data root (folder of target folders)"),
                 depth: int = typer.Option(2, help="folder levels to search")):
    """Target folders and spectrum tables under ROOT."""
    from .source import scan_sources
    found = scan_sources(root, depth=depth)
    t = Table(title=f"{len(found)} sources under {root}")
    for c in ("source", "x1d", "s3d", "path"):
        t.add_column(c)
    for e in found:
        t.add_row(e.name, str(e.n_x1d or ""), str(e.n_s3d or ""), e.path)
    rprint(t)


@source_app.command("info")
def info(path: str = typer.Argument(..., help="source folder or file (or example:FZ_Tau)"),
         preload: bool = typer.Option(False, "--preload", help="read all cubes (in parallel) and report the time"),
         workers: int = typer.Option(4, help="parallel cube readers for --preload"),
         no_dq: bool = typer.Option(False, "--no-dq", help="keep DQ DO_NOT_USE pixels"),
         keep_zeros: bool = typer.Option(False, "--keep-zeros", help="SCI = 0 is data, not undefined")):
    """What a source contains: sub-bands (x1d / s3d), header facts (programme, CAL_VER, CRDS context), position."""
    import time
    from .source import open_source
    src = open_source(path, dq_mask=not no_dq, zero_is_nan=not keep_zeros)
    if preload and src.has_cubes:
        t0 = time.time()
        src.preload(workers=workers)
        rprint(f"[green]read {len(src.cubes.loaded())} cubes in {time.time() - t0:.1f} s with {workers} readers "
               f"({src.memory_mb():.0f} MB)[/green]")
    rprint(src.describe())
    cov = src.coverage()
    t = Table(title="MRS sub-bands")
    for c in ("band", "x1d", "s3d", "λ range [µm]", "cube (z×y×x)"):
        t.add_column(c)
    for r in cov.itertuples():
        if r.x1d or r.s3d:
            t.add_row(r.band, "✓" if r.x1d else "", "✓" if r.s3d else "",
                      f"{r.wmin_um:.3f}–{r.wmax_um:.3f}" if r.s3d else "", r.cube)
    rprint(t)


@source_app.command("set")
def set_settings(path: str = typer.Argument(..., help="source folder or file"),
                 distance: Optional[float] = typer.Option(None, "--distance", help="distance [pc]"),
                 rv: Optional[float] = typer.Option(None, "--rv", help="systemic RV [km/s]"),
                 ra: Optional[str] = typer.Option(None, help="source RA (deg or hh:mm:ss) for apertures / regions"),
                 dec: Optional[str] = typer.Option(None, help="source Dec (deg or ±dd:mm:ss)"),
                 name: Optional[str] = typer.Option(None, help="display name"),
                 clear: bool = typer.Option(False, "--clear", help="forget all remembered settings first")):
    """Remember distance, RV, position or name for a source (jalebi_source.yaml next to the data, or ~/.jalebi/sources/)."""
    from .source import open_source
    src = open_source(path)
    if clear:
        src.settings = {}
    kw = {k: v for k, v in dict(distance_pc=distance, rv_kms=rv, ra=ra, dec=dec, name=name).items() if v is not None}
    f = src.save_settings(**kw)
    rprint(f"[green]written {f}[/green]")
    rprint(src.describe())


@source_app.command("spectrum")
def spectrum(path: str = typer.Argument(..., help="source folder or file"),
             out: str = typer.Argument(..., help="output CSV / .h5"),
             source: str = typer.Option("auto", help="x1d | s3d (aperture on the cubes) | auto"),
             aperture: float = typer.Option(1.5, help="s3d aperture radius [× FWHM]"),
             apcorr: str = typer.Option("mrs", help="s3d aperture correction: mrs | gaussian | none | x1d")):
    """Write the 1-D spectrum of a source (what the LTE slab fit uses)."""
    from .source import open_source
    src = open_source(path)
    spec = src.spectrum(source, dict(aperture_fwhm_scale=aperture, apcorr=apcorr))
    spec.save(out)
    rprint(f"[green]{spec.name}: {len(spec.wave)} pixels in {', '.join(spec.bands)} → {out}[/green]")
