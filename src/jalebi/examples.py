"""Bundled example data and the example scripts.

* ``example_path(name)`` — path of a bundled data set, e.g. ``example_path("FZ_Tau")`` (the 12 MIRI-MRS
  x1d files of FZ Tau) or ``example_path("synthetic/synthetic_miri_ch3.csv")``.
* Any config or CLI path written as ``example:<name>`` is resolved the same way, so the example
  configs run from any folder:  ``target: {path: example:FZ_Tau}``.
* ``copy_examples(dest)`` copies the example scripts, configs and notebook to a folder of yours
  (``jalebi examples ./my_examples`` from the terminal).
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
DATA_DIR = PKG_DIR / "example_data"
PREFIX = "example:"


def list_examples() -> list[str]:
    """Names of the bundled example data sets (folders and files under example_data/)."""
    if not DATA_DIR.exists():
        return []
    out = []
    for p in sorted(DATA_DIR.iterdir()):
        if p.name.startswith((".", "_")):
            continue
        out.append(p.name)
        if p.is_dir():
            out += [f"{p.name}/{q.name}" for q in sorted(p.iterdir()) if q.is_file() and not q.name.endswith((".fits", ".fits.gz"))]
    return out


def example_path(name: str = "") -> str:
    """Absolute path of a bundled example data set or file (``""`` = the example_data folder)."""
    name = name[len(PREFIX):] if name.startswith(PREFIX) else name
    p = DATA_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"no bundled example '{name}'. Available: {', '.join(list_examples())}")
    return str(p)


def resolve_path(path: str | os.PathLike | None) -> str:
    """Expand ``~`` and environment variables, and map ``example:<name>`` to the bundled data."""
    if path is None:
        return ""
    s = str(path)
    if s.startswith(PREFIX):
        return example_path(s)
    return os.path.expandvars(os.path.expanduser(s))


def examples_source() -> Path | None:
    """Folder holding the example scripts: inside the installed wheel (``jalebi/_examples``) or,
    for a source checkout / editable install, the repository's top-level ``examples/``."""
    for cand in (PKG_DIR / "_examples", PKG_DIR.parents[1] / "examples"):
        if cand.is_dir() and any(cand.glob("*.py")):
            return cand
    return None


def copy_examples(dest: str | os.PathLike, overwrite: bool = False) -> list[str]:
    """Copy the example scripts, configs and notebooks to `dest`; returns the files written."""
    src = examples_source()
    if src is None:
        raise FileNotFoundError("example scripts not found in this installation; get them from the GitHub repository "
                                "(examples/ folder)")
    dest = Path(dest).expanduser()
    written = []
    for f in sorted(src.rglob("*")):
        if f.is_dir() or "__pycache__" in f.parts or f.name.startswith(".") or "results" in f.relative_to(src).parts:
            continue
        out = dest / f.relative_to(src)
        if out.exists() and not overwrite:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out)
        written.append(str(out))
    return written
