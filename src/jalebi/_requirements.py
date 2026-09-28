"""Single list of JALEBI's requirements (standard library only).

Read by `jalebi doctor` and by the interactive installer (`install.py`), which runs before the
package or any of its dependencies are installed.  Keep in sync with pyproject.toml
(tests/test_packaging.py checks that they agree).

Each entry: distribution name -> (minimum version, import name, what it is used for).
"""
from __future__ import annotations

PYTHON_MIN = (3, 10)

CORE = {
    "numpy":         ("1.24", "numpy", "arrays"),
    "scipy":         ("1.11", "scipy", "sparse operators, NNLS, differential evolution"),
    "pandas":        ("2.0", "pandas", "tables"),
    "astropy":       ("5.3", "astropy", "FITS input, units, coordinates"),
    "pyarrow":       ("14", "pyarrow", "Parquet line-list cache"),
    "PyYAML":        ("6.0", "yaml", "configuration files"),
    "pydantic":      ("2.5", "pydantic", "validated configs"),
    "emcee":         ("3.1", "emcee", "MCMC sampler"),
    "corner":        ("2.2", "corner", "corner plots"),
    "matplotlib":    ("3.7", "matplotlib", "figures"),
    "pybaselines":   ("1.1", "pybaselines", "continuum baselines (IRSQR, ASLS)"),
    "joblib":        ("1.3", "joblib", "parallel grids and batch runs"),
    "threadpoolctl": ("3.0", "threadpoolctl", "one BLAS thread per worker"),
    "h5py":          ("3.8", "h5py", "MCMC checkpoints (HDF5)"),
    "typer":         ("0.9", "typer", "command-line interface"),
    "rich":          ("13.0", "rich", "terminal output"),
    "tqdm":          ("4.66", "tqdm", "progress bars"),
    "hitran-api":    ("1.2", "hapi", "TIPS-2021 partition functions, HITRAN downloads"),
}

OPTIONAL = {
    "app": {
        "panel": ("1.4", "panel", "interactive web app"),
        "bokeh": ("3.4", "bokeh", "interactive plots"),
    },
    "fetch": {
        "astroquery": ("0.4.7", "astroquery", "download new HITRAN line lists"),
    },
    "notebook": {
        "jupyterlab": ("4.0", "jupyterlab", "notebooks"),
        "ipykernel": ("6.25", "ipykernel", "Jupyter kernel"),
        "ipympl": ("0.9", "ipympl", "interactive matplotlib in notebooks"),
    },
    "jwst": {
        "jwst": ("1.16", "jwst", "STScI pipeline (reduce MIRI data yourself)"),
    },
    "dev": {
        "pytest": ("7.0", "pytest", "test suite"),
        "ruff": ("0.4", "ruff", "linting"),
        "build": ("1.0", "build", "building wheels"),
        "twine": ("5.0", "twine", "uploading to PyPI"),
    },
}

# what the installer offers, in order, with a default and a short description
EXTRAS_MENU = [
    ("app", True, "Web app (Panel + Bokeh): live sliders, continuum clicking, corner plots"),
    ("fetch", True, "HITRAN downloads (astroquery): fetch line lists beyond the bundled ones"),
    ("notebook", False, "Jupyter Lab + kernel 'Python (JALEBI)' for the example notebook"),
    ("jwst", False, "STScI jwst pipeline (large; only if you reduce MIRI data yourself)"),
    ("dev", False, "Developer tools: pytest, ruff, build, twine"),
]


def parse_version(v: str) -> tuple:
    """Loose version -> comparable tuple of ints ('2.1.0rc1' -> (2, 1, 0))."""
    out = []
    for part in str(v).replace("-", ".").split("."):
        num = ""
        for ch in part:
            if ch.isdigit():
                num += ch
            else:
                break
        if num == "":
            break
        out.append(int(num))
    return tuple(out) or (0,)


def version_ok(installed: str | None, minimum: str) -> bool:
    if installed is None:
        return False
    a, b = parse_version(installed), parse_version(minimum)
    n = max(len(a), len(b))
    return a + (0,) * (n - len(a)) >= b + (0,) * (n - len(b))
