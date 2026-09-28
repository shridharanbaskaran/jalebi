# Contributing to JALEBI

Thank you for helping. Bug reports, new molecules, continuum methods, samplers and documentation fixes are all welcome.

## Set up a development copy

```bash
git clone https://github.com/shridharanbaskaran/jalebi.git
cd jalebi
python install.py --extras app,fetch,dev        # editable install: your edits take effect immediately
pytest -q                                        # about 20 s, offline
ruff check src tests examples install.py
```

## Where things live

| Folder | What |
| --- | --- |
| `src/jalebi/` | the package (physics in `model.py`, `instrument.py`, `linedata.py`; fitting in `fit.py`, `pipeline.py`, `detect.py`; interfaces in `cli.py`, `app.py`) |
| `src/jalebi/linedata/` | bundled line lists (Parquet) |
| `src/jalebi/example_data/` | FZ Tau x1d files and the synthetic spectrum |
| `src/jalebi/_requirements.py` | the single list of requirements; keep it in sync with `pyproject.toml` (a test checks this) |
| `examples/` | example scripts, configs and notebook (also shipped inside the wheel) |
| `tests/` | pytest suite |

## Guidelines

- Keep the physics tested. A new model feature needs a test against an analytic limit, like the thin-line
  flux and flux-conservation tests in `tests/test_core.py`.
- The same YAML config must work in the CLI, the Python API and the web app. New options go in `config.py`.
- Units are µm, cm⁻², K, au and Jy everywhere.
- Add an entry to `CHANGELOG.md`.

## Adding a molecule

1. Add it to `MOLECULES` in `molecules.py` (HITRAN molecule and isotopologue number, mass, label, colour;
   `parent` and `default_ratio` for isotopologues).
2. Add `DEFAULT_WINDOWS` and, if it has a Q-branch or band head, `FEATURES`.
3. Fetch or import its line list: `jalebi linedata fetch <MOL>` or `jalebi linedata import <MOL> file.par --release <tag>`.
4. For detection, set a template column in `detect._TEMPLATE_LOGN`.

## Releasing

1. Update the version in `src/jalebi/__init__.py` and `CITATION.cff`, and add a CHANGELOG entry.
2. `python -m build && twine check dist/*`
3. Tag and publish a GitHub release. The `publish` workflow uploads it to PyPI.
