# Putting JALEBI on GitHub (and PyPI)

This guide takes the `jalebi/` folder from your computer to a public GitHub repository, then optionally to PyPI
so people can `pip install jalebi`. The GitHub account is `shridharanbaskaran`, and the repository address
`https://github.com/shridharanbaskaran/jalebi` is already filled in to `pyproject.toml`, `CITATION.cff`,
`README.md` and `CONTRIBUTING.md`.

## 0. Before you start

- A GitHub account, and `git` on your machine (`git --version`).
- Optional: the GitHub CLI `gh` (https://cli.github.com), which creates the repository from the terminal.
- Check that the name is free: https://github.com/shridharanbaskaran/jalebi should give a 404, and
  https://pypi.org/project/jalebi/ should not exist yet. It was free on PyPI on 2026-09-28.

## 1. Username

Already done: every file uses `shridharanbaskaran`. If you ever move the repository to another account
or organisation, replace it everywhere with

```bash
grep -rl "shridharanbaskaran" . --include=*.md --include=*.toml --include=*.cff --include=*.yml | \
  xargs sed -i 's/shridharanbaskaran/new-name/g'       # macOS: sed -i '' 's/.../.../g'
```

## 1b. Restore the `.github` folder, if needed

The CI workflows and the issue templates must be in a folder called `.github`. If your copy has a
`dot-github/` folder instead, as the copy written to your computer by Claude does (hidden `.github` paths
cannot be written remotely), rename it:

```bash
mv dot-github .github
```

## 2. Check that everything works locally

```bash
python install.py --extras app,fetch,dev     # or: pip install -e ".[app,fetch,dev]"
pytest -q                                    # 54 tests, ~20 s
ruff check src tests examples install.py
jalebi doctor
```

## 3. Create the repository and push

With the GitHub CLI:

```bash
git init -b main
git add .
git commit -m "JALEBI 0.9.0: first public release"
gh repo create jalebi --public --source . --push \
   --description "JWST Analysis of Line Emission with Bayesian Inference: simultaneous LTE slab fitting of JWST/MIRI disk spectra"
```

Or in the browser: create an **empty** repository called `jalebi` at https://github.com/new (no README, no
license, no .gitignore, since this folder already has them), then run

```bash
git init -b main
git add .
git commit -m "JALEBI 0.9.0: first public release"
git remote add origin https://github.com/shridharanbaskaran/jalebi.git
git push -u origin main
```

The repository is about 30 MB (129 files), mostly the line lists and the FZ Tau example spectrum. The largest single
file is `H2O_hitemp.parquet` (19 MB), well below GitHub's 100 MB per-file limit, so Git LFS is not needed.
`.gitattributes` marks the Parquet, FITS and NPZ files as binary.

## 4. On GitHub

- **About** (the gear icon on the repository page): add the description and the topics `jwst`, `miri`,
  `protoplanetary-disks`, `astronomy`, `spectroscopy`, `mcmc`, `python`.
- **Actions**: the `tests` workflow runs on every push (Linux and macOS, Python 3.10–3.14). A green tick
  on the first push means the package installs and passes its tests on a clean machine.
- **Cite this repository** appears automatically because of `CITATION.cff`.
- Optional: turn on **Discussions** for user questions.

## 5. A release, and a DOI

1. **Releases → Draft a new release**, tag `v0.9.0`, title `JALEBI 0.9.0`, paste the 0.9.0 section of
   `CHANGELOG.md`, then publish.
2. For a citable DOI, connect the repository at https://zenodo.org/account/settings/github/ **before**
   publishing the release. Zenodo archives every release and assigns a DOI. Add the DOI badge to the
   README and the `doi:` field to `CITATION.cff`.

## 6. Optional: publish on PyPI

The `publish` workflow uploads a release to PyPI with *trusted publishing* (no password or token stored
on GitHub). The one-time setup:

1. Create an account at https://pypi.org and turn on two-factor authentication.
2. Open https://pypi.org/manage/account/publishing/ → *Add a new pending publisher*: project `jalebi`,
   owner `shridharanbaskaran`, repository `jalebi`, workflow `publish.yml`, environment `pypi`.
3. On GitHub: **Settings → Environments → New environment** named `pypi`.
4. Publish a GitHub release (step 5). The workflow builds the wheel and uploads it, and
   `pip install jalebi` then works for everyone. The wheel includes the line lists, the example data and
   the example scripts (`jalebi examples ./dir`).

To upload by hand instead: `python -m build && twine upload dist/*`. Test first with
`twine upload --repository testpypi dist/*`.

## 7. Sending it to people

- Send them to the repository: the README starts with `git clone … && python install.py`.
- Or send the file `jalebi-0.9.0.tar.gz` (the sdist): `pip install jalebi-0.9.0.tar.gz`, or unpack it and run
  `python install.py`.
- After a PyPI release: `pip install "jalebi[app]"`.

## 8. Later releases

1. Bump `__version__` in `src/jalebi/__init__.py` and `version:` in `CITATION.cff`, then add a CHANGELOG entry.
2. `git commit -am "Release 0.9.1" && git tag v0.9.1 && git push --tags`.
3. Publish the release on GitHub. PyPI and Zenodo update themselves.
