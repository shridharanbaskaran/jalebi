# Version control and releases

JALEBI lives in **one** git repository (`LTE_fitting/jalebi/`, remote `origin` =
`https://github.com/shridharanbaskaran/jalebi`). Versions are **tags** in that repository, not copies of
the folder, and work in progress lives on **branches**. This page is the routine for every release,
written out for 0.10.1 (the `jalebi.cube` release, on top of 0.9.1 = `fc48114`, tag `v0.9.1`).

## The model in one paragraph

`main` always holds a working, tested JALEBI. New work goes on a branch (`feature/...`, `fix/...`), is
committed in small steps, pushed, checked by CI (`.github/workflows/tests.yml` runs on every push and pull
request), and merged into `main`. When `main` is ready to ship, bump the version, tag it `vX.Y.Z`, push the
tag, and publish a GitHub release from it; `.github/workflows/publish.yml` then uploads the wheel to PyPI
and Zenodo (if connected) archives it with a DOI. Any old version is one command away:
`git switch --detach v0.9.0`.

## Once, after moving to the `jalebi/` folder

The folder was copied from `jalebi_v0.9.0/`, `.git` included, so it has the full history and the same
remote. Two things to fix:

```bash
cd ~/Desktop/Work/LTE_fitting/jalebi
git status                     # should list the 0.10.1 changes, nothing else
git log --oneline -3           # fc48114 0.9.1: one results folder per source · c7b7c1f JALEBI 0.9.0

# the virtual environment inside was built for the old path: re-point the editable install
source .venv-jalebi/bin/activate          # or: conda activate jalebi
python -m pip install -e ".[app,dev]"     # (or: python install.py --env current --extras app,dev)
python -c "import jalebi; print(jalebi.__version__, jalebi.__file__)"   # 0.10.1 .../LTE_fitting/jalebi/src/jalebi/...
```

Then **retire `jalebi_v0.9.0/`**: rename it (`mv jalebi_v0.9.0 jalebi_v0.9.0_archive`) or delete it, and
never commit or push from it. It holds its own copy of `.git`; pushing from two folders is how histories
diverge. The tag `v0.9.0` (below) is the permanent record of that version.

## Releasing 0.10.1

```bash
cd ~/Desktop/Work/LTE_fitting/jalebi

# 1. work on a branch (optional for a one-person project, but CI then checks it before main changes)
git switch -c feature/cube-maps

# 2. check locally
pytest -q                                   # 103 tests, ~1.5 min
ruff check src tests examples install.py
jalebi doctor && jalebi cube demo           # the HV Tau C maps land in jalebi_cube_demo/

# 3. commit
git add -A
git status                                  # review: no results/, no .venv, no large files
git commit -m "0.10.1: jalebi.cube — line maps, velocity maps, channel maps, PV cuts, region spectra"

# 4. push the branch and let CI run
git push -u origin feature/cube-maps
#    GitHub → "Compare & pull request" → wait for the green tick → "Merge pull request"
#    or from the terminal:  gh pr create --fill && gh pr merge --merge
#    (no PR wanted? merge locally:  git switch main && git merge --no-ff feature/cube-maps && git push)

# 5. tag the release on main
git switch main && git pull
git tag -a v0.9.0 c7b7c1f -m "JALEBI 0.9.0"            # once: tag the first release retroactively
git tag -a v0.10.1 -m "JALEBI 0.10.1: jalebi.cube"
git push origin v0.9.0 v0.9.1 v0.10.1                    # v0.9.1 exists locally; this makes sure it is on GitHub

# 6. publish the release (triggers PyPI upload via publish.yml, and Zenodo if connected)
gh release create v0.10.1 --title "JALEBI 0.10.1" --notes "$(awk '/^## \[0.10.1\]/{f=1;next} /^## \[0.9.1\]/{f=0} f' CHANGELOG.md)"
#    or in the browser: Releases → Draft a new release → choose tag v0.10.1 → paste the CHANGELOG section
```

Authentication: `gh auth login` once (HTTPS + browser), or a personal access token as the password when
git asks. `git push` over SSH works too if the remote is `git@github.com:shridharanbaskaran/jalebi.git`.

## Every later release

1. `git switch -c feature/<what>` → work → `git commit` often → `git push`.
2. Bump the version in `src/jalebi/__init__.py` (`__version__`) and `CITATION.cff` (`version`,
   `date-released`); add a `## [X.Y.Z] — date` section at the top of `CHANGELOG.md`.
3. Merge into `main`, then `git tag -a vX.Y.Z -m "JALEBI X.Y.Z"` and `git push origin vX.Y.Z`.
4. Publish the GitHub release for the tag.

Version numbers follow [Semantic Versioning](https://semver.org): `X.Y.Z` = major.minor.patch. A release
that adds a feature bumps the minor number (`jalebi.cube` made 0.9 → 0.10) and a bug-fix-only release
bumps the patch (0.9.1 was one). The version lives in `src/jalebi/__init__.py` (`__version__`, which
`jalebi --version`, pip and the wheel name read) and `CITATION.cff`; a commit message or a tag does not
change it, so bump those two files before you tag. Before 1.0 many projects are relaxed about this; pick one rule and keep it.

## Everyday commands

| Task | Command |
| --- | --- |
| what changed | `git status`, `git diff`, `git diff --stat main` |
| undo edits to one file (uncommitted) | `git restore path/to/file` |
| look at an old version | `git switch --detach v0.9.0` (back: `git switch main`) |
| two versions side by side | `git worktree add ../jalebi-0.9.0 v0.9.0` — a second folder sharing the same repository |
| a bug fix on an old release | `git switch -c fix/x v0.10.1` → fix → tag `v0.10.2` |
| see the branches and tags | `git branch -a`, `git tag -l` |
| throw away a local branch | `git branch -d feature/x` (after merging) |

## What goes into git

Code, tests, docs, the bundled line lists and example data under `src/jalebi/` (all binary files are
marked in `.gitattributes`; none is near GitHub's 100 MB limit, so Git LFS is not needed; the largest is
`H2O_hitemp.parquet`, 19 MB). Not in git (see `.gitignore`): virtual environments, `results/`,
`jalebi_demo/`, `jalebi_cube_demo/`, MCMC chains, `install_log.txt`, caches. Keep science data (full
cubes, your target folders) outside the repository.
