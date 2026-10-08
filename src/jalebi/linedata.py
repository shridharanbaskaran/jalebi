"""Molecular line lists: bundled with the package, downloadable, cached as Parquet.

A LineList holds, per transition: wavelength (micron, vacuum), A_ul (s^-1), g_u, E_u and
E_l (K), quantum labels and the HITRAN ids.  Line strengths are computed from A_ul and
g_u, not from HITRAN's S(296 K), so isotopologue columns stay physical (no terrestrial
abundance factor).

Where line lists live
---------------------
1. the *user cache* — ``$JALEBI_DATA`` (or the ``linedata.data_dir`` of a config), default
   ``~/.jalebi/linedata``.  Downloads and imports are written here.
2. the *bundled lists* shipped inside the package (``jalebi/linedata``): HITRAN 2020 for the
   common MIRI molecules plus HITEMP H2O and CO.  Read-only.
A list in the user cache takes precedence over the bundled one of the same molecule and release.

Supported sources
-----------------
* HITRAN through astroquery (`fetch_hitran`) or HAPI (`fetch_hapi`), cached to Parquet.
* Local files: HITRAN 160-character .par records, or the processed .par format used by
  iSLAT (header with a Q(T) table followed by lambda, A, E_up, E_low, g_up, g_low).
* Parquet written by this module.
"""
from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import activity as act
from .constants import CM1_TO_K
from .molecules import Molecule, get_molecule
from .partition import PartitionFunction, partition_for

COLUMNS = ["wave", "nu", "a", "gu", "gl", "eu", "el", "vup", "vlow", "qup", "qlow", "iso"]


BUNDLED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "linedata")


def normalize_vlabel(s) -> str:
    """HITRAN global quanta label without separators: '0 1 0' and '0_1_0' -> '010'."""
    return re.sub(r"[\s_]+", "", str(s)).strip()


def data_dir() -> str:
    """Writable user cache: $JALEBI_DATA (legacy: $SLABFIT_DATA), default ~/.jalebi/linedata."""
    d = os.environ.get("JALEBI_DATA") or os.environ.get("SLABFIT_DATA") or \
        os.path.join(os.path.expanduser("~"), ".jalebi", "linedata")
    d = os.path.expanduser(d)
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:          # read-only location: still usable for reading
        pass
    return d


def search_dirs() -> list[str]:
    """Folders searched for cached line lists, in order of precedence (user cache, bundled)."""
    dirs = [data_dir()]
    if os.path.isdir(BUNDLED_DIR) and os.path.abspath(BUNDLED_DIR) != os.path.abspath(dirs[0]):
        dirs.append(BUNDLED_DIR)
    return dirs


def find_cached(molecule: str, release: str = "hitran") -> str | None:
    """Path of the cached Parquet file for (molecule, release), or None."""
    for d in search_dirs():
        p = os.path.join(d, f"{molecule}_{release}.parquet")
        if os.path.exists(p):
            return p
    return None


def cache_path(molecule: str, release: str = "hitran", write: bool = False) -> str:
    """Where (molecule, release) is cached: the existing file if there is one (unless `write`),
    else the location in the writable user cache."""
    if not write:
        p = find_cached(molecule, release)
        if p:
            return p
    return os.path.join(data_dir(), f"{molecule}_{release}.parquet")


@dataclass
class LineList:
    molecule: Molecule
    table: pd.DataFrame                  # columns as in COLUMNS, sorted by wave
    partition: PartitionFunction
    source: str = ""
    release: str = "hitran"

    # ---- convenience accessors (numpy views) --------------------------------
    @property
    def wave(self):
        return self.table["wave"].to_numpy()

    @property
    def a(self):
        return self.table["a"].to_numpy()

    @property
    def gu(self):
        return self.table["gu"].to_numpy()

    @property
    def eu(self):
        return self.table["eu"].to_numpy()

    @property
    def el(self):
        return self.table["el"].to_numpy()

    def __len__(self):
        return len(self.table)

    # ---- vibrational structure (for T_vib != T_rot models) ----------------------
    def vib_energies(self) -> dict[str, float]:
        """Energy (K) of every vibrational state in the list, keyed by its normalised global
        quanta label ("010", "100", ...): the lowest level energy found with that label among the
        upper and lower levels of the lines.  For HITRAN/HITEMP water this recovers the band
        origins to < 0.1 cm^-1 (000: 0, 010: 1594.7, 020: 3151.6, 100: 3657.0, 001: 3755.9 cm^-1).
        Unassigned labels (HITEMP's "-2-2-2") are left out: their lines stay in LTE at T_rot."""
        cache = getattr(self, "_vib_cache", None)
        if cache is not None and cache[0] is self.table:
            return cache[1]
        t = self.table
        vu = t["vup"].astype(str).map(normalize_vlabel).to_numpy()
        vl = t["vlow"].astype(str).map(normalize_vlabel).to_numpy()
        eu, el = t["eu"].to_numpy(float), t["el"].to_numpy(float)
        out: dict[str, float] = {}
        for labels, e in ((vu, eu), (vl, el)):
            for lab in np.unique(labels):
                if not lab or lab.startswith("-") or not re.fullmatch(r"[0-9A-Za-z+\-]+", lab):
                    continue
                m = labels == lab
                v = float(np.nanmin(e[m]))
                out[lab] = min(out.get(lab, np.inf), v)
        self._vib_cache = (t, out)
        return out

    def vib_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Vibrational energy (K) of the upper and lower level of every line; NaN when the label is
        unassigned (those lines are treated in LTE)."""
        ev = self.vib_energies()
        t = self.table
        vu = t["vup"].astype(str).map(normalize_vlabel).map(ev).to_numpy(float)
        vl = t["vlow"].astype(str).map(normalize_vlabel).map(ev).to_numpy(float)
        return vu, vl

    def select(self, wmin: float | None = None, wmax: float | None = None, eup_max: float | None = None,
               aul_min: float | None = None, vup=None, vlow=None, qup=None, qlow=None) -> "LineList":
        """Return a new LineList restricted by wavelength, energy, A_ul and quantum labels.

        `vup`, `vlow`, `qup`, `qlow` may be a string or a list of strings; whitespace in
        HITRAN labels is ignored when comparing.
        """
        t = self.table
        m = np.ones(len(t), bool)
        if wmin is not None:
            m &= t["wave"].to_numpy() >= wmin
        if wmax is not None:
            m &= t["wave"].to_numpy() <= wmax
        if eup_max is not None:
            m &= t["eu"].to_numpy() <= eup_max
        if aul_min is not None:
            m &= t["a"].to_numpy() >= aul_min
        for col, sel in (("vup", vup), ("vlow", vlow), ("qup", qup), ("qlow", qlow)):
            if sel is None:
                continue
            sels = [sel] if isinstance(sel, str) else list(sel)
            sels = [re.sub(r"\s+", "", s) for s in sels]
            labels = t[col].astype(str).str.replace(r"\s+", "", regex=True).to_numpy()
            m &= np.isin(labels, sels)
        return LineList(self.molecule, t[m].reset_index(drop=True), self.partition, self.source, self.release)

    def strength_cut(self, T_hi: float = 1500.0, T_lo: float = 100.0, rel: float = 1e-6) -> "LineList":
        """Drop lines whose peak opacity per unit column is below `rel` times the strongest
        line at either end of the temperature range.  Keeps accuracy while trimming HITEMP
        lists by an order of magnitude."""
        keep = np.zeros(len(self), bool)
        for T in (T_lo, T_hi):
            k = self.kappa(T)
            keep |= k > rel * k.max()
        return LineList(self.molecule, self.table[keep].reset_index(drop=True), self.partition,
                        self.source, self.release)

    def kappa(self, T: float) -> np.ndarray:
        """Opacity per unit column density integrated over the line profile:
        kappa = A_ul g_u lambda^3 / (8 pi) * (exp(-E_l/T) - exp(-E_u/T)) / Z(T)   [m^3 ... per molecule m^-2]
        Multiply by N [m^-2] and by the velocity profile phi(v) [s/m] to get tau.
        """
        Z = self.partition(T)
        lam = self.wave * 1e-6
        return self.a * self.gu * lam**3 / (8.0 * np.pi) * (np.exp(-self.el / T) - np.exp(-self.eu / T)) / Z

    def to_parquet(self, path: str):
        self.table.to_parquet(path, index=False)
        # store the partition table next to it for line lists that carry their own
        qpath = path.replace(".parquet", "_Q.npz")
        np.savez(qpath, T=self.partition.T, Q=self.partition.Q, source=self.partition.source)


# ----------------------------------------------------------------------------
# Readers
# ----------------------------------------------------------------------------

def _finish(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("wave").reset_index(drop=True)
    for c in COLUMNS:
        if c not in df:
            df[c] = "" if c in ("vup", "vlow", "qup", "qlow") else np.nan
    return df[COLUMNS]


def read_hitran_par(path: str, molecule: Molecule | None = None) -> pd.DataFrame:
    """Parse HITRAN 160-character .par records (format of the HITRAN and HITEMP downloads)."""
    rows = []
    with open(path) as fh:
        for line in fh:
            if len(line) < 160 or line.startswith("#"):
                continue
            M = int(line[0:2]); I = line[2]
            try:
                I = int(I)
            except ValueError:  # HITRAN uses letters for iso > 9
                I = 10 + "ABCDEFGHIJ".index(I.upper())
            nu = float(line[3:15]); sw = float(line[15:25]); a = float(line[25:35])
            elower = float(line[45:55])
            vup = line[67:82]; vlow = line[82:97]; qup = line[97:112]; qlow = line[112:127]
            gp = float(line[145:153]); gpp = float(line[153:160])
            rows.append((M, I, nu, sw, a, elower, vup, vlow, qup, qlow, gp, gpp))
    df = pd.DataFrame(rows, columns=["M", "iso", "nu", "sw", "a", "elower", "vup", "vlow", "qup", "qlow", "gu", "gl"])
    if molecule is not None:
        df = df[(df["M"] == molecule.hitran_id) & (df["iso"] == molecule.iso)]
    df["wave"] = 1e4 / df["nu"]
    df["el"] = df["elower"] * CM1_TO_K
    df["eu"] = (df["elower"] + df["nu"]) * CM1_TO_K
    for c in ("vup", "vlow", "qup", "qlow"):
        df[c] = df[c].str.strip()
    return _finish(df)


def read_islat_par(path: str) -> tuple[pd.DataFrame, tuple[np.ndarray, np.ndarray] | None, dict]:
    """Parse the processed line-list format distributed with iSLAT (HITEMP/HITRAN/Arabhavi lists).

    Returns (table, (T, Q) partition table or None, header info).
    """
    info = {}
    with open(path) as fh:
        text = fh.read()
    lines = text.splitlines()
    m = re.search(r"id:(\d+);\s*iso:(\d+)", lines[0])
    if m:
        info["hitran_id"] = int(m.group(1)); info["iso"] = int(m.group(2))
    # partition block
    T = []; Q = []
    i = 0
    n_q = None
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("# Number of Partition"):
            n_q = int(lines[i + 1].strip()); i += 2; continue
        if n_q is not None and s and not s.startswith("#") and len(T) < n_q:
            p = s.split()
            if len(p) >= 2:
                T.append(float(p[0])); Q.append(float(p[1]))
            i += 1
            continue
        if s.startswith("Number of lines"):
            i += 2
            break
        i += 1
    rows = []
    for s in lines[i:]:
        if not s.strip() or s.lstrip().startswith("#"):
            continue
        p = s.split()
        # Nr Lev_up Lev_low Lambda Freq A E_up E_low g_up g_low
        try:
            rows.append((p[1], p[2], float(p[3]), float(p[4]), float(p[5]), float(p[6]), float(p[7]),
                         float(p[8]), float(p[9])))
        except (ValueError, IndexError):
            continue
    df = pd.DataFrame(rows, columns=["lev_up", "lev_low", "wave", "freq", "a", "eu", "el", "gu", "gl"])
    df["nu"] = 1e4 / df["wave"]
    df["vup"] = df["lev_up"].str.split("|").str[0]
    df["qup"] = df["lev_up"].str.split("|").str[-1]
    df["vlow"] = df["lev_low"].str.split("|").str[0]
    df["qlow"] = df["lev_low"].str.split("|").str[-1]
    df["iso"] = info.get("iso", 1)
    Qtab = (np.array(T), np.array(Q)) if T else None
    return _finish(df), Qtab, info


def fetch_hitran(molecule: Molecule, wmin: float = 4.5, wmax: float = 30.0) -> pd.DataFrame:
    """Download lines from HITRAN through astroquery (network)."""
    from astroquery.hitran import Hitran
    from astropy import units as u
    tab = Hitran.query_lines(molecule_number=molecule.hitran_id, isotopologue_number=molecule.iso,
                             min_frequency=1e4 / wmax / u.cm, max_frequency=1e4 / wmin / u.cm)
    df = tab.to_pandas()
    df = df.rename(columns={"global_upper_quanta": "vup", "global_lower_quanta": "vlow",
                            "local_upper_quanta": "qup", "local_lower_quanta": "qlow",
                            "gp": "gu", "gpp": "gl"})
    df["wave"] = 1e4 / df["nu"]
    df["el"] = df["elower"] * CM1_TO_K
    df["eu"] = (df["elower"] + df["nu"]) * CM1_TO_K
    df["iso"] = molecule.iso
    for c in ("vup", "vlow", "qup", "qlow"):
        df[c] = df[c].astype(str).str.strip()
    return _finish(df)


def fetch_hapi(molecule: Molecule, wmin: float = 4.5, wmax: float = 30.0, workdir: str | None = None) -> pd.DataFrame:
    """Download lines with HAPI (needs http access to hitran.org; some sites require an API key)."""
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        import hapi
        workdir = workdir or os.path.join(data_dir(), "hapi")
        os.makedirs(workdir, exist_ok=True)
        hapi.db_begin(workdir)
        name = f"{molecule.name}_tmp"
        hapi.fetch(name, molecule.hitran_id, molecule.iso, 1e4 / wmax, 1e4 / wmin)
        cols = ["nu", "sw", "a", "elower", "gp", "gpp", "global_upper_quanta", "global_lower_quanta",
                "local_upper_quanta", "local_lower_quanta"]
        data = hapi.getColumns(name, cols)
    df = pd.DataFrame(dict(zip(cols, data)))
    df = df.rename(columns={"global_upper_quanta": "vup", "global_lower_quanta": "vlow",
                            "local_upper_quanta": "qup", "local_lower_quanta": "qlow", "gp": "gu", "gpp": "gl"})
    df["wave"] = 1e4 / df["nu"]
    df["el"] = df["elower"] * CM1_TO_K
    df["eu"] = (df["elower"] + df["nu"]) * CM1_TO_K
    df["iso"] = molecule.iso
    return _finish(df)


# ----------------------------------------------------------------------------
# Public loader with cache
# ----------------------------------------------------------------------------

def load_linelist(molecule: str | Molecule, release: str = "hitran", path: str | None = None,
                  fetch: bool = True, wmin: float = 4.5, wmax: float = 30.0,
                  qtpy_folder: str | None = None, fallback: bool | None = None, force: bool = False) -> LineList:
    """Load a line list for `molecule`.

    Order: explicit `path` (par/parquet/csv) -> Parquet cache of `release` -> download (if `fetch`,
    HITRAN only) -> any other cached release of the molecule (if `fallback`; default: only when
    `fetch` is False, so that an explicit fetch of a release never silently returns another one).
    `force` re-downloads even if the release is cached.  The returned LineList.release tells which
    release was actually loaded.
    """
    mol = get_molecule(molecule) if isinstance(molecule, str) else molecule
    if path is not None:
        from .examples import resolve_path
        path = resolve_path(path)
    cpath = cache_path(mol.name, release)
    Qtab = None
    if fallback is None:
        fallback = not fetch
    if force and path is None:
        wpath = cache_path(mol.name, release, write=True)
        if os.path.exists(wpath):
            os.remove(wpath)
        cpath = wpath
    if path is not None:
        src = path
        if path.endswith(".parquet"):
            df = pd.read_parquet(path)
            qpath = path.replace(".parquet", "_Q.npz")
            if os.path.exists(qpath):
                z = np.load(qpath); Qtab = (z["T"], z["Q"])
        elif path.endswith(".csv"):
            df = _finish(pd.read_csv(path))
        else:
            with open(path) as fh:
                head = fh.readline()
            if head.startswith("#") or "Number of lines" in open(path).read(20000):
                df, Qtab, _ = read_islat_par(path)
            else:
                df = read_hitran_par(path, mol)
    elif not os.path.exists(cpath) and fallback and any_cached(mol.name):
        cpath = any_cached(mol.name)
        release = os.path.basename(cpath)[len(mol.name) + 1:-len(".parquet")]
        src = cpath
        df = pd.read_parquet(cpath)
        qpath = cpath.replace(".parquet", "_Q.npz")
        if os.path.exists(qpath):
            z = np.load(qpath); Qtab = (z["T"], z["Q"])
    elif os.path.exists(cpath):
        src = cpath
        df = pd.read_parquet(cpath)
        qpath = cpath.replace(".parquet", "_Q.npz")
        if os.path.exists(qpath):
            z = np.load(qpath); Qtab = (z["T"], z["Q"])
    elif fetch:
        if release != "hitran":
            raise FileNotFoundError(f"{mol.name} release '{release}' is not cached ({cpath}); only HITRAN can be downloaded — "
                                    f"import other releases from a .par file: jalebi linedata import {mol.name} file.par --release {release}")
        src = "HITRAN download"
        act.info("linedata", "downloading %s lines from HITRAN (%g–%g µm) …", mol.name, wmin, wmax)
        try:
            df = fetch_hitran(mol, wmin, wmax)
        except Exception as e_astro:
            try:
                df = fetch_hapi(mol, wmin, wmax)
            except Exception as e_hapi:
                raise RuntimeError(f"Could not download {mol.name} lines: astroquery: {e_astro}; HAPI: {e_hapi}")
        cpath = cache_path(mol.name, release, write=True)
        df.to_parquet(cpath, index=False)
    else:
        raise FileNotFoundError(f"No cached line list for {mol.name} ({cpath}) and fetch=False")
    if len(df) == 0:
        raise ValueError(f"Line list for {mol.name} from {src} is empty")
    act.debug("linedata", "line list %s (%s): %d lines from %s", mol.name, release, len(df), src)
    pf = partition_for(mol, table=Qtab, qtpy_folder=qtpy_folder)
    return LineList(mol, df, pf, source=src, release=release)


def any_cached(molecule: str) -> str | None:
    """Path of any cached release for this molecule (preferring 'hitran'; user cache before bundled)."""
    import glob
    files = []
    for d in search_dirs():
        fs = sorted(glob.glob(os.path.join(d, f"{molecule}_*.parquet")), key=os.path.getmtime, reverse=True)
        files += [f for f in fs if not f.endswith("_Q.parquet")]
    for f in files:
        if f.endswith(f"{molecule}_hitran.parquet"):
            return f
    return files[0] if files else None


def available_linelists() -> pd.DataFrame:
    """Table of cached line lists (molecule, release, path, size, location); user cache shadows bundled."""
    import glob
    rows, seen = [], set()
    for d in search_dirs():
        where = "bundled" if os.path.abspath(d) == os.path.abspath(BUNDLED_DIR) else "user"
        for f in sorted(glob.glob(os.path.join(d, "*.parquet"))):
            base = os.path.basename(f)[:-len(".parquet")]
            mol, _, rel = base.partition("_")
            if (mol, rel) in seen:
                continue
            seen.add((mol, rel))
            rows.append({"molecule": mol, "release": rel, "path": f, "MB": os.path.getsize(f) / 1e6, "location": where})
    return pd.DataFrame(rows, columns=["molecule", "release", "path", "MB", "location"])


def import_linelist(molecule: str, path: str, release: str = "hitran") -> str:
    """Convert a local file to the Parquet cache and return the cache path."""
    ll = load_linelist(molecule, release=release, path=path, fetch=False)
    out = cache_path(ll.molecule.name, release, write=True)
    ll.to_parquet(out)
    return out


# ----------------------------------------------------------------------------
# 0.22: local line lists for molecules HITRAN does not have (C6H6, C3H4, ...)
# ----------------------------------------------------------------------------
IMPORT_FORMATS = ("auto", "hitran160", "islat", "csv")
C2 = 1.438776877          # second radiation constant [cm K]


def read_csv_lines(path: str, Q296: float | None = None) -> pd.DataFrame:
    """A CSV / whitespace table with at least (wave [um] or nu [cm-1]), (eu or el) [K or cm-1 with a unit
    suffix in the header: eu_cm1 / el_cm1], gu, and either the Einstein A [s-1] (column a) or the HITRAN
    intensity at 296 K (column sw, cm-1 / (molecule cm-2)) with gu and the partition sum Q(296) to convert:
        A = S * 8 pi c nu^2 * Q(296) * exp(c2 E_l / 296) / (g_u * (1 - exp(-c2 nu / 296)))     [c in cm/s]
    (the relation of Simeckova et al. 2006 used by Arabhavi et al. 2024 for the GEISA benzene lines)."""
    sep = None
    with open(path) as fh:
        head = fh.readline()
    sep = "," if "," in head else r"\s+"
    df = pd.read_csv(path, sep=sep, comment="#", engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    if "nu" not in df and "wave" in df:
        df["nu"] = 1e4 / df["wave"].astype(float)
    if "wave" not in df and "nu" in df:
        df["wave"] = 1e4 / df["nu"].astype(float)
    if "nu" not in df:
        raise ValueError("need a wave [um] or nu [cm-1] column")
    for c in ("eu", "el"):
        if f"{c}_cm1" in df and c not in df:
            df[c] = df[f"{c}_cm1"].astype(float) * CM1_TO_K
    if "el" not in df and "eu" in df:
        df["el"] = df["eu"].astype(float) - df["nu"].astype(float) * CM1_TO_K
    if "eu" not in df and "el" in df:
        df["eu"] = df["el"].astype(float) + df["nu"].astype(float) * CM1_TO_K
    if "eu" not in df:
        raise ValueError("need eu or el (K; or eu_cm1 / el_cm1)")
    if "gu" not in df:
        raise ValueError("need the upper-level degeneracy gu")
    if "gl" not in df:
        df["gl"] = np.nan
    if "a" not in df:
        if "sw" not in df:
            raise ValueError("need the Einstein A (column a) or the 296 K intensity (column sw) with --q296")
        if Q296 is None:
            raise ValueError("converting intensities to Einstein A needs Q(296): pass --q296 or a partition file covering 296 K")
        nu = df["nu"].astype(float).to_numpy(); el_cm1 = df["el"].astype(float).to_numpy() / CM1_TO_K
        c_cgs = 2.99792458e10
        df["a"] = (df["sw"].astype(float).to_numpy() * 8.0 * np.pi * c_cgs * nu ** 2 * Q296 * np.exp(C2 * el_cm1 / 296.0)
                   / (df["gu"].astype(float).to_numpy() * (1.0 - np.exp(-C2 * nu / 296.0))))
    for c in ("vup", "vlow", "qup", "qlow"):
        if c not in df:
            df[c] = ""
        df[c] = df[c].astype(str).str.strip()
    if "iso" not in df:
        df["iso"] = 1
    return _finish(df)


def read_partition_file(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Two columns T [K], Q (CSV or whitespace, '#' comments)."""
    with open(path) as fh:
        head = fh.readline()
    t = pd.read_csv(path, sep="," if "," in head else r"\s+", comment="#", engine="python", header=None if head.strip()[0].isdigit() else 0)
    T = t.iloc[:, 0].astype(float).to_numpy(); Q = t.iloc[:, 1].astype(float).to_numpy()
    return T, Q


def partition_from_levels(df: pd.DataFrame, T: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Q(T) = sum_i g_i exp(-E_i / T) over the distinct levels of the list (upper and lower levels keyed by
    their energy and degeneracy).  Only as complete as the list: lines that start or end on a level are needed
    for it to count, so Q is underestimated at temperatures where levels outside the list are populated.
    Good enough when the list covers the ground state and the bands that matter at the fit temperatures;
    prefer a published Q table (--partition file) whenever one exists."""
    T = np.asarray(T if T is not None else np.concatenate([np.arange(1, 100, 1.0), np.arange(100, 3001, 10.0)]), float)
    lev = pd.concat([pd.DataFrame({"E": df["eu"].astype(float).round(3), "g": df["gu"].astype(float)}),
                     pd.DataFrame({"E": df["el"].astype(float).round(3), "g": df["gl"].astype(float)})]).dropna()
    lev = lev.drop_duplicates(subset=["E", "g"])
    E = lev["E"].to_numpy(); g = lev["g"].to_numpy()
    Q = np.array([float(np.sum(g * np.exp(-E / t))) for t in T])
    return T, Q


def import_local_linelist(molecule: str, path: str, release: str = "local", fmt: str = "auto",
                          partition: str | None = None, q296: float | None = None,
                          T_grid: np.ndarray | None = None) -> dict:
    """0.22 `jalebi linelist import`: read `path` (hitran160 .par, iSLAT list, CSV), attach a partition function
    (a T,Q file; "levels" = direct sum over the list's levels; or the list's own table / TIPS when available)
    and write <cache>/<MOL>_<release>.parquet + _Q.npz.  Returns a dict describing what was written."""
    mol = get_molecule(molecule)
    if fmt not in IMPORT_FORMATS:
        raise ValueError(f"unknown format {fmt!r}: {', '.join(IMPORT_FORMATS)}")
    Qtab = None
    if partition and partition != "levels":
        Qtab = read_partition_file(partition)
    if fmt == "auto":
        if path.endswith(".csv") or path.endswith(".txt"):
            fmt = "csv"
        else:
            with open(path) as fh:
                head = fh.read(20000)
            fmt = "islat" if (head.startswith("#") or "Number of lines" in head) else "hitran160"
    if fmt == "hitran160":
        df = read_hitran_par(path, mol if mol.hitran_id > 0 else None)
    elif fmt == "islat":
        df, Q_own, _ = read_islat_par(path)
        Qtab = Qtab or Q_own
    else:
        if q296 is None and Qtab is not None:
            q296 = float(np.interp(296.0, Qtab[0], Qtab[1]))
        df = read_csv_lines(path, q296)
    if len(df) == 0:
        raise ValueError(f"no lines read from {path} ({fmt})")
    q_source = "file" if (partition and partition != "levels") else ("list" if Qtab is not None else None)
    if Qtab is None and partition == "levels":
        Qtab = partition_from_levels(df, T_grid); q_source = "levels"
    if Qtab is None:
        if mol.hitran_id > 0:
            q_source = "TIPS (HITRAN id %d)" % mol.hitran_id
        else:
            Qtab = partition_from_levels(df, T_grid); q_source = "levels (no Q table given; underestimates Q where the list is incomplete)"
    pf = partition_for(mol, table=Qtab) if Qtab is not None else partition_for(mol)
    ll = LineList(mol, df, pf, source=path, release=release)
    out = cache_path(mol.name, release, write=True)
    ll.to_parquet(out)
    return {"path": out, "n_lines": int(len(df)), "wave_min": float(df["wave"].min()), "wave_max": float(df["wave"].max()),
            "format": fmt, "partition": q_source, "Q296": float(pf(296.0)), "release": release}
