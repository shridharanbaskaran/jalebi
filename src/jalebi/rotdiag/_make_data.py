"""Build the bundled data files of jalebi.rotdiag (run once by the maintainers; not imported at run time).

    python -m jalebi.rotdiag._make_data  ROUEFF_TABLE  [--extinction]

1. H2 line list and level list from Roueff et al. (2019, A&A 630, A58; CDS J/A+A/630/A58, table2.dat or
   the same table as an astropy ECSV/ipac file): all 4712 electric-quadrupole + magnetic-dipole transitions
   of the X state, with energies from Pachucki & Komasa (2018).  Written as
     jalebi/linedata/H2_roueff2019.parquet  (+ _Q.npz: Q(T) as the level sum, nuclear spin included)
     jalebi/data_files/H2_levels_Roueff2019.csv  (all 302 bound levels: v, J, g, E [K], spin o/p)
2. Extinction curves A_lambda/A_V tabulated from the `dust_extinction` package (Gordon et al. 2024, JOSS 9,
   7023; BSD-3): G23 (R_V 3.1 and 5.5), G21_MWAvg, CT06_MWLoc, F11_MWGC, written as
     jalebi/data_files/extinction_curves.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
D0_CM = 36118.0695            # H2 dissociation energy of the (0,0) level used by Roueff et al. (cm^-1)
CM1_TO_K = 1.4387768775039337


def read_roueff(path: str) -> pd.DataFrame:
    if path.endswith(".dat"):           # CDS fixed-width table2.dat
        cols = [(1, 3), (4, 6), (7, 9), (10, 12), (13, 29), (32, 39), (39, 56), (59, 66), (70, 79), (83, 92), (96, 105),
                (109, 118), (120, 136), (140, 147), (148, 159), (161, 164)]
        names = ["vu", "Ju", "vl", "Jl", "sigma", "Dsigma", "lambda", "dlambda", "Aqua", "Ama", "A", "Atot", "Eu", "DEu", "Tu", "gu"]
        t = pd.read_fwf(path, colspecs=cols, names=names)
    else:                               # the ipac-like table distributed with pdrtpy (same content)
        rows = []
        with open(path) as fh:
            for line in fh:
                s = line.strip()
                if not s or s.startswith("|") or s.startswith("#"):
                    continue
                p = s.split()
                rows.append(p[:16])
        names = ["vu", "Ju", "vl", "Jl", "sigma", "Dsigma", "lambda", "dlambda", "Aqua", "Ama", "A", "Atot", "Eu", "DEu", "Tu", "gu"]
        t = pd.DataFrame(rows, columns=names).apply(pd.to_numeric)
    return t


def branch(dJ: int) -> str:
    return {-2: "S", 0: "Q", 2: "O"}.get(dJ, "?")


def make_h2(path: str):
    t = read_roueff(path)
    eu = (t["Eu"].to_numpy() + D0_CM) * CM1_TO_K
    el_cm = t["Eu"].to_numpy() - t["sigma"].to_numpy()
    el = (el_cm + D0_CM) * CM1_TO_K
    gI = lambda J: np.where(np.asarray(J) % 2 == 1, 3.0, 1.0)
    gl = gI(t["Jl"]) * (2 * t["Jl"] + 1)
    lab = [f"{vu}-{vl} {branch(jl - ju)}({jl})" for vu, vl, ju, jl in zip(t.vu, t.vl, t.Ju, t.Jl)]
    df = pd.DataFrame({"wave": t["lambda"].astype(float), "nu": t["sigma"].astype(float), "a": t["A"].astype(float),
                       "gu": t["gu"].astype(float), "gl": gl.astype(float), "eu": eu, "el": el,
                       "vup": t["vu"].astype(str), "vlow": t["vl"].astype(str),
                       "qup": [f"J={j}" for j in t["Ju"]], "qlow": lab, "iso": 1})
    df = df.sort_values("wave").reset_index(drop=True)
    # levels: every (v, J) that appears as an upper or a lower state
    lev = {}
    for v, J, E in zip(t.vu, t.Ju, eu):
        lev[(int(v), int(J))] = E
    for v, J, E in zip(t.vl, t.Jl, el):
        lev.setdefault((int(v), int(J)), E)
    L = pd.DataFrame([(v, J, float(gI(J) * (2 * J + 1)), E, "o" if J % 2 else "p") for (v, J), E in lev.items()],
                     columns=["v", "J", "g", "E_K", "spin"]).sort_values("E_K").reset_index(drop=True)
    L.loc[(L.v == 0) & (L.J == 0), "E_K"] = 0.0
    T = np.arange(1.0, 5001.0)
    Q = np.array([(L.g * np.exp(-L.E_K / x)).sum() for x in T])
    out_ll = os.path.join(PKG, "linedata", "H2_roueff2019.parquet")
    df.to_parquet(out_ll, index=False)
    np.savez(out_ll.replace(".parquet", "_Q.npz"), T=T, Q=Q, source="level sum of the 302 X-state levels of Roueff et al. (2019)")
    out_lev = os.path.join(PKG, "data_files", "H2_levels_Roueff2019.csv")
    with open(out_lev, "w") as fh:
        fh.write("# H2 X-state rovibrational levels from Roueff et al. (2019, A&A 630, A58; CDS J/A+A/630/A58); "
                 "E_K above (v=0, J=0); g = g_I (2J+1), g_I = 3 (ortho, odd J) or 1 (para)\n")
        L.to_csv(fh, index=False, float_format="%.6f")
    print(f"{len(df)} lines -> {out_ll}; {len(L)} levels -> {out_lev}; Q(100 K) = {Q[99]:.4f}, Q(1000 K) = {Q[999]:.4f}")


def make_extinction():
    import astropy.units as u
    from dust_extinction.averages import CT06_MWLoc, F11_MWGC, G21_MWAvg
    from dust_extinction.parameter_averages import G23
    w = np.unique(np.concatenate([np.geomspace(0.8, 32.0, 700), np.linspace(8.0, 12.0, 161)]))
    cols = {"wave_um": w}
    for name, model in (("G23_Rv3.1", G23(Rv=3.1)), ("G23_Rv5.5", G23(Rv=5.5)), ("G21_MWAvg", G21_MWAvg()),
                        ("CT06_MWLoc", CT06_MWLoc()), ("F11_MWGC", F11_MWGC())):
        lo, hi = model.x_range           # 1/micron
        x = 1.0 / w
        ok = (x >= lo) & (x <= hi)
        v = np.full(len(w), np.nan)
        v[ok] = model(w[ok] * u.micron)
        cols[name] = v
    df = pd.DataFrame(cols)
    out = os.path.join(PKG, "data_files", "extinction_curves.csv")
    with open(out, "w") as fh:
        fh.write("# A_lambda/A_V tabulated from dust_extinction 1.7 (Gordon et al. 2024, JOSS 9, 7023; BSD-3): "
                 "G23 = Gordon et al. 2023 (ApJ 950, 86), G21_MWAvg = Gordon et al. 2021 (ApJ 916, 33), "
                 "CT06_MWLoc = Chiar & Tielens 2006 (ApJ 637, 774) local ISM, F11_MWGC = Fritz et al. 2011 (ApJ 737, 73); "
                 "NaN outside each curve's range\n")
        df.to_csv(fh, index=False, float_format="%.6g")
    print(f"extinction curves -> {out}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        make_h2(args[0])
    if "--extinction" in sys.argv:
        make_extinction()
