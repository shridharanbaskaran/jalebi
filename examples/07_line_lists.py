"""Example 7 — line lists: what is bundled, how to add more, and per-component releases.

    python 07_line_lists.py               # list and inspect (offline)
    python 07_line_lists.py --fetch HCN   # download a HITRAN list into your cache (needs internet + astroquery)

From the terminal:
    jalebi linedata list
    jalebi linedata fetch H2O --release hitran --wmin 4.9 --wmax 28
    jalebi linedata import C6H6 my_C6H6.par --release arabhavi     # HITRAN .par or iSLAT-format files
"""
import argparse

from jalebi.linedata import available_linelists, data_dir, load_linelist

ap = argparse.ArgumentParser()
ap.add_argument("--fetch", nargs="*", default=[], help="molecules to download from HITRAN (e.g. HCN OH)")
args = ap.parse_args()

tab = available_linelists()
print(f"user cache: {data_dir()}  (set $JALEBI_DATA to move it)\n")
print(tab[["molecule", "release", "MB", "location"]].to_string(index=False))

# a line list is a table of transitions plus a partition function Z(T)
ll = load_linelist("CO2", release="hitran", fetch=False)
print(f"\nCO2 ({ll.source}): {len(ll)} lines, {ll.wave.min():.2f}–{ll.wave.max():.2f} µm, Z(500 K) = {ll.partition(500.0):.1f} "
      f"[{ll.partition.source}]")
q = ll.select(14.9, 15.05, vup=None)                 # select by wavelength (also E_up, A_ul, quantum labels)
k = q.kappa(500.0)
print(f"strongest line near the Q-branch: {q.wave[k.argmax()]:.4f} µm, A = {q.a[k.argmax()]:.3g} s^-1, "
      f"E_up = {q.eu[k.argmax()]:.0f} K")

# water comes in two releases: HITEMP (complete at high E_up, for hot gas) and HITRAN (cold gas)
for rel in ("hitemp", "hitran"):
    w = load_linelist("H2O", release=rel, fetch=False)
    print(f"H2O {rel:6s}: {len(w):>7} lines")
print("choose per component in a config:  - {name: H2O_cold, molecule: H2O, ..., linelist_release: hitran}")

for mol in args.fetch:
    ll = load_linelist(mol, release="hitran", fetch=True, fallback=False)
    print(f"fetched {mol}: {len(ll)} lines -> {data_dir()}")
