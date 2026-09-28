"""Example 3 — continuum methods on real and synthetic data.

(a) The FZ Tau MIRI spectrum (bundled x1d files) with every continuum method around the 13–17.5 µm
    organics region, where line forests make the "continuum" hard to define.
(b) The synthetic spectrum, whose true continuum is known: how far off is each method?
    A continuum that sits too high eats line flux and biases T and N low for blended bands (C2H2, HCN).

Run:  python 03_continuum_methods.py      (about 30 s; writes results/03_continuum_methods.png)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jalebi.continuum import ContinuumSettings, estimate_continuum
from jalebi.data import load_spectrum, spike_filter
from jalebi.examples import example_path

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
# label -> (method, settings); see jalebi.continuum.ContinuumSettings for every knob
METHODS = {"irsqr (25 px knots, q=0.10)": ("irsqr", {}),
           "irsqr (50 px knots, q=0.05)": ("irsqr", {"knot_spacing": 50, "quantile": 0.05}),
           "median_sg": ("median_sg", {}), "asls": ("asls", {}), "rolling_min": ("rolling_min", {}),
           "convex_hull": ("convex_hull", {}), "banzatti": ("banzatti", {})}

# (a) FZ Tau ---------------------------------------------------------------------------------------
fz = spike_filter(load_spectrum(example_path("FZ_Tau"), distance_pc=130.0).to_rest_frame(20.0))
fig, axes = plt.subplots(2, 1, figsize=(15, 9))
ax = axes[0]
sel = (fz.wave > 13.0) & (fz.wave < 17.5)
for b in fz.bands:
    i = fz.band_slice(b); i = i[sel[i]]
    if len(i):
        ax.plot(fz.wave[i], fz.flux[i], color="k", lw=0.5, label="FZ Tau" if b == "3B" else None)
for label, (meth, kw) in METHODS.items():
    c = estimate_continuum(fz, ContinuumSettings(method=meth, **kw))
    for b in fz.bands:
        i = fz.band_slice(b); i = i[sel[i]]
        if len(i):
            ax.plot(fz.wave[i], c[i], lw=1.0, label=label if b == "3B" else None)
ax.set_xlim(13.0, 17.5); ax.set_ylabel("F_ν [Jy]"); ax.legend(fontsize=8, ncol=4)
ax.set_title("(a) FZ Tau: continuum methods in the organics region")

# (b) synthetic spectrum with a known continuum ----------------------------------------------------
path = example_path("synthetic/synthetic_miri_ch3.csv")
syn = load_spectrum(path)
truth = pd.read_csv(path, comment="#")["continuum_true"].to_numpy()
rows = []
ax = axes[1]
for label, (meth, kw) in METHODS.items():
    c = estimate_continuum(syn, ContinuumSettings(method=meth, **kw))
    d = c - truth
    s1 = (syn.wave > 13.6) & (syn.wave < 14.2)
    s2 = (syn.wave > 13.6) & (syn.wave < 16.3)
    rows.append({"method": label, "offset_C2H2_mJy": 1e3 * np.nanmean(d[s1]), "rms_fit_window_mJy": 1e3 * np.sqrt(np.nanmean(d[s2] ** 2))})
    for b in syn.bands:
        i = syn.band_slice(b)
        ax.plot(syn.wave[i], 1e3 * d[i], lw=0.8, label=label if b == "3B" else None)
ax.axhline(0, color="k", lw=0.6)
ax.axhspan(-1e3 * np.median(syn.err), 1e3 * np.median(syn.err), color="0.85", label="±1σ noise")
ax.set_xlabel("wavelength [µm]"); ax.set_ylabel("estimated − true continuum [mJy]")
ax.set_title("(b) synthetic spectrum: continuum error per method (positive = too high = line flux lost)")
ax.legend(fontsize=8, ncol=4)
fig.tight_layout(); fig.savefig(OUT / "03_continuum_methods.png", dpi=110)
print(pd.DataFrame(rows).round(2).to_string(index=False))
print(f"wrote {OUT / '03_continuum_methods.png'}")
