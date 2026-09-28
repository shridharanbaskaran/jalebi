"""Example 1 — LTE slab models, no data needed.

Shows what the forward model does:
  (a) CO2 at three column densities: optically thin -> thick (the Q-branch saturates first),
  (b) water at three temperatures: hotter gas lights up high-E_up lines,
  (c) the curve of growth: line flux grows as N while thin, then only slowly once lines saturate.

Run:  python 01_quick_model.py         (about 20 s; writes results/01_quick_model.png)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jalebi.model import Component, build_model
from jalebi.synthetic import band_pixels

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

wave = band_pixels("3B")              # MIRI-MRS channel 3B pixel grid, 13.34–15.57 µm
d_pc = 140.0                           # distance

fig, axes = plt.subplots(3, 1, figsize=(12, 11))

# (a) CO2: thin -> thick ----------------------------------------------------------------------
ax = axes[0]
for k, logN in enumerate((15.0, 17.0, 19.0)):
    m = build_model([Component("CO2", "CO2", logN=logN, T=500.0, logR=-0.5)], wave, d_pc)
    f = m.evaluate()
    ax.plot(wave, f / f.max() + 1.1 * k, lw=0.8,
            label=f"log N = {logN:.0f} cm⁻²: peak {1e3 * f.max():.3g} mJy, τ_max = {m.tau_flags()['CO2']:.2g}")
ax.set_xlim(14.3, 15.3)
ax.set_ylabel("F_ν / peak (offset)")
ax.set_title("(a) CO₂ at 500 K, R = 0.32 au, d = 140 pc: the ν₂ Q-branch at 14.98 µm saturates first, "
             "so the P/R lines catch up")
ax.legend(fontsize=8)

# (b) water at three temperatures ------------------------------------------------------------------
ax = axes[1]
for k, T in enumerate((300.0, 600.0, 1000.0)):
    # HITEMP: complete at the high upper-level energies that hot water populates
    m = build_model([Component("H2O", "H2O", logN=18.0, T=T, logR=-0.3, linelist_release="hitemp")], wave, d_pc,
                    windows=[(13.4, 15.5)])
    f = m.evaluate()
    ax.plot(wave, f / f.max() + 1.1 * k, lw=0.7, label=f"T = {T:.0f} K: peak {1e3 * f.max():.3g} mJy")
ax.set_xlim(13.4, 15.5)
ax.set_ylabel("F_ν / peak (offset)")
ax.set_title("(b) H₂O (HITEMP), log N = 18: hotter gas is brighter and lights up more high-E_up lines")
ax.legend(fontsize=8)

# (c) curve of growth ------------------------------------------------------------------------------
ax = axes[2]
logNs = np.arange(13.0, 21.01, 0.5)
for T in (300.0, 600.0, 1000.0):
    m = build_model([Component("CO2", "CO2", logN=17.0, T=T, logR=-0.5)], wave, d_pc, windows=[(14.3, 15.3)])
    flux = [m.evaluate({"CO2": {"logN": lN}}).sum() for lN in logNs]
    ax.plot(logNs, flux, "o-", ms=3, label=f"T = {T:.0f} K")
ax.plot(logNs, 10 ** (logNs - 17.0) * ax.get_lines()[1].get_ydata()[8], "k:", lw=0.8, label="∝ N (optically thin)")
ax.set_yscale("log")
ax.set_ylim(1e-3, None)
ax.set_xlabel("log N [cm⁻²]")
ax.set_ylabel("Σ F_ν over 14.3–15.3 µm [Jy]")
ax.set_title("(c) Curve of growth: flux ∝ N while thin, then it flattens once the lines saturate")
ax.legend(fontsize=8)

axes[0].set_xlabel("wavelength [µm]"); axes[1].set_xlabel("wavelength [µm]")
fig.tight_layout()
fig.savefig(OUT / "01_quick_model.png", dpi=110)
print(f"wrote {OUT / '01_quick_model.png'}")
