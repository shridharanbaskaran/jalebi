"""Example 4 — which molecules does FZ Tau show?  Automatic detection by simultaneous template fitting.

Every molecule with a line list (bundled or in your cache) gets LTE templates; one non-negative
least-squares solve fits all of them at once; a molecule is "detected" when removing it raises
chi2 in its own windows by more than the BIC penalty (ΔBIC > 10).  The detected components, their
starting values and the matching fit windows are written to a config you can fit directly.

Run:  python 04_detect_molecules.py        (about 1 min; writes results/FZ_Tau_detected.yaml)
      jalebi fit results/FZ_Tau_detected.yaml --stages grid,optimise
"""
from pathlib import Path

from jalebi.config import ProjectConfig
from jalebi.detect import apply_detection, detect_molecules
from jalebi.pipeline import prepare

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"; OUT.mkdir(exist_ok=True)

cfg = ProjectConfig.load(HERE / "configs" / "FZ_Tau_autodetect.yaml")
spec = prepare(cfg)                                   # continuum-subtracted, masked spectrum
det = detect_molecules(spec, threshold=cfg.fit.detect.threshold,
                       progress=lambda msg, f: print(f"  {msg}"))
print(f"\n{spec.name}: {det.n_pixels} pixels in the detection windows")
print(det.table[["candidate", "T", "logN", "logR", "delta_chi2", "delta_BIC", "detected", "windows"]]
      .round(2).to_string(index=False))

new = apply_detection(cfg, det)
new.fit.auto_detect = False          # the components are now explicit
new.output = "results/{target}/detected"       # -> results/FZ_Tau/detected
new.save(OUT / "FZ_Tau_detected.yaml")
print("\nsuggested components:", ", ".join(f"{c.name} ({c.T:.0f} K)" for c in det.components))
print("suggested windows:", det.windows)
print(f"wrote {OUT / 'FZ_Tau_detected.yaml'}")
