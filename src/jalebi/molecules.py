"""Molecule registry: HITRAN identifiers, masses, display names and colours."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Molecule:
    name: str            # key used everywhere in configs (e.g. "H2O", "13CO2")
    hitran_id: int       # HITRAN molecule number M
    iso: int             # isotopologue number I (1 = most abundant)
    mass_amu: float
    label: str           # pretty label for plots
    colour: str
    parent: str | None = None   # for isotopologues: name of the parent molecule
    default_ratio: float | None = None   # parent/iso abundance ratio for the tie prior
    hitran_global_id: int | None = None


_M = [
    Molecule("H2O", 1, 1, 18.010565, "H₂O", "#1f77b4", hitran_global_id=1),
    Molecule("H2_18O", 1, 2, 20.014811, "H₂¹⁸O", "#6baed6", parent="H2O", default_ratio=500.0, hitran_global_id=2),
    Molecule("CO2", 2, 1, 43.98983, "CO₂", "#d62728", hitran_global_id=7),
    Molecule("13CO2", 2, 2, 44.993185, "¹³CO₂", "#ff9896", parent="CO2", default_ratio=70.0, hitran_global_id=8),
    Molecule("CO", 5, 1, 27.994915, "CO", "#2ca02c", hitran_global_id=26),
    Molecule("13CO", 5, 2, 28.99827, "¹³CO", "#98df8a", parent="CO", default_ratio=70.0, hitran_global_id=27),
    Molecule("CH4", 6, 1, 16.0313, "CH₄", "#8c564b", hitran_global_id=32),
    Molecule("NH3", 11, 1, 17.026549, "NH₃", "#e377c2", hitran_global_id=45),
    Molecule("OH", 13, 1, 17.00274, "OH", "#7f7f7f", hitran_global_id=48),
    Molecule("HCN", 23, 1, 27.010899, "HCN", "#9467bd", hitran_global_id=70),
    Molecule("H13CN", 23, 2, 28.014254, "H¹³CN", "#c5b0d5", parent="HCN", default_ratio=70.0, hitran_global_id=71),
    Molecule("C2H2", 26, 1, 26.01565, "C₂H₂", "#ff7f0e", hitran_global_id=76),
    Molecule("13CCH2", 26, 2, 27.019005, "¹³CCH₂", "#ffbb78", parent="C2H2", default_ratio=35.0, hitran_global_id=77),
    Molecule("C2H6", 27, 1, 30.04695, "C₂H₆", "#bcbd22", hitran_global_id=78),
    Molecule("C2H4", 38, 1, 28.0313, "C₂H₄", "#dbdb8d", hitran_global_id=90),
    Molecule("C4H2", 43, 1, 50.01565, "C₄H₂", "#17becf", hitran_global_id=116),
    Molecule("HC3N", 44, 1, 51.010899, "HC₃N", "#9edae5", hitran_global_id=109),
    Molecule("H2", 45, 1, 2.01565, "H₂", "#aec7e8", hitran_global_id=103),
    Molecule("C6H6", 0, 1, 78.04695, "C₆H₆", "#393b79"),   # not in HITRAN: needs a local line list
    Molecule("C3H4", 0, 1, 40.0313, "C₃H₄", "#637939"),     # not in HITRAN: needs a local line list
]
MOLECULES: dict[str, Molecule] = {m.name: m for m in _M}


def get_molecule(name: str) -> Molecule:
    """Look up a molecule by config name; accepts a few common aliases."""
    aliases = {"h2o": "H2O", "water": "H2O", "co2": "CO2", "c2h2": "C2H2", "hcn": "HCN",
               "co": "CO", "oh": "OH", "13co2": "13CO2", "ch4": "CH4", "nh3": "NH3"}
    key = aliases.get(name.lower(), name)
    if key not in MOLECULES:
        raise KeyError(f"Unknown molecule '{name}'. Known: {', '.join(MOLECULES)}")
    return MOLECULES[key]


# Wavelength ranges (micron) where each molecule has its main MIRI features; used as
# default fit windows in the app and CLI.  Chosen from MINDS / JDISCS practice.
DEFAULT_WINDOWS: dict[str, list[tuple[float, float]]] = {
    "H2O":   [(5.0, 8.0), (12.0, 17.5), (17.5, 27.5)],
    "CO":    [(4.9, 5.35)],
    "CO2":   [(14.6, 16.4)],
    "13CO2": [(15.3, 15.5)],
    "13CO":  [(4.9, 5.35)],
    "13CCH2": [(13.65, 13.85)],
    "H13CN": [(14.05, 14.3)],
    "C2H2":  [(13.5, 14.2)],
    "HCN":   [(13.8, 14.3)],
    "OH":    [(9.0, 13.0), (13.0, 27.0)],
    "CH4":   [(7.4, 7.9)],
    "NH3":   [(9.5, 11.5)],
    "C2H4":  [(10.3, 10.7)],
    "C2H6":  [(11.8, 12.4)],
    "C4H2":  [(15.8, 16.0)],
    "HC3N":  [(15.0, 15.2)],
    "C6H6":  [(14.7, 14.9)],
    "C3H4":  [(15.7, 16.0)],
    "H2":    [(5.0, 28.0)],
}


# Spectral landmarks (micron): Q-branches, band heads and band centres by which each molecule is
# recognised by eye in MIRI spectra (MINDS / JDISCS practice).  Used for the shaded markers in the app
# and for the detection windows.  H2O has no entry on purpose: its lines are everywhere.
# Each entry is (lo, hi, label).
FEATURES: dict[str, list[tuple[float, float, str]]] = {
    "CO":     [(4.90, 5.30, "CO v=1–0 P branch")],
    "CO2":    [(13.85, 13.90, "CO₂ hot band Q"), (14.93, 15.02, "CO₂ ν₂ Q"), (16.16, 16.21, "CO₂ hot band Q")],
    "13CO2":  [(15.40, 15.44, "¹³CO₂ Q")],
    "C2H2":   [(13.69, 13.73, "C₂H₂ ν₅ Q")],
    "13CCH2": [(13.73, 13.75, "¹³CCH₂ Q")],
    "HCN":    [(14.00, 14.06, "HCN ν₂ Q"), (14.29, 14.33, "HCN hot band Q")],
    "H13CN":  [(14.14, 14.18, "H¹³CN Q")],
    "CH4":    [(7.64, 7.68, "CH₄ ν₄ Q")],
    "NH3":    [(10.30, 10.80, "NH₃ ν₂")],
    "C2H4":   [(10.51, 10.55, "C₂H₄ ν₇ Q")],
    "C2H6":   [(12.14, 12.20, "C₂H₆ ν₉ Q")],
    "C4H2":   [(15.90, 15.94, "C₄H₂ ν₈ Q")],
    "HC3N":   [(15.06, 15.10, "HC₃N ν₅ Q")],
    "C6H6":   [(14.83, 14.87, "C₆H₆ ν₄ Q")],
    "C3H4":   [(15.78, 15.82, "C₃H₄ ν₉ Q")],
    "OH":     [(9.0, 13.0, "OH prompt emission")],
    "H2":     [(5.506, 5.516, "H₂ S(7)"), (6.104, 6.114, "H₂ S(6)"), (6.904, 6.914, "H₂ S(5)"), (8.020, 8.030, "H₂ S(4)"),
               (9.660, 9.670, "H₂ S(3)"), (12.274, 12.284, "H₂ S(2)"), (17.030, 17.040, "H₂ S(1)"), (28.21, 28.23, "H₂ S(0)")],
}
