"""The JALEBI banner, colours and fun facts (standard library only; shared by the CLI and install.py)."""
from __future__ import annotations

import math
import os
import sys

TITLE = "J A L E B I"
# the name, spelled out: (initial, rest of the word) -> the initials read J-A-L-E-B-I down the column
ACRONYM = [("J", "WST"), ("A", "nalysis of"), ("L", "ine"), ("E", "mission with"), ("B", "ayesian"), ("I", "nference")]
EXPANSION = ["JWST Analysis of Line Emission", "with Bayesian Inference"]
FULL_NAME = "JWST Analysis of Line Emission with Bayesian Inference"
TAGLINE = "simultaneous LTE slab fitting of molecular emission in JWST/MIRI disk spectra"

# "JALEBI" in a 3-row box-drawing font (one column of glyphs per letter), and an ASCII fallback
_BIG = {"J": ["  ╻", "  ┃", "┗━┛"], "A": ["┏━┓", "┣━┫", "╹ ╹"], "L": ["╻  ", "┃  ", "┗━╸"],
        "E": ["┏━╸", "┣╸ ", "┗━╸"], "B": ["┏┓ ", "┣┻┓", "┗━┛"], "I": ["╻", "┃", "╹"]}
_BIG_ASCII = {"J": ["  |", "  |", r"\_/"], "A": ["/-\\", "|-|", "| |"], "L": ["|  ", "|  ", "|__"],
              "E": ["|--", "|- ", "|__"], "B": ["|-\\", "|-<", "|_/"], "I": ["|", "|", "|"]}

# 256-colour codes from golden syrup (inner) to fried orange (outer)
_SYRUP = [226, 220, 214, 208, 202, 166]


def supports_colour(stream=None) -> bool:
    stream = stream or sys.stdout
    if os.environ.get("NO_COLOR") or os.environ.get("JALEBI_NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(stream, "isatty") and stream.isatty() and os.environ.get("TERM") != "dumb"


def supports_unicode(stream=None) -> bool:
    enc = (getattr(stream or sys.stdout, "encoding", None) or "").lower()
    return "utf" in enc


def c256(code: int, text: str, on: bool = True, bold: bool = False) -> str:
    if not on:
        return text
    return f"\033[{'1;' if bold else ''}38;5;{code}m{text}\033[0m"


def spiral(width: int = 32, height: int = 12, turns: float = 2.6, colour: bool = False, char: str | None = None) -> list[str]:
    """An Archimedean spiral (r = a + b·θ) drawn in characters, coloured by radius."""
    char = char or ("●" if supports_unicode() else "o")
    grid = [[None] * width for _ in range(height)]
    cx, cy = (width - 1) / 2, (height - 1) / 2
    n = 6000
    r_max = 0.6 + 0.3555 * turns * 2 * math.pi
    for i in range(n):
        t = i / n * turns * 2 * math.pi
        r = 0.6 + 0.3555 * t
        x = cx + 2.0 * r * math.cos(t)
        y = cy + r * math.sin(t)
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < width and 0 <= yi < height:
            grid[yi][xi] = r / r_max
    lines = []
    for row in grid:
        s = ""
        for v in row:
            if v is None:
                s += " "
            else:
                code = _SYRUP[min(int(v * len(_SYRUP)), len(_SYRUP) - 1)]
                s += c256(code, char, colour)
        lines.append(s)
    # drop empty rows
    return [ln for ln in lines if ln.strip()]


def big_title(colour: bool = False, unicode: bool | None = None) -> list[str]:
    """JALEBI in large letters, each letter a shade of syrup."""
    unicode = supports_unicode() if unicode is None else unicode
    font = _BIG if unicode else _BIG_ASCII
    shades = [226, 220, 214, 208, 202, 214]
    rows = []
    for r in range(3):
        rows.append(" ".join(c256(shades[i], font[ch][r], colour, bold=True) for i, ch in enumerate("JALEBI")))
    return rows


def acronym_rows(colour: bool = False) -> list[str]:
    """The expansion, one word per row, with the initials highlighted so they spell JALEBI downwards."""
    return [c256(214, ini, colour, bold=True) + c256(252, rest, colour) for ini, rest in ACRONYM]


def acronym_line(colour: bool = False) -> str:
    """'JWST Analysis of Line Emission with Bayesian Inference' on one line, initials highlighted."""
    return " ".join(c256(214, ini, colour, bold=True) + c256(252, rest, colour) for ini, rest in ACRONYM)


def acronym_rich() -> str:
    """The same for rich/Typer markup."""
    return " ".join(f"[bold #f5b543]{ini}[/]{rest}" for ini, rest in ACRONYM)


def acronym_html(accent: str = "#f5b543") -> str:
    """The same as HTML (web app header)."""
    return " ".join(f'<b style="color:{accent}">{ini}</b>{rest}' for ini, rest in ACRONYM)


def banner(version: str = "", colour: bool | None = None) -> str:
    """Spiral on the left; on the right JALEBI in large letters, the version, and the name spelled out."""
    colour = supports_colour() if colour is None else colour
    art = spiral(colour=colour)
    right = big_title(colour)
    right.append(c256(245, (f"v{version}  ·  " if version else "") + "LTE slab fitting for JWST/MIRI", colour))
    right.append("")
    right += acronym_rows(colour)
    pad = max(len(art) - len(right), 0)
    right = [""] * (pad // 2) + right
    out = []
    for i in range(max(len(art), len(right))):
        left = art[i] if i < len(art) else " " * 32
        out.append("  " + left + "  " + (right[i] if i < len(right) else ""))
    return "\n".join(out)


def compact_banner(version: str = "", colour: bool | None = None) -> str:
    """Two lines for narrow terminals and log headers."""
    colour = supports_colour() if colour is None else colour
    head = c256(214, "JALEBI", colour, bold=True) + (c256(245, f" v{version}", colour) if version else "")
    return head + "  " + acronym_line(colour)


FUN_FACTS = [
    "A jalebi is a spiral of batter soaked in syrup; an inner-disk MIRI spectrum is a spiral of water lines soaked in continuum.",
    "The bundled HITEMP water list has about 580 000 lines between 4.9 and 28.6 µm.",
    "The CO₂ ν₂ Q-branch at 14.98 µm stacks hundreds of lines into a few MIRI resolution elements.",
    "Optically thick slab: the flux mostly measures T and the emitting area, and N is only a lower limit.",
    "Optically thin slab: only the product N × area is constrained. JALEBI can sample log(N·A) directly.",
    "MIRI-MRS resolving power falls from about 4000 at 5 µm to about 1100 at 27 µm (R ≈ 4603 − 128 λ[µm]).",
    "emcee chains should run for at least ~50 autocorrelation times. JALEBI reports τ, R̂ and the acceptance fraction for you.",
    "HCN's Q-branch sits at 14.0 µm and C₂H₂'s at 13.7 µm, close neighbours on the spiral.",
    "¹²C/¹³C ≈ 70 locally, so the ¹³CO₂ Q-branch at 15.4 µm is a good probe of the CO₂ optical depth.",
    "Cold water (T ≲ 300 K) shows up mostly longward of ~20 µm, in MRS channel 4.",
    "Each model call is a sparse matrix-vector product, typically a few to a few tens of milliseconds.",
    "The emitting areas enter the model linearly, so JALEBI solves them exactly by non-negative least squares.",
    "Isotopologue ties: ¹³CO₂ shares T and area with CO₂, and its column is N(CO₂)/ratio.",
    "OH lines at 9–13 µm are prompt emission from water photodissociation, not LTE, so they are masked by default.",
]
