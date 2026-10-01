"""Matplotlib figures of a rotation-diagram run: the diagram, the posterior (corner) and the line fits."""
from __future__ import annotations

import numpy as np

from .fit import RotFit, diagram_points, model_curves, model_points

SPIN_COLOURS = {"o": "#d62728", "p": "#1f77b4", "": "#333333", "op": "#7f3c8d"}
SPIN_NAMES = {"o": "ortho", "p": "para", "": "", "op": "o+p blend"}
LADDER_COLOURS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf"]


def _groups(pts):
    """Colour groups: spin species if present, else ladders / vibrational bands."""
    if pts["spin"].astype(str).str.len().gt(0).any():
        keys = pts["spin"].astype(str)
        return keys, {k: SPIN_COLOURS.get(k, "#333") for k in keys.unique()}, {k: SPIN_NAMES.get(k, k) for k in keys.unique()}
    keys = pts["ladder"].astype(str)
    u = list(dict.fromkeys(keys))
    return keys, {k: LADDER_COLOURS[i % len(LADDER_COLOURS)] for i, k in enumerate(u)}, {k: k for k in u}


def plot_rotation_diagram(features, geometry, res: RotFit | None = None, ax=None, deredden: bool = True, title: str | None = None,
                          label_points: bool = True, molecule_label: str = ""):
    """ln(N_u/g_u) vs E_u: points (de-reddened by the fitted A_V), upper limits, model curves and model points."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, (ax, axr) = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True, gridspec_kw=dict(height_ratios=[3.2, 1], hspace=0.05))
    else:
        fig, axr = ax.figure, None
    av = res.best.get("Av", 0.0) if (res is not None and deredden) else 0.0
    R = 10 ** res.best.get("logR", 0.0) if res is not None else geometry.R_au
    curve = res.model.curve.name if res is not None else "KP5"
    pts = diagram_points(features, geometry, av, curve, R)
    keys, cols, names = _groups(pts)
    for k in cols:
        s = (keys == k).to_numpy()
        d = s & pts["detected"].to_numpy() & pts["use"].to_numpy()
        u = s & ~pts["detected"].to_numpy() & pts["use"].to_numpy()
        x = s & ~pts["use"].to_numpy()
        ax.errorbar(pts["eu"][d], pts["y"][d], yerr=pts["yerr"][d], fmt="o", ms=5, color=cols[k], ecolor=cols[k], capsize=2,
                    label=names[k] or "data", zorder=3)
        if u.any():
            ax.plot(pts["eu"][u], pts["y_ul"][u], "v", ms=6, color=cols[k], mfc="none", zorder=3,
                    label=(names[k] + " " if names[k] else "") + "upper limit")
        if x.any():
            yy = np.where(pts["detected"][x], pts["y"][x], pts["y_ul"][x])
            ax.plot(pts["eu"][x], yy, "x", ms=6, color="#999999", zorder=2)
    if label_points:
        for r in pts.itertuples():
            y = r.y if (r.detected and np.isfinite(r.y)) else r.y_ul
            if np.isfinite(y):
                ax.annotate(str(r.label).split(" +")[0], (r.eu, y), xytext=(4, 3), textcoords="offset points", fontsize=7, color="#555")
    if res is not None:
        mc = model_curves(res)
        E = mc["E"][0]
        ls = ["--", ":", "-."]
        for spin, curves in mc.items():
            if spin == "E":
                continue
            c = SPIN_COLOURS.get(spin, "#000") if spin else "#000"
            if len(curves) > 2:
                for i, cc in enumerate(curves[:-1]):
                    ax.plot(E, cc, ls[i % 3], color=c, lw=1, alpha=0.7)
            ax.plot(E, curves[-1], "-", color=c, lw=1.6, alpha=0.9, label="model" + (f" ({SPIN_NAMES[spin]})" if spin else ""))
        ym = model_points(res, deredden=deredden)
        use = pts["use"].to_numpy()
        ax.plot(pts["eu"][use], ym[use], "s", ms=7, mfc="none", mec="#f5a623", mew=1.2, label="model (per line)", zorder=4)
        if axr is not None:
            det = use & pts["detected"].to_numpy()
            ax_r = (pts["y"] - ym).to_numpy()
            for k in cols:
                s = det & (keys == k).to_numpy()
                axr.errorbar(pts["eu"][s], ax_r[s], yerr=pts["yerr"][s], fmt="o", ms=4, color=cols[k], capsize=2)
            axr.axhline(0, color="#999", lw=0.8)
            axr.set_ylabel("data − model")
    ylab = "ln(N_u / g_u)" + ("  [cm⁻²]" if geometry.column else "  [molecules]")
    ax.set_ylabel(ylab + (f", de-reddened A_V = {av:.1f}" if av else ""))
    (axr if axr is not None else ax).set_xlabel("E_u / k  [K]")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    if title is None and res is not None:
        t = res.table().set_index("param")["median"]
        bits = [f"{p}={t[p]:.3g}" for p in res.free]
        title = f"{molecule_label} {res.model.kind}: " + ", ".join(bits)
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.2)
    return fig


def plot_corner(res: RotFit):
    import corner
    if res.samples is None or len(res.samples) < 10:
        return None
    labels = [res.model.params[k].label or k for k in res.free]
    fig = corner.corner(res.samples, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".3g",
                        title_kwargs=dict(fontsize=9), label_kwargs=dict(fontsize=9))
    return fig


def plot_line_fits(features, stamps: dict, ncols: int = 5, max_panels: int = 40):
    """One panel per measured feature: data − baseline, the fitted profile and the feature's own component."""
    import matplotlib.pyplot as plt
    ids = [int(i) for i in features["id"] if int(i) in stamps][:max_panels]
    if not ids:
        return None
    n = len(ids)
    nr = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nr, ncols, figsize=(2.6 * ncols, 2.0 * nr), squeeze=False)
    fr = features.set_index("id")
    for ax, i in zip(axes.flat, ids):
        s = stamps[i]
        r = fr.loc[i]
        x = s["wave"]
        ax.step(x, s["data"] - s["base"], where="mid", color="#444", lw=0.8)
        ax.plot(x, s["model"] - s["base"], color="#e4572e", lw=1.0)
        ax.plot(x, s["own"], color="#2e86ab", lw=1.0, ls="--")
        ax.axhline(0, color="#bbb", lw=0.6)
        for wl in np.atleast_1d(s["lines"]):
            ax.axvline(wl * (s["center"] / r["wave"]), color="#ccc", lw=0.5, zorder=0)
        tag = "" if r["detected"] else "  (n.d.)"
        ax.set_title(f"{str(r['label'])[:22]}  S/N {r['snr']:.1f}{tag}", fontsize=7.5)
        ax.tick_params(labelsize=6)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------------------------
# publication-style excitation diagram: log10 left axis, ln right axis, observed vs de-reddened points
# per vibrational band, warm / hot components, parameter box; and the model panels side by side
# ------------------------------------------------------------------------------------------------

BAND_STYLE = [("o", "#1f77b4"), ("s", "#d62728"), ("D", "#2ca02c"), ("^", "#9467bd"), ("v", "#ff7f0e")]
_PNAMES = {"T": "T", "T1": "T_{warm}", "T2": "T_{hot}", "Tmin": "T_{min}", "Tmax": "T_{max}", "b": "b", "Av": "A_V",
           "OPR": "OPR", "fwhm": r"\Delta v", "logR": r"\log R"}


def _pm(med, lo, hi, prec=1):
    if not np.isfinite(lo):
        return f"{med:.{prec}f}"
    if abs(lo - hi) < 0.15 * max(lo, hi, 1e-300):
        return f"{med:.{prec}f} \\pm {0.5 * (lo + hi):.{prec}f}"
    return f"{med:.{prec}f}^{{+{hi:.{prec}f}}}_{{-{lo:.{prec}f}}}"


def _pm_log(med, lo, hi, unit):
    """log10 value with errors -> (a ± b) × 10^c."""
    v = 10 ** med
    e = 0.5 * (10 ** (med + hi) - 10 ** (med - lo)) if np.isfinite(lo) else np.nan
    c = int(np.floor(np.log10(v)))
    s = f"{v / 10 ** c:.2f}" + (f" \\pm {e / 10 ** c:.2f}" if np.isfinite(e) else "")
    return f"({s}) \\times 10^{{{c}}}\\,{unit}" if np.isfinite(e) else f"{s} \\times 10^{{{c}}}\\,{unit}"


def param_box_text(res: RotFit) -> str:
    t = res.table().set_index("param")
    col = res.model.geometry.column
    unit = r"\mathrm{cm^{-2}}" if col else ""
    lines = []
    order = {"single": ["T", "logN"], "two": ["T1", "logN1", "T2", "logN2"], "powerlaw": ["b", "Tmin", "Tmax", "logN"]}[res.model.kind]
    for k in order + ["Av", "OPR", "fwhm", "logR"]:
        if k not in t.index:
            continue
        r = t.loc[k]
        if not r["free"] and k in ("fwhm", "logR", "Av", "OPR") and not (k == "Av" and r["median"] > 0):
            continue
        if k.startswith("logN"):
            name = {"logN": "N_{tot}" if res.model.kind == "powerlaw" else "N", "logN1": "N_{warm}", "logN2": "N_{hot}"}[k]
            if not col:
                name = name.replace("N", r"\mathcal{N}", 1)
            lines.append(f"${name} = {_pm_log(r['median'], r['err_lo'], r['err_hi'], unit)}$")
        else:
            prec = 2 if k in ("b", "OPR", "Av") else 1
            u = r"\,\mathrm{K}" if k.startswith("T") else ""
            fixed = " (fixed)" if not r["free"] else ""
            lines.append(f"${_PNAMES.get(k, k)} = {_pm(r['median'], r['err_lo'], r['err_hi'], prec)}{u}$" + fixed)
    lines.append(rf"$\chi^2_{{red}} = {res.chi2_red:.2f}$")
    return "\n".join(lines)


def plot_excitation(features, res: RotFit, ax=None, title: str | None = None, molecule_label: str = "H_2",
                    show_observed: bool = True, label_points: bool = True, box: bool = True, legend: bool = True):
    """Excitation diagram in the usual paper style: log10(N_u/g_u) (left) and ln (right) vs E_u/k; faint
    observed points and solid de-reddened points (also corrected to the para ladder when the OPR is fitted),
    one marker per vibrational band, labelled lines, warm / hot components dashed, the total in green and a
    box with the parameters."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.6, 5.4))
    fig = ax.figure
    geo = res.model.geometry
    av = res.best.get("Av", 0.0)
    R = 10 ** res.best.get("logR", 0.0)
    obs = diagram_points(features, geo, 0.0, res.model.curve.name, R)
    der = diagram_points(features, geo, av, res.model.curve.name, R)
    L10 = np.log(10.0)
    mc = model_curves(res)
    E = mc["E"][0]
    spins = [k for k in mc if k != "E"]
    ref = "p" if "p" in spins else spins[0]
    # ortho points shifted onto the reference (para) ladder when the OPR is a parameter
    shift = np.zeros(len(der))
    if "o" in spins and "p" in spins:
        d = mc["o"][-1] - mc["p"][-1]
        sp = features["spin"].astype(str).to_numpy()
        shift = np.where(sp == "o", -np.interp(der["eu"], E, d), 0.0)
    bands = sorted(dict.fromkeys(features.get("band_v", features.get("ladder")).astype(str)))     # 0-0 first
    use = der["use"].to_numpy(bool)
    for bi, bnd in enumerate(bands):
        mk, colr = BAND_STYLE[bi % len(BAND_STYLE)]
        s = (features.get("band_v", features.get("ladder")).astype(str) == bnd).to_numpy() & use
        det = s & der["detected"].to_numpy()
        ul = s & ~der["detected"].to_numpy()
        vlab = f"$v={bnd.replace('-', '{-}')}$" if bnd[:1].isdigit() else bnd
        if show_observed and det.any():
            ax.errorbar(obs["eu"][det], obs["y"][det] / L10, yerr=obs["yerr"][det] / L10, fmt=mk, ms=5, color=colr, alpha=0.35,
                        mec=colr, capsize=0, zorder=2, label=f"${molecule_label}$({vlab}) observed")
        if det.any():
            y = (der["y"][det] + shift[det]) / L10
            ax.errorbar(der["eu"][det], y, yerr=der["yerr"][det] / L10, fmt=mk, ms=6.5, color=colr, mec="black", mew=0.7, capsize=2,
                        zorder=4, label=f"${molecule_label}$({vlab}) de-reddened" if av else f"${molecule_label}$({vlab})")
        if ul.any():
            ax.plot(der["eu"][ul], (der["y_ul"][ul] + shift[ul]) / L10, "v", ms=6, color=colr, mfc="none", zorder=3)
        if label_points:
            for x, y0, lab, dt, yul in zip(der["eu"][s], (der["y"][s] + shift[s]) / L10, der["label"][s], der["detected"][s],
                                         (der["y_ul"][s] + shift[s]) / L10):
                yy = y0 if dt else yul
                if np.isfinite(yy):
                    ax.annotate(str(lab).split(" +")[0].split(" ")[-1], (x, yy), xytext=(0, 7), textcoords="offset points",
                                fontsize=8.5, color=colr, style="italic", fontweight="bold", ha="center")
    curves = mc[ref]
    if res.model.kind == "two":
        ax.plot(E, curves[0] / L10, "--", color="#1f77b4", lw=1.3, label="warm component")
        ax.plot(E, curves[1] / L10, "--", color="#d62728", lw=1.3, label="hot component")
    ax.plot(E, curves[-1] / L10, "-", color="#2ca02c", lw=1.8, label="total best fit" if res.model.kind == "two" else "model fit")
    yy = np.concatenate([(der["y"][use] + shift[use]) / L10, obs["y"][use] / L10 if show_observed else []])
    yy = yy[np.isfinite(yy)]
    if len(yy):
        pad = 0.08 * np.ptp(yy) + 0.2
        ax.set_ylim(yy.min() - pad, yy.max() + 0.35 * np.ptp(yy) + pad)
    ax.set_xlim(0, max(float(np.nanmax(der["eu"])) * 1.06, 1000))
    unit = r"[cm$^{-2}$]" if geo.column else r"[molecules]"
    ax.set_xlabel(r"$E_u / k_B$  [K]")
    ax.set_ylabel(r"$\log_{10}(N_u/g_u)$  " + unit)
    ax2 = ax.twinx()
    lo, hi = ax.get_ylim()
    ax2.set_ylim(lo * L10, hi * L10)
    ax2.set_ylabel(r"$\ln(N_u/g_u)$  " + unit)
    ax.tick_params(direction="in", top=True); ax2.tick_params(direction="in")
    if legend:
        ax.legend(fontsize=7.5, loc="upper center", ncol=2, frameon=True, framealpha=0.9)
    if box:
        ax.text(0.97, 0.62, param_box_text(res), transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.5", fc="#f7efe2", ec="#c8b27a", alpha=0.95))
    if title is None:
        title = {"single": "One-temperature fit", "two": "Two-component model fit",
                 "powerlaw": r"Power-law temperature distribution fit ($dN \propto T^{-b}\,dT$)"}[res.model.kind]
    ax.set_title(title, fontsize=11)
    return fig


def plot_model_panels(features, fits: dict, molecule_label: str = "H_2"):
    """The fitted models side by side, e.g. {"two": RotFit, "powerlaw": RotFit} -> (a) two-component, (b) power law."""
    import matplotlib.pyplot as plt
    keys = [k for k in ("single", "two", "powerlaw") if k in fits and fits[k] is not None]
    fig, axes = plt.subplots(1, len(keys), figsize=(7.6 * len(keys), 5.6), squeeze=False)
    for i, (ax, k) in enumerate(zip(axes[0], keys)):
        plot_excitation(features, fits[k], ax=ax, molecule_label=molecule_label)
        ax.text(0.5, -0.16, f"({'abc'[i]}) " + {"single": "One-temperature fit.", "two": "Two-component model fit.",
                                                   "powerlaw": "Power-law model fit."}[k],
                transform=ax.transAxes, ha="center", fontsize=11)
    fig.tight_layout()
    return fig
