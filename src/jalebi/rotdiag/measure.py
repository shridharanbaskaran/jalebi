"""Measure the line fluxes of the rotation-diagram features in a 1-D spectrum.

For every feature, in the sub-band where it sits furthest from the edges:

* a window of ± window_fwhm instrumental FWHM around the line;
* profile: the member lines as pixel-integrated Gaussians at their rest wavelengths shifted by a common
  velocity v, width = width_scale × instrumental σ (R(λ) of MIRI-MRS, Argyriou et al. 2023, or a constant R),
  weighted by their thin-LTE intensity ratios at t_ref (so a blend is one profile of unit area);
* neighbouring features of the molecule and known lines of other species within joint_fwhm FWHM are fitted
  simultaneously (their own amplitudes), so a partial blend does not leak into the flux;
* a polynomial baseline (order cont_order) fitted at the same time; pixels of stronger unmodelled lines
  (e.g. water in a disk spectrum) are clipped from the fit iteratively;
* the fit is linear in the amplitudes and the baseline, so the flux error comes from the covariance matrix —
  with pixel errors max(pipeline error, the local scatter), scaled by sqrt(χ²_red) when that exceeds 1.

``method="integrate"`` sums (F − baseline) Δλ over the feature's window (Banzatti et al. 2025 windows for the
curated H2O list, else ± integrate_fwhm FWHM).  ``method="gauss_free"`` lets each line's centre and width
float (a single Gaussian per feature).  v and the width scale are measured on the strongest isolated lines
(``velocity="auto"``, ``width="auto"``) or given.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import erf

from ..lines import C_KMS, C_UM_S, fit_gaussian_batch
from .features import _contaminant_catalogue, fwhm_kms

SQ2 = np.sqrt(2.0)
FWHM2SIG = 1.0 / 2.354820045


@dataclass
class MeasureConfig:
    method: str = "gauss"                # gauss | gauss_free | integrate
    velocity: str | float = "auto"       # km/s of the lines relative to the spectrum's frame, or "auto"
    width: str | float = "auto"          # line width / instrumental width, or "auto"
    window_fwhm: float = 8.0
    joint_fwhm: float = 2.0
    core_fwhm: float = 1.2
    cont_order: int = 1
    continuum: str = "local"             # local | spectrum (use spec.continuum, subtracted first)
    snr_detect: float = 3.0
    integrate_fwhm: float = 1.2
    center_shift_kms: float = 60.0
    sigma_range: tuple[float, float] = (0.4, 2.0)
    clip_sigma: float = 3.0
    contaminants: bool = True            # fit known lines of other species inside the window
    blend_rel: float = 0.02              # lines of the molecule outside the selection that are fitted as blends
    scale_errors: bool = True
    refine_snr: float = 15.0             # lines above this S/N get their own velocity and width (0 = never)
    refine_kms: float = 80.0             # ± search range of the per-line velocity
    R_model: str | float = "argyriou2023"
    flux_unit: str = "Jy"                # Jy | MJy/sr (then the fluxes are intensities, W m^-2 sr^-1)


def area_to_si(area, wave_um, flux_unit="Jy"):
    """∫F_ν dλ [unit µm] -> W m^-2 (Jy) or W m^-2 sr^-1 (MJy/sr)."""
    f = 1e-26 if flux_unit.lower() == "jy" else 1e-20
    return np.asarray(area) * C_UM_S * f / np.asarray(wave_um) ** 2


def _edges(w):
    m = 0.5 * (w[1:] + w[:-1])
    return np.concatenate([[w[0] - (m[0] - w[0])], m, [w[-1] + (w[-1] - m[-1])]])


def _gauss_pix(edges, mu, sig):
    """Pixel-averaged unit-area Gaussian (per µm), evaluated on the pixels between `edges`."""
    c = 0.5 * (1.0 + erf((edges[:, None] - mu[None, :]) / (SQ2 * sig[None, :])))
    return (c[1:] - c[:-1]) / np.diff(edges)[:, None]


def _band_pixels(spec, band, wave, half):
    if band and spec.band is not None and (spec.band == band).any():
        i = spec.band_slice(band)
    else:
        i = np.arange(len(spec.wave))
    i = i[np.abs(spec.wave[i] - wave) <= half]
    return i[np.argsort(spec.wave[i])]


def _profile_parts(members, fid, t_ref_rel=True):
    m = members[members["feature"] == fid]
    return m["wave"].to_numpy(), m["rel"].to_numpy() / m["rel"].sum()


def _components(k, r, F, members, fw, contam, cfg, molecule):
    """Profiles fitted together in a feature's window: the feature itself, neighbouring features, blends of the
    molecule that are not features (other bands) and known lines of other species."""
    wf = F["wave"].to_numpy()
    comps = [("self", *_profile_parts(members, r.id))]
    for j in np.flatnonzero((np.abs(wf - r.wave) < cfg.joint_fwhm * fw) & (np.arange(len(F)) != k)):
        comps.append((f"feature {F['top'].iloc[j]}", *_profile_parts(members, F["id"].iloc[j])))
    ow, orel = np.asarray(F.attrs.get("other_wave", []), float), np.asarray(F.attrs.get("other_rel", []), float)
    frl = np.asarray(F.attrs.get("feature_rel", []), float)
    frel = float(frl[k]) if len(frl) == len(F) else 1.0
    for j in np.flatnonzero((np.abs(ow - r.wave) < cfg.joint_fwhm * fw) & (orel >= cfg.blend_rel * frel)):
        if not any(np.min(np.abs(c[1] - ow[j])) < 0.3 * fw for c in comps):
            comps.append((f"{molecule} {ow[j]:.4f}", np.array([ow[j]]), np.array([1.0])))
    if contam is not None and len(contam):
        cw = contam["wave"].to_numpy()
        for j in np.flatnonzero(np.abs(cw - r.wave) < cfg.joint_fwhm * fw):
            if not any(np.min(np.abs(c[1] - cw[j])) < 0.3 * fw for c in comps):
                comps.append((contam["name"].iloc[j], np.array([cw[j]]), np.array([1.0])))
    return comps


def _prep_window(spec, r, fw, cfg, noise, fac=1.0):
    i = _band_pixels(spec, r.band, r.wave * fac, cfg.window_fwhm * fw)
    if len(i) < 8:
        return None
    x = spec.wave[i]
    y = spec.flux[i] - (spec.continuum[i] if cfg.continuum == "spectrum" else 0.0)
    e = np.fmax(np.where(np.isfinite(spec.err[i]) & (spec.err[i] > 0), spec.err[i], np.nan),
                noise[i] if noise is not None else np.nan)
    good = np.isfinite(y) & np.isfinite(e) & spec.mask[i]
    if good.sum() < 8:
        return None
    return x, y, e, good


def _profile_min(grid, chi):
    j = int(np.argmin(chi))
    if 0 < j < len(grid) - 1:
        y0, y1, y2 = chi[j - 1], chi[j], chi[j + 1]
        den = y0 - 2 * y1 + y2
        return float(grid[j] + (0.5 * (y0 - y2) / den if den > 0 else 0.0) * (grid[1] - grid[0]))
    return float(grid[j])


def calibrate(spec, features, members, cfg: MeasureConfig, noise=None, molecule: str = ""):
    """Common line velocity [km/s] and width scale (per MRS channel) from the strongest features.

    Both come from the profile likelihood of the measurement template itself (pixel-integrated Gaussians of
    s × the instrumental σ at velocity v, with the neighbours, blends and the baseline fitted): v on a grid of
    ±max(center_shift_kms, 100) km/s, then s on sigma_range for each channel with >= 2 calibration lines (the
    MRS resolving power departs from any R(λ) law differently in each channel), then v again.  Returns
    (v, {channel: s, "all": s}, n_lines)."""
    v0 = 0.0 if cfg.velocity == "auto" else float(cfg.velocity)
    s0 = 1.0 if cfg.width == "auto" else float(cfg.width)
    if cfg.velocity != "auto" and cfg.width != "auto":
        return v0, {"all": s0}, 0
    contam = _contaminant_catalogue(molecule) if cfg.contaminants else None
    pos = {int(i): j for j, i in enumerate(features["id"])}
    cal = []
    for r in features.nlargest(20, "rel_strength").itertuples():
        fw = float(fwhm_kms(r.wave, cfg.R_model)) / C_KMS * r.wave
        win = _prep_window(spec, r, fw, cfg, noise, 1.0 + v0 / C_KMS)
        if win is None:
            continue
        comps = _components(pos[int(r.id)], r, features, members, fw, contam, cfg, molecule)
        res = _gauss_linear(*win, fw, s0 * fw * FWHM2SIG, 1.0 + v0 / C_KMS, cfg, comps)
        if res is None or not (res[1] > 0) or res[0] < 10 * res[1]:
            continue
        cal.append((r, fw, win, comps, res[6]))            # res[6]: the pixels used (fixed for the profiles)
        if len(cal) >= 12:
            break
    if not cal:
        return v0, {"all": s0}, 0

    def chi_line(item, v, sc):
        r, fw, win, comps, use = item
        res = _gauss_linear(*win, fw, sc * fw * FWHM2SIG, 1.0 + v / C_KMS, cfg, comps, fixed_use=use)
        return np.inf if res is None else res[2] * max(res[3] - len(comps) - cfg.cont_order - 1, 1)

    def chi_of(v, sc_of):
        return sum(chi_line(item, v, sc_of(item[0])) for item in cal)

    scales = {"all": s0}
    chan = lambda r: str(r.band)[:1] if isinstance(r.band, str) and r.band else "all"
    sc_of = lambda r: scales.get(chan(r), scales["all"])
    span = max(cfg.center_shift_kms, 100.0)
    vgrid = np.arange(-span, span + 1e-9, 5.0)
    v = v0
    for it in range(2):
        if cfg.velocity == "auto":
            chis = np.array([chi_of(vv, sc_of) for vv in vgrid])
            if np.isfinite(chis).any():
                v = _profile_min(vgrid, np.where(np.isfinite(chis), chis, np.nanmax(chis[np.isfinite(chis)]) * 10))
        if cfg.width == "auto":
            sgrid = np.linspace(cfg.sigma_range[0], cfg.sigma_range[1], 36)
            per = {}
            for item in cal:
                c = np.array([chi_line(item, v, sc) for sc in sgrid])
                if np.isfinite(c).all():
                    per.setdefault(chan(item[0]), []).append(c - c.min())
            allc = [c for cs in per.values() for c in cs]
            if allc:
                scales["all"] = _profile_min(sgrid, np.sum(allc, axis=0))
                for ch, cs in per.items():
                    if ch != "all" and len(cs) >= 2:
                        scales[ch] = _profile_min(sgrid, np.sum(cs, axis=0))
        if cfg.velocity != "auto" or cfg.width != "auto":
            break
    return float(v), scales, len(cal)


def measure_features(spec, features: pd.DataFrame, members: pd.DataFrame, cfg: MeasureConfig | None = None,
                     molecule: str | None = None, noise=None) -> tuple[pd.DataFrame, dict]:
    """Fluxes of the features.  Returns (features + flux columns, {feature id: arrays for plotting})."""
    cfg = cfg or MeasureConfig()
    F = features.copy()
    if noise is None:
        from ..data import estimate_noise
        noise = estimate_noise(spec)
    noise = np.where(np.isfinite(noise) & (noise > 0), noise, np.nan)
    molecule = molecule or F.attrs.get("molecule", "")
    v, scales, ncal = calibrate(spec, F, members, cfg, noise, molecule)
    contam = _contaminant_catalogue(molecule) if cfg.contaminants else None
    out = {k: np.full(len(F), np.nan) for k in ("area", "area_err", "flux", "flux_err", "snr", "v_kms", "fwhm_kms",
                                                   "chi2_red", "cont_level", "n_pix")}
    ok_fit = np.zeros(len(F), bool)
    notes = [""] * len(F)
    stamps: dict[int, dict] = {}
    fac = 1.0 + v / C_KMS
    for k, r in enumerate(F.itertuples()):
        fw = float(fwhm_kms(r.wave, cfg.R_model)) / C_KMS * r.wave
        s = scales.get(str(r.band)[:1] if isinstance(r.band, str) and r.band else "all", scales["all"])
        sig = s * fw * FWHM2SIG
        i = _band_pixels(spec, r.band, r.wave * fac, cfg.window_fwhm * fw)
        if len(i) < 6:
            notes[k] = "outside the spectrum"; continue
        x = spec.wave[i]
        y = spec.flux[i] - (spec.continuum[i] if cfg.continuum == "spectrum" else 0.0)
        e = np.fmax(np.where(np.isfinite(spec.err[i]) & (spec.err[i] > 0), spec.err[i], np.nan), noise[i])
        good = np.isfinite(y) & np.isfinite(e) & spec.mask[i]
        if good.sum() < 6:
            notes[k] = "masked"; continue
        if not good[np.abs(x - r.wave * fac) < fw].all():
            notes[k] = "masked pixels in the line"
        comps = _components(k, r, F, members, fw, contam, cfg, molecule)
        cores = [c[1] for c in comps]
        if cfg.method == "integrate":
            res = _integrate(x, y, e, good, r, fw, fac, cfg, cores)
        elif cfg.method == "gauss_free":
            res = _gauss_free(x, y, e, good, r, fw, sig, fac, cfg, comps)
        else:
            res = _gauss_linear(x, y, e, good, fw, sig, fac, cfg, comps)
            if (res is not None and cfg.refine_snr > 0 and res[1] > 0 and (res[0] / res[1] >= cfg.refine_snr or res[2] > 3.0)
                    and len(comps) == 1):                       # not next to another line: v could slide onto it
                # a strong line: refine its own velocity and width on a grid (sub-band velocity offsets, lines
                # narrower or wider than the R(λ) law), keeping the linear template
                res = _refine_line(x, y, e, good, fw, cfg, comps, res, s, v)
        if res is None:
            notes[k] = notes[k] or "fit failed"; continue
        area, aerr, chi2r, npix, model, base, used, vk, fwk, own = res
        out["area"][k], out["area_err"][k], out["chi2_red"][k], out["n_pix"][k] = area, aerr, chi2r, npix
        out["flux"][k] = float(area_to_si(area, r.wave, cfg.flux_unit)); out["flux_err"][k] = float(area_to_si(aerr, r.wave, cfg.flux_unit))
        out["snr"][k] = area / aerr if aerr > 0 else np.nan
        out["v_kms"][k], out["fwhm_kms"][k] = vk, fwk
        out["cont_level"][k] = float(np.nanmedian(base)) if np.size(base) else np.nan
        ok_fit[k] = True
        stamps[int(r.id)] = dict(wave=x, data=y, err=e, base=base, model=model, own=own, used=used,
                                 lines=np.concatenate(cores), center=r.wave * fac, fwhm_um=fw)
    for c, a in out.items():
        F[c] = a
    F["detected"] = ok_fit & (F["snr"] >= cfg.snr_detect)
    F["upper_limit"] = np.where(ok_fit, cfg.snr_detect * F["flux_err"], np.nan)
    F["measured"] = ok_fit
    F["note"] = notes
    F["use"] = ok_fit & np.isfinite(F["flux"]) & np.isfinite(F["flux_err"]) & (F["flux_err"] > 0)
    if "blended" in F:                                     # unresolvable blends with other species stay out of the fit
        F["use"] &= ~F["blended"].astype(bool)
        F.loc[F["blended"].astype(bool), "note"] = ["blended with " + c for c in F.loc[F["blended"].astype(bool), "contaminants"]]
    F.attrs.update(features.attrs)
    F["width_scale"] = [scales.get(str(b)[:1] if isinstance(b, str) and b else "all", scales["all"]) for b in F["band"]]
    F.attrs.update(velocity_kms=v, width_scale=scales["all"], width_scales={k: round(float(x), 4) for k, x in scales.items()},
                   n_calibration=ncal, method=cfg.method, flux_unit=cfg.flux_unit)
    return F, stamps


def _refine_line(x, y, e, good, fw, cfg, comps, res0, s0, v0):
    """Profile likelihood of one line over (v, width scale) with the pixels of the first fit held fixed."""
    use0 = res0[6]
    vgrid = np.arange(v0 - cfg.refine_kms, v0 + cfg.refine_kms + 1e-9, 4.0)
    sgrid = np.linspace(cfg.sigma_range[0], cfg.sigma_range[1], 33)
    best = (np.inf, v0, s0)
    chi_v = np.full(len(vgrid), np.inf)
    for i, vv in enumerate(vgrid):
        r = _gauss_linear(x, y, e, good, fw, s0 * fw * FWHM2SIG, 1.0 + vv / C_KMS, cfg, comps, fixed_use=use0)
        if r is not None:
            chi_v[i] = r[2] * max(r[3] - len(comps) - cfg.cont_order - 1, 1)
    if np.isfinite(chi_v).any():
        v1 = _profile_min(vgrid, np.where(np.isfinite(chi_v), chi_v, np.nanmax(chi_v[np.isfinite(chi_v)]) * 10))
    else:
        v1 = v0
    chi_s = np.full(len(sgrid), np.inf)
    for j, sc in enumerate(sgrid):
        r = _gauss_linear(x, y, e, good, fw, sc * fw * FWHM2SIG, 1.0 + v1 / C_KMS, cfg, comps, fixed_use=use0)
        if r is not None:
            chi_s[j] = r[2] * max(r[3] - len(comps) - cfg.cont_order - 1, 1)
    s1 = _profile_min(sgrid, np.where(np.isfinite(chi_s), chi_s, np.nanmax(chi_s[np.isfinite(chi_s)]) * 10)) if np.isfinite(chi_s).any() else s0
    best = (0.0, v1, s1)
    r = _gauss_linear(x, y, e, good, fw, best[2] * fw * FWHM2SIG, 1.0 + best[1] / C_KMS, cfg, comps)
    return r if r is not None and r[2] <= res0[2] else res0


def _design_baseline(x, x0, order, scale):
    t = (x - x0) / scale
    return np.stack([t ** p for p in range(order + 1)], axis=1)


def _gauss_linear(x, y, e, good, fw, sig, fac, cfg, comps, fixed_use=None):
    edges = _edges(x)
    cols = []
    for _, wl, wt in comps:
        cols.append((_gauss_pix(edges, wl * fac, np.full(len(wl), sig)) * wt[None, :]).sum(axis=1))
    G = np.stack(cols, axis=1)
    x0 = float(comps[0][1] @ comps[0][2]) * fac
    B = _design_baseline(x, x0, cfg.cont_order, cfg.window_fwhm * fw)
    X = np.concatenate([G, B], axis=1)
    use = good.copy() if fixed_use is None else fixed_use.copy()
    # pixels near the modelled lines are never clipped (within 2.5 FWHM: the wings of a bright line exceed many σ
    # when the template width or velocity is slightly off, and the refinement step takes care of that)
    core = np.zeros(len(x), bool)
    for _, wl, _ in comps:
        core |= np.min(np.abs(x[:, None] - wl[None, :] * fac), axis=1) < max(2.5, cfg.core_fwhm) * fw
    for _ in range(6 if fixed_use is None else 1):
        if use.sum() <= X.shape[1] + 2:
            return None
        W = 1.0 / e[use] ** 2
        A = X[use] * W[:, None]
        try:
            C = np.linalg.inv(X[use].T @ A)
        except np.linalg.LinAlgError:
            return None
        p = C @ (A.T @ y[use])
        r = (y - X @ p) / e
        # clip unmodelled lines (positive outliers) and gross outliers away from the modelled lines; the threshold
        # also allows for a residual of a few per cent of the brightest modelled line (template imperfections)
        amp = float(np.nanmax(np.abs(X[:, :G.shape[1]] @ p[:G.shape[1]]))) if G.shape[1] else 0.0
        thr = cfg.clip_sigma * e + 0.02 * amp
        resid = y - X @ p
        new = good & ~(((resid > thr) | (np.abs(resid) > 3 * thr)) & ~core)
        if fixed_use is not None or np.array_equal(new, use):
            break
        use = new
    dof = max(use.sum() - X.shape[1], 1)
    chi2r = float(np.sum(r[use] ** 2) / dof)
    Cs = C * (max(chi2r, 1.0) if cfg.scale_errors else 1.0)
    area, aerr = float(p[0]), float(np.sqrt(Cs[0, 0]))
    model = X @ p
    base = B @ p[G.shape[1]:]
    own = G[:, 0] * p[0]
    fwhm = sig / FWHM2SIG / x0 * C_KMS
    return area, aerr, chi2r, int(use.sum()), model, base, use, (fac - 1) * C_KMS, fwhm, own


def _baseline_only(x, y, e, good, cores, fw, fac, cfg):
    core = np.zeros(len(x), bool)
    for wl in cores:
        core |= np.min(np.abs(x[:, None] - wl[None, :] * fac), axis=1) < cfg.core_fwhm * fw
    use = good & ~core
    x0 = float(np.mean(cores[0])) * fac
    B = _design_baseline(x, x0, cfg.cont_order, cfg.window_fwhm * fw)
    for _ in range(6):
        if use.sum() <= B.shape[1] + 2:
            return None
        W = 1.0 / e[use] ** 2
        C = np.linalg.inv(B[use].T @ (B[use] * W[:, None]))
        p = C @ ((B[use] * W[:, None]).T @ y[use])
        r = (y - B @ p) / e
        new = good & ~core & (np.abs(r) < cfg.clip_sigma)
        if np.array_equal(new, use):
            break
        use = new
    return B, p, C, use


def _integrate(x, y, e, good, r, fw, fac, cfg, cores):
    bl = _baseline_only(x, y, e, good, cores, fw, fac, cfg)
    if bl is None:
        return None
    B, p, C, use = bl
    base = B @ p
    if np.isfinite(getattr(r, "xmin", np.nan)) and np.isfinite(getattr(r, "xmax", np.nan)):
        lo, hi = r.xmin * fac, r.xmax * fac
    else:
        lo, hi = (r.wave - cfg.integrate_fwhm * fw) * fac, (r.wave + cfg.integrate_fwhm * fw) * fac
    edges = _edges(x)
    # fractional pixel overlap with [lo, hi]
    ov = np.clip(np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo), 0.0, None)
    ov = np.where(good, ov, 0.0)
    if ov.sum() <= 0:
        return None
    d = y - base
    area = float(np.nansum(d * ov))
    var_pix = float(np.nansum((e * ov) ** 2))
    gB = ov @ B                                   # baseline covariance propagated into the sum
    var_b = float(gB @ C @ gB)
    chi2r = float(np.sum(((y - base) / e)[use] ** 2) / max(use.sum() - B.shape[1], 1))
    aerr = np.sqrt(var_pix + var_b * (max(chi2r, 1.0) if cfg.scale_errors else 1.0))
    own = np.where(ov > 0, d, 0.0)
    return area, float(aerr), chi2r, int((ov > 0).sum()), base + own, base, use, (fac - 1) * C_KMS, np.nan, own


def _gauss_free(x, y, e, good, r, fw, sig, fac, cfg, comps):
    bl = _baseline_only(x, y, e, good, [c[1] for c in comps], fw, fac, cfg)
    if bl is None:
        return None
    B, p, C, use = bl
    base = B @ p
    d = y - base
    near = good & (np.abs(x - r.wave * fac) < 2.5 * fw)
    if near.sum() < 4:
        return None
    dmu = r.wave * cfg.center_shift_kms / C_KMS
    sig0 = fw * FWHM2SIG
    amp0 = max(float(np.nanmax(d[near])), 1e-30)
    p0 = np.array([[amp0, r.wave * fac, sig, 0.0]])
    lo = np.array([0.0, r.wave * fac - dmu, cfg.sigma_range[0] * sig0]); hi = np.array([np.inf, r.wave * fac + dmu, cfg.sigma_range[1] * sig0])
    bf = fit_gaussian_batch(x[near], d[near][None], (1.0 / e[near])[None], p0, lo, hi)
    if not bf.ok[0]:
        return None
    A, mu, sg = bf.amp[0], bf.mu[0], bf.sigma[0]
    own = A * np.exp(-0.5 * ((x - mu) / sg) ** 2)
    res = (d - own)[near] / e[near]
    chi2r = float(np.sum(res ** 2) / max(near.sum() - 3, 1))
    aerr = float(bf.area_err[0]) * (np.sqrt(max(chi2r, 1.0)) if cfg.scale_errors else 1.0)
    return (float(bf.area[0]), aerr, chi2r, int(near.sum()), base + own, base, use | near,
            (mu - r.wave) / r.wave * C_KMS, sg / FWHM2SIG / mu * C_KMS, own)
