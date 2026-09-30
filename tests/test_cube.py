"""jalebi.cube: line catalogue, batched Gaussian fits, synthetic cubes with known answers, PSF removal,
velocity maps, stacking, channel maps, PV cuts, regions, FITS/WCS output, config, CLI and the bundled
HV Tau C example."""
import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from jalebi import cube
from jalebi.cli import app
from jalebi.lines import (C_KMS, area_to_cgs_sr, area_to_W_m2, expand_lines, fit_gaussian_batch, fit_line,
                          gaussian_start, get_line, lsf_sigma_um)

runner = CliRunner()


# ---------------------------------------------------------------------------------------------- lines
def test_line_lookup_aliases():
    assert get_line("H2 S(1)").wave == pytest.approx(17.034846)
    assert get_line("h2 s3").name == "H2 S(3)"
    assert get_line("S(2)").name == "H2 S(2)"
    assert get_line("[NeII]").name == "[Ne II] 12.81"
    assert get_line("NeII 12.81").wave == pytest.approx(12.813548)
    assert get_line("feii 17.9").name == "[Fe II] 17.94"
    assert get_line(12.8135).name == "[Ne II] 12.81"
    assert get_line("CO P(10)=4.9876").wave == pytest.approx(4.9876)
    assert [ln.name for ln in expand_lines(["H2 low-J"])] == ["H2 S(1)", "H2 S(2)", "H2 S(3)"]
    assert get_line("[Ne II] 12.81").tag == "NeII_12.81" and get_line("H2 S(1)").tag == "H2_S1"
    with pytest.raises(KeyError):
        get_line("H2")


def test_unit_conversions():
    # 1 Jy um at 10 um = c / lambda^2 * 1e-26 W m^-2
    assert area_to_W_m2(1.0, 10.0) == pytest.approx(2.99792458e14 / 100 * 1e-26)
    assert area_to_cgs_sr(1.0, 10.0) == pytest.approx(2.99792458e14 / 100 * 1e-17)


def test_batch_gaussian_errors_are_calibrated():
    rng = np.random.default_rng(3)
    lam = 12.813548; sig = float(lsf_sigma_um(lam))
    w = np.linspace(12.70, 12.93, 93)
    v_true = 25.0; mu = lam * (1 + v_true / C_KMS)
    n = 3000
    Y = 0.05 * np.exp(-0.5 * ((w - mu) / sig) ** 2)[None] + rng.normal(size=(n, w.size)) * 0.004
    dmu = lam * 200 / C_KMS
    bf = fit_gaussian_batch(w, Y, np.full(Y.shape, 1 / 0.004), gaussian_start(w, Y, lam, sig),
                            [0, lam - dmu, 0.8 * sig], [1, lam + dmu, 2.5 * sig])
    v = C_KMS * (bf.mu - lam) / lam
    pull = (v - v_true) / (bf.mu_err / lam * C_KMS)
    assert bf.ok.mean() > 0.97
    assert abs(np.mean(v) - v_true) < 0.5
    assert 0.9 < np.std(pull) < 1.1


def test_fit_line_1d_flux_and_velocity():
    rng = np.random.default_rng(1)
    lam = 12.813548; sig = float(lsf_sigma_um(lam))
    w = np.linspace(12.70, 12.93, 93)
    mu = lam * (1 + 25.0 / C_KMS); A = 0.05
    y = 0.3 + 0.1 * (w - 12.8) + A * np.exp(-0.5 * ((w - mu) / sig) ** 2) + rng.normal(size=w.size) * 0.004
    r = fit_line(w, y, np.full(w.size, 0.004), "[Ne II]", n_mc=200)
    true_area = A * sig * np.sqrt(2 * np.pi)
    assert r.detected
    assert abs(r.area - true_area) < 3 * r.area_err
    assert abs(r.v_kms - 25.0) < 3 * r.v_err_kms
    assert r.flux_W_m2 == pytest.approx(area_to_W_m2(r.area, r.center_um))


# ---------------------------------------------------------------------------------------------- synthetic cubes
@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth_cube")
    truth = cube.make_synthetic_cube_set(str(d), seed=2)
    return d, truth


def test_cube_set_reads_bands(synth):
    d, _ = synth
    cs = cube.CubeSet(str(d))
    assert cs.bands == ["2B", "3A", "3C"]
    assert {ln.name for ln in cs.lines_covered()} >= {"H2 S(1)", "H2 S(2)", "H2 S(3)", "[Ne II] 12.81"}
    c = cs.cube("3C")
    assert c.north_up and c.pixscale == pytest.approx(0.20, rel=1e-3)
    assert c.pixar_sr == pytest.approx((0.2 / 206264.806) ** 2, rel=1e-6)


def test_source_position_is_subpixel(synth):
    d, truth = synth
    lc = cube.prepare_line(cube.CubeSet(str(d)), "H2 S(1)")
    xt, yt = truth["3C"]["star_pix"]
    assert abs(lc.center_pix[0] - xt) < 0.05 and abs(lc.center_pix[1] - yt) < 0.05


def test_continuum_is_removed_in_line_free_channels(synth):
    d, _ = synth
    lc = cube.prepare_line(cube.CubeSet(str(d)), "H2 S(2)")
    resid = lc.line_data[lc.line_free]
    # residual continuum is at the noise level (15 MJy/sr) although the source peak is ~1e4 MJy/sr
    assert np.nanmedian(np.abs(np.nanmean(resid, axis=0))) < 5.0


@pytest.mark.parametrize("line,band", [("H2 S(2)", "3A"), ("H2 S(3)", "2B"), ("[Ne II] 12.81", "3A")])
def test_point_source_and_extended_flux_recovered(synth, line, band):
    d, truth = synth
    lm = cube.line_maps(cube.prepare_line(cube.CubeSet(str(d)), line), n_mc=0)
    t = truth[band]["lines"][get_line(line).name]
    s = lm.summary
    assert s["point_source_line_flux_W_m2"] == pytest.approx(t["point_W_m2"], rel=0.12)
    assert s["extended_flux_W_m2"] == pytest.approx(t["extended_W_m2"], rel=0.12)
    tot = np.nansum(lm["mom0"]) * 1e-3 * lm.lc.cube.pixar_sr
    assert tot == pytest.approx(t["point_W_m2"] + t["extended_W_m2"], rel=0.05)


def test_velocity_map_recovers_rotation_with_calibrated_errors(synth):
    d, truth = synth
    lm = cube.line_maps(cube.prepare_line(cube.CubeSet(str(d)), "H2 S(2)"), n_mc=60)
    tm = truth["3A"]["maps"]["H2 S(2)"]
    core = lm.lc.psf.core_mask
    ok = np.isfinite(lm["vcen"]) & (tm["mom0_cgs"] > 0.3 * np.nanmax(tm["mom0_cgs"])) & ~core
    assert ok.sum() > 40
    d_v = (lm["vcen"] - tm["mom1_kms"])[ok]
    pull = d_v / lm["vcen_err"][ok]
    assert np.median(np.abs(d_v)) < 3.0
    assert 0.6 < 1.4826 * np.median(np.abs(pull - np.median(pull))) < 1.6
    # the two sides of the ring have opposite signs (Keplerian rotation, PA 30 deg)
    assert np.nanmax(lm["vcen"]) > 25 and np.nanmin(lm["vcen"]) < -25


def test_band_offset_and_zero_point(tmp_path):
    comps = [{"kind": "point", "line": "H2 S(1)", "flux_W_m2": 2e-17},
             {"kind": "ring", "line": "H2 S(1)", "flux_W_m2": 3e-17, "r_arcsec": 1.0, "width_arcsec": 0.3, "incl_deg": 50,
              "pa_deg": 30, "v_kep_kms": 60}]
    c, truth = cube.make_synthetic_cube("3C", comps, v_offset_kms=8.0, noise_mjysr=10.0, seed=4)
    p = c.write(str(tmp_path / "Level3_ch3-long_s3d.fits"))
    cs = cube.CubeSet(p)
    raw = cube.line_maps(cube.prepare_line(cs, "H2 S(1)"), n_mc=0)
    assert raw.summary["source_velocity_kms"] == pytest.approx(8.0, abs=1.5)
    fixed = cube.line_maps(cube.prepare_line(cs, "H2 S(1)", band_offsets_kms={"3C": 8.0}), n_mc=0)
    assert fixed.summary["source_velocity_kms"] == pytest.approx(0.0, abs=1.5)
    zp = cube.line_maps(cube.prepare_line(cs, "H2 S(1)"), n_mc=0, zero_point="star")
    assert zp.summary["zero_point_kms"] == pytest.approx(8.0, abs=1.5)
    assert abs(np.nanmedian(zp["vcen"])) < 5


def test_resampling_wiggles_are_absorbed_by_the_noise_model(tmp_path):
    comps = [{"kind": "point", "line": "[Ne II] 12.81", "flux_W_m2": 1e-17}]
    c, _ = cube.make_synthetic_cube("3A", comps, wiggle_amp=0.03, noise_mjysr=5.0, seed=5)
    lc = cube.prepare_line(c, "[Ne II] 12.81", continuum={"order": 2})
    m = cube.line_maps(lc, n_mc=0)
    # no point-source line is left as fake extended emission, even with 3 % wiggles on a bright continuum
    ext = np.nansum(np.where(m["snr_ext"] >= 5, m["mom0_ext"], 0)) * 1e-3 * lc.cube.pixar_sr
    assert abs(ext) < 0.1e-17


def test_stack_raises_velocity_precision(synth):
    d, _ = synth
    cs = cube.CubeSet(str(d))
    one = cube.line_maps(cube.prepare_line(cs, "H2 S(2)"), n_mc=40)
    st = cube.line_maps(cube.stack_lines(cs, ["H2 S(1)", "H2 S(2)", "H2 S(3)"], name="H2"), n_mc=40)
    assert st.summary["total_flux_W_m2_snr_masked"] is None          # normalised units
    both = np.isfinite(one["vcen_err"]) & np.isfinite(st["vcen_err"])
    assert np.nanmedian(st["vcen_err"][both]) < np.nanmedian(one["vcen_err"][both])
    assert np.nanmax(st["vcen"]) > 20 and np.nanmin(st["vcen"]) < -20


def test_channel_maps_and_pv(synth, tmp_path):
    d, _ = synth
    lc = cube.prepare_line(cube.CubeSet(str(d)), "[Ne II] 12.81")
    cm = cube.channel_maps(lc, -300, 300, dv=60)
    assert cm.data.shape[0] == 10
    p = cm.write_fits(str(tmp_path / "ch.fits"))
    from astropy.io import fits
    h = fits.getheader(p)
    assert h["CTYPE3"] == "VRAD" and h["CDELT3"] == pytest.approx(60)
    # the jet knots (PA -60, +150 km/s at positive offsets) show opposite velocities on either side
    pv = cube.pv_diagram(lc, pa_deg=-60.0, length_arcsec=4.0)
    pos = np.nansum(np.clip(pv.data[:, pv.offset > 0.5], 0, None), axis=1)
    neg = np.nansum(np.clip(pv.data[:, pv.offset < -0.5], 0, None), axis=1)
    vpos = np.sum(pos * pv.velocity) / pos.sum(); vneg = np.sum(neg * pv.velocity) / neg.sum()
    assert vpos > 60 and vneg < -60
    pv.write_fits(str(tmp_path / "pv.fits"))


def test_regions_weights_ds9_and_spectrum(synth, tmp_path):
    d, truth = synth
    cs = cube.CubeSet(str(d))
    c = cs.cube("3C")
    ra, dec = truth["3C"]["star_radec"]
    circ = cube.CircleRegion(ra, dec, 1.0)
    assert circ.weights(c).sum() * c.pixscale ** 2 == pytest.approx(np.pi, rel=0.02)
    ell = cube.EllipseRegion(ra, dec, 1.0, 0.5, 30.0)
    assert ell.weights(c).sum() * c.pixscale ** 2 == pytest.approx(np.pi * 0.5, rel=0.03)
    poly = cube.offset_region("polygon", (ra, dec), -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5)
    assert poly.weights(c).sum() * c.pixscale ** 2 == pytest.approx(1.0, rel=0.03)
    for r in (circ, ell, poly, cube.AnnulusRegion(ra, dec, 0.5, 1.0)):
        back = cube.parse_ds9(r.to_ds9())[0]
        assert type(back) is type(r)
        assert back.area_arcsec2() == pytest.approx(r.area_arcsec2(), rel=1e-3)
    assert cube.parse_ds9(ell.to_ds9())[0].pa == pytest.approx(30.0)
    big = cube.CircleRegion(ra, dec, 3.5)
    spec = cube.region_spectrum(cs, big)
    assert spec.bands == ["2B", "3A", "3C"]
    # a large aperture holds (nearly) all of the injected continuum
    i = spec.band_slice("3C")
    assert np.nanmedian(spec.flux[i]) == pytest.approx(truth["3C"]["continuum_Jy_at_center"], rel=0.05)
    p = tmp_path / "reg.csv"; spec.save(str(p))
    from jalebi.data import load_spectrum
    assert len(load_spectrum(str(p)).wave) == len(spec.wave)


def test_fits_maps_keep_the_wcs(synth, tmp_path):
    d, _ = synth
    lm = cube.line_maps(cube.prepare_line(cube.CubeSet(str(d)), "H2 S(3)"), n_mc=0)
    files = lm.write(str(tmp_path), formats=("fits",))
    from astropy.io import fits
    from astropy.wcs import WCS
    f = next(x for x in files if x.endswith("_mom0.fits"))
    h = fits.getheader(f)
    w = WCS(h)
    ra, dec = w.all_pix2world(10.0, 12.0, 0)
    ra2, dec2 = lm.lc.cube.pix_to_world(10.0, 12.0)
    assert ra == pytest.approx(float(ra2), abs=1e-9) and dec == pytest.approx(float(dec2), abs=1e-9)
    assert h["BUNIT"] == "erg s-1 cm-2 sr-1" and h["LINE"] == "H2 S(3)"
    summ = json.loads(Path(next(x for x in files if x.endswith("_summary.json"))).read_text())
    assert summ["line"] == "H2 S(3)"


def test_config_roundtrip_and_run(synth, tmp_path):
    d, _ = synth
    cfg = cube.example_config("synthetic")
    cfg.path = str(d); cfg.kinematics.n_mc = 20; cfg.formats = ["fits"]
    cfg.regions = [cube.config.CubeRegionConfig(name="east", shape="circle", offset=[0.8, 0.0], r=0.4)]
    p = tmp_path / "c.yaml"; cfg.save(str(p))
    cfg2 = cube.CubeConfig.load(str(p))
    assert cfg2 == cfg
    run = cube.run_cube(cfg2, outdir=str(tmp_path / "out"), verbose=False)
    assert set(run.maps) == {"H2 S(1)", "H2 S(2)", "H2 S(3)", "[Ne II] 12.81"} and "H2" in run.stacks
    assert "east" in run.region_spectra and (tmp_path / "out" / "line_summary.csv").exists()


# ---------------------------------------------------------------------------------------------- bundled example + CLI
def test_bundled_hv_tau_c_example():
    cs = cube.CubeSet("example:HV_Tau_C_cube")
    assert {ln.name for ln in cs.lines_covered()} == {"[Fe II] 5.34", "H2 S(1)", "H2 S(2)", "H2 S(3)", "[Ne II] 12.81"}
    lm = cube.line_maps(cube.prepare_line(cs, "[Fe II] 5.34"), n_mc=0, zero_point="star")
    s = lm.summary
    assert s["extended_flux_W_m2"] > 2 * s["point_source_line_flux_W_m2"]         # the jet dominates
    # the two jet lobes (PA ~ 25 deg) have opposite velocities
    dx, dy = lm.lc.cube.offsets(lm.lc.center_radec)
    along = dx * np.sin(np.deg2rad(25)) + dy * np.cos(np.deg2rad(25))
    good = np.isfinite(lm["vcen"]) & (lm["snr_ext"] > 8)
    north = np.nanmedian(lm["vcen"][good & (along > 0.3)]); south = np.nanmedian(lm["vcen"][good & (along < -0.3)])
    assert north - south > 20


def test_cli_cube_commands(synth, tmp_path):
    d, _ = synth
    r = runner.invoke(app, ["cube", "info", str(d)])
    assert r.exit_code == 0, r.output
    assert "H2 S(1)" in r.output
    r = runner.invoke(app, ["cube", "lines", "--species", "H2"])
    assert r.exit_code == 0 and "17.03485" in r.output
    out = tmp_path / "maps"
    r = runner.invoke(app, ["cube", "maps", str(d), "-l", "H2 S(3)", "--n-mc", "0", "--out", str(out), "--no-png"])
    assert r.exit_code == 0, r.output
    assert (out / "H2_S3" / "SYNTH-DISK_H2_S3_vcen.fits").exists()
    csv = tmp_path / "reg.csv"
    r = runner.invoke(app, ["cube", "region", str(d), "--circle", "0.8 0 0.4", "--offsets", "--out", str(csv), "--no-png"])
    assert r.exit_code == 0, r.output
    assert csv.exists()
    cfgp = tmp_path / "cube.yaml"
    r = runner.invoke(app, ["cube", "init", str(cfgp), "--example", "hv_tau_c"])
    assert r.exit_code == 0 and cfgp.exists()


def test_web_app_cube_workspace_builds():
    pytest.importorskip("panel")
    from jalebi.app import JalebiApp
    a = JalebiApp(start_tab="cube")
    ws = a.cube_ws
    assert a.module == "cube"
    assert "[Fe II] 5.34" in ws.line.options
    ws.n_mc.value = 0
    ws.make_maps()                                 # runs inline without a server
    assert ws.lm is not None and ws.lm.line.name == "[Fe II] 5.34"
    for p in ws.product.options:
        ws.product.value = p
    ws.update_region()
    assert ws.region is not None and "jalebi cube maps" in ws.code.object


def test_zero_point_frame_is_shared_and_idempotent(synth):
    d, _ = synth
    lc = cube.prepare_line(cube.CubeSet(str(d)), "H2 S(1)", rv_kms=12.0)
    v_before = lc.vel.copy()
    m1 = cube.line_maps(lc, n_mc=0, zero_point="star")
    zp = m1.summary["zero_point_kms"]
    assert zp == pytest.approx(-12.0, abs=1.5)
    assert np.allclose(lc.vel, v_before - zp)                 # channel maps / PV / spectra use the same frame
    m2 = cube.line_maps(lc, n_mc=0, zero_point="star")         # a second call does not shift again
    assert np.allclose(lc.vel, v_before - m2.summary["zero_point_kms"])
    sp = m1.spectra["integrated"]
    peak_v = sp["v_kms"][np.nanargmax(sp["line_Jy"])]
    assert abs(peak_v) < 30


def test_stack_is_not_biased_by_missing_channels(synth, tmp_path):
    d, _ = synth
    import shutil
    from astropy.io import fits
    d2 = tmp_path / "holes"; shutil.copytree(d, d2)
    f = d2 / "Level3_ch3-long_s3d.fits"
    with fits.open(f, mode="update") as h:
        w = h["SCI"].header["CRVAL3"] + h["SCI"].header["CDELT3"] * np.arange(h["SCI"].data.shape[0])
        core = np.abs(w - 17.0348) < 0.004
        h["SCI"].data[core, 15:20, 15:20] = np.nan
    good = cube.stack_lines(cube.CubeSet(str(d)), ["H2 S(1)", "H2 S(2)"])
    holes = cube.stack_lines(cube.CubeSet(str(d2)), ["H2 S(1)", "H2 S(2)"])
    sel = np.abs(good.vel) < 100
    a = np.nansum(good.line_data[sel][:, 15:20, 15:20]); b = np.nansum(holes.line_data[sel][:, 15:20, 15:20])
    assert b == pytest.approx(a, rel=0.25)


def test_web_app_rotated_cube_uses_sky_offsets(tmp_path):
    pytest.importorskip("panel")
    comps = [{"kind": "point", "line": "H2 S(1)", "flux_W_m2": 2e-17},
             {"kind": "ring", "line": "H2 S(1)", "flux_W_m2": 3e-17, "r_arcsec": 1.0, "width_arcsec": 0.3, "incl_deg": 50,
              "pa_deg": 30, "v_kep_kms": 60}]
    c, _ = cube.make_synthetic_cube("3C", comps, noise_mjysr=10.0, seed=6)
    t = np.deg2rad(30.0)
    c.wcs.wcs.pc = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    c.write(str(tmp_path / "Level3_ch3-long_s3d.fits"))
    from jalebi.app import JalebiApp
    a = JalebiApp(start_tab="cube", cube_path=str(tmp_path))
    ws = a.cube_ws
    ws.n_mc.value = 0
    ws.line.value = "H2 S(1)"
    ws.make_maps()
    assert not ws.lc.cube.north_up
    for p in ws.product.options:
        ws.product.value = p
    # a click converts sky offsets -> pixel through the WCS, and back
    x, y = ws._offset_to_pix(0.0, 0.0)
    assert x == pytest.approx(ws.lc.center_pix[0], abs=0.05) and y == pytest.approx(ws.lc.center_pix[1], abs=0.05)
    ws.poly_src.data = dict(xs=[[-0.5, 0.5, 0.5, -0.5]], ys=[[-0.5, -0.5, 0.5, 0.5]])
    ws.shape.value = "polygon"; ws.update_region()
    assert ws.region.area_arcsec2() == pytest.approx(1.0, rel=0.02)
    ws.out.value = str(tmp_path / "out")
    ws.make_pv(); ws.write_products()
    assert (tmp_path / "out" / "H2_S1" / "SYNTH-DISK_H2_S1_vcen.fits").exists()


# ---------------------------------------------------------------------------------------------- cube_maps.py recipe
def _crop(c, half):
    """A small copy of a cube around its source (fast tests of the per-spaxel baselines)."""
    _, _, x, y = c.find_source()
    x, y = int(round(x)), int(round(y))
    ys, xs = slice(y - half, y + half + 1), slice(x - half, x + half + 1)
    return cube.Cube(c.sci[:, ys, xs].copy(), c.err[:, ys, xs].copy(), c.wave.copy(), c.wcs[ys, xs], c.header, c.primary,
                     c.band, c.path, c.name, dict(c.meta))


@pytest.fixture(scope="module")
def mini_hv(tmp_path_factory):
    """HV Tau C, [Fe II] 5.34 (ch1-short) and [Ne II] 12.81 (ch3-short) cutouts, ~3" wide, in the old
    cube_maps.py folder layout DATA/HVTAUC/Level3_{channel}_s3d.fits; one spaxel has a NaN channel."""
    d = tmp_path_factory.mktemp("cube_maps_data")
    src = d / "HVTAUC"; src.mkdir()
    cs = cube.CubeSet("example:HV_Tau_C_cube", dq_mask=False, zero_is_nan=False)
    fe = _crop(cs.load(cs.choose(5.3402).path), 10)
    fe.sci[20, 3, 4] = np.nan
    fe.write(str(src / "Level3_ch1-short_s3d.fits"))
    ne = [ci for ci in cs.info if "NeII" in ci.path][0]
    _crop(cs.load(ne.path), 7).write(str(src / "Level3_ch3-short_s3d.fits"))
    return d


def _reference_moment0(path, line_wave, dlambda, component):
    """cube_maps.py make_moment0, re-written with astropy + pybaselines only (no jalebi code): spectral_slab
    (nearest channels), aspls lam=5e6 per spaxel (NaN if any channel is NaN), channel windows, sum x dlambda[m]."""
    from astropy.io import fits
    from pybaselines import Baseline
    with fits.open(path) as h:
        hd = h["SCI"].header
        data = np.array(h["SCI"].data, float)
    w = (hd["CRVAL3"] + hd["CDELT3"] * (np.arange(hd["NAXIS3"]) + 1 - hd["CRPIX3"])) * 1e-6          # metres
    lo = int(np.argmin(np.abs(w - (line_wave - dlambda) * 1e-6))); hi = int(np.argmin(np.abs(w - (line_wave + dlambda) * 1e-6)))
    slab = data[lo:hi + 1]; ws = w[lo:hi + 1]
    cont = np.full_like(slab, np.nan)
    for j in range(slab.shape[1]):
        for i in range(slab.shape[2]):
            y = slab[:, j, i]
            if np.isfinite(y).all():
                cont[:, j, i] = Baseline(ws, check_finite=False).aspls(y, lam=5e6)[0]
    sub = slab - cont
    ci = int(np.argmin(np.abs(ws - line_wave * 1e-6))); n = len(ws)
    if component == "full":
        idx = list(range(max(0, ci - 4), min(n, ci + 5)))
    elif component == "slow":
        idx = list(range(max(0, ci - 2), min(n, ci + 3)))
    else:
        idx = list(range(max(0, ci - 4), max(0, ci - 2))) + list(range(min(n, ci + 3), min(n, ci + 5)))
    part = sub[idx]
    m0 = np.nansum(part, axis=0) * abs(hd["CDELT3"]) * 1e-6
    m0[~np.isfinite(part).any(axis=0)] = np.nan
    return m0


@pytest.mark.parametrize("component", ["full", "slow", "fast"])
def test_cube_maps_make_moment0_reproduces_the_old_recipe(mini_hv, tmp_path, component):
    from astropy.io import fits
    from jalebi.cube import cube_maps as cm
    m, fname = cm.make_moment0([5.3402], ["FeII"], "HVTAUC", 0.1, component=component, save=True, folder_path=str(mini_hv),
                               output_path=str(tmp_path), verbose=False)
    assert fname == str(tmp_path / "moment0_maps" / "HVTAUC" / f"5.3402_FeII_{component}_moment0.fits")
    ref = _reference_moment0(str(mini_hv / "HVTAUC" / "Level3_ch1-short_s3d.fits"), 5.3402, 0.1, component)
    got = fits.getdata(fname)
    assert np.array_equal(np.isnan(ref), np.isnan(got))            # incl. the spaxel with one NaN channel
    assert np.isnan(got[3, 4])
    ok = np.isfinite(ref)
    assert np.nanmax(np.abs(got[ok] - ref[ok])) <= 1e-6 * np.nanmax(np.abs(ref))
    assert len(m.channels) == {"full": 9, "slow": 5, "fast": 4}[component]
    assert fits.getheader(fname)["BUNIT"] == "MJy m / sr"
    # physical units and the jalebi maps of the same window agree with it
    lc = cm.prepare_classic(str(mini_hv / "HVTAUC" / "Level3_ch1-short_s3d.fits"), 5.3402, 0.1)
    mo = cube.moments(lc, component=component, min_valid=1)
    assert np.allclose(mo["mom0"] * cube.native_factor(5.3402, "MJy/sr m"), got, equal_nan=True, rtol=1e-6, atol=0)
    assert cube.native_factor(5.3402, "MJy/sr um") == pytest.approx(1e6 * cube.native_factor(5.3402, "MJy/sr m"))


def test_moment_windows_add_up(mini_hv):
    from jalebi.cube import cube_maps as cm
    lc = cm.prepare_classic(str(mini_hv / "HVTAUC" / "Level3_ch1-short_s3d.fits"), 5.3402, 0.1)
    w = {c: cube.moment_window(lc, component=c)[0] for c in ("full", "slow", "fast")}
    assert (w["full"].sum(), w["slow"].sum(), w["fast"].sum()) == (9, 5, 4)
    assert np.array_equal(w["full"], w["slow"] | w["fast"]) and not (w["slow"] & w["fast"]).any()
    m = {c: cube.moments(lc, component=c, min_valid=1)["mom0"] for c in w}
    ok = np.isfinite(m["full"])
    assert np.allclose(m["full"][ok], (m["slow"] + m["fast"])[ok], rtol=1e-10, atol=1e-30)
    # zero_point="star" shifts the velocities but not the channels of the window
    lc.zero_point_kms = 25.0; lc.vel = lc.vel - 25.0
    assert np.array_equal(cube.moment_window(lc, component="full")[0], w["full"])


def test_get_channel_matches_cube_maps():
    table = [(5.0, "ch1-short"), (5.7, "ch1-medium"), (6.6, "ch1-long"), (8.0, "ch2-short"), (9.66, "ch2-medium"),
             (10.5, "ch2-long"), (12.81, "ch3-short"), (14.0, "ch3-medium"), (17.03, "ch3-long"), (18.0, "ch4-short"),
             (22.0, "ch4-medium"), (25.0, "ch4-long")]
    for w, ch in table:
        assert cube.get_channel(w) == ch
    assert cube.get_channel(5.66) == "ch1-medium"                 # the old code sent exact boundaries to ch4-long
    assert cube.channel_band("ch3-short") == "3A" and cube.band_channel("3A") == "ch3-short"
    cs = cube.CubeSet("example:HV_Tau_C_cube")
    assert cs.choose(12.8135, band="nominal").band == "3A"
    assert cs.choose(12.8135, band="ch3-short").band == "3A"
    with pytest.raises(ValueError):
        cs.choose(12.8135, band="1A")


def test_rms_and_background_masks():
    rng = np.random.default_rng(3)
    img = rng.normal(0, 1.0, (40, 40)); img[18:23, 18:23] += 50
    masked, thr, circ = cube.rms_mask(img, 8, 8, 5, sigma_thresh=3, mode="rms")
    assert thr == pytest.approx(3 * np.nanstd(img[circ])) and circ.sum() == 81
    assert np.isfinite(masked[20, 20]) and np.isnan(masked).mean() > 0.9
    assert cube.region_level(img, 8, 8, 5, "median") == pytest.approx(np.median(img[circ]))
    with pytest.raises(ValueError):
        cube.rms_mask(img, 45, 8, 3)
    with pytest.warns(UserWarning):
        cube.rms_mask(img, 1, 1, 3)                                   # circle over the edge
    bn = cube.background2d(img, 5, (3, 3), exact=False)
    assert bn.shape == img.shape and np.nanmax(np.abs(bn)) < 1.0      # the source is clipped out of the background
    try:
        import photutils  # noqa: F401
    except ImportError:
        return
    img[:3] = np.nan
    bp = cube.background2d(img, 5, (3, 3), exact=True)
    assert np.allclose(bp, cube.background2d(img, 5, (3, 3), exact=False), rtol=1e-10, atol=1e-12)


def test_ratio_map_reprojects_and_masks(mini_hv, tmp_path):
    from astropy.io import fits
    from jalebi.cube import cube_maps as cm
    kw = dict(folder_path=str(mini_hv), output_path=str(tmp_path), verbose=False)
    fig, rm = cm.make_ratio_plot([5.3402, 12.8135], ["FeII", "NeII"], "HVTAUC", center_x=3, center_y=3, radius=2,
                                 sigma_thresh=[3, 3], showfig=False, **kw)
    assert rm.reprojected and rm.ratio.shape == rm.masked1.shape
    f1 = tmp_path / "moment0_maps" / "HVTAUC" / "5.3402_FeII_full_moment0.fits"
    f2 = tmp_path / "moment0_maps" / "HVTAUC" / "12.8135_NeII_full_moment0.fits"
    a1 = fits.getdata(f1)
    thr = 3 * np.nanstd(a1[cube.circle_mask(a1.shape, 3, 3, 2)])
    assert rm.thresholds[0] == pytest.approx(thr)
    assert np.array_equal(np.isfinite(rm.masked1), a1 >= thr)
    ok = rm.valid
    assert ok.sum() > 10 and np.allclose(rm.ratio[ok], rm.masked1[ok] / rm.masked2[ok])
    # the scipy reprojection is the same bilinear sampling as reproject_interp
    r_np = cube.ratio_map(fits.open(f1)[0], fits.open(f2)[0], rms_region=(3, 3, 2), sigma_thresh=(3, 3), exact_reproject=False)
    assert np.array_equal(np.isfinite(r_np.ratio), np.isfinite(rm.ratio))
    assert np.allclose(r_np.ratio[ok], rm.ratio[ok], rtol=1e-6)
    # from LineMaps: the flux ratio = cube_maps.py's ratio x (lambda2 / lambda1)^2
    kw2 = dict(window_um=0.1, continuum={"method": "aspls", "nan_policy": "propagate"}, psf={"enabled": False}, band="nominal")
    cs = cube.CubeSet(str(mini_hv / "HVTAUC"), dq_mask=False, zero_is_nan=False)
    rc = cube.ratio_from_cubes(cs, "5.3402=FeII", "12.8135=NeII", kw2, {"component": "full", "min_valid": 1},
                               rms_region=(3, 3, 2), sigma_thresh=(3, 3))
    both = rc.valid & ok
    assert both.sum() > 10
    assert np.allclose(rc.ratio[both], rm.ratio[both] * (12.8135 / 5.3402) ** 2, rtol=1e-5)
    p = rc.write_fits(str(tmp_path / "ratio.fits"))
    with fits.open(p) as h:
        assert [x.name for x in h] == ["RATIO", "LINE1", "LINE2"] and h[0].header["CTYPE1"].startswith("RA")


def test_channel_slices_like_get_channel_maps(mini_hv, tmp_path):
    import pandas as pd
    from astropy.io import fits
    from jalebi.cube import cube_maps as cm
    df = pd.DataFrame({"Lab wavelength(μm)": [5.3402], "Transitions,J": ["FeII"]})
    root = tmp_path / "proj"; (root / "HVTAUC" / "cubes").mkdir(parents=True)
    import shutil
    shutil.copy(mini_hv / "HVTAUC" / "Level3_ch1-short_s3d.fits", root / "HVTAUC" / "cubes" / "Level3_ch1-short_wcs1_s3d.fits")
    files = cm.get_channel_maps(df, str(root), "HVTAUC", 0.1)
    assert len(files) == 9 and all(Path(f).parent == root / "HVTAUC" / "channel_maps" / "ch_maps_ch1-short" / "5.3402" for f in files)
    lc = cm.prepare_classic(str(root / "HVTAUC" / "cubes" / "Level3_ch1-short_wcs1_s3d.fits"), 5.3402, 0.1)
    sl = cube.channel_slices(lc, "full", source="line")
    assert [Path(f).name for f in files] == sl.filenames() == [f"{float(np.around(w, 4))}.fits" for w in sl.wave_um]
    assert np.allclose(fits.getdata(files[4]), lc.line_data[sl.index[4]], equal_nan=True)
    out = cm.get_moment0(df, str(root), "HVTAUC", 0.1, "slow")
    assert Path(out[0]) == root / "HVTAUC" / "moment0_maps" / "slow" / "5.3402_FeII_moment0.fits"


def test_cli_moment0_ratio_and_recipe_options(mini_hv, tmp_path):
    from astropy.io import fits
    src = str(mini_hv / "HVTAUC")
    r = runner.invoke(app, ["cube", "moment0", src, "-l", "5.3402=FeII", "--component", "fast", "--out", str(tmp_path), "--png"])
    assert r.exit_code == 0, r.output
    f = tmp_path / "moment0_maps" / "HVTAUC" / "5.3402_FeII_fast_moment0.fits"
    assert f.exists() and f.with_suffix(".png").exists()
    ref = _reference_moment0(str(mini_hv / "HVTAUC" / "Level3_ch1-short_s3d.fits"), 5.3402, 0.1, "fast")
    assert np.allclose(fits.getdata(f), ref, equal_nan=True, rtol=1e-6)
    out = tmp_path / "ratio_run"
    r = runner.invoke(app, ["cube", "ratio", src, "-l", "5.3402=FeII", "-l", "12.8135=NeII", "--rms-region", "3 3 2", "--sigma", "3,3",
                            "--continuum", "aspls", "--component", "full", "--window-um", "0.1", "--band", "nominal", "--no-psf",
                            "--no-dq", "--out", str(out)])
    assert r.exit_code == 0, r.output
    assert list((out / "ratios").glob("*_FeII_over_NeII.fits")) and list((out / "ratios").glob("*.png"))
    r = runner.invoke(app, ["cube", "maps", src, "-l", "5.3402=FeII", "--continuum", "aspls", "--component", "slow", "--window-um", "0.1",
                            "--rms-region", "3 3 2", "--no-psf", "--n-mc", "0", "--no-png", "--out", str(tmp_path / "maps")])
    assert r.exit_code == 0, r.output
    summ = json.loads(next((tmp_path / "maps").rglob("*_summary.json")).read_text())
    assert summ["moment_window"]["component"] == "slow" and summ["rms_region"]["radius"] == 2.0
    assert summ["settings"]["continuum"]["method"] == "aspls" and summ["settings"]["window_um"] == 0.1
    assert next((tmp_path / "maps").rglob("*_mom0_masked.fits"))


def test_cube_maps_example_config_validates():
    cfg = cube.example_config("cube_maps")
    assert cfg.continuum.method == "aspls" and cfg.moments.component == "full" and cfg.window_um == 0.1
    assert cfg.band == "nominal" and not cfg.dq_mask and not cfg.psf.enabled and cfg.ratios[0].unit == "MJy/sr m"
    assert cube.CubeConfig.model_validate(cfg.model_dump()) == cfg


def test_web_app_cube_maps_recipe_and_ratio():
    pytest.importorskip("panel")
    from jalebi.app import JalebiApp
    ws = JalebiApp(start_tab="cube").cube_ws
    ws.use_cube_maps_recipe()
    ws.n_mc.value = 0; ws.rms_on.value = True
    ws.make_maps()
    assert ws.lm.summary["moment_window"]["component"] == "full" and "mom0_masked" in ws.lm.maps
    ws.ratio_line.value = "[Ne II] 12.81"
    ws.make_ratio()
    assert ws.ratio is not None and ws.product.value == "ratio"
    for p in ws.product.options:
        ws.product.value = p
    assert "jalebi cube ratio" in ws.code.object and "--continuum aspls" in ws.code.object
    cfg = ws.cube_config()
    assert cfg.ratios and cfg.moments.rms_region == [12.0, 8.0, 3.0] and cfg.continuum.method == "aspls"


def _parse_cli(cmdline):
    """Parse a `jalebi cube ...` command line with the real CLI (options checked, nothing run)."""
    import shlex

    import typer
    from jalebi.cube.cli import cube_app
    args = shlex.split(cmdline)
    assert args[:2] == ["jalebi", "cube"], cmdline
    grp = typer.main.get_command(cube_app)
    ctx = grp.make_context("cube", args[2:], resilient_parsing=False)
    name, cmd, rest = grp.resolve_command(ctx, args[2:])
    cmd.make_context(name, rest)                   # raises on an unknown option
    return name


def test_cli_recipe_equals_make_moment0(mini_hv, tmp_path):
    from astropy.io import fits
    from jalebi.cube import cube_maps as cm
    src = str(mini_hv / "HVTAUC")
    m, _ = cm.make_moment0([5.3402], ["FeII"], "HVTAUC", 0.1, folder_path=str(mini_hv), output_path=str(tmp_path), verbose=False)
    r = runner.invoke(app, ["cube", "maps", src, "-l", "5.3402=FeII", "--recipe", "cube_maps", "--n-mc", "0", "--no-png",
                            "--out", str(tmp_path / "maps")])
    assert r.exit_code == 0, r.output
    got = fits.getdata(next((tmp_path / "maps").rglob("*_FeII_mom0.fits")))
    got = got * cube.native_factor(5.3402, "MJy/sr m")
    assert np.array_equal(np.isnan(got), np.isnan(m.value)) and np.nanmax(np.abs(got - m.value)) <= 1e-6 * np.nanmax(m.value)
    # every option combination the recipe implies parses in each command
    for c in ("maps", "stack", "ratio", "channels"):
        extra = {"maps": '-l "[Fe II] 5.34"', "stack": '-l "H2 S(1)" -l "H2 S(2)"', "ratio": '-l "[Fe II] 5.34" -l "[Ne II] 12.81"',
                 "channels": '-l "[Fe II] 5.34"'}[c]
        assert _parse_cli(f'jalebi cube {c} x {extra} --recipe cube_maps --zeros-valid --nan-policy propagate') == c


def test_cli_ratio_of_gaussian_fluxes_and_clear_errors(mini_hv, tmp_path):
    src = str(mini_hv / "HVTAUC")
    r = runner.invoke(app, ["cube", "ratio", src, "-l", "5.3402=FeII", "-l", "12.8135=NeII", "--key", "gflux", "--sigma", "3,3",
                            "--no-psf", "--out", str(tmp_path)])
    assert r.exit_code == 0, r.output
    assert "spaxels, median" in r.output
    # an RMS circle with no data in it gives a NaN threshold: no ratio, but still a valid FITS file
    r = runner.invoke(app, ["cube", "ratio", src, "-l", "5.3402=FeII", "-l", "12.8135=NeII", "--key", "gflux", "--rms-region", "3 3 2",
                            "--no-psf", "--out", str(tmp_path / "b")])
    assert r.exit_code == 0, r.output
    r = runner.invoke(app, ["cube", "ratio", src, "-l", "5.3402=FeII", "-l", "12.8135=NeII", "--key", "mom0_ext", "--no-psf",
                            "--out", str(tmp_path)])
    assert r.exit_code != 0 and "point source" in r.output
    lm = cube.line_maps(cube.prepare_line(cube.CubeSet(src), "5.3402=FeII", psf={"enabled": False}), kinematics=False)
    with pytest.raises(KeyError, match="velocity fits were off"):
        cube.ratio_map(lm, lm, key="gflux", rms_region=(3, 3, 2))


def test_truncated_windows_warn_and_units_label(mini_hv):
    from jalebi.cube import cube_maps as cm
    path = str(mini_hv / "HVTAUC" / "Level3_ch1-short_s3d.fits")
    lc = cm.prepare_classic(path, 5.3402, 0.1)
    w0 = float(lc.cube.wave[2])                       # a "line" 2 channels from the slab edge
    lc2 = cm.prepare_classic(path, w0, 0.1)
    with pytest.warns(UserWarning, match="stopped with an error"):
        sel, _ = cube.moment_window(lc2, component="fast")
    assert 0 < sel.sum() < 4
    for u, lab in (("MJy sr-1 m", "MJy m / sr"), ("MJy/sr*um", "MJy um / sr"), ("erg/s/cm2/sr", "erg s-1 cm-2 sr-1")):
        m = cm.moment0_map(path, 5.3402, 0.1, unit=u, lc=lc)
        assert m.header["BUNIT"] == lab
    with pytest.raises(ValueError):
        cube.native_factor(5.0, "Jy")
    with pytest.raises(ValueError):
        cube.background2d(np.ones((20, 20)), 5, (2, 2), exact=False)


def test_pipeline_slices_without_channel_maps_and_fixed_band(mini_hv, tmp_path):
    cfg = cube.CubeConfig(path=str(mini_hv / "HVTAUC"), lines=["5.3402=FeII", "12.8135=NeII"], band="1A", psf={"enabled": False},
                          kinematics={"enabled": False}, channels={"slices": True}, formats=["fits"])
    run = cube.run_cube(cfg, outdir=str(tmp_path), verbose=False)
    assert list(run.maps) == ["FeII"]                  # NeII is not in 1A: skipped, not an error
    assert len(list((tmp_path / "FeII" / "channel_slices").glob("*.fits"))) == 9


def test_web_app_keeps_the_line_and_makes_the_ratio_with_line_1_settings(tmp_path):
    pytest.importorskip("panel")
    from jalebi.app import JalebiApp
    ws = JalebiApp(start_tab="cube").cube_ws
    ws.n_mc.value = 0
    ws.line.value = "[Ne II] 12.81"
    ws.use_cube_maps_recipe()                          # changes the DQ / zero rules -> the cubes are reopened
    ws.make_maps()
    assert ws.lm.line.name == "[Ne II] 12.81" and ws.line.value == "[Ne II] 12.81"
    assert ws.lm.summary["settings"]["continuum"]["nan_policy"] == "propagate"
    # settings changed after the maps were made must not leak into line 2 of the ratio
    ws.window_kind.value = "slow (5 ch)"; ws.cont_method.value = "median"
    ws.ratio_line.value = "H2 S(1)"; ws.rms_on.value = False
    ws.make_ratio()
    line1, kw, mk = ws.made
    mk = dict(mk, kinematics=False, rms_region=None)
    lm2 = cube.line_maps(cube.prepare_line(ws.cs, "H2 S(1)", **kw), **mk)
    expect = cube.ratio_map(ws.lm, lm2, sigma_thresh=(5, 5))
    assert np.array_equal(np.isfinite(expect.ratio), np.isfinite(ws.ratio.ratio))
    assert np.allclose(expect.ratio[expect.valid], ws.ratio.ratio[expect.valid])
    # the commands in the code panel are accepted by the CLI
    import html
    import re
    text = html.unescape(re.sub("<[^>]+>", "\n", ws.code.object))
    cmds = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("jalebi cube")]
    assert {_parse_cli(c) for c in cmds} >= {"maps", "ratio"}
    ws.line.value = "stack: H2 S(1) + H2 S(2) + H2 S(3)"
    text = html.unescape(re.sub("<[^>]+>", "\n", ws.code.object))
    assert "stack" in {_parse_cli(ln.strip()) for ln in text.splitlines() if ln.strip().startswith("jalebi cube")}


def test_web_app_ratio_of_a_custom_line_saves_a_runnable_config(tmp_path):
    pytest.importorskip("panel")
    from jalebi.app import JalebiApp
    ws = JalebiApp(start_tab="cube").cube_ws
    ws.n_mc.value = 0; ws.kin_snr.value = 1e9
    ws.custom.value = "12.2786=myH2"
    ws.make_maps()
    assert ws.lm.line.name == "myH2"
    ws.ratio_line.value = "[Ne II] 12.81"
    ws.make_ratio()
    cfg = ws.cube_config()
    assert cfg.lines == ["12.2786=myH2"] and cfg.ratios[0].lines == ["12.2786=myH2", "[Ne II] 12.81"]
    cfg.kinematics.enabled = False; cfg.formats = ["fits"]
    run = cube.run_cube(cube.CubeConfig.model_validate(cfg.model_dump()), outdir=str(tmp_path / "run"), verbose=False)
    assert list(run.ratios) == ["myH2_over_NeII_12.81"]
    ws.out.value = str(tmp_path / "app"); ws.write_products()
    assert (tmp_path / "app" / "ratios" / "myH2_over_NeII_12.81.fits").exists()
