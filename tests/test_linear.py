"""0.17 linear parameters: the emitting areas profiled (NNLS) or marginalised (Gaussian) out of the MCMC."""
import numpy as np
import pytest

from jalebi.config import MCMCConfig
from jalebi.fit import FitProblem, default_free_params
from jalebi.linear import LinearProblem, linear_audit, normalise_mode
from jalebi.model import Component
from jalebi.synthetic import make_synthetic_spectrum

TRUTH = [{"name": "HCN", "molecule": "HCN", "logN": 17.5, "T": 650.0, "logR": -0.7},
         {"name": "CO2", "molecule": "CO2", "logN": 17.0, "T": 500.0, "logR": -0.5}]
WINDOWS = [(13.7, 14.1), (14.8, 15.05)]


def _two_molecule_problem(noiseless=False, start_offset=False, area_param="logNA", fit_noise_scale=True):
    spec, t = make_synthetic_spectrum(components=TRUTH, bands=("3A", "3B"), snr=100.0, seed=4, oversample=3)
    spec.continuum = t["continuum"]
    if noiseless:                      # "Asimov" data: the posterior is centred on the truth
        spec.flux = t["continuum"] + t["line_flux"]
    comps = [Component(**({**c, "logN": c["logN"] + 0.3, "T": c["T"] + 60, "logR": c["logR"] - 0.1}
                          if start_offset else c)) for c in TRUTH]
    return FitProblem(spec, comps, WINDOWS, default_free_params(comps, area_param=area_param), oversample=3,
                      use_pipeline_err=True, fit_noise_scale=fit_noise_scale, area_param=area_param)


def _structured_problem():
    """Ties, an opacity group, annuli, an absorbing screen, a fixed area and a prior on an area."""
    spec, t = make_synthetic_spectrum(components=TRUTH, bands=("3A", "3B"), snr=100.0, seed=2, oversample=3)
    spec.continuum = t["continuum"]
    comps = [Component("CO2", "CO2", logN=17.0, T=500.0, logR=-0.5),
             Component("13CO2", "13CO2", tie_to="CO2", ratio=70.0),
             Component("C2H2", "C2H2", logN=16.5, T=600.0, logR=-0.9, group="acet"),
             Component("13CCH2", "13CCH2", logN=15.0, T=600.0, logR=-0.9, group="acet"),
             Component("HCN", "HCN", logN=17.0, T=700.0, logR=-0.6, kind="annuli", logRin=-1.3, n_annuli=6),
             Component("HCNabs", "HCN", logN=15.5, T=300.0, kind="absorption", fc=0.5, rv=-5.0),
             Component("H2O", "H2O", logN=17.5, T=500.0, logR=-0.7, linelist_release="hitran"),
             Component("OH", "OH", logN=16.0, T=900.0, logR=-1.0)]
    free = default_free_params(comps, area_param="logR")
    free = [p for p in free if p.key != "H2O.logR"]                       # a fixed area
    for p in free:
        if p.key == "OH.logR":
            p.gauss = (-1.0, 0.2)                                           # a prior on an area
    return FitProblem(spec, comps, [(13.6, 15.2)], free, oversample=3, use_pipeline_err=True,
                      fit_noise_scale=True)


# ---------------------------------------------------------------------------------------------------
def test_defaults_keep_the_old_behaviour():
    m = MCMCConfig()
    assert (m.linear, m.linear_prior, m.linear_prior_scale) == ("sample", "log", None)
    assert normalise_mode("marginalize") == "marginalise" and normalise_mode(None) == "sample"
    with pytest.raises(ValueError):
        normalise_mode("hmc")
    prob = _two_molecule_problem()
    a = prob.mcmc(prob.theta0(), nsteps=20, seed=3, nwalkers=16)
    b = prob.mcmc(prob.theta0(), nsteps=20, seed=3, nwalkers=16, linear="sample")
    assert np.array_equal(a.chain, b.chain) and a.linear == "sample" and a.sampled.all() and a.blobs is None


def test_audit_finds_exactly_the_linear_areas():
    prob = _structured_problem()
    aud = linear_audit(prob).set_index("parameter")
    lin = set(aud.index[aud.linear])
    assert lin == {"CO2.logR", "C2H2.logR"}                    # slab, and the group's single area
    assert not aud.loc["HCN.logR", "linear"] and "outer radius" in aud.loc["HCN.logR", "reason"]
    assert not aud.loc["HCNabs.fc", "linear"]
    assert not aud.loc["13CO2.ratio", "linear"]
    assert not aud.loc["OH.logR", "linear"] and "prior" in aud.loc["OH.logR", "reason"]
    assert "13CCH2.logR" not in aud.index and "H2O.logR" not in aud.index     # group member / fixed area
    lp = LinearProblem(prob, "profile", prob.theta0())
    assert lp.lin_units == ["CO2", "acet"] and lp.ndim == prob.ndim - 2
    assert all(p.name not in ("logR", "logNA") or p.comp in ("HCN", "OH") for p in lp.free)


def test_profile_reproduces_the_full_likelihood_with_ties_groups_annuli_screens():
    """The columns of the linear solve are right: ln L(profile) equals the ordinary ln L at the solved
    areas, with every non-linear unit (annuli, screen, fixed area, prior'd area) subtracted correctly."""
    prob = _structured_problem()
    rng = np.random.default_rng(0)
    th0 = prob.theta0()
    lp = LinearProblem(prob, "profile", th0)
    for _ in range(3):
        th = np.clip(th0 + 0.01 * (prob.hi - prob.lo) * rng.standard_normal(prob.ndim), prob.lo + 1e-6, prob.hi - 1e-6)
        r = lp.reduce(th)
        s = lp.solve(r)
        lo, hi = lp.area_bounds(r)
        assert np.all(s["a"] >= lo * (1 - 1e-9)) and np.all(s["a"] <= hi * (1 + 1e-9))
        full = lp.full_theta(r, s["a"])
        assert abs(prob.log_like(full) - s["lnL"]) < 1e-6 * abs(s["lnL"])
        assert s["lnL"] >= prob.log_like(th) - 1e-6                   # a maximum over the areas


def test_marginal_matches_a_numerical_integral():
    """Gaussian marginal with pixel weights != 1 and noise scale s != 1, against brute-force quadrature."""
    spec, t = make_synthetic_spectrum(components=TRUTH[:1], bands=("3B",), snr=60.0, seed=1, oversample=3)
    spec.continuum = t["continuum"]
    comps = [Component(**TRUTH[0])]
    prob = FitProblem(spec, comps, [(13.7, 13.95), (13.95, 14.1)], default_free_params(comps), oversample=3,
                      use_pipeline_err=True, fit_noise_scale=True, window_weights={1: 3.0})
    th = prob.theta0(); th[-1] = 0.15                                          # log s
    assert len(prob.y) > 100 and set(np.unique(prob.weights)) == {1.0, 3.0}
    for scale in (None, 0.05):
        lp = LinearProblem(prob, "marginalise", th, prior_scale=scale, prior="gaussian")
        r = lp.reduce(th)
        s = lp.solve(r)
        # the default prior adds exactly -ln a_hat (Jacobian of a prior uniform in log R)
        s_log = LinearProblem(prob, "marginalise", th, prior_scale=scale).solve(r)
        assert abs(s_log["lnL"] - (s["lnL"] - np.log(s["a"][0]))) < 1e-9
        sd = float(np.sqrt(1.0 / (s["chol"][0, 0] ** 2)))
        grid = np.linspace(s["a"][0] - 8 * sd, s["a"][0] + 8 * sd, 801)
        lam = lp.prior_scale[0]
        v = np.array([prob.log_like(lp.full_theta(r, [x])) for x in np.maximum(grid, 1e-300)])
        v += -0.5 * grid ** 2 / lam ** 2 - 0.5 * np.log(2 * np.pi * lam ** 2)
        m = v.max()
        num = m + np.log(np.trapezoid(np.exp(v - m), grid))
        assert abs(num - s["lnL"]) < 1e-5
    # the noise scale enters the Occam term: ln L changes with s even at a fixed residual
    lp = LinearProblem(prob, "marginalise", th, prior="gaussian")
    r = lp.reduce(th); r2 = r.copy(); r2[-1] = 0.3
    assert lp.solve(r)["lnL"] != lp.solve(r2)["lnL"]


@pytest.mark.parametrize("mode", ["profile", "marginalise"])
def test_two_molecule_recovery_within_one_sigma(mode):
    """Noiseless two-molecule spectrum, started 0.3 dex / 60 K / 0.1 dex away: the posterior medians of T,
    log N and R are within 1 sigma of the injected values."""
    prob = _two_molecule_problem(noiseless=True, start_offset=True, fit_noise_scale=False)
    opt = prob.optimise(maxiter=40, popsize=10, seed=0)
    res = prob.mcmc(opt.theta, nsteps=700, seed=1, moves="de", init="scaled", linear=mode)
    assert res.linear == mode and res.chain.shape[2] == prob.ndim
    assert list(res.sampled) == [p.name not in ("logNA",) for p in prob.free]
    s = res.summary().set_index("parameter")
    for c in TRUTH:
        for q in ("T", "logN", "logR"):
            r = s.loc[f"{c['name']}.{q}"]
            assert abs(r["median"] - c[q]) <= 0.5 * (r["minus"] + r["plus"]), (mode, c["name"], q, r["median"])
    assert {"HCN.R_au", "HCN.logNmol", "CO2.logR", "CO2.logNA"} <= set(s.index)


def test_profile_and_marginalise_agree_with_sample_on_converged_runs():
    """Noisy spectrum; every run converged (>= 50 tau); T within 30 K and log N within 0.1 dex of sample.
    The slowest test of the suite (~2.5 min): the 7-parameter sample run needs 3400 steps for 50 tau."""
    prob = _two_molecule_problem()
    th = prob.theta0()
    runs = {"sample": prob.mcmc(th, nsteps=3400, seed=1, moves="de", init="scaled"),
            "profile": prob.mcmc(th, nsteps=1200, seed=1, moves="de", init="scaled", linear="profile"),
            "marginalise": prob.mcmc(th, nsteps=1200, seed=1, moves="de", init="scaled", linear="marginalise")}
    med = {}
    for k, r in runs.items():
        tau = r.meta.get("tau_sampled", r.autocorr_time())
        assert r.nsteps >= 50 * np.nanmax(tau), (k, np.nanmax(tau))
        med[k] = r.summary().set_index("parameter")["median"]
    for k in ("profile", "marginalise"):
        for c in ("HCN", "CO2"):
            assert abs(med[k][f"{c}.T"] - med["sample"][f"{c}.T"]) <= 30.0
            assert abs(med[k][f"{c}.logN"] - med["sample"][f"{c}.logN"]) <= 0.1
            assert abs(med[k][f"{c}.logR"] - med["sample"][f"{c}.logR"]) <= 0.1
    # fewer dimensions -> fewer walkers and a shorter autocorrelation time
    assert runs["profile"].chain.shape[1] <= runs["sample"].chain.shape[1]
    assert np.nanmax(runs["profile"].meta["tau_sampled"]) < np.nanmax(runs["sample"].autocorr_time())


def test_blocks_auto_and_a_block_with_no_nonlinear_parameter_left():
    truth = [{**TRUTH[0], "windows": [[13.70, 14.10]]}, {**TRUTH[1], "windows": [[14.80, 15.05]]}]
    spec, t = make_synthetic_spectrum(components=truth, bands=("3B",), snr=150.0, seed=4, oversample=3)
    spec.continuum = t["continuum"]
    comps = [Component(**c) for c in truth]
    prob = FitProblem(spec, comps, WINDOWS, default_free_params(comps), oversample=3, use_pipeline_err=True,
                      fit_noise_scale=True)
    th = prob.theta0()
    res = prob.mcmc(th, nsteps=40, seed=1, moves="de", init="scaled", blocks="auto", linear="profile")
    assert len(res.meta["blocks"]) == 2 and res.chain.shape[2] == prob.ndim
    assert np.all(np.isfinite(res.chain))
    i = res.chain.shape[1] - 1
    # ln P of the merged chain = profile ln P of the sample (exact: the blocks share no pixel)
    lp = LinearProblem(prob, "profile", th)
    assert abs(res.log_prob[-1, i] - lp.log_prob(lp.reduce(res.chain[-1, i]))) < 1e-3 * abs(res.log_prob[-1, i])
    # CO2 with log N and T fixed: only its (linear) area is left -> its block disappears, its area is still
    # solved at every call and filled into the chain
    free = [p for p in default_free_params(comps) if p.key not in ("CO2.logN", "CO2.T")]
    prob2 = FitProblem(spec, comps, WINDOWS, free, oversample=3, use_pipeline_err=True, fit_noise_scale=True)
    res2 = prob2.mcmc(prob2.theta0(), nsteps=40, seed=1, moves="de", init="scaled", blocks="auto", linear="marginalise")
    j = [p.key for p in prob2.free].index("CO2.logR")
    col = res2.chain[:, :, j]
    assert np.all(np.isfinite(col)) and np.std(col) > 0 and abs(np.median(col) + 0.5) < 0.05
    assert all("CO2" not in k for g in res2.meta["blocks"] for k in g)


def test_chain_file_layout_is_backward_compatible(tmp_path):
    prob = _two_molecule_problem()
    res = prob.mcmc(prob.theta0(), nsteps=30, seed=0, linear="marginalise")
    f = tmp_path / "chain.npz"
    res.save(str(f))
    z = np.load(f, allow_pickle=False)
    assert {"chain", "log_prob", "acceptance", "runtime_s", "names", "labels"} <= set(z.files)
    assert z["chain"].shape[2] == len(z["names"]) == prob.ndim
    assert [str(x) for x in z["names"]] == [p.key for p in prob.free]
    assert str(z["linear"]) == "marginalise" and z["sampled"].sum() == prob.ndim - 2
    d = res.diagnostics()
    assert d["linear"] == "marginalise" and set(d["linear_neg_frac"]) == {"HCN", "CO2"}
    assert d["linear_params"] == ["HCN.logNA", "CO2.logNA"] and "HCN.logNA" not in d["sampled"]


def test_pipeline_option_reaches_the_sampler(tmp_path):
    from jalebi.config import ComponentConfig, ProjectConfig
    from jalebi.pipeline import RunResult, build_problem, run_mcmc_stage, save_results
    spec, t = make_synthetic_spectrum(components=TRUTH, bands=("3A", "3B"), snr=100.0, seed=4, oversample=3)
    spec.continuum = t["continuum"]
    cfg = ProjectConfig(components=[ComponentConfig(**c) for c in TRUTH])
    cfg.fit.windows = [list(w) for w in WINDOWS]
    cfg.fit.oversample = 3; cfg.fit.use_pipeline_err = True; cfg.fit.area_param = "logNA"
    cfg.fit.mcmc.nsteps = 30; cfg.fit.mcmc.linear = "profile"; cfg.fit.mcmc.checkpoint = False
    prob = build_problem(cfg, spec)
    run = RunResult(cfg, spec, prob)
    run_mcmc_stage(run, verbose=False)
    assert run.mcmc.linear == "profile" and any("profiled (NNLS)" in line for line in run.log)
    save_results(run, str(tmp_path))
    import pandas as pd
    summ = pd.read_csv(tmp_path / "summary.csv")
    assert {"HCN.logNA", "HCN.R_au", "HCN.logNmol", "CO2.logR"} <= set(summ.parameter)
    z = np.load(tmp_path / "chain.npz")
    assert str(z["linear"]) == "profile"
