"""End-to-end checks: a synthetic spectrum is fitted back to its input parameters; the CLI works."""
from typer.testing import CliRunner

from jalebi.cli import app
from jalebi.fit import FitProblem, default_free_params
from jalebi.model import Component
from jalebi.synthetic import make_synthetic_spectrum

runner = CliRunner()


def test_synthetic_co2_recovered_by_optimiser():
    """CO2 alone on channel 3B, noise S/N 200, true continuum known: DE + NNLS areas recover T, N, R."""
    truth = [{"name": "CO2", "molecule": "CO2", "logN": 17.3, "T": 520.0, "logR": -0.6}]
    spec, t = make_synthetic_spectrum(components=truth, bands=("3B",), snr=200.0, seed=3, oversample=4)
    spec.continuum = t["continuum"]
    comps = [Component("CO2", "CO2", logN=16.5, T=350.0, logR=-0.3)]
    free = default_free_params(comps)
    prob = FitProblem(spec, comps, [(14.3, 15.5)], free, oversample=4, use_pipeline_err=True)
    opt = prob.optimise(maxiter=40, popsize=10, seed=1, polish=True)
    P, _ = prob.params_from_theta(opt.theta)
    assert abs(P["CO2"]["T"] - 520.0) < 40.0
    assert abs(P["CO2"]["logN"] - 17.3) < 0.25
    assert abs(P["CO2"]["logR"] + 0.6) < 0.1
    assert 0.7 < opt.chi2_red < 1.4


def test_short_mcmc_runs_and_summarises():
    truth = [{"name": "HCN", "molecule": "HCN", "logN": 16.8, "T": 600.0, "logR": -0.7}]
    spec, t = make_synthetic_spectrum(components=truth, bands=("3B",), snr=100.0, seed=5, oversample=3)
    spec.continuum = t["continuum"]
    comps = [Component("HCN", "HCN", logN=16.8, T=600.0, logR=-0.7)]
    prob = FitProblem(spec, comps, [(13.8, 14.4)], default_free_params(comps), oversample=3, use_pipeline_err=True)
    res = prob.mcmc(prob.theta0(), nwalkers=12, nsteps=40, seed=0)
    summ = res.summary()
    assert {"HCN.logN", "HCN.T", "HCN.logR", "HCN.logNA", "HCN.R_au"} <= set(summ.parameter)
    d = res.diagnostics()
    assert 0 < d["acceptance"] < 1
    assert res.correlation().shape == (3, 3)


def test_cli_version_and_banner():
    r = runner.invoke(app, ["--version"])
    assert r.exit_code == 0 and "jalebi" in r.output
    r = runner.invoke(app, [])
    assert r.exit_code == 0 and "Analysis of" in r.output and "Bayesian" in r.output and "Inference" in r.output


def test_cli_doctor_quick():
    r = runner.invoke(app, ["doctor", "--quick"])
    assert r.exit_code == 0, r.output


def test_cli_init_model_synth(tmp_path):
    cfg = tmp_path / "c.yaml"
    r = runner.invoke(app, ["init", str(cfg), "--example", "synthetic"])
    assert r.exit_code == 0 and cfg.exists()
    out = tmp_path / "m.csv"
    r = runner.invoke(app, ["model", "--molecule", "CO2", "--wmin", "14.5", "--wmax", "15.2", "--out", str(out)])
    assert r.exit_code == 0 and out.exists()
    syn = tmp_path / "s.csv"
    r = runner.invoke(app, ["synth", str(syn), "--bands", "3B"])
    assert r.exit_code == 0 and syn.exists() and (tmp_path / "s_truth.yaml").exists()


def test_cli_linedata_list():
    r = runner.invoke(app, ["linedata", "list"])
    assert r.exit_code == 0 and "CO2" in r.output


def test_doctor_full_report():
    from jalebi.doctor import collect
    rep = collect()
    assert rep["ready"], rep.get("fix")
    assert rep["linelists"]["status"] == "ok"
    assert rep["examples"]["FZ_Tau_x1d_files"] == 12
    assert rep["benchmark"]["ms_per_model"] > 0


def test_banner_spells_the_acronym():
    from jalebi._banner import ACRONYM, acronym_line, banner
    assert "".join(i for i, _ in ACRONYM) == "JALEBI"
    assert acronym_line() == "JWST Analysis of Line Emission with Bayesian Inference"
    txt = banner("9.9.9", colour=False)
    assert "v9.9.9" in txt and all(f"{i}{r}" in txt for i, r in ACRONYM)
