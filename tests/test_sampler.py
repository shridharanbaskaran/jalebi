"""0.16 sampler options: DE moves, walkers started at the local posterior widths, independent parameter blocks."""
import numpy as np
import pytest

from jalebi.config import MCMCConfig
from jalebi.fit import FitProblem, default_free_params, make_moves
from jalebi.model import Component
from jalebi.synthetic import make_synthetic_spectrum


def _two_region_problem():
    truth = [{"name": "HCN", "molecule": "HCN", "logN": 16.8, "T": 600.0, "logR": -0.7, "windows": [[13.80, 14.20]]},
             {"name": "CO2", "molecule": "CO2", "logN": 17.2, "T": 500.0, "logR": -0.6, "windows": [[14.80, 15.10]]}]
    spec, t = make_synthetic_spectrum(components=truth, bands=("3B",), snr=150.0, seed=4, oversample=3)
    spec.continuum = t["continuum"]
    comps = [Component(**c) for c in truth]
    prob = FitProblem(spec, comps, [(13.8, 14.2), (14.8, 15.1)], default_free_params(comps), oversample=3,
                      use_pipeline_err=True, fit_noise_scale=True)
    return prob


def test_defaults_keep_the_old_sampler():
    m = MCMCConfig()
    assert (m.moves, m.init, m.blocks) == ("stretch", "ball", "joint")


def test_make_moves():
    assert make_moves("stretch") is None
    mv = make_moves("de")
    assert len(mv) == 2 and abs(sum(w for _, w in mv) - 1.0) < 1e-12
    assert len(make_moves("de+stretch")) == 3
    with pytest.raises(ValueError):
        make_moves("hamiltonian")


def test_blocks_split_disjoint_components_and_noise_goes_to_largest():
    prob = _two_region_problem()
    th = prob.theta0()
    groups = prob.independent_blocks(th)
    comps = [{prob.free[j].comp for j in g} for g in groups]
    assert len(groups) == 2
    assert {"HCN"} <= comps[0] | comps[1] and {"CO2"} <= comps[0] | comps[1]
    assert sum(len(g) for g in groups) == prob.ndim
    assert "global" in comps[0]                       # the noise scale goes to the first (largest) group
    # an ordering constraint links them
    prob.ordering = [("HCN", "CO2")]
    assert len(prob.independent_blocks(th)) == 1


def test_local_widths_and_scaled_start():
    prob = _two_region_problem()
    th = prob.theta0()
    w = prob.local_widths(th)
    span = prob.hi - prob.lo
    assert np.all(np.isfinite(w)) and np.all(w > 0) and np.all(w <= 0.1 * span + 1e-12)
    p0, scale = prob.initial_walkers(th, 16, np.random.default_rng(0), init="scaled", widths=w)
    assert p0.shape == (16, prob.ndim)
    assert all(np.isfinite(prob.log_prob(p)) for p in p0)


def test_block_mcmc_merges_into_one_chain():
    prob = _two_region_problem()
    th = prob.theta0()
    res = prob.mcmc(th, nsteps=30, seed=1, moves="de", init="scaled", blocks="auto")
    assert res.chain.shape[0] == 30 and res.chain.shape[2] == prob.ndim
    assert len(res.meta["blocks"]) == 2
    assert res.meta["moves"] == "de"
    # merged ln P equals the full ln P for a sample (exact because the groups share no pixel)
    i = res.chain.shape[1] - 1
    assert abs(res.log_prob[-1, i] - prob.log_prob(res.chain[-1, i])) < 1e-3 * max(1.0, abs(res.log_prob[-1, i]))
    d = res.diagnostics()
    assert d["moves"] == "de" and len(d["blocks"]) == 2
    summ = res.summary()
    assert {"HCN.T", "CO2.T", "global.log_s"} <= set(summ.parameter)


def test_bounds_by_molecule_reach_detected_components():
    from jalebi.config import ComponentConfig, ProjectConfig
    from jalebi.pipeline import free_params
    cfg = ProjectConfig(components=[ComponentConfig(name="CO", molecule="CO", T=1400.0),
                                    ComponentConfig(name="HCN", molecule="HCN"),
                                    ComponentConfig(name="CO_b", molecule="CO", bounds={"T": [500.0, 2000.0]})])
    cfg.fit.bounds_by_molecule = {"CO": {"T": [100.0, 3000.0]}}
    b = {p.key: (p.lo, p.hi) for p in free_params(cfg)}
    assert b["CO.T"] == (100.0, 3000.0)
    assert b["HCN.T"] == (100.0, 1500.0)
    assert b["CO_b.T"] == (500.0, 2000.0)          # the component's own bounds win
