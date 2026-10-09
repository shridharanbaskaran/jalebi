"""0.22.1: the two failures of the 0.21 survey.

1. Five targets stopped with "At least one parameter value was NaN" inside emcee's DESnookerMove: identical
   walkers make |s - z| = 0 and the move divides 0 by 0.  Identical walkers came from
   FitProblem.initial_walkers putting every walker whose 100 jittered starts had ln P = -inf at theta0.
2. 54 targets fell back from the emulator to the exact model on a spot check that the exact model at
   fit.oversample fails far worse (median 14x) -- 78 % of the survey's sampling time for a less accurate model.
"""
import numpy as np
import pytest
import emcee

from jalebi.fit import FitProblem, SafeDESnookerMove, dedupe_walkers, make_moves
from jalebi.emulator import EmulatorSettings
from jalebi.emulator_shared import fallback_decision


# ----------------------------------------------------------------------------------------------------- walkers
def _lp(x):
    return -0.5 * float(np.sum(x ** 2))


def test_emcee_snooker_crashes_on_identical_walkers_and_the_safe_move_does_not():
    crashed = 0
    for seed in range(12):
        rng = np.random.default_rng(seed); np.random.seed(seed)
        p0 = rng.normal(size=(32, 3)); p0[:16] = p0[0]
        s = emcee.EnsembleSampler(32, 3, _lp, moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)])
        try:
            with np.errstate(all="ignore"):
                s.run_mcmc(p0, 200, skip_initial_state_check=True)
        except ValueError as e:
            assert "NaN" in str(e)
            crashed += 1
        np.random.seed(seed)
        s = emcee.EnsembleSampler(32, 3, _lp, moves=[(emcee.moves.DEMove(), 0.8), (SafeDESnookerMove(), 0.2)])
        s.run_mcmc(p0, 200, skip_initial_state_check=True)           # never raises
        assert np.all(np.isfinite(s.get_chain()))
    assert crashed > 0                                                 # the bug reproduces with emcee's own move


def test_safe_snooker_equals_emcee_snooker_for_distinct_walkers():
    rng = np.random.default_rng(3)
    s = rng.normal(size=(8, 4)); c = [rng.normal(size=(8, 4))]
    a, b = emcee.moves.DESnookerMove(), SafeDESnookerMove()
    qa, ma = a.get_proposal(s, c * 3, np.random.RandomState(5))
    qb, mb = b.get_proposal(s, c * 3, np.random.RandomState(5))
    assert np.array_equal(qa, qb) and np.allclose(ma, mb)


def test_safe_snooker_one_dimensional_block_is_finite():
    s = np.zeros((4, 1)); c = [np.zeros((4, 1))] * 3                   # everything identical, ndim 1
    q, m = SafeDESnookerMove().get_proposal(s, c, np.random.RandomState(0))
    assert np.array_equal(q, s) and np.all(np.isneginf(m))


def test_make_moves_uses_the_safe_snooker():
    mv = make_moves("de", 5)
    assert type(mv[1][0]).__name__ == "SafeDESnookerMove"
    assert make_moves("stretch") is None


class _Narrow:
    """ln P finite only in a tiny box around the optimum (like a corner fit pinned at two bounds)."""
    ndim = 3
    lo = np.array([0.0, 0.0, 0.0]); hi = np.array([1.0, 1.0, 1.0])

    def log_prob(self, x):
        x = np.asarray(x)
        if np.any(x <= self.lo) or np.any(x >= self.hi) or np.any(np.abs(x - 0.999) > 3e-6):
            return -np.inf
        return 0.0

    def local_widths(self, theta):
        return np.full(3, 0.05)


def test_initial_walkers_never_identical_even_when_no_jitter_is_finite():
    prob = _Narrow()
    th0 = np.array([0.999, 0.999, 0.999])
    with pytest.warns(UserWarning, match="ln P = -inf"):
        p0, _ = FitProblem.initial_walkers(prob, th0, 32, np.random.default_rng(0), init="scaled")
    assert len(np.unique(p0, axis=0)) == 32
    assert prob.init_info["non_finite_starts"] > 0
    # the whole run goes through with the safe move
    s = emcee.EnsembleSampler(32, 3, prob.log_prob, moves=make_moves("de", 3))
    np.random.seed(0)
    with np.errstate(all="ignore"):
        s.run_mcmc(p0, 100, skip_initial_state_check=True)
    assert np.all(np.isfinite(s.get_chain()))


def test_dedupe_walkers():
    rng = np.random.default_rng(1)
    p = np.tile([0.5, 0.5], (10, 1))
    q = dedupe_walkers(p, np.zeros(2), np.ones(2), rng)
    assert len(np.unique(q, axis=0)) == 10 and np.abs(q - 0.5).max() < 1e-4
    r = rng.uniform(size=(10, 2))
    assert np.array_equal(dedupe_walkers(r, np.zeros(2), np.ones(2), rng), r)   # nothing to do: unchanged


# ------------------------------------------------------------------------------------------------- emulator
def _sc(es, ef, xs=None, xf=0.0):
    d = {"max_sigma": es, "max_flux": ef, "oversample": 6}
    if xs is not None:
        d["vs_fit_oversample"] = {"oversample": 3, "max_sigma": xs, "max_flux": xf}
    return d


@pytest.mark.parametrize("mode,sc,fall", [
    ("relative", _sc(0.05, 1e-5, 0.9), False),          # passes the targets
    ("relative", _sc(0.24, 2e-5, 3.97, 3e-4), False),   # GD-362 H2O_rovib: exact at oversample 3 >= 3.7 sigma off -> keep
    ("relative", _sc(5.2, 2e-5, 51.7, 3e-4), False),    # HST10 H2O_rovib
    ("relative", _sc(0.90, 0.020, 0.90, 0.0203), True), # a biased table: emulator ~ as far from both exact models
    ("relative", _sc(0.30, 2e-5, 0.35), True),          # exact(fit) may be only 0.05 sigma off: fall back
    ("relative", _sc(0.05, 5e-3, 0.9, 5.1e-3), True),   # integrated flux off by 0.5 %, exact(fit) maybe 0.01 %
    ("relative", _sc(0.24, 2e-5), True),                # fit runs at the table's oversample: exact is the reference
    ("absolute", _sc(0.24, 2e-5, 3.9), True),           # the 0.21 rule
    ("never", _sc(9.0, 1e-2, 0.1), False),
])
def test_fallback_decision(mode, sc, fall):
    st = EmulatorSettings(fallback=mode)
    f, note = fallback_decision(sc, st)
    assert f is fall, note
    if not fall and mode == "relative" and sc["max_sigma"] > st.target_sigma:
        assert note.startswith("kept") and "fit.oversample" in note


def test_fallback_decision_rejects_unknown_mode():
    with pytest.raises(ValueError):
        fallback_decision(_sc(0.2, 0.0, 1.0), EmulatorSettings(fallback="sometimes"))


def test_config_fallback_reaches_the_settings():
    from jalebi.config import EmulatorConfig
    assert EmulatorConfig().settings().fallback == "relative"
    assert EmulatorConfig(fallback="absolute").settings().fallback == "absolute"
