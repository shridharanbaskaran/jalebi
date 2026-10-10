"""0.23.2: truncated area draws get the full 50 rejection attempts (k was passed as `tries`)."""
import numpy as np

from jalebi.linear import LinearProblem


def test_draw_uses_all_tries_and_truncates_only_areas():
    rng = np.random.default_rng(0)
    mean = np.array([0.05, 1.0, 0.0])            # first area close to zero, one continuum coefficient (mean 0)
    chol = np.eye(3) / 0.1                       # cov = 0.01 I
    n_clip = 0
    for _ in range(400):
        a = LinearProblem._draw(mean, chol, rng, k=2)
        assert np.all(a[:2] > 0)
        n_clip += int(np.isclose(a[0], 1e-12))
    assert n_clip == 0                            # p(a0 > 0) ~ 0.69: 50 tries never all fail
