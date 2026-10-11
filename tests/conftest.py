"""Test-suite defaults.

0.24 runs two post-processing stages after every fit by default (continuum robustness, ~20 refits; H I accretion,
a water fit per line).  They would multiply the run time of every pipeline test, so they are switched off here
except in tests marked `post_stages` (tests/test_0240_robustness_accretion.py)."""
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "post_stages: run the 0.24 post-processing stages inside run_pipeline")


@pytest.fixture(autouse=True)
def _no_post_stages(request, monkeypatch):
    if request.node.get_closest_marker("post_stages") is None:
        import jalebi.pipeline as pl
        monkeypatch.setattr(pl, "run_post_stages", lambda run, outdir: None)
    yield
