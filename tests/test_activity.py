"""The terminal activity log of the web app (jalebi.activity)."""
import io
import logging

import pytest

import jalebi.activity as act


@pytest.fixture()
def log():
    buf = io.StringIO()
    act.configure("trace", stream=buf, colour=False)
    yield buf
    act.reset()


def test_quiet_unless_configured(capsys):
    act.reset()
    assert not act.enabled(logging.INFO)
    with act.step("nothing to see", "test"):
        pass
    act.info("test", "hidden")
    bar = act.Progress("hidden bar", total=3)
    bar.update(n=3); bar.close()
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""


def test_step_success_and_failure(log):
    with act.step("reading things", "test") as st:
        st.note("3 things")
    with pytest.raises(ValueError):
        with act.step("breaking things", "test"):
            raise ValueError("boom")
    txt = log.getvalue()
    assert "▶ reading things" in txt and "✓ reading things — " in txt and "· 3 things" in txt
    assert "✗ breaking things failed after" in txt and "ValueError: boom" in txt and "Traceback" in txt
    assert " INF test " in txt and " ERR test " in txt


def test_progress_lines_without_a_terminal(log, monkeypatch):
    t = [0.0]
    monkeypatch.setattr(act.time, "perf_counter", lambda: t[0])
    bar = act.Progress("reading cubes", total=10, unit="cube", area="source")
    for i in range(1, 11):
        t[0] += 3.0
        bar.update(n=i, extra=f"cube {i}")
    bar.close()
    lines = [ln for ln in log.getvalue().splitlines() if "reading cubes ▕" in ln]
    assert 5 <= len(lines) <= 11 and "100%" in lines[-1] and "10/10 cube" in lines[-1]
    assert "✓ reading cubes: 10/10 cube in" in log.getvalue()


def test_stages_close_the_previous_stage_complete(log):
    st = act.Stages("fit X", area="fit")
    st.update("grid A", frac=0.9)
    st.update("optimise", frac=0.5)
    st.close()
    txt = log.getvalue()
    assert "✓ fit X · grid A: 100 %" in txt and "✓ fit X · optimise: 50 %" in txt


def test_traced_methods_and_session_tags(log):
    @act.traced("demo", hot=("tick",), skip=("quiet",))
    class Thing:
        def run(self, x, flag=True):
            act.info("demo", "inside run")
            return x * 2

        def tick(self):
            return 1

        def quiet(self):
            return 2

        def fail(self):
            raise RuntimeError("nope")

    t = Thing()
    assert t.run(3, flag=False) == 6 and t.tick() == 1 and t.quiet() == 2
    with pytest.raises(RuntimeError):
        t.fail()
    txt = log.getvalue()
    assert "→ Thing.run(3, flag=False)" in txt and "← Thing.run" in txt
    assert "    inside run" in txt or "  inside run" in txt          # indented under the call
    assert " TRC demo " in txt and "→ Thing.tick()" in txt
    assert "Thing.quiet" not in txt
    assert "✗ Thing.fail raised RuntimeError: nope" in txt
    assert "│" in txt and " — " in txt                             # session column ("—": not in a browser session)


def test_thread_target_carries_the_session(log):
    import threading

    import panel as pn
    from bokeh.document import Document
    pn.state.curdoc = Document()                       # as if a browser session were running this callback
    try:
        tag = act.current_session()
        assert tag.startswith("s")

        def work():
            act.info("demo", "from a worker")
        th = threading.Thread(target=act.thread_target(work))
        th.start(); th.join()
    finally:
        pn.state.curdoc = None
    assert f" {tag}·bg│ from a worker" in log.getvalue().replace("  ", " ").replace(" │", "│")

    def other():                                       # a worker nobody bound: plain "bg"
        act.info("demo", "unbound worker")
    th = threading.Thread(target=other)
    th.start(); th.join()
    line = [ln for ln in log.getvalue().splitlines() if "unbound worker" in ln][0]
    assert " bg " in line and "·bg" not in line


def test_callback_errors_still_raise_when_the_log_is_off():
    import panel as pn
    act.reset()
    act.install_web_hooks()
    with pytest.raises(ZeroDivisionError):
        pn.config.exception_handler(ZeroDivisionError("x"))


def test_table_selection_and_password_typing(log):
    import pandas as pd
    import panel as pn
    t = pn.widgets.Tabulator(pd.DataFrame({"a": [1, 2]}), name="targets")
    act.install_web_hooks()
    t._process_events({"indices": [1]})
    pw = pn.widgets.PasswordInput(name="token")
    pw._process_events({"value_input": "hunter2"})
    pw._process_events({"value": "hunter2"})
    txt = log.getvalue()
    assert "👤 selected rows [1] in 'targets'" in txt
    assert "hunter2" not in txt and "typed into 'token' (hidden)" in txt


def test_describe_and_strip_html():
    import numpy as np
    assert act.describe(np.zeros((3, 4))) == "array(3, 4)"
    assert act.describe(b"x" * 2500) == "2.5 kB"
    assert act.describe("a" * 200, 20).endswith("…") and len(act.describe("a" * 200, 20)) == 20
    assert act.strip_html('<div class="x">✓ <b>FZ Tau</b> ready&nbsp;now</div>') == "✓ FZ Tau ready now"
    assert act.fmt_time(0.0123) == "12 ms" and act.fmt_time(75) == "75.00 s" and act.fmt_time(3700).startswith("1 h")


def test_watch_text_echoes_status_panes(log):
    import panel as pn
    pane = pn.pane.HTML("")
    act.watch_text(pane, "demo", "status")
    pane.object = '<div class="sf-kv">✓ <b>done</b> in 1 s</div>'
    pane.object = '<div class="sf-kv">✓ <b>done</b> in 1 s</div>'          # unchanged: not repeated
    pane.object = '<div class="sf-note">⚠ could not open: missing</div>'
    txt = log.getvalue()
    assert txt.count("status: ✓ done in 1 s") == 1
    assert " WRN demo " in txt and "status: ⚠ could not open: missing" in txt


def test_user_actions_in_the_app_are_logged(log):
    from bokeh.events import ButtonClick
    from jalebi.app import JalebiApp
    app = JalebiApp()
    act.install_web_hooks()
    card = app.cards[0]
    card.logN._slider._process_events({"value": 18.1})                 # dragging: trace
    card.logN._slider._process_events({"value_throttled": 18.3})       # released: info
    app.tabs._process_events({"active": 2})
    app.plot_theme_btn._process_events({"active": 1})
    app.fit_stages._process_events({"active": [0, 2]})
    app.cfg_upload._process_events({"filename": "my.yaml"})
    app.scan_btn._process_event(ButtonClick(model=None))
    txt = log.getvalue()
    assert "👤 dragging 'H2O_hot · log N [cm⁻²]' → 18.1" in txt and " TRC model " in txt
    assert "👤 set 'H2O_hot · log N [cm⁻²]' = 18.3" in txt
    assert "👤 opened the 'Model' tab" in txt
    assert "👤 chose 'plot theme' = 'light plots'" in txt
    assert "👤 chose 'stages' = ['grid', 'mcmc']" in txt
    assert "👤 uploaded file 'my.yaml' (config YAML upload)" in txt
    assert "👤 clicked 'Scan'" in txt and "scanned " in txt
    assert "new browser tab: building the app" in txt and "app ready in" in txt


def test_source_open_and_cache_hits_are_logged(log):
    from jalebi import source as S
    S.forget_sources()
    src = S.open_source("example:HV_Tau_C_cube")
    src.preload(workers=2)
    S.open_source("example:HV_Tau_C_cube")
    src.x1d() if src.has_x1d else None
    txt = log.getvalue()
    assert "opening " in txt and "HV_Tau_C_cube" in txt
    assert "✓ reading the headers of 5 cubes" in txt
    assert "reading 5 cubes with 2 parallel readers" in txt and txt.count("read Level3_") == 5
    assert "is already open in this server (cache hit" in txt
    S.forget_sources()


def test_log_file(tmp_path):
    f = tmp_path / "serve.log"
    act.configure("info", stream=io.StringIO(), colour=False, log_file=str(f))
    try:
        act.info("demo", "to the file")
    finally:
        act.reset()
    assert "to the file" in f.read_text()
