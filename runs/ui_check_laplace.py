"""Headless browser check of the 0.19 Laplace UI (Playwright + Chromium).

    python runs/ui_check_laplace.py [--out ui_check] [--port 5077]

Serves the jalebi app with a synthetic HCN + CO2 spectrum already fitted by the optimiser (no MCMC), opens it in
headless Chromium, clicks "Quick errors (Laplace)" in the Fit workspace, waits for the background job, opens
Results and checks that the Laplace table (one row per parameter + derived), the corner plot and the correlation
heatmap are on the page.  Then repeats with an MCMC chain present (contours on top).  Screenshots go to --out.
Exit code 0 = all checks passed.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time


def build_app(with_mcmc: bool):
    from jalebi.app import JalebiApp
    from jalebi.config import ComponentConfig
    from jalebi.pipeline import RunResult, build_problem, run_mcmc_stage, run_optimise_stage
    from jalebi.synthetic import make_synthetic_spectrum
    truth = [{"name": "HCN", "molecule": "HCN", "logN": 17.5, "T": 650.0, "logR": -0.7},
             {"name": "CO2", "molecule": "CO2", "logN": 17.0, "T": 500.0, "logR": -0.5}]
    app = JalebiApp(start_module="lte", start_tab="fit")
    spec, t = make_synthetic_spectrum(components=truth, bands=("3B",), snr=100.0, seed=4, oversample=3)
    spec.continuum = t["continuum"]
    cfg = app.cfg.model_copy(deep=True)
    cfg.components = [ComponentConfig(**c) for c in truth]
    cfg.fit.windows = [[13.7, 14.1], [14.8, 15.05]]; cfg.fit.oversample = 3; cfg.fit.use_pipeline_err = True
    cfg.fit.optimise.maxiter = 15; cfg.fit.optimise.popsize = 8
    cfg.fit.mcmc.nsteps = 400; cfg.fit.mcmc.moves = "de"; cfg.fit.mcmc.checkpoint = False
    run = RunResult(cfg, spec, build_problem(cfg, spec))
    run_optimise_stage(run)
    if with_mcmc:
        run_mcmc_stage(run, verbose=False)
    app.run = run
    app.show_results(run)
    app.laplace_btn.disabled = False
    return app


def serve(port: int):
    """One server and one route, as `jalebi serve`; ?mcmc=1 builds the session with an MCMC chain."""
    # import the app module before the server starts, as `jalebi serve` does: imported inside the first session, its
    # CSS registration (pn.config) would belong to that session only and later browser tabs would be unstyled
    import jalebi.app  # noqa: F401
    import panel as pn

    def page():
        with_mcmc = pn.state.session_args.get("mcmc", [b"0"])[0] == b"1"
        return build_app(with_mcmc).servable()
    pn.serve({"app": page}, port=port, show=False, threaded=True,
             websocket_origin=[f"localhost:{port}", f"127.0.0.1:{port}"])


def check(port: int, out: str, tag: str) -> list[str]:
    from playwright.sync_api import sync_playwright
    fails = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1500, "height": 1000})
        page.goto(f"http://localhost:{port}/app" + ("?mcmc=1" if tag == "mcmc" else ""), timeout=180_000)
        page.wait_for_selector("text=Run fit", timeout=180_000)
        # the Fit workspace tab (Panel tabs are plain elements: select them by their exact text)
        page.get_by_text("Fit", exact=True).first.click()
        btn = page.get_by_role("button", name="Quick errors (Laplace)")
        btn.wait_for(timeout=60_000)
        if btn.is_disabled():
            fails.append("button disabled after the optimiser")
        page.screenshot(path=os.path.join(out, f"{tag}_1_fit_tab.png"))
        btn.click()
        # the job runs in a thread; its label changes while it runs and comes back when the results are on the page
        page.wait_for_selector("text=Quick errors … running", timeout=30_000)
        page.wait_for_selector("text=Quick errors (Laplace)", timeout=180_000)
        page.get_by_text("Results", exact=True).first.click()
        page.wait_for_selector("text=Quick errors — Laplace approximation at the optimum", timeout=60_000)
        time.sleep(2)
        rows = page.locator(".tabulator-row").count()
        if rows < 6:
            fails.append(f"Laplace table has {rows} rows")
        for title in ("Laplace error ellipses", "Laplace correlations"):
            if page.locator(f"text={title}").count() == 0:
                fails.append(f"missing: {title}")
        imgs = page.locator("img").count()
        if imgs < 3:
            fails.append(f"only {imgs} images on the Results page")
        page.locator("text=Laplace error ellipses").first.scroll_into_view_if_needed()
        time.sleep(3)
        page.screenshot(path=os.path.join(out, f"{tag}_2_results.png"))
        # the app's CSS (dark cards) must still be applied: sample the sidebar's first card
        from PIL import Image
        px = Image.open(os.path.join(out, f"{tag}_2_results.png")).convert("RGB").getpixel((30, 105))
        if not (px[2] > px[0] + 15):            # styled sidebar card: navy (51, 59, 80); unstyled: grey (43, 48, 53)
            fails.append(f"the app's panel styling is missing (sidebar pixel {px})")
        bg = page.evaluate("getComputedStyle(document.body).backgroundColor")
        if bg in ("rgb(255, 255, 255)", "rgba(0, 0, 0, 0)"):
            fails.append(f"the app theme is missing (body background {bg})")
        if tag == "mcmc" and page.locator("text=MCMC contours in blue").count() == 0:
            fails.append("the MCMC overlay note is missing")
        b.close()
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ui_check")
    ap.add_argument("--port", type=int, default=5077)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fails = []
    threading.Thread(target=serve, args=(a.port,), daemon=True).start()
    time.sleep(3)
    for tag in ("optimiser", "mcmc"):
        f = check(a.port, a.out, tag)
        print(f"{tag}: {'OK' if not f else 'FAILED: ' + '; '.join(f)}", flush=True)
        fails += f
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
