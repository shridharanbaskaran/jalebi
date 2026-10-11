# JALEBI landing page

The project website, published to <https://shridharanbaskaran.github.io/jalebi/> by
`.github/workflows/pages.yml` whenever something under `website/` changes on `main`.

| file | what it is |
|---|---|
| `index.html` | the whole page: scroll-driven tour (pinned chart on the right on desktop, on top on phones), video, quickstart, citation, team |
| `data.json` | the real JALEBI output the charts are drawn from (FZ Tau spectrum, continuum, detections, fit, posterior samples; HV Tau C velocity map and H₂ rotation diagram) |
| `media/` | logo spiral, video poster, web-app screenshot and the 90-second explainer video |

No build step: it is static HTML + JS (fonts from Google Fonts). To preview locally:

```bash
cd website && python -m http.server 8000   # then open http://localhost:8000
```

One-time setup on GitHub: **Settings → Pages → Build and deployment → Source: GitHub Actions**.
`website/` is not part of the Python package (the sdist and wheel include only what `pyproject.toml` lists).
