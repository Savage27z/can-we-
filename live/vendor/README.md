# Vendored third-party code

`lightweight-charts.standalone.production.js` is **TradingView Lightweight Charts™ v4.2.3**,
unmodified, from the official npm package `lightweight-charts`
(`https://cdn.jsdelivr.net/npm/lightweight-charts@4.2.3/dist/lightweight-charts.standalone.production.js`).
Copyright (c) TradingView, Inc. Licensed under the Apache License 2.0; the licence text is in
`LICENSE-lightweight-charts`. The package ships no NOTICE file. Charts drawn with it carry the
attribution line "Chart: TradingView Lightweight Charts™ · tradingview.com", as the library asks.

It is used only by `live/chart_tv.py`, the optional TradingView-style chart renderer
(`CHART_RENDERER=tradingview`, off by default). That renderer inlines this file into a local page and
draws it in a headless browser; nothing is fetched at run time.

To update: download the new version from the same package, replace the file, bump the version above,
and run `tests/test_chart_tv.py` (with `CHART_BROWSER_CHANNEL=chrome` to include the browser test).

## Turning the renderer on

1. `pip install -r requirements-tv.txt`, plus a browser: `playwright install chromium`, or set
   `CHART_BROWSER_CHANNEL=chrome` to use an installed Chrome.
2. Set `CHART_RENDERER=tradingview`.

Any failure falls back to the matplotlib chart. **Not enabled in the production Docker image:**
Chromium adds several hundred MB and a render needs a few hundred MB of memory, which can get a small
container killed. Measure that on the target plan before adding it to the Dockerfile.
