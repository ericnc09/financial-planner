# Performance Validation — artifact index

Everything here answers one question: **do the system's decisions actually
work?** — tested honestly, with no hindsight and no cherry-picking.

## TL;DR (honest headline)

On a bias-controlled, point-in-time backtest of **2023 insider open-market
buys**, the model's top-15 conviction picks **lagged** the S&P 500 and Nasdaq-100
(2023–2026 was a historic mega-cap bull). The conviction score did **not** predict
returns in this period, and the concentrated selection gave up the breadth edge
that a diversified basket of all insider buys captured. The live since-launch
record (Mar 2026→) is similarly behind the benchmarks so far.

This is a *validation*, and it did its job: it shows the methodology is sound and
the current model needs work. See [METHODOLOGY.md](METHODOLOGY.md) for why the
test is trustworthy, and [VALIDATION_SUMMARY.md](VALIDATION_SUMMARY.md) for the
numbers.

## Files

| Path | What it is |
|------|------------|
| `METHODOLOGY.md` | How the validation works and its limitations (the rigour) |
| `VALIDATION_SUMMARY.md` | Auto-generated results + the 15-pick table |
| `backtest_2023_results.json` | Full 2023 backtest output (picks, returns, equity curve) |
| `track_record_2026_results.json` | Real since-launch record |
| `charts/*.png` | Equity curve, per-pick bars, conviction scatter, live record |
| `site/index.html` | Self-contained demo page for ericcosta.ca (rigour-first framing) |
| `cache/` | Raw EDGAR filings + price data (so results are reproducible) |

## Regenerate from scratch

```bash
python -m scripts.validation.collect_insider_2023     # 2023 EDGAR insider buys (cached/resumable)
python -m scripts.validation.run_backtest_2023        # point-in-time backtest
python -m scripts.validation.run_track_record_2026    # real live record
python -m scripts.validation.make_charts              # PNG charts
python -m scripts.validation.summarize                # VALIDATION_SUMMARY.md
python -m scripts.validation.build_site               # site/index.html
```

## Deploy the demo page to ericcosta.ca

`site/index.html` is one self-contained file (only external dependency is the
Chart.js CDN). To publish:

1. **Wire up signups:** replace `REPLACE_WITH_YOUR_FORM_ENDPOINT` in the `<form>`
   with a Formspree / Buttondown / ConvertKit endpoint. Until then it falls back
   to opening an email draft.
2. **Host it:** drop the file on any static host — Netlify/Vercel drag-and-drop,
   GitHub Pages, Cloudflare Pages, or your existing ericcosta.ca host. No backend
   or API keys are exposed.
3. **Refresh numbers:** re-run the pipeline above and redeploy whenever you want
   the live record updated.

*Not investment advice. Backtested/hypothetical results have inherent limitations;
past performance does not guarantee future results.*
