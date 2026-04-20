---
description: Generate an 8-section institutional equity research report for a ticker, backed by this repo's quant pipeline (Monte Carlo, HMM, GARCH, Fama-French, copula, Bayesian decay, ensemble scoring).
argument-hint: <TICKER> [--profile conservative|moderate|aggressive] [--portfolio <dollars>]
---

# /equity-report

You are about to generate a full equity research report for the user using this repo's statistical pipeline. The target ticker and optional flags are in `$ARGUMENTS`.

## Parse arguments

From `$ARGUMENTS`, extract:

- **TICKER** — first positional token, uppercase. Required. If missing, ask the user for a ticker and stop.
- **--profile** — one of `conservative`, `moderate`, `aggressive`. Defaults to `moderate`.
- **--portfolio** — dollar amount for position sizing. Defaults to `100000`.

If the user provides anything that does not parse (e.g. an unsupported profile), respond with the allowed options and stop.

## Preferred path: hit the local API

If a local dev server is already running, call the report endpoint:

```bash
curl -s "http://localhost:8000/api/export/report/${TICKER}?format=md&risk_profile=${PROFILE}&portfolio_value=${PORTFOLIO}"
```

Render the markdown body directly in chat. Do not summarise or paraphrase — the report is meant to be read verbatim, disclaimer included.

If the server returns 404 (no analysis), tell the user the ticker is not in the analysis universe and ask whether they want to run a fresh analysis via Python (see fallback below).

If the `curl` call fails because the server is not running (connection refused), fall back to running it in-process.

## Fallback: run the pipeline directly

Execute the report generator inside the repo's virtualenv:

```bash
.venv/bin/python - <<'PY'
import asyncio
from src.api.api import _run_full_analysis
from src.reporting.markdown_report import generate_report

TICKER = "${TICKER}"
PROFILE = "${PROFILE}"
PORTFOLIO = ${PORTFOLIO}

async def main():
    analysis = await _run_full_analysis(TICKER)
    if not analysis:
        print(f"No analysis available for {TICKER}")
        return
    md = await generate_report(
        TICKER, analysis,
        risk_profile=PROFILE,
        portfolio_value=PORTFOLIO,
    )
    print(md)

asyncio.run(main())
PY
```

Stream the resulting markdown to the user. Again, do not paraphrase.

## After the report

End your response with a single line asking whether the user wants a follow-up (e.g. different profile, peer drill-down, catalyst deep-dive). Do not append any additional analysis beyond the report itself — the disclaimer must be the last substantive block.

## Constraints

- Never recommend trade execution. Never add language like "you should buy" or "this is a guaranteed winner" — the compliance filter will flag those terms and they would be redundant with the report body.
- Do not generate a report if `$ARGUMENTS` is empty. Ask for a ticker.
- If the pipeline warns that model coverage is below 50% (fewer than ~5 models returned data), flag that clearly before the report body — the conviction number is less reliable with sparse coverage.
