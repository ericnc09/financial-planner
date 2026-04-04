"""
Formats analysis results into Slack Block Kit payloads.
"""


def _verdict_emoji(verdict: str) -> str:
    return {
        "BUY": ":large_green_circle:",
        "SELL": ":red_circle:",
        "HOLD": ":white_circle:",
    }.get(verdict, ":white_circle:")


def _verdict_color(verdict: str) -> str:
    return {
        "BUY": "#22c55e",
        "SELL": "#ef4444",
        "HOLD": "#94a3b8",
    }.get(verdict, "#94a3b8")


def format_result(r: dict) -> dict:
    """Format a single ticker's analysis into a Slack attachment."""
    ticker = r["ticker"]
    verdict = r["verdict"]
    price = r["price"]
    buy = r["buy_score"]
    sell = r["sell_score"]
    n = r["n_models"]
    conf = r["confidence"]

    mom30 = f"{r['momentum_30d']:+.1f}%" if r.get("momentum_30d") is not None else "N/A"
    mom90 = f"{r['momentum_90d']:+.1f}%" if r.get("momentum_90d") is not None else "N/A"
    dd = f"-{r['drawdown_52w']:.1f}%" if r.get("drawdown_52w") is not None else "N/A"

    # Build reasoning
    reasons = _build_reasoning(r)

    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{_verdict_emoji(verdict)} {ticker} — {verdict}  (Score: {buy:.0f}/100)",
            },
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Price:* ${price:.2f}"},
                {"type": "mrkdwn", "text": f"*Models:* {n}/5 ({conf:.0%} confidence)"},
                {"type": "mrkdwn", "text": f"*30d Momentum:* {mom30}"},
                {"type": "mrkdwn", "text": f"*90d Momentum:* {mom90}"},
                {"type": "mrkdwn", "text": f"*52w Drawdown:* {dd}"},
                {"type": "mrkdwn", "text": f"*Buy/Sell:* {buy:.0f} / {sell:.0f}"},
            ],
        },
    ]

    # Model component scores
    buy_comps = r.get("buy_components", {})
    if buy_comps:
        comp_lines = []
        name_map = {
            "monte_carlo": "Monte Carlo",
            "hmm_regime": "HMM Regime",
            "garch": "GARCH",
            "fama_french": "Fama-French",
            "copula_tail": "Copula Tail",
        }
        for key, label in name_map.items():
            val = buy_comps.get(key)
            if val is not None:
                comp_lines.append(f"{label}: {val:.0f}")
        if comp_lines:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Model Scores:* {' | '.join(comp_lines)}",
                },
            })

    # Reasoning
    if reasons:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Why {verdict}:*\n{reasons}",
            },
        })

    return {
        "color": _verdict_color(verdict),
        "blocks": blocks,
    }


def _build_reasoning(r: dict) -> str:
    """Generate a human-readable explanation of the verdict."""
    lines = []
    models = r.get("models", {})
    verdict = r["verdict"]
    buy = r["buy_score"]
    sell = r["sell_score"]

    # Monte Carlo reasoning
    mc = models.get("monte_carlo")
    if mc:
        h63 = mc.get("horizons", {}).get(63) or mc.get("horizons", {}).get("63")
        if h63:
            prob = h63["probability_of_profit"] * 100
            exp_ret = h63["expected_return"] * 100
            lines.append(f"Monte Carlo: {prob:.0f}% probability of profit over 63 days (E[return] = {exp_ret:+.1f}%)")

    # HMM reasoning
    hmm = models.get("hmm")
    if hmm:
        state = hmm.get("current_state", "unknown").upper()
        bull_p = hmm.get("prob_bull", 0) * 100
        bear_p = hmm.get("prob_bear", 0) * 100
        lines.append(f"HMM Regime: Currently in {state} state (P(bull)={bull_p:.0f}%, P(bear)={bear_p:.0f}%)")

    # GARCH reasoning
    garch = models.get("garch")
    if garch:
        curr_vol = garch.get("current_conditional_vol_annual", 0) * 100
        f5 = garch.get("forecasts", {}).get(5, {})
        interp = f5.get("interpretation", "stable").replace("_", " ")
        lines.append(f"GARCH: Current vol {curr_vol:.0f}% annualized, 5d forecast {interp}")

    # Fama-French reasoning
    ff = models.get("fama_french")
    if ff:
        alpha = ff.get("alpha_annual", 0) * 100
        beta = ff.get("beta_market", 0)
        lines.append(f"Fama-French: Alpha {alpha:+.1f}%/yr, Beta {beta:.2f}")

    # Copula reasoning
    cop = models.get("copula")
    if cop:
        score = cop.get("tail_risk_score", 0)
        cvar = cop.get("cvar_95", 0) * 100
        label = "low" if score < 30 else "moderate" if score < 60 else "elevated"
        lines.append(f"Copula: {label} tail risk (score {score:.0f}/100, CVaR95 = {cvar:.1f}%)")

    # Verdict summary
    if verdict == "BUY":
        lines.append(f"Buy signal dominant ({buy:.0f} vs sell {sell:.0f}) with >10pt spread")
    elif verdict == "SELL":
        lines.append(f"Sell signal dominant ({sell:.0f} vs buy {buy:.0f}) with >10pt spread")
    else:
        lines.append(f"No clear directional edge (buy {buy:.0f}, sell {sell:.0f})")

    return "\n".join(f"• {l}" for l in lines)


def format_results(results: list[dict], failed: list[str] | None = None) -> dict:
    """Format multiple ticker results into a full Slack message payload."""
    n = len(results)
    buys = sum(1 for r in results if r["verdict"] == "BUY")
    sells = sum(1 for r in results if r["verdict"] == "SELL")
    holds = sum(1 for r in results if r["verdict"] == "HOLD")

    summary = f":bar_chart: *Analysis Complete* — {n} tickers analyzed"
    if buys:
        summary += f" | :large_green_circle: {buys} BUY"
    if holds:
        summary += f" | :white_circle: {holds} HOLD"
    if sells:
        summary += f" | :red_circle: {sells} SELL"

    blocks = [
        {"type": "section", "text": {"type": "mrkdwn", "text": summary}},
        {"type": "divider"},
    ]

    attachments = [format_result(r) for r in results]

    if failed:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f":warning: Could not analyze: {', '.join(failed)}",
            },
        })

    return {"blocks": blocks, "attachments": attachments}
