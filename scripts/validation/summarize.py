"""
Render reports/VALIDATION_SUMMARY.md from the results JSON files.

Plain-language results with the key tables, auto-filled so the summary always
matches the latest run. Pairs with METHODOLOGY.md (which explains the *how*).
"""

import json
from pathlib import Path

BT = Path("reports/backtest_2023_results.json")
TR = Path("reports/track_record_2026_results.json")
OUT = Path("reports/VALIDATION_SUMMARY.md")


def pct(x, plus=True):
    if x is None:
        return "—"
    return f"{x*100:+.1f}%" if plus else f"{x*100:.1f}%"


def main():
    lines = ["# Smart Money Follows — Validation Results", "",
             "*Auto-generated from the latest backtest run. See "
             "[METHODOLOGY.md](METHODOLOGY.md) for how these are computed and "
             "their limitations.*", ""]

    if BT.exists():
        bt = json.loads(BT.read_text())
        s = bt["summary"]
        lines += [
            "## Pillar 1 — 2023 point-in-time backtest", "",
            f"- **Universe:** {bt['universe']} — {bt['n_raw_events']} open-market "
            f"insider buys scanned, {bt['n_scored_events']} scored.",
            f"- **Portfolio (top 15, equal weight, to date):** "
            f"**{pct(s['portfolio_to_date'])}**",
            f"- **S&P 500 (matched cash, same windows):** {pct(s['sp500_to_date_matched'])}",
            f"- **Nasdaq-100 (matched cash, same windows):** {pct(s['nasdaq_to_date_matched'])}",
            f"- **Picks that beat the S&P 500:** {s['picks_beating_sp500']}",
            f"- **Unfiltered universe avg (all {s['unfiltered_universe_n']} names, to date):** "
            f"{pct(s['unfiltered_universe_avg_to_date'])} "
            f"— vs top-15 avg {pct(s['avg_pick_to_date'])} "
            f"(does the conviction ranking add value?)",
            "",
            "**Average return by horizon (top 15):**", "",
            "| Horizon | 1m | 3m | 6m | 12m | To date |",
            "|---|---|---|---|---|---|",
            "| Avg | " + " | ".join(pct(s["avg_return"].get(h)) for h in
                                     ["1m", "3m", "6m", "12m", "to_date"]) + " |",
            "| Median | " + " | ".join(pct(s["median_return"].get(h)) for h in
                                        ["1m", "3m", "6m", "12m", "to_date"]) + " |",
            "",
            "**The 15 picks:**", "",
            "| # | Ticker | Conviction | Entry | To-date | S&P (same window) | Beat? |",
            "|---|--------|-----------|-------|---------|-------------------|-------|",
        ]
        picks = sorted([p for p in bt["picks"] if p["returns"]["entry_date"]],
                       key=lambda p: p["conviction"], reverse=True)
        for i, p in enumerate(picks, 1):
            r = p["returns"]["to_date"]
            b = (p.get("benchmark_to_date") or {}).get("sp500")
            beat = "✅" if (r is not None and b is not None and r > b) else "❌"
            lines.append(f"| {i} | {p['ticker']} | {p['conviction']:.3f} | "
                         f"{p['returns']['entry_date']} | {pct(r)} | {pct(b)} | {beat} |")
        lines += ["", f"_Methodology: {bt['methodology']}_", ""]

    if TR.exists():
        tr = json.loads(TR.read_text())
        s = tr["summary"]
        lines += [
            "## Pillar 2 — real since-launch track record (Mar–Jun 2026)", "",
            f"- **Actionable buy signals (conviction ≥ {tr['threshold']}):** "
            f"{s['n_signals_rows']} rows → {s['n_distinct_tickers']} tickers "
            f"({s.get('data_hygiene_skipped', 0)} bad records dropped)",
            f"- **Live portfolio (to date):** **{pct(s['portfolio_to_date'])}**",
            f"- **S&P 500 / Nasdaq-100 (matched):** {pct(s['sp500_to_date_matched'])} / "
            f"{pct(s['nasdaq_to_date_matched'])}",
            f"- **Beating the S&P 500:** {s['beating_sp500']}",
            f"- _{s['note']}_", "",
        ]

    lines += [
        "## Charts", "",
        "![Equity curve 2023](charts/equity_curve_2023.png)", "",
        "![Picks vs S&P 500](charts/picks_2023.png)", "",
        "![Conviction vs return](charts/conviction_scatter.png)", "",
        "![2026 live record](charts/track_record_2026.png)", "",
        "---", "",
        "*Not investment advice. Backtested/hypothetical results have inherent "
        "limitations and do not represent actual trading. Past performance does "
        "not guarantee future results.*",
    ]

    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
