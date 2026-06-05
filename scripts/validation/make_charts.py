"""
Render the validation charts (PNG) from the results JSON files.

  1. equity_curve_2023.png  - growth of matched capital: basket vs SP500 vs Nasdaq
  2. picks_2023.png         - per-pick to-date return vs SP500 over same window
  3. conviction_scatter.png - does the conviction score predict realized return?
  4. track_record_2026.png  - the real, short, since-launch record

These are designed to drop straight onto a marketing page later, but for now
they are standalone artifacts for review.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import date

CHARTS = Path("reports/charts")
CHARTS.mkdir(parents=True, exist_ok=True)

BG = "#0f1117"
FG = "#e6e6e6"
GRID = "#2a2f3a"
ACCENT = "#28c76f"      # smart money green
SP = "#5b8def"          # s&p blue
NQ = "#f6c343"          # nasdaq amber
LOSS = "#ea5455"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
    "axes.edgecolor": GRID, "grid.color": GRID, "font.size": 11,
    "axes.titlesize": 14, "axes.titleweight": "bold",
})


def _pct(x):
    return f"{x*100:+.1f}%"


def equity_curve(bt: dict):
    c = bt.get("equity_curve") or {}
    if not c:
        return
    dates = [date.fromisoformat(d) for d in c["dates"]]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(dates, c["portfolio"], color=ACCENT, lw=2.4, label="Smart Money basket (15 picks)")
    ax.plot(dates, c["sp500"], color=SP, lw=1.8, label="S&P 500 (matched cash)")
    ax.plot(dates, c["nasdaq"], color=NQ, lw=1.8, label="Nasdaq-100 (matched cash)")
    ax.axhline(100, color=GRID, lw=1, ls="--")
    s = bt["summary"]
    ax.set_title(f"Growth of matched capital — 2023 insider-buy backtest\n"
                 f"basket {_pct(s['portfolio_to_date'])}   "
                 f"S&P {_pct(s['sp500_to_date_matched'])}   "
                 f"Nasdaq {_pct(s['nasdaq_to_date_matched'])}", loc="left")
    ax.set_ylabel("Value (start = 100)")
    ax.legend(loc="upper left", framealpha=0.1)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(CHARTS / "equity_curve_2023.png", dpi=150)
    plt.close(fig)


def picks_bar(bt: dict):
    picks = [p for p in bt["picks"] if p["returns"]["entry_date"] and p["returns"]["to_date"] is not None]
    picks.sort(key=lambda p: p["returns"]["to_date"])
    tickers = [p["ticker"] for p in picks]
    rets = [p["returns"]["to_date"] for p in picks]
    spy = [(p["benchmark_to_date"] or {}).get("sp500") for p in picks]
    colors = [ACCENT if (s is not None and r > s) else LOSS for r, s in zip(rets, spy)]
    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(tickers))
    ax.barh(y, [r * 100 for r in rets], color=colors, alpha=0.9)
    for i, s in enumerate(spy):
        if s is not None:
            ax.plot([s * 100], [i], marker="D", color=SP, ms=7,
                    label="S&P 500 (same window)" if i == 0 else None)
    ax.set_yticks(y)
    ax.set_yticklabels(tickers)
    ax.axvline(0, color=FG, lw=1)
    ax.set_xlabel("Total return since entry (%)")
    ax.set_title("Each 2023 pick vs the S&P 500 over the identical holding window\n"
                 "green = beat the market   red = lagged   ◆ = S&P over same window", loc="left")
    ax.legend(loc="lower right", framealpha=0.1)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS / "picks_2023.png", dpi=150)
    plt.close(fig)


def conviction_scatter(bt: dict):
    pts = [(p["conviction"], p["returns"]["to_date"]) for p in bt["picks"]
           if p["returns"]["to_date"] is not None]
    if len(pts) < 3:
        return
    x = np.array([a for a, _ in pts])
    yv = np.array([b * 100 for _, b in pts])
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x, yv, color=ACCENT, s=70, alpha=0.85, edgecolor=BG)
    if len(set(x)) > 1:
        m, b = np.polyfit(x, yv, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, m * xs + b, color=SP, lw=2, ls="--",
                label=f"trend: {m:.0f}% per conviction point")
        r = np.corrcoef(x, yv)[0, 1]
        ax.set_title(f"Does the conviction score predict realized return?  (corr={r:+.2f})", loc="left")
        ax.legend(loc="best", framealpha=0.1)
    else:
        ax.set_title("Conviction vs realized return", loc="left")
    ax.axhline(0, color=GRID, lw=1, ls="--")
    ax.set_xlabel("Conviction score at signal")
    ax.set_ylabel("Total return since entry (%)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS / "conviction_scatter.png", dpi=150)
    plt.close(fig)


def track_2026(tr: dict):
    c = tr.get("equity_curve") or {}
    if not c:
        return
    dates = [date.fromisoformat(d) for d in c["dates"]]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(dates, c["portfolio"], color=ACCENT, lw=2.4, label="Live signals (actioned buys)")
    ax.plot(dates, c["sp500"], color=SP, lw=1.8, label="S&P 500")
    ax.plot(dates, c["nasdaq"], color=NQ, lw=1.8, label="Nasdaq-100")
    ax.axhline(100, color=GRID, lw=1, ls="--")
    s = tr["summary"]
    ax.set_title(f"Real since-launch record (Mar–Jun 2026, window still open)\n"
                 f"live {_pct(s['portfolio_to_date'])}   "
                 f"S&P {_pct(s['sp500_to_date_matched'])}   "
                 f"Nasdaq {_pct(s['nasdaq_to_date_matched'])}", loc="left")
    ax.set_ylabel("Value (start = 100)")
    ax.legend(loc="best", framealpha=0.1)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(CHARTS / "track_record_2026.png", dpi=150)
    plt.close(fig)


def main():
    bt_path = Path("reports/backtest_2023_results.json")
    tr_path = Path("reports/track_record_2026_results.json")
    if bt_path.exists():
        bt = json.loads(bt_path.read_text())
        equity_curve(bt)
        picks_bar(bt)
        conviction_scatter(bt)
        print("wrote 2023 charts")
    if tr_path.exists():
        tr = json.loads(tr_path.read_text())
        track_2026(tr)
        print("wrote 2026 chart")
    print(f"charts -> {CHARTS}")


if __name__ == "__main__":
    main()
