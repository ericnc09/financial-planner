"""
Generate a single self-contained demo page for ericcosta.ca.

Framing is rigour-first, not returns-first: the differentiator is radical
transparency and bias control, and the real (currently market-lagging) results
are shown in full. Data is inlined so the output is one static file you can
drop on any host. Re-run after each backtest to refresh the numbers.

  python -m scripts.validation.build_site   ->   reports/site/index.html
"""

import json
from pathlib import Path

BT = Path("reports/backtest_2023_results.json")
TR = Path("reports/track_record_2026_results.json")
OUT = Path("reports/site/index.html")


def pct(x, d=1):
    return "—" if x is None else f"{x*100:+.{d}f}%"


def downsample(xs, target=160):
    if len(xs) <= target:
        return xs
    step = max(1, len(xs) // target)
    return xs[::step]


def build():
    bt = json.loads(BT.read_text()) if BT.exists() else {}
    tr = json.loads(TR.read_text()) if TR.exists() else {}
    bs = bt.get("summary", {})
    ts = tr.get("summary", {})

    # ---- chart data (downsampled) ----
    ec = bt.get("equity_curve", {})
    step = max(1, len(ec.get("dates", [1])) // 160)
    chart = {
        "labels": downsample(ec.get("dates", [])),
        "portfolio": downsample(ec.get("portfolio", [])),
        "sp500": downsample(ec.get("sp500", [])),
        "nasdaq": downsample(ec.get("nasdaq", [])),
    }
    tec = tr.get("equity_curve", {})
    tchart = {
        "labels": downsample(tec.get("dates", [])),
        "portfolio": downsample(tec.get("portfolio", [])),
        "sp500": downsample(tec.get("sp500", [])),
        "nasdaq": downsample(tec.get("nasdaq", [])),
    }

    # ---- picks table ----
    picks = [p for p in bt.get("picks", []) if p["returns"]["entry_date"]]
    picks.sort(key=lambda p: p["conviction"], reverse=True)
    rows = ""
    for i, p in enumerate(picks, 1):
        r = p["returns"]["to_date"]
        b = (p.get("benchmark_to_date") or {}).get("sp500")
        beat = (r is not None and b is not None and r > b)
        rows += (f"<tr><td>{i}</td><td class=tk>{p['ticker']}</td>"
                 f"<td>{p['conviction']:.3f}</td><td>{p['returns']['entry_date']}</td>"
                 f"<td class='{'pos' if (r or 0)>=0 else 'neg'}'>{pct(r)}</td>"
                 f"<td class=dim>{pct(b)}</td>"
                 f"<td>{'✅' if beat else '—'}</td></tr>\n")

    data_js = json.dumps({"bt": chart, "tr": tchart})

    html = TEMPLATE
    repl = {
        "__DATA__": data_js,
        "__BT_PORT__": pct(bs.get("portfolio_to_date")),
        "__BT_SP__": pct(bs.get("sp500_to_date_matched")),
        "__BT_NQ__": pct(bs.get("nasdaq_to_date_matched")),
        "__BT_BEAT__": str(bs.get("picks_beating_sp500", "—")),
        "__BT_UNIV__": str(bt.get("universe", "S&P SmallCap 600")),
        "__BT_NRAW__": str(bt.get("n_raw_events", "—")),
        "__BT_UNFILT__": pct(bs.get("unfiltered_universe_avg_to_date")),
        "__TR_PORT__": pct(ts.get("portfolio_to_date")),
        "__TR_SP__": pct(ts.get("sp500_to_date_matched")),
        "__TR_NQ__": pct(ts.get("nasdaq_to_date_matched")),
        "__TR_N__": str(ts.get("n_distinct_tickers", "—")),
        "__TR_BEAT__": str(ts.get("beating_sp500", "—")),
        "__ROWS__": rows,
    }
    for k, v in repl.items():
        html = html.replace(k, v)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(f"wrote {OUT}  ({len(html):,} bytes)")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Smart Money Follows — radically transparent signal validation</title>
<meta name="description" content="A smart-money trading signal system validated with a bias-controlled, point-in-time backtest. We publish the real results — even when they lag the market.">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root{
    --bg:#0f1117; --panel:#161a23; --line:#2a2f3a; --fg:#e6e6e6; --dim:#9aa3b2;
    --green:#28c76f; --blue:#5b8def; --amber:#f6c343; --red:#ea5455;
  }
  *{box-sizing:border-box} html{scroll-behavior:smooth}
  body{margin:0;background:var(--bg);color:var(--fg);
    font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
  .wrap{max-width:980px;margin:0 auto;padding:0 20px}
  a{color:var(--blue)}
  h1,h2,h3{line-height:1.2;font-weight:800;letter-spacing:-.02em}
  h2{font-size:28px;margin-top:0}
  section{padding:64px 0;border-top:1px solid var(--line)}
  .pill{display:inline-block;font-size:12px;font-weight:700;letter-spacing:.08em;
    text-transform:uppercase;color:var(--green);background:rgba(40,199,111,.1);
    padding:6px 12px;border-radius:999px;border:1px solid rgba(40,199,111,.3)}
  /* hero */
  .hero{padding:90px 0 64px;text-align:center}
  .hero h1{font-size:48px;margin:18px 0 10px;background:linear-gradient(90deg,#fff,#9aa3b2);
    -webkit-background-clip:text;background-clip:text;color:transparent}
  .hero p.sub{font-size:20px;color:var(--dim);max-width:680px;margin:0 auto 28px}
  .cta{display:inline-block;background:var(--green);color:#06210f;font-weight:800;
    padding:14px 26px;border-radius:10px;text-decoration:none}
  .cta.alt{background:transparent;color:var(--fg);border:1px solid var(--line);margin-left:10px}
  /* stat strip */
  .stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:40px}
  .stat{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px}
  .stat .n{font-size:26px;font-weight:800}
  .stat .l{font-size:12px;color:var(--dim);text-transform:uppercase;letter-spacing:.05em}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:20px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:22px}
  .card h3{margin:0 0 8px;font-size:18px}
  .card p{margin:0;color:var(--dim);font-size:15px}
  .chartbox{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px;margin-top:18px}
  table{width:100%;border-collapse:collapse;font-size:14px;margin-top:14px}
  th,td{padding:9px 10px;text-align:left;border-bottom:1px solid var(--line)}
  th{color:var(--dim);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
  td.tk{font-weight:700} td.pos{color:var(--green)} td.neg{color:var(--red)} td.dim{color:var(--dim)}
  .callout{background:rgba(91,141,239,.08);border:1px solid rgba(91,141,239,.3);
    border-radius:12px;padding:18px 22px;margin-top:20px}
  .callout.warn{background:rgba(246,195,67,.08);border-color:rgba(246,195,67,.3)}
  form{display:flex;gap:10px;max-width:520px;margin:22px auto 0;flex-wrap:wrap;justify-content:center}
  input[type=email]{flex:1;min-width:240px;background:#0c0e14;border:1px solid var(--line);
    color:var(--fg);padding:14px;border-radius:10px;font-size:16px}
  .muted{color:var(--dim);font-size:13px}
  footer{padding:40px 0 70px;color:var(--dim);font-size:13px;border-top:1px solid var(--line)}
  @media(max-width:760px){.stats{grid-template-columns:1fr 1fr}.grid2{grid-template-columns:1fr}.hero h1{font-size:34px}}
</style>
</head>
<body>
<div class="wrap">

  <div class="hero">
    <span class="pill">Smart Money Follows</span>
    <h1>We don't sell hope. We publish proof.</h1>
    <p class="sub">A signal system that tracks insider &amp; smart-money buying — validated with a
      bias-controlled, point-in-time backtest. We show you the real results, even when they
      lag the market. <b>Especially</b> when they lag the market.</p>
    <a class="cta" href="#signup">Follow the experiment</a>
    <a class="cta alt" href="#method">See the method</a>
    <div class="stats">
      <div class="stat"><div class="n">__BT_NRAW__</div><div class="l">real 2023 buys analyzed</div></div>
      <div class="stat"><div class="n">100%</div><div class="l">point-in-time, no look-ahead</div></div>
      <div class="stat"><div class="n">0</div><div class="l">cherry-picked tickers</div></div>
      <div class="stat"><div class="n">live</div><div class="l">track record, updated</div></div>
    </div>
  </div>

  <section id="why">
    <span class="pill">Why this is different</span>
    <h2 style="margin-top:14px">Most "stock pick" sites show you the winners. We show you everything.</h2>
    <div class="grid2" style="margin-top:24px">
      <div class="card"><h3>No hindsight</h3><p>Every decision is frozen as of the day the trade became
        public. Entry is the next trading day — never the trade date. Prices, momentum and macro use
        only data that existed at the time.</p></div>
      <div class="card"><h3>No cherry-picking</h3><p>We score <i>every</i> open-market insider buy in a
        fixed, recognized universe (S&amp;P SmallCap 600) for the whole year, then let the model rank.
        We never hand-pick tickers after the fact.</p></div>
      <div class="card"><h3>No survivorship games</h3><p>Delisted names are kept and exited at their last
        traded price. Transaction costs are deducted. Returns are measured against the S&amp;P 500 and
        Nasdaq-100 over the identical window.</p></div>
      <div class="card"><h3>No spin</h3><p>The current model <b>underperformed</b> a historic mega-cap
        bull market. We're telling you that on the homepage, with the chart, because that's what honest
        validation looks like.</p></div>
    </div>
  </section>

  <section id="results">
    <span class="pill">The 2023 backtest — unedited</span>
    <h2 style="margin-top:14px">Growth of matched capital, 2023 → today</h2>
    <p class="muted">Top 15 conviction picks, equal weight, vs. the same dollars in the S&amp;P 500 / Nasdaq-100
      over the identical windows. Universe: __BT_UNIV__.</p>
    <div class="chartbox"><canvas id="ec"></canvas></div>
    <div class="stats" style="margin-top:18px">
      <div class="stat"><div class="n">__BT_PORT__</div><div class="l">our 15 picks</div></div>
      <div class="stat"><div class="n">__BT_SP__</div><div class="l">S&amp;P 500</div></div>
      <div class="stat"><div class="n">__BT_NQ__</div><div class="l">Nasdaq-100</div></div>
      <div class="stat"><div class="n">__BT_BEAT__</div><div class="l">picks that beat the S&amp;P</div></div>
    </div>
    <div class="callout warn"><b>What this tells us.</b> Our conviction picks lagged the benchmarks.
      Interestingly, a naive equal-weight basket of <i>all</i> insider buys returned __BT_UNFILT__ —
      the edge in this period was breadth, not concentration. The model's job is to beat that, and right
      now it doesn't. That's the gap we're closing in public.</div>

    <h3 style="margin-top:40px">The 15 picks, every one</h3>
    <table>
      <thead><tr><th>#</th><th>Ticker</th><th>Conviction</th><th>Entry</th>
        <th>Return to date</th><th>S&amp;P (same window)</th><th>Beat?</th></tr></thead>
      <tbody>__ROWS__</tbody>
    </table>
  </section>

  <section id="live">
    <span class="pill">Live track record</span>
    <h2 style="margin-top:14px">Since launch — real signals, updated</h2>
    <p class="muted">The actual buy signals the running system flagged (conviction ≥ 0.60) since March 2026.
      Short window, still open — shown raw.</p>
    <div class="chartbox"><canvas id="tec"></canvas></div>
    <div class="stats" style="margin-top:18px">
      <div class="stat"><div class="n">__TR_PORT__</div><div class="l">live signals</div></div>
      <div class="stat"><div class="n">__TR_SP__</div><div class="l">S&amp;P 500</div></div>
      <div class="stat"><div class="n">__TR_NQ__</div><div class="l">Nasdaq-100</div></div>
      <div class="stat"><div class="n">__TR_BEAT__</div><div class="l">beating the S&amp;P</div></div>
    </div>
  </section>

  <section id="method">
    <span class="pill">The method</span>
    <h2 style="margin-top:14px">How the validation works</h2>
    <p class="muted">Full write-up in <code>METHODOLOGY.md</code>. The short version:</p>
    <div class="grid2" style="margin-top:18px">
      <div class="card"><h3>Signal</h3><p>SEC Form 4 open-market purchases (code P only — cash bets, not
        grants). Scored on actor seniority, size, cluster buying, disclosure speed and consensus.</p></div>
      <div class="card"><h3>Context</h3><p>Price momentum, RSI, drawdown and liquidity computed as of the
        disclosure date; macro regime reconstructed from FRED as of that date.</p></div>
      <div class="card"><h3>Execution</h3><p>Next-day entry, liquidity-tier transaction costs, delistings
        exited at last price. No leverage, no shorting in the backtest.</p></div>
      <div class="card"><h3>Honesty about limits</h3><p>One bull-market regime; insider-only (congressional
        data was inaccessible); valuation/short-interest held neutral to avoid look-ahead. Small sample.</p></div>
    </div>
  </section>

  <section id="signup" style="text-align:center">
    <span class="pill">Follow the experiment</span>
    <h2 style="margin-top:14px">Watch us try to beat the market — in public</h2>
    <p class="sub" style="margin:10px auto 0">No performance promises. You'll get the live signals, every
      backtest we run, and an honest log of what's working and what isn't.</p>
    <form action="REPLACE_WITH_YOUR_FORM_ENDPOINT" method="POST"
          onsubmit="if(this.action.indexOf('REPLACE_WITH')>-1){window.location.href='mailto:niloyericcosta@gmail.com?subject=Smart%20Money%20signup&body='+encodeURIComponent(this.email.value);return false;}">
      <input type="email" name="email" placeholder="you@email.com" required>
      <button class="cta" type="submit">Keep me posted</button>
    </form>
    <p class="muted" style="margin-top:12px">Connect the form to your email provider (Formspree, Buttondown,
      etc.) by replacing the <code>action</code> URL. Until then it falls back to an email draft.</p>
  </section>

  <footer>
    <b>Not investment advice.</b> Backtested / hypothetical results have inherent limitations and do not
    represent actual trading; they may not reflect the impact of material economic and market factors.
    Past performance — simulated or real — does not guarantee future results. This page is a research
    demonstration. © Eric Costa.
  </footer>

</div>

<script>
const D = __DATA__;
const palette={green:'#28c76f',blue:'#5b8def',amber:'#f6c343',line:'#2a2f3a',dim:'#9aa3b2'};
function mk(id, d, labels){
  const el=document.getElementById(id); if(!el||!d.labels||!d.labels.length)return;
  new Chart(el,{type:'line',data:{labels:d.labels,datasets:[
    {label:labels[0],data:d.portfolio,borderColor:palette.green,borderWidth:2.4,pointRadius:0,tension:.1},
    {label:'S&P 500',data:d.sp500,borderColor:palette.blue,borderWidth:1.6,pointRadius:0,tension:.1},
    {label:'Nasdaq-100',data:d.nasdaq,borderColor:palette.amber,borderWidth:1.6,pointRadius:0,tension:.1},
  ]},options:{responsive:true,interaction:{mode:'index',intersect:false},
    plugins:{legend:{labels:{color:palette.dim}}},
    scales:{x:{ticks:{color:palette.dim,maxTicksLimit:8},grid:{color:palette.line}},
      y:{ticks:{color:palette.dim},grid:{color:palette.line},title:{display:true,text:'Value (start = 100)',color:palette.dim}}}}});
}
mk('ec', D.bt, ['Smart Money basket']);
mk('tec', D.tr, ['Live signals']);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    build()
