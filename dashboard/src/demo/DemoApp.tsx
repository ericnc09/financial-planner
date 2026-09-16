import { useEffect, useMemo, useRef, useState } from 'react';
import type { DemoEvent, DemoMonteCarlo, DemoSnapshot, DemoTicker } from './types';
import { downloadSnapshot, formatDate, loadSnapshot, money, percent } from './data';
import './demo.css';

type View = 'explore' | 'methodology' | 'case-study';
const REPO = 'https://github.com/ericnc09/financial-planner';
const modelLabels: Record<string, string> = { bull: 'Rising', bear: 'Falling', sideways: 'Sideways' };

function Icon({ name, size = 20 }: { name: 'explore' | 'methodology' | 'case-study' | 'arrow' | 'search' | 'download' | 'check'; size?: number }) {
  const paths = {
    explore: <><rect x="3" y="3" width="7" height="7" rx="1.5" /><rect x="14" y="3" width="7" height="7" rx="1.5" /><rect x="3" y="14" width="7" height="7" rx="1.5" /><rect x="14" y="14" width="7" height="7" rx="1.5" /></>,
    methodology: <><path d="M12 3 3 7v6c0 4 9 8 9 8s9-4 9-8V7l-9-4Z" /><path d="m8 12 3 3 5-6" /></>,
    'case-study': <><rect x="4" y="3" width="16" height="18" rx="2" /><path d="M8 8h8M8 12h8M8 16h5" /></>,
    arrow: <><path d="M6 18 18 6M7 6h11v11" /></>,
    search: <><circle cx="10.5" cy="10.5" r="6.5" /><path d="m16 16 5 5" /></>,
    download: <><path d="M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5" /></>,
    check: <><path d="m5 12 4 4L19 6" /></>,
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>;
}

function route() {
  const [rawView, ticker] = window.location.hash.replace(/^#\/?/, '').split('/');
  const view: View = rawView === 'methodology' || rawView === 'case-study' ? rawView : 'explore';
  return { view, ticker: ticker || '' };
}

export default function DemoApp() {
  const [snapshot, setSnapshot] = useState<DemoSnapshot | null>(null);
  const [error, setError] = useState('');
  const [location, setLocation] = useState(route);
  const [loading, setLoading] = useState(true);
  const mainRef = useRef<HTMLElement>(null);

  function load() {
    setLoading(true); setError('');
    loadSnapshot().then(setSnapshot).catch((e: Error) => setError(e.message)).finally(() => setLoading(false));
  }
  useEffect(load, []);
  useEffect(() => {
    const listener = () => setLocation(route());
    window.addEventListener('hashchange', listener);
    return () => window.removeEventListener('hashchange', listener);
  }, []);
  useEffect(() => {
    const name = location.view === 'explore' ? 'Disclosure research' : location.view === 'methodology' ? 'Data & methodology' : 'Product case study';
    document.title = `finAlyse — ${name}`;
  }, [location.view]);

  return <div className="demo-shell">
    <a className="skip-link" href="#main-content" onClick={e => { e.preventDefault(); mainRef.current?.focus(); }}>Skip to content</a>
    <aside className="sidebar">
      <a className="wordmark" href="#explore" aria-label="finAlyse home">finAlyse<span>.</span></a>
      <span className="sidebar-caption">EQUITY RESEARCH PROTOTYPE</span>
      <nav className="main-nav" aria-label="Main navigation">
        {([['explore', 'Explore disclosures'], ['methodology', 'Data & methodology'], ['case-study', 'Product case study']] as const).map(([view, label]) =>
          <a key={view} href={`#${view}`} aria-current={location.view === view ? 'page' : undefined}>
            <Icon name={view} /><span>{label}</span>
          </a>)}
      </nav>
      <div className="sidebar-bottom">
        <div className="sidebar-note"><span className="tiny-label">BUILT BY</span><strong>Eric Costa</strong><span>A product exploration in<br />financial research & trust.</span></div>
        <a href={REPO} target="_blank" rel="noreferrer">View the code <Icon name="arrow" size={15} /></a>
        <a href="https://ericcosta.ca" target="_blank" rel="noreferrer">Back to portfolio <Icon name="arrow" size={15} /></a>
      </div>
    </aside>
    <div className="workspace">
      <header className="topbar"><span>Research workspace <span className="topbar-slash">/</span> <strong>{location.view === 'explore' ? 'Explore' : location.view === 'methodology' ? 'Methodology' : 'Case study'}</strong></span><span className="prototype-label">Historical demo</span></header>
      <main id="main-content" ref={mainRef} tabIndex={-1}>
        {loading && <div className="loading-state" role="status"><span className="loading-line" /><h1>Opening the research workspace</h1><p>Loading a saved historical dataset.</p></div>}
        {error && <div className="error-state" role="alert"><h1>The dataset couldn’t be loaded.</h1><p>{error} Please try again.</p><button className="primary-button" onClick={load}>Try again</button></div>}
        {snapshot && !loading && !error && <>
          <div className="snapshot-banner"><span className="snapshot-symbol" aria-hidden="true">◷</span><p><strong>Historical snapshot · {formatDate(snapshot.metadata.as_of)}</strong><span>Saved research outputs. No live prices or trading recommendations.</span></p><a href="#methodology">About the data <span aria-hidden="true">↗</span></a></div>
          {location.view === 'explore' && <Explorer snapshot={snapshot} routeTicker={location.ticker} />}
          {location.view === 'methodology' && <Methodology snapshot={snapshot} />}
          {location.view === 'case-study' && <CaseStudy snapshot={snapshot} />}
          <footer className="workspace-footer"><span>finAlyse · Research & education only</span><span>Model outputs are estimates, not financial advice.</span></footer>
        </>}
      </main>
    </div>
  </div>;
}

function Explorer({ snapshot, routeTicker }: { snapshot: DemoSnapshot; routeTicker: string }) {
  const [query, setQuery] = useState('');
  const [sector, setSector] = useState('all');
  const [direction, setDirection] = useState('all');
  const [windowDays, setWindowDays] = useState('all');
  const sectors = [...new Set(snapshot.tickers.map(t => t.sector))].sort();
  const cutOff = windowDays === 'all' ? -Infinity : Date.parse(snapshot.metadata.as_of.slice(0, 10)) - Number(windowDays) * 86400000;
  const matches = (event: DemoEvent) => (direction === 'all' || event.direction === direction) && Date.parse(event.disclosure_date) >= cutOff;
  const visible = snapshot.tickers.filter(t => (sector === 'all' || t.sector === sector)
    && `${t.ticker} ${t.company_name}`.toLowerCase().includes(query.toLowerCase().trim()) && t.events.some(matches));
  const selected = visible.find(t => t.ticker === routeTicker) ?? visible[0];
  const shownEvents = selected?.events.filter(matches) ?? [];
  const count = visible.reduce((sum, ticker) => sum + ticker.events.filter(matches).length, 0);
  const hasFilter = query || sector !== 'all' || direction !== 'all' || windowDays !== 'all';
  function clear() { setQuery(''); setSector('all'); setDirection('all'); setWindowDays('all'); }

  return <>
    <section className="page-heading"><div><span className="eyebrow">FROM DISCLOSURE TO CONTEXT</span><h1>Explore the evidence.</h1><p>Start with a reported trade. Understand the context. See what’s missing.</p></div><a className="text-link" href="#case-study">Why I built this <Icon name="arrow" size={17} /></a></section>
    <ol className="journey" aria-label="Research journey"><li><span>01</span>Choose a company</li><li><span>02</span>Inspect its disclosures</li><li><span>03</span>Explore model context</li></ol>
    <section className="research-controls" aria-label="Filter research">
      <label className="search-field"><span className="sr-only">Search companies or tickers</span><Icon name="search" size={18} /><input type="search" placeholder="Search company or ticker" value={query} onChange={e => setQuery(e.target.value)} /></label>
      <label className="select-field"><span>Sector</span><select value={sector} onChange={e => setSector(e.target.value)}><option value="all">All sectors</option>{sectors.map(s => <option key={s}>{s}</option>)}</select></label>
      <label className="select-field"><span>Reported trade</span><select value={direction} onChange={e => setDirection(e.target.value)}><option value="all">All transactions</option><option value="buy">Purchases</option><option value="sell">Sales</option></select></label>
      <label className="select-field"><span>Disclosed before snapshot</span><select value={windowDays} onChange={e => setWindowDays(e.target.value)}><option value="all">All saved dates</option><option value="90">Last 90 days</option><option value="30">Last 30 days</option></select></label>
    </section>
    <div className="results-heading"><p aria-live="polite"><strong>{visible.length}</strong> companies <span>·</span> <strong>{count}</strong> disclosures</p>{hasFilter ? <button className="text-button" onClick={clear}>Clear filters</button> : <span>Selected for data coverage, not returns</span>}</div>
    {visible.length ? <div className="research-grid">
      <section className="company-list" aria-label="Companies">
        {visible.map(ticker => <button key={ticker.ticker} className={`company-option ${selected?.ticker === ticker.ticker ? 'selected' : ''}`} onClick={() => { window.location.hash = `explore/${ticker.ticker}`; }} aria-pressed={selected?.ticker === ticker.ticker}>
          <span className="company-option-top"><span className="ticker-symbol">{ticker.ticker}</span><span className="selection-arrow" aria-hidden="true">↗</span></span>
          <strong>{ticker.company_name}</strong><span className="company-sector">{ticker.sector}</span>
          <span className="company-option-bottom"><span>{ticker.events.filter(matches).length} disclosure{ticker.events.filter(matches).length !== 1 ? 's' : ''}</span><span>{ticker.availability.filter(m => m.available).length}/{ticker.availability.length} models available</span></span>
        </button>)}
        <div className="list-note"><Icon name="methodology" size={19} /><p>A reported purchase or sale describes a disclosure. It is not a recommendation.</p></div>
      </section>
      {selected && <TickerWorkspace key={`${selected.ticker}-${direction}-${windowDays}`} ticker={selected} events={shownEvents} />}
    </div> : <section className="empty-state"><Icon name="search" size={32} /><h2>No disclosures match these filters.</h2><p>This demo contains {snapshot.tickers.length} selected companies. Date filters are measured from {formatDate(snapshot.metadata.as_of)}, not today.</p><button className="primary-button" onClick={clear}>Show all companies</button></section>}
  </>;
}

function TickerWorkspace({ ticker, events }: { ticker: DemoTicker; events: DemoEvent[] }) {
  const [eventId, setEventId] = useState(events[0]?.id);
  const [tab, setTab] = useState<'disclosures' | 'models'>('disclosures');
  const event = events.find(e => e.id === eventId) ?? events[0];
  const available = ticker.availability.filter(model => model.available).length;
  return <article className="ticker-workspace" aria-label={`${ticker.ticker} research`}>
    <div className="ticker-header"><div><span className="eyebrow">COMPANY RESEARCH</span><h2>{ticker.company_name}</h2><div className="ticker-meta"><strong>{ticker.ticker}</strong><span>{ticker.sector}</span></div></div><span className="coverage-pill">{available}/{ticker.availability.length} models available</span></div>
    <div className="research-tabs" role="tablist" aria-label="Company research views" onKeyDown={e => {
      if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) {
        e.preventDefault(); const next = e.key === 'Home' ? 'disclosures' : e.key === 'End' ? 'models' : tab === 'disclosures' ? 'models' : 'disclosures';
        setTab(next); document.getElementById(`tab-${next}`)?.focus();
      }
    }}><button id="tab-disclosures" role="tab" aria-selected={tab === 'disclosures'} aria-controls="disclosures-panel" tabIndex={tab === 'disclosures' ? 0 : -1} onClick={() => setTab('disclosures')}>Disclosures <span>{events.length}</span></button><button id="tab-models" role="tab" aria-selected={tab === 'models'} aria-controls="models-panel" tabIndex={tab === 'models' ? 0 : -1} onClick={() => setTab('models')}>Model context</button></div>
    {tab === 'disclosures' && <div id="disclosures-panel" role="tabpanel" aria-labelledby="tab-disclosures" className="disclosures-panel">
      <div className="panel-intro"><h3>What was reported?</h3><p>Congressional disclosures aggregated by Capitol Trades. Dates reflect the saved source record.</p></div>
      <div className="event-list" aria-label="Select disclosure">{events.map(item => <button key={item.id} className={`event-option ${item.id === event?.id ? 'selected' : ''}`} onClick={() => setEventId(item.id)} aria-pressed={item.id === event?.id}>
        <span className={`trade-badge ${item.direction}`}>{item.direction === 'buy' ? 'Purchase' : 'Sale'}</span><span className="event-person"><strong>{item.actor}</strong><span>Traded {formatDate(item.trade_date, true)}</span></span><span className="event-date"><span>Disclosed</span><strong>{formatDate(item.disclosure_date, true)}</strong></span>
      </button>)}</div>
      {event && <div className="disclosure-detail">
        <div className="detail-heading"><h3>The disclosure timeline</h3><span className="neutral-badge">{event.disclosure_lag_days} day reporting gap</span></div>
        <div className="timeline"><div><span className="timeline-node" /><span className="tiny-label">TRADE DATE</span><strong>{formatDate(event.trade_date, true)}</strong><p>Transaction reported by the source.</p></div><div><span className="timeline-node" /><span className="tiny-label">DISCLOSURE DATE</span><strong>{formatDate(event.disclosure_date, true)}</strong><p>Reported disclosure date; exact public availability may differ.</p></div></div>
        <div className="insight-note"><strong>Why the gap matters</strong><p>The trade had already happened when it was disclosed. A historical trade date is not a price or entry point a researcher could have acted on.</p></div>
        <dl className="event-facts"><div><dt>Reported owner</dt><dd>{event.owner || 'Not recorded'}</dd></div><div><dt>Transaction amount</dt><dd>Not verified in this snapshot</dd></div><div><dt>Collected</dt><dd>{formatDate(event.collected_at, true)}</dd></div><div><dt>Original filing link</dt><dd>{event.source_url ? <a href={event.source_url} target="_blank" rel="noreferrer">View original filing ↗</a> : 'Not retained by the collector'}</dd></div></dl>
        <div className="source-row"><div><strong>{event.source_label}</strong><p>{event.source_note}</p></div><a className="secondary-button" href={event.source_portal_url} target="_blank" rel="noreferrer">Open source portal <Icon name="arrow" size={16} /></a></div>
      </div>}
      <button className="context-cta" onClick={() => { setTab('models'); document.getElementById('tab-models')?.focus(); }}><span><strong>Put this disclosure in context</strong><span>Explore saved volatility, regime, and simulation outputs.</span></span><span aria-hidden="true">→</span></button>
    </div>}
    {tab === 'models' && <div id="models-panel" role="tabpanel" aria-labelledby="tab-models" className="models-panel">
      <div className="panel-intro"><h3>Context, with uncertainty.</h3><p>These model runs may postdate the disclosures. They are retrospective research, not predictions known at the time of the trade.</p></div>
      <div className="model-stat-grid"><div><span className="tiny-label">ESTIMATED VOLATILITY</span><strong>{percent(ticker.analysis.garch?.current_vol_annual)}</strong><p>Annualized GARCH estimate</p><span className="model-date">Run {formatDate(ticker.analysis.garch?.run_date, true)}</span></div><div><span className="tiny-label">ESTIMATED REGIME</span><strong>{ticker.analysis.hmm ? modelLabels[ticker.analysis.hmm.current_state] || ticker.analysis.hmm.current_state : 'Unavailable'}</strong><p>Historical price-pattern model</p><span className="model-date">Run {formatDate(ticker.analysis.hmm?.run_date, true)}</span></div><div><span className="tiny-label">MARKET SENSITIVITY</span><strong>{ticker.analysis.factors?.beta_market == null ? 'Unavailable' : `${ticker.analysis.factors.beta_market.toFixed(2)}×`}</strong><p>Estimated market-factor beta</p><span className="model-date">Run {formatDate(ticker.analysis.factors?.run_date, true)}</span></div></div>
      {ticker.analysis.monte_carlo ? <SimulationChart model={ticker.analysis.monte_carlo} ticker={ticker.ticker} /> : <div className="inline-empty">No saved simulation is available for this company.</div>}
      <details className="explanation"><summary>How should I read these models?</summary><div><p><strong>Volatility</strong> estimates the scale of price variation, not its direction. The annualized estimate is not an expected return.</p><p><strong>Regime</strong> summarizes patterns in past prices. “Rising” describes the model’s fitted state, not a promise that prices will rise.</p><p><strong>Market beta</strong> estimates sensitivity to the market factor. A value above 1 suggests greater modeled sensitivity, not greater quality.</p><p><strong>Simulations</strong> depend on assumptions fitted to past prices. Their percentiles do not establish a calibrated real-world chance of profit.</p></div></details>
      <div className="availability-section"><div className="detail-heading"><h3>What’s available?</h3><span className="tiny-label">SAVED MODEL OUTPUTS</span></div>{ticker.availability.map(model => <div key={model.key} className="availability-row"><span className={`availability-icon ${model.available ? 'yes' : 'no'}`}>{model.available ? <Icon name="check" size={14} /> : '—'}</span><div><strong>{model.label}</strong><span>{model.available ? `Run ${formatDate(model.as_of, true)}` : 'Unavailable in this snapshot'}</span></div><span>{model.available ? 'Available' : 'Missing'}</span></div>)}<p className="field-note">Availability means a saved result exists. It does not measure predictive accuracy.</p></div>
      <div className="inline-empty"><strong>Historical price chart unavailable</strong><p>{ticker.price_history.note}</p></div>
    </div>}
  </article>;
}

function SimulationChart({ model, ticker }: { model: DemoMonteCarlo; ticker: string }) {
  const [days, setDays] = useState<30 | 90>(30);
  const horizon = model.horizons.find(h => h.days === days) ?? model.horizons[0];
  const gradientId = `band-${ticker}`;
  const bounds = useMemo(() => {
    const values = [model.current_price, ...model.horizons.flatMap(h => [h.p10, h.p90])];
    const low = Math.min(...values); const high = Math.max(...values); const padding = Math.max((high - low) * .16, high * .02, 1);
    return { low: Math.max(0, low - padding), high: high + padding };
  }, [model]);
  if (!horizon) return <div className="inline-empty">Simulation percentiles are unavailable.</div>;
  const y = (value: number) => 188 - (value - bounds.low) / (bounds.high - bounds.low) * 162;
  const ticks = Array.from({ length: 5 }, (_, i) => bounds.low + (bounds.high - bounds.low) * i / 4);
  return <section className="simulation-card" aria-label="Saved price simulation">
    <div className="simulation-heading"><div><span className="eyebrow">SIMULATED · NOT OBSERVED PRICES</span><h3>A range of possible outcomes</h3></div><div className="segmented-control" aria-label="Simulation horizon">{([30, 90] as const).map(d => <button key={d} aria-pressed={days === d} onClick={() => setDays(d)}>{d} days</button>)}</div></div>
    <p className="chart-description">{model.n_simulations.toLocaleString()} simulated paths · model run {formatDate(model.run_date, true)}</p>
    <svg className="simulation-chart" viewBox="0 0 580 225" role="img" aria-label={`${ticker}, ${days} day simulated prices: tenth percentile ${money(horizon.p10)}, median ${money(horizon.p50)}, ninetieth percentile ${money(horizon.p90)}. Baseline ${money(model.current_price)}.`}>
      <defs><linearGradient id={gradientId} x1="0" x2="1"><stop stopColor="#deece9" /><stop offset="1" stopColor="#8cbeb2" /></linearGradient></defs>
      {ticks.map((v, i) => <g key={i}><line x1="63" y1={y(v)} x2="527" y2={y(v)} stroke="#e4eae8" /><text x="52" y={y(v) + 4} textAnchor="end" fill="#617674" fontSize="11">${v.toFixed(0)}</text></g>)}
      <path d={`M95 ${y(model.current_price)} L436 ${y(horizon.p10)} L436 ${y(horizon.p90)} Z`} fill={`url(#${gradientId})`} opacity=".48" />
      <path d={`M95 ${y(model.current_price)} L436 ${y(horizon.p25)} L436 ${y(horizon.p75)} Z`} fill="#72ab9e" opacity=".36" />
      <line x1="63" x2="527" y1={y(model.current_price)} y2={y(model.current_price)} stroke="#8faaa4" strokeDasharray="4 5" />
      <path d={`M95 ${y(model.current_price)} L436 ${y(horizon.p50)}`} stroke="#246b5c" strokeWidth="2.5" strokeDasharray="6 4" />
      <line x1="436" x2="436" y1={y(horizon.p10)} y2={y(horizon.p90)} stroke="#246b5c" strokeWidth="1.5" />
      {[horizon.p10, horizon.p50, horizon.p90].map((v, i) => <g key={i}><circle cx="436" cy={y(v)} r={i === 1 ? 4 : 3} fill="#246b5c" /><text x="450" y={y(v) + 4} fill="#294e45" fontSize="11">{['P10', 'P50', 'P90'][i]}</text></g>)}
      <circle cx="95" cy={y(model.current_price)} r="4" fill="#246b5c" />
      <text x="95" y="216" textAnchor="middle" fontSize="12" fill="#617674">Saved baseline</text><text x="436" y="216" textAnchor="middle" fontSize="12" fill="#617674">{days}-day horizon</text>
    </svg>
    <div className="chart-legend"><span><i className="legend-wide" />10th–90th percentile</span><span><i className="legend-narrow" />25th–75th percentile</span><span><i className="legend-median" />Median</span></div>
    <dl className="simulation-values"><div><dt>Saved baseline</dt><dd>{money(model.current_price)}</dd></div><div><dt>10th percentile</dt><dd>{money(horizon.p10)}</dd></div><div><dt>Median</dt><dd>{money(horizon.p50)}</dd></div><div><dt>90th percentile</dt><dd>{money(horizon.p90)}</dd></div></dl>
    <p className="field-note">Only the saved baseline and horizon percentiles are available. Connecting bands are illustrative; they do not show a simulated daily path. {model.note}</p>
  </section>;
}

function Methodology({ snapshot }: { snapshot: DemoSnapshot }) {
  return <>
    <section className="page-heading"><div><span className="eyebrow">TRANSPARENCY BY DESIGN</span><h1>Know what you’re looking at.</h1><p>The data, the assumptions, and the limits of this research prototype.</p></div><button className="secondary-button" onClick={() => downloadSnapshot(snapshot)}><Icon name="download" size={17} />Download dataset</button></section>
    <div className="method-summary"><div><span className="tiny-label">CURATED UNIVERSE</span><strong>{snapshot.metadata.source_counts.tickers} companies</strong><p>{snapshot.metadata.source_counts.events} saved congressional disclosures</p></div><div><span className="tiny-label">HISTORICAL CUTOFF</span><strong>{formatDate(snapshot.metadata.as_of, true)}</strong><p>Individual records have their own dates</p></div><div><span className="tiny-label">RESEARCH STATUS</span><strong>Experimental</strong><p>Model accuracy is not established</p></div></div>
    <div className="method-grid"><section className="content-card"><span className="section-number">01 / DATA</span><h2>A selected historical dataset</h2><p>{snapshot.metadata.selection_note}</p><p>Company identity comes from matching source records. Saved model outputs are joined by ticker. We exclude enrichment fields whose event association could not be trusted.</p><div className="data-flow"><span>Public disclosures</span><span aria-hidden="true">↓</span><span>Saved collector records</span><span aria-hidden="true">↓</span><span>Reviewed field selection</span><span aria-hidden="true">↓</span><span>This versioned snapshot</span></div><p>Original filing links and verified transaction amounts were not retained for these records. Source buttons open the aggregator portal, not an original filing.</p></section>
      <section className="content-card"><span className="section-number">02 / EVALUATION</span><h2>What the model audit tells us</h2><div className="audit-score"><strong>{snapshot.validation.passed}<span>/{snapshot.validation.total}</span></strong><div>research checks passed<small>Audit dated {formatDate(snapshot.validation.generated_at, true)}</small></div></div><p>These are checks from a saved research report. They are not a certification of predictive performance.</p><ul className="gate-list">{snapshot.validation.gates.map(gate => <li key={gate.key}><span className={gate.passed ? 'gate-pass' : 'gate-fail'}>{gate.passed ? 'Passed' : 'Not passed'}</span><span>{gate.label}</span></li>)}</ul><p className="field-note">XGBoost: {snapshot.validation.xgboost_status.replace(/_/g, ' ')}.</p></section>
    </div>
    <section className="content-card limitations-card"><span className="section-number">03 / LIMITS</span><h2>Where the evidence stops</h2><div className="limit-grid"><div><h3>Information arrives late</h3><p>A transaction date is earlier than its disclosure. Saved model runs may be later still. These outputs cannot be presented as decisions available at the original trade date.</p></div><div><h3>Availability is not accuracy</h3><p>A saved model result demonstrates the research workflow. It does not establish reliable stock selection, a calibrated probability, or future returns.</p></div><div><h3>Missing stays missing</h3><p>We do not invent price history, source documents, or unavailable models. GEHC provides an example with incomplete model coverage.</p></div><div><h3>Returns need further validation</h3><p>Historical evaluation has unresolved timing and data-quality issues. Return leaderboards, backtest Sharpe ratios, and buy recommendations are excluded from this demo.</p></div></div><details className="explanation"><summary>Read the dataset and evaluation caveats</summary><ul>{[...new Set([...snapshot.metadata.limitations, ...snapshot.validation.caveats])].map(note => <li key={note}>{note}</li>)}</ul></details></section>
    {snapshot.macro && <section className="content-card"><span className="section-number">04 / MARKET CONTEXT</span><h2>A separate macro snapshot</h2><p>{snapshot.macro.note}</p><div className="macro-grid">{snapshot.macro.indicators.map(indicator => <div key={indicator.key}><span className="tiny-label">{indicator.label}</span><strong>{indicator.value == null ? 'Unavailable' : `${indicator.value.toLocaleString('en-US', { maximumFractionDigits: 2 })}${indicator.unit === '%' ? '%' : indicator.unit ? ` ${indicator.unit}` : ''}`}</strong><span>Saved {formatDate(indicator.as_of, true)}</span><a href={indicator.source_url} target="_blank" rel="noreferrer">Source series ↗</a><p>{indicator.note}</p></div>)}</div></section>}
    <div className="snapshot-details"><span>Snapshot {snapshot.metadata.snapshot_id.slice(0, 16)}</span><span>Schema {snapshot.metadata.schema_version}</span><a href={REPO} target="_blank" rel="noreferrer">Inspect the implementation ↗</a></div>
  </>;
}

function CaseStudy({ snapshot }: { snapshot: DemoSnapshot }) {
  return <>
    <section className="page-heading case-heading"><div><span className="eyebrow">A PRODUCT EXPLORATION BY ERIC COSTA</span><h1>Making research easier to question.</h1><p>How a broad stock-analysis project became a focused exploration of evidence, uncertainty, and trust.</p></div></section>
    <div className="case-intro"><div><span className="tiny-label">PRODUCT</span><strong>finAlyse</strong><p>Equity-research prototype</p></div><div><span className="tiny-label">CURRENT RELEASE</span><strong>Interactive historical demo</strong><p>{snapshot.metadata.source_counts.tickers} companies · dated source records</p></div><div><span className="tiny-label">NEXT VALIDATION</span><strong>Task-based user research</strong><p>Usability and demand remain hypotheses</p></div></div>
    <section className="case-section"><span className="section-number">01 / THE PROBLEM HYPOTHESIS</span><h2>A trade headline leaves a lot unsaid.</h2><p>A reported purchase can look like a clear signal. But who reported it, when did it become public, and what does the available evidence actually support?</p><p>The working audience is self-directed investors researching public disclosures. The hypothesis is that a single, explainable workflow can help them inspect evidence and uncertainty with less effort. This audience and need still require direct user validation.</p></section>
    <section className="case-section"><span className="section-number">02 / THE MVP</span><h2>One complete research task.</h2><p>Choose a company, inspect a disclosure, then explore model context and missing information. The demo makes that journey available without accounts, API keys, or waiting for a model to run.</p><a className="primary-button" href="#explore">Try the research workflow <span aria-hidden="true">→</span></a></section>
    <section className="case-section"><span className="section-number">03 / PRODUCT DECISIONS</span><h2>Three choices that define this release.</h2><div className="decision-grid"><article><span>01</span><h3>Evidence before ranking</h3><p>Research checks exposed weaknesses in conviction ranking and historical evaluation. The demo centers on disclosures and context; unsupported performance claims are withheld.</p><strong>Tradeoff</strong><p>Less immediate certainty, with clearer boundaries around what the product can support.</p></article><article><span>02</span><h3>A dependable first session</h3><p>A versioned snapshot keeps exploration independent of live providers. Every view carries the historical date, and the interface retains original record timestamps.</p><strong>Tradeoff</strong><p>Freshness is limited in exchange for a reproducible, fast research experience.</p></article><article><span>03</span><h3>Show the gaps</h3><p>Missing model results, original filing links, and price histories are explicit. Model coverage describes available data, not confidence in a recommendation.</p><strong>Tradeoff</strong><p>A narrower feature set makes limitations visible and actionable.</p></article></div></section>
    <section className="case-section"><span className="section-number">04 / WHAT EXISTS TODAY</span><h2>Working software. Open research questions.</h2><div className="evidence-grid"><div><h3>Implemented</h3><ul><li>Disclosure ingestion and saved research outputs</li><li>Interactive, dated research explorer</li><li>Model availability and validation reporting</li><li>Reproducible snapshot export and automated checks</li></ul></div><div><h3>Not established</h3><ul><li>Demand from the proposed user segment</li><li>Improvement in research-task completion</li><li>Reliable out-of-sample investment performance</li><li>Readiness for personalized financial decisions</li></ul></div></div></section>
    <section className="case-section"><span className="section-number">05 / THE NEXT EXPERIMENT</span><h2>Can people explain what they found?</h2><p>Proposed study: 5–8 participants matching the working audience complete the same disclosure-research task with their current tools and this prototype. Ask them to explain the source, reporting delay, and one reason the model output could be misleading.</p><div className="measurement-grid"><div><strong>Task completion</strong><p>Can they identify a disclosure and its supporting evidence?</p></div><div><strong>Time to evidence</strong><p>How long does it take to find and explain useful context?</p></div><div><strong>Understanding</strong><p>Do they distinguish observed facts from simulated estimates?</p></div></div><p className="field-note">This study is planned. No participant outcomes, adoption metrics, or time savings are claimed.</p></section>
    <div className="case-bottom"><div><h2>Explore the work.</h2><p>The implementation and research audit are open for inspection.</p></div><a className="secondary-button" href={REPO} target="_blank" rel="noreferrer">GitHub <Icon name="arrow" size={17} /></a><a className="primary-button" href="#methodology">Read the methodology</a></div>
  </>;
}
