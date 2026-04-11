import React, { useState } from 'react';
import type { BacktestResult, PeriodMetrics } from '../types';
import { api } from '../api/client';

const HOLD_PERIODS = ['7', '14', '30', '60', '90'];

const fmt = (v: number, pct = false) => {
  if (pct) return `${(v * 100).toFixed(1)}%`;
  return v.toFixed(3);
};

const deltaColor = (delta: number) => (delta > 0 ? '#22c55e' : delta < 0 ? '#ef4444' : '#8b949e');

export const BacktestPanel: React.FC = () => {
  const today = new Date().toISOString().slice(0, 10);
  const oneYearAgo = new Date(Date.now() - 365 * 86400000).toISOString().slice(0, 10);

  const [startDate, setStartDate] = useState(oneYearAgo);
  const [endDate, setEndDate] = useState(today);
  const [threshold, setThreshold] = useState(0.6);
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runBacktestWithDates = async (start: string, end: string, thresh: number) => {
    setStartDate(start);
    setEndDate(end);
    setThreshold(thresh);
    setLoading(true);
    setError(null);
    try {
      const data = await api.runBacktest(start, end, thresh);
      setResult(data);
    } catch (e: any) {
      setError(e.message || 'Backtest failed');
    } finally {
      setLoading(false);
    }
  };

  const runBacktest = () => runBacktestWithDates(startDate, endDate, threshold);

  const renderEquityCurve = (res: BacktestResult) => {
    const periods = HOLD_PERIODS.filter(p => res.filtered_metrics[p] || res.unfiltered_metrics[p]);
    if (periods.length === 0) return null;

    const filteredReturns = periods.map(p => res.filtered_metrics[p]?.avg_return || 0);
    const unfilteredReturns = periods.map(p => res.unfiltered_metrics[p]?.avg_return || 0);
    const allVals = [...filteredReturns, ...unfilteredReturns];
    const minY = Math.min(0, ...allVals);
    const maxY = Math.max(0.01, ...allVals);
    const range = maxY - minY || 0.01;

    const w = 600, h = 180, pad = 40;
    const pw = w - pad * 2, ph = h - pad * 2;

    const toX = (i: number) => pad + (i / (periods.length - 1 || 1)) * pw;
    const toY = (v: number) => pad + ph - ((v - minY) / range) * ph;

    const fLine = filteredReturns.map((v, i) => `${toX(i)},${toY(v)}`).join(' ');
    const uLine = unfilteredReturns.map((v, i) => `${toX(i)},${toY(v)}`).join(' ');
    const zeroY = toY(0);

    return (
      <svg width="100%" viewBox={`0 0 ${w} ${h}`} style={{ marginTop: 12 }}>
        {/* Zero line */}
        <line x1={pad} y1={zeroY} x2={w - pad} y2={zeroY} stroke="#30363d" strokeDasharray="4" />
        {/* Grid labels */}
        {periods.map((p, i) => (
          <text key={p} x={toX(i)} y={h - 8} textAnchor="middle" fill="#8b949e" fontSize={10}>{p}d</text>
        ))}
        {/* Unfiltered line */}
        <polyline points={uLine} fill="none" stroke="#8b949e" strokeWidth={2} strokeOpacity={0.6} />
        {unfilteredReturns.map((v, i) => (
          <circle key={`u${i}`} cx={toX(i)} cy={toY(v)} r={3} fill="#8b949e" />
        ))}
        {/* Filtered line */}
        <polyline points={fLine} fill="none" stroke="#22c55e" strokeWidth={2.5} />
        {filteredReturns.map((v, i) => (
          <circle key={`f${i}`} cx={toX(i)} cy={toY(v)} r={4} fill="#22c55e" />
        ))}
        {/* Legend */}
        <circle cx={pad + 10} cy={12} r={4} fill="#22c55e" />
        <text x={pad + 20} y={16} fill="#c9d1d9" fontSize={10}>Filtered</text>
        <circle cx={pad + 90} cy={12} r={3} fill="#8b949e" />
        <text x={pad + 100} y={16} fill="#8b949e" fontSize={10}>Unfiltered</text>
      </svg>
    );
  };

  const renderMetricsRow = (label: string, f: PeriodMetrics | undefined, u: PeriodMetrics | undefined) => {
    if (!f && !u) return null;
    const fm = f || {} as PeriodMetrics;
    const um = u || {} as PeriodMetrics;

    const metrics: { key: keyof PeriodMetrics; label: string; pct: boolean }[] = [
      { key: 'total_trades', label: 'Trades', pct: false },
      { key: 'win_rate', label: 'Win Rate', pct: true },
      { key: 'avg_return', label: 'Avg Return', pct: true },
      { key: 'sharpe_ratio', label: 'Sharpe', pct: false },
      { key: 'sortino_ratio', label: 'Sortino', pct: false },
      { key: 'profit_factor', label: 'Profit Factor', pct: false },
      { key: 'max_drawdown', label: 'Max DD', pct: true },
    ];

    return (
      <tr key={label}>
        <td style={{ padding: '8px 12px', borderBottom: '1px solid #21262d', fontWeight: 600, color: '#c9d1d9' }}>{label}</td>
        {metrics.map(m => {
          const fv = (fm[m.key] as number) || 0;
          const uv = (um[m.key] as number) || 0;
          const delta = fv - uv;
          return (
            <td key={m.key} style={{ padding: '8px 6px', borderBottom: '1px solid #21262d', textAlign: 'right', fontSize: 12 }}>
              <div style={{ color: '#c9d1d9' }}>{fmt(fv, m.pct)}</div>
              <div style={{ color: '#8b949e', fontSize: 10 }}>{fmt(uv, m.pct)}</div>
              {m.key !== 'total_trades' && (
                <div style={{ color: deltaColor(m.key === 'max_drawdown' ? -delta : delta), fontSize: 10 }}>
                  {delta >= 0 ? '+' : ''}{fmt(delta, m.pct)}
                </div>
              )}
            </td>
          );
        })}
      </tr>
    );
  };

  return (
    <div style={{ background: '#0d1117', borderRadius: 12, border: '1px solid #30363d', padding: 20, marginBottom: 20 }}>
      <h3 style={{ color: '#e1e4e8', marginBottom: 16, fontSize: 16 }}>Backtest Engine</h3>

      {/* Controls */}
      <div style={{ display: 'flex', gap: 16, alignItems: 'flex-end', flexWrap: 'wrap', marginBottom: 16 }}>
        <div>
          <label style={{ fontSize: 11, color: '#8b949e', display: 'block', marginBottom: 4 }}>Start Date</label>
          <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)}
            style={{ padding: '6px 10px', borderRadius: 6, border: '1px solid #30363d', background: '#161b22', color: '#c9d1d9', fontSize: 13 }} />
        </div>
        <div>
          <label style={{ fontSize: 11, color: '#8b949e', display: 'block', marginBottom: 4 }}>End Date</label>
          <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)}
            style={{ padding: '6px 10px', borderRadius: 6, border: '1px solid #30363d', background: '#161b22', color: '#c9d1d9', fontSize: 13 }} />
        </div>
        <div>
          <label style={{ fontSize: 11, color: '#8b949e', display: 'block', marginBottom: 4 }}>
            Conviction Threshold: {threshold.toFixed(2)}
          </label>
          <input type="range" min={0} max={1} step={0.05} value={threshold} onChange={e => setThreshold(Number(e.target.value))}
            style={{ width: 160, accentColor: '#6366f1' }} />
        </div>
        <button onClick={runBacktest} disabled={loading} style={{
          padding: '8px 20px', borderRadius: 8, border: 'none', background: loading ? '#30363d' : '#6366f1',
          color: '#fff', cursor: loading ? 'not-allowed' : 'pointer', fontSize: 13, fontWeight: 600,
        }}>
          {loading ? 'Running...' : 'Run Backtest'}
        </button>
        <button onClick={() => {
          const now = new Date();
          const thirtyAgo = new Date(now.getTime() - 30 * 86400000);
          runBacktestWithDates(thirtyAgo.toISOString().slice(0, 10), now.toISOString().slice(0, 10), threshold);
        }} disabled={loading} style={{
          padding: '8px 20px', borderRadius: 8, border: '1px solid #30363d',
          background: loading ? '#30363d' : '#21262d', color: '#c9d1d9',
          cursor: loading ? 'not-allowed' : 'pointer', fontSize: 13,
        }}>
          Last 30 Days
        </button>
      </div>

      {error && <div style={{ color: '#ef4444', fontSize: 13, marginBottom: 12 }}>{error}</div>}

      {result && (
        <>
          {/* Summary */}
          <div style={{ display: 'flex', gap: 12, marginBottom: 16, flexWrap: 'wrap' }}>
            {[
              { label: 'Total Signals', value: String(result.total_signals), color: '#c9d1d9' },
              { label: 'Filtered Signals', value: String(result.filtered_signals), color: '#22c55e' },
              { label: 'Threshold', value: result.conviction_threshold.toFixed(2), color: '#6366f1' },
              { label: 'Date Range', value: `${result.date_range[0]} to ${result.date_range[1]}`, color: '#8b949e' },
            ].map(s => (
              <div key={s.label} style={{ background: '#161b22', borderRadius: 8, padding: '10px 16px', border: '1px solid #30363d' }}>
                <div style={{ fontSize: 10, color: '#8b949e' }}>{s.label}</div>
                <div style={{ fontSize: 16, fontWeight: 600, color: s.color }}>{s.value}</div>
              </div>
            ))}
          </div>

          {/* Equity Curve */}
          <div style={{ background: '#161b22', borderRadius: 8, padding: 12, border: '1px solid #30363d', marginBottom: 16 }}>
            <div style={{ fontSize: 12, color: '#8b949e', marginBottom: 4 }}>Average Return by Hold Period</div>
            {renderEquityCurve(result)}
          </div>

          {/* Metrics Table */}
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #30363d' }}>
                  <th style={{ textAlign: 'left', padding: '8px 12px', color: '#8b949e', fontSize: 11 }}>Period</th>
                  {['Trades', 'Win Rate', 'Avg Return', 'Sharpe', 'Sortino', 'Profit Factor', 'Max DD'].map(h => (
                    <th key={h} style={{ textAlign: 'right', padding: '8px 6px', color: '#8b949e', fontSize: 11 }}>
                      {h}
                      <div style={{ fontSize: 9, fontWeight: 400 }}>filtered / unfiltered / delta</div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {HOLD_PERIODS.map(p =>
                  renderMetricsRow(`${p}d`, result.filtered_metrics[p], result.unfiltered_metrics[p])
                )}
              </tbody>
            </table>
          </div>
        </>
      )}

      {!result && !loading && (
        <div style={{ color: '#8b949e', fontSize: 13, textAlign: 'center', padding: 20 }}>
          Configure parameters and click "Run Backtest" to validate signal performance historically.
        </div>
      )}
    </div>
  );
};
