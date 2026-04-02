import React, { useEffect, useState } from 'react';
import { api } from '../api/client';

interface PerformanceSummary {
  total_signals: number;
  tracked_signals: number;
  win_rate_5d: number | null;
  win_rate_20d: number | null;
  avg_return_5d: number | null;
  avg_return_20d: number | null;
  avg_return_60d: number | null;
  by_direction: Record<string, any> | null;
  by_source: Record<string, any> | null;
  by_conviction_bucket: Record<string, any> | null;
  top_winners: { ticker: string; direction: string; return_20d: number; conviction: number | null }[];
  top_losers: { ticker: string; direction: string; return_20d: number; conviction: number | null }[];
}

const pct = (v: number | null | undefined) => v != null ? `${(v * 100).toFixed(1)}%` : '-';
const retColor = (v: number | null | undefined) => {
  if (v == null) return '#8b949e';
  return v >= 0 ? '#22c55e' : '#ef4444';
};

export const PerformancePanel: React.FC = () => {
  const [data, setData] = useState<PerformanceSummary | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/performance/summary')
      .then(r => r.json())
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return null;
  if (!data || data.tracked_signals === 0) return null;

  const dirs = data.by_direction || {};
  const sources = data.by_source || {};
  const buckets = data.by_conviction_bucket || {};

  return (
    <div style={{
      background: '#0d1117', border: '1px solid #30363d', borderRadius: 12,
      padding: 20, marginBottom: 20,
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <div>
          <div style={{ fontSize: 15, fontWeight: 700, color: '#e1e4e8' }}>Signal Performance</div>
          <div style={{ fontSize: 11, color: '#8b949e' }}>
            {data.tracked_signals} of {data.total_signals} signals tracked
          </div>
        </div>
        <button onClick={() => fetch('/api/performance/update', { method: 'POST' })} style={{
          padding: '4px 12px', borderRadius: 6, border: '1px solid #30363d',
          background: '#21262d', color: '#c9d1d9', cursor: 'pointer', fontSize: 11,
        }}>Update</button>
      </div>

      {/* Overall stats */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 16, flexWrap: 'wrap' }}>
        {[
          { label: 'Win Rate 5d', value: data.win_rate_5d, fmt: pct },
          { label: 'Win Rate 20d', value: data.win_rate_20d, fmt: pct },
          { label: 'Avg Return 5d', value: data.avg_return_5d, fmt: pct },
          { label: 'Avg Return 20d', value: data.avg_return_20d, fmt: pct },
          { label: 'Avg Return 60d', value: data.avg_return_60d, fmt: pct },
        ].map(({ label, value, fmt }) => (
          <div key={label} style={{
            background: '#161b22', borderRadius: 8, padding: '10px 14px',
            border: '1px solid #30363d', flex: '1 1 100px', minWidth: 100,
          }}>
            <div style={{ fontSize: 10, color: '#8b949e', marginBottom: 2 }}>{label}</div>
            <div style={{ fontSize: 18, fontWeight: 700, color: retColor(value) }}>
              {fmt(value)}
            </div>
          </div>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
        {/* By direction */}
        <div style={{ background: '#161b22', borderRadius: 8, padding: 12, border: '1px solid #30363d' }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: '#c9d1d9', marginBottom: 8 }}>By Direction</div>
          {Object.entries(dirs).map(([dir, stats]) => (
            <div key={dir} style={{ marginBottom: 6, fontSize: 11 }}>
              <span style={{ color: dir === 'buy' ? '#22c55e' : '#ef4444', fontWeight: 600, textTransform: 'uppercase' }}>
                {dir}
              </span>
              <span style={{ color: '#8b949e' }}> ({stats.n}) — </span>
              <span style={{ color: retColor(stats.win_rate_20d) }}>
                {pct(stats.win_rate_20d)} win
              </span>
              <span style={{ color: '#8b949e' }}>, </span>
              <span style={{ color: retColor(stats.avg_return_20d) }}>
                {pct(stats.avg_return_20d)} avg
              </span>
            </div>
          ))}
        </div>

        {/* By source */}
        <div style={{ background: '#161b22', borderRadius: 8, padding: 12, border: '1px solid #30363d' }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: '#c9d1d9', marginBottom: 8 }}>By Source</div>
          {Object.entries(sources).map(([src, stats]) => (
            <div key={src} style={{ marginBottom: 6, fontSize: 11 }}>
              <span style={{ color: src === 'congressional' ? '#818cf8' : '#fbbf24', fontWeight: 600 }}>
                {src}
              </span>
              <span style={{ color: '#8b949e' }}> ({stats.n}) — </span>
              <span style={{ color: retColor(stats.win_rate_20d) }}>
                {pct(stats.win_rate_20d)} win
              </span>
              <span style={{ color: '#8b949e' }}>, </span>
              <span style={{ color: retColor(stats.avg_return_20d) }}>
                {pct(stats.avg_return_20d)} avg
              </span>
            </div>
          ))}
        </div>

        {/* By conviction bucket */}
        <div style={{ background: '#161b22', borderRadius: 8, padding: 12, border: '1px solid #30363d' }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: '#c9d1d9', marginBottom: 8 }}>By Conviction</div>
          {Object.entries(buckets).map(([bucket, stats]) => (
            <div key={bucket} style={{ marginBottom: 6, fontSize: 11 }}>
              <span style={{ color: bucket === 'high' ? '#22c55e' : bucket === 'medium' ? '#f59e0b' : '#8b949e', fontWeight: 600 }}>
                {bucket}
              </span>
              <span style={{ color: '#8b949e' }}> ({stats.n}) — </span>
              <span style={{ color: retColor(stats.win_rate_20d) }}>
                {pct(stats.win_rate_20d)} win
              </span>
              <span style={{ color: '#8b949e' }}>, </span>
              <span style={{ color: retColor(stats.avg_return_20d) }}>
                {pct(stats.avg_return_20d)} avg
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Top winners/losers */}
      {(data.top_winners.length > 0 || data.top_losers.length > 0) && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginTop: 12 }}>
          <div style={{ background: '#161b22', borderRadius: 8, padding: 12, border: '1px solid #30363d' }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: '#22c55e', marginBottom: 6 }}>Top Winners (20d)</div>
            {data.top_winners.map((w, i) => (
              <div key={i} style={{ fontSize: 11, display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                <span style={{ color: '#c9d1d9' }}>{w.ticker} <span style={{ color: '#8b949e' }}>{w.direction}</span></span>
                <span style={{ color: '#22c55e', fontWeight: 600 }}>+{(w.return_20d * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
          <div style={{ background: '#161b22', borderRadius: 8, padding: 12, border: '1px solid #30363d' }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: '#ef4444', marginBottom: 6 }}>Top Losers (20d)</div>
            {data.top_losers.map((w, i) => (
              <div key={i} style={{ fontSize: 11, display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                <span style={{ color: '#c9d1d9' }}>{w.ticker} <span style={{ color: '#8b949e' }}>{w.direction}</span></span>
                <span style={{ color: '#ef4444', fontWeight: 600 }}>{(w.return_20d * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
