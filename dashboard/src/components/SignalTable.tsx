import React, { useState } from 'react';
import type { Signal } from '../types';
import { ConvictionBar } from './ConvictionBar';

interface Props {
  signals: Signal[];
}

type SortKey = 'conviction' | 'trade_date' | 'ticker';

export const SignalTable: React.FC<Props> = ({ signals }) => {
  const [sortKey, setSortKey] = useState<SortKey>('conviction');
  const [sourceFilter, setSourceFilter] = useState<string>('all');
  const [expandedId, setExpandedId] = useState<number | null>(null);

  const filtered = signals.filter(s => sourceFilter === 'all' || s.source_type === sourceFilter);
  const sorted = [...filtered].sort((a, b) => {
    if (sortKey === 'conviction') return (b.conviction ?? 0) - (a.conviction ?? 0);
    if (sortKey === 'trade_date') return new Date(b.trade_date).getTime() - new Date(a.trade_date).getTime();
    return a.ticker.localeCompare(b.ticker);
  });

  const dirColor = (d: string) => d === 'buy' ? '#22c55e' : '#ef4444';
  const fmt = (n: number | null) => n !== null ? `$${n.toLocaleString(undefined, { maximumFractionDigits: 0 })}` : '-';

  return (
    <div style={{ background: '#161b22', borderRadius: 12, padding: 20, border: '1px solid #30363d' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <h3 style={{ fontSize: 14, color: '#c9d1d9' }}>Smart Money Signals</h3>
        <div style={{ display: 'flex', gap: 8 }}>
          {['all', 'congressional', 'insider'].map(f => (
            <button key={f} onClick={() => setSourceFilter(f)} style={{
              padding: '4px 10px', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 11,
              background: sourceFilter === f ? '#6366f1' : '#21262d', color: sourceFilter === f ? '#fff' : '#8b949e',
            }}>
              {f.charAt(0).toUpperCase() + f.slice(1)}
            </button>
          ))}
        </div>
      </div>

      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
        <thead>
          <tr style={{ borderBottom: '1px solid #30363d', color: '#8b949e', fontSize: 11, textAlign: 'left' }}>
            {[
              { key: 'ticker' as SortKey, label: 'Ticker' },
              { key: 'conviction' as SortKey, label: 'Conviction' },
            ].map(col => (
              <th key={col.key} onClick={() => setSortKey(col.key)} style={{ padding: '8px 12px', cursor: 'pointer' }}>
                {col.label} {sortKey === col.key ? '▼' : ''}
              </th>
            ))}
            <th style={{ padding: '8px 12px' }}>Direction</th>
            <th style={{ padding: '8px 12px' }}>Actor</th>
            <th style={{ padding: '8px 12px' }}>Size</th>
            <th style={{ padding: '8px 12px', cursor: 'pointer' }} onClick={() => setSortKey('trade_date')}>
              Date {sortKey === 'trade_date' ? '▼' : ''}
            </th>
            <th style={{ padding: '8px 12px' }}>Source</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map(s => (
            <React.Fragment key={s.id}>
              <tr onClick={() => setExpandedId(expandedId === s.id ? null : s.id)} style={{
                borderBottom: '1px solid #21262d', cursor: 'pointer',
                background: expandedId === s.id ? '#1c2129' : 'transparent',
              }}>
                <td style={{ padding: '10px 12px', fontWeight: 600 }}>{s.ticker}</td>
                <td style={{ padding: '10px 12px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <div style={{
                      width: 60, height: 6, borderRadius: 3, background: '#21262d', overflow: 'hidden',
                    }}>
                      <div style={{
                        width: `${(s.conviction ?? 0) * 100}%`, height: '100%', borderRadius: 3,
                        background: (s.conviction ?? 0) >= 0.6 ? '#22c55e' : (s.conviction ?? 0) >= 0.4 ? '#f59e0b' : '#ef4444',
                      }} />
                    </div>
                    <span style={{ fontSize: 12, color: '#c9d1d9' }}>{((s.conviction ?? 0) * 100).toFixed(0)}%</span>
                  </div>
                </td>
                <td style={{ padding: '10px 12px' }}>
                  <span style={{ color: dirColor(s.direction), fontWeight: 600, textTransform: 'uppercase', fontSize: 11 }}>
                    {s.direction}
                  </span>
                </td>
                <td style={{ padding: '10px 12px', color: '#8b949e', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {s.actor}
                </td>
                <td style={{ padding: '10px 12px', color: '#8b949e' }}>{fmt(s.size_estimate)}</td>
                <td style={{ padding: '10px 12px', color: '#8b949e' }}>{new Date(s.trade_date).toLocaleDateString()}</td>
                <td style={{ padding: '10px 12px' }}>
                  <span style={{
                    padding: '2px 8px', borderRadius: 10, fontSize: 10, fontWeight: 500,
                    background: s.source_type === 'congressional' ? '#6366f122' : '#f59e0b22',
                    color: s.source_type === 'congressional' ? '#818cf8' : '#fbbf24',
                  }}>
                    {s.source_type}
                  </span>
                </td>
              </tr>
              {expandedId === s.id && s.signal_score !== null && (
                <tr>
                  <td colSpan={7} style={{ padding: '12px 24px', background: '#1c2129' }}>
                    <ConvictionBar
                      signalScore={s.signal_score ?? 0}
                      fundamentalScore={s.fundamental_score ?? 0}
                      macroModifier={s.macro_modifier ?? 1}
                      conviction={s.conviction ?? 0}
                    />
                    <div style={{ marginTop: 8, fontSize: 11, color: '#8b949e', display: 'flex', gap: 16 }}>
                      {s.sector && <span>Sector: {s.sector}</span>}
                      {s.price_at_signal && <span>Price: ${s.price_at_signal.toFixed(2)}</span>}
                      <span>Passes: {s.passes_threshold ? 'Yes' : 'No'}</span>
                    </div>
                  </td>
                </tr>
              )}
            </React.Fragment>
          ))}
        </tbody>
      </table>
      {sorted.length === 0 && (
        <div style={{ textAlign: 'center', padding: 40, color: '#8b949e' }}>No signals found</div>
      )}
    </div>
  );
};
