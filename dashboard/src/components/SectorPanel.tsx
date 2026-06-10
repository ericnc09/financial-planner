import React, { useEffect, useState } from 'react';

interface SectorSummary {
  sector: string;
  n_signals: number;
  n_buys: number;
  n_sells: number;
  n_tickers: number;
  tickers: string[];
  avg_conviction: number | null;
  avg_ensemble_score: number | null;
  top_ticker: string | null;
  top_ticker_conviction: number | null;
  latest_signal_date: string | null;
}

interface Props {
  selectedSector: string | null;
  onSelectSector: (sector: string | null) => void;
  onTickerClick?: (ticker: string) => void;
}

export const SectorPanel: React.FC<Props> = ({ selectedSector, onSelectSector, onTickerClick }) => {
  const [sectors, setSectors] = useState<SectorSummary[]>([]);

  useEffect(() => {
    fetch('/api/sectors?days=30')
      .then(r => (r.ok ? r.json() : []))
      .then(setSectors)
      .catch(() => setSectors([]));
  }, []);

  if (sectors.length === 0) return null;

  const selected = sectors.find(s => s.sector === selectedSector);

  return (
    <div style={{ background: '#161b22', borderRadius: 12, padding: 20, border: '1px solid #30363d', marginBottom: 20 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <h3 style={{ fontSize: 14, color: '#c9d1d9', margin: 0 }}>Industries — pick one to see what happened there</h3>
        {selectedSector && (
          <button onClick={() => onSelectSector(null)} style={{
            padding: '4px 10px', borderRadius: 6, border: '1px solid #30363d', cursor: 'pointer',
            fontSize: 11, background: '#21262d', color: '#8b949e',
          }}>
            Clear filter ✕
          </button>
        )}
      </div>

      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        {sectors.map(s => {
          const active = s.sector === selectedSector;
          const buyRatio = s.n_signals > 0 ? s.n_buys / s.n_signals : 0;
          return (
            <button
              key={s.sector}
              onClick={() => onSelectSector(active ? null : s.sector)}
              style={{
                padding: '8px 14px', borderRadius: 8, cursor: 'pointer', textAlign: 'left',
                border: active ? '1px solid #6366f1' : '1px solid #30363d',
                background: active ? '#6366f122' : '#21262d',
              }}
            >
              <div style={{ fontSize: 12, fontWeight: 600, color: active ? '#a5b4fc' : '#c9d1d9' }}>
                {s.sector}
              </div>
              <div style={{ fontSize: 10, color: '#8b949e', marginTop: 2 }}>
                {s.n_signals} signals · {s.n_tickers} tickers ·{' '}
                <span style={{ color: buyRatio >= 0.5 ? '#22c55e' : '#ef4444' }}>
                  {s.n_buys}B/{s.n_sells}S
                </span>
              </div>
            </button>
          );
        })}
      </div>

      {selected && (
        <div style={{ marginTop: 14, padding: 14, background: '#1c2129', borderRadius: 8, fontSize: 12, color: '#c9d1d9' }}>
          <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', marginBottom: 8 }}>
            <span>Avg conviction: <b>{selected.avg_conviction !== null ? `${(selected.avg_conviction * 100).toFixed(0)}%` : '—'}</b></span>
            <span>Avg ensemble: <b>{selected.avg_ensemble_score !== null ? `${selected.avg_ensemble_score.toFixed(0)}/100` : '—'}</b></span>
            <span>
              Top ticker:{' '}
              {selected.top_ticker ? (
                <b style={{ color: '#818cf8', cursor: 'pointer', textDecoration: 'underline' }}
                   onClick={() => onTickerClick?.(selected.top_ticker!)}>
                  {selected.top_ticker}
                </b>
              ) : '—'}
              {selected.top_ticker_conviction !== null && ` (${(selected.top_ticker_conviction * 100).toFixed(0)}%)`}
            </span>
            <span>Latest signal: <b>{selected.latest_signal_date ? new Date(selected.latest_signal_date).toLocaleDateString() : '—'}</b></span>
          </div>
          <div style={{ color: '#8b949e' }}>
            Tickers: {selected.tickers.map((t, i) => (
              <React.Fragment key={t}>
                {i > 0 && ', '}
                <span style={{ color: '#818cf8', cursor: 'pointer' }} onClick={() => onTickerClick?.(t)}>{t}</span>
              </React.Fragment>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
