import React, { useState, useEffect, useRef } from 'react';
import type { PriceHistoryData } from '../types';
import { api } from '../api/client';

interface Props {
  tickers: string[];
}

const TIME_RANGES = [
  { label: '30D', days: 30 },
  { label: '90D', days: 90 },
  { label: '1Y', days: 365 },
  { label: '2Y', days: 730 },
];

const formatPrice = (v: number) => v >= 100 ? `$${v.toFixed(0)}` : `$${v.toFixed(2)}`;

const formatDate = (d: string, rangeDays: number) => {
  const date = new Date(d + 'T00:00:00');
  if (rangeDays <= 90) return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  return date.toLocaleDateString('en-US', { month: 'short', year: '2-digit' });
};

export const StockPriceChart: React.FC<Props> = ({ tickers }) => {
  const [selectedTicker, setSelectedTicker] = useState<string>(tickers[0] || '');
  const [rangeDays, setRangeDays] = useState(365);
  const [data, setData] = useState<PriceHistoryData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout>>();

  useEffect(() => {
    if (!selectedTicker) return;
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(async () => {
      setLoading(true);
      setError(null);
      try {
        const result = await api.getPriceHistory(selectedTicker, rangeDays);
        setData(result);
      } catch (e: any) {
        setError(e.message || 'Failed to fetch price data');
        setData(null);
      } finally {
        setLoading(false);
      }
    }, 300);
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current); };
  }, [selectedTicker, rangeDays]);

  // Update selected ticker if tickers list changes and current selection is gone
  useEffect(() => {
    if (tickers.length > 0 && !tickers.includes(selectedTicker)) {
      setSelectedTicker(tickers[0]);
    }
  }, [tickers]);

  const renderChart = () => {
    if (!data || data.closes.length < 2) return null;

    const { dates, closes } = data;
    const W = 700, H = 240;
    const pad = { top: 20, right: 20, bottom: 36, left: 60 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    const minPrice = Math.min(...closes) * 0.98;
    const maxPrice = Math.max(...closes) * 1.02;
    const priceRange = maxPrice - minPrice || 1;

    const xScale = (i: number) => pad.left + (i / (closes.length - 1)) * plotW;
    const yScale = (v: number) => pad.top + (1 - (v - minPrice) / priceRange) * plotH;

    // Line path
    const pathD = closes.map((c, i) =>
      `${i === 0 ? 'M' : 'L'}${xScale(i).toFixed(1)},${yScale(c).toFixed(1)}`
    ).join(' ');

    // Area fill path
    const areaD = `${pathD} L${xScale(closes.length - 1).toFixed(1)},${yScale(minPrice).toFixed(1)} L${xScale(0).toFixed(1)},${yScale(minPrice).toFixed(1)} Z`;

    const periodReturn = (closes[closes.length - 1] - closes[0]) / closes[0];
    const lineColor = periodReturn >= 0 ? '#22c55e' : '#ef4444';
    const fillColor = periodReturn >= 0 ? 'rgba(34,197,94,0.12)' : 'rgba(239,68,68,0.12)';

    // Y-axis labels (5 ticks)
    const yTicks = Array.from({ length: 5 }, (_, i) => minPrice + (priceRange * i) / 4);

    // X-axis labels (6 evenly spaced)
    const xLabelCount = 6;
    const xIndices = Array.from({ length: xLabelCount }, (_, i) =>
      Math.round((i / (xLabelCount - 1)) * (dates.length - 1))
    );

    // Grid lines
    const gridY = yTicks.map(v => yScale(v));

    return (
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ marginTop: 8 }}>
        {/* Horizontal grid */}
        {gridY.map((y, i) => (
          <line key={i} x1={pad.left} y1={y} x2={W - pad.right} y2={y} stroke="#21262d" strokeWidth={1} />
        ))}

        {/* Area fill */}
        <path d={areaD} fill={fillColor} />

        {/* Price line */}
        <path d={pathD} fill="none" stroke={lineColor} strokeWidth={2} />

        {/* Y-axis labels */}
        {yTicks.map((v, i) => (
          <text key={i} x={pad.left - 6} y={yScale(v) + 4} textAnchor="end" fill="#8b949e" fontSize={10}>
            {formatPrice(v)}
          </text>
        ))}

        {/* X-axis labels */}
        {xIndices.map((idx, i) => (
          <text key={i} x={xScale(idx)} y={H - 8} textAnchor="middle" fill="#8b949e" fontSize={10}>
            {formatDate(dates[idx], rangeDays)}
          </text>
        ))}
      </svg>
    );
  };

  const renderStats = () => {
    if (!data || data.closes.length < 2) return null;
    const { closes } = data;
    const current = closes[closes.length - 1];
    const first = closes[0];
    const change = current - first;
    const changePct = (change / first) * 100;
    const high = Math.max(...closes);
    const low = Math.min(...closes);
    const positive = change >= 0;

    const stats = [
      { label: 'Current', value: formatPrice(current), color: '#e1e4e8' },
      { label: 'Change', value: `${positive ? '+' : ''}${formatPrice(change)} (${positive ? '+' : ''}${changePct.toFixed(2)}%)`, color: positive ? '#22c55e' : '#ef4444' },
      { label: 'Period High', value: formatPrice(high), color: '#c9d1d9' },
      { label: 'Period Low', value: formatPrice(low), color: '#c9d1d9' },
    ];

    return (
      <div style={{ display: 'flex', gap: 12, marginTop: 12, flexWrap: 'wrap' }}>
        {stats.map(s => (
          <div key={s.label} style={{ background: '#161b22', borderRadius: 8, padding: '8px 14px', border: '1px solid #30363d' }}>
            <div style={{ fontSize: 10, color: '#8b949e' }}>{s.label}</div>
            <div style={{ fontSize: 14, fontWeight: 600, color: s.color }}>{s.value}</div>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div style={{ background: '#0d1117', borderRadius: 12, border: '1px solid #30363d', padding: 20, marginBottom: 20 }}>
      <h3 style={{ color: '#e1e4e8', marginBottom: 16, fontSize: 16 }}>Stock Price History</h3>

      {/* Controls */}
      <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap', marginBottom: 12 }}>
        <select
          value={selectedTicker}
          onChange={e => setSelectedTicker(e.target.value)}
          style={{
            padding: '6px 10px', borderRadius: 6, border: '1px solid #30363d',
            background: '#161b22', color: '#c9d1d9', fontSize: 13,
          }}
        >
          {tickers.map(t => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>

        <div style={{ display: 'flex', gap: 4 }}>
          {TIME_RANGES.map(r => (
            <button
              key={r.days}
              onClick={() => setRangeDays(r.days)}
              style={{
                padding: '4px 12px', borderRadius: 6, cursor: 'pointer', fontSize: 11, fontWeight: 600,
                border: rangeDays === r.days ? 'none' : '1px solid #30363d',
                background: rangeDays === r.days ? '#6366f1' : '#161b22',
                color: rangeDays === r.days ? '#fff' : '#8b949e',
              }}
            >
              {r.label}
            </button>
          ))}
        </div>
      </div>

      {/* Chart area */}
      {loading && (
        <div style={{ color: '#8b949e', fontSize: 13, textAlign: 'center', padding: 40 }}>Loading price data...</div>
      )}
      {error && (
        <div style={{ color: '#ef4444', fontSize: 13, textAlign: 'center', padding: 40 }}>{error}</div>
      )}
      {!loading && !error && data && (
        <div style={{ background: '#161b22', borderRadius: 8, padding: 12, border: '1px solid #30363d' }}>
          {renderChart()}
        </div>
      )}
      {!loading && !error && !data && (
        <div style={{ color: '#8b949e', fontSize: 13, textAlign: 'center', padding: 40 }}>
          Select a ticker to view price history.
        </div>
      )}

      {/* Stats */}
      {!loading && !error && data && renderStats()}
    </div>
  );
};
