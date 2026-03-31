import React from 'react';
import type { ExtendedMacroData } from '../types';

interface Props {
  data: ExtendedMacroData;
}

export const ExtendedMacro: React.FC<Props> = ({ data }) => {
  const indicators = [
    {
      label: 'VIX',
      value: data.vix,
      format: (v: number) => v.toFixed(1),
      color: v => v > 30 ? '#ef4444' : v > 20 ? '#f59e0b' : '#22c55e',
      desc: 'Fear gauge',
    },
    {
      label: 'Consumer Sentiment',
      value: data.consumer_sentiment,
      format: (v: number) => v.toFixed(1),
      color: v => v < 60 ? '#ef4444' : v < 80 ? '#f59e0b' : '#22c55e',
      desc: 'UMich survey',
    },
    {
      label: 'M2 Supply',
      value: data.money_supply_m2,
      format: (v: number) => `$${(v / 1000).toFixed(1)}T`,
      color: () => '#6366f1',
      desc: 'Money supply',
    },
    {
      label: 'Housing Starts',
      value: data.housing_starts,
      format: (v: number) => `${v.toFixed(0)}k`,
      color: v => v < 1200 ? '#ef4444' : v < 1500 ? '#f59e0b' : '#22c55e',
      desc: 'New construction',
    },
    {
      label: 'Industrial Prod.',
      value: data.industrial_production,
      format: (v: number) => v.toFixed(1),
      color: () => '#c9d1d9',
      desc: 'Output index',
    },
  ];

  return (
    <div style={{
      background: '#161b22', borderRadius: 12, padding: 16,
      border: '1px solid #30363d', marginBottom: 20,
    }}>
      <div style={{ fontSize: 13, fontWeight: 600, color: '#e1e4e8', marginBottom: 12 }}>
        Extended Macro Indicators
      </div>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
        {indicators.map(({ label, value, format, color, desc }) => (
          <div key={label} style={{
            flex: '1 1 120px', textAlign: 'center', padding: '8px 4px',
            background: '#0d1117', borderRadius: 8, border: '1px solid #30363d',
          }}>
            <div style={{ fontSize: 10, color: '#8b949e' }}>{label}</div>
            <div style={{
              fontSize: 16, fontWeight: 700,
              color: value != null ? (color as any)(value) : '#484f58',
            }}>
              {value != null ? format(value) : '—'}
            </div>
            <div style={{ fontSize: 9, color: '#484f58' }}>{desc}</div>
          </div>
        ))}
      </div>
    </div>
  );
};
