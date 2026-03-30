import React from 'react';
import type { MacroData } from '../types';

interface Props {
  macro: MacroData;
}

const regimeColors: Record<string, string> = {
  expansion: '#22c55e',
  transition: '#f59e0b',
  recession: '#ef4444',
};

const Indicator: React.FC<{ label: string; value: string | number | null; unit?: string }> = ({ label, value, unit }) => (
  <div style={{ textAlign: 'center', flex: 1 }}>
    <div style={{ fontSize: 11, color: '#8b949e', marginBottom: 2 }}>{label}</div>
    <div style={{ fontSize: 16, fontWeight: 600 }}>
      {value !== null && value !== undefined ? `${typeof value === 'number' ? value.toFixed(2) : value}${unit || ''}` : 'N/A'}
    </div>
  </div>
);

export const MacroGauge: React.FC<Props> = ({ macro }) => {
  const regime = macro.regime || 'transition';
  const color = regimeColors[regime] || '#8b949e';
  const score = macro.regime_score ?? 0.5;
  const angle = -90 + score * 180; // -90 (expansion) to 90 (recession)

  return (
    <div style={{ background: '#161b22', borderRadius: 12, padding: 20, border: '1px solid #30363d' }}>
      <h3 style={{ fontSize: 14, marginBottom: 12, color: '#c9d1d9' }}>Macro Regime</h3>

      {/* Gauge */}
      <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 16 }}>
        <svg width="160" height="90" viewBox="0 0 160 90">
          {/* Background arc */}
          <path d="M 10 80 A 70 70 0 0 1 150 80" fill="none" stroke="#21262d" strokeWidth="12" strokeLinecap="round" />
          {/* Colored arc based on score */}
          <path d="M 10 80 A 70 70 0 0 1 150 80" fill="none" stroke={color} strokeWidth="12" strokeLinecap="round"
            strokeDasharray={`${score * 220} 220`} opacity="0.8" />
          {/* Needle */}
          <line x1="80" y1="80" x2={80 + 50 * Math.cos((angle * Math.PI) / 180)} y2={80 + 50 * Math.sin((angle * Math.PI) / 180)}
            stroke="#e1e4e8" strokeWidth="2" strokeLinecap="round" />
          <circle cx="80" cy="80" r="4" fill="#e1e4e8" />
          {/* Labels */}
          <text x="10" y="88" fontSize="9" fill="#8b949e">Expand</text>
          <text x="126" y="88" fontSize="9" fill="#8b949e">Recess</text>
        </svg>
      </div>

      <div style={{ textAlign: 'center', marginBottom: 16 }}>
        <span style={{
          display: 'inline-block', padding: '4px 12px', borderRadius: 20,
          background: color + '22', color, fontSize: 13, fontWeight: 600, textTransform: 'uppercase',
        }}>
          {regime}
        </span>
      </div>

      <div style={{ display: 'flex', gap: 8 }}>
        <Indicator label="Yield Spread" value={macro.yield_spread} unit="%" />
        <Indicator label="Jobless Claims" value={macro.unemployment_claims ? Math.round(macro.unemployment_claims / 1000) : null} unit="k" />
        <Indicator label="CPI YoY" value={macro.cpi_yoy} unit="%" />
        <Indicator label="Fed Rate" value={macro.fed_funds_rate} unit="%" />
      </div>
    </div>
  );
};
