import React from 'react';
import type { MonteCarloData } from '../types';

interface Props {
  data: MonteCarloData;
}

export const MonteCarloChart: React.FC<Props> = ({ data }) => {
  const renderHorizon = (label: string, horizon: string) => {
    const h = data.horizons[horizon];
    if (!h) return null;

    const { percentiles: p } = h;
    const range = p.p90 - p.p10;
    const scale = (val: number) => ((val - p.p10) / range) * 100;

    const currentPos = scale(data.current_price);
    const profitPct = (h.probability_of_profit * 100).toFixed(0);
    const returnPct = (h.expected_return * 100).toFixed(1);
    const varPct = (h.value_at_risk_95 * 100).toFixed(1);

    return (
      <div style={{ marginBottom: 16 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
          <span style={{ fontSize: 12, fontWeight: 600, color: '#e1e4e8' }}>{label}</span>
          <span style={{ fontSize: 11, color: '#8b949e' }}>
            P(profit): <span style={{ color: Number(profitPct) > 50 ? '#22c55e' : '#ef4444' }}>{profitPct}%</span>
            {' | '}E[r]: {returnPct}% | VaR₉₅: {varPct}%
          </span>
        </div>
        {/* Price fan visualization */}
        <div style={{ position: 'relative', height: 32, background: '#0d1117', borderRadius: 4, overflow: 'hidden' }}>
          {/* P10-P90 range (lightest) */}
          <div style={{
            position: 'absolute', left: '0%', right: '0%', top: 4, bottom: 4,
            background: '#6366f122', borderRadius: 4,
          }} />
          {/* P25-P75 range */}
          <div style={{
            position: 'absolute',
            left: `${scale(p.p25)}%`, width: `${scale(p.p75) - scale(p.p25)}%`,
            top: 4, bottom: 4, background: '#6366f144', borderRadius: 4,
          }} />
          {/* Current price marker */}
          <div style={{
            position: 'absolute', left: `${Math.min(100, Math.max(0, currentPos))}%`,
            top: 0, bottom: 0, width: 2, background: '#fff', zIndex: 2,
          }} />
          {/* Median marker */}
          <div style={{
            position: 'absolute', left: `${scale(p.p50)}%`,
            top: 2, bottom: 2, width: 2, background: '#6366f1', zIndex: 1,
          }} />
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#8b949e', marginTop: 2 }}>
          <span>${p.p10.toFixed(0)}</span>
          <span>${p.p25.toFixed(0)}</span>
          <span style={{ color: '#6366f1' }}>${p.p50.toFixed(0)}</span>
          <span>${p.p75.toFixed(0)}</span>
          <span>${p.p90.toFixed(0)}</span>
        </div>
      </div>
    );
  };

  return (
    <div style={{ background: '#161b22', borderRadius: 8, padding: 16, border: '1px solid #30363d' }}>
      <div style={{ fontSize: 13, fontWeight: 600, color: '#e1e4e8', marginBottom: 4 }}>
        Monte Carlo Simulation
      </div>
      <div style={{ fontSize: 11, color: '#8b949e', marginBottom: 12 }}>
        {data.n_simulations.toLocaleString()} paths | Vol: {((data.annual_volatility || 0) * 100).toFixed(0)}% ann.
        | Current: ${data.current_price.toFixed(2)}
      </div>
      {renderHorizon('30 Day', '30')}
      {renderHorizon('90 Day', '90')}
      <div style={{ fontSize: 10, color: '#484f58', marginTop: 4 }}>
        White = current price | Purple = median forecast | Bands = P10-P90
      </div>
    </div>
  );
};
