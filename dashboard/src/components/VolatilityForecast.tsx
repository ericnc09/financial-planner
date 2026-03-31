import React from 'react';
import type { GARCHData } from '../types';

interface Props {
  data: GARCHData;
}

export const VolatilityForecast: React.FC<Props> = ({ data }) => {
  const volColor = (interp: string | null) =>
    interp === 'volatility_expanding' ? '#ef4444'
    : interp === 'volatility_contracting' ? '#22c55e'
    : '#f59e0b';

  const volLabel = (interp: string | null) =>
    interp === 'volatility_expanding' ? 'Expanding'
    : interp === 'volatility_contracting' ? 'Contracting'
    : 'Stable';

  const fmtPct = (v: number | null) => v != null ? `${(v * 100).toFixed(1)}%` : '—';

  const maxVol = Math.max(
    data.historical_vol_60d || 0,
    data.forecast_5d_vol || 0,
    data.forecast_20d_vol || 0,
    data.long_run_vol_annual || 0,
  );

  const barWidth = (v: number | null) => v != null && maxVol > 0 ? `${(v / maxVol) * 100}%` : '0%';

  return (
    <div style={{ background: '#161b22', borderRadius: 8, padding: 16, border: '1px solid #30363d' }}>
      <div style={{ fontSize: 13, fontWeight: 600, color: '#e1e4e8', marginBottom: 4 }}>
        GARCH Volatility Forecast
      </div>
      <div style={{ fontSize: 11, color: '#8b949e', marginBottom: 12 }}>
        Persistence: {data.persistence?.toFixed(3) || '—'} | {data.n_observations || 0} observations
      </div>

      {/* Volatility bars */}
      {[
        { label: 'Hist 60d', val: data.historical_vol_60d, color: '#8b949e' },
        { label: 'Current', val: data.current_vol_annual, color: '#c9d1d9' },
        { label: '5d Forecast', val: data.forecast_5d_vol, color: volColor(data.forecast_5d_interpretation) },
        { label: '20d Forecast', val: data.forecast_20d_vol, color: volColor(data.forecast_20d_interpretation) },
        { label: 'Long-run', val: data.long_run_vol_annual, color: '#6366f1' },
      ].map(({ label, val, color }) => (
        <div key={label} style={{ marginBottom: 8 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 2 }}>
            <span style={{ color: '#8b949e' }}>{label}</span>
            <span style={{ color }}>{fmtPct(val)}</span>
          </div>
          <div style={{ height: 6, background: '#0d1117', borderRadius: 3 }}>
            <div style={{ height: '100%', width: barWidth(val), background: color, borderRadius: 3 }} />
          </div>
        </div>
      ))}

      {/* Interpretation */}
      <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
        <div style={{
          flex: 1, textAlign: 'center', padding: '6px 4px', borderRadius: 6,
          background: volColor(data.forecast_5d_interpretation) + '22',
          border: `1px solid ${volColor(data.forecast_5d_interpretation)}`,
        }}>
          <div style={{ fontSize: 10, color: '#8b949e' }}>5-Day</div>
          <div style={{ fontSize: 12, fontWeight: 600, color: volColor(data.forecast_5d_interpretation) }}>
            {volLabel(data.forecast_5d_interpretation)}
          </div>
          <div style={{ fontSize: 10, color: '#8b949e' }}>Ratio: {data.forecast_5d_ratio?.toFixed(2) || '—'}x</div>
        </div>
        <div style={{
          flex: 1, textAlign: 'center', padding: '6px 4px', borderRadius: 6,
          background: volColor(data.forecast_20d_interpretation) + '22',
          border: `1px solid ${volColor(data.forecast_20d_interpretation)}`,
        }}>
          <div style={{ fontSize: 10, color: '#8b949e' }}>20-Day</div>
          <div style={{ fontSize: 12, fontWeight: 600, color: volColor(data.forecast_20d_interpretation) }}>
            {volLabel(data.forecast_20d_interpretation)}
          </div>
          <div style={{ fontSize: 10, color: '#8b949e' }}>Ratio: {data.forecast_20d_ratio?.toFixed(2) || '—'}x</div>
        </div>
      </div>
    </div>
  );
};
