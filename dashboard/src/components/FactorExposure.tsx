import React from 'react';
import type { FamaFrenchData } from '../types';

interface Props {
  data: FamaFrenchData;
}

export const FactorExposure: React.FC<Props> = ({ data }) => {
  const factors = [
    { label: 'Market (Beta)', value: data.beta_market, desc: 'Sensitivity to market returns' },
    { label: 'Size (SMB)', value: data.beta_smb, desc: 'Small vs large cap tilt' },
    { label: 'Value (HML)', value: data.beta_hml, desc: 'Value vs growth tilt' },
    { label: 'Profitability (RMW)', value: data.beta_rmw, desc: 'Robust vs weak profits' },
    { label: 'Investment (CMA)', value: data.beta_cma, desc: 'Conservative vs aggressive' },
  ];

  const maxAbs = Math.max(...factors.map(f => Math.abs(f.value || 0)), 0.1);

  return (
    <div style={{ background: '#161b22', borderRadius: 8, padding: 16, border: '1px solid #30363d' }}>
      <div style={{ fontSize: 13, fontWeight: 600, color: '#e1e4e8', marginBottom: 4 }}>
        Fama-French Factor Exposure
      </div>
      <div style={{ fontSize: 11, color: '#8b949e', marginBottom: 12 }}>
        Alpha: <span style={{ color: (data.alpha_annual || 0) > 0 ? '#22c55e' : '#ef4444' }}>
          {((data.alpha_annual || 0) * 100).toFixed(1)}% ann.
        </span>
        {' | '}R²: {((data.r_squared || 0) * 100).toFixed(0)}%
      </div>

      {factors.map(({ label, value, desc }) => {
        const v = value || 0;
        const isPositive = v >= 0;
        const barWidth = (Math.abs(v) / maxAbs) * 50;
        const color = isPositive ? '#6366f1' : '#ef4444';

        return (
          <div key={label} style={{ marginBottom: 10 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 2 }}>
              <span style={{ color: '#c9d1d9' }}>{label}</span>
              <span style={{ color, fontWeight: 600 }}>{v.toFixed(2)}</span>
            </div>
            <div style={{ position: 'relative', height: 8, background: '#0d1117', borderRadius: 4 }}>
              {/* Center line */}
              <div style={{ position: 'absolute', left: '50%', top: 0, bottom: 0, width: 1, background: '#30363d' }} />
              {/* Bar */}
              <div style={{
                position: 'absolute',
                left: isPositive ? '50%' : `${50 - barWidth}%`,
                width: `${barWidth}%`,
                top: 0, bottom: 0, background: color, borderRadius: 4,
              }} />
            </div>
          </div>
        );
      })}
    </div>
  );
};
