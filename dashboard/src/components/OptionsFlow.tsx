import React from 'react';
import type { OptionsFlowData } from '../types';

interface Props {
  data: OptionsFlowData;
}

const pcrColor = (pcr: number | null) => {
  if (pcr === null) return '#8b949e';
  if (pcr < 0.7) return '#22c55e';
  if (pcr <= 1.0) return '#f59e0b';
  return '#ef4444';
};

const pcrLabel = (pcr: number | null) => {
  if (pcr === null) return 'N/A';
  if (pcr < 0.7) return 'Bullish';
  if (pcr <= 1.0) return 'Neutral';
  return 'Bearish';
};

const skewColor = (skew: number | null) => {
  if (skew === null) return '#8b949e';
  if (skew > 0.1) return '#ef4444';
  if (skew < -0.1) return '#22c55e';
  return '#f59e0b';
};

export const OptionsFlow: React.FC<Props> = ({ data }) => {
  const totalVol = (data.total_call_volume || 0) + (data.total_put_volume || 0);
  const callPct = totalVol > 0 ? ((data.total_call_volume || 0) / totalVol) * 100 : 50;
  const totalOI = (data.total_call_oi || 0) + (data.total_put_oi || 0);
  const callOIPct = totalOI > 0 ? ((data.total_call_oi || 0) / totalOI) * 100 : 50;

  return (
    <div style={{ background: '#0d1117', borderRadius: 12, border: '1px solid #30363d', padding: 20, marginBottom: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <h4 style={{ color: '#e1e4e8', margin: 0, fontSize: 15 }}>Options Flow</h4>
        <span style={{ fontSize: 11, color: '#8b949e' }}>
          Expiry: {data.nearest_expiry || 'N/A'}
        </span>
      </div>

      {/* Stat Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10, marginBottom: 16 }}>
        {/* PCR */}
        <div style={{ background: '#161b22', borderRadius: 8, padding: 12, border: '1px solid #30363d' }}>
          <div style={{ fontSize: 10, color: '#8b949e', marginBottom: 4 }}>Put/Call Ratio</div>
          <div style={{ fontSize: 20, fontWeight: 700, color: pcrColor(data.pcr) }}>
            {data.pcr !== null ? data.pcr.toFixed(2) : 'N/A'}
          </div>
          <div style={{ fontSize: 10, color: pcrColor(data.pcr), marginTop: 2 }}>{pcrLabel(data.pcr)}</div>
        </div>

        {/* Unusual Volume */}
        <div style={{ background: '#161b22', borderRadius: 8, padding: 12, border: '1px solid #30363d' }}>
          <div style={{ fontSize: 10, color: '#8b949e', marginBottom: 4 }}>Unusual Volume</div>
          <div style={{ fontSize: 20, fontWeight: 700, color: (data.unusual_volume_score || 0) > 0.5 ? '#f59e0b' : '#c9d1d9' }}>
            {data.unusual_volume_score !== null ? `${(data.unusual_volume_score * 100).toFixed(0)}%` : 'N/A'}
          </div>
          <div style={{
            height: 4, borderRadius: 2, background: '#21262d', marginTop: 6,
          }}>
            <div style={{
              height: '100%', borderRadius: 2, background: (data.unusual_volume_score || 0) > 0.5 ? '#f59e0b' : '#6366f1',
              width: `${(data.unusual_volume_score || 0) * 100}%`,
            }} />
          </div>
        </div>

        {/* IV Skew */}
        <div style={{ background: '#161b22', borderRadius: 8, padding: 12, border: '1px solid #30363d' }}>
          <div style={{ fontSize: 10, color: '#8b949e', marginBottom: 4 }}>IV Skew</div>
          <div style={{ fontSize: 20, fontWeight: 700, color: skewColor(data.iv_skew) }}>
            {data.iv_skew !== null ? `${(data.iv_skew * 100).toFixed(1)}%` : 'N/A'}
          </div>
          <div style={{ fontSize: 10, color: skewColor(data.iv_skew), marginTop: 2 }}>
            {data.iv_skew === null ? '' : data.iv_skew > 0.1 ? 'Bearish Skew' : data.iv_skew < -0.1 ? 'Bullish Skew' : 'Balanced'}
          </div>
        </div>

        {/* Max Pain */}
        <div style={{ background: '#161b22', borderRadius: 8, padding: 12, border: '1px solid #30363d' }}>
          <div style={{ fontSize: 10, color: '#8b949e', marginBottom: 4 }}>Max Pain</div>
          <div style={{ fontSize: 20, fontWeight: 700, color: '#c9d1d9' }}>
            {data.max_pain !== null ? `$${data.max_pain.toFixed(0)}` : 'N/A'}
          </div>
          <div style={{ fontSize: 10, color: '#8b949e', marginTop: 2 }}>Strike target</div>
        </div>
      </div>

      {/* Volume Breakdown */}
      <div style={{ marginBottom: 10 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#8b949e', marginBottom: 4 }}>
          <span>Calls: {(data.total_call_volume || 0).toLocaleString()}</span>
          <span>Volume</span>
          <span>Puts: {(data.total_put_volume || 0).toLocaleString()}</span>
        </div>
        <div style={{ display: 'flex', height: 8, borderRadius: 4, overflow: 'hidden' }}>
          <div style={{ width: `${callPct}%`, background: '#22c55e', transition: 'width 0.3s' }} />
          <div style={{ width: `${100 - callPct}%`, background: '#ef4444', transition: 'width 0.3s' }} />
        </div>
      </div>

      {/* Open Interest Breakdown */}
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#8b949e', marginBottom: 4 }}>
          <span>Calls: {(data.total_call_oi || 0).toLocaleString()}</span>
          <span>Open Interest</span>
          <span>Puts: {(data.total_put_oi || 0).toLocaleString()}</span>
        </div>
        <div style={{ display: 'flex', height: 8, borderRadius: 4, overflow: 'hidden' }}>
          <div style={{ width: `${callOIPct}%`, background: '#22c55e', opacity: 0.7, transition: 'width 0.3s' }} />
          <div style={{ width: `${100 - callOIPct}%`, background: '#ef4444', opacity: 0.7, transition: 'width 0.3s' }} />
        </div>
      </div>
    </div>
  );
};
