import React from 'react';
import type { BayesianDecayData } from '../types';

interface Props {
  data: BayesianDecayData;
}

const qualityColor: Record<string, string> = {
  slow_decay: '#22c55e',
  moderate_decay: '#60a5fa',
  fast_decay: '#f59e0b',
  flash: '#ef4444',
  no_alpha: '#8b949e',
  no_signal: '#8b949e',
};

const qualityLabel: Record<string, string> = {
  slow_decay: 'Slow Decay',
  moderate_decay: 'Moderate',
  fast_decay: 'Fast Decay',
  flash: 'Flash',
  no_alpha: 'No Alpha',
  no_signal: 'No Signal',
};

export const BayesianDecayChart: React.FC<Props> = ({ data }) => {
  const halfLife = data.posterior_half_life ?? 0;
  const quality = data.decay_quality ?? 'no_signal';
  const color = qualityColor[quality] ?? '#8b949e';

  // Draw exponential decay curve
  const W = 200;
  const H = 80;
  const pad = { left: 30, right: 10, top: 10, bottom: 20 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  const maxDay = 40;
  const lambda = halfLife > 0 ? Math.log(2) / halfLife : 0.1;
  const points: string[] = [];
  for (let d = 0; d <= maxDay; d++) {
    const x = pad.left + (d / maxDay) * plotW;
    const y = pad.top + (1 - Math.exp(-lambda * d)) * plotH;
    points.push(`${d === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`);
  }

  const halfLifeX = pad.left + (halfLife / maxDay) * plotW;
  const halfLifeY = pad.top + 0.5 * plotH;

  return (
    <div style={{ background: '#161b22', borderRadius: 8, padding: 16, border: '1px solid #30363d' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: '#e1e4e8' }}>Signal Decay</div>
        <div style={{
          padding: '2px 8px', borderRadius: 10, fontSize: 10, fontWeight: 600,
          background: color + '22', color, border: `1px solid ${color}44`,
        }}>
          {qualityLabel[quality] ?? quality}
        </div>
      </div>

      <svg width={W} height={H} style={{ display: 'block', margin: '0 auto' }}>
        {/* Decay curve */}
        <path d={points.join(' ')} fill="none" stroke={color} strokeWidth={2} />

        {/* Half-life marker */}
        {halfLife > 0 && halfLife <= maxDay && (
          <>
            <line x1={halfLifeX} y1={pad.top} x2={halfLifeX} y2={H - pad.bottom}
              stroke={color} strokeWidth={1} strokeDasharray="3,3" opacity={0.5} />
            <text x={halfLifeX} y={H - pad.bottom + 12} textAnchor="middle" fill={color} fontSize={9}>
              t½={halfLife.toFixed(0)}d
            </text>
          </>
        )}

        {/* 50% line */}
        <line x1={pad.left} x2={W - pad.right} y1={halfLifeY} y2={halfLifeY}
          stroke="#30363d" strokeWidth={1} strokeDasharray="2,2" />

        {/* Y-axis labels */}
        <text x={pad.left - 4} y={pad.top + 4} textAnchor="end" fill="#8b949e" fontSize={8}>100%</text>
        <text x={pad.left - 4} y={halfLifeY + 3} textAnchor="end" fill="#8b949e" fontSize={8}>50%</text>
        <text x={pad.left - 4} y={H - pad.bottom} textAnchor="end" fill="#8b949e" fontSize={8}>0%</text>
      </svg>

      {/* Stats */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginTop: 8, fontSize: 11 }}>
        <div>
          <span style={{ color: '#8b949e' }}>CAR: </span>
          <span style={{ color: (data.total_car ?? 0) >= 0 ? '#22c55e' : '#ef4444', fontWeight: 600 }}>
            {((data.total_car ?? 0) * 100).toFixed(2)}%
          </span>
        </div>
        <div>
          <span style={{ color: '#8b949e' }}>IR: </span>
          <span style={{ color: '#c9d1d9', fontWeight: 600 }}>{(data.annualized_ir ?? 0).toFixed(2)}</span>
        </div>
        <div>
          <span style={{ color: '#8b949e' }}>Entry: </span>
          <span style={{ color: '#c9d1d9' }}>&lt;{(data.entry_window_days ?? 0).toFixed(0)}d</span>
        </div>
        <div>
          <span style={{ color: '#8b949e' }}>Exit: </span>
          <span style={{ color: '#c9d1d9' }}>&gt;{(data.exit_window_days ?? 0).toFixed(0)}d</span>
        </div>
      </div>
    </div>
  );
};
