import React from 'react';
import type { CopulaTailRiskData } from '../types';

interface Props {
  data: CopulaTailRiskData;
}

export const TailRiskGauge: React.FC<Props> = ({ data }) => {
  const score = data.tail_risk_score ?? 0;
  const color = score >= 70 ? '#ef4444' : score >= 40 ? '#f59e0b' : '#22c55e';
  const label = score >= 70 ? 'High' : score >= 40 ? 'Moderate' : 'Low';

  // Gauge arc
  const W = 160;
  const H = 90;
  const cx = W / 2;
  const cy = H - 10;
  const r = 60;
  const startAngle = Math.PI;
  const endAngle = 0;
  const scoreAngle = startAngle - (score / 100) * Math.PI;

  const arcPath = (from: number, to: number) => {
    const x1 = cx + r * Math.cos(from);
    const y1 = cy - r * Math.sin(from);
    const x2 = cx + r * Math.cos(to);
    const y2 = cy - r * Math.sin(to);
    const large = from - to > Math.PI ? 1 : 0;
    return `M${x1},${y1} A${r},${r} 0 ${large} 1 ${x2},${y2}`;
  };

  const needleX = cx + (r - 10) * Math.cos(scoreAngle);
  const needleY = cy - (r - 10) * Math.sin(scoreAngle);

  return (
    <div style={{ background: '#161b22', borderRadius: 8, padding: 16, border: '1px solid #30363d' }}>
      <div style={{ fontSize: 13, fontWeight: 600, color: '#e1e4e8', marginBottom: 8 }}>
        Copula Tail Risk
      </div>

      <svg width={W} height={H} style={{ display: 'block', margin: '0 auto' }}>
        {/* Background arc */}
        <path d={arcPath(startAngle, endAngle)} fill="none" stroke="#21262d" strokeWidth={12} strokeLinecap="round" />
        {/* Score arc */}
        <path d={arcPath(startAngle, scoreAngle)} fill="none" stroke={color} strokeWidth={12} strokeLinecap="round" />
        {/* Needle */}
        <line x1={cx} y1={cy} x2={needleX} y2={needleY} stroke="#e1e4e8" strokeWidth={2} />
        <circle cx={cx} cy={cy} r={4} fill="#e1e4e8" />
        {/* Score text */}
        <text x={cx} y={cy - 20} textAnchor="middle" fill={color} fontSize={22} fontWeight={700}>
          {score.toFixed(0)}
        </text>
        <text x={cx} y={cy - 5} textAnchor="middle" fill="#8b949e" fontSize={10}>
          {label} Risk
        </text>
      </svg>

      {/* Risk metrics */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginTop: 8, fontSize: 11 }}>
        <div>
          <span style={{ color: '#8b949e' }}>VaR 95%: </span>
          <span style={{ color: '#ef4444', fontWeight: 600 }}>{((data.var_95 ?? 0) * 100).toFixed(2)}%</span>
        </div>
        <div>
          <span style={{ color: '#8b949e' }}>CVaR 95%: </span>
          <span style={{ color: '#ef4444', fontWeight: 600 }}>{((data.cvar_95 ?? 0) * 100).toFixed(2)}%</span>
        </div>
        <div>
          <span style={{ color: '#8b949e' }}>Tail Dep: </span>
          <span style={{ color: '#c9d1d9', fontWeight: 600 }}>{((data.tail_dep_lower ?? 0) * 100).toFixed(1)}%</span>
        </div>
        <div>
          <span style={{ color: '#8b949e' }}>Crash x Indep: </span>
          <span style={{ color: '#f59e0b', fontWeight: 600 }}>{data.tail_dep_ratio ?? 0}x</span>
        </div>
      </div>
    </div>
  );
};
