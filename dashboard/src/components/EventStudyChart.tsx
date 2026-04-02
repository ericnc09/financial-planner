import React from 'react';
import type { EventStudyData } from '../types';

interface Props {
  events: EventStudyData[];
}

export const EventStudyChart: React.FC<Props> = ({ events }) => {
  if (!events.length) return null;

  // Average daily CARs across all events for the ticker
  const maxLen = Math.max(...events.map(e => e.daily_cars.length));
  const avgCars: number[] = [];
  for (let i = 0; i < maxLen; i++) {
    const vals = events.map(e => e.daily_cars[i]).filter(v => v !== undefined);
    avgCars.push(vals.reduce((a, b) => a + b, 0) / vals.length);
  }

  // Day labels: event window is [-5, +20]
  const days = avgCars.map((_, i) => i - 5);

  // Chart dimensions
  const W = 320;
  const H = 140;
  const pad = { top: 20, right: 16, bottom: 28, left: 48 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  const minY = Math.min(0, ...avgCars) * 1.2;
  const maxY = Math.max(0, ...avgCars) * 1.2 || 0.01;
  const rangeY = maxY - minY || 0.01;

  const x = (i: number) => pad.left + (i / (avgCars.length - 1 || 1)) * plotW;
  const y = (v: number) => pad.top + (1 - (v - minY) / rangeY) * plotH;

  const zeroY = y(0);
  const pathD = avgCars.map((v, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');

  // Fill area between line and zero
  const areaD = `${pathD} L${x(avgCars.length - 1).toFixed(1)},${zeroY.toFixed(1)} L${x(0).toFixed(1)},${zeroY.toFixed(1)} Z`;

  const lastCar = avgCars[avgCars.length - 1];
  const isPositive = lastCar >= 0;
  const lineColor = isPositive ? '#22c55e' : '#ef4444';
  const fillColor = isPositive ? '#22c55e15' : '#ef444415';

  // Summary stats
  const sigCount = events.filter(e => e.is_significant).length;
  const avgCar5d = events.filter(e => e.car_5d !== null).reduce((s, e) => s + (e.car_5d ?? 0), 0) / events.length;
  const avgCar20d = events.filter(e => e.car_20d !== null).reduce((s, e) => s + (e.car_20d ?? 0), 0) / events.length;

  return (
    <div style={{ background: '#161b22', borderRadius: 8, padding: 16, border: '1px solid #30363d' }}>
      <div style={{ fontSize: 13, fontWeight: 600, color: '#e1e4e8', marginBottom: 8 }}>
        Event Study — Cumulative Abnormal Returns
      </div>
      <div style={{ fontSize: 11, color: '#8b949e', marginBottom: 12 }}>
        {events.length} event{events.length > 1 ? 's' : ''} | {sigCount} significant (p&lt;0.05)
      </div>

      <svg width={W} height={H} style={{ display: 'block', margin: '0 auto' }}>
        {/* Zero line */}
        <line x1={pad.left} x2={W - pad.right} y1={zeroY} y2={zeroY}
          stroke="#30363d" strokeWidth={1} strokeDasharray="4,3" />

        {/* Day 0 vertical line */}
        {days.includes(0) && (
          <line x1={x(5)} x2={x(5)} y1={pad.top} y2={H - pad.bottom}
            stroke="#6366f144" strokeWidth={1} strokeDasharray="3,3" />
        )}

        {/* Fill area */}
        <path d={areaD} fill={fillColor} />

        {/* CAR line */}
        <path d={pathD} fill="none" stroke={lineColor} strokeWidth={2} />

        {/* Endpoint dot */}
        <circle cx={x(avgCars.length - 1)} cy={y(lastCar)} r={3} fill={lineColor} />

        {/* Y-axis labels */}
        {[minY, 0, maxY].map((v, i) => (
          <text key={i} x={pad.left - 4} y={y(v)} textAnchor="end" fill="#8b949e"
            fontSize={9} dominantBaseline="middle">
            {(v * 100).toFixed(1)}%
          </text>
        ))}

        {/* X-axis labels */}
        {[0, 5, 10, 15, 20].filter(d => d - 5 >= days[0] && d - 5 <= days[days.length - 1]).map(dayIdx => (
          <text key={dayIdx} x={x(dayIdx)} y={H - pad.bottom + 14} textAnchor="middle"
            fill="#8b949e" fontSize={9}>
            {dayIdx === 5 ? 'T' : `T+${dayIdx - 5}`}
          </text>
        ))}

        {/* Endpoint label */}
        <text x={x(avgCars.length - 1)} y={y(lastCar) - 8} textAnchor="middle"
          fill={lineColor} fontSize={10} fontWeight={600}>
          {(lastCar * 100).toFixed(2)}%
        </text>
      </svg>

      {/* Stats row */}
      <div style={{ display: 'flex', justifyContent: 'space-around', marginTop: 12, fontSize: 11 }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ color: '#8b949e' }}>CAR +5d</div>
          <div style={{ color: avgCar5d >= 0 ? '#22c55e' : '#ef4444', fontWeight: 600 }}>
            {(avgCar5d * 100).toFixed(2)}%
          </div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ color: '#8b949e' }}>CAR +20d</div>
          <div style={{ color: avgCar20d >= 0 ? '#22c55e' : '#ef4444', fontWeight: 600 }}>
            {(avgCar20d * 100).toFixed(2)}%
          </div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ color: '#8b949e' }}>Significant</div>
          <div style={{ color: '#c9d1d9', fontWeight: 600 }}>{sigCount}/{events.length}</div>
        </div>
      </div>
    </div>
  );
};
