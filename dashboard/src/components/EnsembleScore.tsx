import React from 'react';
import type { EnsembleScoreData } from '../types';

interface Props {
  data: EnsembleScoreData;
}

const COMPONENT_LABELS: Record<string, string> = {
  monte_carlo: 'Monte Carlo',
  hmm_regime: 'HMM Regime',
  garch: 'GARCH Vol',
  fama_french: 'Fama-French',
  copula_tail: 'Copula Tail',
  bayesian_decay: 'Signal Decay',
  event_study: 'Event Study',
};

const scoreColor = (s: number) => s >= 70 ? '#22c55e' : s >= 45 ? '#f59e0b' : '#ef4444';
const recColor = (r: string | null) => {
  if (!r) return '#8b949e';
  if (r.includes('strong')) return '#22c55e';
  if (r === 'buy' || r === 'sell') return '#60a5fa';
  if (r === 'hold') return '#f59e0b';
  return '#ef4444';
};

export const EnsembleScore: React.FC<Props> = ({ data }) => {
  const barH = 12;
  const entries = Object.entries(data.components).sort((a, b) => b[1] - a[1]);

  return (
    <div style={{ background: '#161b22', borderRadius: 8, padding: 16, border: '1px solid #30363d' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: '#e1e4e8' }}>Ensemble Score</div>
        <div style={{
          padding: '2px 10px', borderRadius: 12, fontSize: 11, fontWeight: 600,
          background: recColor(data.recommendation) + '22',
          color: recColor(data.recommendation),
          border: `1px solid ${recColor(data.recommendation)}44`,
          textTransform: 'uppercase',
        }}>
          {data.recommendation ?? 'N/A'}
        </div>
      </div>

      {/* Big score */}
      <div style={{ textAlign: 'center', marginBottom: 12 }}>
        <span style={{ fontSize: 36, fontWeight: 700, color: scoreColor(data.total_score) }}>
          {data.total_score.toFixed(0)}
        </span>
        <span style={{ fontSize: 14, color: '#8b949e' }}>/100</span>
        <div style={{ fontSize: 11, color: '#8b949e' }}>
          {data.n_models ?? 0} models | {((data.confidence ?? 0) * 100).toFixed(0)}% confidence
        </div>
      </div>

      {/* Component bars */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {entries.map(([key, score]) => (
          <div key={key} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div style={{ width: 80, fontSize: 10, color: '#8b949e', textAlign: 'right', flexShrink: 0 }}>
              {COMPONENT_LABELS[key] ?? key}
            </div>
            <div style={{ flex: 1, position: 'relative', height: barH, background: '#0d1117', borderRadius: 4 }}>
              <div style={{
                position: 'absolute', left: 0, top: 0, height: '100%', borderRadius: 4,
                width: `${Math.min(100, score)}%`, background: scoreColor(score),
                opacity: 0.7,
              }} />
            </div>
            <div style={{ width: 28, fontSize: 10, color: scoreColor(score), fontWeight: 600, textAlign: 'right' }}>
              {score.toFixed(0)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
