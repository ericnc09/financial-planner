import React from 'react';

interface Props {
  signalScore: number;
  fundamentalScore: number;
  macroModifier: number;
  conviction: number;
}

const colors = {
  signal: '#6366f1',
  fundamental: '#22c55e',
  macro: '#f59e0b',
};

export const ConvictionBar: React.FC<Props> = ({ signalScore, fundamentalScore, macroModifier, conviction }) => {
  const total = signalScore + fundamentalScore;
  const sigPct = total > 0 ? (signalScore / total) * 100 : 33;
  const fundPct = total > 0 ? (fundamentalScore / total) * 100 : 33;

  return (
    <div style={{ width: '100%' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4, fontSize: 11, color: '#8b949e' }}>
        <span>Conviction: {(conviction * 100).toFixed(1)}%</span>
        <span>Macro: {macroModifier.toFixed(2)}x</span>
      </div>
      <div style={{ display: 'flex', height: 8, borderRadius: 4, overflow: 'hidden', background: '#21262d' }}>
        <div style={{ width: `${sigPct}%`, background: colors.signal }} title={`Signal: ${signalScore.toFixed(2)}`} />
        <div style={{ width: `${fundPct}%`, background: colors.fundamental }} title={`Fundamental: ${fundamentalScore.toFixed(2)}`} />
      </div>
      <div style={{ display: 'flex', gap: 12, marginTop: 4, fontSize: 10, color: '#8b949e' }}>
        <span><span style={{ color: colors.signal }}>&#9632;</span> Signal {signalScore.toFixed(2)}</span>
        <span><span style={{ color: colors.fundamental }}>&#9632;</span> Fundamental {fundamentalScore.toFixed(2)}</span>
      </div>
    </div>
  );
};
