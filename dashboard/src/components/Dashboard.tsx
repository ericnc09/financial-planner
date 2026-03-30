import React from 'react';
import type { DashboardData } from '../types';
import { MacroGauge } from './MacroGauge';
import { SignalTable } from './SignalTable';

interface Props {
  data: DashboardData;
  onRefresh: () => void;
  onRunPipeline: () => void;
  loading: boolean;
}

const StatCard: React.FC<{ label: string; value: string; color?: string }> = ({ label, value, color }) => (
  <div style={{
    background: '#161b22', borderRadius: 12, padding: 16, border: '1px solid #30363d', flex: 1, minWidth: 140,
  }}>
    <div style={{ fontSize: 11, color: '#8b949e', marginBottom: 4 }}>{label}</div>
    <div style={{ fontSize: 22, fontWeight: 700, color: color || '#e1e4e8' }}>{value}</div>
  </div>
);

export const Dashboard: React.FC<Props> = ({ data, onRefresh, onRunPipeline, loading }) => {
  const { signals, macro } = data;

  const passingSignals = signals.filter(s => s.passes_threshold).length;
  const buySignals = signals.filter(s => s.direction === 'buy').length;
  const sellSignals = signals.filter(s => s.direction === 'sell').length;
  const regime = macro?.regime || 'unknown';

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto', padding: 24 }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <div>
          <h1 style={{ fontSize: 24, fontWeight: 700, color: '#e1e4e8' }}>Smart Money Follows</h1>
          <p style={{ fontSize: 12, color: '#8b949e', marginTop: 4 }}>
            Congressional & insider trade signals with conviction scoring — you decide, you execute
          </p>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button onClick={onRefresh} disabled={loading} style={{
            padding: '8px 16px', borderRadius: 8, border: '1px solid #30363d', background: '#21262d',
            color: '#c9d1d9', cursor: 'pointer', fontSize: 12,
          }}>
            {loading ? 'Loading...' : 'Refresh'}
          </button>
          <button onClick={onRunPipeline} style={{
            padding: '8px 16px', borderRadius: 8, border: 'none', background: '#6366f1',
            color: '#fff', cursor: 'pointer', fontSize: 12, fontWeight: 600,
          }}>
            Run Pipeline
          </button>
        </div>
      </div>

      {/* Stats Row */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 20, flexWrap: 'wrap' }}>
        <StatCard label="Total Signals" value={String(signals.length)} />
        <StatCard label="High Conviction" value={String(passingSignals)} color="#22c55e" />
        <StatCard label="Buy Signals" value={String(buySignals)} color="#22c55e" />
        <StatCard label="Sell Signals" value={String(sellSignals)} color="#ef4444" />
        <StatCard label="Macro Regime" value={regime.charAt(0).toUpperCase() + regime.slice(1)}
          color={regime === 'expansion' ? '#22c55e' : regime === 'recession' ? '#ef4444' : '#f59e0b'} />
      </div>

      {/* Macro Gauge */}
      {macro && (
        <div style={{ marginBottom: 20 }}>
          <MacroGauge macro={macro} />
        </div>
      )}

      {/* Signals Table */}
      <div style={{ marginBottom: 20 }}>
        <SignalTable signals={signals} />
      </div>
    </div>
  );
};
