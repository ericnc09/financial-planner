import React, { useEffect, useState } from 'react';
import type { MeanVarianceData } from '../types';
import { api } from '../api/client';

export const MeanVarianceChart: React.FC = () => {
  const [data, setData] = useState<MeanVarianceData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getMeanVariance()
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div style={{ color: '#8b949e', fontSize: 13, padding: 16 }}>Loading portfolio...</div>;
  if (!data) return null;

  const { max_sharpe, min_variance, equal_weight, efficient_frontier, risk_contribution, tickers } = data;

  // --- Efficient Frontier Chart ---
  const W = 340;
  const H = 160;
  const pad = { top: 16, right: 16, bottom: 28, left: 48 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  const allVols = efficient_frontier.map(p => p.volatility).concat([
    max_sharpe.volatility, min_variance.volatility, equal_weight.volatility,
  ]);
  const allRets = efficient_frontier.map(p => p.return).concat([
    max_sharpe.expected_return, min_variance.expected_return, equal_weight.expected_return,
  ]);

  const minVol = Math.min(...allVols) * 0.9;
  const maxVol = Math.max(...allVols) * 1.1 || 0.01;
  const minRet = Math.min(...allRets) * (Math.min(...allRets) < 0 ? 1.2 : 0.8);
  const maxRet = Math.max(...allRets) * 1.1 || 0.01;

  const x = (vol: number) => pad.left + ((vol - minVol) / (maxVol - minVol)) * plotW;
  const y = (ret: number) => pad.top + (1 - (ret - minRet) / (maxRet - minRet)) * plotH;

  const frontierPath = efficient_frontier
    .sort((a, b) => a.volatility - b.volatility)
    .map((p, i) => `${i === 0 ? 'M' : 'L'}${x(p.volatility).toFixed(1)},${y(p.return).toFixed(1)}`)
    .join(' ');

  // --- Weight bars ---
  const sortedWeights = Object.entries(max_sharpe.weights)
    .sort((a, b) => b[1] - a[1])
    .filter(([, w]) => w > 0.001);

  const weightColors = ['#6366f1', '#22c55e', '#f59e0b', '#ef4444', '#60a5fa', '#a78bfa', '#f472b6', '#34d399'];

  return (
    <div style={{
      background: '#0d1117', border: '1px solid #30363d', borderRadius: 12,
      padding: 20, marginBottom: 20,
    }}>
      <div style={{ fontSize: 15, fontWeight: 700, color: '#e1e4e8', marginBottom: 4 }}>
        Portfolio Optimization
      </div>
      <div style={{ fontSize: 11, color: '#8b949e', marginBottom: 16 }}>
        Mean-Variance across {data.n_assets} signal tickers
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        {/* Left: Efficient Frontier */}
        <div style={{ background: '#161b22', borderRadius: 8, padding: 12, border: '1px solid #30363d' }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: '#c9d1d9', marginBottom: 8 }}>Efficient Frontier</div>
          <svg width={W} height={H} style={{ display: 'block' }}>
            {/* Grid lines */}
            {[0.25, 0.5, 0.75].map(f => {
              const yPos = pad.top + f * plotH;
              return <line key={f} x1={pad.left} x2={W - pad.right} y1={yPos} y2={yPos} stroke="#21262d" strokeWidth={1} />;
            })}

            {/* Frontier curve */}
            {frontierPath && <path d={frontierPath} fill="none" stroke="#6366f180" strokeWidth={2} />}

            {/* Frontier dots */}
            {efficient_frontier.map((p, i) => (
              <circle key={i} cx={x(p.volatility)} cy={y(p.return)} r={2.5} fill="#6366f1" opacity={0.5} />
            ))}

            {/* Min Variance point */}
            <circle cx={x(min_variance.volatility)} cy={y(min_variance.expected_return)}
              r={5} fill="none" stroke="#f59e0b" strokeWidth={2} />
            <text x={x(min_variance.volatility) + 8} y={y(min_variance.expected_return) + 3}
              fill="#f59e0b" fontSize={9}>MinVar</text>

            {/* Max Sharpe point */}
            <circle cx={x(max_sharpe.volatility)} cy={y(max_sharpe.expected_return)}
              r={5} fill="#22c55e" />
            <text x={x(max_sharpe.volatility) + 8} y={y(max_sharpe.expected_return) + 3}
              fill="#22c55e" fontSize={9}>MaxSharpe</text>

            {/* Equal Weight point */}
            <circle cx={x(equal_weight.volatility)} cy={y(equal_weight.expected_return)}
              r={4} fill="none" stroke="#8b949e" strokeWidth={1.5} strokeDasharray="2,2" />

            {/* Axis labels */}
            <text x={W / 2} y={H - 2} textAnchor="middle" fill="#8b949e" fontSize={9}>Volatility</text>
            <text x={8} y={H / 2} textAnchor="middle" fill="#8b949e" fontSize={9}
              transform={`rotate(-90, 8, ${H / 2})`}>Return</text>

            {/* Y ticks */}
            {[minRet, (minRet + maxRet) / 2, maxRet].map((v, i) => (
              <text key={i} x={pad.left - 4} y={y(v)} textAnchor="end" fill="#8b949e" fontSize={8}
                dominantBaseline="middle">{(v * 100).toFixed(0)}%</text>
            ))}
          </svg>
        </div>

        {/* Right: Weights + Stats */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {/* Portfolio stats */}
          <div style={{ background: '#161b22', borderRadius: 8, padding: 12, border: '1px solid #30363d' }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: '#c9d1d9', marginBottom: 8 }}>Max Sharpe Portfolio</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, fontSize: 11 }}>
              <div>
                <div style={{ color: '#8b949e' }}>Return</div>
                <div style={{ color: '#22c55e', fontWeight: 600 }}>{(max_sharpe.expected_return * 100).toFixed(1)}%</div>
              </div>
              <div>
                <div style={{ color: '#8b949e' }}>Volatility</div>
                <div style={{ color: '#f59e0b', fontWeight: 600 }}>{(max_sharpe.volatility * 100).toFixed(1)}%</div>
              </div>
              <div>
                <div style={{ color: '#8b949e' }}>Sharpe</div>
                <div style={{ color: '#6366f1', fontWeight: 600 }}>{max_sharpe.sharpe_ratio.toFixed(2)}</div>
              </div>
            </div>
          </div>

          {/* Weight allocation bars */}
          <div style={{ background: '#161b22', borderRadius: 8, padding: 12, border: '1px solid #30363d', flex: 1 }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: '#c9d1d9', marginBottom: 8 }}>Allocation</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
              {sortedWeights.map(([ticker, weight], i) => (
                <div key={ticker} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <div style={{ width: 40, fontSize: 10, color: '#c9d1d9', fontWeight: 600, textAlign: 'right' }}>
                    {ticker}
                  </div>
                  <div style={{ flex: 1, height: 10, background: '#0d1117', borderRadius: 4, position: 'relative' }}>
                    <div style={{
                      position: 'absolute', left: 0, top: 0, height: '100%', borderRadius: 4,
                      width: `${Math.min(100, weight * 100)}%`,
                      background: weightColors[i % weightColors.length],
                      opacity: 0.75,
                    }} />
                  </div>
                  <div style={{ width: 36, fontSize: 10, color: '#8b949e', textAlign: 'right' }}>
                    {(weight * 100).toFixed(0)}%
                  </div>
                </div>
              ))}
              {sortedWeights.length === 0 && (
                <div style={{ fontSize: 11, color: '#8b949e' }}>No allocations yet</div>
              )}
            </div>
          </div>

          {/* Risk contribution */}
          {Object.keys(risk_contribution).length > 0 && (
            <div style={{ background: '#161b22', borderRadius: 8, padding: 12, border: '1px solid #30363d' }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: '#c9d1d9', marginBottom: 6 }}>Risk Contribution</div>
              <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                {Object.entries(risk_contribution)
                  .filter(([, v]) => v > 0.01)
                  .sort((a, b) => b[1] - a[1])
                  .map(([ticker, pct], i) => (
                    <div key={ticker} style={{
                      padding: '2px 8px', borderRadius: 10, fontSize: 10,
                      background: weightColors[i % weightColors.length] + '22',
                      color: weightColors[i % weightColors.length],
                      border: `1px solid ${weightColors[i % weightColors.length]}44`,
                    }}>
                      {ticker} {(pct * 100).toFixed(0)}%
                    </div>
                  ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
