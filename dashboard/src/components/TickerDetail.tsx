import React, { useEffect, useState } from 'react';
import type { TickerAnalysis } from '../types';
import { api } from '../api/client';
import { MonteCarloChart } from './MonteCarloChart';
import { VolatilityForecast } from './VolatilityForecast';
import { EventStudyChart } from './EventStudyChart';
import { FactorExposure } from './FactorExposure';
import { TailRiskGauge } from './TailRiskGauge';
import { EnsembleScore } from './EnsembleScore';
import { BayesianDecayChart } from './BayesianDecayChart';
import { OptionsFlow } from './OptionsFlow';

interface Props {
  ticker: string;
  onClose: () => void;
}

export const TickerDetail: React.FC<Props> = ({ ticker, onClose }) => {
  const [data, setData] = useState<TickerAnalysis | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    api.getTickerAnalysis(ticker)
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, [ticker]);

  const hasData = data && (
    data.monte_carlo || data.hmm || data.garch || data.fama_french ||
    data.copula_tail_risk || data.event_studies?.length ||
    data.bayesian_decay?.length || data.ensemble_scores?.length
  );

  return (
    <div style={{
      background: '#0d1117', border: '1px solid #30363d', borderRadius: 12,
      padding: 24, marginBottom: 20,
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <h2 style={{ fontSize: 18, fontWeight: 700, color: '#e1e4e8', margin: 0 }}>
          Analysis: {ticker}
        </h2>
        <div style={{ display: 'flex', gap: 8 }}>
          <a href={`/api/export/analysis/${ticker}?format=csv`} download style={{
            background: '#21262d', border: '1px solid #30363d', borderRadius: 6,
            color: '#8b949e', textDecoration: 'none', padding: '4px 12px', fontSize: 12, lineHeight: '20px',
          }}>Export CSV</a>
          <button onClick={onClose} style={{
            background: '#21262d', border: '1px solid #30363d', borderRadius: 6,
            color: '#c9d1d9', cursor: 'pointer', padding: '4px 12px', fontSize: 12,
          }}>Close</button>
        </div>
      </div>

      {loading && <div style={{ color: '#8b949e', fontSize: 13 }}>Loading analysis...</div>}

      {!loading && !hasData && (
        <div style={{ color: '#8b949e', fontSize: 13 }}>
          No analysis data yet. Run the pipeline to generate model results.
        </div>
      )}

      {!loading && data && hasData && (
        <>
          {/* Ensemble score banner if available */}
          {data.ensemble_scores && data.ensemble_scores.length > 0 && (
            <div style={{ marginBottom: 16 }}>
              <EnsembleScore data={data.ensemble_scores[0]} />
            </div>
          )}

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            {data.monte_carlo && <MonteCarloChart data={data.monte_carlo} />}
            {data.garch && <VolatilityForecast data={data.garch} />}
            {data.hmm && (
              <div style={{ background: '#161b22', borderRadius: 8, padding: 16, border: '1px solid #30363d' }}>
                <div style={{ fontSize: 13, fontWeight: 600, color: '#e1e4e8', marginBottom: 12 }}>HMM Regime</div>
                <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
                  {(['bull', 'bear', 'sideways'] as const).map(state => {
                    const prob = state === 'bull' ? data.hmm!.prob_bull
                      : state === 'bear' ? data.hmm!.prob_bear
                      : data.hmm!.prob_sideways;
                    const color = state === 'bull' ? '#22c55e' : state === 'bear' ? '#ef4444' : '#f59e0b';
                    const isCurrent = data.hmm!.current_state === state;
                    return (
                      <div key={state} style={{
                        flex: 1, textAlign: 'center', padding: '8px 4px', borderRadius: 6,
                        background: isCurrent ? color + '22' : '#0d1117',
                        border: `1px solid ${isCurrent ? color : '#30363d'}`,
                      }}>
                        <div style={{ fontSize: 11, color: '#8b949e', textTransform: 'capitalize' }}>{state}</div>
                        <div style={{ fontSize: 16, fontWeight: 700, color }}>{((prob || 0) * 100).toFixed(0)}%</div>
                      </div>
                    );
                  })}
                </div>
                <div style={{ fontSize: 11, color: '#8b949e' }}>
                  Transition from {data.hmm.current_state}:
                  Bull {((data.hmm.trans_to_bull || 0) * 100).toFixed(0)}% |
                  Bear {((data.hmm.trans_to_bear || 0) * 100).toFixed(0)}% |
                  Sideways {((data.hmm.trans_to_sideways || 0) * 100).toFixed(0)}%
                </div>
              </div>
            )}
            {data.fama_french && <FactorExposure data={data.fama_french} />}
            {data.copula_tail_risk && <TailRiskGauge data={data.copula_tail_risk} />}
            {data.bayesian_decay && data.bayesian_decay.length > 0 && (
              <BayesianDecayChart data={data.bayesian_decay[0]} />
            )}
            {data.event_studies && data.event_studies.length > 0 && (
              <EventStudyChart events={data.event_studies} />
            )}
            {data.options_flow && <OptionsFlow data={data.options_flow} />}
          </div>
        </>
      )}
    </div>
  );
};
