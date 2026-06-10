import React, { useEffect, useState } from 'react';

interface TopPick {
  ticker: string;
  score: number;
  conviction: number | null;
  ensemble_score: number | null;
  ensemble_recommendation: string | null;
  mc_prob_profit_30d: number | null;
  n_signals: number;
  n_distinct_actors: number;
  sector: string | null;
  price_at_signal: number | null;
  latest_actor: string | null;
  latest_trade_date: string | null;
  source_type: string | null;
  runner_ups: { ticker: string; score: number }[];
  lookback_days: number;
}

interface Props {
  onTickerClick?: (ticker: string) => void;
}

export const TopPickCard: React.FC<Props> = ({ onTickerClick }) => {
  const [pick, setPick] = useState<TopPick | null>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    // Prefer this week's signals; fall back to a wider window (labeled via
    // lookback_days in the response) so the card still renders on quiet weeks.
    fetch('/api/top-pick')
      .then(r => (r.ok ? r.json() : null))
      .then(p => p ?? fetch('/api/top-pick?lookback_days=30').then(r => (r.ok ? r.json() : null)))
      .then(setPick)
      .catch(() => setPick(null))
      .finally(() => setLoaded(true));
  }, []);

  if (!loaded || !pick) return null;

  const Metric: React.FC<{ label: string; value: string }> = ({ label, value }) => (
    <div>
      <div style={{ fontSize: 10, color: '#8b949e' }}>{label}</div>
      <div style={{ fontSize: 15, fontWeight: 600, color: '#e1e4e8' }}>{value}</div>
    </div>
  );

  return (
    <div style={{
      background: 'linear-gradient(135deg, #1c1917 0%, #161b22 60%)',
      borderRadius: 12, padding: 20, border: '1px solid #f59e0b55', marginBottom: 20,
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
        <div>
          <div style={{ fontSize: 11, color: '#fbbf24', fontWeight: 600, letterSpacing: 1, marginBottom: 4 }}>
            🌅 MORNING TOP PICK
          </div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 10 }}>
            <span
              onClick={() => onTickerClick?.(pick.ticker)}
              style={{ fontSize: 28, fontWeight: 700, color: '#e1e4e8', cursor: 'pointer', textDecoration: 'underline' }}
            >
              {pick.ticker}
            </span>
            {pick.sector && <span style={{ fontSize: 12, color: '#8b949e' }}>{pick.sector}</span>}
          </div>
          <div style={{ fontSize: 11, color: '#8b949e', marginTop: 4 }}>
            Latest: {pick.latest_actor} ({pick.source_type}) · {pick.latest_trade_date?.slice(0, 10)}
            {' · '}{pick.n_signals} buy(s) from {pick.n_distinct_actors} actor(s) in {pick.lookback_days}d
          </div>
        </div>
        <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
          <Metric label="Composite" value={pick.score.toFixed(2)} />
          {pick.conviction !== null && <Metric label="Conviction" value={`${(pick.conviction * 100).toFixed(0)}%`} />}
          {pick.ensemble_score !== null && <Metric label="Ensemble" value={`${pick.ensemble_score.toFixed(0)}/100`} />}
          {pick.mc_prob_profit_30d !== null && <Metric label="P(profit 30d)" value={`${(pick.mc_prob_profit_30d * 100).toFixed(0)}%`} />}
        </div>
      </div>
      {pick.runner_ups.length > 0 && (
        <div style={{ fontSize: 11, color: '#8b949e', marginTop: 10 }}>
          Runners-up: {pick.runner_ups.map(r => `${r.ticker} (${r.score.toFixed(2)})`).join(' · ')}
        </div>
      )}
      <div style={{ fontSize: 10, color: '#6b7280', marginTop: 8 }}>
        Research signal, not financial advice.
      </div>
    </div>
  );
};
