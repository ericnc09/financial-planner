export interface Signal {
  id: number;
  ticker: string;
  actor: string;
  direction: 'buy' | 'sell';
  size_estimate: number | null;
  trade_date: string;
  source_type: 'congressional' | 'insider';
  signal_score: number | null;
  fundamental_score: number | null;
  macro_modifier: number | null;
  conviction: number | null;
  passes_threshold: boolean | null;
  sector: string | null;
  price_at_signal: number | null;
}

export interface MacroData {
  snapshot_date: string;
  yield_spread: number | null;
  unemployment_claims: number | null;
  cpi_yoy: number | null;
  fed_funds_rate: number | null;
  regime: 'expansion' | 'transition' | 'recession' | null;
  regime_score: number | null;
}

export interface DashboardData {
  signals: Signal[];
  macro: MacroData | null;
}
