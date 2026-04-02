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

export interface ExtendedMacroData {
  snapshot_date: string;
  vix: number | null;
  consumer_sentiment: number | null;
  money_supply_m2: number | null;
  housing_starts: number | null;
  industrial_production: number | null;
  put_call_ratio: number | null;
}

export interface MonteCarloHorizon {
  percentiles: { p10: number; p25: number; p50: number; p75: number; p90: number };
  probability_of_profit: number;
  expected_return: number;
  value_at_risk_95: number;
}

export interface MonteCarloData {
  ticker: string;
  run_date: string;
  current_price: number;
  annual_drift: number | null;
  annual_volatility: number | null;
  n_simulations: number;
  horizons: Record<string, MonteCarloHorizon>;
}

export interface HMMData {
  ticker: string;
  run_date: string;
  current_state: 'bull' | 'bear' | 'sideways';
  prob_bull: number | null;
  prob_bear: number | null;
  prob_sideways: number | null;
  trans_to_bull: number | null;
  trans_to_bear: number | null;
  trans_to_sideways: number | null;
}

export interface GARCHData {
  ticker: string;
  run_date: string;
  persistence: number | null;
  current_vol_annual: number | null;
  long_run_vol_annual: number | null;
  historical_vol_20d: number | null;
  historical_vol_60d: number | null;
  forecast_5d_vol: number | null;
  forecast_5d_ratio: number | null;
  forecast_5d_interpretation: string | null;
  forecast_20d_vol: number | null;
  forecast_20d_ratio: number | null;
  forecast_20d_interpretation: string | null;
}

export interface FamaFrenchData {
  ticker: string;
  run_date: string;
  alpha_annual: number | null;
  beta_market: number | null;
  beta_smb: number | null;
  beta_hml: number | null;
  beta_rmw: number | null;
  beta_cma: number | null;
  r_squared: number | null;
}

export interface EventStudyData {
  ticker: string;
  event_id: number;
  direction: 'buy' | 'sell';
  source_type: 'insider' | 'congressional' | null;
  car_1d: number | null;
  car_5d: number | null;
  car_10d: number | null;
  car_20d: number | null;
  t_statistic: number | null;
  p_value: number | null;
  is_significant: boolean | null;
  daily_cars: number[];
}

export interface TickerAnalysis {
  ticker: string;
  monte_carlo: MonteCarloData | null;
  hmm: HMMData | null;
  garch: GARCHData | null;
  fama_french: FamaFrenchData | null;
  event_studies: EventStudyData[] | null;
}

export interface DashboardData {
  signals: Signal[];
  macro: MacroData | null;
  extended_macro: ExtendedMacroData | null;
}
