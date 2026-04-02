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

export interface CopulaTailRiskData {
  ticker: string;
  run_date: string;
  gaussian_rho: number | null;
  student_t_rho: number | null;
  student_t_nu: number | null;
  tail_dep_lower: number | null;
  tail_dep_upper: number | null;
  joint_crash_prob: number | null;
  tail_dep_ratio: number | null;
  var_95: number | null;
  var_99: number | null;
  cvar_95: number | null;
  cvar_99: number | null;
  conditional_var_95: number | null;
  conditional_cvar_95: number | null;
  tail_risk_score: number | null;
}

export interface BayesianDecayData {
  ticker: string;
  event_id: number;
  direction: 'buy' | 'sell';
  total_car: number | null;
  posterior_half_life: number | null;
  entry_window_days: number | null;
  exit_window_days: number | null;
  annualized_ir: number | null;
  decay_quality: string | null;
  signal_strength_5d: number | null;
  signal_strength_20d: number | null;
}

export interface MeanVarianceData {
  run_date: string;
  n_assets: number;
  tickers: string[];
  max_sharpe: { weights: Record<string, number>; expected_return: number; volatility: number; sharpe_ratio: number };
  min_variance: { weights: Record<string, number>; expected_return: number; volatility: number };
  equal_weight: { expected_return: number; volatility: number; sharpe_ratio: number };
  efficient_frontier: { return: number; volatility: number }[];
  risk_contribution: Record<string, number>;
}

export interface EnsembleScoreData {
  ticker: string;
  event_id: number;
  direction: 'buy' | 'sell';
  total_score: number;
  confidence: number | null;
  recommendation: string | null;
  n_models: number | null;
  components: Record<string, number>;
}

export interface TickerAnalysis {
  ticker: string;
  monte_carlo: MonteCarloData | null;
  hmm: HMMData | null;
  garch: GARCHData | null;
  fama_french: FamaFrenchData | null;
  copula_tail_risk: CopulaTailRiskData | null;
  event_studies: EventStudyData[] | null;
  bayesian_decay: BayesianDecayData[] | null;
  ensemble_scores: EnsembleScoreData[] | null;
}

export interface DashboardData {
  signals: Signal[];
  macro: MacroData | null;
  extended_macro: ExtendedMacroData | null;
}
