/** Public allowlisted contract emitted by scripts/export_demo_snapshot.py. */
export type DemoDirection = 'buy' | 'sell';
export interface DemoEvent {
  id: number;
  ticker: string;
  actor: string;
  direction: DemoDirection;
  transaction_type: string;
  direction_note: string;
  trade_date: string;
  disclosure_date: string;
  collected_at: string;
  disclosure_lag_days: number;
  source_type: 'congressional';
  source_label: string;
  source_url: string | null;
  source_portal_url: string;
  source_note: string;
  size_lower: number | null;
  size_upper: number | null;
  size_note: string;
  owner: string | null;
}
export interface DemoMonteCarlo {
  run_date: string;
  current_price: number;
  n_simulations: number;
  annual_volatility: number | null;
  horizons: { days: 30 | 90; p10: number; p25: number; p50: number; p75: number; p90: number }[];
  note: string;
}
export interface DemoGarch {
  run_date: string;
  current_vol_annual: number | null;
  historical_vol_20d: number | null;
  forecast_5d_vol: number | null;
  forecast_20d_vol: number | null;
  forecast_20d_interpretation: string | null;
  n_observations: number | null;
}
export interface DemoHmm {
  run_date: string;
  current_state: string;
  prob_bull: number | null;
  prob_bear: number | null;
  prob_sideways: number | null;
  n_observations: number | null;
}
export interface DemoFactors {
  run_date: string;
  beta_market: number | null;
  beta_smb: number | null;
  beta_hml: number | null;
  beta_rmw: number | null;
  beta_cma: number | null;
  r_squared: number | null;
}
export interface DemoCopula {
  run_date: string;
  n_observations: number | null;
  tail_dep_lower: number | null;
  tail_dep_upper: number | null;
  student_t_rho: number | null;
}
export type DemoModelKey = 'monte_carlo' | 'garch' | 'hmm' | 'factors' | 'copula';
export interface DemoTicker {
  ticker: string;
  company_name: string;
  sector: string;
  identity_source: string;
  events: DemoEvent[];
  analysis: {
    monte_carlo: DemoMonteCarlo | null;
    garch: DemoGarch | null;
    hmm: DemoHmm | null;
    factors: DemoFactors | null;
    copula: DemoCopula | null;
  };
  availability: { key: DemoModelKey; label: string; available: boolean; as_of: string | null; note: string }[];
  price_history: { available: false; points: []; note: string };
  limitations: string[];
}
export interface DemoMacro {
  as_of: string;
  regime: string | null;
  indicators: { key: string; label: string; value: number | null; unit: string; as_of: string; source_url: string; note: string }[];
  note: string;
}
export interface DemoSnapshot {
  metadata: {
    schema_version: 1;
    snapshot_id: string;
    as_of: string;
    generated_at: string;
    /** Repository HEAD at export time; uncommitted code may differ. */
    code_commit: string;
    /** Whether tracked or untracked workspace changes existed at export time. */
    code_dirty: boolean;
    /** Exact SHA-256 digest of the exporter script used to produce this file. */
    exporter_sha256: string;
    selection_note: string;
    limitations: string[];
    source_counts: { tickers: number; events: number; congressional: number; buy: number; sell: number };
    data_quality: { excluded_enrichment_rows: number; orphan_enrichment_rows: number; excluded_non_congressional_events: number };
  };
  tickers: DemoTicker[];
  macro: DemoMacro | null;
  validation: {
    generated_at: string;
    status: 'experimental';
    passed: number;
    total: number;
    gates: { key: string; label: string; passed: boolean }[];
    table_counts: Record<string, number>;
    xgboost_status: string;
    caveats: string[];
  };
}
