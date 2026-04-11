import type { DashboardData, Signal, MacroData, TickerAnalysis, HMMData, ExtendedMacroData, MeanVarianceData, EnsembleScoreData, BacktestResult, PriceHistoryData } from '../types';

const BASE = '/api';

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`);
  return res.json();
}

export const api = {
  getDashboard: () => fetchJson<DashboardData>('/dashboard'),
  getSignals: (days = 14, minConviction?: number) => {
    const params = new URLSearchParams({ days: String(days) });
    if (minConviction !== undefined) params.set('min_conviction', String(minConviction));
    return fetchJson<Signal[]>(`/signals?${params}`);
  },
  getMacro: () => fetchJson<MacroData>('/macro'),
  getMacroHistory: (days = 90) => fetchJson<MacroData[]>(`/macro/history?days=${days}`),
  getExtendedMacro: () => fetchJson<ExtendedMacroData>('/macro/extended'),
  getTickerAnalysis: (ticker: string) => fetchJson<TickerAnalysis>(`/analysis/${ticker}`),
  getAllHMM: () => fetchJson<HMMData[]>('/analysis/hmm/all'),
  getEventStudySummary: () => fetchJson<any>('/analysis/event-study/summary'),
  getMeanVariance: () => fetchJson<MeanVarianceData>('/analysis/mean-variance'),
  getAllEnsembleScores: () => fetchJson<EnsembleScoreData[]>('/analysis/ensemble/all'),
  triggerPipeline: () => fetch(`${BASE}/pipeline/run`, { method: 'POST' }).then(r => r.json()),
  runBacktest: (startDate: string, endDate: string, threshold: number): Promise<BacktestResult> =>
    fetch(`${BASE}/backtest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ start_date: startDate, end_date: endDate, conviction_threshold: threshold }),
    }).then(r => r.json()),
  getPriceHistory: (ticker: string, days = 365) =>
    fetchJson<PriceHistoryData>(`/prices/${ticker}?days=${days}`),
};
