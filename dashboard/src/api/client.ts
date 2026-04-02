import type { DashboardData, Signal, MacroData, TickerAnalysis, HMMData, ExtendedMacroData, MeanVarianceData, EnsembleScoreData } from '../types';

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
};
