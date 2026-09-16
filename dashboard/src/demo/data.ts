import type { DemoSnapshot } from './types';

let pending: Promise<DemoSnapshot> | undefined;

export function loadSnapshot(): Promise<DemoSnapshot> {
  if (!pending) {
    pending = fetch(`${import.meta.env.BASE_URL}demo/snapshot.json`)
      .then(async response => {
        if (!response.ok) throw new Error('The historical dataset could not be loaded.');
        const data = await response.json() as DemoSnapshot;
        if (data.metadata?.schema_version !== 1 || !Array.isArray(data.tickers) || !data.tickers.length
          || !data.metadata.as_of || !data.validation || data.tickers.some(ticker => !ticker.events?.length)) {
          throw new Error('This dataset is not compatible with the demo.');
        }
        return data;
      })
      .catch(error => { pending = undefined; throw error; });
  }
  return pending;
}

export function formatDate(value: string | null | undefined, short = false): string {
  if (!value) return 'Not recorded';
  const date = new Date(`${value.slice(0, 10)}T12:00:00Z`);
  if (!Number.isFinite(date.getTime())) return 'Not recorded';
  return new Intl.DateTimeFormat('en-CA', {
    day: 'numeric', month: short ? 'short' : 'long', year: 'numeric', timeZone: 'UTC',
  }).format(date);
}

export const percent = (value: number | null | undefined, digits = 1) => value == null
  ? 'Unavailable' : `${(value * 100).toFixed(digits)}%`;
export const money = (value: number) => new Intl.NumberFormat('en-US', {
  style: 'currency', currency: 'USD', maximumFractionDigits: 2,
}).format(value);

export function downloadSnapshot(snapshot: DemoSnapshot) {
  const blob = new Blob([JSON.stringify(snapshot, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `finalyse-${snapshot.metadata.as_of.slice(0, 10)}.json`;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
