#!/usr/bin/env python3
"""Export an allowlisted, historical portfolio demo without network or DB writes.

    python3 scripts/export_demo_snapshot.py --as-of 2026-06-22

This is a publication adapter, not an ingestion or financial-validation pipeline.
Dates are preserved; results collected after the explicit cutoff are excluded.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, time, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import tempfile
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TICKERS = ('PINS', 'CARR', 'TSM', 'NEE', 'GEHC', 'GS', 'MAR', 'AESI')
MODEL_SPECS = {
    'monte_carlo': ('monte_carlo_results', 'Price simulations', ('current_price', 'n_simulations', 'annual_volatility')),
    'garch': ('garch_forecasts', 'Volatility estimate', ('current_vol_annual', 'historical_vol_20d', 'forecast_5d_vol', 'forecast_20d_vol', 'forecast_20d_interpretation', 'n_observations')),
    'hmm': ('hmm_regime_states', 'Market regime estimate', ('current_state', 'prob_bull', 'prob_bear', 'prob_sideways', 'n_observations')),
    'factors': ('fama_french_exposures', 'Factor exposures', ('beta_market', 'beta_smb', 'beta_hml', 'beta_rmw', 'beta_cma', 'r_squared')),
    'copula': ('copula_tail_risk_results', 'Shared downside estimate', ('n_observations', 'tail_dep_lower', 'tail_dep_upper', 'student_t_rho')),
}
SECTORS = {
    'communication-services': 'Communication services', 'consumer-discretionary': 'Consumer discretionary',
    'consumer-staples': 'Consumer staples', 'energy': 'Energy', 'financials': 'Financials',
    'health-care': 'Health care', 'industrials': 'Industrials', 'information-technology': 'Information technology',
    'materials': 'Materials', 'real-estate': 'Real estate', 'utilities': 'Utilities',
}
GATE_LABELS = {
    'minimum_20d_samples': 'Minimum historical sample count',
    'ensemble_rank_ic_positive': 'Positive ensemble rank correlation',
    'conviction_rank_ic_positive': 'Positive conviction rank correlation',
    'buy_bucket_beats_hold': 'Buy-labelled group exceeds hold-labelled group',
    'sell_bucket_positive': 'Sell-labelled group has positive directional returns',
    'high_conviction_beats_other': 'Higher-conviction group exceeds other events',
    'all_claimed_components_present': 'All claimed components are present',
}
LIMITATIONS = [
    'Historical research snapshot, not live market data or investment advice. Selection is based on source identity and model coverage, not returns.',
    'Only congressional records with matching saved issuer, transaction type, and dates are included. Original filing URLs were not retained; the source link opens the provider portal.',
    'All event-linked enrichment and score joins are excluded: the saved database contains orphan enrichment rows and inconsistent ticker-sector associations.',
    'Model outputs were fitted on their displayed run dates and may postdate the disclosures. This demo does not recreate what a researcher could have known at the trade date.',
    'Price history is unavailable in the saved database. The chart uses saved simulated price quantiles, not historical observations or validated forecasts.',
    'Congressional amount bounds are omitted because the saved equal bounds come from a provider value and do not establish the original filing range.',
]
VALIDATION_CAVEATS = [
    'This is the saved project audit, not a new validation run. Its counts describe the full research database, not the selected demo universe.',
    'Historical evaluation uses latest scores and may include information unavailable at entry time; missing-provider returns and horizon conventions also require correction.',
    'Passing a gate does not establish predictive accuracy or investment performance. Return, win-rate, Sharpe, and recommendation claims are excluded from this demo.',
]
ALLOWED_URL_HOSTS = {'www.capitoltrades.com', 'fred.stlouisfed.org'}
MACRO_SPECS = (
    ('yield_spread_10y2y', '10-year minus 2-year yield', 'percentage points', 'T10Y2Y'),
    ('unemployment_claims', 'Initial unemployment claims', 'claims', 'ICSA'),
    ('fed_funds_rate', 'Federal funds rate', '%', 'FEDFUNDS'),
    ('cpi_yoy', 'Consumer price inflation', '% year over year', 'CPIAUCSL'),
)


def timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{label}: a timestamp is required')
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as exc:
        raise ValueError(f'{label}: malformed timestamp {value!r}') from exc
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def iso(value: object, label: str) -> str:
    return timestamp(value, label).isoformat().replace('+00:00', 'Z')


def cutoff(as_of: str) -> datetime:
    try:
        parsed = date.fromisoformat(as_of)
    except (ValueError, TypeError) as exc:
        raise ValueError('as_of must be an explicit YYYY-MM-DD date') from exc
    if parsed.isoformat() != as_of:
        raise ValueError('as_of must be an explicit YYYY-MM-DD date')
    return datetime.combine(parsed, time.max, timezone.utc)


def rows_before(conn: sqlite3.Connection, table: str, field: str, bound: datetime) -> list[dict]:
    # All identifiers are internal constants. Parse dates instead of relying on
    # mixed SQLite timestamp text formats or silently dropping malformed dates.
    rows = []
    for record in conn.execute(f'SELECT * FROM {table}'):
        row = dict(record)
        observed = timestamp(row[field], f'{table}.{field}')
        created = timestamp(row.get('created_at', row[field]), f'{table}.created_at')
        if observed <= bound and created <= bound:
            rows.append(row)
    return rows


def latest(rows: list[dict], field: str) -> dict | None:
    return max(rows, key=lambda r: (timestamp(r[field], field), r['id']), default=None)


def model_result(key: str, row: dict | None) -> dict | None:
    if row is None:
        return None
    result = {'run_date': iso(row['run_date'], 'model.run_date')}
    for field in MODEL_SPECS[key][2]:
        result[field] = row[field]
    if key == 'monte_carlo':
        result['horizons'] = [
            {'days': days, **{q: row[f'h{days}_{q}'] for q in ('p10', 'p25', 'p50', 'p75', 'p90')}}
            for days in (30, 90)
        ]
        result['note'] = 'Saved simulation price quantiles in USD. Baseline is the model input price at run time; this is not a price-history chart or a validated forecast.'
    return result


def public_event(row: dict) -> tuple[dict, dict]:
    try:
        raw = json.loads(row['raw_payload'])
    except (ValueError, TypeError) as exc:
        raise ValueError(f'Event {row["id"]}: missing or malformed source payload') from exc
    if not isinstance(raw, dict) or not isinstance(raw.get('issuer'), dict):
        raise ValueError(f'Event {row["id"]}: missing issuer identity')
    issuer = raw['issuer']
    if issuer.get('issuerTicker') != f'{row["ticker"]}:US':
        raise ValueError(f'Event {row["id"]}: source ticker does not match event')
    if raw.get('txType') != row['direction']:
        raise ValueError(f'Event {row["id"]}: source direction does not match event')
    trade = timestamp(row['trade_date'], 'trade_date')
    disclosure = timestamp(row['disclosure_date'], 'disclosure_date')
    if timestamp(raw.get('txDate'), 'source.txDate').date() != trade.date():
        raise ValueError(f'Event {row["id"]}: source trade date does not match event')
    if timestamp(raw.get('pubDate'), 'source.pubDate') != disclosure:
        raise ValueError(f'Event {row["id"]}: source disclosure date does not match event')
    if disclosure < trade:
        raise ValueError(f'Event {row["id"]}: disclosure predates trade')
    event = {
        'id': row['id'], 'ticker': row['ticker'], 'actor': row['actor'],
        'direction': row['direction'], 'transaction_type': raw['txType'],
        'direction_note': 'Direction reported by the disclosure provider. This may concern assets owned by a spouse; it does not establish the reason for a trade.',
        'trade_date': iso(row['trade_date'], 'trade_date'),
        'disclosure_date': iso(row['disclosure_date'], 'disclosure_date'),
        'collected_at': iso(row['created_at'], 'created_at'),
        'disclosure_lag_days': (disclosure.date() - trade.date()).days,
        'source_type': 'congressional', 'source_label': 'Capitol Trades — saved congressional disclosure',
        'source_url': None, 'source_portal_url': 'https://www.capitoltrades.com/trades',
        'source_note': 'Provider portal, not a direct filing link. The original disclosure URL was not retained in the saved record.',
        'size_lower': None, 'size_upper': None,
        'size_note': 'Original filing amount range unavailable. The saved provider value is not presented as an exact trade amount.',
        'owner': raw.get('owner') if raw.get('owner') in ('self', 'spouse', 'joint', 'dependent') else None,
    }
    identity = {'company_name': issuer.get('issuerName'), 'sector': SECTORS.get(issuer.get('sector'), 'Unavailable')}
    return event, identity


def snapshot_identity(snapshot: dict) -> str:
    stable = {**snapshot, 'metadata': {k: v for k, v in snapshot['metadata'].items() if k not in ('snapshot_id', 'generated_at')}}
    canonical = json.dumps(stable, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return 'demo-' + hashlib.sha256(canonical).hexdigest()[:16]


def build_snapshot(database: Path, audit_path: Path, as_of: str, tickers=DEFAULT_TICKERS, *, generated_at: str | None = None, code_commit: str | None = None, code_dirty: bool | None = None) -> dict:
    bound = cutoff(as_of)
    if not tickers or len(tickers) != len(set(tickers)) or any(not re.fullmatch(r'[A-Z][A-Z0-9.-]{0,9}', t) for t in tickers):
        raise ValueError('Provide unique uppercase ticker symbols')
    audit = json.loads(audit_path.read_text())
    audit_date = timestamp(audit.get('generated_at'), 'audit.generated_at')
    if audit_date > bound:
        raise ValueError('Audit was generated after the selected as_of boundary')
    conn = sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute('PRAGMA query_only = ON')
        events = rows_before(conn, 'smart_money_events', 'trade_date', bound)
        eligible = []
        for row in events:
            # A missing disclosure is not evidence of a publication date.
            if row['source_type'] == 'congressional':
                if timestamp(row['disclosure_date'], 'event.disclosure_date') <= bound:
                    eligible.append(row)
        model_rows = {key: rows_before(conn, spec[0], 'run_date', bound) for key, spec in MODEL_SPECS.items()}
        entries = []
        for symbol in tickers:
            chosen = sorted((e for e in eligible if e['ticker'] == symbol), key=lambda e: (timestamp(e['disclosure_date'], 'disclosure_date'), e['id']), reverse=True)
            if not chosen:
                raise ValueError(f'{symbol}: no eligible congressional disclosures before {as_of}')
            normalized = [public_event(e) for e in chosen]
            identities = {json.dumps(identity, sort_keys=True) for _, identity in normalized}
            if len(identities) != 1:
                raise ValueError(f'{symbol}: inconsistent saved issuer identity')
            identity = normalized[0][1]
            analyses = {key: model_result(key, latest([r for r in rows if r['ticker'] == symbol], 'run_date')) for key, rows in model_rows.items()}
            entries.append({
                'ticker': symbol, **identity,
                'identity_source': 'Issuer name and sector from the matching saved congressional source payload; enrichment joins excluded.',
                'events': [e for e, _ in normalized], 'analysis': analyses,
                'availability': [{'key': key, 'label': spec[1], 'available': analyses[key] is not None,
                                  'as_of': analyses[key]['run_date'] if analyses[key] else None,
                                  'note': 'Saved experimental model output.' if analyses[key] else 'Unavailable in this snapshot.'}
                                 for key, spec in MODEL_SPECS.items()],
                'price_history': {'available': False, 'points': [], 'note': 'No local historical price series is stored. Simulations are shown separately and are not observed prices.'},
                'limitations': ['Model coverage counts stored components, not prediction confidence.', 'Disclosure publication dates can be later than transaction dates.'],
            })
        macro_row = latest(rows_before(conn, 'macro_snapshots', 'snapshot_date', bound), 'snapshot_date')
        macro = None
        if macro_row:
            macro_date = iso(macro_row['snapshot_date'], 'macro.snapshot_date')
            macro = {'as_of': macro_date, 'regime': macro_row['regime'],
                     'indicators': [{'key': field, 'label': label, 'value': macro_row[field], 'unit': unit,
                                     'as_of': macro_date, 'source_url': 'https://fred.stlouisfed.org/series/' + series,
                                     'note': 'Saved pipeline snapshot date; source observation and release dates were not retained.'}
                                    for field, label, unit, series in MACRO_SPECS],
                     'note': 'Historical pipeline values. The regime is a rule-based research label; individual indicator observation dates may be older than this snapshot.'}
        gate_data = audit['validation_20d']['acceptance_gates']
        gates = [{'key': key, 'label': GATE_LABELS.get(key, key.replace('_', ' ').capitalize()), 'passed': value} for key, value in gate_data['gates'].items()]
        exported_events = [event for entry in entries for event in entry['events']]
        directions = Counter(event['direction'] for event in exported_events)
        if code_commit is None:
            code_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        if code_dirty is None:
            code_dirty = bool(subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=normal'], cwd=ROOT, text=True).strip())
        snapshot = {
            'metadata': {'schema_version': 1, 'snapshot_id': '', 'as_of': as_of,
                         'generated_at': iso(generated_at or datetime.now(timezone.utc).isoformat(), 'generated_at'),
                         'code_commit': code_commit,
                         'code_dirty': code_dirty,
                         'exporter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         'selection_note': 'Eight congressional-disclosure examples selected for matching source identities, sector variety, both trade directions, and saved model coverage. GEHC retains incomplete coverage. This is a curated research workflow, not a representative or performance-ranked sample.' if tuple(tickers) == DEFAULT_TICKERS else 'Explicitly selected congressional tickers with matching saved source identities; selection is not a performance ranking.',
                         'limitations': LIMITATIONS,
                         'source_counts': {'tickers': len(entries), 'events': len(exported_events), 'congressional': len(exported_events), 'buy': directions['buy'], 'sell': directions['sell']},
                         'data_quality': {'excluded_enrichment_rows': conn.execute('SELECT COUNT(*) FROM enrichments').fetchone()[0],
                                          'orphan_enrichment_rows': conn.execute('SELECT COUNT(*) FROM enrichments e LEFT JOIN smart_money_events s ON s.id=e.event_id WHERE s.id IS NULL').fetchone()[0],
                                          'excluded_non_congressional_events': sum(e['source_type'] != 'congressional' for e in events)}},
            'tickers': entries, 'macro': macro,
            'validation': {'generated_at': iso(audit['generated_at'], 'audit.generated_at'), 'status': 'experimental',
                           'passed': gate_data['passed'], 'total': gate_data['total'], 'gates': gates,
                           'table_counts': audit['table_counts'], 'xgboost_status': audit['xgboost']['status'], 'caveats': VALIDATION_CAVEATS},
        }
        snapshot['metadata']['snapshot_id'] = snapshot_identity(snapshot)
        validate_snapshot(snapshot)
        return snapshot
    finally:
        conn.close()


def require_keys(value: object, keys: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f'{label}: missing or unexpected fields')
    return value


def string(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{label}: nonempty string required')


def number(value: object, label: str, minimum=None, maximum=None, nullable=False) -> None:
    if value is None and nullable:
        return
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{label}: finite number required')
    if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise ValueError(f'{label}: out of bounds')


def validate_snapshot(snapshot: dict) -> None:
    """Reject unsafe links, incomplete schema, future/malformed dates, and bad joins."""
    require_keys(snapshot, {'metadata', 'tickers', 'macro', 'validation'}, 'snapshot')
    meta = require_keys(snapshot['metadata'], {'schema_version', 'snapshot_id', 'as_of', 'generated_at', 'code_commit', 'code_dirty', 'exporter_sha256', 'selection_note', 'limitations', 'source_counts', 'data_quality'}, 'metadata')
    if type(meta['schema_version']) is not int or meta['schema_version'] != 1:
        raise ValueError('Unsupported schema_version')
    bound = cutoff(meta['as_of'])
    timestamp(meta['generated_at'], 'generated_at')
    for key in ('snapshot_id', 'code_commit', 'selection_note'):
        string(meta[key], key)
    if not re.fullmatch(r'[a-f0-9]{40}', meta['code_commit']):
        raise ValueError('code_commit must be a full Git SHA')
    if type(meta['code_dirty']) is not bool:
        raise ValueError('code_dirty must be a boolean')
    if not isinstance(meta['exporter_sha256'], str) or not re.fullmatch(r'[a-f0-9]{64}', meta['exporter_sha256']):
        raise ValueError('exporter_sha256 must be a SHA-256 hex digest')

    def before(value, label):
        parsed = timestamp(value, label)
        if parsed > bound:
            raise ValueError(f'{label}: after as_of boundary')
        return parsed

    def walk(value, path='snapshot'):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f'{path}: non-finite number')
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {'raw_payload', 'database', 'model_path'}:
                    raise ValueError(f'{path}: private field {key}')
                if key.endswith('_url') and item is not None:
                    url = urlsplit(item)
                    if url.scheme != 'https' or url.hostname not in ALLOWED_URL_HOSTS or url.username or url.password or url.query or url.fragment or url.port:
                        raise ValueError(f'{path}.{key}: unsafe or unapproved public URL')
                walk(item, f'{path}.{key}')
        elif isinstance(value, list):
            for item in value:
                walk(item, path)
    walk(snapshot)
    if not isinstance(snapshot['tickers'], list) or not snapshot['tickers']:
        raise ValueError('At least one ticker is required')
    tickers, ids, events = set(), set(), []
    for item in snapshot['tickers']:
        require_keys(item, {'ticker', 'company_name', 'sector', 'identity_source', 'events', 'analysis', 'availability', 'price_history', 'limitations'}, 'ticker')
        if not isinstance(item['ticker'], str) or not re.fullmatch(r'[A-Z][A-Z0-9.-]{0,9}', item['ticker']) or item['ticker'] in tickers:
            raise ValueError('Invalid or duplicate ticker')
        tickers.add(item['ticker'])
        for key in ('company_name', 'sector', 'identity_source'):
            string(item[key], key)
        if not isinstance(item['events'], list) or not item['events']:
            raise ValueError('Each ticker requires an event')
        for event in item['events']:
            require_keys(event, {'id', 'ticker', 'actor', 'direction', 'transaction_type', 'direction_note', 'trade_date', 'disclosure_date', 'collected_at', 'disclosure_lag_days', 'source_type', 'source_label', 'source_url', 'source_portal_url', 'source_note', 'size_lower', 'size_upper', 'size_note', 'owner'}, 'event')
            if type(event['id']) is not int or event['id'] < 1 or event['id'] in ids or event['ticker'] != item['ticker']:
                raise ValueError('Event referential integrity failure')
            ids.add(event['id']); events.append(event)
            for key in ('actor', 'direction_note', 'source_label', 'source_note', 'source_portal_url', 'size_note'):
                string(event[key], key)
            if event['source_type'] != 'congressional' or event['direction'] not in ('buy', 'sell') or event['transaction_type'] != event['direction']:
                raise ValueError('Invalid event source or direction')
            trade = before(event['trade_date'], 'trade_date'); disclosure = before(event['disclosure_date'], 'disclosure_date')
            collected = before(event['collected_at'], 'collected_at')
            if trade > disclosure or disclosure > collected or event['disclosure_lag_days'] != (disclosure.date() - trade.date()).days:
                raise ValueError('Inconsistent event date order or disclosure lag')
            if event['source_url'] is not None or event['size_lower'] is not None or event['size_upper'] is not None:
                raise ValueError('Original filing URL and amount range were not verified')
            if event['owner'] not in (None, 'self', 'spouse', 'joint', 'dependent'):
                raise ValueError('Invalid ownership label')
        require_keys(item['analysis'], set(MODEL_SPECS), 'analysis')
        if len(item['availability']) != len(MODEL_SPECS):
            raise ValueError('Missing availability metadata')
        seen_availability = set()
        for availability in item['availability']:
            require_keys(availability, {'key', 'label', 'available', 'as_of', 'note'}, 'availability')
            key = availability['key']
            if key not in MODEL_SPECS or key in seen_availability:
                raise ValueError('Invalid availability model key')
            seen_availability.add(key)
            row = item['analysis'][key]
            if type(availability['available']) is not bool or availability['available'] != (row is not None) or availability['as_of'] != (row['run_date'] if row else None):
                raise ValueError('Inconsistent model availability')
            for field in ('label', 'note'):
                string(availability[field], field)
        for key, row in item['analysis'].items():
            if row is None:
                continue
            fields = {'run_date', *MODEL_SPECS[key][2]}
            if key == 'monte_carlo': fields |= {'horizons', 'note'}
            require_keys(row, fields, f'analysis.{key}')
            before(row['run_date'], f'{key}.run_date')
            for field in MODEL_SPECS[key][2]:
                if field in ('current_state', 'forecast_20d_interpretation'):
                    if row[field] is not None: string(row[field], field)
                else:
                    number(row[field], field, nullable=field not in ('current_price', 'n_simulations'))
                    if field.startswith(('prob_', 'tail_dep_')) or field == 'r_squared':
                        number(row[field], field, 0, 1, nullable=True)
                    if field in ('n_observations', 'n_simulations') and row[field] is not None:
                        if type(row[field]) is not int or row[field] < 1: raise ValueError(f'{field}: positive integer required')
            if key == 'hmm' and row['current_state'] not in ('bull', 'bear', 'sideways'):
                raise ValueError('Invalid HMM state')
            if key == 'monte_carlo':
                number(row['current_price'], 'current_price', minimum=0.0000001)
                if not isinstance(row['horizons'], list) or [h['days'] for h in row['horizons']] != [30, 90]:
                    raise ValueError('Expected 30 and 90 day simulation horizons')
                for horizon in row['horizons']:
                    require_keys(horizon, {'days', 'p10', 'p25', 'p50', 'p75', 'p90'}, 'horizon')
                    values = [horizon[p] for p in ('p10', 'p25', 'p50', 'p75', 'p90')]
                    for value in values: number(value, 'simulation quantile', minimum=0)
                    if values != sorted(values): raise ValueError('Simulation quantiles must be ordered')
        require_keys(item['price_history'], {'available', 'points', 'note'}, 'price_history')
        if item['price_history']['available'] is not False or item['price_history']['points'] != []:
            raise ValueError('No verified price history is included in schema version 1')
    counts = {'tickers': len(tickers), 'events': len(events), 'congressional': len(events), 'buy': sum(e['direction'] == 'buy' for e in events), 'sell': sum(e['direction'] == 'sell' for e in events)}
    if meta['source_counts'] != counts: raise ValueError('Source counts do not match selected events')
    require_keys(meta['data_quality'], {'excluded_enrichment_rows', 'orphan_enrichment_rows', 'excluded_non_congressional_events'}, 'data_quality')
    for value in meta['data_quality'].values():
        if type(value) is not int or value < 0: raise ValueError('Invalid data-quality count')
    macro = snapshot['macro']
    if macro is not None:
        require_keys(macro, {'as_of', 'regime', 'indicators', 'note'}, 'macro')
        before(macro['as_of'], 'macro.as_of')
        expected_sources = {field: 'https://fred.stlouisfed.org/series/' + series for field, _, _, series in MACRO_SPECS}
        if len(macro['indicators']) != len(expected_sources) or {i['key'] for i in macro['indicators']} != set(expected_sources):
            raise ValueError('Missing or duplicate macro indicators')
        for indicator in macro['indicators']:
            require_keys(indicator, {'key', 'label', 'value', 'unit', 'as_of', 'source_url', 'note'}, 'macro indicator')
            before(indicator['as_of'], 'indicator.as_of')
            number(indicator['value'], 'indicator.value', nullable=True)
            if indicator['source_url'] != expected_sources[indicator['key']]:
                raise ValueError('Macro source does not match the collected FRED series')
    audit = require_keys(snapshot['validation'], {'generated_at', 'status', 'passed', 'total', 'gates', 'table_counts', 'xgboost_status', 'caveats'}, 'validation')
    before(audit['generated_at'], 'validation.generated_at')
    if audit['status'] != 'experimental': raise ValueError('Validation status must be experimental')
    seen_gates = set()
    for gate in audit['gates']:
        require_keys(gate, {'key', 'label', 'passed'}, 'gate')
        string(gate['key'], 'gate.key'); string(gate['label'], 'gate.label')
        if type(gate['passed']) is not bool or gate['key'] in seen_gates: raise ValueError('Invalid or duplicate audit gate')
        seen_gates.add(gate['key'])
    if audit['passed'] != sum(g['passed'] for g in audit['gates']) or audit['total'] != len(audit['gates']) or audit['total'] == 0:
        raise ValueError('Audit gate counts are inconsistent')
    if not isinstance(audit['table_counts'], dict) or not audit['table_counts'] or any(type(v) is not int or v < 0 for v in audit['table_counts'].values()):
        raise ValueError('Invalid audit table counts')
    for strings in [meta['limitations'], audit['caveats'], *[t['limitations'] for t in snapshot['tickers']]]:
        if not isinstance(strings, list) or not strings: raise ValueError('Limitations must be present')
        for value in strings: string(value, 'limitation')
    if meta['snapshot_id'] != snapshot_identity(snapshot):
        raise ValueError('Snapshot identity does not match content')


def export_snapshot(snapshot: dict, output: Path) -> None:
    """Validate before writing; an invalid refresh cannot replace a working file."""
    validate_snapshot(snapshot)
    payload = json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=output.parent, prefix='.snapshot-', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary and temporary.exists(): temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--as-of', required=True, help='Explicit inclusive UTC date, YYYY-MM-DD')
    parser.add_argument('--database', type=Path, default=ROOT / 'smart_money.db')
    parser.add_argument('--audit', type=Path, default=ROOT / 'reports/model_status_report.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'dashboard/public/demo/snapshot.json')
    parser.add_argument('--tickers', nargs='+', default=DEFAULT_TICKERS)
    args = parser.parse_args()
    try:
        snapshot = build_snapshot(args.database, args.audit, args.as_of, args.tickers)
        export_snapshot(snapshot, args.output)
    except (ValueError, KeyError, TypeError, OSError, sqlite3.Error) as exc:
        parser.exit(1, f'Snapshot not published: {exc}\n')
    print(f'Exported {snapshot["metadata"]["snapshot_id"]}: {len(snapshot["tickers"])} tickers / {snapshot["metadata"]["source_counts"]["events"]} events as of {args.as_of} to {args.output}')


if __name__ == '__main__':
    main()
