"""Contract and publication safety checks for the historical demo exporter."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import sqlite3

import pytest

from scripts.export_demo_snapshot import (
    MODEL_SPECS, build_snapshot, export_snapshot, snapshot_identity, validate_snapshot,
)


@pytest.fixture()
def saved_inputs(tmp_path):
    database = tmp_path / 'research.sqlite'
    conn = sqlite3.connect(database)
    conn.executescript('''
        CREATE TABLE smart_money_events (
          id INTEGER PRIMARY KEY, ticker TEXT, actor TEXT, direction TEXT,
          trade_date TEXT, disclosure_date TEXT, source_type TEXT,
          raw_payload TEXT, created_at TEXT);
        CREATE TABLE enrichments (id INTEGER PRIMARY KEY, event_id INTEGER, sector TEXT);
        CREATE TABLE macro_snapshots (id INTEGER PRIMARY KEY, snapshot_date TEXT,
          created_at TEXT, regime TEXT, yield_spread_10y2y REAL,
          unemployment_claims REAL, fed_funds_rate REAL, cpi_yoy REAL);
    ''')
    for key, (table, _, fields) in MODEL_SPECS.items():
        columns = ['id INTEGER PRIMARY KEY', 'ticker TEXT', 'run_date TEXT', 'created_at TEXT']
        columns.extend(f'{field} {"INTEGER" if field in ("n_simulations", "n_observations") else "TEXT" if field in ("current_state", "forecast_20d_interpretation") else "REAL"}' for field in fields)
        if key == 'monte_carlo':
            columns.extend(f'h{days}_{quantile} REAL' for days in (30, 90) for quantile in ('p10', 'p25', 'p50', 'p75', 'p90'))
        conn.execute(f'CREATE TABLE {table} ({",".join(columns)})')
    raw = {'issuer': {'issuerTicker': 'TEST:US', 'issuerName': 'Example Issuer', 'sector': 'financials'},
           'txType': 'buy', 'txDate': '2026-01-02', 'pubDate': '2026-01-05T12:00:00Z',
           'owner': 'spouse', 'politician': {'dob': '1990-01-01'}, 'comment': 'DO NOT EXPORT'}
    conn.execute('INSERT INTO smart_money_events VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)',
                 ('TEST', 'Example Actor', 'buy', '2026-01-02', '2026-01-05 12:00:00', 'congressional', json.dumps(raw), '2026-01-05T12:30:00Z'))
    conn.executemany('INSERT INTO enrichments VALUES (?, ?, ?)', [(1, 1, 'Wrong sector'), (2, 999, 'Orphan')])
    conn.execute("INSERT INTO macro_snapshots VALUES (1, '2026-01-10T12:00:00Z', '2026-01-10T12:00:00Z', 'transition', 0.4, 215000, 3.63, NULL)")
    columns = [x[1] for x in conn.execute('PRAGMA table_info(monte_carlo_results)')]
    for index, run_date, baseline in [(1, '2026-01-10T12:00:00Z', 100), (2, '2026-02-10T12:00:00Z', 999)]:
        values = {'id': index, 'ticker': 'TEST', 'run_date': run_date, 'created_at': run_date,
                  'current_price': baseline, 'n_simulations': 10000, 'annual_volatility': .2}
        values.update({f'h{days}_{q}': baseline + offset for days in (30, 90) for q, offset in zip(('p10', 'p25', 'p50', 'p75', 'p90'), (-20, -10, 0, 10, 20))})
        conn.execute(f'INSERT INTO monte_carlo_results VALUES ({",".join("?" for _ in columns)})', [values[c] for c in columns])
    conn.commit(); conn.close()
    audit = tmp_path / 'audit.json'
    audit.write_text(json.dumps({'generated_at': '2026-01-20T10:00:00Z', 'table_counts': {'events': 1},
        'validation_20d': {'acceptance_gates': {'passed': 1, 'total': 2, 'gates': {'minimum_20d_samples': True, 'all_claimed_components_present': False}}},
        'xgboost': {'status': 'not_trained', 'model_path': '/private/should-not-export.pkl'}}))
    return database, audit


def build(saved_inputs, **kwargs):
    return build_snapshot(*saved_inputs, '2026-01-31', ('TEST',), code_commit='a' * 40,
                          generated_at=kwargs.pop('generated_at', '2026-09-15T12:00:00Z'), code_dirty=False, **kwargs)


def resign(snapshot):
    snapshot['metadata']['snapshot_id'] = snapshot_identity(snapshot)
    return snapshot


def test_export_is_read_only_and_preserves_dated_evidence(saved_inputs):
    database, _ = saved_inputs
    before = database.read_bytes()
    snapshot = build(saved_inputs)
    assert database.read_bytes() == before
    ticker = snapshot['tickers'][0]
    assert ticker['sector'] == 'Financials'  # Identity from matching payload, never enrichment.
    assert ticker['analysis']['monte_carlo']['current_price'] == 100  # Feb result excluded.
    assert ticker['analysis']['monte_carlo']['run_date'] == '2026-01-10T12:00:00Z'
    assert ticker['events'][0]['trade_date'] == '2026-01-02T00:00:00Z'
    assert ticker['events'][0]['disclosure_lag_days'] == 3
    assert ticker['analysis']['factors'] is None
    assert ticker['price_history']['available'] is False
    assert snapshot['metadata']['data_quality']['orphan_enrichment_rows'] == 1
    serialized = json.dumps(snapshot)
    for excluded in ('raw_payload', 'politician', '1990-01-01', 'DO NOT EXPORT', 'should-not-export'):
        assert excluded not in serialized


def test_identity_is_stable_across_export_time(saved_inputs):
    first = build(saved_inputs)
    second = build(saved_inputs, generated_at='2026-09-16T14:00:00Z')
    assert first['metadata']['snapshot_id'] == second['metadata']['snapshot_id']
    changed = deepcopy(first)
    changed['tickers'][0]['company_name'] = 'Changed issuer'
    assert snapshot_identity(changed) != first['metadata']['snapshot_id']


def test_exporter_provenance_distinguishes_base_commit_from_exact_script(saved_inputs):
    snapshot = build(saved_inputs)
    exporter = Path(__file__).resolve().parents[1] / 'scripts/export_demo_snapshot.py'
    assert snapshot['metadata']['code_commit'] == 'a' * 40
    assert snapshot['metadata']['code_dirty'] is False
    assert snapshot['metadata']['exporter_sha256'] == hashlib.sha256(exporter.read_bytes()).hexdigest()


def test_macro_sources_match_the_collector_series(saved_inputs):
    indicators = build(saved_inputs)['macro']['indicators']
    assert {i['key']: i['source_url'].split('/')[-1] for i in indicators} == {
        'yield_spread_10y2y': 'T10Y2Y', 'unemployment_claims': 'ICSA',
        'fed_funds_rate': 'FEDFUNDS', 'cpi_yoy': 'CPIAUCSL',
    }


@pytest.mark.parametrize('mutate, message', [
    (lambda s: s['tickers'][0]['events'][0].update(ticker='OTHER'), 'referential integrity'),
    (lambda s: s['tickers'][0]['events'][0].update(disclosure_date='yesterday'), 'malformed timestamp'),
    (lambda s: s['tickers'][0]['analysis']['monte_carlo'].update(run_date='2026-02-01T00:00:00Z'), 'availability|after as_of'),
    (lambda s: s['tickers'][0]['events'][0].pop('actor'), 'missing or unexpected'),
    (lambda s: s['tickers'][0]['events'][0].update(source_portal_url='https://evil.example/filing'), 'unsafe'),
    (lambda s: s['tickers'][0]['events'][0].update(source_portal_url='javascript:alert(1)'), 'unsafe'),
    (lambda s: s['tickers'][0]['analysis']['monte_carlo']['horizons'][0].update(p50=500), 'ordered'),
    (lambda s: s['metadata']['source_counts'].update(events=99), 'Source counts'),
    (lambda s: s['validation'].update(passed=2), 'gate counts'),
    (lambda s: s['tickers'][0].update(price_history={'available': True, 'points': [], 'note': 'invented'}), 'No verified price history'),
    (lambda s: s['metadata'].update(exporter_sha256='not-a-hash'), 'exporter_sha256'),
    (lambda s: s['metadata'].update(code_dirty='false'), 'code_dirty'),
    (lambda s: s['macro']['indicators'][2].update(source_url='https://fred.stlouisfed.org/series/DFF'), 'collected FRED series'),
])
def test_invalid_contract_rejected(saved_inputs, mutate, message):
    snapshot = build(saved_inputs)
    mutate(snapshot)
    resign(snapshot)
    with pytest.raises(ValueError, match=message):
        validate_snapshot(snapshot)


def test_failed_refresh_preserves_last_working_file(saved_inputs, tmp_path):
    snapshot = build(saved_inputs)
    output = tmp_path / 'public' / 'snapshot.json'
    export_snapshot(snapshot, output)
    original = output.read_bytes()
    snapshot['tickers'][0]['analysis']['monte_carlo']['horizons'][0]['p50'] = float('nan')
    with pytest.raises(ValueError, match='non-finite'):
        export_snapshot(snapshot, output)
    assert output.read_bytes() == original
    assert list(output.parent.glob('.snapshot-*.tmp')) == []


@pytest.mark.parametrize('column, value, message', [
    ('run_date', 'bad-date', 'malformed timestamp'),
    ('created_at', 'bad-date', 'malformed timestamp'),
])
def test_malformed_saved_dates_fail_closed(saved_inputs, column, value, message):
    with sqlite3.connect(saved_inputs[0]) as conn:
        conn.execute(f'UPDATE monte_carlo_results SET {column}=? WHERE id=1', (value,))
    with pytest.raises(ValueError, match=message):
        build(saved_inputs)


def test_collection_after_cutoff_excludes_model(saved_inputs):
    with sqlite3.connect(saved_inputs[0]) as conn:
        conn.execute("UPDATE monte_carlo_results SET created_at='2026-02-01T00:00:00Z' WHERE id=1")
    assert build(saved_inputs)['tickers'][0]['analysis']['monte_carlo'] is None


def test_source_identity_mismatch_fails_closed(saved_inputs):
    with sqlite3.connect(saved_inputs[0]) as conn:
        raw = json.loads(conn.execute('SELECT raw_payload FROM smart_money_events').fetchone()[0])
        raw['issuer']['issuerTicker'] = 'OTHER:US'
        conn.execute('UPDATE smart_money_events SET raw_payload=?', (json.dumps(raw),))
    with pytest.raises(ValueError, match='source ticker'):
        build(saved_inputs)


def test_future_audit_and_missing_input_fail_without_output(saved_inputs, tmp_path):
    database, audit_path = saved_inputs
    audit = json.loads(audit_path.read_text())
    audit['generated_at'] = '2026-02-02T00:00:00Z'
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match='Audit was generated after'):
        build(saved_inputs)
    missing = tmp_path / 'missing.sqlite'
    audit['generated_at'] = '2026-01-20T00:00:00Z'
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(sqlite3.OperationalError):
        build_snapshot(missing, audit_path, '2026-01-31', ('TEST',), code_commit='a' * 40)
    assert not missing.exists()


def test_bundled_snapshot_is_valid_and_contains_real_missing_coverage():
    snapshot = json.loads((Path(__file__).resolve().parents[1] / 'dashboard/public/demo/snapshot.json').read_text())
    validate_snapshot(snapshot)
    assert snapshot['metadata']['source_counts']['tickers'] == 8
    assert snapshot['metadata']['source_counts']['events'] == 9
    assert len({entry['sector'] for entry in snapshot['tickers']}) == 8
    assert {event['direction'] for entry in snapshot['tickers'] for event in entry['events']} == {'buy', 'sell'}
    gehc = next(entry for entry in snapshot['tickers'] if entry['ticker'] == 'GEHC')
    assert gehc['analysis']['factors'] is None
    assert gehc['analysis']['copula'] is None
