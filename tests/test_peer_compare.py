"""Tests for peer comparison (relative-valuation deltas)."""

from src.analysis.peer_compare import (
    SECTOR_PEER_UNIVERSE,
    _compute_deltas,
    _safe_float,
    _sector_medians,
    format_peer_table,
)


def test_sector_peer_universe_has_all_main_sectors():
    expected = {
        "Technology", "Communication Services", "Consumer Cyclical",
        "Consumer Defensive", "Financial Services", "Healthcare",
        "Industrials", "Energy", "Utilities", "Basic Materials",
        "Real Estate",
    }
    assert expected.issubset(SECTOR_PEER_UNIVERSE.keys())


def test_sector_peer_universe_all_nonempty():
    for sector, peers in SECTOR_PEER_UNIVERSE.items():
        assert len(peers) >= 4, f"sector {sector} has too few peers"


def test_safe_float_handles_nan_and_none():
    assert _safe_float(None) is None
    assert _safe_float("abc") is None
    assert _safe_float(float("nan")) is None
    assert _safe_float(12.345) == 12.345


def test_sector_medians_computes_per_field():
    peers = [
        {"pe": 20, "ps": 5, "ev_ebitda": 15, "gross_margin": 0.4, "operating_margin": 0.2},
        {"pe": 25, "ps": 6, "ev_ebitda": 18, "gross_margin": 0.5, "operating_margin": 0.25},
        {"pe": 30, "ps": 7, "ev_ebitda": 20, "gross_margin": 0.6, "operating_margin": 0.3},
    ]
    m = _sector_medians(peers)
    assert m["pe"] == 25
    assert m["ps"] == 6
    assert m["gross_margin"] == 0.5


def test_sector_medians_skips_none_values():
    peers = [
        {"pe": 20, "ps": None, "ev_ebitda": 15, "gross_margin": 0.4, "operating_margin": None},
        {"pe": 30, "ps": 6, "ev_ebitda": 20, "gross_margin": 0.6, "operating_margin": 0.3},
    ]
    m = _sector_medians(peers)
    assert m["pe"] == 25
    assert m["ps"] == 6  # only one value, median is that value
    assert m["operating_margin"] == 0.3


def test_compute_deltas_pct_and_bps():
    target = {"pe": 30, "ps": 6, "ev_ebitda": 18,
              "gross_margin": 0.6, "operating_margin": 0.25}
    medians = {"pe": 20, "ps": 4, "ev_ebitda": 15,
               "gross_margin": 0.4, "operating_margin": 0.20}
    d = _compute_deltas(target, medians)
    assert d["pe_delta_pct"] == 50.0  # (30-20)/20 * 100
    assert d["ps_delta_pct"] == 50.0
    assert d["gross_margin_delta_bps"] == 2000  # (0.6-0.4) * 10000
    assert d["operating_margin_delta_bps"] == 500


def test_compute_deltas_handles_zero_median():
    target = {"pe": 25}
    medians = {"pe": 0}
    d = _compute_deltas(target, medians)
    assert d["pe_delta_pct"] is None


def test_format_peer_table_renders_all_peers():
    comparison = {
        "ticker": "AAPL",
        "sector": "Technology",
        "target": {
            "ticker": "AAPL", "market_cap": 3e12,
            "pe": 30, "ps": 7, "ev_ebitda": 22,
            "gross_margin": 0.45, "operating_margin": 0.3,
        },
        "peers": [
            {
                "ticker": "MSFT", "market_cap": 3e12,
                "pe": 32, "ps": 12, "ev_ebitda": 24,
                "gross_margin": 0.7, "operating_margin": 0.4,
            }
        ],
        "sector_medians": {
            "pe": 32, "ps": 12, "ev_ebitda": 24,
            "gross_margin": 0.7, "operating_margin": 0.4,
        },
        "deltas": {
            "pe_delta_pct": -6.3, "ps_delta_pct": -41.7, "ev_ebitda_delta_pct": -8.3,
            "gross_margin_delta_bps": -2500, "operating_margin_delta_bps": -1000,
        },
        "n_peers": 1,
    }
    out = format_peer_table(comparison)
    assert "AAPL" in out
    assert "MSFT" in out
    assert "Sector median" in out
    assert "P/E" in out
    assert "premium" in out.lower() or "discount" in out.lower()


def test_format_peer_table_empty_input():
    assert "No peer comparison" in format_peer_table({})
