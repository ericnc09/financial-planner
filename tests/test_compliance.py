"""Tests for prohibited-terms scanner and disclaimer enforcement."""

from src.reporting.compliance import (
    DISCLAIMER_MARKER,
    PROHIBITED_TERMS,
    compliance_check,
    ensure_disclaimer,
)


def test_compliance_replaces_guaranteed():
    text = "This is a guaranteed winner based on our models."
    cleaned, violations = compliance_check(text)
    assert "guaranteed" not in cleaned.lower().split("disclaimer")[0]
    assert any(v["term"] == "guaranteed" for v in violations)


def test_compliance_replaces_risk_free():
    text = "A risk-free opportunity with great upside."
    cleaned, violations = compliance_check(text)
    assert "risk-free" not in cleaned.lower().split("disclaimer")[0]
    assert len(violations) >= 1


def test_compliance_replaces_sure_thing():
    text = "This pick is a sure thing for Q2."
    cleaned, violations = compliance_check(text)
    assert "sure thing" not in cleaned.lower().split("disclaimer")[0]
    assert any(v["term"] == "sure thing" for v in violations)


def test_compliance_case_insensitive():
    text = "GUARANTEED returns, and also a Guaranteed beat."
    cleaned, violations = compliance_check(text)
    pre_disclaimer = cleaned.split("Disclaimer:")[0]
    assert "guaranteed" not in pre_disclaimer.lower()
    assert len(violations) == 2


def test_compliance_adds_disclaimer_when_absent():
    text = "# Some report\n\nBuy recommendation."
    cleaned, _ = compliance_check(text)
    assert DISCLAIMER_MARKER in cleaned


def test_compliance_does_not_duplicate_disclaimer():
    text = "# Report\n\n" + DISCLAIMER_MARKER + " already here."
    cleaned, _ = compliance_check(text)
    assert cleaned.count(DISCLAIMER_MARKER) == 1


def test_ensure_disclaimer_idempotent():
    body = "Some body text."
    once = ensure_disclaimer(body)
    twice = ensure_disclaimer(once)
    assert once == twice


def test_compliance_clean_report_passes_through():
    text = "# Report\n\nThe analysis suggests potential upside."
    cleaned, violations = compliance_check(text)
    assert violations == []
    assert "potential upside" in cleaned


def test_compliance_violations_include_context():
    text = "Our analysis shows this is a guaranteed win for holders."
    _, violations = compliance_check(text)
    assert violations
    assert "context" in violations[0]
    assert len(violations[0]["context"]) > 10


def test_prohibited_terms_nonempty():
    assert len(PROHIBITED_TERMS) >= 3
