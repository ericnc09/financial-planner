"""Tests for the ModelArtifact registry."""

import json
import os
import pickle
import tempfile

import pytest

from src.models.database import init_db, get_session_factory
from src.models.registry import (
    ModelArtifact,
    build_artifact_record,
    latest_artifact,
    list_artifacts,
    record_artifact,
)


@pytest.fixture()
def session_factory(tmp_path):
    """In-memory-ish SQLite DB for each test."""
    db_path = tmp_path / "registry_test.db"
    engine = init_db(f"sqlite:///{db_path}")
    return get_session_factory(engine)


@pytest.fixture()
def fake_artifact_file(tmp_path):
    """Pickle a tiny payload to disk so SHA256/size are computable."""
    path = tmp_path / "fake_model.pkl"
    with open(path, "wb") as f:
        pickle.dump({"model": "fake", "trained_at": "2026-04-22"}, f)
    return path


def test_build_artifact_record_captures_file_hash(fake_artifact_file):
    rec = build_artifact_record(
        "test_model",
        fake_artifact_file,
        cv_auc=0.65,
        feature_cols=["a", "b", "c"],
    )
    assert rec.name == "test_model"
    assert rec.path == str(fake_artifact_file)
    assert rec.file_sha256 is not None
    assert len(rec.file_sha256) == 64
    assert rec.file_size_bytes > 0
    assert rec.cv_auc == 0.65
    assert rec.feature_cols == ["a", "b", "c"]


def test_build_artifact_record_handles_missing_file(tmp_path):
    rec = build_artifact_record(
        "missing", tmp_path / "does_not_exist.pkl", cv_auc=0.5
    )
    assert rec.file_sha256 is None
    assert rec.file_size_bytes is None


def test_build_artifact_record_captures_git_commit(fake_artifact_file):
    """In a git repo, git_commit should be a 40-char SHA."""
    rec = build_artifact_record("git_test", fake_artifact_file)
    if rec.git_commit is not None:
        assert len(rec.git_commit) == 40
        assert rec.git_dirty in {"clean", "dirty"}


def test_record_artifact_persists_to_db(session_factory, fake_artifact_file):
    session = session_factory()
    try:
        artifact = record_artifact(
            session,
            name="xgboost_signal_classifier",
            path=str(fake_artifact_file),
            cv_auc=0.62,
            cv_metric_name="cv_auc_mean",
            cv_metric_value=0.62,
            feature_cols=["pe_ratio", "momentum_30d"],
            extra={"model_type": "xgboost", "n_samples": 423},
        )
        assert artifact.id is not None
        assert artifact.name == "xgboost_signal_classifier"
        assert artifact.cv_auc == 0.62
        # JSON columns are serialized strings
        assert json.loads(artifact.feature_cols) == ["pe_ratio", "momentum_30d"]
        assert json.loads(artifact.extra_json)["model_type"] == "xgboost"
    finally:
        session.close()


def test_latest_artifact_returns_most_recent(session_factory, fake_artifact_file):
    session = session_factory()
    try:
        record_artifact(session, "model_a", str(fake_artifact_file), cv_auc=0.55)
        record_artifact(session, "model_a", str(fake_artifact_file), cv_auc=0.62)
        record_artifact(session, "model_b", str(fake_artifact_file), cv_auc=0.50)

        latest = latest_artifact(session, "model_a")
        assert latest is not None
        assert latest.cv_auc == 0.62
    finally:
        session.close()


def test_list_artifacts_filters_by_name(session_factory, fake_artifact_file):
    session = session_factory()
    try:
        record_artifact(session, "alpha", str(fake_artifact_file))
        record_artifact(session, "beta", str(fake_artifact_file))
        record_artifact(session, "alpha", str(fake_artifact_file))

        all_a = list_artifacts(session, name="alpha")
        all_any = list_artifacts(session)
        assert len(all_a) == 2
        assert len(all_any) == 3
        assert all(r.name == "alpha" for r in all_a)
    finally:
        session.close()


def test_repr_shows_short_commit(fake_artifact_file):
    rec = build_artifact_record("test", fake_artifact_file, cv_auc=0.5)
    artifact = rec.to_orm()
    s = repr(artifact)
    assert "ModelArtifact" in s
    assert "name=test" in s
