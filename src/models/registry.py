"""
Model artifact registry.

Every persisted model (xgboost_model.pkl, isotonic calibrator, etc.) gets a
`ModelArtifact` row recording what was serving when. Without this, when live
results drift you can't tell which model produced them — only that something
changed.

Usage:
    from src.models.registry import ModelArtifact, record_artifact

    artifact = record_artifact(
        session,
        name="xgboost_signal_classifier",
        path="src/analysis/xgboost_model.pkl",
        cv_auc=0.62,
        feature_cols=["pe_ratio", "momentum_30d"],
        extra={"model_type": "xgboost", "n_samples": 423},
    )
    # → returns the persisted ModelArtifact, with id, git_commit, file_sha256
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Session

from src.models.database import Base


# ── ORM table ────────────────────────────────────────────────────────────────


class ModelArtifact(Base):
    """
    Registry row for one persisted model artifact.

    Tracks: name, path, when it was created, what git commit produced it,
    a file content hash (sha256), the headline CV metric, and the feature
    schema. `extra_json` captures anything model-specific.
    """

    __tablename__ = "model_artifacts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, index=True)
    path = Column(String(500), nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Provenance
    git_commit = Column(String(40), nullable=True)
    git_dirty = Column(String(10), nullable=True)  # "clean" / "dirty" / "unknown"
    file_sha256 = Column(String(64), nullable=True)
    file_size_bytes = Column(Integer, nullable=True)

    # Headline metric (whatever's most relevant for this model type)
    cv_auc = Column(Float, nullable=True)
    cv_metric_name = Column(String(40), nullable=True)
    cv_metric_value = Column(Float, nullable=True)

    # Feature schema (JSON list)
    feature_cols = Column(Text, nullable=True)

    # Free-form metadata (n_samples, model_type, hyperparameters, etc.)
    extra_json = Column(Text, nullable=True)

    __table_args__ = (
        Index("ix_model_artifacts_name_created", "name", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<ModelArtifact id={self.id} name={self.name} "
            f"git={(self.git_commit or '?')[:8]} cv_auc={self.cv_auc}>"
        )


# ── Provenance helpers ───────────────────────────────────────────────────────


def _git_commit() -> tuple[str | None, str]:
    """Return (commit_sha, dirty_flag). Falls back to (None, 'unknown') off-repo."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        diff = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return sha, ("dirty" if diff else "clean")
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None, "unknown"


def _file_sha256(path: str | os.PathLike) -> tuple[str | None, int | None]:
    """Stream-hash a file. Returns (hex digest, size_bytes) or (None, None)."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None, None
    h = hashlib.sha256()
    size = 0
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


# ── Lightweight dataclass mirror (for in-memory / non-DB usage) ─────────────


@dataclass
class ModelArtifactRecord:
    """
    In-memory mirror of ModelArtifact — useful when you want to capture
    artifact metadata before deciding whether to persist (e.g. during a
    training experiment).
    """

    name: str
    path: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    git_commit: str | None = None
    git_dirty: str | None = None
    file_sha256: str | None = None
    file_size_bytes: int | None = None
    cv_auc: float | None = None
    cv_metric_name: str | None = None
    cv_metric_value: float | None = None
    feature_cols: list[str] | None = None
    extra: dict | None = None

    def to_orm(self) -> ModelArtifact:
        """Convert to a SQLAlchemy ModelArtifact ready to add to a session."""
        return ModelArtifact(
            name=self.name,
            path=self.path,
            created_at=self.created_at,
            git_commit=self.git_commit,
            git_dirty=self.git_dirty,
            file_sha256=self.file_sha256,
            file_size_bytes=self.file_size_bytes,
            cv_auc=self.cv_auc,
            cv_metric_name=self.cv_metric_name,
            cv_metric_value=self.cv_metric_value,
            feature_cols=(
                json.dumps(self.feature_cols) if self.feature_cols is not None else None
            ),
            extra_json=(json.dumps(self.extra) if self.extra is not None else None),
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        return d


# ── Public API ───────────────────────────────────────────────────────────────


def build_artifact_record(
    name: str,
    path: str | os.PathLike,
    *,
    cv_auc: float | None = None,
    cv_metric_name: str | None = None,
    cv_metric_value: float | None = None,
    feature_cols: list[str] | None = None,
    extra: dict | None = None,
) -> ModelArtifactRecord:
    """
    Build a ModelArtifactRecord from disk + git, without touching the DB.

    Useful for tests, dry-runs, or when you want to inspect provenance
    before persisting.
    """
    sha, dirty = _git_commit()
    file_hash, size = _file_sha256(path)
    return ModelArtifactRecord(
        name=name,
        path=str(path),
        git_commit=sha,
        git_dirty=dirty,
        file_sha256=file_hash,
        file_size_bytes=size,
        cv_auc=cv_auc,
        cv_metric_name=cv_metric_name,
        cv_metric_value=cv_metric_value,
        feature_cols=list(feature_cols) if feature_cols is not None else None,
        extra=dict(extra) if extra is not None else None,
    )


def record_artifact(
    session: Session,
    name: str,
    path: str | os.PathLike,
    *,
    cv_auc: float | None = None,
    cv_metric_name: str | None = None,
    cv_metric_value: float | None = None,
    feature_cols: list[str] | None = None,
    extra: dict | None = None,
) -> ModelArtifact:
    """
    Build a ModelArtifactRecord from disk + git, persist via `session`,
    and return the ORM row (with id assigned).
    """
    record = build_artifact_record(
        name,
        path,
        cv_auc=cv_auc,
        cv_metric_name=cv_metric_name,
        cv_metric_value=cv_metric_value,
        feature_cols=feature_cols,
        extra=extra,
    )
    artifact = record.to_orm()
    session.add(artifact)
    session.commit()
    session.refresh(artifact)
    return artifact


def latest_artifact(session: Session, name: str) -> ModelArtifact | None:
    """Return the most-recently-recorded artifact with this name, or None."""
    return (
        session.query(ModelArtifact)
        .filter(ModelArtifact.name == name)
        .order_by(ModelArtifact.created_at.desc())
        .first()
    )


def list_artifacts(
    session: Session,
    name: str | None = None,
    limit: int = 20,
) -> list[ModelArtifact]:
    """List recent artifacts, optionally filtered by name."""
    q = session.query(ModelArtifact).order_by(ModelArtifact.created_at.desc())
    if name:
        q = q.filter(ModelArtifact.name == name)
    return q.limit(limit).all()
