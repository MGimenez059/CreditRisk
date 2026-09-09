"""Frozen selection manifest and content hashes for final evaluation."""

import hashlib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class FrozenSelection(BaseModel):
    """Validated artifact identity and partitions fixed before test scoring."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    protocol: str = "phase4-v1"
    frozen_at: str
    dataset_sha256: str
    artifact_sha256: str
    metadata_sha256: str
    evidence_sha256: str
    protocol_sha256: str
    feature_columns: list[str]
    positions: dict[str, list[int]]
    candidate: str
    calibration: str
    threshold: float = Field(ge=0, le=1)


def file_sha256(path: Path) -> str:
    """Hash exact file bytes for reproducibility and accidental-change detection."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
