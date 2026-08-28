"""Violation detection service: crossing + scan correlation.

Layout:
    scan_store  — ScanStore + FastAPI webhook handler (Stage 4 ← external).
    rules       — Violation rules: scan-in-window match + tailgating.
    service     — ViolationService orchestrator (consumes crossing event queue
                  and writes violations to a repository).
    repository  — Postgres-backed ViolationRepository (SQLAlchemy). If no
                  DB is configured, the service uses an in-memory NoopRepository.
    evidence    — EvidenceRecorder + clip storage + disk-space / memory-budget
                  guards. Triggered by ViolationService.on_violation.
"""

from .scan_store import Scan, ScansInWindow, ScanStore, ScanWebhookHandler
from .rules import (
    CrossingRecord,
    DensityRule,
    HighDensityViolation,
    ScanSnapshot,
    Violation,
    ViolationKind,
    ViolationService,
    ZoneOccupancySnapshot,
    density_snapshots_from_zone_counts,
)
from .repository import (
    HighDensityRepository,
    InMemoryHighDensityRepository,
    InMemoryViolationRepository,
    SQLAlchemyViolationRepository,
    ViolationRepository,
)
from .evidence import (
    ClipBuilder,
    ClipStorage,
    DiskSpaceError,
    DiskSpaceGuard,
    DiskSpaceStatus,
    EvidenceRecorder,
    LiveStreamPost,
    LocalClip,
    LocalDiskClipStorage,
    MemoryBudgetExceeded,
    MemoryBudgetGuard,
    NoopClipStorage,
    StreamPost,
    SyntheticStreamPost,
)
from .models import Base, HighDensityViolationRecord, ViolationRecord  # noqa: F401

__all__ = [
    "Scan",
    "ScansInWindow",
    "ScanStore",
    "ScanWebhookHandler",
    "CrossingRecord",
    "DensityRule",
    "HighDensityViolation",
    "HighDensityRepository",
    "HighDensityViolationRecord",
    "InMemoryHighDensityRepository",
    "ScanSnapshot",
    "ZoneOccupancySnapshot",
    "Violation",
    "ViolationKind",
    "ViolationService",
    "density_snapshots_from_zone_counts",
    "ViolationRepository",
    "InMemoryViolationRepository",
    "SQLAlchemyViolationRepository",
    "ViolationRecord",
    "Base",
    # evidence
    "ClipBuilder",
    "ClipStorage",
    "DiskSpaceError",
    "DiskSpaceGuard",
    "DiskSpaceStatus",
    "EvidenceRecorder",
    "LiveStreamPost",
    "LocalClip",
    "LocalDiskClipStorage",
    "MemoryBudgetExceeded",
    "MemoryBudgetGuard",
    "NoopClipStorage",
    "StreamPost",
    "SyntheticStreamPost",
]
