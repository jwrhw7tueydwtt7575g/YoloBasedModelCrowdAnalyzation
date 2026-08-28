"""Tests for ScanStore and webhook ingestion."""

from __future__ import annotations

import pytest

from mctracker.violations import (
    Scan,
    ScanStore,
    ScanWebhookHandler,
)


def test_add_and_query_window_basic():
    store = ScanStore()
    s1 = Scan(zone_id="A", timestamp=100.0, identity="alice")
    s2 = Scan(zone_id="A", timestamp=105.0, identity="bob")
    s3 = Scan(zone_id="A", timestamp=130.0, identity="carol")
    store.add(s1)
    store.add(s2)
    store.add(s3)

    # Query around t=100 with ±5s window.
    out = store.query_window("A", 100.0, 5.0)
    assert {s.identity for s in out.scans} == {"alice", "bob"}

    # Query around t=105 with ±1s.
    out = store.query_window("A", 105.0, 1.0)
    assert {s.identity for s in out.scans} == {"bob"}

    # Query around t=130 with ±5s → only carol.
    out = store.query_window("A", 130.0, 5.0)
    assert {s.identity for s in out.scans} == {"carol"}


def test_query_window_exact_boundary():
    """A scan exactly at ±window is included (inclusive bounds)."""
    store = ScanStore()
    store.add(Scan(zone_id="A", timestamp=100.0, identity="alice"))
    # Crossing at t=110, window=10 → scan at 100 should match.
    out = store.query_window("A", 110.0, 10.0)
    assert out.matched
    assert out.scans[0].identity == "alice"


def test_per_zone_isolation():
    store = ScanStore()
    store.add(Scan(zone_id="A", timestamp=100.0, identity="alice"))
    store.add(Scan(zone_id="B", timestamp=100.0, identity="bob"))
    out = store.query_window("A", 100.0, 5.0)
    assert len(out) == 1
    assert out.scans[0].identity == "alice"


def test_empty_zone_returns_empty():
    store = ScanStore()
    assert not store.query_window("nowhere", 100.0, 5.0).matched


def test_window_must_be_non_negative():
    store = ScanStore()
    with pytest.raises(ValueError):
        store.query_window("A", 100.0, -1.0)


def test_webhook_ingest_valid_payload():
    store = ScanStore()
    handler = ScanWebhookHandler(store)
    scan = handler.ingest({"zone_id": "A", "timestamp": 100.0, "identity": "alice"})
    assert scan.zone_id == "A"
    assert scan.timestamp == 100.0
    assert scan.identity == "alice"
    assert store.scan_count_by_zone("A") == 1


def test_webhook_ingest_iso_timestamp():
    """ISO 8601 timestamps should be accepted and converted to epoch seconds."""
    store = ScanStore()
    handler = ScanWebhookHandler(store)
    # Use a known second: 2024-01-01T00:00:00Z → 1704067200
    payload = {
        "zone_id": "A",
        "timestamp": "2024-01-01T00:00:00Z",
        "identity": "alice",
    }
    scan = handler.ingest(payload)
    assert abs(scan.timestamp - 1704067200.0) < 1.0


def test_webhook_ingest_rejects_missing_fields():
    store = ScanStore()
    handler = ScanWebhookHandler(store)
    with pytest.raises(ValueError):
        handler.ingest({"zone_id": "A", "timestamp": 1.0})  # no identity
    with pytest.raises(ValueError):
        handler.ingest({"zone_id": "A", "identity": "x"})  # no timestamp


def test_webhook_build_app_exposes_post_scans():
    """FastAPI integration (skipped if FastAPI isn't installed)."""
    fastapi = pytest.importorskip("fastapi")
    httpx = pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    store = ScanStore()
    handler = ScanWebhookHandler(store)
    app = handler.build_app()
    client = TestClient(app)

    resp = client.post("/scans", json={"zone_id": "A", "timestamp": 1.0, "identity": "x"})
    assert resp.status_code == 200
    assert resp.json() == {"zone_id": "A", "timestamp": 1.0, "identity": "x"}

    # And bad payload → 400.
    resp = client.post("/scans", json={"zone_id": "A", "timestamp": "not-a-number"})
    assert resp.status_code == 400

    # /health should be fine.
    assert client.get("/health").json() == {"status": "ok"}
