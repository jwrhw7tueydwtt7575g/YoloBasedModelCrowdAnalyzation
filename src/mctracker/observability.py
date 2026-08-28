"""Metrics and structured logging hooks.

Two parallel surfaces:

* ``Metrics`` — in-process counters + histograms, thread-safe under a single
  lock. Always available (no extra deps). Tests can read values without any
  Prometheus client installed.
* ``start_prometheus_endpoint(port)`` — opt-in HTTP server, only imported if
  ``prometheus_client`` is available. If not installed, the call is a no-op
  with a logged warning; the in-process counters still work.

The point of having a pure-Python fallback is that the benchmark and unit
tests must be able to assert metric values without installing the heavy
``prometheus_client`` extra.

Histogram buckets are fixed-width in milliseconds to keep the implementation
small. Eight buckets per the plan: 1, 2.5, 5, 10, 25, 50, 100, 250 ms. Larger
observations land in ``+Inf``.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# Latency histogram bucket upper-bounds in seconds.
LATENCY_BUCKETS_S: Tuple[float, ...] = (
    0.001, 0.0025, 0.005, 0.010, 0.025, 0.050, 0.100, 0.250,
)


class _Counter:
    """Integer counter, optionally labelled."""

    __slots__ = ("_value", "_labels")

    def __init__(self) -> None:
        self._value: int = 0
        # label-tuple -> count (e.g. ("detect",) -> 3)
        self._labels: Dict[Tuple[str, ...], int] = {}

    def inc(self, n: int = 1, labels: Tuple[str, ...] = ()) -> None:
        self._value += n
        self._labels[labels] = self._labels.get(labels, 0) + n

    @property
    def value(self) -> int:
        return self._value

    def labelled(self) -> Dict[Tuple[str, ...], int]:
        return dict(self._labels)


class _Histogram:
    """Fixed-bucket histogram in seconds. Stores observations under their
    bucket bound, plus a count and sum."""

    __slots__ = ("_buckets", "_count", "_sum", "_labels")

    def __init__(self, buckets: Tuple[float, ...] = LATENCY_BUCKETS_S) -> None:
        # bucket upper-bound -> cumulative count
        self._buckets: Dict[float, int] = {b: 0 for b in buckets}
        self._count: int = 0
        self._sum: float = 0.0
        self._labels: Dict[Tuple[str, ...], Dict[float, int]] = {}

    def observe(self, value: float, labels: Tuple[str, ...] = ()) -> None:
        self._count += 1
        self._sum += float(value)
        for ub in self._buckets:
            if value <= ub:
                self._buckets[ub] += 1
        if labels:
            lab = self._labels.setdefault(labels, {b: 0 for b in self._buckets})
            lab_count = self._labels[labels]
            for ub in lab_count:
                if value <= ub:
                    lab_count[ub] += 1
            self._labels[labels] = lab_count

    def snapshot(self) -> Dict[str, float]:
        """Return plain counters (count, sum, cumulative bucket counts)."""
        out = {"count": float(self._count), "sum": float(self._sum)}
        for ub in self._buckets:
            out[f"le_{ub}"] = float(self._buckets[ub])
        return out

    @property
    def count(self) -> int:
        return self._count


class Metrics:
    """The single global metrics registry for a pipeline run.

    Thread-safe under one lock. The lock is only held for the duration of a
    counter inc or histogram observe — both O(1) operations — so contention
    is negligible at any realistic frame rate.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Counters
        self.frames_processed_total = _Counter()
        self.frames_dropped_total = _Counter()
        self.stream_reconnects_total = _Counter()
        self.stream_build_failures_total = _Counter()
        self.id_switches_total = _Counter()
        self.violations_total = _Counter()
        self.stage_failures_total = _Counter()
        # Histograms (one per pipeline stage + one for end-to-end)
        self.detect_seconds = _Histogram()
        self.track_seconds = _Histogram()
        self.zone_seconds = _Histogram()
        self.tripwire_seconds = _Histogram()

    # ----- recorders -----

    def inc_frame(self, stream_id: str, dropped: bool = False) -> None:
        with self._lock:
            if dropped:
                self.frames_dropped_total.inc(labels=(stream_id,))
            else:
                self.frames_processed_total.inc(labels=(stream_id,))

    def inc_reconnect(self, stream_id: str) -> None:
        with self._lock:
            self.stream_reconnects_total.inc(labels=(stream_id,))

    def inc_build_failure(self, stream_id: str) -> None:
        with self._lock:
            self.stream_build_failures_total.inc(labels=(stream_id,))

    def inc_id_switch(self, stream_id: str) -> None:
        with self._lock:
            self.id_switches_total.inc(labels=(stream_id,))

    def inc_violation(self, kind: str) -> None:
        with self._lock:
            self.violations_total.inc(labels=(kind,))

    def inc_stage_failure(self, stream_id: str, stage: str) -> None:
        with self._lock:
            self.stage_failures_total.inc(labels=(stream_id, stage))

    def observe(self, stage: str, seconds: float, stream_id: str = "") -> None:
        hist: _Histogram
        if stage == "detect":
            hist = self.detect_seconds
        elif stage == "track":
            hist = self.track_seconds
        elif stage == "zone":
            hist = self.zone_seconds
        elif stage == "tripwire":
            hist = self.tripwire_seconds
        else:
            return
        with self._lock:
            hist.observe(seconds, labels=(stream_id,) if stream_id else ())

    # ----- text rendering -----

    def to_prometheus_text(self) -> str:
        """Render the registry as Prometheus text format (0.0.4)."""
        lines: List[str] = []
        counters = [
            ("mctracker_frames_processed_total", self.frames_processed_total),
            ("mctracker_frames_dropped_total", self.frames_dropped_total),
            ("mctracker_stream_reconnects_total", self.stream_reconnects_total),
            ("mctracker_stream_build_failures_total", self.stream_build_failures_total),
            ("mctracker_id_switches_total", self.id_switches_total),
            ("mctracker_violations_total", self.violations_total),
            ("mctracker_stage_failures_total", self.stage_failures_total),
        ]
        with self._lock:
            for name, c in counters:
                lines.append(f"# TYPE {name} counter")
                if not c._labels:
                    lines.append(f"{name} {c.value}")
                else:
                    for labels, val in c._labels.items():
                        rendered = ",".join(
                            f'{k}="{v}"' for k, v in zip(("stream_id", "stage", "kind"), labels)
                        )
                        lines.append(f"{name}{{{rendered}}} {val}")
            histograms = [
                ("mctracker_detect_seconds", self.detect_seconds),
                ("mctracker_track_seconds", self.track_seconds),
                ("mctracker_zone_seconds", self.zone_seconds),
                ("mctracker_tripwire_seconds", self.tripwire_seconds),
            ]
            for name, h in histograms:
                lines.append(f"# TYPE {name} histogram")
                lines.append(f"{name}_count {h._count}")
                lines.append(f"{name}_sum {h._sum:.6f}")
                for ub in h._buckets:
                    lines.append(f"{name}_bucket{{le=\"{ub}\"}} {h._buckets[ub]}")
                lines.append(f"{name}_bucket{{le=\"+Inf\"}} {h._count}")
        return "\n".join(lines) + "\n"

    # ----- snapshot for tests -----

    def snapshot(self) -> Dict[str, object]:
        """Return a JSON-serializable snapshot of all counters and histograms."""
        with self._lock:
            return {
                "counters": {
                    "frames_processed_total": self.frames_processed_total.value,
                    "frames_dropped_total": self.frames_dropped_total.value,
                    "stream_reconnects_total": self.stream_reconnects_total.value,
                    "stream_build_failures_total": self.stream_build_failures_total.value,
                    "id_switches_total": self.id_switches_total.value,
                    "violations_total": self.violations_total.value,
                    "stage_failures_total": self.stage_failures_total.value,
                },
                "histograms": {
                    "detect_count": self.detect_seconds.count,
                    "track_count": self.track_seconds.count,
                    "zone_count": self.zone_seconds.count,
                    "tripwire_count": self.tripwire_seconds.count,
                },
            }


# Global singleton for convenience. Modules can `from .observability import METRICS`
# and call METRICS.inc_frame(...) without plumbing a metrics instance through
# their constructor. Tests instantiate a fresh `Metrics()` and monkeypatch
# ``mctracker.observability.METRICS`` if they want to observe isolated values.
METRICS = Metrics()


def reset_metrics(metrics: Optional[Metrics] = None) -> Metrics:
    """Replace the global METRICS instance. Used by tests and pipeline setup."""
    global METRICS
    METRICS = metrics or Metrics()
    return METRICS


def start_prometheus_endpoint(port: int) -> bool:
    """Optionally expose ``/metrics`` on the given TCP port.

    Returns ``True`` if the endpoint is live, ``False`` if ``prometheus_client``
    is not installed (the in-process counters continue to work). Multiple
    calls on the same port are idempotent: only the first call starts a
    server; subsequent calls are logged and ignored.
    """
    if port <= 0:
        return False
    try:
        from prometheus_client import start_http_server  # type: ignore[import]
    except Exception:
        log.warning(
            "prometheus_client not installed; /metrics endpoint disabled. "
            "Install with `pip install -e .[prometheus]`."
        )
        return False
    start_http_server(port)
    log.info("prometheus /metrics endpoint listening on :%d", port)
    return True


# Convenience helpers (Stage 5 wants a `time_stage` context manager).
class time_stage:
    """Context manager that observes wall-clock time into the right histogram.

        with time_stage("detect", stream_id="cam0"):
            ...
    """

    __slots__ = ("_stage", "_stream_id", "_t0")

    def __init__(self, stage: str, stream_id: str = "") -> None:
        self._stage = stage
        self._stream_id = stream_id
        self._t0 = 0.0

    def __enter__(self) -> "time_stage":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        elapsed = time.perf_counter() - self._t0
        METRICS.observe(self._stage, elapsed, self._stream_id)