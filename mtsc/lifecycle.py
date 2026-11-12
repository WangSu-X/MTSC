"""Per-request lifecycle timing for MTSC.

Tracing is opt-in: export ``MTSC_LIFECYCLE_PROFILE_FILE`` (JSONL path) before
starting vLLM, and every hook appends one line per event.  When the variable is
absent ``recorder()`` returns ``None`` and each hook is a single ``is None``
check, so the production path is unchanged.

One line looks like::

    {"ts_ns": 123, "wall_ns": 456, "pid": 7, "stack": "MTSC",
     "component": "scheduler", "role": "P", "kind": "queued",
     "request_ids": ["chatcmpl-..."], "queue_ms": 12.5, ...}

``ts_ns`` is ``time.monotonic_ns`` (durations) and ``wall_ns`` is
``time.time_ns`` (cross-process ordering), so P and D files can be merged by
request_id and, loosely, by wall clock.
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import threading
import time
from typing import Any

_STOP = object()
_MAX_EVENTS_ENV = "MTSC_LIFECYCLE_MAX_EVENTS"


class LifecycleRecorder:
    """Background JSONL writer; the request path only enqueues."""

    def __init__(self, path: str, component: str, role: str, max_events: int) -> None:
        self.path = path
        self.component = component
        self.role = role
        self._pid = os.getpid()
        self._events: queue.Queue[dict[str, Any] | object] = queue.Queue(
            maxsize=max(256, max_events)
        )
        self._dropped = 0
        self._drop_lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(
            target=self._write_events,
            name="mtsc-lifecycle",
            daemon=True,
        )
        self._thread.start()

    def record(
        self,
        kind: str,
        request_ids: list[str] | tuple[str, ...] | str | None = None,
        **fields: Any,
    ) -> None:
        if self._closed:
            return
        event: dict[str, Any] = {
            "ts_ns": time.monotonic_ns(),
            "wall_ns": time.time_ns(),
            "pid": self._pid,
            "stack": "MTSC",
            "component": self.component,
            "role": self.role,
            "kind": kind,
        }
        if request_ids:
            if isinstance(request_ids, str):
                event["request_ids"] = [request_ids]
            else:
                event["request_ids"] = list(request_ids)
        event.update(fields)
        try:
            self._events.put_nowait(event)
        except queue.Full:
            with self._drop_lock:
                self._dropped += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._events.put(_STOP, timeout=1.0)
        except queue.Full:
            return
        self._thread.join(timeout=5.0)

    def _write_events(self) -> None:
        try:
            parent = os.path.dirname(self.path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self.path, "a", encoding="utf-8", buffering=1) as output:
                batch: list[dict[str, Any]] = []
                while True:
                    try:
                        item = self._events.get(timeout=0.25)
                    except queue.Empty:
                        if batch:
                            _flush(output, batch)
                            batch.clear()
                        continue
                    if item is _STOP:
                        break
                    batch.append(item)  # type: ignore[arg-type]
                    if len(batch) >= 64:
                        _flush(output, batch)
                        batch.clear()
                if batch:
                    _flush(output, batch)
                with self._drop_lock:
                    dropped = self._dropped
                _flush(
                    output,
                    [
                        {
                            "ts_ns": time.monotonic_ns(),
                            "wall_ns": time.time_ns(),
                            "pid": self._pid,
                            "stack": "MTSC",
                            "component": self.component,
                            "role": self.role,
                            "kind": "recorder_summary",
                            "dropped_events": dropped,
                        }
                    ],
                )
        except OSError:
            return


def _flush(output: Any, events: list[dict[str, Any]]) -> None:
    for event in events:
        output.write(json.dumps(event, separators=(",", ":")) + "\n")


_LOCK = threading.Lock()
_RECORDER: LifecycleRecorder | None = None
_INITIALIZED = False


def recorder(component: str, role: str) -> LifecycleRecorder | None:
    """Return this process' recorder, or ``None`` when tracing is off."""
    global _INITIALIZED, _RECORDER
    if _INITIALIZED:
        return _RECORDER
    with _LOCK:
        if _INITIALIZED:
            return _RECORDER
        _INITIALIZED = True
        path = os.getenv("MTSC_LIFECYCLE_PROFILE_FILE", "").strip()
        if not path:
            return None
        try:
            max_events = int(os.getenv(_MAX_EVENTS_ENV, "65536"))
        except ValueError:
            max_events = 65536
        _RECORDER = LifecycleRecorder(path, component, role, max_events)
        atexit.register(_RECORDER.close)
        return _RECORDER


def now_ns() -> int:
    """Monotonic clock for durations."""
    return time.monotonic_ns()


def queue_ms(arrival_time: float | None) -> float:
    """Millis a request waited before the connector first saw it."""
    if not arrival_time:
        return -1.0
    return max(0.0, (time.time() - arrival_time) * 1000.0)


def elapsed_ms(start_ns: int) -> float:
    if start_ns <= 0:
        return -1.0
    return (time.monotonic_ns() - start_ns) / 1e6
