"""Remote KV cache contract and Mooncake Store implementation."""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import math
import os
import socket
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any, Protocol

import torch
import zmq
from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed import (
    get_dcp_group,
    get_pcp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import rdma_utils
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
    get_mooncake_dp_engine_index,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.coordinator import (
    ExternalCachedBlockPool,
    MooncakeStoreCoordinator,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    ChunkedTokenDatabase,
    KeyMetadata,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.utils.network_utils import get_ip, make_zmq_socket
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.v1.core.kv_cache_utils import BlockHash, resolve_kv_cache_block_sizes

from .utils import (
    BlockIds,
    KVLayoutAdapter,
    cache_tensors,
    is_npu_platform,
    npu_kv_nz_enabled,
    npu_registration_regions,
)


class KVReadyEvent(Protocol):
    """Device completion signal; waited on by I/O workers, not the caller."""

    def synchronize(self) -> None: ...


@dataclass(frozen=True)
class LoadEvent:
    """Load [start_load, end_load) into already allocated local blocks.

    block_ids describes the full request prefix, grouped by KV cache group.
    block_hashes contains prefix hashes through end_load. Both bounds are
    token offsets. The pool derives each group's chunk/block mapping from
    its configured layout.
    """

    request_id: str
    block_ids: BlockIds
    block_hashes: tuple[BlockHash, ...]
    start_load: int
    end_load: int


@dataclass(frozen=True)
class SaveEvent:
    """Save complete cache chunks in [start_save, end_save).

    Both bounds are token offsets into the full request prefix described by
    block_hashes and block_ids. Repeated saves for the same request may arrive
    before earlier saves finish; the pool queues and coalesces pending work.

    The caller keeps source blocks valid and unchanged until completion or
    preempt() returns. ready_event=None means the data is already ready.
    """

    request_id: str
    block_ids: BlockIds
    block_hashes: tuple[BlockHash, ...]
    start_save: int
    end_save: int
    ready_event: KVReadyEvent | None = None
    prompt_tokens: int | None = None


@dataclass(frozen=True)
class LoadResult:
    """Terminal result for the request's one logical load.

    A backend may split the load into group/chunk I/O subtasks. Publish this
    result only after every subtask is terminal and no destination writes
    remain, including on failure. Completion does not imply full success.

    loaded_tokens is the contiguous usable prefix length, INCLUDING the
    event's start_load, valid across all cache groups. A miss may yield
    only start_load with error=None; an I/O failure sets error. Bytes beyond
    loaded_tokens must not be used, even if some chunks were copied.
    Invalid destination block IDs are reported separately by take_errors().
    """

    request_id: str
    loaded_tokens: int
    error: str | None = None


@dataclass(frozen=True)
class SaveResult:
    """Result after all currently submitted saves for a request are terminal.

    Covers every save accepted since the previous SaveResult was collected.
    No running or pending save remains at collection time, including events
    not yet converted into futures. Terminal work includes successful, failed
    and safely cancelled operations; source reads must have stopped in every
    case. preempt() instead clears the work without publishing a result.
    error=None means
    every selected chunk succeeded; errors from earlier work must not be
    hidden by a later successful save. Existing chunks count as success.
    On failure, valid chunks already saved may remain in storage; atomic
    multi-chunk commit is not required. The request may submit more saves
    later; this result does not signal request completion.
    """

    request_id: str
    error: str | None = None


@dataclass
class PoolPollResult:
    loads: list[LoadResult] = field(default_factory=list)
    saves: list[SaveResult] = field(default_factory=list)


class KVCachePool(ABC):
    """Remote KV cache wrapper with pluggable storage backends.

    Backends may include Mooncake Store, MongoDB, FileSystem or memory.
    The pool owns storage lookup and I/O; the connector owns request
    scheduling, block allocation and P-D fallback. The pool tracks pending
    work by request_id, with one result per load and aggregated save results.

    load/save submit work without waiting for I/O. A queue or worker thread
    is an implementation choice, not part of this contract. Results,
    including immediate completions, are collected through poll().
    Normally a request has one logical load in an allocation lifecycle,
    rather than one load per decode step. It may have multiple incremental
    saves as prefill/decode produces more tokens. At most one uncollected
    load is allowed; duplicate loads raise ValueError. Repeated saves are
    accepted, ordered per request and coalesced where safe. A retry requires
    the previous load and block errors to be consumed or preempt() to fence
    and clear them before resubmission.

    Internal tracking is an implementation choice. A typical layout is
    request_id -> one load future (aggregating any load subtasks), and
    request_id -> save state containing running work, pending events and
    accumulated errors. A list of save futures is also valid, but its
    completion check must include pending events not yet submitted to I/O.
    In the common contiguous-save case, one running save and one merged
    pending event suffice; incompatible snapshots remain separate.
    Events are immutable metadata snapshots; no KV tensor copy is required.
    The caller keeps registered tensors and task blocks alive until terminal
    completion or preempt() returns. Public calls are serialized by the
    caller; implementations synchronize their own background workers.
    Request IDs must not be reused while work, results or block errors remain
    outstanding. Error block IDs must be consumed before their blocks are reused;
    preempt() fences old work and clears results before resubmission.
    """

    @abstractmethod
    def register(
        self,
        kv_caches: dict[
            str, torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]
        ],
    ) -> None:
        """Register local KV memory before load/save.

        Layout information comes from pool configuration. Registered tensors
        remain alive until close(); re-registration with pending tasks is
        unsupported.
        """

    @abstractmethod
    def lookup(self, token_count: int, hashes: Sequence[BlockHash]) -> int:
        """Return the remotely available contiguous prefix length in tokens.

        Synchronous metadata query, without copying KV or allocating blocks.
        Returns 0 on a miss, at most token_count, aligned to usable cache
        chunks across all groups. Backend failures raise an exception.
        The caller may schedule this query asynchronously if needed.
        A hit is advisory: data can be evicted before load(), whose result
        remains authoritative. Lookup does not require registered KV memory.
        """

    @abstractmethod
    def load(self, load_event: LoadEvent) -> None:
        """Submit KV loading into the event's destination blocks.

        Non-blocking; misses and I/O failures are reported through poll().
        If split into multiple I/O subtasks, collect and fence all of them
        before reporting the request's LoadResult, even if one fails early.
        The connector decides whether to recompute or use P-D fallback.
        """

    @abstractmethod
    def save(self, save_event: SaveEvent) -> None:
        """Submit KV saving; completion is reported through poll().

        Non-blocking. Wait for ready_event before reading source KV memory.
        Save only complete chunks; do not persist an unfinished token tail.
        Keep an executing save's snapshot unchanged. Pending adjacent or
        overlapping ranges may be merged using their union and the newest
        block/hash snapshot, only if it covers the union and preserves each
        token position's source mapping. Wait for readiness covering all merged
        data; replacing older events is safe only if the newer signal fences
        their writes too. Otherwise retain separate pending snapshots.
        For example, while [0, 64) runs, pending [64, 96) and [96, 128)
        can become [64, 128). Never merge across block reuse after preemption.
        A new submission before a save result is collected extends the work
        covered by that result; do not emit stale completion for earlier work.
        """

    @abstractmethod
    def poll(self) -> PoolPollResult:
        """Drain completed load/save operations without waiting for I/O.

        Each result is returned once. LoadResult fences that load's writes;
        SaveResult fences all accepted saves for that request and is returned
        only when no running or pending save remains. Save completion must be
        rechecked when polled, since new submissions can extend pending work.
        Checking all existing futures is insufficient if a pending event
        has not been submitted yet. Completion means all related work is
        terminal, not necessarily successful; collect errors without returning
        early while other I/O can still access local memory.
        Operation bookkeeping is released when results and block errors are
        consumed; the pool needs no request-end signal. If a block is shared
        by load and save work, both must finish before the caller reuses it.
        Publishing a LoadResult also publishes its invalid block IDs for
        take_errors(); the caller drains those in the same polling cycle.
        SaveResult while the request is still running only updates the
        connector's save status; it must not directly become finished_sending.
        A later save submission makes that request's save status pending again.
        After request_finished(), the connector aggregates block-release
        requirements: request ended, no outstanding saves, and any required
        P-D send terminal. This applies to Store-only Decode saves too;
        send requirements describe block-release dependencies, not only PD.
        """

    @abstractmethod
    def take_errors(self) -> set[int]:
        """Drain invalid local destination block IDs from polled loads.

        Non-blocking; returns an empty set when no errors are pending. Include
        blocks whose requested KV is unusable after a miss, partial load or
        I/O failure, even when LoadResult.error is None. Exclude the existing
        local prefix and all save failures: failed persistence does not
        invalidate valid local KV. IDs are physical local block IDs, not
        hashes or per-request block indices.

        The connector may recover these blocks through P-D fallback before
        reporting remaining failures via get_block_ids_with_load_errors().
        The worker output carries them as invalid_block_ids for scheduler
        recovery. This set carries no additional completion notification;
        local memory access is fenced by the corresponding LoadResult.
        """

    @abstractmethod
    def preempt(self, request_id: str) -> None:
        """Cancel and fence all load/save work for a request before block reuse.

        May block if backend I/O cannot safely be cancelled. On return the
        request's work no longer reads or writes local KV memory, uncollected
        results, pending block errors and request bookkeeping are cleared,
        and resubmission is safe.
        Calls with no tracked request are harmless. Already persisted valid
        KV need not be rolled back.
        """

    @abstractmethod
    def close(self) -> None:
        """Stop accepting work, cancel queued tasks and fence running I/O.

        Release registered memory/resources only after workers stop accessing
        them. Idempotent; subsequent submissions raise RuntimeError.
        """


class _CompatBlobBlockHashes(Sequence[BlockHash]):
    """Lazy fixed-width hash view for vLLM versions before 0.26."""

    def __init__(self, blob: memoryview, hash_len: int) -> None:
        self._blob = blob
        self._hash_len = hash_len
        self._length = len(blob) // hash_len if hash_len else 0

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(self._length))]
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(index)
        offset = index * self._hash_len
        return BlockHash(bytes(self._blob[offset : offset + self._hash_len]))


try:
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
        BlobBlockHashes,
    )
except ImportError:
    BlobBlockHashes = _CompatBlobBlockHashes  # type: ignore[misc]

logger = init_logger(__name__)

_LOOKUP = b"lookup"

_RESET = b"reset"

_OK = b"ok"

_ERR = b"error"


def _parse_size(value: Any) -> int:
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    for suffix, scale in (("gb", 1 << 30), ("mb", 1 << 20), ("kb", 1 << 10), ("b", 1)):
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)]) * scale)
    return int(text)


def _key_prefix(
    metadata: KeyMetadata,
    namespace: str,
    *,
    tp_rank: int | None = None,
    pcp_rank: int | None = None,
    dcp_rank: int | None = None,
    pp_rank: int | None = None,
) -> str:
    """Build a Store key prefix without relying on version-specific helpers."""
    base = (
        f"{metadata.model_name}"
        f"@tp_rank:{metadata.tp_rank if tp_rank is None else tp_rank}"
        f"@pcp{metadata.pcp_rank if pcp_rank is None else pcp_rank}"
        f"@dcp{metadata.dcp_rank if dcp_rank is None else dcp_rank}"
        f"@pp_rank:{metadata.pp_rank if pp_rank is None else pp_rank}"
        f"@group:{metadata.group_id}"
    )
    return f"{namespace}@{base}" if namespace else base


def _key_string(prefix: str, block_hash: BlockHash) -> str:
    return f"{prefix}@{block_hash.hex()}"


def _make_external_cached_pool(
    hash_block_size: int, present: set[tuple[int, bytes]]
) -> ExternalCachedBlockPool:
    parameters = inspect.signature(ExternalCachedBlockPool).parameters
    if "hash_block_size" in parameters:
        return ExternalCachedBlockPool(hash_block_size, present)
    return ExternalCachedBlockPool(present)


def store_tp_layout(
    *,
    use_mla: bool,
    total_num_kv_heads: int,
    tp_size: int,
    dcp_size: int,
    tp_rank: int,
) -> tuple[int, int, int]:
    """Return ``(key_heads, put_step, key_tp_rank)`` for Store I/O.

    MLA KV is replicated across TP ranks, so it is represented as one KV head
    in Store.  TP ranks share one key and take turns writing chunks; every rank
    can subsequently load that key into its own replicated cache.
    """
    num_kv_heads = 1 if use_mla else total_num_kv_heads
    if num_kv_heads <= 0 or tp_size <= 0 or not 0 <= tp_rank < tp_size:
        raise ValueError("invalid KV head count or TP topology")
    if num_kv_heads < tp_size and dcp_size <= 1:
        if tp_size % num_kv_heads:
            raise ValueError("TP size must be divisible by the KV head count")
        put_step = tp_size // num_kv_heads
        return num_kv_heads, put_step, tp_rank // put_step
    return num_kv_heads, 1, tp_rank


def store_topology_namespace(
    vllm_config: VllmConfig,
    groups: Sequence[Any],
    *,
    tp_size: int,
    pp_size: int,
    pcp_size: int,
    dcp_size: int,
    block_size: int,
    hash_block_size: int,
) -> str:
    """Build a stable namespace that excludes incompatible Store objects."""
    model = vllm_config.model_config
    cache = vllm_config.cache_config

    def spec_signature(spec: Any) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": f"{type(spec).__module__}.{type(spec).__qualname__}"
        }
        for name in (
            "block_size",
            "page_size_bytes",
            "sliding_window",
            "num_kv_heads",
            "head_size",
            "dtype",
        ):
            value = getattr(spec, name, None)
            if value is not None:
                result[name] = str(value)
        nested = getattr(spec, "kv_cache_specs", None)
        if nested:
            unique = {
                json.dumps(spec_signature(item), sort_keys=True, separators=(",", ":"))
                for item in nested.values()
            }
            result["nested"] = sorted(unique)
        return result

    if is_npu_platform():
        layout = "npu-nz" if npu_kv_nz_enabled(vllm_config) else "npu-normal"
    else:
        layout = str(get_kv_cache_layout())
    payload = {
        "version": 1,
        "model_id": str(model.model),
        "model_revision": str(getattr(model, "revision", None) or ""),
        "tp_size": tp_size,
        "pp_size": pp_size,
        "pcp_size": pcp_size,
        "dcp_size": dcp_size,
        "block_size": block_size,
        "hash_block_size": hash_block_size,
        "kv_layout": layout,
        "kv_cache_dtype": str(
            model.dtype if cache.cache_dtype == "auto" else cache.cache_dtype
        ),
        "is_mla": bool(model.use_mla),
        "group_schema": [spec_signature(group.kv_cache_spec) for group in groups],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    signature = hashlib.sha256(canonical.encode()).hexdigest()[:24]
    assert vllm_config.kv_transfer_config is not None
    extra = vllm_config.kv_transfer_config.kv_connector_extra_config
    user_prefix = str(extra.get("cache_prefix", ""))
    enabled = extra.get("mtsc_store_topology_namespace", True)
    if enabled is None:
        enabled = True
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() not in {"0", "false", "no", "off"}
    if not enabled:
        return user_prefix
    namespace = f"mtsc-v1-{signature}"
    return f"{user_prefix}:{namespace}" if user_prefix else namespace


@dataclass(frozen=True)
class StoreConfig:
    metadata_server: str
    master_server_address: str
    protocol: str = "rdma"
    device_name: str = ""
    global_segment_size: int = 4 << 30
    local_buffer_size: int = 4 << 30

    @classmethod
    def load(cls) -> StoreConfig:
        path = os.getenv("MOONCAKE_CONFIG_PATH")
        if not path:
            raise ValueError("MOONCAKE_CONFIG_PATH is not set")
        with open(path) as stream:
            raw = json.load(stream)
        return cls(
            metadata_server=raw.get("metadata_server", ""),
            master_server_address=raw.get("master_server_address", ""),
            protocol=raw.get("protocol", "rdma"),
            device_name=raw.get("device_name", ""),
            global_segment_size=_parse_size(raw.get("global_segment_size", 4 << 30)),
            local_buffer_size=_parse_size(raw.get("local_buffer_size", 4 << 30)),
        )


def lookup_rpc_path(vllm_config: VllmConfig) -> str:
    """Use lookup_rpc_port as an IPC path identifier, not a TCP port."""
    assert vllm_config.kv_transfer_config is not None
    extra = vllm_config.kv_transfer_config.kv_connector_extra_config
    port = extra.get("lookup_rpc_port", 0)
    dp_rank = get_mooncake_dp_engine_index(vllm_config.parallel_config)
    return (
        f"ipc://{envs.VLLM_RPC_BASE_PATH}/mtsc_lookup_{port}_"
        f"host_{socket.gethostname()}_dp_rank{dp_rank}"
    )


class StoreLookupClient:
    """Scheduler-side async lookup client with per-request futures."""

    def __init__(self, vllm_config: VllmConfig) -> None:
        assert vllm_config.kv_transfer_config is not None
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        self._path = lookup_rpc_path(vllm_config)
        timeout = float(extra.get("mtsc_store_lookup_timeout_seconds", 10.0))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(
                "mtsc_store_lookup_timeout_seconds must be finite and positive"
            )
        if timeout > (2**31 - 1) / 1000:
            raise ValueError(
                "mtsc_store_lookup_timeout_seconds exceeds ZMQ's millisecond limit"
            )
        self._timeout_ms = max(1, int(timeout * 1000))
        self._ctx = zmq.Context()  # type: ignore[attr-defined]
        self._socket = self._make_socket()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mtsc-lookup"
        )
        self._futures: dict[str, Future[int]] = {}

    def _make_socket(self):
        socket = make_zmq_socket(self._ctx, self._path, zmq.REQ, bind=False)
        socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        return socket

    def _reset_socket(self) -> None:
        self._socket.close(linger=0)
        self._socket = self._make_socket()

    def _lookup(self, token_count: int, hashes: list[BlockHash]) -> int:
        hash_len = len(hashes[0]) if hashes else 0
        try:
            self._socket.send_multipart(
                (
                    _LOOKUP,
                    token_count.to_bytes(4, "big"),
                    hash_len.to_bytes(2, "big"),
                    b"".join(hashes),
                ),
                copy=False,
            )
            response = self._socket.recv()
            if response == _ERR:
                raise RuntimeError("Pool lookup backend failed")
            hit = int.from_bytes(response, "big")
            if not 0 <= hit <= token_count:
                raise ValueError("Pool lookup returned an invalid prefix")
            return hit
        except zmq.Again as exc:
            self._reset_socket()
            raise TimeoutError(
                f"MTSC Store lookup timed out after {self._timeout_ms / 1000:.3f}s"
            ) from exc

    def lookup(
        self,
        request_id: str,
        token_count: int,
        hashes: list[BlockHash],
        *,
        asynchronous: bool,
    ) -> int | None:
        future = self._futures.get(request_id)
        if future is None:
            future = self._executor.submit(self._lookup, token_count, list(hashes))
            self._futures[request_id] = future
        if asynchronous and not future.done():
            return None
        try:
            return future.result()
        except Exception as exc:  # noqa: BLE001 - worker/RPC boundary
            logger.warning(
                "MTSC Store lookup failed: request_id=%s error=%s", request_id, exc
            )
            return 0
        finally:
            self._futures.pop(request_id, None)

    def discard(self, request_id: str) -> None:
        future = self._futures.pop(request_id, None)
        if future is not None:
            future.cancel()

    def reset(self) -> bool:
        def call() -> bool:
            try:
                self._socket.send(_RESET)
                return self._socket.recv() == _OK
            except zmq.Again as exc:
                self._reset_socket()
                raise TimeoutError(
                    f"MTSC Store reset timed out after {self._timeout_ms / 1000:.3f}s"
                ) from exc

        try:
            return self._executor.submit(call).result()
        except Exception as exc:  # noqa: BLE001 - admin RPC boundary
            logger.warning("MTSC Store reset failed: error=%s", exc)
            return False

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._socket.close(linger=0)
        self._ctx.term()


class StoreLookupServer:
    def __init__(self, owner: MooncakeKVCachePool, vllm_config: VllmConfig) -> None:
        self._owner = owner
        self._ctx = zmq.Context()  # type: ignore[attr-defined]
        self._path = lookup_rpc_path(vllm_config)
        ipc_path = self._path.removeprefix("ipc://")
        if os.path.exists(ipc_path):
            os.unlink(ipc_path)
        self._socket = make_zmq_socket(self._ctx, self._path, zmq.REP, bind=True)
        self._socket.setsockopt(zmq.RCVTIMEO, 200)
        self._running = True
        self._thread = threading.Thread(
            target=self._serve, name="mtsc-store-lookup", daemon=True
        )
        self._thread.start()

    def _serve(self) -> None:
        while self._running:
            try:
                frames = self._socket.recv_multipart(copy=False)
            except zmq.Again:
                continue
            except zmq.ZMQError:
                return
            kind = bytes(frames[0])
            if kind == _LOOKUP:
                token_count = int.from_bytes(frames[1], "big")
                hash_len = int.from_bytes(frames[2], "big")
                hashes = BlobBlockHashes(frames[3].buffer, hash_len)
                try:
                    hit = self._owner.lookup(token_count, hashes)
                    self._socket.send(hit.to_bytes(4, "big"))
                except Exception:
                    # Keep REP alive after a backend failure; the scheduler
                    # client degrades query errors to a cache miss.
                    logger.exception("MTSC Pool lookup failed")
                    self._socket.send(_ERR)
            elif kind == _RESET:
                try:
                    self._owner.wait_for_all_saves()
                    self._owner.store.remove_all(force=True)
                    self._socket.send(_OK)
                except Exception:
                    logger.exception("MTSC Store reset failed")
                    self._socket.send(_ERR)
            else:
                self._socket.send(_ERR)

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=2)
        self._socket.close(linger=0)
        self._ctx.term()
        ipc_path = self._path.removeprefix("ipc://")
        if os.path.exists(ipc_path):
            os.unlink(ipc_path)


@dataclass(frozen=True)
class _ReadySignals:
    signals: tuple[KVReadyEvent, ...]

    def synchronize(self) -> None:
        for signal in self.signals:
            signal.synchronize()


@dataclass
class _SaveState:
    running: Future[None] | None = None
    pending: list[SaveEvent] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _LoadOutcome:
    loaded_tokens: int
    error: str | None
    invalid: set[int]


@dataclass(slots=True)
class _LoadState:
    """One submitted load: immutable event, I/O fence and timeout tracking."""

    event: LoadEvent
    future: Future[_LoadOutcome]
    started_at: float
    timed_out: bool = False


class MooncakeKVCachePool(KVLayoutAdapter, KVCachePool):
    """Own Mooncake memory registration, lookup and fenced load/save tasks."""

    def __init__(self, config, kv_cache_config) -> None:
        self._closed = False
        self._registered = False
        self._task_lock = threading.RLock()
        self._save_states: dict[str, _SaveState] = {}
        self._invalid: dict[str, set[int]] = {}
        try:
            from mooncake.store import MooncakeDistributedStore, ReplicateConfig
        except ImportError as exc:
            raise ImportError("Mooncake Python store bindings are required") from exc

        assert config.kv_transfer_config is not None
        model = config.model_config
        parallel = config.parallel_config
        extra = config.kv_transfer_config.kv_connector_extra_config
        recv_workers = max(1, int(extra.get("mtsc_store_load_workers", 2)))
        self.load_timeout = float(extra.get("mtsc_store_get_timeout_seconds", 180.0))
        if not math.isfinite(self.load_timeout) or self.load_timeout <= 0:
            raise ValueError(
                "mtsc_store_get_timeout_seconds must be finite and positive"
            )
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        pp_size = parallel.pipeline_parallel_size
        pp_rank = (parallel.rank // self.tp_size) % pp_size
        pcp = get_pcp_group()
        dcp = get_dcp_group()
        pcp_size, pcp_rank = (
            pcp.world_size,
            pcp.rank_in_group if pcp.world_size > 1 else 0,
        )
        dcp_size, dcp_rank = (
            dcp.world_size,
            dcp.rank_in_group if dcp.world_size > 1 else 0,
        )
        block_size, self.hash_block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, config
        )
        if config.cache_config.num_gpu_blocks is None:
            raise ValueError(
                "num_gpu_blocks must be initialized before MooncakeKVCachePool"
            )
        self.num_blocks = config.cache_config.num_gpu_blocks
        num_kv_heads, self.put_step, key_tp_rank = store_tp_layout(
            use_mla=model.use_mla,
            total_num_kv_heads=model.get_total_num_kv_heads(),
            tp_size=self.tp_size,
            dcp_size=dcp_size,
            tp_rank=self.tp_rank,
        )

        groups = list(kv_cache_config.kv_cache_groups)
        if len(groups) == 1 and groups[0].kv_cache_spec.block_size != block_size:
            group = groups[0]
            groups = [
                dataclasses.replace(
                    group,
                    kv_cache_spec=dataclasses.replace(
                        group.kv_cache_spec, block_size=block_size
                    ),
                )
            ]
        self.groups = groups
        spec_cfg = getattr(config, "speculative_config", None)
        use_eagle = bool(
            spec_cfg.use_eagle()
            if spec_cfg is not None and callable(getattr(spec_cfg, "use_eagle", None))
            else False
        )
        coordinator_kwargs: dict[str, Any] = {
            "scheduler_block_size": block_size,
            "hash_block_size": self.hash_block_size,
            "use_eagle": use_eagle,
        }
        if (
            "retention_interval"
            in inspect.signature(MooncakeStoreCoordinator).parameters
        ):
            coordinator_kwargs["retention_interval"] = (
                envs.VLLM_PREFIX_CACHE_RETENTION_INTERVAL
            )
        self.coordinator = MooncakeStoreCoordinator(groups, **coordinator_kwargs)
        self.cache_namespace = store_topology_namespace(
            config,
            groups,
            tp_size=self.tp_size,
            pp_size=pp_size,
            pcp_size=pcp_size,
            dcp_size=dcp_size,
            block_size=block_size,
            hash_block_size=self.hash_block_size,
        )
        metadata = KeyMetadata(
            model_name=model.model.rstrip("/").split("/")[-1],
            tp_rank=key_tp_rank,
            pcp_rank=pcp_rank,
            dcp_rank=dcp_rank,
            pp_rank=pp_rank,
        )
        self.databases = [
            ChunkedTokenDatabase(
                dataclasses.replace(metadata, group_id=i),
                group.kv_cache_spec.block_size,
                self.hash_block_size,
            )
            for i, group in enumerate(groups)
        ]
        self._init_lookup_prefixes(
            num_kv_heads, pp_size=pp_size, pcp_size=pcp_size, dcp_size=dcp_size
        )

        cfg = StoreConfig.load()
        self.store = MooncakeDistributedStore()
        hostname = rdma_utils.get_requester_local_hostname(get_ip())
        ret = self.store.setup(
            hostname,
            cfg.metadata_server,
            cfg.global_segment_size,
            cfg.local_buffer_size,
            cfg.protocol,
            cfg.device_name,
            cfg.master_server_address,
        )
        if ret != 0:
            raise RuntimeError(f"Mooncake Store setup failed: {ret}")
        self.replicate_config = ReplicateConfig()
        preferred = rdma_utils.get_configured_preferred_segment(extra)
        if preferred is not None:
            self.replicate_config.preferred_segment = preferred

        # PUTs use one ordered lane; GETs remain parallel.
        save_workers = 1
        self._load_pool = ThreadPoolExecutor(
            max_workers=recv_workers, thread_name_prefix="mtsc-store-get"
        )
        self._save_pool = ThreadPoolExecutor(
            max_workers=save_workers, thread_name_prefix="mtsc-store-put"
        )
        self._loads: dict[str, _LoadState] = {}
        self._lookup_server = (
            StoreLookupServer(self, config) if parallel.rank == 0 else None
        )
        self.device_id = torch.accelerator.current_device_index()
        self.kv_caches = {}
        self.npu_kv_nz = npu_kv_nz_enabled(config)
        self.topology = SimpleNamespace(
            block_size=block_size,
            is_mla=config.model_config.use_mla,
            total_num_kv_heads=config.model_config.get_total_num_kv_heads(),
        )
        self.layer_groups = {}
        self.layer_specs = {}
        for index, group in enumerate(kv_cache_config.kv_cache_groups):
            specs = getattr(group.kv_cache_spec, "kv_cache_specs", {})
            for layer in group.layer_names:
                self.layer_groups[layer] = index
                self.layer_specs[layer] = specs.get(layer, group.kv_cache_spec)

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("KV cache pool is closed")

    def register(self, kv_caches) -> None:
        self._check_open()
        if self._registered:
            raise RuntimeError("KV cache memory is already registered")
        if not kv_caches:
            raise RuntimeError("No KV caches supplied")
        seen: set[int] = set()
        addresses: list[int] = []
        block_lengths: list[int] = []
        if is_npu_platform():
            pointers, lengths = npu_registration_regions(kv_caches)
            for pointer, length in zip(pointers, lengths, strict=True):
                ret = self.store.register_buffer(pointer, length)
                if ret != 0:
                    raise RuntimeError(f"Mooncake register_buffer failed: {ret}")
        for raw in kv_caches.values():
            for tensor in cache_tensors(raw):
                storage = tensor.untyped_storage()
                storage_base, storage_size = storage.data_ptr(), storage.nbytes()
                if not is_npu_platform() and storage_base not in seen:
                    seen.add(storage_base)
                    ret = self.store.register_buffer(storage_base, storage_size)
                    if ret != 0:
                        raise RuntimeError(f"Mooncake register_buffer failed: {ret}")
                if is_npu_platform():
                    if tensor.ndim == 0 or tensor.shape[0] < self.num_blocks:
                        raise ValueError(
                            "Ascend KV cache must have a leading block dimension"
                        )
                    addresses.append(tensor.data_ptr())
                    block_lengths.append(tensor.stride(0) * tensor.element_size())
                    continue
                # CUDA may expose a packed storage with outer K/V or layer
                # dimensions. Preserve the original storage expansion there.
                if tensor.data_ptr() != storage_base:
                    continue
                page = storage_size // self.num_blocks
                outer = [
                    d
                    for d in range(tensor.ndim)
                    if tensor.stride(d) * tensor.element_size() > page
                ]
                if not outer:
                    addresses.append(storage_base)
                    block_lengths.append(page)
                else:
                    stride = tensor.stride(outer[0]) * tensor.element_size()
                    for index in range(tensor.shape[outer[0]]):
                        addresses.append(storage_base + index * stride)
                        block_lengths.append(stride // self.num_blocks)
        for database in self.databases:
            database.set_kv_caches_base_addr(addresses)
            database.set_block_len(block_lengths)
        logger.info("MTSC Store registered %d regions", len(addresses))
        self.kv_caches = dict(kv_caches)
        self._registered = True

    def lookup(self, token_count: int, hashes: Sequence[BlockHash]) -> int:
        self._check_open()
        if token_count <= 0 or not hashes:
            return 0
        keys: list[str] = []
        candidates: list[tuple[int, bytes]] = []
        masks = self._lookup_masks(token_count)
        for group_index, database in enumerate(self.databases):
            group_hashes = self.coordinator.block_hashes_for_spec(
                hashes, self.groups[group_index].kv_cache_spec
            )
            mask = masks[group_index]
            limit = min(len(group_hashes), cdiv(token_count, database.block_size))
            if mask is not None:
                limit = min(limit, len(mask))
            for index in range(limit):
                if mask is not None and not mask[index]:
                    continue
                block_hash = group_hashes[index]
                keys.extend(
                    _key_string(prefix, block_hash)
                    for prefix in self._lookup_prefixes[group_index]
                )
                candidates.append((group_index, bytes(block_hash)))
        if not keys:
            return 0
        result = self.store.batch_is_exist(keys)
        present = {
            candidate
            for index, candidate in enumerate(candidates)
            if all(
                result[index * self._lookup_rank_count + rank] == 1
                for rank in range(self._lookup_rank_count)
            )
        }
        _, hit = self.coordinator.find_longest_cache_hit(
            hashes,
            token_count,
            _make_external_cached_pool(self.hash_block_size, present),
        )
        return hit

    def _validate(self, event, start: int, end: int) -> None:
        self._check_open()
        if not self._registered:
            raise RuntimeError("Register KV memory before submitting I/O")
        if start < 0 or end < start:
            raise ValueError("Invalid KV token range")
        if len(event.block_ids) != len(self.databases):
            raise ValueError("KV group count does not match registered layout")
        for ids in event.block_ids:
            if any(block >= self.num_blocks for block in ids):
                raise ValueError("KV block ID exceeds registered memory")

    def load(self, event: LoadEvent) -> None:
        self._validate(event, event.start_load, event.end_load)
        with self._task_lock:
            if event.request_id in self._loads or event.request_id in self._invalid:
                raise ValueError(f"Uncollected load for {event.request_id}")
            started_at = time.monotonic()
            future = self._load_pool.submit(self._load_work, event)
            self._loads[event.request_id] = _LoadState(event, future, started_at)

    def _load_work(self, event: LoadEvent) -> _LoadOutcome:
        error = None
        try:
            failed = self._load(event)
            if failed:
                error = "Mooncake load missed or failed"
        except Exception as exc:  # noqa: BLE001 - I/O result boundary
            failed = self._request_load_blocks(event)
            error = str(exc)
        actual = self._loaded_prefix(event, failed)
        try:
            # Conversion is part of the aggregate I/O future. poll() never
            # waits for device conversion; success fences its writes too.
            self.reformat_npu_blocks(
                self._range_blocks(event, event.start_load, actual),
                self.tp_size,
            )
        except Exception as exc:  # noqa: BLE001 - device conversion result boundary
            actual = event.start_load
            error = str(exc)
        invalid = {
            block
            for group in self._range_blocks(event, actual, event.end_load)
            for block in group
        }
        return _LoadOutcome(actual, error, invalid)

    @staticmethod
    def _merge(old: SaveEvent, new: SaveEvent, block_sizes) -> SaveEvent | None:
        if old.prompt_tokens != new.prompt_tokens:
            return None
        if max(old.start_save, new.start_save) > min(old.end_save, new.end_save):
            return None
        end = max(old.end_save, new.end_save)
        if new.end_save < end or len(new.block_hashes) < len(old.block_hashes):
            return None
        if new.block_hashes[: len(old.block_hashes)] != old.block_hashes:
            return None
        for older, newer, size in zip(
            old.block_ids, new.block_ids, block_sizes, strict=True
        ):
            required = cdiv(end, size)
            if len(newer) < required or newer[: len(older)] != older:
                return None
        signals = []
        for ready in (old.ready_event, new.ready_event):
            values = ready.signals if isinstance(ready, _ReadySignals) else (ready,)
            for value in values:
                if value is not None and all(value is not other for other in signals):
                    signals.append(value)
        return replace(
            new,
            start_save=min(old.start_save, new.start_save),
            end_save=end,
            ready_event=_ReadySignals(tuple(signals)) if signals else None,
        )

    def save(self, event: SaveEvent) -> None:
        self._validate(event, event.start_save, event.end_save)
        with self._task_lock:
            state = self._save_states.setdefault(event.request_id, _SaveState())
            if state.pending:
                merged = self._merge(
                    state.pending[-1],
                    event,
                    tuple(database.block_size for database in self.databases),
                )
                if merged is not None:
                    state.pending[-1] = merged
                else:
                    state.pending.append(event)
            else:
                state.pending.append(event)
            self._advance_save(state)

    def _advance_save(self, state: _SaveState) -> None:
        if state.running is not None:
            if not state.running.done():
                return
            try:
                state.running.result()
            except Exception as exc:  # noqa: BLE001 - background future boundary
                state.errors.append(str(exc))
            state.running = None
        if state.pending:
            event = state.pending.pop(0)
            state.running = self._save_pool.submit(self._save, event)

    def _loaded_prefix(self, event: LoadEvent, failed: set[int]) -> int:
        actual = event.end_load
        for ids, database in zip(event.block_ids, self.databases, strict=True):
            start = cdiv(event.start_load, database.block_size)
            end = min(cdiv(event.end_load, database.block_size), len(ids))
            for index in range(start, end):
                if ids[index] in failed:
                    actual = min(actual, index * database.block_size)
                    break
        return max(event.start_load, actual)

    def _range_blocks(self, event: LoadEvent, start: int, end: int):
        return [
            [
                block
                for block in ids[cdiv(start, db.block_size) : cdiv(end, db.block_size)]
                if block >= 0
            ]
            for ids, db in zip(event.block_ids, self.databases, strict=True)
        ]

    def poll(self) -> PoolPollResult:
        result = PoolPollResult()
        with self._task_lock:
            for request_id, state in list(self._loads.items()):
                event, future = state.event, state.future
                if not future.done():
                    if time.monotonic() - state.started_at >= self.load_timeout:
                        state.timed_out = True
                    continue
                try:
                    outcome = future.result()
                except Exception as exc:  # noqa: BLE001 - background future boundary
                    outcome = _LoadOutcome(
                        event.start_load, str(exc), self._request_load_blocks(event)
                    )
                if state.timed_out:
                    outcome = _LoadOutcome(
                        event.start_load,
                        "Mooncake load timed out (I/O fenced)",
                        self._request_load_blocks(event),
                    )
                if outcome.invalid:
                    self._invalid[request_id] = outcome.invalid
                result.loads.append(
                    LoadResult(request_id, outcome.loaded_tokens, outcome.error)
                )
                del self._loads[request_id]
            for request_id, state in list(self._save_states.items()):
                self._advance_save(state)
                if state.running is None and not state.pending:
                    result.saves.append(
                        SaveResult(
                            request_id,
                            "; ".join(state.errors) or None,
                        )
                    )
                    del self._save_states[request_id]
        return result

    def take_errors(self) -> set[int]:
        with self._task_lock:
            errors = {block for blocks in self._invalid.values() for block in blocks}
            self._invalid.clear()
            return errors

    def preempt(self, request_id: str) -> None:
        with self._task_lock:
            state = self._save_states.pop(request_id, None)
            if state is not None:
                state.pending.clear()
                if state.running is not None and not state.running.cancel():
                    try:
                        state.running.result()
                    except Exception as exc:  # noqa: BLE001 - preemption I/O fence
                        # Failed persistence does not invalidate source KV.
                        logger.debug("MTSC PUT failed during preemption: %s", exc)
            self._fence_loads({request_id})
            self._invalid.pop(request_id, None)

    def wait_for_all_saves(self) -> None:
        with self._task_lock:
            for state in self._save_states.values():
                while state.running is not None or state.pending:
                    self._advance_save(state)
                    if state.running is not None:
                        try:
                            state.running.result()
                        except Exception:  # noqa: BLE001, S110 - _advance_save retains the error
                            pass

    def close(self) -> None:
        with self._task_lock:
            if self._closed:
                return
            self._closed = True
            for state in self._save_states.values():
                state.pending.clear()
        if self._lookup_server is not None:
            self._lookup_server.close()
        # Running Mooncake calls cannot be cancelled safely: they still own
        # registered GPU addresses and the Store handle. Fence them before
        # closing either resource.
        self._load_pool.shutdown(wait=True, cancel_futures=True)
        self._save_pool.shutdown(wait=True, cancel_futures=True)
        store, self.store = self.store, None
        if store is not None:
            store.close()
        self._save_states.clear()
        self._loads.clear()
        self._invalid.clear()
        self.kv_caches.clear()

    def _init_lookup_prefixes(
        self, num_kv_heads: int, *, pp_size: int, pcp_size: int, dcp_size: int
    ) -> None:
        if dcp_size > 1:
            ranks = tuple(
                (tp, pcp, tp % dcp_size, pp)
                for pcp in range(pcp_size)
                for tp in range(self.tp_size)
                for pp in range(pp_size)
            )
        else:
            ranks = tuple(
                (tp, pcp, 0, pp)
                for pcp in range(pcp_size)
                for tp in range(min(self.tp_size, num_kv_heads))
                for pp in range(pp_size)
            )
        self._lookup_prefixes = tuple(
            tuple(
                _key_prefix(
                    db.metadata,
                    self.cache_namespace,
                    tp_rank=tp,
                    pcp_rank=pcp,
                    dcp_rank=dcp,
                    pp_rank=pp,
                )
                for tp, pcp, dcp, pp in ranks
            )
            for db in self.databases
        )
        self._lookup_rank_count = len(ranks)

    def _lookup_masks(self, token_count: int) -> tuple[list[bool] | None, ...]:
        lookup_mask = getattr(self.coordinator, "lookup_mask", None)
        if lookup_mask is None:
            # vLLM 0.23 did not expose lookup_mask. Querying every key is a
            # conservative superset; find_longest_cache_hit still decides the
            # valid continuous prefix.
            return tuple(None for _ in self.databases)
        return lookup_mask(token_count)

    def _store_masks(
        self, token_count: int, save_from: int, prompt_tokens: int | None
    ) -> tuple[list[bool] | None, ...]:
        store_mask = self.coordinator.store_mask
        if "start_token" in inspect.signature(store_mask).parameters:
            return store_mask(token_count, save_from, num_prompt_tokens=prompt_tokens)
        # vLLM 0.23 returns masks for [0, token_count). Convert them to the
        # suffix-relative masks expected by MTSC's incremental save path.
        masks = store_mask(token_count)
        return tuple(
            None if mask is None else mask[cdiv(save_from, database.block_size) :]
            for mask, database in zip(masks, self.databases, strict=True)
        )

    def _database_key(self, database: ChunkedTokenDatabase, value: Any) -> str:
        # vLLM 0.23 process_tokens yields PoolKey; newer versions yield
        # BlockHash and expose key_for(). Namespace locally in both cases so
        # topology isolation does not depend on KeyMetadata.cache_prefix.
        if hasattr(value, "to_string"):
            base = value.to_string()
        else:
            base = database.key_for(value)
        return f"{self.cache_namespace}@{base}" if self.cache_namespace else base

    def _process_tokens(
        self,
        database: ChunkedTokenDatabase,
        token_count: int,
        block_hashes: Sequence[BlockHash],
        start_token: int,
        *,
        chunk_mask: list[bool] | None = None,
        put_step: int = 1,
        put_step_rank: int = 0,
    ) -> Iterator[tuple[int, int, str]]:
        """Normalize vLLM 0.23 and newer ChunkedTokenDatabase APIs."""
        if put_step <= 0:
            raise ValueError("put_step must be positive")
        start_chunk = cdiv(start_token, database.block_size)
        for start, end, value in database.process_tokens(
            token_count, block_hashes, start_token
        ):
            chunk = start // database.block_size
            relative = chunk - start_chunk
            if chunk_mask is not None and (
                relative < 0 or relative >= len(chunk_mask) or not chunk_mask[relative]
            ):
                continue
            if chunk % put_step != put_step_rank:
                continue
            yield start, end, self._database_key(database, value)

    def _request_load_blocks(self, event: LoadEvent) -> set[int]:
        result: set[int] = set()
        for group, database in zip(event.block_ids, self.databases, strict=True):
            start = cdiv(event.start_load, database.block_size)
            end = min(cdiv(event.end_load, database.block_size), len(group))
            result.update(block for block in group[start:end] if block >= 0)
        return result

    def _load(self, event: LoadEvent) -> set[int]:
        failed: set[int] = set()
        masks = self.coordinator.load_mask(event.block_hashes, event.end_load)
        keys: list[str] = []
        addresses: list[list[int]] = []
        sizes: list[list[int]] = []
        block_ids: list[int] = []
        for group_index, database in enumerate(self.databases):
            for start, end, key in self._process_tokens(
                database,
                event.end_load,
                event.block_hashes,
                event.start_load,
            ):
                chunk = start // database.block_size
                if chunk >= len(masks[group_index]) or not masks[group_index][chunk]:
                    continue
                address, size, block_id = database.prepare_value(
                    start, end, event.block_ids[group_index]
                )
                keys.append(key)
                addresses.append(address)
                sizes.append(size)
                block_ids.append(block_id)
        if keys:
            try:
                result = self.store.batch_get_into_multi_buffers(keys, addresses, sizes)
                failed.update(
                    block_id
                    for block_id, status in zip(block_ids, result, strict=True)
                    if status < 0
                )
            except Exception as exc:  # noqa: BLE001 - Mooncake binding boundary
                failed.update(block_ids)
                logger.warning(
                    "MTSC Store GET failed: request_id=%s error=%s",
                    event.request_id,
                    exc,
                )
        return failed

    def _save(self, event: SaveEvent) -> None:
        token_count = (
            event.end_save
            // self.coordinator.lcm_block_size
            * self.coordinator.lcm_block_size
        )
        save_from = event.start_save
        if token_count <= save_from:
            return
        masks = self._store_masks(token_count, save_from, event.prompt_tokens)
        keys: list[str] = []
        addresses: list[list[int]] = []
        sizes: list[list[int]] = []
        for group_index, database in enumerate(self.databases):
            phase = (self.tp_rank + group_index) % self.put_step
            for start, end, key in self._process_tokens(
                database,
                token_count,
                event.block_hashes,
                save_from,
                chunk_mask=masks[group_index],
                put_step=self.put_step,
                put_step_rank=phase,
            ):
                address, size, _ = database.prepare_value(
                    start, end, event.block_ids[group_index]
                )
                keys.append(key)
                addresses.append(address)
                sizes.append(size)
        if not keys:
            return
        exists = self.store.batch_is_exist(keys)
        missing = [index for index, status in enumerate(exists) if status != 1]
        if not missing:
            return
        if event.ready_event is not None:
            event.ready_event.synchronize()
        result = self.store.batch_put_from_multi_buffers(
            [keys[i] for i in missing],
            [addresses[i] for i in missing],
            [sizes[i] for i in missing],
            self.replicate_config,
        )
        if any(status < 0 for status in result):
            logger.warning(
                "MTSC Store PUT partially failed: request_id=%s", event.request_id
            )
            raise RuntimeError("Mooncake Store PUT partially failed")

    def _fence_loads(self, request_ids: set[str]) -> None:
        """Fence Store writes before a preempted request's blocks are reused."""
        for request_id in request_ids:
            state = self._loads.pop(request_id, None)
            if state is not None and not state.future.cancel():
                try:
                    state.future.result()
                except Exception as exc:  # noqa: BLE001 - I/O fence
                    logger.warning(
                        "MTSC Store GET failed while fencing preemption: "
                        "request_id=%s error=%s",
                        request_id,
                        exc,
                    )
