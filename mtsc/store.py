"""MTSC-owned Mooncake Store data path.

Only the low-level ``MooncakeDistributedStore`` API and vLLM's pure key/mask
helpers are reused.  Queueing, futures, completion and request ownership live
in MTSC.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import os
import socket
import threading
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

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
from vllm.v1.kv_cache_interface import KVCacheConfig


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

from .device import (
    DeviceEvent,
    cache_tensors,
    is_npu_platform,
    npu_kv_nz_enabled,
    npu_registration_regions,
)
from .protocol import StoreRequest

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
        self._ctx = zmq.Context()  # type: ignore[attr-defined]
        assert vllm_config.kv_transfer_config is not None
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        self._path = lookup_rpc_path(vllm_config)
        self._timeout_ms = max(
            1, int(float(extra.get("mtsc_store_lookup_timeout_seconds", 10.0)) * 1000)
        )
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
            return int.from_bytes(self._socket.recv(), "big")
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
                    "MTSC Store reset timed out after "
                    f"{self._timeout_ms / 1000:.3f}s"
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
    def __init__(self, owner: StoreIO, vllm_config: VllmConfig) -> None:
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
                self._socket.send(
                    self._owner.lookup(token_count, hashes).to_bytes(4, "big")
                )
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


class StoreIO:
    """Worker-side asynchronous GET/PUT engine owned by MTSC."""

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig) -> None:
        try:
            from mooncake.store import MooncakeDistributedStore, ReplicateConfig
        except ImportError as exc:
            raise ImportError("Mooncake Python store bindings are required") from exc

        assert vllm_config.kv_transfer_config is not None
        model = vllm_config.model_config
        parallel = vllm_config.parallel_config
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.pp_size = parallel.pipeline_parallel_size
        self.pp_rank = (parallel.rank // self.tp_size) % self.pp_size
        pcp = get_pcp_group()
        dcp = get_dcp_group()
        self.pcp_size, self.pcp_rank = (
            pcp.world_size,
            pcp.rank_in_group if pcp.world_size > 1 else 0,
        )
        self.dcp_size, self.dcp_rank = (
            dcp.world_size,
            dcp.rank_in_group if dcp.world_size > 1 else 0,
        )
        self.block_size, self.hash_block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )
        if vllm_config.cache_config.num_gpu_blocks is None:
            raise ValueError("num_gpu_blocks must be initialized before StoreIO")
        self.num_blocks = vllm_config.cache_config.num_gpu_blocks
        num_kv_heads, self.put_step, key_tp_rank = store_tp_layout(
            use_mla=model.use_mla,
            total_num_kv_heads=model.get_total_num_kv_heads(),
            tp_size=self.tp_size,
            dcp_size=self.dcp_size,
            tp_rank=self.tp_rank,
        )

        groups = list(kv_cache_config.kv_cache_groups)
        if len(groups) == 1 and groups[0].kv_cache_spec.block_size != self.block_size:
            group = groups[0]
            groups = [
                dataclasses.replace(
                    group,
                    kv_cache_spec=dataclasses.replace(
                        group.kv_cache_spec, block_size=self.block_size
                    ),
                )
            ]
        self.groups = groups
        spec_cfg = getattr(vllm_config, "speculative_config", None)
        use_eagle = bool(
            spec_cfg.use_eagle()
            if spec_cfg is not None and callable(getattr(spec_cfg, "use_eagle", None))
            else False
        )
        coordinator_kwargs: dict[str, Any] = {
            "scheduler_block_size": self.block_size,
            "hash_block_size": self.hash_block_size,
            "use_eagle": use_eagle,
        }
        if "retention_interval" in inspect.signature(
            MooncakeStoreCoordinator
        ).parameters:
            coordinator_kwargs["retention_interval"] = (
                envs.VLLM_PREFIX_CACHE_RETENTION_INTERVAL
            )
        self.coordinator = MooncakeStoreCoordinator(groups, **coordinator_kwargs)
        self.cache_namespace = store_topology_namespace(
            vllm_config,
            groups,
            tp_size=self.tp_size,
            pp_size=self.pp_size,
            pcp_size=self.pcp_size,
            dcp_size=self.dcp_size,
            block_size=self.block_size,
            hash_block_size=self.hash_block_size,
        )
        metadata = KeyMetadata(
            model_name=model.model.rstrip("/").split("/")[-1],
            tp_rank=key_tp_rank,
            pcp_rank=self.pcp_rank,
            dcp_rank=self.dcp_rank,
            pp_rank=self.pp_rank,
        )
        self.databases = [
            ChunkedTokenDatabase(
                dataclasses.replace(metadata, group_id=i),
                group.kv_cache_spec.block_size,
                self.hash_block_size,
            )
            for i, group in enumerate(groups)
        ]
        self._init_lookup_prefixes(num_kv_heads)

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

        recv_workers = max(1, int(extra.get("mtsc_store_load_workers", 2)))
        self.load_timeout = float(extra.get("mtsc_store_get_timeout_seconds", 180.0))
        if self.load_timeout <= 0:
            raise ValueError("mtsc_store_get_timeout_seconds must be positive")
        # One ordered PUT lane keeps each request's saved high-water mark
        # deterministic. GETs remain parallel.
        save_workers = 1
        self._load_pool = ThreadPoolExecutor(
            max_workers=recv_workers, thread_name_prefix="mtsc-store-get"
        )
        self._save_pool = ThreadPoolExecutor(
            max_workers=save_workers, thread_name_prefix="mtsc-store-put"
        )
        self._loads: dict[str, Future[set[int]]] = {}
        self._load_requests: dict[str, StoreRequest] = {}
        self._load_started_at: dict[str, float] = {}
        self._load_timed_out: set[str] = set()
        self._saves: dict[str, list[Future[None]]] = {}
        self._save_offsets: dict[str, int] = {}
        self._finished_save_requests: set[str] = set()
        self._errors: set[int] = set()
        self._lookup_server = (
            StoreLookupServer(self, vllm_config) if parallel.rank == 0 else None
        )

    def _init_lookup_prefixes(self, num_kv_heads: int) -> None:
        if self.dcp_size > 1:
            ranks = tuple(
                (tp, pcp, tp % self.dcp_size, pp)
                for pcp in range(self.pcp_size)
                for tp in range(self.tp_size)
                for pp in range(self.pp_size)
            )
        else:
            ranks = tuple(
                (tp, pcp, 0, pp)
                for pcp in range(self.pcp_size)
                for tp in range(min(self.tp_size, num_kv_heads))
                for pp in range(self.pp_size)
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

    def register(
        self,
        kv_caches: dict[
            str, torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]
        ],
    ) -> None:
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
            return store_mask(
                token_count, save_from, num_prompt_tokens=prompt_tokens
            )
        # vLLM 0.23 returns masks for [0, token_count). Convert them to the
        # suffix-relative masks expected by MTSC's incremental save path.
        masks = store_mask(token_count)
        return tuple(
            None
            if mask is None
            else mask[cdiv(save_from, database.block_size) :]
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
                relative < 0
                or relative >= len(chunk_mask)
                or not chunk_mask[relative]
            ):
                continue
            if chunk % put_step != put_step_rank:
                continue
            yield start, end, self._database_key(database, value)

    def lookup(self, token_count: int, hashes: Sequence[BlockHash]) -> int:
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

    def enqueue_load(self, request: StoreRequest) -> None:
        if request.request_id not in self._loads:
            self._loads[request.request_id] = self._load_pool.submit(
                self._load, request
            )
            self._load_requests[request.request_id] = request
            self._load_started_at[request.request_id] = time.monotonic()

    def _request_load_blocks(self, request: StoreRequest) -> set[int]:
        assert request.load is not None
        result: set[int] = set()
        for group, database in zip(request.block_ids, self.databases, strict=True):
            start = cdiv(request.load.local_tokens, database.block_size)
            end = min(cdiv(request.load.store_tokens, database.block_size), len(group))
            result.update(block for block in group[start:end] if block >= 0)
        return result

    def _load(self, request: StoreRequest) -> set[int]:
        assert request.load is not None
        failed: set[int] = set()
        masks = self.coordinator.load_mask(
            request.block_hashes, request.load.store_tokens
        )
        keys: list[str] = []
        addresses: list[list[int]] = []
        sizes: list[list[int]] = []
        block_ids: list[int] = []
        for group_index, database in enumerate(self.databases):
            for start, end, key in self._process_tokens(
                database,
                request.load.store_tokens,
                request.block_hashes,
                request.load.local_tokens,
            ):
                chunk = start // database.block_size
                if chunk >= len(masks[group_index]) or not masks[group_index][chunk]:
                    continue
                address, size, block_id = database.prepare_value(
                    start, end, request.block_ids[group_index]
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
                    request.request_id,
                    exc,
                )
        return failed

    def enqueue_save(self, request: StoreRequest, event: DeviceEvent | None) -> None:
        self._saves.setdefault(request.request_id, []).append(
            self._save_pool.submit(self._save, request, event)
        )

    def _save(self, request: StoreRequest, event: DeviceEvent | None) -> None:
        token_count = (
            request.token_count
            // self.coordinator.lcm_block_size
            * self.coordinator.lcm_block_size
        )
        save_from = max(
            request.save_from, self._save_offsets.get(request.request_id, 0)
        )
        if token_count <= save_from:
            return
        masks = self._store_masks(token_count, save_from, request.prompt_tokens)
        keys: list[str] = []
        addresses: list[list[int]] = []
        sizes: list[list[int]] = []
        for group_index, database in enumerate(self.databases):
            phase = (self.tp_rank + group_index) % self.put_step
            for start, end, key in self._process_tokens(
                database,
                token_count,
                request.block_hashes,
                save_from,
                chunk_mask=masks[group_index],
                put_step=self.put_step,
                put_step_rank=phase,
            ):
                address, size, _ = database.prepare_value(
                    start, end, request.block_ids[group_index]
                )
                keys.append(key)
                addresses.append(address)
                sizes.append(size)
        if not keys:
            self._save_offsets[request.request_id] = token_count
            return
        exists = self.store.batch_is_exist(keys)
        missing = [index for index, status in enumerate(exists) if status != 1]
        if not missing:
            self._save_offsets[request.request_id] = token_count
            return
        if event is not None:
            event.synchronize()
        result = self.store.batch_put_from_multi_buffers(
            [keys[i] for i in missing],
            [addresses[i] for i in missing],
            [sizes[i] for i in missing],
            self.replicate_config,
        )
        if any(status < 0 for status in result):
            logger.warning(
                "MTSC Store PUT partially failed: request_id=%s", request.request_id
            )
            return
        self._save_offsets[request.request_id] = token_count

    def poll(self, finished_request_ids: set[str]) -> tuple[set[str], set[str]]:
        recv_done: set[str] = set()
        for request_id, future in list(self._loads.items()):
            started_at = self._load_started_at.get(request_id, time.monotonic())
            if (
                not future.done()
                and request_id not in self._load_timed_out
                and time.monotonic() - started_at >= self.load_timeout
            ):
                # Mooncake's synchronous Store GET has no safe cancellation
                # primitive. Record the timeout now, but keep the request
                # fenced until the call returns so it cannot write late into a
                # recycled block.
                self._load_timed_out.add(request_id)
                logger.warning(
                    "MTSC Store GET timed out; waiting for safe fence: request_id=%s",
                    request_id,
                )
            if not future.done():
                continue
            try:
                errors = future.result()
                if request_id in self._load_timed_out:
                    request = self._load_requests[request_id]
                    errors.update(self._request_load_blocks(request))
                self._errors.update(errors)
            finally:
                del self._loads[request_id]
                self._load_requests.pop(request_id, None)
                self._load_started_at.pop(request_id, None)
                self._load_timed_out.discard(request_id)
                recv_done.add(request_id)
        self._finished_save_requests.update(
            request_id
            for request_id in finished_request_ids
            if request_id in self._saves
        )
        send_done: set[str] = set()
        for request_id in list(self._finished_save_requests):
            futures = self._saves.get(request_id, [])
            if not all(future.done() for future in futures):
                continue
            for future in futures:
                try:
                    future.result()
                except Exception as exc:  # noqa: BLE001 - background future boundary
                    logger.warning(
                        "MTSC Store PUT failed: request_id=%s error=%s", request_id, exc
                    )
            self._saves.pop(request_id, None)
            self._save_offsets.pop(request_id, None)
            self._finished_save_requests.remove(request_id)
            send_done.add(request_id)
        return send_done, recv_done

    def take_errors(self) -> set[int]:
        errors, self._errors = self._errors, set()
        return errors

    def wait_for_all_saves(self) -> None:
        for futures in self._saves.values():
            for future in futures:
                future.result()

    def finish_preempted_saves(self, request_ids: set[str]) -> None:
        """Fence GPU reads before vLLM reuses a preempted request's blocks."""
        for request_id in request_ids:
            futures = self._saves.pop(request_id, [])
            for future in futures:
                try:
                    future.result()
                except Exception as exc:  # noqa: BLE001 - background future boundary
                    logger.warning(
                        "MTSC Store PUT failed while fencing preemption: "
                        "request_id=%s error=%s",
                        request_id,
                        exc,
                    )
            self._save_offsets.pop(request_id, None)
            self._finished_save_requests.discard(request_id)

    def finish_preempted_loads(self, request_ids: set[str]) -> None:
        """Fence Store writes before a preempted request's blocks are reused."""
        for request_id in request_ids:
            future = self._loads.pop(request_id, None)
            if future is not None and not future.cancel():
                try:
                    future.result()
                except Exception as exc:  # noqa: BLE001 - I/O fence
                    logger.warning(
                        "MTSC Store GET failed while fencing preemption: "
                        "request_id=%s error=%s",
                        request_id,
                        exc,
                    )
            self._load_requests.pop(request_id, None)
            self._load_started_at.pop(request_id, None)
            self._load_timed_out.discard(request_id)

    def close(self) -> None:
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
