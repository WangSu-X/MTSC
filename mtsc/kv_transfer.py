"""Worker KV transfer contract and Mooncake Transfer Engine implementation."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import threading
import time
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import httpx
import msgspec
import torch
import uvicorn
import zmq
import zmq.asyncio
from fastapi import FastAPI, HTTPException
from packaging.version import InvalidVersion, Version
from pydantic import BaseModel
from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import (
    TransferTopology,
    get_current_attn_backends,
)
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_ip, make_zmq_path
from vllm.v1.kv_cache_interface import MambaSpec, MLAAttentionSpec, SlidingWindowMLASpec

from .protocol import (
    KVTransferRequest,
    KVTransferResponse,
    KVTransferSchema,
    KVTransferStatus,
)
from .utils import (
    BlockIds,
    KVLayoutAdapter,
    cache_tensors,
    effective_tp,
    is_npu_platform,
    npu_kv_nz_enabled,
    npu_registration_regions,
)


def _require_tcp_write_ack(protocol: str) -> None:
    """Legacy TCP WRITE completes before remote memory is safe to consume."""
    if protocol != "tcp":
        return
    requirement = "mooncake-transfer-engine>=0.3.13.post1"
    try:
        installed = Version(version("mooncake-transfer-engine"))
    except (PackageNotFoundError, InvalidVersion) as exc:
        raise RuntimeError(
            f"MTSC TCP requires {requirement} with receiver ACK"
        ) from exc
    if installed < Version("0.3.13.post1"):
        raise RuntimeError(
            f"MTSC TCP requires {requirement} on both P and D; found {installed}. "
            "Legacy TCP reports completion before remote GPU writes finish."
        )
    if os.environ.get("MC_TCP_PROTO") == "1":
        raise RuntimeError(
            "MTSC TCP requires receiver ACK; unset MC_TCP_PROTO=1 "
            "to enable acknowledged protocol v2."
        )


@dataclass(frozen=True)
class SendEvent:
    """Publish the request's ready source KV for P-D transfer.

    block_ids is the full source block table, grouped and ordered by KV cache
    group. All listed KV must be safe to read when send() is called. D selects
    a suffix by providing its destination block table in RecvEvent.
    Source blocks remain valid and unchanged until the send result is polled
    or preempt() returns.
    """

    request_id: str
    transfer_id: str
    block_ids: BlockIds


@dataclass(frozen=True)
class RecvEvent:
    """Request a source KV suffix into D's allocated destination blocks.

    block_ids contains only blocks to receive, grouped and ordered by KV cache
    group; it excludes blocks already populated locally. For each group, P
    copies the last len(destination_group) source blocks in order. Empty groups
    require no transfer. Source and destination tables describe the same prefix
    endpoint with compatible group/block layouts; arbitrary ranges are unsupported.
    The backend resolves matching P workers through bootstrap_addr and publishes
    destination memory information derived from register().
    """

    request_id: str
    transfer_id: str
    block_ids: BlockIds
    bootstrap_addr: str
    remote_engine_id: str
    remote_dp_rank: int


@dataclass(frozen=True)
class TransferResult:
    """Terminal result for one local send or recv, identified by request_id.

    error=None indicates success. A send result means source KV is no longer
    being read; a recv result means destination KV is no longer being written.
    Only a successful recv guarantees all requested destination KV is usable.
    Failed recv block IDs are exposed separately through take_errors().
    """

    request_id: str
    error: str | None = None


@dataclass
class TransferPollResult:
    sends: list[TransferResult] = field(default_factory=list)
    recvs: list[TransferResult] = field(default_factory=list)


class KVTransfer(ABC):
    """D-initiated P-D direct transfer, implemented by P writing into D.

    P: register -> prefill completes -> send(ready source KV) -> poll().
    D: register -> recv(destination + bootstrap) -> poll(recv completion).
    send() and the remote recv request may arrive in either order. Transfer
    starts only after send() publishes ready KV and a receiver request is present.
    P writes the requested KV, then notifies D of completion or failure.

    request_id identifies local work and results, without a task handle.
    transfer_id matches P and D requests, whose local request IDs may differ.
    It is a unique cross-worker attempt ID supplied by the connector; retries
    use a new transfer_id so delayed messages cannot match new block ownership.
    It is not a backend-generated task ID requiring a result-to-request map.

    At most one uncollected send and one uncollected recv per local request;
    duplicate submissions of the same kind raise ValueError. Events are
    immutable metadata snapshots. Layout, topology, endpoint registration,
    protocol timeouts and peer discovery caching belong to backend setup.
    The backend validates peer identity, layout and block-table compatibility and
    derives TP/PP pairing and complete destination coverage before success.
    Callers serialize public calls; the backend synchronizes background work.

    This interface owns transfer and memory-access fences. Request scheduling,
    block allocation, persistence and cache lookup belong to the connector
    and KVCachePool. There is no request-end notification or public state machine.
    """

    @abstractmethod
    def register(
        self,
        kv_caches: dict[
            str, torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]
        ],
    ) -> None:
        """Register local KV memory on both P and D before send/recv.

        On P, also expose the worker control endpoint through bootstrap.
        Memory descriptors and transport addresses are backend-owned; callers
        only provide block IDs. Tensors stay alive until close(). Re-registering
        while transfers are pending is unsupported.
        """

    @abstractmethod
    def prepare(self, request_id: str, transfer_id: str) -> None:
        """Associate an unready P session so it can be cancelled before send.

        This registers identity only; it never publishes source memory.
        Repeated registration of the same association is harmless.
        """

    @abstractmethod
    def cancel(self, request_id: str, transfer_id: str) -> None:
        """Permanently retire a P session, including one not yet prepared.

        Wake early receivers with failure and fence any active source reads.
        Unlike an unready P preemption, this session cannot be resumed.
        """

    @abstractmethod
    def send(self, send_event: SendEvent) -> None:
        """Publish ready source KV without waiting for a receiver or I/O.

        Calling send() marks this source ready; the caller ensures device writes
        are complete and visible to the transport before calling. P must
        retain source blocks while waiting and during writes. Fan-out to D
        workers is managed internally: publish one send result only after all
        expected targets are terminal and all source reads have stopped.
        A source with no receiver may expire according to backend policy;
        retire its transfer_id before reporting failure so late pulls cannot
        access released blocks.
        """

    @abstractmethod
    def recv(self, recv_event: RecvEvent) -> None:
        """Submit a receive request without waiting for discovery or I/O.

        Resolve matching P workers through bootstrap_addr, scoped by remote
        engine and DP identity. Send them the transfer_id, destination block
        table and local memory descriptors. P waits for send() and writes
        directly into D; D waits for every required worker's terminal response.
        Report success only after all destination KV is ready for local use,
        including any required device synchronization or layout conversion.
        """

    def recv_batch(self, recv_events: list[RecvEvent]) -> None:
        """Submit the receives made ready by one worker step."""
        for event in recv_events:
            self.recv(event)

    @abstractmethod
    def poll(self) -> TransferPollResult:
        """Drain terminal send/recv results without waiting for I/O.

        Return each result once, including failures. Source block release is
        authorized by send completion; destination reuse by recv completion.
        If blocks are used by other operations, those must also be fenced.
        Publish failed recv block IDs with their result and drain take_errors()
        in the same polling cycle before block/request reuse.
        """

    @abstractmethod
    def take_errors(self) -> set[int]:
        """Drain invalid physical local block IDs from polled recv failures.

        Non-blocking. Report unusable blocks from RecvEvent.block_ids; if partial
        coverage cannot be proved, invalidate all listed destination blocks.
        Preserve the existing local prefix, which is not in that table. Send failures
        do not invalidate local source KV. The connector reports these IDs
        through get_block_ids_with_load_errors() for scheduler recovery.
        """

    @abstractmethod
    def preempt(self, request_id: str) -> None:
        """Cancel and fence the request's local send/recv before block reuse.

        Unpublished P sessions have no source memory and may retain early D
        waiters for resumed prefill. Published send and recv attempts are retired;
        retries of those attempts need a new transfer_id. Cancel queued work,
        and fence active transfers. On D, remote P writes must stop before
        returning; a local timeout or discarded response alone is insufficient.
        May block if safe cancellation is unavailable. On return no transfer
        accesses the request's local blocks; pending results, block errors and
        local bookkeeping are cleared. Calls with no tracked request are safe.
        """

    @abstractmethod
    def close(self) -> None:
        """Stop accepting work and fence local reads and remote writes.

        Notify pending peers as needed and release registered memory/transport
        resources only after transfers stop accessing them. Idempotent;
        subsequent send/recv submissions raise RuntimeError.
        """


logger = init_logger(__name__)
logger.setLevel(logging.INFO)


def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


def _bootstrap_address(config: VllmConfig) -> tuple[str, int]:
    parallel = config.parallel_config
    if parallel.local_engines_only:
        host = "127.0.0.1"
    elif parallel.nnodes_within_dp > 1:
        host = parallel.master_addr
    else:
        host = parallel.data_parallel_master_ip
    return host, envs.VLLM_MOONCAKE_BOOTSTRAP_PORT


def _launch_bootstrap(config: VllmConfig) -> bool:
    parallel = config.parallel_config
    if get_tensor_model_parallel_rank() != 0 or get_pp_group().rank_in_group != 0:
        return False
    if parallel.local_engines_only:
        return parallel.data_parallel_rank_local == 0
    return parallel.data_parallel_index == 0


class WorkerRegistration(BaseModel):
    engine_id: str
    dp_rank: int
    tp_rank: int
    tp_size: int
    pp_rank: int
    pp_size: int
    address: str


class BootstrapServer:
    """Small registry owned by MTSC; no retry/routing policy lives here."""

    def __init__(self, port: int) -> None:
        self.workers: dict[int, dict[str, Any]] = {}
        app = FastAPI()
        app.post("/register")(self.register)
        app.get("/query")(self.query)
        self.server = uvicorn.Server(
            uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
        )
        self.thread = threading.Thread(
            target=self.server.run, name="mtsc-bootstrap", daemon=True
        )

    def start(self) -> None:
        self.thread.start()
        while not self.server.started:
            time.sleep(0.05)

    async def register(self, payload: WorkerRegistration) -> dict[str, str]:
        entry = self.workers.setdefault(
            payload.dp_rank,
            {
                "engine_id": payload.engine_id,
                "tp_size": payload.tp_size,
                "pp_size": payload.pp_size,
                "worker_addr": {},
            },
        )
        if entry["engine_id"] != payload.engine_id:
            raise HTTPException(400, "engine_id mismatch")
        if entry["tp_size"] != payload.tp_size or entry["pp_size"] != payload.pp_size:
            raise HTTPException(400, "worker topology mismatch")
        if not 0 <= payload.tp_rank < payload.tp_size:
            raise HTTPException(400, "invalid tp_rank")
        if not 0 <= payload.pp_rank < payload.pp_size:
            raise HTTPException(400, "invalid pp_rank")
        tp_entry = entry["worker_addr"].setdefault(payload.tp_rank, {})
        if payload.pp_rank in tp_entry:
            if tp_entry[payload.pp_rank] == payload.address:
                return {"status": "ok"}
            raise HTTPException(400, "worker rank already registered")
        tp_entry[payload.pp_rank] = payload.address
        return {"status": "ok"}

    async def query(self) -> dict[int, dict[str, Any]]:
        return self.workers

    def close(self) -> None:
        if self.server.started:
            self.server.should_exit = True
            self.thread.join(timeout=5)


@dataclass(frozen=True)
class TransferRegion:
    layer_name: str
    layer_index: int
    group_index: int
    base_address: int
    block_length: int
    kv_block_length: int


@dataclass
class _NPUTransferTopology:
    """The subset of TransferTopology needed by the Ascend direct path.

    Ascend attention backends expose split K/V tensors whose backend shape has
    a leading K/V dimension.  The upstream CUDA-oriented TransferTopology
    validates a single blocks-first tensor, so it cannot be constructed for
    that layout.
    """

    tp_rank: int
    tp_size: int
    block_size: int
    is_mla: bool
    total_num_kv_heads: int

    def handshake_target_ranks(self, remote_tp_size: int) -> list[int]:
        if self.tp_size >= remote_tp_size:
            if self.tp_size % remote_tp_size:
                raise ValueError("P/D TP sizes must have an integer ratio")
            return [self.tp_rank // (self.tp_size // remote_tp_size)]
        if remote_tp_size % self.tp_size:
            raise ValueError("P/D TP sizes must have an integer ratio")
        ratio = remote_tp_size // self.tp_size
        return [self.tp_rank * ratio + offset for offset in range(ratio)]


def kv_slice_plan(p_rank, p_size, d_rank, d_size, p_bytes, d_bytes, kv_heads, is_mla):
    """Return copy flag, byte slices and effective source/target shard counts.

    Rank discovery uses physical TP ranks. Byte slicing uses unique KV shards,
    selecting one sender per replicated shard within each D rank's peer set.
    """
    if max(p_size, d_size) % min(p_size, d_size):
        raise ValueError("P/D TP sizes must have an integer ratio")
    p_shards = effective_tp(p_size, kv_heads, is_mla)
    d_shards = effective_tp(d_size, kv_heads, is_mla)
    p_copies, d_copies = p_size // p_shards, d_size // d_shards
    if p_size >= d_size:
        first_peer = d_rank * (p_size // d_size)
        representative = max(first_peer, (p_rank // p_copies) * p_copies)
        if p_rank != representative:
            return False, 0, 0, 0, p_shards, d_shards
    p_shard, d_shard = p_rank // p_copies, d_rank // d_copies
    if p_shards >= d_shards:
        ratio = p_shards // d_shards
        if p_shard // ratio != d_shard:
            raise ValueError("P/D KV shard pairing mismatch")
        return True, 0, (p_shard % ratio) * p_bytes, p_bytes, p_shards, d_shards
    ratio = d_shards // p_shards
    if d_shard // ratio != p_shard:
        raise ValueError("P/D KV shard pairing mismatch")
    return True, (d_shard % ratio) * d_bytes, 0, d_bytes, p_shards, d_shards


@dataclass
class _Session:
    request_id: str
    transfer_id: str
    block_ids: BlockIds = ()
    ready: threading.Event = field(default_factory=threading.Event)
    ready_waiters: set[asyncio.Future] = field(default_factory=set)
    published: bool = False
    abort: bool = False
    expected: int = 0
    completed: int = 0
    terminal: int = 0
    active_writes: int = 0
    expires_at: float = float("inf")
    peer: tuple | None = None
    targets: dict[tuple, asyncio.Future] = field(default_factory=dict)
    descriptors: dict[tuple, tuple] = field(default_factory=dict)


class MooncakeKVTransfer(KVLayoutAdapter, KVTransfer):
    """Own TE registration, parallel mapping, WRITE and fenced sessions.

    Retired attempt IDs prevent late pulls from reaching recycled memory.
    Unpublished P sessions survive preemption so early D waiters can resume.
    """

    def __init__(self, config: VllmConfig, kv_cache_config: Any) -> None:
        try:
            from mooncake.engine import TransferEngine
        except ImportError as exc:
            raise ImportError("Mooncake TransferEngine bindings are required") from exc
        assert config.kv_transfer_config is not None
        transfer = config.kv_transfer_config
        assert transfer.engine_id is not None
        self.num_blocks = config.cache_config.num_gpu_blocks
        self._num_blocks_resolved = config.cache_config.num_gpu_blocks is not None
        self.engine_id = transfer.engine_id
        self.is_producer = transfer.kv_role == "kv_producer"
        extra = transfer.kv_connector_extra_config
        self.timeout = float(
            extra.get(
                "mtsc_pd_timeout_seconds", envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT
            )
        )
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("mtsc_pd_timeout_seconds must be finite and positive")
        max_workers = max(1, int(extra.get("num_workers", 10)))
        self.num_sender_tasks = max_workers * 2
        self.device_id = torch.accelerator.current_device_index()
        current_platform.set_device(self.device_id)
        default_protocol = "ascend" if is_npu_platform() else "rdma"
        protocol = extra.get("mooncake_protocol", default_protocol)
        _require_tcp_write_ack(protocol)
        self.protocol = protocol
        self._tcp_write_locks: dict[str, asyncio.Lock] = {}
        self.engine = TransferEngine()
        self.hostname = get_ip()
        ret = self.engine.initialize(
            self.hostname,
            "P2PHANDSHAKE",
            protocol,
            extra.get("device_name", ""),
        )
        if ret != 0:
            raise RuntimeError(f"Mooncake TransferEngine initialization failed: {ret}")
        self.rpc_port = self.engine.get_rpc_port()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.pp_rank = get_pp_group().rank_in_group
        self.pp_size = config.parallel_config.pipeline_parallel_size
        parallel = config.parallel_config
        self.dp_rank = (
            parallel.data_parallel_rank_local
            if parallel.local_engines_only
            else parallel.data_parallel_index
        )
        model = config.model_config
        topology_args = {
            "tp_rank": self.tp_rank,
            "tp_size": self.tp_size,
            "block_size": config.cache_config.block_size,
            "is_mla": model.use_mla,
            "total_num_kv_heads": model.get_total_num_kv_heads(),
        }
        if is_npu_platform():
            self.topology = _NPUTransferTopology(**topology_args)
        else:
            self.topology = TransferTopology(
                **topology_args,
                engine_id=self.engine_id,
                is_mamba=kv_cache_config.has_mamba_layers,
                attn_backends=get_current_attn_backends(config),
            )
        self.npu_kv_nz = npu_kv_nz_enabled(config)
        if is_npu_platform():
            cache_layout = "npu-nz" if self.npu_kv_nz else "npu-normal"
        else:
            cache_layout = "mla" if model.use_mla else "hnd"
        self.schema = KVTransferSchema(
            # Versions 6/7 add batched pulls and incremental terminal responses.
            # Older receivers cannot fence a multi-response transfer.
            topology_version=7 if protocol == "tcp" else 6,
            pcp_size=getattr(parallel, "prefill_context_parallel_size", 1),
            dcp_size=getattr(parallel, "decode_context_parallel_size", 1),
            model_id=str(model.model),
            model_revision=str(getattr(model, "revision", None) or ""),
            cache_dtype=str(
                model.dtype
                if config.cache_config.cache_dtype == "auto"
                else config.cache_config.cache_dtype
            ),
            cache_layout=cache_layout,
            block_size=config.cache_config.block_size,
            is_mla=model.use_mla,
        )
        self.kv_caches: dict[
            str, torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]
        ] = {}
        self.layer_specs: dict[str, Any] = {}
        self.layer_groups: dict[str, int] = {}
        for group_index, group in enumerate(kv_cache_config.kv_cache_groups):
            specs = getattr(group.kv_cache_spec, "kv_cache_specs", {})
            for layer in group.layer_names:
                self.layer_specs[layer] = specs.get(layer, group.kv_cache_spec)
                self.layer_groups[layer] = group_index
        self.regions: list[TransferRegion] = []
        self._registered_storage: list[int] = []
        self._sources: dict[str, _Session] = {}
        self._source_lock = threading.Lock()
        self._finished_send: set[str] = set()
        self._failed_recv: set[str] = set()
        self._result_lock = threading.Lock()
        self._remote_workers: dict[tuple[str, str, int], dict[int, dict[int, str]]] = {}
        self._receive_futures: dict[str, Future[None]] = {}
        self._receive_lock = threading.Lock()
        self._encoder = msgspec.msgpack.Encoder()
        self._request_decoder = msgspec.msgpack.Decoder(KVTransferRequest)
        self._response_decoder = msgspec.msgpack.Decoder(KVTransferResponse)
        self._ctx = zmq.asyncio.Context()
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=_run_loop, args=(self._loop,), name="mtsc-pd", daemon=True
        )
        self._loop_thread.start()
        self._listener_future = None
        self._serve_tasks: set[asyncio.Task[None]] = set()
        self.sender_worker_queue: asyncio.Queue[tuple[bytes, bytes]] = asyncio.Queue()
        self._send_pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="mtsc-te-write",
            initializer=lambda: current_platform.set_device(self.device_id),
        )
        self._bootstrap = None
        if self.is_producer:
            host, port = _bootstrap_address(config)
            self._registration_url = make_zmq_path("http", host, port) + "/register"
            if _launch_bootstrap(config):
                self._bootstrap = BootstrapServer(port)
                self._bootstrap.start()
        self._source_changed = threading.Condition(self._source_lock)
        self._retired: set[str] = set()
        self._prepared: dict[str, str] = {}
        self._pending_sends: set[str] = set()
        self._recv_events: dict[str, RecvEvent] = {}
        self._send_failures: dict[str, str] = {}
        self._invalid: dict[str, set[int]] = {}
        self._registered = False
        self._closed = False

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("KV transfer is closed")

    def register(self, kv_caches) -> None:
        self._check_open()
        if self._registered:
            raise RuntimeError("KV transfer memory is already registered")
        self.kv_caches = kv_caches
        if not self._num_blocks_resolved:
            for raw in kv_caches.values():
                for tensor in cache_tensors(raw):
                    if tensor.ndim > 0 and tensor.shape[0] > 0:
                        self.num_blocks = int(tensor.shape[0])
                        self._num_blocks_resolved = True
                        break
                if self._num_blocks_resolved:
                    break
            if not self._num_blocks_resolved:
                raise ValueError(
                    "num_gpu_blocks must be initialized before MooncakeKVTransfer"
                )
            logger.info(
                "MTSC KVTransfer inferred num_blocks=%d from KV cache tensors",
                self.num_blocks,
            )
        is_npu = is_npu_platform()
        if is_npu:
            pointers, lengths = npu_registration_regions(kv_caches)
        else:
            pointers, lengths = [], []
        seen: set[int] = set()
        for layer_name, raw in kv_caches.items():
            spec = self.layer_specs.get(layer_name)
            if spec is None:
                continue
            if isinstance(spec, MambaSpec):
                if not isinstance(raw, (list, tuple)) or not raw:
                    raise TypeError(
                        f"Mamba cache {layer_name!r} must be a non-empty sequence"
                    )
                # The convolution state is transferred. The SSM state is
                # recomputed from the final prompt token on Decode.
                cache_list = [raw[0]]
            elif is_npu:
                cache_list = cache_tensors(raw)
            else:
                if not isinstance(raw, torch.Tensor):
                    raise TypeError(f"Attention cache {layer_name!r} must be a tensor")
                cache_list = self.topology.get_transfer_cache_regions(raw, spec)
            if isinstance(cache_list, torch.Tensor):
                cache_list = [cache_list]
            for cache in cache_list:
                storage = cache.untyped_storage()
                if not is_npu and storage.data_ptr() not in seen:
                    seen.add(storage.data_ptr())
                    pointers.append(storage.data_ptr())
                    lengths.append(storage.nbytes())
                block_length = cache.stride(0) * cache.element_size()
                if is_npu:
                    # Ascend exposes K/V (or MLA nope/rope) as independent
                    # blocks-first tensors. Each region transfers its own page.
                    kv_length = block_length
                elif isinstance(spec, (MLAAttentionSpec, SlidingWindowMLASpec)):
                    kv_length = spec.page_size_bytes
                elif self.topology.virtually_split_kv_in_blocks and not isinstance(
                    spec, MambaSpec
                ):
                    kv_length = block_length // 2
                else:
                    kv_length = block_length
                self.regions.append(
                    TransferRegion(
                        layer_name,
                        extract_layer_index(layer_name),
                        self.layer_groups[layer_name],
                        cache.data_ptr(),
                        block_length,
                        kv_length,
                    )
                )
                if (
                    not is_npu
                    and self.topology.virtually_split_kv_in_blocks
                    and not isinstance(
                        spec, (MambaSpec, MLAAttentionSpec, SlidingWindowMLASpec)
                    )
                ):
                    # Blocks-first caches pack K and V into one page. Publish
                    # both halves as regions so TP slicing copies each half.
                    self.regions.append(
                        TransferRegion(
                            layer_name,
                            extract_layer_index(layer_name),
                            self.layer_groups[layer_name],
                            cache.data_ptr() + kv_length,
                            block_length,
                            kv_length,
                        )
                    )
        if not pointers:
            raise RuntimeError("No PD KV regions registered")
        if is_npu:
            # Mooncake's Ascend wrapper registers HCCL regions one by one;
            # batch_register_memory is the CUDA/RDMA path.
            for pointer, length in zip(pointers, lengths, strict=True):
                ret = self.engine.register_memory(pointer, length)
                if ret != 0:
                    raise RuntimeError(f"Mooncake TE memory registration failed: {ret}")
                self._registered_storage.append(pointer)
        else:
            ret = self.engine.batch_register_memory(pointers, lengths)
            if ret != 0:
                raise RuntimeError(f"Mooncake TE memory registration failed: {ret}")
            self._registered_storage = pointers
        if self.is_producer:
            ready = threading.Event()
            self._listener_future = asyncio.run_coroutine_threadsafe(
                self._listen(ready), self._loop
            )
            if not ready.wait(timeout=self.timeout):
                self._listener_future.cancel()
                raise TimeoutError("MTSC P listener startup timed out")
            if self._listener_future.done():
                # Do not turn bootstrap/listener initialization errors into a
                # silent, permanent engine startup hang.
                self._listener_future.result()
        self._registered = True

    def prepare(self, request_id: str, transfer_id: str) -> None:
        self._check_open()
        if not self.is_producer:
            raise ValueError("Only P can prepare a source session")
        with self._source_lock:
            previous = self._prepared.get(request_id)
            if previous is not None and previous != transfer_id:
                raise ValueError("Request already belongs to another transfer attempt")
            if transfer_id in self._retired:
                raise ValueError("Transfer attempt is retired")
            source = self._sources.setdefault(
                transfer_id, _Session(request_id, transfer_id)
            )
            if source.request_id and source.request_id != request_id:
                raise ValueError("Transfer attempt belongs to another request")
            source.request_id = request_id
            self._prepared[request_id] = transfer_id

    def send(self, event: SendEvent) -> None:
        self._check_open()
        if not self._registered:
            raise RuntimeError("Register KV memory before send")
        if event.request_id in self._pending_sends:
            raise ValueError("Uncollected send for request")
        self.prepare(event.request_id, event.transfer_id)
        with self._source_lock:
            source = self._sources[event.transfer_id]
            source.block_ids = event.block_ids
            source.published = True
            source.expires_at = time.monotonic() + self.timeout
            self._pending_sends.add(event.request_id)
            self._signal_ready_locked(source)
            if source.expected > 0 and source.terminal >= source.expected:
                self._retire_locked(source)

    def recv(self, event: RecvEvent) -> None:
        self.recv_batch([event])

    def recv_batch(self, events: list[RecvEvent]) -> None:
        self._check_open()
        if not self._registered:
            raise RuntimeError("Register KV memory before recv")
        if not events:
            return
        request_ids: set[str] = set()
        transfer_ids: set[str] = set()
        groups: defaultdict[tuple[str, str, int], list[RecvEvent]] = defaultdict(list)
        for event in events:
            if (
                event.request_id in self._recv_events
                or event.request_id in self._invalid
                or event.request_id in request_ids
            ):
                raise ValueError("Uncollected recv or block errors for request")
            if event.transfer_id in self._retired or event.transfer_id in transfer_ids:
                raise ValueError("Transfer attempt is retired or duplicated")
            request_ids.add(event.request_id)
            transfer_ids.add(event.transfer_id)
            groups[
                (event.bootstrap_addr, event.remote_engine_id, event.remote_dp_rank)
            ].append(event)
        futures = {event.request_id: Future() for event in events}
        with self._receive_lock:
            self._receive_futures.update(futures)
        self._recv_events.update((event.request_id, event) for event in events)
        try:
            asyncio.run_coroutine_threadsafe(
                self._receive_groups(groups, futures), self._loop
            )
        except Exception:
            with self._receive_lock:
                for event in events:
                    self._receive_futures.pop(event.request_id, None)
                    self._recv_events.pop(event.request_id, None)
            raise

    @staticmethod
    def _signal_ready_locked(source: _Session) -> None:
        source.ready.set()

        def wake(waiter):
            if not waiter.done():
                waiter.set_result(None)

        for waiter in source.ready_waiters:
            waiter.get_loop().call_soon_threadsafe(wake, waiter)

    async def _wait_source_ready(self, source: _Session) -> bool:
        with self._source_lock:
            if source.ready.is_set():
                return True
            waiter = asyncio.get_running_loop().create_future()
            source.ready_waiters.add(waiter)
        try:
            await waiter
            return True
        finally:
            with self._source_lock:
                source.ready_waiters.discard(waiter)

    def _retire_locked(self, source: _Session, *, report: bool = True) -> None:
        if source.active_writes:
            raise RuntimeError("Cannot retire a source with active WRITEs")
        source.abort = True
        self._signal_ready_locked(source)
        self._retired.add(source.transfer_id)
        self._sources.pop(source.transfer_id, None)
        if report and source.published:
            with self._result_lock:
                if (
                    source.completed < source.expected
                    or source.completed != source.terminal
                ):
                    self._send_failures[source.request_id] = (
                        "Transfer target failed or timed out"
                    )
                self._finished_send.add(source.request_id)

    async def _serve(self, identity, payload, socket) -> None:
        waiting = {}
        owners = {}
        existing_targets = []
        try:
            request = self._request_decoder.decode(payload)
            target = (
                request.engine_id,
                request.dp_rank,
                request.tp_rank,
                request.pp_rank,
            )
            # Snapshot the entire batch before validation can fail or this
            # handler creates targets of its own. A batch-wide failure must
            # fence existing WRITEs regardless of request iteration order.
            with self._source_lock:
                for transfer_id in request.requests:
                    source = self._sources.get(transfer_id)
                    if source is not None:
                        previous = source.targets.get(target)
                        if previous is not None:
                            existing_targets.append(previous)
            if self._closed:
                raise RuntimeError("P transfer is closing")
            logger.info(
                "MTSC P _serve got request: transfer_ids=%s from engine=%s dp=%d",
                list(request.requests.keys()),
                request.engine_id,
                request.dp_rank,
            )
            if request.schema != self.schema:
                raise ValueError("P/D transfer schema mismatch")
            if (
                request.remote_engine_id != self.engine_id
                or request.remote_dp_rank != self.dp_rank
            ):
                raise ValueError("P engine/DP identity mismatch")
            if not request.engine_id:
                raise ValueError("D engine identity is missing")
            if request.destination_num_blocks <= 0:
                raise ValueError("D registered block capacity is missing")
            ranks = self.topology.handshake_target_ranks(request.tp_size)
            if request.tp_rank not in ranks:
                raise ValueError("D TP rank is not paired with this P rank")
            if request.pp_size <= 0 or not 0 <= request.pp_rank < request.pp_size:
                raise ValueError("Invalid D PP identity")
            if self.pp_size == request.pp_size and self.pp_rank != request.pp_rank:
                raise ValueError("D PP rank is not paired with this P rank")
            peer = (
                request.engine_id,
                request.dp_rank,
                request.tp_size,
                request.pp_size,
            )
            for transfer_id, (request_id, _) in request.requests.items():
                descriptor = (
                    request_id,
                    request.hostname,
                    request.rpc_port,
                    tuple(tuple(ids) for ids in request.requests[transfer_id][1]),
                    tuple(request.region_base_addresses),
                    tuple(request.block_lengths),
                    tuple(request.kv_block_lengths),
                    tuple(request.layer_names),
                    tuple(request.layer_indices),
                    tuple(request.group_indices),
                )
                with self._source_lock:
                    if transfer_id in self._retired:
                        source = None
                    else:
                        source = self._sources.setdefault(
                            transfer_id, _Session("", transfer_id)
                        )
                        if source.peer is not None and source.peer != peer:
                            raise ValueError(
                                "Transfer attempt belongs to another D replica/topology"
                            )
                        source.peer = peer
                        source.expected = len(ranks) * (
                            1 if self.pp_size == request.pp_size else request.pp_size
                        )
                        previous = source.targets.get(target)
                        if (
                            previous is not None
                            and source.descriptors[target] != descriptor
                        ):
                            raise ValueError(
                                "Duplicate target changed destination memory"
                            )
                        if previous is None:
                            previous = asyncio.get_running_loop().create_future()
                            source.targets[target] = previous
                            source.descriptors[target] = descriptor
                            owner = True
                        else:
                            owner = False
                if source is None:
                    task = asyncio.get_running_loop().create_future()
                    task.set_result((False, set()))
                elif owner:
                    owners[transfer_id] = (source, previous)
                    task = asyncio.create_task(self._wait_source_ready(source))
                else:
                    # A duplicate pull joins the original fence. It must never
                    # write again or increment source completion twice.
                    task = asyncio.shield(previous)
                waiting[task] = transfer_id

            while waiting:
                done, _ = await asyncio.wait(
                    waiting,
                    timeout=self.timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                timed_out = not done
                if timed_out:
                    # Like MooncakeConnector, abort the pending ready waits
                    # when a whole round times out. Duplicate pulls still join
                    # their original WRITE fence before reporting a result.
                    done = set(waiting)
                    for task, tid in waiting.items():
                        if tid in owners:
                            task.cancel()
                    await asyncio.gather(*done, return_exceptions=True)
                results = {}
                ready_sources = {}
                for task in done:
                    transfer_id = waiting.pop(task)
                    if transfer_id in owners:
                        source, previous = owners[transfer_id]
                        if not timed_out and task.result():
                            ready_sources[transfer_id] = source
                        else:
                            self._finish_target(source, False, False)
                            results[transfer_id] = (False, set())
                    else:
                        results[transfer_id] = task.result()
                if ready_sources:
                    results.update(
                        await self._write_ready_targets(ready_sources, request)
                    )
                for transfer_id, result in results.items():
                    owned = owners.pop(transfer_id, None)
                    if owned is not None:
                        owned[1].set_result(result)
                completed = [tid for tid, (ok, _) in results.items() if ok]
                failed = [tid for tid, (ok, _) in results.items() if not ok]
                response = KVTransferResponse(
                    KVTransferStatus.IN_PROGRESS
                    if waiting
                    else KVTransferStatus.COMPLETE,
                    completed or None,
                    failed or None,
                    "transfer failed" if failed else None,
                    {tid: sorted(results[tid][1]) for tid in completed} or None,
                )
                await socket.send_multipart((identity, self._encoder.encode(response)))
            if not request.requests:
                await socket.send_multipart(
                    (
                        identity,
                        self._encoder.encode(
                            KVTransferResponse(KVTransferStatus.COMPLETE)
                        ),
                    )
                )
        except Exception as exc:  # noqa: BLE001 - control response boundary
            logger.warning("MTSC P _serve failed: error=%s", exc)
            # A malformed sibling must not turn a duplicate's in-flight WRITE
            # into an early terminal response authorizing destination reuse.
            if existing_targets:
                await asyncio.gather(
                    *(asyncio.shield(future) for future in existing_targets),
                    return_exceptions=True,
                )
            response = KVTransferResponse(
                KVTransferStatus.FAILED, error_message=str(exc)
            )
            await socket.send_multipart((identity, self._encoder.encode(response)))
        finally:
            for task in waiting:
                task.cancel()
            if waiting:
                await asyncio.gather(*waiting, return_exceptions=True)
            for source, previous in owners.values():
                self._finish_target(source, False, False)
                if not previous.done():
                    previous.set_result((False, set()))

    async def _write_target(self, source, request, transfer_id):
        try:
            await asyncio.wait_for(self._wait_source_ready(source), self.timeout)
        except asyncio.TimeoutError:
            self._finish_target(source, False, False)
            return False, set()
        return (await self._write_ready_targets({transfer_id: source}, request))[
            transfer_id
        ]

    async def _write_ready_targets(self, sources, request):
        results = {tid: (False, set()) for tid in sources}
        with self._source_lock:
            active = {
                tid: source
                for tid, source in sources.items()
                if not source.abort and source.transfer_id not in self._retired
            }
            for source in active.values():
                source.active_writes += 1
        try:
            if active:
                results.update(await self._write_batch(active, request))
        except Exception as exc:  # noqa: BLE001 - native WRITE result boundary
            logger.warning(
                "MTSC Transfer WRITE failed: transfer_ids=%s error=%s",
                list(active),
                exc,
            )
        finally:
            for tid, source in sources.items():
                self._finish_target(source, tid in active, results[tid][0])
        return results

    def _finish_target(self, source, active, ok):
        with self._source_changed:
            if active:
                source.active_writes -= 1
            source.terminal += 1
            source.completed += int(ok)
            if source.active_writes == 0:
                self._source_changed.notify_all()
                if (
                    self._sources.get(source.transfer_id) is source
                    and source.published
                    and source.terminal >= source.expected
                ):
                    self._retire_locked(source)

    def poll(self) -> TransferPollResult:
        result = TransferPollResult()
        with self._source_lock:
            for source in list(self._sources.values()):
                if (
                    source.published
                    and source.active_writes == 0
                    and source.expires_at < time.monotonic()
                ):
                    # No receiver is a failure too, although source memory is safe.
                    source.expected = max(source.expected, source.completed + 1)
                    self._retire_locked(source)
        with self._receive_lock:
            terminal = {
                rid for rid, future in self._receive_futures.items() if future.done()
            }
        with self._result_lock:
            for request_id in self._finished_send:
                result.sends.append(
                    TransferResult(
                        request_id,
                        self._send_failures.pop(request_id, None),
                    )
                )
                self._pending_sends.discard(request_id)
                self._prepared.pop(request_id, None)
            self._finished_send.clear()
            for request_id in terminal:
                event = self._recv_events.pop(request_id, None)
                if event is None:
                    continue
                failed = request_id in self._failed_recv
                if failed:
                    self._invalid[request_id] = {
                        block for ids in event.block_ids for block in ids if block >= 0
                    }
                result.recvs.append(
                    TransferResult(
                        request_id,
                        "Transfer receive failed" if failed else None,
                    )
                )
                self._failed_recv.discard(request_id)
                self._retired.add(event.transfer_id)
        with self._receive_lock:
            for request_id in terminal:
                self._receive_futures.pop(request_id, None)
        return result

    def take_errors(self) -> set[int]:
        result = {block for ids in self._invalid.values() for block in ids}
        self._invalid.clear()
        return result

    def cancel(self, request_id: str, transfer_id: str) -> None:
        with self._source_changed:
            source = self._sources.get(transfer_id)
            if source is not None:
                if source.request_id and source.request_id != request_id:
                    raise ValueError("Cannot cancel another request's session")
                source.abort = True
                self._signal_ready_locked(source)
                while source.active_writes:
                    self._source_changed.wait()
                self._retire_locked(source, report=False)
            self._retired.add(transfer_id)
            self._prepared.pop(request_id, None)
            self._pending_sends.discard(request_id)
        with self._result_lock:
            self._finished_send.discard(request_id)
            self._send_failures.pop(request_id, None)

    def preempt(self, request_id: str) -> None:
        event = self._recv_events.get(request_id)
        self._fence_receives({request_id})
        if event is not None:
            self._retired.add(event.transfer_id)
            self._recv_events.pop(request_id, None)
        self._invalid.pop(request_id, None)
        transfer_id = self._prepared.get(request_id)
        if transfer_id is not None:
            with self._source_lock:
                source = self._sources.get(transfer_id)
                unready = source is not None and not source.published
            if not unready:
                self.cancel(request_id, transfer_id)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._source_lock:
            for source in self._sources.values():
                source.abort = True
                self._signal_ready_locked(source)
        # D may have already published destination addresses. A remote P can
        # legally keep writing until its terminal response arrives, so fence
        # every receive before registered cache memory can be torn down.
        with self._receive_lock:
            receive_ids = set(self._receive_futures)
        self._fence_receives(receive_ids)
        if self._loop.is_running():

            async def close_context() -> None:
                # ZMQ sockets belong to this event-loop thread. Destroying the
                # context from the vLLM main thread can trip libzmq's signaler
                # assertion during process shutdown.
                # Drain accepted messages and native WRITEs before stopping
                # the fixed sender workers or closing the response socket.
                await self.sender_worker_queue.join()
                # Keep the ROUTER alive until terminal responses have been
                # delivered. Closing it earlier can strand D's receive fence.
                if self._listener_future is not None:
                    self._listener_future.cancel()
                    await asyncio.sleep(0)
                if self._serve_tasks:
                    await asyncio.gather(
                        *tuple(self._serve_tasks), return_exceptions=True
                    )
                self._ctx.destroy(linger=0)
                await asyncio.sleep(0)
                await self._loop.shutdown_default_executor()

            future = asyncio.run_coroutine_threadsafe(close_context(), self._loop)
            future.result()
            # Native writes are fenced; pool shutdown can no longer strand D.
            self._send_pool.shutdown(wait=True, cancel_futures=True)
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join()
            self._loop.close()
        else:
            self._send_pool.shutdown(wait=True, cancel_futures=True)
        if self._bootstrap is not None:
            self._bootstrap.close()
        if self._registered_storage:
            try:
                if is_npu_platform() or not hasattr(
                    self.engine, "batch_unregister_memory"
                ):
                    results = [
                        self.engine.unregister_memory(pointer)
                        for pointer in self._registered_storage
                    ]
                    if any(result != 0 for result in results):
                        logger.warning("MTSC TE memory unregistration partially failed")
                else:
                    ret = self.engine.batch_unregister_memory(self._registered_storage)
                    if ret != 0:
                        logger.warning(
                            "MTSC TE memory unregistration failed: ret=%s", ret
                        )
            except Exception as exc:  # noqa: BLE001 - binding teardown boundary
                logger.warning("MTSC TE memory unregistration failed: error=%s", exc)
            finally:
                self._registered_storage = []
        self._sources.clear()
        self._prepared.clear()
        self._pending_sends.clear()
        self._recv_events.clear()
        self._invalid.clear()
        self._retired.clear()
        self._finished_send.clear()
        self._failed_recv.clear()
        self._send_failures.clear()
        self.kv_caches.clear()

    async def _register_worker(self, side_port: int) -> None:
        payload = WorkerRegistration(
            engine_id=self.engine_id,
            dp_rank=self.dp_rank,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            address=make_zmq_path("tcp", self.hostname, side_port),
        )
        # Bootstrap is an internal control-plane hop. Never route it through
        # process-wide HTTP(S)_PROXY settings.
        async with httpx.AsyncClient(trust_env=False) as client:
            for _ in range(120):
                try:
                    response = await client.post(
                        self._registration_url, json=payload.model_dump()
                    )
                    response.raise_for_status()
                    return
                except httpx.ConnectError:
                    await asyncio.sleep(0.25)
        raise RuntimeError(f"MTSC bootstrap unavailable: {self._registration_url}")

    async def _listen(self, ready: threading.Event) -> None:
        socket = self._ctx.socket(zmq.ROUTER)
        try:
            port = socket.bind_to_random_port(f"tcp://{self.hostname}")
            await self._register_worker(port)
            self._serve_tasks.update(
                asyncio.create_task(self._sender_worker(socket))
                for _ in range(self.num_sender_tasks)
            )
            ready.set()
            while True:
                identity, payload = await socket.recv_multipart()
                await self.sender_worker_queue.put((identity, payload))
        except (asyncio.CancelledError, zmq.ContextTerminated):
            pass
        except Exception:
            ready.set()
            raise
        finally:
            # Cancelling an asyncio worker cannot stop its native TE thread.
            # Fence accepted batches and deliver their terminal responses
            # before cancelling idle workers or closing the ROUTER.
            await self.sender_worker_queue.join()
            for task in self._serve_tasks:
                task.cancel()
            if self._serve_tasks:
                await asyncio.gather(*self._serve_tasks, return_exceptions=True)
            self._serve_tasks.clear()
            socket.close(linger=0)

    async def _sender_worker(self, socket) -> None:
        while True:
            identity, payload = await self.sender_worker_queue.get()
            try:
                await self._serve(identity, payload, socket)
            except Exception as exc:  # noqa: BLE001 - sender worker boundary
                logger.warning("MTSC sender worker failed: error=%s", exc)
            finally:
                self.sender_worker_queue.task_done()

    def _aligned_regions(
        self, request: KVTransferRequest
    ) -> list[tuple[TransferRegion, TransferRegion, int]]:
        remote = [
            TransferRegion(name, index, group, base, block, kv)
            for name, index, group, base, block, kv in zip(
                request.layer_names,
                request.layer_indices,
                request.group_indices,
                request.region_base_addresses,
                request.block_lengths,
                request.kv_block_lengths,
                strict=True,
            )
        ]
        by_key: dict[tuple[str, int], tuple[TransferRegion, int]] = {}
        counts: defaultdict[str, int] = defaultdict(int)
        for remote_index, region in enumerate(remote):
            key = (region.layer_name, counts[region.layer_name])
            counts[region.layer_name] += 1
            by_key[key] = (region, remote_index)
        result: list[tuple[TransferRegion, TransferRegion, int]] = []
        counts.clear()
        for region in self.regions:
            key = (region.layer_name, counts[region.layer_name])
            counts[region.layer_name] += 1
            match = by_key.get(key)
            if match is None:
                # Different PP partitions legitimately have non-overlapping
                # local layers. D validates the union of all P responses.
                continue
            other, remote_index = match
            if (
                other.layer_index != region.layer_index
                or other.group_index != region.group_index
            ):
                raise ValueError(
                    f"P/D region identity mismatch for {region.layer_name!r}"
                )
            result.append((region, other, remote_index))
        return result

    @staticmethod
    def _validate_region_plan(
        local: TransferRegion,
        remote: TransferRegion,
        local_size: int,
        remote_size: int,
        replicated: bool,
        source_offset: int,
        destination_offset: int,
        length: int,
    ) -> None:
        if local.block_length <= 0 or remote.block_length <= 0 or length <= 0:
            raise ValueError("P/D region lengths must be positive")
        if source_offset + length > local.block_length:
            raise ValueError("P source slice exceeds its registered block")
        if destination_offset + length > remote.block_length:
            raise ValueError("D destination slice exceeds its registered block")
        if replicated:
            if local.kv_block_length != remote.kv_block_length:
                raise ValueError("Replicated P/D KV page sizes do not match")
            return
        if local_size == remote_size:
            valid = local.kv_block_length == remote.kv_block_length
        elif local_size > remote_size:
            valid = (
                local.kv_block_length * (local_size // remote_size)
                == remote.kv_block_length
            )
        else:
            valid = (
                remote.kv_block_length * (remote_size // local_size)
                == local.kv_block_length
            )
        if not valid:
            raise ValueError("P/D KV page lengths do not match their TP ratio")

    async def _write_one(
        self, transfer_id: str, source: _Session, request: KVTransferRequest
    ) -> tuple[bool, set[int]]:
        return (await self._write_batch({transfer_id: source}, request))[transfer_id]

    async def _write_batch(self, sources, request):
        src, dst, sizes = [], [], []
        results = {tid: (False, set()) for tid in sources}
        coverage = {}
        for transfer_id, source in sources.items():
            try:
                local_src, local_dst, local_sizes, covered = (
                    self._build_request_transfer_params(transfer_id, source, request)
                )
            except Exception as exc:  # noqa: BLE001 - per-request validation
                logger.warning(
                    "MTSC transfer plan failed: transfer_id=%s error=%s",
                    transfer_id,
                    exc,
                )
                continue
            src.extend(local_src)
            dst.extend(local_dst)
            sizes.extend(local_sizes)
            coverage[transfer_id] = covered
        ret = 0
        if src:
            session = f"{request.hostname}:{request.rpc_port}"
            logger.info(
                "MTSC P WRITE batch: transfer_ids=%s session=%s descriptors=%d",
                list(coverage),
                session,
                len(src),
            )
            ret = await self._write_buffers(session, src, dst, sizes)
            if ret != 0:
                logger.warning(
                    "MTSC TE WRITE failed: transfer_ids=%s ret=%s", list(coverage), ret
                )
        for transfer_id, covered in coverage.items():
            results[transfer_id] = (ret == 0, covered if ret == 0 else set())
        return results

    def _build_request_transfer_params(self, transfer_id, source, request):
        _, destination_groups = request.requests[transfer_id]
        if not any(destination_groups):
            return [], [], [], set()
        if len(source.block_ids) != len(destination_groups):
            raise ValueError("P/D KV group count mismatch")
        source_groups: list[list[int]] = []
        for local, remote in zip(source.block_ids, destination_groups, strict=True):
            if len(local) < len(remote):
                raise ValueError("P has fewer source blocks than D requested")
            source_groups.append(local[-len(remote) :] if remote else [])
        src: list[int] = []
        dst: list[int] = []
        sizes: list[int] = []
        covered: set[int] = set()
        for local_region, remote_region, remote_index in self._aligned_regions(request):
            group = local_region.group_index
            if group >= len(destination_groups):
                raise ValueError(
                    f"P/D region group {group} exceeds request group count"
                )
            if not any(block >= 0 for block in destination_groups[group]):
                continue
            copy, src_offset, dst_offset, length, p_shards, d_shards = kv_slice_plan(
                self.tp_rank,
                self.tp_size,
                request.tp_rank,
                request.tp_size,
                local_region.kv_block_length,
                remote_region.kv_block_length,
                self.topology.total_num_kv_heads,
                self.topology.is_mla,
            )
            if not copy:
                continue
            self._validate_region_plan(
                local_region,
                remote_region,
                p_shards,
                d_shards,
                False,
                src_offset,
                dst_offset,
                length,
            )
            covered.add(remote_index)
            region_start = len(src)
            can_coalesce = (
                src_offset == 0
                and dst_offset == 0
                and length == local_region.block_length
                and length == remote_region.block_length
            )
            for source_block, destination_block in zip(
                source_groups[group], destination_groups[group], strict=True
            ):
                if destination_block < 0:
                    continue  # Sliding-window placeholder, not writable memory.
                if source_block < 0:
                    raise ValueError("Requested source block is a placeholder")
                if source_block >= self.num_blocks:
                    raise ValueError("Source block exceeds registered memory")
                if (
                    request.destination_num_blocks
                    and destination_block >= request.destination_num_blocks
                ):
                    raise ValueError("Destination block exceeds registered memory")
                src_address = (
                    local_region.base_address
                    + source_block * local_region.block_length
                    + src_offset
                )
                dst_address = (
                    remote_region.base_address
                    + destination_block * remote_region.block_length
                    + dst_offset
                )
                if (
                    can_coalesce
                    and len(src) > region_start
                    and src[-1] + sizes[-1] == src_address
                    and dst[-1] + sizes[-1] == dst_address
                ):
                    sizes[-1] += length
                else:
                    src.append(src_address)
                    dst.append(dst_address)
                    sizes.append(length)
        return src, dst, sizes, covered

    async def _write_buffers(self, session, src, dst, sizes) -> int:
        async def write(start, end):
            return await self._loop.run_in_executor(
                self._send_pool,
                self.engine.batch_transfer_sync_write,
                session,
                src[start:end],
                dst[start:end],
                sizes[start:end],
            )

        if self.protocol != "tcp":
            return await write(0, len(src))
        # TCP has a bounded descriptor queue per peer. Serialize each peer's
        # batches and wait for receiver ACKs before admitting more descriptors.
        # Requests to other peers can still progress independently.
        lock = self._tcp_write_locks.setdefault(session, asyncio.Lock())
        async with lock:
            for start in range(0, len(src), 256):
                ret = await write(start, start + 256)
                if ret != 0:
                    return ret
        return 0

    async def _query_workers(
        self, address: str, engine_id: str, dp_rank: int
    ) -> dict[int, dict[int, str]]:
        key = (address.rstrip("/"), engine_id, dp_rank)
        if key in self._remote_workers:
            return self._remote_workers[key]
        async with httpx.AsyncClient(trust_env=False) as client:
            response = await client.get(address.rstrip("/") + "/query")
            response.raise_for_status()
            entries = response.json()
        entry = entries.get(str(dp_rank))
        if entry is None:
            raise KeyError(f"Remote DP rank {dp_rank} is not registered")
        if entry.get("engine_id") != engine_id:
            raise KeyError(
                f"Remote DP rank {dp_rank} belongs to engine "
                f"{entry.get('engine_id')!r}, not {engine_id!r}"
            )
        workers = {
            int(tp): {int(pp): addr for pp, addr in pp_map.items()}
            for tp, pp_map in entry["worker_addr"].items()
        }
        tp_size = int(entry.get("tp_size", 0))
        pp_size = int(entry.get("pp_size", 0))
        logger.info(
            "MTSC D _query_workers: engine=%s dp=%d tp_size=%d workers=%s",
            engine_id,
            dp_rank,
            tp_size,
            {k: v for k, v in workers.items()},
        )
        if tp_size <= 0 or pp_size <= 0:
            raise ValueError("Remote P topology dimensions are missing")
        if sorted(workers) != list(range(tp_size)):
            raise ValueError("Remote P TP workers are not fully registered")
        expected_pp = list(range(pp_size))
        if any(sorted(pp_map) != expected_pp for pp_map in workers.values()):
            raise ValueError("Remote P PP workers are not fully registered")
        self._remote_workers[key] = workers
        return workers

    def _validate_coverage(
        self,
        transfer_id: str,
        block_ids: list[list[int]],
        responses: list[KVTransferResponse],
        expected: int,
    ) -> None:
        """Require every data-carrying D region to have exactly one TP fan-in."""
        coverage: Counter[int] = Counter()
        for response in responses:
            coverage.update((response.covered_regions or {}).get(transfer_id, []))
        required = {
            index
            for index, region in enumerate(self.regions)
            if region.group_index < len(block_ids)
            and any(block >= 0 for block in block_ids[region.group_index])
        }
        invalid = [index for index in required if coverage[index] != expected]
        unexpected = [index for index in coverage if index not in required]
        if invalid or unexpected:
            raise ValueError(
                "P/D region coverage mismatch: "
                f"expected={expected} invalid={invalid} unexpected={unexpected}"
            )

    async def _receive_groups(self, groups, futures) -> None:
        await asyncio.gather(
            *(self._receive_batch(events, futures) for events in groups.values())
        )

    async def _receive(
        self,
        request_id: str,
        transfer_id: str,
        block_ids: list[list[int]],
        remote_engine_id: str,
        bootstrap: str,
        remote_dp_rank: int,
    ) -> None:
        await self._receive_batch(
            [
                RecvEvent(
                    request_id,
                    transfer_id,
                    tuple(tuple(ids) for ids in block_ids),
                    bootstrap,
                    remote_engine_id,
                    remote_dp_rank,
                )
            ]
        )

    def _finish_receive(self, event, failed, future=None) -> None:
        logger.info(
            "MTSC D receive done: request_id=%s failed=%s", event.request_id, failed
        )
        if failed:
            with self._result_lock:
                self._failed_recv.add(event.request_id)
        if future is not None and not future.done():
            future.set_result(None)

    async def _receive_batch(self, events, futures=None) -> None:
        futures = futures or {}
        finished: set[str] = set()
        by_id = {event.transfer_id: event for event in events}
        if not events:
            return
        first = events[0]
        try:
            workers = await self._query_workers(
                first.bootstrap_addr, first.remote_engine_id, first.remote_dp_rank
            )
            target_tp = self.topology.handshake_target_ranks(len(workers))
            addresses: list[str] = []
            for tp_rank in target_tp:
                pp_map = workers[tp_rank]
                pp_ranks = (
                    [self.pp_rank]
                    if len(pp_map) == self.pp_size and self.pp_rank in pp_map
                    else sorted(pp_map)
                )
                addresses.extend(pp_map[rank] for rank in pp_ranks)
            if not addresses:
                raise RuntimeError("No matching P workers in bootstrap response")
            logger.info(
                "MTSC D receive batch: transfer_ids=%s target_tp=%s addresses=%s",
                list(by_id),
                target_tp,
                addresses,
            )
            request = KVTransferRequest(
                hostname=self.hostname,
                rpc_port=self.rpc_port,
                tp_size=self.tp_size,
                tp_rank=self.tp_rank,
                pp_size=self.pp_size,
                pp_rank=self.pp_rank,
                schema=self.schema,
                requests={
                    event.transfer_id: (
                        event.request_id,
                        [list(ids) for ids in event.block_ids],
                    )
                    for event in events
                },
                region_base_addresses=[region.base_address for region in self.regions],
                block_lengths=[region.block_length for region in self.regions],
                kv_block_lengths=[region.kv_block_length for region in self.regions],
                layer_names=[region.layer_name for region in self.regions],
                layer_indices=[region.layer_index for region in self.regions],
                group_indices=[region.group_index for region in self.regions],
                engine_id=self.engine_id,
                dp_rank=self.dp_rank,
                remote_engine_id=first.remote_engine_id,
                remote_dp_rank=first.remote_dp_rank,
                destination_num_blocks=self.num_blocks,
            )
            payload = self._encoder.encode(request)
            responses = {tid: {} for tid in by_id}
            p_shards = effective_tp(
                len(workers), self.topology.total_num_kv_heads, self.schema.is_mla
            )
            d_shards = effective_tp(
                self.tp_size, self.topology.total_num_kv_heads, self.schema.is_mla
            )
            expected = max(1, p_shards // d_shards)

            def record(address, transfer_id, response):
                peer_responses = responses[transfer_id]
                peer_responses[address] = response
                if len(peer_responses) != len(addresses):
                    return
                event = by_id[transfer_id]
                failed = any(
                    transfer_id not in (result.completed_transfer_ids or [])
                    or transfer_id in (result.failed_transfer_ids or [])
                    or result.status == KVTransferStatus.FAILED
                    for result in peer_responses.values()
                )
                if not failed:
                    try:
                        if any(event.block_ids):
                            self._validate_coverage(
                                transfer_id,
                                event.block_ids,
                                list(peer_responses.values()),
                                expected,
                            )
                        self.reformat_npu_blocks(event.block_ids, len(workers))
                    except Exception as exc:  # noqa: BLE001 - completion boundary
                        logger.warning(
                            "MTSC D completion failed: transfer_id=%s error=%s",
                            transfer_id,
                            exc,
                        )
                        failed = True
                self._finish_receive(event, failed, futures.get(event.request_id))
                finished.add(transfer_id)

            async def call(address: str) -> None:
                socket = self._ctx.socket(zmq.DEALER)
                remaining = set(by_id)
                try:
                    socket.setsockopt(zmq.LINGER, 0)
                    socket.connect(address)
                    await socket.send(payload)
                    # A local timeout cannot fence published destination memory.
                    # Each response is terminal for the IDs it reports, while
                    # IN_PROGRESS keeps this socket open for the other IDs.
                    while True:
                        response = self._response_decoder.decode(await socket.recv())
                        if response.status == KVTransferStatus.FAILED:
                            raise RuntimeError(
                                response.error_message or "P peer failed"
                            )
                        if response.status not in (
                            KVTransferStatus.IN_PROGRESS,
                            KVTransferStatus.COMPLETE,
                        ):
                            raise ValueError("Invalid P response status")
                        completed = set(response.completed_transfer_ids or [])
                        failed = set(response.failed_transfer_ids or [])
                        reported = completed | failed
                        if completed & failed or not reported <= remaining:
                            raise ValueError(
                                "P response contains conflicting or repeated transfer IDs"
                            )
                        for tid in reported:
                            record(address, tid, response)
                        remaining.difference_update(reported)
                        if response.status == KVTransferStatus.COMPLETE:
                            if remaining:
                                raise ValueError(
                                    "P final response omitted pending transfer IDs"
                                )
                            break
                except Exception as exc:  # noqa: BLE001 - peer response boundary
                    logger.warning(
                        "MTSC D peer pull failed: address=%s error=%s", address, exc
                    )
                    response = KVTransferResponse(
                        KVTransferStatus.FAILED, error_message=str(exc)
                    )
                    for tid in remaining:
                        record(address, tid, response)
                finally:
                    socket.close(linger=0)

            # A failed peer must not leave sibling WRITEs unfenced.
            outcomes = await asyncio.gather(
                *(call(address) for address in addresses), return_exceptions=True
            )
            errors = [
                outcome for outcome in outcomes if isinstance(outcome, BaseException)
            ]
            if errors:
                raise RuntimeError(f"P peer request failed: {errors[0]}")
        except Exception as exc:  # noqa: BLE001 - async transport boundary
            logger.warning(
                "MTSC D pull batch failed: transfer_ids=%s error=%s", list(by_id), exc
            )
        finally:
            for transfer_id, event in by_id.items():
                if transfer_id not in finished:
                    self._finish_receive(event, True, futures.get(event.request_id))

    def _fence_receives(self, request_ids: set[str]) -> None:
        """Fence in-flight P writes before vLLM can recycle D blocks."""
        futures: list[tuple[str, Future[None]]] = []
        with self._receive_lock:
            for request_id in request_ids:
                future = self._receive_futures.pop(request_id, None)
                if future is not None:
                    futures.append((request_id, future))
        for request_id, future in futures:
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 - transport fence
                logger.warning(
                    "MTSC PD receive fence failed: request_id=%s error=%s",
                    request_id,
                    exc,
                )
        with self._result_lock:
            self._failed_recv.difference_update(request_ids)
