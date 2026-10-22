# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Region and TP helpers follow the checked-in vLLM Mooncake connector.
"""Mooncake Transfer Engine implementation for MTSC workers."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import threading
import time
from collections import defaultdict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import httpx
import msgspec
import numpy as np
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
    is_local_first_rank,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_ip, make_zmq_path
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.v1.worker.utils import select_common_block_size

from .protocol import (
    KVTransferRequest,
    KVTransferResponse,
    KVTransferStatus,
)
from .utils import (
    BlockIds,
    cache_tensors,
    is_npu_platform,
)

TransferId = str


def _resolve_transfer_transport(extra: Mapping[str, Any]) -> tuple[str, str]:
    """Resolve PD transport: explicit extra config > Store JSON > defaults.

    None is treated as unset. An empty device_name is an explicit override
    that clears the JSON's device whitelist. A configured but invalid JSON
    must fail rather than silently selecting a different transport.
    """
    protocol = extra.get("mooncake_protocol")
    device_name = extra.get("device_name")
    if (protocol is None or device_name is None) and os.getenv("MOONCAKE_CONFIG_PATH"):
        from .kv_cache_pool import StoreConfig

        config = StoreConfig.load()
        if protocol is None:
            protocol = config.protocol
        if device_name is None:
            device_name = config.device_name
    if protocol is None:
        protocol = "rdma"
    if device_name is None:
        device_name = ""
    if not isinstance(protocol, str) or not protocol.strip():
        raise ValueError("Mooncake direct TE protocol must be a non-empty string")
    if not isinstance(device_name, str):
        raise ValueError("Mooncake direct TE device_name must be a string")
    return protocol.strip(), device_name.strip()


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


class _ReceiveFenceError(RuntimeError):
    """The worker must stop: a published destination may still be written."""


logger = init_logger(__name__)
logger.setLevel(logging.INFO)


def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


def _bootstrap_address(config: VllmConfig) -> tuple[str, int]:
    parallel = config.parallel_config
    if parallel.local_engines_only:
        host = "127.0.0.1"
    else:
        host = parallel.data_parallel_master_ip
    return host, envs.VLLM_MOONCAKE_BOOTSTRAP_PORT


def _launch_bootstrap(config: VllmConfig) -> bool:
    parallel = config.parallel_config
    return is_local_first_rank() and (
        parallel.local_engines_only or parallel.data_parallel_index == 0
    )


class WorkerRegistration(BaseModel):
    engine_id: str
    dp_rank: int
    tp_rank: int
    pp_rank: int
    addr: str


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
                "worker_addr": {},
            },
        )
        if entry["engine_id"] != payload.engine_id:
            raise HTTPException(400, "engine_id mismatch")
        tp_entry = entry["worker_addr"].setdefault(payload.tp_rank, {})
        if payload.pp_rank in tp_entry:
            if tp_entry[payload.pp_rank] == payload.addr:
                return {"status": "ok"}
            raise HTTPException(400, "worker rank already registered")
        tp_entry[payload.pp_rank] = payload.addr
        return {"status": "ok"}

    async def query(self) -> dict[int, dict[str, Any]]:
        return self.workers

    def close(self) -> None:
        if self.server.started:
            self.server.should_exit = True
            self.thread.join(timeout=5)


@dataclass(frozen=True)
class TransferRegion:
    base_addr: int
    block_len: int
    kv_block_len: int


def _get_tp_ratio(local_tp_size: int, remote_tp_size: int) -> int:
    """Return the TP ratio used by heterogeneous TP transfer planning.

    Positive values mean one local rank maps into a larger remote KV region.
    Negative values mean one local rank must gather from multiple remote KV
    regions.
    """
    if local_tp_size >= remote_tp_size:
        assert local_tp_size % remote_tp_size == 0, (
            f"Local tensor parallel size {local_tp_size} is not divisible "
            f"by remote tensor parallel size {remote_tp_size}."
        )
        return local_tp_size // remote_tp_size

    assert remote_tp_size % local_tp_size == 0, (
        f"Remote tensor parallel size {remote_tp_size} is not divisible "
        f"by local tensor parallel size {local_tp_size}."
    )
    return -(remote_tp_size // local_tp_size)


def _expand_transfer_regions(
    base_addrs: list[int],
    block_lens: list[int],
    is_kv_layout_blocks_first: bool,
) -> list[TransferRegion]:
    """Expand registered KV tensors into the regions transferred by Mooncake."""
    assert len(base_addrs) == len(block_lens), (
        "Mooncake transfer regions require matching numbers of base addresses "
        f"and block lengths, got {len(base_addrs)} and {len(block_lens)}."
    )
    regions: list[TransferRegion] = []
    for base_addr, block_len in zip(base_addrs, block_lens):
        kv_block_len = block_len // 2 if is_kv_layout_blocks_first else block_len
        regions.append(
            TransferRegion(
                base_addr=base_addr,
                block_len=block_len,
                kv_block_len=kv_block_len,
            )
        )
        if is_kv_layout_blocks_first:
            regions.append(
                TransferRegion(
                    base_addr=base_addr + kv_block_len,
                    block_len=block_len,
                    kv_block_len=kv_block_len,
                )
            )
    return regions


def _compute_sender_transfer_plan(
    local_tp_rank: int,
    local_tp_size: int,
    remote_tp_rank: int,
    remote_tp_size: int,
    local_kv_block_len: int,
    remote_kv_block_len: int,
    producer_cache_replicated: bool,
) -> tuple[bool, int, int, int]:
    """Plan one producer-rank to one consumer-rank copy for heterogeneous TP."""
    tp_ratio = _get_tp_ratio(local_tp_size, remote_tp_size)

    if tp_ratio == 1:
        return True, 0, 0, local_kv_block_len

    if tp_ratio > 0:
        if producer_cache_replicated:
            return local_tp_rank % tp_ratio == 0, 0, 0, local_kv_block_len
        return (
            True,
            0,
            (local_tp_rank % tp_ratio) * local_kv_block_len,
            local_kv_block_len,
        )

    if producer_cache_replicated:
        return True, 0, 0, local_kv_block_len

    ratio_abs = -tp_ratio
    return (
        True,
        (remote_tp_rank % ratio_abs) * remote_kv_block_len,
        0,
        remote_kv_block_len,
    )


def _can_coalesce_block_transfers(
    local_region_block_len: int,
    remote_region_block_len: int,
    src_region_offset: int,
    dst_region_offset: int,
    transfer_len: int,
) -> bool:
    """Whether a contiguous block group can be emitted as one larger copy."""
    return (
        src_region_offset == 0
        and dst_region_offset == 0
        and transfer_len == local_region_block_len
        and transfer_len == remote_region_block_len
    )


def _validate_asymmetric_region_lengths(
    local_regions: list[TransferRegion],
    remote_regions: list[TransferRegion],
    local_tp_size: int,
    remote_tp_size: int,
    producer_cache_replicated: bool,
) -> str | None:
    """Validate transfer-region metadata for a fixed producer/consumer pair.

    This checks registered KV regions, not per-request block counts. A region
    corresponds to one registered KV tensor, or one K/V half after expansion
    for layouts that store K and V together.
    """
    if len(local_regions) != len(remote_regions):
        return (
            "Mooncake asymmetric TP requires matching KV region counts between "
            "producer and consumer."
        )

    if producer_cache_replicated:
        return None

    tp_ratio = _get_tp_ratio(local_tp_size, remote_tp_size)
    for idx, (local_region, remote_region) in enumerate(
        zip(local_regions, remote_regions)
    ):
        if tp_ratio == 1:
            if local_region.kv_block_len != remote_region.kv_block_len:
                return (
                    "Mooncake KV region length mismatch for homogeneous TP at "
                    f"region {idx}: local={local_region.kv_block_len}, "
                    f"remote={remote_region.kv_block_len}."
                )
        elif tp_ratio > 0:
            if remote_region.kv_block_len != local_region.kv_block_len * tp_ratio:
                return (
                    "Mooncake destination KV region length does not match the "
                    "producer TP ratio at region "
                    f"{idx}: local={local_region.kv_block_len}, "
                    f"remote={remote_region.kv_block_len}, tp_ratio={tp_ratio}."
                )
        else:
            ratio_abs = -tp_ratio
            if local_region.kv_block_len != remote_region.kv_block_len * ratio_abs:
                return (
                    "Mooncake source KV region length does not match the "
                    "consumer TP ratio at region "
                    f"{idx}: local={local_region.kv_block_len}, "
                    f"remote={remote_region.kv_block_len}, tp_ratio={tp_ratio}."
                )

    return None


def group_concurrent_contiguous(
    src_indices: list[int], dst_indices: list[int]
) -> tuple[list[list[int]], list[list[int]]]:
    """Vectorised NumPy implementation."""
    if len(src_indices) == 0:
        return [], []

    brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
    src_groups = np.split(src_indices, brk)
    dst_groups = np.split(dst_indices, brk)

    src_groups = [g.tolist() for g in src_groups]
    dst_groups = [g.tolist() for g in dst_groups]

    return src_groups, dst_groups


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

    @property
    def local_replicates_kv_cache(self) -> bool:
        return self.is_mla or self.tp_size > self.total_num_kv_heads

    virtually_split_kv_in_blocks = False
    split_k_and_v = False

    def handshake_target_ranks(self, remote_tp_size: int) -> list[int]:
        if self.tp_size >= remote_tp_size:
            if self.tp_size % remote_tp_size:
                raise ValueError("P/D TP sizes must have an integer ratio")
            return [self.tp_rank // (self.tp_size // remote_tp_size)]
        if remote_tp_size % self.tp_size:
            raise ValueError("P/D TP sizes must have an integer ratio")
        ratio = remote_tp_size // self.tp_size
        return [self.tp_rank * ratio + offset for offset in range(ratio)]


@dataclass
class SendBlockMeta:
    p_req_id: str
    transfer_id: TransferId
    local_block_ids: list[list[int]] = field(default_factory=list)
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    expire_time: float = float("inf")
    need_send: int = 0
    sent: int = 0
    sending: int = 0


@dataclass
class PullReqMeta:
    d_req_id: str
    transfer_id: str
    local_block_ids: list[list[int]]
    remote_engine_id: str
    remote_bootstrap_addr: str
    expire_time: float = float("inf")
    pull_tasks_count: int = 0


class MooncakeKVTransfer:
    """Own TE registration, parallel mapping, WRITE and fenced sessions.

    Sender state follows Mooncake's ready/need_send/sent/sending counters.
    Unpublished P requests survive preemption so early D waiters can resume.
    """

    def __init__(self, config: VllmConfig, kv_cache_config: Any) -> None:
        try:
            from mooncake.engine import TransferEngine
        except ImportError as exc:
            raise ImportError("Mooncake TransferEngine bindings are required") from exc
        assert config.kv_transfer_config is not None
        transfer = config.kv_transfer_config
        assert transfer.engine_id is not None
        if config.parallel_config.pipeline_parallel_size > 1:
            raise ValueError("Mooncake transfer does not support pipeline parallelism")
        if kv_cache_config.has_mamba_layers:
            raise ValueError("Mooncake transfer does not support Mamba KV cache")
        self.num_blocks = 0
        self._receive_error: _ReceiveFenceError | None = None
        self.engine_id = transfer.engine_id
        self.vllm_config = config
        self.block_size = config.cache_config.block_size
        self.is_producer = transfer.kv_role != "kv_consumer"
        self.is_consumer = transfer.kv_role != "kv_producer"
        extra = transfer.kv_connector_extra_config
        self.timeout = float(
            extra.get(
                "mtsc_pd_timeout_seconds", envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT
            )
        )
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("mtsc_pd_timeout_seconds must be finite and positive")
        self._sync_block_size_with_kernel()
        self.num_sender_workers = max(1, int(extra.get("num_workers", 10)))
        self.num_sender_tasks = self.num_sender_workers * 2
        protocol, device_name = _resolve_transfer_transport(extra)
        _require_tcp_write_ack(protocol)
        self.device_id = torch.accelerator.current_device_index()
        current_platform.set_device(self.device_id)
        self._setup_ascend_local_comm_res()

        # Share transport defaults with Store while retaining upstream's
        # explicit extra-config overrides and the PD metadata handshake.
        logger.info("MTSC direct TE: protocol=%s device_name=%r", protocol, device_name)
        self.protocol = protocol
        self.engine = TransferEngine()
        self.hostname = get_ip()
        ret = self.engine.initialize(
            self.hostname,
            "P2PHANDSHAKE",
            protocol,
            device_name,
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
            "block_size": self.block_size,
            "is_mla": model.use_mla,
            "total_num_kv_heads": model.get_total_num_kv_heads(),
        }
        if is_npu_platform():
            self.transfer_topo = _NPUTransferTopology(**topology_args)
        else:
            self.transfer_topo = TransferTopology(
                **topology_args,
                engine_id=self.engine_id,
                is_mamba=False,
                attn_backends=get_current_attn_backends(config),
            )
        self.kv_caches: dict[
            str, torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]
        ] = {}
        self.kv_caches_base_addr: list[int] = []
        self.block_len_per_layer: list[int] = []
        self._registered_storage: list[int] = []
        self.reqs_need_send: dict[TransferId, SendBlockMeta] = {}
        self.finished_sending_reqs: set[str] = set()
        self.finished_recving_reqs: set[str] = set()
        self._failed_recv: set[str] = set()
        self._remote_agents: dict[str, dict[int, dict[int, str]]] = {}
        self._remote_tp_size: dict[str, int] = {}
        self._pending_bootstrap_queries: dict[str, asyncio.Event] = {}
        self._receive_futures: dict[str, asyncio.Future[None]] = {}
        self._encoder = msgspec.msgpack.Encoder()
        self._request_decoder = msgspec.msgpack.Decoder(KVTransferRequest)
        self._response_decoder = msgspec.msgpack.Decoder(KVTransferResponse)
        self._ctx = zmq.asyncio.Context()
        self.sender_loop: asyncio.AbstractEventLoop | None = None
        self.receiver_loop: asyncio.AbstractEventLoop | None = None
        self._sender_listener_t: threading.Thread | None = None
        self._receiver_t: threading.Thread | None = None
        self._sender_executor: ThreadPoolExecutor | None = None
        self._sender_listener_future = None
        self.sender_tasks: list[asyncio.Task[None]] = []
        self._bootstrap = None
        if self.is_producer:
            self._sender_executor = ThreadPoolExecutor(
                max_workers=self.num_sender_workers,
                thread_name_prefix="mtsc-te-write",
                initializer=self._bind_sender_thread_device,
            )
            self.sender_worker_queue: asyncio.Queue[tuple[bytes, bytes]] = (
                asyncio.Queue()
            )
            self.sender_loop = asyncio.new_event_loop()
            sender_started = threading.Event()
            self.sender_loop.call_soon(sender_started.set)
            self._sender_listener_t = threading.Thread(
                target=_run_loop,
                args=(self.sender_loop,),
                name="mtsc-mooncake-sender",
                daemon=True,
            )
            self._sender_listener_t.start()
            sender_started.wait()
            host, port = _bootstrap_address(config)
            self._registration_url = make_zmq_path("http", host, port) + "/register"
            if _launch_bootstrap(config):
                self._bootstrap = BootstrapServer(port)
                self._bootstrap.start()
        if self.is_consumer:
            self.receiver_loop = asyncio.new_event_loop()
            receiver_started = threading.Event()
            self.receiver_loop.call_soon(receiver_started.set)
            self._receiver_t = threading.Thread(
                target=_run_loop,
                args=(self.receiver_loop,),
                name="mtsc-mooncake-receiver",
                daemon=True,
            )
            self._receiver_t.start()
            receiver_started.wait()
        self._retired_recvs: set[str] = set()
        self._recv_events: dict[str, RecvEvent] = {}
        self._invalid: dict[str, set[int]] = {}
        self._registered = False
        self._closed = False

    def _bind_sender_thread_device(self) -> None:
        current_platform.set_device(self.device_id)

    def _setup_ascend_local_comm_res(self) -> None:
        """按 device 设置 ASCEND_LOCAL_COMM_RES（HIXL 机内 UB endpoint）。

        与 vllm-ascend 的 setup_ascend_local_comm_res 相同：
        读 /etc/hixlep/ub_endpoint_npu_{device}.json -> ASCEND_LOCAL_COMM_RES env。
        缺失该 env 时 HIXL 生成错误 rank_table -> 103900。
        """
        if os.getenv("ASCEND_LOCAL_COMM_RES"):
            return
        res_path = "/etc/hixlep"
        visible_devices = os.getenv("ASCEND_RT_VISIBLE_DEVICES")
        if not visible_devices:
            return
        devices = [int(x) for x in visible_devices.split(",") if x.strip()]
        if not devices:
            return
        current_dev = (
            int(torch.npu.current_device()) if hasattr(torch, "npu") else self.device_id
        )
        if current_dev not in devices:
            return
        endpoint_file = os.path.join(res_path, f"ub_endpoint_npu_{current_dev}.json")
        try:
            with open(endpoint_file, encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return
        os.environ["ASCEND_LOCAL_COMM_RES"] = json.dumps(
            data, ensure_ascii=False, separators=(",", ":")
        )

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("KV transfer is closed")

    async def _check_receive_error(self) -> None:
        if self._receive_error is not None:
            raise self._receive_error

    def _sync_block_size_with_kernel(self) -> None:
        backends = get_current_attn_backends(self.vllm_config)
        kernel_block_size = select_common_block_size(self.block_size, backends)
        if self.block_size != kernel_block_size:
            assert self.block_size > kernel_block_size
            self.block_size = kernel_block_size

    def register(self, kv_caches) -> None:
        self._check_open()
        if self._registered:
            raise RuntimeError("KV transfer memory is already registered")
        self.kv_caches = kv_caches
        self.kv_caches_base_addr, self.block_len_per_layer = [], []
        pointers, lengths = [], []
        seen = set()
        for raw in kv_caches.values():
            if is_npu_platform():
                caches = cache_tensors(raw)
            else:
                caches = raw if self.transfer_topo.split_k_and_v else [raw]
            for cache in caches:
                pointer = cache.data_ptr()
                if pointer in seen:
                    continue
                seen.add(pointer)
                if not pointers:
                    self.num_blocks = int(cache.shape[0])
                assert cache.shape[0] == self.num_blocks, (
                    "All kv cache tensors must have the same number of blocks"
                )
                stride = cache.stride(0) * cache.element_size()
                pointers.append(pointer)
                lengths.append(self.num_blocks * stride)
                self.kv_caches_base_addr.append(pointer)
                self.block_len_per_layer.append(stride)
        if not pointers or self.num_blocks <= 0:
            raise RuntimeError("No PD KV regions registered")
        if is_npu_platform():
            for pointer, length in zip(pointers, lengths, strict=True):
                if self.engine.register_memory(pointer, length) != 0:
                    raise RuntimeError("Mooncake TE memory registration failed")
                self._registered_storage.append(pointer)
        else:
            if self.engine.batch_register_memory(pointers, lengths) != 0:
                raise RuntimeError("Mooncake TE memory registration failed")
            self._registered_storage = pointers
        if self.is_producer:
            ready = threading.Event()
            self._sender_listener_future = asyncio.run_coroutine_threadsafe(
                self._sender_listener(ready), self.sender_loop
            )
            if not ready.wait(timeout=self.timeout):
                self._sender_listener_future.cancel()
                raise TimeoutError("MTSC P listener startup timed out")
            if self._sender_listener_future.done():
                self._sender_listener_future.result()
        self._registered = True

    def _get_transfer_regions(self, addresses, lengths):
        return _expand_transfer_regions(
            addresses, lengths, self.transfer_topo.virtually_split_kv_in_blocks
        )

    def _prepare_transfer_regions(self, request):
        local = self._get_transfer_regions(
            self.kv_caches_base_addr, self.block_len_per_layer
        )
        remote = self._get_transfer_regions(
            request.kv_caches_base_addr, request.block_lens
        )
        error = _validate_asymmetric_region_lengths(
            local,
            remote,
            self.tp_size,
            request.remote_tp_size,
            self.transfer_topo.local_replicates_kv_cache,
        )
        if error:
            raise ValueError(error)
        if (
            is_npu_platform()
            and not self.transfer_topo.is_mla
            and self.tp_size != request.remote_tp_size
        ):
            if get_kv_cache_layout() != "HND":
                raise ValueError("Heterogeneous TP requires HND KV cache layout")
        return remote

    async def _prepare_send(self, request_id, transfer_id):
        send_meta = self.reqs_need_send.setdefault(
            transfer_id, SendBlockMeta(request_id, transfer_id)
        )
        send_meta.p_req_id = request_id
        return send_meta

    def prepare(self, request_id: str, transfer_id: str) -> None:
        self._check_open()
        if self.sender_loop is None:
            raise ValueError("Only P can send KV")
        asyncio.run_coroutine_threadsafe(
            self._prepare_send(request_id, transfer_id), self.sender_loop
        ).result()

    async def _publish_send(self, event):
        send_meta = await self._prepare_send(event.request_id, event.transfer_id)
        send_meta.local_block_ids = [list(group) for group in event.block_ids]
        send_meta.expire_time = time.perf_counter() + self.timeout
        send_meta.ready.set()

    def send(self, event: SendEvent) -> None:
        self._check_open()
        if self.sender_loop is None:
            raise ValueError("Only P can send KV")
        asyncio.run_coroutine_threadsafe(
            self._publish_send(event), self.sender_loop
        ).result()

    async def _start_receive(self, events: list[RecvEvent]):
        await self._check_receive_error()
        request_ids = set(self._recv_events) | self._invalid.keys()
        transfer_ids = self._retired_recvs | {
            event.transfer_id for event in self._recv_events.values()
        }
        for event in events:
            if event.request_id in request_ids:
                raise ValueError("Uncollected recv or block errors for request")
            if event.transfer_id in transfer_ids:
                raise ValueError("Transfer attempt is retired or duplicated")
            request_ids.add(event.request_id)
            transfer_ids.add(event.transfer_id)

        reqs_to_recv: dict[str, dict[str, PullReqMeta]] = defaultdict(dict)
        for event in events:
            self._receive_futures[event.request_id] = (
                asyncio.get_running_loop().create_future()
            )
            self._recv_events[event.request_id] = event
            reqs_to_recv[event.remote_engine_id][event.request_id] = PullReqMeta(
                event.request_id,
                event.transfer_id,
                [list(ids) for ids in event.block_ids],
                event.remote_engine_id,
                event.bootstrap_addr,
            )
        await self._start_load_kv(reqs_to_recv)

    async def _start_load_kv(self, reqs_to_recv: dict[str, dict[str, PullReqMeta]]):
        for remote_engine_id, pull_metas in reqs_to_recv.items():
            if remote_engine_id not in self._remote_agents:
                asyncio.create_task(
                    self.handle_new_engine_id(remote_engine_id, pull_metas)
                )
            else:
                self.receive_kv(remote_engine_id, pull_metas)

    def recv(self, events: list[RecvEvent]) -> None:
        self._check_open()
        if self.receiver_loop is None:
            raise ValueError("Only D can receive KV")
        if not self._registered:
            raise RuntimeError("Register KV memory before recv")
        asyncio.run_coroutine_threadsafe(
            self._start_receive(events), self.receiver_loop
        ).result()

    async def send_kv_to_decode(self, identity, payload, socket) -> None:
        try:
            meta = self._request_decoder.decode(payload)
            remote_tp_ranks = self.transfer_topo.handshake_target_ranks(
                meta.remote_tp_size
            )
            if meta.remote_tp_rank not in remote_tp_ranks:
                raise ValueError("D TP rank is not paired with this P rank")
            remote_regions = self._prepare_transfer_regions(meta)
        except Exception as exc:  # noqa: BLE001 - control response boundary
            response = KVTransferResponse(KVTransferStatus.ERROR, err_msg=str(exc))
            await socket.send_multipart((identity, self._encoder.encode(response)))
            return

        pending_reqs: dict[str, SendBlockMeta] = {}
        for d_req_id, (transfer_id, _) in meta.req_blocks.items():
            pending_reqs[d_req_id] = self.reqs_need_send.setdefault(
                transfer_id, SendBlockMeta("", transfer_id)
            )

        async def wait_and_ret(d_req_id, send_meta):
            await send_meta.ready.wait()
            return d_req_id, send_meta

        wait_tasks = [
            asyncio.create_task(wait_and_ret(d_req_id, send_meta))
            for d_req_id, send_meta in pending_reqs.items()
        ]
        try:
            while wait_tasks:
                done, pending = await asyncio.wait(
                    wait_tasks,
                    timeout=self.timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    logger.warning(
                        "Timeout waiting for P side ready: %s", list(pending_reqs)
                    )
                    response = KVTransferResponse(
                        KVTransferStatus.FINISH,
                        err_reqs=list(pending_reqs),
                        err_msg="Timeout waiting for P side ready.",
                    )
                    await socket.send_multipart(
                        (identity, self._encoder.encode(response))
                    )
                    break

                wait_tasks = list(pending)
                response_status = (
                    KVTransferStatus.CONTINUE if wait_tasks else KVTransferStatus.FINISH
                )
                ready_reqs = []
                expired_reqs = []
                for task in done:
                    d_req_id, send_meta = task.result()
                    del pending_reqs[d_req_id]
                    if self.reqs_need_send.get(send_meta.transfer_id) is send_meta:
                        send_meta.sending += 1
                        if not send_meta.need_send:
                            self.resolve_need_send(send_meta, remote_tp_ranks)
                        ready_reqs.append((d_req_id, send_meta))
                    else:
                        # MTSC cancel() wakes an early receiver after removing
                        # its source metadata. No source memory is published.
                        expired_reqs.append(d_req_id)

                try:
                    (
                        src,
                        dst,
                        lengths,
                        err_reqs,
                        err_msg,
                    ) = await self._build_transfer_params(
                        ready_reqs, meta, remote_regions
                    )
                    err_req_set = set(err_reqs)
                    ok_reqs = [rid for rid, _ in ready_reqs if rid not in err_req_set]
                    if src:
                        ret = await self._write_buffers(
                            f"{meta.remote_hostname}:{meta.remote_port}",
                            src,
                            dst,
                            lengths,
                        )
                        if ret != 0:
                            err_msg = f"Mooncake transfer engine returned {ret}"
                            err_reqs.extend(ok_reqs)
                            err_req_set.update(ok_reqs)
                            ok_reqs = []

                    for d_req_id, send_meta in ready_reqs:
                        if d_req_id in err_req_set:
                            continue
                        send_meta.sent += 1
                        if (
                            send_meta.sent == send_meta.need_send
                            and self.reqs_need_send.pop(send_meta.transfer_id, None)
                            is not None
                        ):
                            self.finished_sending_reqs.add(send_meta.p_req_id)
                finally:
                    for _, send_meta in ready_reqs:
                        send_meta.sending -= 1

                response = KVTransferResponse(
                    response_status,
                    ok_reqs=ok_reqs or None,
                    err_reqs=err_reqs + expired_reqs or None,
                    err_msg=err_msg
                    or (
                        "Source request is no longer available."
                        if expired_reqs
                        else None
                    ),
                )
                await socket.send_multipart((identity, self._encoder.encode(response)))
        finally:
            for task in wait_tasks:
                task.cancel()
            if wait_tasks:
                await asyncio.gather(*wait_tasks, return_exceptions=True)

    def resolve_need_send(self, send_meta, remote_tp_ranks):
        send_meta.need_send = len(remote_tp_ranks)

    async def fetch_finished_sending_reqs(self) -> list[TransferResult]:
        results = [TransferResult(rid) for rid in self.finished_sending_reqs]
        self.finished_sending_reqs.clear()
        now = time.perf_counter()
        for transfer_id, send_meta in list(self.reqs_need_send.items()):
            if (
                send_meta.p_req_id
                and send_meta.expire_time < now
                and send_meta.sending == 0
            ):
                logger.warning(
                    "P request %s expired before sending completed", send_meta.p_req_id
                )
                results.append(TransferResult(send_meta.p_req_id, "Transfer timed out"))
                del self.reqs_need_send[transfer_id]
        return results

    async def fetch_finished_recving_reqs(self) -> list[TransferResult]:
        """Collect receiver-owned completion state on receiver_loop."""
        await self._check_receive_error()
        finished = self.finished_recving_reqs
        self.finished_recving_reqs = set()
        results = []
        for request_id in finished:
            event = self._recv_events.pop(request_id)
            failed = request_id in self._failed_recv
            if failed:
                self._invalid[request_id] = {
                    block for ids in event.block_ids for block in ids if block >= 0
                }
            results.append(
                TransferResult(
                    request_id, "Transfer receive failed" if failed else None
                )
            )
            self._failed_recv.discard(request_id)
            self._retired_recvs.add(event.transfer_id)
            self._receive_futures.pop(request_id, None)
        return results

    def poll(self) -> TransferPollResult:
        result = TransferPollResult()
        if self.is_producer:
            result.sends = asyncio.run_coroutine_threadsafe(
                self.fetch_finished_sending_reqs(), self.sender_loop
            ).result()
        if self.is_consumer:
            result.recvs = asyncio.run_coroutine_threadsafe(
                self.fetch_finished_recving_reqs(), self.receiver_loop
            ).result()
        return result

    async def _take_receive_errors(self):
        await self._check_receive_error()
        result = {block for ids in self._invalid.values() for block in ids}
        self._invalid.clear()
        return result

    def take_errors(self) -> set[int]:
        if self.is_consumer:
            return asyncio.run_coroutine_threadsafe(
                self._take_receive_errors(), self.receiver_loop
            ).result()
        return set()

    async def _cancel_send(self, request_id, transfer_id):
        send_meta = self.reqs_need_send.pop(transfer_id, None)
        if send_meta is not None:
            # Wake early D waiters; removal prevents new source reads.
            send_meta.ready.set()
            # Fence reads before preemption reuses the source blocks.
            while send_meta.sending:
                await asyncio.sleep(0.001)
        self.finished_sending_reqs.discard(request_id)

    def cancel(self, request_id: str, transfer_id: str) -> None:
        self._check_open()
        if self.sender_loop is None:
            raise ValueError("Only P can send KV")
        asyncio.run_coroutine_threadsafe(
            self._cancel_send(request_id, transfer_id), self.sender_loop
        ).result()

    async def _preempt_send(self, request_id):
        transfer_ids = [
            tid
            for tid, meta in self.reqs_need_send.items()
            if meta.p_req_id == request_id and meta.ready.is_set()
        ]
        for transfer_id in transfer_ids:
            await self._cancel_send(request_id, transfer_id)

    def preempt(self, request_id: str) -> None:
        if self.is_consumer:
            self._fence_receives({request_id})
        if self.is_producer:
            asyncio.run_coroutine_threadsafe(
                self._preempt_send(request_id), self.sender_loop
            ).result()

    async def _abort_all_sends(self):
        pending = list(self.reqs_need_send.values())
        self.reqs_need_send.clear()
        for send_meta in pending:
            send_meta.ready.set()

    def close(self) -> None:
        if self._closed:
            return
        if self.is_consumer:
            # Check on receiver_loop before changing lifecycle state.
            asyncio.run_coroutine_threadsafe(
                self._check_receive_error(), self.receiver_loop
            ).result()
        self._closed = True
        if self.is_producer:
            asyncio.run_coroutine_threadsafe(
                self._abort_all_sends(), self.sender_loop
            ).result()
        # D may have already published destination addresses. A remote P can
        # legally keep writing until its terminal response arrives, so fence
        # every receive before registered cache memory can be torn down.
        if self.is_consumer:
            self._fence_receives()
        if self.sender_loop is not None:

            async def drain_sender():
                await self.sender_worker_queue.join()
                if self._sender_listener_future is not None:
                    self._sender_listener_future.cancel()
                    await asyncio.sleep(0)
                if self.sender_tasks:
                    await asyncio.gather(
                        *tuple(self.sender_tasks), return_exceptions=True
                    )

            asyncio.run_coroutine_threadsafe(drain_sender(), self.sender_loop).result()
        if self._sender_executor is not None:
            self._sender_executor.shutdown(wait=True, cancel_futures=True)
        owner = self.receiver_loop or self.sender_loop
        if owner is not None:

            async def destroy_context():
                self._ctx.destroy(linger=0)
                await asyncio.sleep(0)

            asyncio.run_coroutine_threadsafe(destroy_context(), owner).result()
        else:
            self._ctx.destroy(linger=0)
        for loop, thread in (
            (self.sender_loop, self._sender_listener_t),
            (self.receiver_loop, self._receiver_t),
        ):
            if loop is None:
                continue
            asyncio.run_coroutine_threadsafe(
                loop.shutdown_default_executor(), loop
            ).result()
            loop.call_soon_threadsafe(loop.stop)
            thread.join()
            loop.close()
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
        self.reqs_need_send.clear()
        self.finished_sending_reqs.clear()
        self.kv_caches.clear()

    async def _register_worker(self, side_port: int) -> None:
        payload = WorkerRegistration(
            engine_id=self.engine_id,
            dp_rank=self.dp_rank,
            tp_rank=self.tp_rank,
            pp_rank=self.pp_rank,
            addr=make_zmq_path("tcp", self.hostname, side_port),
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

    async def _sender_listener(self, ready: threading.Event) -> None:
        socket = self._ctx.socket(zmq.ROUTER)
        try:
            port = socket.bind_to_random_port(f"tcp://{self.hostname}")
            await self._register_worker(port)
            self.sender_tasks.extend(
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
            for task in self.sender_tasks:
                task.cancel()
            if self.sender_tasks:
                await asyncio.gather(*self.sender_tasks, return_exceptions=True)
            self.sender_tasks.clear()
            socket.close(linger=0)

    async def _sender_worker(self, socket) -> None:
        while True:
            identity, payload = await self.sender_worker_queue.get()
            try:
                await self.send_kv_to_decode(identity, payload, socket)
            except Exception as exc:  # noqa: BLE001 - sender worker boundary
                logger.warning("MTSC sender worker failed: error=%s", exc)
            finally:
                self.sender_worker_queue.task_done()

    async def _build_transfer_params(self, ready_reqs, request, remote_regions):
        src, dst, lengths, err_reqs = [], [], [], []
        err_msg = None
        for d_req_id, send_meta in ready_reqs:
            try:
                local_src, local_dst, local_lengths = (
                    self._build_request_transfer_params(
                        d_req_id, send_meta, request, remote_regions
                    )
                )
            except Exception as exc:  # noqa: BLE001 - per-request validation
                logger.warning(
                    "MTSC transfer plan failed: request_id=%s error=%s", d_req_id, exc
                )
                err_reqs.append(d_req_id)
                if err_msg is None:
                    err_msg = str(exc)
                continue
            src.extend(local_src)
            dst.extend(local_dst)
            lengths.extend(local_lengths)
        return src, dst, lengths, err_reqs, err_msg

    def _build_request_transfer_params(
        self, request_id, send_meta, request, remote_regions=None
    ):
        if remote_regions is None:
            remote_regions = self._prepare_transfer_regions(request)
        if remote_regions is None:
            raise ValueError("Peer layout is missing")
        _, requested_groups = request.req_blocks[request_id]
        destination_groups = [list(group) for group in requested_groups]
        if len(send_meta.local_block_ids) != len(destination_groups):
            raise ValueError("P/D KV group count mismatch")
        # Match Mooncake: trim each group independently, then use the same
        # flattened block table for every registered transfer region.
        local_ids, remote_ids = [], []
        for local, remote in zip(
            send_meta.local_block_ids, destination_groups, strict=True
        ):
            if len(local) < len(remote):
                raise ValueError("P has fewer source blocks than D requested")
            local_ids.extend(local[-len(remote) :] if remote else [])
            remote_ids.extend(remote)
        blocks = [(s, d) for s, d in zip(local_ids, remote_ids, strict=True) if d >= 0]
        src, dst, sizes = [], [], []
        for local, remote in zip(
            self._get_transfer_regions(
                self.kv_caches_base_addr, self.block_len_per_layer
            ),
            remote_regions,
            strict=True,
        ):
            if any(s < 0 or s >= self.num_blocks for s, _ in blocks):
                raise ValueError("Requested source block is outside registered memory")
            source_runs, destination_runs = group_concurrent_contiguous(
                [s for s, _ in blocks],
                [d for _, d in blocks],
            )
            copy, src_offset, dst_offset, length = _compute_sender_transfer_plan(
                self.tp_rank,
                self.tp_size,
                request.remote_tp_rank,
                request.remote_tp_size,
                local.kv_block_len,
                remote.kv_block_len,
                self.transfer_topo.local_replicates_kv_cache,
            )
            if not copy:
                continue
            assert src_offset + length <= local.kv_block_len, (
                "Computed source transfer region exceeds local KV block size."
            )
            assert dst_offset + length <= remote.kv_block_len, (
                "Computed destination transfer region exceeds remote KV block size."
            )
            coalesce = _can_coalesce_block_transfers(
                local.block_len,
                remote.block_len,
                src_offset,
                dst_offset,
                length,
            )
            for source_run, destination_run in zip(source_runs, destination_runs):
                if coalesce:
                    src.append(local.base_addr + source_run[0] * local.block_len)
                    dst.append(remote.base_addr + destination_run[0] * remote.block_len)
                    sizes.append(length * len(source_run))
                else:
                    for source_block, destination_block in zip(
                        source_run, destination_run
                    ):
                        src.append(
                            local.base_addr
                            + source_block * local.block_len
                            + src_offset
                        )
                        dst.append(
                            remote.base_addr
                            + destination_block * remote.block_len
                            + dst_offset
                        )
                        sizes.append(length)
        return src, dst, sizes

    async def _write_buffers(self, session, src, dst, sizes) -> int:
        assert self.sender_loop is not None
        return await self.sender_loop.run_in_executor(
            self._sender_executor,
            self.engine.batch_transfer_sync_write,
            session,
            src,
            dst,
            sizes,
        )

    async def _connect_to_prefiller_bootstrap(self, remote_bootstrap_addr: str):
        try:
            async with httpx.AsyncClient(trust_env=False) as client:
                response = await client.get(remote_bootstrap_addr + "/query")
                response.raise_for_status()
                for entry in response.json().values():
                    remote_engine_id = entry["engine_id"]
                    workers = {
                        int(tp): {int(pp): addr for pp, addr in pp_map.items()}
                        for tp, pp_map in entry["worker_addr"].items()
                    }
                    if not workers or sorted(workers) != list(range(len(workers))):
                        raise ValueError("Remote P TP workers are not fully registered")
                    if any(sorted(pp_map) != [0] for pp_map in workers.values()):
                        raise ValueError("Mooncake transfer requires PP=1")
                    self._remote_agents[remote_engine_id] = workers
                    self._remote_tp_size[remote_engine_id] = len(workers)
        except Exception as exc:
            logger.error(
                "Failed to connect to bootstrap server %s: %s",
                remote_bootstrap_addr,
                exc,
            )
        finally:
            # Wake all engines sharing this bootstrap, including on failure.
            self._pending_bootstrap_queries.pop(remote_bootstrap_addr).set()

    async def handle_new_engine_id(
        self, remote_engine_id: str, pull_metas: dict[str, PullReqMeta]
    ):
        remote_bootstrap_addr = next(
            iter(pull_metas.values())
        ).remote_bootstrap_addr.rstrip("/")
        if remote_bootstrap_addr not in self._pending_bootstrap_queries:
            self._pending_bootstrap_queries[remote_bootstrap_addr] = asyncio.Event()
            await self._connect_to_prefiller_bootstrap(remote_bootstrap_addr)
        else:
            await self._pending_bootstrap_queries[remote_bootstrap_addr].wait()

        if remote_engine_id not in self._remote_agents:
            logger.error(
                "Failed to find remote engine_id %s from bootstrap server %s",
                remote_engine_id,
                remote_bootstrap_addr,
            )
            for meta in pull_metas.values():
                self._finish_receive(
                    self._recv_events[meta.d_req_id],
                    True,
                    self._receive_futures.get(meta.d_req_id),
                )
            return

        self.receive_kv(remote_engine_id, pull_metas)

    def _finish_receive(self, event, failed, future=None) -> None:
        logger.info(
            "MTSC D receive done: request_id=%s failed=%s", event.request_id, failed
        )
        if failed:
            self._failed_recv.add(event.request_id)
        self.finished_recving_reqs.add(event.request_id)
        if future is not None and not future.done():
            future.set_result(None)

    def receive_kv(self, remote_engine_id: str, pull_metas: dict[str, PullReqMeta]):
        remote_tp_ranks = self.transfer_topo.handshake_target_ranks(
            self._remote_tp_size[remote_engine_id]
        )
        for meta in pull_metas.values():
            meta.pull_tasks_count = len(remote_tp_ranks)
        for remote_tp_rank in remote_tp_ranks:
            worker_addr = self._remote_agents[remote_engine_id][remote_tp_rank][0]
            asyncio.create_task(
                self.receive_kv_from_single_worker(worker_addr, pull_metas)
            )

    def process_pulling_result(self, response, pull_metas, remaining):
        ok_reqs = response.ok_reqs or []
        if response.status in (KVTransferStatus.ERROR, KVTransferStatus.FINISH):
            err_reqs = remaining.difference(ok_reqs)
        else:
            err_reqs = response.err_reqs or []
        self._failed_recv.update(err_reqs)

        for request_id in [*ok_reqs, *err_reqs]:
            meta = pull_metas[request_id]
            meta.pull_tasks_count -= 1
            if meta.pull_tasks_count == 0:
                self._finish_receive(
                    self._recv_events[request_id],
                    request_id in self._failed_recv,
                    self._receive_futures.get(request_id),
                )
        remaining.difference_update(ok_reqs)
        remaining.difference_update(err_reqs)

    async def receive_kv_from_single_worker(self, worker_addr, pull_metas):
        request = KVTransferRequest(
            remote_hostname=self.hostname,
            remote_port=self.rpc_port,
            remote_tp_size=self.tp_size,
            remote_tp_rank=self.tp_rank,
            req_blocks={
                rid: (meta.transfer_id, meta.local_block_ids)
                for rid, meta in pull_metas.items()
            },
            kv_caches_base_addr=self.kv_caches_base_addr,
            block_lens=self.block_len_per_layer,
        )
        socket = self._ctx.socket(zmq.DEALER)
        remaining, published = set(pull_metas), False
        try:
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVTIMEO, int((self.timeout + 60) * 1000))
            socket.connect(worker_addr)
            published = True
            await socket.send(self._encoder.encode(request))
            while True:
                response = self._response_decoder.decode(await socket.recv())
                self.process_pulling_result(response, pull_metas, remaining)
                if response.status in (KVTransferStatus.ERROR, KVTransferStatus.FINISH):
                    break
        except (Exception, asyncio.CancelledError) as exc:
            logger.warning(
                "MTSC D peer pull failed: address=%s error=%s", worker_addr, exc
            )
            if published and remaining:
                if self._receive_error is None:
                    self._receive_error = _ReceiveFenceError(
                        f"Lost P completion fence at {worker_addr}: {exc!r}; "
                        "worker must stop before KV blocks can be reused"
                    )
                # Wake receiver-loop fences. They check the fatal error before
                # authorizing reuse; these signals are not successful receives.
                for future in self._receive_futures.values():
                    if not future.done():
                        future.set_result(None)
            else:
                self.process_pulling_result(
                    KVTransferResponse(KVTransferStatus.ERROR, err_msg=str(exc)),
                    pull_metas,
                    remaining,
                )
        finally:
            socket.close(linger=0)

    async def _wait_for_receives(self, request_ids=None):
        """Fence and discard receiver-owned state before block reuse."""
        await self._check_receive_error()
        drain_all = request_ids is None
        if request_ids is None:
            request_ids = set(self._receive_futures)
        futures = [
            self._receive_futures[rid]
            for rid in request_ids
            if rid in self._receive_futures
        ]
        if futures:
            await asyncio.gather(*futures)
        await self._check_receive_error()
        for request_id in request_ids:
            self._receive_futures.pop(request_id, None)
            event = self._recv_events.pop(request_id, None)
            if event is not None:
                self._retired_recvs.add(event.transfer_id)
            self._invalid.pop(request_id, None)
        self.finished_recving_reqs.difference_update(request_ids)
        self._failed_recv.difference_update(request_ids)
        if drain_all:
            self._invalid.clear()
            self._retired_recvs.clear()

    def _fence_receives(self, request_ids=None) -> None:
        asyncio.run_coroutine_threadsafe(
            self._wait_for_receives(request_ids), self.receiver_loop
        ).result()
