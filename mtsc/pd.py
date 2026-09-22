"""MTSC-owned direct P/D control plane and Mooncake TE data path."""

from __future__ import annotations

import asyncio
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import httpx
import msgspec
import torch
import uvicorn
import zmq
import zmq.asyncio
from fastapi import FastAPI, HTTPException
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

from .device import (
    cache_tensors,
    is_npu_platform,
    npu_kv_nz_enabled,
    npu_registration_regions,
)
from .protocol import (
    PDResponseStatus,
    PDSendUpdate,
    PDTransferRequest,
    PDTransferResponse,
    PDTransferSchema,
)

logger = init_logger(__name__)


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
        self.app = FastAPI()
        self.app.post("/register")(self.register)
        self.app.get("/query")(self.query)
        self.server = uvicorn.Server(
            uvicorn.Config(self.app, host="0.0.0.0", port=port, log_level="warning")
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
        if (
            entry["tp_size"] != payload.tp_size
            or entry["pp_size"] != payload.pp_size
        ):
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
class _Source:
    request_id: str
    transfer_id: str
    block_ids: tuple[list[int], ...] = ()
    ready: threading.Event = field(default_factory=threading.Event)
    published: bool = False
    abort: bool = False
    expected: int = 0
    completed: int = 0
    terminal: int = 0
    active_writes: int = 0
    expires_at: float = float("inf")


def sender_transfer_plan(
    local_rank: int,
    local_size: int,
    remote_rank: int,
    remote_size: int,
    local_length: int,
    remote_length: int,
    replicated: bool,
) -> tuple[bool, int, int, int]:
    """Return (copy, source offset, destination offset, bytes)."""
    if local_size >= remote_size:
        if local_size % remote_size:
            raise ValueError("P/D TP sizes must have an integer ratio")
        ratio = local_size // remote_size
    else:
        if remote_size % local_size:
            raise ValueError("P/D TP sizes must have an integer ratio")
        ratio = -(remote_size // local_size)
    if ratio == 1:
        return True, 0, 0, local_length
    if ratio > 0:
        if replicated:
            return local_rank % ratio == 0, 0, 0, local_length
        return True, 0, (local_rank % ratio) * local_length, local_length
    if replicated:
        return True, 0, 0, local_length
    return True, (remote_rank % -ratio) * remote_length, 0, remote_length


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
    engine_id: str
    is_mla: bool
    total_num_kv_heads: int
    virtually_split_kv_in_blocks: bool = False

    @property
    def local_replicates_kv_cache(self) -> bool:
        return self.is_mla or self.tp_size > self.total_num_kv_heads

    def handshake_target_ranks(self, remote_tp_size: int) -> list[int]:
        if self.tp_size >= remote_tp_size:
            if self.tp_size % remote_tp_size:
                raise ValueError("P/D TP sizes must have an integer ratio")
            return [self.tp_rank // (self.tp_size // remote_tp_size)]
        if remote_tp_size % self.tp_size:
            raise ValueError("P/D TP sizes must have an integer ratio")
        ratio = remote_tp_size // self.tp_size
        return [self.tp_rank * ratio + offset for offset in range(ratio)]


def _transpose_npu_cache_blocks(
    cache: torch.Tensor, block_ids: list[int], block_size: int, tp_ratio: int
) -> None:
    """Restore [token, TP-split, ...] after multiple P shards were appended."""
    if tp_ratio <= 1 or not block_ids:
        return
    if cache.ndim < 2 or cache.shape[1] != block_size:
        raise ValueError(
            "Ascend heterogeneous TP requires [blocks, tokens, ...] KV layout"
        )
    ids = torch.tensor(sorted(set(block_ids)), dtype=torch.long, device=cache.device)
    selected = cache.index_select(0, ids)
    if selected[0].numel() % (tp_ratio * block_size):
        raise ValueError("Ascend KV block cannot be evenly split across P TP ranks")
    transposed = (
        selected.reshape(len(ids), tp_ratio, block_size, -1)
        .transpose(1, 2)
        .contiguous()
        .reshape_as(selected)
    )
    cache.index_copy_(0, ids, transposed)


class PDTransfer:
    """One TE per worker plus MTSC's own listener/receiver state machines."""

    def __init__(self, config: VllmConfig, kv_cache_config: Any) -> None:
        try:
            from mooncake.engine import TransferEngine
        except ImportError as exc:
            raise ImportError("Mooncake TransferEngine bindings are required") from exc
        assert config.kv_transfer_config is not None
        transfer = config.kv_transfer_config
        assert transfer.engine_id is not None
        self.config = config
        self.engine_id = transfer.engine_id
        self.is_producer = transfer.kv_role == "kv_producer"
        self.is_consumer = transfer.kv_role == "kv_consumer"
        self.extra = transfer.kv_connector_extra_config
        self.timeout = float(
            self.extra.get(
                "mtsc_pd_timeout_seconds", envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT
            )
        )
        self.device_id = torch.accelerator.current_device_index()
        current_platform.set_device(self.device_id)
        self.engine = TransferEngine()
        self.hostname = get_ip()
        default_protocol = "ascend" if is_npu_platform() else "rdma"
        ret = self.engine.initialize(
            self.hostname,
            "P2PHANDSHAKE",
            self.extra.get("mooncake_protocol", default_protocol),
            self.extra.get("device_name", ""),
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
            "engine_id": self.engine_id,
            "is_mla": model.use_mla,
            "total_num_kv_heads": model.get_total_num_kv_heads(),
        }
        if is_npu_platform():
            self.topology = _NPUTransferTopology(**topology_args)
        else:
            self.topology = TransferTopology(
                **topology_args,
                is_mamba=kv_cache_config.has_mamba_layers,
                attn_backends=get_current_attn_backends(config),
            )
        self.npu_kv_nz = npu_kv_nz_enabled(config)
        if is_npu_platform():
            cache_layout = "npu-nz" if self.npu_kv_nz else "npu-normal"
        else:
            cache_layout = "mla" if model.use_mla else "hnd"
        self.schema = PDTransferSchema(
            topology_version=1,
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
        self._sources: dict[str, _Source] = {}
        self._source_lock = threading.Lock()
        self._finished_send: set[str] = set()
        self._finished_recv: set[str] = set()
        self._failed_recv: set[str] = set()
        self._result_lock = threading.Lock()
        self._remote_workers: dict[tuple[str, str, int], dict[int, dict[int, str]]] = {}
        self._receive_futures: dict[str, Future[None]] = {}
        self._receive_lock = threading.Lock()
        self._encoder = msgspec.msgpack.Encoder()
        self._request_decoder = msgspec.msgpack.Decoder(PDTransferRequest)
        self._response_decoder = msgspec.msgpack.Decoder(PDTransferResponse)
        self._ctx = zmq.asyncio.Context()
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=_run_loop, args=(self._loop,), name="mtsc-pd", daemon=True
        )
        self._loop_thread.start()
        self._listener_future = None
        self._serve_tasks: set[asyncio.Task[None]] = set()
        self._closing = False
        self._send_pool = ThreadPoolExecutor(
            max_workers=max(1, int(self.extra.get("num_workers", 10))),
            thread_name_prefix="mtsc-te-write",
            initializer=lambda: current_platform.set_device(self.device_id),
        )
        self._bootstrap = None
        if self.is_producer and _launch_bootstrap(config):
            _, port = _bootstrap_address(config)
            self._bootstrap = BootstrapServer(port)
            self._bootstrap.start()

    def register(
        self,
        kv_caches: dict[
            str, torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]
        ],
    ) -> None:
        self.kv_caches = kv_caches
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

    def _npu_nz_pair(self, tensors: list[torch.Tensor], block_ids: list[int]) -> None:
        if len(tensors) != 2 or not block_ids:
            raise ValueError("Ascend NZ conversion requires one K/V tensor pair")
        try:
            import torch_npu
        except ImportError as exc:
            raise RuntimeError("Ascend NZ conversion requires torch_npu") from exc
        k_cache, v_cache = tensors
        if k_cache.ndim != 4 or v_cache.ndim != 4:
            raise ValueError("Ascend NZ conversion requires 4D K/V caches")
        ids = torch.tensor(
            sorted(set(block_ids)), dtype=torch.int32, device=k_cache.device
        )
        num_blocks = ids.numel()
        num_tokens = num_blocks * self.topology.block_size
        num_heads = k_cache.shape[2]
        k_head_dim, v_head_dim = k_cache.shape[3], v_cache.shape[3]
        block_table = ids.view(1, -1)
        block_len = torch.tensor([num_tokens], dtype=torch.int32, device=k_cache.device)
        seq_start = torch.tensor([0], dtype=torch.int32, device=k_cache.device)
        k_buffer = torch.empty(
            (num_tokens, num_heads, k_head_dim),
            dtype=k_cache.dtype,
            device=k_cache.device,
        )
        v_buffer = torch.empty(
            (num_tokens, num_heads, v_head_dim),
            dtype=v_cache.dtype,
            device=v_cache.device,
        )
        torch.npu.synchronize()
        torch_npu.npu_gather_pa_kv_cache(
            k_cache,
            v_cache,
            block_table,
            block_len,
            seq_offset=seq_start,
            key=k_buffer,
            value=v_buffer,
        )
        offsets = torch.arange(
            self.topology.block_size, dtype=torch.int32, device=k_cache.device
        )
        slots = (
            offsets.view(1, -1) + ids.view(-1, 1) * self.topology.block_size
        ).flatten()
        nz = 16
        if k_head_dim * num_heads % nz or v_head_dim * num_heads % nz:
            raise ValueError("Ascend NZ dimensions must be divisible by 16")
        k_nz = k_cache.view(
            -1,
            k_head_dim * num_heads // nz,
            self.topology.block_size,
            nz,
        )
        v_nz = v_cache.view(
            -1,
            v_head_dim * num_heads // nz,
            self.topology.block_size,
            nz,
        )
        torch_npu.npu_scatter_pa_kv_cache(k_buffer, v_buffer, k_nz, v_nz, slots)

    @torch.no_grad()
    def reformat_npu_blocks(
        self, block_ids: list[list[int]], remote_tp_size: int
    ) -> None:
        if not is_npu_platform() or not any(block_ids):
            return
        current_platform.set_device(self.device_id)
        producer_replicated = (
            self.topology.is_mla or remote_tp_size > self.topology.total_num_kv_heads
        )
        ratio = (
            remote_tp_size // self.tp_size
            if remote_tp_size > self.tp_size and not producer_replicated
            else 1
        )
        if remote_tp_size > self.tp_size and remote_tp_size % self.tp_size:
            raise ValueError("P/D TP sizes must have an integer ratio")
        if ratio == 1 and not self.npu_kv_nz:
            return
        reformatted: set[tuple[int, ...]] = set()
        for layer_name, raw in self.kv_caches.items():
            group = self.layer_groups.get(layer_name)
            spec = self.layer_specs.get(layer_name)
            if group is None or group >= len(block_ids) or isinstance(spec, MambaSpec):
                continue
            group_blocks = [block for block in block_ids[group] if block >= 0]
            if not group_blocks:
                continue
            tensors = cache_tensors(raw)
            identity = tuple(tensor.data_ptr() for tensor in tensors)
            if identity in reformatted:
                continue
            reformatted.add(identity)
            if ratio > 1:
                for tensor in tensors:
                    _transpose_npu_cache_blocks(
                        tensor, group_blocks, self.topology.block_size, ratio
                    )
            if self.npu_kv_nz:
                self._npu_nz_pair(tensors, group_blocks)

    async def _register_worker(self, side_port: int) -> None:
        host, port = _bootstrap_address(self.config)
        payload = WorkerRegistration(
            engine_id=self.engine_id,
            dp_rank=self.dp_rank,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            address=make_zmq_path("tcp", self.hostname, side_port),
        )
        url = make_zmq_path("http", host, port) + "/register"
        # Bootstrap is an internal control-plane hop. Never route it through
        # process-wide HTTP(S)_PROXY settings.
        async with httpx.AsyncClient(trust_env=False) as client:
            for _ in range(120):
                try:
                    response = await client.post(url, json=payload.model_dump())
                    response.raise_for_status()
                    return
                except httpx.ConnectError:
                    await asyncio.sleep(0.25)
        raise RuntimeError(f"MTSC bootstrap unavailable: {url}")

    async def _listen(self, ready: threading.Event) -> None:
        socket = self._ctx.socket(zmq.ROUTER)
        try:
            port = socket.bind_to_random_port(f"tcp://{self.hostname}")
            await self._register_worker(port)
            ready.set()
            while True:
                identity, payload = await socket.recv_multipart()
                task = asyncio.create_task(self._serve(identity, payload, socket))
                self._serve_tasks.add(task)
                task.add_done_callback(self._serve_tasks.discard)
        except (asyncio.CancelledError, zmq.ContextTerminated):
            pass
        except Exception:
            ready.set()
            raise
        finally:
            socket.close(linger=0)

    async def _serve(
        self, identity: bytes, payload: bytes, socket: zmq.asyncio.Socket
    ) -> None:
        request_ids: list[str] = []
        failed: list[str] = []
        covered_regions: dict[str, list[int]] = {}
        try:
            if getattr(self, "_closing", False):
                raise RuntimeError("MTSC P listener is closing")
            request = self._request_decoder.decode(payload)
            if request.schema != self.schema:
                raise ValueError(
                    f"P/D transfer schema mismatch: P={self.schema!r} "
                    f"D={request.schema!r}"
                )
            target_ranks = self.topology.handshake_target_ranks(request.tp_size)
            if request.tp_rank not in target_ranks:
                raise ValueError(
                    f"D rank {request.tp_rank} is not paired with P rank {self.tp_rank}"
                )
            if request.pp_size <= 0 or not 0 <= request.pp_rank < request.pp_size:
                raise ValueError("Invalid D pipeline-parallel identity")
            if self.pp_size == request.pp_size and self.pp_rank != request.pp_rank:
                raise ValueError(
                    f"D PP rank {request.pp_rank} is not paired with "
                    f"P PP rank {self.pp_rank}"
                )
            pp_fanout = 1 if self.pp_size == request.pp_size else request.pp_size
            for decode_id, (transfer_id, _) in request.requests.items():
                with self._source_lock:
                    source = self._sources.setdefault(
                        transfer_id, _Source("", transfer_id)
                    )
                    source.expected = max(
                        source.expected, len(target_ranks) * pp_fanout
                    )
                is_ready = await asyncio.to_thread(source.ready.wait, self.timeout)
                if not is_ready or source.abort:
                    failed.append(decode_id)
                    self._finish_source_target(transfer_id, source, False)
                    continue
                with self._source_lock:
                    source.active_writes += 1
                ok = False
                covered: set[int] = set()
                try:
                    ok, covered = await self._write_one(decode_id, source, request)
                except Exception as exc:  # noqa: BLE001 - one target boundary
                    logger.warning(
                        "MTSC P write failed: request_id=%s error=%s",
                        decode_id,
                        exc,
                    )
                finally:
                    self._finish_source_target(
                        transfer_id, source, ok, active_write=True
                    )
                if ok:
                    request_ids.append(decode_id)
                    covered_regions[decode_id] = sorted(covered)
                else:
                    failed.append(decode_id)
            response = PDTransferResponse(
                PDResponseStatus.FINISH,
                request_ids or None,
                failed or None,
                "transfer failed" if failed else None,
                covered_regions or None,
            )
        except Exception as exc:
            logger.exception("MTSC P transfer request failed")
            response = PDTransferResponse(PDResponseStatus.ERROR, error=str(exc))
        await socket.send_multipart((identity, self._encoder.encode(response)))

    def _finish_source_target(
        self,
        transfer_id: str,
        source: _Source,
        successful: bool,
        *,
        active_write: bool = False,
    ) -> None:
        request_id = ""
        with self._source_lock:
            if active_write:
                source.active_writes -= 1
            source.terminal += 1
            if successful:
                source.completed += 1
            if (
                source.active_writes == 0
                and source.terminal >= source.expected
                and (source.published or source.abort)
            ):
                if self._sources.get(transfer_id) is source:
                    self._sources.pop(transfer_id, None)
                if source.published and not source.abort:
                    request_id = source.request_id
        if request_id:
            with self._result_lock:
                self._finished_send.add(request_id)

    def _aligned_regions(
        self, request: PDTransferRequest
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
        self, decode_id: str, source: _Source, request: PDTransferRequest
    ) -> tuple[bool, set[int]]:
        _, destination_groups = request.requests[decode_id]
        if not any(destination_groups):
            return True, set()
        if len(source.block_ids) != len(destination_groups):
            return False, set()
        source_groups: list[list[int]] = []
        for local, remote in zip(source.block_ids, destination_groups, strict=True):
            if len(local) < len(remote):
                return False, set()
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
            if not destination_groups[group]:
                continue
            copy, src_offset, dst_offset, length = sender_transfer_plan(
                self.tp_rank,
                self.tp_size,
                request.tp_rank,
                request.tp_size,
                local_region.kv_block_length,
                remote_region.kv_block_length,
                self.topology.local_replicates_kv_cache,
            )
            if not copy:
                continue
            self._validate_region_plan(
                local_region,
                remote_region,
                self.tp_size,
                request.tp_size,
                self.topology.local_replicates_kv_cache,
                src_offset,
                dst_offset,
                length,
            )
            covered.add(remote_index)
            for source_block, destination_block in zip(
                source_groups[group], destination_groups[group], strict=True
            ):
                src.append(
                    local_region.base_address
                    + source_block * local_region.block_length
                    + src_offset
                )
                dst.append(
                    remote_region.base_address
                    + destination_block * remote_region.block_length
                    + dst_offset
                )
                sizes.append(length)
        if not src:
            return True, covered
        session = f"{request.hostname}:{request.rpc_port}"
        ret = await self._loop.run_in_executor(
            self._send_pool,
            self.engine.batch_transfer_sync_write,
            session,
            src,
            dst,
            sizes,
        )
        if ret != 0:
            logger.warning("MTSC TE WRITE failed: request_id=%s ret=%s", decode_id, ret)
        return ret == 0, covered if ret == 0 else set()

    def apply_updates(self, updates: list[PDSendUpdate]) -> None:
        if not self.is_producer:
            return
        completed: set[str] = set()
        with self._source_lock:
            for update in updates:
                source = self._sources.setdefault(
                    update.transfer_id, _Source(update.request_id, update.transfer_id)
                )
                if update.abort:
                    source.abort = True
                    source.request_id = update.request_id
                    source.expires_at = min(source.expires_at, time.monotonic())
                    source.ready.set()
                elif update.source_ready:
                    source.request_id = update.request_id
                    source.block_ids = update.block_ids
                    source.published = True
                    source.expires_at = time.monotonic() + self.timeout
                    source.ready.set()
                    if (
                        source.expected > 0
                        and source.terminal >= source.expected
                        and source.active_writes == 0
                    ):
                        self._sources.pop(update.transfer_id, None)
                        completed.add(update.request_id)
        if completed:
            with self._result_lock:
                self._finished_send.update(completed)

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
        if tp_size <= 0 or pp_size <= 0:
            raise ValueError("Remote P topology dimensions are missing")
        if sorted(workers) != list(range(tp_size)):
            raise ValueError("Remote P TP workers are not fully registered")
        expected_pp = list(range(pp_size))
        if any(sorted(pp_map) != expected_pp for pp_map in workers.values()):
            raise ValueError("Remote P PP workers are not fully registered")
        self._remote_workers[key] = workers
        return workers

    def receive(
        self,
        request_id: str,
        transfer_id: str,
        block_ids: list[list[int]],
        remote_engine_id: str,
        bootstrap_address: str,
        remote_dp_rank: int,
    ) -> None:
        future = asyncio.run_coroutine_threadsafe(
            self._receive(
                request_id,
                transfer_id,
                block_ids,
                remote_engine_id,
                bootstrap_address,
                remote_dp_rank,
            ),
            self._loop,
        )
        with self._receive_lock:
            previous = self._receive_futures.get(request_id)
            if previous is not None and not previous.done():
                future.cancel()
                raise RuntimeError(f"Duplicate PD receive for request {request_id}")
            self._receive_futures[request_id] = future

    def _validate_coverage(
        self,
        request_id: str,
        block_ids: list[list[int]],
        responses: list[PDTransferResponse],
        expected: int,
    ) -> None:
        """Require every data-carrying D region to have exactly one TP fan-in."""
        coverage: Counter[int] = Counter()
        for response in responses:
            coverage.update((response.covered_regions or {}).get(request_id, []))
        required = {
            index
            for index, region in enumerate(self.regions)
            if region.group_index < len(block_ids) and block_ids[region.group_index]
        }
        invalid = [index for index in required if coverage[index] != expected]
        unexpected = [index for index in coverage if index not in required]
        if invalid or unexpected:
            raise ValueError(
                "P/D region coverage mismatch: "
                f"expected={expected} invalid={invalid} unexpected={unexpected}"
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
        failed = False
        try:
            workers = await self._query_workers(
                bootstrap, remote_engine_id, remote_dp_rank
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
            request = PDTransferRequest(
                hostname=self.hostname,
                rpc_port=self.rpc_port,
                tp_size=self.tp_size,
                tp_rank=self.tp_rank,
                pp_size=self.pp_size,
                pp_rank=self.pp_rank,
                schema=self.schema,
                requests={request_id: (transfer_id, block_ids)},
                region_base_addresses=[region.base_address for region in self.regions],
                block_lengths=[region.block_length for region in self.regions],
                kv_block_lengths=[region.kv_block_length for region in self.regions],
                layer_names=[region.layer_name for region in self.regions],
                layer_indices=[region.layer_index for region in self.regions],
                group_indices=[region.group_index for region in self.regions],
            )
            payload = self._encoder.encode(request)

            async def call(address: str) -> PDTransferResponse:
                socket = self._ctx.socket(zmq.DEALER)
                socket.setsockopt(zmq.LINGER, 0)
                socket.connect(address)
                try:
                    await socket.send(payload)
                    # Once P has the destination addresses it may already be
                    # writing. Do not time out locally and release D blocks;
                    # wait for P's terminal response to fence the DMA.
                    raw = await socket.recv()
                    return self._response_decoder.decode(raw)
                finally:
                    socket.close(linger=0)

            responses = await asyncio.gather(*(call(address) for address in addresses))
            failed = any(
                response.status != PDResponseStatus.FINISH
                or request_id not in (response.completed or [])
                or request_id in (response.failed or [])
                for response in responses
            )
            if not failed and any(block_ids):
                producer_replicated = (
                    self.schema.is_mla
                    or len(workers) > self.topology.total_num_kv_heads
                )
                expected = 1 if producer_replicated else len(target_tp)
                self._validate_coverage(
                    request_id, block_ids, responses, expected
                )
            if not failed:
                self.reformat_npu_blocks(block_ids, len(workers))
        except Exception as exc:  # noqa: BLE001 - async transport boundary
            failed = True
            logger.warning(
                "MTSC D pull failed: request_id=%s error=%s", request_id, exc
            )
        with self._result_lock:
            (self._failed_recv if failed else self._finished_recv).add(request_id)

    def finish_receives(self, request_ids: set[str]) -> None:
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
            self._finished_recv.difference_update(request_ids)
            self._failed_recv.difference_update(request_ids)

    def poll(self) -> tuple[set[str], set[str], set[str]]:
        with self._source_lock:
            now = time.monotonic()
            for transfer_id, source in list(self._sources.items()):
                if (
                    source.ready.is_set()
                    and source.expires_at < now
                    and source.active_writes == 0
                    and (
                        source.abort
                        or (
                            source.published
                            and (
                                source.expected == 0
                                or source.terminal < source.expected
                            )
                        )
                    )
                ):
                    self._sources.pop(transfer_id, None)
                    if source.published and not source.abort:
                        with self._result_lock:
                            self._finished_send.add(source.request_id)
                    logger.warning(
                        "MTSC P source timed out: request_id=%s terminal=%d expected=%d",
                        source.request_id,
                        source.terminal,
                        source.expected,
                    )
        with self._result_lock:
            sends, recvs, failed = (
                self._finished_send,
                self._finished_recv,
                self._failed_recv,
            )
            self._finished_send, self._finished_recv, self._failed_recv = (
                set(),
                set(),
                set(),
            )
        terminal_recvs = recvs | failed
        if terminal_recvs:
            with self._receive_lock:
                for request_id in terminal_recvs:
                    future = self._receive_futures.get(request_id)
                    if future is not None and future.done():
                        self._receive_futures.pop(request_id, None)
        return sends, recvs, failed

    def close(self) -> None:
        self._closing = True
        with self._source_lock:
            for source in self._sources.values():
                source.abort = True
                source.ready.set()
        # D may have already published destination addresses. A remote P can
        # legally keep writing until its terminal response arrives, so fence
        # every receive before registered cache memory can be torn down.
        with self._receive_lock:
            receive_ids = set(self._receive_futures)
        self.finish_receives(receive_ids)
        # A running WRITE still uses both registered GPU memory and the TE.
        self._send_pool.shutdown(wait=True, cancel_futures=True)
        if self._listener_future is not None:
            self._listener_future.cancel()
        if self._loop.is_running():

            async def close_context() -> None:
                # ZMQ sockets belong to this event-loop thread. Destroying the
                # context from the vLLM main thread can trip libzmq's signaler
                # assertion during process shutdown.
                if self._serve_tasks:
                    await asyncio.gather(
                        *tuple(self._serve_tasks), return_exceptions=True
                    )
                self._ctx.destroy(linger=0)
                await asyncio.sleep(0)

            future = asyncio.run_coroutine_threadsafe(close_context(), self._loop)
            try:
                future.result(timeout=5)
            except Exception:  # noqa: BLE001 - best-effort teardown
                logger.warning("Timed out while closing MTSC PD ZMQ context")
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join(timeout=5)
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
                    ret = self.engine.batch_unregister_memory(
                        self._registered_storage
                    )
                    if ret != 0:
                        logger.warning(
                            "MTSC TE memory unregistration failed: ret=%s", ret
                        )
            except Exception as exc:  # noqa: BLE001 - binding teardown boundary
                logger.warning("MTSC TE memory unregistration failed: error=%s", exc)
            finally:
                self._registered_storage = []
