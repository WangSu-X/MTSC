# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Region alignment and TP planning track vLLM da9134bc89364465f68aa679564435936d157a14.
"""Worker KV transfer contract and Mooncake Transfer Engine implementation."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
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
from vllm.v1.kv_cache_interface import MambaSpec

from .protocol import (
    KVTransferRequest,
    KVTransferResponse,
    KVTransferStatus,
)
from .utils import (
    BlockIds,
    cache_tensors,
    is_npu_platform,
    npu_kv_nz_enabled,
    npu_registration_regions,
)


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
    derives TP pairing and waits for all required producer replies.
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
        If the transport loses a remote WRITE fence, raise a fatal worker error
        instead of publishing an ordinary failed recv or authorizing reuse.
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
        A lost receive fence raises and retains registered memory until worker
        teardown; a timeout alone cannot prove remote WRITEs have stopped.
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
    layer_name: str = ""
    layer_index: int = -1
    group_index: int = 0
    shared_group_ids: tuple[int, ...] = ()
    row_offset: int = -1


_SHARED_REGION_GROUP_ID = -1


def _block_ids_for_region(
    block_ids_by_group: list[list[int]],
    group_index: int,
    shared_group_ids: tuple[int, ...] = (),
) -> list[int]:
    """Follow NIXL `_block_ids_by_region`.

    -1 flattens only the groups that share this allocation. NIXL can flatten
    every group because its packed flag means they all share one storage;
    Mooncake can still have a separate Mamba group in the same request.
    """
    if group_index == _SHARED_REGION_GROUP_ID:
        assert shared_group_ids, "Shared packed region is missing its group ids."
        return [
            block for group in shared_group_ids for block in block_ids_by_group[group]
        ]
    assert 0 <= group_index < len(block_ids_by_group), (
        "Transfer region references a missing KV group."
    )
    return list(block_ids_by_group[group_index])


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
    kv_block_lens: list[int],
    layer_names: list[str],
    layer_indices: list[int],
    group_indices: list[int] | None = None,
    shared_group_ids: list[tuple[int, ...]] | None = None,
    row_offsets: list[int] | None = None,
) -> list[TransferRegion]:
    """Expand registered KV tensors into the regions transferred by Mooncake."""
    assert (
        len(base_addrs)
        == len(block_lens)
        == len(kv_block_lens)
        == len(layer_names)
        == len(layer_indices)
    ), (
        "Mooncake transfer regions require matching metadata lengths, got "
        f"base_addrs={len(base_addrs)}, block_lens={len(block_lens)}, "
        f"kv_block_lens={len(kv_block_lens)}, "
        f"layer_names={len(layer_names)}, "
        f"layer_indices={len(layer_indices)}."
    )
    if group_indices is None:
        group_indices = [0] * len(layer_names)
    assert len(group_indices) == len(layer_names), (
        "Mooncake transfer regions require matching group metadata lengths, "
        f"got group_indices={len(group_indices)}, layer_names={len(layer_names)}."
    )
    if shared_group_ids is None:
        shared_group_ids = [() for _ in layer_names]
    assert len(shared_group_ids) == len(layer_names)
    # Missing offsets must not look like row starts, or every run would be
    # promoted to the full row.
    if row_offsets is None:
        row_offsets = [-1] * len(layer_names)
    assert len(row_offsets) == len(layer_names)
    regions: list[TransferRegion] = []
    for (
        base_addr,
        block_len,
        kv_block_len,
        layer_name,
        layer_index,
        group_index,
        region_groups,
        row_offset,
    ) in zip(
        base_addrs,
        block_lens,
        kv_block_lens,
        layer_names,
        layer_indices,
        group_indices,
        shared_group_ids,
        row_offsets,
    ):
        regions.append(
            TransferRegion(
                layer_name=layer_name,
                layer_index=layer_index,
                base_addr=base_addr,
                block_len=block_len,
                kv_block_len=kv_block_len,
                group_index=group_index,
                shared_group_ids=region_groups,
                row_offset=row_offset,
            )
        )
    return regions


def _legacy_transfer_regions(
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
    total_num_kv_heads: int | None = None,
) -> tuple[bool, int, int, int]:
    """Plan one producer-rank to one consumer-rank copy for heterogeneous TP."""
    tp_ratio = _get_tp_ratio(local_tp_size, remote_tp_size)

    if tp_ratio == 1:
        return True, 0, 0, local_kv_block_len

    if total_num_kv_heads is not None:
        consumer_cache_replicated = remote_tp_size > total_num_kv_heads
        if producer_cache_replicated != consumer_cache_replicated:
            local_head_count = max(total_num_kv_heads // local_tp_size, 1)
            local_replica_count = max(local_tp_size // total_num_kv_heads, 1)

            local_head = local_tp_rank * total_num_kv_heads // local_tp_size
            remote_head = remote_tp_rank * total_num_kv_heads // remote_tp_size
            bytes_per_head = local_kv_block_len // local_head_count

            return (
                local_tp_rank % local_replica_count == 0,
                max(remote_head - local_head, 0) * bytes_per_head,
                max(local_head - remote_head, 0) * bytes_per_head,
                bytes_per_head,
            )

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
    total_num_kv_heads: int | None = None,
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

    if total_num_kv_heads is not None:
        # TP ranks beyond the KV-head count replicate existing shards.
        local_tp_size = min(local_tp_size, total_num_kv_heads)
        remote_tp_size = min(remote_tp_size, total_num_kv_heads)
    elif producer_cache_replicated:
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


def _align_transfer_regions(
    local_regions: list[TransferRegion],
    remote_regions: list[TransferRegion],
    *,
    allow_partial_layers: bool = False,
) -> tuple[list[TransferRegion], list[TransferRegion], str | None]:
    """Align KV transfer regions by registered layer-name occurrence.

    PP shards own different layer subsets. Positional matching is therefore
    wrong once producer and consumer have different PP layouts. Multiple
    registered transfer buffers for the same layer are represented by repeated
    layer names and matched by occurrence order.
    """

    def keyed_regions(
        regions: list[TransferRegion],
    ) -> list[tuple[tuple[str, int], TransferRegion]]:
        counts: dict[str, int] = defaultdict(int)
        keyed: list[tuple[tuple[str, int], TransferRegion]] = []
        for region in regions:
            occurrence = counts[region.layer_name]
            counts[region.layer_name] += 1
            keyed.append(((region.layer_name, occurrence), region))
        return keyed

    local_keyed = keyed_regions(local_regions)
    remote_keyed = keyed_regions(remote_regions)
    remote_by_key = dict(remote_keyed)
    aligned_local: list[TransferRegion] = []
    aligned_remote: list[TransferRegion] = []
    for key, local_region in local_keyed:
        remote_region = remote_by_key.get(key)
        if remote_region is None:
            if (
                allow_partial_layers
                and (local_region.layer_name, 0) not in remote_by_key
            ):
                continue
            return (
                [],
                [],
                (
                    "Mooncake producer registered layer has no matching "
                    f"consumer occurrence: {key[0]} occurrence {key[1]}."
                ),
            )
        if local_region.layer_index != remote_region.layer_index:
            return (
                [],
                [],
                (
                    "Mooncake registered layer index mismatch for "
                    f"{local_region.layer_name}: producer="
                    f"{local_region.layer_index}, consumer="
                    f"{remote_region.layer_index}."
                ),
            )
        if local_region.group_index != remote_region.group_index:
            return (
                [],
                [],
                (
                    "Mooncake registered group index mismatch for "
                    f"{local_region.layer_name}: producer="
                    f"{local_region.group_index}, consumer="
                    f"{remote_region.group_index}."
                ),
            )
        if (
            local_region.shared_group_ids
            and remote_region.shared_group_ids
            and local_region.shared_group_ids != remote_region.shared_group_ids
        ):
            return (
                [],
                [],
                (
                    "Mooncake shared-group set mismatch for "
                    f"{local_region.layer_name}: producer="
                    f"{local_region.shared_group_ids}, consumer="
                    f"{remote_region.shared_group_ids}."
                ),
            )
        aligned_local.append(local_region)
        aligned_remote.append(remote_region)

    return aligned_local, aligned_remote, None


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
    peer: int | None = None
    targets: dict[tuple, asyncio.Future] = field(default_factory=dict)
    descriptors: dict[tuple, tuple] = field(default_factory=dict)


class MooncakeKVTransfer(KVTransfer):
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
        if config.parallel_config.pipeline_parallel_size > 1:
            raise ValueError("Mooncake transfer does not support pipeline parallelism")
        self.num_blocks = 0
        self.has_mamba = kv_cache_config.has_mamba_layers
        self.npu_kv_nz = npu_kv_nz_enabled(config)
        self._head_resharding_errors: list[str] = []
        self._receive_error: _ReceiveFenceError | None = None
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
        self.kv_caches_base_addr: list[int] = []
        self.block_lens: list[int] = []
        self.regions: list[TransferRegion] = []
        self._registered_storage: list[int] = []
        self._sources: dict[str, _Session] = {}
        self._source_lock = threading.Lock()
        self._finished_send: set[str] = set()
        self._failed_recv: set[str] = set()
        self._result_lock = threading.Lock()
        self._remote_workers: dict[tuple[str, str, int], dict[int, dict[int, str]]] = {}
        self._remote_layouts: OrderedDict[tuple, list[TransferRegion]] = OrderedDict()
        self._pending_bootstrap_queries: dict[str, asyncio.Future[dict[str, Any]]] = {}
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
        self._check_receive_error()
        if self._closed:
            raise RuntimeError("KV transfer is closed")

    def _check_receive_error(self) -> None:
        if self._receive_error is not None:
            raise self._receive_error

    def _cache_regions(self, cache, spec):
        """Describe payloads independently from their physical block strides."""
        stride = cache.stride(0) * cache.element_size()
        if is_npu_platform() and not isinstance(spec, MambaSpec):
            if getattr(self, "npu_kv_nz", False):
                if not self.topology.is_mla:
                    self._head_resharding_errors.append(
                        "Ascend NZ KV-head re-sharding is unsupported"
                    )
                return [TransferRegion(cache.data_ptr(), stride, stride)]
            if not self.topology.is_mla:
                # Standard Ascend caches are [blocks, tokens, heads, dim].
                # A token's heads are contiguous; the whole block's heads are
                # not. Expose each token as an upstream transfer region so
                # both splitting and gathering preserve token/head order.
                if (
                    cache.ndim == 4
                    and cache.shape[1] == self.topology.block_size
                    and cache.stride(3) == 1
                    and cache.stride(2) == cache.shape[3]
                    and cache.stride(1) >= cache.shape[2] * cache.shape[3]
                ):
                    payload = cache.shape[2] * cache.shape[3] * cache.element_size()
                    return [
                        TransferRegion(
                            cache.data_ptr()
                            + token * cache.stride(1) * cache.element_size(),
                            stride,
                            payload,
                        )
                        for token in range(cache.shape[1])
                    ]
                # A logical NHD view over contiguous HND storage is already
                # head-major and can be sliced as one payload.
                if not (cache.ndim == 4 and cache.transpose(1, 2)[0].is_contiguous()):
                    self._head_resharding_errors.append(
                        "Unsupported Ascend KV-head re-sharding layout"
                    )
            return [TransferRegion(cache.data_ptr(), stride, stride)]
        if isinstance(spec, MambaSpec):
            return [TransferRegion(cache.data_ptr(), stride, spec.page_size_bytes)]
        parts = (
            [cache[:, index] for index in range(2)]
            if self.topology.virtually_split_kv_in_blocks
            else [cache]
        )
        regions = []
        for part in parts:
            row = part[0]
            span = (
                1
                + sum((size - 1) * step for size, step in zip(row.shape, row.stride()))
            ) * part.element_size()
            storage = part.untyped_storage()
            if span == row.nbytes and span <= stride:
                regions.append(TransferRegion(part.data_ptr(), stride, span))
            elif self.num_blocks * stride == storage.nbytes():
                # Non-contiguous packed layer views cannot name a sub-span.
                # Match upstream's opaque-row branch and deduplicate the row
                # across only the KV groups sharing this allocation.
                regions.append(
                    TransferRegion(storage.data_ptr(), stride, stride, row_offset=0)
                )
            else:
                raise ValueError(
                    "KV cache blocks must be dense or share a block-major packed row"
                )
        return regions

    def register(self, kv_caches) -> None:
        self._check_open()
        if self._registered:
            raise RuntimeError("KV transfer memory is already registered")
        self.kv_caches = kv_caches
        self.kv_caches_base_addr, self.block_lens, self.regions = [], [], []
        self._head_resharding_errors = []
        self.mamba_groups = set()
        is_npu = is_npu_platform()
        pointers, lengths = npu_registration_regions(kv_caches) if is_npu else ([], [])
        seen_storage = set()
        views = {}
        for layer_name, raw in kv_caches.items():
            spec = self.layer_specs.get(layer_name)
            if spec is None:
                continue
            if isinstance(spec, MambaSpec):
                if not isinstance(raw, (list, tuple)) or not raw:
                    raise TypeError(
                        f"Mamba cache {layer_name!r} must be a non-empty sequence"
                    )
                cache_list = [raw[0]]
            elif is_npu:
                cache_list = cache_tensors(raw)
            else:
                if not isinstance(raw, torch.Tensor):
                    raise TypeError(f"Attention cache {layer_name!r} must be a tensor")
                cache_list = self.topology.get_transfer_cache_regions(raw, spec)
            if isinstance(cache_list, torch.Tensor):
                # vLLM v0.23 returns the K/V-first tensor itself when the
                # caller must iterate its leading K/V dimension.
                cache_list = (
                    list(cache_list)
                    if getattr(self.topology, "split_k_and_v", False)
                    else [cache_list]
                )
            group = self.layer_groups.get(layer_name, 0)
            if isinstance(spec, MambaSpec):
                self.mamba_groups.add(group)
            page = getattr(spec, "page_size_bytes", 0)
            if page > getattr(spec, "unpadded_page_size_bytes", page):
                self._head_resharding_errors.append(
                    "KV-head re-sharding is unsupported for padded KV pages"
                )
            if getattr(getattr(spec, "kv_quant_mode", None), "is_nvfp4", False):
                self._head_resharding_errors.append(
                    "KV-head re-sharding is unsupported for NVFP4 KV cache"
                )
            indices = re.findall(r"(?:^|\.)(\d+)(?:\.|$)", layer_name)
            layer_index = int(indices[0]) if indices else -1
            for cache in cache_list:
                if not self.regions:
                    self.num_blocks = int(cache.shape[0])
                assert cache.shape[0] == self.num_blocks, (
                    "All kv cache tensors must have the same number of blocks"
                )
                storage = cache.untyped_storage()
                storage_base = storage.data_ptr()
                if not is_npu and storage_base not in seen_storage:
                    seen_storage.add(storage_base)
                    pointers.append(storage_base)
                    lengths.append(storage.nbytes())
                for region in self._cache_regions(cache, spec):
                    identity = (region.base_addr, region.block_len, region.kv_block_len)
                    previous = views.get(identity)
                    if previous is not None:
                        # Packed HMA rows can be shared by several groups.
                        # Flatten only those groups for this one region.
                        existing = self.regions[previous]
                        groups = tuple(sorted(set(existing.shared_group_ids) | {group}))
                        self.regions[previous] = replace(
                            existing,
                            group_index=-1 if len(groups) > 1 else group,
                            shared_group_ids=groups,
                        )
                        continue
                    views[identity] = len(self.regions)
                    self.regions.append(
                        replace(
                            region,
                            layer_name=layer_name,
                            layer_index=layer_index,
                            group_index=group,
                            shared_group_ids=(group,),
                        )
                    )
        if not pointers or not self.regions or self.num_blocks <= 0:
            raise RuntimeError("No PD KV regions registered")
        self.kv_caches_base_addr = [r.base_addr for r in self.regions]
        self.block_lens = [r.block_len for r in self.regions]
        if is_npu:
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
                self._listener_future.result()
        self._registered = True

    def _head_count(self) -> int | None:
        if getattr(self.topology, "is_mla", False) or getattr(self, "has_mamba", False):
            return None
        return self.topology.total_num_kv_heads

    @staticmethod
    def _layout_key(request):
        return (
            request.remote_hostname,
            request.remote_port,
            request.remote_tp_size,
            request.remote_tp_rank,
            request.remote_pp_size,
            request.layout_id,
        )

    def _prepare_transfer_regions(self, request):
        """Align upstream layout metadata before planning any native WRITE."""
        if request.remote_pp_size != 1:
            raise ValueError("MTSC transfer requires PP=1 on both peers")
        cache = self._remote_layouts
        key = self._layout_key(request)
        if request.layout_id and not request.kv_caches_base_addr:
            remote = cache.get(key)
            if remote is None:
                return None
            cache.move_to_end(key)
            return remote
        if request.kv_block_lens:
            count = len(request.kv_caches_base_addr)
            names = request.registered_layer_names
            indices = request.registered_layer_indices
            # The upstream fields default to empty, but a named local layout
            # cannot safely match an unnamed one by position.
            if not names and all(not region.layer_name for region in self.regions):
                names, indices = [""] * count, [-1] * count
            groups = request.registered_group_indices or [
                self.layer_groups.get(name, 0) for name in names
            ]
            remote = _expand_transfer_regions(
                request.kv_caches_base_addr,
                request.block_lens,
                request.kv_block_lens,
                names,
                indices,
                groups,
                [tuple(ids) for ids in request.registered_shared_group_ids] or None,
                request.registered_row_offsets or None,
            )
            if any(
                r.base_addr <= 0
                or r.block_len <= 0
                or not 0 < r.kv_block_len <= r.block_len
                for r in remote
            ):
                raise ValueError(
                    "Invalid peer KV region address, stride or payload length"
                )
            _, remote, error = _align_transfer_regions(self.regions, remote)
            if error:
                raise ValueError(error)
        else:
            # v0.23 compatibility: its wire layout has implicit K/V halves and
            # positional regions. Never infer a modern layout this way.
            remote = _legacy_transfer_regions(
                request.kv_caches_base_addr,
                request.block_lens,
                self.topology.virtually_split_kv_in_blocks,
            )
            if len(remote) == len(self.regions):
                remote = [
                    replace(
                        remote_region,
                        layer_name=local_region.layer_name,
                        layer_index=local_region.layer_index,
                        group_index=local_region.group_index,
                        shared_group_ids=local_region.shared_group_ids,
                    )
                    for local_region, remote_region in zip(
                        self.regions, remote, strict=True
                    )
                ]
        heads = self._head_count()
        if heads is not None and min(self.tp_size, heads) != min(
            request.remote_tp_size, heads
        ):
            errors = getattr(self, "_head_resharding_errors", [])
            if errors:
                raise ValueError(errors[0])
        error = _validate_asymmetric_region_lengths(
            self.regions,
            remote,
            self.tp_size,
            request.remote_tp_size,
            self.topology.local_replicates_kv_cache,
            heads,
        )
        if error:
            raise ValueError(error)
        if heads is None and any(
            local_region.kv_block_len != remote_region.kv_block_len
            for local_region, remote_region in zip(self.regions, remote, strict=True)
        ):
            raise ValueError("Replicated P/D KV page sizes do not match")
        if request.layout_id:
            cache[key] = remote
            cache.move_to_end(key)
            while len(cache) > 1024:
                cache.popitem(last=False)
        return remote

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
                request.remote_hostname,
                request.remote_port,
                request.remote_tp_rank,
            )
            with self._source_lock:
                for transfer_id, _ in request.req_blocks.values():
                    source = self._sources.get(transfer_id)
                    if source is not None and target in source.targets:
                        existing_targets.append(source.targets[target])
            if self._closed:
                raise RuntimeError("P transfer is closing")
            ranks = self.topology.handshake_target_ranks(request.remote_tp_size)
            if request.remote_tp_rank not in ranks:
                raise ValueError("D TP rank is not paired with this P rank")
            remote_regions = self._prepare_transfer_regions(request)
            if remote_regions is None:
                await socket.send_multipart(
                    (
                        identity,
                        self._encoder.encode(
                            KVTransferResponse(
                                KVTransferStatus.LAYOUT_MISS,
                                layout_id=request.layout_id,
                            )
                        ),
                    )
                )
                return
            for request_id, (transfer_id, blocks) in request.req_blocks.items():
                descriptor = (
                    request_id,
                    tuple(tuple(ids) for ids in blocks),
                    tuple(request.kv_caches_base_addr),
                    tuple(request.block_lens),
                    tuple(remote_regions),
                )
                with self._source_lock:
                    if transfer_id in self._retired:
                        source = None
                    else:
                        source = self._sources.setdefault(
                            transfer_id, _Session("", transfer_id)
                        )
                        if (
                            source.peer is not None
                            and source.peer != request.remote_tp_size
                        ):
                            raise ValueError("Transfer attempt changed D TP size")
                        source.peer = request.remote_tp_size
                        source.expected = len(ranks)
                        previous = source.targets.get(target)
                        if (
                            previous is not None
                            and source.descriptors[target] != descriptor
                        ):
                            raise ValueError(
                                "Duplicate target changed destination memory"
                            )
                        owner = previous is None
                        if owner:
                            previous = asyncio.get_running_loop().create_future()
                            source.targets[target] = previous
                            source.descriptors[target] = descriptor
                if source is None:
                    task = asyncio.get_running_loop().create_future()
                    task.set_result(False)
                elif owner:
                    owners[request_id] = (source, previous)
                    task = asyncio.create_task(self._wait_source_ready(source))
                else:
                    task = asyncio.shield(previous)
                waiting[task] = request_id

            while waiting:
                done, _ = await asyncio.wait(
                    waiting,
                    timeout=self.timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                timed_out = not done
                if timed_out:
                    done = set(waiting)
                    for task, request_id in waiting.items():
                        if request_id in owners:
                            task.cancel()
                    await asyncio.gather(*done, return_exceptions=True)
                results, ready_sources = {}, {}
                for task in done:
                    request_id = waiting.pop(task)
                    if request_id in owners:
                        source, previous = owners[request_id]
                        if not timed_out and task.result():
                            ready_sources[request_id] = source
                        else:
                            self._finish_target(source, False, False)
                            results[request_id] = False
                    else:
                        results[request_id] = task.result()
                if ready_sources:
                    results.update(
                        await self._write_ready_targets(
                            ready_sources, request, remote_regions
                        )
                    )
                for request_id, result in results.items():
                    owned = owners.pop(request_id, None)
                    if owned is not None:
                        owned[1].set_result(result)
                failed = [rid for rid, ok in results.items() if not ok]
                response = KVTransferResponse(
                    KVTransferStatus.CONTINUE if waiting else KVTransferStatus.FINISH,
                    [rid for rid, ok in results.items() if ok] or None,
                    failed or None,
                    "transfer failed" if failed else None,
                    layout_id=request.layout_id,
                )
                await socket.send_multipart((identity, self._encoder.encode(response)))
            if not request.req_blocks:
                await socket.send_multipart(
                    (
                        identity,
                        self._encoder.encode(
                            KVTransferResponse(
                                KVTransferStatus.FINISH, layout_id=request.layout_id
                            )
                        ),
                    )
                )
        except Exception as exc:  # noqa: BLE001 - control response boundary
            logger.warning("MTSC P transfer failed: %s", exc)
            # An error reply must not release memory while an earlier duplicate
            # is still writing to it.
            if existing_targets:
                await asyncio.gather(
                    *(asyncio.shield(future) for future in existing_targets),
                    return_exceptions=True,
                )
            await socket.send_multipart(
                (
                    identity,
                    self._encoder.encode(
                        KVTransferResponse(KVTransferStatus.ERROR, err_msg=str(exc))
                    ),
                )
            )
        finally:
            for task in waiting:
                task.cancel()
            if waiting:
                await asyncio.gather(*waiting, return_exceptions=True)
            for source, previous in owners.values():
                self._finish_target(source, False, False)
                if not previous.done():
                    previous.set_result(False)

    async def _write_ready_targets(self, sources, request, remote_regions=None):
        results = dict.fromkeys(sources, False)
        with self._source_lock:
            active = {
                rid: source
                for rid, source in sources.items()
                if not source.abort and source.transfer_id not in self._retired
            }
            for source in active.values():
                source.active_writes += 1
        try:
            if active:
                results.update(await self._write_batch(active, request, remote_regions))
        except Exception as exc:  # noqa: BLE001 - native WRITE result boundary
            logger.warning("MTSC Transfer WRITE failed: %s", exc)
        finally:
            for rid, source in sources.items():
                self._finish_target(source, rid in active, results[rid])
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
        self._check_receive_error()
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
                rid: future
                for rid, future in self._receive_futures.items()
                if future.done()
            }
        for future in terminal.values():
            future.result()
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
        self._check_receive_error()
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
        self._check_receive_error()
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
        self._remote_layouts.clear()
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

    async def _write_batch(self, sources, request, remote_regions=None):
        src, dst, sizes = [], [], []
        results = dict.fromkeys(sources, False)
        valid = []
        if remote_regions is None:
            remote_regions = self._prepare_transfer_regions(request)
        for request_id, source in sources.items():
            try:
                local_src, local_dst, local_sizes = self._build_request_transfer_params(
                    request_id,
                    source,
                    request,
                    remote_regions,
                )
            except Exception as exc:  # noqa: BLE001 - per-request validation
                logger.warning(
                    "MTSC transfer plan failed: request_id=%s error=%s", request_id, exc
                )
                continue
            src.extend(local_src)
            dst.extend(local_dst)
            sizes.extend(local_sizes)
            valid.append(request_id)
        ret = 0
        if src:
            ret = await self._write_buffers(
                f"{request.remote_hostname}:{request.remote_port}",
                src,
                dst,
                sizes,
            )
        for request_id in valid:
            results[request_id] = ret == 0
        return results

    def _build_request_transfer_params(
        self, request_id, source, request, remote_regions=None
    ):
        if remote_regions is None:
            remote_regions = self._prepare_transfer_regions(request)
        if remote_regions is None:
            raise ValueError("Peer layout is missing")
        _, requested_groups = request.req_blocks[request_id]
        destination_groups = [list(group) for group in requested_groups]
        if len(source.block_ids) != len(destination_groups):
            raise ValueError("P/D KV group count mismatch")
        source_groups = []
        for index, (local, remote) in enumerate(
            zip(source.block_ids, destination_groups, strict=True)
        ):
            if index in getattr(self, "mamba_groups", set()):
                local = [block for block in local if block != 0]
                remote = [block for block in remote if block != 0]
                destination_groups[index] = remote
            if len(local) < len(remote):
                raise ValueError("P has fewer source blocks than D requested")
            source_groups.append(list(local[-len(remote) :]) if remote else [])
        src, dst, sizes = [], [], []
        for local, remote in zip(self.regions, remote_regions, strict=True):
            local_ids = _block_ids_for_region(
                source_groups, local.group_index, local.shared_group_ids
            )
            remote_ids = _block_ids_for_region(
                destination_groups, remote.group_index, local.shared_group_ids
            )
            blocks = [
                (s, d) for s, d in zip(local_ids, remote_ids, strict=True) if d >= 0
            ]
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
                self.topology.local_replicates_kv_cache,
                self._head_count(),
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
        return await self._loop.run_in_executor(
            self._send_pool,
            self.engine.batch_transfer_sync_write,
            session,
            src,
            dst,
            sizes,
        )

    async def _query_workers(
        self, address: str, engine_id: str, dp_rank: int
    ) -> dict[int, dict[int, str]]:
        address = address.rstrip("/")
        key = (address, engine_id, dp_rank)
        if key in self._remote_workers:
            return self._remote_workers[key]
        query = self._pending_bootstrap_queries.get(address)
        if query is None:
            query = asyncio.get_running_loop().create_future()
            self._pending_bootstrap_queries[address] = query
            try:
                async with httpx.AsyncClient(trust_env=False) as client:
                    response = await client.get(address + "/query")
                    response.raise_for_status()
                    entries = response.json()
                query.set_result(entries)
            except (Exception, asyncio.CancelledError) as exc:
                query.set_exception(exc)
                # The owner observes this error even if no follower remains.
                query.exception()
                raise
            finally:
                self._pending_bootstrap_queries.pop(address, None)
        else:
            # Cancelling one follower must not cancel the shared query.
            entries = await asyncio.shield(query)
        if key in self._remote_workers:
            return self._remote_workers[key]
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
        if not workers or sorted(workers) != list(range(len(workers))):
            raise ValueError("Remote P TP workers are not fully registered")
        if any(sorted(pp_map) != [0] for pp_map in workers.values()):
            raise ValueError("Mooncake transfer requires PP=1")
        self._remote_workers[key] = workers
        return workers

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
        by_id = {event.request_id: event for event in events}
        if not events:
            return
        first = events[0]
        try:
            workers = await self._query_workers(
                first.bootstrap_addr, first.remote_engine_id, first.remote_dp_rank
            )
            remote_tp_size = len(workers)
            target_tp = self.topology.handshake_target_ranks(remote_tp_size)
            addresses = [workers[tp_rank][0] for tp_rank in target_tp]
            if not addresses:
                raise RuntimeError("No matching P workers in bootstrap response")
            request = KVTransferRequest(
                remote_hostname=self.hostname,
                remote_port=self.rpc_port,
                remote_tp_size=self.tp_size,
                remote_tp_rank=self.tp_rank,
                req_blocks={
                    event.request_id: (
                        event.transfer_id,
                        [list(ids) for ids in event.block_ids],
                    )
                    for event in events
                },
                kv_caches_base_addr=[r.base_addr for r in self.regions],
                block_lens=[r.block_len for r in self.regions],
                kv_block_lens=[r.kv_block_len for r in self.regions],
                registered_layer_names=[r.layer_name for r in self.regions],
                registered_layer_indices=[r.layer_index for r in self.regions],
                registered_group_indices=[r.group_index for r in self.regions],
                registered_shared_group_ids=[
                    list(r.shared_group_ids) for r in self.regions
                ],
                registered_row_offsets=[r.row_offset for r in self.regions],
                remote_pp_size=self.pp_size,
            )
            payload = self._encoder.encode(request)
            pending = dict.fromkeys(by_id, len(addresses))
            failed_requests = set()

            def record(request_id, response):
                if (
                    request_id not in (response.ok_reqs or [])
                    or request_id in (response.err_reqs or [])
                    or response.status == KVTransferStatus.ERROR
                ):
                    failed_requests.add(request_id)
                pending[request_id] -= 1
                if pending[request_id]:
                    return
                event = by_id[request_id]
                failed = request_id in failed_requests
                self._finish_receive(event, failed, futures.get(event.request_id))
                finished.add(request_id)

            async def call(address: str) -> None:
                socket = self._ctx.socket(zmq.DEALER)
                remaining = set(by_id)
                published = False
                try:
                    socket.setsockopt(zmq.LINGER, 0)
                    # Match upstream: let P's ready wait time out first.
                    socket.setsockopt(zmq.RCVTIMEO, int((self.timeout + 60) * 1000))
                    socket.connect(address)
                    published = True
                    await socket.send(payload)
                    # Each response is terminal for the IDs it reports, while
                    # CONTINUE keeps this socket open for the other IDs.
                    while True:
                        response = self._response_decoder.decode(await socket.recv())
                        if response.status == KVTransferStatus.ERROR:
                            for request_id in remaining:
                                record(request_id, response)
                            remaining.clear()
                            break
                        if response.status not in (
                            KVTransferStatus.CONTINUE,
                            KVTransferStatus.FINISH,
                        ):
                            raise ValueError("Invalid P response status")
                        completed = set(response.ok_reqs or [])
                        failed = set(response.err_reqs or [])
                        reported = completed | failed
                        if completed & failed or not reported <= remaining:
                            raise ValueError(
                                "P response contains conflicting or repeated request IDs"
                            )
                        for tid in reported:
                            record(tid, response)
                        remaining.difference_update(reported)
                        if response.status == KVTransferStatus.FINISH:
                            # FINISH fences the whole message, including IDs
                            # the peer omitted from its success list.
                            for request_id in remaining:
                                record(request_id, response)
                            remaining.clear()
                            break
                except (Exception, asyncio.CancelledError) as exc:
                    logger.warning(
                        "MTSC D peer pull failed: address=%s error=%s", address, exc
                    )
                    if published and remaining:
                        # Timeout/disconnect is not a remote WRITE fence. Fail
                        # the worker instead of returning blocks for recompute.
                        with self._result_lock:
                            if self._receive_error is None:
                                self._receive_error = _ReceiveFenceError(
                                    f"Lost P completion fence at {address}: {exc!r}; "
                                    "worker must stop before KV blocks can be reused"
                                )
                            error = self._receive_error
                        # Wake a main-thread preemption/close waiting on any
                        # batch, even when a different peer lost its fence.
                        with self._receive_lock:
                            pending_futures = list(self._receive_futures.values())
                        for future in pending_futures:
                            if not future.done():
                                future.set_exception(error)
                        for request_id in remaining:
                            finished.add(request_id)
                            future = futures.get(request_id)
                            if future is not None and not future.done():
                                future.set_exception(error)
                    else:
                        response = KVTransferResponse(
                            KVTransferStatus.ERROR, err_msg=str(exc)
                        )
                        for request_id in remaining:
                            record(request_id, response)
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
                "MTSC D pull batch failed: request_ids=%s error=%s", list(by_id), exc
            )
        finally:
            for request_id, event in by_id.items():
                if request_id not in finished:
                    self._finish_receive(event, True, futures.get(event.request_id))

    def _fence_receives(self, request_ids: set[str]) -> None:
        """Fence in-flight P writes before vLLM can recycle D blocks."""
        self._check_receive_error()
        futures: list[tuple[str, Future[None]]] = []
        with self._receive_lock:
            for request_id in request_ids:
                future = self._receive_futures.get(request_id)
                if future is not None:
                    futures.append((request_id, future))
        for request_id, future in futures:
            try:
                future.result()
            except _ReceiveFenceError:
                raise
            except Exception as exc:  # noqa: BLE001 - transport fence
                logger.warning(
                    "MTSC PD receive fence failed: request_id=%s error=%s",
                    request_id,
                    exc,
                )
        self._check_receive_error()
        with self._receive_lock:
            for request_id, _ in futures:
                self._receive_futures.pop(request_id, None)
        with self._result_lock:
            self._failed_recv.difference_update(request_ids)
