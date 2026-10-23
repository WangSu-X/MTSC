# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MTSC-owned scheduler/worker metadata and transfer wire protocols."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import msgspec
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)
from vllm.v1.core.kv_cache_utils import BlockHash


@dataclass(frozen=True)
class KVPoolLoadSpec:
    """Pool lookup decision awaiting local block allocation."""

    computed_tokens: int  # Already computed tokens (local)
    pool_hit_tokens: int  # Tokens available in pool


@dataclass
class KVPoolLoadRequest:
    """Load [start_token, end_token) from the pool into allocated blocks."""

    request_id: str
    block_ids: tuple[list[int], ...]
    block_hashes: list[BlockHash]
    start_token: int
    end_token: int
    load_id: int = 0


@dataclass
class KVPoolSaveRequest:
    """Save complete cache chunks in [start_token, end_token) to the pool."""

    request_id: str
    block_ids: tuple[list[int], ...]
    block_hashes: list[BlockHash]
    start_token: int
    end_token: int
    prompt_tokens: int | None = None


@dataclass(frozen=True)
class KVTransferPlan:
    """Immutable plan for pool loading followed by direct transfer."""

    request_id: str
    transfer_id: str
    computed_tokens: int  # Already computed tokens (local)
    pool_hit_tokens: int  # Tokens available in pool
    target_tokens: int  # Target total tokens to load
    all_block_ids: tuple[list[int], ...]
    external_block_ids: tuple[list[int], ...]
    group_block_sizes: tuple[int, ...]
    blocks_per_sliding_window: tuple[int, ...]
    transfer_params: dict[str, Any]
    load_id: int = 0


@dataclass(frozen=True)
class MTSCLoadReport:
    """One worker's terminal load record. Token counts are logical, not TP sums.

    pd_tokens describes the requested suffix; only status=ok confirms that
    the suffix is usable. Store tokens describe the actual usable prefix.
    """

    request_id: str
    load_id: int
    worker_rank: int
    local_tokens: int
    store_tokens: int
    pd_tokens: int
    store_ms: float
    pd_ms: float
    total_ms: float
    store_status: str
    status: str


@dataclass
class MTSCWorkerMetadata(KVConnectorWorkerMetadata):
    """Completed load records, drained once per model-runner step."""

    load_reports: list[MTSCLoadReport] = field(default_factory=list)

    def aggregate(self, other: KVConnectorWorkerMetadata) -> MTSCWorkerMetadata:
        assert isinstance(other, MTSCWorkerMetadata)
        return MTSCWorkerMetadata(self.load_reports + other.load_reports)


@dataclass(frozen=True)
class KVTransferSourceState:
    """Producer state delta; empty blocks register a placeholder."""

    request_id: str
    transfer_id: str
    source_block_ids: tuple[list[int], ...] = ()
    ready: bool = False
    cancelled: bool = False


@dataclass(frozen=True)
class FinishedWait:
    """Operations that must finish before a completed request releases blocks."""

    store_save: bool = False
    kv_transfer: bool = False


@dataclass
class MTSCConnectorMetadata(KVConnectorMetadata):
    """Per-step execution plan sent from Scheduler to Worker."""

    pool_loads: list[KVPoolLoadRequest] = field(default_factory=list)
    pool_saves: list[KVPoolSaveRequest] = field(default_factory=list)
    transfer_plans: list[KVTransferPlan] = field(default_factory=list)
    transfer_states: list[KVTransferSourceState] = field(default_factory=list)
    finished_waits: dict[str, FinishedWait] = field(default_factory=dict)
    finished_request_ids: set[str] = field(default_factory=set)
    preempted_request_ids: set[str] = field(default_factory=set)


class KVTransferStatus(IntEnum):
    FINISH = 0
    CONTINUE = 1
    ERROR = 2


class KVTransferRequest(msgspec.Struct, omit_defaults=True):  # type: ignore[call-arg]
    """Official Mooncake D -> P metadata; req_blocks is keyed by D request ID."""

    remote_hostname: str
    remote_port: int
    remote_tp_size: int
    remote_tp_rank: int
    req_blocks: dict[str, tuple[str, list[list[int]]]]
    kv_caches_base_addr: list[int]
    block_lens: list[int]


class KVTransferResponse(msgspec.Struct, omit_defaults=True):  # type: ignore[call-arg]
    """Official Mooncake response, keyed by D request ID."""

    status: KVTransferStatus
    ok_reqs: list[str] | None = None
    err_reqs: list[str] | None = None
    err_msg: str | None = None
