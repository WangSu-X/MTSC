# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MTSC-owned scheduler/worker metadata and transfer wire protocols."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import msgspec
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
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
    LAYOUT_MISS = 3


class KVTransferRequest(msgspec.Struct, omit_defaults=True):  # type: ignore[call-arg]
    """Official Mooncake D -> P metadata; req_blocks is keyed by D request ID."""

    remote_hostname: str
    remote_port: int
    remote_tp_size: int
    remote_tp_rank: int
    req_blocks: dict[str, tuple[str, list[list[int]]]]
    kv_caches_base_addr: list[int]
    block_lens: list[int]
    # vLLM da9134bc layout metadata. The default permits receiving v0.23
    # requests; MTSC always sends explicit payload lengths and layer metadata.
    kv_block_lens: list[int] = msgspec.field(default_factory=list)
    registered_layer_names: list[str] = msgspec.field(default_factory=list)
    registered_layer_indices: list[int] = msgspec.field(default_factory=list)
    registered_group_indices: list[int] = msgspec.field(default_factory=list)
    registered_shared_group_ids: list[list[int]] = msgspec.field(default_factory=list)
    registered_row_offsets: list[int] = msgspec.field(default_factory=list)
    remote_pp_size: int = 1
    layout_id: int = 0


class KVTransferResponse(msgspec.Struct, omit_defaults=True):  # type: ignore[call-arg]
    """Official Mooncake response, keyed by D request ID."""

    status: KVTransferStatus
    ok_reqs: list[str] | None = None
    err_reqs: list[str] | None = None
    err_msg: str | None = None
    layout_id: int = 0
