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

    local_tokens: int
    pool_tokens: int


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
    local_tokens: int
    pool_tokens: int
    target_tokens: int
    all_block_ids: tuple[list[int], ...]
    external_block_ids: tuple[list[int], ...]
    group_block_sizes: tuple[int, ...]
    blocks_per_sliding_window: tuple[int, ...]
    transfer_enabled: bool
    wait_for_completion: bool
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
    # COMPLETE ends the control message; IN_PROGRESS reports terminal results
    # for a subset of its transfer IDs while the remaining IDs are pending.
    COMPLETE = 0
    IN_PROGRESS = 1
    FAILED = 2


class KVTransferSchema(msgspec.Struct, frozen=True):
    """Layout invariants that must match before P writes into D memory."""

    topology_version: int
    model_id: str
    model_revision: str
    cache_dtype: str
    cache_layout: str
    block_size: int
    is_mla: bool
    pcp_size: int = 1
    dcp_size: int = 1


class KVTransferRequest(msgspec.Struct, omit_defaults=True):  # type: ignore[call-arg]
    """D -> P control message; P writes bytes into the listed D regions."""

    hostname: str
    rpc_port: int
    tp_size: int
    tp_rank: int
    pp_size: int
    pp_rank: int
    schema: KVTransferSchema
    # {transfer_id: (consumer_request_id, destination_block_groups)}
    requests: dict[str, tuple[str, list[list[int]]]]
    region_base_addresses: list[int]
    block_lengths: list[int]
    kv_block_lengths: list[int]
    layer_names: list[str] = msgspec.field(default_factory=list)
    layer_indices: list[int] = msgspec.field(default_factory=list)
    group_indices: list[int] = msgspec.field(default_factory=list)
    # Explicit replica identities for the concrete KVTransfer backend.
    engine_id: str = ""
    dp_rank: int = 0
    remote_engine_id: str = ""
    remote_dp_rank: int = 0
    destination_num_blocks: int = 0


class KVTransferResponse(msgspec.Struct, omit_defaults=True):  # type: ignore[call-arg]
    """Per-round terminal results, including destination coverage for each ID."""

    status: KVTransferStatus
    completed_transfer_ids: list[str] | None = None
    failed_transfer_ids: list[str] | None = None
    error_message: str | None = None
    # Destination region indices covered by each transfer_id.
    covered_regions: dict[str, list[int]] | None = None
