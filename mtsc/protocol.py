"""MTSC-owned scheduler/worker and P/D wire protocols.

The types in this module deliberately do not depend on either vLLM Mooncake
connector.  MTSC owns their lifecycle and only the data plane talks to the
Mooncake Python libraries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import msgspec
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.v1.core.kv_cache_utils import BlockHash


@dataclass
class StoreLoadSpec:
    local_tokens: int
    store_tokens: int
    enabled: bool = False


@dataclass
class StoreRequest:
    request_id: str
    token_count: int
    block_ids: tuple[list[int], ...]
    block_hashes: list[BlockHash]
    load: StoreLoadSpec | None = None
    save: bool = False
    save_from: int = 0
    token_ids: list[int] | None = None
    prompt_tokens: int | None = None


@dataclass(frozen=True)
class DTwoStageLoadPlan:
    """One Decode request's immutable, once-allocated load plan."""

    request_id: str
    transfer_id: str
    local_prefix_tokens: int
    store_candidate_tokens: int
    target_prefix_tokens: int
    all_block_ids: tuple[list[int], ...]
    external_block_ids: tuple[list[int], ...]
    group_block_sizes: tuple[int, ...]
    blocks_per_sliding_window: tuple[int, ...]
    pd_enabled: bool
    wait_for_completion: bool
    kv_transfer_params: dict[str, Any]


@dataclass(frozen=True)
class PDSendUpdate:
    """P-side state delta. Empty blocks register a placeholder."""

    request_id: str
    transfer_id: str
    block_ids: tuple[list[int], ...] = ()
    source_ready: bool = False
    abort: bool = False


@dataclass(frozen=True)
class SendRequirement:
    store: bool = False
    pd: bool = False


@dataclass
class MTSCConnectorMetadata(KVConnectorMetadata):
    """A per-step delta wholly owned by MTSC."""

    store_requests: list[StoreRequest] = field(default_factory=list)
    decode_plans: list[DTwoStageLoadPlan] = field(default_factory=list)
    pd_send_updates: list[PDSendUpdate] = field(default_factory=list)
    send_requirements: dict[str, SendRequirement] = field(default_factory=dict)
    finished_request_ids: set[str] = field(default_factory=set)
    preempted_request_ids: set[str] = field(default_factory=set)


class PDResponseStatus(IntEnum):
    FINISH = 0
    CONTINUE = 1
    ERROR = 2


class PDTransferSchema(msgspec.Struct, frozen=True):
    """Layout invariants that must match before P writes into D memory."""

    topology_version: int
    model_id: str
    model_revision: str
    cache_dtype: str
    cache_layout: str
    block_size: int
    is_mla: bool


class PDTransferRequest(msgspec.Struct, omit_defaults=True):  # type: ignore[call-arg]
    """D -> P control message; P writes bytes into the listed D regions."""

    hostname: str
    rpc_port: int
    tp_size: int
    tp_rank: int
    pp_size: int
    pp_rank: int
    schema: PDTransferSchema
    requests: dict[str, tuple[str, list[list[int]]]]
    region_base_addresses: list[int]
    block_lengths: list[int]
    kv_block_lengths: list[int]
    layer_names: list[str] = msgspec.field(default_factory=list)
    layer_indices: list[int] = msgspec.field(default_factory=list)
    group_indices: list[int] = msgspec.field(default_factory=list)


class PDTransferResponse(msgspec.Struct, omit_defaults=True):  # type: ignore[call-arg]
    status: PDResponseStatus
    completed: list[str] | None = None
    failed: list[str] | None = None
    error: str | None = None
    covered_regions: dict[str, list[int]] | None = None
