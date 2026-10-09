"""Unified MTSC worker and Decode two-stage state machine."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv

from .kv_cache_pool import KVCachePool, LoadEvent, SaveEvent
from .kv_transfer import KVTransfer, RecvEvent, SendEvent
from .protocol import (
    FinishedWait,
    KVPoolLoadRequest,
    KVPoolSaveRequest,
    KVTransferPlan,
    MTSCConnectorMetadata,
)
from .utils import new_device_event

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)
logger.setLevel(logging.INFO)


def _backend(config, key: str) -> str:
    extra = config.kv_transfer_config.kv_connector_extra_config
    backend = extra.get(key, "mooncake")
    if backend != "mooncake":
        raise ValueError(f"Unsupported {key}: {backend!r}")
    return backend


def _create_pool(config, kv_cache_config) -> KVCachePool:
    _backend(config, "mtsc_pool_backend")
    from .kv_cache_pool import MooncakeKVCachePool

    return MooncakeKVCachePool(config, kv_cache_config)


def _create_transfer(config, kv_cache_config) -> KVTransfer:
    _backend(config, "mtsc_transfer_backend")
    from .kv_transfer import MooncakeKVTransfer

    return MooncakeKVTransfer(config, kv_cache_config)


@dataclass
class _TransferState:
    """Tracks STORE_PENDING -> STORE_DONE -> TRANSFER_PENDING -> DONE."""

    plan: KVTransferPlan
    stage: str
    suffix_block_ids: list[list[int]] = field(default_factory=list)


@dataclass
class _FinishedWaitState:
    """Aggregates required operations before source blocks can be released."""

    requirements: FinishedWait
    pool_save_done: bool = False
    kv_transfer_done: bool = False

    @property
    def complete(self) -> bool:
        return (not self.requirements.store_save or self.pool_save_done) and (
            not self.requirements.kv_transfer or self.kv_transfer_done
        )


class MTSCWorker:
    """Executes batch operations and advances pool + transfer loading."""

    def __init__(self, config: VllmConfig, kv_cache_config: KVCacheConfig) -> None:
        self.pool: KVCachePool = _create_pool(config, kv_cache_config)
        try:
            self.transfer: KVTransfer = _create_transfer(config, kv_cache_config)
        except Exception:
            self.pool.close()
            raise
        # Every submitted pool operation is retained until its terminal result.
        self._on_load_requests: dict[str, KVPoolLoadRequest] = {}
        self._on_save_requests: dict[str, KVPoolSaveRequest] = {}
        # Transfers include the initial pool-load stage.
        self._on_transfer: dict[str, _TransferState] = {}
        self._finished_waits: dict[str, _FinishedWaitState] = {}
        self._ignored_recvs: set[str] = set()
        self._load_errors: set[int] = set()
        self._closed = False

    def register_kv_caches(
        self,
        kv_caches: dict[
            str, torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]
        ],
    ) -> None:
        self.pool.register(kv_caches)
        try:
            self.transfer.register(kv_caches)
        except Exception:
            try:
                self.transfer.close()
            finally:
                self.pool.close()
            raise
        logger.info("MTSC registered pool and transfer data planes")

    def handle_preemptions(self, metadata: MTSCConnectorMetadata) -> None:
        request_ids = metadata.preempted_request_ids
        if not request_ids:
            return
        # vLLM invokes this hook before the current model step can overwrite
        # recycled blocks. A PUT reads GPU memory asynchronously, so it must be
        # fenced here rather than in wait_for_save().
        for request_id in request_ids:
            self.pool.preempt(request_id)
            self.transfer.preempt(request_id)
            self._on_save_requests.pop(request_id, None)
            self._on_transfer.pop(request_id, None)
            self._on_load_requests.pop(request_id, None)
            self._finished_waits.pop(request_id, None)
            self._ignored_recvs.discard(request_id)
        # Error sets are consumed in the same cycle as their terminal result;
        # preempt() clears any backend-owned errors before block reuse.

    def _apply_metadata(self, metadata: MTSCConnectorMetadata) -> None:
        for load_meta in metadata.pool_loads:
            self.pool.load(
                LoadEvent(
                    load_meta.request_id,
                    tuple(tuple(group) for group in load_meta.block_ids),
                    tuple(bytes(value) for value in load_meta.block_hashes),
                    load_meta.start_token,
                    load_meta.end_token,
                )
            )
            # All loads, including the first transfer stage, share this tracker.
            self._on_load_requests[load_meta.request_id] = load_meta
        save_event = None
        if metadata.pool_saves:
            save_event = new_device_event()
            save_event.record()
        for save_meta in metadata.pool_saves:
            self.pool.save(
                SaveEvent(
                    save_meta.request_id,
                    tuple(tuple(group) for group in save_meta.block_ids),
                    tuple(bytes(value) for value in save_meta.block_hashes),
                    save_meta.start_token,
                    save_meta.end_token,
                    save_event,
                    save_meta.prompt_tokens,
                )
            )
            self._on_save_requests[save_meta.request_id] = save_meta
            state = self._finished_waits.get(save_meta.request_id)
            if state is not None:
                state.pool_save_done = False
        ready_updates = [
            source_state
            for source_state in metadata.transfer_states
            if source_state.ready
        ]
        if ready_updates:
            # send() publishes readable source memory. This fences preceding
            # device writes before the transport can access those addresses.
            ready = new_device_event()
            ready.record()
            ready.synchronize()
        for source_state in metadata.transfer_states:
            if source_state.cancelled:
                self.transfer.cancel(source_state.request_id, source_state.transfer_id)
            elif source_state.ready:
                self.transfer.send(
                    SendEvent(
                        source_state.request_id,
                        source_state.transfer_id,
                        tuple(tuple(group) for group in source_state.source_block_ids),
                    )
                )
            else:
                self.transfer.prepare(source_state.request_id, source_state.transfer_id)
        for request_id, requirement in metadata.finished_waits.items():
            self._finished_waits.setdefault(
                request_id,
                _FinishedWaitState(
                    requirement,
                    pool_save_done=request_id not in self._on_save_requests,
                ),
            )
        for plan in metadata.transfer_plans:
            if plan.request_id in self._on_transfer:
                continue
            stage = (
                "STORE_PENDING"
                if plan.pool_tokens > plan.local_tokens
                else "STORE_DONE"
            )
            self._on_transfer[plan.request_id] = _TransferState(plan, stage)
            logger.info(
                "MTSC transfer state created: request_id=%s stage=%s L=%d H=%d T=%d",
                plan.request_id,
                stage,
                plan.local_tokens,
                plan.pool_tokens,
                plan.target_tokens,
            )

    @staticmethod
    def _pool_candidate_block_ids(plan: KVTransferPlan) -> set[int]:
        result: set[int] = set()
        for group_ids, block_size in zip(
            plan.all_block_ids, plan.group_block_sizes, strict=True
        ):
            start = cdiv(plan.local_tokens, block_size)
            end = min(cdiv(plan.pool_tokens, block_size), len(group_ids))
            result.update(group_ids[start:end])
        return result

    @staticmethod
    def _suffix_blocks(
        plan: KVTransferPlan, actual_pool_prefix: int
    ) -> list[list[int]]:
        offset = actual_pool_prefix - plan.local_tokens
        result: list[list[int]] = []
        for group, block_size, window in zip(
            plan.external_block_ids,
            plan.group_block_sizes,
            plan.blocks_per_sliding_window,
            strict=True,
        ):
            suffix = group[cdiv(offset, block_size) :]
            if window:
                suffix = suffix[-window:]
            result.append(suffix)
        return result

    def _start_transfer(
        self,
        state: _TransferState,
        actual_pool_prefix: int,
        receives: list[RecvEvent] | None = None,
    ) -> bool:
        plan = state.plan
        suffix = self._suffix_blocks(plan, actual_pool_prefix)
        state.suffix_block_ids = suffix
        if not plan.transfer_enabled:
            if actual_pool_prefix < plan.target_tokens:
                for group in suffix:
                    self._load_errors.update(block for block in group if block >= 0)
                logger.warning(
                    "MTSC pool miss has no transfer source: request_id=%s",
                    plan.request_id,
                )
            state.stage = "DONE"
            return False
        params = plan.transfer_params
        event = RecvEvent(
            plan.request_id,
            plan.transfer_id,
            tuple(tuple(group) for group in suffix),
            str(params["remote_bootstrap_addr"]),
            str(params["remote_engine_id"]),
            int(params.get("remote_dp_rank", 0)),
        )
        if receives is None:
            self.transfer.recv(event)
        else:
            receives.append(event)
        state.stage = "TRANSFER_PENDING"
        if not plan.wait_for_completion:
            self._ignored_recvs.add(plan.request_id)
        logger.info(
            "MTSC transfer suffix started: request_id=%s A=%d T=%d blocks=%d",
            plan.request_id,
            actual_pool_prefix,
            plan.target_tokens,
            sum(len(group) for group in suffix),
        )
        return True

    def _aggregate_sends(
        self, pool_save_done: set[str], kv_transfer_done: set[str]
    ) -> set[str]:
        for request_id in pool_save_done:
            self._on_save_requests.pop(request_id, None)
            state = self._finished_waits.get(request_id)
            if state is not None:
                state.pool_save_done = True
        for request_id in kv_transfer_done:
            state = self._finished_waits.get(request_id)
            if state is not None:
                state.kv_transfer_done = True
        completed = set()
        # Only request_finished creates a release dependency. Intermediate
        # backend results never release blocks of an active request.
        for request_id, state in list(self._finished_waits.items()):
            if state.complete:
                completed.add(request_id)
                del self._finished_waits[request_id]
        return completed

    def get_finished(
        self, finished_req_ids: set[str], metadata: MTSCConnectorMetadata
    ) -> tuple[set[str] | None, set[str] | None]:
        self._apply_metadata(metadata)
        pool_results = self.pool.poll()
        pool_errors = self.pool.take_errors()
        staged_ids = set(self._on_transfer)
        staged_blocks = {
            block
            for state in self._on_transfer.values()
            for block in self._pool_candidate_block_ids(state.plan)
        }
        self._load_errors.update(pool_errors - staged_blocks)
        loads = {result.request_id: result for result in pool_results.loads}
        finished_recv = set()
        for request_id in loads:
            self._on_load_requests.pop(request_id, None)
            if request_id not in staged_ids:
                finished_recv.add(request_id)

        receives: list[RecvEvent] = []
        for request_id, state in list(self._on_transfer.items()):
            if state.stage == "STORE_DONE":
                actual = state.plan.local_tokens
            elif state.stage == "STORE_PENDING" and request_id in loads:
                actual = loads[request_id].loaded_tokens
                if not state.plan.local_tokens <= actual <= state.plan.pool_tokens:
                    raise ValueError("Pool returned an invalid loaded prefix")
                logger.info(
                    "MTSC pool load terminal: request_id=%s L=%d H=%d A=%d",
                    request_id,
                    state.plan.local_tokens,
                    state.plan.pool_tokens,
                    actual,
                )
            else:
                continue
            if not self._start_transfer(state, actual, receives):
                if state.plan.wait_for_completion:
                    finished_recv.add(request_id)
                del self._on_transfer[request_id]

        if receives:
            self.transfer.recv_batch(receives)

        transfer_results = self.transfer.poll()
        transfer_errors = self.transfer.take_errors()
        self._load_errors.update(transfer_errors)
        for result in transfer_results.recvs:
            request_id = result.request_id
            if request_id in self._ignored_recvs:
                self._ignored_recvs.discard(request_id)
                self._on_transfer.pop(request_id, None)
                continue
            state = self._on_transfer.pop(request_id, None)
            if state is not None:
                if result.error is not None:
                    # Preserve fallback even if the backend could not prove
                    # which subset of its requested regions is usable.
                    self._load_errors.update(
                        block
                        for group in state.suffix_block_ids
                        for block in group
                        if block >= 0
                    )
                if state.plan.wait_for_completion:
                    finished_recv.add(request_id)

        finished_send = self._aggregate_sends(
            {result.request_id for result in pool_results.saves},
            {result.request_id for result in transfer_results.sends},
        )
        return finished_send or None, finished_recv or None

    def get_block_ids_with_load_errors(self) -> set[int]:
        errors, self._load_errors = self._load_errors, set()
        return errors

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.transfer.close()
        finally:
            self.pool.close()
