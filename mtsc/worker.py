"""Unified MTSC worker and Decode two-stage state machine."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from vllm import envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv

from .kv_cache_pool import LoadEvent, LoadResult, MooncakeKVCachePool, SaveEvent
from .kv_transfer import MooncakeKVTransfer, RecvEvent, SendEvent, TransferResult
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
logger.setLevel(envs.VLLM_LOGGING_LEVEL)


def _backend(config, key: str) -> str:
    extra = config.kv_transfer_config.kv_connector_extra_config
    backend = extra.get(key, "mooncake")
    if backend != "mooncake":
        raise ValueError(f"Unsupported {key}: {backend!r}")
    return backend


def _create_pool(config, kv_cache_config) -> MooncakeKVCachePool:
    _backend(config, "mtsc_pool_backend")
    return MooncakeKVCachePool(config, kv_cache_config)


def _create_transfer(config, kv_cache_config) -> MooncakeKVTransfer:
    _backend(config, "mtsc_transfer_backend")
    return MooncakeKVTransfer(config, kv_cache_config)


@dataclass
class _TransferState:
    """Tracks POOL_LOADING -> POOL_DONE -> TRANSFER_LOADING -> DONE."""

    plan: KVTransferPlan
    stage: str
    suffix_block_ids: list[list[int]] = field(default_factory=list)
    load_started_at: float = field(default_factory=time.monotonic)
    store_ms: float = 0.0
    store_status: str = "skipped"
    pd_started_at: float = 0.0


@dataclass
class _FinishedWaitState:
    """Aggregates required operations before source blocks can be released."""

    wait_for: FinishedWait
    pool_save_done: bool = False
    kv_transfer_done: bool = False

    @property
    def is_finished(self) -> bool:
        return (not self.wait_for.store_save or self.pool_save_done) and (
            not self.wait_for.kv_transfer or self.kv_transfer_done
        )


class MTSCWorker:
    """Executes batch operations and advances pool + transfer loading."""

    def __init__(self, config: VllmConfig, kv_cache_config: KVCacheConfig) -> None:
        self.pool = _create_pool(config, kv_cache_config)
        try:
            self.transfer = _create_transfer(config, kv_cache_config)
        except Exception:
            self.pool.close()
            raise
        # Every submitted pool operation is retained until its terminal result.
        self._on_load_requests: dict[str, KVPoolLoadRequest] = {}
        self._on_save_requests: dict[str, KVPoolSaveRequest] = {}
        # Transfers include the initial pool-load stage.
        self._on_transfer: dict[str, _TransferState] = {}
        self._finished_waits: dict[str, _FinishedWaitState] = {}
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
        logger.debug("[MTSC] register: pool=ready transfer=ready")

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
        # Error sets are consumed in the same cycle as their terminal result;
        # preempt() clears any backend-owned errors before block reuse.

    @staticmethod
    def _pool_candidate_block_ids(plan: KVTransferPlan) -> set[int]:
        result: set[int] = set()
        for group_ids, block_size in zip(
            plan.all_block_ids, plan.group_block_sizes, strict=True
        ):
            start = cdiv(plan.computed_tokens, block_size)
            end = min(cdiv(plan.pool_hit_tokens, block_size), len(group_ids))
            result.update(group_ids[start:end])
        return result

    @staticmethod
    def _suffix_blocks(
        plan: KVTransferPlan, actual_pool_prefix: int
    ) -> list[list[int]]:
        offset = actual_pool_prefix - plan.computed_tokens
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
        recv_events: list[RecvEvent],
    ):
        plan = state.plan
        suffix = self._suffix_blocks(plan, actual_pool_prefix)
        state.suffix_block_ids = suffix
        params = plan.transfer_params
        event = RecvEvent(
            plan.request_id,
            plan.transfer_id,
            tuple(tuple(group) for group in suffix),
            str(params["remote_bootstrap_addr"]),
            str(params["remote_engine_id"]),
            int(params.get("remote_dp_rank", 0)),
        )
        recv_events.append(event)
        state.stage = "TRANSFER_LOADING"
        state.pd_started_at = time.monotonic()
        logger.debug(
            "[MTSC] pd_recv_start: request_id=%s A=%d T=%d suffix_blocks=%d",
            plan.request_id,
            actual_pool_prefix,
            plan.target_tokens,
            sum(len(group) for group in suffix),
        )

    def _aggregate_send_finished(
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
        send_finished = set()
        # Only request_finished creates a release dependency. Intermediate
        # backend results never release blocks of an active request.
        for request_id, state in list(self._finished_waits.items()):
            if state.is_finished:
                send_finished.add(request_id)
                del self._finished_waits[request_id]
        return send_finished

    def _aggregate_recv_finished(
        self,
        pool_load_results: list[LoadResult],
        transfer_recv_result: list[TransferResult],
    ) -> set[str]:
        pool_load_results_dict = {
            result.request_id: result for result in pool_load_results
        }
        finished_recv = set()
        for request_id in pool_load_results_dict:
            self._on_load_requests.pop(request_id, None)
            if request_id not in self._on_transfer:
                finished_recv.add(request_id)

        recv_events: list[RecvEvent] = []
        for request_id, state in list(self._on_transfer.items()):
            if state.stage == "POOL_DONE":
                actual_pool_prefix = state.plan.computed_tokens
                logger.debug(
                    "[MTSC] store_load: role=D request_id=%s tp_rank=%d "
                    "elapsed_ms=0.000 loaded_tokens=0 status=skipped",
                    request_id,
                    self.transfer.tp_rank,
                )
            elif state.stage == "POOL_LOADING" and request_id in pool_load_results_dict:
                load_result = pool_load_results_dict[request_id]
                actual_pool_prefix = load_result.loaded_tokens
                state.store_ms = (time.monotonic() - state.load_started_at) * 1000
                state.store_status = "error" if load_result.error else "ok"
                logger.debug(
                    "[MTSC] store_load_complete: request_id=%s L=%d H=%d A=%d",
                    request_id,
                    state.plan.computed_tokens,
                    state.plan.pool_hit_tokens,
                    actual_pool_prefix,
                )
            else:
                continue
            self._start_transfer(state, actual_pool_prefix, recv_events)

        if recv_events:
            self.transfer.recv(recv_events)

        for result in transfer_recv_result:
            request_id = result.request_id
            state = self._on_transfer.pop(request_id, None)
            if state is not None:
                finished_at = time.monotonic()
                logger.info(
                    "[MTSC] load_complete: role=D request_id=%s transfer_id=%s "
                    "tp_rank=%d store_ms=%.3f pd_ms=%.3f total_ms=%.3f "
                    "store_status=%s suffix_blocks=%d status=%s",
                    request_id,
                    state.plan.transfer_id,
                    self.transfer.tp_rank,
                    state.store_ms,
                    (finished_at - state.pd_started_at) * 1000,
                    (finished_at - state.load_started_at) * 1000,
                    state.store_status,
                    sum(len(group) for group in state.suffix_block_ids),
                    "error" if result.error else "ok",
                )
                if result.error is not None:
                    # Record suffix blocks as load errors when transfer fails
                    self._load_errors.update(
                        block
                        for group in state.suffix_block_ids
                        for block in group
                        if block >= 0
                    )
                if state.plan.target_tokens > state.plan.computed_tokens:
                    finished_recv.add(request_id)
        return finished_recv

    def update_load_metadata(self, metadata: MTSCConnectorMetadata) -> None:
        load_started_at = time.monotonic()
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

        for transfer_plan in metadata.transfer_plans:
            if transfer_plan.request_id in self._on_transfer:
                logger.warning(
                    "[MTSC] pd_plan: request_id=%s status=duplicate_ignored",
                    transfer_plan.request_id,
                )
                continue
            stage = (
                "POOL_LOADING"
                if transfer_plan.pool_hit_tokens > transfer_plan.computed_tokens
                else "POOL_DONE"
            )
            self._on_transfer[transfer_plan.request_id] = _TransferState(
                transfer_plan, stage, load_started_at=load_started_at
            )
            logger.debug(
                "[MTSC] pd_plan: request_id=%s stage=%s L=%d H=%d T=%d",
                transfer_plan.request_id,
                stage,
                transfer_plan.computed_tokens,
                transfer_plan.pool_hit_tokens,
                transfer_plan.target_tokens,
            )

    def update_save_metadata(self, metadata: MTSCConnectorMetadata) -> None:
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

    def update_finished_metadata(self, metadata: MTSCConnectorMetadata) -> None:
        # for P side update transfer state
        has_transfer_ready_updates = any(
            source_state.ready for source_state in metadata.transfer_states
        )
        if has_transfer_ready_updates:
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
        # for request wait finished
        for request_id, wait_for in metadata.finished_waits.items():
            self._finished_waits.setdefault(
                request_id,
                _FinishedWaitState(
                    wait_for,
                    pool_save_done=request_id not in self._on_save_requests,
                ),
            )

    def get_finished(
        self, finished_req_ids: set[str], metadata: MTSCConnectorMetadata
    ) -> tuple[set[str] | None, set[str] | None]:
        # step 1 update metadata & send or save
        self.update_load_metadata(metadata)
        self.update_save_metadata(metadata)
        self.update_finished_metadata(metadata)

        # step 2 poll results
        pool_results = self.pool.poll()
        load_errors_block_ids = self.pool.take_errors()
        transfer_results = self.transfer.poll()
        transfer_errors = self.transfer.take_errors()
        can_transfer_block_ids = {
            block_ids
            for state in self._on_transfer.values()
            for block_ids in self._pool_candidate_block_ids(state.plan)
        }
        self._load_errors.update(load_errors_block_ids - can_transfer_block_ids)
        self._load_errors.update(transfer_errors)

        # step 3 process loaded
        finished_recv = self._aggregate_recv_finished(
            pool_results.loads, transfer_results.recvs
        )

        # step4 process save and send
        finished_send = self._aggregate_send_finished(
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
