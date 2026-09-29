"""Unified MTSC worker and Decode two-stage state machine."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv

from .kv_cache_pool import KVCachePool, LoadEvent, SaveEvent
from .kv_transfer import KVTransfer, RecvEvent, SendEvent
from .protocol import (
    DTwoStageLoadPlan,
    MTSCConnectorMetadata,
    SendRequirement,
    StoreRequest,
)
from .utils import new_device_event

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


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
class _DLoadState:
    plan: DTwoStageLoadPlan
    stage: str
    created_at: float = field(default_factory=time.monotonic)
    pd_started_at: float | None = None
    suffix_block_ids: list[list[int]] = field(default_factory=list)


@dataclass
class _SendState:
    required: SendRequirement
    store_done: bool = False
    pd_done: bool = False

    @property
    def complete(self) -> bool:
        return (not self.required.store or self.store_done) and (
            not self.required.pd or self.pd_done
        )


class MTSCWorker:
    def __init__(self, config: VllmConfig, kv_cache_config: KVCacheConfig) -> None:
        self.pool: KVCachePool = _create_pool(config, kv_cache_config)
        try:
            self.transfer: KVTransfer = _create_transfer(config, kv_cache_config)
        except Exception:
            self.pool.close()
            raise
        assert config.kv_transfer_config is not None
        self.is_producer = config.kv_transfer_config.kv_role == "kv_producer"
        self.is_consumer = config.kv_transfer_config.kv_role == "kv_consumer"
        self.timeout = float(
            config.kv_transfer_config.kv_connector_extra_config.get(
                "mtsc_pd_timeout_seconds", 180.0
            )
        )
        self._decode: dict[str, _DLoadState] = {}
        self._plain_store_loads: dict[str, StoreRequest] = {}
        self._send: dict[str, _SendState] = {}
        self._ignored_pd_recvs: set[str] = set()
        self._load_errors: set[int] = set()
        self._save_pending: set[str] = set()
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
        logger.info("MTSC registered Store and PD data planes")

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
            self._save_pending.discard(request_id)
            self._decode.pop(request_id, None)
            self._plain_store_loads.pop(request_id, None)
            self._send.pop(request_id, None)
            self._ignored_pd_recvs.discard(request_id)
        # Error sets are consumed in the same cycle as their terminal result;
        # preempt() clears any backend-owned errors before block reuse.

    def _accept_metadata(self, metadata: MTSCConnectorMetadata) -> None:
        decode_ids = set(self._decode)
        decode_ids.update(plan.request_id for plan in metadata.decode_plans)
        save_requests = [request for request in metadata.store_requests if request.save]
        save_event = None
        if save_requests:
            save_event = new_device_event()
            save_event.record()
        for request in metadata.store_requests:
            block_ids = tuple(tuple(group) for group in request.block_ids)
            hashes = tuple(bytes(value) for value in request.block_hashes)
            if request.load is not None and request.load.enabled:
                self.pool.load(
                    LoadEvent(
                        request.request_id,
                        block_ids,
                        hashes,
                        request.load.local_tokens,
                        request.load.store_tokens,
                    )
                )
                if request.request_id not in decode_ids:
                    self._plain_store_loads[request.request_id] = request
            if request.save:
                self.pool.save(
                    SaveEvent(
                        request.request_id,
                        block_ids,
                        hashes,
                        request.save_from,
                        request.token_count,
                        save_event,
                        request.prompt_tokens,
                    )
                )
                self._save_pending.add(request.request_id)
                state = self._send.get(request.request_id)
                if state is not None:
                    state.store_done = False
        ready_updates = [
            update for update in metadata.pd_send_updates if update.source_ready
        ]
        if ready_updates:
            # send() publishes readable source memory. This fences preceding
            # device writes before the transport can access those addresses.
            ready = new_device_event()
            ready.record()
            ready.synchronize()
        for update in metadata.pd_send_updates:
            if update.abort:
                self.transfer.cancel(update.request_id, update.transfer_id)
            elif update.source_ready:
                self.transfer.send(
                    SendEvent(
                        update.request_id,
                        update.transfer_id,
                        tuple(tuple(group) for group in update.block_ids),
                    )
                )
            else:
                self.transfer.prepare(update.request_id, update.transfer_id)
        for request_id, requirement in metadata.send_requirements.items():
            self._send.setdefault(
                request_id,
                _SendState(
                    requirement,
                    store_done=request_id not in self._save_pending,
                ),
            )
        for plan in metadata.decode_plans:
            if plan.request_id in self._decode:
                continue
            stage = (
                "STORE_PENDING"
                if plan.store_candidate_tokens > plan.local_prefix_tokens
                else "STORE_DONE"
            )
            self._decode[plan.request_id] = _DLoadState(plan, stage)
            logger.info(
                "MTSC D state created: request_id=%s stage=%s L=%d H=%d T=%d",
                plan.request_id,
                stage,
                plan.local_prefix_tokens,
                plan.store_candidate_tokens,
                plan.target_prefix_tokens,
            )

    @staticmethod
    def _store_candidate_block_ids(plan: DTwoStageLoadPlan) -> set[int]:
        result: set[int] = set()
        for group_ids, block_size in zip(
            plan.all_block_ids, plan.group_block_sizes, strict=True
        ):
            start = cdiv(plan.local_prefix_tokens, block_size)
            end = min(cdiv(plan.store_candidate_tokens, block_size), len(group_ids))
            result.update(group_ids[start:end])
        return result

    @staticmethod
    def _suffix_blocks(
        plan: DTwoStageLoadPlan, actual_store_prefix: int
    ) -> list[list[int]]:
        offset = actual_store_prefix - plan.local_prefix_tokens
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

    def _start_pd(self, state: _DLoadState, actual_store_prefix: int) -> bool:
        plan = state.plan
        suffix = self._suffix_blocks(plan, actual_store_prefix)
        state.suffix_block_ids = suffix
        if not plan.pd_enabled:
            if actual_store_prefix < plan.target_prefix_tokens:
                for group in suffix:
                    self._load_errors.update(block for block in group if block >= 0)
                logger.warning(
                    "MTSC D Store miss has no P source: request_id=%s", plan.request_id
                )
            state.stage = "DONE"
            return False
        params = plan.kv_transfer_params
        self.transfer.recv(
            RecvEvent(
                plan.request_id,
                plan.transfer_id,
                tuple(tuple(group) for group in suffix),
                str(params["remote_bootstrap_addr"]),
                str(params["remote_engine_id"]),
                int(params.get("remote_dp_rank", 0)),
            )
        )
        state.stage = "PD_PENDING"
        state.pd_started_at = time.monotonic()
        if not plan.wait_for_completion:
            self._ignored_pd_recvs.add(plan.request_id)
        logger.info(
            "MTSC D PD suffix started: request_id=%s A=%d T=%d blocks=%d",
            plan.request_id,
            actual_store_prefix,
            plan.target_prefix_tokens,
            sum(len(group) for group in suffix),
        )
        return True

    def _aggregate_sends(self, store_done: set[str], pd_done: set[str]) -> set[str]:
        self._save_pending.difference_update(store_done)
        for request_id in store_done:
            state = self._send.get(request_id)
            if state is not None:
                state.store_done = True
        for request_id in pd_done:
            state = self._send.get(request_id)
            if state is not None:
                state.pd_done = True
        completed = set()
        # Only request_finished creates a release dependency. Intermediate
        # backend results never release blocks of an active request.
        for request_id, state in list(self._send.items()):
            if state.complete:
                completed.add(request_id)
                del self._send[request_id]
        return completed

    def get_finished(
        self, finished_req_ids: set[str], metadata: MTSCConnectorMetadata
    ) -> tuple[set[str] | None, set[str] | None]:
        self._accept_metadata(metadata)
        pool_results = self.pool.poll()
        store_errors = self.pool.take_errors()
        staged_ids = set(self._decode)
        staged_blocks = {
            block
            for state in self._decode.values()
            for block in self._store_candidate_block_ids(state.plan)
        }
        self._load_errors.update(store_errors - staged_blocks)
        loads = {result.request_id: result for result in pool_results.loads}
        finished_recv = set()
        for request_id in loads.keys() - staged_ids:
            self._plain_store_loads.pop(request_id, None)
            finished_recv.add(request_id)

        for request_id, state in list(self._decode.items()):
            if state.stage == "STORE_DONE":
                actual = state.plan.local_prefix_tokens
            elif state.stage == "STORE_PENDING" and request_id in loads:
                actual = loads[request_id].loaded_tokens
                if (
                    not state.plan.local_prefix_tokens
                    <= actual
                    <= state.plan.store_candidate_tokens
                ):
                    raise ValueError("Pool returned an invalid loaded prefix")
                logger.info(
                    "MTSC D Pool terminal: request_id=%s L=%d H=%d A=%d",
                    request_id,
                    state.plan.local_prefix_tokens,
                    state.plan.store_candidate_tokens,
                    actual,
                )
            else:
                continue
            if not self._start_pd(state, actual):
                if state.plan.wait_for_completion:
                    finished_recv.add(request_id)
                del self._decode[request_id]

        transfer_results = self.transfer.poll()
        transfer_errors = self.transfer.take_errors()
        self._load_errors.update(transfer_errors)
        for result in transfer_results.recvs:
            request_id = result.request_id
            if request_id in self._ignored_pd_recvs:
                self._ignored_pd_recvs.discard(request_id)
                self._decode.pop(request_id, None)
                continue
            state = self._decode.pop(request_id, None)
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
