"""Unified MTSC worker and Decode two-stage state machine."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv

from .device import new_device_event
from .pd import PDTransfer
from .protocol import (
    DTwoStageLoadPlan,
    MTSCConnectorMetadata,
    SendRequirement,
    StoreRequest,
)
from .store import StoreIO

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


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
        self.store = StoreIO(config, kv_cache_config)
        try:
            self.pd = PDTransfer(config, kv_cache_config)
        except Exception:
            self.store.close()
            raise
        assert config.kv_transfer_config is not None
        self.is_producer = config.kv_transfer_config.kv_role == "kv_producer"
        self.is_consumer = config.kv_transfer_config.kv_role == "kv_consumer"
        self.timeout = float(
            config.kv_transfer_config.kv_connector_extra_config.get(
                "mtsc_pd_timeout_seconds", 180.0
            )
        )
        self.group_block_sizes = tuple(
            group.kv_cache_spec.block_size for group in kv_cache_config.kv_cache_groups
        )
        self._decode: dict[str, _DLoadState] = {}
        self._plain_store_loads: dict[str, StoreRequest] = {}
        self._send: dict[str, _SendState] = {}
        self._ignored_pd_recvs: set[str] = set()
        self._load_errors: set[int] = set()
        self._closed = False

    def register_kv_caches(
        self,
        kv_caches: dict[
            str, torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]
        ],
    ) -> None:
        self.store.register(kv_caches)
        try:
            self.pd.register(kv_caches)
        except Exception:
            try:
                self.pd.close()
            finally:
                self.store.close()
            raise
        logger.info("MTSC registered Store and PD data planes")

    def handle_preemptions(self, metadata: MTSCConnectorMetadata) -> None:
        request_ids = metadata.preempted_request_ids
        if not request_ids:
            return
        # vLLM invokes this hook before the current model step can overwrite
        # recycled blocks. A PUT reads GPU memory asynchronously, so it must be
        # fenced here rather than in wait_for_save().
        self.store.finish_preempted_saves(request_ids)
        self.store.finish_preempted_loads(request_ids)
        self.pd.finish_receives(request_ids)
        for request_id in request_ids:
            self._decode.pop(request_id, None)
            self._plain_store_loads.pop(request_id, None)
            self._send.pop(request_id, None)
            self._ignored_pd_recvs.discard(request_id)

    def _accept_metadata(self, metadata: MTSCConnectorMetadata) -> None:
        decode_ids = set(self._decode)
        decode_ids.update(plan.request_id for plan in metadata.decode_plans)
        save_requests = [request for request in metadata.store_requests if request.save]
        save_event = None
        if save_requests:
            save_event = new_device_event()
            save_event.record()
        for request in metadata.store_requests:
            if request.load is not None and request.load.enabled:
                self.store.enqueue_load(request)
                if request.request_id not in decode_ids:
                    self._plain_store_loads[request.request_id] = request
            if request.save:
                self.store.enqueue_save(request, save_event)
        self.pd.apply_updates(metadata.pd_send_updates)
        for request_id, requirement in metadata.send_requirements.items():
            self._send.setdefault(request_id, _SendState(requirement))
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
    def _actual_store_prefix(
        plan: DTwoStageLoadPlan, invalid_block_ids: set[int]
    ) -> int:
        if not invalid_block_ids:
            return plan.store_candidate_tokens
        actual = plan.store_candidate_tokens
        for group_ids, block_size in zip(
            plan.all_block_ids, plan.group_block_sizes, strict=True
        ):
            start = cdiv(plan.local_prefix_tokens, block_size)
            end = cdiv(plan.store_candidate_tokens, block_size)
            for block_index in range(start, min(end, len(group_ids))):
                if group_ids[block_index] in invalid_block_ids:
                    actual = min(actual, block_index * block_size)
                    break
        return max(plan.local_prefix_tokens, actual)

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
        self.pd.receive(
            plan.request_id,
            plan.transfer_id,
            suffix,
            str(params["remote_engine_id"]),
            str(params["remote_bootstrap_addr"]),
            int(params.get("remote_dp_rank", 0)),
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
        completed: set[str] = set()
        for request_id in store_done:
            state = self._send.get(request_id)
            if state is None:
                completed.add(request_id)
            else:
                state.store_done = True
        for request_id in pd_done:
            state = self._send.get(request_id)
            if state is None:
                completed.add(request_id)
            else:
                state.pd_done = True
        for request_id, state in list(self._send.items()):
            if state.complete:
                completed.add(request_id)
                del self._send[request_id]
        return completed

    def get_finished(
        self, finished_req_ids: set[str], metadata: MTSCConnectorMetadata
    ) -> tuple[set[str] | None, set[str] | None]:
        self._accept_metadata(metadata)
        all_finished = finished_req_ids | metadata.finished_request_ids
        store_send, store_recv = self.store.poll(all_finished)
        store_errors = self.store.take_errors()
        staged_blocks: set[int] = set()
        for state in self._decode.values():
            staged_blocks.update(self._store_candidate_block_ids(state.plan))
        self._load_errors.update(store_errors - staged_blocks)

        for request_id in store_recv:
            request = self._plain_store_loads.pop(request_id, None)
            if request is not None:
                self.pd.reformat_npu_blocks(
                    self._plain_store_loaded_blocks(request, store_errors),
                    self.pd.tp_size,
                )

        finished_recv: set[str] = set()
        for request_id, state in list(self._decode.items()):
            if state.stage == "STORE_DONE":
                started = self._start_pd(state, state.plan.local_prefix_tokens)
                if not started:
                    if state.plan.wait_for_completion:
                        finished_recv.add(request_id)
                    del self._decode[request_id]
            elif state.stage == "STORE_PENDING" and request_id in store_recv:
                actual = self._actual_store_prefix(state.plan, store_errors)
                logger.info(
                    "MTSC D Store terminal: request_id=%s L=%d H=%d A=%d",
                    request_id,
                    state.plan.local_prefix_tokens,
                    state.plan.store_candidate_tokens,
                    actual,
                )
                self.pd.reformat_npu_blocks(
                    self._store_loaded_blocks(state.plan, actual), self.pd.tp_size
                )
                started = self._start_pd(state, actual)
                if not started:
                    if state.plan.wait_for_completion:
                        finished_recv.add(request_id)
                    del self._decode[request_id]

        pd_send, pd_recv, pd_failed = self.pd.poll()
        ignored = (pd_recv | pd_failed) & self._ignored_pd_recvs
        self._ignored_pd_recvs.difference_update(ignored)
        pd_recv -= ignored
        pd_failed -= ignored
        for request_id, state in list(self._decode.items()):
            if state.stage != "PD_PENDING":
                continue
            failed = request_id in pd_failed
            if request_id not in pd_recv and not failed:
                continue
            if failed:
                for group in state.suffix_block_ids:
                    self._load_errors.update(block for block in group if block >= 0)
                logger.warning(
                    "MTSC D PD suffix failed: request_id=%s reason=transfer_error",
                    request_id,
                )
            if state.plan.wait_for_completion:
                finished_recv.add(request_id)
            del self._decode[request_id]

        # Plain P-side Store loads are not represented by a D state.
        finished_recv.update(store_recv - set(self._decode))
        finished_send = self._aggregate_sends(store_send, pd_send)
        return finished_send or None, finished_recv or None

    @staticmethod
    def _store_loaded_blocks(
        plan: DTwoStageLoadPlan, actual_store_prefix: int
    ) -> list[list[int]]:
        result: list[list[int]] = []
        for group, block_size in zip(
            plan.all_block_ids, plan.group_block_sizes, strict=True
        ):
            start = cdiv(plan.local_prefix_tokens, block_size)
            end = min(cdiv(actual_store_prefix, block_size), len(group))
            result.append(group[start:end])
        return result

    def _plain_store_loaded_blocks(
        self, request: StoreRequest, invalid_block_ids: set[int]
    ) -> list[list[int]]:
        if request.load is None:
            return [[] for _ in request.block_ids]
        result: list[list[int]] = []
        for group, block_size in zip(
            request.block_ids, self.group_block_sizes, strict=True
        ):
            start = cdiv(request.load.local_tokens, block_size)
            end = min(cdiv(request.load.store_tokens, block_size), len(group))
            result.append(
                [
                    block
                    for block in group[start:end]
                    if block >= 0 and block not in invalid_block_ids
                ]
            )
        return result

    def get_block_ids_with_load_errors(self) -> set[int]:
        errors, self._load_errors = self._load_errors, set()
        return errors

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.pd.close()
        finally:
            self.store.close()
