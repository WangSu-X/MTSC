"""MTSC-owned scheduler state for Store + direct P/D transfer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
from vllm.v1.kv_cache_interface import SlidingWindowSpec
from vllm.v1.request import RequestStatus

from .protocol import (
    DTwoStageLoadPlan,
    MTSCConnectorMetadata,
    PDSendUpdate,
    SendRequirement,
    StoreLoadSpec,
    StoreRequest,
)
from .store import StoreLookupClient

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class _LookupDecision:
    local_tokens: int
    store_tokens: int
    target_tokens: int


@dataclass
class _TrackedRequest:
    request: Request
    block_ids: tuple[list[int], ...]
    token_count: int = 0
    saved_tokens: int = 0


def _groups(
    value: tuple[list[int], ...] | list[int] | list[list[int]],
) -> tuple[list[int], ...]:
    if isinstance(value, tuple):
        return tuple(group.copy() for group in value)
    if value and isinstance(value[0], list):
        return tuple(group.copy() for group in value)  # type: ignore[union-attr]
    return (value.copy(),)


class MTSCScheduler:
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        decode_save: bool,
    ) -> None:
        assert vllm_config.kv_transfer_config is not None
        transfer = vllm_config.kv_transfer_config
        extra = transfer.kv_connector_extra_config
        self.kv_role = transfer.kv_role
        self.is_producer = self.kv_role == "kv_producer"
        self.is_consumer = self.kv_role == "kv_consumer"
        self.decode_save = decode_save and self.is_consumer
        self.lookup_async = bool(extra.get("lookup_async", False))
        self.lookup_client = StoreLookupClient(vllm_config)
        self.block_size, self.hash_block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )
        self.group_block_sizes = tuple(
            group.kv_cache_spec.block_size for group in kv_cache_config.kv_cache_groups
        )
        self.blocks_per_sliding_window = tuple(
            cdiv(group.kv_cache_spec.sliding_window, group.kv_cache_spec.block_size) + 1
            if isinstance(group.kv_cache_spec, SlidingWindowSpec)
            else 0
            for group in kv_cache_config.kv_cache_groups
        )
        self.has_mamba = kv_cache_config.has_mamba_layers

        self._decisions: dict[str, _LookupDecision] = {}
        self._load_specs: dict[str, StoreLoadSpec] = {}
        self._tracked: dict[str, _TrackedRequest] = {}
        self._pending_store: list[StoreRequest] = []
        self._pending_decode: dict[str, DTwoStageLoadPlan] = {}
        self._pending_pd: list[PDSendUpdate] = []
        self._pending_requirements: dict[str, SendRequirement] = {}
        self._save_issued: set[str] = set()
        self._pd_registered: set[str] = set()
        self._delayed: set[str] = set()

    @staticmethod
    def _valid_pd(params: dict[str, Any]) -> bool:
        return all(
            params.get(key)
            for key in ("transfer_id", "remote_engine_id", "remote_bootstrap_addr")
        )

    def _truncate_mamba_prefill(self, request: Request) -> None:
        params = request.kv_transfer_params or {}
        if (
            not self.has_mamba
            or params.get("_mtsc_p_truncated")
            or request.num_prompt_tokens <= 1
        ):
            return
        if request.prompt_token_ids is not None:
            request.prompt_token_ids.pop()
        elif request.prompt_embeds is not None:
            request.prompt_embeds = request.prompt_embeds[:-1]
        else:
            return
        request._all_token_ids.pop()
        request.num_prompt_tokens -= 1
        request.max_tokens = 1
        params["_mtsc_p_truncated"] = True

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        params = request.kv_transfer_params or {}
        if self.is_producer and params.get("do_remote_decode"):
            self._truncate_mamba_prefill(request)

        lookup_tokens = request.num_tokens // self.block_size * self.block_size
        if lookup_tokens < self.block_size:
            store_hit = 0
        else:
            hit = self.lookup_client.lookup(
                request.request_id,
                lookup_tokens,
                request.block_hashes,
                asynchronous=self.lookup_async,
            )
            if hit is None:
                logger.debug(
                    "MTSC Store lookup pending: request_id=%s", request.request_id
                )
                return None, False
            store_hit = hit
            if store_hit == request.num_tokens:
                store_hit = max(
                    0, (request.num_tokens - 1) // self.block_size * self.block_size
                )

        store_hit = max(num_computed_tokens, store_hit)
        if not (self.is_consumer and params.get("do_remote_prefill")):
            external = max(0, store_hit - num_computed_tokens)
            if external:
                self._load_specs[request.request_id] = StoreLoadSpec(
                    num_computed_tokens, store_hit
                )
            return external, external > 0

        target = request.num_prompt_tokens - (
            1 if self.has_mamba and request.num_prompt_tokens > 1 else 0
        )
        target = max(num_computed_tokens, min(target, request.num_tokens))
        store_hit = min(store_hit, target)
        pd_enabled = self._valid_pd(params)
        if not pd_enabled:
            logger.warning(
                "MTSC invalid decode transfer spec; using Store only: request_id=%s",
                request.request_id,
            )
            target = store_hit
        self._decisions[request.request_id] = _LookupDecision(
            num_computed_tokens, store_hit, target
        )
        if store_hit > num_computed_tokens:
            self._load_specs[request.request_id] = StoreLoadSpec(
                num_computed_tokens, store_hit
            )
        logger.info(
            "MTSC D load planned: request_id=%s L=%d H=%d T=%d",
            request.request_id,
            num_computed_tokens,
            store_hit,
            target,
        )
        external = target - num_computed_tokens
        return external, external > 0

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ) -> None:
        params = request.kv_transfer_params or {}
        all_blocks = _groups(blocks.get_block_ids())
        tracked = self._tracked.get(request.request_id)
        if tracked is None:
            tracked = _TrackedRequest(request, all_blocks)
            self._tracked[request.request_id] = tracked
        elif all_blocks:
            tracked.block_ids = all_blocks

        decision = self._decisions.pop(request.request_id, None)
        load_spec = self._load_specs.pop(request.request_id, None)
        if decision is not None:
            external_blocks = _groups(blocks.get_unhashed_block_ids_all_groups())
            if load_spec is not None:
                load_spec.enabled = True
                self._pending_store.append(
                    StoreRequest(
                        request.request_id,
                        decision.store_tokens,
                        all_blocks,
                        request.block_hashes,
                        load=load_spec,
                    )
                )
            transfer_id = str(params.get("transfer_id", request.request_id))
            self._pending_decode[request.request_id] = DTwoStageLoadPlan(
                request_id=request.request_id,
                transfer_id=transfer_id,
                local_prefix_tokens=decision.local_tokens,
                store_candidate_tokens=decision.store_tokens,
                target_prefix_tokens=decision.target_tokens,
                all_block_ids=all_blocks,
                external_block_ids=external_blocks,
                group_block_sizes=self.group_block_sizes,
                blocks_per_sliding_window=self.blocks_per_sliding_window,
                pd_enabled=self._valid_pd(params),
                wait_for_completion=num_external_tokens > 0,
                kv_transfer_params=dict(params),
            )
            params["do_remote_prefill"] = False
            return

        if load_spec is not None and num_external_tokens > 0:
            load_spec.enabled = True
            self._pending_store.append(
                StoreRequest(
                    request.request_id,
                    load_spec.store_tokens,
                    all_blocks,
                    request.block_hashes,
                    load=load_spec,
                )
            )

        if (
            self.is_producer
            and params.get("do_remote_decode")
            and request.request_id not in self._pd_registered
        ):
            transfer_id = params.get("transfer_id")
            if transfer_id:
                self._pending_pd.append(
                    PDSendUpdate(request.request_id, str(transfer_id))
                )
                self._pd_registered.add(request.request_id)
            else:
                logger.warning(
                    "MTSC producer request lacks transfer_id: request_id=%s",
                    request.request_id,
                )

    def _append_blocks(
        self, tracked: _TrackedRequest, value: tuple[list[int], ...] | list[int] | None
    ) -> None:
        if value is None:
            return
        incoming = _groups(value)
        if not tracked.block_ids:
            tracked.block_ids = incoming
            return
        if len(incoming) != len(tracked.block_ids):
            raise ValueError("KV group count changed")
        for current, new in zip(tracked.block_ids, incoming, strict=True):
            current.extend(new)

    def _save(
        self, tracked: _TrackedRequest, token_count: int, *, decode: bool
    ) -> StoreRequest | None:
        complete = token_count // self.block_size * self.block_size
        start = tracked.saved_tokens
        if decode:
            prompt_end = (
                cdiv(tracked.request.num_prompt_tokens, self.block_size)
                * self.block_size
            )
            start = max(start, prompt_end)
        if complete <= start:
            return None
        tracked.saved_tokens = complete
        self._save_issued.add(tracked.request.request_id)
        return StoreRequest(
            tracked.request.request_id,
            complete,
            tuple(group.copy() for group in tracked.block_ids),
            tracked.request.block_hashes,
            save=True,
            save_from=start,
            token_ids=list(tracked.request.all_token_ids[:complete]),
            prompt_tokens=tracked.request.num_prompt_tokens,
        )

    def build_connector_meta(self, output: SchedulerOutput) -> MTSCConnectorMetadata:
        store_requests, self._pending_store = self._pending_store, []
        loading = {
            request.request_id
            for request in store_requests
            if request.load is not None and request.load.enabled
        }
        allow_save = self.is_producer or self.decode_save
        if allow_save:
            for scheduled in output.scheduled_new_reqs:
                tracked = self._tracked.get(scheduled.req_id)
                if tracked is None or scheduled.req_id in loading:
                    continue
                tracked.block_ids = _groups(scheduled.block_ids)
                token_count = (
                    scheduled.num_computed_tokens
                    + output.num_scheduled_tokens[scheduled.req_id]
                )
                tracked.token_count = token_count
                request = self._save(tracked, token_count, decode=self.is_consumer)
                if request is not None:
                    store_requests.append(request)
            cached = output.scheduled_cached_reqs
            for index, request_id in enumerate(cached.req_ids):
                tracked = self._tracked.get(request_id)
                if tracked is None:
                    continue
                self._append_blocks(tracked, cached.new_block_ids[index])
                token_count = (
                    cached.num_computed_tokens[index]
                    + output.num_scheduled_tokens[request_id]
                )
                tracked.token_count = token_count
                request = self._save(tracked, token_count, decode=self.is_consumer)
                if request is not None:
                    store_requests.append(request)

        finished = set(output.finished_req_ids)
        preempted = set(output.preempted_req_ids or set())
        if preempted:
            # A preempted request's blocks may be recycled by this very step.
            # Do not publish newly queued reads/writes or stale P placeholders
            # after the worker-side preemption fence has already run.
            store_requests = [
                request
                for request in store_requests
                if request.request_id not in preempted
            ]
            self._pending_pd = [
                update
                for update in self._pending_pd
                if update.request_id not in preempted
            ]
            for request_id in preempted:
                self._pending_requirements.pop(request_id, None)
        for request_id in finished | preempted:
            self.lookup_client.discard(request_id)
            self._load_specs.pop(request_id, None)
            self._decisions.pop(request_id, None)
            self._pending_decode.pop(request_id, None)

        metadata = MTSCConnectorMetadata(
            store_requests=store_requests,
            decode_plans=list(self._pending_decode.values()),
            pd_send_updates=self._pending_pd,
            send_requirements=self._pending_requirements,
            finished_request_ids=finished,
            preempted_request_ids=preempted,
        )
        self._pending_decode.clear()
        self._pending_pd = []
        self._pending_requirements = {}
        for request_id in preempted:
            self._tracked.pop(request_id, None)
            self._save_issued.discard(request_id)
            self._pd_registered.discard(request_id)
        return metadata

    def request_finished(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        params = request.kv_transfer_params or {}
        pd_delay = False
        if (
            self.is_producer
            and params.get("transfer_id")
            and params.get("do_remote_decode")
        ):
            if request.status == RequestStatus.FINISHED_LENGTH_CAPPED and any(
                block_ids
            ):
                self._pending_pd.append(
                    PDSendUpdate(
                        request.request_id,
                        str(params["transfer_id"]),
                        _groups(block_ids),
                        source_ready=True,
                    )
                )
                pd_delay = True
            else:
                self._pending_pd.append(
                    PDSendUpdate(
                        request.request_id, str(params["transfer_id"]), abort=True
                    )
                )
        store_delay = request.request_id in self._save_issued and any(block_ids)
        delay = store_delay or pd_delay
        if delay:
            self._pending_requirements[request.request_id] = SendRequirement(
                store_delay, pd_delay
            )
            self._delayed.add(request.request_id)
        if not delay:
            self._tracked.pop(request.request_id, None)
        return delay, None

    def update_connector_output(self, output: KVConnectorOutput) -> None:
        completed = output.finished_sending or set()
        self._delayed.difference_update(completed)
        for request_id in completed:
            self._tracked.pop(request_id, None)
            self._save_issued.discard(request_id)
            self._pd_registered.discard(request_id)

    def has_pending_push_work(self) -> bool:
        return bool(self._delayed)

    def reset_store(self) -> bool:
        return self.lookup_client.reset()

    def close(self) -> None:
        self.lookup_client.close()
