"""MTSC-owned scheduler state for Store + direct P/D transfer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm import envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
from vllm.v1.kv_cache_interface import SlidingWindowSpec
from vllm.v1.request import RequestStatus

from .kv_cache_pool import StoreLookupClient
from .protocol import (
    FinishedWait,
    KVPoolLoadRequest,
    KVPoolLoadSpec,
    KVPoolSaveRequest,
    KVTransferPlan,
    KVTransferSourceState,
    MTSCConnectorMetadata,
    MTSCLoadReport,
    MTSCWorkerMetadata,
)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

logger = init_logger("vllm.mtsc.scheduler")
logger.setLevel(envs.VLLM_LOGGING_LEVEL)


@dataclass
class TransferSpec:
    """Lookup boundaries awaiting allocation for a pool + transfer load."""

    computed_tokens: int  # Already computed tokens (local)
    pool_hit_tokens: int  # Tokens available in pool
    target_tokens: int  # Target total tokens to load


@dataclass
class _TrackedRequest:
    request: Request
    block_ids: tuple[list[int], ...]
    saved_tokens: int = 0
    is_decode: bool = False


@dataclass
class _LoadTrace:
    load_id: int
    role: str
    prompt_tokens: int
    target_tokens: int
    transfer_id: str | None
    reports: dict[int, MTSCLoadReport] = field(default_factory=dict)


def _groups(
    value: tuple[list[int], ...] | list[int] | list[list[int]],
) -> tuple[list[int], ...]:
    if isinstance(value, tuple):
        return tuple(group.copy() for group in value)
    if value and isinstance(value[0], list):
        return tuple(group.copy() for group in value)  # type: ignore[union-attr]
    return (value.copy(),)


class MTSCScheduler:
    """Plans per-step operations and retains requests until block release."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        decode_save: bool,
    ) -> None:
        assert vllm_config.kv_transfer_config is not None
        transfer = vllm_config.kv_transfer_config
        extra = transfer.kv_connector_extra_config
        self.is_producer = transfer.kv_role != "kv_consumer"
        self.is_consumer = transfer.kv_role != "kv_producer"
        self.decode_save = decode_save and self.is_consumer
        lookup_async = extra.get("lookup_async", False)
        if isinstance(lookup_async, str):
            lookup_async = lookup_async.strip().lower() not in {
                "0",
                "false",
                "no",
                "off",
            }
        self.lookup_async = bool(lookup_async)
        self.lookup_client = StoreLookupClient(vllm_config)
        self.block_size, _ = resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)
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

        # Retained through execution and any delayed block release.
        self._tracked_requests: dict[str, _TrackedRequest] = {}
        self._load_traces: dict[str, _LoadTrace] = {}
        self._next_load_id = 0
        self._expected_load_workers = vllm_config.parallel_config.world_size
        # Lookup decisions are consumed once local blocks are allocated.
        self._transfer_decisions: dict[str, TransferSpec] = {} # for requests that need to transfer blocks from a remote engine e.g. D side reqeust pull P kv cache
        self._load_decisions: dict[str, KVPoolLoadSpec] = {}
        # Per-step operations; detached after metadata is built.
        self._batch_pool_saves: list[KVPoolSaveRequest] = []
        self._batch_pool_loads: list[KVPoolLoadRequest] = []
        self._batch_transfers: list[KVTransferPlan] = []
        self._batch_transfer_state: list[KVTransferSourceState] = []
        # Release dependencies awaiting publication to Worker.
        self._finished_waits: dict[str, FinishedWait] = {}
        # Save issuance is conservative: Worker reports aggregate completion
        # only after request_finished, never for intermediate saves.
        self._saving_requests: set[str] = set()
        self._transferring_requests: set[str] = set()
        self._delayed_releases: set[str] = set()

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

    @staticmethod
    def _validate_transfer_params(request: Request) -> None:
        params = request.kv_transfer_params or {}
        invalid_fields = [
            name
            for name in ("remote_bootstrap_addr", "remote_engine_id")
            if not isinstance(params.get(name), str) or not params[name].strip()
        ]
        if invalid_fields:
            raise ValueError(
                "MTSC remote prefill requires non-empty string transfer parameters: "
                f"request_id={request.request_id}, "
                f"invalid_fields={', '.join(invalid_fields)}"
            )

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        params = request.kv_transfer_params or {}
        if self.is_consumer and params.get("do_remote_prefill"):
            self._validate_transfer_params(request)
        if self.is_producer and params.get("do_remote_decode"):
            self._truncate_mamba_prefill(request)

        target = request.num_prompt_tokens - (
            1 if self.has_mamba and request.num_prompt_tokens > 1 else 0
        )

        lookup_tokens = target // self.block_size * self.block_size
        if lookup_tokens < self.block_size:
            pool_hit = 0
        else:
            pool_hit = self.lookup_client.lookup(
                request.request_id,
                lookup_tokens,
                request.block_hashes,
                asynchronous=self.lookup_async,
            )
            if pool_hit is None:
                logger.debug(
                    "[MTSC] store_lookup: request_id=%s status=pending",
                    request.request_id,
                )
                return None, False

        # A retry may resolve to a different hit before allocation.
        self._load_decisions.pop(request.request_id, None)
        # can load from poll
        if pool_hit > num_computed_tokens:
            self._load_decisions[request.request_id] = KVPoolLoadSpec(
                computed_tokens=num_computed_tokens,
                pool_hit_tokens=pool_hit
            )
        if not (self.is_consumer and params.get("do_remote_prefill")):
            external = max(0, pool_hit - num_computed_tokens)
            return external, external > 0

        # for decode side pd transfer
        self._transfer_decisions.pop(request.request_id, None)
        self._transfer_decisions[request.request_id] = TransferSpec(
            computed_tokens=num_computed_tokens,
            pool_hit_tokens=pool_hit,
            target_tokens=target
        )
        logger.debug(
            "[MTSC] pd_plan: request_id=%s L=%d H=%d T=%d",
            request.request_id,
            num_computed_tokens,
            pool_hit,
            target,
        )
        external = target - num_computed_tokens
        return external, external > 0

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ) -> None:
        params = request.kv_transfer_params or {}
        if request.request_id in self._transfer_decisions:
            self._validate_transfer_params(request)
        all_blocks = blocks.get_block_ids()
        tracked = self._tracked_requests.get(request.request_id)
        if tracked is None:
            tracked = _TrackedRequest(
                request,
                all_blocks,
                is_decode=self.is_consumer and not bool(params.get("do_remote_decode")),
            )
            self._tracked_requests[request.request_id] = tracked
        elif all_blocks:
            tracked.block_ids = all_blocks

        load_spec = self._load_decisions.pop(request.request_id, None)
        transfer_spec = self._transfer_decisions.pop(request.request_id, None)
        load_id = 0
        if load_spec is not None or transfer_spec is not None:
            self._next_load_id += 1
            load_id = self._next_load_id
            self._load_traces[request.request_id] = _LoadTrace(
                load_id=load_id,
                role="D" if tracked.is_decode else "P",
                prompt_tokens=request.num_prompt_tokens,
                target_tokens=(
                    transfer_spec.target_tokens
                    if transfer_spec is not None
                    else request.num_prompt_tokens
                ),
                transfer_id=params.get("transfer_id"),
            )
        if load_spec is not None:
            self._batch_pool_loads.append(
                KVPoolLoadRequest(
                    request.request_id,
                    all_blocks,
                    list(request.block_hashes),
                    load_spec.computed_tokens,
                    load_spec.pool_hit_tokens,
                    load_id=load_id,
                )
            )
        elif self.is_producer and params.get("do_remote_decode"):
            logger.info(
                "[MTSC] store_load: role=P request_id=%s scope=decision "
                "prompt_tokens=%d store_tokens=0 store_ms_max=0.000 status=skipped",
                request.request_id,
                request.num_prompt_tokens,
            )

        if transfer_spec is not None:
            external_blocks = _groups(blocks.get_unhashed_block_ids_all_groups())
            transfer_id = str(params.get("transfer_id", request.request_id))
            self._batch_transfers.append(
                KVTransferPlan(
                    request_id=request.request_id,
                    transfer_id=transfer_id,
                    computed_tokens=transfer_spec.computed_tokens,
                    pool_hit_tokens=transfer_spec.pool_hit_tokens,
                    target_tokens=transfer_spec.target_tokens,
                    all_block_ids=all_blocks,
                    external_block_ids=external_blocks,
                    group_block_sizes=self.group_block_sizes,
                    blocks_per_sliding_window=self.blocks_per_sliding_window,
                    transfer_params=dict(params),
                    load_id=load_id,
                )
            )
            params["do_remote_prefill"] = False
            return

        if (
            self.is_producer
            and params.get("do_remote_decode")
            and request.request_id not in self._transferring_requests
        ):
            transfer_id = params.get("transfer_id")
            if transfer_id:
                self._batch_transfer_state.append(
                    KVTransferSourceState(request.request_id, str(transfer_id))
                )
                self._transferring_requests.add(request.request_id)
            else:
                logger.warning(
                    "[MTSC] pd_prepare: request_id=%s status=missing_transfer_id",
                    request.request_id,
                )

    def _append_blocks(
        self,
        tracked_req: _TrackedRequest,
        new_block_ids: tuple[list[int], ...] | list[int] | None,
    ) -> None:
        if new_block_ids is None:
            return
        incoming = _groups(new_block_ids)
        if not tracked_req.block_ids:
            tracked_req.block_ids = incoming
            return
        if len(incoming) != len(tracked_req.block_ids):
            raise ValueError("KV group count changed")
        for current, new in zip(tracked_req.block_ids, incoming, strict=True):
            current.extend(new)

    def _build_save_meta(
        self, tracked_req: _TrackedRequest, token_count: int, *, decode: bool
    ) -> KVPoolSaveRequest | None:
        # Like Mooncake Store's all_token_ids slice, only advance through
        # known tokens, excluding unverified drafts and async placeholders.
        token_count = min(token_count, tracked_req.request.num_tokens)
        complete = token_count // self.block_size * self.block_size
        start = tracked_req.saved_tokens
        if decode:
            prompt_end = (
                cdiv(tracked_req.request.num_prompt_tokens, self.block_size)
                * self.block_size
            )
            start = max(start, prompt_end)
        if complete <= start:
            return None
        tracked_req.saved_tokens = complete
        self._saving_requests.add(tracked_req.request.request_id)
        return KVPoolSaveRequest(
            tracked_req.request.request_id,
            tuple(group.copy() for group in tracked_req.block_ids),
            list(tracked_req.request.block_hashes),
            start_token=start,
            end_token=complete,
            prompt_tokens=tracked_req.request.num_prompt_tokens,
        )

    def build_connector_meta(self, output: SchedulerOutput) -> MTSCConnectorMetadata:
        pool_loading_req_ids = {request.request_id for request in self._batch_pool_loads}
        allow_save = self.is_producer or self.decode_save
        if allow_save:
            # 1. requests newly scheduled this step
            for req in output.scheduled_new_reqs:
                tracked_req = self._tracked_requests.get(req.req_id)
                if (
                    tracked_req is None
                    or req.req_id in pool_loading_req_ids
                    or (tracked_req.is_decode and not self.decode_save)
                ):
                    continue
                tracked_req.block_ids = _groups(req.block_ids)
                token_count = (
                    req.num_computed_tokens + output.num_scheduled_tokens[req.req_id]
                )
                save_meta = self._build_save_meta(
                    tracked_req, token_count, decode=tracked_req.is_decode
                )
                if save_meta is not None:
                    self._batch_pool_saves.append(save_meta)
            # 2. requests already running (cached) that continue this step
            cached = output.scheduled_cached_reqs
            for index, request_id in enumerate(cached.req_ids):
                tracked_req = self._tracked_requests.get(request_id)
                if tracked_req is None or (
                    tracked_req.is_decode and not self.decode_save
                ):
                    continue
                self._append_blocks(tracked_req, cached.new_block_ids[index])
                token_count = (
                    cached.num_computed_tokens[index]
                    + output.num_scheduled_tokens[request_id]
                )
                save_meta = self._build_save_meta(
                    tracked_req, token_count, decode=tracked_req.is_decode
                )
                if save_meta is not None:
                    self._batch_pool_saves.append(save_meta)

        finished = set(output.finished_req_ids)
        preempted = set(output.preempted_req_ids or set())
        # Preemption fences have already run before the current model step.
        # Never republish operations on blocks that can now be recycled.
        self._batch_pool_loads = [
            load_meta
            for load_meta in self._batch_pool_loads
            if load_meta.request_id not in preempted
        ]
        self._batch_pool_saves = [
            save_meta
            for save_meta in self._batch_pool_saves
            if save_meta.request_id not in preempted
        ]
        self._batch_transfers = [
            plan
            for plan in self._batch_transfers
            if plan.request_id not in finished | preempted
        ]
        self._batch_transfer_state = [
            source_state
            for source_state in self._batch_transfer_state
            if source_state.request_id not in preempted
        ]
        for request_id in finished | preempted:
            self.lookup_client.discard(request_id)
            self._load_decisions.pop(request_id, None)
            self._transfer_decisions.pop(request_id, None)
        for request_id in preempted:
            self._load_traces.pop(request_id, None)
            self._finished_waits.pop(request_id, None)
            self._delayed_releases.discard(request_id)
            self._tracked_requests.pop(request_id, None)
            self._saving_requests.discard(request_id)
            self._transferring_requests.discard(request_id)

        metadata = MTSCConnectorMetadata(
            pool_loads=self._batch_pool_loads,
            pool_saves=self._batch_pool_saves,
            transfer_plans=self._batch_transfers,
            transfer_states=self._batch_transfer_state,
            finished_waits=dict(self._finished_waits),
            finished_request_ids=finished,
            preempted_request_ids=preempted,
        )
        self._batch_pool_loads = []
        self._batch_pool_saves = []
        self._batch_transfers = []
        self._batch_transfer_state = []
        # Publish release dependencies once; Worker owns their progress and
        # _delayed_releases retains Scheduler state until the terminal result.
        self._finished_waits = {}
        return metadata

    def request_finished(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        params = request.kv_transfer_params or {}
        transfer_delay = False
        if (
            self.is_producer
            and params.get("transfer_id")
            and params.get("do_remote_decode")
        ):
            if request.status == RequestStatus.FINISHED_LENGTH_CAPPED and any(
                block_ids
            ):
                self._batch_transfer_state.append(
                    KVTransferSourceState(
                        request.request_id,
                        str(params["transfer_id"]),
                        _groups(block_ids),
                        ready=True,
                    )
                )
                transfer_delay = True
            else:
                self._batch_transfer_state.append(
                    KVTransferSourceState(
                        request.request_id, str(params["transfer_id"]), cancelled=True
                    )
                )

        pool_save_delay = request.request_id in self._saving_requests and any(block_ids)
        delay = pool_save_delay or transfer_delay
        if delay:
            self._finished_waits[request.request_id] = FinishedWait(
                store_save=pool_save_delay, kv_transfer=transfer_delay
            )
            self._delayed_releases.add(request.request_id)
        if not delay:
            self._load_traces.pop(request.request_id, None)
            self._tracked_requests.pop(request.request_id, None)
            self._saving_requests.discard(request.request_id)
            self._transferring_requests.discard(request.request_id)
        return delay, None

    def update_connector_output(self, output: KVConnectorOutput) -> None:
        metadata = output.kv_connector_worker_meta
        if metadata is not None:
            assert isinstance(metadata, MTSCWorkerMetadata)
            for report in metadata.load_reports:
                trace = self._load_traces.get(report.request_id)
                if trace is None or trace.load_id != report.load_id:
                    # A preempted/cancelled attempt may finish after a retry.
                    continue
                trace.reports[report.worker_rank] = report
                if len(trace.reports) == self._expected_load_workers:
                    self._log_load_complete(report.request_id, trace)
                    del self._load_traces[report.request_id]

        completed = output.finished_sending or set()
        self._delayed_releases.difference_update(completed)
        for request_id in completed:
            self._load_traces.pop(request_id, None)
            self._finished_waits.pop(request_id, None)
            self._tracked_requests.pop(request_id, None)
            self._saving_requests.discard(request_id)
            self._transferring_requests.discard(request_id)

    def _log_load_complete(self, request_id: str, trace: _LoadTrace) -> None:
        reports = list(trace.reports.values())

        def token_range(name: str) -> str:
            values = [getattr(report, name) for report in reports]
            low, high = min(values), max(values)
            return str(low) if low == high else f"{low}..{high}"

        logger.info(
            "[MTSC] %s: role=%s request_id=%s transfer_id=%s load_id=%d "
            "prompt_tokens=%d target_tokens=%d local_tokens=%s "
            "store_tokens=%s pd_tokens=%s store_ms_max=%.3f "
            "pd_ms_max=%.3f total_ms_max=%.3f workers=%d/%d "
            "store_status=%s status=%s",
            "load_complete" if trace.role == "D" else "store_load",
            trace.role,
            request_id,
            trace.transfer_id,
            trace.load_id,
            trace.prompt_tokens,
            trace.target_tokens,
            token_range("local_tokens"),
            token_range("store_tokens"),
            token_range("pd_tokens"),
            max(report.store_ms for report in reports),
            max(report.pd_ms for report in reports),
            max(report.total_ms for report in reports),
            len(reports),
            self._expected_load_workers,
            ",".join(sorted({report.store_status for report in reports})),
            "error" if any(report.status == "error" for report in reports) else "ok",
        )

    def has_pending_push_work(self) -> bool:
        return bool(self._delayed_releases)

    def reset_store(self) -> bool:
        return self.lookup_client.reset()

    def close(self) -> None:
        self._load_traces.clear()
        self.lookup_client.close()
