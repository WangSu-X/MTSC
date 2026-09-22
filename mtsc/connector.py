"""vLLM external connector entry point for the self-owned MTSC stack."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.outputs import KVConnectorOutput

from .device import is_npu_platform, npu_kv_nz_enabled
from .protocol import MTSCConnectorMetadata
from .scheduler import MTSCScheduler
from .worker import MTSCWorker

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


def _config_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def _require_recompute_policy(config: VllmConfig) -> None:
    transfer = config.kv_transfer_config
    if transfer is None or transfer.kv_load_failure_policy != "recompute":
        raise ValueError(
            "MTSC requires kv_load_failure_policy=recompute so terminal "
            "Store/PD load failures fall back to local computation"
        )


class MTSCConnector(KVConnectorBase_V1, SupportsHMA):
    """One connector that owns Store GET/PUT and direct P/D transfer."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        if vllm_config.kv_transfer_config is None:
            raise ValueError("kv_transfer_config is required")
        _require_recompute_policy(vllm_config)
        if vllm_config.kv_transfer_config.kv_role not in {
            "kv_producer",
            "kv_consumer",
        }:
            raise ValueError("MTSC MVP requires kv_role=kv_producer or kv_consumer")

        extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        if not _config_bool(extra.get("load_async"), True):
            raise ValueError("MTSC requires kv_connector_extra_config.load_async=true")
        if not _config_bool(extra.get("mtsc_store_topology_namespace"), True):
            logger.warning(
                "MTSC Store topology namespace is disabled; incompatible "
                "deployments must use distinct cache_prefix values"
            )

        self.scheduler: MTSCScheduler | None = None
        self.worker: MTSCWorker | None = None

        if role == KVConnectorRole.SCHEDULER:
            decode_save = _config_bool(extra.get("mtsc_decode_save"), True)
            if decode_save and npu_kv_nz_enabled(vllm_config):
                decode_save = False
                logger.warning(
                    "MTSC disables Decode Store save for Ascend NZ cache layout"
                )
            self.scheduler = MTSCScheduler(vllm_config, kv_cache_config, decode_save)
        else:
            timeout = float(extra.get("mtsc_pd_timeout_seconds", 180.0))
            if timeout <= 0:
                raise ValueError("mtsc_pd_timeout_seconds must be positive")
            self.worker = MTSCWorker(vllm_config, kv_cache_config)
        logger.info(
            "Initialized MTSC connector: process_role=%s kv_role=%s",
            role.name,
            vllm_config.kv_transfer_config.kv_role,
        )

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig) -> str | None:
        if (
            is_npu_platform()
            or vllm_config.model_config is None
            or vllm_config.model_config.use_mla
        ):
            return None
        return "HND"

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        # Direct PD needs per-layer registration in this first implementation.
        return False

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        assert self.scheduler is not None
        return self.scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        assert self.scheduler is not None
        self.scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> MTSCConnectorMetadata:
        assert self.scheduler is not None
        return self.scheduler.build_connector_meta(scheduler_output)

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        if self.scheduler is not None:
            self.scheduler.update_connector_output(connector_output)

    def request_finished(
        self, request: Request, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        return self.request_finished_all_groups(request, (block_ids,))

    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.scheduler is not None
        return self.scheduler.request_finished(request, block_ids)

    def has_pending_push_work(self) -> bool:
        return self.scheduler is not None and self.scheduler.has_pending_push_work()

    def take_events(self) -> Iterable[KVCacheEvent]:
        return iter(())

    def reset_cache(self) -> bool | None:
        return self.scheduler.reset_store() if self.scheduler is not None else False

    def register_kv_caches(
        self,
        kv_caches: dict[
            str, torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]
        ],
    ) -> None:
        assert self.worker is not None
        self.worker.register_kv_caches(kv_caches)

    def handle_preemptions(self, kv_connector_metadata: MTSCConnectorMetadata) -> None:
        assert self.worker is not None
        if not isinstance(kv_connector_metadata, MTSCConnectorMetadata):
            raise TypeError(
                f"Expected MTSCConnectorMetadata, got {type(kv_connector_metadata)!r}"
            )
        self.worker.handle_preemptions(kv_connector_metadata)

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        # Store and PD publication are ordered in get_finished().
        return

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        return

    def wait_for_save(self) -> None:
        return

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        assert self.worker is not None
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, MTSCConnectorMetadata):
            raise TypeError(f"Expected MTSCConnectorMetadata, got {type(metadata)!r}")
        return self.worker.get_finished(finished_req_ids, metadata)

    def get_block_ids_with_load_errors(self) -> set[int]:
        assert self.worker is not None
        return self.worker.get_block_ids_with_load_errors()

    def get_kv_connector_kv_cache_events(self):
        return None

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        return None

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> KVConnectorStats | None:
        return None

    def shutdown(self) -> None:
        if self.worker is not None:
            self.worker.close()
            self.worker = None
        if self.scheduler is not None:
            self.scheduler.close()
            self.scheduler = None

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:  # noqa: BLE001
            # Interpreter teardown can invalidate imported modules first.
            return
