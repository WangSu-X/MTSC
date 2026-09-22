"""Small accelerator compatibility helpers used by MTSC data paths."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Protocol

import torch
from vllm.platforms import current_platform

_NPU_REGISTER_MERGE_GAP_BYTES = 4096
_MAX_NPU_REGISTER_REGIONS = 256


class DeviceEvent(Protocol):
    def record(self) -> None: ...

    def synchronize(self) -> None: ...


def is_npu_platform() -> bool:
    return current_platform.device_type == "npu"


def new_device_event() -> DeviceEvent:
    if is_npu_platform():
        npu = getattr(torch, "npu", None)
        if npu is None or not hasattr(npu, "Event"):
            raise RuntimeError("Ascend platform requires torch.npu.Event")
        return npu.Event()
    return torch.cuda.Event()


def cache_tensors(raw: Any) -> list[torch.Tensor]:
    values = list(raw) if isinstance(raw, (list, tuple)) else [raw]
    if not values or not all(isinstance(value, torch.Tensor) for value in values):
        raise TypeError("KV cache must be a tensor or a non-empty tensor sequence")
    return values


def npu_registration_regions(
    kv_caches: dict[str, Any],
) -> tuple[list[int], list[int]]:
    """Collect aligned logical tensor ranges, merging shared-storage views."""
    ranges: OrderedDict[int, list[tuple[int, int]]] = OrderedDict()
    for raw in kv_caches.values():
        for tensor in cache_tensors(raw):
            if tensor.numel() == 0:
                continue
            storage_key = tensor.untyped_storage().data_ptr()
            start = tensor.data_ptr()
            ranges.setdefault(storage_key, []).append((start, start + tensor.nbytes))

    pointers: list[int] = []
    lengths: list[int] = []
    for storage_ranges in ranges.values():
        storage_ranges.sort()
        start, end = storage_ranges[0]
        for next_start, next_end in storage_ranges[1:]:
            # Mirror vLLM-Ascend's registration coalescing. HCCL limits the
            # number of registered regions, and a small unused gap inside a
            # shared allocation is safe to include.
            if next_start <= end + _NPU_REGISTER_MERGE_GAP_BYTES:
                end = max(end, next_end)
            else:
                pointers.append(start)
                lengths.append(end - start)
                start, end = next_start, next_end
        pointers.append(start)
        lengths.append(end - start)
    if len(pointers) > _MAX_NPU_REGISTER_REGIONS:
        raise RuntimeError(
            "Mooncake register_buffer region count "
            f"{len(pointers)} exceeds the HCCL per-process limit "
            f"{_MAX_NPU_REGISTER_REGIONS}"
        )
    return pointers, lengths


def npu_kv_nz_enabled(config: Any) -> bool:
    if not is_npu_platform():
        return False
    additional = getattr(config, "additional_config", None)
    if isinstance(additional, dict) and "enable_kv_nz" in additional:
        return bool(additional["enable_kv_nz"])
    try:
        from vllm_ascend.ascend_config import get_ascend_config

        return bool(get_ascend_config().enable_kv_nz)
    except (ImportError, AttributeError, RuntimeError):
        return False
