"""Shared device, memory registration and KV layout helpers."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Protocol

import torch
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import MambaSpec

BlockIds = tuple[tuple[int, ...], ...]

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
        enabled = additional["enable_kv_nz"]
        if isinstance(enabled, str):
            enabled = enabled.strip().lower() not in {"0", "false", "no", "off"}
        return bool(enabled)
    try:
        from vllm_ascend.ascend_config import get_ascend_config

        return bool(get_ascend_config().enable_kv_nz)
    except (ImportError, AttributeError, RuntimeError):
        return False


def effective_tp(tp_size: int, kv_heads: int, is_mla: bool) -> int:
    if tp_size <= 0 or kv_heads <= 0:
        raise ValueError("Invalid TP size or KV head count")
    heads = 1 if is_mla else kv_heads
    if max(tp_size, heads) % min(tp_size, heads):
        raise ValueError("KV heads and TP size must have an integer ratio")
    return min(tp_size, heads)


def _transpose_npu_cache_blocks(
    cache: torch.Tensor, block_ids: list[int], block_size: int, tp_ratio: int
) -> None:
    """Restore [token, TP-split, ...] after multiple P shards were appended."""
    if tp_ratio <= 1 or not block_ids:
        return
    if cache.ndim < 2 or cache.shape[1] != block_size:
        raise ValueError(
            "Ascend heterogeneous TP requires [blocks, tokens, ...] KV layout"
        )
    ids = torch.tensor(sorted(set(block_ids)), dtype=torch.long, device=cache.device)
    selected = cache.index_select(0, ids)
    if selected[0].numel() % (tp_ratio * block_size):
        raise ValueError("Ascend KV block cannot be evenly split across P TP ranks")
    transposed = (
        selected.reshape(len(ids), tp_ratio, block_size, -1)
        .transpose(1, 2)
        .contiguous()
        .reshape_as(selected)
    )
    cache.index_copy_(0, ids, transposed)


class KVLayoutAdapter:
    def _npu_nz_pair(self, tensors: list[torch.Tensor], block_ids: list[int]) -> None:
        if len(tensors) != 2 or not block_ids:
            raise ValueError("Ascend NZ conversion requires one K/V tensor pair")
        try:
            import torch_npu
        except ImportError as exc:
            raise RuntimeError("Ascend NZ conversion requires torch_npu") from exc
        k_cache, v_cache = tensors
        if k_cache.ndim != 4 or v_cache.ndim != 4:
            raise ValueError("Ascend NZ conversion requires 4D K/V caches")
        ids = torch.tensor(
            sorted(set(block_ids)), dtype=torch.int32, device=k_cache.device
        )
        num_blocks = ids.numel()
        num_tokens = num_blocks * self.topology.block_size
        num_heads = k_cache.shape[2]
        k_head_dim, v_head_dim = k_cache.shape[3], v_cache.shape[3]
        block_table = ids.view(1, -1)
        block_len = torch.tensor([num_tokens], dtype=torch.int32, device=k_cache.device)
        seq_start = torch.tensor([0], dtype=torch.int32, device=k_cache.device)
        k_buffer = torch.empty(
            (num_tokens, num_heads, k_head_dim),
            dtype=k_cache.dtype,
            device=k_cache.device,
        )
        v_buffer = torch.empty(
            (num_tokens, num_heads, v_head_dim),
            dtype=v_cache.dtype,
            device=v_cache.device,
        )
        torch.npu.synchronize()
        torch_npu.npu_gather_pa_kv_cache(
            k_cache,
            v_cache,
            block_table,
            block_len,
            seq_offset=seq_start,
            key=k_buffer,
            value=v_buffer,
        )
        offsets = torch.arange(
            self.topology.block_size, dtype=torch.int32, device=k_cache.device
        )
        slots = (
            offsets.view(1, -1) + ids.view(-1, 1) * self.topology.block_size
        ).flatten()
        nz = 16
        if k_head_dim * num_heads % nz or v_head_dim * num_heads % nz:
            raise ValueError("Ascend NZ dimensions must be divisible by 16")
        k_nz = k_cache.view(
            -1,
            k_head_dim * num_heads // nz,
            self.topology.block_size,
            nz,
        )
        v_nz = v_cache.view(
            -1,
            v_head_dim * num_heads // nz,
            self.topology.block_size,
            nz,
        )
        torch_npu.npu_scatter_pa_kv_cache(k_buffer, v_buffer, k_nz, v_nz, slots)

    @torch.no_grad()
    def reformat_npu_blocks(
        self, block_ids: list[list[int]], remote_tp_size: int
    ) -> None:
        if not is_npu_platform() or not any(block_ids):
            return
        current_platform.set_device(self.device_id)
        remote_shards = effective_tp(
            remote_tp_size,
            self.topology.total_num_kv_heads,
            self.topology.is_mla,
        )
        local_shards = effective_tp(
            self.tp_size,
            self.topology.total_num_kv_heads,
            self.topology.is_mla,
        )
        ratio = max(1, remote_shards // local_shards)
        if remote_tp_size > self.tp_size and remote_tp_size % self.tp_size:
            raise ValueError("P/D TP sizes must have an integer ratio")
        if ratio == 1 and not self.npu_kv_nz:
            return
        try:
            reformatted: set[tuple[int, ...]] = set()
            for layer_name, raw in self.kv_caches.items():
                group = self.layer_groups.get(layer_name)
                spec = self.layer_specs.get(layer_name)
                if (
                    group is None
                    or group >= len(block_ids)
                    or isinstance(spec, MambaSpec)
                ):
                    continue
                group_blocks = [block for block in block_ids[group] if block >= 0]
                if not group_blocks:
                    continue
                tensors = cache_tensors(raw)
                identity = tuple(tensor.data_ptr() for tensor in tensors)
                if identity in reformatted:
                    continue
                reformatted.add(identity)
                if ratio > 1:
                    for tensor in tensors:
                        _transpose_npu_cache_blocks(
                            tensor, group_blocks, self.topology.block_size, ratio
                        )
                if self.npu_kv_nz:
                    self._npu_nz_pair(tensors, group_blocks)
        finally:
            # Fence already-issued device work on both success and failure.
            torch.npu.synchronize()
