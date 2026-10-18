"""Regression coverage for startup, Store GET and upstream PD layouts."""

from __future__ import annotations

import ctypes
import json
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import msgspec
import pytest
import torch
from reference_mooncake import MooncakeXferMetadata, MooncakeXferResponse
from test_backend_setup import _caches, _config
from test_backends import _pool, _Socket, _transfer
from vllm.distributed.kv_transfer.kv_connector.utils import TransferTopology
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

from mtsc.connector import MTSCConnector
from mtsc.kv_cache_pool import LoadEvent
from mtsc.kv_transfer import (
    MooncakeKVTransfer,
    RecvEvent,
    SendEvent,
    _NPUTransferTopology,
    _resolve_transfer_transport,
    _Session,
)
from mtsc.protocol import KVTransferRequest, KVTransferResponse, KVTransferStatus


def _registered(
    caches, *, tp=1, rank=0, heads=4, npu=False, groups=None, nz=False, topology=None
):
    transfer = _transfer()
    transfer._registered = False
    transfer.is_producer = False
    transfer.npu_kv_nz = nz
    transfer._registered_storage = []
    transfer.tp_size, transfer.tp_rank = tp, rank
    transfer.topology = topology or _NPUTransferTopology(rank, tp, 3, False, heads)
    if topology is None:
        transfer.topology.get_transfer_cache_regions = lambda raw, spec: [raw]
    transfer.layer_specs = {name: SimpleNamespace(block_size=3) for name in caches}
    transfer.layer_groups = groups or dict.fromkeys(caches, 0)
    transfer.engine = Mock()
    transfer.engine.register_memory.return_value = 0
    transfer.engine.batch_register_memory.return_value = 0
    with patch("mtsc.kv_transfer.is_npu_platform", return_value=npu):
        transfer.register(caches)
    return transfer


def _request(receiver, blocks, *, layout_id=0):
    regions = receiver.regions
    return KVTransferRequest(
        "d",
        123,
        receiver.tp_size,
        receiver.tp_rank,
        {"d-r": ("x", blocks)},
        [r.base_addr for r in regions],
        [r.block_len for r in regions],
        kv_block_lens=[r.kv_block_len for r in regions],
        registered_layer_names=[r.layer_name for r in regions],
        registered_layer_indices=[r.layer_index for r in regions],
        registered_group_indices=[r.group_index for r in regions],
        registered_shared_group_ids=[list(r.shared_group_ids) for r in regions],
        registered_row_offsets=[r.row_offset for r in regions],
        layout_id=layout_id,
    )


def _copy(sender, receiver, source_blocks, destination_blocks):
    request = _request(receiver, destination_blocks)
    src, dst, sizes = sender._build_request_transfer_params(
        "d-r",
        _Session("p-r", "x", block_ids=source_blocks),
        request,
    )
    for source, target, size in zip(src, dst, sizes, strict=True):
        ctypes.memmove(target, source, size)


@pytest.mark.parametrize(
    "decode_save,expected", [(None, True), ("false", False), (True, True)]
)
def test_scheduler_connector_initializes_with_decode_save_setting(
    decode_save, expected
):
    config = _config()
    config.kv_transfer_config.kv_load_failure_policy = "recompute"
    if decode_save is not None:
        config.kv_transfer_config.kv_connector_extra_config["mtsc_decode_save"] = (
            decode_save
        )
    with patch("mtsc.connector.MTSCScheduler") as scheduler:
        connector = MTSCConnector(config, KVConnectorRole.SCHEDULER, _caches())
        scheduler.assert_called_once_with(config, _caches(), expected)
        connector.shutdown()


def test_store_success_is_usable_without_a_layout_conversion_method():
    pool = _pool()
    try:
        event = LoadEvent("r", ((10, 11),), (b"a", b"b"), 0, 32)
        pool.load(event)
        pool._loads["r"].future.result(timeout=1)
        result = pool.poll().loads[0]
        assert result.loaded_tokens == 32
        assert result.error is None
        assert pool.take_errors() == set()
    finally:
        pool.close()


@pytest.mark.parametrize(
    "p_size,d_size",
    [(1, 1), (2, 1), (1, 2), (4, 2), (2, 4), (8, 1), (1, 8), (8, 2), (2, 8)],
)
@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_npu_head_resharding_preserves_token_order(p_size, d_size, layout):
    heads, tokens = 4, 3
    full = torch.arange(tokens * heads * 2, dtype=torch.int32).reshape(tokens, heads, 2)
    for d_rank in range(d_size):
        d_heads = max(heads // d_size, 1)
        d_start = d_rank * heads // d_size
        target = torch.full((6, tokens, d_heads, 2), -1, dtype=torch.int32)
        if layout == "HND":
            target = torch.full(
                (6, d_heads, tokens, 2), -1, dtype=torch.int32
            ).transpose(1, 2)
        receiver = _registered({"layer.0": target}, tp=d_size, rank=d_rank, npu=True)
        peers = receiver.topology.handshake_target_ranks(p_size)
        for p_rank in peers:
            p_heads = max(heads // p_size, 1)
            p_start = p_rank * heads // p_size
            source = torch.zeros((6, tokens, p_heads, 2), dtype=torch.int32)
            if layout == "HND":
                source = torch.zeros(
                    (6, p_heads, tokens, 2), dtype=torch.int32
                ).transpose(1, 2)
            source[1] = full[:, p_start : p_start + p_heads]
            sender = _registered({"layer.0": source}, tp=p_size, rank=p_rank, npu=True)
            _copy(sender, receiver, ((1,),), [[4]])
        assert torch.equal(target[4], full[:, d_start : d_start + d_heads])
        assert torch.all(target[:4] == -1)
        assert torch.all(target[5] == -1)


def test_npu_separate_kv_tensors_preserve_both_payloads():
    full = [
        torch.arange(3 * 4 * dim, dtype=torch.int32).reshape(3, 4, dim) + offset
        for dim, offset in ((2, 0), (3, 1000))
    ]
    target = [torch.full((6, 3, 4, dim), -1, dtype=torch.int32) for dim in (2, 3)]
    receiver = _registered({"layer.0": target}, npu=True)
    for rank in range(2):
        source = [torch.zeros((6, 3, 2, dim), dtype=torch.int32) for dim in (2, 3)]
        for cache, expected in zip(source, full, strict=True):
            cache[1] = expected[:, rank * 2 : rank * 2 + 2]
        sender = _registered({"layer.0": source}, tp=2, rank=rank, npu=True)
        _copy(sender, receiver, ((1,),), [[4]])
    for cache, expected in zip(target, full, strict=True):
        assert torch.equal(cache[4], expected)
        assert torch.all(cache[[0, 1, 2, 3, 5]] == -1)


def test_mamba_null_placeholders_do_not_shift_the_source_suffix():
    source = {
        name: torch.arange(6 * 4, dtype=torch.int32).reshape(6, 4) + offset
        for name, offset in (("layer.0", 0), ("layer.1", 1000))
    }
    target = {name: torch.full_like(cache, -1) for name, cache in source.items()}
    groups = {"layer.0": 0, "layer.1": 1}
    sender, receiver = (
        _registered(caches, groups=groups) for caches in (source, target)
    )
    sender.mamba_groups = {1}
    _copy(sender, receiver, ((1,), (0, 2, 0)), [[3], [0, 4]])
    assert torch.equal(target["layer.1"][4], source["layer.1"][2])
    assert torch.all(target["layer.1"][[0, 1, 2, 3, 5]] == -1)


def test_mismatched_group_metadata_is_rejected_before_write():
    cache = torch.zeros((6, 4), dtype=torch.int32)
    sender = _registered({"layer.0": cache})
    receiver = _registered({"layer.0": torch.full_like(cache, -1)})
    request = msgspec.structs.replace(
        _request(receiver, [[4]]), registered_group_indices=[1]
    )
    with pytest.raises(ValueError, match="group index mismatch"):
        sender._build_request_transfer_params(
            "d-r", _Session("p-r", "x", block_ids=((1,),)), request
        )
    sender.engine.batch_transfer_sync_write.assert_not_called()


@pytest.mark.parametrize("shared", [False, True])
def test_each_region_uses_only_its_own_kv_groups(shared):
    source = torch.arange(6 * 4, dtype=torch.int32).reshape(6, 4)
    target = torch.full_like(source, -1)
    sources = {"layer.0": source, "layer.1": source if shared else source + 1000}
    targets = {
        "layer.0": target,
        "layer.1": target if shared else torch.full_like(target, -1),
    }
    groups = {"layer.0": 0, "layer.1": 1}
    sender, receiver = (
        _registered(values, groups=groups) for values in (sources, targets)
    )
    _copy(sender, receiver, ((1,), (2,)), [[3], [4]])
    for layer, src_id, dst_id in (("layer.0", 1, 3), ("layer.1", 2, 4)):
        assert torch.equal(targets[layer][dst_id], sources[layer][src_id])
        untouched = [0, 1, 2, 5] if shared else [i for i in range(6) if i != dst_id]
        assert torch.all(targets[layer][untouched] == -1)
    assert len(sender.regions) == (1 if shared else 2)


def test_packed_layer_payloads_do_not_copy_the_next_layer_or_the_next_row():
    source = torch.arange(6 * 2 * 4, dtype=torch.int32).reshape(6, 2, 4)
    target = torch.full_like(source, -1)
    sources = {"layer.0": source[:, 0], "layer.1": source[:, 1]}
    targets = {"layer.1": target[:, 1], "layer.0": target[:, 0]}
    groups = {"layer.0": 0, "layer.1": 1}
    sender, receiver = (
        _registered(values, groups=groups) for values in (sources, targets)
    )
    _copy(sender, receiver, ((1,), (2,)), [[3], [4]])
    assert torch.equal(target[3, 0], source[1, 0])
    assert torch.equal(target[4, 1], source[2, 1])
    assert torch.all(target[3, 1] == -1)
    assert torch.all(target[4, 0] == -1)
    assert torch.all(target[[0, 1, 2, 5]] == -1)


def test_unsupported_npu_nz_resharding_is_rejected_before_memory_access():
    source = torch.zeros((6, 3, 2, 2), dtype=torch.int32)
    target = torch.full((6, 3, 4, 2), -1, dtype=torch.int32)
    sender = _registered({"layer.0": source}, tp=2, npu=True, nz=True)
    receiver = _registered({"layer.0": target}, npu=True, nz=True)
    with pytest.raises(ValueError, match="NZ KV-head re-sharding"):
        _copy(sender, receiver, ((1,),), [[4]])
    assert torch.all(target == -1)


@pytest.mark.asyncio
async def test_upstream_layout_id_cache_and_layer_matching():
    cache = torch.zeros((6, 4), dtype=torch.int32)
    sender = _registered({"layer.0": cache})
    sender.is_producer = True
    sender.send(SendEvent("p-r", "x", ((1,),)))
    receiver = _registered({"layer.0": torch.full_like(cache, -1)})
    request = _request(receiver, [[4]], layout_id=123)
    sender._write_batch = AsyncMock(return_value={"d-r": True})
    socket = _Socket()
    # Decode the actual request with the pinned upstream schema first.
    upstream = msgspec.msgpack.decode(
        msgspec.msgpack.encode(request), type=MooncakeXferMetadata
    )
    await sender._serve(b"peer", msgspec.msgpack.encode(upstream), socket)
    assert socket.messages[0].ok_reqs == ["d-r"]
    assert socket.messages[0].layout_id == 123
    compact = msgspec.structs.replace(
        request,
        kv_caches_base_addr=[],
        block_lens=[],
        kv_block_lens=[],
        registered_layer_names=[],
        registered_layer_indices=[],
    )
    assert sender._prepare_transfer_regions(compact) is not None
    missing = msgspec.structs.replace(compact, layout_id=456)
    socket = _Socket()
    await sender._serve(b"peer", msgspec.msgpack.encode(missing), socket)
    assert socket.messages[0].status == KVTransferStatus.LAYOUT_MISS
    assert socket.messages[0].layout_id == 456


@pytest.mark.asyncio
async def test_generated_decode_request_is_accepted_by_current_upstream_schema():
    receiver = _registered({"layer.0": torch.zeros((6, 4), dtype=torch.int32)})
    receiver.hostname, receiver.rpc_port = "d", 123
    receiver._query_workers = AsyncMock(return_value={0: {0: "p"}})
    receiver._response_decoder = msgspec.msgpack.Decoder(KVTransferResponse)
    captured = []

    class Client:
        def setsockopt(self, *args):
            pass

        def connect(self, address):
            pass

        async def send(self, payload):
            captured.append(msgspec.msgpack.decode(payload, type=MooncakeXferMetadata))

        async def recv(self):
            return msgspec.msgpack.encode(MooncakeXferResponse(0, ["d-r"]))

        def close(self, **kwargs):
            pass

    receiver._ctx = SimpleNamespace(socket=lambda kind: Client())
    future = Future()
    await receiver._receive_batch(
        [RecvEvent("d-r", "x", ((4,),), "bootstrap", "p", 0)], {"d-r": future}
    )
    assert future.result() is None
    assert captured[0].kv_block_lens == [16]
    assert captured[0].registered_layer_names == ["layer.0"]
    assert captured[0].registered_group_indices == [0]


@pytest.mark.parametrize(
    "extra,store_json,expected",
    [
        (
            {"mooncake_protocol": "rdma", "device_name": "mlx5_2"},
            "invalid",
            ("rdma", "mlx5_2"),
        ),
        ({"mooncake_protocol": "tcp", "device_name": ""}, None, ("tcp", "")),
        ({"mooncake_protocol": "ascend", "device_name": ""}, None, ("ascend", "")),
        ({}, {"protocol": "tcp", "device_name": "json-device"}, ("tcp", "json-device")),
        (
            {"mooncake_protocol": "rdma"},
            {"protocol": "tcp", "device_name": "json-device"},
            ("rdma", "json-device"),
        ),
        (
            {"device_name": ""},
            {"protocol": "tcp", "device_name": "json-device"},
            ("tcp", ""),
        ),
        (
            {"device_name": "extra-device"},
            {"protocol": "tcp", "device_name": "json-device"},
            ("tcp", "extra-device"),
        ),
        (
            {"mooncake_protocol": None, "device_name": None},
            {"protocol": "tcp"},
            ("tcp", ""),
        ),
        ({}, {}, ("rdma", "")),
        ({}, None, ("rdma", "")),
        ({"mooncake_protocol": "ascend"}, None, ("ascend", "")),
    ],
)
def test_direct_te_transport_precedence(
    extra, store_json, expected, tmp_path, monkeypatch
):
    monkeypatch.delenv("MOONCAKE_CONFIG_PATH", raising=False)
    if store_json is not None:
        path = tmp_path / "mooncake.json"
        path.write_text(
            json.dumps(store_json) if isinstance(store_json, dict) else store_json
        )
        monkeypatch.setenv("MOONCAKE_CONFIG_PATH", str(path))
    engine = Mock()
    engine.initialize.return_value = 0
    engine.get_rpc_port.return_value = 123
    config = _config()
    config.parallel_config.pipeline_parallel_size = 1
    config.kv_transfer_config.kv_connector_extra_config.update(extra)
    with (
        patch.dict(
            "sys.modules",
            {"mooncake.engine": SimpleNamespace(TransferEngine=lambda: engine)},
        ),
        patch("mtsc.kv_transfer.get_tensor_model_parallel_rank", return_value=0),
        patch("mtsc.kv_transfer.get_tensor_model_parallel_world_size", return_value=1),
        patch(
            "mtsc.kv_transfer.get_pp_group",
            return_value=SimpleNamespace(rank_in_group=0),
        ),
        patch("mtsc.kv_transfer.is_npu_platform", return_value=True),
        patch("mtsc.kv_transfer.current_platform"),
        patch("mtsc.kv_transfer._launch_bootstrap", return_value=False),
        patch("mtsc.kv_transfer.version", return_value="0.3.13.post1"),
        patch("torch.accelerator.current_device_index", return_value=0),
        patch.dict("os.environ", {"MC_TCP_PROTO": "2"}),
    ):
        transfer = MooncakeKVTransfer(config, _caches())
        try:
            assert transfer.protocol == expected[0]
            assert engine.initialize.call_args.args[1:] == ("P2PHANDSHAKE", *expected)
        finally:
            transfer.close()


@pytest.mark.parametrize("content", [None, "invalid JSON"])
def test_invalid_configured_store_json_does_not_silently_select_rdma(
    content, tmp_path, monkeypatch
):
    path = tmp_path / "mooncake.json"
    if content is not None:
        path.write_text(content)
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", str(path))
    with pytest.raises((FileNotFoundError, json.JSONDecodeError)):
        _resolve_transfer_transport({})


@pytest.mark.parametrize(
    "extra", [{"mooncake_protocol": ""}, {"mooncake_protocol": 1}, {"device_name": 1}]
)
def test_invalid_explicit_transport_does_not_fall_back(extra, monkeypatch):
    monkeypatch.delenv("MOONCAKE_CONFIG_PATH", raising=False)
    with pytest.raises(ValueError, match="Mooncake direct TE"):
        _resolve_transfer_transport(extra)


def test_real_vllm_topology_splits_kv_first_cache_before_inferring_blocks():
    class KVFirstBackend:
        @staticmethod
        def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size):
            return (2, num_blocks, block_size, num_kv_heads, head_size)

    def topology():
        return TransferTopology(
            tp_rank=0,
            tp_size=1,
            block_size=3,
            engine_id="p",
            is_mla=False,
            is_mamba=False,
            total_num_kv_heads=4,
            attn_backends=[KVFirstBackend],
        )

    source = torch.arange(2 * 6 * 3 * 4 * 2, dtype=torch.int32).reshape(2, 6, 3, 4, 2)
    target = torch.full_like(source, -1)
    sender = _registered({"layer.0": source}, topology=topology())
    receiver = _registered({"layer.0": target}, topology=topology())
    _copy(sender, receiver, ((1,),), [[4]])
    assert sender.num_blocks == 6
    assert len(sender.regions) == 2
    assert torch.equal(target[:, 4], source[:, 1])
    assert torch.all(target[:, [0, 1, 2, 3, 5]] == -1)
