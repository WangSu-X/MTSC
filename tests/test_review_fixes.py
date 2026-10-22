"""Regression coverage for startup, Store GET and upstream PD layouts."""

from __future__ import annotations

import ctypes
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import msgspec
import pytest
import torch
from test_backend_setup import _caches, _config
from test_backends import _pool, _transfer
from vllm.distributed.kv_transfer.kv_connector.utils import TransferTopology
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

from mtsc.connector import MTSCConnector
from mtsc.kv_cache_pool import LoadEvent
from mtsc.kv_transfer import (
    MooncakeKVTransfer,
    SendBlockMeta,
    _NPUTransferTopology,
    _resolve_transfer_transport,
)
from mtsc.protocol import KVTransferRequest


def _registered(caches, *, tp=1, rank=0, heads=4, npu=False, nz=False, topology=None):
    transfer = _transfer()
    transfer._registered = False
    transfer.is_producer = False
    transfer.npu_kv_nz = nz
    transfer._registered_storage = []
    transfer.tp_size, transfer.tp_rank = tp, rank
    transfer.transfer_topo = topology or _NPUTransferTopology(rank, tp, 3, False, heads)
    if topology is None:
        transfer.transfer_topo.get_transfer_cache_regions = lambda raw, spec: [raw]
    transfer.layer_specs = {name: SimpleNamespace(block_size=3) for name in caches}
    transfer.engine = Mock()
    transfer.engine.register_memory.return_value = 0
    transfer.engine.batch_register_memory.return_value = 0
    with patch("mtsc.kv_transfer.is_npu_platform", return_value=npu):
        transfer.register(caches)
    return transfer


def _request(receiver, blocks, *, layout_id=0):
    return KVTransferRequest(
        "d",
        123,
        receiver.tp_size,
        receiver.tp_rank,
        {"d-r": ("x", blocks)},
        receiver.kv_caches_base_addr,
        receiver.block_len_per_layer,
    )


def _copy(sender, receiver, source_blocks, destination_blocks):
    request = _request(receiver, destination_blocks)
    src, dst, sizes = sender._build_request_transfer_params(
        "d-r",
        SendBlockMeta("p-r", "x", local_block_ids=source_blocks),
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
    "p_size,d_size", [(1, 1), (2, 1), (1, 2), (4, 2), (2, 4), (8, 1), (1, 8)]
)
def test_npu_hnd_head_resharding_matches_native_whole_block_plan(p_size, d_size):
    from unittest.mock import patch

    heads, tokens = 8, 3
    full = torch.arange(heads * tokens * 2, dtype=torch.int32).reshape(heads, tokens, 2)
    for d_rank in range(d_size):
        target = torch.full((6, heads // d_size, tokens, 2), -1, dtype=torch.int32)
        receiver = _registered(
            {"layer.0": target}, tp=d_size, rank=d_rank, heads=heads, npu=True
        )
        for p_rank in receiver.transfer_topo.handshake_target_ranks(p_size):
            source = torch.zeros((6, heads // p_size, tokens, 2), dtype=torch.int32)
            source[1] = full[p_rank * heads // p_size : (p_rank + 1) * heads // p_size]
            sender = _registered(
                {"layer.0": source}, tp=p_size, rank=p_rank, heads=heads, npu=True
            )
            with (
                patch("mtsc.kv_transfer.get_kv_cache_layout", return_value="HND"),
                patch("mtsc.kv_transfer.is_npu_platform", return_value=True),
            ):
                _copy(sender, receiver, ((1,),), [[4]])
        assert torch.equal(
            target[4], full[d_rank * heads // d_size : (d_rank + 1) * heads // d_size]
        )
        assert torch.all(target[[0, 1, 2, 3, 5]] == -1)


def test_npu_nhd_heterogeneous_tp_is_rejected_without_token_regions():
    source = torch.zeros((6, 3, 2, 2), dtype=torch.int32)
    target = torch.full((6, 3, 4, 2), -1, dtype=torch.int32)
    sender = _registered({"layer.0": source}, tp=2, npu=True)
    receiver = _registered({"layer.0": target}, npu=True)
    with (
        patch("mtsc.kv_transfer.is_npu_platform", return_value=True),
        patch("mtsc.kv_transfer.get_kv_cache_layout", return_value="NHD"),
    ):
        with pytest.raises(ValueError, match="requires HND"):
            _copy(sender, receiver, ((1,),), [[4]])
    assert torch.all(target == -1)


@pytest.mark.parametrize("shared", [False, True])
def test_all_regions_use_the_flattened_group_block_table(shared):
    source = torch.arange(6 * 4, dtype=torch.int32).reshape(6, 4)
    target = torch.full_like(source, -1)
    sources = {"layer.0": source, "layer.1": source if shared else source + 1000}
    targets = {
        "layer.0": target,
        "layer.1": target if shared else torch.full_like(target, -1),
    }
    sender, receiver = (_registered(values) for values in (sources, targets))
    _copy(sender, receiver, ((0, 1), (2,)), [[3], [4]])
    for layer in sources:
        assert torch.equal(targets[layer][3], sources[layer][1])
        assert torch.equal(targets[layer][4], sources[layer][2])
        assert torch.all(targets[layer][[0, 1, 2, 5]] == -1)
    assert len(sender.kv_caches_base_addr) == (1 if shared else 2)


@pytest.mark.asyncio
async def test_native_metadata_contains_no_layout_extensions():
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
        MooncakeXferMetadata as NativeMeta,
    )

    cache = torch.zeros((6, 4), dtype=torch.int32)
    receiver = _registered({"layer.0": cache})
    request = _request(receiver, [[4]])
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(request), type=NativeMeta)
    assert decoded.req_blocks == request.req_blocks
    assert set(msgspec.to_builtins(request)) == set(msgspec.to_builtins(decoded))


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
@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer", "kv_both"])
def test_direct_te_transport_precedence(
    extra, store_json, expected, role, tmp_path, monkeypatch
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
    config.kv_transfer_config.kv_role = role
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
        patch("mtsc.kv_transfer.get_current_attn_backends", return_value=[]),
        patch(
            "mtsc.kv_transfer.select_common_block_size",
            side_effect=lambda size, backends: size,
        ),
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
            if role == "kv_producer":
                loop, thread = transfer.sender_loop, transfer._sender_listener_t
                assert transfer.receiver_loop is None
                assert transfer._receiver_t is None
                assert transfer._sender_executor is not None
            elif role == "kv_consumer":
                loop, thread = transfer.receiver_loop, transfer._receiver_t
                assert transfer.sender_loop is None
                assert transfer._sender_listener_t is None
                assert transfer._sender_executor is None
                assert not hasattr(transfer, "sender_worker_queue")
            else:
                loop, thread = transfer.receiver_loop, transfer._receiver_t
                assert transfer.sender_loop.is_running()
                assert transfer._sender_listener_t.is_alive()
                assert transfer._sender_executor is not None
            assert loop.is_running()
            assert thread.is_alive()
        finally:
            transfer.close()
        assert loop.is_closed()
        assert not thread.is_alive()
        if role == "kv_both":
            assert transfer.sender_loop.is_closed()
            assert not transfer._sender_listener_t.is_alive()


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
    assert len(sender.kv_caches_base_addr) == 2
    assert torch.equal(target[:, 4], source[:, 1])
    assert torch.all(target[:, [0, 1, 2, 3, 5]] == -1)
