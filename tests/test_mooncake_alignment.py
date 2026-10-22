"""Integration checks against the supplied Mooncake worker contract."""

from __future__ import annotations

import asyncio
import ctypes
import inspect
import sys
import threading
import time
from dataclasses import fields
from types import SimpleNamespace

import msgspec
import pytest
import torch
from test_backend_setup import _caches, _config
from test_backends import _Socket, _transfer
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import (
    mooncake_connector as native,
)

from mtsc.kv_transfer import (
    MooncakeKVTransfer,
    PullReqMeta,
    RecvEvent,
    SendBlockMeta,
    SendEvent,
)
from mtsc.protocol import KVTransferRequest, KVTransferResponse, KVTransferStatus


class CopyEngine:
    def initialize(self, *args):
        return 0

    def get_rpc_port(self):
        return 123

    def register_memory(self, pointer, length):
        return 0

    def unregister_memory(self, pointer):
        return 0

    def batch_transfer_sync_write(self, session, sources, destinations, lengths):
        for source, destination, length in zip(
            sources, destinations, lengths, strict=True
        ):
            ctypes.memmove(destination, source, length)
        return 0


@pytest.fixture
def make_transfer(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "mooncake.engine", SimpleNamespace(TransferEngine=CopyEngine)
    )
    monkeypatch.setattr("mtsc.kv_transfer.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        "mtsc.kv_transfer.get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(
        "mtsc.kv_transfer.get_pp_group", lambda: SimpleNamespace(rank_in_group=0)
    )
    monkeypatch.setattr("mtsc.kv_transfer.is_npu_platform", lambda: True)
    monkeypatch.setattr("mtsc.kv_transfer.get_current_attn_backends", lambda config: [])
    monkeypatch.setattr(
        "mtsc.kv_transfer.select_common_block_size", lambda size, backends: size
    )
    monkeypatch.setattr("mtsc.kv_transfer.get_ip", lambda: "127.0.0.1")
    monkeypatch.setattr("mtsc.kv_transfer._launch_bootstrap", lambda config: False)
    monkeypatch.setattr("torch.accelerator.current_device_index", lambda: 0)
    monkeypatch.setattr(
        "mtsc.kv_transfer.current_platform",
        SimpleNamespace(set_device=lambda device: None),
    )
    instances = []

    def make(role="kv_both", engine_id="engine"):
        config = _config()
        config.parallel_config.pipeline_parallel_size = 1
        config.kv_transfer_config.kv_role = role
        config.kv_transfer_config.engine_id = engine_id
        transfer = MooncakeKVTransfer(config, _caches())
        instances.append(transfer)

        async def register(port):
            transfer.side_port = port

        transfer._register_worker = register
        return transfer

    yield make
    for instance in reversed(instances):
        instance.close()


def test_schemas_and_request_metadata_match_actual_native_classes():
    for ours, theirs in (
        (KVTransferRequest, native.MooncakeXferMetadata),
        (KVTransferResponse, native.MooncakeXferResponse),
    ):
        assert [
            (f.name, f.type if f.name != "status" else "status")
            for f in msgspec.structs.fields(ours)
        ] == [
            (f.name, f.type if f.name != "status" else "status")
            for f in msgspec.structs.fields(theirs)
        ]
    assert [f.name for f in fields(SendBlockMeta)] == [
        f.name for f in fields(native.SendBlockMeta)
    ]
    assert [f.name for f in fields(PullReqMeta)] == [
        f.name for f in fields(native.PullReqMeta)
    ]
    assert [(s.name, int(s)) for s in KVTransferStatus] == [
        (s.name, int(s)) for s in native.MooncakeXferResponseStatus
    ]


def test_kernel_block_size_is_selected_before_topology(make_transfer, monkeypatch):
    seen = []

    def select(size, backends):
        seen.append((size, backends))
        return 8

    monkeypatch.setattr("mtsc.kv_transfer.select_common_block_size", select)
    transfer = make_transfer("kv_consumer")
    assert seen == [(16, [])]
    assert transfer.block_size == transfer.transfer_topo.block_size == 8


@pytest.mark.parametrize("p_size,d_size", [(1, 1), (2, 4), (4, 2)])
@pytest.mark.parametrize("ret", [0, -1])
@pytest.mark.parametrize("invalid_blocks", [False, True])
async def test_sender_lifecycle_matches_native(p_size, d_size, ret, invalid_blocks):
    """Compare real sender functions, descriptors and source release timing."""
    from mtsc.kv_transfer import _NPUTransferTopology

    ours = _transfer()
    ours.tp_size = p_size
    ours.transfer_topo = _NPUTransferTopology(0, p_size, 16, False, 8)
    ours.sender_loop = asyncio.get_running_loop()
    theirs = object.__new__(native.MooncakeConnectorWorker)
    # This test constructs only sender state, without the native runtime.
    theirs.shutdown = lambda: None
    for name in (
        "tp_rank",
        "tp_size",
        "transfer_topo",
        "sender_loop",
        "kv_caches_base_addr",
        "block_len_per_layer",
        "_encoder",
    ):
        setattr(theirs, name, getattr(ours, name))
    theirs._sender_executor = ours._sender_executor = None
    theirs.reqs_need_send = {}
    theirs.finished_sending_reqs = set()
    theirs.xfer_stats = native.MooncakeKVConnectorStats()
    (await ours._publish_send(SendEvent("p", "x", ((1,),))))
    theirs.reqs_need_send["x"] = native.SendBlockMeta(
        "p", "x", [[1]], asyncio.Event(), expire_time=float("inf")
    )
    theirs.reqs_need_send["x"].ready.set()
    sources = [ours.reqs_need_send["x"], theirs.reqs_need_send["x"]]
    calls = [[], []]
    for worker, recorded in zip((ours, theirs), calls):
        worker.engine = SimpleNamespace(
            batch_transfer_sync_write=lambda *args, recorded=recorded: (
                recorded.append(args) or ret
            )
        )

    for rank in ours.transfer_topo.handshake_target_ranks(d_size):
        request = KVTransferRequest(
            "decode",
            123,
            d_size,
            rank,
            {"d": ("x", [[3, 4] if invalid_blocks else [4]])},
            [2000],
            [16 * p_size // d_size],
        )
        sockets = [_Socket(), _Socket()]
        await ours.send_kv_to_decode(b"d", msgspec.msgpack.encode(request), sockets[0])
        await theirs.send_kv_to_decode(
            b"d",
            sockets[1],
            msgspec.msgpack.decode(
                msgspec.msgpack.encode(request), type=native.MooncakeXferMetadata
            ),
        )
        for field in ("status", "ok_reqs", "err_reqs"):
            assert getattr(sockets[0].messages[-1], field) == getattr(
                sockets[1].messages[-1], field
            )
        assert calls[0] == calls[1]
        assert (sources[0].need_send, sources[0].sent, sources[0].sending) == (
            sources[1].need_send,
            sources[1].sent,
            sources[1].sending,
        )
        assert set(ours.reqs_need_send) == set(theirs.reqs_need_send)
        assert ours.finished_sending_reqs == theirs.finished_sending_reqs

    if ret != 0 or invalid_blocks:
        assert (await ours.fetch_finished_sending_reqs()) == []
        for source in sources:
            source.expire_time = 0
        assert (
            {r.request_id for r in (await ours.fetch_finished_sending_reqs())}
            == (await theirs.fetch_finished_sending_reqs())
            == {"p"}
        )
        assert ours.reqs_need_send == theirs.reqs_need_send == {}


def test_sender_state_runs_on_sender_loop(make_transfer):
    transfer = make_transfer("kv_producer")
    caller = threading.get_ident()
    seen = []
    original = transfer._prepare_send

    async def prepare(*args):
        seen.append(threading.get_ident())
        return await original(*args)

    transfer._prepare_send = prepare
    transfer.prepare("request", "attempt")
    assert seen == [transfer._sender_listener_t.ident]
    assert seen[0] != caller
    transfer.cancel("request", "attempt")


def test_receiver_state_runs_on_receiver_loop(make_transfer):
    transfer = make_transfer("kv_consumer")
    transfer.register({"layer.0": torch.zeros((6, 16), dtype=torch.int32)})
    caller = threading.get_ident()
    seen = []

    def record_callback(name):
        callback = getattr(transfer, name)

        if inspect.iscoroutinefunction(callback):

            async def run(*args):
                seen.append((name, threading.get_ident()))
                return await callback(*args)
        else:

            def run(*args):
                seen.append((name, threading.get_ident()))
                return callback(*args)

        setattr(transfer, name, run)

    for name in (
        "_start_receive",
        "_finish_receive",
        "fetch_finished_recving_reqs",
        "_take_receive_errors",
    ):
        record_callback(name)

    wait_for_receives = transfer._wait_for_receives

    async def fence(*args):
        seen.append(("_wait_for_receives", threading.get_ident()))
        await wait_for_receives(*args)

    transfer._wait_for_receives = fence
    release = asyncio.Event()

    async def receive(worker_addr, metas):
        await release.wait()
        ok = [rid for rid in metas if rid != "d-3"]
        failed = [rid for rid in metas if rid == "d-3"]
        transfer.process_pulling_result(
            KVTransferResponse(KVTransferStatus.FINISH, ok, failed),
            metas,
            set(metas),
        )

    transfer._remote_agents["p"] = {0: {0: "producer"}}
    transfer._remote_tp_size["p"] = 1
    transfer.receive_kv_from_single_worker = receive
    transfer.recv(
        [
            RecvEvent(
                f"d-{index}",
                f"x-{index}",
                ((index % 6,),),
                "bootstrap",
                "p",
                0,
            )
            for index in range(16)
        ]
    )
    assert transfer.poll().recvs == []
    transfer.receiver_loop.call_soon_threadsafe(
        transfer.receiver_loop.call_later, 0.01, release.set
    )
    transfer.preempt("d-0")
    results = {}
    deadline = time.monotonic() + 3
    while len(results) < 15 and time.monotonic() < deadline:
        for result in transfer.poll().recvs:
            assert result.request_id not in results
            results[result.request_id] = result
        time.sleep(0.001)
    assert set(results) == {f"d-{index}" for index in range(1, 16)}
    assert {rid for rid, result in results.items() if result.error} == {"d-3"}
    assert transfer.take_errors() == {3}
    assert transfer.take_errors() == set()
    assert {name for name, _ in seen} == {
        "_start_receive",
        "_finish_receive",
        "fetch_finished_recving_reqs",
        "_take_receive_errors",
        "_wait_for_receives",
    }
    assert {thread_id for _, thread_id in seen} == {transfer._receiver_t.ident}
    assert transfer._receiver_t.ident != caller


def test_kv_both_transfers_both_directions(make_transfer):
    left, right = make_transfer(engine_id="left"), make_transfer(engine_id="right")
    left_cache = torch.arange(6 * 16, dtype=torch.int32).reshape(6, 16)
    right_cache = left_cache.clone() + 1000
    left_expected, right_expected = right_cache[2].clone(), left_cache[1].clone()
    left.register({"layer.0": left_cache})
    right.register({"layer.0": right_cache})

    left._remote_agents["right"] = {0: {0: f"tcp://127.0.0.1:{right.side_port}"}}
    left._remote_tp_size["right"] = 1
    right._remote_agents["left"] = {0: {0: f"tcp://127.0.0.1:{left.side_port}"}}
    right._remote_tp_size["left"] = 1
    left.send(SendEvent("left-send", "left-attempt", ((1,),)))
    right.send(SendEvent("right-send", "right-attempt", ((2,),)))
    left.recv([RecvEvent("left-recv", "right-attempt", ((4,),), "right", "right", 0)])
    right.recv([RecvEvent("right-recv", "left-attempt", ((5,),), "left", "left", 0)])
    completed = set()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and len(completed) < 4:
        for transfer in (left, right):
            results = transfer.poll()
            for result in results.sends + results.recvs:
                assert result.error is None
                completed.add(result.request_id)
        time.sleep(0.001)
    assert completed == {"left-send", "right-send", "left-recv", "right-recv"}
    assert torch.equal(left_cache[4], left_expected)
    assert torch.equal(right_cache[5], right_expected)


@pytest.mark.parametrize(
    "worker_cls", [MooncakeKVTransfer, native.MooncakeConnectorWorker]
)
async def test_discovery_and_cached_dispatch_match_native(worker_cls, monkeypatch):
    from unittest.mock import AsyncMock, Mock

    worker = object.__new__(worker_cls)
    worker.shutdown = lambda: None
    worker._remote_agents = {}
    tp_size_attr = "_remote_tp_size" if worker_cls is MooncakeKVTransfer else "_tp_size"
    setattr(worker, tp_size_attr, {})
    worker._pending_bootstrap_queries = {}
    worker.receive_kv = Mock()
    entered, release = asyncio.Event(), asyncio.Event()
    response = Mock()
    response.json.return_value = {
        str(rank): {
            "engine_id": f"p{rank}",
            "worker_addr": {"0": {"0": f"tcp://p{rank}"}},
        }
        for rank in (0, 1)
    }

    async def get(url):
        assert url == "http://bootstrap/query"
        entered.set()
        await release.wait()
        return response

    client = AsyncMock()
    client.get.side_effect = get
    context = AsyncMock()
    context.__aenter__.return_value = client
    monkeypatch.setattr("httpx.AsyncClient", Mock(return_value=context))
    metas = {"d0": PullReqMeta("d0", "x0", [[0]], "p0", "http://bootstrap")}
    await worker._start_load_kv({"p0": metas})
    await asyncio.wait_for(entered.wait(), 1)
    worker.receive_kv.assert_not_called()
    # A second engine on the same bootstrap shares the outstanding query.
    other_metas = {"d1": PullReqMeta("d1", "x1", [[1]], "p1", "http://bootstrap")}
    other = asyncio.create_task(worker.handle_new_engine_id("p1", other_metas))
    await asyncio.sleep(0)
    client.get.assert_awaited_once()
    release.set()
    await asyncio.wait_for(other, 1)
    assert worker._remote_agents == {
        "p0": {0: {0: "tcp://p0"}},
        "p1": {0: {0: "tcp://p1"}},
    }
    assert getattr(worker, tp_size_attr) == {"p0": 1, "p1": 1}
    assert worker._pending_bootstrap_queries == {}
    assert worker.receive_kv.call_count == 2
    await worker._start_load_kv({"p1": other_metas})
    worker.receive_kv.assert_called_with("p1", other_metas)
    client.get.assert_awaited_once()


@pytest.mark.parametrize(
    "worker_cls", [MooncakeKVTransfer, native.MooncakeConnectorWorker]
)
async def test_receive_dispatches_cached_tp_workers_like_native(worker_cls):
    from unittest.mock import AsyncMock, Mock

    worker = object.__new__(worker_cls)
    worker.shutdown = lambda: None
    worker._remote_agents = {"p": {0: {0: "p0"}, 1: {0: "p1"}}}
    tp_size_attr = "_remote_tp_size" if worker_cls is MooncakeKVTransfer else "_tp_size"
    setattr(worker, tp_size_attr, {"p": 2})
    worker.transfer_topo = SimpleNamespace(
        handshake_target_ranks=Mock(return_value=[0, 1])
    )
    worker.receive_kv_from_single_worker = AsyncMock()
    metas = {"d": PullReqMeta("d", "x", [[0]], "p", "http://bootstrap")}
    assert not inspect.iscoroutinefunction(worker.receive_kv)
    worker.receive_kv("p", metas)
    assert metas["d"].pull_tasks_count == 2
    await asyncio.sleep(0)
    worker.transfer_topo.handshake_target_ranks.assert_called_once_with(2)
    assert [
        call.args for call in worker.receive_kv_from_single_worker.await_args_list
    ] == [("p0", metas), ("p1", metas)]
