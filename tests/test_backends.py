from __future__ import annotations

import asyncio
import threading
import time
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import torch
import zmq
import zmq.asyncio

from mtsc.kv_cache_pool import (
    KVCachePool,
    LoadEvent,
    MooncakeKVCachePool,
    PoolPollResult,
    SaveEvent,
    SaveResult,
)
from mtsc.kv_transfer import (
    KVTransfer,
    MooncakeKVTransfer,
    RecvEvent,
    SendEvent,
    TransferPollResult,
    TransferRegion,
    _NPUTransferTopology,
    effective_tp,
    kv_slice_plan,
)
from mtsc.protocol import (
    MTSCConnectorMetadata,
    PDTransferRequest,
    PDTransferResponse,
    PDTransferSchema,
    SendRequirement,
    StoreRequest,
)
from mtsc.worker import MTSCWorker


class _Database:
    block_size = 16

    def process_tokens(self, token_count, hashes, start):
        for index in range((start + 15) // 16, token_count // 16):
            yield index * 16, (index + 1) * 16, hashes[index]

    def key_for(self, value):
        return value.hex()

    def prepare_value(self, start, end, ids):
        block = ids[start // 16]
        return [1000 + block * 16], [16], block


class _Store:
    def __init__(self):
        self.puts = []
        self.gets = []
        self.failed_get = set()
        self.fail_put = False
        self.closed = False

    def batch_is_exist(self, keys):
        return [0] * len(keys)

    def batch_put_from_multi_buffers(self, keys, addresses, sizes, replicate):
        self.puts.append(list(keys))
        return [-1 if self.fail_put else 0] * len(keys)

    def batch_get_into_multi_buffers(self, keys, addresses, sizes):
        self.gets.append(list(keys))
        return [-1 if index in self.failed_get else 0 for index in range(len(keys))]

    def close(self):
        self.closed = True


class _Ready:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def synchronize(self):
        self.entered.set()
        if not self.release.wait(3):
            raise TimeoutError("Test did not release ready signal")


def _pool():
    pool = object.__new__(MooncakeKVCachePool)
    pool._closed = False
    pool._registered = True
    pool._task_lock = threading.RLock()
    pool._save_states = {}
    pool._load_events = {}
    pool._invalid = {}
    pool._loads = {}
    pool._load_requests = {}
    pool._load_started_at = {}
    pool._load_timed_out = set()
    pool._load_pool = ThreadPoolExecutor(max_workers=2)
    pool._save_pool = ThreadPoolExecutor(max_workers=1)
    pool._lookup_server = None
    pool.load_timeout = 30
    pool.databases = [_Database()]
    pool.coordinator = SimpleNamespace(
        lcm_block_size=16,
        load_mask=lambda hashes, count: ([True] * (count // 16),),
    )
    pool._store_masks = lambda *args: (None,)
    pool.store = _Store()
    pool.replicate_config = None
    pool.put_step = 1
    pool.tp_rank = 0
    pool.tp_size = 1
    pool.num_blocks = 100
    pool.cache_namespace = "test"
    pool.kv_caches = {}
    return pool


def _save(start, end, ready=None, request_id="r"):
    return SaveEvent(
        request_id,
        (tuple(range(end // 16)),),
        tuple(bytes([i + 1]) for i in range(end // 16)),
        start,
        end,
        ready,
    )


def _drain(pool, kind):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        values = getattr(pool.poll(), kind)
        if values:
            return values
        time.sleep(0.001)
    raise AssertionError(f"No {kind} completion")


class PoolBackendTest(unittest.TestCase):
    def setUp(self):
        self.pool = _pool()
        self.addCleanup(self.pool.close)

    def test_implements_published_abc(self):
        self.assertIsInstance(self.pool, KVCachePool)
        self.assertTrue(issubclass(MooncakeKVTransfer, KVTransfer))
        self.assertFalse(MooncakeKVTransfer.__abstractmethods__)

    def test_pending_ranges_merge_and_wait_for_all_ready_signals(self):
        first, second, third = _Ready(), _Ready(), _Ready()
        self.addCleanup(first.release.set)
        self.addCleanup(second.release.set)
        self.addCleanup(third.release.set)
        self.pool.save(_save(0, 16, first))
        self.assertTrue(first.entered.wait(1))
        self.pool.save(_save(16, 32, second))
        self.pool.save(_save(32, 48, third))
        pending = self.pool._save_states["r"].pending
        self.assertEqual([(e.start_save, e.end_save) for e in pending], [(16, 48)])
        first.release.set()
        self.pool._save_states["r"].running.result(timeout=1)
        self.assertEqual(self.pool.poll().saves, [])
        self.assertTrue(second.entered.wait(1))
        second.release.set()
        self.assertTrue(third.entered.wait(1))
        self.assertEqual(self.pool.poll().saves, [])
        third.release.set()
        self.assertEqual(_drain(self.pool, "saves"), [SaveResult("r")])
        self.assertEqual([len(keys) for keys in self.pool.store.puts], [1, 2])

    def test_changed_source_mapping_is_not_merged(self):
        first = _Ready()
        self.addCleanup(first.release.set)
        self.pool.save(_save(0, 16, first))
        self.assertTrue(first.entered.wait(1))
        self.pool.save(_save(16, 32))
        self.pool.save(replace(_save(32, 48), block_ids=((0, 99, 2),)))
        self.assertEqual(len(self.pool._save_states["r"].pending), 2)

    def test_new_save_before_poll_extends_completion(self):
        self.pool.save(_save(0, 16))
        self.pool._save_states["r"].running.result(timeout=1)
        ready = _Ready()
        self.addCleanup(ready.release.set)
        self.pool.save(_save(16, 32, ready))
        self.assertTrue(ready.entered.wait(1))
        self.assertEqual(self.pool.poll().saves, [])
        ready.release.set()
        self.assertEqual(_drain(self.pool, "saves"), [SaveResult("r")])

    def test_save_error_survives_later_success(self):
        self.pool.store.fail_put = True
        self.pool.save(_save(0, 16))
        with self.assertRaises(RuntimeError):
            self.pool._save_states["r"].running.result(timeout=1)
        self.pool.store.fail_put = False
        self.pool.save(_save(16, 32))
        result = _drain(self.pool, "saves")[0]
        self.assertIsNotNone(result.error)
        self.assertEqual(self.pool.take_errors(), set())

    def test_save_can_continue_after_intermediate_result(self):
        self.pool.save(_save(0, 16))
        self.assertEqual(_drain(self.pool, "saves"), [SaveResult("r")])
        self.pool.save(_save(16, 32))
        self.assertEqual(_drain(self.pool, "saves"), [SaveResult("r")])
        self.assertEqual(self.pool._save_states, {})

    def test_partial_load_reports_contiguous_prefix_and_invalid_suffix(self):
        self.pool.store.failed_get = {1}
        event = LoadEvent("r", ((10, 11, 12, 13),), (b"a", b"b", b"c", b"d"), 16, 64)
        self.pool.load(event)
        result = _drain(self.pool, "loads")[0]
        self.assertEqual(result.loaded_tokens, 32)
        with self.assertRaises(ValueError):
            self.pool.load(event)
        self.assertEqual(self.pool.take_errors(), {12, 13})
        self.assertEqual(self.pool.take_errors(), set())

    def test_load_timeout_does_not_complete_until_io_stops(self):
        event = LoadEvent("r", ((10, 11),), (b"a", b"b"), 0, 32)
        self.pool.load(event)
        self.pool._loads["r"].result(timeout=1)
        future = Future()
        self.pool._loads["r"] = future
        self.pool._load_started_at["r"] = 0
        self.assertEqual(self.pool.poll().loads, [])
        future.set_result(set())
        result = self.pool.poll().loads[0]
        self.assertEqual(result.loaded_tokens, 0)
        self.assertIsNotNone(result.error)
        self.assertEqual(self.pool.take_errors(), {10, 11})

    def test_load_completion_includes_device_conversion_without_blocking_poll(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def convert(*args):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("Test did not release layout conversion")

        self.pool.reformat_npu_blocks = convert
        self.pool.load(LoadEvent("r", ((10, 11),), (b"a", b"b"), 0, 32))
        self.assertTrue(entered.wait(1))
        self.assertEqual(self.pool.poll().loads, [])
        release.set()
        result = _drain(self.pool, "loads")[0]
        self.assertEqual(result.loaded_tokens, 32)
        self.assertIsNone(result.error)

    def test_delayed_poll_does_not_timeout_already_completed_load(self):
        self.pool.load(LoadEvent("r", ((10, 11),), (b"a", b"b"), 0, 32))
        self.pool._loads["r"].result(timeout=1)
        self.pool._load_started_at["r"] = 0
        result = self.pool.poll().loads[0]
        self.assertEqual(result.loaded_tokens, 32)
        self.assertIsNone(result.error)

    def test_preempt_waits_for_source_reads_and_cancels_pending(self):
        ready = _Ready()
        self.addCleanup(ready.release.set)
        self.pool.save(_save(0, 16, ready))
        self.assertTrue(ready.entered.wait(1))
        self.pool.save(_save(16, 32))
        with ThreadPoolExecutor(max_workers=1) as executor:
            fence = executor.submit(self.pool.preempt, "r")
            self.assertFalse(fence.done())
            ready.release.set()
            fence.result(timeout=1)
        self.assertEqual(self.pool.poll().saves, [])
        self.assertEqual(len(self.pool.store.puts), 1)
        self.pool.save(_save(0, 16))
        self.assertEqual(_drain(self.pool, "saves"), [SaveResult("r")])

    def test_close_is_idempotent_and_rejects_submissions(self):
        self.pool.close()
        self.pool.close()
        with self.assertRaises(RuntimeError):
            self.pool.save(_save(0, 16))


class KVTopologyTest(unittest.TestCase):
    def test_failed_layout_conversion_still_fences_device_writes(self):
        transfer = _transfer()
        transfer.device_id = 0
        transfer.topology = _NPUTransferTopology(0, 1, 2, "d", False, 2)
        transfer.npu_kv_nz = False
        transfer.kv_caches = {
            "layer.0": torch.tensor([[[0, 1], [10, 11]]]),
            "layer.1": torch.tensor(0),
        }
        transfer.layer_groups = {"layer.0": 0, "layer.1": 0}
        transfer.layer_specs = {}
        fences = []
        with (
            patch("mtsc.utils.is_npu_platform", return_value=True),
            patch(
                "mtsc.utils.current_platform",
                SimpleNamespace(set_device=lambda device: None),
            ),
            patch.object(
                torch,
                "npu",
                SimpleNamespace(synchronize=lambda: fences.append(True)),
                create=True,
            ),
            self.assertRaises(ValueError),
        ):
            transfer.reformat_npu_blocks([[0]], 2)
        self.assertEqual(fences, [True])

    def test_every_target_byte_is_written_once_for_mha_gqa_and_mla(self):
        for mla in (False, True):
            for heads in (1, 2, 4, 8):
                for p_size in (1, 2, 4, 8):
                    for d_size in (1, 2, 4, 8):
                        p_shards = effective_tp(p_size, heads, mla)
                        d_shards = effective_tp(d_size, heads, mla)
                        total = 64
                        for d_rank in range(d_size):
                            coverage = [0] * (total // d_shards)
                            values = [None] * len(coverage)
                            peers = (
                                range(
                                    d_rank * (p_size // d_size),
                                    (d_rank + 1) * (p_size // d_size),
                                )
                                if p_size >= d_size
                                else [d_rank // (d_size // p_size)]
                            )
                            for p_rank in peers:
                                copy, src, dst, count, _, _ = kv_slice_plan(
                                    p_rank,
                                    p_size,
                                    d_rank,
                                    d_size,
                                    total // p_shards,
                                    total // d_shards,
                                    heads,
                                    mla,
                                )
                                if copy:
                                    shard_start = (p_rank // (p_size // p_shards)) * (
                                        total // p_shards
                                    )
                                    for i in range(count):
                                        coverage[dst + i] += 1
                                        values[dst + i] = shard_start + src + i
                            expected_start = (d_rank // (d_size // d_shards)) * len(
                                coverage
                            )
                            self.assertEqual(coverage, [1] * len(coverage))
                            self.assertEqual(
                                values,
                                list(
                                    range(
                                        expected_start, expected_start + len(coverage)
                                    )
                                ),
                            )

    def test_nonintegral_tp_ratio_rejected(self):
        with self.assertRaises(ValueError):
            kv_slice_plan(0, 2, 0, 3, 32, 32, 6, False)


def _transfer():
    transfer = object.__new__(MooncakeKVTransfer)
    transfer._closed = transfer._closing = False
    transfer._registered = True
    transfer.is_producer = True
    transfer.engine_id = "p"
    transfer.dp_rank = 2
    transfer.tp_size = transfer.pp_size = 1
    transfer.num_blocks = 6
    transfer.tp_rank = transfer.pp_rank = 0
    transfer.timeout = 1
    transfer.topology = _NPUTransferTopology(0, 1, 16, "p", False, 8)
    transfer.schema = PDTransferSchema(2, "model", "", "float16", "hnd", 16, False)
    transfer._source_lock = threading.Lock()
    transfer._source_changed = threading.Condition(transfer._source_lock)
    transfer._result_lock = threading.Lock()
    transfer._receive_lock = threading.Lock()
    transfer._sources = {}
    transfer._prepared = {}
    transfer._send_events = {}
    transfer._recv_events = {}
    transfer._send_failures = {}
    transfer._invalid = {}
    transfer._retired = set()
    transfer._finished_send = set()
    transfer._finished_recv = set()
    transfer._failed_recv = set()
    transfer._receive_futures = {}
    transfer._request_decoder = msgspec.msgpack.Decoder(PDTransferRequest)
    transfer._encoder = msgspec.msgpack.Encoder()
    return transfer


def _request(transfer, **kwargs):
    return PDTransferRequest(
        "host",
        123,
        1,
        0,
        1,
        0,
        transfer.schema,
        {"d-r": ("x", [[4]])},
        [2000],
        [16],
        [16],
        layer_names=["layer.0"],
        layer_indices=[0],
        group_indices=[0],
        engine_id="d",
        dp_rank=3,
        remote_engine_id="p",
        remote_dp_rank=2,
        destination_num_blocks=6,
        **kwargs,
    )


class _Socket:
    def __init__(self):
        self.messages = []

    async def send_multipart(self, frames):
        self.messages.append(msgspec.msgpack.decode(frames[1], type=PDTransferResponse))


class TransferBackendTest(unittest.IsolatedAsyncioTestCase):
    async def test_shutdown_delivers_terminal_response_before_closing_router(self):
        transfer = _transfer()
        entered, release = threading.Event(), threading.Event()

        def write(*args):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("Test did not release WRITE")
            return 0

        async def register(port):
            transfer.side_port = port

        transfer.engine = SimpleNamespace(batch_transfer_sync_write=write)
        transfer.regions = [TransferRegion("layer.0", 0, 0, 1000, 16, 16)]
        transfer.hostname = "127.0.0.1"
        transfer._register_worker = register
        transfer._ctx = zmq.asyncio.Context()
        transfer._loop = asyncio.new_event_loop()
        transfer._loop_thread = threading.Thread(target=transfer._loop.run_forever)
        transfer._loop_thread.start()
        transfer._serve_tasks = set()
        transfer._send_pool = ThreadPoolExecutor(max_workers=1)
        transfer._bootstrap = None
        transfer._registered_storage = []
        transfer.kv_caches = {}
        ready = threading.Event()
        transfer._listener_future = asyncio.run_coroutine_threadsafe(
            transfer._listen(ready), transfer._loop
        )
        self.assertTrue(await asyncio.to_thread(ready.wait, 1))
        transfer.send(SendEvent("p-r", "x", ((1,),)))
        context = zmq.asyncio.Context()
        client = context.socket(zmq.DEALER)
        client.connect(f"tcp://127.0.0.1:{transfer.side_port}")
        try:
            await client.send(msgspec.msgpack.encode(_request(transfer)))
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            closing = asyncio.create_task(asyncio.to_thread(transfer.close))
            await asyncio.sleep(0.01)
            self.assertFalse(closing.done())
            release.set()
            response = msgspec.msgpack.decode(
                await asyncio.wait_for(client.recv(), 2), type=PDTransferResponse
            )
            self.assertEqual(response.completed, ["d-r"])
            await asyncio.wait_for(closing, 2)
            self.assertFalse(transfer._loop_thread.is_alive())
            self.assertTrue(transfer._loop.is_closed())
            transfer.close()
            with self.assertRaises(RuntimeError):
                transfer.send(SendEvent("p-r", "new", ((1,),)))
        finally:
            release.set()
            client.close(linger=0)
            context.destroy(linger=0)
            if not transfer._closed:
                await asyncio.to_thread(transfer.close)

    async def test_cancel_fences_active_write_and_suppresses_completion(self):
        transfer = _transfer()
        entered, release = asyncio.Event(), asyncio.Event()

        async def write(*args):
            entered.set()
            await release.wait()
            return True, {0}

        transfer._write_one = write
        transfer.send(SendEvent("p-r", "x", ((1,),)))
        request = asyncio.create_task(
            transfer._serve(
                b"a",
                msgspec.msgpack.encode(_request(transfer)),
                _Socket(),
            )
        )
        await asyncio.wait_for(entered.wait(), 1)
        fence = asyncio.create_task(asyncio.to_thread(transfer.cancel, "p-r", "x"))
        await asyncio.sleep(0.01)
        self.assertFalse(fence.done())
        release.set()
        await asyncio.gather(request, fence)
        self.assertEqual(transfer.poll().sends, [])
        self.assertIn("x", transfer._retired)

    async def test_recv_preemption_waits_for_remote_terminal(self):
        transfer = _transfer()
        transfer._recv_events["d-r"] = RecvEvent(
            "d-r", "x", ((4,),), "bootstrap", "p", 2
        )
        future = Future()
        transfer._receive_futures["d-r"] = future
        fence = asyncio.create_task(asyncio.to_thread(transfer.preempt, "d-r"))
        await asyncio.sleep(0.01)
        self.assertFalse(fence.done())
        future.set_result(None)
        await asyncio.wait_for(fence, 1)
        self.assertEqual(transfer.poll().recvs, [])
        self.assertEqual(transfer._recv_events, {})
        self.assertIn("x", transfer._retired)

    async def test_peer_failure_waits_for_other_published_destination(self):
        transfer = _transfer()
        waiting, release = asyncio.Event(), asyncio.Event()

        class Client:
            def connect(self, address):
                self.address = address

            def setsockopt(self, *args):
                pass

            async def send(self, payload):
                request = msgspec.msgpack.decode(payload, type=PDTransferRequest)
                self_identity = (request.engine_id, request.dp_rank)
                if self_identity != ("p", 2):
                    raise AssertionError("Missing local replica identity")

            async def recv(self):
                if self.address == "fail":
                    raise RuntimeError("First peer failed")
                waiting.set()
                await release.wait()
                return msgspec.msgpack.encode(PDTransferResponse(0, ["d-r"]))

            def close(self, **kwargs):
                pass

        async def query(*args):
            return {0: {0: "fail", 1: "wait"}}

        transfer._ctx = SimpleNamespace(socket=lambda kind: Client())
        transfer._query_workers = query
        transfer.regions = []
        transfer.hostname, transfer.rpc_port = "host", 123
        transfer._response_decoder = msgspec.msgpack.Decoder(PDTransferResponse)
        transfer.reformat_npu_blocks = lambda *args: None
        receive = asyncio.create_task(
            transfer._receive("d-r", "x", [[]], "p", "bootstrap", 2)
        )
        await asyncio.wait_for(waiting.wait(), 1)
        self.assertFalse(receive.done())
        self.assertEqual(transfer._failed_recv, set())
        release.set()
        await receive
        self.assertEqual(transfer._failed_recv, {"d-r"})

    async def test_real_region_plan_transfers_heterogeneous_tp_pp_and_mla(self):
        # Simulate TE bytes while executing the production region mapper,
        # source suffix selection, control handlers and completion aggregation.
        for mla in (False, True):
            memory = {}
            d_shards = 1 if mla else 2
            d_page = 64 // d_shards
            for layer in range(2):
                memory[20000 + layer * 1000] = bytearray(d_page * 6)

            def write(session, sources, targets, lengths, memory=memory):
                for src, dst, size in zip(sources, targets, lengths, strict=True):
                    s_base = next(
                        base
                        for base, buf in memory.items()
                        if base <= src < base + len(buf)
                    )
                    d_base = next(
                        base
                        for base, buf in memory.items()
                        if base <= dst < base + len(buf)
                    )
                    memory[d_base][dst - d_base : dst - d_base + size] = memory[s_base][
                        src - s_base : src - s_base + size
                    ]
                return 0

            responses = []
            for p_rank in (0, 1):  # Peers for D TP rank 0 of 2.
                for pp_rank in (0, 1):
                    transfer = _transfer()
                    transfer.tp_size, transfer.tp_rank = 4, p_rank
                    transfer.pp_size, transfer.pp_rank = 2, pp_rank
                    transfer.topology = _NPUTransferTopology(p_rank, 4, 16, "p", mla, 4)
                    transfer.schema = msgspec.structs.replace(
                        transfer.schema, is_mla=mla
                    )
                    p_page = 64 if mla else 16
                    base = 1000 + p_rank * 4000 + pp_rank * 1000
                    shard_start = 0 if mla else p_rank * p_page
                    memory[base] = bytearray(
                        block * 64 + shard_start + offset
                        for block in range(3)
                        for offset in range(p_page)
                    )
                    transfer.regions = [
                        TransferRegion(
                            f"layer.{pp_rank}", pp_rank, 0, base, p_page, p_page
                        )
                    ]
                    transfer.engine = SimpleNamespace(batch_transfer_sync_write=write)
                    transfer._loop = asyncio.get_running_loop()
                    transfer.send(SendEvent("p-r", "x", ((0, 1, 2),)))
                    request = msgspec.structs.replace(
                        _request(transfer),
                        tp_size=2,
                        requests={"d-r": ("x", [[4, 5]])},
                        region_base_addresses=[20000, 21000],
                        block_lengths=[d_page] * 2,
                        kv_block_lengths=[d_page] * 2,
                        layer_names=["layer.0", "layer.1"],
                        layer_indices=[0, 1],
                        group_indices=[0, 0],
                    )
                    socket = _Socket()
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        transfer._send_pool = executor
                        await transfer._serve(
                            b"a", msgspec.msgpack.encode(request), socket
                        )
                    responses.extend(socket.messages)
                    self.assertIsNone(transfer.poll().sends[0].error)
            receiver = _transfer()
            receiver.regions = [
                TransferRegion(
                    f"layer.{layer}", layer, 0, 20000 + layer * 1000, d_page, d_page
                )
                for layer in range(2)
            ]
            receiver._validate_coverage("d-r", [[4, 5]], responses, 1 if mla else 2)
            for layer in range(2):
                data = memory[20000 + layer * 1000]
                for dest, source in ((4, 1), (5, 2)):
                    self.assertEqual(
                        data[dest * d_page : (dest + 1) * d_page],
                        bytearray(range(source * 64, source * 64 + d_page)),
                    )

    async def test_early_recv_and_duplicate_pull_write_once(self):
        transfer = _transfer()
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def write(*args):
            calls.append(args)
            entered.set()
            await release.wait()
            return True, {0}

        transfer._write_one = write
        socket = _Socket()
        payload = msgspec.msgpack.encode(_request(transfer))
        first = asyncio.create_task(transfer._serve(b"a", payload, socket))
        await asyncio.sleep(0)
        transfer.prepare("p-r", "x")
        transfer.send(SendEvent("p-r", "x", ((1,),)))
        await asyncio.wait_for(entered.wait(), 1)
        duplicate = asyncio.create_task(transfer._serve(b"b", payload, socket))
        await asyncio.sleep(0)
        self.assertEqual(transfer.poll().sends, [])
        release.set()
        await asyncio.gather(first, duplicate)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(transfer.poll().sends), 1)
        self.assertEqual(transfer.poll().sends, [])
        late = _Socket()
        await transfer._serve(b"c", payload, late)
        self.assertEqual(late.messages[0].failed, ["d-r"])

    async def test_cancel_unpublished_session_wakes_early_receiver(self):
        transfer = _transfer()
        socket = _Socket()
        task = asyncio.create_task(
            transfer._serve(b"a", msgspec.msgpack.encode(_request(transfer)), socket)
        )
        await asyncio.sleep(0)
        transfer.cancel("p-r", "x")
        await asyncio.wait_for(task, 1)
        self.assertEqual(socket.messages[0].failed, ["d-r"])
        self.assertEqual(transfer.poll().sends, [])
        with self.assertRaises(ValueError):
            transfer.prepare("p-r", "x")

    async def test_preempt_unready_source_keeps_receiver_for_resume(self):
        transfer = _transfer()
        transfer.prepare("p-r", "x")
        transfer.preempt("p-r")
        transfer.send(SendEvent("p-r", "x", ((1,),)))
        self.assertNotIn("x", transfer._retired)

    async def test_wrong_dp_rejected_before_source_access(self):
        transfer = _transfer()
        socket = _Socket()
        request = msgspec.structs.replace(_request(transfer), remote_dp_rank=99)
        await transfer._serve(b"a", msgspec.msgpack.encode(request), socket)
        self.assertEqual(socket.messages[0].error, "P engine/DP identity mismatch")
        self.assertEqual(transfer._sources, {})

    async def test_source_timeout_retires_attempt_and_reports_error(self):
        transfer = _transfer()
        transfer.send(SendEvent("p-r", "x", ((1,),)))
        source = transfer._sources["x"]
        source.expires_at = 0
        source.active_writes = 1
        self.assertEqual(transfer.poll().sends, [])
        source.active_writes = 0
        self.assertIsNotNone(transfer.poll().sends[0].error)
        self.assertIn("x", transfer._retired)

    async def test_empty_recv_still_confirms_send(self):
        transfer = _transfer()
        transfer.send(SendEvent("p-r", "x", ((1,),)))
        request = msgspec.structs.replace(
            _request(transfer), requests={"d-r": ("x", [[]])}
        )
        socket = _Socket()
        await transfer._serve(b"a", msgspec.msgpack.encode(request), socket)
        self.assertEqual(socket.messages[0].completed, ["d-r"])
        self.assertIsNone(transfer.poll().sends[0].error)

    async def test_recv_result_waits_for_future_terminal_and_invalidates_only_targets(
        self,
    ):
        transfer = _transfer()
        event = RecvEvent("d-r", "x", ((4, 5),), "bootstrap", "p", 2)
        transfer._recv_events["d-r"] = event
        future = Future()
        transfer._receive_futures["d-r"] = future
        transfer._failed_recv.add("d-r")
        self.assertEqual(transfer.poll().recvs, [])
        future.set_result(None)
        self.assertIsNotNone(transfer.poll().recvs[0].error)
        self.assertEqual(transfer.take_errors(), {4, 5})


class WorkerReleaseTest(unittest.TestCase):
    def _worker(self):
        worker = object.__new__(MTSCWorker)
        worker.pool = SimpleNamespace(
            poll=lambda: PoolPollResult(), take_errors=lambda: set()
        )
        worker.transfer = SimpleNamespace(
            poll=lambda: TransferPollResult(), take_errors=lambda: set()
        )
        worker._save_pending = set()
        worker._send = {}
        worker._decode = {}
        worker._plain_store_loads = {}
        worker._ignored_pd_recvs = set()
        worker._load_errors = set()
        return worker

    def test_intermediate_save_does_not_release_active_request(self):
        worker = self._worker()
        worker._save_pending.add("r")
        worker.pool.poll = lambda: PoolPollResult(saves=[SaveResult("r")])
        self.assertEqual(
            worker.get_finished(set(), MTSCConnectorMetadata()), (None, None)
        )
        worker.pool.poll = lambda: PoolPollResult()
        metadata = MTSCConnectorMetadata(
            send_requirements={"r": SendRequirement(store=True)}
        )
        self.assertEqual(worker.get_finished({"r"}, metadata), ({"r"}, None))

    def test_new_save_resets_completion_before_request_end(self):
        worker = self._worker()
        worker.pool.save = lambda event: None
        metadata = MTSCConnectorMetadata(
            store_requests=[StoreRequest("r", 16, ([1],), [b"a"], save=True)],
            send_requirements={"r": SendRequirement(store=True)},
        )
        ready = SimpleNamespace(record=lambda: None, synchronize=lambda: None)
        with patch("mtsc.worker.new_device_event", return_value=ready):
            self.assertEqual(worker.get_finished({"r"}, metadata), (None, None))
        worker.pool.poll = lambda: PoolPollResult(saves=[SaveResult("r")])
        self.assertEqual(
            worker.get_finished(set(), MTSCConnectorMetadata()), ({"r"}, None)
        )
