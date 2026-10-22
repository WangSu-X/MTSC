from __future__ import annotations

import asyncio
import ctypes
import threading
import time
import unittest
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import torch
import zmq
import zmq.asyncio

from mtsc.kv_cache_pool import (
    KeyMetadata,
    LoadEvent,
    MooncakeKVCachePool,
    PoolKey,
    PoolPollResult,
    SaveEvent,
    SaveResult,
)
from mtsc.kv_transfer import (
    MooncakeKVTransfer,
    PullReqMeta,
    RecvEvent,
    SendBlockMeta,
    SendEvent,
    TransferPollResult,
    _NPUTransferTopology,
    _require_tcp_write_ack,
)
from mtsc.kv_transfer import (
    _compute_sender_transfer_plan as local_sender_plan,
)
from mtsc.protocol import (
    FinishedWait,
    KVPoolSaveRequest,
    KVTransferRequest,
    KVTransferResponse,
    KVTransferStatus,
    MTSCConnectorMetadata,
)
from mtsc.utils import effective_tp
from mtsc.worker import MTSCWorker


class _Database:
    block_size = 16

    def process_tokens(self, token_count, hashes, start):
        for index in range((start + 15) // 16, token_count // 16):
            yield (
                index * 16,
                (index + 1) * 16,
                PoolKey(KeyMetadata("model", 0, 0, 0, 0), hashes[index].hex()),
            )

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
    pool._task_lock = threading.RLock()
    pool._closed = False
    pool._registered = True
    pool._save_states = {}
    pool._invalid = {}
    pool._loads = {}
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
        self.pool._loads["r"].future.result(timeout=1)
        future = Future()
        self.pool._loads["r"].future = future
        self.pool._loads["r"].started_at = 0
        self.assertEqual(self.pool.poll().loads, [])
        future.set_result(set())
        result = self.pool.poll().loads[0]
        self.assertEqual(result.loaded_tokens, 0)
        self.assertIsNotNone(result.error)
        self.assertEqual(self.pool.take_errors(), {10, 11})

    def test_failed_submission_allows_retry_without_orphaned_load(self):
        event = LoadEvent("r", ((10, 11),), (b"a", b"b"), 0, 32)
        with (
            patch.object(
                self.pool._load_pool,
                "submit",
                side_effect=RuntimeError("submit failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "submit failed"),
        ):
            self.pool.load(event)
        self.assertEqual(self.pool.poll().loads, [])
        self.pool.load(event)
        result = _drain(self.pool, "loads")[0]
        self.assertEqual(result.loaded_tokens, 32)
        self.assertIsNone(result.error)

    def test_load_future_failure_invalidates_only_requested_blocks(self):
        future = Future()
        future.set_exception(RuntimeError("load task failed"))
        event = LoadEvent("r", ((10, 11, 12),), (b"a", b"b", b"c"), 16, 48)
        with patch.object(self.pool._load_pool, "submit", return_value=future):
            self.pool.load(event)
        result = self.pool.poll().loads[0]
        self.assertEqual(result.loaded_tokens, 16)
        self.assertEqual(result.error, "load task failed")
        self.assertEqual(self.pool.take_errors(), {11, 12})
        self.assertEqual(self.pool.poll().loads, [])

    def test_load_completion_waits_for_store_writes_without_blocking_poll(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        store_get = self.pool.store.batch_get_into_multi_buffers

        def get(*args):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("Test did not release Store GET")
            return store_get(*args)

        self.pool.store.batch_get_into_multi_buffers = get
        self.pool.load(LoadEvent("r", ((10, 11),), (b"a", b"b"), 0, 32))
        self.assertTrue(entered.wait(1))
        self.assertEqual(self.pool.poll().loads, [])
        release.set()
        result = _drain(self.pool, "loads")[0]
        self.assertEqual(result.loaded_tokens, 32)
        self.assertIsNone(result.error)

    def test_delayed_poll_does_not_timeout_already_completed_load(self):
        self.pool.load(LoadEvent("r", ((10, 11),), (b"a", b"b"), 0, 32))
        self.pool._loads["r"].future.result(timeout=1)
        self.pool._loads["r"].started_at = 0
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
    def test_every_target_byte_is_written_once_for_mha_and_mla(self):
        for mla in (False, True):
            for heads in (8,):
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
                                copy, src, dst, count = local_sender_plan(
                                    p_rank,
                                    p_size,
                                    d_rank,
                                    d_size,
                                    total // p_shards,
                                    total // d_shards,
                                    mla or p_size > heads,
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
        with self.assertRaises(AssertionError):
            local_sender_plan(0, 2, 0, 3, 32, 32, False or 2 > 6)


def _transfer():
    transfer = object.__new__(MooncakeKVTransfer)
    transfer._remote_layouts = OrderedDict()
    transfer._closed = False
    transfer._receive_error = None
    transfer._registered = True
    transfer.is_producer = True
    transfer.is_consumer = False
    transfer.sender_loop = None
    transfer.receiver_loop = None
    transfer._sender_listener_t = None
    transfer._receiver_t = None
    transfer._sender_executor = None
    transfer.engine_id = "p"
    transfer.dp_rank = 2
    transfer.tp_size = transfer.pp_size = 1
    transfer.num_blocks = 6
    transfer.num_sender_tasks = 2
    transfer.sender_worker_queue = asyncio.Queue()
    transfer.tp_rank = transfer.pp_rank = 0
    transfer.timeout = 1
    transfer.protocol = "rdma"
    transfer.kv_caches_base_addr = [1000]
    transfer.block_len_per_layer = [16]
    transfer.kv_caches_base_addr = [1000]
    transfer.block_len_per_layer = [16]
    transfer.transfer_topo = _NPUTransferTopology(0, 1, 16, False, 8)

    transfer.reqs_need_send = {}
    transfer._recv_events = {}
    transfer._invalid = {}
    transfer._retired_recvs = set()
    transfer.finished_sending_reqs = set()

    transfer._remote_agents = {}
    transfer._remote_tp_size = {}
    transfer._pending_bootstrap_queries = {}
    transfer._failed_recv = set()
    transfer._receive_futures = {}
    transfer.finished_recving_reqs = set()
    transfer._request_decoder = msgspec.msgpack.Decoder(KVTransferRequest)
    transfer._encoder = msgspec.msgpack.Encoder()

    async def receive_events(events, futures=None):
        transfer.is_producer, transfer.is_consumer = False, True
        transfer.receiver_loop = asyncio.get_running_loop()
        futures = futures or {
            event.request_id: asyncio.get_running_loop().create_future()
            for event in events
        }
        transfer._recv_events.update((event.request_id, event) for event in events)
        transfer._receive_futures.update(futures)
        metas = {
            e.request_id: PullReqMeta(
                e.request_id,
                e.transfer_id,
                [list(g) for g in e.block_ids],
                e.remote_engine_id,
                e.bootstrap_addr,
            )
            for e in events
        }
        await transfer._start_load_kv({events[0].remote_engine_id: metas})
        await asyncio.gather(*futures.values())

    transfer._test_receive_events = receive_events
    return transfer


def _request(transfer, **kwargs):
    return KVTransferRequest(
        "host", 123, 1, 0, {"d-r": ("x", [[4]])}, [2000], [16], **kwargs
    )


class _Socket:
    def __init__(self):
        self.messages = []

    async def send_multipart(self, frames):
        self.messages.append(msgspec.msgpack.decode(frames[1], type=KVTransferResponse))


class TCPCompletionRequirementTest(unittest.TestCase):
    def test_legacy_engine_rejected_before_transfer(self):
        with (
            patch("mtsc.kv_transfer.version", return_value="0.3.10.post1"),
            self.assertRaisesRegex(RuntimeError, "both P and D"),
        ):
            _require_tcp_write_ack("tcp")

    def test_acknowledged_engine_and_forced_legacy_mode(self):
        with patch("mtsc.kv_transfer.version", return_value="0.3.13.post1"):
            with patch.dict("os.environ", {"MC_TCP_PROTO": "2"}):
                _require_tcp_write_ack("tcp")
            with (
                patch.dict("os.environ", {"MC_TCP_PROTO": "1"}),
                self.assertRaisesRegex(RuntimeError, "MC_TCP_PROTO=1"),
            ):
                _require_tcp_write_ack("tcp")

    def test_other_transports_do_not_require_tcp_version(self):
        with patch("mtsc.kv_transfer.version") as installed:
            _require_tcp_write_ack("rdma")
            _require_tcp_write_ack("ascend")
            installed.assert_not_called()


class TransferBackendTest(unittest.IsolatedAsyncioTestCase):
    def test_producer_rejects_receive_without_receiver_loop(self):
        transfer = _transfer()
        with self.assertRaisesRegex(ValueError, "Only D can receive"):
            transfer.recv([RecvEvent("d-r", "x", ((4,),), "bootstrap", "p", 2)])
        self.assertEqual(transfer._receive_futures, {})

    async def test_close_waits_for_receiver_fence_before_stopping_loop(self):
        transfer = _transfer()
        transfer.is_producer, transfer.is_consumer = False, True
        transfer.receiver_loop = asyncio.new_event_loop()
        started = threading.Event()
        transfer.receiver_loop.call_soon(started.set)
        transfer._receiver_t = threading.Thread(
            target=transfer.receiver_loop.run_forever
        )
        transfer._receiver_t.start()
        self.assertTrue(await asyncio.to_thread(started.wait, 1))
        transfer._ctx = zmq.asyncio.Context()
        transfer._bootstrap = None
        transfer._registered_storage = []
        transfer.kv_caches = {}

        async def register_pending():
            pending = asyncio.get_running_loop().create_future()
            transfer._receive_futures["d-r"] = pending
            return pending

        pending = asyncio.run_coroutine_threadsafe(
            register_pending(), transfer.receiver_loop
        ).result()
        closing = asyncio.create_task(asyncio.to_thread(transfer.close))
        try:
            await asyncio.sleep(0.01)
            self.assertFalse(closing.done())
            self.assertTrue(transfer.receiver_loop.is_running())
            transfer.receiver_loop.call_soon_threadsafe(pending.set_result, None)
            await asyncio.wait_for(closing, 2)
            self.assertTrue(transfer.receiver_loop.is_closed())
            self.assertFalse(transfer._receiver_t.is_alive())
            self.assertTrue(transfer._ctx.closed)
            transfer.close()
        finally:
            if not pending.done():
                transfer.receiver_loop.call_soon_threadsafe(pending.set_result, None)
            await closing

    async def test_native_write_receives_whole_batch(self):
        transfer = _transfer()
        transfer.sender_loop = asyncio.get_running_loop()
        calls = []
        transfer.engine = SimpleNamespace(
            batch_transfer_sync_write=lambda *args: calls.append(args) or 0
        )
        with ThreadPoolExecutor(max_workers=1) as pool:
            transfer._sender_executor = pool
            self.assertEqual(
                await transfer._write_buffers(
                    "peer", list(range(600)), [0] * 600, [1] * 600
                ),
                0,
            )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], list(range(600)))

    async def test_region_mismatch_is_rejected_before_write(self):
        transfer = _transfer()
        transfer.kv_caches_base_addr = [1000]
        transfer.block_len_per_layer = [16]
        request = msgspec.structs.replace(_request(transfer), block_lens=[32])
        socket = _Socket()
        await transfer.send_kv_to_decode(
            b"peer", msgspec.msgpack.encode(request), socket
        )
        self.assertEqual(socket.messages[0].status, KVTransferStatus.ERROR)
        self.assertIn("length mismatch", socket.messages[0].err_msg)
        self.assertEqual(transfer.reqs_need_send, {})

    async def test_replicated_gqa_uses_native_sender_selection(self):
        transfer = _transfer()
        transfer.tp_size = 8
        transfer.transfer_topo = _NPUTransferTopology(0, 8, 16, False, 2)
        request = msgspec.structs.replace(_request(transfer), block_lens=[32])
        src, dst, sizes = transfer._build_request_transfer_params(
            "d-r", SendBlockMeta("p-r", "x", local_block_ids=((1,),)), request
        )
        self.assertEqual((src, dst, sizes), ([1016], [2128], [16]))
        transfer.tp_rank = 4
        src, dst, sizes = transfer._build_request_transfer_params(
            "d-r", SendBlockMeta("p-r", "x", local_block_ids=((1,),)), request
        )
        self.assertEqual((src, dst, sizes), ([], [], []))

    async def test_blocks_first_write_copies_k_and_v_with_tp_slicing(self):
        class CopyEngine:
            def batch_register_memory(self, pointers, lengths):
                return 0

            def batch_transfer_sync_write(self, session, src, dst, sizes):
                for source, target, size in zip(src, dst, sizes, strict=True):
                    ctypes.memmove(target, source, size)
                return 0

        def registered(cache, tp_size, tp_rank):
            transfer = _transfer()
            transfer._registered = False
            transfer.is_producer = False
            transfer.tp_size, transfer.tp_rank = tp_size, tp_rank
            transfer.transfer_topo = SimpleNamespace(
                virtually_split_kv_in_blocks=True,
                total_num_kv_heads=2,
                local_replicates_kv_cache=False,
                is_mla=False,
                get_transfer_cache_regions=lambda raw, spec: [raw],
                split_k_and_v=False,
            )
            transfer.layer_specs = {"layer.0": SimpleNamespace()}
            transfer.kv_caches_base_addr = []
            transfer.block_len_per_layer = []
            transfer.engine = CopyEngine()
            transfer.sender_loop = asyncio.get_running_loop()
            transfer._sender_executor = None
            transfer.register({"layer.0": cache})
            return transfer

        source = torch.arange(6 * 2 * 2 * 16 * 64, dtype=torch.int32)
        source = source.reshape(6, 2, 2, 16, 64).permute(0, 1, 3, 2, 4)
        for tp_size, tp_rank in [(1, 0), (2, 0), (2, 1)]:
            with self.subTest(tp_size=tp_size, tp_rank=tp_rank):
                target = torch.full(
                    (6, 2, 2 // tp_size, 16, 64), -1, dtype=torch.int32
                ).permute(0, 1, 3, 2, 4)
                sender = registered(source, 1, 0)
                receiver = registered(target, tp_size, tp_rank)
                request = msgspec.structs.replace(
                    _request(sender),
                    remote_tp_size=tp_size,
                    remote_tp_rank=tp_rank,
                    kv_caches_base_addr=receiver.kv_caches_base_addr,
                    block_lens=receiver.block_len_per_layer,
                )
                src, dst, lengths, errors, _ = await sender._build_transfer_params(
                    [("d-r", SendBlockMeta("p-r", "x", local_block_ids=((2,),)))],
                    request,
                    sender._prepare_transfer_regions(request),
                )
                self.assertEqual(errors, [])
                ret = await sender._write_buffers("peer", src, dst, lengths)
                expected = (
                    source[2]
                    if tp_size == 1
                    else source[2, :, :, tp_rank : tp_rank + 1, :]
                )
                self.assertEqual(ret, 0)
                self.assertTrue(torch.equal(target[4], expected))
                self.assertTrue(torch.all(target[:4] == -1))
                self.assertTrue(torch.all(target[5] == -1))

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
        transfer.kv_caches_base_addr = [1000]
        transfer.block_len_per_layer = [16]
        transfer.hostname = "127.0.0.1"
        transfer._register_worker = register
        transfer._ctx = zmq.asyncio.Context()
        transfer.sender_loop = asyncio.new_event_loop()
        transfer._sender_listener_t = threading.Thread(
            target=transfer.sender_loop.run_forever
        )
        transfer._sender_listener_t.start()
        transfer.sender_tasks = []
        transfer._sender_executor = ThreadPoolExecutor(max_workers=1)
        transfer._bootstrap = None
        transfer._registered_storage = []
        transfer.kv_caches = {}
        ready = threading.Event()
        transfer._sender_listener_future = asyncio.run_coroutine_threadsafe(
            transfer._sender_listener(ready), transfer.sender_loop
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
                await asyncio.wait_for(client.recv(), 2), type=KVTransferResponse
            )
            self.assertEqual(response.ok_reqs, ["d-r"])
            await asyncio.wait_for(closing, 2)
            self.assertFalse(transfer._sender_listener_t.is_alive())
            self.assertTrue(transfer.sender_loop.is_closed())
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
        transfer.sender_loop = asyncio.get_running_loop()
        entered, release = asyncio.Event(), asyncio.Event()

        async def write(*args):
            entered.set()
            await release.wait()
            return 0

        transfer._write_buffers = write
        (await transfer._publish_send(SendEvent("p-r", "x", ((1,),))))
        request = asyncio.create_task(
            transfer.send_kv_to_decode(
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
        self.assertEqual((await transfer.fetch_finished_sending_reqs()), [])
        self.assertNotIn("x", transfer.reqs_need_send)

    async def test_recv_preemption_waits_for_remote_terminal(self):
        transfer = _transfer()
        transfer._recv_events["d-r"] = RecvEvent(
            "d-r", "x", ((4,),), "bootstrap", "p", 2
        )
        transfer.is_producer, transfer.is_consumer = False, True
        transfer.receiver_loop = asyncio.get_running_loop()
        future = asyncio.get_running_loop().create_future()
        transfer._receive_futures["d-r"] = future
        fence = asyncio.create_task(asyncio.to_thread(transfer.preempt, "d-r"))
        await asyncio.sleep(0.01)
        self.assertFalse(fence.done())
        future.set_result(None)
        await asyncio.wait_for(fence, 1)
        self.assertEqual((await asyncio.to_thread(transfer.poll)).recvs, [])
        self.assertEqual(transfer._recv_events, {})
        self.assertIn("x", transfer._retired_recvs)

    async def test_peer_failure_waits_for_other_published_destination(self):
        transfer = _transfer()
        waiting, release = asyncio.Event(), asyncio.Event()

        class Client:
            def connect(self, address):
                self.address = address

            def setsockopt(self, *args):
                pass

            async def send(self, payload):
                request = msgspec.msgpack.decode(payload, type=KVTransferRequest)
                if request.req_blocks != {"d-r": ("x", [[]])}:
                    raise AssertionError(
                        "Transfer wire request must be keyed by D request ID"
                    )

            async def recv(self):
                if self.address == "fail":
                    return msgspec.msgpack.encode(
                        KVTransferResponse(
                            KVTransferStatus.ERROR, err_msg="First peer failed"
                        )
                    )
                waiting.set()
                await release.wait()
                return msgspec.msgpack.encode(KVTransferResponse(0, ["d-r"]))

            def close(self, **kwargs):
                pass

        transfer._ctx = SimpleNamespace(socket=lambda kind: Client())
        transfer._remote_agents["p"] = {0: {0: "fail"}, 1: {0: "wait"}}
        transfer._remote_tp_size["p"] = 2
        transfer.kv_caches_base_addr = []
        transfer.block_len_per_layer = []
        transfer.hostname, transfer.rpc_port = "host", 123
        transfer._response_decoder = msgspec.msgpack.Decoder(KVTransferResponse)
        transfer.reformat_npu_blocks = lambda *args: None
        receive = asyncio.create_task(
            transfer._test_receive_events(
                [RecvEvent("d-r", "x", ((),), "bootstrap", "p", 2)]
            )
        )
        await asyncio.wait_for(waiting.wait(), 1)
        self.assertFalse(receive.done())
        self.assertFalse(transfer._receive_futures["d-r"].done())
        release.set()
        await receive
        self.assertEqual(transfer._failed_recv, {"d-r"})

    async def test_real_region_plan_transfers_heterogeneous_tp_and_mla(self):
        # Simulate TE bytes while executing the production region mapper,
        # source suffix selection, control handlers and completion aggregation.
        for mla, p_size, d_size in (
            (mla, p, d) for mla in (False, True) for p, d in ((4, 2), (8, 1))
        ):
            memory = {}
            d_shards = 1 if mla else d_size
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

            for p_rank in range(p_size // d_size):  # Peers for D rank 0; PP=1.
                transfer = _transfer()
                transfer.tp_size, transfer.tp_rank = p_size, p_rank
                transfer.transfer_topo = _NPUTransferTopology(
                    p_rank, p_size, 16, mla, 8
                )
                p_page = 64 if mla else 64 // p_size
                transfer.kv_caches_base_addr = []
                transfer.block_len_per_layer = []
                for layer in range(2):
                    base = 1000 + p_rank * 2000 + layer * 500
                    shard_start = 0 if mla else p_rank * p_page
                    memory[base] = bytearray(
                        block * 64 + shard_start + offset
                        for block in range(3)
                        for offset in range(p_page)
                    )
                    transfer.kv_caches_base_addr.append(base)
                    transfer.block_len_per_layer.append(p_page)
                transfer.engine = SimpleNamespace(batch_transfer_sync_write=write)
                transfer.sender_loop = asyncio.get_running_loop()
                (await transfer._publish_send(SendEvent("p-r", "x", ((0, 1, 2),))))
                request = msgspec.structs.replace(
                    _request(transfer),
                    remote_tp_size=d_size,
                    req_blocks={"d-r": ("x", [[4, 5]])},
                    kv_caches_base_addr=[20000, 21000],
                    block_lens=[d_page] * 2,
                )
                socket = _Socket()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    transfer._sender_executor = executor
                    await transfer.send_kv_to_decode(
                        b"a", msgspec.msgpack.encode(request), socket
                    )
                self.assertEqual(socket.messages[0].ok_reqs, ["d-r"])
                self.assertIsNone(
                    (await transfer.fetch_finished_sending_reqs())[0].error
                )
            for layer in range(2):
                data = memory[20000 + layer * 1000]
                for dest, source in ((4, 1), (5, 2)):
                    self.assertEqual(
                        data[dest * d_page : (dest + 1) * d_page],
                        bytearray(range(source * 64, source * 64 + d_page)),
                    )

    async def test_early_recv_waits_until_source_is_ready(self):
        transfer = _transfer()
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def write(*args):
            calls.append(args)
            entered.set()
            await release.wait()
            return 0

        transfer._write_buffers = write
        socket = _Socket()
        payload = msgspec.msgpack.encode(_request(transfer))
        task = asyncio.create_task(transfer.send_kv_to_decode(b"a", payload, socket))
        await asyncio.sleep(0)
        self.assertEqual(calls, [])
        (await transfer._prepare_send("p-r", "x"))
        (await transfer._publish_send(SendEvent("p-r", "x", ((1,),))))
        await asyncio.wait_for(entered.wait(), 1)
        self.assertEqual((await transfer.fetch_finished_sending_reqs()), [])
        release.set()
        await task
        self.assertEqual(len(calls), 1)
        self.assertEqual(socket.messages[0].ok_reqs, ["d-r"])
        self.assertEqual(len(await transfer.fetch_finished_sending_reqs()), 1)
        self.assertEqual((await transfer.fetch_finished_sending_reqs()), [])

    async def test_cancel_unpublished_session_wakes_early_receiver(self):
        transfer = _transfer()
        socket = _Socket()
        task = asyncio.create_task(
            transfer.send_kv_to_decode(
                b"a", msgspec.msgpack.encode(_request(transfer)), socket
            )
        )
        await asyncio.sleep(0)
        (await transfer._cancel_send("p-r", "x"))
        await asyncio.wait_for(task, 1)
        self.assertEqual(socket.messages[0].err_reqs, ["d-r"])
        self.assertEqual((await transfer.fetch_finished_sending_reqs()), [])
        self.assertNotIn("x", transfer.reqs_need_send)

    async def test_preempt_unready_source_keeps_receiver_for_resume(self):
        transfer = _transfer()
        (await transfer._prepare_send("p-r", "x"))
        (await transfer._preempt_send("p-r"))
        (await transfer._publish_send(SendEvent("p-r", "x", ((1,),))))
        self.assertTrue(transfer.reqs_need_send["x"].ready.is_set())

    async def test_unpaired_tp_rejected_before_source_access(self):
        transfer = _transfer()
        socket = _Socket()
        request = msgspec.structs.replace(_request(transfer), remote_tp_rank=1)
        await transfer.send_kv_to_decode(b"a", msgspec.msgpack.encode(request), socket)
        self.assertEqual(
            socket.messages[0].err_msg, "D TP rank is not paired with this P rank"
        )
        self.assertEqual(transfer.reqs_need_send, {})

    async def test_source_timeout_removes_metadata_and_reports_error(self):
        transfer = _transfer()
        (await transfer._publish_send(SendEvent("p-r", "x", ((1,),))))
        source = transfer.reqs_need_send["x"]
        source.expire_time = 0
        source.sending = 1
        self.assertEqual((await transfer.fetch_finished_sending_reqs()), [])
        source.sending = 0
        self.assertIsNotNone((await transfer.fetch_finished_sending_reqs())[0].error)
        self.assertNotIn("x", transfer.reqs_need_send)

    async def test_empty_recv_still_confirms_send(self):
        transfer = _transfer()
        (await transfer._publish_send(SendEvent("p-r", "x", ((1,),))))
        request = msgspec.structs.replace(
            _request(transfer), req_blocks={"d-r": ("x", [[]])}
        )
        socket = _Socket()
        await transfer.send_kv_to_decode(b"a", msgspec.msgpack.encode(request), socket)
        self.assertEqual(socket.messages[0].ok_reqs, ["d-r"])
        self.assertIsNone((await transfer.fetch_finished_sending_reqs())[0].error)

    async def test_recv_result_waits_for_future_terminal_and_invalidates_only_targets(
        self,
    ):
        transfer = _transfer()
        event = RecvEvent("d-r", "x", ((4, 5),), "bootstrap", "p", 2)
        transfer._recv_events["d-r"] = event
        transfer.is_producer, transfer.is_consumer = False, True
        transfer.receiver_loop = asyncio.get_running_loop()
        future = asyncio.get_running_loop().create_future()
        transfer._receive_futures["d-r"] = future
        self.assertEqual((await asyncio.to_thread(transfer.poll)).recvs, [])
        transfer._finish_receive(event, True, future)
        self.assertIsNotNone((await asyncio.to_thread(transfer.poll)).recvs[0].error)
        self.assertEqual((await asyncio.to_thread(transfer.take_errors)), {4, 5})


class WorkerReleaseTest(unittest.TestCase):
    def _worker(self):
        worker = object.__new__(MTSCWorker)
        worker.pool = SimpleNamespace(
            poll=lambda: PoolPollResult(), take_errors=lambda: set()
        )
        worker.transfer = SimpleNamespace(
            poll=lambda: TransferPollResult(),
            take_errors=lambda: set(),
            recv=lambda event: None,
        )
        worker._on_save_requests = {}
        worker._finished_waits = {}
        worker._on_transfer = {}
        worker._on_load_requests = {}
        worker._ignored_recvs = set()
        worker._load_errors = set()
        return worker

    def test_intermediate_save_does_not_release_active_request(self):
        worker = self._worker()
        worker._on_save_requests["r"] = KVPoolSaveRequest("r", ([1],), [], 0, 16)
        worker.pool.poll = lambda: PoolPollResult(saves=[SaveResult("r")])
        self.assertEqual(
            worker.get_finished(set(), MTSCConnectorMetadata()), (None, None)
        )
        worker.pool.poll = lambda: PoolPollResult()
        metadata = MTSCConnectorMetadata(
            finished_waits={"r": FinishedWait(store_save=True)}
        )
        self.assertEqual(worker.get_finished({"r"}, metadata), ({"r"}, None))

    def test_new_save_resets_completion_before_request_end(self):
        worker = self._worker()
        worker.pool.save = lambda event: None
        metadata = MTSCConnectorMetadata(
            pool_saves=[KVPoolSaveRequest("r", ([1],), [b"a"], 0, 16)],
            finished_waits={"r": FinishedWait(store_save=True)},
        )
        ready = SimpleNamespace(record=lambda: None, synchronize=lambda: None)
        with patch("mtsc.worker.new_device_event", return_value=ready):
            self.assertEqual(worker.get_finished({"r"}, metadata), (None, None))
        worker.pool.poll = lambda: PoolPollResult(saves=[SaveResult("r")])
        self.assertEqual(
            worker.get_finished(set(), MTSCConnectorMetadata()), ({"r"}, None)
        )
