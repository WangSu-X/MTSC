from __future__ import annotations

import asyncio
import ctypes
import threading
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import msgspec
import torch
import zmq
import zmq.asyncio
from test_backends import _request, _Socket, _transfer

from mtsc.kv_transfer import RecvEvent, SendEvent, TransferRegion, _Session
from mtsc.protocol import KVTransferRequest, KVTransferResponse, KVTransferStatus


class BatchedTransferTest(unittest.IsolatedAsyncioTestCase):
    def _sender(self, write):
        transfer = _transfer()
        transfer._loop = asyncio.get_running_loop()
        transfer.regions = [TransferRegion("layer.0", 0, 0, 1000, 16, 16)]
        transfer.engine = SimpleNamespace(batch_transfer_sync_write=write)
        return transfer

    @staticmethod
    def _two_requests(transfer):
        return msgspec.structs.replace(
            _request(transfer), requests={"x": ("d-x", [[4]]), "y": ("d-y", [[5]])}
        )

    async def test_ready_requests_share_one_native_write_and_batch_failure(self):
        for ret in (0, -1):
            with self.subTest(ret=ret):
                calls = []

                def write(*args, calls=calls, ret=ret):
                    calls.append(args)
                    return ret

                transfer = self._sender(write)
                transfer.send(SendEvent("p-x", "x", ((1,),)))
                transfer.send(SendEvent("p-y", "y", ((2,),)))
                socket = _Socket()
                with ThreadPoolExecutor(max_workers=2) as pool:
                    transfer._send_pool = pool
                    await transfer._serve(
                        b"d",
                        msgspec.msgpack.encode(self._two_requests(transfer)),
                        socket,
                    )
                self.assertEqual(len(calls), 1)
                session, src, dst, sizes = calls[0]
                self.assertEqual(session, "host:123")
                self.assertEqual(
                    set(zip(src, dst, sizes)), {(1016, 2064, 16), (1032, 2080, 16)}
                )
                self.assertEqual(len(socket.messages), 1)
                response = socket.messages[0]
                self.assertEqual(response.status, KVTransferStatus.COMPLETE)
                ids = (
                    response.completed_transfer_ids
                    if ret == 0
                    else response.failed_transfer_ids
                )
                self.assertEqual(set(ids), {"x", "y"})
                results = transfer.poll().sends
                self.assertEqual(
                    {result.request_id for result in results}, {"p-x", "p-y"}
                )
                self.assertTrue(
                    all((result.error is None) == (ret == 0) for result in results)
                )

    async def test_unready_request_does_not_delay_ready_request(self):
        calls = []

        def write(*args):
            calls.append(args)
            return 0

        class Socket(_Socket):
            async def send_multipart(self, frames):
                await super().send_multipart(frames)
                first_response.set()

        first_response = asyncio.Event()
        transfer = self._sender(write)
        transfer.send(SendEvent("p-x", "x", ((1,),)))
        transfer.prepare("p-y", "y")
        socket = Socket()
        with ThreadPoolExecutor(max_workers=2) as pool:
            transfer._send_pool = pool
            serving = asyncio.create_task(
                transfer._serve(
                    b"d", msgspec.msgpack.encode(self._two_requests(transfer)), socket
                )
            )
            try:
                await asyncio.wait_for(first_response.wait(), 1)
                self.assertFalse(serving.done())
                self.assertEqual(len(calls), 1)
                self.assertEqual(
                    socket.messages[0].status, KVTransferStatus.IN_PROGRESS
                )
                self.assertEqual(socket.messages[0].completed_transfer_ids, ["x"])
                self.assertEqual(
                    {result.request_id for result in transfer.poll().sends}, {"p-x"}
                )
                await asyncio.to_thread(transfer.send, SendEvent("p-y", "y", ((2,),)))
                await asyncio.wait_for(serving, 1)
                self.assertEqual(len(calls), 2)
                self.assertEqual(socket.messages[-1].status, KVTransferStatus.COMPLETE)
                self.assertEqual(socket.messages[-1].completed_transfer_ids, ["y"])
            finally:
                transfer.cancel("p-y", "y")
                await serving

    async def test_invalid_request_is_excluded_from_native_batch(self):
        calls = []

        def write(*args):
            calls.append(args)
            return 0

        transfer = self._sender(write)
        transfer.send(SendEvent("p-x", "x", ((1,),)))
        transfer.send(SendEvent("p-y", "y", ((2,),)))
        request = msgspec.structs.replace(
            self._two_requests(transfer),
            requests={"x": ("d-x", [[4]]), "y": ("d-y", [[6]])},
        )
        socket = _Socket()
        with ThreadPoolExecutor(max_workers=1) as pool:
            transfer._send_pool = pool
            await transfer._serve(b"d", msgspec.msgpack.encode(request), socket)
        self.assertEqual(calls, [("host:123", [1016], [2064], [16])])
        self.assertEqual(socket.messages[0].completed_transfer_ids, ["x"])
        self.assertEqual(socket.messages[0].failed_transfer_ids, ["y"])

    async def test_ready_wait_timeout_finishes_all_pending_ids_without_writes(self):
        calls = []
        transfer = self._sender(lambda *args: calls.append(args) or 0)
        transfer.timeout = 0.01
        transfer.prepare("p-x", "x")
        transfer.prepare("p-y", "y")
        socket = _Socket()
        await asyncio.wait_for(
            transfer._serve(
                b"d", msgspec.msgpack.encode(self._two_requests(transfer)), socket
            ),
            1,
        )
        self.assertEqual(calls, [])
        self.assertEqual(socket.messages[0].status, KVTransferStatus.COMPLETE)
        self.assertEqual(set(socket.messages[0].failed_transfer_ids), {"x", "y"})
        self.assertEqual(transfer.poll().sends, [])
        for tid in ("x", "y"):
            self.assertEqual(transfer._sources[tid].terminal, 1)
            self.assertEqual(transfer._sources[tid].ready_waiters, set())
            transfer.send(SendEvent(f"p-{tid}", tid, ((1,),)))
        self.assertTrue(
            all(result.error is not None for result in transfer.poll().sends)
        )

    async def test_coalescing_requires_full_contiguous_blocks_on_both_sides(self):
        transfer = self._sender(lambda *args: 0)
        source = _Session("p-x", "x", block_ids=((0, 1),))
        request = msgspec.structs.replace(
            _request(transfer), requests={"x": ("d-x", [[2, 3]])}
        )
        src, dst, sizes, coverage = transfer._build_request_transfer_params(
            "x", source, request
        )
        self.assertEqual((src, dst, sizes, coverage), ([1000], [2032], [32], {0}))

        source.block_ids = ((0, 2),)
        self.assertEqual(
            transfer._build_request_transfer_params("x", source, request)[2], [16, 16]
        )
        source.block_ids = ((0, 1),)
        request = msgspec.structs.replace(request, requests={"x": ("d-x", [[2, 4]])})
        self.assertEqual(
            transfer._build_request_transfer_params("x", source, request)[2], [16, 16]
        )

        # Padding and TP slices cannot be copied as a contiguous full page.
        transfer.regions = [TransferRegion("layer.0", 0, 0, 1000, 32, 16)]
        request = msgspec.structs.replace(
            request, block_lengths=[32], requests={"x": ("d-x", [[2, 3]])}
        )
        self.assertEqual(
            transfer._build_request_transfer_params("x", source, request)[2], [16, 16]
        )
        transfer.regions = [TransferRegion("layer.0", 0, 0, 1000, 32, 32)]
        request = msgspec.structs.replace(
            request, tp_size=2, tp_rank=1, block_lengths=[16]
        )
        self.assertEqual(
            transfer._build_request_transfer_params("x", source, request)[2], [16, 16]
        )

    async def test_cancel_one_source_fences_shared_batch_and_preserves_sibling(self):
        entered, release = threading.Event(), threading.Event()

        def write(*args):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("Test did not release native WRITE")
            return 0

        transfer = self._sender(write)
        transfer.send(SendEvent("p-x", "x", ((1,),)))
        transfer.send(SendEvent("p-y", "y", ((2,),)))
        with ThreadPoolExecutor(max_workers=1) as pool:
            transfer._send_pool = pool
            serving = asyncio.create_task(
                transfer._serve(
                    b"d",
                    msgspec.msgpack.encode(self._two_requests(transfer)),
                    _Socket(),
                )
            )
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                self.assertEqual(transfer._sources["x"].active_writes, 1)
                self.assertEqual(transfer._sources["y"].active_writes, 1)
                cancelling = asyncio.create_task(
                    asyncio.to_thread(transfer.cancel, "p-x", "x")
                )
                await asyncio.sleep(0.01)
                self.assertFalse(cancelling.done())
                release.set()
                await asyncio.wait_for(asyncio.gather(serving, cancelling), 1)
                self.assertEqual(
                    {result.request_id for result in transfer.poll().sends}, {"p-y"}
                )
            finally:
                release.set()
                await serving

    async def test_malformed_sibling_waits_for_duplicate_write_fence(self):
        await self._check_malformed_batch_write_fence(("x", "y"))

    async def test_malformed_sibling_before_duplicate_waits_for_write_fence(self):
        await self._check_malformed_batch_write_fence(("y", "x"))

    async def test_malformed_batch_does_not_wait_for_its_own_unready_target(self):
        await self._check_malformed_batch_write_fence(("z", "y", "x"))

    async def _check_malformed_batch_write_fence(self, order):
        entered, release = threading.Event(), threading.Event()
        calls = []

        def write(*args):
            calls.append(args)
            entered.set()
            if not release.wait(3):
                raise TimeoutError("Test did not release native WRITE")
            return 0

        transfer = self._sender(write)
        transfer.send(SendEvent("p-x", "x", ((1,),)))
        source = transfer._sources["x"]
        original_request = msgspec.structs.replace(
            _request(transfer), requests={"x": ("d-x", [[4]])}
        )
        transfer.prepare("p-y", "y")
        transfer._sources["y"].peer = ("another-engine", 3, 1, 1)
        requests = {"x": ("d-x", [[4]]), "y": ("d-y", [[5]]), "z": ("d-z", [[3]])}
        malformed_request = msgspec.structs.replace(
            original_request, requests={tid: requests[tid] for tid in order}
        )
        original_socket, malformed_socket = _Socket(), _Socket()
        with ThreadPoolExecutor(max_workers=1) as pool:
            transfer._send_pool = pool
            original = asyncio.create_task(
                transfer._serve(
                    b"original",
                    msgspec.msgpack.encode(original_request),
                    original_socket,
                )
            )
            malformed = None
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                malformed = asyncio.create_task(
                    transfer._serve(
                        b"duplicate",
                        msgspec.msgpack.encode(malformed_request),
                        malformed_socket,
                    )
                )
                await asyncio.sleep(0.01)
                self.assertFalse(malformed.done())
                self.assertEqual(malformed_socket.messages, [])
                self.assertEqual(source.active_writes, 1)
                release.set()
                await asyncio.wait_for(asyncio.gather(original, malformed), 1)
                self.assertEqual(len(calls), 1)
                self.assertEqual(source.terminal, 1)
                self.assertEqual(
                    original_socket.messages[0].completed_transfer_ids, ["x"]
                )
                self.assertEqual(
                    malformed_socket.messages[0].status, KVTransferStatus.FAILED
                )
                if "z" in order:
                    owned = transfer._sources["z"]
                    self.assertEqual(owned.terminal, 1)
                    self.assertEqual(owned.ready_waiters, set())
                    self.assertTrue(
                        all(future.done() for future in owned.targets.values())
                    )
            finally:
                release.set()
                await original
                if malformed is not None:
                    await malformed

    async def test_incremental_receive_finishes_each_request_after_all_peers(self):
        peer_one_ready, peer_two_ready, finish_y = (
            asyncio.Event(),
            asyncio.Event(),
            asyncio.Event(),
        )
        payloads = []

        class Client:
            def __init__(self):
                self.round = 0

            def setsockopt(self, *args):
                pass

            def connect(self, address):
                self.address = address

            async def send(self, payload):
                payloads.append(msgspec.msgpack.decode(payload, type=KVTransferRequest))

            async def recv(self):
                self.round += 1
                if self.round == 1:
                    if self.address == "p0":
                        peer_one_ready.set()
                    else:
                        await peer_two_ready.wait()
                    response = KVTransferResponse(KVTransferStatus.IN_PROGRESS, ["x"])
                else:
                    await finish_y.wait()
                    response = KVTransferResponse(KVTransferStatus.COMPLETE, ["y"])
                return msgspec.msgpack.encode(response)

            def close(self, **kwargs):
                pass

        async def query(*args):
            return {0: {0: "p0", 1: "p1"}}

        transfer = _transfer()
        transfer.hostname, transfer.rpc_port = "d", 123
        transfer.regions = []
        transfer._ctx = SimpleNamespace(socket=lambda kind: Client())
        transfer._query_workers = query
        transfer._response_decoder = msgspec.msgpack.Decoder(KVTransferResponse)
        transfer.reformat_npu_blocks = lambda *args: None
        events = [
            RecvEvent(f"d-{tid}", tid, ((),), "bootstrap", "p", 2) for tid in ("x", "y")
        ]
        futures = {event.request_id: Future() for event in events}
        receiving = asyncio.create_task(transfer._receive_batch(events, futures))
        try:
            await asyncio.wait_for(peer_one_ready.wait(), 1)
            self.assertFalse(futures["d-x"].done())
            peer_two_ready.set()
            await asyncio.wait_for(asyncio.wrap_future(futures["d-x"]), 1)
            self.assertFalse(futures["d-y"].done())
            self.assertFalse(receiving.done())
            self.assertEqual(len(payloads), 2)
            self.assertTrue(
                all(set(request.requests) == {"x", "y"} for request in payloads)
            )
            finish_y.set()
            await asyncio.wait_for(receiving, 1)
            self.assertTrue(futures["d-y"].done())
            self.assertEqual(transfer._failed_recv, set())
        finally:
            peer_two_ready.set()
            finish_y.set()
            await receiving

    async def test_recv_batch_groups_by_remote_replica(self):
        payloads = []

        class Client:
            def setsockopt(self, *args):
                pass

            def connect(self, address):
                pass

            async def send(self, payload):
                self.request = msgspec.msgpack.decode(payload, type=KVTransferRequest)
                payloads.append(self.request)

            async def recv(self):
                return msgspec.msgpack.encode(
                    KVTransferResponse(
                        KVTransferStatus.COMPLETE, list(self.request.requests)
                    )
                )

            def close(self, **kwargs):
                pass

        async def query(*args):
            return {0: {0: "p"}}

        transfer = _transfer()
        transfer._loop = asyncio.get_running_loop()
        transfer.hostname, transfer.rpc_port = "d", 123
        transfer.regions = []
        transfer._ctx = SimpleNamespace(socket=lambda kind: Client())
        transfer._query_workers = query
        transfer._response_decoder = msgspec.msgpack.Decoder(KVTransferResponse)
        transfer.reformat_npu_blocks = lambda *args: None
        events = [
            RecvEvent(f"d-{tid}", tid, ((),), "bootstrap", "p", 2) for tid in ("x", "y")
        ]
        events.append(
            replace(events[0], request_id="d-z", transfer_id="z", remote_dp_rank=3)
        )
        transfer.recv_batch(events)
        await asyncio.wait_for(
            asyncio.gather(
                *(
                    asyncio.wrap_future(future)
                    for future in transfer._receive_futures.values()
                )
            ),
            1,
        )
        self.assertEqual(len(payloads), 2)
        self.assertEqual(
            {request.remote_dp_rank: set(request.requests) for request in payloads},
            {2: {"x", "y"}, 3: {"z"}},
        )
        self.assertEqual(
            {result.request_id for result in transfer.poll().recvs},
            {"d-x", "d-y", "d-z"},
        )

    async def test_partial_failure_and_preemption_preserve_other_batch_request(self):
        release = asyncio.Event()

        class Client:
            def __init__(self):
                self.round = 0
                self.closed = False

            def setsockopt(self, *args):
                pass

            def connect(self, address):
                pass

            async def send(self, payload):
                request = msgspec.msgpack.decode(payload, type=KVTransferRequest)
                self.request_ids = set(request.requests)

            async def recv(self):
                self.round += 1
                if self.round == 1:
                    response = KVTransferResponse(
                        KVTransferStatus.IN_PROGRESS, ["x"], covered_regions={"x": [0]}
                    )
                else:
                    await release.wait()
                    response = KVTransferResponse(
                        KVTransferStatus.COMPLETE, failed_transfer_ids=["y"]
                    )
                return msgspec.msgpack.encode(response)

            def close(self, **kwargs):
                self.closed = True

        async def query(*args):
            return {0: {0: "p"}}

        transfer = _transfer()
        transfer._loop = asyncio.get_running_loop()
        transfer.hostname, transfer.rpc_port = "d", 123
        transfer.regions = [TransferRegion("layer.0", 0, 0, 2000, 16, 16)]
        client = Client()
        transfer._ctx = SimpleNamespace(socket=lambda kind: client)
        transfer._query_workers = query
        transfer._response_decoder = msgspec.msgpack.Decoder(KVTransferResponse)
        transfer.reformat_npu_blocks = lambda *args: None
        transfer.recv_batch(
            [
                RecvEvent("d-x", "x", ((4,),), "bootstrap", "p", 2),
                RecvEvent("d-y", "y", ((5,),), "bootstrap", "p", 2),
            ]
        )
        first, second = (
            transfer._receive_futures["d-x"],
            transfer._receive_futures["d-y"],
        )
        try:
            await asyncio.wait_for(asyncio.wrap_future(first), 1)
            results = transfer.poll().recvs
            self.assertEqual([result.request_id for result in results], ["d-x"])
            self.assertIsNone(results[0].error)
            self.assertFalse(second.done())
            transfer.preempt("d-x")
            self.assertFalse(client.closed)
            self.assertEqual(transfer.take_errors(), set())
            release.set()
            await asyncio.wait_for(asyncio.wrap_future(second), 1)
            results = transfer.poll().recvs
            self.assertEqual([result.request_id for result in results], ["d-y"])
            self.assertIsNotNone(results[0].error)
            self.assertEqual(transfer.take_errors(), {5})
            self.assertEqual(client.request_ids, {"x", "y"})
        finally:
            release.set()
            await asyncio.wrap_future(second)

    async def test_listener_uses_fixed_workers_and_drains_message_queue(self):
        transfer = _transfer()
        transfer.hostname = "127.0.0.1"
        transfer._serve_tasks = set()
        transfer._ctx = zmq.asyncio.Context()
        ready = threading.Event()
        release, entered = asyncio.Event(), asyncio.Event()
        active, peak = 0, 0

        async def register(port):
            transfer.side_port = port

        async def serve(identity, payload, socket):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if active == transfer.num_sender_tasks:
                entered.set()
            try:
                await release.wait()
                await socket.send_multipart((identity, b"done"))
            finally:
                active -= 1

        transfer._register_worker = register
        transfer._serve = serve
        listener = asyncio.create_task(transfer._listen(ready))
        client = transfer._ctx.socket(zmq.DEALER)
        client.setsockopt(zmq.LINGER, 0)
        try:
            self.assertTrue(await asyncio.to_thread(ready.wait, 1))
            client.connect(f"tcp://127.0.0.1:{transfer.side_port}")
            for _ in range(20):
                await client.send(b"request")
            await asyncio.wait_for(entered.wait(), 1)
            self.assertEqual(len(transfer._serve_tasks), 2)
            release.set()
            for _ in range(20):
                self.assertEqual(await asyncio.wait_for(client.recv(), 1), b"done")
            await asyncio.wait_for(transfer.sender_worker_queue.join(), 1)
            self.assertEqual(peak, 2)
        finally:
            release.set()
            listener.cancel()
            await listener
            client.close(linger=0)
            transfer._ctx.destroy(linger=0)

    async def test_sender_worker_survives_a_failed_control_response(self):
        transfer = _transfer()
        processed = []

        async def serve(identity, payload, socket):
            processed.append(payload)
            if payload == b"fail":
                raise RuntimeError("Control response failed")

        transfer._serve = serve
        worker = asyncio.create_task(transfer._sender_worker(None))
        try:
            await transfer.sender_worker_queue.put((b"d", b"fail"))
            await transfer.sender_worker_queue.put((b"d", b"next"))
            await asyncio.wait_for(transfer.sender_worker_queue.join(), 1)
            self.assertEqual(processed, [b"fail", b"next"])
            self.assertFalse(worker.done())
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    async def test_listener_shutdown_fences_inflight_batch_before_source_release(self):
        entered, release = threading.Event(), threading.Event()

        def write(*args):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("Test did not release native WRITE")
            return 0

        transfer = self._sender(write)
        transfer.hostname = "127.0.0.1"
        transfer._serve_tasks = set()
        transfer._ctx = zmq.asyncio.Context()
        transfer.send(SendEvent("p-x", "x", ((1,),)))
        transfer.send(SendEvent("p-y", "y", ((2,),)))
        sources = list(transfer._sources.values())

        async def register(port):
            transfer.side_port = port

        transfer._register_worker = register
        ready = threading.Event()
        client = transfer._ctx.socket(zmq.DEALER)
        client.setsockopt(zmq.LINGER, 0)
        with ThreadPoolExecutor(max_workers=1) as pool:
            transfer._send_pool = pool
            listener = asyncio.create_task(transfer._listen(ready))
            try:
                self.assertTrue(await asyncio.to_thread(ready.wait, 1))
                client.connect(f"tcp://127.0.0.1:{transfer.side_port}")
                await client.send(msgspec.msgpack.encode(self._two_requests(transfer)))
                self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                listener.cancel()
                await asyncio.sleep(0.01)
                self.assertFalse(listener.done())
                self.assertTrue(all(source.active_writes == 1 for source in sources))
                self.assertEqual(transfer.poll().sends, [])
                release.set()
                response = msgspec.msgpack.decode(
                    await asyncio.wait_for(client.recv(), 1), type=KVTransferResponse
                )
                self.assertEqual(set(response.completed_transfer_ids), {"x", "y"})
                await asyncio.wait_for(listener, 1)
                self.assertTrue(all(source.active_writes == 0 for source in sources))
                self.assertTrue(all(source.terminal == 1 for source in sources))
            finally:
                release.set()
                listener.cancel()
                await listener
                client.close(linger=0)
                transfer._ctx.destroy(linger=0)

    async def test_batched_pull_over_zmq_copies_both_requests_with_one_te_call(self):
        source = torch.arange(96, dtype=torch.uint8).reshape(6, 16)
        destination = torch.zeros_like(source)
        calls = []

        def write(session, src, dst, sizes):
            calls.append((session, src, dst, sizes))
            for source_address, target_address, size in zip(
                src, dst, sizes, strict=True
            ):
                ctypes.memmove(target_address, source_address, size)
            return 0

        producer = self._sender(write)
        producer.hostname = "127.0.0.1"
        producer.regions = [TransferRegion("layer.0", 0, 0, source.data_ptr(), 16, 16)]
        producer._serve_tasks = set()
        producer._ctx = zmq.asyncio.Context()
        producer.send(SendEvent("p-x", "x", ((1,),)))
        producer.send(SendEvent("p-y", "y", ((2,),)))

        async def register(port):
            producer.side_port = port

        async def query(*args):
            return {0: {0: f"tcp://127.0.0.1:{producer.side_port}"}}

        producer._register_worker = register
        consumer = _transfer()
        consumer._loop = asyncio.get_running_loop()
        consumer.hostname, consumer.rpc_port = "d", 456
        consumer.engine_id, consumer.dp_rank = "d", 3
        consumer.regions = [
            TransferRegion("layer.0", 0, 0, destination.data_ptr(), 16, 16)
        ]
        consumer._ctx = zmq.asyncio.Context()
        consumer._query_workers = query
        consumer._response_decoder = msgspec.msgpack.Decoder(KVTransferResponse)
        consumer.reformat_npu_blocks = lambda *args: None
        ready = threading.Event()
        with ThreadPoolExecutor(max_workers=2) as pool:
            producer._send_pool = pool
            listener = asyncio.create_task(producer._listen(ready))
            try:
                self.assertTrue(await asyncio.to_thread(ready.wait, 1))
                consumer.recv_batch(
                    [
                        RecvEvent("d-x", "x", ((4,),), "bootstrap", "p", 2),
                        RecvEvent("d-y", "y", ((5,),), "bootstrap", "p", 2),
                    ]
                )
                await asyncio.wait_for(
                    asyncio.gather(
                        *(
                            asyncio.wrap_future(future)
                            for future in consumer._receive_futures.values()
                        )
                    ),
                    2,
                )
                self.assertEqual(len(calls), 1)
                self.assertTrue(torch.equal(destination[4], source[1]))
                self.assertTrue(torch.equal(destination[5], source[2]))
                self.assertTrue(torch.all(destination[:4] == 0))
                self.assertTrue(
                    all(result.error is None for result in consumer.poll().recvs)
                )
                self.assertEqual(
                    {result.request_id for result in producer.poll().sends},
                    {"p-x", "p-y"},
                )
                await asyncio.wait_for(producer.sender_worker_queue.join(), 1)
            finally:
                listener.cancel()
                await listener
                producer._ctx.destroy(linger=0)
                consumer._ctx.destroy(linger=0)
