"""Wire compatibility with the vLLM Mooncake implementation."""

from __future__ import annotations

import asyncio
import sys
import unittest
from concurrent.futures import Future
from itertools import product
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import msgspec
import torch
import zmq
from reference_mooncake import (
    MooncakeXferMetadata,
    MooncakeXferResponse,
    MooncakeXferResponseStatus,
)
from test_backends import _transfer
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    MooncakeConnectorWorker,
    get_mooncake_bootstrap_addr,
    should_launch_bootstrap_server,
)

from mtsc.kv_transfer import (
    MooncakeKVTransfer,
    RecvEvent,
    _bootstrap_address,
    _launch_bootstrap,
    _ReceiveFenceError,
    _Session,
)
from mtsc.protocol import KVTransferRequest, KVTransferResponse, KVTransferStatus


class OfficialTransferTest(unittest.TestCase):
    def test_request_wire_compatible_in_both_directions(self):
        fields = {
            "remote_hostname": "decode",
            "remote_port": 123,
            "remote_tp_size": 1,
            "remote_tp_rank": 0,
            "req_blocks": {"decode-request": ("transfer-attempt", [[4, 5], []])},
            "kv_caches_base_addr": [1000, 2000],
            "block_lens": [16, 32],
            "kv_block_lens": [16, 32],
            "registered_layer_names": ["layer.0", "layer.1"],
            "registered_layer_indices": [0, 1],
            "registered_group_indices": [0, 1],
            "registered_shared_group_ids": [[0], [1]],
            "registered_row_offsets": [-1, -1],
            "layout_id": 123,
        }
        ours, official = KVTransferRequest(**fields), MooncakeXferMetadata(**fields)
        self.assertEqual(msgspec.msgpack.encode(ours), msgspec.msgpack.encode(official))
        for source, target in (
            (ours, MooncakeXferMetadata),
            (official, KVTransferRequest),
        ):
            decoded = msgspec.msgpack.decode(
                msgspec.msgpack.encode(source), type=target
            )
            self.assertEqual(decoded.req_blocks, fields["req_blocks"])

    def test_response_wire_compatible_for_all_statuses(self):
        for status in KVTransferStatus:
            ours = KVTransferResponse(status, ["decode-ok"], ["decode-error"], "error")
            official = MooncakeXferResponse(
                MooncakeXferResponseStatus(status),
                ["decode-ok"],
                ["decode-error"],
                "error",
            )
            self.assertEqual(
                msgspec.msgpack.encode(ours), msgspec.msgpack.encode(official)
            )
            self.assertEqual(
                msgspec.msgpack.decode(
                    msgspec.msgpack.encode(official), type=KVTransferResponse
                ),
                ours,
            )
            self.assertEqual(
                msgspec.msgpack.decode(
                    msgspec.msgpack.encode(ours), type=MooncakeXferResponse
                ),
                official,
            )

    def test_pipeline_parallelism_rejected_before_engine_initialization(self):
        factory = Mock()
        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(engine_id="p"),
            parallel_config=SimpleNamespace(pipeline_parallel_size=2),
        )
        with patch.dict(
            sys.modules, {"mooncake.engine": SimpleNamespace(TransferEngine=factory)}
        ):
            with self.assertRaisesRegex(ValueError, "pipeline parallelism"):
                MooncakeKVTransfer(config, None)
        factory.assert_not_called()


class OfficialBootstrapTest(unittest.TestCase):
    def test_bootstrap_address_matches_official(self):
        for local_only, nodes in product((False, True), (1, 2, 4)):
            with self.subTest(local_only=local_only, nodes=nodes):
                config = SimpleNamespace(
                    parallel_config=SimpleNamespace(
                        local_engines_only=local_only,
                        nnodes_within_dp=nodes,
                        master_addr="10.0.0.1",
                        data_parallel_master_ip="10.0.0.2",
                    )
                )
                self.assertEqual(
                    _bootstrap_address(config), get_mooncake_bootstrap_addr(config)
                )

    def test_bootstrap_launch_matches_official_rank_selection(self):
        official_module = (
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector"
        )
        for first, local_only, dp_rank, local_dp, tp_rank in product(
            (False, True),
            (False, True),
            (0, 1),
            (0, 1, None),
            (0, 4),
        ):
            with self.subTest(
                first=first,
                local_only=local_only,
                dp_rank=dp_rank,
                local_dp=local_dp,
                tp_rank=tp_rank,
            ):
                config = SimpleNamespace(
                    parallel_config=SimpleNamespace(
                        local_engines_only=local_only,
                        data_parallel_index=dp_rank,
                        data_parallel_rank_local=local_dp,
                    )
                )
                with (
                    patch("mtsc.kv_transfer.is_local_first_rank", return_value=first),
                    patch(official_module + ".is_local_first_rank", return_value=first),
                    patch(
                        "mtsc.kv_transfer.get_tensor_model_parallel_rank",
                        return_value=tp_rank,
                    ),
                ):
                    self.assertEqual(
                        _launch_bootstrap(config),
                        should_launch_bootstrap_server(config),
                    )


class BootstrapQueryTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.transfer = object.__new__(MooncakeKVTransfer)
        self.transfer._remote_workers = {}
        self.transfer._pending_bootstrap_queries = {}
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.response = Mock()
        self.response.json.return_value = {
            str(rank): {
                "engine_id": f"p{rank}",
                "worker_addr": {"0": {"0": f"tcp://p{rank}"}},
            }
            for rank in (0, 1)
        }

        async def get(url):
            self.entered.set()
            await self.release.wait()
            return self.response

        self.client = AsyncMock()
        self.client.get.side_effect = get
        context = AsyncMock()
        context.__aenter__.return_value = self.client
        patcher = patch("mtsc.kv_transfer.httpx.AsyncClient", return_value=context)
        self.factory = patcher.start()
        self.addCleanup(patcher.stop)

    async def test_twenty_concurrent_queries_share_one_get_across_dp_ranks(self):
        queries = [
            asyncio.create_task(
                self.transfer._query_workers(
                    "http://bootstrap" + "/" * (index % 3),
                    f"p{index % 2}",
                    index % 2,
                )
            )
            for index in range(20)
        ]
        try:
            await asyncio.wait_for(self.entered.wait(), 1)
            await asyncio.sleep(0)
            self.client.get.assert_awaited_once_with("http://bootstrap/query")
            self.release.set()
            results = await asyncio.wait_for(asyncio.gather(*queries), 1)
            self.assertEqual(
                results, [{0: {0: f"tcp://p{index % 2}"}} for index in range(20)]
            )
            self.assertEqual(self.transfer._pending_bootstrap_queries, {})
            for rank in (0, 1):
                await self.transfer._query_workers(
                    "http://bootstrap/", f"p{rank}", rank
                )
            self.client.get.assert_awaited_once()
            self.factory.assert_called_once_with(trust_env=False)
        finally:
            self.release.set()
            await asyncio.gather(*queries, return_exceptions=True)

    async def test_failed_query_releases_all_waiters_and_can_retry(self):
        async def fail(url):
            self.entered.set()
            await self.release.wait()
            raise ConnectionError("bootstrap unavailable")

        self.client.get.side_effect = fail
        queries = [
            asyncio.create_task(
                self.transfer._query_workers("http://bootstrap", "p0", 0)
            )
            for _ in range(5)
        ]
        try:
            await asyncio.wait_for(self.entered.wait(), 1)
            await asyncio.sleep(0)
            self.release.set()
            results = await asyncio.wait_for(
                asyncio.gather(*queries, return_exceptions=True), 1
            )
            self.assertTrue(
                all(isinstance(result, ConnectionError) for result in results)
            )
            self.client.get.assert_awaited_once()
            self.assertEqual(self.transfer._pending_bootstrap_queries, {})
            self.assertEqual(self.transfer._remote_workers, {})
            self.client.get.side_effect = None
            self.client.get.return_value = self.response
            self.assertEqual(
                await self.transfer._query_workers("http://bootstrap", "p0", 0),
                {0: {0: "tcp://p0"}},
            )
            self.assertEqual(self.client.get.await_count, 2)
        finally:
            self.release.set()
            await asyncio.gather(*queries, return_exceptions=True)

    async def test_cancelled_follower_does_not_cancel_shared_query(self):
        owner = asyncio.create_task(
            self.transfer._query_workers("http://bootstrap", "p0", 0)
        )
        follower = None
        try:
            await asyncio.wait_for(self.entered.wait(), 1)
            follower = asyncio.create_task(
                self.transfer._query_workers("http://bootstrap", "p0", 0)
            )
            await asyncio.sleep(0)
            follower.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await follower
            self.assertFalse(owner.done())
            self.release.set()
            self.assertEqual(await asyncio.wait_for(owner, 1), {0: {0: "tcp://p0"}})
            self.client.get.assert_awaited_once()
            self.assertEqual(self.transfer._pending_bootstrap_queries, {})
        finally:
            self.release.set()
            await asyncio.gather(
                owner, *([follower] if follower else []), return_exceptions=True
            )

    async def test_cancelled_owner_wakes_followers_and_allows_retry(self):
        owner = asyncio.create_task(
            self.transfer._query_workers("http://bootstrap", "p0", 0)
        )
        follower = None
        try:
            await asyncio.wait_for(self.entered.wait(), 1)
            follower = asyncio.create_task(
                self.transfer._query_workers("http://bootstrap", "p0", 0)
            )
            await asyncio.sleep(0)
            owner.cancel()
            results = await asyncio.wait_for(
                asyncio.gather(owner, follower, return_exceptions=True), 1
            )
            self.assertTrue(
                all(isinstance(result, asyncio.CancelledError) for result in results)
            )
            self.assertEqual(self.transfer._pending_bootstrap_queries, {})
            self.release.set()
            self.assertEqual(
                await self.transfer._query_workers("http://bootstrap", "p0", 0),
                {0: {0: "tcp://p0"}},
            )
            self.assertEqual(self.client.get.await_count, 2)
        finally:
            self.release.set()
            await asyncio.gather(
                owner, *([follower] if follower else []), return_exceptions=True
            )

    async def test_bad_engine_identity_does_not_poison_other_waiters(self):
        wrong = asyncio.create_task(
            self.transfer._query_workers("http://bootstrap", "wrong", 0)
        )
        correct = asyncio.create_task(
            self.transfer._query_workers("http://bootstrap", "p1", 1)
        )
        try:
            await asyncio.wait_for(self.entered.wait(), 1)
            await asyncio.sleep(0)
            self.release.set()
            results = await asyncio.wait_for(
                asyncio.gather(wrong, correct, return_exceptions=True), 1
            )
            self.assertIsInstance(results[0], KeyError)
            self.assertEqual(results[1], {0: {0: "tcp://p1"}})
            self.assertNotIn(
                ("http://bootstrap", "wrong", 0), self.transfer._remote_workers
            )
            self.client.get.assert_awaited_once()
        finally:
            self.release.set()
            await asyncio.gather(wrong, correct, return_exceptions=True)

    async def test_different_bootstrap_servers_query_independently(self):
        queries = [
            asyncio.create_task(self.transfer._query_workers(address, "p0", 0))
            for address in ("http://first", "http://second")
        ]
        try:
            await asyncio.wait_for(self.entered.wait(), 1)
            await asyncio.sleep(0)
            self.assertEqual(self.client.get.await_count, 2)
            self.release.set()
            await asyncio.wait_for(asyncio.gather(*queries), 1)
            self.assertEqual(self.transfer._pending_bootstrap_queries, {})
        finally:
            self.release.set()
            await asyncio.gather(*queries, return_exceptions=True)


class OfficialRegistrationTest(unittest.TestCase):
    @staticmethod
    def _register(caches, split=False):
        transfer = _transfer()
        transfer._registered = False
        transfer.is_producer = False
        transfer.num_blocks = 999  # Tensor geometry takes precedence over config.
        transfer.layer_specs = {name: SimpleNamespace() for name in caches}
        transfer.engine = Mock()
        transfer.engine.batch_register_memory.return_value = 0
        transfer.topology = SimpleNamespace(
            virtually_split_kv_in_blocks=False,
            local_replicates_kv_cache=False,
            total_num_kv_heads=8,
            is_mla=False,
            get_transfer_cache_regions=lambda cache, spec: (
                list(cache) if split else [cache]
            ),
        )
        with patch("mtsc.kv_transfer.is_npu_platform", return_value=False):
            transfer.register(caches)
        return transfer

    def _compare_official(self, caches, split=False):
        transfer = self._register(caches, split)
        official = SimpleNamespace(
            use_mla=False,
            transfer_topo=SimpleNamespace(split_k_and_v=split),
            _log_debug_cache_registration=lambda *args: None,
            engine=Mock(),
            is_kv_consumer=True,
        )
        official.engine.batch_register_memory.return_value = 0
        MooncakeConnectorWorker.register_kv_caches(official, caches)
        self.assertEqual(transfer.num_blocks, official.num_blocks)
        self.assertEqual(transfer.kv_caches_base_addr, official.kv_caches_base_addr)
        self.assertEqual(transfer.block_lens, official.block_len_per_layer)
        # Current upstream registers each whole storage once, whereas v0.23
        # registered the K/V views separately. The transfer addresses agree.
        storages = {
            cache.untyped_storage().data_ptr(): cache.untyped_storage().nbytes()
            for cache in caches.values()
        }
        transfer.engine.batch_register_memory.assert_called_once_with(
            list(storages), list(storages.values())
        )
        return transfer

    def test_kv_first_cache_infers_blocks_after_splitting(self):
        cache = torch.empty((2, 6, 16))
        transfer = self._compare_official({"layer.0": cache}, split=True)
        self.assertEqual(transfer.num_blocks, 6)
        self.assertEqual(len(transfer.kv_caches_base_addr), 2)
        request = KVTransferRequest(
            "d",
            123,
            1,
            0,
            {"d": ("x", [[4]])},
            transfer.kv_caches_base_addr,
            transfer.block_lens,
        )
        src, _, lengths = transfer._build_request_transfer_params(
            "d",
            _Session("p", "x", block_ids=((3,),)),
            request,
        )
        self.assertEqual(src, [part.data_ptr() + 3 * 64 for part in cache])
        self.assertEqual(lengths, [64, 64])

    def test_shared_cache_is_registered_and_transferred_once(self):
        cache = torch.empty((6, 16))
        transfer = self._compare_official({"layer.0": cache, "layer.1": cache})
        self.assertEqual(len(transfer.regions), 1)
        self.assertEqual(len(transfer._registered_storage), 1)

    def test_padded_cache_registers_stride_sized_blocks(self):
        cache = torch.empty((6, 32))[:, :16]
        transfer = self._compare_official({"layer.0": cache})
        self.assertEqual(transfer.block_lens, [128])
        transfer.engine.batch_register_memory.assert_called_once_with(
            [cache.data_ptr()],
            [6 * 128],
        )

    def test_inconsistent_block_counts_are_rejected(self):
        with self.assertRaisesRegex(AssertionError, "same number of blocks"):
            self._register(
                {"layer.0": torch.empty((6, 16)), "layer.1": torch.empty((5, 16))}
            )


class ReceiveTimeoutTest(unittest.IsolatedAsyncioTestCase):
    async def test_real_zmq_silent_peer_has_bounded_receive_wait(self):
        import zmq.asyncio

        context = zmq.asyncio.Context()
        router = context.socket(zmq.ROUTER)
        port = router.bind_to_random_port("tcp://127.0.0.1")
        configured_timeouts = []

        class ShortTimeoutSocket:
            def __init__(self, socket):
                self.socket = socket

            def __getattr__(self, name):
                return getattr(self.socket, name)

            def setsockopt(self, option, value):
                if option == zmq.RCVTIMEO:
                    configured_timeouts.append(value)
                    value = 20  # Exercise actual ZMQ expiry without a 61s test.
                self.socket.setsockopt(option, value)

        transfer = _transfer()
        transfer.hostname, transfer.rpc_port = "d", 123
        transfer._ctx = SimpleNamespace(
            socket=lambda kind: ShortTimeoutSocket(context.socket(kind))
        )
        transfer._query_workers = AsyncMock(
            return_value={0: {0: f"tcp://127.0.0.1:{port}"}}
        )
        transfer._response_decoder = msgspec.msgpack.Decoder(KVTransferResponse)
        event = RecvEvent("d", "x", ((4,),), "bootstrap", "p", 0)
        future = Future()
        transfer._receive_futures["d"] = future
        try:
            await asyncio.wait_for(transfer._receive_batch([event], {"d": future}), 1)
            self.assertEqual(configured_timeouts, [61000])
            with self.assertRaises(_ReceiveFenceError):
                future.result()
            with self.assertRaises(_ReceiveFenceError):
                transfer.poll()
        finally:
            router.close(linger=0)
            context.destroy(linger=0)

    async def test_lost_fence_stops_worker_and_wakes_other_batch_preemption(self):
        for transport_error in (zmq.Again(), ConnectionError("P disconnected")):
            with self.subTest(error=type(transport_error).__name__):
                release, receiving = asyncio.Event(), asyncio.Event()
                options = {}

                class Client:
                    def setsockopt(self, option, value, options=options):
                        options[option] = value

                    def connect(self, address):
                        pass

                    async def send(self, payload):
                        pass

                    async def recv(
                        self,
                        receiving=receiving,
                        release=release,
                        transport_error=transport_error,
                    ):
                        receiving.set()
                        await release.wait()
                        raise transport_error

                    def close(self, **kwargs):
                        pass

                transfer = _transfer()
                transfer.hostname, transfer.rpc_port = "d", 123
                transfer._ctx = SimpleNamespace(socket=lambda kind: Client())
                transfer._query_workers = AsyncMock(return_value={0: {0: "p"}})
                transfer._response_decoder = msgspec.msgpack.Decoder(KVTransferResponse)
                transfer.engine = Mock()
                events = [RecvEvent("d", "x", ((4,),), "bootstrap", "p", 0)]
                current, other = Future(), Future()
                transfer._receive_futures = {"d": current, "other": other}
                transfer._recv_events = {"d": events[0]}
                task = asyncio.create_task(
                    transfer._receive_batch(events, {"d": current})
                )
                fence = None
                try:
                    await asyncio.wait_for(receiving.wait(), 1)
                    self.assertEqual(options[zmq.RCVTIMEO], 61000)
                    fence = asyncio.create_task(
                        asyncio.to_thread(transfer.preempt, "other")
                    )
                    await asyncio.sleep(0.01)
                    self.assertFalse(fence.done())
                    release.set()
                    await asyncio.wait_for(task, 1)
                    with self.assertRaises(_ReceiveFenceError):
                        await asyncio.wait_for(fence, 1)
                    for future in (current, other):
                        with self.assertRaises(_ReceiveFenceError):
                            future.result()
                    for operation in (
                        transfer.poll,
                        transfer.take_errors,
                        transfer.close,
                    ):
                        with self.assertRaisesRegex(
                            _ReceiveFenceError, "worker must stop"
                        ):
                            operation()
                    self.assertEqual(transfer._invalid, {})
                    self.assertEqual(transfer._failed_recv, set())
                    self.assertIn("d", transfer._recv_events)
                    transfer.engine.batch_unregister_memory.assert_not_called()
                finally:
                    release.set()
                    if not other.done():
                        other.set_result(None)
                    await asyncio.gather(
                        task, *([fence] if fence else []), return_exceptions=True
                    )

    async def test_bootstrap_failure_before_publishing_is_recoverable(self):
        transfer = _transfer()
        transfer._query_workers = AsyncMock(
            side_effect=ConnectionError("bootstrap down")
        )
        event = RecvEvent("d", "x", ((4,),), "bootstrap", "p", 0)
        future = Future()
        transfer._receive_futures["d"] = future
        transfer._recv_events["d"] = event
        await transfer._receive_batch([event], {"d": future})
        self.assertIsNone(transfer._receive_error)
        self.assertIsNone(future.result())
        self.assertIsNotNone(transfer.poll().recvs[0].error)
        self.assertEqual(transfer.take_errors(), {4})
