"""Wire compatibility with the vLLM Mooncake implementation."""

from __future__ import annotations

import asyncio
import sys
import unittest
from itertools import product
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import msgspec
import torch
import zmq
from test_backends import _transfer
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    MooncakeConnectorWorker,
    MooncakeXferMetadata,
    MooncakeXferResponse,
    MooncakeXferResponseStatus,
    get_mooncake_bootstrap_addr,
    should_launch_bootstrap_server,
)

from mtsc.kv_transfer import (
    MooncakeKVTransfer,
    PullReqMeta,
    RecvEvent,
    SendBlockMeta,
    TransferRegion,
    _bootstrap_address,
    _launch_bootstrap,
    _ReceiveFenceError,
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

    def test_mamba_rejected_before_engine_initialization(self):
        factory = Mock()
        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(engine_id="p"),
            parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        )
        with patch.dict(
            sys.modules, {"mooncake.engine": SimpleNamespace(TransferEngine=factory)}
        ):
            with self.assertRaisesRegex(ValueError, "Mamba KV cache"):
                MooncakeKVTransfer(config, SimpleNamespace(has_mamba_layers=True))
        factory.assert_not_called()


class GroupTransferPlanTest(unittest.IsolatedAsyncioTestCase):
    async def test_group_trimming_and_flattening_matches_official(self):
        for p_size, p_rank, d_size, d_rank in (
            (1, 0, 1, 0),
            (2, 0, 4, 1),
            (4, 1, 2, 0),
        ):
            with self.subTest(p_size=p_size, d_size=d_size):
                transfer = _transfer()
                transfer.tp_size, transfer.tp_rank = p_size, p_rank
                local_len, remote_len = 64 // p_size, 64 // d_size
                transfer.kv_caches_base_addr = [1000, 2000]
                transfer.block_len_per_layer = [local_len, local_len]
                remote_regions = [
                    TransferRegion(base, remote_len, remote_len)
                    for base in (3000, 4000)
                ]
                send_meta = SendBlockMeta(
                    "p", "x", local_block_ids=((0, 1, 2), (), (3,))
                )
                request = KVTransferRequest(
                    "d",
                    123,
                    d_size,
                    d_rank,
                    {"d": ("x", [[4], [], [5]])},
                    [r.base_addr for r in remote_regions],
                    [r.block_len for r in remote_regions],
                )
                official = SimpleNamespace(
                    tp_size=p_size,
                    tp_rank=p_rank,
                    _producer_cache_is_replicated=lambda: False,
                )
                official._get_sender_transfer_plan = (
                    MooncakeConnectorWorker._get_sender_transfer_plan.__get__(official)
                )
                expected = await MooncakeConnectorWorker._build_transfer_params(
                    official,
                    [("d", send_meta)],
                    request,
                    transfer._get_transfer_regions(
                        transfer.kv_caches_base_addr, transfer.block_len_per_layer
                    ),
                    remote_regions,
                )
                actual = transfer._build_request_transfer_params(
                    "d", send_meta, request, remote_regions
                )
                self.assertEqual(actual, expected[:3])
                self.assertEqual(expected[3:], ([], None))

    def test_empty_destination_group_does_not_copy_source_blocks(self):
        transfer = _transfer()
        send_meta = SendBlockMeta("p", "x", local_block_ids=((0, 1), (2, 3)))
        request = KVTransferRequest(
            "d", 123, 1, 0, {"d": ("x", [[], [4]])}, [2000], [16]
        )
        self.assertEqual(
            transfer._build_request_transfer_params("d", send_meta, request),
            ([1048], [2064], [16]),
        )

    def test_group_count_mismatch_is_rejected_before_write(self):
        transfer = _transfer()
        send_meta = SendBlockMeta("p", "x", local_block_ids=((1,), (2,)))
        request = KVTransferRequest("d", 123, 1, 0, {"d": ("x", [[4]])}, [2000], [16])
        with self.assertRaisesRegex(ValueError, "KV group count mismatch"):
            transfer._build_request_transfer_params("d", send_meta, request)


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
        self.transfer = _transfer()
        self.transfer.receive_kv = Mock()
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

    def pull_metas(self, request_id="d", engine_id="p0", address="http://bootstrap"):
        event = RecvEvent(request_id, request_id, ((0,),), address, engine_id, 0)
        self.transfer._recv_events[request_id] = event
        self.transfer._receive_futures[request_id] = (
            asyncio.get_running_loop().create_future()
        )
        return {
            request_id: PullReqMeta(request_id, request_id, [[0]], engine_id, address)
        }

    async def test_twenty_concurrent_engines_share_one_get_and_cache_all_engines(self):
        queries = [
            asyncio.create_task(
                self.transfer.handle_new_engine_id(
                    f"p{index % 2}",
                    self.pull_metas(
                        f"d{index}",
                        f"p{index % 2}",
                        "http://bootstrap" + "/" * (index % 3),
                    ),
                )
            )
            for index in range(20)
        ]
        await asyncio.wait_for(self.entered.wait(), 1)
        await asyncio.sleep(0)
        self.client.get.assert_awaited_once_with("http://bootstrap/query")
        self.release.set()
        await asyncio.wait_for(asyncio.gather(*queries), 1)
        self.assertEqual(self.transfer._pending_bootstrap_queries, {})
        self.assertEqual(
            self.transfer._remote_agents,
            {
                "p0": {0: {0: "tcp://p0"}},
                "p1": {0: {0: "tcp://p1"}},
            },
        )
        self.assertEqual(self.transfer._remote_tp_size, {"p0": 1, "p1": 1})
        self.assertEqual(self.transfer.receive_kv.call_count, 20)
        for rank in (0, 1):
            metas = self.pull_metas(f"cached{rank}", f"p{rank}")
            await self.transfer._start_load_kv({f"p{rank}": metas})
            self.transfer.receive_kv.assert_called_with(f"p{rank}", metas)
        self.client.get.assert_awaited_once()
        self.factory.assert_called_once_with(trust_env=False)

    async def test_failed_query_releases_all_waiters_and_can_retry(self):
        async def fail(url):
            self.entered.set()
            await self.release.wait()
            raise ConnectionError("bootstrap unavailable")

        self.client.get.side_effect = fail
        queries = [
            asyncio.create_task(
                self.transfer.handle_new_engine_id("p0", self.pull_metas(f"d{index}"))
            )
            for index in range(5)
        ]
        await asyncio.wait_for(self.entered.wait(), 1)
        await asyncio.sleep(0)
        self.release.set()
        await asyncio.wait_for(asyncio.gather(*queries), 1)
        self.client.get.assert_awaited_once()
        self.assertEqual(self.transfer._pending_bootstrap_queries, {})
        self.assertEqual(self.transfer._remote_agents, {})
        self.assertEqual(self.transfer._failed_recv, {f"d{i}" for i in range(5)})
        self.assertTrue(all(f.done() for f in self.transfer._receive_futures.values()))
        self.transfer.receive_kv.assert_not_called()
        self.client.get.side_effect = None
        self.client.get.return_value = self.response
        metas = self.pull_metas("retry")
        await self.transfer.handle_new_engine_id("p0", metas)
        self.transfer.receive_kv.assert_called_once_with("p0", metas)
        self.assertEqual(self.client.get.await_count, 2)

    async def test_cancelled_follower_does_not_cancel_shared_query(self):
        owner = asyncio.create_task(
            self.transfer.handle_new_engine_id("p0", self.pull_metas("owner"))
        )
        await asyncio.wait_for(self.entered.wait(), 1)
        follower = asyncio.create_task(
            self.transfer.handle_new_engine_id("p0", self.pull_metas("follower"))
        )
        await asyncio.sleep(0)
        follower.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await follower
        self.assertFalse(owner.done())
        self.release.set()
        await asyncio.wait_for(owner, 1)
        self.client.get.assert_awaited_once()
        self.assertEqual(self.transfer._pending_bootstrap_queries, {})
        self.assertEqual(self.transfer.receive_kv.call_count, 1)

    async def test_cancelled_owner_wakes_followers_and_allows_retry(self):
        owner = asyncio.create_task(
            self.transfer.handle_new_engine_id("p0", self.pull_metas("owner"))
        )
        await asyncio.wait_for(self.entered.wait(), 1)
        follower = asyncio.create_task(
            self.transfer.handle_new_engine_id("p0", self.pull_metas("follower"))
        )
        await asyncio.sleep(0)
        owner.cancel()
        results = await asyncio.wait_for(
            asyncio.gather(owner, follower, return_exceptions=True), 1
        )
        self.assertIsInstance(results[0], asyncio.CancelledError)
        self.assertIsNone(results[1])
        self.assertEqual(self.transfer._failed_recv, {"follower"})
        self.assertEqual(self.transfer._pending_bootstrap_queries, {})
        self.release.set()
        metas = self.pull_metas("retry")
        await self.transfer.handle_new_engine_id("p0", metas)
        self.transfer.receive_kv.assert_called_once_with("p0", metas)
        self.assertEqual(self.client.get.await_count, 2)

    async def test_missing_engine_does_not_poison_other_waiters(self):
        wrong = asyncio.create_task(
            self.transfer.handle_new_engine_id(
                "wrong", self.pull_metas("wrong", "wrong")
            )
        )
        metas = self.pull_metas("correct", "p1")
        correct = asyncio.create_task(self.transfer.handle_new_engine_id("p1", metas))
        await asyncio.wait_for(self.entered.wait(), 1)
        await asyncio.sleep(0)
        self.release.set()
        await asyncio.wait_for(asyncio.gather(wrong, correct), 1)
        self.assertEqual(self.transfer._failed_recv, {"wrong"})
        self.assertNotIn("wrong", self.transfer._remote_agents)
        self.transfer.receive_kv.assert_called_once_with("p1", metas)
        self.client.get.assert_awaited_once()

    async def test_different_bootstrap_servers_query_independently(self):
        queries = [
            asyncio.create_task(
                self.transfer.handle_new_engine_id(
                    "p0", self.pull_metas(f"d{index}", address=address)
                )
            )
            for index, address in enumerate(("http://first", "http://second"))
        ]
        await asyncio.wait_for(self.entered.wait(), 1)
        await asyncio.sleep(0)
        self.assertEqual(self.client.get.await_count, 2)
        self.release.set()
        await asyncio.wait_for(asyncio.gather(*queries), 1)
        self.assertEqual(self.transfer._pending_bootstrap_queries, {})

    async def test_start_load_schedules_discovery_without_waiting_for_http(self):
        metas = self.pull_metas()
        await self.transfer._start_load_kv({"p0": metas})
        await asyncio.wait_for(self.entered.wait(), 1)
        self.transfer.receive_kv.assert_not_called()
        self.release.set()
        for _ in range(10):
            if self.transfer.receive_kv.called:
                break
            await asyncio.sleep(0)
        self.transfer.receive_kv.assert_called_once_with("p0", metas)


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
        transfer.transfer_topo = SimpleNamespace(
            virtually_split_kv_in_blocks=False,
            split_k_and_v=split,
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
        self.assertEqual(transfer.block_len_per_layer, official.block_len_per_layer)
        storages = dict(
            zip(
                official.kv_caches_base_addr,
                [
                    official.num_blocks * length
                    for length in official.block_len_per_layer
                ],
            )
        )
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
            transfer.block_len_per_layer,
        )
        src, _, lengths = transfer._build_request_transfer_params(
            "d",
            SendBlockMeta("p", "x", local_block_ids=((3,),)),
            request,
        )
        self.assertEqual(src, [part.data_ptr() + 3 * 64 for part in cache])
        self.assertEqual(lengths, [64, 64])

    def test_shared_cache_is_registered_and_transferred_once(self):
        cache = torch.empty((6, 16))
        transfer = self._compare_official({"layer.0": cache, "layer.1": cache})
        self.assertEqual(len(transfer.kv_caches_base_addr), 1)
        self.assertEqual(len(transfer._registered_storage), 1)

    def test_padded_cache_registers_stride_sized_blocks(self):
        cache = torch.empty((6, 32))[:, :16]
        transfer = self._compare_official({"layer.0": cache})
        self.assertEqual(transfer.block_len_per_layer, [128])
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
        transfer._remote_agents["p"] = {0: {0: f"tcp://127.0.0.1:{port}"}}
        transfer._remote_tp_size["p"] = 1
        transfer._response_decoder = msgspec.msgpack.Decoder(KVTransferResponse)
        event = RecvEvent("d", "x", ((4,),), "bootstrap", "p", 0)
        future = asyncio.get_running_loop().create_future()
        transfer._receive_futures["d"] = future
        try:
            await asyncio.wait_for(
                transfer._test_receive_events([event], {"d": future}), 1
            )
            self.assertEqual(configured_timeouts, [61000])
            # The receiver-loop fence wakes, then poll propagates the fatal error.
            self.assertTrue(future.done())
            with self.assertRaises(_ReceiveFenceError):
                (await asyncio.to_thread(transfer.poll))
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
                transfer._remote_agents["p"] = {0: {0: "p"}}
                transfer._remote_tp_size["p"] = 1
                transfer._response_decoder = msgspec.msgpack.Decoder(KVTransferResponse)
                transfer.engine = Mock()
                events = [RecvEvent("d", "x", ((4,),), "bootstrap", "p", 0)]
                current, other = (
                    asyncio.get_running_loop().create_future() for _ in range(2)
                )
                transfer._receive_futures = {"d": current, "other": other}
                transfer._recv_events = {"d": events[0]}
                task = asyncio.create_task(
                    transfer._test_receive_events(events, {"d": current})
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
                    self.assertTrue(current.done())
                    self.assertTrue(other.done())
                    for operation in (
                        transfer.poll,
                        transfer.take_errors,
                        transfer.close,
                    ):
                        with self.assertRaisesRegex(
                            _ReceiveFenceError, "worker must stop"
                        ):
                            await asyncio.to_thread(operation)
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
        with patch("mtsc.kv_transfer.httpx.AsyncClient") as client:
            client.return_value.__aenter__.side_effect = ConnectionError(
                "bootstrap down"
            )
            event = RecvEvent("d", "x", ((4,),), "bootstrap", "p", 0)
            future = asyncio.get_running_loop().create_future()
            transfer._receive_futures["d"] = future
            transfer._recv_events["d"] = event
            await transfer._test_receive_events([event], {"d": future})
            self.assertIsNone(transfer._receive_error)
            self.assertIsNone(future.result())
            self.assertIsNotNone(
                (await asyncio.to_thread(transfer.poll)).recvs[0].error
            )
            self.assertEqual((await asyncio.to_thread(transfer.take_errors)), {4})
