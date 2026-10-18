from __future__ import annotations

import asyncio
import threading
import time
import unittest
from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch
import zmq
from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    _compute_sender_transfer_plan,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from mtsc.connector import _require_recompute_policy
from mtsc.kv_cache_pool import (
    KeyMetadata,
    LoadEvent,
    LoadResult,
    MooncakeKVCachePool,
    PoolKey,
    PoolPollResult,
    StoreLookupClient,
    _CompatBlobBlockHashes,
    _LoadOutcome,
    _LoadState,
    _SaveState,
)
from mtsc.kv_transfer import (
    MooncakeKVTransfer,
    SendEvent,
    TransferPollResult,
    TransferRegion,
    TransferResult,
    _NPUTransferTopology,
    _Session,
    _validate_asymmetric_region_lengths,
)
from mtsc.kv_transfer import (
    _compute_sender_transfer_plan as local_sender_plan,
)
from mtsc.protocol import (
    FinishedWait,
    KVPoolLoadRequest,
    KVPoolSaveRequest,
    KVTransferPlan,
    MTSCConnectorMetadata,
)
from mtsc.scheduler import _groups
from mtsc.utils import (
    cache_tensors,
    new_device_event,
    npu_registration_regions,
)
from mtsc.worker import MTSCWorker, _FinishedWaitState, _TransferState
from proxy.pd_proxy import (
    DecodeEndpoint,
    PDProxy,
    PrefillEndpoint,
    RequestMetrics,
)


def _plan() -> KVTransferPlan:
    return KVTransferPlan(
        request_id="d-1",
        transfer_id="xfer-1",
        computed_tokens=32,
        pool_hit_tokens=80,
        target_tokens=112,
        all_block_ids=([10, 11, 12, 13, 14, 15, 16],),
        external_block_ids=([12, 13, 14, 15, 16],),
        group_block_sizes=(16,),
        blocks_per_sliding_window=(0,),
        transfer_params={
            "transfer_id": "xfer-1",
            "remote_engine_id": "p0",
            "remote_bootstrap_addr": "http://p:8998",
        },
    )


class TwoStageBoundaryTest(unittest.TestCase):
    @staticmethod
    def _loaded_prefix(plan: KVTransferPlan, invalid: set[int]) -> int:
        pool = object.__new__(MooncakeKVCachePool)
        pool.databases = [
            SimpleNamespace(block_size=size) for size in plan.group_block_sizes
        ]
        event = LoadEvent(
            plan.request_id,
            tuple(tuple(ids) for ids in plan.all_block_ids),
            (),
            plan.computed_tokens,
            plan.pool_hit_tokens,
        )
        return pool._loaded_prefix(event, invalid)

    def test_hma_unhashed_group_shape_is_preserved(self) -> None:
        self.assertEqual(_groups([[1, 2], [3]]), ([1, 2], [3]))

    def test_store_success_uses_candidate_boundary(self) -> None:
        self.assertEqual(self._loaded_prefix(_plan(), set()), 80)

    def test_first_failed_store_block_becomes_pd_boundary(self) -> None:
        self.assertEqual(self._loaded_prefix(_plan(), {13}), 48)

    def test_failure_before_candidate_is_clamped_to_local_prefix(self) -> None:
        self.assertEqual(self._loaded_prefix(_plan(), {10}), 80)

    def test_candidate_block_ids_only_cover_store_interval(self) -> None:
        self.assertEqual(MTSCWorker._pool_candidate_block_ids(_plan()), {12, 13, 14})

    def test_full_local_hit_still_notifies_prefill(self) -> None:
        class PD:
            def __init__(self) -> None:
                self.requests = []

            def recv_batch(self, events) -> None:
                self.requests.extend(events)

        worker = object.__new__(MTSCWorker)
        worker.transfer = PD()
        plan = replace(
            _plan(),
            pool_hit_tokens=32,
            target_tokens=32,
            external_block_ids=([],),
        )
        state = _TransferState(plan=plan, stage="POOL_DONE")
        receives = []
        worker._start_transfer(state, 32, receives)
        # Even with empty external_block_ids, we should still create a recv event
        self.assertEqual(len(receives), 1)
        self.assertEqual(receives[0].request_id, plan.request_id)

    def test_mla_replicated_tp_only_uses_one_sender(self) -> None:
        self.assertEqual(
            local_sender_plan(0, 2, 0, 1, 128, 128, True or 2 > 1)[:4],
            (True, 0, 0, 128),
        )
        self.assertEqual(
            local_sender_plan(1, 2, 0, 1, 128, 128, True or 2 > 1)[:4],
            (False, 0, 0, 128),
        )

    def test_partial_prompt_tail_is_pulled_from_prefill(self) -> None:
        plan = replace(
            _plan(),
            pool_hit_tokens=48,
            target_tokens=50,
            all_block_ids=([10, 11, 12, 13],),
            external_block_ids=([12, 13],),
        )
        # Store owns [32, 48); P must still transfer block 13, which contains
        # the two-token [48, 50) tail even though that block is not full.
        self.assertEqual(MTSCWorker._suffix_blocks(plan, 48), [[13]])


class FailureFallbackTest(unittest.TestCase):
    class _Store:
        def __init__(
            self, recv: set[str] | None = None, errors: set[int] | None = None
        ) -> None:
            self.recv = recv or set()
            self.errors = errors or set()

        def poll(self) -> PoolPollResult:
            loads = [LoadResult(rid, 48 if self.errors else 80) for rid in self.recv]
            return PoolPollResult(loads=loads)

        def take_errors(self) -> set[int]:
            errors, self.errors = self.errors, set()
            return errors

    class _PD:
        def __init__(self, failed: set[str] | None = None) -> None:
            self.failed = failed or set()
            self.receives: list[tuple] = []
            self.reformatted: list[tuple[list[list[int]], int]] = []
            self.tp_size = 1

        def recv(self, event) -> None:
            self.receives.append(event)

        def recv_batch(self, events) -> None:
            for event in events:
                self.recv(event)

        def poll(self) -> TransferPollResult:
            return TransferPollResult(
                recvs=[TransferResult(rid, "failed") for rid in self.failed]
            )

        def take_errors(self) -> set[int]:
            return set()

    @staticmethod
    def _worker(store, pd) -> MTSCWorker:
        worker = object.__new__(MTSCWorker)
        worker.pool = store
        worker.transfer = pd
        worker._on_save_requests = {}
        worker._on_transfer = {}
        worker._on_load_requests = {}
        worker._finished_waits = {}
        worker._ignored_recvs = set()
        worker._load_errors = set()
        worker.group_block_sizes = (16,)
        return worker

    def test_store_partial_failure_falls_back_to_pd_suffix(self) -> None:
        store = self._Store({"d-1"}, {13})
        pd = self._PD()
        worker = self._worker(store, pd)
        worker._on_transfer["d-1"] = _TransferState(_plan(), "POOL_LOADING")

        sent, received = worker.get_finished(set(), MTSCConnectorMetadata())

        self.assertIsNone(sent)
        self.assertIsNone(received)
        self.assertEqual(len(pd.receives), 1)
        self.assertEqual(pd.receives[0].block_ids, ((13, 14, 15, 16),))
        self.assertEqual(
            pd.reformatted, []
        )  # PD layout conversion belongs to its backend.
        self.assertEqual(worker._on_transfer["d-1"].stage, "TRANSFER_LOADING")
        self.assertEqual(worker.get_block_ids_with_load_errors(), set())

    def test_pd_failure_reports_suffix_for_local_recompute(self) -> None:
        store = self._Store()
        pd = self._PD({"d-1"})
        worker = self._worker(store, pd)
        state = _TransferState(_plan(), "POOL_DONE")
        worker._on_transfer["d-1"] = state
        receives = []
        worker._start_transfer(state, 80, receives)
        pd.recv_batch(receives)
        self.assertEqual(len(pd.receives), 1)
        self.assertEqual(pd.receives[0].block_ids, ((15, 16),))
        self.assertEqual(state.stage, "TRANSFER_LOADING")

        sent, received = worker.get_finished(set(), MTSCConnectorMetadata())

        self.assertIsNone(sent)
        self.assertEqual(received, {"d-1"})
        self.assertEqual(worker.get_block_ids_with_load_errors(), {15, 16})
        self.assertNotIn("d-1", worker._on_transfer)

    def test_plain_pool_load_reports_completion_and_errors(self) -> None:
        store = self._Store({"p-1"}, {11})
        pd = self._PD()
        worker = self._worker(store, pd)
        worker._on_load_requests["p-1"] = KVPoolLoadRequest(
            request_id="p-1",
            block_ids=([10, 11, 12],),
            block_hashes=[],
            start_token=0,
            end_token=48,
        )

        _, received = worker.get_finished(set(), MTSCConnectorMetadata())

        self.assertEqual(received, {"p-1"})
        self.assertEqual(pd.reformatted, [])
        self.assertEqual(worker.get_block_ids_with_load_errors(), {11})

    def test_connector_requires_vllm_recompute_policy(self) -> None:
        config = VllmConfig(
            kv_transfer_config=KVTransferConfig(
                kv_connector="MTSCConnector",
                kv_connector_module_path="mtsc.connector",
                kv_role="kv_consumer",
            )
        )
        with self.assertRaisesRegex(ValueError, "kv_load_failure_policy=recompute"):
            _require_recompute_policy(config)

        config.kv_transfer_config.kv_load_failure_policy = "recompute"
        _require_recompute_policy(config)


class TransferTopologyTest(unittest.TestCase):
    def test_heterogeneous_tp_matches_vllm_mooncake_reference(self) -> None:
        for p_size in (1, 2, 4):
            for d_size in (1, 2, 4):
                for d_rank in range(d_size):
                    peers = (
                        range(
                            d_rank * (p_size // d_size),
                            (d_rank + 1) * (p_size // d_size),
                        )
                        if p_size >= d_size
                        else [d_rank // (d_size // p_size)]
                    )
                    for p_rank in peers:
                        for replicated in (False, True):
                            actual = local_sender_plan(
                                p_rank,
                                p_size,
                                d_rank,
                                d_size,
                                128,
                                128,
                                replicated or p_size > 8,
                            )[:4]
                            expected = _compute_sender_transfer_plan(
                                p_rank, p_size, d_rank, d_size, 128, 128, replicated
                            )
                            self.assertEqual(actual[:3], expected[:3])
                            if actual[0]:
                                self.assertEqual(actual[3], expected[3])

    def test_non_integral_tp_ratio_is_rejected(self) -> None:
        with self.assertRaises(AssertionError):
            local_sender_plan(0, 2, 0, 3, 128, 128, False or 2 > 8)
        with self.assertRaises(AssertionError):
            local_sender_plan(0, 3, 0, 2, 128, 128, False or 3 > 8)

    def test_npu_topology_maps_all_required_prefill_ranks(self) -> None:
        topology = _NPUTransferTopology(
            tp_rank=1,
            tp_size=2,
            block_size=16,
            is_mla=False,
            total_num_kv_heads=8,
        )
        self.assertEqual(topology.handshake_target_ranks(4), [2, 3])

    def test_region_count_mismatch_is_rejected(self) -> None:
        self.assertIsNotNone(
            _validate_asymmetric_region_lengths(
                [TransferRegion(100, 64, 64)],
                [],
                1,
                1,
                False,
            )
        )

    def test_region_length_must_match_tp_ratio(self) -> None:
        error = _validate_asymmetric_region_lengths(
            [TransferRegion(100, 64, 64)],
            [TransferRegion(200, 96, 96)],
            2,
            1,
            False,
        )
        self.assertIn("TP ratio", error)


class NPUDataPathTest(unittest.TestCase):
    def test_device_event_uses_torch_npu(self) -> None:
        class Event:
            pass

        class NPU:
            @staticmethod
            def Event():
                return Event()

        with (
            patch("mtsc.utils.is_npu_platform", return_value=True),
            patch.object(torch, "npu", NPU(), create=True),
        ):
            self.assertIsInstance(new_device_event(), Event)

    def test_cache_tensor_sequence_is_preserved(self) -> None:
        k_cache = torch.empty((4, 16, 2, 8))
        v_cache = torch.empty((4, 16, 2, 8))
        tensors = cache_tensors((k_cache, v_cache))
        self.assertIs(tensors[0], k_cache)
        self.assertIs(tensors[1], v_cache)

    def test_npu_registration_uses_logical_aligned_view(self) -> None:
        allocation = torch.empty(128, dtype=torch.uint8)
        first = allocation[16:48]
        second = allocation[64:96]

        pointers, lengths = npu_registration_regions({"layer": (first, second)})

        self.assertEqual(pointers, [first.data_ptr()])
        self.assertEqual(
            lengths, [second.data_ptr() + second.nbytes - first.data_ptr()]
        )
        self.assertNotEqual(pointers[0], allocation.data_ptr())

    def test_store_registers_both_npu_kv_tensors(self) -> None:
        class Backend:
            def __init__(self) -> None:
                self.regions = []

            def register_buffer(self, address, length) -> int:
                self.regions.append((address, length))
                return 0

        class Database:
            def set_kv_caches_base_addr(self, addresses) -> None:
                self.addresses = addresses

            def set_block_len(self, lengths) -> None:
                self.lengths = lengths

        k_cache = torch.empty((4, 16, 2, 8))
        v_cache = torch.empty((4, 16, 2, 8))
        store = object.__new__(MooncakeKVCachePool)
        store._task_lock = threading.RLock()
        store._closed = store._registered = False
        store.num_blocks = 4
        store._num_blocks_resolved = True
        store.store = Backend()
        database = Database()
        store.databases = [database]

        with patch("mtsc.kv_cache_pool.is_npu_platform", return_value=True):
            store.register({"model.layers.0.self_attn": (k_cache, v_cache)})

        self.assertEqual(database.addresses, [k_cache.data_ptr(), v_cache.data_ptr()])
        self.assertEqual(
            database.lengths,
            [
                k_cache.stride(0) * k_cache.element_size(),
                v_cache.stride(0) * v_cache.element_size(),
            ],
        )
        self.assertEqual(len(store.store.regions), 2)

    def test_pd_registers_both_npu_kv_tensors(self) -> None:
        class Engine:
            def __init__(self) -> None:
                self.regions = []

            def register_memory(self, pointer, length) -> int:
                self.regions.append((pointer, length))
                return 0

        spec = MLAAttentionSpec(
            16,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.float16,
        )
        k_cache = torch.empty((4, 16, 1, 6), dtype=torch.float16)
        v_cache = torch.empty((4, 16, 1, 2), dtype=torch.float16)
        pd = object.__new__(MooncakeKVTransfer)
        pd._closed = pd._registered = False
        pd._receive_error = None
        pd.layer_specs = {"model.layers.0.self_attn": spec}
        pd.layer_groups = {"model.layers.0.self_attn": 0}
        pd.engine = Engine()
        pd.regions = []
        pd.kv_caches_base_addr, pd.block_lens = [], []
        pd.topology = _NPUTransferTopology(0, 1, 16, True, 1)
        pd._registered_storage = []
        pd.is_producer = False
        pd.kv_caches = {}

        with patch("mtsc.kv_transfer.is_npu_platform", return_value=True):
            pd.register({"model.layers.0.self_attn": (k_cache, v_cache)})

        self.assertEqual(len(pd.regions), 2)
        self.assertEqual(
            [region.kv_block_len for region in pd.regions],
            [
                k_cache.stride(0) * k_cache.element_size(),
                v_cache.stride(0) * v_cache.element_size(),
            ],
        )
        self.assertEqual(len(pd.engine.regions), 2)


class Vllm023CompatibilityTest(unittest.TestCase):
    def test_blob_block_hashes_is_a_lazy_fixed_width_sequence(self) -> None:
        hashes = _CompatBlobBlockHashes(memoryview(b"aaaabbbbcccc"), 4)

        self.assertEqual(len(hashes), 3)
        self.assertEqual(bytes(hashes[0]), b"aaaa")
        self.assertEqual(bytes(hashes[-1]), b"cccc")
        self.assertEqual([bytes(value) for value in hashes[1:]], [b"bbbb", b"cccc"])
        self.assertEqual(b"".join(hashes), b"aaaabbbbcccc")
        with self.assertRaises(IndexError):
            _ = hashes[3]

    def test_pool_key_serializes_rank_and_group(self) -> None:
        metadata = KeyMetadata(
            model_name="model",
            tp_rank=0,
            pcp_rank=0,
            dcp_rank=0,
            pp_rank=0,
            group_id=2,
        )
        key = PoolKey(replace(metadata, tp_rank=3, pp_rank=1), "0102")

        self.assertEqual(
            key.to_string(),
            "model@tp_rank:3@pcp0@dcp0@pp_rank:1@group:2@0102",
        )

    def test_database_pool_keys_and_put_striping_are_preserved(self) -> None:
        class OldPoolKey:
            def __init__(self, value: str) -> None:
                self.value = value

            def to_string(self) -> str:
                return self.value

        class OldDatabase:
            block_size = 16

            def process_tokens(self, token_count, block_hashes, mask_num=0):
                for chunk, block_hash in enumerate(block_hashes):
                    start = chunk * self.block_size
                    if start >= token_count:
                        break
                    if start < mask_num:
                        continue
                    yield (
                        start,
                        min(start + self.block_size, token_count),
                        OldPoolKey(f"base@{block_hash.hex()}"),
                    )

        store = object.__new__(MooncakeKVCachePool)
        store._task_lock = threading.RLock()
        rows = list(
            store._process_tokens(
                OldDatabase(),
                64,
                [b"a", b"b", b"c", b"d"],
                0,
                chunk_mask=[True, True, True, True],
                put_step=2,
                put_step_rank=0,
            )
        )

        self.assertEqual(
            [(start, end, key.to_string()) for start, end, key in rows],
            [
                (0, 16, "base@61"),
                (32, 48, "base@63"),
            ],
        )

    def test_old_full_store_mask_is_sliced_for_incremental_save(self) -> None:
        class OldCoordinator:
            @staticmethod
            def store_mask(token_count):
                self.assertEqual(token_count, 64)
                return ([True, False, True, True],)

        store = object.__new__(MooncakeKVCachePool)
        store._task_lock = threading.RLock()
        store.coordinator = OldCoordinator()
        store.databases = [SimpleNamespace(block_size=16)]

        self.assertEqual(store._store_masks(64, 32, None), ([True, True],))


class MLALayoutTest(unittest.TestCase):
    def test_pd_register_uses_mla_physical_block_stride(self) -> None:
        class Engine:
            def __init__(self) -> None:
                self.registrations = []

            def batch_register_memory(self, pointers, lengths) -> int:
                self.registrations.append((pointers, lengths))
                return 0

        class Topology:
            virtually_split_kv_in_blocks = False

            def get_transfer_cache_regions(self, raw, _spec):
                return [raw]

        spec = MLAAttentionSpec(
            16,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.float16,
        )
        cache = torch.empty((4, 16, 16), dtype=torch.float16)[:, :, :8]
        pd = object.__new__(MooncakeKVTransfer)
        pd._closed = pd._registered = False
        pd._receive_error = None
        pd.layer_specs = {"model.layers.0.self_attn": spec}
        pd.layer_groups = {"model.layers.0.self_attn": 0}
        pd.topology = Topology()
        pd.engine = Engine()
        pd.regions = []
        pd.kv_caches_base_addr, pd.block_lens = [], []
        pd._registered_storage = []
        pd.is_producer = False

        pd.register({"model.layers.0.self_attn": cache})

        self.assertEqual(len(pd.regions), 1)
        region = pd.regions[0]
        self.assertEqual(region.block_len, cache.stride(0) * cache.element_size())
        self.assertEqual(region.kv_block_len, cache.stride(0) * cache.element_size())
        self.assertGreater(region.kv_block_len, spec.page_size_bytes)
        self.assertEqual(len(pd.engine.registrations), 1)


class CompletionAggregationTest(unittest.TestCase):
    def test_store_and_pd_are_both_required(self) -> None:
        worker = object.__new__(MTSCWorker)
        worker._on_save_requests = {"p-1": KVPoolSaveRequest("p-1", ([1],), [], 0, 16)}
        worker._finished_waits = {
            "p-1": _FinishedWaitState(FinishedWait(store_save=True, kv_transfer=True))
        }
        # When only pool save completes, request is not finished
        self.assertEqual(worker._aggregate_send_finished({"p-1"}, set()), set())
        # When kv transfer also completes, request is finished
        self.assertEqual(worker._aggregate_send_finished(set(), {"p-1"}), {"p-1"})


class PreemptionLifecycleTest(unittest.TestCase):
    def test_store_save_is_fenced_and_state_is_removed(self) -> None:
        future: Future[None] = Future()
        future.set_running_or_notify_cancel()
        future.set_result(None)
        store = object.__new__(MooncakeKVCachePool)
        store._task_lock = threading.RLock()
        store._save_states = {"r1": _SaveState(running=future), "r2": _SaveState()}
        store._loads = {}
        store._invalid = {}

        store.preempt("r1")

        self.assertNotIn("r1", store._save_states)
        self.assertIn("r2", store._save_states)

    def test_prefill_preemption_preserves_early_decode_waiter(self) -> None:
        class Store:
            @staticmethod
            def preempt(_request_id) -> None:
                return

        class PD:
            def __init__(self) -> None:
                self.source = _Session("r1", "x1")

            @staticmethod
            def preempt(_request_id) -> None:
                return

        worker = object.__new__(MTSCWorker)
        worker.pool = Store()
        worker.transfer = PD()
        worker._on_save_requests = {}
        worker._on_transfer = {}
        worker._on_load_requests = {}
        worker._finished_waits = {}
        worker._ignored_recvs = set()
        metadata = MTSCConnectorMetadata(preempted_request_ids={"r1"})

        worker.handle_preemptions(metadata)

        self.assertEqual(worker.transfer.source.transfer_id, "x1")

    @staticmethod
    def _polling_pd(source: _Session) -> MooncakeKVTransfer:
        pd = object.__new__(MooncakeKVTransfer)
        pd._closed = False
        pd._receive_error = None
        pd._registered = True
        pd._source_lock = threading.Lock()
        pd._source_changed = threading.Condition(pd._source_lock)
        pd._result_lock = threading.Lock()
        pd._receive_lock = threading.Lock()
        pd._sources = {source.transfer_id: source}
        pd._finished_send = set()

        pd._failed_recv = set()
        pd._receive_futures = {}
        pd._retired = set()
        pd._prepared = {}
        pd._pending_sends = set()
        pd._recv_events = {}
        pd._send_failures = {}
        pd._invalid = {}
        return pd

    def test_partial_fanout_expires_after_active_writes_finish(self) -> None:
        source = _Session(
            "r1", "x1", published=True, expected=2, completed=1, terminal=1
        )
        source.ready.set()
        source.expires_at = 0
        pd = self._polling_pd(source)

        sent = {result.request_id for result in pd.poll().sends}

        self.assertEqual(sent, {"r1"})
        self.assertNotIn("x1", pd._sources)

    def test_source_timeout_does_not_release_active_write(self) -> None:
        source = _Session("r1", "x1", expected=1, active_writes=1)
        source.ready.set()
        source.expires_at = 0
        pd = self._polling_pd(source)

        sent = {result.request_id for result in pd.poll().sends}

        self.assertEqual(sent, set())
        self.assertIn("x1", pd._sources)

    def test_abort_without_decode_request_is_cleaned_without_completion(self) -> None:
        source = _Session("r1", "x1", abort=True, expires_at=0)
        source.ready.set()
        pd = self._polling_pd(source)

        pd.cancel("r1", "x1")

        self.assertEqual(pd.poll().sends, [])
        self.assertNotIn("x1", pd._sources)
        self.assertIn("x1", pd._retired)

    def test_active_write_completion_is_accounted_atomically(self) -> None:
        source = _Session("r1", "x1", published=True, expected=1)
        source.ready.set()
        pd = self._polling_pd(source)
        pd.timeout = 1

        async def failed_write(*_args):
            self.assertEqual(source.active_writes, 1)
            return dict.fromkeys(_args[0], False)

        pd._write_batch = failed_write
        asyncio.run(pd._write_ready_targets({"d1": source}, None))

        self.assertEqual(source.active_writes, 0)
        self.assertEqual(source.terminal, 1)
        self.assertNotIn("x1", pd._sources)
        self.assertEqual(pd._finished_send, {"r1"})

    def test_early_decode_timeout_waits_for_prefill_terminal(self) -> None:
        import msgspec
        from test_backends import _request, _Socket, _transfer

        source = _Session("r1", "x1", expected=1)
        pd = _transfer()
        pd._sources["x1"] = source
        pd.timeout = 0.001
        request = msgspec.structs.replace(
            _request(pd), req_blocks={"d1": ("x1", [[4]])}
        )
        asyncio.run(pd._serve(b"d", msgspec.msgpack.encode(request), _Socket()))

        self.assertIn("x1", pd._sources)
        self.assertEqual(pd._finished_send, set())
        pd.send(SendEvent("r1", "x1", ((1,),)))
        self.assertNotIn("x1", pd._sources)
        self.assertEqual(pd._finished_send, {"r1"})
        self.assertIsNotNone(pd.poll().sends[0].error)

    def test_published_source_without_decode_expires(self) -> None:
        source = _Session("r1", "x1", published=True, expires_at=0)
        source.ready.set()
        pd = self._polling_pd(source)

        sent = {result.request_id for result in pd.poll().sends}

        self.assertEqual(sent, {"r1"})
        self.assertNotIn("x1", pd._sources)

    def test_timed_out_store_get_is_fenced_then_invalidated(self) -> None:
        future: Future[_LoadOutcome] = Future()
        store = object.__new__(MooncakeKVCachePool)
        store._task_lock = threading.RLock()
        store.load_timeout = 1.0
        store.databases = [SimpleNamespace(block_size=16)]
        store._loads = {
            "r1": _LoadState(
                LoadEvent("r1", ((10, 11),), (), 0, 32),
                future,
                time.monotonic() - 2,
            )
        }
        store._invalid = {}
        store._save_states = {}

        self.assertEqual(store.poll().loads, [])
        self.assertTrue(store._loads["r1"].timed_out)
        self.assertIn("r1", store._loads)

        future.set_result(_LoadOutcome(32, None, set()))
        received = store.poll().loads
        self.assertEqual([result.request_id for result in received], ["r1"])
        self.assertEqual(received[0].loaded_tokens, 0)
        self.assertEqual(store.take_errors(), {10, 11})

    def test_store_lookup_timeout_resets_req_socket(self) -> None:
        class Socket:
            @staticmethod
            def send_multipart(*_args, **_kwargs) -> None:
                raise zmq.Again()

        client = object.__new__(StoreLookupClient)
        client._socket = Socket()
        client._timeout_ms = 25
        reset = []
        client._reset_socket = lambda: reset.append(True)  # type: ignore[method-assign]

        with self.assertRaisesRegex(TimeoutError, "0.025s"):
            client._lookup(16, [])

        self.assertEqual(reset, [True])


class ProxySpecTest(unittest.TestCase):
    def setUp(self) -> None:
        self.proxy = PDProxy(
            [PrefillEndpoint("http://p", "p0", "http://p:8998", 0)],
            [DecodeEndpoint("http://d")],
            metrics_file=None,
            backend_token=None,
        )

    def test_wire_specs_share_transfer_id(self) -> None:
        class Session:
            transfer_id = "xfer-r1"
            prefill = PrefillEndpoint("http://p", "p0", "http://p:8998", 0)

        original = {
            "model": "m",
            "prompt": "hello",
            "stream": True,
            "max_tokens": 8,
            "stream_options": {"include_usage": True},
        }
        p_body = self.proxy._prefill_body(original, Session())
        d_body = self.proxy._decode_body(original, Session())
        self.assertFalse(p_body["stream"])
        self.assertEqual(p_body["max_tokens"], 1)
        self.assertNotIn("stream_options", p_body)
        self.assertTrue(p_body["kv_transfer_params"]["do_remote_decode"])
        self.assertTrue(d_body["kv_transfer_params"]["do_remote_prefill"])
        self.assertEqual(d_body["kv_transfer_params"]["remote_dp_rank"], 0)
        self.assertEqual(
            p_body["kv_transfer_params"]["transfer_id"],
            d_body["kv_transfer_params"]["transfer_id"],
        )

    def test_metrics_record_has_stable_timing_schema(self) -> None:
        metrics = RequestMetrics(
            request_id="r1",
            client_request_id="client-r1",
            transfer_id="xfer-r1",
            endpoint="/v1/completions",
            prefill=PrefillEndpoint("http://p:8100", "p0", "http://p:8998", 2),
            decode=DecodeEndpoint("http://d:8200"),
        )
        metrics.prefill_start_at = metrics.request_received_at + 0.001
        metrics.prefill_end_at = metrics.request_received_at + 0.003
        metrics.prefill_start_mono = metrics.request_received_mono + 0.001
        metrics.prefill_end_mono = metrics.request_received_mono + 0.003
        metrics.request_end_at = metrics.request_received_at + 0.01
        metrics.request_end_mono = metrics.request_received_mono + 0.01
        metrics.outcome = "success"

        record = metrics.to_record()

        self.assertEqual(record["schema_version"], 1)
        self.assertEqual(record["prefill_dp_rank"], 2)
        self.assertEqual(record["prefill_duration_ms"], 2.0)
        self.assertIn("+00:00", record["request_received_at"])
        self.assertIn("error_stage", record)


class AsyncControlPlaneTest(unittest.IsolatedAsyncioTestCase):
    async def test_bootstrap_query_selects_exact_dp_and_engine(self) -> None:
        class Response:
            @staticmethod
            def raise_for_status() -> None:
                return

            @staticmethod
            def json():
                return {
                    "0": {
                        "engine_id": "p0",
                        "tp_size": 1,
                        "pp_size": 1,
                        "worker_addr": {"0": {"0": "tcp://p0"}},
                    },
                    "1": {
                        "engine_id": "p1",
                        "tp_size": 1,
                        "pp_size": 1,
                        "worker_addr": {"0": {"0": "tcp://p1"}},
                    },
                }

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args) -> None:
                return None

            async def get(self, _url):
                return Response()

        pd = object.__new__(MooncakeKVTransfer)
        pd._remote_workers = {}
        pd._pending_bootstrap_queries = {}
        with patch("mtsc.kv_transfer.httpx.AsyncClient", return_value=Client()):
            workers = await pd._query_workers("http://bootstrap", "p1", 1)
            self.assertEqual(workers, {0: {0: "tcp://p1"}})
            with self.assertRaisesRegex(KeyError, "not 'p0'"):
                await pd._query_workers("http://bootstrap", "p0", 1)

    async def test_prefill_error_wins_while_decode_headers_are_pending(self) -> None:
        class RawRequest:
            def __init__(self) -> None:
                self.headers = {"content-type": "application/json"}

            @staticmethod
            async def json():
                return {"model": "m", "prompt": "hello", "stream": True}

        class Client:
            def __init__(self, *_args, **_kwargs) -> None:
                self.decode_started = asyncio.Event()
                self.decode_cancelled = False
                self.closed = False

            async def post(self, *_args, **_kwargs):
                self.decode_started.set()
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    self.decode_cancelled = True
                    raise

            async def close(self) -> None:
                self.closed = True

        proxy = PDProxy(
            [PrefillEndpoint("http://p", "p0", "http://p:8998", 0)],
            [DecodeEndpoint("http://d")],
            metrics_file=None,
            backend_token=None,
        )
        client = Client()

        async def fail_prefill(*_args, **_kwargs) -> None:
            await client.decode_started.wait()
            raise RuntimeError("prefill failed")

        proxy._run_prefill = fail_prefill  # type: ignore[method-assign]
        with patch("proxy.pd_proxy.aiohttp.ClientSession", return_value=client):
            response = await proxy._handle(RawRequest(), "/v1/completions")

        self.assertEqual(response.status_code, 502)
        self.assertTrue(client.decode_cancelled)
        self.assertTrue(client.closed)
        self.assertEqual(proxy.sessions, {})


class ExternalConnectorTest(unittest.TestCase):
    def test_vllm_factory_loads_external_connector(self) -> None:
        config = KVTransferConfig(
            kv_connector="MTSCConnector",
            kv_connector_module_path="mtsc.connector",
            kv_role="kv_producer",
            engine_id="p0",
        )
        connector = KVConnectorFactory.get_connector_class(config)
        self.assertEqual(connector.__name__, "MTSCConnector")
        self.assertTrue(KVConnectorFactory.supports_hma_config(config))


if __name__ == "__main__":
    unittest.main()
