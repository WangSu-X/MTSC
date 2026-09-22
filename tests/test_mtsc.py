from __future__ import annotations

import asyncio
import sys
import threading
import time
import unittest
from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import torch
import zmq
from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    _compute_sender_transfer_plan,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from mtsc.connector import _require_recompute_policy
from mtsc.device import cache_tensors, new_device_event, npu_registration_regions
from mtsc.pd import (
    PDTransfer,
    TransferRegion,
    _NPUTransferTopology,
    _Source,
    _transpose_npu_cache_blocks,
    sender_transfer_plan,
)
from mtsc.protocol import (
    DTwoStageLoadPlan,
    MTSCConnectorMetadata,
    PDResponseStatus,
    PDSendUpdate,
    PDTransferRequest,
    PDTransferResponse,
    PDTransferSchema,
    SendRequirement,
    StoreLoadSpec,
    StoreRequest,
)
from mtsc.scheduler import _groups
from mtsc.store import (
    StoreIO,
    StoreLookupClient,
    store_topology_namespace,
    store_tp_layout,
)
from mtsc.worker import MTSCWorker, _DLoadState, _SendState
from proxy.pd_proxy import (
    DecodeEndpoint,
    PDProxy,
    PrefillEndpoint,
    RequestMetrics,
)


def _plan() -> DTwoStageLoadPlan:
    return DTwoStageLoadPlan(
        request_id="d-1",
        transfer_id="xfer-1",
        local_prefix_tokens=32,
        store_candidate_tokens=80,
        target_prefix_tokens=112,
        all_block_ids=([10, 11, 12, 13, 14, 15, 16],),
        external_block_ids=([12, 13, 14, 15, 16],),
        group_block_sizes=(16,),
        blocks_per_sliding_window=(0,),
        pd_enabled=True,
        wait_for_completion=True,
        kv_transfer_params={
            "transfer_id": "xfer-1",
            "remote_engine_id": "p0",
            "remote_bootstrap_addr": "http://p:8998",
        },
    )


class TwoStageBoundaryTest(unittest.TestCase):
    def test_hma_unhashed_group_shape_is_preserved(self) -> None:
        self.assertEqual(_groups([[1, 2], [3]]), ([1, 2], [3]))

    def test_store_success_uses_candidate_boundary(self) -> None:
        self.assertEqual(MTSCWorker._actual_store_prefix(_plan(), set()), 80)

    def test_first_failed_store_block_becomes_pd_boundary(self) -> None:
        self.assertEqual(MTSCWorker._actual_store_prefix(_plan(), {13}), 48)

    def test_failure_before_candidate_is_clamped_to_local_prefix(self) -> None:
        self.assertEqual(MTSCWorker._actual_store_prefix(_plan(), {10}), 80)

    def test_candidate_block_ids_only_cover_store_interval(self) -> None:
        self.assertEqual(MTSCWorker._store_candidate_block_ids(_plan()), {12, 13, 14})

    def test_full_local_hit_still_notifies_prefill(self) -> None:
        class PD:
            def __init__(self) -> None:
                self.requests = []

            def receive(self, *args) -> None:
                self.requests.append(args)

        worker = object.__new__(MTSCWorker)
        worker.pd = PD()
        worker._ignored_pd_recvs = set()
        plan = replace(
            _plan(),
            store_candidate_tokens=32,
            target_prefix_tokens=32,
            external_block_ids=([],),
            wait_for_completion=False,
        )
        state = _DLoadState(plan=plan, stage="STORE_DONE")
        self.assertTrue(worker._start_pd(state, 32))
        self.assertEqual(len(worker.pd.requests), 1)
        self.assertIn(plan.request_id, worker._ignored_pd_recvs)

    def test_mla_replicated_tp_only_uses_one_sender(self) -> None:
        self.assertEqual(
            sender_transfer_plan(0, 2, 0, 1, 128, 128, True),
            (True, 0, 0, 128),
        )
        self.assertEqual(
            sender_transfer_plan(1, 2, 0, 1, 128, 128, True),
            (False, 0, 0, 128),
        )

    def test_partial_prompt_tail_is_pulled_from_prefill(self) -> None:
        plan = replace(
            _plan(),
            store_candidate_tokens=48,
            target_prefix_tokens=50,
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

        def poll(self, _finished: set[str]) -> tuple[set[str], set[str]]:
            return set(), set(self.recv)

        def take_errors(self) -> set[int]:
            errors, self.errors = self.errors, set()
            return errors

    class _PD:
        def __init__(self, failed: set[str] | None = None) -> None:
            self.failed = failed or set()
            self.receives: list[tuple] = []
            self.reformatted: list[tuple[list[list[int]], int]] = []
            self.tp_size = 1

        def apply_updates(self, _updates) -> None:
            return

        def receive(self, *args) -> None:
            self.receives.append(args)

        def poll(self) -> tuple[set[str], set[str], set[str]]:
            return set(), set(), set(self.failed)

        def reformat_npu_blocks(self, block_ids, remote_tp_size) -> None:
            self.reformatted.append((block_ids, remote_tp_size))

    @staticmethod
    def _worker(store, pd) -> MTSCWorker:
        worker = object.__new__(MTSCWorker)
        worker.store = store
        worker.pd = pd
        worker.timeout = 180.0
        worker._decode = {}
        worker._plain_store_loads = {}
        worker._send = {}
        worker._ignored_pd_recvs = set()
        worker._load_errors = set()
        worker.group_block_sizes = (16,)
        return worker

    def test_store_partial_failure_falls_back_to_pd_suffix(self) -> None:
        store = self._Store({"d-1"}, {13})
        pd = self._PD()
        worker = self._worker(store, pd)
        worker._decode["d-1"] = _DLoadState(_plan(), "STORE_PENDING")

        sent, received = worker.get_finished(set(), MTSCConnectorMetadata())

        self.assertIsNone(sent)
        self.assertIsNone(received)
        self.assertEqual(len(pd.receives), 1)
        self.assertEqual(pd.receives[0][2], [[13, 14, 15, 16]])
        self.assertEqual(pd.reformatted, [([[12]], 1)])
        self.assertEqual(worker._decode["d-1"].stage, "PD_PENDING")
        self.assertEqual(worker.get_block_ids_with_load_errors(), set())

    def test_pd_failure_reports_suffix_for_local_recompute(self) -> None:
        store = self._Store()
        pd = self._PD({"d-1"})
        worker = self._worker(store, pd)
        state = _DLoadState(_plan(), "STORE_DONE")
        worker._decode["d-1"] = state
        self.assertTrue(worker._start_pd(state, 80))

        sent, received = worker.get_finished(set(), MTSCConnectorMetadata())

        self.assertIsNone(sent)
        self.assertEqual(received, {"d-1"})
        self.assertEqual(worker.get_block_ids_with_load_errors(), {15, 16})
        self.assertNotIn("d-1", worker._decode)

    def test_plain_store_load_reformats_only_successful_blocks(self) -> None:
        store = self._Store({"p-1"}, {11})
        pd = self._PD()
        worker = self._worker(store, pd)
        worker._plain_store_loads["p-1"] = StoreRequest(
            request_id="p-1",
            token_count=48,
            block_ids=([10, 11, 12],),
            block_hashes=[],
            load=StoreLoadSpec(0, 48, enabled=True),
        )

        _, received = worker.get_finished(set(), MTSCConnectorMetadata())

        self.assertEqual(received, {"p-1"})
        self.assertEqual(pd.reformatted, [([[10, 12]], 1)])
        self.assertEqual(worker.get_block_ids_with_load_errors(), {11})

    def test_connector_requires_vllm_recompute_policy(self) -> None:
        config = VllmConfig(
            kv_transfer_config=KVTransferConfig(
                kv_connector="MTSCConnector",
                kv_connector_module_path="mtsc.connector",
                kv_role="kv_consumer",
                engine_id="d0",
            )
        )
        with self.assertRaisesRegex(ValueError, "kv_load_failure_policy=recompute"):
            _require_recompute_policy(config)

        config.kv_transfer_config.kv_load_failure_policy = "recompute"
        _require_recompute_policy(config)


class TransferTopologyTest(unittest.TestCase):
    def test_heterogeneous_tp_matches_vllm_mooncake_reference(self) -> None:
        cases = 0
        for local_size in (1, 2, 4):
            for remote_size in (1, 2, 4):
                if max(local_size, remote_size) % min(local_size, remote_size):
                    continue
                for local_rank in range(local_size):
                    for remote_rank in range(remote_size):
                        for replicated in (False, True):
                            args = (
                                local_rank,
                                local_size,
                                remote_rank,
                                remote_size,
                                128,
                                128,
                                replicated,
                            )
                            self.assertEqual(
                                sender_transfer_plan(*args),
                                _compute_sender_transfer_plan(*args),
                            )
                            cases += 1
        self.assertEqual(cases, 98)

    def test_non_integral_tp_ratio_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            sender_transfer_plan(0, 2, 0, 3, 128, 128, False)
        with self.assertRaises(ValueError):
            sender_transfer_plan(0, 3, 0, 2, 128, 128, False)

    def test_npu_topology_maps_all_required_prefill_ranks(self) -> None:
        topology = _NPUTransferTopology(
            tp_rank=1,
            tp_size=2,
            block_size=16,
            engine_id="d0",
            is_mla=False,
            total_num_kv_heads=8,
        )
        self.assertEqual(topology.handshake_target_ranks(4), [2, 3])
        self.assertFalse(topology.local_replicates_kv_cache)

    @staticmethod
    def _schema() -> PDTransferSchema:
        return PDTransferSchema(1, "model", "", "torch.float16", "hnd", 16, False)

    def test_pp_alignment_uses_layer_intersection(self) -> None:
        pd = object.__new__(PDTransfer)
        pd.regions = [
            TransferRegion("layers.0.attn", 0, 0, 100, 64, 64),
            TransferRegion("layers.2.attn", 2, 0, 200, 64, 64),
        ]
        request = PDTransferRequest(
            hostname="d",
            rpc_port=1,
            tp_size=1,
            tp_rank=0,
            pp_size=1,
            pp_rank=0,
            schema=self._schema(),
            requests={"r": ("x", [[1]])},
            region_base_addresses=[300, 400],
            block_lengths=[64, 64],
            kv_block_lengths=[64, 64],
            layer_names=["layers.0.attn", "layers.1.attn"],
            layer_indices=[0, 1],
            group_indices=[0, 0],
        )

        aligned = pd._aligned_regions(request)

        self.assertEqual(len(aligned), 1)
        self.assertEqual(aligned[0][0].layer_name, "layers.0.attn")
        self.assertEqual(aligned[0][2], 0)

    def test_region_length_must_match_tp_ratio(self) -> None:
        local = TransferRegion("layers.0.attn", 0, 0, 100, 64, 64)
        remote = TransferRegion("layers.0.attn", 0, 0, 200, 96, 96)
        with self.assertRaisesRegex(ValueError, "TP ratio"):
            PDTransfer._validate_region_plan(local, remote, 2, 1, False, 0, 0, 64)

    def test_region_coverage_rejects_missing_duplicate_and_unexpected(self) -> None:
        pd = object.__new__(PDTransfer)
        pd.regions = [
            TransferRegion("layers.0.attn", 0, 0, 100, 64, 64),
            TransferRegion("layers.1.attn", 1, 0, 200, 64, 64),
        ]
        valid = [
            PDTransferResponse(
                PDResponseStatus.FINISH, ["r"], covered_regions={"r": [0, 1]}
            )
        ]
        pd._validate_coverage("r", [[10]], valid, expected=1)

        duplicate = [
            PDTransferResponse(
                PDResponseStatus.FINISH,
                ["r"],
                covered_regions={"r": [0, 0, 1]},
            )
        ]
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            pd._validate_coverage("r", [[10]], duplicate, expected=1)

        missing = [
            PDTransferResponse(
                PDResponseStatus.FINISH, ["r"], covered_regions={"r": [0]}
            )
        ]
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            pd._validate_coverage("r", [[10]], missing, expected=1)


class NPUDataPathTest(unittest.TestCase):
    def test_device_event_uses_torch_npu(self) -> None:
        class Event:
            pass

        class NPU:
            @staticmethod
            def Event():
                return Event()

        with (
            patch("mtsc.device.is_npu_platform", return_value=True),
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
        store = object.__new__(StoreIO)
        store.num_blocks = 4
        store.store = Backend()
        database = Database()
        store.databases = [database]

        with patch("mtsc.store.is_npu_platform", return_value=True):
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
        pd = object.__new__(PDTransfer)
        pd.layer_specs = {"model.layers.0.self_attn": spec}
        pd.layer_groups = {"model.layers.0.self_attn": 0}
        pd.engine = Engine()
        pd.regions = []
        pd._registered_storage = []
        pd.is_producer = False
        pd.kv_caches = {}

        with patch("mtsc.pd.is_npu_platform", return_value=True):
            pd.register({"model.layers.0.self_attn": (k_cache, v_cache)})

        self.assertEqual(len(pd.regions), 2)
        self.assertEqual(
            [region.kv_block_length for region in pd.regions],
            [
                k_cache.stride(0) * k_cache.element_size(),
                v_cache.stride(0) * v_cache.element_size(),
            ],
        )
        self.assertEqual(len(pd.engine.regions), 2)

    def test_heterogeneous_tp_transposes_appended_npu_shards(self) -> None:
        cache = torch.tensor([[[0, 1], [10, 11]]])

        _transpose_npu_cache_blocks(cache, [0], block_size=2, tp_ratio=2)

        self.assertEqual(cache.tolist(), [[[0, 10], [1, 11]]])

    def test_npu_nz_path_uses_gather_and_scatter_ops(self) -> None:
        calls = []

        def gather(*_args, **_kwargs) -> None:
            calls.append("gather")

        def scatter(*_args, **_kwargs) -> None:
            calls.append("scatter")

        fake_torch_npu = SimpleNamespace(
            npu_gather_pa_kv_cache=gather,
            npu_scatter_pa_kv_cache=scatter,
        )
        pd = object.__new__(PDTransfer)
        pd.topology = _NPUTransferTopology(
            tp_rank=0,
            tp_size=1,
            block_size=4,
            engine_id="d0",
            is_mla=True,
            total_num_kv_heads=1,
        )
        k_cache = torch.empty((2, 4, 1, 16))
        v_cache = torch.empty((2, 4, 1, 16))

        with (
            patch.dict(sys.modules, {"torch_npu": fake_torch_npu}),
            patch.object(
                torch,
                "npu",
                SimpleNamespace(synchronize=lambda: calls.append("sync")),
                create=True,
            ),
        ):
            pd._npu_nz_pair([k_cache, v_cache], [0])

        self.assertEqual(calls, ["sync", "gather", "scatter"])


class MLALayoutTest(unittest.TestCase):
    def test_store_namespace_separates_incompatible_topologies(self) -> None:
        config = SimpleNamespace(
            model_config=SimpleNamespace(
                model="org/model",
                revision="rev1",
                dtype=torch.float16,
                use_mla=False,
            ),
            cache_config=SimpleNamespace(cache_dtype="auto"),
            kv_transfer_config=SimpleNamespace(
                kv_connector_extra_config={"cache_prefix": "tenant"}
            ),
        )
        groups = [
            SimpleNamespace(
                kv_cache_spec=SimpleNamespace(block_size=16, page_size_bytes=64)
            )
        ]
        kwargs = {
            "pp_size": 1,
            "pcp_size": 1,
            "dcp_size": 1,
            "block_size": 16,
            "hash_block_size": 16,
        }
        with (
            patch("mtsc.store.is_npu_platform", return_value=False),
            patch("mtsc.store.get_kv_cache_layout", return_value="HND"),
        ):
            tp1 = store_topology_namespace(config, groups, tp_size=1, **kwargs)
            tp1_again = store_topology_namespace(config, groups, tp_size=1, **kwargs)
            tp2 = store_topology_namespace(config, groups, tp_size=2, **kwargs)

        self.assertEqual(tp1, tp1_again)
        self.assertNotEqual(tp1, tp2)
        self.assertTrue(tp1.startswith("tenant:mtsc-v1-"))

    def test_store_uses_one_shared_key_and_striped_puts(self) -> None:
        for tp_size in (1, 2, 4, 8):
            for tp_rank in range(tp_size):
                self.assertEqual(
                    store_tp_layout(
                        use_mla=True,
                        total_num_kv_heads=128,
                        tp_size=tp_size,
                        dcp_size=1,
                        tp_rank=tp_rank,
                    ),
                    (1, tp_size, 0),
                )

    def test_pd_register_uses_mla_page_size_as_transfer_length(self) -> None:
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
        cache = torch.empty((4, 16, 8), dtype=torch.float16)
        pd = object.__new__(PDTransfer)
        pd.layer_specs = {"model.layers.0.self_attn": spec}
        pd.layer_groups = {"model.layers.0.self_attn": 0}
        pd.topology = Topology()
        pd.engine = Engine()
        pd.regions = []
        pd._registered_storage = []
        pd.is_producer = False

        pd.register({"model.layers.0.self_attn": cache})

        self.assertEqual(len(pd.regions), 1)
        region = pd.regions[0]
        self.assertEqual(region.block_length, cache.stride(0) * cache.element_size())
        self.assertEqual(region.kv_block_length, spec.page_size_bytes)
        self.assertEqual(len(pd.engine.registrations), 1)


class CompletionAggregationTest(unittest.TestCase):
    def test_store_and_pd_are_both_required(self) -> None:
        worker = object.__new__(MTSCWorker)
        worker._send = {"p-1": _SendState(SendRequirement(store=True, pd=True))}
        self.assertEqual(worker._aggregate_sends({"p-1"}, set()), set())
        self.assertEqual(worker._aggregate_sends(set(), {"p-1"}), {"p-1"})


class PreemptionLifecycleTest(unittest.TestCase):
    def test_store_save_is_fenced_and_state_is_removed(self) -> None:
        future: Future[None] = Future()
        future.set_result(None)
        store = object.__new__(StoreIO)
        store._saves = {"r1": [future], "r2": []}
        store._save_offsets = {"r1": 32, "r2": 16}
        store._finished_save_requests = {"r1"}

        store.finish_preempted_saves({"r1"})

        self.assertNotIn("r1", store._saves)
        self.assertNotIn("r1", store._save_offsets)
        self.assertNotIn("r1", store._finished_save_requests)
        self.assertIn("r2", store._saves)

    def test_prefill_preemption_preserves_early_decode_waiter(self) -> None:
        class Store:
            @staticmethod
            def finish_preempted_saves(_request_ids) -> None:
                return

            @staticmethod
            def finish_preempted_loads(_request_ids) -> None:
                return

        class PD:
            def __init__(self) -> None:
                self.source = _Source("r1", "x1")

            @staticmethod
            def finish_receives(_request_ids) -> None:
                return

        worker = object.__new__(MTSCWorker)
        worker.store = Store()
        worker.pd = PD()
        worker._decode = {}
        worker._plain_store_loads = {}
        worker._send = {}
        worker._ignored_pd_recvs = set()
        metadata = MTSCConnectorMetadata(preempted_request_ids={"r1"})

        worker.handle_preemptions(metadata)

        self.assertEqual(worker.pd.source.transfer_id, "x1")

    @staticmethod
    def _polling_pd(source: _Source) -> PDTransfer:
        pd = object.__new__(PDTransfer)
        pd._source_lock = threading.Lock()
        pd._result_lock = threading.Lock()
        pd._receive_lock = threading.Lock()
        pd._sources = {source.transfer_id: source}
        pd._finished_send = set()
        pd._finished_recv = set()
        pd._failed_recv = set()
        pd._receive_futures = {}
        return pd

    def test_partial_fanout_expires_after_active_writes_finish(self) -> None:
        source = _Source(
            "r1", "x1", published=True, expected=2, completed=1, terminal=1
        )
        source.ready.set()
        source.expires_at = 0
        pd = self._polling_pd(source)

        sent, _, _ = pd.poll()

        self.assertEqual(sent, {"r1"})
        self.assertNotIn("x1", pd._sources)

    def test_source_timeout_does_not_release_active_write(self) -> None:
        source = _Source("r1", "x1", expected=1, active_writes=1)
        source.ready.set()
        source.expires_at = 0
        pd = self._polling_pd(source)

        sent, _, _ = pd.poll()

        self.assertEqual(sent, set())
        self.assertIn("x1", pd._sources)

    def test_abort_without_decode_request_is_cleaned_without_completion(self) -> None:
        source = _Source("r1", "x1", abort=True, expires_at=0)
        source.ready.set()
        pd = self._polling_pd(source)

        sent, _, _ = pd.poll()

        self.assertEqual(sent, set())
        self.assertNotIn("x1", pd._sources)

    def test_active_write_completion_is_accounted_atomically(self) -> None:
        source = _Source("r1", "x1", published=True, expected=1, active_writes=1)
        source.ready.set()
        pd = self._polling_pd(source)

        pd._finish_source_target("x1", source, False, active_write=True)

        self.assertEqual(source.active_writes, 0)
        self.assertEqual(source.terminal, 1)
        self.assertNotIn("x1", pd._sources)
        self.assertEqual(pd._finished_send, {"r1"})

    def test_early_decode_timeout_waits_for_prefill_terminal(self) -> None:
        source = _Source("r1", "x1", expected=1)
        pd = self._polling_pd(source)
        pd.is_producer = True
        pd.timeout = 10.0

        pd._finish_source_target("x1", source, False)

        self.assertIn("x1", pd._sources)
        self.assertEqual(pd._finished_send, set())

        pd.apply_updates(
            [
                PDSendUpdate(
                    request_id="r1",
                    transfer_id="x1",
                    block_ids=([1],),
                    source_ready=True,
                )
            ]
        )

        self.assertNotIn("x1", pd._sources)
        self.assertEqual(pd._finished_send, {"r1"})

    def test_published_source_without_decode_expires(self) -> None:
        source = _Source("r1", "x1", published=True, expires_at=0)
        source.ready.set()
        pd = self._polling_pd(source)

        sent, _, _ = pd.poll()

        self.assertEqual(sent, {"r1"})
        self.assertNotIn("x1", pd._sources)

    def test_timed_out_store_get_is_fenced_then_invalidated(self) -> None:
        request = StoreRequest(
            request_id="r1",
            token_count=32,
            block_ids=([10, 11],),
            block_hashes=[],
            load=StoreLoadSpec(0, 32, enabled=True),
        )
        future: Future[set[int]] = Future()
        store = object.__new__(StoreIO)
        store.load_timeout = 1.0
        store.databases = [SimpleNamespace(block_size=16)]
        store._loads = {"r1": future}
        store._load_requests = {"r1": request}
        store._load_started_at = {"r1": time.monotonic() - 2}
        store._load_timed_out = set()
        store._errors = set()
        store._saves = {}
        store._finished_save_requests = set()
        store._save_offsets = {}

        _, received = store.poll(set())

        self.assertEqual(received, set())
        self.assertIn("r1", store._load_timed_out)
        self.assertIn("r1", store._loads)

        future.set_result(set())
        _, received = store.poll(set())

        self.assertEqual(received, {"r1"})
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

        pd = object.__new__(PDTransfer)
        pd._remote_workers = {}
        with patch("mtsc.pd.httpx.AsyncClient", return_value=Client()):
            workers = await pd._query_workers("http://bootstrap", "p1", 1)
            self.assertEqual(workers, {0: {0: "tcp://p1"}})
            with self.assertRaisesRegex(KeyError, "not 'p0'"):
                await pd._query_workers("http://bootstrap", "p0", 1)

    async def test_schema_mismatch_is_rejected_before_write(self) -> None:
        class Socket:
            def __init__(self) -> None:
                self.frames = None

            async def send_multipart(self, frames) -> None:
                self.frames = frames

        schema = PDTransferSchema(1, "model", "", "float16", "hnd", 16, False)
        request = PDTransferRequest(
            hostname="d",
            rpc_port=1,
            tp_size=1,
            tp_rank=0,
            pp_size=1,
            pp_rank=0,
            schema=PDTransferSchema(
                1, "different-model", "", "float16", "hnd", 16, False
            ),
            requests={"r": ("x", [[1]])},
            region_base_addresses=[],
            block_lengths=[],
            kv_block_lengths=[],
        )
        pd = object.__new__(PDTransfer)
        pd.schema = schema
        pd._request_decoder = msgspec.msgpack.Decoder(PDTransferRequest)
        pd._response_decoder = msgspec.msgpack.Decoder(PDTransferResponse)
        pd._encoder = msgspec.msgpack.Encoder()
        socket = Socket()

        await pd._serve(b"d", pd._encoder.encode(request), socket)

        response = pd._response_decoder.decode(socket.frames[1])
        self.assertEqual(response.status, PDResponseStatus.ERROR)
        self.assertIn("schema mismatch", response.error)

    async def test_source_completion_counts_heterogeneous_pp_fanout(self) -> None:
        class Socket:
            async def send_multipart(self, _frames) -> None:
                return

        class Topology:
            @staticmethod
            def handshake_target_ranks(_remote_tp_size):
                return [0]

        schema = PDTransferSchema(1, "model", "", "float16", "hnd", 16, False)
        request = PDTransferRequest(
            hostname="d",
            rpc_port=1,
            tp_size=1,
            tp_rank=0,
            pp_size=2,
            pp_rank=0,
            schema=schema,
            requests={"d-r": ("x", [[]])},
            region_base_addresses=[],
            block_lengths=[],
            kv_block_lengths=[],
        )
        source = _Source("p-r", "x", abort=True)
        source.ready.set()
        pd = object.__new__(PDTransfer)
        pd.schema = schema
        pd.topology = Topology()
        pd.tp_rank = 0
        pd.pp_size = 1
        pd.pp_rank = 0
        pd.timeout = 0.01
        pd._sources = {"x": source}
        pd._source_lock = threading.Lock()
        pd._result_lock = threading.Lock()
        pd._finished_send = set()
        pd._request_decoder = msgspec.msgpack.Decoder(PDTransferRequest)
        pd._encoder = msgspec.msgpack.Encoder()

        await pd._serve(b"d", pd._encoder.encode(request), Socket())

        self.assertEqual(source.expected, 2)
        self.assertEqual(source.terminal, 1)
        self.assertIn("x", pd._sources)

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
