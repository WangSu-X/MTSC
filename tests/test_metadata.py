from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import msgspec
from vllm.v1.request import RequestStatus

from mtsc.kv_cache_pool import LoadResult, PoolPollResult, SaveResult
from mtsc.kv_transfer import TransferPollResult, TransferResult
from mtsc.protocol import (
    FinishedWait,
    KVPoolLoadRequest,
    KVPoolSaveRequest,
    KVTransferResponse,
    KVTransferSourceState,
    KVTransferStatus,
    MTSCConnectorMetadata,
)
from mtsc.scheduler import MTSCScheduler
from mtsc.worker import MTSCWorker


def _scheduler(*, consumer=True, hit=80, extra=None):
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_role="kv_consumer" if consumer else "kv_producer",
            kv_connector_extra_config=extra if extra is not None else {},
        )
    )
    caches = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16))],
        has_mamba_layers=False,
    )
    lookup = Mock()
    lookup.lookup.return_value = hit
    with (
        patch("mtsc.scheduler.StoreLookupClient", return_value=lookup),
        patch("mtsc.scheduler.resolve_kv_cache_block_sizes", return_value=(16, 16)),
    ):
        scheduler = MTSCScheduler(config, caches, decode_save=True)
    return scheduler, config


def _request(*, params=None, tokens=112, prompt=112):
    return SimpleNamespace(
        request_id="request-1",
        kv_transfer_params=params
        if params is not None
        else {
            "do_remote_prefill": True,
            "transfer_id": "transfer-1",
            "remote_engine_id": "producer",
            "remote_bootstrap_addr": "http://producer:8998",
        },
        num_tokens=tokens,
        num_prompt_tokens=prompt,
        block_hashes=[b"hash"] * (tokens // 16),
        status=RequestStatus.FINISHED_LENGTH_CAPPED,
    )


def _blocks():
    return SimpleNamespace(
        get_block_ids=lambda: ([10, 11, 12, 13, 14, 15, 16],),
        get_unhashed_block_ids_all_groups=lambda: ([12, 13, 14, 15, 16],),
    )


def _output(**kwargs):
    fields = {
        "scheduled_new_reqs": [],
        "scheduled_cached_reqs": SimpleNamespace(
            req_ids=[], new_block_ids=[], num_computed_tokens=[]
        ),
        "num_scheduled_tokens": {},
        "finished_req_ids": set(),
        "preempted_req_ids": set(),
    }
    fields.update(kwargs)
    return SimpleNamespace(**fields)


def _worker(config):
    pool, transfer = Mock(), Mock()
    pool.poll.return_value = PoolPollResult()
    pool.take_errors.return_value = set()
    transfer.poll.return_value = TransferPollResult()
    transfer.take_errors.return_value = set()
    with (
        patch("mtsc.worker._create_pool", return_value=pool),
        patch("mtsc.worker._create_transfer", return_value=transfer),
    ):
        return MTSCWorker(config, None)


class MetadataFlowTest(unittest.TestCase):
    def test_worker_submits_ready_transfers_in_one_batch(self):
        _, config, metadata = self._decode_metadata()
        first = metadata.transfer_plans[0]
        second = replace(first, request_id="request-2", transfer_id="transfer-2")
        metadata.transfer_plans = [first, second]
        worker = _worker(config)
        worker.get_finished(set(), metadata)
        worker.pool.poll.return_value = PoolPollResult(
            loads=[LoadResult(first.request_id, 48), LoadResult(second.request_id, 64)]
        )
        worker.get_finished(set(), MTSCConnectorMetadata())
        worker.transfer.recv_batch.assert_called_once()
        events = worker.transfer.recv_batch.call_args.args[0]
        self.assertEqual(
            [event.request_id for event in events],
            [first.request_id, second.request_id],
        )
        self.assertEqual(
            [event.block_ids for event in events],
            [((13, 14, 15, 16),), ((14, 15, 16),)],
        )

    def test_lookup_boolean_config_reaches_lookup_with_correct_mode(self):
        for value, expected in (
            (False, False),
            (None, False),
            (0, False),
            ("false", False),
            (" OFF ", False),
            ("no", False),
            ("0", False),
            (True, True),
            (1, True),
            ("true", True),
            (" ON ", True),
        ):
            with self.subTest(value=value):
                scheduler, _ = _scheduler(hit=0, extra={"lookup_async": value})
                self.assertEqual(
                    scheduler.get_num_new_matched_tokens(_request(params={}), 0),
                    (0, False),
                )
                self.assertIs(
                    scheduler.lookup_client.lookup.call_args.kwargs["asynchronous"],
                    expected,
                )

    def _decode_metadata(self):
        scheduler, config = _scheduler()
        request = _request()
        external, asynchronous = scheduler.get_num_new_matched_tokens(request, 32)
        self.assertEqual((external, asynchronous), (80, True))
        scheduler.update_state_after_alloc(request, _blocks(), external)
        return scheduler, config, scheduler.build_connector_meta(_output())

    def test_two_stage_load_tracks_pool_then_transfers_actual_suffix(self):
        scheduler, config, metadata = self._decode_metadata()
        self.assertEqual(scheduler._load_decisions, {})
        self.assertEqual(scheduler._transfer_decisions, {})
        self.assertEqual(scheduler._batch_transfers, [])
        self.assertEqual(len(metadata.pool_loads), 1)
        self.assertEqual(metadata.pool_saves, [])
        worker = _worker(config)

        self.assertEqual(worker.get_finished(set(), metadata), (None, None))
        load = worker.pool.load.call_args.args[0]
        self.assertEqual((load.start_load, load.end_load), (32, 80))
        self.assertIs(worker._on_load_requests[load.request_id], metadata.pool_loads[0])
        worker.transfer.recv_batch.assert_not_called()

        worker.pool.poll.return_value = PoolPollResult(
            loads=[LoadResult(load.request_id, 48, "partial miss")]
        )
        worker.pool.take_errors.return_value = {13, 14}
        self.assertEqual(
            worker.get_finished(set(), MTSCConnectorMetadata()), (None, None)
        )
        self.assertEqual(worker._on_load_requests, {})
        recv = worker.transfer.recv_batch.call_args.args[0][0]
        self.assertEqual(recv.block_ids, ((13, 14, 15, 16),))
        self.assertEqual(recv.transfer_id, "transfer-1")
        self.assertEqual(worker.get_block_ids_with_load_errors(), set())

        worker.pool.poll.return_value = PoolPollResult()
        worker.pool.take_errors.return_value = set()
        worker.transfer.poll.return_value = TransferPollResult(
            recvs=[TransferResult(load.request_id)]
        )
        self.assertEqual(
            worker.get_finished(set(), MTSCConnectorMetadata()),
            (None, {load.request_id}),
        )
        self.assertEqual(worker._on_transfer, {})

    def test_plain_pool_load_uses_same_tracker_without_transfer_plan(self):
        scheduler, config = _scheduler(consumer=False)
        request = _request(params={})
        external, _ = scheduler.get_num_new_matched_tokens(request, 32)
        scheduler.update_state_after_alloc(request, _blocks(), external)
        metadata = scheduler.build_connector_meta(_output())
        self.assertEqual(metadata.transfer_plans, [])
        worker = _worker(config)
        worker.get_finished(set(), metadata)
        self.assertIn(request.request_id, worker._on_load_requests)
        worker.pool.poll.return_value = PoolPollResult(
            loads=[LoadResult(request.request_id, 80)]
        )
        self.assertEqual(
            worker.get_finished(set(), MTSCConnectorMetadata()),
            (None, {request.request_id}),
        )
        worker.transfer.recv_batch.assert_not_called()
        self.assertEqual(worker._on_load_requests, {})

    def test_full_local_hit_still_notifies_transfer_source(self):
        scheduler, config = _scheduler(hit=112)
        request = _request()
        self.assertEqual(scheduler.get_num_new_matched_tokens(request, 112), (0, False))
        blocks = _blocks()
        blocks.get_unhashed_block_ids_all_groups = lambda: ([],)
        scheduler.update_state_after_alloc(request, blocks, 0)
        metadata = scheduler.build_connector_meta(_output())
        self.assertEqual(metadata.pool_loads, [])
        worker = _worker(config)
        worker.get_finished(set(), metadata)
        recv = worker.transfer.recv_batch.call_args.args[0][0]
        self.assertEqual(recv.block_ids, ((),))
        worker.transfer.poll.return_value = TransferPollResult(
            recvs=[TransferResult(request.request_id)]
        )
        self.assertEqual(
            worker.get_finished(set(), MTSCConnectorMetadata()), (None, None)
        )
        self.assertEqual(worker._ignored_recvs, set())
        self.assertEqual(worker._on_transfer, {})

    def test_pool_only_partial_load_invalidates_unusable_suffix(self):
        _, config, metadata = self._decode_metadata()
        metadata.transfer_plans[0] = replace(
            metadata.transfer_plans[0],
            transfer_enabled=False,
            target_tokens=80,
            external_block_ids=([12, 13, 14],),
        )
        worker = _worker(config)
        worker.pool.poll.return_value = PoolPollResult(
            loads=[LoadResult("request-1", 48)]
        )
        worker.pool.take_errors.return_value = {13, 14}
        self.assertEqual(worker.get_finished(set(), metadata), (None, {"request-1"}))
        self.assertEqual(worker.get_block_ids_with_load_errors(), {13, 14})
        worker.transfer.recv_batch.assert_not_called()

    def test_lookup_retry_drops_stale_load_decision(self):
        scheduler, _ = _scheduler(consumer=False)
        request = _request(params={})
        self.assertEqual(scheduler.get_num_new_matched_tokens(request, 32), (48, True))
        scheduler.lookup_client.lookup.return_value = 0
        self.assertEqual(scheduler.get_num_new_matched_tokens(request, 32), (0, False))
        self.assertEqual(scheduler._load_decisions, {})

    def test_worker_preemption_clears_active_load_and_transfer(self):
        _, config, metadata = self._decode_metadata()
        worker = _worker(config)
        worker.get_finished(set(), metadata)
        self.assertIn("request-1", worker._on_load_requests)
        self.assertIn("request-1", worker._on_transfer)
        worker.handle_preemptions(
            MTSCConnectorMetadata(preempted_request_ids={"request-1"})
        )
        worker.pool.preempt.assert_called_once_with("request-1")
        worker.transfer.preempt.assert_called_once_with("request-1")
        self.assertEqual(worker._on_load_requests, {})
        self.assertEqual(worker._on_transfer, {})

    def test_aborted_producer_cancels_transfer_without_delaying_release(self):
        scheduler, config = _scheduler(consumer=False, hit=0)
        request = _request(
            params={"do_remote_decode": True, "transfer_id": "transfer-1"}
        )
        scheduler.update_state_after_alloc(request, _blocks(), 0)
        worker = _worker(config)
        worker.get_finished(set(), scheduler.build_connector_meta(_output()))
        request.status = RequestStatus.FINISHED_ABORTED
        self.assertEqual(
            scheduler.request_finished(request, _blocks().get_block_ids()),
            (False, None),
        )
        metadata = scheduler.build_connector_meta(
            _output(finished_req_ids={request.request_id})
        )
        self.assertTrue(metadata.transfer_states[0].cancelled)
        worker.get_finished({request.request_id}, metadata)
        worker.transfer.cancel.assert_called_once_with(request.request_id, "transfer-1")
        self.assertEqual(scheduler._tracked_requests, {})
        self.assertEqual(scheduler._transferring_requests, set())
        self.assertFalse(scheduler.has_pending_push_work())

    def test_decode_save_preserves_prompt_boundary_and_incremental_range(self):
        scheduler, config = _scheduler(hit=0)
        request = _request(params={}, tokens=112, prompt=50)
        scheduler.update_state_after_alloc(request, _blocks(), 0)
        new = SimpleNamespace(
            req_id=request.request_id,
            block_ids=_blocks().get_block_ids(),
            num_computed_tokens=0,
        )
        metadata = scheduler.build_connector_meta(
            _output(
                scheduled_new_reqs=[new], num_scheduled_tokens={request.request_id: 83}
            )
        )
        save = metadata.pool_saves[0]
        self.assertEqual((save.start_token, save.end_token), (64, 80))
        self.assertEqual(save.prompt_tokens, 50)
        worker = _worker(config)
        with patch("mtsc.worker.new_device_event", return_value=Mock()):
            worker.get_finished(set(), metadata)
        event = worker.pool.save.call_args.args[0]
        self.assertEqual(
            (event.start_save, event.end_save, event.prompt_tokens), (64, 80, 50)
        )
        self.assertIs(worker._on_save_requests[request.request_id], save)
        next_output = _output(
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=[request.request_id],
                new_block_ids=[None],
                num_computed_tokens=[83],
            ),
            num_scheduled_tokens={request.request_id: 29},
        )
        next_save = scheduler.build_connector_meta(next_output).pool_saves[0]
        self.assertEqual((next_save.start_token, next_save.end_token), (80, 112))

    def test_preemption_discards_queued_operations_and_all_request_state(self):
        scheduler, _ = _scheduler()
        request = _request()
        external, _ = scheduler.get_num_new_matched_tokens(request, 32)
        scheduler.update_state_after_alloc(request, _blocks(), external)
        scheduler._batch_pool_saves.append(
            KVPoolSaveRequest(request.request_id, ([1],), [], 0, 16)
        )
        scheduler._batch_transfer_state.append(
            KVTransferSourceState(request.request_id, "transfer-1")
        )
        scheduler._saving_requests.add(request.request_id)
        scheduler._transferring_requests.add(request.request_id)
        scheduler._finished_waits[request.request_id] = FinishedWait(True, True)
        scheduler._delayed_releases.add(request.request_id)
        metadata = scheduler.build_connector_meta(
            _output(preempted_req_ids={request.request_id})
        )
        self.assertEqual(
            (
                metadata.pool_loads,
                metadata.pool_saves,
                metadata.transfer_plans,
                metadata.transfer_states,
            ),
            ([], [], [], []),
        )
        self.assertEqual(metadata.finished_waits, {})
        self.assertEqual(scheduler._tracked_requests, {})
        self.assertEqual(scheduler._saving_requests, set())
        self.assertEqual(scheduler._transferring_requests, set())
        self.assertFalse(scheduler.has_pending_push_work())
        scheduler.lookup_client.discard.assert_called_once_with(request.request_id)

    def test_finished_request_waits_for_both_save_and_send_and_publishes_once(self):
        scheduler, config = _scheduler(consumer=False, hit=0)
        request = _request(
            params={"do_remote_decode": True, "transfer_id": "transfer-1"}
        )
        scheduler.update_state_after_alloc(request, _blocks(), 0)
        worker = _worker(config)
        worker.get_finished(set(), scheduler.build_connector_meta(_output()))
        worker.transfer.prepare.assert_called_once_with(
            request.request_id, "transfer-1"
        )
        scheduler._batch_pool_saves.append(
            scheduler._build_save_meta(
                scheduler._tracked_requests[request.request_id], 80, decode=False
            )
        )
        self.assertEqual(
            scheduler.request_finished(request, _blocks().get_block_ids()), (True, None)
        )
        metadata = scheduler.build_connector_meta(
            _output(finished_req_ids={request.request_id})
        )
        self.assertEqual(
            metadata.finished_waits[request.request_id], FinishedWait(True, True)
        )
        with patch("mtsc.worker.new_device_event", return_value=Mock()):
            self.assertEqual(
                worker.get_finished({request.request_id}, metadata), (None, None)
            )
        self.assertEqual(scheduler.build_connector_meta(_output()).finished_waits, {})
        worker.pool.poll.return_value = PoolPollResult(
            saves=[SaveResult(request.request_id)]
        )
        self.assertEqual(
            worker.get_finished(set(), MTSCConnectorMetadata()), (None, None)
        )
        worker.pool.poll.return_value = PoolPollResult()
        worker.transfer.poll.return_value = TransferPollResult(
            sends=[TransferResult(request.request_id)]
        )
        self.assertEqual(
            worker.get_finished(set(), MTSCConnectorMetadata()),
            ({request.request_id}, None),
        )
        scheduler.update_connector_output(
            SimpleNamespace(finished_sending={request.request_id})
        )
        self.assertFalse(scheduler.has_pending_push_work())
        self.assertEqual(scheduler._tracked_requests, {})
        self.assertEqual(worker._on_save_requests, {})
        self.assertEqual(worker._finished_waits, {})

    def test_metadata_roundtrip_keeps_load_save_and_transfer_separate(self):
        _, _, metadata = self._decode_metadata()
        metadata.pool_saves = [
            KVPoolSaveRequest(
                "save-1",
                ([9],),
                [b"hash"],
                0,
                16,
                prompt_tokens=16,
            )
        ]
        metadata.transfer_states = [
            KVTransferSourceState("source-1", "transfer-1", ([1],), ready=True)
        ]
        metadata.finished_waits = {"save-1": FinishedWait(store_save=True)}
        decoded = msgspec.msgpack.decode(
            msgspec.msgpack.encode(metadata), type=MTSCConnectorMetadata
        )
        self.assertEqual(decoded, metadata)
        self.assertIsInstance(decoded.pool_loads[0], KVPoolLoadRequest)
        self.assertIsInstance(decoded.pool_saves[0], KVPoolSaveRequest)

    def test_wire_response_roundtrip_reports_transfer_ids(self):
        response = KVTransferResponse(
            KVTransferStatus.COMPLETE,
            completed_transfer_ids=["transfer-1"],
            failed_transfer_ids=["transfer-2"],
            error_message="partial failure",
            covered_regions={"transfer-1": [0, 1]},
        )
        encoded = msgspec.msgpack.encode(response)
        decoded = msgspec.msgpack.decode(encoded, type=KVTransferResponse)
        self.assertEqual(decoded, response)
        self.assertEqual(
            msgspec.msgpack.decode(encoded)["completed_transfer_ids"], ["transfer-1"]
        )
