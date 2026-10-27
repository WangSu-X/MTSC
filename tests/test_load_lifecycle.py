"""Exercise the vLLM load/forward/completion hook ordering."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from test_metadata import _blocks, _output, _request, _scheduler, _worker
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

from mtsc.connector import MTSCConnector
from mtsc.kv_cache_pool import LoadResult, PoolPollResult
from mtsc.kv_transfer import TransferPollResult, TransferResult
from mtsc.protocol import MTSCConnectorMetadata


def _decode(hit):
    scheduler, config = _scheduler(hit=hit)
    request = _request()
    external, _ = scheduler.get_num_new_matched_tokens(request, 32)
    scheduler.update_state_after_alloc(request, _blocks(), external)
    worker = _worker(config)
    connector = object.__new__(MTSCConnector)
    connector.scheduler = None
    connector.worker = worker
    connector.bind_connector_metadata(scheduler.build_connector_meta(_output()))
    return connector, worker, request


def test_store_load_starts_before_forward_and_is_not_reissued_at_completion():
    connector, worker, request = _decode(hit=80)
    connector.start_load_kv(None)
    worker.pool.load.assert_called_once()
    worker.pool.poll.assert_not_called()
    worker.pool.save.assert_not_called()
    worker.transfer.recv.assert_not_called()
    worker.transfer.send.assert_not_called()

    # The GET may finish during forward; only get_finished reports it and
    # starts PD at the actual prefix, not the original lookup boundary.
    worker.pool.poll.return_value = PoolPollResult(
        loads=[LoadResult(request.request_id, 48, "partial miss")]
    )
    worker.pool.take_errors.return_value = {13, 14}
    assert connector.get_finished(set()) == (None, None)
    worker.pool.load.assert_called_once()
    worker.transfer.recv.assert_called_once()
    assert worker.transfer.recv.call_args.args[0][0].block_ids == ((13, 14, 15, 16),)
    assert connector.get_block_ids_with_load_errors() == set()

    worker.pool.poll.return_value = PoolPollResult()
    worker.pool.take_errors.return_value = set()
    worker.transfer.poll.return_value = TransferPollResult(
        recvs=[TransferResult(request.request_id)]
    )
    assert connector.get_finished(set()) == (None, {request.request_id})
    worker.transfer.poll.return_value = TransferPollResult()
    assert connector.get_finished(set()) == (None, None)
    worker.pool.load.assert_called_once()
    worker.transfer.recv.assert_called_once()


def test_store_miss_starts_pd_before_forward_and_completes_on_empty_step():
    connector, worker, request = _decode(hit=0)
    connector.start_load_kv(None)
    worker.pool.load.assert_not_called()
    worker.transfer.recv.assert_called_once()
    assert worker.transfer.recv.call_args.args[0][0].block_ids == (
        (12, 13, 14, 15, 16),
    )
    assert connector.get_finished(set()) == (None, None)
    connector.clear_connector_metadata()

    connector.bind_connector_metadata(MTSCConnectorMetadata())
    connector.start_load_kv(None)
    worker.transfer.poll.return_value = TransferPollResult(
        recvs=[TransferResult(request.request_id)]
    )
    assert connector.get_finished(set()) == (None, {request.request_id})
    worker.transfer.recv.assert_called_once()


def test_preemption_fences_old_load_before_starting_reallocated_attempt():
    connector, worker, request = _decode(hit=80)
    metadata = connector._get_connector_metadata()
    connector.start_load_kv(None)
    connector.handle_preemptions(
        MTSCConnectorMetadata(preempted_request_ids={request.request_id})
    )
    worker.pool.preempt.assert_called_once_with(request.request_id)
    worker.transfer.preempt.assert_called_once_with(request.request_id)
    retry = MTSCConnectorMetadata(
        pool_loads=[replace(metadata.pool_loads[0], load_id=2)],
        transfer_plans=[replace(metadata.transfer_plans[0], load_id=2)],
    )
    connector.clear_connector_metadata()
    connector.bind_connector_metadata(retry)
    connector.start_load_kv(None)
    assert worker.pool.load.call_count == 2
    assert worker._on_transfer[request.request_id].plan.load_id == 2
    worker.transfer.recv.assert_not_called()


@pytest.mark.parametrize("role", ["kv_consumer", "kv_both", "kv_producer"])
@pytest.mark.parametrize(
    "setting,enabled",
    [
        ({}, False),
        ({"mtsc_decode_save": False}, False),
        ({"mtsc_decode_save": "false"}, False),
        ({"mtsc_decode_save": True}, True),
        ({"mtsc_decode_save": "true"}, True),
    ],
)
def test_decode_save_default_and_overrides_reach_scheduler(role, setting, enabled):
    _, config = _scheduler(role=role, extra=setting)
    config.kv_transfer_config.kv_load_failure_policy = "recompute"
    caches = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16))],
        has_mamba_layers=False,
    )
    with (
        patch("mtsc.scheduler.StoreLookupClient", return_value=Mock()),
        patch("mtsc.scheduler.resolve_kv_cache_block_sizes", return_value=(16, 16)),
    ):
        connector = MTSCConnector(config, KVConnectorRole.SCHEDULER, caches)
    scheduler = connector.scheduler
    assert scheduler.decode_save == (enabled and role != "kv_producer")
    request = _request(params={}, tokens=112, prompt=50)
    scheduler.update_state_after_alloc(request, _blocks(), 0)
    metadata = scheduler.build_connector_meta(
        _output(
            scheduled_new_reqs=[
                SimpleNamespace(
                    req_id=request.request_id,
                    block_ids=_blocks().get_block_ids(),
                    num_computed_tokens=0,
                )
            ],
            num_scheduled_tokens={request.request_id: 83},
        )
    )
    if role == "kv_producer":
        assert metadata.pool_saves[0].start_token == 0
    elif enabled:
        assert metadata.pool_saves[0].start_token == 64
    else:
        assert metadata.pool_saves == []
        assert scheduler.request_finished(request, _blocks().get_block_ids()) == (
            False,
            None,
        )
    connector.shutdown()
