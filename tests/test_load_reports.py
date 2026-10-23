"""Request-level load reporting through vLLM's real output aggregator."""

from dataclasses import replace
from unittest.mock import patch

import pytest
from test_metadata import _blocks, _output, _request, _scheduler, _worker
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput

from mtsc.kv_cache_pool import LoadResult, PoolPollResult
from mtsc.kv_transfer import TransferPollResult, TransferResult
from mtsc.protocol import MTSCConnectorMetadata, MTSCLoadReport, MTSCWorkerMetadata


def _allocated(*, consumer=True, hit=80, workers=2):
    scheduler, config = _scheduler(consumer=consumer, hit=hit)
    config.parallel_config.world_size = workers
    scheduler._expected_load_workers = workers
    request = (
        _request()
        if consumer
        else _request(params={"do_remote_decode": True, "transfer_id": "p-x"})
    )
    external, _ = scheduler.get_num_new_matched_tokens(request, 32)
    scheduler.update_state_after_alloc(request, _blocks(), external)
    return scheduler, config, request, scheduler.build_connector_meta(_output())


def _report(request_id, load_id, rank, **changes):
    report = MTSCLoadReport(
        request_id, load_id, rank, 32, 48, 32, 100.0, 10.0, 110.0, "ok", "ok"
    )
    return replace(report, **changes)


def _aggregate(aggregator, *reports):
    outputs = [
        ModelRunnerOutput.with_kv_conn_output_only(
            KVConnectorOutput(
                kv_connector_worker_meta=MTSCWorkerMetadata([report])
                if report
                else None
            )
        )
        for report in reports
    ]
    return aggregator.aggregate(outputs).kv_connector_output


def test_rank_reports_across_steps_print_once_and_keep_maxima_separate():
    scheduler, _, request, metadata = _allocated()
    load_id = metadata.transfer_plans[0].load_id
    first = _report(request.request_id, load_id, 0)
    second = _report(request.request_id, load_id, 1, store_ms=10.0, pd_ms=100.0)
    aggregator = KVOutputAggregator(expected_finished_count=2)
    with patch("mtsc.scheduler.logger.info") as log:
        scheduler.update_connector_output(_aggregate(aggregator, first, None))
        log.assert_not_called()
        scheduler.update_connector_output(_aggregate(aggregator, None, second))
        log.assert_called_once()
        message = log.call_args.args[0] % log.call_args.args[1:]
        assert "workers=2/2" in message
        assert "store_ms_max=100.000 pd_ms_max=100.000 total_ms_max=110.000" in message
        assert "local_tokens=32 store_tokens=48 pd_tokens=32" in message
        assert "prompt_tokens=112 target_tokens=112" in message
        assert scheduler._load_traces == {}
        # A duplicated old record cannot print the completed request again.
        scheduler.update_connector_output(_aggregate(aggregator, first, second))
        log.assert_called_once()


def test_different_rank_prefixes_are_ranges_not_token_sums():
    scheduler, _, request, metadata = _allocated()
    load_id = metadata.transfer_plans[0].load_id
    reports = [
        _report(request.request_id, load_id, 0),
        _report(
            request.request_id,
            load_id,
            1,
            store_tokens=16,
            pd_tokens=64,
            store_status="error",
        ),
    ]
    with patch("mtsc.scheduler.logger.info") as log:
        scheduler.update_connector_output(
            KVConnectorOutput(kv_connector_worker_meta=MTSCWorkerMetadata(reports))
        )
        message = log.call_args.args[0] % log.call_args.args[1:]
        assert "store_tokens=16..48 pd_tokens=32..64" in message
        assert "store_status=error,ok status=ok" in message


def test_preempted_rank_record_does_not_complete_new_load_attempt():
    scheduler, _, request, metadata = _allocated()
    old_id = metadata.transfer_plans[0].load_id
    scheduler.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=MTSCWorkerMetadata(
                [_report(request.request_id, old_id, 0)]
            )
        )
    )
    scheduler.build_connector_meta(_output(preempted_req_ids={request.request_id}))
    assert scheduler._load_traces == {}
    retry = _request()
    external, _ = scheduler.get_num_new_matched_tokens(retry, 32)
    scheduler.update_state_after_alloc(retry, _blocks(), external)
    new_id = scheduler.build_connector_meta(_output()).transfer_plans[0].load_id
    assert new_id != old_id
    with patch("mtsc.scheduler.logger.info") as log:
        scheduler.update_connector_output(
            KVConnectorOutput(
                kv_connector_worker_meta=MTSCWorkerMetadata(
                    [
                        _report(request.request_id, old_id, 1),
                        _report(request.request_id, new_id, 0),
                    ]
                )
            )
        )
        log.assert_not_called()
        scheduler.update_connector_output(
            KVConnectorOutput(
                kv_connector_worker_meta=MTSCWorkerMetadata(
                    [
                        _report(request.request_id, new_id, 1),
                    ]
                )
            )
        )
        log.assert_called_once()
        assert scheduler._load_traces == {}


def test_p_load_report_keeps_elapsed_time_and_actual_prefix():
    _, config, request, metadata = _allocated(consumer=False)
    worker = _worker(config)
    worker.get_finished(set(), metadata)
    worker.pool.poll.return_value = PoolPollResult(
        loads=[LoadResult(request.request_id, 64, "partial", elapsed_ms=12.5)]
    )
    worker.get_finished(set(), MTSCConnectorMetadata())
    report = worker.build_connector_worker_meta().load_reports[0]
    assert report.load_id == metadata.pool_loads[0].load_id
    assert report.store_tokens == 32  # Actual [32,64), not lookup [32,80).
    assert report.local_tokens == 32
    assert report.store_ms == report.total_ms == 12.5
    assert report.status == report.store_status == "error"
    assert report.pd_tokens == 0
    assert worker.build_connector_worker_meta() is None


@pytest.mark.parametrize("loaded,error", [(48, "partial"), (80, None)])
def test_d_report_uses_actual_store_result_and_logical_pd_suffix(loaded, error):
    _, config, request, metadata = _allocated()
    worker = _worker(config)
    worker.get_finished(set(), metadata)
    worker.pool.poll.return_value = PoolPollResult(
        loads=[LoadResult(request.request_id, loaded, error, elapsed_ms=7.5)]
    )
    worker.get_finished(set(), MTSCConnectorMetadata())
    assert worker.build_connector_worker_meta() is None
    worker.pool.poll.return_value = PoolPollResult()
    worker.transfer.poll.return_value = TransferPollResult(
        recvs=[TransferResult(request.request_id)]
    )
    worker.get_finished(set(), MTSCConnectorMetadata())
    report = worker.build_connector_worker_meta().load_reports[0]
    assert report.store_ms == 7.5
    assert report.store_tokens == loaded - 32
    assert report.pd_tokens == 112 - loaded
    assert report.local_tokens + report.store_tokens + report.pd_tokens == 112
    assert report.store_status == ("error" if error else "ok")
    assert report.status == "ok"  # PD can recover the Store miss.
    assert report.total_ms >= report.pd_ms >= 0


def test_prompt_snapshot_is_used_before_request_release_cleanup():
    scheduler, _, request, metadata = _allocated(consumer=False)
    load_id = metadata.pool_loads[0].load_id
    request.num_tokens += 10
    scheduler._tracked_requests.clear()
    with patch("mtsc.scheduler.logger.info") as log:
        scheduler.update_connector_output(
            KVConnectorOutput(
                finished_sending={request.request_id},
                kv_connector_worker_meta=MTSCWorkerMetadata(
                    [
                        _report(request.request_id, load_id, rank, pd_tokens=0)
                        for rank in range(2)
                    ]
                ),
            )
        )
        log.assert_called_once()
        message = log.call_args.args[0] % log.call_args.args[1:]
        assert "[MTSC] store_load: role=P" in message
        assert "prompt_tokens=112" in message
        assert scheduler._load_traces == {}


def test_store_miss_is_one_scheduler_log_without_worker_reports():
    scheduler, _ = _scheduler(consumer=False, hit=0)
    request = _request(params={"do_remote_decode": True, "transfer_id": "p-x"})
    external, _ = scheduler.get_num_new_matched_tokens(request, 0)
    with patch("mtsc.scheduler.logger.info") as log:
        scheduler.update_state_after_alloc(request, _blocks(), external)
        log.assert_called_once()
        message = log.call_args.args[0] % log.call_args.args[1:]
        assert "prompt_tokens=112 store_tokens=0" in message
        assert "status=skipped" in message
        assert scheduler._load_traces == {}
