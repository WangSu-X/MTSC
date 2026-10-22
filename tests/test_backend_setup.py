from __future__ import annotations

import asyncio
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import torch

from mtsc.kv_cache_pool import (
    MooncakeKVCachePool,
    PoolKey,
    StoreConfig,
    StoreLookupClient,
    lookup_rpc_path,
)
from mtsc.kv_transfer import MooncakeKVTransfer


def _config():
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_role="kv_producer", engine_id="producer", kv_connector_extra_config={}
        ),
        cache_config=SimpleNamespace(
            num_gpu_blocks=100, block_size=16, cache_dtype="auto"
        ),
        model_config=SimpleNamespace(
            model="model",
            revision=None,
            dtype=torch.float16,
            use_mla=False,
            get_total_num_kv_heads=lambda: 8,
        ),
        parallel_config=SimpleNamespace(
            rank=5,
            pipeline_parallel_size=2,
            local_engines_only=True,
            data_parallel_rank_local=0,
        ),
    )


def _caches():
    return SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=SimpleNamespace(block_size=16), layer_names=["layer.0"]
            )
        ],
        has_mamba_layers=False,
    )


class BackendSetupTest(unittest.TestCase):
    def test_pool_setup_maps_replicated_kv_to_shared_keys(self):
        store = Mock()
        store.setup.return_value = 0
        binding = SimpleNamespace(
            MooncakeDistributedStore=lambda: store, ReplicateConfig=Mock
        )
        rank = Mock()
        world_size = Mock()
        dcp_group = Mock()
        cases = [
            (True, 128, 1, tp_size, tp_rank, 1, tp_size, 0)
            for tp_size in (1, 2, 4, 8)
            for tp_rank in range(tp_size)
        ]
        cases += [(False, 2, 1, 8, tp_rank, 2, 4, tp_rank // 4) for tp_rank in range(8)]
        cases += [(False, 8, 1, 4, tp_rank, 8, 1, tp_rank) for tp_rank in range(4)]
        cases += [(value, 8, 1, 4, 1, 8, 1, 1) for value in (None, 1, "true")]
        cases += [(True, 128, 2, 8, tp_rank, 1, 1, tp_rank) for tp_rank in range(8)]
        cases += [(False, 2, 2, 8, tp_rank, 2, 1, tp_rank) for tp_rank in range(8)]
        with (
            patch.dict(sys.modules, {"mooncake.store": binding}),
            patch("torch.accelerator.current_device_index", return_value=0),
            patch.multiple(
                "mtsc.kv_cache_pool",
                get_tensor_model_parallel_rank=rank,
                get_tensor_model_parallel_world_size=world_size,
                get_pcp_group=Mock(
                    return_value=SimpleNamespace(world_size=1, rank_in_group=0)
                ),
                get_dcp_group=dcp_group,
                resolve_kv_cache_block_sizes=Mock(return_value=(16, 16)),
                MooncakeStoreCoordinator=Mock(),
                StoreLookupServer=Mock(),
                get_requester_local_hostname=Mock(return_value="host"),
                get_configured_preferred_segment=Mock(return_value=None),
            ),
            patch(
                "mtsc.kv_cache_pool.StoreConfig.load",
                return_value=StoreConfig("metadata", "master"),
            ),
        ):
            for mla, heads, dcp, tp, local_rank, key_heads, step, key_rank in cases:
                with self.subTest(mla=mla, dcp=dcp, tp=tp, rank=local_rank):
                    rank.return_value = local_rank
                    world_size.return_value = tp
                    dcp_group.return_value = SimpleNamespace(
                        world_size=dcp, rank_in_group=local_rank % dcp
                    )
                    config = _config()
                    config.model_config.use_mla = mla
                    config.model_config.get_total_num_kv_heads = lambda heads=heads: (
                        heads
                    )
                    pool = MooncakeKVCachePool(config, _caches())
                    try:
                        self.assertEqual(pool.use_mla, mla is True)
                        self.assertEqual(pool.num_kv_head, key_heads)
                        self.assertEqual(pool.tp_rank, local_rank)
                        self.assertEqual(pool.tp_size, tp)
                        self.assertEqual(pool.pp_size, 2)
                        self.assertEqual(pool.pp_rank, (5 // tp) % 2)
                        self.assertEqual(pool.pcp_size, 1)
                        self.assertEqual(pool.pcp_rank, 0)
                        self.assertEqual(pool.dcp_size, dcp)
                        self.assertEqual(pool.dcp_rank, local_rank % dcp)
                        self.assertEqual(pool.block_size, 16)
                        self.assertEqual(pool.put_step, step)
                        self.assertEqual(pool.head_or_tp_rank, key_rank)
                        self.assertEqual(pool.metadata.tp_rank, key_rank)
                        self.assertEqual(pool.metadata.pp_rank, pool.pp_rank)
                        self.assertEqual(pool.metadata.pcp_rank, pool.pcp_rank)
                        self.assertEqual(pool.metadata.dcp_rank, pool.dcp_rank)
                        self.assertEqual(pool.databases[0].metadata.tp_rank, key_rank)
                        # GET/PUT and lookup must address the same logical rank.
                        keys = {
                            PoolKey(metadata, "61").to_string()
                            for metadata in pool._lookup_metadata[0]
                        }
                        self.assertIn(
                            pool.databases[0]._make_key_by_hash("61").to_string(),
                            keys,
                        )
                        self.assertEqual(
                            len(keys),
                            (tp if dcp > 1 else min(tp, key_heads)) * pool.pp_size,
                        )
                    finally:
                        pool.close()

    def test_invalid_store_timeout_fails_before_store_initialization(self):
        factory = Mock()
        binding = SimpleNamespace(
            MooncakeDistributedStore=factory, ReplicateConfig=Mock
        )
        with patch.dict(sys.modules, {"mooncake.store": binding}):
            for value in (
                0,
                -1,
                float("nan"),
                float("inf"),
                -float("inf"),
                "nan",
                "inf",
                "1e309",
            ):
                with self.subTest(value=value):
                    config = _config()
                    config.kv_transfer_config.kv_connector_extra_config[
                        "mtsc_store_get_timeout_seconds"
                    ] = value
                    with self.assertRaisesRegex(
                        ValueError, "mtsc_store_get_timeout_seconds"
                    ):
                        MooncakeKVCachePool(config, _caches())
        factory.assert_not_called()

    def test_invalid_pd_timeout_fails_before_engine_initialization(self):
        factory = Mock()
        with (
            patch.dict(
                sys.modules,
                {"mooncake.engine": SimpleNamespace(TransferEngine=factory)},
            ),
            patch("mtsc.kv_transfer.current_platform") as platform,
        ):
            for value in (
                0,
                -1,
                float("nan"),
                float("inf"),
                -float("inf"),
                "nan",
                "inf",
                "1e309",
            ):
                with self.subTest(value=value):
                    config = _config()
                    config.parallel_config.pipeline_parallel_size = 1
                    config.kv_transfer_config.kv_connector_extra_config[
                        "mtsc_pd_timeout_seconds"
                    ] = value
                    with self.assertRaisesRegex(ValueError, "mtsc_pd_timeout_seconds"):
                        MooncakeKVTransfer(config, _caches())
        factory.assert_not_called()
        platform.set_device.assert_not_called()

    def test_invalid_lookup_timeout_fails_before_creating_zmq_context(self):
        with (
            patch(
                "mtsc.kv_cache_pool.lookup_rpc_path",
                return_value="ipc:///tmp/mtsc-test",
            ),
            patch("mtsc.kv_cache_pool.zmq.Context") as context,
        ):
            for value in (0, -1, "nan", "inf", "-inf", "1e309", 2**31 / 1000):
                with self.subTest(value=value):
                    config = _config()
                    config.kv_transfer_config.kv_connector_extra_config[
                        "mtsc_store_lookup_timeout_seconds"
                    ] = value
                    with self.assertRaisesRegex(
                        ValueError, "mtsc_store_lookup_timeout_seconds"
                    ):
                        StoreLookupClient(config)
        context.assert_not_called()

    def test_lookup_timeout_defaults_and_submillisecond_values(self):
        with (
            patch(
                "mtsc.kv_cache_pool.lookup_rpc_path",
                return_value="ipc:///tmp/mtsc-test",
            ),
            patch("mtsc.kv_cache_pool.zmq.Context"),
            patch("mtsc.kv_cache_pool.StoreLookupClient._make_socket"),
        ):
            for extra, expected in (
                ({}, 10000),
                ({"mtsc_store_lookup_timeout_seconds": "0.025"}, 25),
                ({"mtsc_store_lookup_timeout_seconds": 0.0001}, 1),
            ):
                with self.subTest(extra=extra):
                    config = _config()
                    config.kv_transfer_config.kv_connector_extra_config.update(extra)
                    client = StoreLookupClient(config)
                    try:
                        self.assertEqual(client._timeout_ms, expected)
                    finally:
                        client.close()

    def test_lookup_rpc_identifier_is_preserved_in_ipc_path(self):
        with (
            patch("mtsc.kv_cache_pool.get_mooncake_dp_engine_index", return_value=2),
            patch("mtsc.kv_cache_pool.socket.gethostname", return_value="host"),
            patch("mtsc.kv_cache_pool.envs.VLLM_RPC_BASE_PATH", "/tmp"),
        ):
            for value in (0, "001", 8100):
                with self.subTest(value=value):
                    config = _config()
                    config.kv_transfer_config.kv_connector_extra_config[
                        "lookup_rpc_port"
                    ] = value
                    self.assertEqual(
                        lookup_rpc_path(config),
                        f"ipc:///tmp/mtsc_lookup_{value}_host_host_dp_rank2",
                    )

    def test_pool_setup_keeps_cross_rank_lookup_keys_after_localizing_topology(self):
        store = Mock()
        store.setup.return_value = 0
        store.batch_is_exist.side_effect = lambda keys: [1] * len(keys)
        coordinator = Mock()
        coordinator.lookup_mask.return_value = (None,)
        coordinator.block_hashes_for_spec.return_value = [b"a", b"b"]
        coordinator.find_longest_cache_hit.return_value = ([], 32)
        binding = SimpleNamespace(
            MooncakeDistributedStore=lambda: store, ReplicateConfig=Mock
        )
        with (
            patch.dict(sys.modules, {"mooncake.store": binding}),
            patch("torch.accelerator.current_device_index", return_value=0),
            patch("mtsc.kv_cache_pool.get_tensor_model_parallel_rank", return_value=1),
            patch(
                "mtsc.kv_cache_pool.get_tensor_model_parallel_world_size",
                return_value=4,
            ),
            patch(
                "mtsc.kv_cache_pool.get_pcp_group",
                return_value=SimpleNamespace(world_size=2, rank_in_group=1),
            ),
            patch(
                "mtsc.kv_cache_pool.get_dcp_group",
                return_value=SimpleNamespace(world_size=2, rank_in_group=1),
            ),
            patch(
                "mtsc.kv_cache_pool.resolve_kv_cache_block_sizes", return_value=(16, 16)
            ),
            patch(
                "mtsc.kv_cache_pool.MooncakeStoreCoordinator", return_value=coordinator
            ),
            patch(
                "mtsc.kv_cache_pool.StoreConfig.load",
                return_value=StoreConfig("metadata", "master"),
            ),
            patch(
                "mtsc.kv_cache_pool.get_requester_local_hostname",
                return_value="host",
            ),
            patch(
                "mtsc.kv_cache_pool.get_configured_preferred_segment",
                return_value=None,
            ),
            patch("torch.accelerator.current_device_index", return_value=0),
        ):
            pool = MooncakeKVCachePool(_config(), _caches())
            self.addCleanup(pool.close)
            config = _config()
            config.kv_transfer_config.kv_connector_extra_config.update(
                mtsc_store_get_timeout_seconds="12.5", mtsc_store_load_workers=-2
            )
            override = MooncakeKVCachePool(config, _caches())
            try:
                self.assertEqual(override.load_timeout, 12.5)
                self.assertEqual(override._load_pool._max_workers, 1)
            finally:
                override.close()
        with patch("mtsc.kv_cache_pool._make_external_cached_pool") as cached_pool:
            self.assertEqual(pool.lookup(32, [b"a", b"b"]), 32)
            self.assertEqual(cached_pool.call_args.args[1], {(0, b"a"), (0, b"b")})
        keys = store.batch_is_exist.call_args.args[0]
        # Each hash must exist on all TP/PP/PCP/DCP participants.
        self.assertEqual(len(keys), 32)
        missing = "model@tp_rank:3@pcp1@dcp1@pp_rank:1@group:0@62"
        self.assertIn(missing, keys)

        # One missing participant must exclude the whole candidate block.
        store.batch_is_exist.side_effect = lambda keys: [
            int(key != missing) for key in keys
        ]
        with patch("mtsc.kv_cache_pool._make_external_cached_pool") as cached_pool:
            pool.lookup(32, [b"a", b"b"])
            self.assertEqual(cached_pool.call_args.args[1], {(0, b"a")})

        # A serializer change must affect lookup and GET/PUT identically.
        serialize = PoolKey.to_string
        with (
            patch.object(PoolKey, "to_string", lambda key: f"shared@{serialize(key)}"),
            patch("mtsc.kv_cache_pool._make_external_cached_pool"),
        ):
            pool.lookup(32, [b"a", b"b"])
            lookup_keys = set(store.batch_is_exist.call_args.args[0])
            io_keys = {
                key.to_string()
                for _, _, key in pool._process_tokens(
                    pool.databases[0], 32, [b"a", b"b"], 0
                )
            }
            self.assertTrue(io_keys)
            self.assertTrue(io_keys <= lookup_keys)
            self.assertTrue(all(key.startswith("shared@") for key in lookup_keys))

        self.assertEqual(pool.databases[0].metadata.pp_rank, 1)
        self.assertEqual(pool.databases[0].metadata.pcp_rank, 1)
        self.assertEqual(pool.databases[0].metadata.dcp_rank, 1)
        self.assertEqual((pool.pcp_size, pool.pcp_rank), (2, 1))
        self.assertEqual((pool.dcp_size, pool.dcp_rank), (2, 1))
        self.assertEqual((pool.pp_size, pool.pp_rank), (2, 1))
        self.assertEqual(pool.load_timeout, 180)
        self.assertEqual(pool._load_pool._max_workers, 2)

    def test_transfer_registers_worker_with_cached_url_and_closes_loop(self):
        engine = Mock()
        engine.initialize.return_value = 0
        engine.get_rpc_port.return_value = 123
        with (
            patch.dict(
                sys.modules,
                {"mooncake.engine": SimpleNamespace(TransferEngine=lambda: engine)},
            ),
            patch("mtsc.kv_transfer.get_tensor_model_parallel_rank", return_value=0),
            patch(
                "mtsc.kv_transfer.get_tensor_model_parallel_world_size", return_value=1
            ),
            patch(
                "mtsc.kv_transfer.get_pp_group",
                return_value=SimpleNamespace(rank_in_group=0),
            ),
            patch("mtsc.kv_transfer.is_npu_platform", return_value=True),
            patch("mtsc.kv_transfer.get_current_attn_backends", return_value=[]),
            patch(
                "mtsc.kv_transfer.select_common_block_size",
                side_effect=lambda size, backends: size,
            ),
            patch("mtsc.kv_transfer.current_platform"),
            patch(
                "mtsc.kv_transfer._bootstrap_address", return_value=("127.0.0.1", 8998)
            ),
            patch("mtsc.kv_transfer._launch_bootstrap", return_value=False),
            patch("mtsc.kv_transfer.envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", 37.5),
            patch("torch.accelerator.current_device_index", return_value=0),
        ):
            config = _config()
            config.parallel_config.pipeline_parallel_size = 1
            transfer = MooncakeKVTransfer(config, _caches())
            self.addCleanup(transfer.close)
            self.assertEqual(transfer.timeout, 37.5)
            self.assertEqual(transfer._sender_executor._max_workers, 10)
            self.assertEqual(transfer.num_sender_tasks, 20)
            config = _config()
            config.parallel_config.pipeline_parallel_size = 1
            config.kv_transfer_config.kv_connector_extra_config.update(
                mtsc_pd_timeout_seconds="12.5", num_workers=0
            )
            override = MooncakeKVTransfer(config, _caches())
            try:
                self.assertEqual(override.timeout, 12.5)
                self.assertEqual(override._sender_executor._max_workers, 1)
                self.assertEqual(override.num_sender_tasks, 2)
            finally:
                override.close()
        client = AsyncMock()
        client.post.return_value = Mock()
        context = AsyncMock()
        context.__aenter__.return_value = client
        with patch("mtsc.kv_transfer.httpx.AsyncClient", return_value=context):
            asyncio.run(transfer._register_worker(12345))
        url = client.post.call_args.args[0]
        payload = client.post.call_args.kwargs["json"]
        self.assertEqual(url, "http://127.0.0.1:8998/register")
        self.assertEqual(payload["engine_id"], "producer")
        self.assertTrue(payload["addr"].endswith(":12345"))
        transfer.close()
        self.assertFalse(transfer._sender_listener_t.is_alive())
        self.assertTrue(transfer.sender_loop.is_closed())
