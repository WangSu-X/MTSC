"""Store key parity with the local vLLM Mooncake Store implementation."""

from __future__ import annotations

from dataclasses import asdict

import pytest
import torch
from test_backends import _drain, _pool
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    ChunkedTokenDatabase as OfficialDatabase,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    KeyMetadata as OfficialMetadata,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

from mtsc.kv_cache_pool import (
    ChunkedTokenDatabase,
    KeyMetadata,
    LoadEvent,
    MooncakeKVCachePool,
    MooncakeStoreCoordinator,
    SaveEvent,
)


@pytest.mark.parametrize(
    "block_size,start_token", [(16, 0), (16, 16), (32, 0), (32, 32)]
)
def test_process_tokens_matches_official_keys(block_size, start_token):
    metadata = KeyMetadata("model", 3, 1, 2, 1, group_id=2)
    hashes = [bytes([index]) * 32 for index in range(1, 5)]
    ours = ChunkedTokenDatabase(metadata, block_size, 16)
    official = OfficialDatabase(OfficialMetadata(**asdict(metadata)), block_size, 16)
    pool = object.__new__(MooncakeKVCachePool)
    actual = [
        (start, end, key.to_string())
        for start, end, key in pool._process_tokens(ours, 64, hashes, start_token)
    ]
    expected = [
        (start, end, key.to_string())
        for start, end, key in official.process_tokens(64, hashes, start_token)
    ]
    assert actual == expected
    assert all(
        key.startswith("model@tp_rank:3@pcp1@dcp2@pp_rank:1@group:2@")
        for _, _, key in actual
    )


def test_process_tokens_passes_through_key_objects_without_serializing():
    class UnserializedKey:
        def to_string(self):
            raise AssertionError("Token iteration must preserve the key object")

    key = UnserializedKey()

    class Database:
        block_size = 16

        def process_tokens(self, token_count, hashes, start_token):
            yield 0, 16, key

    pool = object.__new__(MooncakeKVCachePool)
    assert list(pool._process_tokens(Database(), 16, [b"hash"], 0)) == [(0, 16, key)]


def test_lookup_put_and_get_use_the_same_official_keys_across_groups():
    pool = _pool()
    hashes = tuple(bytes([index]) * 32 for index in range(1, 5))
    groups = [
        KVCacheGroupSpec(
            [f"layer.{index}"],
            FullAttentionSpec(
                block_size=size,
                num_kv_heads=1,
                head_size=8,
                dtype=torch.float16,
            ),
        )
        for index, size in enumerate((16, 32))
    ]
    pool.groups = groups
    pool.hash_block_size = 16
    pool.coordinator = MooncakeStoreCoordinator(groups, 32, 16)
    pool.databases = [
        ChunkedTokenDatabase(
            KeyMetadata("model", 0, 0, 0, 0, group_id=index),
            group.kv_cache_spec.block_size,
            16,
        )
        for index, group in enumerate(groups)
    ]
    pool._store_masks = MooncakeKVCachePool._store_masks.__get__(pool)
    for index, database in enumerate(pool.databases):
        database.set_kv_caches_base_addr([1000 + index * 1000])
        database.set_block_len([database.block_size])
    pool.tp_size = 1
    pool.num_kv_head = 1
    pool.pp_size = pool.pcp_size = pool.dcp_size = 1
    pool._init_lookup_metadata()
    saved = set()

    def put(keys, addresses, sizes, replicate_config):
        assert all(isinstance(key, str) for key in keys)
        pool.store.puts.append(list(keys))
        saved.update(keys)
        return [0] * len(keys)

    def get(keys, addresses, sizes):
        pool.store.gets.append(list(keys))
        return [0 if key in saved else -1 for key in keys]

    pool.store.batch_is_exist = lambda keys: [int(key in saved) for key in keys]
    pool.store.batch_put_from_multi_buffers = put
    pool.store.batch_get_into_multi_buffers = get
    blocks = ((10, 11, 12, 13), (20, 21))
    try:
        pool.save(SaveEvent("save", blocks, hashes, 0, 64))
        assert _drain(pool, "saves")[0].error is None
        expected = {
            key.to_string()
            for index, size in enumerate((16, 32))
            for _, _, key in OfficialDatabase(
                OfficialMetadata("model", 0, 0, 0, 0, group_id=index), size, 16
            ).process_tokens(64, hashes)
        }
        assert saved == expected
        assert pool.lookup(64, hashes) == 64
        pool.load(LoadEvent("load", blocks, hashes, 0, 64))
        result = _drain(pool, "loads")[0]
        assert result.loaded_tokens == 64
        assert result.error is None
        assert set(pool.store.gets[0]) == expected
        assert pool.take_errors() == set()
    finally:
        pool.close()
