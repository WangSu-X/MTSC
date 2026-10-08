"""Exercise Store lookup with vLLM's Mooncake connector unavailable."""

import subprocess
import sys
from pathlib import Path


def test_store_without_vllm_mooncake_connector():
    script = """
import importlib.abc
import sys

class BlockMooncakeConnector(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake"
        ):
            raise ModuleNotFoundError(fullname)

sys.meta_path.insert(0, BlockMooncakeConnector())

import torch
from mtsc.kv_cache_pool import MooncakeKVCachePool
from mtsc.connector import MTSCConnector
from mtsc.kv_cache_pool import ExternalCachedBlockPool, MooncakeStoreCoordinator
from mtsc.kv_cache_pool import ChunkedTokenDatabase, KeyMetadata
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

spec = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=8,
                         dtype=torch.float16)
groups = [KVCacheGroupSpec(["layer0"], spec),
          KVCacheGroupSpec(["layer1"], spec)]
coordinator = MooncakeStoreCoordinator(groups, 16, 16)
hashes = [BlockHash(b"a" * 32), BlockHash(b"b" * 32)]
present = {(0, bytes(hashes[0])), (1, bytes(hashes[0])),
           (0, bytes(hashes[1]))}
_, hit = coordinator.find_longest_cache_hit(
    hashes, 32, ExternalCachedBlockPool(present))
assert hit == 16, hit

database = ChunkedTokenDatabase(KeyMetadata("model", 0, 0, 0, 0), 32, 16)
chunks = list(database.process_tokens(32, hashes))
assert len(chunks) == 1
assert chunks[0][2].chunk_hash == (hashes[0] + hashes[1]).hex()
database.set_kv_caches_base_addr([1000, 2000])
database.set_block_len([64, 64])
assert database.prepare_value(0, 32, [3]) == ([1192, 2192], [64, 64], 3)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
