"""Run isolated GPU PD correctness checks and replay existing trace sessions.

Use the same arguments with two immutable --source snapshots. Services use GPUs
0/1 (P TP2), 2/3 (D TP2), private ports, eager FP16 and no local prefix cache.
No installed packages or existing experiment services are modified.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import re
import signal
import statistics
import subprocess
import time
from pathlib import Path

import aiohttp


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def percentile(values, fraction):
    values = sorted(values)
    return values[min(len(values) - 1, int((len(values) - 1) * fraction))]


class Experiment:
    def __init__(self, args):
        self.args = args
        self.output = args.output.resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.processes = []
        self.logs = []
        self.env = {
            **os.environ,
            "PYTHONPATH": str(args.source.resolve()),
            "PYTHONHASHSEED": "0",
            "PYTHONUNBUFFERED": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "MC_RPC_PROTOCOL": "tcp",
            "MC_STORE_MEMCPY": "0",
            "MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER": "65535",
            "MC_TCP_MAX_PENDING_ADMISSIONS_PER_PEER": "65535",
            "MC_TCP_ADMISSION_TIMEOUT_MS": "60000",
            "MC_TCP_SLICE_SIZE": "1048576",
            "MOONCAKE_REQUESTER_LOCAL_HOSTNAME": "127.0.0.1",
            "VLLM_SERVER_DEV_MODE": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_MOONCAKE_BOOTSTRAP_PORT": "8970",
        }
        self.env.pop("MC_TCP_PROTO", None)
        config = self.output / "mooncake.json"
        dump(
            config,
            {
                "metadata_server": "http://127.0.0.1:8097/metadata",
                "master_server_address": "127.0.0.1:50097",
                "protocol": "tcp",
                "global_segment_size": 8 << 30,
                "local_buffer_size": 128 << 20,
            },
        )
        self.env["MOONCAKE_CONFIG_PATH"] = str(config)
        dump(
            self.output / "manifest.json",
            {
                "source": str(args.source.resolve()),
                "source_sha256": {
                    str(path.relative_to(args.source)): hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                    for package in ("mtsc", "proxy")
                    for path in sorted((args.source / package).glob("*.py"))
                },
                "model": str(args.model),
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in (
                        "vllm",
                        "torch",
                        "mooncake-transfer-engine",
                        "transformers",
                    )
                },
                "decode_save": args.decode_save,
                "concurrency": args.concurrency,
                "rounds": args.rounds,
                "gpu_ids": [0, 1, 2, 3],
                "dataset_sha256": {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(args.dataset_dir.glob("*.jsonl"))
                }
                if args.dataset_dir
                else {},
            },
        )

    def launch(self, name, command, **env):
        log = (self.output / f"{name}.log").open("w")
        self.logs.append(log)
        process = subprocess.Popen(
            command,
            env={**self.env, **env},
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.processes.append((name, process))
        dump(self.output / f"{name}.command.json", command)
        (self.output / f"{name}.pid").write_text(str(process.pid))
        return process

    async def wait_ready(self, client, url, process):
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Service exited ({process.returncode}): {url}")
            try:
                async with client.get(
                    url, timeout=aiohttp.ClientTimeout(total=2)
                ) as response:
                    if response.status == 200:
                        return
            except (aiohttp.ClientError, TimeoutError):
                pass
            await asyncio.sleep(1)
        raise TimeoutError(f"Service startup timed out: {url}")

    def close(self):
        for _name, process in reversed(self.processes):
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
        for log in self.logs:
            log.close()

    async def reset(self, client):
        # P's reset fences pending Store saves before clearing this private Store.
        for port, external in ((8170, "true"), (8270, "false")):
            async with client.post(
                f"http://127.0.0.1:{port}/reset_prefix_cache?reset_external={external}"
            ) as response:
                response.raise_for_status()
                await response.read()

    async def complete(self, client, port, payload):
        started = time.perf_counter()
        async with client.post(
            f"http://127.0.0.1:{port}/v1/completions", json=payload
        ) as response:
            body = await response.json()
            if response.status != 200:
                raise RuntimeError(body)
        return {"response": body, "elapsed_ms": (time.perf_counter() - started) * 1000}

    async def correctness(self, client):
        log_offset = (self.output / "decode.log").stat().st_size
        results = []
        for length in (1024, 1031):
            prompt = [1000 + i % 2000 for i in range(length)]
            payload = {
                "model": "mtsc-qwen",
                "prompt": prompt,
                "max_tokens": 16,
                "min_tokens": 16,
                "temperature": 0,
                "ignore_eos": True,
            }
            await self.reset(client)
            full = await self.complete(
                client,
                8270,
                {
                    **payload,
                    "cache_salt": f"independent-full-{length}",
                },
            )
            cold = await self.complete(client, 8370, payload)
            await asyncio.sleep(1)
            warm = await self.complete(client, 8370, payload)
            texts = [x["response"]["choices"][0]["text"] for x in (full, cold, warm)]
            record = {
                "prompt_tokens": length,
                "full": full,
                "cold": cold,
                "warm": warm,
                "same_output": len(set(texts)) == 1,
            }
            results.append(record)
            dump(self.output / "e2e.json", results)
            if not record["same_output"]:
                raise AssertionError(f"PD/full-compute mismatch for {length} tokens")
        with (self.output / "decode.log").open() as stream:
            stream.seek(log_offset)
            reports = [
                dict(re.findall(r"(\w+)=([^ ]+)", line.strip()))
                for line in stream
                if "[MTSC] load_complete: role=D" in line
            ]
        dump(self.output / "e2e-load-reports.json", reports)
        if len(reports) != 4 or any(r["status"] != "ok" for r in reports):
            raise AssertionError("Missing successful rank-aggregated PD load reports")
        if not any(int(r["store_tokens"]) > 0 for r in reports):
            raise AssertionError("Warm E2E did not exercise Store GET")
        if not any(int(r["pd_tokens"]) > 0 for r in reports):
            raise AssertionError("Cold E2E did not exercise PD transfer")
        print("E2E: cold/warm, aligned/partial prompts match full compute", flush=True)

    async def stream(self, client, payload):
        started = time.perf_counter()
        first = None
        usage = None
        pieces = []
        async with client.post(
            "http://127.0.0.1:8370/v1/chat/completions",
            json={
                **payload,
                "model": "mtsc-qwen",
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        ) as response:
            if response.status != 200:
                raise RuntimeError(await response.text())
            async for raw in response.content:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                item = json.loads(line[6:])
                if "error" in item:
                    raise RuntimeError(item)
                if item.get("usage"):
                    usage = item["usage"]
                for choice in item.get("choices", []):
                    delta = choice.get("delta", {})
                    content = delta.get("content") or delta.get("reasoning") or ""
                    if content:
                        if first is None:
                            first = time.perf_counter()
                        pieces.append(content)
        ended = time.perf_counter()
        if first is None or usage is None:
            raise AssertionError("Missing generated content or server token usage")
        return {
            "ttft_ms": (first - started) * 1000,
            "latency_ms": (ended - started) * 1000,
            "usage": usage,
            "text_sha256": hashlib.sha256("".join(pieces).encode()).hexdigest(),
        }

    async def replay(self, client, index):
        await self.reset(client)
        offsets = {
            name: (self.output / f"{name}.log").stat().st_size
            for name in ("prefill", "decode")
        }
        semaphore = asyncio.Semaphore(self.args.concurrency)
        records = []

        async def session(path):
            for turn, line in enumerate(path.read_text().splitlines()):
                payload = json.loads(line)
                async with semaphore:
                    result = await self.stream(client, payload)
                records.append({"session": path.name, "turn": turn, **result})

        started = time.perf_counter()
        await asyncio.gather(
            *(session(path) for path in sorted(self.args.dataset_dir.glob("*.jsonl")))
        )
        elapsed = time.perf_counter() - started
        for name in ("prefill", "decode"):
            async with client.get(
                f"http://127.0.0.1:{8170 if name == 'prefill' else 8270}/metrics"
            ) as response:
                (self.output / f"round-{index}.{name}.metrics.txt").write_text(
                    await response.text()
                )
        result = {
            "round": index,
            "requests": len(records),
            "elapsed_s": elapsed,
            "request_throughput": len(records) / elapsed,
            "output_token_throughput": sum(
                r["usage"]["completion_tokens"] for r in records
            )
            / elapsed,
            "ttft_mean_ms": statistics.mean(r["ttft_ms"] for r in records),
            "ttft_p95_ms": percentile([r["ttft_ms"] for r in records], 0.95),
            "latency_mean_ms": statistics.mean(r["latency_ms"] for r in records),
            "log_offsets": offsets,
            "records": records,
        }
        dump(self.output / f"round-{index}.json", result)
        print(
            json.dumps(
                {k: v for k, v in result.items() if k not in {"records", "log_offsets"}}
            ),
            flush=True,
        )
        return result

    async def run(self):
        timeout = aiohttp.ClientTimeout(total=600)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as client:
            master = self.launch(
                "master",
                [
                    str(self.args.master),
                    "--rpc_address=127.0.0.1",
                    "--rpc_port=50097",
                    "--enable_http_metadata_server=true",
                    "--http_metadata_server_host=127.0.0.1",
                    "--http_metadata_server_port=8097",
                    "--metrics_host=127.0.0.1",
                    "--metrics_port=9027",
                    "--logtostderr=true",
                ],
            )
            await self.wait_ready(client, "http://127.0.0.1:8097/health", master)
            for name, port, gpu, role in (
                ("prefill", 8170, "0,1", "kv_producer"),
                ("decode", 8270, "2,3", "kv_consumer"),
            ):
                extra = {
                    "lookup_async": True,
                    "lookup_rpc_port": port,
                    "mooncake_protocol": "tcp",
                    "num_workers": 10,
                }
                if self.args.decode_save != "default":
                    extra["mtsc_decode_save"] = self.args.decode_save == "on"
                transfer = {
                    "kv_connector": "MTSCConnector",
                    "kv_connector_module_path": "mtsc.connector",
                    "kv_role": role,
                    "engine_id": name + "-perf",
                    "kv_load_failure_policy": "recompute",
                    "kv_connector_extra_config": extra,
                }
                process = self.launch(
                    name,
                    [
                        str(self.args.python),
                        "-m",
                        "vllm.entrypoints.openai.api_server",
                        "--model",
                        str(self.args.model),
                        "--served-model-name",
                        "mtsc-qwen",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(port),
                        "--tensor-parallel-size",
                        "2",
                        "--dtype",
                        "float16",
                        "--max-model-len",
                        "40960",
                        "--max-num-batched-tokens",
                        "4096",
                        "--max-num-seqs",
                        "4",
                        "--gpu-memory-utilization",
                        "0.9",
                        "--block-size",
                        "128",
                        "--enforce-eager",
                        "--enable-chunked-prefill",
                        "--no-enable-prefix-caching",
                        "--shutdown-timeout",
                        "10",
                        "--kv-transfer-config",
                        json.dumps(transfer),
                    ],
                    CUDA_VISIBLE_DEVICES=gpu,
                )
                await self.wait_ready(
                    client, f"http://127.0.0.1:{port}/health", process
                )
                print(f"{name} ready", flush=True)
            proxy = self.launch(
                "proxy",
                [
                    str(self.args.python),
                    "-m",
                    "proxy.pd_proxy",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8370",
                    "--prefill",
                    "http://127.0.0.1:8170,prefill-perf,http://127.0.0.1:8970,0",
                    "--decode",
                    "http://127.0.0.1:8270",
                    "--metrics-file",
                    str(self.output / "requests.jsonl"),
                    "--first-token-timeout",
                    "600",
                ],
            )
            await self.wait_ready(client, "http://127.0.0.1:8370/status", proxy)
            await self.correctness(client)
            rounds = [
                await self.replay(client, index)
                for index in range(1, self.args.rounds + 1)
            ]
            dump(
                self.output / "summary.json",
                [
                    {
                        k: v
                        for k, v in result.items()
                        if k not in {"records", "log_offsets"}
                    }
                    for result in rounds
                ],
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--decode-save", choices=("default", "on", "off"), default="off"
    )
    args = parser.parse_args()
    if args.rounds and (
        args.dataset_dir is None or not list(args.dataset_dir.glob("*.jsonl"))
    ):
        parser.error("Nonempty --dataset-dir is required when --rounds > 0")
    experiment = Experiment(args)

    def terminate(*_):
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, terminate)
    try:
        asyncio.run(experiment.run())
    finally:
        experiment.close()


if __name__ == "__main__":
    main()
