"""Compare round-level metrics and validate successful two-stage load reports."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

METRICS = (
    "elapsed_s",
    "request_throughput",
    "output_token_throughput",
    "ttft_mean_ms",
    "ttft_p95_ms",
    "latency_mean_ms",
)


def read_variant(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    summaries = json.loads((folder / "summary.json").read_text())
    rounds = [
        json.loads((folder / f"round-{r['round']}.json").read_text()) for r in summaries
    ]
    for index, result in enumerate(rounds):
        with (folder / "decode.log").open("rb") as stream:
            start = result["log_offsets"]["decode"]
            stream.seek(start)
            end = (
                rounds[index + 1]["log_offsets"]["decode"]
                if index + 1 < len(rounds)
                else None
            )
            lines = (
                stream.read(end - start if end is not None else -1)
                .decode()
                .splitlines()
            )
        reports = [
            dict(re.findall(r"(\w+)=([^ ]+)", line.strip()))
            for line in lines
            if "[MTSC] load_complete: role=D" in line
        ]
        if len(reports) != result["requests"]:
            raise AssertionError(
                f"{folder}: missing D load reports in round {index + 1}"
            )
        if any(
            r["status"] != "ok" or r["store_status"] not in {"ok", "skipped"}
            for r in reports
        ):
            raise AssertionError(f"{folder}: Store/PD failure in round {index + 1}")
        if any(r["workers"] != "2/2" for r in reports):
            raise AssertionError(f"{folder}: incomplete TP rank aggregation")
        if any(
            int(r["store_tokens"]) + int(r["pd_tokens"]) + int(r["local_tokens"])
            != int(r["target_tokens"])
            for r in reports
        ):
            raise AssertionError(f"{folder}: incomplete token coverage")
        if not any(int(r["store_tokens"]) > 0 for r in reports):
            raise AssertionError(
                f"{folder}: no Store-hit requests in round {index + 1}"
            )
        records = result["records"]
        expected = {
            (path, turn) for path in manifest["dataset_sha256"] for turn in range(3)
        }
        if {(r["session"], r["turn"]) for r in records} != expected:
            raise AssertionError(f"{folder}: incomplete three-turn session replay")
        if len(records) != len(expected):
            raise AssertionError(f"{folder}: duplicated replay requests")
        result["loads"] = {
            "requests": len(reports),
            "store_hit_requests": sum(int(r["store_tokens"]) > 0 for r in reports),
            "store_tokens": sum(int(r["store_tokens"]) for r in reports),
            "pd_tokens": sum(int(r["pd_tokens"]) for r in reports),
            "store_ms_mean": statistics.mean(float(r["store_ms_max"]) for r in reports),
            "pd_ms_mean": statistics.mean(float(r["pd_ms_max"]) for r in reports),
            "total_ms_mean": statistics.mean(float(r["total_ms_max"]) for r in reports),
        }
        result["input_tokens"] = sum(r["usage"]["prompt_tokens"] for r in records)
        result["output_tokens"] = sum(r["usage"]["completion_tokens"] for r in records)
    e2e = json.loads((folder / "e2e.json").read_text())
    if not all(r["same_output"] for r in e2e):
        raise AssertionError(f"{folder}: E2E output mismatch")
    return {
        "manifest": manifest,
        "rounds": [
            {k: v for k, v in r.items() if k not in {"records", "log_offsets"}}
            for r in rounds
        ],
        "means": {
            metric: statistics.mean(r[metric] for r in rounds) for metric in METRICS
        },
        "stdev": {
            metric: statistics.stdev(r[metric] for r in rounds)
            if len(rounds) > 1
            else 0
            for metric in METRICS
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = read_variant(args.baseline)
    candidate = read_variant(args.candidate)
    if (
        baseline["manifest"]["dataset_sha256"]
        != candidate["manifest"]["dataset_sha256"]
    ):
        raise AssertionError("Dataset snapshots differ")
    workloads = {
        (r["input_tokens"], r["output_tokens"])
        for r in baseline["rounds"] + candidate["rounds"]
    }
    if len(workloads) != 1:
        raise AssertionError("Server-counted token workloads differ")
    output = {
        "baseline": baseline,
        "candidate": candidate,
        "change_percent": {
            metric: (candidate["means"][metric] / baseline["means"][metric] - 1) * 100
            for metric in METRICS
        },
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print("| Metric | Before | After | Change |")
    print("| --- | ---: | ---: | ---: |")
    for metric in METRICS:
        print(
            f"| {metric} | {baseline['means'][metric]:.3f} | {candidate['means'][metric]:.3f} | {output['change_percent'][metric]:+.2f}% |"
        )


if __name__ == "__main__":
    main()
