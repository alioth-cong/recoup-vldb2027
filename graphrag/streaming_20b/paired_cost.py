"""Measure the exact successful-token difference on a matched move trajectory."""
import argparse
import asyncio
from collections import Counter
import hashlib
from pathlib import Path

import httpx

from prepare_stream import read, sha, write
from run_inference import SUMMARY, REVISION, infer

MATCHED = ("deferred-000", "deferred-005", "deferred-020")


def prepare(directory):
    base = directory / "report-plan"
    counts = {}
    final_counts = {}
    evidence = {}
    traces = {arm: sha(directory / "native" / arm / "commit_trace.tsv") for arm in MATCHED}
    if len(set(traces.values())) != 1:
        raise ValueError("Matched-cost comparison requires identical native move traces")
    for arm in MATCHED:
        counts[arm] = Counter()
        final_counts[arm] = Counter()
        for path in sorted((base / arm).glob("*.json")):
            target = final_counts[arm] if path.stem == "final-settlement" else counts[arm]
            for job in read(path)["jobs"].values():
                text = job["evidence"]
                key = hashlib.sha256(text.encode()).hexdigest()
                target[key] += 1
                evidence.setdefault(key, text)
    residual = {}
    for arm in MATCHED[1:]:
        online = counts[arm].copy()
        online.subtract(counts[MATCHED[0]])
        settled = online.copy()
        settled.update(final_counts[arm])
        settled.subtract(final_counts[MATCHED[0]])
        residual[arm] = {"online": {k: v for k, v in online.items() if v},
                         "after_settlement": {k: v for k, v in settled.items() if v}}
    needed = {key for arm in residual.values() for scope in arm.values() for key in scope}
    result = {"reference": MATCHED[0], "matched_trace_sha256": traces[MATCHED[0]],
              "reports": {k: evidence[k] for k in sorted(needed)}, "multiplicities": residual,
              "logical_update_jobs": {arm: sum(counts[arm].values()) for arm in MATCHED},
              "logical_final_settlement_jobs": {arm: sum(final_counts[arm].values()) for arm in MATCHED}}
    write(directory / "paired-cost-plan.json", result)
    print("Matched cost needs", len(needed), "distinct model calls", flush=True)


async def measure(directory, url, transport=None):
    plan_path = directory / "paired-cost-plan.json"
    plan = read(plan_path)
    out = directory / ("mock-cost" if transport else "inference")
    dest = out / "paired-cost-results.json"
    fingerprint = {"plan_sha256": sha(plan_path), "revision": REVISION, "summary_instruction": SUMMARY}
    if dest.exists() and read(dest)["fingerprint"] != fingerprint:
        raise ValueError("Configuration changed across a paired-cost resume")
    semaphore = asyncio.Semaphore(4)
    async with httpx.AsyncClient(base_url=url, timeout=300, transport=transport, trust_env=False) as client:
        records = await asyncio.gather(*(infer(client, semaphore, out / "reports" / (key + ".json"),
                                                SUMMARY, evidence, 2048)
                                        for key, evidence in plan["reports"].items()))
    input_usage = {key: r["attempts"][-1]["usage"] for key, r in zip(plan["reports"], records)}
    delta = {arm: {scope: {field: sum(n * input_usage[k][field] for k, n in multiplicities.items())
                           for field in ("prompt_tokens", "completion_tokens", "total_tokens")}
                   for scope, multiplicities in sections.items()}
             for arm, sections in plan["multiplicities"].items()}
    write(dest, {"fingerprint": fingerprint, "mock": transport is not None,
                 "reference": plan["reference"], "actual_unique_requests": len(records),
                 "token_delta": delta, "per_prompt_usage": input_usage,
                 "scope": "Signed differences from successful logical report refreshes. Identical prompts cancel."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--url")
    args = parser.parse_args()
    if args.url is None:
        prepare(args.run.resolve())
    elif not args.url.startswith("http://127.0.0.1:"):
        parser.error("Expected a loopback inference service")
    else:
        asyncio.run(measure(args.run.resolve(), args.url))
