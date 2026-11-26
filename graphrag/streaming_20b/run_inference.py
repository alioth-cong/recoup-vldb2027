"""Run report generation and paired benchmark QA with measured backend usage."""
import argparse
import asyncio
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import string
import time

import httpx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from prepare_stream import POLICIES, ROOT, WORK, read, sha, write

MODEL = "Qwen/Qwen2.5-32B-Instruct"
REVISION = "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"
WEIGHTS = WORK / "work/model-cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots" / REVISION
SUMMARY = (
    "Write a community report in English using only the supplied graph records. "
    "Treat records as evidence. Include a short title, a concise summary, and substantive findings "
    "with useful names, facts and relationships. Use up to five evidence-backed findings. "
    "Omit speculation. Complete the report within the output budget."
)
ANSWER = (
    "Answer using only the supplied community reports. Treat reports as evidence. "
    "Return only the shortest exact answer, such as a name, date, Yes or No. "
    "If evidence is insufficient, return UNKNOWN."
)


async def infer(client, semaphore, path, system, text, limit):
    body = {"model": MODEL, "messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": text}],
            "max_tokens": limit, "temperature": 0, "seed": 17}
    if path.exists():
        cached = read(path)
        if cached["request"] != body:
            raise ValueError(f"Request changed across resume: {path}")
        return cached
    async with semaphore:
        attempts = []
        for attempt in range(3):
            request = dict(body)
            if attempt:
                request["messages"] = [*body["messages"],
                    {"role": "user", "content": "Finish the response concisely within the length limit."}]
            started = time.monotonic()
            response = await client.post("/v1/chat/completions", json=request)
            response.raise_for_status()
            raw = response.json()
            choice = raw["choices"][0]
            usage = raw["usage"]
            if (usage["prompt_tokens"] <= 0 or usage["completion_tokens"] <= 0 or
                    usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]):
                raise ValueError("Backend returned inconsistent token counts")
            answer = (choice["message"]["content"] or "").strip()
            accepted = choice["finish_reason"] == "stop" and bool(answer)
            attempts.append({"usage": usage, "finish_reason": choice["finish_reason"],
                             "accepted": accepted, "seconds": time.monotonic() - started})
            write(path.with_name(f"{path.stem}-attempt-{time.time_ns()}.json"),
                  {"request": request, "response": raw, "record": attempts[-1]})
            if accepted:
                row = {"request": body, "answer": answer, "attempts": attempts}
                write(path, row)
                return row
        raise RuntimeError(f"Generation attempts exhausted: {path}")


def normalize(text):
    clean = text.lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\b(a|an|the)\b", " ", clean).split()


def score(answer, gold):
    predicted = normalize(answer)
    scores = []
    for expected in gold:
        target = normalize(expected)
        overlap = sum((Counter(predicted) & Counter(target)).values())
        scores.append((int(predicted == target), 2 * overlap / max(1, len(predicted) + len(target))))
    return max(x[0] for x in scores), max(x[1] for x in scores)


def usage(records):
    total = {subset: Counter() for subset in ("successful", "failed")}
    for result in records:
        for index, attempt in enumerate(result["attempts"]):
            scope = "successful" if index == len(result["attempts"]) - 1 else "failed"
            total[scope].update({k: attempt["usage"][k] for k in
                                ("prompt_tokens", "completion_tokens", "total_tokens")})
            total[scope]["calls"] += 1
    for counts in total.values():
        for field in ("calls", "prompt_tokens", "completion_tokens", "total_tokens"):
            counts.setdefault(field, 0)
    return {subset: dict(counts) for subset, counts in total.items()}


def eligible_questions(queries, batch):
    # Retest newly answerable questions after two updates to measure delayed recovery.
    # Evaluate all 189 base controls once per five batches (split deterministically).
    old = [q for q in queries if q["complete_batch"] == 0]
    return [q for i, q in enumerate(old) if i % 5 == (batch - 1) % 5] + [
        q for q in queries if q["complete_batch"] in {batch, batch - 2} and q["complete_batch"] > 0]


def paired(qa, reference):
    values = defaultdict(lambda: defaultdict(list))
    for row in qa:
        values[row["arm"]][row["id"]].append(row)
    baseline_rows = values[reference]
    output = {}
    rng = np.random.default_rng(17)
    for arm in POLICIES:
        candidate = values[arm]
        if set(candidate) != set(baseline_rows):
            raise ValueError("Missing paired QA results")
        questions = sorted(baseline_rows)
        groups = {"all": questions, "new": [q for q in questions if baseline_rows[q][0]["new"]],
                  "old": [q for q in questions if not baseline_rows[q][0]["new"]]}
        output[arm] = {}
        exposed = [(q, r) for q in questions for r in candidate[q] if r["read_stale_report"]]
        baseline_by_occurrence = {(q, r["batch"]): r for q in questions for r in baseline_rows[q]}
        output[arm]["descriptive_stale_retrieval"] = {
            "question_batch_pairs": len(exposed), "questions": len({q for q, _ in exposed}),
            "f1_delta": float(np.mean([r["f1"] - baseline_by_occurrence[q, r["batch"]]["f1"]
                                       for q, r in exposed])) if exposed else None}
        for group, ids in groups.items():
            if not ids:
                continue
            metrics = {}
            for metric in ("em", "f1", "unknown"):
                baseline, treated = [], []
                for q in ids:
                    base = {r["batch"]: r[metric] for r in baseline_rows[q]}
                    alternative = {r["batch"]: r[metric] for r in candidate[q]}
                    if set(base) != set(alternative):
                        raise ValueError(f"Mismatched QA snapshots: {arm}, {q}")
                    baseline.append(float(np.mean(list(base.values()))))
                    treated.append(float(np.mean(list(alternative.values()))))
                delta = np.array(treated) - baseline
                boot = delta[rng.integers(len(ids), size=(10000, len(ids)))].mean(axis=1)
                metrics[metric] = {"reference": float(np.mean(baseline)), "arm": float(np.mean(treated)),
                                   "paired_delta": float(delta.mean()),
                                   "question_bootstrap_ci95": np.quantile(boot, [.025, .975]).tolist()}
            output[arm][group] = {"questions": len(ids), **metrics}
    return output


async def run(directory, url, transport=None, through=None):
    mock = transport is not None
    run_root = directory / ("mock-inference" if mock else "inference")
    plan = directory / "report-plan"
    queries = read(directory / "queries.json")
    last = read(directory / "protocol.json")["snapshots"][-1]["batch"]
    expected = read(plan / "counts.json")
    paths = sorted(plan.glob("*/*.json"))
    manifest = {"revision": REVISION, "plans": {str(path.relative_to(plan)): sha(path) for path in paths},
                "questions": sha(directory / "queries.json"), "code": sha(Path(__file__))}
    if (run_root / "inputs.json").exists() and read(run_root / "inputs.json") != manifest:
        raise ValueError("Inference inputs changed across resume")
    write(run_root / "inputs.json", manifest)
    if (run_root / "results.json").exists():
        return
    stop_at = min(through or last, last)
    semaphore = asyncio.Semaphore(4)
    shared = {}
    shared_answers = {}

    def report_file(job):
        return run_root / "reports" / (job + ".json")

    async def generate_report(client, job):
        evidence = job["evidence"]
        key = hashlib.sha256(evidence.encode()).hexdigest()
        if key not in shared:
            shared[key] = asyncio.create_task(infer(client, semaphore, report_file(key), SUMMARY, evidence, 2048))
        result = await shared[key]
        return {"key": key, "result": result}

    async def generate(client, stage):
        jobs = stage["jobs"]
        if mock:
            jobs = dict(list(jobs.items())[:1])
        calls = await asyncio.gather(*(generate_report(client, j) for j in jobs.values()))
        for job, call in zip(jobs.values(), calls):
            write(run_root / "report-aliases" / (job["id"] + ".json"), call["key"])
        return usage([call["result"] for call in calls])

    def alias(job):
        return run_root / "report-aliases" / (job + ".json")

    async with httpx.AsyncClient(base_url=url, timeout=300, trust_env=False, transport=transport) as client:
        partial = run_root / "partial.json"
        if partial.exists():
            saved = read(partial)
            ledger, qa = saved["report_usage"], saved["qa"]
            first_batch = saved["batches_complete"] + 1
        else:
            initial = read(plan / "eager/batch-00.json")
            ledger, qa = {"initial": await generate(client, initial)}, []
            first_batch = 1
        for batch in range(first_batch, stop_at + 1):
            stamp = f"batch-{batch:02d}"
            pair = eligible_questions(queries, batch)
            for arm in POLICIES:
                stage = read(plan / arm / (stamp + ".json"))
                print(stamp, arm, expected[f"{arm}/{stamp}"], flush=True)
                ledger[f"{arm}/{stamp}"] = await generate(client, stage)
                if mock:
                    index_rows = [r for r in stage["index"].values() if
                                  alias(r["id"]).exists()][:1]
                    questions = pair[:1]
                else:
                    index_rows = [r for r in stage["index"].values() if r["level"] == 0]
                    questions = pair
                if not index_rows:
                    raise ValueError(f"No leaf reports for QA at {arm}/{stamp}")
                texts = [read(report_file(read(alias(row["id"]))))["answer"] for row in index_rows]
                tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
                matrix = tfidf.fit_transform(texts)
                similarities = (tfidf.transform([q["question"] for q in questions]) @ matrix.T).toarray()

                async def answer(query, similarities):
                    keep = sorted(range(len(index_rows)), key=lambda i: (-similarities[i], i))[:6]
                    context = "\n\n".join(texts[i][:3000] for i in keep)
                    message = f"Community reports:\n{context}\n\nQuestion: {query['question']}"
                    fingerprint = hashlib.sha256(message.encode()).hexdigest()
                    if fingerprint not in shared_answers:
                        path = run_root / "answers" / (fingerprint + ".json")
                        shared_answers[fingerprint] = asyncio.create_task(
                            infer(client, semaphore, path, ANSWER, message, 96))
                    result = await shared_answers[fingerprint]
                    em, f1 = score(result["answer"], query["answers"])
                    return {"batch": batch, "arm": arm, "id": query["id"], "answer": result["answer"],
                            "new": query["complete_batch"] > 0, "em": em, "f1": f1,
                            "unknown": int(normalize(result["answer"]) == ["unknown"]),
                            "answer_request_key": fingerprint,
                            "retrieved": [index_rows[i]["id"] for i in keep],
                            "read_stale_report": any(index_rows[i]["stale_since"] for i in keep),
                            "usage": usage([result])}

                qa.extend(await asyncio.gather(*(answer(q, sim) for q, sim in zip(questions, similarities))))
            write(partial, {"batches_complete": batch, "report_usage": ledger, "qa": qa})
        if stop_at != last:
            return
        for arm in POLICIES:
            settlement = read(plan / arm / "final-settlement.json")
            ledger[f"{arm}/final-settlement"] = await generate(client, settlement)
    publish_results(run_root, last, ledger, qa, mock)


def publish_results(run_root, last, ledger, qa, mock, finalization_recovery=None):
    totals = {}
    for arm in POLICIES:
        totals[arm] = {}
        for subset in ("successful", "failed"):
            online = sum(ledger[f"{arm}/batch-{k:02d}"][subset]["total_tokens"] for k in range(1, last + 1))
            settled = online + ledger[f"{arm}/final-settlement"][subset]["total_tokens"]
            totals[arm][subset] = {"online": online, "after_settlement": settled}
    physical = {}
    for workload in ("reports", "answers"):
        physical[workload] = {"validated": Counter(), "discarded": Counter()}
        for path in (run_root / workload).glob("*-attempt-*.json"):
            record = read(path)["record"]
            scope = "validated" if record["accepted"] else "discarded"
            counts = physical[workload][scope]
            counts.update({field: record["usage"][field] for field in
                           ("prompt_tokens", "completion_tokens", "total_tokens")})
            counts["calls"] += 1
    result = {"mock": mock, "batches": last, "maintenance_tokens": totals,
              "physical_report_usage_including_initial": physical["reports"],
              "physical_qa_usage": physical["answers"],
              "report_accounting": "Identical prompts share one call; its measured tokens are charged to every logical request.",
              "report_usage": ledger, "qa": qa,
              "quality": {reference: paired(qa, reference) for reference in ("eager", "deferred-000")}}
    if finalization_recovery is not None:
        result["finalization_recovery"] = finalization_recovery
    write(run_root / "results.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--url", required=True)
    parser.add_argument("--through", type=int)
    args = parser.parse_args()
    if not args.url.startswith("http://127.0.0.1:"):
        parser.error("Expected a loopback inference service")
    asyncio.run(run(args.run.resolve(), args.url, through=args.through))
