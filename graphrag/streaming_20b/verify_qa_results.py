"""Check real QA, or an explicitly requested completed prefix, against raw evidence."""
import argparse
from collections import Counter, defaultdict
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path

from prepare_stream import POLICIES, read, sha, write
from run_inference import eligible_questions, normalize, paired, score, usage as request_usage

BASE = Path(__file__).resolve().parent / "warm-hierarchy"
TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
USAGE_FIELDS = ("calls", *TOKEN_FIELDS)


def verify_maintenance(inference, results, batches, prefix):
    ledger = results["report_usage"]
    plan_root = BASE / "report-plan"
    columns = ("stage", "arm", "status", *USAGE_FIELDS,
               *(f"cumulative_{field}" for field in USAGE_FIELDS))

    @lru_cache(maxsize=None)
    def measured_usage(key):
        report = read(inference / "reports" / f"{key}.json")
        source = report["request"]["messages"][1]["content"]
        if hashlib.sha256(source.encode()).hexdigest() != key:
            raise ValueError(f"Mismatched report evidence: {key}")
        attempts = report["attempts"]
        if not attempts or [a["accepted"] for a in attempts] != [False] * (len(attempts) - 1) + [True]:
            raise ValueError(f"Invalid generation outcomes: {key}")
        for attempt in attempts:
            tokens = attempt["usage"]
            if (min(tokens[field] for field in TOKEN_FIELDS) <= 0
                    or tokens["prompt_tokens"] + tokens["completion_tokens"] != tokens["total_tokens"]):
                raise ValueError(f"Invalid maintenance tokens: {key}")
        return request_usage([report])

    def verify_stage(arm, stage, ledger_key):
        jobs = read(plan_root / arm / f"{stage}.json")["jobs"]
        totals = {name: Counter(dict.fromkeys(USAGE_FIELDS, 0)) for name in ("successful", "failed")}
        for job in jobs.values():
            key = hashlib.sha256(job["evidence"].encode()).hexdigest()
            if read(inference / "report-aliases" / f"{job['id']}.json") != key:
                raise ValueError(f"Mismatched report alias: {job['id']}")
            actual = measured_usage(key)
            for status, counts in totals.items():
                counts.update(actual[status])
        if totals != ledger[ledger_key]:
            raise ValueError(f"Maintenance usage does not match model records: {ledger_key}")
        return totals

    initial = verify_stage("eager", "batch-00", "initial")
    rows = []

    def add_row(stage, arm, status, counts, cumulative):
        rows.append({"stage": stage, "arm": arm, "status": status, **counts,
                     **{f"cumulative_{field}": cumulative[field] for field in USAGE_FIELDS}})

    for status, counts in initial.items():
        add_row("initial", "common", status, counts, counts)
    stages = [f"batch-{batch:02d}" for batch in range(1, batches + 1)]
    if not prefix:
        stages.append("final-settlement")
    summary = {}
    for arm in POLICIES:
        cumulative = {name: Counter(dict.fromkeys(USAGE_FIELDS, 0)) for name in initial}
        online = None
        for stage in stages:
            if stage == "final-settlement":
                online = {name: dict(counts) for name, counts in cumulative.items()}
            totals = verify_stage(arm, stage, f"{arm}/{stage}")
            for status, counts in totals.items():
                cumulative[status].update(counts)
                add_row(stage, arm, status, counts, cumulative[status])
        current = {name: dict(counts) for name, counts in cumulative.items()}
        summary[arm] = {"online": online if online is not None else current}
        if not prefix:
            summary[arm]["after_settlement"] = current
            stored = results["maintenance_tokens"][arm]
            expected = {status: {scope: totals[status]["total_tokens"]
                                 for scope, totals in summary[arm].items()} for status in current}
            if stored != expected:
                raise ValueError(f"Cumulative maintenance usage mismatch: {arm}")
    if not prefix:
        matched = read(inference / "paired-cost-results.json")["token_delta"]
        for arm, scopes in matched.items():
            for scope, expected in scopes.items():
                observed = {field: summary[arm][scope]["successful"][field]
                            - summary["deferred-000"][scope]["successful"][field] for field in TOKEN_FIELDS}
                if observed != expected:
                    raise ValueError(f"Matched maintenance costs disagree with 20-prompt measurement: {arm}/{scope}")
    return summary, columns, rows


def main(prefix=False):
    inference = BASE / "inference"
    watcher = read(BASE / "watch-status.json")
    if prefix:
        results = read(inference / "partial.json")
        batches = results["batches_complete"]
        if not 1 <= batches <= 20:
            raise ValueError("Expected at least one completed QA batch")
    else:
        results = read(inference / "results.json")
        batches = 20
        if results["mock"] or results["batches"] != batches or watcher["state"] != "complete":
            raise ValueError("Expected completed real 20-batch inference and verified release")

    @lru_cache(maxsize=None)
    def report_prefix(job):
        key = read(inference / "report-aliases" / f"{job}.json")
        report = read(inference / "reports" / f"{key}.json")
        source = report["request"]["messages"][1]["content"]
        if hashlib.sha256(source.encode()).hexdigest() != key:
            raise ValueError(f"Corrupt report request: {job}")
        return report["answer"][:3000]

    queries = read(BASE / "queries.json")
    entries = defaultdict(dict)
    for row in results["qa"]:
        key = (row["arm"], row["batch"])
        if row["id"] in entries[key]:
            raise ValueError(f"Duplicate answer: {key}/{row['id']}")
        entries[key][row["id"]] = row
    same_request = Counter()
    exposures = defaultdict(Counter)
    usage = defaultdict(Counter)
    stages = 0
    for batch in range(1, batches + 1):
        eligible = {r["id"]: r for r in eligible_questions(queries, batch)}
        for arm in POLICIES:
            expected = entries[arm, batch]
            if expected.keys() != eligible.keys():
                raise ValueError(f"Question set mismatch at {arm}/{batch}")
            plan = read(BASE / "report-plan" / arm / f"batch-{batch:02d}.json")
            leaf_reports = {r["id"]: r for r in plan["index"].values() if r["level"] == 0}
            stale_leaf_ids = {rid for rid, report in leaf_reports.items() if report["stale_since"] is not None}
            for qid, row in expected.items():
                ids = row["retrieved"]
                if len(ids) != 6 or len(set(ids)) != 6 or not set(ids) <= leaf_reports.keys():
                    raise ValueError(f"Invalid retrieval: {arm}/{batch}/{qid}")
                key = row["answer_request_key"]
                answer = read(inference / "answers" / f"{key}.json")
                input_text = answer["request"]["messages"][1]["content"]
                context = "\n\n".join(report_prefix(job) for job in ids)
                expected_input = f"Community reports:\n{context}\n\nQuestion: {eligible[qid]['question']}"
                if (hashlib.sha256(input_text.encode()).hexdigest() != key
                        or input_text != expected_input or answer["answer"] != row["answer"]
                        or request_usage([answer]) != row["usage"]):
                    raise ValueError(f"Answer corrupted: {arm}/{batch}/{qid}")
                actual = bool(stale_leaf_ids.intersection(ids))
                em, f1 = score(row["answer"], eligible[qid]["answers"])
                if (row["read_stale_report"], row["em"], row["f1"], row["unknown"]) != (
                        actual, em, f1, int(normalize(row["answer"]) == ["unknown"])):
                    raise ValueError(f"Metric or exposure mismatch: {arm}/{batch}/{qid}")
                exposures[arm].update(answers=1, stale_retrievals=int(actual),
                                      answers_with_stale_leaf_available=int(bool(stale_leaf_ids)))
                for section in ("successful", "failed"):
                    billing = row["usage"][section]
                    usage[arm].update({field: billing[field] for field in
                                       ("prompt_tokens", "completion_tokens", "total_tokens")})
                other = entries["deferred-000", batch][qid]
                if key == other["answer_request_key"]:
                    assert row["answer"] == other["answer"]
                    same_request[arm] += 1
            stages += 1
    if len(entries) != stages:
        raise ValueError("Unexpected treatment or batch")
    references = ("eager", "deferred-000")
    metrics_by_reference = {reference: paired(results["qa"], reference) for reference in references}
    if not prefix and results["quality"] != metrics_by_reference:
        raise ValueError("Paired estimates or bootstrap intervals do not reproduce")
    audits = [json.loads(line) for line in (BASE / "worker-releases.jsonl").read_text().splitlines()]
    for event in audits if prefix else [*audits, watcher["release"]]:
        if event["worker_main_pid"] or event["worker_gpu_pids"] or event["worker_cgroup_pids"]:
            raise ValueError("GPU release was not verified")
    completed = {row["through"] for row in audits if row["phase"] == "qa" and not row["exit_status"]}
    recovery = None if prefix else results.get("finalization_recovery")
    if recovery is not None:
        checkpoint = inference / "partial.json"
        saved = read(checkpoint)
        released = [event for event in audits if event["checked_at"] == recovery["gpu_release_checked_at"]]
        if (recovery["checkpoint_sha256"] != sha(checkpoint)
                or recovery["inference_code_sha256"] != read(inference / "inputs.json")["code"]
                or saved["batches_complete"] != batches or saved["qa"] != results["qa"]
                or len(released) != 1 or released[0]["phase"] != "qa" or released[0]["through"] != batches
                or released[0]["exit_status"] != 1 or released[0]["gpu"]["status"] != "interrupted_or_failed"):
            raise ValueError("Recovered summary does not match the saved QA or final GPU release")
        if any(read(BASE / "report-plan" / arm / "final-settlement.json")["jobs"] for arm in POLICIES):
            raise ValueError("CPU-only finalization requires an empty settlement plan")
        completed.add(batches)
    expected_releases = set(range(1, batches + 1))
    if not expected_releases <= completed or (not prefix and completed != expected_releases):
        raise ValueError("Worker releases do not match the completed batches")
    if not any(row["phase"] == "cost" and not row["exit_status"] for row in audits):
        raise ValueError("Missing worker release for the fixed-trajectory cost stage")
    maintenance, maintenance_columns, maintenance_rows = verify_maintenance(inference, results, batches, prefix)
    checked = {"batches_verified": batches, "interim_prefix": prefix,
               "logical_answers": len(results["qa"]), "verified_release_records": len(audits),
               "logical_maintenance_excluding_initial": maintenance,
               "arms": {arm: {**counts, "same_question_and_context_as_eps_0": same_request[arm],
                               "mean_query_tokens": {k: v / counts["answers"] for k, v in usage[arm].items()}}
                        for arm, counts in exposures.items()}}
    if recovery is not None:
        checked["finalization_recovery"] = recovery
    audit_name = f"qa-prefix-audit-{batches:02d}.json" if prefix else "final-qa-audit.json"
    write(BASE / audit_name, {**checked, "metrics": metrics_by_reference} if prefix else checked)
    maintenance_name = f"qa-prefix-maintenance-{batches:02d}.csv" if prefix else "qa-maintenance-by-batch.csv"
    with (BASE / maintenance_name).open("w", newline="") as table:
        writer = csv.DictWriter(table, fieldnames=maintenance_columns)
        writer.writeheader()
        writer.writerows(maintenance_rows)
    metrics = []
    for reference, treated in metrics_by_reference.items():
        for arm, cohorts in treated.items():
            for cohort in ("all", "old", "new"):
                if cohort not in cohorts:
                    continue
                for metric in ("em", "f1", "unknown"):
                    result = cohorts[cohort][metric]
                    low, high = result["question_bootstrap_ci95"]
                    metrics.append({
                        "reference": reference, "arm": arm, "cohort": cohort, "metric": metric,
                        "questions": cohorts[cohort]["questions"], "reference_mean": result["reference"],
                        "arm_mean": result["arm"], "paired_difference": result["paired_delta"],
                        "lower_95": low, "upper_95": high,
                    })
    metrics_name = f"qa-prefix-metrics-{batches:02d}.csv" if prefix else "qa-paired-metrics.csv"
    with (BASE / metrics_name).open("w", newline="") as table:
        writer = csv.DictWriter(table, fieldnames=metrics[0])
        writer.writeheader()
        writer.writerows(metrics)
    print(json.dumps(checked, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", action="store_true", help="Audit only the last fully completed checkpoint")
    main(prefix=parser.parse_args().prefix)
