"""Translate native refresh plans into auditable, prefix-safe report jobs."""
from collections import defaultdict
from dataclasses import dataclass
import argparse
from pathlib import Path

import pandas as pd
import tiktoken

from prepare_stream import POLICIES, read, rows, sha, snapshot, write

ENC = tiktoken.get_encoding("cl100k_base")


@dataclass
class Report:
    members: frozenset
    evidence: dict
    epoch: int
    job: str
    pending_since: int | None = None
    settlements: tuple = ()


def node_sets(directory, batch):
    result = {}
    for path in sorted(directory.glob(f"nodes_{batch}_L*.tsv")):
        level = int(path.stem.rsplit("L", 1)[1])
        for row in rows(path):
            result[level, int(row["community"])] = frozenset(map(int, row["members"].split(",")))
    if not result:
        raise ValueError(f"No snapshot for {directory}, batch {batch}")
    return result


class Evidence:
    def __init__(self, batch, ids):
        ent = pd.read_parquet(snapshot(batch) / "entities.parquet")
        rel = pd.read_parquet(snapshot(batch) / "relationships.parquet")
        self.entities = {ids[r.title]: (r.title, r.type, r.description or "",
                                       tuple(sorted(set(r.text_unit_ids))))
                         for r in ent.itertuples() if r.title in ids}
        self.edges = defaultdict(list)
        self.neighbors = defaultdict(set)
        for row in rel.itertuples():
            if row.source not in ids or row.target not in ids or row.source == row.target:
                continue
            a, b = sorted((ids[row.source], ids[row.target]))
            self.edges[a, b].append((row.source, row.target, row.description or "",
                                     tuple(sorted(set(row.text_unit_ids)))))
            self.neighbors[a].add(b)
        self.edges = {pair: tuple(sorted(set(values))) for pair, values in self.edges.items()}

    def records(self, members):
        edges = {(a, b): self.edges[a, b] for a in members
                 for b in self.neighbors[a] & members}
        if not edges:
            return {}
        return {(v,): self.entities[v] for v in members} | edges


def prompt(records):
    lines = []
    used = 0
    for key, value in sorted(records.items(), key=lambda item: (-len(item[0]), item[0])):
        if len(key) == 1:
            line = "ENTITY " + " | ".join(str(v) for v in value[:3])
        else:
            line = "RELATION " + " ; ".join(" | ".join(str(v) for v in r[:3]) for r in value)
        encoded = ENC.encode(line + "\n", disallowed_special=())
        remaining = 4000 - used
        if remaining <= 0:
            break
        lines.append(ENC.decode(encoded[:remaining]))
        used += min(len(encoded), remaining)
    return "".join(lines)


def plan(directory):
    protocol = read(directory / "protocol.json")
    for filename, digest in protocol["data_sha256"].items():
        if sha(Path(filename)) != digest:
            raise ValueError(f"Input changed: {filename}")
    ids = {name: int(v) for v, name in read(directory / "stream/id2name.json").items()}
    plans = {}
    states = {}
    epochs = {}
    reports = defaultdict(dict)
    batch_summary = {}
    for arm in POLICIES:
        native = directory / "native" / arm
        plans[arm] = defaultdict(dict)
        states[arm] = defaultdict(dict)
        epochs[arm] = {(int(r["level"]), int(r["community"])): int(r["report_version"])
                       for r in rows(native / "initial_report_epochs.tsv")}
        for r in rows(native / "refresh_plan.tsv"):
            plans[arm][int(r["batch"])][int(r["level"]), int(r["community"])] = r
        for r in rows(native / "source_state.tsv"):
            states[arm][int(r["batch"])][0, int(r["source"])] = r

    last = protocol["snapshots"][-1]["batch"]
    plan_dir = directory / "report-plan"
    names = {v: k for k, v in ids.items()}
    for batch in range(last + 1):
        data = Evidence(batch, ids)
        stage = f"batch-{batch:02d}"
        for arm, epsilon in POLICIES.items():
            if batch == 0 and arm != "eager":
                if node_sets(directory / "native" / arm, 0) != nodes or epochs[arm] != epochs["eager"]:
                    raise ValueError("Arms do not start from an identical report index")
                reports[arm] = reports["eager"].copy()
                continue
            nodes = node_sets(directory / "native" / arm, batch)
            old = reports[arm]
            ready = {}
            jobs = {}
            pending = {}
            source_removal = []
            declared = plans[arm][batch]
            for key, state in states[arm][batch].items():
                if key not in old or key not in nodes:
                    continue
                previous = old[key]
                current = nodes[key]
                if (int(state["report_version"]) == previous.epoch and
                        float(state["drift"]) < epsilon and current < previous.members):
                    source_removal.append((key, previous.members, previous.members - current))
            for key, members in sorted(nodes.items()):
                evidence = data.records(members)
                if not evidence:
                    continue
                key_id = f"{key[0]}-{key[1]}"
                previous = old.get(key)
                native = declared.get(key)
                distinct = previous is None or evidence != previous.evidence
                reason = ("initial" if batch == 0 else native["reason"] if native else
                          "new_materialized_report" if previous is None else "changed_evidence_or_membership")
                if batch and native is None and previous is not None:
                    if members == previous.members and not distinct:
                        ready[key] = Report(members, evidence, previous.epoch, previous.job)
                        continue
                    # A source certificate applies to the source leaf and ancestors
                    # whose lost members are entirely explained by certified leaves.
                    certified = [(node, lost) for node, basis, lost in source_removal
                                 if node == key or (key[0] > 0 and basis <= previous.members)]
                    removed = members < previous.members and previous.members - members
                    added_or_updated = any(previous.evidence.get(k) != v for k, v in evidence.items())
                    lost_only = set(previous.evidence) - evidence.keys()
                    retained_same = all(not set(k) <= members for k in lost_only)
                    covered = removed and removed <= set().union(*(lost for _, lost in certified))
                    if covered and not added_or_updated and retained_same:
                        since = previous.pending_since or batch
                        settled_by = tuple(node for node, _ in certified)
                        ready[key] = Report(previous.members, previous.evidence, previous.epoch,
                                            previous.job, since, settled_by)
                        pending[key_id] = {"since": since, "age": batch - since + 1,
                                           "sources": settled_by, "lost_members": len(removed)}
                        continue
                epoch = (int(native["valuation_version"]) if native else
                         previous.epoch if previous else epochs[arm].get(key, 0))
                job_id = f"{'common' if batch == 0 else arm}/{stage}/{key_id}"
                jobs[key_id] = {"id": job_id, "reason": reason, "evidence": prompt(evidence)}
                ready[key] = Report(members, evidence, epoch, job_id)
            reports[arm] = ready
            index = report_index(ready, names)
            batch_summary[f"{arm}/{stage}"] = publish(plan_dir, arm, stage, index, jobs, pending)
            if batch == last:
                settle = ready.copy()
                flush = {}
                for key, previous in ready.items():
                    if previous.pending_since is None:
                        continue
                    key_id = f"{key[0]}-{key[1]}"
                    evidence = data.records(nodes[key])
                    job_id = f"{arm}/final-settlement/{key_id}"
                    flush[key_id] = {"id": job_id, "reason": "final_settlement", "evidence": prompt(evidence)}
                    settle[key] = Report(nodes[key], evidence, previous.epoch, job_id)
                batch_summary[f"{arm}/final-settlement"] = publish(
                    plan_dir, arm, "final-settlement", report_index(settle, names), flush, {})
        print(stage, {a: batch_summary.get(f"{a}/{stage}") for a in POLICIES}, flush=True)
    write(plan_dir / "counts.json", batch_summary)


def report_index(reports, names):
    return {f"{level}-{cid}": {"id": report.job, "stale_since": report.pending_since,
                               "level": level, "members": [names[v] for v in sorted(report.members)]}
            for (level, cid), report in reports.items()}


def publish(directory, arm, stage, index, jobs, pending):
    write(directory / arm / f"{stage}.json", {"jobs": jobs, "index": index, "pending": pending})
    return {"reports": len(index), "generation_jobs": len(jobs), "retained_stale": len(pending),
            "max_stale_age": max((value["age"] for value in pending.values()), default=0)}


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("run", type=Path)
    plan(arg.parse_args().run.resolve())
