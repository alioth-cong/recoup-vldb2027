#!/usr/bin/env python3
"""Reproduce Exp-5 (selector fidelity & joint-pricing ablation) using dylo_std_v2_1."""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import statistics as st
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRV = ROOT / "harness/cpp/dylo/dylo_std_v2_1"
RUNS = Path(os.environ.get("DYLO_RUNS_DIR", "/ssd3/yuheng/dylo-data/rerun/data/runs"))
MHRAG_STREAM = Path(os.environ.get(
    "DYLO_MHRAG_STREAM",
    "/home/yuheng/Documents/Codex/2026-11-20/dylo-deferral-h100/warm-hierarchy/stream",
))
SCRATCH = Path("/dev/shm/yuheng_repo_exp5")

DATASETS = ["facebook", "cit-hepth", "cit-hepph", "epinions", "twitter", "dblp", "flickr", "youtube"]
INSTANCES = ["louvain", "slm", "leiden"]

BASE_ENV = {
    "DYLO_EMIT_HIER": "1",
    "DYLO_COST_UNITS": "normalized_initial",
    "DYLO_AUDIT_K": "8",
    "DYLO_REBUILD_R": "1e18",
    "DYLO_SIGMA_ONLY": "1",
    "DYLO_SIGMA": "0.2",
    "DYLO_FINAL_AUDIT": "1",
    "DYLO_TOPDOWN": "0",
    "DYLO_HIER_GAMMA": "0.1",
    "DYLO_PRUNE": "1",
    "DYLO_PRUNE_MODE": "leaf",
    "DYLO_DEFER_MODE": "quality",
    "DYLO_DEFER_EPS": "0.05",
    "DYLO_TREE_DP": "1",
    "DYLO_LAZY_CERT": "1",
    "DYLO_INC_TREE": "1",
    "DYLO_LAZY_RECLAIM": "1",
    "DYLO_RECLAIM": "1",
    "DYLO_FAST_RECLAIM": "1",
    "DYLO_PEEL": "1",
    "DYLO_CONFLICT_FILTER": "1",
    "DYLO_ITER_PEEL": "1",
    "DYLO_SEQ_SINGLETON": "0",
    "DYLO_NO_COM": "1",
    "DYLO_SEED": "42",
    "OMP_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "",
}


def clean_env(extra: dict[str, str]) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("DYLO_")}
    env.update(BASE_ENV)
    env.update(extra)
    return env


def parse_stderr(err: str) -> dict[str, float | int]:
    m = re.search(
        r"\[selector_fidelity\] prop_rounds=(\d+) direct_acc=(\d+) iter_peel_multi=(\d+) "
        r"singleton_peel=(\d+) committed_rounds=(\d+) multi_move_rounds=(\d+) "
        r"total_moves=(\d+) multi_moves=(\d+)",
        err,
    )
    if not m:
        raise RuntimeError(f"missing [selector_fidelity] in stderr:\n{err[-400:]}")
    out: dict[str, float | int] = {
        "proposed": int(m.group(1)),
        "direct_accepted": int(m.group(2)),
        "iter_peel_multi": int(m.group(3)),
        "singleton_peel": int(m.group(4)),
        "committed_rounds": int(m.group(5)),
        "multi_move_rounds": int(m.group(6)),
        "total_moves": int(m.group(7)),
        "multi_moves": int(m.group(8)),
    }
    out["rejected"] = int(out["proposed"]) - int(out["direct_accepted"])
    out["peel"] = int(out["iter_peel_multi"]) + int(out["singleton_peel"])
    ma = re.search(r"\[df_audit\] commits=(\d+) max_diff=([0-9.eE+-]+)", err)
    out["df_commits"] = int(ma.group(1)) if ma else -1
    out["df_max_diff"] = float(ma.group(2)) if ma else -1.0
    return out


def run_snap_one(args: tuple[str, str, str, int, int]) -> dict:
    ds, inst, lam, defer, audit_df = args
    meta = json.loads((RUNS / ds / "meta.json").read_text())
    nb = int(meta["n_batches"])
    arm = f"defer{defer}-lam{lam}"
    out = SCRATCH / f"{ds}_{inst}_{arm}"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    try:
        extra = {
            "DYLO_LAMBDA": lam,
            "DYLO_DEFER": str(defer),
            "DYLO_AUDIT_DF": str(audit_df),
        }
        p = subprocess.run(
            [str(DRV), inst, str(RUNS / ds / "initial.txt"), str(RUNS / ds / "inc.txt"), str(nb), str(out)],
            env=clean_env(extra), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=True, timeout=3600,
        )
        stats = parse_stderr(p.stderr)
        mrows = [l.split("\t") for l in (out / "metrics.tsv").read_text().splitlines()[1:]]
        q_mean = st.mean(float(r[2]) for r in mrows[1:])
        rrows = [l.split("\t") for l in (out / "resum.tsv").read_text().splitlines()[1:]]
        ex_mean = st.mean(float(r[5]) for r in rrows)
        lrows = [l.split("\t") for l in (out / "ledger.tsv").read_text().splitlines()[1:]]
        max_lerr = max((abs(float(r[-1])) for r in lrows), default=0.0)
        prop = int(stats["proposed"])
        return {
            "id": f"validate_peel__{ds}__{inst}__s42__{arm}",
            "ds": ds,
            "inst": inst,
            "seed": 42,
            "lambda": float(lam),
            "arm": arm,
            "defer": defer,
            "conflict_filter": 1,
            "iter_peel": 1,
            "status": "pass",
            "proposed": prop,
            "direct_accepted": int(stats["direct_accepted"]),
            "rejected": int(stats["rejected"]),
            "iter_peel_multi": int(stats["iter_peel_multi"]),
            "singleton_peel": int(stats["singleton_peel"]),
            "peel": int(stats["peel"]),
            "committed_rounds": int(stats["committed_rounds"]),
            "multi_move_rounds": int(stats["multi_move_rounds"]),
            "total_moves": int(stats["total_moves"]),
            "multi_moves": int(stats["multi_moves"]),
            "direct_acc_rate": round(int(stats["direct_accepted"]) / prop, 6) if prop else 1.0,
            "Q_observed_mean": q_mean,
            "executed_mean": ex_mean,
            "max_ledger_error": max_lerr,
            "df_commits": int(stats["df_commits"]),
            "df_max_diff": float(stats["df_max_diff"]),
        }
    finally:
        shutil.rmtree(out, ignore_errors=True)


def run_mhrag_ablation() -> list[dict]:
    rows = []
    if not (MHRAG_STREAM / "initial.txt").exists():
        print("WARN: MHRAG stream directory not found; skipping MHRAG ablation.", file=sys.stderr)
        return rows
    variants = [
        ("dp", {"DYLO_SEL_MODE": "dp", "DYLO_CONFLICT_FILTER": "1", "DYLO_ITER_PEEL": "1", "DYLO_SEQ_SINGLETON": "0", "DYLO_AUDIT_DF": "1"}),
        ("greedy", {"DYLO_SEL_MODE": "greedy", "DYLO_CONFLICT_FILTER": "1", "DYLO_ITER_PEEL": "1", "DYLO_SEQ_SINGLETON": "0", "DYLO_AUDIT_DF": "1"}),
        ("indep", {"DYLO_SEL_MODE": "indep", "DYLO_CONFLICT_FILTER": "1", "DYLO_ITER_PEEL": "1", "DYLO_SEQ_SINGLETON": "0", "DYLO_AUDIT_DF": "1"}),
        ("seq_singleton", {"DYLO_SEL_MODE": "dp", "DYLO_CONFLICT_FILTER": "0", "DYLO_ITER_PEEL": "0", "DYLO_SEQ_SINGLETON": "1", "DYLO_AUDIT_DF": "0"}),
    ]
    for lam in ("0.1", "0.01"):
        for vname, vextra in variants:
            if lam == "0.01" and vname == "seq_singleton":
                continue
            out = SCRATCH / f"mhrag20_{lam}_{vname}"
            shutil.rmtree(out, ignore_errors=True)
            out.mkdir(parents=True, exist_ok=True)
            try:
                extra = {
                    "DYLO_LAMBDA": lam,
                    "DYLO_DEFER": "1",
                    "DYLO_INIT_PART": str(MHRAG_STREAM / "init_part.txt"),
                    "DYLO_SKIP_INIT_DETECT": "1",
                    **vextra,
                }
                p = subprocess.run(
                    [str(DRV), "leiden", str(MHRAG_STREAM / "initial.txt"), str(MHRAG_STREAM / "inc.txt"), "20", str(out)],
                    env=clean_env(extra), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=True, timeout=1800,
                )
                stats = parse_stderr(p.stderr)
                mrows = [l.split("\t") for l in (out / "metrics.tsv").read_text().splitlines()[1:]]
                q_mean = st.mean(float(r[2]) for r in mrows[1:])
                rrows = [l.split("\t") for l in (out / "resum.tsv").read_text().splitlines()[1:]]
                ex_mean = st.mean(float(r[5]) for r in rrows)
                prop = int(stats["proposed"])
                rows.append({
                    "dataset": "multihop_rag_20b",
                    "lambda": float(lam),
                    "selector": vname,
                    "Q_mean": round(q_mean, 6),
                    "exec_mean": round(ex_mean, 6),
                    "proposed": prop,
                    "direct_accepted": int(stats["direct_accepted"]),
                    "direct_acc_pct": round(100.0 * int(stats["direct_accepted"]) / prop, 2) if prop else 100.0,
                    "iter_peel_multi": int(stats["iter_peel_multi"]),
                    "singleton_peel": int(stats["singleton_peel"]),
                    "committed_rounds": int(stats["committed_rounds"]),
                    "multi_move_rounds": int(stats["multi_move_rounds"]),
                    "total_moves": int(stats["total_moves"]),
                    "multi_moves": int(stats["multi_moves"]),
                    "multi_move_pct": round(100.0 * int(stats["multi_moves"]) / max(1, int(stats["total_moves"])), 2),
                    "df_commits": int(stats["df_commits"]),
                    "df_max_diff": float(stats["df_max_diff"]),
                })
                print(
                    f"[mhrag20] lam={lam:4s} sel={vname:13s}: Q={q_mean:.5f} exec={ex_mean:.5f} "
                    f"dir={stats['direct_accepted']}/{prop} ({rows[-1]['direct_acc_pct']:.1f}%) "
                    f"multiM={stats['multi_moves']}/{stats['total_moves']} ({rows[-1]['multi_move_pct']:.1f}%) "
                    f"df_commits={stats['df_commits']}",
                    flush=True,
                )
            finally:
                shutil.rmtree(out, ignore_errors=True)
    return rows


def main() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    # 8 graphs x 3 instances x (defer=1 lam=0.1, defer=0 lam=0.1, defer=1 lam=0.01) = 72 runs, all with exact DeltaF oracle!
    jobs = []
    for lam, defer in (("0.1", 1), ("0.1", 0), ("0.01", 1)):
        for ds in DATASETS:
            for inst in INSTANCES:
                # Full-graph DeltaF oracle on all graphs at lam=0.1, and on first 6 graphs at lam=0.01
                audit_df = 1 if (lam == "0.1" or ds not in ("flickr", "youtube")) else 0
                jobs.append((ds, inst, lam, defer, audit_df))
    rows = []
    with ProcessPoolExecutor(max_workers=24) as ex:
        for r in ex.map(run_snap_one, jobs):
            rows.append(r)

    rows.sort(key=lambda r: (r["arm"], r["ds"], r["inst"]))
    (ROOT / "results/validate_peel_rate.json").write_text(json.dumps(rows, indent=1) + "\n")

    for arm in ("defer1-lam0.1", "defer0-lam0.1", "defer1-lam0.01"):
        sub = [r for r in rows if r["arm"] == arm]
        prop = sum(r["proposed"] for r in sub)
        dacc = sum(r["direct_accepted"] for r in sub)
        rej = sum(r["rejected"] for r in sub)
        ipm = sum(r["iter_peel_multi"] for r in sub)
        sp = sum(r["singleton_peel"] for r in sub)
        cr = sum(r["committed_rounds"] for r in sub)
        mmr = sum(r["multi_move_rounds"] for r in sub)
        tm = sum(r["total_moves"] for r in sub)
        mm = sum(r["multi_moves"] for r in sub)
        dfc = sum(max(0, r["df_commits"]) for r in sub)
        maxd = max((r["df_max_diff"] for r in sub if r["df_commits"] >= 0), default=0.0)
        print(
            f"[SNAP {arm:14s}] prop={prop} dir_acc={dacc} ({100*dacc/max(1,prop):.2f}%) rej={rej} "
            f"iterM={ipm} singP={sp} | multiR={mmr}/{cr} ({100*mmr/max(1,cr):.1f}%) "
            f"multiM={mm}/{tm} ({100*mm/max(1,tm):.1f}%) | Q={st.mean(r['Q_observed_mean'] for r in sub):.5f} "
            f"exec={st.mean(r['executed_mean'] for r in sub):.5f} | df_audit={dfc} max_diff={maxd:.2e}",
            flush=True,
        )

    mh_rows = run_mhrag_ablation()
    if mh_rows:
        out_csv = ROOT / "results/selector_ablation_w1.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(mh_rows[0].keys()))
            w.writeheader()
            w.writerows(mh_rows)

    shutil.rmtree(SCRATCH, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
