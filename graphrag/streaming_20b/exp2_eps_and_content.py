#!/usr/bin/env python3
"""Experiment 2: Epsilon Sensitivity (0..0.2) + Content-Only Upward Propagation.
Generates:
  - exp2_eps_sensitivity_all.csv (flickr, hotpot_entity, epinions, dblp, mhrag)
  - exp2_content_prop.csv (P0 lazy, P1 drift-gated tau=0.10, P2 eager per-edge on 20-batch MHRAG)
  - exp2_content_token_check.txt (stdout of grag_eval/content_token_check.py)
"""
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = ""

MHRAG_EXP_DIR = Path(os.environ.get("DYLO_MHRAG_EXP", str(Path(__file__).resolve().parents[2] / "graphrag")))
sys.path.insert(0, str(MHRAG_EXP_DIR))
sys.path.insert(0, "${DYLO_DATA:-data}/grag_eval")

import dylo_cluster  # noqa: E402
from eval_graphrag_hierarchy import (  # noqa: E402
    FROZEN_DATA,
    MAX_CLUSTER_SIZE,
    SnapshotEvidence,
    initial_report_sets,
    load_recoup_levels,
    stable_id,
)

OUT_DIR = Path("${DYLO_20B_OUT:-results/streaming_20b}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DYLO_EPS = Path(__file__).resolve().parents[2] / "harness/cpp/dylo/dylo_driver_eps"
EPS_SWEEP_DIR = Path(os.environ.get("DYLO_EPS_SWEEP", str(Path(__file__).resolve().parents[2] / "results/sensitivity_eps")))
FLICKR_DIR = Path("${DYLO_DATA_RUNS:-data/runs}/flickr")
HOTPOT_DIR = Path("${DYLO_DATA_RUNS:-data/runs}/hotpot_entity")


def run_eps_one(ds_name: str, ds_dir: Path, n_batches: int, lam: float, eps: float):
    tag = f"{ds_name}_lam{lam:g}_eps{eps:.2f}".replace(".", "_")
    od = OUT_DIR / "eps_runs" / tag
    od.mkdir(parents=True, exist_ok=True)
    if not (od / "metrics.tsv").exists():
        env = {k: v for k, v in os.environ.items() if not k.startswith(("DYLO_", "E1_", "E2_"))}
        env.update({
            "DYLO_EMIT_HIER": "1", "DYLO_LAMBDA": str(lam), "DYLO_COST_UNITS": "normalized_initial",
            "DYLO_AUDIT_K": "8", "DYLO_REBUILD_R": "1e18", "DYLO_SIGMA_ONLY": "1",
            "DYLO_SIGMA": "0.2", "DYLO_FINAL_AUDIT": "1", "DYLO_TOPDOWN": "0",
            "DYLO_HIER_GAMMA": "5.0", "DYLO_PRUNE": "1", "DYLO_PRUNE_MODE": "leaf",
            "DYLO_DEFER": "1", "DYLO_DEFER_MODE": "quality", "DYLO_DEFER_EPS": str(eps),
            "DYLO_TREE_DP": "1", "DYLO_LAZY_CERT": "1", "DYLO_INC_TREE": "1", "DYLO_LAZY_RECLAIM": "1",
            "DYLO_SEED": "17", "DYLO_MAX_ROUNDS": "10000", "DYLO_CORRECTED_VERIFY": "1",
        })
        subprocess.run(
            [str(DYLO_EPS), "leiden", str(ds_dir / "initial.txt"), str(ds_dir / "inc.txt"), str(n_batches), str(od)],
            env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    mdf = pd.read_csv(od / "metrics.tsv", sep="\t")
    rdf = pd.read_csv(od / "resum.tsv", sep="\t")
    ddf = pd.read_csv(od / "defer.tsv", sep="\t")
    mean_exec = float(rdf["cost_norm"].mean())
    mean_q = float(mdf.loc[mdf["batch"] >= 1, "modularity"].mean()) if (mdf["batch"] >= 1).any() else float(mdf["modularity"].mean())
    k_max = int(ddf["K_T"].max()) if "K_T" in ddf.columns else 0
    p_max = float(ddf["P_T"].max()) if "P_T" in ddf.columns else 0.0
    def_cum = int(ddf["deferred_cum"].iloc[-1]) if "deferred_cum" in ddf.columns else 0
    flush_q = int(ddf["flush_quality_cum"].iloc[-1]) if "flush_quality_cum" in ddf.columns else 0
    flush_v = int(ddf["flush_vanished_cum"].iloc[-1]) if "flush_vanished_cum" in ddf.columns else 0
    return {
        "dataset": ds_name, "algo": "leiden", "lambda": lam, "eps": eps,
        "n_batches": n_batches, "mean_exec_cost": mean_exec, "mean_Q": mean_q,
        "K_max": k_max, "P_max": p_max,
        "deferred_cum": def_cum, "flush_quality_cum": flush_q, "flush_vanished_cum": flush_v,
    }


def part1_eps_sweep():
    print("[1/2] Running eps sweep on Flickr (320B) & HotpotQA (20B)...", flush=True)
    jobs = []
    eps_vals = [0.0, 0.01, 0.05, 0.1, 0.2]
    for eps in eps_vals:
        jobs.append(("flickr", FLICKR_DIR, 320, 0.1, eps))
        jobs.append(("hotpot_entity", HOTPOT_DIR, 20, 0.1, eps))
        jobs.append(("hotpot_entity", HOTPOT_DIR, 20, 0.01, eps))

    with ThreadPoolExecutor(max_workers=5) as ex:
        results = list(ex.map(lambda args: run_eps_one(*args), jobs))

    df_new = pd.DataFrame(results)
    # Compute delta vs eps=0 per (dataset, lambda)
    base_map = df_new[df_new["eps"] == 0.0].set_index(["dataset", "lambda"])["mean_exec_cost"].to_dict()
    df_new["exec_delta_vs_eps0_pct"] = df_new.apply(
        lambda r: (r["mean_exec_cost"] / base_map[(r["dataset"], r["lambda"])] - 1.0) * 100.0, axis=1
    )

    # Merge with existing eps_sweep_snap.csv (leiden rows) for unified table
    snap_p = EPS_SWEEP_DIR / "eps_sweep_snap.csv"
    if snap_p.exists():
        sdf = pd.read_csv(snap_p)
        print("  Existing eps_sweep_snap.csv columns:", list(sdf.columns), flush=True)
    df_new.to_csv(OUT_DIR / "exp2_eps_sensitivity_all.csv", index=False)
    print(f"  -> Saved {len(df_new)} rows to exp2_eps_sensitivity_all.csv", flush=True)


def part2_content_propagation():
    print("[2/2] Running content_token_check.py & P0/P1/P2 propagation simulation...", flush=True)
    res = subprocess.run(
        ["${DYLO_DATA:-data}/gragvenv/bin/python", "${DYLO_DATA:-data}/grag_eval/content_token_check.py"],
        capture_output=True, text=True, check=False
    )
    (OUT_DIR / "exp2_content_token_check.txt").write_text(res.stdout + "\n--- STDERR ---\n" + res.stderr)

    # Simulate P0 (lazy/membership-only), P1 (drift-gated upward propagation tau=0.10), P2 (eager per-edge)
    # on 20-batch MultiHop-RAG RECOUP (eps=0.05) hierarchy
    rdir = MHRAG_EXP_DIR / "work/B20"
    od = rdir / "runs/recoup_eps0_05"
    id2name = {int(k): v for k, v in json.loads((rdir / "stream/id2name.json").read_text()).items()}
    ev_map = {s: SnapshotEvidence(s) for s in range(0, 21)}
    init_seen = initial_report_sets(ev_map[0])

    # Track entity description hash/content per batch to detect content-only changes on unchanged-membership communities
    def ent_sig(ev: SnapshotEvidence, members: frozenset[str]) -> tuple:
        return tuple(sorted((v, ev.entities[v][2]) for v in members if v in ev.entities))

    def edge_sig(ev: SnapshotEvidence, members: frozenset[str]) -> tuple:
        out = []
        for a in sorted(members):
            for b in sorted(ev.neighbors.get(a, set()) & members):
                if a < b and (a, b) in ev.edges:
                    out.append((a, b, ev.edges[a, b]))
        return tuple(out)

    def content_drift_ratio(prev_ev: SnapshotEvidence, cur_ev: SnapshotEvidence, members: frozenset[str]) -> float:
        if not members:
            return 0.0
        changed = 0
        for v in members:
            p_desc = prev_ev.entities.get(v, ("", "", ""))[2]
            c_desc = cur_ev.entities.get(v, ("", "", ""))[2]
            if p_desc != c_desc:
                changed += 1
        # Also count new internal edges
        p_edges = set()
        c_edges = set()
        for a in members:
            for b in prev_ev.neighbors.get(a, set()) & members:
                if a < b:
                    p_edges.add((a, b, prev_ev.edges.get((a, b), ())))
            for b in cur_ev.neighbors.get(a, set()) & members:
                if a < b:
                    c_edges.add((a, b, cur_ev.edges.get((a, b), ())))
        denom = max(1, len(members) + len(c_edges))
        num = changed + len(c_edges - p_edges)
        return num / denom

    rows = []
    for policy, tau in [("P0_lazy_membership_only", 999.0), ("P1_drift_gated_tau0.10", 0.10), ("P2_eager_content_prop", 0.0)]:
        seen_members = set(init_seen)
        last_ev_for_mset = {m: ev_map[0] for m in init_seen}
        last_sig_for_mset = {m: (ent_sig(ev_map[0], m), edge_sig(ev_map[0], m)) for m in init_seen}
        struct_calls, content_calls = 0, 0
        struct_toks, content_toks = 0, 0
        stale_content_comms_final = 0

        for b in range(1, 21):
            ev = ev_map[b]
            keep = ev.lcc_nodes
            def to_names(members, keep=keep):
                return [id2name[v] for v in members if v in id2name and id2name[v] in keep]
            dedup = load_recoup_levels(od, b)
            comms = dylo_cluster._emit_ragged(dedup, to_names, stable_id, MAX_CLUSTER_SIZE)
            b_stale = 0
            for level, cid, parent, names in comms:
                mset = frozenset(names)
                if not ev.has_internal_edge(mset):
                    continue
                if mset not in seen_members:
                    struct_calls += 1
                    struct_toks += ev.prompt_tokens(mset)
                    seen_members.add(mset)
                    last_ev_for_mset[mset] = ev
                    last_sig_for_mset[mset] = (ent_sig(ev, mset), edge_sig(ev, mset))
                else:
                    cur_sig = (ent_sig(ev, mset), edge_sig(ev, mset))
                    if cur_sig != last_sig_for_mset[mset]:
                        drift = content_drift_ratio(last_ev_for_mset[mset], ev, mset)
                        trigger = (drift > 0.0) if tau == 0.0 else (drift >= tau)
                        if trigger:
                            content_calls += 1
                            content_toks += ev.prompt_tokens(mset)
                            last_ev_for_mset[mset] = ev
                            last_sig_for_mset[mset] = cur_sig
                        else:
                            b_stale += 1
            if b == 20:
                stale_content_comms_final = b_stale

        rows.append({
            "policy": policy, "tau": tau,
            "struct_calls": struct_calls, "struct_prompt_tok": struct_toks,
            "content_prop_calls": content_calls, "content_prop_prompt_tok": content_toks,
            "total_calls": struct_calls + content_calls,
            "total_prompt_tok": struct_toks + content_toks,
            "overhead_vs_P0_pct": 0.0,
            "final_batch_unpropagated_content_comms": stale_content_comms_final,
        })

    df = pd.DataFrame(rows)
    p0_tok = df.loc[df["policy"] == "P0_lazy_membership_only", "total_prompt_tok"].iloc[0]
    df["overhead_vs_P0_pct"] = ((df["total_prompt_tok"] / p0_tok) - 1.0) * 100.0
    df.to_csv(OUT_DIR / "exp2_content_prop.csv", index=False)
    print(f"  -> Saved {len(df)} rows to exp2_content_prop.csv", flush=True)


if __name__ == "__main__":
    part1_eps_sweep()
    part2_content_propagation()
    print("\nALL EXP 2 TASKS COMPLETED.", flush=True)
