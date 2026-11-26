#!/usr/bin/env python3
"""Experiment 1: Multi-Batch Streaming End-to-End Evaluation (20 batches).
Generates:
  - mhrag_b20_per_batch.csv (t=1..20 per-batch & cumulative calls/tokens for 11 methods)
  - mhrag_clean_chain_trajectory.csv (real-LLM 13-batch trajectory from clean_chain/cost_*.tsv)
  - hotpot_b20_per_batch.csv (HotpotQA 20-batch per-batch structural refresh & estimated tokens)
  - pending_liability_pt.csv (P_t, B_t, E_t, Z_t, conservation_error on MHRAG, HotpotQA, Epinions)
  - mhrag_qa_batches_1_5_10_15_20.csv (32B QA EM/F1 + 95% paired CI at Batches 1, 5, 10, 15, 20)
  - mhrag_retrieval_recall_batches.csv (TF-IDF Retrieval Recall@6 at Batches 1, 5, 10, 15, 20)
"""
import csv
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from graspologic.partition import hierarchical_leiden
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

os.environ["CUDA_VISIBLE_DEVICES"] = ""

MHRAG_EXP_DIR = Path(os.environ.get("DYLO_MHRAG_EXP", str(Path(__file__).resolve().parents[2] / "graphrag")))
sys.path.insert(0, str(MHRAG_EXP_DIR))
sys.path.insert(0, "${DYLO_DATA:-data}/grag_eval")

import dylo_cluster  # noqa: E402
from eval_graphrag_hierarchy import (  # noqa: E402
    FROZEN_DATA,
    MAX_CLUSTER_SIZE,
    SnapshotEvidence,
    eval_communities,
    initial_report_sets,
    load_recoup_levels,
    read_com_file,
    stable_id,
)

OUT_DIR = Path("${DYLO_20B_OUT:-results/streaming_20b}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DYLO_V21 = Path("harness/cpp/dylo/dylo_std_v2_1")
DYLO_EPS = Path(__file__).resolve().parents[2] / "harness/cpp/dylo/dylo_driver_eps"
WARM_H100 = Path("${DYLO_20B_H100:-graphrag/streaming_20b}/warm-hierarchy")
HOTPOT_DIR = Path("${DYLO_DATA_RUNS:-data/runs}/hotpot_entity")
EPINIONS_DIR = Path("${DYLO_DATA_RUNS:-data/runs}/epinions")

BASELINES_B20 = [
    ("QCA", "QCA"),
    ("GreMod", "GreMod"),
    ("DynaMo", "DynaMo"),
    ("Batch", "Batch"),
    ("HIT-Leiden", "HIT-Leiden"),
    ("DF-Louvain", "DF-Louvain"),
    ("Delta-screen", "Delta-screen"),
    ("DF-Leiden", "DF-Leiden"),
]


def part1_mhrag_b20_per_batch():
    rdir = MHRAG_EXP_DIR / "work/B20"
    id2name = {int(k): v for k, v in json.loads((rdir / "stream/id2name.json").read_text()).items()}
    if (OUT_DIR / "mhrag_b20_per_batch.csv").exists():
        print("[1/6] Reusing existing mhrag_b20_per_batch.csv; loading snapshots [1,5,10,15,20] for part 6...", flush=True)
        snaps = [1, 5, 10, 15, 20]
        ev_map = {s: SnapshotEvidence(s) for s in snaps}
        fs_comms = {}
        for s in snaps:
            ev = ev_map[s]
            mapping = hierarchical_leiden(ev.lcc_graph, max_cluster_size=MAX_CLUSTER_SIZE, random_seed=17)
            lvl = defaultdict(lambda: defaultdict(list))
            parent = {}
            for p in mapping:
                lvl[p.level][p.cluster].append(p.node)
                parent[p.cluster] = p.parent_cluster if p.parent_cluster is not None else -1
            out = []
            for level in sorted(lvl):
                for cid, names in lvl[level].items():
                    if names:
                        out.append((level, cid, parent.get(cid, -1), names))
            fs_comms[s] = out
        return ev_map, id2name, fs_comms
    print("[1/6] Computing MultiHop-RAG B20 per-batch calls & tokens...", flush=True)
    ev_map = {s: SnapshotEvidence(s) for s in range(0, 21)}
    init_seen = initial_report_sets(ev_map[0])

    # Precompute hierarchical_leiden for Rebuild / Reuse-aware Rebuild
    fs_comms = {}
    for s in range(1, 21):
        ev = ev_map[s]
        mapping = hierarchical_leiden(ev.lcc_graph, max_cluster_size=MAX_CLUSTER_SIZE, random_seed=17)
        lvl = defaultdict(lambda: defaultdict(list))
        parent = {}
        for p in mapping:
            lvl[p.level][p.cluster].append(p.node)
            parent[p.cluster] = p.parent_cluster if p.parent_cluster is not None else -1
        out = []
        for level in sorted(lvl):
            for cid, names in lvl[level].items():
                if names:
                    out.append((level, cid, parent.get(cid, -1), names))
        fs_comms[s] = out

    rows = []

    # 1. Rebuild & Reuse-aware Rebuild
    for rebuild_mode in (True, False):
        mname = "Rebuild" if rebuild_mode else "Reuse-aware Rebuild"
        seen = set(init_seen)
        cum_calls, cum_toks = 0, 0
        for b in range(1, 21):
            c, t, mat_tot, mat_l01 = eval_communities(fs_comms[b], ev_map[b], seen, rebuild=rebuild_mode)
            cum_calls += c
            cum_toks += t
            rows.append({
                "batch": b, "method": mname,
                "batch_calls": c, "batch_prompt_tok": t,
                "cum_calls": cum_calls, "cum_prompt_tok": cum_toks,
                "mat_total": mat_tot, "mat_l01": mat_l01,
            })

    # 2. RECOUP (eps=0, 0.05, 0.20)
    for eps, mname in [(0.0, "RECOUP (eps=0.00)"), (0.05, "RECOUP (eps=0.05)"), (0.20, "RECOUP (eps=0.20)")]:
        arm = f"recoup_eps{eps:.2f}".replace(".", "_")
        od = rdir / "runs" / arm
        seen = set(init_seen)
        cum_calls, cum_toks = 0, 0
        deferred_by_batch = defaultdict(set)
        if eps > 0 and (od / "source_state.tsv").exists():
            with (od / "source_state.tsv").open() as f:
                for row in csv.DictReader(f, delimiter="\t"):
                    deferred_by_batch[int(row["batch"])].add(int(row["source"]))
        for b in range(1, 21):
            ev = ev_map[b]
            keep = ev.lcc_nodes
            def to_names(members, keep=keep):
                return [id2name[v] for v in members if v in id2name and id2name[v] in keep]
            dedup = load_recoup_levels(od, b)
            comms = dylo_cluster._emit_ragged(dedup, to_names, stable_id, MAX_CLUSTER_SIZE)
            def_srcs = deferred_by_batch.get(b, set())
            if def_srcs and b < 20:
                def_msets = set()
                with (od / f"nodes_{b}_L0.tsv").open() as f:
                    for row in csv.DictReader(f, delimiter="\t"):
                        if int(row["community"]) in def_srcs:
                            m = frozenset(to_names([int(x) for x in row["members"].split(",") if x]))
                            if m:
                                def_msets.add(m)
                filtered_comms = [cm for cm in comms if frozenset(cm[3]) not in def_msets]
            else:
                filtered_comms = comms
            c, t, mat_tot, mat_l01 = eval_communities(filtered_comms, ev, seen, rebuild=False)
            cum_calls += c
            cum_toks += t
            rows.append({
                "batch": b, "method": mname,
                "batch_calls": c, "batch_prompt_tok": t,
                "cum_calls": cum_calls, "cum_prompt_tok": cum_toks,
                "mat_total": mat_tot, "mat_l01": mat_l01,
            })

    # 3. 8 Incremental Baselines
    for bname, dname in BASELINES_B20:
        bdir = rdir / "runs" / dname
        seen = set(init_seen)
        cum_calls, cum_toks = 0, 0
        for b in range(1, 21):
            ev = ev_map[b]
            keep = ev.lcc_nodes
            def to_names(members, keep=keep):
                return [id2name[v] for v in members if v in id2name and id2name[v] in keep]
            raw_comms = read_com_file(bdir / f"com_{b}.txt")
            comms = dylo_cluster._emit_subdivided(raw_comms, ev.lcc_graph, to_names, MAX_CLUSTER_SIZE, 17)
            c, t, mat_tot, mat_l01 = eval_communities(comms, ev, seen, rebuild=False)
            cum_calls += c
            cum_toks += t
            rows.append({
                "batch": b, "method": bname,
                "batch_calls": c, "batch_prompt_tok": t,
                "cum_calls": cum_calls, "cum_prompt_tok": cum_toks,
                "mat_total": mat_tot, "mat_l01": mat_l01,
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "mhrag_b20_per_batch.csv", index=False)
    print(f"  -> Saved {len(df)} rows to mhrag_b20_per_batch.csv", flush=True)
    return ev_map, id2name, fs_comms


def part2_clean_chain_trajectory():
    print("[2/6] Parsing real-LLM 13-batch clean_chain cost_*.tsv...", flush=True)
    cdir = Path("${DYLO_DATA:-data}/mhrag/clean_chain")
    rows = []
    for p in sorted(cdir.glob("cost_*.tsv")):
        with p.open() as f:
            agg = defaultdict(lambda: [0, 0])  # [calls, tokens]
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 5:
                    continue
                tag = parts[0]
                try:
                    tok = int(parts[4])
                except ValueError:
                    continue
                agg[tag][0] += 1
                agg[tag][1] += tok
            for tag, (calls, toks) in sorted(agg.items()):
                rows.append({
                    "file": p.name, "tag": tag,
                    "calls": calls, "prompt_tokens": toks,
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "mhrag_clean_chain_trajectory.csv", index=False)
    print(f"  -> Saved {len(df)} rows to mhrag_clean_chain_trajectory.csv", flush=True)


def part3_hotpot_b20():
    print("[3/6] Computing HotpotQA 20-batch per-batch metrics + RECOUP...", flush=True)
    # Run dylo_driver_eps on hotpot_entity with eps=0.05 (lambda=0.1)
    hp_out = OUT_DIR / "hotpot_recoup_eps005"
    hp_out.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DYLO_", "E1_", "E2_"))}
    env.update({
        "DYLO_EMIT_HIER": "1", "DYLO_LAMBDA": "0.1", "DYLO_COST_UNITS": "normalized_initial",
        "DYLO_AUDIT_K": "8", "DYLO_REBUILD_R": "1e18", "DYLO_SIGMA_ONLY": "1",
        "DYLO_SIGMA": "0.2", "DYLO_FINAL_AUDIT": "1", "DYLO_TOPDOWN": "0",
        "DYLO_HIER_GAMMA": "5.0", "DYLO_PRUNE": "1", "DYLO_PRUNE_MODE": "leaf",
        "DYLO_DEFER": "1", "DYLO_DEFER_MODE": "quality", "DYLO_DEFER_EPS": "0.05",
        "DYLO_TREE_DP": "1", "DYLO_LAZY_CERT": "1", "DYLO_INC_TREE": "1", "DYLO_LAZY_RECLAIM": "1",
        "DYLO_SEED": "17", "DYLO_MAX_ROUNDS": "10000", "DYLO_CORRECTED_VERIFY": "1",
    })
    subprocess.run(
        [str(DYLO_EPS), "leiden", str(HOTPOT_DIR / "initial.txt"), str(HOTPOT_DIR / "inc.txt"), "20", str(hp_out)],
        env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    # Also read hotpot_tokens.json for published per-batch token averages
    hp_tok_ref = {r["method"]: r for r in json.loads(Path("${DYLO_DATA:-data}/hotpot_tokens.json").read_text())}

    # Map HotpotQA out_* directories to compute exact per-batch structural refresh & member-set churn
    # Note: hotpot_entity has 20 batches: com_0.txt .. com_19.txt
    method_dirs = {
        "From-scratch Leiden": HOTPOT_DIR / "out_leiden-cpp-static",
        "QCA": HOTPOT_DIR / "out_qca",
        "GreMod": HOTPOT_DIR / "out_gremod",
        "HIT-Leiden": HOTPOT_DIR / "out_hit-leiden",
        "DynaMo": HOTPOT_DIR / "out_dynamo",
        "Batch": HOTPOT_DIR / "out_batch",
        "DF-Louvain": HOTPOT_DIR / "out_df-louvain",
        "$\\Delta$-screen": HOTPOT_DIR / "out_ds-louvain",
        "DF-Leiden": HOTPOT_DIR / "out_df-leiden",
    }

    rows = []
    # Reuse-aware From-scratch Leiden (member-set deduplication)
    for mname, mdir in method_dirs.items():
        if not mdir.exists():
            continue
        is_rebuild = (mname == "From-scratch Leiden")
        seen = set()
        cum_refreshed = 0
        ref_avg_tok = hp_tok_ref.get(mname, {}).get("tokens", np.nan)
        ref_avg_comms = hp_tok_ref.get(mname, {}).get("comms", np.nan)
        batch_refreshes = []
        for b in range(20):
            p = mdir / f"com_{b}.txt"
            comms = []
            for line in p.read_text().splitlines():
                toks = [x for x in line.split() if x != "null"]
                if len(toks) >= 2:
                    comms.append(frozenset(int(x) for x in toks))
            if is_rebuild:
                ref = len(comms)
            else:
                ref = sum(1 for c in comms if c not in seen)
                seen.update(comms)
            cum_refreshed += ref
            batch_refreshes.append((b + 1, len(comms), ref, cum_refreshed))
        mean_ref = np.mean([x[2] for x in batch_refreshes])
        tok_per_comm = (ref_avg_tok / mean_ref) if (not np.isnan(ref_avg_tok) and mean_ref > 0) else np.nan
        for b1, n_comms, ref, cum_ref in batch_refreshes:
            est_tok = ref * tok_per_comm if not np.isnan(tok_per_comm) else np.nan
            rows.append({
                "batch": b1, "method": mname,
                "total_comms": n_comms, "refreshed_comms": ref, "cum_refreshed_comms": cum_ref,
                "est_batch_tokens": est_tok, "ref_avg_tokens_per_batch": ref_avg_tok,
                "ref_avg_comms_per_batch": ref_avg_comms,
            })

    # RECOUP (eps=0.05) from hp_out/metrics.tsv & resum.tsv
    # Note: resum.tsv has exact refreshed_nodes / total_nodes; metrics.tsv has executed_cost, Q, etc.
    # Also compute member-set deduplicated communities from nodes_{b}_L*.tsv if present, or from resum.tsv
    hp_metrics = pd.read_csv(hp_out / "metrics.tsv", sep="\t")
    hp_metrics = hp_metrics[hp_metrics["batch"] >= 1].reset_index(drop=True)
    hp_resum = pd.read_csv(hp_out / "resum.tsv", sep="\t")
    dylo_ref_tok = hp_tok_ref.get("DyLo-Leiden", {}).get("tokens", 51826.95)
    dylo_ref_comms = hp_tok_ref.get("DyLo-Leiden", {}).get("comms", 321.9)
    # Use cost_norm ratio from resum.tsv to distribute tokens across batches proportionally, anchored to total 20-batch tokens
    exec_costs = hp_resum["cost_norm"].values
    sum_exec = np.sum(exec_costs) if np.sum(exec_costs) > 0 else 1.0
    tot_dylo_tok = dylo_ref_tok * 20.0
    cum_ref = 0
    for idx, row in hp_metrics.iterrows():
        b1 = int(row["batch"])
        # Approximate refreshed comms proportional to executed_cost
        est_tok = tot_dylo_tok * (exec_costs[idx] / sum_exec)
        ref_comms = dylo_ref_comms * 20.0 * (exec_costs[idx] / sum_exec)
        cum_ref += ref_comms
        rows.append({
            "batch": b1, "method": "RECOUP (eps=0.05)",
            "total_comms": np.nan, "refreshed_comms": ref_comms, "cum_refreshed_comms": cum_ref,
            "est_batch_tokens": est_tok, "ref_avg_tokens_per_batch": dylo_ref_tok,
            "ref_avg_comms_per_batch": dylo_ref_comms,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "hotpot_b20_per_batch.csv", index=False)
    print(f"  -> Saved {len(df)} rows to hotpot_b20_per_batch.csv", flush=True)


def part4_pending_liability_pt():
    print("[4/6] Running dylo_std_v2_1 ledger extraction for P_t dynamics...", flush=True)
    all_rows = []
    # 1. Existing MHRAG B20 runs (eps=0.05, 0.20)
    for eps in (0.05, 0.20):
        arm = f"recoup_eps{eps:.2f}".replace(".", "_")
        ledger_p = MHRAG_EXP_DIR / "work/B20/runs" / arm / "ledger.tsv"
        df = pd.read_csv(ledger_p, sep="\t")
        for _, r in df.iterrows():
            all_rows.append({
                "dataset": f"MultiHop-RAG (eps={eps:g})",
                "batch": int(r["batch"]),
                "executed_raw": float(r["executed_raw"]),
                "pending_start": float(r["pending_start"]),
                "pending_end_Pt": float(r["pending_end"]),
                "booked_Bt": float(r["booked"]),
                "retired_Zt": float(r["retired"]),
                "canceled": float(r["canceled"]),
                "cumulative_executed": float(r["cumulative_executed"]),
                "conservation_error": float(r["conservation_error"]),
            })

    # 2. HotpotQA (20 batches) & Epinions (41 batches) via dylo_std_v2_1
    for ds_name, ds_dir, n_batches in [
        ("HotpotQA (eps=0.05)", HOTPOT_DIR, 20),
        ("Epinions (eps=0.05)", EPINIONS_DIR, 41),
    ]:
        out_d = OUT_DIR / ("ledger_" + ds_name.split()[0].lower())
        out_d.mkdir(parents=True, exist_ok=True)
        env = {k: v for k, v in os.environ.items() if not k.startswith(("DYLO_", "E1_", "E2_"))}
        env.update({
            "DYLO_EMIT_HIER": "1", "DYLO_LAMBDA": "0.1", "DYLO_COST_UNITS": "normalized_initial",
            "DYLO_AUDIT_K": "8", "DYLO_REBUILD_R": "1e18", "DYLO_SIGMA_ONLY": "1",
            "DYLO_SIGMA": "0.2", "DYLO_FINAL_AUDIT": "1", "DYLO_TOPDOWN": "0",
            "DYLO_HIER_GAMMA": "5.0", "DYLO_PRUNE": "1", "DYLO_PRUNE_MODE": "leaf",
            "DYLO_DEFER": "1", "DYLO_DEFER_MODE": "quality", "DYLO_DEFER_EPS": "0.05",
            "DYLO_TREE_DP": "1", "DYLO_LAZY_CERT": "1", "DYLO_INC_TREE": "1", "DYLO_LAZY_RECLAIM": "1",
            "DYLO_SEED": "17", "DYLO_MAX_ROUNDS": "10000", "DYLO_CORRECTED_VERIFY": "1",
        })
        subprocess.run(
            [str(DYLO_V21), "leiden", str(ds_dir / "initial.txt"), str(ds_dir / "inc.txt"), str(n_batches), str(out_d)],
            env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        df = pd.read_csv(out_d / "ledger.tsv", sep="\t")
        for _, r in df.iterrows():
            all_rows.append({
                "dataset": ds_name,
                "batch": int(r["batch"]),
                "executed_raw": float(r["executed_raw"]),
                "pending_start": float(r["pending_start"]),
                "pending_end_Pt": float(r["pending_end"]),
                "booked_Bt": float(r["booked"]),
                "retired_Zt": float(r["retired"]),
                "canceled": float(r["canceled"]),
                "cumulative_executed": float(r["cumulative_executed"]),
                "conservation_error": float(r["conservation_error"]),
            })

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(OUT_DIR / "pending_liability_pt.csv", index=False)
    print(f"  -> Saved {len(out_df)} rows to pending_liability_pt.csv", flush=True)


def part5_qa_batches():
    print("[5/6] Extracting 32B QA EM/F1 at Batches 1, 5, 10, 15, 20...", flush=True)
    res = json.loads((WARM_H100 / "inference/results.json").read_text())["qa"]
    df = pd.DataFrame(res)
    rng = np.random.default_rng(42)

    rows = []
    target_batches = [1, 5, 10, 15, 20]
    for b in target_batches + ["ALL"]:
        sub = df if b == "ALL" else df[df["batch"] == b]
        eager_sub = sub[sub["arm"] == "eager"].sort_values(["batch", "id"])
        for arm in ["eager", "deferred-000", "deferred-005", "deferred-020"]:
            arm_sub = sub[sub["arm"] == arm].sort_values(["batch", "id"])
            n_q = len(arm_sub)
            em_mean = float(arm_sub["em"].mean()) * 100.0
            f1_mean = float(arm_sub["f1"].mean()) * 100.0
            unk_mean = float(arm_sub["unknown"].mean()) * 100.0
            stale_hits = int(arm_sub["read_stale_report"].sum())

            # Paired bootstrap vs eager
            if arm == "eager":
                df1, ci_lo, ci_hi = 0.0, 0.0, 0.0
            else:
                diff = (arm_sub["f1"].values - eager_sub["f1"].values) * 100.0
                df1 = float(diff.mean())
                idx = rng.integers(0, len(diff), size=(2000, len(diff)))
                boot = diff[idx].mean(axis=1)
                ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

            rows.append({
                "batch": str(b), "arm": arm, "n_questions": n_q,
                "em_pct": round(em_mean, 2), "f1_pct": round(f1_mean, 2),
                "unknown_pct": round(unk_mean, 2), "stale_hits": stale_hits,
                "delta_f1_vs_eager_pp": round(df1, 2),
                "ci95_lo_pp": round(ci_lo, 2), "ci95_hi_pp": round(ci_hi, 2),
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_DIR / "mhrag_qa_batches_1_5_10_15_20.csv", index=False)
    print(f"  -> Saved {len(out_df)} rows to mhrag_qa_batches_1_5_10_15_20.csv", flush=True)


def part6_retrieval_recall(ev_map, id2name, fs_comms):
    print("[6/6] Computing TF-IDF Retrieval Recall@6 at Batches 1, 5, 10, 15, 20...", flush=True)
    queries = json.loads((WARM_H100 / "queries.json").read_text())
    rdir = MHRAG_EXP_DIR / "work/B20"
    target_batches = [1, 5, 10, 15, 20]

    # Map document_id -> set of entity titles from batch_20 text_units / documents
    doc2ents = defaultdict(set)
    # Build from entities.parquet + text_units.parquet in batch_20
    tu_df = pd.read_parquet(FROZEN_DATA / "batch_20/text_units.parquet")
    ent_df = pd.read_parquet(FROZEN_DATA / "batch_20/entities.parquet")
    doc_df = pd.read_parquet(FROZEN_DATA / "batch_20/documents.parquet")
    # Check doc id mapping
    tu_id2doc = {}
    for row in tu_df.itertuples():
        dids = getattr(row, "document_ids", [])
        if dids is not None and len(dids) > 0:
            tu_id2doc[row.id] = dids[0]
    doc_idx_map = {row.id: idx for idx, row in enumerate(doc_df.itertuples())}
    for row in ent_df.itertuples():
        tids = getattr(row, "text_unit_ids", [])
        if tids is not None:
            for tid in tids:
                did = tu_id2doc.get(tid)
                if did in doc_idx_map:
                    doc2ents[doc_idx_map[did]].add(row.title)
                if did is not None:
                    doc2ents[did].add(row.title)

    # Pre-load deferred sources for RECOUP eps=0.05
    od_recoup = rdir / "runs/recoup_eps0_05"
    deferred_by_batch = defaultdict(set)
    if (od_recoup / "source_state.tsv").exists():
        with (od_recoup / "source_state.tsv").open() as f:
            for row in csv.DictReader(f, delimiter="\t"):
                deferred_by_batch[int(row["batch"])].add(int(row["source"]))

    rows = []
    methods = ["Rebuild", "RECOUP (eps=0.05)", "QCA", "GreMod", "HIT-Leiden"]
    for b in target_batches:
        ev = ev_map[b]
        keep = ev.lcc_nodes
        def to_names(members, keep=keep):
            return [id2name[v] for v in members if v in id2name and id2name[v] in keep]

        active_qs = [q for q in queries if q["complete_batch"] <= b]
        # Gold entities per query (entities in query's documents that exist in batch b)
        q_gold_ents = []
        valid_qs = []
        for q in active_qs:
            gents = set()
            for d in q["documents"]:
                gents |= doc2ents.get(d, set())
            gents &= set(ev.entities.keys())
            if gents:
                valid_qs.append(q)
                q_gold_ents.append(gents)

        for mname in methods:
            if mname == "Rebuild":
                comms = fs_comms[b]
            elif mname == "RECOUP (eps=0.05)":
                dedup = load_recoup_levels(od_recoup, b)
                comms = dylo_cluster._emit_ragged(dedup, to_names, stable_id, MAX_CLUSTER_SIZE)
            else:
                raw = read_com_file(rdir / "runs" / mname / f"com_{b}.txt")
                comms = dylo_cluster._emit_subdivided(raw, ev.lcc_graph, to_names, MAX_CLUSTER_SIZE, 17)

            # Build materialized community texts (only communities with internal edges)
            comm_texts = []
            comm_ents = []
            for level, cid, parent, names in comms:
                mset = frozenset(names)
                if not ev.has_internal_edge(mset):
                    continue
                txt = " ".join(
                    f"{v} {ev.entities[v][2]}" for v in names if v in ev.entities
                )
                comm_texts.append(txt)
                comm_ents.append(set(names))

            if not comm_texts or not valid_qs:
                continue

            vec = TfidfVectorizer(max_features=12000, stop_words="english")
            X_c = vec.fit_transform(comm_texts)
            X_q = vec.transform([q["question"] for q in valid_qs])
            sims = cosine_similarity(X_q, X_c)

            recalls_6 = []
            hit_any_6 = []
            for qi, gents in enumerate(q_gold_ents):
                top6 = np.argsort(-sims[qi])[:6]
                ret_ents = set()
                for ci in top6:
                    ret_ents |= comm_ents[ci]
                recalls_6.append(len(ret_ents & gents) / len(gents))
                hit_any_6.append(1.0 if (ret_ents & gents) else 0.0)

            rows.append({
                "batch": b, "method": mname, "n_active_queries": len(valid_qs),
                "n_mat_comms": len(comm_texts),
                "entity_recall_at_6_pct": round(float(np.mean(recalls_6)) * 100.0, 2),
                "hit_any_at_6_pct": round(float(np.mean(hit_any_6)) * 100.0, 2),
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_DIR / "mhrag_retrieval_recall_batches.csv", index=False)
    print(f"  -> Saved {len(out_df)} rows to mhrag_retrieval_recall_batches.csv", flush=True)


if __name__ == "__main__":
    ev_map, id2name, fs_comms = part1_mhrag_b20_per_batch()
    part2_clean_chain_trajectory()
    part3_hotpot_b20()
    part4_pending_liability_pt()
    part5_qa_batches()
    part6_retrieval_recall(ev_map, id2name, fs_comms)
    print("\nALL EXP 1 TASKS COMPLETED.", flush=True)
