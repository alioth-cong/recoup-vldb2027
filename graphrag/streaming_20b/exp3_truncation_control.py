#!/usr/bin/env python3
"""Experiment 3: Truncation-Bias-Free Exp-9 Controls & Decoupled Architecture Quantification.
Generates:
  - exp3_arena_matched.csv (am_gpt & am_opus matched-exposure 453-report 120Q win rates + 95% bootstrap CIs, 2-dim & 4-dim)
  - exp3_arena_matched_lengths.csv (mean word count & abstention rate per arm in arena_matched)
  - exp3_32b_full_audit.csv (32B gpt-oss-120b 8-arm full 60Q audit: win rates, zero query-time truncation, C0_reports, answer words)
  - exp3_decoupled_arch.csv (decoupled real-time graph + async batched LLM quantification)
"""
import csv
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("${DYLO_20B_OUT:-results/streaming_20b}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

AM_GPT = Path("${DYLO_DATA:-data}/mhrag/clean_chain/am_gpt")
AM_OPUS = Path("${DYLO_DATA:-data}/mhrag/clean_chain/am_opus")
ARENA_MATCHED_MD = Path("${DYLO_DATA:-data}/mhrag/clean_chain/arena_matched")
Q32_DIR = Path("${DYLO_CASE_ROOT:-graphrag}/outputs/qwen32-std4-20261017/quality")


def eval_arena_judge(judge_name: str, jdir: Path, files: list[tuple[int, str]], rng: np.random.Generator):
    key = json.loads((jdir / "key.json").read_text())
    arms = key["arms"]
    letters = key["letters"]
    orders = key["orders"]

    pair_scores = defaultdict(lambda: defaultdict(list))  # (criterion, baseline) -> qid -> [scores]
    pair_wins = defaultdict(lambda: [0, 0, 0])  # [win, tie, loss]

    for order_idx, fname in files:
        order_map = orders[order_idx]
        with (jdir / fname).open() as f:
            for row in csv.DictReader(f):
                qid_str = row["qid"].lstrip("q")
                crit = row["criterion"].strip().lower()
                ranking = [x.strip() for x in row["ranking"].split(">") if x.strip()]
                perm = order_map[qid_str]
                letter_to_arm = {letters[i]: arms[perm[i]] for i in range(len(letters))}
                rank_pos = {letter_to_arm[L]: pos for pos, L in enumerate(ranking) if L in letter_to_arm}
                if "m-dylo" not in rank_pos:
                    continue
                d_pos = rank_pos["m-dylo"]
                for b_arm in arms:
                    if b_arm == "m-dylo" or b_arm not in rank_pos:
                        continue
                    b_pos = rank_pos[b_arm]
                    if d_pos < b_pos:
                        sc = 1.0
                        pair_wins[(crit, b_arm)][0] += 1
                    elif d_pos == b_pos:
                        sc = 0.5
                        pair_wins[(crit, b_arm)][1] += 1
                    else:
                        sc = 0.0
                        pair_wins[(crit, b_arm)][2] += 1
                    pair_scores[(crit, b_arm)][int(qid_str)].append(sc)

    rows = []
    for (crit, b_arm), qdict in sorted(pair_scores.items()):
        q_means = np.array([np.mean(v) for v in qdict.values()], dtype=float)
        n_q = len(q_means)
        win_rate = float(q_means.mean()) * 100.0
        idx = rng.integers(0, n_q, size=(10000, n_q))
        boot = q_means[idx].mean(axis=1) * 100.0
        ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        w, t, l = pair_wins[(crit, b_arm)]
        rows.append({
            "judge": judge_name, "criterion": crit, "baseline": b_arm,
            "n_questions": n_q, "n_judgments": w + t + l,
            "dylo_win_rate_pct": round(win_rate, 2),
            "ci95_lo_pct": round(ci_lo, 2), "ci95_hi_pct": round(ci_hi, 2),
            "wins": w, "ties": t, "losses": l,
        })

    all_baselines = sorted({b for (_, b) in pair_scores})
    for label, crits in [
        ("overall_2dim", ["comprehensiveness", "diversity"]),
        ("overall_4dim", ["comprehensiveness", "diversity", "empowerment", "directness"]),
    ]:
        for b_arm in all_baselines:
            q_sets = [set(pair_scores[(c, b_arm)].keys()) for c in crits]
            qids = sorted(set.intersection(*q_sets)) if q_sets else []
            if not qids:
                continue
            q_means = np.array([
                float(np.mean([np.mean(pair_scores[(c, b_arm)][q]) for c in crits]))
                for q in qids
            ], dtype=float)
            n_q = len(q_means)
            win_rate = float(q_means.mean()) * 100.0
            idx = rng.integers(0, n_q, size=(10000, n_q))
            boot = q_means[idx].mean(axis=1) * 100.0
            ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
            w = sum(pair_wins[(c, b_arm)][0] for c in crits)
            t = sum(pair_wins[(c, b_arm)][1] for c in crits)
            l = sum(pair_wins[(c, b_arm)][2] for c in crits)
            rows.append({
                "judge": judge_name, "criterion": label, "baseline": b_arm,
                "n_questions": n_q, "n_judgments": w + t + l,
                "dylo_win_rate_pct": round(win_rate, 2),
                "ci95_lo_pct": round(ci_lo, 2), "ci95_hi_pct": round(ci_hi, 2),
                "wins": w, "ties": t, "losses": l,
            })
    return rows


def part1_arena_matched():
    print("[1/3] Tallying Arena Matched (am_gpt & am_opus, 453 aligned reports)...", flush=True)
    rng = np.random.default_rng(42)
    rows = []
    rows.extend(eval_arena_judge("GPT-5", AM_GPT, [(0, "order1.csv"), (1, "order2.csv")], rng))
    rows.extend(eval_arena_judge("Claude-Opus", AM_OPUS, [(0, "order1-part1.csv"), (1, "order2-part1.csv")], rng))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "exp3_arena_matched.csv", index=False)
    print(f"  -> Saved {len(df)} rows to exp3_arena_matched.csv", flush=True)

    # Also parse order1-part*.md to verify answer lengths & abstention rates per arm
    key = json.loads((AM_GPT / "key.json").read_text())
    arms = key["arms"]
    letters = key["letters"]
    order1_map = key["orders"][0]

    arm_words = defaultdict(list)
    arm_abstain = defaultdict(int)
    for md_path in sorted(ARENA_MATCHED_MD.glob("order1-part*.md")):
        txt = md_path.read_text()
        q_blocks = re.split(r"(?m)^## q(\d+)\s*$", txt)
        for i in range(1, len(q_blocks), 2):
            qid_str = q_blocks[i]
            body = q_blocks[i + 1]
            if qid_str not in order1_map:
                continue
            perm = order1_map[qid_str]
            letter_to_arm = {letters[j]: arms[perm[j]] for j in range(len(letters))}
            ans_parts = re.split(r"\*\*Answer ([A-H]):\*\*", body)
            for j in range(1, len(ans_parts), 2):
                L = ans_parts[j]
                ans_txt = ans_parts[j + 1].strip()
                arm = letter_to_arm[L]
                wc = len(ans_txt.split())
                arm_words[arm].append(wc)
                if "unable to answer" in ans_txt.lower() or "i am sorry" in ans_txt.lower():
                    arm_abstain[arm] += 1

    len_rows = []
    for arm in arms:
        wcs = arm_words[arm]
        len_rows.append({
            "arm": arm,
            "n_questions": len(wcs),
            "mean_words": round(float(np.mean(wcs)), 1) if wcs else 0.0,
            "median_words": round(float(np.median(wcs)), 1) if wcs else 0.0,
            "abstention_count": arm_abstain[arm],
            "abstention_pct": round(100.0 * arm_abstain[arm] / max(1, len(wcs)), 2),
        })
    len_df = pd.DataFrame(len_rows)
    len_df.to_csv(OUT_DIR / "exp3_arena_matched_lengths.csv", index=False)
    print(f"  -> Saved {len(len_df)} rows to exp3_arena_matched_lengths.csv", flush=True)


def part2_32b_full_audit():
    print("[2/3] Auditing 32B Exp-9 (gpt-oss-120b, 60Q, all 8 arms)...", flush=True)
    # 1. Answer & query trace stats per arm
    arm_stats = {}
    for arm in ["dylo", "rebuild", "qca", "batch", "df-louvain", "ds-louvain", "dynamo", "hit-leiden"]:
        c0s, words, rc_words = [], [], []
        all_cut_inc, trunc_cnt = 0, 0
        for q in range(60):
            ap = Q32_DIR / "answers" / arm / f"q{q:03d}.json"
            tp = Q32_DIR / "query_traces" / arm / f"q{q:03d}.json"
            if ap.exists():
                ad = json.loads(ap.read_text())
                c0s.append(int(ad.get("C0_reports", 0)))
                words.append(int(ad.get("words", 0)))
            if tp.exists():
                td = json.loads(tp.read_text())
                rc = td.get("reduce_context") or ""
                nrc = td.get("native_reduce_context_before_exact_packing") or ""
                rc_words.append(len(rc.split()))
                if td.get("all_cut_reports_included", False):
                    all_cut_inc += 1
                if len(rc) < len(nrc):
                    trunc_cnt += 1
        arm_stats[arm] = {
            "mean_C0_reports": round(float(np.mean(c0s)), 1) if c0s else 0.0,
            "mean_answer_words": round(float(np.mean(words)), 1) if words else 0.0,
            "mean_reduce_ctx_words": round(float(np.mean(rc_words)), 1) if rc_words else 0.0,
            "all_cut_included_cnt": all_cut_inc,
            "reduce_truncated_cnt": trunc_cnt,
        }

    # 2. Read all 7 quality-full-{baseline}.json
    rows = []
    for b_arm in ["rebuild", "qca", "batch", "df-louvain", "ds-louvain", "dynamo", "hit-leiden"]:
        qp = Q32_DIR / f"quality-full-{b_arm}.json"
        if not qp.exists():
            continue
        qd = json.loads(qp.read_text())
        dim_map = {r["dimension"]: r for r in qd["rows"]}
        # Compute 2-dim and 4-dim from verdicts-full-{b_arm}.json
        vp = Q32_DIR / f"verdicts-full-{b_arm}.json"
        vdata = json.loads(vp.read_text())
        by_q = defaultdict(lambda: defaultdict(list))
        for it in vdata:
            by_q[int(it["qid"])][it["dimension"]].append(float(it["dylo_score"]))
        rng = np.random.default_rng(42)
        q2 = np.array([
            0.5 * (np.mean(by_q[q]["comprehensiveness"]) + np.mean(by_q[q]["diversity"]))
            for q in range(60)
        ])
        q4 = np.array([
            0.25 * sum(np.mean(by_q[q][d]) for d in ("comprehensiveness", "diversity", "empowerment", "directness"))
            for q in range(60)
        ])
        idx = rng.integers(0, 60, size=(10000, 60))
        ci2 = np.percentile(q2[idx].mean(axis=1) * 100.0, [2.5, 97.5])
        ci4 = np.percentile(q4[idx].mean(axis=1) * 100.0, [2.5, 97.5])

        rows.append({
            "baseline": b_arm,
            "n_questions": 60,
            "comp_win_pct": round(dim_map["comprehensiveness"]["dylo_preference_score"], 2),
            "div_win_pct": round(dim_map["diversity"]["dylo_preference_score"], 2),
            "emp_win_pct": round(dim_map["empowerment"]["dylo_preference_score"], 2),
            "dir_win_pct": round(dim_map["directness"]["dylo_preference_score"], 2),
            "overall_2dim_pct": round(float(q2.mean() * 100.0), 2),
            "ci95_2dim_lo": round(float(ci2[0]), 2),
            "ci95_2dim_hi": round(float(ci2[1]), 2),
            "overall_4dim_pct": round(float(q4.mean() * 100.0), 2),
            "ci95_4dim_lo": round(float(ci4[0]), 2),
            "ci95_4dim_hi": round(float(ci4[1]), 2),
            "dylo_C0_reports": arm_stats["dylo"]["mean_C0_reports"],
            "base_C0_reports": arm_stats[b_arm]["mean_C0_reports"],
            "dylo_reduce_ctx_words": arm_stats["dylo"]["mean_reduce_ctx_words"],
            "base_reduce_ctx_words": arm_stats[b_arm]["mean_reduce_ctx_words"],
            "dylo_answer_words": arm_stats["dylo"]["mean_answer_words"],
            "base_answer_words": arm_stats[b_arm]["mean_answer_words"],
            "dylo_all_cut_included": arm_stats["dylo"]["all_cut_included_cnt"],
            "base_all_cut_included": arm_stats[b_arm]["all_cut_included_cnt"],
            "dylo_reduce_truncated": arm_stats["dylo"]["reduce_truncated_cnt"],
            "base_reduce_truncated": arm_stats[b_arm]["reduce_truncated_cnt"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "exp3_32b_full_audit.csv", index=False)
    print(f"  -> Saved {len(df)} rows to exp3_32b_full_audit.csv", flush=True)


def part3_decoupled_arch():
    print("[3/3] Building decoupled architecture quantification table...", flush=True)
    rows = [
        {
            "component": "Real-time Graph Side (Incremental Detection + Selector)",
            "regime": "Per-batch online SLA (ms-scale)",
            "baseline_value": "8.4–156.9 ms (Eager Selector)",
            "recoup_value": "1.6–4.0 ms (Deferred Selector)",
            "improvement": "5.0x–39.0x faster selector; zero-blocking graph state update",
        },
        {
            "component": "Asynchronous LLM Settlement (MultiHop-RAG 20-Batch Stream)",
            "regime": "Batched background LLM calls",
            "baseline_value": "45,578 calls / 51.31M tok (Rebuild); 10,586 calls / 19.68M tok (QCA)",
            "recoup_value": "1,810 calls / 3.29M tok (RECOUP eps=0.05)",
            "improvement": "93.6% token saving vs Rebuild; 83.3% token saving vs QCA",
        },
        {
            "component": "Asynchronous LLM Settlement (HotpotQA 20-Batch Stream)",
            "regime": "Batched background LLM calls",
            "baseline_value": "317,662 tok/batch (Rebuild); 90,379 tok/batch (GreMod)",
            "recoup_value": "51,827 tok/batch (RECOUP/DyLo)",
            "improvement": "83.7% token saving vs Rebuild; 42.7% token saving vs GreMod",
        },
        {
            "component": "Single-Batch Physical LLM Wall-Clock (Clean Chain)",
            "regime": "64-worker concurrent vLLM (Qwen2.5-32B)",
            "baseline_value": "1,901 s / batch (Full Rebuild); 1,039 s (Reuse-aware Rebuild)",
            "recoup_value": "116 s / batch (RECOUP)",
            "improvement": "16.4x faster than Rebuild; 9.0x faster than Reuse-aware Rebuild",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "exp3_decoupled_arch.csv", index=False)
    print(f"  -> Saved {len(df)} rows to exp3_decoupled_arch.csv", flush=True)


if __name__ == "__main__":
    part1_arena_matched()
    part2_32b_full_audit()
    part3_decoupled_arch()
    print("\nALL EXP 3 TASKS COMPLETED.", flush=True)
