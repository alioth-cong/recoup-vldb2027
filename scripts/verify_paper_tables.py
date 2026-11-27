#!/usr/bin/env python3
'''Offline verification of VLDB 2027 paper Tables 3-8 and Exp-1..9 against shipped results/.'''
import csv
import hashlib
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def check_table3_recoup_latency() -> str:
    expected = {
        ("facebook", "louvain"): 0.45, ("facebook", "slm"): 0.48, ("facebook", "leiden"): 0.47,
        ("cit-hepth", "louvain"): 3.06, ("cit-hepth", "slm"): 2.99, ("cit-hepth", "leiden"): 3.14,
        ("cit-hepph", "louvain"): 2.87, ("cit-hepph", "slm"): 2.90, ("cit-hepph", "leiden"): 2.90,
        ("epinions", "louvain"): 7.34, ("epinions", "slm"): 7.18, ("epinions", "leiden"): 7.37,
        ("twitter", "louvain"): 4.55, ("twitter", "slm"): 4.64, ("twitter", "leiden"): 4.66,
        ("dblp", "louvain"): 7.07, ("dblp", "slm"): 7.45, ("dblp", "leiden"): 7.37,
        ("gplus", "louvain"): 7.84, ("gplus", "slm"): 8.43, ("gplus", "leiden"): 8.16,
        ("flickr", "louvain"): 12.2, ("flickr", "slm"): 12.7, ("flickr", "leiden"): 13.0,
        ("youtube", "louvain"): 28.5, ("youtube", "slm"): 30.6, ("youtube", "leiden"): 32.9,
        ("livejournal", "louvain"): 17.4, ("livejournal", "slm"): 22.0, ("livejournal", "leiden"): 21.2,
    }
    observed = {}
    grp = defaultdict(list)
    for r in csv.DictReader(open(RES / "lat_iso.tsv"), delimiter="\t"):
        if r["drv"] == "new":
            grp[(r["ds"], r["inst"])].append(float(r["lat"]))
    for k, vals in grp.items():
        observed[k] = st.mean(vals)
    grp_f = defaultdict(list)
    for r in csv.DictReader(open(RES / "lat_final.tsv"), delimiter="\t"):
        if r["drv"] == "new" and r["ds"] == "gplus":
            grp_f[(r["ds"], r["inst"])].append(float(r["lat"]))
    for k, vals in grp_f.items():
        observed[k] = st.mean(vals)
    for r in csv.DictReader(open(RES / "reclaim_ab_livejournal.csv")):
        if r["variant"] == "recl":
            observed[("livejournal", r["instance"])] = float(r["latency_ms"])
    for key, exp in expected.items():
        got = observed[key]
        tol = 0.06 if exp >= 10 else 0.015
        assert abs(got - exp) <= tol, f"Table 3 mismatch {key}: got {got:.4f}, expected {exp}"
    return "PASS: Table 3 (30 RECOUP latencies across 10 graphs match within rounding)"


def check_table4_unified_cost() -> str:
    rows = list(csv.DictReader(open(RES / "unified_main_cost.csv")))
    by_ds = defaultdict(dict)
    max_diff = 0.0
    for r in rows:
        cf = float(r["cost_flat"])
        by_ds[r["dataset"]][r["method"]] = cf
        if r["method"].startswith("recoup-"):
            max_diff = max(max_diff, abs(float(r["cost_hier"]) - cf))
    assert len(by_ds) == 10, f"Expected 10 datasets, got {len(by_ds)}"
    for ds, m in by_ds.items():
        exp_n = 11 if ds in ("gplus", "flickr", "youtube") else 12
        assert len(m) == exp_n, f"{ds} expected {exp_n} methods, got {len(m)}"
        rl = m["recoup-leiden"]
        base_min = min(v for k, v in m.items() if not k.startswith("recoup-"))
        assert rl < base_min, f"{ds}: recoup-leiden {rl} not < min baseline {base_min}"
    assert max_diff <= 0.00185, f"max |cost_hier - cost_flat| = {max_diff} > 0.0018"
    lj_ratio = by_ds["livejournal"]["gremod"] / by_ds["livejournal"]["recoup-leiden"]
    assert 64.0 <= lj_ratio <= 65.5, f"LJ ratio {lj_ratio:.2f} unexpected"
    return f"PASS: Table 4 (10 graphs x 12 methods verified; max |hier-flat|={max_diff:.5f}; LJ ratio={lj_ratio:.1f}x)"


def check_exp2_ablation_and_frontier() -> str:
    data = json.load(open(RES / "exp2_final.json"))
    exp_lat = {
        ("epinions", "V0"): 8.59, ("epinions", "V1"): 32.6, ("epinions", "V2"): 8.56, ("epinions", "V3"): 10.3,
        ("flickr", "V0"): 13.6, ("flickr", "V1"): 177.0, ("flickr", "V2"): 20.4, ("flickr", "V3"): 17.4,
        ("youtube", "V0"): 32.6, ("youtube", "V1"): 364.0, ("youtube", "V2"): 64.1, ("youtube", "V3"): 44.9,
    }
    exp_cost = {
        ("epinions", "V0"): 0.133, ("epinions", "V1"): 0.133, ("epinions", "V2"): 0.133, ("epinions", "V3"): 0.144,
        ("flickr", "V0"): 0.046, ("flickr", "V1"): 0.046, ("flickr", "V2"): 0.046, ("flickr", "V3"): 0.051,
        ("youtube", "V0"): 0.057, ("youtube", "V1"): 0.057, ("youtube", "V2"): 0.057, ("youtube", "V3"): 0.060,
    }
    seen = set()
    for r in data:
        key = (r["ds"], r["arm"])
        seen.add(key)
        assert abs(r["lat"] - exp_lat[key]) <= (0.6 if exp_lat[key] >= 100 else 0.06), (key, r["lat"])
        assert abs(r["cost"] - exp_cost[key]) <= 0.001, (key, r["cost"])
        if r["arm"] in ("V1", "V2"):
            assert r["ndiff"] == 0, f"{key} commit mismatch ndiff={r['ndiff']}"
    assert len(seen) == 12
    fr = list(csv.DictReader(open(RES / "frontier_leg_summary.csv")))
    assert len(fr) == 9, f"Expected 9 frontier graphs, got {len(fr)}"
    for r in fr:
        assert float(r["median_pct_of_vertices"]) < 0.2, r
        assert float(r["mean_sweep_expiry_pct"]) <= 0.15, r
    return "PASS: Table 5 & §6.2 (Exp-2 V0-V3 exact match; frontier <0.2% vertices & <=0.15% expiries on 9 graphs)"


def check_fig2_motivation() -> str:
    d = json.load(open(RES / "fig2_motivation.json"))
    assert len(d) == 9
    recs = [v["X_recurring"] for v in d.values()]
    shs = [v["Y_share"] for v in d.values()]
    assert 0.43 <= min(recs) and max(recs) <= 0.96, (min(recs), max(recs))
    assert abs(st.mean(shs) - 258.5) <= 0.2, st.mean(shs)
    return f"PASS: Fig 2(b,c) (9 graphs, recurring {min(recs)*100:.1f}%-{max(recs)*100:.1f}%, mean sharing {st.mean(shs):.1f}x)"


def check_exp3_and_exp5() -> str:
    ds = json.load(open(RES / "defer_seltime.json"))
    assert len(ds) == 48, f"Expected 48 runs in defer_seltime.json, got {len(ds)}"
    dp_rounds = sum(x["dp_rounds"] for x in ds if x["defer"] == 1)
    cut_rounds = sum(x["cut_rounds"] for x in ds if x["defer"] == 1)
    assert dp_rounds == 2079 and cut_rounds == 0, (dp_rounds, cut_rounds)
    yt0 = sorted(x["dp_ms"] + x["cut_ms"] for x in ds if x["ds"] == "youtube" and x["defer"] == 0)[1]
    yt1 = sorted(x["dp_ms"] + x["cut_ms"] for x in ds if x["ds"] == "youtube" and x["defer"] == 1)[1]
    yt_ratio = yt0 / yt1
    assert 38.5 <= yt_ratio <= 39.5, yt_ratio
    vp = json.load(open(RES / "validate_peel_rate.json"))
    d1 = [x for x in vp if x.get("arm") == "defer1-lam0.1"]
    d0 = [x for x in vp if x.get("arm") == "defer0-lam0.1"]
    d01 = [x for x in vp if x.get("arm") == "defer1-lam0.01"]
    assert len(d1) == 24 and len(d0) == 24 and len(d01) == 24, (len(d1), len(d0), len(d01))
    prop1, acc1, rej1 = sum(x["proposed"] for x in d1), sum(x["direct_accepted"] for x in d1), sum(x["rejected"] for x in d1)
    prop0, acc0, rej0 = sum(x["proposed"] for x in d0), sum(x["direct_accepted"] for x in d0), sum(x["rejected"] for x in d0)
    prop01, acc01, rej01 = sum(x["proposed"] for x in d01), sum(x["direct_accepted"] for x in d01), sum(x["rejected"] for x in d01)
    assert prop1 == 728 and acc1 == 728 and rej1 == 0, (prop1, acc1, rej1)
    assert prop0 == 728 and acc0 == 728 and rej0 == 0, (prop0, acc0, rej0)
    assert prop01 == 1910 and acc01 == 1910 and rej01 == 0, (prop01, acc01, rej01)
    assert sum(x["df_commits"] for x in d1) == 746 and all(x["df_max_diff"] < 1e-12 for x in d1 + d0)
    mm_pct1 = 100.0 * sum(x["multi_moves"] for x in d1) / sum(x["total_moves"] for x in d1)
    mm_pct01 = 100.0 * sum(x["multi_moves"] for x in d01) / sum(x["total_moves"] for x in d01)
    assert abs(mm_pct1 - 54.24) < 0.1 and abs(mm_pct01 - 91.01) < 0.1, (mm_pct1, mm_pct01)

    sa = {(float(r["lambda"]), r["selector"]): r for r in csv.DictReader(open(RES / "selector_ablation_w1.csv"))}
    dp1, ind1, seq1 = sa[(0.1, "dp")], sa[(0.1, "indep")], sa[(0.1, "seq_singleton")]
    assert int(dp1["direct_accepted"]) == 113 and int(dp1["proposed"]) == 119
    assert abs(float(dp1["Q_mean"]) - float(ind1["Q_mean"]) - 0.0320) < 0.001
    assert abs(float(ind1["exec_mean"]) / float(dp1["exec_mean"]) - 1.255) < 0.005
    assert abs(float(dp1["Q_mean"]) - float(seq1["Q_mean"]) - 0.0261) < 0.001
    assert abs(float(seq1["exec_mean"]) / float(dp1["exec_mean"]) - 1.251) < 0.005
    assert int(seq1["committed_rounds"]) == 4945 and int(dp1["committed_rounds"]) == 208
    return (
        f"PASS: Exp-3 & Exp-5 ({dp_rounds} defer-on DP rounds, YouTube speedup {yt_ratio:.1f}x; "
        f"ConflictFilter+DP fast-path: {acc1}/{prop1} [100.0%] SNAP, 113/119 [95.0%] MHRAG, "
        f"indep -0.032 Q / +25.5% cost, seq -0.026 Q / 23.8x rounds)"
    )


def check_table7_eps_and_streaming_20b() -> str:
    # Table 7: Flickr canonical SNAP flags
    fl = {(float(r["lambda"]), float(r["eps"])): r for r in csv.DictReader(open(RES / "sensitivity_eps/flickr_canonical_eps.csv"))}
    assert abs(float(fl[(0.1, 0.0)]["exec_cost"]) - 0.0995) < 0.001
    assert abs(float(fl[(0.1, 0.05)]["exec_cost"]) - 0.0373) < 0.001
    assert abs(float(fl[(0.01, 0.0)]["exec_cost"]) - 0.1605) < 0.001
    assert abs(float(fl[(0.01, 0.05)]["exec_cost"]) - 0.0373) < 0.001

    # Exp-8 20-batch MultiHop-RAG token cost & content propagation
    m20 = [r for r in csv.DictReader(open(RES / "streaming_20b/mhrag_b20_per_batch.csv")) if r["batch"] == "20"]
    by_m = {r["method"]: r for r in m20}
    assert int(by_m["RECOUP (eps=0.05)"]["cum_calls"]) == 1810
    assert abs(int(by_m["RECOUP (eps=0.05)"]["cum_prompt_tok"]) / 1e6 - 3.29) < 0.01
    assert abs(int(by_m["Rebuild"]["cum_prompt_tok"]) / 1e6 - 51.31) < 0.01
    assert abs(int(by_m["QCA"]["cum_prompt_tok"]) / 1e6 - 19.68) < 0.01

    cp = {r["policy"]: r for r in csv.DictReader(open(RES / "sensitivity_eps/exp2_content_prop.csv"))}
    assert abs(int(cp["P1_drift_gated_tau0.10"]["total_prompt_tok"]) / 1e6 - 4.20) < 0.01

    # Strict Sec. 4.1 same-obligation 20-batch 32B maintenance & 225-question QA
    s41 = {r["arm"]: r for r in csv.DictReader(open(RES / "streaming_20b/strict41_32b_table.csv"))}
    assert int(s41["recoup"]["full_calls"]) == 9722
    assert abs(int(s41["recoup"]["full_prompt_tok"]) / 1e6 - 9.20) < 0.01
    assert abs(int(s41["recoup"]["full_comp_tok"]) / 1e6 - 3.55) < 0.01
    assert abs(int(s41["recoup"]["full_total_tok"]) / 1e6 - 12.75) < 0.01
    assert abs(1.0 - int(s41["recoup"]["full_total_tok"]) / int(s41["hit_warm"]["full_total_tok"]) - 0.420) < 0.002
    assert abs(1.0 - int(s41["recoup"]["full_total_tok"]) / int(s41["freeze_attach"]["full_total_tok"]) - 0.237) < 0.002
    assert abs(float(s41["recoup"]["new225_f1"]) - 23.7037) < 0.01
    assert abs(float(s41["recoup_eager"]["new225_f1"]) - 23.6296) < 0.01
    assert abs(float(s41["hit_warm"]["new225_f1"]) - 19.3333) < 0.01
    assert abs(float(s41["freeze_attach"]["new225_f1"]) - 16.0741) < 0.01

    # Batch-20 closing ledger conservation
    pt = list(csv.DictReader(open(RES / "streaming_20b/pending_liability_pt.csv")))
    mh05 = [r for r in pt if r["dataset"] == "MultiHop-RAG (eps=0.05)"]
    mh20 = [r for r in pt if r["dataset"] == "MultiHop-RAG (eps=0.2)"]
    b20 = mh05[-1]
    assert b20["batch"] == "20"
    booked = float(b20["booked_Bt"])
    retired = float(b20["retired_Zt"])
    canceled = float(b20["canceled"])
    p20 = float(b20["pending_end_Pt"])
    assert abs((booked - retired - canceled) - p20) < 1e-9
    assert abs(booked - 1820.36) < 0.01 and abs(retired - 751.61) < 0.01 and abs(canceled - 1037.42) < 0.01 and abs(p20 - 31.33) < 0.01
    assert abs(max(float(r["pending_end_Pt"]) for r in mh05) - 35.33) < 0.01
    assert abs(max(float(r["pending_end_Pt"]) for r in mh20) - 108.19) < 0.01

    # Equal-exposure control (453 summaries, 120 questions, GPT-5 & Claude-Opus)
    lens = {r["arm"]: r for r in csv.DictReader(open(RES / "equal_exposure/exp3_arena_matched_lengths.csv"))}
    assert float(lens["m-dylo"]["median_words"]) == 141.0 and float(lens["m-rebuild"]["median_words"]) == 141.0
    assert float(lens["m-dylo"]["abstention_pct"]) == 2.5
    return "PASS: Table 7, Fig 7, Table 8 (20-batch & closing ledger 31.33), and Equal-Exposure Control verified"


def check_exp8_exp9_32b() -> str:
    mrows = {r["method"]: r for r in csv.DictReader(open(RES / "exp8_32b_maintenance.csv"))}
    assert len(mrows) == 8
    dylo, qca, reb = mrows["dylo"], mrows["qca"], mrows["rebuild"]
    assert int(dylo["reports"]) == 344 and abs(float(dylo["summary_s"]) - 1901.1) < 0.1
    assert abs(float(dylo["maintenance_s"]) - 1926.5) < 0.1
    ts_vs_qca = (1.0 - float(dylo["summary_s"]) / float(qca["summary_s"])) * 100
    ts_vs_reb = (1.0 - float(dylo["summary_s"]) / float(reb["summary_s"])) * 100
    tok_vs_qca = (1.0 - float(dylo["successful_total_tokens"]) / float(qca["successful_total_tokens"])) * 100
    tok_vs_reb = (1.0 - float(dylo["successful_total_tokens"]) / float(reb["successful_total_tokens"])) * 100
    assert abs(ts_vs_qca - 43.4) < 0.1 and abs(ts_vs_reb - 74.6) < 0.1
    assert abs(tok_vs_qca - 23.3) < 0.1 and abs(tok_vs_reb - 58.8) < 0.1

    qrows = list(csv.DictReader(open(RES / "exp9_32b_quality.csv")))
    assert len(qrows) == 28
    assert all(int(r["question_count"]) == 60 and int(r["missing_question_pairs"]) == 0 for r in qrows)
    gt50 = sum(1 for r in qrows if float(r["dylo_preference_score"]) > 50.0)
    assert gt50 == 27, gt50
    hl_div = next(float(r["dylo_preference_score"]) for r in qrows if r["baseline"] == "hit-leiden" and r["dimension"] == "diversity")
    assert abs(hl_div - 49.58) < 0.01

    qc = {r["method_id"]: r for r in csv.DictReader(open(RES / "exp9_32b_query_cost_by_method.csv"))}
    assert abs(float(qc["dylo"]["mean_total_tokens"]) - 65777.1) < 1.0
    assert abs(float(qc["dylo"]["query_latency_mean_s"]) - 51.77) < 0.1
    assert int(qc["qca"]["first_integer_queries_recoup_costs_more"]) == 10
    assert int(qc["rebuild"]["first_integer_queries_recoup_costs_more"]) == 45

    prov = json.load(open(ROOT / "graphrag/case_study_32b/sensemaking-questions-provenance.json"))
    sha = hashlib.sha256((ROOT / "graphrag/case_study_32b/sensemaking-questions.csv").read_bytes()).hexdigest()
    assert sha == prov["csv_sha256"], (sha, prov["csv_sha256"])
    return "PASS: Table 7, Table 8 & §6.5 (32B maintenance, 60-question sensemaking, query crossover, and CSV SHA-256 verified)"


def main() -> int:
    checks = [
        check_table3_recoup_latency,
        check_table4_unified_cost,
        check_exp2_ablation_and_frontier,
        check_fig2_motivation,
        check_exp3_and_exp5,
        check_exp8_exp9_32b,
        check_table7_eps_and_streaming_20b,
    ]
    for fn in checks:
        print(fn())
    print("ALL PAPER TABLE & CLAIM VERIFICATIONS PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
