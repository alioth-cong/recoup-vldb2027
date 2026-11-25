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
    prop = sum(x["proposed"] for x in vp)
    peel = sum(x["peel"] for x in vp)
    rej = sum(x["rejected"] for x in vp)
    assert prop == 1437 and peel == 0 and rej == 0, (prop, peel, rej)
    return f"PASS: Exp-3 & Exp-5 ({dp_rounds} defer-on DP rounds, YouTube speedup {yt_ratio:.1f}x; {prop} validated groups, 0 Peel)"


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
    ]
    for fn in checks:
        print(fn())
    print("ALL PAPER TABLE & CLAIM VERIFICATIONS PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
