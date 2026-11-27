# RECOUP: Balancing Modularity Gains and Report Refresh Costs in GraphRAG Community Indexes — Artifact (VLDB 2027)

- **Venue:** VLDB 2027 / PVLDB Volume 20 (Regular Research Track)
- **Public Repository URL:** https://github.com/alioth-cong/recoup-vldb2027
- **Archival DOI:** `[Zenodo DOI — archive release tag before final submission]`
- **Primary Contact:** `[Author Name <email@institution.edu>]`

> **Naming note:** Internal driver binaries, environment variables (`DYLO_*`), and early Python symbols retain the project's historical codename `dylo` for CLI compatibility; the exact-certificate C++ implementation (`dylo_std_v2_1` and `dylo_driver_std4`, both using `exact_group_gain_dense` in `corrected_state.hpp`) and curated results correspond 1-to-1 to **RECOUP** in the VLDB 2027 paper.

---

## 1. Artifact Overview & Directory Structure

This repository contains the C++17 RECOUP implementation, benchmark harness, third-party baseline adapters, dataset preparation utilities, verification suites, plotting scripts, GraphRAG 2.7.2 integration scripts, and all canonical numerical results reported in **Tables 2–9** and **Figures 2, 5, 6, 7** (**Exp-1–Exp-9**) of the VLDB 2027 paper.

Every reported claim and table cell can be verified offline directly from `results/` via `scripts/verify_paper_tables.py` in under two seconds, or regenerated from source as described below.

```text
recoup-vldb27/
├── README.md                              # This file
├── LICENSE                                # MIT License
├── Makefile                               # Builds all 10 C++17 drivers + runs verify/smoke/figures
├── MANIFEST.json                          # Sorted SHA-256 manifest of all shipped files
├── requirements.txt                       # Python dependencies
├── .gitignore
├── data/
│   └── datasets.tsv                       # Public URLs & descriptions of the 10 graph benchmarks
├── harness/
│   ├── cpp/
│   │   ├── dylo/                          # RECOUP C++17 headers & drivers (zero external deps)
│   │   │   ├── corrected_state.hpp        # Exact group gain, Ledger, IncrementalDP
│   │   │   ├── detector.hpp, detector_fixed.hpp, dp.hpp, closure.hpp, localized_closure.hpp
│   │   │   ├── driver_std_v2_1.cxx        # Production RECOUP driver (std_v2_1)
│   │   │   ├── driver_std4.cxx            # Exact-certificate driver (+ oracle & no-cert arms)
│   │   │   ├── driver_{std4sel,std4w,seltime,eps,seedrep,b3fix,r2greedy}.cxx
│   │   │   ├── unit_tests.cxx             # K4 counterexample, 10k joint-gain, 6k DP, 12 ledger tests
│   │   │   ├── selector_verify.cxx        # Exp-3(a): enumeration vs. min-cut vs. tree DP
│   │   │   └── hit_leiden_driver_inc.cxx  # Incremental HIT-Leiden C++ baseline
│   │   ├── dylo_driver.cxx                # Wrapper for C++ Louvain dynamic baselines (DF/delta-screen)
│   │   └── dylo_leiden_driver.cxx         # Wrapper for C++ DF-Leiden baseline
│   ├── java/                              # Wrapper & build script for Java baselines (DynaMo/Batch/QCA/GreMod/SLM4J)
│   └── *.py                               # Graph benchmark runners, stream prep, and scoring scripts
├── graphrag/
│   ├── case_study_32b/
│   │   ├── protocol.json                  # Complete 8-arm 32B GraphRAG execution & audit protocol
│   │   ├── sensemaking-questions.csv      # Exact 60 frozen MultiHop-RAG synthesis questions (q0-q59)
│   │   └── sensemaking-questions-provenance.json
│   ├── streaming_20b/                     # 20-batch MultiHop-RAG/HotpotQA, eps-sensitivity & equal-exposure scripts
│   ├── community_report_graph.txt         # Structured 32B community summarization prompt
│   ├── settings.yaml                      # GraphRAG configuration template
│   └── *.py, run_dylo_driver.sh           # GraphRAG 2.7.2 cluster/report patching and token metering
├── tools/
│   ├── download_datasets.sh               # Downloads & normalizes the 10 public graph benchmarks
│   ├── fetch_baselines.sh                 # Clones third-party baselines at pinned commits
│   ├── build_{dynamo,puzzlef,puzzlef_driver,puzzlef_leiden_driver,slm4j}.sh
│   └── install_py.sh                      # Creates local Python venv if needed
├── scripts/
│   ├── verify_paper_tables.py             # Offline consistency verifier for Tables 3-8 & Exp-1..9
│   ├── cert_oracle_test.py                # Exact-certificate oracle test (counterexample + 200 random graphs)
│   ├── smoke_test.sh                      # End-to-end Facebook run checking Q, cost, and oracle=0
│   ├── plot_fig2_motivation.py            # Regenerates Figure 2(b,c) from results/fig2_motivation.json
│   └── plot_fig5_fig6.py                  # Regenerates Figure 5(a,b) and Figure 6(a-c)
└── results/
    ├── unified_main_cost.csv              # Table 4 & Table 6: common leaf-only cost & modularity (10 graphs)
    ├── lat_iso.tsv, lat_final.tsv         # Table 3: isolated RECOUP latencies across 9 graphs
    ├── reclaim_ab.csv, reclaim_ab_livejournal.csv, lj_summary.json  # LiveJournal 200-batch metrics & latency
    ├── latency.csv, quality.csv           # Long-format baseline latency & modularity records
    ├── fig2_motivation.json               # Figure 2(b,c): HIT-Leiden recurring & shared refresh stats
    ├── exp2_final.json, frontier_leg_summary.csv  # Table 5 & §6.2: V0-V3 ablations & frontier size
    ├── openset_grid.json, scalability.csv, defer_seltime.json  # Figure 5 (Exp-3): selector scaling & deferral timing
    ├── validate_peel_rate.json            # Exp-5: 1,082 groups, 1,053 (97.3%) intercepted & Peel-repaired (normalized_initial, defer=1)
    ├── std_v2_1_paired_main.csv           # Exp-1/4: 10-graph x 3-instance std_v2_1 verification
    ├── batchsize_multi.csv, batchsize.csv, lambda.csv  # Figure 6 (Exp-6 & Exp-7)
    ├── sensitivity_eps/                   # Table 7: eps sweep & tau=0.10 content propagation
    ├── streaming_20b/                     # Figure 7 & Table 8: 20-batch MultiHop-RAG/HotpotQA, ledger, 32B QA
    ├── equal_exposure/                    # §6.5 equal-exposure control (453 summaries, GPT-5 & Claude-Opus)
    ├── exp8_32b_{maintenance.{csv,json},phase_times.csv}  # Table 8 (1-batch H100): 32B GraphRAG maintenance
    └── exp9_32b_{quality.{csv,json},query_cost_by_method.csv,query_cost_by_question.csv}  # Table 9 & §6.5
```

---

## 2. Hardware & Software Requirements

### Hardware
- **Graph-stream benchmarks (Exp-1–Exp-7):** Paper measurements ran single-threaded in memory on a dual-socket **AMD EPYC 9654** host with **1.1 TiB RAM** (Ubuntu 22.04 LTS).
  - Graphs other than LiveJournal run comfortably on an **8–16 GB RAM** workstation.
  - Full baseline runs on **LiveJournal** require **>= 64 GB RAM** (RECOUP peaks at **9.5 GB RSS**; baselines reach up to **54.5 GB RSS**).
- **32B GraphRAG case study (Exp-8 & Exp-9):** Single **NVIDIA H100 (80 GB)** GPU serving `Qwen/Qwen2.5-32B-Instruct` via vLLM (and `gpt-oss-120b` for local pairwise judging).

### Software
- **C++ compiler:** `g++ >= 11` with C++17 (`-O2 -std=c++17`). **RECOUP itself has zero external C++ dependencies.**
- **Python:** `Python >= 3.10`. Install packages via `pip install -r requirements.txt` (or `bash tools/install_py.sh`). Core stack: `numpy`, `pandas`, `pyarrow`, `scipy`, `matplotlib`, `networkx`, `python-igraph`, `leidenalg==0.10.2`.
- **Baselines (optional if reproducing external baselines):** OpenJDK 21 + Maven (Java baselines: DynaMo, Batch, QCA, GreMod, SLM4J) and OpenMP (C++ baselines: DF-Louvain, $\Delta$-screening, DF-Leiden).
- **End-to-end GraphRAG (optional for Exp-8 / Exp-9):** `graphrag==2.7.2`, vLLM serving `Qwen/Qwen2.5-32B-Instruct` (revision `5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`, BF16, 4 concurrent requests, response/prefix caching disabled) and `BAAI/bge-small-en-v1.5`.

---

## 3. Installation & Quick Verification (≈ 2 minutes)

```bash
# 1. Compile RECOUP C++17 drivers and baseline helpers (no external dependencies)
make

# 2. Verify all reported paper tables & numbers against shipped results/ (< 2 seconds)
python3 scripts/verify_paper_tables.py

# 3. Verify exact affine drift certificates against an independent oracle (counterexample + 200 random graphs)
python3 scripts/cert_oracle_test.py

# 4. Run end-to-end smoke test on Facebook (downloads Facebook if absent, checks Q, cost, oracle=0)
bash scripts/smoke_test.sh
```

Expected outcomes:
- `verify_paper_tables.py` prints `PASS` for every paper table/claim check and exits `0`.
- `cert_oracle_test.py` reports **0** live-certificate false negatives in exact mode (while catching positive violations when `DYLO_CERT_LEGACY=1` is enabled for contrast).
- `smoke_test.sh` matches `results/unified_main_cost.csv` (`facebook`, `recoup-leiden`) with `mean Q ≈ 0.83593`, `mean cost ≈ 0.06770`, and `oracle viol 0`.

---

## 4. Datasets & Workloads

### Graph Benchmarks (Table 2)

| Dataset | Source | $|V|$ | $|E|$ | Batches ($|\Delta E|=10^3$) | URL |
|---|---|---:|---:|---:|---|
| Facebook (FB) | SNAP | 4,039 | 88,234 | 9 | `https://snap.stanford.edu/data/facebook_combined.txt.gz` |
| GPlus (G+) | SNAP | 107,614 | 12,238,285 | 1,224 | `https://snap.stanford.edu/data/gplus_combined.txt.gz` |
| Twitter (Tw) | SNAP | 81,306 | 1,342,296 | 135 | `https://snap.stanford.edu/data/twitter_combined.txt.gz` |
| Epinions (Ep) | SNAP | 75,879 | 405,740 | 41 | `https://snap.stanford.edu/data/soc-Epinions1.txt.gz` |
| Cit-HepPh (Ph) | SNAP | 34,546 | 420,877 | 43 | `https://snap.stanford.edu/data/cit-HepPh.txt.gz` |
| Cit-HepTh (Th) | SNAP | 27,769 | 352,285 | 36 | `https://snap.stanford.edu/data/cit-HepTh.txt.gz` |
| DBLP | SNAP | 317,080 | 1,049,866 | 105 | `https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz` |
| Flickr (Fl) | NetworkRepository | 513,969 | 3,190,452 | 320 | `https://nrvis.com/download/data/soc/soc-flickr.zip` |
| YouTube (YT) | SNAP | 1,134,890 | 2,987,624 | 299 | `https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz` |
| LiveJournal (LJ) | SNAP | 4,846,609 | 42,851,237 | 4,286 (first 200 evaluated) | `https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz` |

Download, normalize to undirected simple edge lists, and construct the 90%/10% seed-42 dynamic streams:

```bash
bash tools/download_datasets.sh
python3 harness/prep_data.py --name facebook   # writes data/runs/facebook/{initial.txt,inc_*.txt,meta.json}
# Repeat for cit-hepth, cit-hepph, epinions, twitter, dblp, gplus, flickr, youtube, livejournal
```

### MultiHop-RAG News Corpus (Exp-8 & Exp-9)
Documents come from the public **MultiHop-RAG** English news corpus (Tang and Yang, COLM 2024): 304 initial articles plus one 23-article insertion batch (merged graph: 327 documents, 13,264 entities, 12,925 relationships, 833 text units; semantic SHA-256 `cee9752bf403ba8390692cb8928ad5ce0d3037983b3778a86fbf2caca8c196f9`). The exact 60 evaluated synthesis questions (`q0`–`q59`) are bundled in `graphrag/case_study_32b/sensemaking-questions.csv` with SHA-256 provenance.

---

## 5. Paper-to-Artifact Provenance Map

| Paper Item | Content | Binary / Script | Canonical Result File(s) |
|---|---|---|---|
| **Table 2** | 10 graph dataset statistics | `tools/download_datasets.sh`, `harness/prep_data.py` | `data/datasets.tsv` |
| **Figure 2(a)** | Latency vs. structural cost motivation | `harness/run.py`, `harness/recost_reclaim_ab.py` | `results/unified_main_cost.csv`, `results/latency.csv` |
| **Figure 2(b,c)** | HIT-Leiden recurring & shared refreshes (40 batches) | `scripts/plot_fig2_motivation.py` | `results/fig2_motivation.json` |
| **Table 3** (Exp-1) | Graph-maintenance latency across 10 graphs | `dylo_driver_std4`, `harness/run.py`, `harness/lj_run_all.py` | `results/lat_iso.tsv`, `results/lat_final.tsv`, `results/reclaim_ab_livejournal.csv`, `results/latency.csv` |
| **Table 4** (Exp-1) | Common active-vertex leaf-only refresh fraction | `harness/dirty_cost.py`, `harness/recost_reclaim_ab.py`, `harness/recost_all.py` | `results/unified_main_cost.csv` |
| **Table 5 & §6.2** (Exp-2) | Component ablation ($V_0$–$V_3$) & certified frontier stats | `dylo_driver_std4` (`DYLO_NO_CERT=1`, `DYLO_FRONTIER_LOG=1`) | `results/exp2_final.json`, `results/frontier_leg_summary.csv` |
| **Figure 5(a,b)** (Exp-3) | Synthetic tree scaling & native-stream deferral timing | `selector_verify`, `dylo_driver_seltime`, `harness/defer_seltime.py`, `scripts/plot_fig5_fig6.py` | `results/openset_grid.json`, `results/scalability.csv`, `results/defer_seltime.json` |
| **Table 6** (Exp-4) | Stream-mean modularity $\bar Q$ across 10 graphs | `harness/recost_reclaim_ab.py`, `harness/run.py` | `results/unified_main_cost.csv`, `results/quality.csv`, `results/reclaim_ab.csv`, `results/reclaim_ab_livejournal.csv` |
| **§6.3 Exp-5** | Selector fidelity (1,082 groups, 1,053 / 97.3% intercepted & Peel-repaired, 0 $\Delta F$ viol.) | `harness/validate_peel_rate.py`, `dylo_std_v2_1` | `results/validate_peel_rate.json` |
| **§6.5 Exp-8 (same-rule)** | Strict Sec. 4.1 20-batch 32B maintenance + 225-question QA | `graphrag/streaming_20b/strict41_same_obligation_eval.py` | `results/streaming_20b/strict41_32b_table.csv`, `results/streaming_20b/strict41_32b_summary.json` |
| **Figure 6(a)** (Exp-6) | Batch-size vs. batch-closing delay frontier | `harness/exp7_batchsize_multi.py`, `scripts/plot_fig5_fig6.py` | `results/batchsize_multi.csv`, `results/batchsize.csv` |
| **Figure 6(b,c)** (Exp-7) | Exchange-rate $\lambda$ sensitivity (cost & $\bar Q$ loss) | `harness/exp5_lambda.py`, `scripts/plot_fig5_fig6.py` | `results/lambda.csv` |
| **Table 7** (Exp-7 cont.) | Deferral threshold $\varepsilon$ sensitivity & $\tau{=}0.10$ content prop. | `graphrag/streaming_20b/{exp2_eps_and_content,run_flickr_canonical_eps}.py` | `results/sensitivity_eps/` |
| **Figure 7 & Table 8** (Exp-8) | 20-batch MultiHop-RAG/HotpotQA & 1-batch H100 maintenance | `graphrag/streaming_20b/exp1_multibatch.py`, `graphrag/dylo_patch.py` | `results/streaming_20b/`, `results/exp8_32b_*` |
| **Table 9 & §6.5** (Exp-9) | 32B sensemaking, equal-exposure control & query crossover | `graphrag/case_study_32b/`, `graphrag/streaming_20b/exp3_truncation_control.py` | `results/exp9_32b_*`, `results/equal_exposure/` |

---

## 6. Experiment Workflow (Step-by-Step)

All Python scripts resolve paths relative to the repository root (`DYLO_ROOT` overrides the root, `DYLO_SCRATCH` sets temporary output directory default `/tmp/dylo`, `DYLO_MHRAG` points to GraphRAG index storage).

### 6.1 Running RECOUP Directly

```bash
harness/cpp/dylo/dylo_driver_std4 <instance> <initial.txt> <inc.txt> <numBatches> <outDir>
# instance in {louvain, slm, leiden}; append -fs (e.g., leiden-fs) for from-scratch baseline
```

Default paper configuration ($\lambda=0.1$, source deferral with $\varepsilon=0.05$, exact affine drift certificates with direct $k_v$ dependency, singleton reclamation $\sigma=0.2$, tree DP):

```bash
export DYLO_EMIT_HIER=1 DYLO_LAMBDA=1e-1 DYLO_AUDIT_K=8 DYLO_REBUILD_R=1e18 \
       DYLO_SIGMA_ONLY=1 DYLO_SIGMA=0.2 DYLO_FINAL_AUDIT=1 DYLO_TOPDOWN=0 \
       DYLO_HIER_GAMMA=0.1 DYLO_PRUNE=1 DYLO_PRUNE_MODE=leaf \
       DYLO_DEFER=1 DYLO_DEFER_EPS=0.05 DYLO_LAZY_CERT=1 DYLO_INC_TREE=1 \
       DYLO_LAZY_RECLAIM=1 DYLO_RECLAIM=1 DYLO_FAST_RECLAIM=1 DYLO_SEED_LEGACY=1
```

Key switches:
- `DYLO_CERT_ORACLE=1`: Runs an independent exact-gain oracle over all live certificates (`violations=0`).
- `DYLO_CERT_LEGACY=1`: Restores the earlier $0.97$ / $4k_v$-gated heuristic strictly for regression/contrast testing.
- `DYLO_NO_CERT=1`: Disables drift certificates (`V_1` ablation arm; rechecks all certified pairs each batch).
- `DYLO_FRONTIER_LOG=1`: Emits per-batch `frontier.tsv` (unique evaluated vertices and sweep expiries).
- Presence flags (`getenv != nullptr`): `DYLO_RECLAIM`, `DYLO_FAST_RECLAIM`, `DYLO_SIGMA_ONLY`, `DYLO_FINAL_AUDIT`, `DYLO_NO_COM` (unset to disable).
- Value flags (`atoi != 0`): `DYLO_LAZY_CERT`, `DYLO_INC_TREE`, `DYLO_PRUNE`, `DYLO_DEFER`, `DYLO_TREE_DP`, `DYLO_SRC_CHARGE` (`=0` disables).

| Binary | Role |
|---|---|
| `dylo_driver_std4` (`dylo_driver`) | Main RECOUP driver (exact certificates, oracle check, `V_1`, frontier telemetry) |
| `dylo_driver_seltime` | Per-round selection timer (`dp_rounds`/`dp_ms` vs. `cut_rounds`/`cut_ms`, Exp-3(b)) |
| `dylo_driver_eps` | Per-batch deferral and drift logs (`defer.tsv`, `drift.tsv`) |
| `dylo_driver_seedrep` | Parameterized base detector seed via `DYLO_SEED` |
| `dylo_driver_{std4sel,r2greedy,b3fix,std4w}` | Selector, greedy, marginal-gamma, and weighted ablation variants |
| `selector_verify` | Exp-3(a): synthetic tree benchmark (exhaustive enumeration vs. min-cut vs. tree DP) |
| `hit_leiden_driver_inc` | Incremental HIT-Leiden C++ baseline |

### 6.2 Building & Running External Baselines

```bash
bash tools/fetch_baselines.sh          # clones pinned upstream commits into external/
bash tools/build_dynamo.sh             # builds Java DynaMo / Batch / QCA / GreMod
bash tools/build_slm4j.sh && bash harness/java/build.sh
bash tools/build_puzzlef.sh && bash tools/build_puzzlef_driver.sh && bash tools/build_puzzlef_leiden_driver.sh

python3 harness/run.py --method <m> --dataset <name>
# m in {leiden-static, louvain-static, slm-static, leiden-hier, dynamo, batch, qca, gremod, df-louvain, ds-louvain, df-leiden}
```

### 6.3 Reproducing Main Experiments & Regenerating Plots

```bash
# Exp-1 & Exp-4 (RECOUP across Louvain/SLM/Leiden + common leaf-only scoring)
python3 harness/recost_reclaim_ab.py --datasets facebook,cit-hepth,cit-hepph,epinions,twitter,dblp,flickr,youtube --instances louvain,slm,leiden --variants recl --defer 1

# Exp-3(b) (selector runtime across 8 native streams), Exp-5 (Validate/Peel fidelity), Exp-6 (batch size), Exp-7 (lambda sweep)
python3 harness/defer_seltime.py
python3 harness/validate_peel_rate.py
python3 harness/exp7_batchsize_multi.py
python3 harness/exp5_lambda.py --instances louvain,slm,leiden --datasets facebook,cit-hepth,cit-hepph,epinions,dblp,flickr

# Regenerate Figure 2(b,c), Figure 5(a,b), and Figure 6(a-c) into results/figures_regen/
make figures
```

### 6.4 32B GraphRAG Case Study (Exp-8 & Exp-9)

`graphrag/dylo_patch.py` replaces GraphRAG 2.7.2's `update_communities` and `update_community_reports` stages: `graphrag/dylo_cluster.py` injects each incremental partition (with `DYLO_DEFER=0` eager source refresh and $\lambda=0.1$ for RECOUP), reuses stored reports whose entity-membership set is unchanged (materializing only internal-edge communities), and meters input/output tokens via `graphrag/cost_meter.py`. Exact frozen configuration, SHA-256 checksums, 60 questions, and 32B audit tables are in `graphrag/case_study_32b/protocol.json` and `results/exp{8,9}_32b_*`.

---

## 7. Notes on Scope & Methodology

1. **Exact certificates:** All RECOUP binaries default to the sound affine drift certificates with direct $k_v$ dependency described in Section 5 of the paper (`cert_oracle_test.py` confirms 0 violations across the counterexample and 200 dynamic test graphs).
2. **1-batch vs. 20-batch GraphRAG protocols:** Graph-stream experiments (Exp-1–Exp-7) and the **20-batch MultiHop-RAG/HotpotQA streaming evaluation** (Exp-8, `DYLO_DEFER=1`, $\varepsilon=0.05$) enable source deferral, evaluate $\tau=0.10$ drift-gated content propagation, and evaluate 32B QA via top-6 leaf retrieval (`~2.35k` tokens/query). The **1-batch H100 case study** (Exp-8/Exp-9, $304{\to}327$) replays the archived `DYLO_DEFER=0` member-set-reuse run, where unaligned root-cut global search (`151` vs. `67` roots) breaks even at `10`/`45` global queries against QCA/Rebuild (`results/exp9_32b_query_cost_by_method.csv`); the **equal-exposure control** (`results/equal_exposure/`) aligns all arms at `453` level-$\le 1$ summaries.
3. **LiveJournal scope:** Per Section 6.1, LiveJournal (42.85M edges) is evaluated over its first 200 update batches (single run); full 4,286-batch from-scratch rebuild is infeasible within time limits.
4. **Third-party code:** External baseline implementations are not redistributed inside this repository; `tools/fetch_baselines.sh` retrieves their exact commits under their original licenses.
5. **Integrity:** Every shipped file and its SHA-256 digest are listed in `MANIFEST.json`.

---

## 8. License & Citation

This artifact is released under the **MIT License** (see `LICENSE`).

```bibtex
@article{recoup2027vldb,
  title   = {{RECOUP}: Balancing Modularity Gains and Report Refresh Costs in GraphRAG Community Indexes},
  journal = {Proceedings of the VLDB Endowment (PVLDB)},
  volume  = {20},
  number  = {1},
  year    = {2027},
  note    = {Artifact repository; replace with camera-ready author list and DOI}
}
```
