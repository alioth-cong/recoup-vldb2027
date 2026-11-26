#!/usr/bin/env python3
"""Run canonical SNAP env epsilon sweep on Flickr (320 batches) for lambda in {0.1, 0.01}."""
import csv
import os
import shutil
import statistics
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = ""

BIN_EPS = Path(__file__).resolve().parents[2] / "harness/cpp/dylo/dylo_driver_eps"
FLICKR_DIR = Path("${DYLO_DATA_RUNS:-data/runs}/flickr")
OUT_BASE = Path("${DYLO_20B_OUT:-results/streaming_20b}")


def run_one(task):
    lam, eps = task
    eps_tag = f"{eps:g}".replace(".", "p").replace("-", "m")
    out = OUT_BASE / "flickr_canon_runs" / f"lam{lam:g}_eps{eps_tag}"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("DYLO_")}
    env.update(
        DYLO_EMIT_HIER="1", DYLO_LAMBDA=str(lam), DYLO_AUDIT_K="8", DYLO_REBUILD_R="1e18",
        DYLO_SIGMA_ONLY="1", DYLO_SIGMA="0.2", DYLO_FINAL_AUDIT="1", DYLO_TOPDOWN="0",
        DYLO_HIER_GAMMA="0.1", DYLO_PRUNE="1", DYLO_PRUNE_MODE="leaf",
        DYLO_DEFER="1", DYLO_DEFER_EPS=str(eps), DYLO_LAZY_CERT="1", DYLO_INC_TREE="1",
        DYLO_LAZY_RECLAIM="1", DYLO_RECLAIM="1", DYLO_FAST_RECLAIM="1",
        DYLO_NO_COM="1", DYLO_SEED_LEGACY="1", OMP_NUM_THREADS="1",
    )
    subprocess.run(
        [str(BIN_EPS), "leiden", str(FLICKR_DIR / "initial.txt"), str(FLICKR_DIR / "inc.txt"), "320", str(out)],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=600,
    )
    with (out / "metrics.tsv").open() as f:
        mrows = list(csv.DictReader(f, delimiter="\t"))[1:]
    with (out / "resum.tsv").open() as f:
        rrows = list(csv.DictReader(f, delimiter="\t"))
    with (out / "defer.tsv").open() as f:
        drows = list(csv.DictReader(f, delimiter="\t"))
    qs = [float(r["modularity"]) for r in mrows]
    costs = [float(r["cost_norm"]) for r in rrows]
    kts = [int(r["K_T"]) for r in drows]
    shutil.rmtree(out, ignore_errors=True)
    return {
        "dataset": "flickr", "lambda": lam, "eps": eps,
        "mean_Q": statistics.mean(qs), "exec_cost": statistics.mean(costs),
        "K_max": max(kts), "flush_quality_cum": int(drows[-1]["flush_quality_cum"]),
        "flush_vanished_cum": int(drows[-1]["flush_vanished_cum"]),
    }


def main():
    tasks = [(lam, eps) for lam in (0.1, 0.01) for eps in (0.0, 0.01, 0.05, 0.1, 0.2)]
    rows = []
    with ProcessPoolExecutor(max_workers=10) as ex:
        futs = [ex.submit(run_one, t) for t in tasks]
        for fut in as_completed(futs):
            rows.append(fut.result())
    df = pd.DataFrame(rows).sort_values(["lambda", "eps"])
    df.to_csv(OUT_BASE / "flickr_canonical_eps.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
