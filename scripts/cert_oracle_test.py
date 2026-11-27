#!/usr/bin/env python3
"""Exact certificate oracle test on the C++ driver (dylo_driver_std4).

Runs DYLO_CERT_ORACLE=1 on:
  (a) the exact counterexample graph scaled to integer multi-edges; and
  (b) 200 randomized small dynamic graphs mixing near-zero margins, hub bursts,
      third-community insertions, remote m drift, chained migration, and arrivals.

Asserts 0 live-certificate violations across all oracle checks.
"""
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(os.environ.get("DYLO_ROOT", Path(__file__).resolve().parent.parent))
DRV = ROOT / "harness/cpp/dylo/dylo_driver_std4"
DRV_V2 = ROOT / "harness/cpp/dylo/dylo_std_v2_1"


def ensure_binary() -> None:
    if not DRV.exists():
        subprocess.run(["make", "-s", "harness/cpp/dylo/dylo_driver_std4"], cwd=ROOT, check=True)
    if not DRV_V2.exists():
        subprocess.run(["make", "-s", "harness/cpp/dylo/dylo_std_v2_1"], cwd=ROOT, check=True)


def run_driver(case_dir: Path, nbt: int, inst: str, env_vars: dict[str, str], drv: Path = DRV) -> tuple[int, int]:
    out = case_dir / f"out_{drv.name}"
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, **env_vars, DYLO_CERT_ORACLE="1", DYLO_NO_COM="1")
    p = subprocess.run(
        [str(drv), inst, str(case_dir / "initial.txt"), str(case_dir / "inc.txt"), str(nbt), str(out)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
    )
    if p.returncode != 0:
        raise RuntimeError(f"driver failed: {p.stderr[-400:]}")
    m = re.search(r"violations=(\d+)", p.stderr)
    mc = re.search(r"checks=(\d+)", p.stderr)
    if not m or not mc:
        raise RuntimeError(f"missing [certfix] oracle line: {p.stderr[-400:]}")
    return int(m.group(1)), int(mc.group(1))


def run_counterexample(work: Path, drv: Path = DRV) -> tuple[int, int]:
    """Integer-scaled multi-edge realization of the counterexample in the paper."""
    d = work / f"counterexample_{drv.name}"
    d.mkdir(parents=True, exist_ok=True)
    init_edges = [(0, 1)] * 60 + [(0, 3)] * 30 + [(0, 5)] * 110 + [(1, 2)] * 470 + [(3, 4)] * 195 + [(5, 6)] * 1135
    with open(d / "initial.txt", "w") as f:
        for u, v in init_edges:
            f.write(f"{u}\t{v}\t0\n")
    with open(d / "inc_1.txt", "w") as f:
        for _ in range(8):
            f.write("0\t5\t1\n")
    env = dict(
        DYLO_EMIT_HIER="1", DYLO_LAMBDA="1e-1", DYLO_AUDIT_K="8", DYLO_REBUILD_R="1e18",
        DYLO_SIGMA_ONLY="1", DYLO_SIGMA="0.2", DYLO_FINAL_AUDIT="1", DYLO_TOPDOWN="0",
        DYLO_HIER_GAMMA="0.1", DYLO_PRUNE="1", DYLO_PRUNE_MODE="leaf", DYLO_DEFER="0",
        DYLO_LAZY_CERT="1", DYLO_INC_TREE="1", DYLO_LAZY_RECLAIM="1",
    )
    ve, ce = run_driver(d, 1, "leiden", env, drv)
    shutil.rmtree(d, ignore_errors=True)
    return ve, ce


def run_random_case(args: tuple[int, Path]) -> tuple[int, int]:
    seed, work = args
    r = random.Random(seed)
    d = work / f"case_{seed}"
    d.mkdir(parents=True, exist_ok=True)
    nb = r.randint(2, 6)
    n = r.randint(20, 120)
    blk = [i % nb for i in range(n)]
    pin, pout = r.uniform(0.15, 0.6), r.uniform(0.005, 0.08)
    edges = [(u, v) for u in range(n) for v in range(u + 1, n)
             if r.random() < (pin if blk[u] == blk[v] else pout)]
    r.shuffle(edges)
    nbt = r.randint(3, 10)
    per = r.randint(4, 32)
    nxt = n
    batches = []
    for _ in range(nbt):
        es = []
        for _ in range(per):
            k = r.random()
            if k < 0.20:
                h = r.randrange(n)
                es += [(h, r.randrange(n)) for _ in range(r.randint(2, 6))]
            elif k < 0.40:
                u = r.randrange(n)
                others = [x for x in range(n) if blk[x] != blk[u]] or [0]
                es.append((u, r.choice(others)))
            elif k < 0.55:
                u = r.randrange(n)
                same = [x for x in range(n) if blk[x] == blk[u]]
                es.append((u, r.choice(same)))
            elif k < 0.75:
                u = r.randrange(n)
                t = r.randrange(nb)
                tgt = [x for x in range(n) if blk[x] == t]
                es += [(u, x) for x in r.sample(tgt, min(3, len(tgt)))]
            elif k < 0.90:
                t = r.randrange(nb)
                tgt = [x for x in range(n) if blk[x] == t]
                es += [(nxt, x) for x in r.sample(tgt, min(len(tgt), r.randint(1, 4)))]
                nxt += 1
            else:
                es.append((r.randrange(nxt), r.randrange(nxt)))
        batches.append([(u, v) for u, v in es if u != v])

    with open(d / "initial.txt", "w") as f:
        for u, v in edges:
            f.write(f"{u}\t{v}\t0\n")
    for b, es in enumerate(batches, 1):
        with open(d / f"inc_{b}.txt", "w") as f:
            for u, v in es:
                f.write(f"{u}\t{v}\t{b}\n")

    env = dict(
        DYLO_EMIT_HIER="1",
        DYLO_LAMBDA=r.choice(["0", "1e-3", "1e-2", "1e-1", "1"]),
        DYLO_AUDIT_K=str(r.choice([2, 4, 8])),
        DYLO_REBUILD_R="1e18",
        DYLO_SIGMA_ONLY="1",
        DYLO_SIGMA=r.choice(["0.05", "0.2", "0.5"]),
        DYLO_FINAL_AUDIT="1",
        DYLO_TOPDOWN="0",
        DYLO_HIER_GAMMA="0.1",
        DYLO_PRUNE=r.choice(["0", "1"]),
        DYLO_PRUNE_MODE="leaf",
        DYLO_DEFER="0",
        DYLO_LAZY_CERT="1",
        DYLO_INC_TREE="1",
        DYLO_LAZY_RECLAIM="1",
    )
    if r.random() < 0.7:
        env.update(DYLO_RECLAIM="1", DYLO_FAST_RECLAIM="1")
    inst = r.choice(["leiden", "louvain", "slm"])

    ve, ce = run_driver(d, nbt, inst, env, DRV)
    ve2, ce2 = run_driver(d, nbt, inst, {**env, "DYLO_AUDIT_DF": "1"}, DRV_V2)
    shutil.rmtree(d, ignore_errors=True)
    return ve + ve2, ce + ce2


def run_v2_cold_and_scaling_checks(work: Path) -> None:
    """Verify dylo_std_v2_1 catches disconnected m-drift at lam=0 and is x32 scale-invariant."""
    # 1) Cold-start 14-node 2-component m-drift test (lam=0)
    dc = work / "v2_cold"
    dc.mkdir(parents=True, exist_ok=True)
    c_init = [
        (0,1),(0,2),(1,2),(1,3),(2,3),(2,4),(3,4),(3,5),(4,5),(4,6),(5,6),(0,6),(1,5),
        (7,8),(7,9),(8,9),(8,10),(9,10),(9,11),(10,11),(10,12),(11,12),(11,13),(12,13),(7,13),
    ]
    with open(dc / "initial.txt", "w") as f:
        for u, v in c_init:
            f.write(f"{u}\t{v}\t0\n")
    with open(dc / "inc_1.txt", "w") as f:
        f.write("7\t12\t1\n")
    base_env = dict(
        DYLO_EMIT_HIER="1", DYLO_LAMBDA="0", DYLO_AUDIT_K="8", DYLO_REBUILD_R="1e18",
        DYLO_SIGMA_ONLY="1", DYLO_SIGMA="0.2", DYLO_FINAL_AUDIT="1", DYLO_TOPDOWN="0",
        DYLO_HIER_GAMMA="0.1", DYLO_PRUNE="1", DYLO_PRUNE_MODE="leaf", DYLO_DEFER="0",
        DYLO_LAZY_CERT="1", DYLO_INC_TREE="1", DYLO_LAZY_RECLAIM="1", DYLO_NO_COM="1",
        DYLO_AUDIT_DF="1",
    )
    o_cert = dc / "out_cert"
    o_nocert = dc / "out_nocert"
    o_cert.mkdir()
    o_nocert.mkdir()
    subprocess.run([str(DRV_V2), "leiden", str(dc / "initial.txt"), str(dc / "inc.txt"), "1", str(o_cert)],
                   env=dict(os.environ, **base_env), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
    subprocess.run([str(DRV_V2), "leiden", str(dc / "initial.txt"), str(dc / "inc.txt"), "1", str(o_nocert)],
                   env=dict(os.environ, **base_env, DYLO_NO_CERT="1"), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
    q_cert = float((o_cert / "metrics.tsv").read_text().splitlines()[-1].split("\t")[2])
    q_nocert = float((o_nocert / "metrics.tsv").read_text().splitlines()[-1].split("\t")[2])
    assert abs(q_cert - q_nocert) < 1e-9, f"cold-start m-drift mismatch: {q_cert} vs {q_nocert}"

    # 2) x32 parallel-edge scaling invariance under default normalized_initial (lam=0.1)
    init_e = [(0,1),(0,2),(1,2),(2,3),(3,4),(3,5),(4,5)]
    b1_e, b2_e = [(0,3),(1,4)], [(2,5)]
    qs = []
    traces = []
    for mult in (1, 32):
        ds = work / f"v2_scale_{mult}"
        ds.mkdir(parents=True, exist_ok=True)
        with open(ds / "initial.txt", "w") as f:
            for u, v in init_e:
                for _ in range(mult):
                    f.write(f"{u}\t{v}\t0\n")
        for bi, es in ((1, b1_e), (2, b2_e)):
            with open(ds / f"inc_{bi}.txt", "w") as f:
                for u, v in es:
                    for _ in range(mult):
                        f.write(f"{u}\t{v}\t{bi}\n")
        os_out = ds / "out"
        os_out.mkdir()
        subprocess.run([str(DRV_V2), "leiden", str(ds / "initial.txt"), str(ds / "inc.txt"), "2", str(os_out)],
                       env=dict(os.environ, **{k: v for k, v in base_env.items() if k != "DYLO_LAMBDA"}, DYLO_LAMBDA="0.1"),
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        traces.append((os_out / "commit_trace.tsv").read_text())
        qs.append(float((os_out / "metrics.tsv").read_text().splitlines()[-1].split("\t")[2]))
    assert traces[0] == traces[1] and abs(qs[0] - qs[1]) < 1e-9, (qs, traces)


def main() -> None:
    ensure_binary()
    n_cases = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    workers = min(16, os.cpu_count() or 4)
    scratch = os.environ.get("DYLO_SCRATCH")
    if scratch:
        Path(scratch).mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="recoup_oracle_", dir=scratch if scratch else None))
    try:
        ce_ve, ce_ce = run_counterexample(work, DRV)
        ce_ve2, ce_ce2 = run_counterexample(work, DRV_V2)
        run_v2_cold_and_scaling_checks(work)
        print(f"Counterexample (scaled multi-edge): std4_viol={ce_ve} ({ce_ce}), v2_1_viol={ce_ve2} ({ce_ce2}); v2_1 cold-start & x32 scaling: PASSED")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(run_random_case, [(s, work) for s in range(1000, 1000 + n_cases)]))
        exact_viol = sum(r[0] for r in results)
        exact_checks = sum(r[1] for r in results)
        print(f"Random small graphs ({n_cases} cases, std4+std_v2_1, {exact_checks} oracle checks): exact_viol={exact_viol}")
        ok = (ce_ve == 0) and (ce_ve2 == 0) and (exact_viol == 0) and (ce_ce + ce_ce2 + exact_checks > 0)
        print("CERT ORACLE TEST", "PASSED" if ok else "FAILED")
        sys.exit(0 if ok else 1)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
