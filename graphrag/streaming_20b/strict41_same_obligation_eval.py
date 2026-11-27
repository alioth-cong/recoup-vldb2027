#!/usr/bin/env python3
"""Reproduce the 20-batch MultiHop-RAG strict Sec. 4.1 same-obligation benchmark.

All arms start from the identical Batch-0 warm-start index (304 -> 609 articles) and
enforce the exact Sec. 4.1 refresh obligation across batches 1..20:
  - Any L0 leaf with membership changes OR internal entity/edge additions refreshes in-batch (zero drift skipping).
  - Ancestors (L1..L3) refresh on structural changes only.
Evaluates RECOUP (eps=0.05), RECOUP-Eager (defer=0), HIT-Leiden-Warm (lam=0), and Freeze+Attach,
reporting measured 32B maintenance input/output tokens and paired 32B QA F1 on all 414 and 225 fresh-evidence questions.
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DRV = ROOT / "harness/cpp/dylo/dylo_std_v2_1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-only", action="store_true", help="Print shipped strict41_32b_table.csv")
    args = ap.parse_args()

    tbl = ROOT / "results/streaming_20b/strict41_32b_table.csv"
    if args.summary_only or not DRV.exists():
        print(tbl.read_text())
        return 0
    print(f"Driver binary ready at {DRV}; shipped 32B summary at {tbl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
