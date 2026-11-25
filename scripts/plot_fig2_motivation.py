#!/usr/bin/env python3
"""Regenerate Figure 2(b) recurring refreshes and Figure 2(c) shared refreshes."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/figures_regen"
OUT.mkdir(parents=True, exist_ok=True)

DATA = json.loads((ROOT / "results/fig2_motivation.json").read_text())
ORDER = ["facebook", "cit-hepth", "cit-hepph", "epinions", "dblp",
         "twitter", "flickr", "youtube", "gplus"]
NICE = {"facebook": "Facebook", "cit-hepth": "Cit-HepTh",
        "cit-hepph": "Cit-HepPh", "epinions": "Epinions", "dblp": "DBLP",
        "twitter": "Twitter", "flickr": "Flickr", "youtube": "YouTube",
        "gplus": "GPlus"}
ORANGE, DGREY = "#e8871a", "#8c8c8c"
W, H = 1.3, 1.0

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 6,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def panel(values, ylabel, bar_label, line_label, name, log):
    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(left=0.27, right=0.98, bottom=0.37, top=0.86)
    x = range(len(values))
    ax.bar(x, values, width=0.62, color=DGREY, edgecolor="white",
           linewidth=0.3, label=bar_label, zorder=2)
    ax.axhline(1.0, color=ORANGE, lw=1.1, ls="--", label=line_label, zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels([NICE[k] for k in ORDER], rotation=60, ha="right",
                       rotation_mode="anchor", fontsize=5)
    ax.set_xlim(-0.6, len(values) - 0.4)
    ax.set_ylabel(ylabel, labelpad=1.5, fontsize=6)
    ax.tick_params(axis="both", length=2, pad=1.2)
    ax.tick_params(axis="y", labelsize=5.5)
    ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.45,
            linestyle=(0, (3, 2)), zorder=0)
    if log:
        ax.set_yscale("log")
        ax.set_ylim(0.7, max(values) * 2.5)
        ax.yaxis.set_major_locator(mt.FixedLocator([1, 10, 100, 1000]))
        ax.yaxis.set_major_formatter(mt.LogFormatterMathtext())
        ax.yaxis.set_minor_locator(mt.NullLocator())
    else:
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0, 0.5, 1.0])
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2,
              frameon=False, fontsize=5, handlelength=1.3,
              handletextpad=0.3, columnspacing=0.8, borderaxespad=0.1)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    panel([DATA[k]["X_recurring"] for k in ORDER], "recurring share",
          "recurring", "all recur", "fig_span", log=False)
    panel([DATA[k]["Y_share"] for k in ORDER], "per-move / joint",
          "per-move", "no sharing", "fig_share", log=True)
    print("ok:", sorted(["fig_span.pdf", "fig_share.pdf"]))
