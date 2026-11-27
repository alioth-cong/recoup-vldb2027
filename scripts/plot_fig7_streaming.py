import sys
#!/usr/bin/env python3
"""Render PVLDB-style single-column 3-panel Figure 7 (20-batch streaming E2E)."""
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_PDF = Path(sys.argv[1]) / "fig7_streaming_e2e.pdf" if len(sys.argv) > 1 else ROOT / "results/figures_regen/fig7_streaming_e2e.pdf"
OUT_PNG = OUT_PDF.with_suffix(".png")
DATA = Path(__file__).resolve().parent.parent / "results/streaming_20b"

LIB = Path(os.environ.get(
    "TEXLIVE_LIBERTINE_DIR",
    "/usr/local/texlive/2024/texmf-dist/fonts/opentype/public/libertine",
))
HAS_LIB = all((LIB / f).exists() for f in (
    "LinBiolinum_R.otf", "LinBiolinum_RB.otf", "LinBiolinum_RI.otf",
    "LinLibertine_R.otf", "LinLibertine_RB.otf",
))
if HAS_LIB:
    for f in ("LinBiolinum_R.otf", "LinBiolinum_RB.otf", "LinBiolinum_RI.otf",
              "LinLibertine_R.otf", "LinLibertine_RB.otf"):
        fm.fontManager.addfont(str(LIB / f))

INK, AXIS, GRID, MUTED = "#1a1a1a", "#3a3a3a", "#e6e6e6", "#6b6b6b"
TICK, LABEL, NOTE, SUBCAP = 6.2, 7.0, 5.8, 7.8
PLOT_FONT = "Linux Biolinum O" if HAS_LIB else "DejaVu Sans"
SUBCAP_FONT = (
    fm.FontProperties(fname=str(LIB / "LinLibertine_RB.otf"), size=SUBCAP)
    if HAS_LIB
    else fm.FontProperties(family="DejaVu Serif", weight="bold", size=SUBCAP)
)
plt.rcParams.update({
    "font.family": PLOT_FONT,
    "mathtext.fontset": "custom" if HAS_LIB else "dejavusans",
    **({"mathtext.rm": "Linux Biolinum O",
        "mathtext.it": "Linux Biolinum O:italic",
        "mathtext.bf": "Linux Biolinum O:bold"} if HAS_LIB else {}),
    "pdf.fonttype": 42,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": AXIS, "ytick.color": AXIS,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "xtick.major.width": 0.55, "ytick.major.width": 0.55,
    "xtick.major.size": 2.4, "ytick.major.size": 2.4,
    "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    "xtick.labelsize": TICK, "ytick.labelsize": TICK,
    "axes.labelsize": LABEL, "axes.labelpad": 1.8,
    "legend.fontsize": NOTE, "legend.frameon": False,
    "lines.solid_capstyle": "round", "lines.dash_capstyle": "round",
})

BLUE, VERM, ORANGE, GREEN, PURPLE, GRAY = (
    "#0072B2", "#D55E00", "#E69F00", "#009E73", "#9467BD", "#7f7f7f"
)


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", which="major", color=GRID, lw=0.45, zorder=0)
    ax.set_axisbelow(True)


def main():
    mhrag = pd.read_csv(DATA / "mhrag_b20_per_batch.csv")
    pt = pd.read_csv(DATA / "pending_liability_pt.csv")
    s41 = pd.read_csv(DATA / "strict41_32b_table.csv").set_index("arm")

    W, H = 3.36, 1.26
    fig = plt.figure(figsize=(W, H))
    w, h, y0 = 0.215, 0.50, 0.34
    xs = (0.092, 0.422, 0.758)
    axa, axb, axc = (fig.add_axes([x, y0, w, h]) for x in xs)

    # (a) 20-batch cumulative prompt tokens
    series_a = [
        ("Rebuild", GRAY, "--", 1.0, "Rebuild"),
        ("DynaMo", ORANGE, "-.", 0.9, "DynaMo"),
        ("GreMod", PURPLE, ":", 0.95, "GreMod"),
        ("QCA", BLUE, "-.", 0.9, "QCA"),
        ("Reuse-aware Rebuild", "#56B4E9", "--", 0.9, "Reuse"),
        ("RECOUP (eps=0.05)", GREEN, "-", 1.45, "RECOUP"),
    ]
    for m, col, ls, lw, lbl in series_a:
        sub = mhrag[mhrag["method"] == m].sort_values("batch")
        axa.plot(sub["batch"], sub["cum_prompt_tok"] / 1e6, color=col, ls=ls, lw=lw, label=lbl, zorder=4 if m.startswith("RECOUP") else 3)
    axa.set(xlim=(1, 20), ylim=(0, 55))
    axa.set_xticks([1, 5, 10, 15, 20])
    axa.set_yticks([0, 20, 40])
    clean(axa)
    axa.set_xlabel("Batch $t$", labelpad=1.5)
    axa.set_ylabel("Cum. tokens (M)", labelpad=1.5)
    axa.annotate("-93.6%", xy=(19.8, 3.8), xytext=(12.2, 8.8),
                 arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.7),
                 fontsize=NOTE - 0.2, fontweight="bold", color=GREEN)
    axa.legend(loc="upper left", handlelength=1.4, labelspacing=0.15, borderpad=0.05, handletextpad=0.35)

    # (b) Liability P_t
    for ds, col, ls, lbl in [
        ("MultiHop-RAG (eps=0.05)", GREEN, "-", r"$\varepsilon{=}0.05$"),
        ("MultiHop-RAG (eps=0.2)", VERM, "--", r"$\varepsilon{=}0.20$"),
    ]:
        sub = pt[pt["dataset"] == ds].sort_values("batch")
        axb.plot(sub["batch"], sub["pending_end_Pt"], color=col, ls=ls, lw=1.2, label=lbl, zorder=4)
    axb.set(xlim=(1, 20), ylim=(0, 118))
    axb.set_xticks([1, 5, 10, 15, 20])
    axb.set_yticks([0, 50, 100])
    clean(axb)
    axb.set_xlabel("Batch $t$", labelpad=1.5)
    axb.set_ylabel("Pending $P_t$", labelpad=1.5)
    axb.legend(loc="upper left", handlelength=1.4, labelspacing=0.2, borderpad=0.05, handletextpad=0.35)
    axb.text(13.0, 40, "35.3", color=GREEN, fontsize=NOTE - 0.2, fontweight="bold")
    axb.text(9.6, 107, "108.2", color=VERM, fontsize=NOTE - 0.2, fontweight="bold")

    # (c) Same-rule 32B QA F1
    x = np.arange(2)
    bw = 0.18
    arms_c = [
        ("freeze_attach", ORANGE, "Freeze"),
        ("hit_warm", BLUE, "HIT"),
        ("recoup_eager", GRAY, "Eager"),
        ("recoup", GREEN, "RECOUP"),
    ]
    for idx, (arm_k, col, lbl) in enumerate(arms_c):
        vals = [float(s41.loc[arm_k, "new225_f1"]), float(s41.loc[arm_k, "all414_f1"])]
        offs = (idx - 1.5) * bw
        axc.bar(x + offs, vals, width=bw, color=col, label=lbl, zorder=4 if arm_k == "recoup" else 3)
    axc.set(ylim=(0, 38))
    axc.set_xticks(x)
    axc.set_xticklabels(["New-225", "All-414"], fontsize=TICK)
    axc.set_yticks([0, 15, 30])
    clean(axc)
    axc.set_xlabel("Question set", labelpad=1.5)
    axc.set_ylabel("QA $F_1$ (%)", labelpad=1.5)
    axc.legend(loc="upper left", ncol=2, columnspacing=0.45, handlelength=0.8, labelspacing=0.12, borderpad=0.05, handletextpad=0.25, fontsize=NOTE - 0.5)

    for ax, sc in ((axa, "(a) 20-batch tokens."), (axb, "(b) Liability $P_t$."), (axc, "(c) Same-rule 32B $F_1$.")):
        pos = ax.get_position()
        fig.text(pos.x0 + pos.width / 2, 0.035, sc, ha="center", va="bottom", fontproperties=SUBCAP_FONT)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT_PNG, dpi=240, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved {OUT_PDF} and {OUT_PNG}")


if __name__ == "__main__":
    main()
