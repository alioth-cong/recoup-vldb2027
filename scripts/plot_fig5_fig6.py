#!/usr/bin/env python3
'''Regenerate Figure 5 (Exp-3: selector scaling & deferral timing) and Figure 6 (Exp-6/7).'''
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/figures_regen"
OUT.mkdir(parents=True, exist_ok=True)

LIB = Path(os.environ.get(
    "TEXLIVE_LIBERTINE_DIR",
    "/usr/local/texlive/2024/texmf-dist/fonts/opentype/public/libertine",
))
HAS_LIBERTINE = all((LIB / f).exists() for f in (
    "LinBiolinum_R.otf", "LinBiolinum_RB.otf", "LinBiolinum_RI.otf",
    "LinLibertine_R.otf", "LinLibertine_RB.otf",
))
if HAS_LIBERTINE:
    for f in ("LinBiolinum_R.otf", "LinBiolinum_RB.otf", "LinBiolinum_RI.otf",
              "LinLibertine_R.otf", "LinLibertine_RB.otf"):
        fm.fontManager.addfont(str(LIB / f))

INK, AXIS, GRID, MUTED = "#1a1a1a", "#3a3a3a", "#e6e6e6", "#6b6b6b"
TICK, LABEL, NOTE, SUBCAP = 6.5, 7.2, 6.3, 7.97
PLOT_FONT = "Linux Biolinum O" if HAS_LIBERTINE else "DejaVu Sans"
SUBCAP_FONT = (
    fm.FontProperties(fname=str(LIB / "LinLibertine_RB.otf"), size=SUBCAP)
    if HAS_LIBERTINE
    else fm.FontProperties(family="DejaVu Serif", weight="bold", size=SUBCAP)
)
plt.rcParams.update({
    "font.family": PLOT_FONT,
    "mathtext.fontset": "custom" if HAS_LIBERTINE else "dejavusans",
    **({"mathtext.rm": "Linux Biolinum O",
        "mathtext.it": "Linux Biolinum O:italic",
        "mathtext.bf": "Linux Biolinum O:bold"} if HAS_LIBERTINE else {}),
    "pdf.fonttype": 42,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": AXIS, "ytick.color": AXIS,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.45, "ytick.minor.width": 0.45,
    "xtick.major.size": 2.6, "ytick.major.size": 2.6,
    "xtick.minor.size": 1.4, "ytick.minor.size": 1.4,
    "xtick.major.pad": 1.6, "ytick.major.pad": 1.6,
    "xtick.labelsize": TICK, "ytick.labelsize": TICK,
    "axes.labelsize": LABEL, "axes.labelpad": 2.0,
    "legend.fontsize": NOTE, "legend.frameon": False,
    "lines.solid_capstyle": "round", "lines.dash_capstyle": "round",
})

N_SMALL = [8, 12, 16, 20, 22]
ENUM_T = [0.0021, 0.0242, 0.1927, 3.9625, 13.681]
CUT_S = [0.0007, 0.0010, 0.0012, 0.0018, 0.0021]
DP_S = [0.0004, 0.0004, 0.0005, 0.0009, 0.0011]
N_LARGE = [50, 200, 1000, 5000, 20000]
CUT_L = [0.0076, 0.0316, 0.1816, 1.0172, 5.6968]
DP_L = [0.0027, 0.0108, 0.0500, 0.2873, 1.2407]

GRAPHS = ["Epinions", "DBLP", "Flickr", "YouTube"]
SEL_OFF = [2.07, 0.71, 19.0, 155.4]
SEL_ON = [0.19, 0.15, 1.90, 3.99]

BATCH = [1, 10, 100, 1000]
DELAY = [0.083, 0.828, 8.278, 82.781]
MH_EAGER = [113.5442, 49.2932, 26.4956, 7.7388]
MH_LAZY = [115.5214, 45.7760, 25.2219, 7.7208]
PH_EAGER = [58.3974, 57.8606, 53.2983, 26.3030]
PH_LAZY = [33.4460, 33.3073, 31.4761, 21.0177]

LAMBDAS = [1e-3, 1e-2, 1e-1, 0.3, 1, 3, 10]
COST = {
    "Cit-HepPh": [0.1152, 0.1125, 0.1125, 0.1125, 0.1125, 0.1125, 0.1125],
    "Cit-HepTh": [0.2545, 0.1167, 0.1167, 0.1167, 0.1167, 0.1167, 0.1167],
    "DBLP": [0.1490, 0.1375, 0.1375, 0.1375, 0.1375, 0.1375, 0.1375],
    "Epinions": [0.8043, 0.1101, 0.1095, 0.1094, 0.1094, 0.1094, 0.1094],
    "Facebook": [0.8164, 0.4019, 0.3974, 0.3974, 0.3974, 0.3974, 0.3974],
    "Flickr": [0.1363, 0.1303, 0.1301, 0.1301, 0.1301, 0.1301, 0.1301],
}
QLOSS = {
    "Cit-HepPh": [0.09, 0.09, 0.09, 0.09, 0.10, 0.10, 0.10],
    "Cit-HepTh": [0.09, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14],
    "DBLP": [0.18, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19],
    "Epinions": [0.08, 0.67, 0.69, 0.70, 0.70, 0.70, 0.70],
    "Facebook": [0.00, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06],
    "Flickr": [0.07, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08],
}

BLUE, VERM, ORANGE, GREEN, SKY, PURPLE = (
    "#0072B2", "#D55E00", "#E69F00", "#009E73", "#56B4E9", "#9467BD")
SERIES7 = [
    ("Cit-HepPh", BLUE, "o"),
    ("Cit-HepTh", ORANGE, "s"),
    ("DBLP", GREEN, "^"),
    ("Epinions", VERM, "D"),
    ("Facebook", PURPLE, "v"),
    ("Flickr", SKY, "X"),
]


def clean(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid, which="major", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def log_minor(axis):
    axis.set_minor_locator(mt.LogLocator(base=10, subs=range(2, 10), numticks=30))
    axis.set_minor_formatter(mt.NullFormatter())


def fig5a(ax):
    n = N_SMALL + N_LARGE
    ax.plot(N_SMALL, ENUM_T, color=ORANGE, marker="^", ms=3.6, lw=1.2, mec="white", mew=0.5, zorder=4)
    ax.plot(n, CUT_S + CUT_L, color=VERM, marker="s", ms=3.0, lw=1.2, mec="white", mew=0.5, zorder=3)
    ax.plot(n, DP_S + DP_L, color=BLUE, marker="o", ms=3.3, lw=1.5, mec="white", mew=0.5, zorder=5)
    ax.set(xscale="log", yscale="log", xlim=(5.5, 7e4), ylim=(2e-4, 90))
    ax.xaxis.set_major_locator(mt.FixedLocator([10, 1e2, 1e3, 1e4]))
    ax.yaxis.set_major_locator(mt.FixedLocator([1e-3, 1e-2, 1e-1, 1, 10]))
    log_minor(ax.xaxis); log_minor(ax.yaxis)
    clean(ax, grid="both")
    ax.set_xlabel(r"Tree size $n$")
    ax.set_ylabel("Time per solve (ms)")
    ax.text(7.0, 30, "Enumeration", color="#B07800", fontsize=NOTE, va="bottom")
    ax.text(900, 0.36, "Min-cut", color=VERM, fontsize=NOTE, ha="right", va="bottom")
    ax.text(1500, 0.016, "Tree DP", color=BLUE, fontsize=NOTE, fontweight="bold", ha="left", va="top")
    x0 = 30
    ax.annotate("", xy=(x0, DP_S[-1]), xytext=(x0, ENUM_T[-1]),
                arrowprops=dict(arrowstyle="<|-|>", color="#B07800", lw=0.6,
                                mutation_scale=4.5, shrinkA=0, shrinkB=0))
    ax.plot([22, x0], [ENUM_T[-1]] * 2, color="#B07800", lw=0.4, ls=(0, (1, 1)))
    ax.plot([22, x0], [DP_S[-1]] * 2, color="#B07800", lw=0.4, ls=(0, (1, 1)))
    ax.text(x0 * 1.25, 0.12, r"12,699$\times$", color="#B07800", fontsize=NOTE, fontweight="bold", va="center")
    x1 = 3.4e4
    ax.annotate("", xy=(x1, DP_L[-1]), xytext=(x1, CUT_L[-1]),
                arrowprops=dict(arrowstyle="<|-|>", color=VERM, lw=0.6,
                                mutation_scale=4.5, shrinkA=0, shrinkB=0))
    ax.text(x1, CUT_L[-1] * 1.6, r"4.6$\times$", color=VERM, fontsize=NOTE, fontweight="bold", ha="center", va="bottom")


def fig5b(ax):
    bw, gap = 0.36, 0.03
    off_c, on_c = "#E8A49C", BLUE
    for i in range(4):
        ax.bar(i - bw / 2 - gap / 2, SEL_OFF[i], bw, color=off_c, edgecolor="#B5534A", lw=0.5, zorder=3)
        ax.bar(i + bw / 2 + gap / 2, SEL_ON[i], bw, color=on_c, edgecolor="#00517F", lw=0.5, zorder=3)
        r = SEL_OFF[i] / SEL_ON[i]
        ax.text(i, SEL_OFF[i] * 1.35, f"{r:.0f}" + r"$\times$", ha="center", va="bottom",
                fontsize=NOTE, fontweight="bold" if i == 3 else "normal",
                color=BLUE if i == 3 else INK)
    ax.set_yscale("log")
    ax.set_ylim(0.06, 900)
    ax.yaxis.set_major_locator(mt.FixedLocator([0.1, 1, 10, 100]))
    log_minor(ax.yaxis)
    ax.set_xlim(-0.6, 3.6)
    ax.set_xticks(range(4))
    ax.set_xticklabels(GRAPHS, rotation=35, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=0, pad=2.5)
    clean(ax, grid="y")
    ax.set_ylabel("Selection time (ms)")
    ax.legend(handles=[Patch(fc=off_c, ec="#B5534A", lw=0.5, label="Deferral off"),
                       Patch(fc=on_c, ec="#00517F", lw=0.5, label="Deferral on")],
              loc="upper left", bbox_to_anchor=(-0.02, 1.04), ncol=1,
              handlelength=1.0, handleheight=0.8, handletextpad=0.4,
              labelspacing=0.25, borderaxespad=0.0)


def build_fig5():
    W, H = 1.61, 1.42
    margins = dict(left=0.255, right=0.975, bottom=0.215, top=0.975)
    for name, draw in (("exp2b_scaling", fig5a), ("exp2b_seltime", fig5b)):
        fig, ax = plt.subplots(figsize=(W, H))
        fig.subplots_adjust(**margins)
        draw(ax)
        fig.savefig(OUT / f"{name}.pdf")
        plt.close(fig)


def fig6a(ax):
    for d in DELAY:
        ax.axvline(d, color="#d9d9d9", lw=0.5, ls=(0, (1.5, 1.5)), zorder=1)
    kw = dict(lw=1.2, ms=3.3, zorder=4)
    ax.plot(DELAY, MH_EAGER, color=BLUE, marker="o", mec="white", mew=0.5, **kw)
    ax.plot(DELAY, MH_LAZY, color=BLUE, marker="o", ls=(0, (3, 1.6)), mfc="white", mec=BLUE, mew=0.9, **kw)
    ax.plot(DELAY, PH_EAGER, color=VERM, marker="s", mec="white", mew=0.5, **kw)
    ax.plot(DELAY, PH_LAZY, color=VERM, marker="s", ls=(0, (3, 1.6)), mfc="white", mec=VERM, mew=0.9, **kw)
    xl = (0.05, 160)
    ax.set(xscale="log", yscale="log", xlim=xl, ylim=(5, 260))
    ax.xaxis.set_major_locator(mt.FixedLocator([0.1, 1, 10, 100]))
    ax.yaxis.set_major_locator(mt.FixedLocator([10, 100]))
    log_minor(ax.xaxis); log_minor(ax.yaxis)
    clean(ax, grid="y")
    ax.set_xlabel("Batch-closing delay (s)")
    ax.set_ylabel("Cost per edge")
    top = ax.twiny()
    top.set(xscale="log", xlim=xl)
    top.set_xticks(DELAY)
    top.set_xticklabels([f"{b:,}" for b in BATCH])
    top.xaxis.set_minor_locator(mt.NullLocator())
    for s in ("right", "left", "bottom"):
        top.spines[s].set_visible(False)
    top.spines["top"].set_color(MUTED)
    top.tick_params(axis="x", colors=MUTED, labelcolor=MUTED, length=2.0, pad=1.2, labelsize=TICK - 0.3)
    top.set_xlabel(r"Batch size $E$ (edges)", color=MUTED, fontsize=TICK, labelpad=2.0)
    ax.text(0.11, 140, "MultiHop-RAG", color=BLUE, fontsize=NOTE, va="bottom")
    ax.text(2.2, 64, "Cit-HepPh", color=VERM, fontsize=NOTE, va="bottom")
    key = [Line2D([0], [0], color="#555555", lw=1.1, marker="o", ms=3.0, mec="white", mew=0.5, label="eager"),
           Line2D([0], [0], color="#555555", lw=1.1, ls=(0, (3, 1.6)), marker="o", ms=3.0,
                  mfc="white", mec="#555555", mew=0.9, label="lazy")]
    ax.legend(handles=key, loc="lower left", bbox_to_anchor=(0.0, 0.0),
              handlelength=2.0, handletextpad=0.4, labelspacing=0.2, borderaxespad=0.25)


def fig6_sweep(ax, data, ylabel, log):
    ax.axvline(1e-2, color=MUTED, lw=0.7, ls=(0, (1.2, 1.2)), zorder=2)
    order = ["Epinions", "Facebook", "DBLP", "Cit-HepTh", "Flickr", "Cit-HepPh"]
    style = {n: (c, m) for n, c, m in SERIES7}
    for z, name in enumerate(order):
        c, m = style[name]
        ax.plot(LAMBDAS, data[name], color=c, marker=m, lw=1.1,
                ms=3.4 if m != "X" else 3.8, mec="white", mew=0.45, zorder=3 + z)
    ax.set_xscale("log")
    ax.set_xlim(6e-4, 16)
    ax.xaxis.set_major_locator(mt.FixedLocator([1e-3, 1e-2, 1e-1, 1, 10]))
    ax.xaxis.set_minor_locator(mt.NullLocator())
    if log:
        ax.set_yscale("log")
        ax.set_ylim(0.085, 1.15)
        ax.yaxis.set_major_locator(mt.FixedLocator([0.1, 0.2, 0.5, 1.0]))
        ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
        log_minor(ax.yaxis)
        ax.text(1.25e-2, 1.08, "knee", color=MUTED, fontsize=NOTE - 0.3, style="italic", va="top")
    else:
        ax.set_ylim(-0.03, 0.82)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.text(1.25e-2, 0.80, "knee", color=MUTED, fontsize=NOTE - 0.3, style="italic", va="top")
    clean(ax, grid="y")
    ax.set_xlabel(r"Exchange rate $\lambda$")
    ax.set_ylabel(ylabel)


def fig6_key(ax):
    ax.set_axis_off()
    handles = [Line2D([0], [0], color=c, marker=m, lw=1.1,
                      ms=3.4 if m != "X" else 3.8, mec="white", mew=0.45, label=n)
               for n, c, m in SERIES7]
    handles.append(Line2D([0], [0], color=MUTED, lw=0.8, ls=(0, (1.2, 1.2)),
                          label=r"knee, $\lambda=10^{-2}$"))
    ax.legend(handles=handles, loc="center", ncol=1, fontsize=TICK,
              handlelength=2.0, handletextpad=0.5, labelspacing=0.32, borderaxespad=0.0)


def subcap(fig, ax, text, dy):
    b = ax.get_position()
    fig.text((b.x0 + b.x1) / 2, b.y0 - dy, text, ha="center", va="top",
             fontproperties=SUBCAP_FONT, color=INK)


def build_fig6():
    W, H = 3.36, 3.10
    fig = plt.figure(figsize=(W, H))
    lx, rx, w = 0.135, 0.635, 0.345
    ty, by, h = 0.625, 0.165, 0.270
    axa = fig.add_axes([lx, ty, w, h])
    axb = fig.add_axes([rx, ty, w, h])
    axc = fig.add_axes([lx, by, w, h])
    axk = fig.add_axes([rx - 0.07, by, w + 0.10, h])
    fig6a(axa)
    fig6_sweep(axb, COST, r"Cost$(\lambda)$ / Cost$(0)$", log=True)
    fig6_sweep(axc, QLOSS, r"$[Q(0)-Q(\lambda)]\times10^{3}$", log=False)
    fig6_key(axk)
    subcap(fig, axa, "(a) Exp-6: batching.", 0.125)
    subcap(fig, axb, "(b) Exp-7: cost.", 0.125)
    subcap(fig, axc, r"(c) Exp-7: $\bar{Q}$ loss.", 0.125)
    fig.savefig(OUT / "fig6_sensitivity.pdf")
    plt.close(fig)


if __name__ == "__main__":
    build_fig5()
    build_fig6()
    print("ok:", sorted(["exp2b_scaling.pdf", "exp2b_seltime.pdf", "fig6_sensitivity.pdf"]))
