"""Generate comparison plots: Bonbon vs Beaini/Boltz-2 on RxRx3 phenomics."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT_DIR = "/home/adimash/cellular_experiment/results/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Color palette
BONBON_COLOR = "#2563EB"      # blue
BONBON_CLEAN = "#60A5FA"      # lighter blue
BEAINI_COLOR = "#DC2626"      # red
CHANCE_COLOR = "#9CA3AF"      # gray

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


def fig1_cc_auroc():
    """Main comparison: CC AUROC apples-to-apples."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)

    # @0.4 threshold
    ax = axes[0]
    methods = ["Beaini\n(Boltz-2)", "Bonbon\n(pair-CV)", "Bonbon\n(compound-CV\nclean)"]
    values = [71.0, 78.22, 66.61]
    colors = [BEAINI_COLOR, BONBON_COLOR, BONBON_CLEAN]
    bars = ax.bar(methods, values, color=colors, width=0.55, edgecolor="white", linewidth=1.5)
    ax.set_title("CC AUROC @ 0.4 threshold")
    ax.set_ylabel("AUROC (%)")
    ax.set_ylim(50, 90)
    ax.axhline(50, color=CHANCE_COLOR, linestyle=":", linewidth=1, label="Chance")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.8, f"{val:.1f}%",
                ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.annotate("No CV\n(train=test)", xy=(0, 71.0), xytext=(0.3, 75),
                fontsize=9, color=BEAINI_COLOR, ha="center",
                arrowprops=dict(arrowstyle="->", color=BEAINI_COLOR, lw=1.2))

    # @0.6 threshold
    ax = axes[1]
    values = [76.0, 81.15, 68.08]
    bars = ax.bar(methods, values, color=colors, width=0.55, edgecolor="white", linewidth=1.5)
    ax.set_title("CC AUROC @ 0.6 threshold")
    ax.set_ylim(50, 90)
    ax.axhline(50, color=CHANCE_COLOR, linestyle=":", linewidth=1)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.8, f"{val:.1f}%",
                ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.annotate("No CV\n(train=test)", xy=(0, 76.0), xytext=(0.3, 80),
                fontsize=9, color=BEAINI_COLOR, ha="center",
                arrowprops=dict(arrowstyle="->", color=BEAINI_COLOR, lw=1.2))

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=BEAINI_COLOR, label="Beaini/Boltz-2 (no CV, in-distribution)"),
        mpatches.Patch(facecolor=BONBON_COLOR, label="Bonbon (5-fold pair-level CV)"),
        mpatches.Patch(facecolor=BONBON_CLEAN, label="Bonbon (5-fold compound-level CV, strictest)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.02), frameon=True, edgecolor="#E5E7EB")

    fig.suptitle("Compound-Compound AUROC: Bonbon vs. Boltz-2", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig1_cc_auroc_comparison.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved fig1_cc_auroc_comparison.png")


def fig2_per_target():
    """Per-target comparison."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    methods = [
        "Beaini\n(Boltz-2)\nzero-shot",
        "Bonbon\nzero-shot\n(best config)",
        "Bonbon\ntrained\n(global model)",
    ]
    values = [53.9, 58.78, 86.49]
    colors = [BEAINI_COLOR, BONBON_COLOR, BONBON_COLOR]
    hatches = ["", "", "//"]

    bars = ax.bar(methods, values, color=colors, width=0.5, edgecolor="white", linewidth=1.5)
    bars[2].set_hatch("//")
    bars[2].set_edgecolor(BONBON_COLOR)

    ax.set_ylabel("Median per-target AUROC (%)")
    ax.set_ylim(40, 100)
    ax.axhline(50, color=CHANCE_COLOR, linestyle=":", linewidth=1, label="Chance (50%)")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.0, f"{val:.1f}%",
                ha="center", va="bottom", fontweight="bold", fontsize=12)

    # Annotations
    ax.annotate("Uses transcriptomics\n+ PPI + 11 cell lines\n+ 500K ligands",
                xy=(0, 53.9), xytext=(0.6, 45),
                fontsize=9, color=BEAINI_COLOR, ha="center",
                arrowprops=dict(arrowstyle="->", color=BEAINI_COLOR, lw=1.2))
    ax.annotate("Bonbon embeddings\nonly (1 checkpoint,\n1 cell line)",
                xy=(1, 58.78), xytext=(1.55, 47),
                fontsize=9, color=BONBON_COLOR, ha="center",
                arrowprops=dict(arrowstyle="->", color=BONBON_COLOR, lw=1.2))
    ax.annotate("Target-level 5-fold CV\n(not apples-to-apples\nwith Beaini)",
                xy=(2, 86.49), xytext=(2, 95),
                fontsize=8, color="#6B7280", ha="center",
                arrowprops=dict(arrowstyle="->", color="#6B7280", lw=1))

    ax.set_title("Per-Target AUROC: Bonbon vs. Boltz-2", fontweight="bold", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig2_per_target_comparison.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved fig2_per_target_comparison.png")


def fig3_compute():
    """Compute efficiency comparison."""
    fig, ax = plt.subplots(figsize=(9, 4.5))

    categories = ["Embedding\ncompute", "PPI\ncompute", "CC AUROC\n@ 0.6", "Per-target\nzero-shot"]

    # Beaini
    beaini_y = [0, 1, 2, 3]
    beaini_vals = ["12 months\nBioHive-1", "4M co-foldings\n(months)", "76.0%", "53.9%"]

    # Bonbon
    bonbon_vals = ["12.5 min\n1x H200", "Seconds\n(cosine sim)", "81.2%", "58.8%"]

    row_height = 0.35
    for i, cat in enumerate(categories):
        # Beaini bar
        ax.barh(i + row_height / 2, 1, height=row_height, color=BEAINI_COLOR, alpha=0.85)
        ax.text(0.5, i + row_height / 2, beaini_vals[i], ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")
        # Bonbon bar
        ax.barh(i - row_height / 2, 1, height=row_height, color=BONBON_COLOR, alpha=0.85)
        ax.text(0.5, i - row_height / 2, bonbon_vals[i], ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")

    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories, fontsize=11)
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.invert_yaxis()

    legend_elements = [
        mpatches.Patch(facecolor=BONBON_COLOR, alpha=0.85, label="Bonbon (dark-snowball-245)"),
        mpatches.Patch(facecolor=BEAINI_COLOR, alpha=0.85, label="Beaini/Boltz-2"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10, frameon=True, edgecolor="#E5E7EB")

    ax.set_title("Compute & Performance: Bonbon vs. Boltz-2", fontweight="bold", fontsize=14)
    ax.grid(False)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig3_compute_comparison.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved fig3_compute_comparison.png")


def fig4_eval_rigor():
    """Evaluation methodology comparison — showing our numbers hold under stricter evaluation."""
    fig, ax = plt.subplots(figsize=(10, 5))

    methods = [
        "Beaini\ntrain=test\n(no CV)",
        "Bonbon\npair-level CV\n(5-fold)",
        "Bonbon\ncompound-CV\nleaky sig",
        "Bonbon\ncompound-CV\nclean sig",
    ]
    vals_04 = [71.0, 78.22, 68.07, 66.61]
    vals_06 = [76.0, 81.15, 72.28, 68.08]

    x = np.arange(len(methods))
    w = 0.32

    bars1 = ax.bar(x - w / 2, vals_04, w, label="@0.4 threshold", color=BONBON_COLOR, alpha=0.7)
    bars2 = ax.bar(x + w / 2, vals_06, w, label="@0.6 threshold", color=BONBON_COLOR, alpha=1.0)

    # Color Beaini differently
    bars1[0].set_color(BEAINI_COLOR)
    bars1[0].set_alpha(0.7)
    bars2[0].set_color(BEAINI_COLOR)
    bars2[0].set_alpha(1.0)

    for bars in [bars1, bars2]:
        for bar in bars:
            val = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5, f"{val:.1f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("CC AUROC (%)")
    ax.set_ylim(50, 90)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.axhline(50, color=CHANCE_COLOR, linestyle=":", linewidth=1)
    ax.legend(loc="upper right")

    # Arrow showing increasing rigor
    ax.annotate("", xy=(3.4, 55), xytext=(0.4, 55),
                arrowprops=dict(arrowstyle="->", color="#374151", lw=2))
    ax.text(1.9, 52.5, "Increasing evaluation rigor  →", ha="center", fontsize=10,
            color="#374151", fontstyle="italic")

    ax.set_title("CC AUROC Under Increasing Evaluation Rigor", fontweight="bold", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig4_eval_rigor.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved fig4_eval_rigor.png")


if __name__ == "__main__":
    print("Generating plots...")
    fig1_cc_auroc()
    fig2_per_target()
    fig3_compute()
    fig4_eval_rigor()
    print("Done!")
