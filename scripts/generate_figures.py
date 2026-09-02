"""
scripts/generate_figures.py

Generates all 4 publication-ready paper figures from the results/ directory.

Figures produced:
  Figure 1: 4-panel forgetting heatmaps (C2/C3/C4/C5)
  Figure 2: Mean forgetting curves with shaded error bands
  Figure 3: Transfer-gap bar chart (key research result)
  Figure 4: Adapter Subspace Overlap Trajectory (Frobenius norm evolution)

Output: figures/ directory (PDF + PNG, 300 DPI)
"""
import json
import os
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#cccccc",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
})

os.makedirs("figures", exist_ok=True)

# ── Palette — perceptually distinct, colorblind-friendly ────────────────────
CONFIG_COLORS = {
    "C2": "#d62728",  # Brick red
    "C3": "#2ca02c",  # Forest green
    "C4": "#ff7f0e",  # Vivid orange
    "C5": "#1f77b4",  # Steel blue
}
CONFIG_LABELS = {
    "C2": "C2: Vanilla LoRA (Baseline)",
    "C3": "C3: O-LoRA (Planned)",
    "C4": "C4: O-LoRA (Self-Gen)",
    "C5": "C5: LoRI (Self-Gen)",
}
# Distinct line styles so curves stay distinguishable even in B&W print
CONFIG_LINESTYLES = {
    "C2": (0, ()),               # solid
    "C3": (0, (6, 2)),           # dashed
    "C4": (0, (3, 1, 1, 1)),     # dash-dot
    "C5": (0, (1, 1)),           # densely dotted
}
CONFIG_MARKERS = {
    "C2": "o",
    "C3": "s",
    "C4": "^",
    "C5": "D",
}


# ── Data loading helpers ──────────────────────────────────────────────────────

def load_all_seeds(config_prefix: str) -> list:
    """Load run_log.json for all seeds of a config."""
    pattern = os.path.join("results", f"{config_prefix}*", "run_log.json")
    files = glob.glob(pattern)
    if not files:
        print(f"  [WARNING] No results found for config prefix '{config_prefix}'")
        return []
    runs = []
    for f in files:
        with open(f) as fp:
            runs.append(json.load(fp))
    print(f"  Loaded {len(runs)} seed(s) for {config_prefix}")
    return runs


def extract_accuracy_matrix(run: dict) -> dict:
    """Extract {step_num: {fact_id: accuracy}} from a run log."""
    matrix = {}
    for step_str, step_data in run.get("steps", {}).items():
        if "accuracy_matrix" in step_data:
            matrix[int(step_str)] = step_data["accuracy_matrix"]
    return matrix


def aggregate_across_seeds(runs: list) -> tuple:
    """Returns (mean_per_step, std_per_step) dicts."""
    if not runs:
        return {}, {}

    all_step_means = {}
    for run in runs:
        matrix = extract_accuracy_matrix(run)
        for step, fact_accs in matrix.items():
            mean_acc = sum(fact_accs.values()) / len(fact_accs)
            all_step_means.setdefault(step, []).append(mean_acc)

    mean_per_step = {s: np.mean(v) for s, v in all_step_means.items()}
    std_per_step  = {s: np.std(v) if len(v) > 1 else 0.0 for s, v in all_step_means.items()}
    return mean_per_step, std_per_step


def get_final_accuracies_per_seed(runs: list) -> list:
    """Extract the final-step mean accuracy for each seed."""
    finals = []
    for run in runs:
        matrix = extract_accuracy_matrix(run)
        if matrix:
            last_step = max(matrix.keys())
            facts = matrix[last_step]
            finals.append(sum(facts.values()) / len(facts))
    return finals


# ── Figure 1: Forgetting Heatmaps ────────────────────────────────────────────

def plot_heatmaps(all_config_runs: dict):
    """
    4-panel heatmaps — one per row (vertical stack) for a clean,
    uncluttered layout.  Each row shows per-fact QA accuracy across
    self-edit steps for one configuration.
    """
    configs_to_plot = [k for k in ["C2", "C3", "C4", "C5"] if all_config_runs.get(k)]
    if not configs_to_plot:
        return

    n = len(configs_to_plot)

    # Vertical stacking keeps labels readable and avoids cramped horizontal panels
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.6 * n),
                             constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, config_key in zip(axes, configs_to_plot):
        runs = all_config_runs[config_key]
        matrix = extract_accuracy_matrix(runs[0])
        if not matrix:
            continue

        steps = sorted(matrix.keys())
        facts  = sorted(matrix[steps[0]].keys())

        # y-tick decimation: show at most ~10 fact labels to avoid clutter
        fact_labels = [f"f{int(f.split('_')[1]):02d}" for f in facts]
        n_facts = len(facts)
        label_step = max(1, n_facts // 10)
        y_labels = [fact_labels[i] if i % label_step == 0 else "" for i in range(n_facts)]

        # x-tick decimation: show at most ~12 step labels
        n_steps = len(steps)
        x_step = max(1, n_steps // 12)
        x_labels = [str(s) if i % x_step == 0 else "" for i, s in enumerate(steps)]

        data = np.array([
            [matrix[s].get(f, np.nan) for f in facts]
            for s in steps
        ])  # shape: (steps, facts)

        sns.heatmap(
            data.T, ax=ax,
            vmin=0, vmax=1,
            cmap="YlOrRd_r",            # light=forgot (yellow), dark=retained (red)
            xticklabels=x_labels,
            yticklabels=y_labels,
            cbar=True,
            cbar_kws={"label": "QA Accuracy", "shrink": 0.85, "pad": 0.01},
            linewidths=0,               # no cell gridlines — cleaner look
            rasterized=True,            # faster PDF rendering for large matrices
        )

        # Panel title coloured by config
        ax.set_title(
            f"[{config_key}]  {CONFIG_LABELS[config_key]}",
            fontsize=13, fontweight="bold", loc="left", pad=8,
            color=CONFIG_COLORS[config_key],
        )
        ax.set_xlabel("Self-Edit Step", labelpad=4)
        ax.set_ylabel("Fact ID", labelpad=4)
        ax.tick_params(axis="x", rotation=0, labelsize=9)
        ax.tick_params(axis="y", rotation=0, labelsize=8)
        ax.collections[0].colorbar.ax.tick_params(labelsize=9)

    fig.suptitle(
        "Per-Fact QA Accuracy Heatmaps Across Self-Edit Steps",
        fontsize=15, fontweight="bold", y=1.01,
    )

    out = "figures/fig1_forgetting_heatmaps.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    print(f"[Fig 1] Saved: {out}")
    plt.close()


# ── Figure 2: Forgetting Trajectory Curves ────────────────────────────────────

def plot_forgetting_curves(all_config_runs: dict):
    """
    Mean accuracy trajectories across steps for all 4 configs.
    Uses distinct line styles AND markers so curves remain separable
    even in greyscale or with colourblind readers.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    has_data = False
    for config_key in ["C2", "C3", "C4", "C5"]:
        runs = all_config_runs.get(config_key, [])
        if not runs:
            continue
        means, stds = aggregate_across_seeds(runs)
        steps = sorted(means.keys())
        if not steps:
            continue

        y_mean = np.array([means[s] for s in steps])
        y_std  = np.array([stds[s]  for s in steps])

        color = CONFIG_COLORS[config_key]
        ls    = CONFIG_LINESTYLES[config_key]
        mk    = CONFIG_MARKERS[config_key]

        ax.plot(
            steps, y_mean,
            label=CONFIG_LABELS[config_key],
            color=color, linewidth=2.4,
            linestyle=ls,
            marker=mk, markersize=6, markevery=2,
            zorder=3,
        )
        if np.any(y_std > 0):
            ax.fill_between(
                steps,
                y_mean - y_std, y_mean + y_std,
                alpha=0.13, color=color, zorder=2,
            )
        has_data = True

    if not has_data:
        plt.close()
        return

    ax.set_xlabel("Self-Edit Step  ($t$)", labelpad=6)
    ax.set_ylabel("Mean Retention Accuracy", labelpad=6)
    ax.set_title("Catastrophic Forgetting Dynamics under Self-Editing")

    # Integer x-ticks only — avoids the ugly 0.0 / 2.5 / 5.0 float labels
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_ylim(bottom=-0.02)

    ax.legend(
        loc="upper right", frameon=True,
        facecolor="white", edgecolor="#cccccc",
        ncol=1, handlelength=3.0,
    )

    plt.tight_layout()
    out = "figures/fig2_forgetting_curves.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    print(f"[Fig 2] Saved: {out}")
    plt.close()


# ── Figure 3: Transfer Gap Bar Chart ─────────────────────────────────────────

def plot_transfer_gap(all_config_runs: dict):
    """Bar chart illustrating final accuracy and the Curriculum Transfer Gap."""
    c2_finals = get_final_accuracies_per_seed(all_config_runs.get("C2", []))
    c3_finals = get_final_accuracies_per_seed(all_config_runs.get("C3", []))
    c4_finals = get_final_accuracies_per_seed(all_config_runs.get("C4", []))
    c5_finals = get_final_accuracies_per_seed(all_config_runs.get("C5", []))

    if not (c2_finals and c3_finals and c4_finals):
        print("[Fig 3] Insufficient data — skipping transfer gap plot")
        return

    c2_m = np.mean(c2_finals)
    c3_m = np.mean(c3_finals)
    c4_m = np.mean(c4_finals)
    c5_m = np.mean(c5_finals) if c5_finals else 0.0

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bar_labels  = ["Vanilla LoRA\n(C2)", "O-LoRA Planned\n(C3)", "O-LoRA Self-Gen\n(C4)", "LoRI Self-Gen\n(C5)"]
    bar_heights = [c2_m, c3_m, c4_m, c5_m]
    bar_colors  = [CONFIG_COLORS[k] for k in ["C2", "C3", "C4", "C5"]]
    y_max = max(bar_heights)

    bars = ax.bar(
        bar_labels, bar_heights,
        color=bar_colors, alpha=0.85,
        edgecolor="white", linewidth=1.5,
        width=0.55,
    )

    # Value labels above each bar
    # For zero/tiny bars, nudge the label up so it's still readable
    for bar in bars:
        yval = bar.get_height()
        label_y = max(yval, y_max * 0.015) + y_max * 0.022
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            label_y,
            f"{yval:.4f}",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold",
        )

    # ── Transfer Gap annotation ────────────────────────────────────────────
    # We draw the gap bracket OUTSIDE the bars, to the right of C3 & C4.
    # Layout:
    #   horizontal dash at C3 top  ──┐
    #   vertical double arrow        │  "Transfer Gap = +X"
    #   horizontal dash at C4 top  ──┘
    gap = c3_m - c4_m

    c3_bar = bars[1]   # O-LoRA Planned
    c4_bar = bars[2]   # O-LoRA Self-Gen

    # Right edge of C4 bar + a small margin
    right_edge = c4_bar.get_x() + c4_bar.get_width()
    bx = right_edge + 0.10     # x where the bracket sits
    arm = 0.12                  # length of horizontal dashes

    y_top = c3_m   # C3 bar top
    y_bot = c4_m   # C4 bar top

    # Horizontal arm from C3 bar-right → bracket
    ax.annotate("", xy=(bx, y_top), xytext=(c3_bar.get_x() + c3_bar.get_width(), y_top),
                arrowprops=dict(arrowstyle="-", lw=1.5, color="#c0392b"), zorder=5)
    # Horizontal arm from C4 bar-right → bracket
    ax.annotate("", xy=(bx, y_bot), xytext=(right_edge, y_bot),
                arrowprops=dict(arrowstyle="-", lw=1.5, color="#c0392b"), zorder=5)

    # Double-headed vertical arrow along the bracket spine
    ax.annotate(
        "",
        xy=(bx, y_bot),
        xytext=(bx, y_top),
        arrowprops=dict(arrowstyle="<->", lw=2.0, color="#c0392b",
                        mutation_scale=16),
        zorder=5,
    )

    # Gap label to the right of the bracket midpoint
    mid_y = (y_top + y_bot) / 2.0
    ax.text(
        bx + 0.07, mid_y,
        f"Transfer Gap\n= {gap:+.4f}",
        ha="left", va="center",
        fontsize=9.5, fontweight="bold", color="#c0392b",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff5f5",
                  edgecolor="#c0392b", lw=1.0),
        zorder=6,
    )

    ax.set_ylabel("Final Mean Retention Accuracy (Step 18)", labelpad=6)
    ax.set_title("Curriculum Transfer Gap in PEFT Continual Learning")
    ax.set_ylim(0, y_max * 1.40)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = "figures/fig3_transfer_gap.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    print(f"[Fig 3] Saved: {out}")
    print(f"  C2 Vanilla: {c2_m:.4f} | C3 O-LoRA Planned: {c3_m:.4f} | C4 O-LoRA SelfGen: {c4_m:.4f} | C5 LoRI SelfGen: {c5_m:.4f}")
    print(f"  Transfer Gap (C3 - C4): {gap:+.4f}")
    plt.close()


# ── Figure 4: Adapter Subspace Overlap Evolution ────────────────────────────

def plot_subspace_overlap(all_config_runs: dict):
    """
    Plots the cumulative orthogonality penalty across steps.
    Uses distinct line styles/markers for C3 vs C4 so they remain
    visually separable even if values are similar.
    The long formula is placed in a compact in-plot text box rather
    than squeezing it into the y-axis label.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    has_data = False
    for config_key in ["C3", "C4"]:
        runs = all_config_runs.get(config_key, [])
        if not runs:
            continue
        run = runs[0]
        penalty_vals = []
        steps = []
        for step_str, step_data in run.get("steps", {}).items():
            if "loss" in step_data and "orth_penalty" in step_data["loss"]:
                steps.append(int(step_str))
                penalty_vals.append(step_data["loss"]["orth_penalty"])

        if not steps:
            continue

        # Sort by step index
        order = np.argsort(steps)
        steps = [steps[i] for i in order]
        penalty_vals = [penalty_vals[i] for i in order]

        color = CONFIG_COLORS[config_key]
        ls    = CONFIG_LINESTYLES[config_key]
        mk    = CONFIG_MARKERS[config_key]

        ax.plot(
            steps, penalty_vals,
            label=CONFIG_LABELS[config_key],
            color=color, linewidth=2.5,
            linestyle=ls,
            marker=mk, markersize=6, markevery=1,
            zorder=3,
        )
        has_data = True

    if not has_data:
        plt.close()
        return

    ax.set_xlabel("Self-Edit Step  ($t$)", labelpad=6)
    # Short, clean y-axis label — full formula goes in the in-plot note below
    ax.set_ylabel(r"Orthogonality Penalty  $\mathcal{L}_{\mathrm{orth}}$", labelpad=6)
    ax.set_title("Adapter Subspace Overlap Evolution Across Self-Edit Steps")

    # Integer x-ticks
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax.legend(
        loc="upper left", frameon=True,
        facecolor="white", edgecolor="#cccccc",
        handlelength=3.0,
    )

    # Full formula as a tidy in-plot annotation — keeps the y-axis label clean
    ax.text(
        0.98, 0.05,
        r"$\mathcal{L}_{\mathrm{orth}} = \lambda \sum_{i<t} \|A_t^{\top} A_i\|_F^2$",
        transform=ax.transAxes,
        ha="right", va="bottom", fontsize=9.5, color="#555555",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f8f8",
                  edgecolor="#cccccc", lw=0.8),
    )

    plt.tight_layout()
    out = "figures/fig4_subspace_overlap.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    print(f"[Fig 4] Saved: {out}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Generating 4 Publication Paper Figures")
    print("=" * 60)

    all_config_runs = {
        "C2": load_all_seeds("C2_vanilla_lora"),
        "C3": load_all_seeds("C3_olora_planned"),
        "C4": load_all_seeds("C4_olora_selfgen"),
        "C5": load_all_seeds("C5_lori_selfgen"),
    }

    plot_heatmaps(all_config_runs)
    plot_forgetting_curves(all_config_runs)
    plot_transfer_gap(all_config_runs)
    plot_subspace_overlap(all_config_runs)

    print("\nAll 4 figures generated successfully in figures/")


if __name__ == "__main__":
    main()
