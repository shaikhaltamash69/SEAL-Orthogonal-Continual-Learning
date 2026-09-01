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
import seaborn as sns
from scipy import stats

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 300,
})
os.makedirs("figures", exist_ok=True)

CONFIG_COLORS = {
    "C2": "#e74c3c",  # Red
    "C3": "#27ae60",  # Green
    "C4": "#e67e22",  # Orange
    "C5": "#2980b9",  # Blue
}
CONFIG_LABELS = {
    "C2": "C2: Vanilla LoRA (Baseline)",
    "C3": "C3: O-LoRA (Planned)",
    "C4": "C4: O-LoRA (Self-Gen)",
    "C5": "C5: LoRI (Self-Gen)",
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
    """4-panel heatmaps comparing per-fact accuracy across steps."""
    configs_to_plot = [k for k in ["C2", "C3", "C4", "C5"] if all_config_runs.get(k)]
    if not configs_to_plot:
        return

    n = len(configs_to_plot)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5.5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, config_key in zip(axes, configs_to_plot):
        runs = all_config_runs[config_key]
        matrix = extract_accuracy_matrix(runs[0])
        if not matrix:
            continue

        steps = sorted(matrix.keys())
        facts  = sorted(matrix[steps[0]].keys())
        data   = np.array([
            [matrix[s].get(f, np.nan) for f in facts]
            for s in steps
        ])

        sns.heatmap(
            data.T, ax=ax,
            vmin=0, vmax=1, cmap="YlGnBu",
            xticklabels=[str(s) for s in steps],
            yticklabels=[f"f_{int(f.split('_')[1]):02d}" for f in facts],
            cbar=(ax == axes[-1]),
            cbar_kws={"label": "QA Accuracy"},
            linewidths=0.2, linecolor="#eeeeee",
        )
        ax.set_title(CONFIG_LABELS[config_key], fontsize=10, fontweight="bold")
        ax.set_xlabel("Self-Edit Step")
        if ax == axes[0]:
            ax.set_ylabel("Fact Identifier")
        ax.tick_params(axis="y", labelsize=7)
        ax.tick_params(axis="x", labelsize=7)

    plt.tight_layout()
    out = "figures/fig1_forgetting_heatmaps.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    print(f"[Fig 1] Saved: {out}")
    plt.close()


# ── Figure 2: Forgetting Trajectory Curves ────────────────────────────────────

def plot_forgetting_curves(all_config_runs: dict):
    """Mean accuracy trajectories across steps for all 4 configs."""
    fig, ax = plt.subplots(figsize=(8, 4.8))

    for config_key in ["C2", "C3", "C4", "C5"]:
        runs = all_config_runs.get(config_key, [])
        if not runs:
            continue
        means, stds = aggregate_across_seeds(runs)
        steps = sorted(means.keys())
        y_mean = [means[s] for s in steps]
        y_std  = [stds[s]  for s in steps]

        color = CONFIG_COLORS[config_key]
        ax.plot(steps, y_mean, label=CONFIG_LABELS[config_key],
                color=color, linewidth=2.2, marker="o", markersize=4)
        if any(s > 0 for s in y_std):
            ax.fill_between(steps,
                            np.array(y_mean) - np.array(y_std),
                            np.array(y_mean) + np.array(y_std),
                            alpha=0.15, color=color)

    ax.set_xlabel("Self-Edit Sequence Step ($t$)")
    ax.set_ylabel("Mean Retention Accuracy")
    ax.set_title("Catastrophic Forgetting Dynamics under Self-Editing", fontweight="bold")
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")
    ax.set_ylim(-0.02, 0.60)
    ax.grid(True, linestyle="--", alpha=0.4)
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

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bar_labels  = ["Vanilla LoRA\n(C2)", "O-LoRA Planned\n(C3)", "O-LoRA Self-Gen\n(C4)", "LoRI Self-Gen\n(C5)"]
    bar_heights = [c2_m, c3_m, c4_m, c5_m]
    bar_colors  = [CONFIG_COLORS["C2"], CONFIG_COLORS["C3"], CONFIG_COLORS["C4"], CONFIG_COLORS["C5"]]

    bars = ax.bar(bar_labels, bar_heights, color=bar_colors, alpha=0.88, edgecolor="black", width=0.55)

    # Annotate value on top of each bar
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f"{yval:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Annotate Curriculum Transfer Gap arrow between C3 and C4
    gap = c3_m - c4_m
    ax.annotate(
        f"Transfer Gap: {gap:+.4f}",
        xy=(1.5, max(c3_m, c4_m) + 0.015),
        xytext=(1.5, max(c3_m, c4_m) + 0.035),
        arrowprops=dict(arrowstyle="->", lw=1.2, color="crimson"),
        ha="center", fontsize=9, color="crimson", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff0f0", edgecolor="crimson", lw=0.8)
    )

    ax.set_ylabel("Final Mean Retention Accuracy (Step 18)")
    ax.set_title("Quantifying the Curriculum Transfer Gap in PEFT", fontweight="bold")
    ax.set_ylim(0, max(bar_heights) + 0.06)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
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
    """Plots the cumulative orthogonality penalty (subspace overlap) across steps."""
    fig, ax = plt.subplots(figsize=(7, 4.2))

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

        if steps:
            color = CONFIG_COLORS[config_key]
            ax.plot(steps, penalty_vals, label=f"{CONFIG_LABELS[config_key]} Orth Penalty",
                    color=color, linewidth=2.0, linestyle="-", marker="s", markersize=3.5)

    ax.set_xlabel("Self-Edit Sequence Step ($t$)")
    ax.set_ylabel(r"Orthogonality Penalty $\mathcal{L}_{\mathrm{orth}} = \lambda \sum \|A_t^T A_i\|_F^2$")
    ax.set_title("Adapter Subspace Overlap Evolution Across Steps", fontweight="bold")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.4)
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
