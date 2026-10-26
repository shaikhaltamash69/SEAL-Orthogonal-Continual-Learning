"""
scripts/render_workflow_diagram.py

Renders a publication-quality, high-resolution workflow diagram matching the
Eraser-style architectural blueprint for SEAL:
1. Data & Verification
2. Curriculum Selection
3. Self-Edit Loop (19 iterations)

Outputs:
  - figures/workflow_diagram.png (300 DPI)
  - figures/workflow_diagram.pdf
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_workflow_diagram():
    os.makedirs("figures", exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 13), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 110)
    ax.axis("off")

    # Title
    ax.text(
        50, 106,
        "SEAL: Curriculum Transfer Gap in Self-Editing Language Models",
        ha="center", va="center",
        fontsize=16, fontweight="bold", color="#1e293b",
        fontfamily="sans-serif"
    )

    # -- Box 1: Data & Verification (Top Left) --------------------------
    b1 = patches.FancyBboxPatch(
        (4, 53), 38, 48,
        boxstyle="round,pad=1.2,rounding_size=2.5",
        linewidth=1.5, edgecolor="#93c5fd", facecolor="#eff6ff"
    )
    ax.add_patch(b1)
    ax.text(6, 98, "1 · Data & Verification", fontsize=12, fontweight="bold", color="#1e40af")

    # Sub-card 1.1: Fact Pool
    sc1 = patches.FancyBboxPatch(
        (7, 72), 32, 22,
        boxstyle="round,pad=1.0,rounding_size=2",
        linewidth=1.2, edgecolor="#bfdbfe", facecolor="#ffffff"
    )
    ax.add_patch(sc1)
    ax.text(23, 90, "Fact Pool", ha="center", fontsize=11, fontweight="bold", color="#0f172a")
    ax.text(23, 86, "data/fact_pool.json", ha="center", fontsize=8.5, color="#64748b", fontfamily="monospace")
    ax.text(23, 81.5, "19 verified post-2024 facts", ha="center", fontsize=9, fontweight="bold", color="#334155")
    ax.text(23, 76.5, "tech · science · politics · sports\npassage + title + test QA pairs", ha="center", fontsize=8, color="#475569")

    # Sub-card 1.2: Novelty Verification
    sc2 = patches.FancyBboxPatch(
        (7, 56), 32, 12,
        boxstyle="round,pad=1.0,rounding_size=2",
        linewidth=1.2, edgecolor="#bfdbfe", facecolor="#ffffff"
    )
    ax.add_patch(sc2)
    ax.text(23, 65, "Novelty Verification", ha="center", fontsize=10.5, fontweight="bold", color="#0f172a")
    ax.text(23, 61.5, "verify_novelty.py", ha="center", fontsize=8.5, color="#64748b", fontfamily="monospace")
    ax.text(23, 58, "Qwen2.5-1.5B-Instruct scores\n0% prior knowledge", ha="center", fontsize=8, color="#334155")

    # Arrow 1.1 -> 1.2
    ax.annotate(
        "", xy=(23, 68), xytext=(23, 72),
        arrowprops=dict(arrowstyle="->", lw=1.8, color="#0f172a")
    )

    # -- Box 2: Curriculum Selection (Top Right) ------------------------
    b2 = patches.FancyBboxPatch(
        (46, 51), 50, 50,
        boxstyle="round,pad=1.2,rounding_size=2.5",
        linewidth=1.5, edgecolor="#c4b5fd", facecolor="#f5f3ff"
    )
    ax.add_patch(b2)
    ax.text(48, 98, "2 · Curriculum Selection", fontsize=12, fontweight="bold", color="#5b21b6")

    # Diamond / Decision node
    diamond = patches.RegularPolygon(
        (71, 86), numVertices=4, radius=8, orientation=0,
        linewidth=1.2, edgecolor="#a78bfa", facecolor="#ffffff"
    )
    ax.add_patch(diamond)
    ax.text(71, 86.8, "Curriculum", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#312e81")
    ax.text(71, 84, "Strategy", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#312e81")

    # Sub-card 2.1: PlannedCurriculum
    sc21 = patches.FancyBboxPatch(
        (49, 69), 44, 11,
        boxstyle="round,pad=1.0,rounding_size=2",
        linewidth=1.2, edgecolor="#ddd6fe", facecolor="#ffffff"
    )
    ax.add_patch(sc21)
    ax.text(71, 76.5, "PlannedCurriculum (Human)", ha="center", fontsize=10, fontweight="bold", color="#0f172a")
    ax.text(71, 72, "Fixed static order · used in C3", ha="center", fontsize=8.5, color="#475569")

    # Sub-card 2.2: SelfGeneratedCurriculum
    sc22 = patches.FancyBboxPatch(
        (49, 54), 44, 12.5,
        boxstyle="round,pad=1.0,rounding_size=2",
        linewidth=1.2, edgecolor="#ddd6fe", facecolor="#ffffff"
    )
    ax.add_patch(sc22)
    ax.text(71, 62.5, "SelfGeneratedCurriculum (Model)", ha="center", fontsize=10, fontweight="bold", color="#0f172a")
    ax.text(71, 58, "LLM picks next fact + logs reasoning\nused in C2, C4, C5", ha="center", fontsize=8.5, color="#475569")

    # Connectors in Box 2
    # From Data & Verification to Curriculum Strategy
    ax.annotate(
        "", xy=(63, 86), xytext=(39, 62),
        arrowprops=dict(arrowstyle="->", lw=1.8, color="#0f172a", connectionstyle="arc3,rad=-0.15")
    )
    # From Strategy Diamond to Planned
    ax.annotate(
        "", xy=(71, 80), xytext=(71, 81.5),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="#0f172a")
    )
    # From Strategy Diamond to Self-Generated
    ax.annotate(
        "", xy=(49, 60), xytext=(63, 86),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="#0f172a", connectionstyle="arc3,rad=0.2")
    )

    # -- Box 3: Self-Edit Loop (19 iterations) (Bottom) ----------------
    b3 = patches.FancyBboxPatch(
        (4, 5), 92, 43,
        boxstyle="round,pad=1.2,rounding_size=2.5",
        linewidth=1.5, edgecolor="#86efac", facecolor="#f0fdf4"
    )
    ax.add_patch(b3)
    ax.text(6, 45, "3 · Self-Edit Loop (19 iterations)", fontsize=12, fontweight="bold", color="#15803d")

    # Sub-card 3.1: Synthetic Data Gen
    sc31 = patches.FancyBboxPatch(
        (7, 26), 26, 15,
        boxstyle="round,pad=1.0,rounding_size=2",
        linewidth=1.2, edgecolor="#bbf7d0", facecolor="#ffffff"
    )
    ax.add_patch(sc31)
    ax.text(20, 36.5, "Synthetic Data Gen", ha="center", fontsize=10.5, fontweight="bold", color="#0f172a")
    ax.text(20, 32.5, "generate_synthetic_data", ha="center", fontsize=8, color="#64748b", fontfamily="monospace")
    ax.text(20, 28.5, "4 QA pairs from passage", ha="center", fontsize=8.5, color="#334155")

    # Diamond for SFT Adaptation Method
    d_sft = patches.RegularPolygon(
        (20, 16), numVertices=4, radius=6.5, orientation=0,
        linewidth=1.2, edgecolor="#86efac", facecolor="#ffffff"
    )
    ax.add_patch(d_sft)
    ax.text(20, 16.8, "SFT Adaptation", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#166534")
    ax.text(20, 14.5, "Method", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#166534")

    ax.annotate(
        "", xy=(20, 21), xytext=(20, 26),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="#0f172a")
    )

    # Sub-card 3.2a: C2 Vanilla LoRA
    sc32a = patches.FancyBboxPatch(
        (37, 33), 26, 8.5,
        boxstyle="round,pad=0.8,rounding_size=1.8",
        linewidth=1.2, edgecolor="#cbd5e1", facecolor="#ffffff"
    )
    ax.add_patch(sc32a)
    ax.text(50, 38.5, "C2 · Vanilla LoRA", ha="center", fontsize=9.5, fontweight="bold", color="#0f172a")
    ax.text(50, 35, "Standard LoRA SFT, no constraint", ha="center", fontsize=7.5, color="#475569")

    # Sub-card 3.2b: C3/C4 O-LoRA
    sc32b = patches.FancyBboxPatch(
        (37, 22), 26, 9.5,
        boxstyle="round,pad=0.8,rounding_size=1.8",
        linewidth=1.2, edgecolor="#cbd5e1", facecolor="#ffffff"
    )
    ax.add_patch(sc32b)
    ax.text(50, 28.5, "C3/C4 · O-LoRA", ha="center", fontsize=9.5, fontweight="bold", color="#0f172a")
    ax.text(50, 25.5, "src/orthogonality.py · λ=0.1", ha="center", fontsize=7.5, color="#64748b", fontfamily="monospace")
    ax.text(50, 23, "L_SFT + λ Σ ||A_t^T A_i||²_F", ha="center", fontsize=7.5, color="#1e293b", fontweight="bold")

    # Sub-card 3.2c: C5 LoRI
    sc32c = patches.FancyBboxPatch(
        (37, 9), 26, 11,
        boxstyle="round,pad=0.8,rounding_size=1.8",
        linewidth=1.2, edgecolor="#cbd5e1", facecolor="#ffffff"
    )
    ax.add_patch(sc32c)
    ax.text(50, 16.5, "C5 · LoRI", ha="center", fontsize=9.5, fontweight="bold", color="#0f172a")
    ax.text(50, 13.5, "A frozen random projection", ha="center", fontsize=7.5, color="#475569")
    ax.text(50, 10.5, "top-k mask on B (80% sparse)", ha="center", fontsize=7.5, color="#475569")

    # Connectors from SFT Diamond to C2, C3/C4, C5
    ax.annotate("", xy=(37, 37), xytext=(26, 17), arrowprops=dict(arrowstyle="->", lw=1.3, color="#0f172a", connectionstyle="arc3,rad=-0.1"))
    ax.annotate("", xy=(37, 26), xytext=(26.5, 16), arrowprops=dict(arrowstyle="->", lw=1.3, color="#0f172a", connectionstyle="arc3,rad=-0.05"))
    ax.annotate("", xy=(37, 14), xytext=(26.5, 15), arrowprops=dict(arrowstyle="->", lw=1.3, color="#0f172a"))

    # Sub-card 3.3: Cumulative QA Evaluation
    sc33 = patches.FancyBboxPatch(
        (66, 19), 22, 17,
        boxstyle="round,pad=1.0,rounding_size=2",
        linewidth=1.2, edgecolor="#86efac", facecolor="#ffffff"
    )
    ax.add_patch(sc33)
    ax.text(77, 31, "Cumulative QA", ha="center", fontsize=10.5, fontweight="bold", color="#0f172a")
    ax.text(77, 27.5, "Evaluation", ha="center", fontsize=10.5, fontweight="bold", color="#0f172a")
    ax.text(77, 23, "Greedy decode over", ha="center", fontsize=8, color="#475569")
    ax.text(77, 20.5, "all facts 0 ... t", ha="center", fontsize=8, color="#475569")

    # Connectors from Methods to QA Eval
    ax.annotate("", xy=(66, 29), xytext=(63, 37), arrowprops=dict(arrowstyle="->", lw=1.3, color="#0f172a", connectionstyle="arc3,rad=-0.1"))
    ax.annotate("", xy=(66, 27), xytext=(63, 27), arrowprops=dict(arrowstyle="->", lw=1.3, color="#0f172a"))
    ax.annotate("", xy=(66, 25), xytext=(63, 15), arrowprops=dict(arrowstyle="->", lw=1.3, color="#0f172a", connectionstyle="arc3,rad=0.1"))

    # Connectors from Curriculum to Synthetic Data Gen
    ax.annotate("", xy=(7, 34), xytext=(93, 74), arrowprops=dict(arrowstyle="->", lw=1.5, color="#0f172a", connectionstyle="arc3,rad=0.4"))
    ax.annotate("", xy=(7, 31), xytext=(93, 60), arrowprops=dict(arrowstyle="->", lw=1.5, color="#0f172a", connectionstyle="arc3,rad=0.4"))

    plt.tight_layout()
    plt.savefig("figures/workflow_diagram.png", dpi=300, bbox_inches="tight")
    plt.savefig("figures/workflow_diagram.pdf", bbox_inches="tight")
    plt.close()
    print("[SUCCESS] High-resolution workflow diagram generated at figures/workflow_diagram.png and .pdf")

if __name__ == "__main__":
    generate_workflow_diagram()
