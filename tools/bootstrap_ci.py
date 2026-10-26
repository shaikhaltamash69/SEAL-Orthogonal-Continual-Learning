"""
tools/bootstrap_ci.py

WP4: Clustered bootstrap confidence intervals for continual learning evaluation.
Resamples facts with replacement, then questions within facts, to compute
empirical 95% bootstrap intervals for:
  1. Single config final accuracy
  2. Curriculum Transfer Gap (CTG) differences between configs (e.g. C3 - C4, C6 - C5)
Supports both new runs (with per-step qa_correct logs) and retroactive v1 runs.

Usage:
    python tools/bootstrap_ci.py --config C3
    python tools/bootstrap_ci.py --diff C3 C4
    python tools/bootstrap_ci.py --diff C6 C5
    python tools/bootstrap_ci.py --config C4 --split new
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.config_registry import all_configs, config_for_run_id, SEEDS_V1
from compile_results import load_baseline, scan_results, load_pool_size


def extract_question_correctness(run_id: str, results_dir: Path = ROOT / "results") -> dict[str, list[int]]:
    """
    Extract final-step per-question correctness {fact_id: [0/1, ...]} for a completed run.
    If qa_correct is present in step log, uses it; otherwise reconstructs from accuracy_matrix.
    """
    log_path = results_dir / run_id / "run_log.json"
    with open(log_path) as f:
        log = json.load(f)

    steps = log.get("steps", {})
    last_step = str(max(int(k) for k in steps.keys()))
    step_data = steps[last_step]

    # Case 1: qa_correct is explicitly logged
    if "qa_correct" in step_data:
        return {fid: [int(x) for x in res] for fid, res in step_data["qa_correct"].items()}

    # Case 2: reconstruct from accuracy_matrix (each fact has 2 questions in pool)
    acc_matrix = step_data.get("accuracy_matrix", {})
    reconstructed = {}
    for fid, acc in acc_matrix.items():
        if math.isclose(acc, 1.0, abs_tol=1e-3):
            reconstructed[fid] = [1, 1]
        elif math.isclose(acc, 0.5, abs_tol=1e-3):
            reconstructed[fid] = [1, 0]
        else:
            reconstructed[fid] = [0, 0]
    return reconstructed


def bootstrap_config(
    cfg_id: str,
    runs_map: dict[int, dict],
    fact_ids: list[str],
    n_boot: int = 2000,
    rng: random.Random | None = None,
) -> list[float]:
    """
    Run bootstrap resampling for a single config across its available seeds.
    Returns list of bootstrap replicate mean accuracies.
    """
    if rng is None:
        rng = random.Random(42)

    # Pre-extract correctness per seed: {seed: {fact_id: [0/1, ...]}}
    seed_data = {}
    for seed, r in runs_map.items():
        seed_data[seed] = extract_question_correctness(r["run_id"])

    seeds = list(seed_data.keys())
    N = len(fact_ids)
    boot_means = []

    for _ in range(n_boot):
        # 1. Sample facts with replacement
        sampled_facts = [rng.choice(fact_ids) for _ in range(N)]
        
        seed_accs = []
        for s in seeds:
            total_correct = 0
            total_q = 0
            for fid in sampled_facts:
                q_list = seed_data[s].get(fid, [0, 0])
                # 2. Sample questions within fact with replacement
                sampled_q = [rng.choice(q_list) for _ in range(len(q_list))]
                total_correct += sum(sampled_q)
                total_q += len(sampled_q)
            seed_accs.append(total_correct / total_q if total_q > 0 else 0.0)

        boot_means.append(sum(seed_accs) / len(seed_accs))

    return boot_means


def bootstrap_diff(
    cfg_a: str,
    cfg_b: str,
    runs_a: dict[int, dict],
    runs_b: dict[int, dict],
    fact_ids: list[str],
    n_boot: int = 2000,
    rng: random.Random | None = None,
) -> list[float]:
    """
    Run paired bootstrap resampling for the difference (cfg_a - cfg_b).
    """
    if rng is None:
        rng = random.Random(42)

    seed_data_a = {s: extract_question_correctness(r["run_id"]) for s, r in runs_a.items()}
    seed_data_b = {s: extract_question_correctness(r["run_id"]) for s, r in runs_b.items()}
    seeds_a = list(seed_data_a.keys())
    seeds_b = list(seed_data_b.keys())
    N = len(fact_ids)

    boot_diffs = []

    for _ in range(n_boot):
        # Shared resampled facts
        sampled_facts = [rng.choice(fact_ids) for _ in range(N)]

        # Config A
        accs_a = []
        for s in seeds_a:
            corr = 0
            tot = 0
            for fid in sampled_facts:
                ql = seed_data_a[s].get(fid, [0, 0])
                res = [rng.choice(ql) for _ in range(len(ql))]
                corr += sum(res)
                tot += len(res)
            accs_a.append(corr / tot if tot > 0 else 0.0)
        mean_a = sum(accs_a) / len(accs_a)

        # Config B
        accs_b = []
        for s in seeds_b:
            corr = 0
            tot = 0
            for fid in sampled_facts:
                ql = seed_data_b[s].get(fid, [0, 0])
                res = [rng.choice(ql) for _ in range(len(ql))]
                corr += sum(res)
                tot += len(res)
            accs_b.append(corr / tot if tot > 0 else 0.0)
        mean_b = sum(accs_b) / len(accs_b)

        boot_diffs.append(mean_a - mean_b)

    return boot_diffs


def compute_ci(replicates: list[float], alpha: float = 0.05) -> tuple[float, float, float, float]:
    """
    Compute point estimate, standard error, and percentile [low, high] CI.
    """
    replicates.sort()
    n = len(replicates)
    mean_val = sum(replicates) / n
    variance = sum((x - mean_val) ** 2 for x in replicates) / (n - 1)
    se = math.sqrt(variance)
    low_idx = int((alpha / 2.0) * n)
    high_idx = int((1.0 - alpha / 2.0) * n)
    low = replicates[min(low_idx, n - 1)]
    high = replicates[min(high_idx, n - 1)]
    return mean_val, se, low, high


def main():
    parser = argparse.ArgumentParser(description="Clustered bootstrap confidence intervals for continual learning.")
    parser.add_argument("--config", type=str, default=None, help="Config ID (e.g. C3)")
    parser.add_argument("--diff", type=str, nargs=2, default=None, help="Compute difference between two configs (e.g. --diff C3 C4)")
    parser.add_argument("--protocol", type=int, default=1, choices=[1, 2], help="Protocol version (default: 1)")
    parser.add_argument("--split", type=str, choices=["all", "new", "pre"], default="all",
                        help="Fact split to evaluate: all, new (prior acc=0), pre (prior acc>0)")
    parser.add_argument("--n-boot", type=int, default=2000, help="Number of bootstrap resamples (default: 2000)")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level (default: 0.05 -> 95% CI)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    args = parser.parse_args()

    pool_size = load_pool_size()
    found_runs = scan_results(args.protocol, pool_size)
    baseline = load_baseline()

    # Determine fact subset
    all_facts = [f"fact_{i:03d}" for i in range(1, pool_size + 2) if f"fact_{i:03d}" in baseline]
    if args.split == "new":
        eval_facts = [fid for fid in all_facts if baseline.get(fid, 0.0) == 0.0]
    elif args.split == "pre":
        eval_facts = [fid for fid in all_facts if baseline.get(fid, 0.0) > 0.0]
    else:
        eval_facts = all_facts

    rng = random.Random(args.seed)

    print(f"\n{'='*75}")
    print(f"BOOTSTRAP CONFIDENCE INTERVALS (Protocol {args.protocol}, Split: {args.split.upper()})")
    print(f"  Facts in split: {len(eval_facts)}  |  Bootstrap replicates: {args.n_boot}  |  Alpha: {args.alpha}")
    print(f"{'='*75}")

    if args.diff:
        cfg_a, cfg_b = args.diff
        if cfg_a not in found_runs or cfg_b not in found_runs:
            print(f"[ERROR] Runs for {cfg_a} or {cfg_b} not found.")
            sys.exit(1)
        runs_a = found_runs[cfg_a]
        runs_b = found_runs[cfg_b]
        diffs = bootstrap_diff(cfg_a, cfg_b, runs_a, runs_b, eval_facts, n_boot=args.n_boot, rng=rng)
        mean_d, se_d, low_d, high_d = compute_ci(diffs, alpha=args.alpha)
        ci_pct = int((1.0 - args.alpha) * 100)
        print(f"\nDifference ({cfg_a} - {cfg_b}):")
        print(f"  Mean diff: {mean_d:+.4f} ({mean_d*100:+.2f} pp)")
        print(f"  Std Error: {se_d:.4f}")
        print(f"  {ci_pct}% CI:   [{low_d:+.4f}, {high_d:+.4f}] ([{low_d*100:+.2f} pp, {high_d*100:+.2f} pp])")
        signif = "Significant (CI excludes 0)" if (low_d > 0 or high_d < 0) else "Not significant (CI spans 0)"
        print(f"  Status:    {signif}")

    elif args.config:
        cfg = args.config
        if cfg not in found_runs:
            print(f"[ERROR] Runs for {cfg} not found.")
            sys.exit(1)
        runs = found_runs[cfg]
        boots = bootstrap_config(cfg, runs, eval_facts, n_boot=args.n_boot, rng=rng)
        mean_v, se_v, low_v, high_v = compute_ci(boots, alpha=args.alpha)
        ci_pct = int((1.0 - args.alpha) * 100)
        print(f"\nConfig {cfg}:")
        print(f"  Mean accuracy: {mean_v:.4f} ({mean_v*100:.2f}%)")
        print(f"  Std Error:     {se_v:.4f}")
        print(f"  {ci_pct}% CI:        [{low_v:.4f}, {high_v:.4f}] ([{low_v*100:.2f}%, {high_v*100:.2f}%])")
    else:
        # Default: print C3, C4, and difference C3-C4
        print("\n[Default run: C3 vs C4 and C6 vs C5 CTG differences]")
        for c_a, c_b in [("C3", "C4"), ("C6", "C5")]:
            if c_a in found_runs and c_b in found_runs:
                diffs = bootstrap_diff(c_a, c_b, found_runs[c_a], found_runs[c_b], eval_facts, n_boot=args.n_boot, rng=rng)
                mean_d, se_d, low_d, high_d = compute_ci(diffs, alpha=args.alpha)
                print(f"  {c_a} - {c_b}: mean={mean_d:+.4f} [{low_d:+.4f}, {high_d:+.4f}] (95% CI)")

    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
