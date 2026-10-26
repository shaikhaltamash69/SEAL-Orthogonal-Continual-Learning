"""
scripts/run_multi_seeds.py

Run experimental configs across multiple seeds and collect mean / min-max accuracy.
Import configs and labels from scripts.config_registry (WP1/WP3 single source of truth).

Usage:
    python scripts/run_multi_seeds.py --seeds 42 7 123
    python scripts/run_multi_seeds.py --protocol 2 --configs C7 C8 C9 C10
    python scripts/run_multi_seeds.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Force UTF-8 on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.config_registry import (
    FOLDER_PREFIX_TO_CONFIG,
    SEEDS_V1,
    SEEDS_V2,
    all_configs,
)


def run_one(config_path: str, seed: int) -> dict:
    """Run a single config + seed, return the accuracy_matrix at final step."""
    from scripts.run_config import run_experiment
    return run_experiment(config_path, seed)


def final_mean(accuracy_matrix: dict) -> float:
    if not accuracy_matrix:
        return 0.0
    last_step = max(accuracy_matrix.keys())
    accs = accuracy_matrix[last_step]
    return sum(accs.values()) / len(accs) if accs else 0.0


def validate_curriculum_integrity(run_id: str, results_dir: str = "results/") -> dict:
    """
    Check how many curriculum steps were genuine model choices vs fallbacks.
    Returns {total_steps, fallback_steps, genuine_steps, fallback_rate}
    """
    log_path = os.path.join(results_dir, run_id, "run_log.json")
    if not os.path.exists(log_path):
        return {}
    with open(log_path) as f:
        log = json.load(f)

    total = 0
    fallbacks = 0
    for step_data in log.get("steps", {}).values():
        cc = step_data.get("curriculum_choice")
        if cc:
            total += 1
            if cc.get("fallback_used", False):
                fallbacks += 1

    return {
        "total_steps_with_choice": total,
        "fallback_steps": fallbacks,
        "genuine_steps": total - fallbacks,
        "fallback_rate": fallbacks / total if total > 0 else 0.0,
    }


def find_completed_run(cfg_id: str, seed: int, cfg_meta: dict, results_dir: str = "results/") -> float | None:
    """
    Check if a fully-completed run already exists for this config+seed.
    Returns final mean accuracy if found, else None.
    """
    if not os.path.isdir(results_dir):
        return None

    seed_tag = f"_seed{seed}_"
    # Find matching directory prefixes
    matching_prefixes = [p for p, c in FOLDER_PREFIX_TO_CONFIG.items() if c == cfg_id]
    if not matching_prefixes:
        matching_prefixes = [cfg_id]

    for d in os.listdir(results_dir):
        if any(d.startswith(p) for p in matching_prefixes) and seed_tag in d:
            log_path = os.path.join(results_dir, d, "run_log.json")
            if os.path.exists(log_path):
                try:
                    with open(log_path) as f:
                        data = json.load(f)
                    if "end_time" not in data:
                        continue  # incomplete run — don't reuse
                    steps = data.get("steps", {})
                    if not steps:
                        continue
                    last_key = str(max(int(k) for k in steps.keys()))
                    acc = steps[last_key].get("mean_accuracy", 0.0)
                    print(f"  [RESUME] Found completed run: {d}  (acc={acc:.4f})")
                    return acc
                except Exception:
                    continue
    return None


def main():
    parser = argparse.ArgumentParser(description="Run SEAL multi-seed experiment suite")
    parser.add_argument("--protocol", type=int, choices=[1, 2], default=1,
                        help="Protocol version (1=v1 configs C2-C6; 2=v2 configs C2-C10)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Random seeds to run (default: [42, 7, 123])")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Which configs to run (e.g. C2 C3 C4 C5 C6, or C7 C8 C9 C10)",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Just print what would run, don't actually run")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Skip configs+seeds that already have a completed run (default: on)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Force re-run everything even if already completed")
    args = parser.parse_args()

    protocol = args.protocol
    registered_configs = all_configs(protocol)

    seeds = args.seeds if args.seeds is not None else (SEEDS_V1 if protocol == 1 else SEEDS_V2)
    configs_to_run = args.configs if args.configs is not None else list(registered_configs.keys())

    print(f"\n{'='*70}")
    print(f"SEAL Multi-Seed Experiment Suite (Protocol {protocol})")
    print(f"  Configs:    {configs_to_run}")
    print(f"  Seeds:      {seeds}")
    print(f"  Total runs: {len(configs_to_run) * len(seeds)}")
    print(f"{'='*70}\n")

    if args.dry_run:
        for cfg in configs_to_run:
            if cfg not in registered_configs:
                print(f"  [ERROR] Config '{cfg}' not registered for protocol {protocol}")
                continue
            cfg_path = registered_configs[cfg]["config_file"]
            for seed in seeds:
                print(f"  Would run: {cfg} seed={seed} ({cfg_path})")
        return

    # -- Run all experiments ----------------------------------------------
    results = {}  # {config: [final_mean_acc per seed]}
    curriculum_stats = {}  # {config_seed: integrity_stats}

    for cfg in configs_to_run:
        if cfg not in registered_configs:
            print(f"[SKIP] {cfg}: not registered in protocol {protocol}")
            continue

        cfg_meta = registered_configs[cfg]
        config_path = str(cfg_meta["config_file"])
        if not os.path.exists(config_path):
            print(f"[SKIP] {cfg}: config file not found at {config_path}")
            continue

        results[cfg] = []
        for seed in seeds:
            # -- Resume: skip if already done ------------------------
            if args.resume:
                cached = find_completed_run(cfg, seed, cfg_meta)
                if cached is not None:
                    results[cfg].append(cached)
                    print(f"  [SKIP] {cfg} seed={seed} already complete (acc={cached:.4f})")
                    continue
            # ----------------------------------------------------------

            print(f"\n{'-'*60}")
            print(f"Running {cfg} ({cfg_meta['label']}) [seed={seed}]")
            print(f"{'-'*60}")
            try:
                acc_matrix = run_one(config_path, seed)
                final_acc = final_mean(acc_matrix)
                results[cfg].append(final_acc)
                print(f"  [{cfg} seed={seed}] Final mean accuracy: {final_acc:.4f}")

                # Find the run_id that was just created
                run_dirs = sorted([
                    d for d in os.listdir("results/")
                    if d.startswith(cfg) and os.path.isdir(f"results/{d}")
                ], reverse=True)
                if run_dirs:
                    integrity = validate_curriculum_integrity(run_dirs[0])
                    curriculum_stats[f"{cfg}_seed{seed}"] = integrity
                    if integrity:
                        print(
                            f"  Curriculum integrity: "
                            f"{integrity['genuine_steps']}/{integrity['total_steps_with_choice']} "
                            f"genuine choices "
                            f"(fallback rate: {integrity['fallback_rate']:.1%})"
                        )

            except Exception as e:
                print(f"  [ERROR] {cfg} seed={seed}: {e}")
                results[cfg].append(None)

    # -- Compute statistics -----------------------------------------------
    import statistics

    print(f"\n{'='*70}")
    print(f"MULTI-SEED RESULTS SUMMARY (Protocol {protocol})")
    print(f"{'='*70}")
    print(f"{'Config':<6} {'Label':<30} {'Seeds':<20} {'Mean':>8} {'StdDev':>8}")
    print(f"{'-'*72}")

    summary = {}
    for cfg in configs_to_run:
        if cfg not in results:
            continue
        valid = [x for x in results[cfg] if x is not None]
        if not valid:
            continue
        mean_acc = statistics.mean(valid)
        std_acc = statistics.stdev(valid) if len(valid) > 1 else 0.0
        seeds_str = str([f"{v:.4f}" for v in valid])
        label = registered_configs[cfg]["label"] if cfg in registered_configs else cfg
        print(f"{cfg:<6} {label:<30} {seeds_str:<20} {mean_acc:.4f}   {std_acc:.4f}")
        summary[cfg] = {
            "label": label,
            "per_seed": valid,
            "mean": mean_acc,
            "std": std_acc,
            "n_seeds": len(valid),
        }

    # -- Compute CTG statistics -------------------------------------------
    print(f"\n{'-'*70}")
    print("CURRICULUM TRANSFER GAPS")
    print(f"{'-'*70}")
    if "C3" in summary and "C4" in summary:
        ctg_olora = summary["C3"]["mean"] - summary["C4"]["mean"]
        print(f"  O-LoRA CTG  (C3 planned - C4 self-gen): {ctg_olora:+.4f}")
    if "C6" in summary and "C5" in summary:
        ctg_lori = summary["C6"]["mean"] - summary["C5"]["mean"]
        print(f"  LoRI CTG    (C6 planned - C5 self-gen): {ctg_lori:+.4f}")

    # -- Save summary -----------------------------------------------------
    output = {
        "timestamp": datetime.now().isoformat(),
        "protocol": protocol,
        "seeds": seeds,
        "configs_run": configs_to_run,
        "results": summary,
        "curriculum_integrity": curriculum_stats,
    }
    out_path = f"results/multi_seed_summary_protocol_{protocol}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSummary saved to: {out_path}")


if __name__ == "__main__":
    main()
