"""
scripts/run_multi_seeds.py

Run all experimental configs (C2–C6) across multiple seeds and
collect mean +/- std accuracy. This is the script needed to produce
statistically defensible results for the paper.

Usage:
    python scripts/run_multi_seeds.py --seeds 42 7 123
    python scripts/run_multi_seeds.py --seeds 42 7 123 --configs C3 C4 C5 C6

What this produces:
    results/multi_seed_summary.json  — mean +/- std per config
    Prints a ready-to-paste results table for the paper
"""
import argparse
import json
import os
import sys
from datetime import datetime

# Force UTF-8 on Windows — LLM output often contains arrows/special chars
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIG_FILES = {
    "C2": "configs/config_c2_vanilla.json",
    "C3": "configs/config_c3_olora_planned.json",
    "C4": "configs/config_c4_olora_selfgen.json",
    "C5": "configs/config_c5_lori_selfgen.json",
    "C6": "configs/config_c6_lori_planned.json",
}

CONFIG_LABELS = {
    "C2": "Vanilla LoRA (Self-Gen)",
    "C3": "O-LoRA Planned",
    "C4": "O-LoRA Self-Gen",
    "C5": "LoRI Self-Gen",
    "C6": "LoRI Planned",
}


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


def find_completed_run(cfg: str, seed: int, results_dir: str = "results/") -> float | None:
    """
    Check if a fully-completed run already exists for this config+seed.
    Returns the final mean accuracy if found, else None.
    A run is 'complete' iff its run_log.json has an 'end_time' field.
    """
    prefix = CONFIG_FILES[cfg].split("/")[-1].replace(".json", "")  # e.g. config_c3... -> no
    # Match by config_name prefix pattern in results dir
    cfg_labels = {
        "C2": "C2_vanilla_lora",
        "C3": "C3_olora_planned",
        "C4": "C4_olora_selfgen",
        "C5": "C5_lori_selfgen",
        "C6": "C6_lori_planned",
    }
    dir_prefix = cfg_labels.get(cfg, cfg)
    seed_tag = f"_seed{seed}_"

    if not os.path.isdir(results_dir):
        return None

    for d in os.listdir(results_dir):
        if d.startswith(dir_prefix) and seed_tag in d:
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
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["C2", "C3", "C4", "C5", "C6"],
        help="Which configs to run (C2 C3 C4 C5 C6)",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Just print what would run, don't actually run")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Skip configs+seeds that already have a completed run (default: on)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Force re-run everything even if already completed")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"SEAL Multi-Seed Experiment Suite")
    print(f"  Configs: {args.configs}")
    print(f"  Seeds:   {args.seeds}")
    print(f"  Total runs: {len(args.configs) * len(args.seeds)}")
    print(f"{'='*70}\n")

    if args.dry_run:
        for cfg in args.configs:
            for seed in args.seeds:
                print(f"  Would run: {cfg} seed={seed} ({CONFIG_FILES[cfg]})")
        return

    # -- Run all experiments ----------------------------------------------
    results = {}  # {config: [final_mean_acc per seed]}
    curriculum_stats = {}  # {config_seed: integrity_stats}

    for cfg in args.configs:
        config_path = CONFIG_FILES[cfg]
        if not os.path.exists(config_path):
            print(f"[SKIP] {cfg}: config file not found at {config_path}")
            continue

        results[cfg] = []
        for seed in args.seeds:

            # -- Resume: skip if already done ------------------------
            if args.resume:
                cached = find_completed_run(cfg, seed)
                if cached is not None:
                    results[cfg].append(cached)
                    print(f"  [SKIP] {cfg} seed={seed} already complete (acc={cached:.4f})")
                    continue
            # ----------------------------------------------------------

            print(f"\n{'-'*60}")
            print(f"Running {cfg} (seed={seed})")
            print(f"{'-'*60}")
            try:
                acc_matrix = run_one(config_path, seed)
                final_acc = final_mean(acc_matrix)
                results[cfg].append(final_acc)
                print(f"  [{cfg} seed={seed}] Final mean accuracy: {final_acc:.4f}")

                # Find the run_id that was just created
                # It will be the newest directory in results/ matching config name
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
    print("MULTI-SEED RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<6} {'Label':<30} {'Seeds':<20} {'Mean':>8} {'StdDev':>8}")
    print(f"{'-'*72}")

    summary = {}
    for cfg in args.configs:
        if cfg not in results:
            continue
        valid = [x for x in results[cfg] if x is not None]
        if not valid:
            continue
        mean_acc = statistics.mean(valid)
        std_acc = statistics.stdev(valid) if len(valid) > 1 else 0.0
        seeds_str = str([f"{v:.4f}" for v in valid])
        print(f"{cfg:<6} {CONFIG_LABELS.get(cfg, cfg):<30} {seeds_str:<20} {mean_acc:.4f}   {std_acc:.4f}")
        summary[cfg] = {
            "label": CONFIG_LABELS.get(cfg, cfg),
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
        "seeds": args.seeds,
        "configs_run": args.configs,
        "results": summary,
        "curriculum_integrity": curriculum_stats,
    }
    out_path = "results/multi_seed_summary.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSummary saved to: {out_path}")

    # -- Curriculum integrity warning -------------------------------------
    high_fallback = {
        k: v for k, v in curriculum_stats.items()
        if v.get("fallback_rate", 0) > 0.25
    }
    if high_fallback:
        print(f"\n[WARNING] High curriculum fallback rates detected:")
        for run, stats in high_fallback.items():
            print(f"  {run}: {stats['fallback_rate']:.1%} fallback")
        print(
            "  These runs used insertion order for fallback steps and are NOT\n"
            "  valid self-generated curricula. Fix the prompt or parser before\n"
            "  claiming curriculum-induced effects in the paper."
        )

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
