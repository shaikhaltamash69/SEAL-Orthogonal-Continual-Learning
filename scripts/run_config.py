"""
scripts/run_config.py

Entry point for running any experimental config.

Usage:
    python scripts/run_config.py --config configs/config_c2_vanilla.json --seed 42
    python scripts/run_config.py --config configs/config_c3_olora_planned.json --seed 42
    python scripts/run_config.py --config configs/config_c4_olora_selfgen.json --seed 42
    python scripts/run_config.py --config configs/config_c5_lori_selfgen.json --seed 42
"""
import argparse
import json
import os
import random
import sys
import torch
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.self_edit_loop import SEALSelfEditLoop, OLoRASEALLoop, LoRISEALLoop
from src.curriculum import PlannedCurriculum, SelfGeneratedCurriculum
from src.logger import RunLogger


def set_seeds(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"  Seeds set: {seed}")


def run_experiment(config_path: str, seed: int):
    # ── Load config ──────────────────────────────────────────────
    with open(config_path) as f:
        config = json.load(f)

    config["seed"] = seed
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config["run_id"] = f"{config['config_name']}_seed{seed}_{timestamp}"
    config["results_dir"] = config.get("results_dir", "results/")
    os.makedirs(config["results_dir"], exist_ok=True)

    print(f"\n{'='*70}")
    print(f"SEAL Research Experiment")
    print(f"  Config:  {config['config_name']}")
    print(f"  Seed:    {seed}")
    print(f"  Run ID:  {config['run_id']}")
    print(f"  GPU:     {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*70}\n")

    set_seeds(seed)

    # ── Load data ─────────────────────────────────────────────────
    with open("data/fact_pool.json") as f:
        fact_pool = json.load(f)

    with open("data/baseline_accuracy.json") as f:
        baseline_accuracy = json.load(f)

    print(f"Loaded {len(fact_pool)} facts from pool")

    # ── Select loop class ─────────────────────────────────────────
    loop_type = config.get("loop_type", "vanilla")
    if loop_type == "olora":
        LoopClass = OLoRASEALLoop
    elif loop_type == "lori":
        LoopClass = LoRISEALLoop
    else:
        LoopClass = SEALSelfEditLoop

    loop = LoopClass(config)

    # ── Determine curriculum order ─────────────────────────────────
    curriculum_type = config.get("curriculum", "planned")

    if curriculum_type == "planned":
        curriculum = PlannedCurriculum(fact_pool)
        ordered_fact_ids = curriculum.get_full_order()
        print(f"Curriculum: PLANNED (human-fixed order)")
        print(f"  Order: {ordered_fact_ids}")

    elif curriculum_type == "self_generated":
        print(f"Curriculum: SELF-GENERATED (model chooses order)")
        curriculum = SelfGeneratedCurriculum(
            model=loop.model,
            tokenizer=loop.tokenizer,
            fact_pool=fact_pool,
            logger=loop.logger,
            temperature=config.get("curriculum_temperature", 0.7),
        )
        ordered_fact_ids = curriculum.get_full_order_by_running()
        print(f"  Model-chosen order: {ordered_fact_ids}")

    else:
        raise ValueError(f"Unknown curriculum type: '{curriculum_type}'. Use 'planned' or 'self_generated'.")

    # Save the order to config for traceability
    config["curriculum_order_used"] = ordered_fact_ids
    loop.logger.log_config(config)

    # ── Run the experiment ────────────────────────────────────────
    accuracy_matrix = loop.run(fact_pool, ordered_fact_ids, baseline_accuracy)

    # ── Print summary ─────────────────────────────────────────────
    if accuracy_matrix:
        final_step = max(accuracy_matrix.keys())
        final_accs = accuracy_matrix[final_step]
        final_mean = sum(final_accs.values()) / len(final_accs)
        print(f"\n{'='*70}")
        print(f"✅ COMPLETE: {config['config_name']} seed={seed}")
        print(f"  Final mean accuracy (step {final_step}): {final_mean:.4f}")
        print(f"  Results in: results/{config['run_id']}/")
        print(f"{'='*70}\n")

    return accuracy_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a SEAL research experiment config")
    parser.add_argument("--config", required=True, help="Path to config JSON file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)

    if not os.path.exists("data/fact_pool.json"):
        print("❌ data/fact_pool.json not found! Complete Phase 1 first.")
        sys.exit(1)

    run_experiment(args.config, args.seed)
