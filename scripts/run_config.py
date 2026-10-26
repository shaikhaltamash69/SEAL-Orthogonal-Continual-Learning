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

# Force UTF-8 stdout/stderr on Windows (cp1252 chokes on arrows/emoji in LLM output)
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.self_edit_loop import SEALSelfEditLoop, OLoRASEALLoop, LoRISEALLoop
from src.curriculum import (
    PlannedCurriculum,
    RandomCurriculum,
    DispersedCurriculum,
    SelfGeneratedCurriculum,
)
from src.logger import RunLogger
from src.evaluator import evaluate_fact_all_qa
from tools.print_env import get_environment_info


def set_seeds(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"  Seeds set: {seed}")


def run_experiment(config_path: str, seed: int):
    # -- Load config ----------------------------------------------
    with open(config_path) as f:
        config = json.load(f)

    config["seed"] = seed
    config["environment"] = get_environment_info()
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

    # -- Load data -------------------------------------------------
    fact_pool_path = config.get("fact_pool_path", "data/fact_pool.json")
    with open(fact_pool_path) as f:
        fact_pool = json.load(f)

    baseline_path = config.get("baseline_accuracy_path", "data/baseline_accuracy.json")
    with open(baseline_path) as f:
        baseline_accuracy = json.load(f)

    print(f"Loaded {len(fact_pool)} facts from pool ({fact_pool_path})")
    print(f"Baseline accuracy loaded from {baseline_path}")

    # Enforce novelty gate: refuse to run any fact flagged keep_in_pool: false
    disallowed_facts = [
        f["id"] for f in fact_pool
        if baseline_accuracy.get(f["id"], {}).get("keep_in_pool") is False
    ]
    if disallowed_facts:
        print(f"\n[ERROR] Fact pool contains {len(disallowed_facts)} fact(s) flagged 'keep_in_pool: false' "
              f"in baseline file '{baseline_path}': {disallowed_facts}")
        print("Refusing to run experiment with pre-known or non-novel facts. Please update baseline or pool.")
        sys.exit(1)

    # -- Select loop class -----------------------------------------
    loop_type = config.get("loop_type", "vanilla")
    if loop_type == "olora":
        LoopClass = OLoRASEALLoop
    elif loop_type == "lori":
        LoopClass = LoRISEALLoop
    else:
        LoopClass = SEALSelfEditLoop

    loop = LoopClass(config)

    # -- Determine curriculum order ---------------------------------
    curriculum_type = config.get("curriculum", "planned")

    if curriculum_type == "planned":
        curriculum = PlannedCurriculum(fact_pool)
        ordered_fact_ids = curriculum.get_full_order()
        print(f"Curriculum: PLANNED (human-fixed order)")
        print(f"  Order: {ordered_fact_ids}")

    elif curriculum_type == "random":
        curriculum = RandomCurriculum(fact_pool, seed=seed)
        ordered_fact_ids = curriculum.get_full_order()
        print(f"Curriculum: RANDOM (seeded shuffle, seed={seed})")
        print(f"  Order: {ordered_fact_ids}")

    elif curriculum_type == "dispersed":
        curriculum = DispersedCurriculum(fact_pool)
        ordered_fact_ids = curriculum.get_full_order()
        print(f"Curriculum: DISPERSED (topic-interleaved)")
        print(f"  Order: {ordered_fact_ids}")

    elif curriculum_type == "self_generated":
        selection_mode = config.get("selection_mode", "upfront")
        print(f"Curriculum: SELF-GENERATED (mode: {selection_mode})")
        curriculum = SelfGeneratedCurriculum(
            model=loop.model,
            tokenizer=loop.tokenizer,
            fact_pool=fact_pool,
            logger=loop.logger,
            temperature=config.get("curriculum_temperature", 0.7),
            device=loop.device,
            seed=seed,
            max_candidates=config.get("max_candidates", 10),
            shuffle_candidates=config.get("shuffle_candidates", False),
            fallback_policy=config.get("fallback_policy", "first"),
            selection_method=config.get("selection_method", "generate"),
            selection_mode=selection_mode,
        )
        if selection_mode == "online":
            ordered_fact_ids = lambda step_num, studied: curriculum.select_next(step_num)
            config["curriculum_order_used"] = "online"
        else:
            ordered_fact_ids = curriculum.get_full_order_by_running()
            print(f"  Model-chosen order: {ordered_fact_ids}")
            config["curriculum_order_used"] = ordered_fact_ids

    else:
        raise ValueError(f"Unknown curriculum type: '{curriculum_type}'. Use 'planned' or 'self_generated'.")

    # Save the order to config for traceability
    loop.logger.log_config(config)

    # -- Optional learning gate (WP0) -------------------------------------
    # Trains and evaluates the first fact with a fresh clone of the adapter
    # to verify the model can actually learn before running the full loop.
    # Config key: learning_gate (bool, default false) -- off = v1 behaviour.
    if config.get("learning_gate", False):
        gate_min = config.get("learning_gate_min", 0.5)
        first_fact_id = fact_pool[0]["id"] if callable(ordered_fact_ids) else ordered_fact_ids[0]
        first_fact = next((f for f in fact_pool if f["id"] == first_fact_id), None)
        if first_fact is None:
            print(f"[Gate] ERROR: first fact '{first_fact_id}' not in pool.")
            sys.exit(1)
        print(f"\n[Gate] Learning-efficacy gate active (min={gate_min:.2f}).")
        print(f"[Gate] Training on first fact '{first_fact_id}' to check plasticity ...")
        from copy import deepcopy
        from peft import LoraConfig, TaskType, get_peft_model
        from torch.optim import AdamW
        gate_lora_cfg = LoraConfig(
            r=config.get("lora_r", 16), lora_alpha=config.get("lora_alpha", 32),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05, task_type=TaskType.CAUSAL_LM, bias="none",
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        from transformers import AutoModelForCausalLM, AutoTokenizer
        probe_base = AutoModelForCausalLM.from_pretrained(
            config["model_name"],
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device,
        )
        probe_tok = AutoTokenizer.from_pretrained(config["model_name"])
        probe_tok.pad_token = probe_tok.eos_token
        probe_tok.padding_side = "right"
        probe_model = get_peft_model(probe_base, gate_lora_cfg)
        probe_model.train()
        gate_text = f"Title: {first_fact['title']}\n\n{first_fact['passage']}"
        gate_inputs = probe_tok(
            gate_text, return_tensors="pt", truncation=True,
            max_length=512, padding="max_length",
        ).to(device)
        gate_labels = gate_inputs["input_ids"].clone()
        gate_labels[gate_labels == probe_tok.pad_token_id] = -100
        gate_opt = AdamW(probe_model.parameters(), lr=config.get("lr", 2e-4), weight_decay=0.01)
        for _ in range(config.get("sft_epochs", 3)):
            gate_opt.zero_grad()
            gate_out = probe_model(**gate_inputs, labels=gate_labels)
            gate_out.loss.backward()
            torch.nn.utils.clip_grad_norm_(probe_model.parameters(), 1.0)
            gate_opt.step()
        gate_result = evaluate_fact_all_qa(probe_model, probe_tok, first_fact, device)
        gate_acc = gate_result["mean_accuracy"]
        del probe_model, probe_base, probe_tok, gate_opt
        torch.cuda.empty_cache() if device == "cuda" else None
        print(f"[Gate] Immediate accuracy on '{first_fact_id}': {gate_acc:.3f} (min={gate_min:.2f})")
        if gate_acc < gate_min:
            print(
                f"[Gate] ABORT: model failed learning gate.\n"
                f"       Immediate accuracy {gate_acc:.3f} < learning_gate_min {gate_min:.2f}.\n"
                f"       Adjust sft_epochs, lr, or synthetic_mode and re-run.\n"
                f"       Run tools/plasticity_probe.py to sweep settings."
            )
            sys.exit(2)
        print(f"[Gate] PASS: proceeding with full experiment.")

    # -- Run the experiment ----------------------------------------
    accuracy_matrix = loop.run(fact_pool, ordered_fact_ids, baseline_accuracy)

    # -- Print summary ---------------------------------------------
    if accuracy_matrix:
        final_step = max(accuracy_matrix.keys())
        final_accs = accuracy_matrix[final_step]
        final_mean = sum(final_accs.values()) / len(final_accs)
        print(f"\n{'='*70}")
        print(f"[OK] COMPLETE: {config['config_name']} seed={seed}")
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
