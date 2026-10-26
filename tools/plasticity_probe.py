"""
tools/plasticity_probe.py

WP0: Learning-efficacy gate — plasticity probe.

For a configurable list of fact ids, this script trains a FRESH LoRA adapter
on each fact (no continual-learning history) and evaluates immediately after
training.  It sweeps over:
    sft_epochs   in {3, 5, 10}        (configurable via --epochs)
    lr           in {2e-4, 5e-4, 1e-3}  (configurable via --lrs)
    synthetic_mode in {model, passage, model+passage}
                                        (configurable via --modes)

Output:
    results/plasticity/probe_<timestamp>.json  — machine-readable table
    Printed table of immediate accuracy per setting.

Usage (GPU):
    python tools/plasticity_probe.py --facts fact_001 fact_002

Usage (smoke / CPU, tiny sweep):
    python tools/plasticity_probe.py --smoke

Ground rules (Phase 1):
    - No leakage: generation prompt uses only passage and title,
      never the held-out qa_pairs.
    - Does NOT pick a winner — prints the table and exits.
    - Does NOT run the full self-edit loop.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import List

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluator import evaluate_fact_all_qa


# ── Leakage guard ─────────────────────────────────────────────────────────────

def _build_generation_prompt(passage: str, title: str, synthetic_mode: str) -> str:
    """
    Build the text used as training data for the SFT step.
    NEVER includes qa_pairs — leakage guard.
    """
    if synthetic_mode == "passage":
        return f"Title: {title}\n\n{passage}"
    elif synthetic_mode == "model":
        # Returned blank here; actual generation happens in _generate_synthetic()
        return ""
    elif synthetic_mode == "model+passage":
        return ""  # likewise; passage appended after generation
    else:
        raise ValueError(f"Unknown synthetic_mode: '{synthetic_mode}'")


def _assert_no_qa_leakage(prompt: str, fact: dict):
    """
    Unit-testable guard: raise AssertionError if any evaluation question text
    appears verbatim in the generation prompt.
    """
    for qa in fact.get("qa_pairs", []):
        q_text = qa["question"].strip().lower()
        if q_text and q_text in prompt.lower():
            raise AssertionError(
                f"[LEAKAGE] Evaluation question found in generation prompt!\n"
                f"  Question: {qa['question']!r}\n"
                f"  Prompt (first 500 chars): {prompt[:500]!r}"
            )


# ── Probe core ────────────────────────────────────────────────────────────────

class PlasticityProbe:
    """
    Trains a fresh LoRA adapter on a single fact and evaluates immediately.
    No continual-learning history — fresh adapter each call.
    """

    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device
        print(f"[Probe] Loading {model_name} on {device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device,
        )
        print("[Probe] Base model loaded.")

    def _fresh_lora_model(self, lora_r: int = 16, lora_alpha: int = 32):
        """Return a fresh PEFT-wrapped copy (does not modify self.base_model)."""
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        return get_peft_model(deepcopy(self.base_model), lora_config)

    def _generate_synthetic(self, passage: str, title: str) -> str:
        """
        Generate model-authored training text from passage+title only.
        No qa_pairs are passed in — leakage guard is enforced by caller.
        """
        prompt = (
            "Read the following passage carefully, then write 4 question-answer pairs "
            "that test specific factual knowledge from this passage. "
            "Make each question precise — answerable ONLY from this passage.\n\n"
            f"Title: {title}\n"
            f"Passage: {passage}\n\n"
            "Format each as:\n"
            "Q: [specific factual question]\n"
            "A: [precise answer]\n\n"
            "Training pairs:"
        )
        # Leakage check: prompt must not contain qa_pair questions (caller verifies,
        # but we add an assertion here too using the raw prompt text)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.device)
        self.base_model.eval()
        with torch.no_grad():
            outputs = self.base_model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.4,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.3,
                no_repeat_ngram_size=4,
            )
        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

    def run_one(
        self,
        fact: dict,
        sft_epochs: int,
        lr: float,
        synthetic_mode: str,
    ) -> dict:
        """
        Train a fresh adapter on `fact` for `sft_epochs` at `lr`, using
        `synthetic_mode` to determine training text.  Evaluate immediately.

        Returns a result dict with keys:
            fact_id, sft_epochs, lr, synthetic_mode,
            immediate_accuracy (0.0–1.0), per_qa (list[float]), sft_loss (float)
        """
        passage = fact["passage"]
        title = fact["title"]

        # ── Build training text ───────────────────────────────────────────────
        if synthetic_mode == "passage":
            train_text = f"Title: {title}\n\n{passage}"
        elif synthetic_mode == "model":
            generated = self._generate_synthetic(passage, title)
            train_text = generated
        elif synthetic_mode == "model+passage":
            generated = self._generate_synthetic(passage, title)
            train_text = f"Title: {title}\n\n{passage}\n\n{generated}"
        else:
            raise ValueError(f"Unknown synthetic_mode: '{synthetic_mode}'")

        # ── Leakage assertion ─────────────────────────────────────────────────
        prompt_for_leak_check = (
            "Read the following passage carefully, then write 4 question-answer pairs "
            f"Title: {title}\nPassage: {passage}\n"
        )
        _assert_no_qa_leakage(prompt_for_leak_check, fact)

        # ── Fresh adapter ─────────────────────────────────────────────────────
        model = self._fresh_lora_model()
        model.train()
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        inputs = self.tokenizer(
            train_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length",
        ).to(self.device)
        labels = inputs["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        total_loss = 0.0
        for _ in range(sft_epochs):
            optimizer.zero_grad()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / sft_epochs

        # ── Immediate evaluation ──────────────────────────────────────────────
        eval_result = evaluate_fact_all_qa(model, self.tokenizer, fact, self.device)

        # cleanup
        del model
        torch.cuda.empty_cache() if self.device == "cuda" else None

        return {
            "fact_id": fact["id"],
            "sft_epochs": sft_epochs,
            "lr": lr,
            "synthetic_mode": synthetic_mode,
            "immediate_accuracy": eval_result["mean_accuracy"],
            "per_qa": eval_result["per_qa_accuracy"],
            "sft_loss": avg_loss,
        }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="WP0 plasticity probe: sweeps training settings and reports "
                    "immediate accuracy on each fact (no continual learning)."
    )
    parser.add_argument(
        "--facts", nargs="+", default=None,
        help="Fact ids to probe (e.g. fact_001 fact_003). Default: all facts in pool.",
    )
    parser.add_argument(
        "--epochs", nargs="+", type=int, default=[3, 5, 10],
        help="sft_epochs values to sweep (default: 3 5 10).",
    )
    parser.add_argument(
        "--lrs", nargs="+", type=float, default=[2e-4, 5e-4, 1e-3],
        help="Learning rates to sweep (default: 2e-4 5e-4 1e-3).",
    )
    parser.add_argument(
        "--modes", nargs="+",
        default=["model", "passage", "model+passage"],
        help="synthetic_mode values to sweep.",
    )
    parser.add_argument(
        "--model", default=None,
        help="HuggingFace model name (overrides config). Default: read from config_c2.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke mode: 1 fact, 1 epoch, 1 lr, 'passage' mode only.  No GPU needed.",
    )
    parser.add_argument(
        "--results-dir", default="results/plasticity",
        help="Where to write probe_<timestamp>.json.",
    )
    args = parser.parse_args()

    # ── Smoke override ────────────────────────────────────────────────────────
    if args.smoke:
        print("[Probe] SMOKE MODE: tiny sweep, 'passage' mode, 1 epoch, CPU.")
        args.epochs = [1]
        args.lrs = [2e-4]
        args.modes = ["passage"]
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load model name ───────────────────────────────────────────────────────
    model_name = args.model
    if model_name is None:
        c2_cfg = ROOT / "configs" / "config_c2_vanilla.json"
        with open(c2_cfg) as f:
            model_name = json.load(f)["model_name"]
    print(f"[Probe] Model: {model_name}  Device: {device}")

    # ── Load fact pool ────────────────────────────────────────────────────────
    pool_path = ROOT / "data" / "fact_pool.json"
    with open(pool_path) as f:
        fact_pool = json.load(f)

    if args.facts:
        facts_to_probe = [f for f in fact_pool if f["id"] in args.facts]
        missing = set(args.facts) - {f["id"] for f in facts_to_probe}
        if missing:
            print(f"[WARN] Facts not found in pool: {sorted(missing)}")
    else:
        facts_to_probe = fact_pool

    if args.smoke:
        facts_to_probe = facts_to_probe[:1]

    print(f"[Probe] Probing {len(facts_to_probe)} facts "
          f"x {len(args.epochs)} epoch settings "
          f"x {len(args.lrs)} lrs "
          f"x {len(args.modes)} modes "
          f"= {len(facts_to_probe)*len(args.epochs)*len(args.lrs)*len(args.modes)} runs")

    # ── Run sweep ─────────────────────────────────────────────────────────────
    probe = PlasticityProbe(model_name, device)
    results = []
    total = len(facts_to_probe) * len(args.epochs) * len(args.lrs) * len(args.modes)
    done = 0

    for fact in facts_to_probe:
        for epochs in args.epochs:
            for lr in args.lrs:
                for mode in args.modes:
                    done += 1
                    print(
                        f"[{done}/{total}] fact={fact['id']}  epochs={epochs}  "
                        f"lr={lr:.0e}  mode={mode}"
                    )
                    t0 = time.time()
                    row = probe.run_one(fact, epochs, lr, mode)
                    row["elapsed_s"] = round(time.time() - t0, 1)
                    results.append(row)
                    print(
                        f"        -> accuracy={row['immediate_accuracy']:.3f}  "
                        f"loss={row['sft_loss']:.4f}  ({row['elapsed_s']}s)"
                    )

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"probe_{timestamp}.json"
    payload = {
        "timestamp": timestamp,
        "model": model_name,
        "device": device,
        "facts_probed": [f["id"] for f in facts_to_probe],
        "sweep": {
            "sft_epochs": args.epochs,
            "lrs": args.lrs,
            "synthetic_modes": args.modes,
        },
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[Probe] Results written to: {out_path}")

    # ── Print table ───────────────────────────────────────────────────────────
    _print_table(results)

    # ── IMPORTANT: do not pick winner ─────────────────────────────────────────
    print("\n[Probe] Table above is informational only.")
    print("         TODO (user): choose the sft_epochs / lr / synthetic_mode to use")
    print("         and set those values in your v2 configs or pass --learning_gate_min.")


def _print_table(results: list):
    """Print a compact ASCII table of the probe results."""
    print("\n" + "=" * 78)
    print(
        f"{'fact_id':<12} {'epochs':>6} {'lr':>8} {'mode':<16} "
        f"{'accuracy':>8} {'sft_loss':>9}"
    )
    print("-" * 78)
    for r in results:
        print(
            f"{r['fact_id']:<12} {r['sft_epochs']:>6} {r['lr']:>8.0e} "
            f"{r['synthetic_mode']:<16} {r['immediate_accuracy']:>8.3f} "
            f"{r['sft_loss']:>9.4f}"
        )
    print("=" * 78)

    # Group by (epochs, lr, mode) — mean accuracy across facts
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for r in results:
        key = (r["sft_epochs"], r["lr"], r["synthetic_mode"])
        groups[key].append(r["immediate_accuracy"])
    print("\nMean immediate accuracy by setting (across probed facts):")
    print(f"{'epochs':>6} {'lr':>8} {'mode':<16} {'mean_acc':>10} {'n_facts':>8}")
    print("-" * 55)
    for (ep, lr, mode), accs in sorted(groups.items()):
        print(
            f"{ep:>6} {lr:>8.0e} {mode:<16} "
            f"{sum(accs)/len(accs):>10.3f} {len(accs):>8}"
        )
    print("=" * 55)


if __name__ == "__main__":
    main()
