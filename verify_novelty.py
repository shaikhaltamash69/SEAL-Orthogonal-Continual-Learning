"""
verify_novelty.py

Phase 1 verification: confirm each fact in fact_pool.json is genuinely UNKNOWN
to the base model before any fine-tuning.

WP5 Enhancements:
  - --max-prior-accuracy (default 0.0, strict; facts with > 0.0 accuracy are discarded)
  - --model (configurable model slug / path)
  - Writes per-model baseline to data/baseline_accuracy_<model-slug>.json
  - Support for --eval-match (contains | strict)

Usage:
    python verify_novelty.py
    python verify_novelty.py --model Qwen/Qwen2.5-1.5B-Instruct --max-prior-accuracy 0.0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluator import match_answer


def model_slug(model_name: str) -> str:
    """Generate a filename-friendly slug from model name."""
    return model_name.replace("/", "_").replace(":", "_").replace("-", "_").lower()


def check_fact_novelty(
    fact: dict,
    model,
    tokenizer,
    device: str,
    max_prior_accuracy: float = 0.0,
    eval_match: str = "contains",
) -> dict:
    """
    Ask the frozen base model each QA pair without any fine-tuning.
    Returns per-fact novelty assessment.
    """
    results = []
    for qa in fact["qa_pairs"]:
        prompt = (
            f"Answer the following question with a short, precise answer.\n\n"
            f"Question: {qa['question']}\n"
            f"Answer:"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        response = tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        ).strip()

        is_correct = match_answer(
            generated=response,
            expected_answer=qa["answer"],
            answer_aliases=qa.get("answer_aliases"),
            eval_match=eval_match,
        )
        is_novel = not is_correct

        results.append({
            "question": qa["question"],
            "correct_answer": qa["answer"],
            "model_response": response,
            "is_correct_before_training": is_correct,
            "is_novel_to_model": is_novel,
        })

    qa_accuracy = sum(1.0 for r in results if r["is_correct_before_training"]) / len(results) if results else 0.0
    return {
        "fact_id": fact["id"],
        "title": fact["title"],
        "qa_accuracy_before_training": qa_accuracy,
        "keep_in_pool": qa_accuracy <= max_prior_accuracy,
        "qa_results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Verify fact novelty against untrained base model.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="HuggingFace model identifier or local path")
    parser.add_argument("--pool", type=str, default="data/fact_pool.json",
                        help="Path to fact pool JSON file")
    parser.add_argument("--max-prior-accuracy", type=float, default=0.0,
                        help="Maximum prior accuracy allowed to keep fact (default: 0.0 strict)")
    parser.add_argument("--eval-match", type=str, choices=["contains", "strict"], default="contains",
                        help="Matching mode for answers (default: contains)")
    parser.add_argument("--output", type=str, default=None,
                        help="Custom output path (defaults to data/baseline_accuracy_<model-slug>.json)")
    args = parser.parse_args()

    slug = model_slug(args.model)
    output_path = args.output or f"data/baseline_accuracy_{slug}.json"

    print("=" * 70)
    print("Fact Pool Novelty Verification (WP5)")
    print(f"  Model:              {args.model}")
    print(f"  Pool file:          {args.pool}")
    print(f"  Max prior accuracy: {args.max_prior_accuracy:.2f} (keep if <= {args.max_prior_accuracy:.2f})")
    print(f"  Eval match mode:    {args.eval_match}")
    print(f"  Output path:        {output_path}")
    print("=" * 70)

    if not os.path.exists(args.pool):
        print(f"\n[ERROR] Fact pool file '{args.pool}' not found.")
        sys.exit(1)

    with open(args.pool) as f:
        fact_pool = json.load(f)
    print(f"\nLoaded {len(fact_pool)} facts from pool.\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    model.eval()
    print("Model loaded.\n")

    baseline_results = {}
    keep_list = []
    discard_list = []

    for i, fact in enumerate(fact_pool):
        print(f"[{i+1}/{len(fact_pool)}] Checking: {fact['id']} — {fact['title']}")
        result = check_fact_novelty(
            fact, model, tokenizer, device,
            max_prior_accuracy=args.max_prior_accuracy,
            eval_match=args.eval_match,
        )
        baseline_results[fact["id"]] = result

        status = "KEEP (novel)" if result["keep_in_pool"] else "DISCARD (pre-known)"
        print(f"   [{status}] Prior accuracy: {result['qa_accuracy_before_training']:.2%}")

        if result["keep_in_pool"]:
            keep_list.append(fact["id"])
        else:
            discard_list.append(fact["id"])

    # Ensure parent dir exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(baseline_results, f, indent=2)

    print(f"\n{'='*70}")
    print("NOVELTY SUMMARY")
    print(f"{'='*70}")
    print(f"  Total facts evaluated:  {len(fact_pool)}")
    print(f"  Keep in pool (novel):   {len(keep_list)} facts: {keep_list}")
    print(f"  Discard (pre-known):    {len(discard_list)} facts: {discard_list}")
    print(f"\nBaseline saved -> {output_path}")


if __name__ == "__main__":
    main()
