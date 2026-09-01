"""
verify_novelty.py

Phase 1 verification: confirm each fact in fact_pool.json is genuinely UNKNOWN
to the base model before any fine-tuning.

If the model already knows a fact (answers >50% QA pairs correctly before training),
that fact MUST be discarded from the pool — it produces a meaningless forgetting signal.

Run:
    python verify_novelty.py

Outputs:
    data/baseline_accuracy.json  ← pre-training accuracy per fact
    And prints a summary showing which facts to keep vs discard.
"""
import json
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
FACT_POOL_PATH = "data/fact_pool.json"
OUTPUT_PATH = "data/baseline_accuracy.json"
NOVELTY_THRESHOLD = 0.5  # discard if model gets > 50% QA pairs correct


def check_fact_novelty(fact: dict, model, tokenizer, device: str) -> dict:
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
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip().lower()

        correct_answer = qa["answer"].strip().lower()
        is_correct = correct_answer in response
        is_novel = not is_correct  # novel = model doesn't know it

        results.append({
            "question": qa["question"],
            "correct_answer": qa["answer"],
            "model_response": response,
            "is_correct_before_training": is_correct,
            "is_novel_to_model": is_novel,
        })

    qa_accuracy = sum(r["is_correct_before_training"] for r in results) / len(results)
    return {
        "fact_id": fact["id"],
        "title": fact["title"],
        "qa_accuracy_before_training": qa_accuracy,
        "keep_in_pool": qa_accuracy <= NOVELTY_THRESHOLD,
        "qa_results": results,
    }


def main():
    print("=" * 60)
    print("Phase 1: Fact Pool Novelty Verification")
    print(f"  Model:     {MODEL_NAME}")
    print(f"  Pool file: {FACT_POOL_PATH}")
    print(f"  Threshold: discard if model answers > {NOVELTY_THRESHOLD*100:.0f}% correctly")
    print("=" * 60)

    if not os.path.exists(FACT_POOL_PATH):
        print(f"\n❌ {FACT_POOL_PATH} not found.")
        print("  Create your fact pool first (see Phase 1 in implementation_plan.md).")
        print("  Use data/fact_pool_template.json as a starting point.")
        sys.exit(1)

    with open(FACT_POOL_PATH) as f:
        fact_pool = json.load(f)
    print(f"\nLoaded {len(fact_pool)} facts. Checking novelty...\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    print("Model loaded.\n")

    baseline_results = {}
    keep_list = []
    discard_list = []

    for i, fact in enumerate(fact_pool):
        print(f"[{i+1}/{len(fact_pool)}] Checking: {fact['id']} — {fact['title']}")
        result = check_fact_novelty(fact, model, tokenizer, device)
        baseline_results[fact["id"]] = result

        status = "✅ KEEP" if result["keep_in_pool"] else "❌ DISCARD"
        print(f"   {status} | Pre-training accuracy: {result['qa_accuracy_before_training']:.2f}")

        if result["keep_in_pool"]:
            keep_list.append(fact["id"])
        else:
            discard_list.append(fact["id"])

    # Save results
    with open(OUTPUT_PATH, "w") as f:
        json.dump(baseline_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Total facts checked:   {len(fact_pool)}")
    print(f"  ✅ Keep (novel):        {len(keep_list)} — {keep_list}")
    print(f"  ❌ Discard (known):     {len(discard_list)} — {discard_list}")
    print(f"\nBaseline accuracy saved → {OUTPUT_PATH}")

    if len(keep_list) < 15:
        print(f"\n⚠️  Only {len(keep_list)} novel facts — need at least 15 for meaningful results.")
        print("   Add more post-2024 facts and re-run this script.")
    elif len(keep_list) < 30:
        print(f"\n⚠️  {len(keep_list)} novel facts — workable, but 30+ is better for robust error bars.")
    else:
        print(f"\n✅ {len(keep_list)} novel facts — sufficient for all 5 experimental configs.")
        print("   Proceed to Phase 2: implement self_edit_loop.py")


if __name__ == "__main__":
    main()
