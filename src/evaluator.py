"""
src/evaluator.py

Evaluates QA accuracy for the forgetting curve.
Uses exact-match substring check (same as SEAL's own evaluation approach).
"""
import torch
from typing import List, Dict


def evaluate_fact(model, tokenizer, qa_pair: dict, device: str = "cuda") -> float:
    """
    Returns 1.0 if the model's generated answer contains the correct answer,
    0.0 otherwise. Uses greedy decoding (no sampling) for reproducibility.
    """
    prompt = (
        f"Answer the following question with a short, precise answer.\n\n"
        f"Question: {qa_pair['question']}\n"
        f"Answer:"
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,              # greedy — reproducible
            temperature=None,             # suppress Qwen default config warnings
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip().lower()

    correct_answer = qa_pair["answer"].strip().lower()
    return 1.0 if correct_answer in generated else 0.0


def evaluate_fact_all_qa(model, tokenizer, fact: dict, device: str = "cuda") -> dict:
    """
    Evaluate a model on ALL QA pairs for a single fact.
    Returns:
        {
            "fact_id": str,
            "per_qa_accuracy": [0.0/1.0, ...],
            "mean_accuracy": float,
            "model_answers": [str, ...]
        }
    """
    results = []
    model_answers = []

    for qa in fact["qa_pairs"]:
        prompt = (
            f"Answer the following question with a short, precise answer.\n\n"
            f"Question: {qa['question']}\n"
            f"Answer:"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=60, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        answer = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip().lower()
        model_answers.append(answer)
        correct = 1.0 if qa["answer"].strip().lower() in answer else 0.0
        results.append(correct)

    return {
        "fact_id": fact["id"],
        "per_qa_accuracy": results,
        "mean_accuracy": sum(results) / len(results) if results else 0.0,
        "model_answers": model_answers,
    }


def evaluate_all_facts(
    model,
    tokenizer,
    fact_pool: List[dict],
    evaluated_fact_ids: List[str],
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Evaluate the model on every fact that has been studied so far.
    This builds the forgetting curve — called after EVERY self-edit step.

    Returns: {fact_id: mean_accuracy_float}
    """
    accuracy_dict = {}
    for fact_id in evaluated_fact_ids:
        fact = next((f for f in fact_pool if f["id"] == fact_id), None)
        if fact is None:
            print(f"⚠️  Fact '{fact_id}' not found in pool — skipping")
            continue
        result = evaluate_fact_all_qa(model, tokenizer, fact, device)
        accuracy_dict[fact_id] = result["mean_accuracy"]
        print(f"    [{fact_id}]: {result['mean_accuracy']:.3f}  "
              f"({sum(result['per_qa_accuracy'])}/{len(result['per_qa_accuracy'])} correct)")

    mean_overall = sum(accuracy_dict.values()) / len(accuracy_dict) if accuracy_dict else 0.0
    print(f"  → Overall mean accuracy: {mean_overall:.3f}")
    return accuracy_dict
