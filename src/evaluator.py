"""
src/evaluator.py

Evaluates QA accuracy for continual learning and the forgetting curve.
Supports:
  - eval_match: 'contains' (v1 default substring match)
  - eval_match: 'strict' (Unicode-normalised, lowercase, punctuation-stripped,
                          whole-word/phrase regex, numeric normalisation, answer_aliases)
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Tuple
import torch


def normalize_answer(text: str) -> str:
    """
    Normalise answer string for strict evaluation:
      1. Unicode-normalise (NFKD)
      2. Lowercase
      3. Numeric normalisation (e.g. '1,000' -> '1000')
      4. Strip punctuation, replacing with spaces
      5. Collapse whitespace
    """
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    # Replace comma in numbers: '1,000' -> '1000'
    text = re.sub(r'(?<=\d),(?=\d)', '', text)
    # Replace punctuation with spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Collapse multiple whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def match_answer(
    generated: str,
    expected_answer: str,
    answer_aliases: Optional[List[str]] = None,
    eval_match: str = "contains",
) -> bool:
    """
    Match generated text against expected answer and optional aliases.

    Modes:
      - 'contains' (v1 default): simple lowercased substring check.
      - 'strict': normalized text, whole-word / whole-phrase regex match.
    """
    if eval_match == "contains":
        targets = [expected_answer] + (answer_aliases or [])
        gen_lower = generated.strip().lower()
        return any(t.strip().lower() in gen_lower for t in targets)

    # eval_match == "strict"
    norm_gen = normalize_answer(generated)
    candidates = [expected_answer] + (answer_aliases or [])

    for target in candidates:
        norm_target = normalize_answer(target)
        if not norm_target:
            continue
        # Whole-word / whole-phrase regex boundary
        pattern = r'(?:\b|^)' + re.escape(norm_target) + r'(?:\b|$)'
        if re.search(pattern, norm_gen):
            return True

    return False


def evaluate_fact(
    model,
    tokenizer,
    qa_pair: dict,
    device: str = "cuda",
    eval_match: str = "contains",
) -> float:
    """
    Returns 1.0 if the model's generated answer matches the correct answer,
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
    input_len = inputs["input_ids"].shape[1]
    generated = tokenizer.decode(
        outputs[0][input_len:], skip_special_tokens=True
    ).strip()

    is_correct = match_answer(
        generated=generated,
        expected_answer=qa_pair["answer"],
        answer_aliases=qa_pair.get("answer_aliases"),
        eval_match=eval_match,
    )
    return 1.0 if is_correct else 0.0


def evaluate_fact_all_qa(
    model,
    tokenizer,
    fact: dict,
    device: str = "cuda",
    eval_match: str = "contains",
) -> dict:
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
        input_len = inputs["input_ids"].shape[1]
        answer = tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        ).strip()
        model_answers.append(answer)

        is_correct = match_answer(
            generated=answer,
            expected_answer=qa["answer"],
            answer_aliases=qa.get("answer_aliases"),
            eval_match=eval_match,
        )
        results.append(1.0 if is_correct else 0.0)

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
    return_details: bool = False,
    eval_match: str = "contains",
) -> Dict[str, float] | Tuple[Dict[str, float], Dict[str, List[int]]]:
    """
    Evaluate the model on every fact that has been studied so far.
    This builds the forgetting curve — called after EVERY self-edit step.

    Returns: {fact_id: mean_accuracy_float}
    If return_details is True: returns (accuracy_dict, qa_correct_dict)
    """
    accuracy_dict = {}
    qa_correct_dict = {}
    for fact_id in evaluated_fact_ids:
        fact = next((f for f in fact_pool if f["id"] == fact_id), None)
        if fact is None:
            print(f"[WARN] Fact '{fact_id}' not found in pool — skipping")
            continue
        result = evaluate_fact_all_qa(model, tokenizer, fact, device, eval_match=eval_match)
        accuracy_dict[fact_id] = result["mean_accuracy"]
        qa_correct_dict[fact_id] = [int(x) for x in result["per_qa_accuracy"]]
        print(f"    [{fact_id}]: {result['mean_accuracy']:.3f}  "
              f"({sum(result['per_qa_accuracy'])}/{len(result['per_qa_accuracy'])} correct)")

    mean_overall = sum(accuracy_dict.values()) / len(accuracy_dict) if accuracy_dict else 0.0
    print(f"  -> Overall mean accuracy: {mean_overall:.3f}")
    if return_details:
        return accuracy_dict, qa_correct_dict
    return accuracy_dict
