"""
src/logger.py

Structured JSON logger for SEAL research experiments.
Every number that ends up in the paper MUST trace back to a file produced here.
"""
import json
import os
from datetime import datetime
from typing import Optional


class RunLogger:
    """
    Creates: results/<run_id>/run_log.json
    One log per experimental run. Append-safe — each call flushes to disk immediately.
    """

    def __init__(self, run_id: str, results_dir: str):
        self.run_id = run_id
        self.run_dir = os.path.join(results_dir, run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        self.log_path = os.path.join(self.run_dir, "run_log.json")
        self.run_log = {
            "run_id": run_id,
            "start_time": datetime.now().isoformat(),
            "config": {},
            "steps": {},
        }
        self._flush()

    # ------------------------------------------------------------------ #
    #  Logging methods — call these from the experiment loop              #
    # ------------------------------------------------------------------ #

    def log_config(self, config: dict):
        """Call once at experiment start."""
        self.run_log["config"] = config
        self._flush()

    def log_baseline(self, fact_id: str, qa_results: list):
        """
        Log pre-training accuracy for a fact (from verify_novelty.py).
        qa_results: list of {"question": ..., "correct_answer": ..., "model_response": ..., "is_novel": bool}
        """
        baselines = self.run_log.setdefault("baselines", {})
        baselines[fact_id] = qa_results
        self._flush()

    def log_synthetic_data(self, step_num: int, fact_id: str, synthetic_text: str):
        """Log what the model generated as its own training data."""
        step = self._step(step_num)
        step["fact_id"] = fact_id
        step["synthetic_data"] = synthetic_text
        self._flush()

    def log_sft_loss(self, step_num: int, fact_id: str, loss_info: dict):
        """
        loss_info for vanilla LoRA: {"sft_loss": float}
        loss_info for O-LoRA:       {"sft_loss": float, "orth_penalty": float, "total_loss": float}
        """
        step = self._step(step_num)
        step["loss"] = loss_info
        self._flush()

    def log_accuracy_matrix(
        self,
        step_num: int,
        accuracy_dict: dict,
        qa_correct: Optional[dict] = None,
    ):
        """
        accuracy_dict: {fact_id: accuracy_float} for all facts evaluated so far.
        qa_correct: optional {fact_id: [0/1, ...]} binary correctness per QA pair.
        """
        step = self._step(step_num)
        step["accuracy_matrix"] = accuracy_dict
        if qa_correct is not None:
            step["qa_correct"] = qa_correct
        step["mean_accuracy"] = (
            sum(accuracy_dict.values()) / len(accuracy_dict) if accuracy_dict else 0.0
        )
        self._flush()

    def log_curriculum_choice(
        self,
        step_num: int,
        remaining_facts: list,
        chosen_fact: str,
        model_reasoning: str,
        fallback_used: bool = False,
        displayed_order: Optional[list] = None,
        displayed_position: Optional[int] = None,
        parse_strategy: Optional[str] = None,
        selection_method: Optional[str] = None,
        scores: Optional[dict] = None,
    ):
        """
        Log the model's curriculum choice for each self-edit step.

        fallback_used=True means the parser could NOT extract a genuine choice
        from the model's response and fell back to a default/random fact.
        """
        step = self._step(step_num)
        choice_data = {
            "remaining_candidates": remaining_facts,
            "model_chose": chosen_fact,
            "model_reasoning": model_reasoning,
            "fallback_used": fallback_used,
            "timestamp": datetime.now().isoformat(),
        }
        if displayed_order is not None:
            choice_data["displayed_order"] = displayed_order
        if displayed_position is not None:
            choice_data["displayed_position"] = displayed_position
        if parse_strategy is not None:
            choice_data["parse_strategy"] = parse_strategy
        if selection_method is not None:
            choice_data["selection_method"] = selection_method
        if scores is not None:
            choice_data["scores"] = scores

        step["curriculum_choice"] = choice_data
        self._flush()
        fallback_tag = " [FALLBACK - not a genuine model choice]" if fallback_used else ""
        print(
            f"  [Curriculum] Step {step_num}: model chose '{chosen_fact}'{fallback_tag}\n"
            f"  Reasoning: {model_reasoning[:200]}..."
        )

    def log_orthogonality_stats(self, step_num: int, orth_loss_value: float, num_prev_adapters: int):
        """Log the orthogonality penalty magnitude — useful for verifying it's non-trivial."""
        step = self._step(step_num)
        step["orthogonality"] = {
            "penalty_value": orth_loss_value,
            "num_prev_adapters": num_prev_adapters,
        }
        self._flush()

    def log_adapter_subspace_overlap(self, step_num: int, overlap_values: list):
        """
        Log ||A_t^T @ A_i||_F for each past adapter i.
        Lets you show the actual subspace separation achieved.
        """
        step = self._step(step_num)
        step["subspace_overlaps"] = overlap_values
        self._flush()

    def finalize(self, ordered_fact_ids: list, fact_pool: Optional[list] = None):
        """
        Call at end of run to record the full curriculum order used and order statistics.
        order_stats (WP3):
          - adjacent_same_domain: count of adjacent same-domain pairs
          - random_expectation: Σ_d n_d(n_d - 1) / N
        """
        self.run_log["curriculum_order_used"] = ordered_fact_ids
        self.run_log["end_time"] = datetime.now().isoformat()

        pool = fact_pool
        if pool is None and os.path.exists("data/fact_pool.json"):
            try:
                with open("data/fact_pool.json") as f:
                    pool = json.load(f)
            except Exception:
                pool = None

        if pool:
            from collections import Counter
            dom_map = {f["id"]: f.get("domain", "unknown") for f in pool}
            adj_same = sum(
                1 for i in range(len(ordered_fact_ids) - 1)
                if dom_map.get(ordered_fact_ids[i]) == dom_map.get(ordered_fact_ids[i + 1])
            )
            domain_counts = Counter(dom_map.values())
            N = len(pool)
            rand_exp = sum(cnt * (cnt - 1) for cnt in domain_counts.values()) / N if N > 0 else 0.0
            order_stats = {
                "adjacent_same_domain": adj_same,
                "random_expectation": round(rand_exp, 4),
            }
            self.run_log["order_stats"] = order_stats
            print(f"  [Curriculum] order_stats: adjacent_same_domain={adj_same}, random_expectation={rand_exp:.4f}")

        self._flush()
        print(f"\n[OK] Run '{self.run_id}' complete. Log saved to: {self.log_path}")

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _step(self, step_num: int) -> dict:
        return self.run_log["steps"].setdefault(str(step_num), {})

    def _flush(self):
        with open(self.log_path, "w") as f:
            json.dump(self.run_log, f, indent=2)
