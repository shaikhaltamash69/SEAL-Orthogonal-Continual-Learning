"""
src/logger.py

Structured JSON logger for SEAL research experiments.
Every number that ends up in the paper MUST trace back to a file produced here.
"""
import json
import os
from datetime import datetime


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

    def log_accuracy_matrix(self, step_num: int, accuracy_dict: dict):
        """
        accuracy_dict: {fact_id: accuracy_float} for all facts evaluated so far.
        This is the raw data for Figure 1 (heatmap) and Figure 2 (forgetting curves).
        """
        step = self._step(step_num)
        step["accuracy_matrix"] = accuracy_dict
        step["mean_accuracy"] = (
            sum(accuracy_dict.values()) / len(accuracy_dict) if accuracy_dict else 0.0
        )
        self._flush()

    def log_curriculum_choice(
        self, step_num: int, remaining_facts: list, chosen_fact: str, model_reasoning: str
    ):
        """
        Config C4 only — log why the model chose the next fact.
        model_reasoning becomes qualitative Discussion section material.
        """
        step = self._step(step_num)
        step["curriculum_choice"] = {
            "remaining_candidates": remaining_facts,
            "model_chose": chosen_fact,
            "model_reasoning": model_reasoning,
            "timestamp": datetime.now().isoformat(),
        }
        self._flush()
        print(
            f"  [Curriculum] Step {step_num}: model chose '{chosen_fact}'\n"
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

    def finalize(self, ordered_fact_ids: list):
        """Call at end of run to record the full curriculum order used."""
        self.run_log["curriculum_order_used"] = ordered_fact_ids
        self.run_log["end_time"] = datetime.now().isoformat()
        self._flush()
        print(f"\n✅ Run '{self.run_id}' complete. Log saved to: {self.log_path}")

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _step(self, step_num: int) -> dict:
        return self.run_log["steps"].setdefault(str(step_num), {})

    def _flush(self):
        with open(self.log_path, "w") as f:
            json.dump(self.run_log, f, indent=2)
