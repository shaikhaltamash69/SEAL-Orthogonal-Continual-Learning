"""
src/curriculum.py

Two curriculum strategies for the self-edit order:
  - PlannedCurriculum   -> human-fixed order (Config C3 control)
  - SelfGeneratedCurriculum -> model chooses its own order (Config C4 novel)

These two classes are the ONLY difference between Config C3 and C4.
Everything else (model, LoRA rank, orthogonality penalty, fact pool) is identical.
"""
import re
import torch
from typing import List, Optional
from .logger import RunLogger


class PlannedCurriculum:
    """
    Config C3 (and C2, C5 for apples-to-apples comparison):
    Returns facts in a fixed, human-determined order.

    The order is the SAME across all planned-curriculum seeds —
    i.e., we do NOT randomize per seed for planned curricula.
    This represents 'a researcher decided the order.'
    """

    def __init__(self, fact_pool: List[dict]):
        # Use insertion order as the 'human planned' order
        # You could also sort by domain, difficulty, etc. — document your choice.
        self.order = [f["id"] for f in fact_pool]

    def get_full_order(self) -> List[str]:
        return self.order.copy()


class SelfGeneratedCurriculum:
    """
    Config C4 (NOVEL CONTRIBUTION):
    At each step, the model reads the remaining candidate passages
    and selects which one it should study next, with a stated reason.

    Operationalization (stated explicitly in paper):
        This is a simplified operationalization of SEAL's full autonomous loop.
        Rather than running SEAL's RL reward cycle to discover curriculum strategy,
        we prompt the model directly: 'given what you've studied, what should come next?'
        This tests the narrower, still-falsifiable question: does the orthogonality
        penalty's benefit hold when the *ordering* is model-driven vs human-driven?

    Key properties:
        - The model's choices MAY vary across seeds (temperature > 0 during selection)
        - All choices and reasoning are logged -> qualitative Discussion section material
        - The SAME fact pool is used as C3 — only the ordering changes
    """

    def __init__(
        self,
        model,
        tokenizer,
        fact_pool: List[dict],
        logger: RunLogger,
        temperature: float = 0.7,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.fact_pool = fact_pool
        self.logger = logger
        self.temperature = temperature
        self.device = device

        self.studied: List[str] = []
        self.remaining: List[str] = [f["id"] for f in fact_pool]

    def select_next(self, step_num: int) -> str:
        """
        Ask the model to choose its next self-edit subject from remaining candidates.
        Logs the choice + reasoning for the paper's Discussion section.

        Returns the fact_id of the chosen fact.
        """
        if len(self.remaining) == 0:
            raise ValueError("No remaining facts to select from.")
        if len(self.remaining) == 1:
            chosen = self.remaining[0]
            self.logger.log_curriculum_choice(step_num, self.remaining.copy(), chosen,
                                              "Only one fact remaining.")
            self._update_state(chosen)
            return chosen

        # Build the candidate list (cap at 10 to avoid prompt overflow)
        candidate_ids = self.remaining[:10]
        candidates_text = "\n".join([
            f"  {i+1}. [{fid}] "
            + next(f["title"] for f in self.fact_pool if f["id"] == fid)
            for i, fid in enumerate(candidate_ids)
        ])

        studied_str = (
            ", ".join(self.studied[-5:]) + " (showing last 5)"
            if len(self.studied) > 5 else
            ", ".join(self.studied) if self.studied else "nothing yet"
        )

        prompt = (
            "You are an autonomous learning agent deciding what to study next.\n\n"
            f"Topics already studied: {studied_str}\n\n"
            "Numbered list of remaining candidates:\n"
            f"{candidates_text}\n\n"
            "Select the single best topic to maximise long-term knowledge retention.\n"
            "Consider: coherence with what you already know, foundational importance, "
            "and avoiding semantic overlap with recently studied topics.\n\n"
            "Reply with ONLY the candidate number (e.g., 3) followed by one sentence of reasoning.\n"
            "Example format:\n"
            "3: This topic builds directly on the prior material.\n\n"
            "Your selection:"
        )

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        # Parse the choice robustly — handles [fact_XXX] id, digit, or true fallback
        chosen_idx, fallback_used = self._parse_choice(
            response, len(candidate_ids), candidate_ids, step_num
        )
        chosen_fact_id = candidate_ids[chosen_idx]

        # Log everything — this is paper material
        self.logger.log_curriculum_choice(
            step_num=step_num,
            remaining_facts=candidate_ids,
            chosen_fact=chosen_fact_id,
            model_reasoning=response,
            fallback_used=fallback_used,
        )

        self._update_state(chosen_fact_id)
        return chosen_fact_id

    def get_full_order_by_running(self) -> List[str]:
        """
        Pre-generate the full curriculum order by running selection for all steps.
        Call this BEFORE the main self-edit loop to log the complete order upfront.
        """
        print("\n[Curriculum] Pre-generating self-selected order for all facts...")
        full_order = []
        # Reset state first
        self.studied = []
        self.remaining = [f["id"] for f in self.fact_pool]
        for step in range(len(self.fact_pool)):
            chosen = self.select_next(step)
            full_order.append(chosen)
        # Reset again — the actual training loop will call select_next() too
        self.studied = []
        self.remaining = [f["id"] for f in self.fact_pool]
        return full_order

    def _parse_choice(
        self,
        response: str,
        num_candidates: int,
        candidate_ids: list,
        step_num: int,
    ) -> tuple:
        """
        Robustly extract the model's chosen candidate index from its response.

        Tries (in order):
          1. Direct [fact_XXX] id match — most reliable, no number ambiguity
          2. 1-indexed digit anywhere in the response string
          3. True fallback to index 0 (first remaining) — logs fallback_used=True

        Returns (idx, fallback_used: bool)
        """
        stripped = response.strip()

        # Strategy 1: model explicitly named a fact id like [fact_007] or fact_007
        for i, fid in enumerate(candidate_ids):
            if fid in stripped:
                return i, False

        # Strategy 2: find any standalone 1-indexed digit in the response
        # Use word-boundary search so "NUMBER" or "NUMBER:" don't match,
        # but "3." or "pick 2" or "Option 1" do.
        digit_match = re.search(r'(?<![A-Za-z])(\d+)(?![A-Za-z])', stripped)
        if digit_match:
            idx = int(digit_match.group(1)) - 1  # 1-indexed -> 0-indexed
            if 0 <= idx < num_candidates:
                return idx, False

        # Strategy 3: true fallback
        print(
            f"  [Curriculum step {step_num}] FALLBACK: could not parse response "
            f"'{stripped[:80]}'. Using first remaining fact."
        )
        return 0, True

    def _update_state(self, chosen_fact_id: str):
        self.remaining.remove(chosen_fact_id)
        self.studied.append(chosen_fact_id)
