"""
src/curriculum.py

Curriculum strategies for the self-edit order:
  - PlannedCurriculum       -> human-fixed order (Config C3 control)
  - SelfGeneratedCurriculum -> model chooses its own order (Config C4/C6 novel)

WP2 Enhancements:
  - max_candidates: configurable cap (null for all remaining)
  - shuffle_candidates: randomize presented order every step with dedicated RNG
  - own RNG: dedicated random.Random(seed) and torch.Generator, isolated from training RNG
  - parser: priority order (id at start, leading digit, earliest id occurrence, anywhere digit, fallback)
  - fallback_policy: 'first' (v1) or 'random'
  - selection_method: 'generate' (free-text) or 'logprob' (token likelihood scoring)
  - selection_mode: 'upfront' or 'online'
"""
import math
import random
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
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
        self.order = [f["id"] for f in fact_pool]

    def get_full_order(self) -> List[str]:
        return self.order.copy()


class RandomCurriculum:
    """
    Config C7 / C9 control curriculum:
    Returns facts in a seeded pseudo-random order.
    """

    def __init__(self, fact_pool: List[dict], seed: int = 42):
        self.rng = random.Random(seed)
        self.order = [f["id"] for f in fact_pool]
        self.rng.shuffle(self.order)

    def get_full_order(self) -> List[str]:
        return self.order.copy()


class DispersedCurriculum:
    """
    Config C8 / C10 control curriculum:
    Deterministic, topic-interleaved curriculum minimizing adjacent same-domain pairs.
    Round-robin over domains, largest domain first, ties by min fact ID.
    """

    def __init__(self, fact_pool: List[dict]):
        from collections import defaultdict
        groups = defaultdict(list)
        for f in fact_pool:
            groups[f.get("domain", "unknown")].append(f["id"])

        for d in groups:
            groups[d].sort()

        # Sort domains: largest domain first; ties by min fact ID
        sorted_domains = sorted(
            groups.keys(),
            key=lambda d: (-len(groups[d]), min(groups[d]) if groups[d] else "")
        )

        order = []
        done = False
        while not done:
            done = True
            for d in sorted_domains:
                if groups[d]:
                    order.append(groups[d].pop(0))
                    done = False

        self.order = order

    def get_full_order(self) -> List[str]:
        return self.order.copy()


class SelfGeneratedCurriculum:
    """
    Config C4 (NOVEL CONTRIBUTION):
    At each step, the model reads candidate passages and selects what to study next.

    WP2 parameters:
      - max_candidates: int or None (null = all remaining)
      - shuffle_candidates: bool (shuffle presented order each step)
      - seed: int (dedicated RNG seed)
      - fallback_policy: 'first' (v1) or 'random'
      - selection_method: 'generate' or 'logprob'
      - selection_mode: 'upfront' or 'online'
    """

    def __init__(
        self,
        model,
        tokenizer,
        fact_pool: List[dict],
        logger: RunLogger,
        temperature: float = 0.7,
        device: str = "cuda",
        seed: int = 42,
        max_candidates: Optional[Union[int, str]] = 10,
        shuffle_candidates: bool = False,
        fallback_policy: str = "first",
        selection_method: str = "generate",
        selection_mode: str = "upfront",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.fact_pool = fact_pool
        self.logger = logger
        self.temperature = temperature
        self.device = device
        self.seed = seed

        # Parse max_candidates (handles None, "null", or integer)
        if max_candidates in (None, "null", "None"):
            self.max_candidates = None
        else:
            self.max_candidates = int(max_candidates)

        self.shuffle_candidates = bool(shuffle_candidates)
        self.fallback_policy = str(fallback_policy)
        self.selection_method = str(selection_method)
        self.selection_mode = str(selection_mode)

        # Dedicated RNG for selection to ensure training RNG is never perturbed
        self.rng = random.Random(seed)
        self.torch_gen = torch.Generator(device="cpu").manual_seed(seed)

        self.studied: List[str] = []
        self.remaining: List[str] = [f["id"] for f in fact_pool]

    def _get_candidates(self) -> List[str]:
        """
        Determine candidate fact IDs for the current selection step.
        If max_candidates is None: all remaining facts are candidates.
        If capped:
          - If shuffle_candidates is True or max_candidates != 10 or fallback_policy != 'first' or selection_method != 'generate':
            take a seeded random sample (WP2 rule 1).
          - Else (v1 default: shuffle=False, max_candidates=10):
            take self.remaining[:10] to preserve v1 exact behavior.
        """
        if self.max_candidates is None or len(self.remaining) <= self.max_candidates:
            candidates = list(self.remaining)
        elif self.shuffle_candidates or self.max_candidates != 10 or self.fallback_policy != "first" or self.selection_method != "generate":
            candidates = self.rng.sample(self.remaining, self.max_candidates)
        else:
            candidates = self.remaining[: self.max_candidates]

        if self.shuffle_candidates:
            self.rng.shuffle(candidates)

        return candidates

    def select_next(self, step_num: int) -> str:
        """
        Ask the model to choose its next self-edit subject from remaining candidates.
        Logs the choice + reasoning. Returns the fact_id of the chosen fact.
        """
        if len(self.remaining) == 0:
            raise ValueError("No remaining facts to select from.")
        if len(self.remaining) == 1:
            chosen = self.remaining[0]
            self.logger.log_curriculum_choice(
                step_num=step_num,
                remaining_facts=self.remaining.copy(),
                chosen_fact=chosen,
                model_reasoning="Only one fact remaining.",
                fallback_used=False,
                displayed_order=[chosen],
                displayed_position=1,
                parse_strategy="id",
                selection_method=self.selection_method,
            )
            self._update_state(chosen)
            return chosen

        displayed_candidates = self._get_candidates()
        candidates_text = "\n".join([
            f"  {i+1}. [{fid}] "
            + next(f["title"] for f in self.fact_pool if f["id"] == fid)
            for i, fid in enumerate(displayed_candidates)
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

        # -----------------------------------------------------------
        # Method 1: Logprob scoring
        # -----------------------------------------------------------
        if self.selection_method == "logprob":
            scores = self._score_candidates_logprob(prompt, displayed_candidates)
            temp = max(self.temperature, 1e-4)
            cand_scores = [scores.get(fid, -100.0) for fid in displayed_candidates]
            max_s = max(cand_scores)
            exp_scores = [math.exp((s - max_s) / temp) for s in cand_scores]
            sum_exp = sum(exp_scores)
            probs = [e / sum_exp for e in exp_scores] if sum_exp > 0 else [1.0 / len(exp_scores)] * len(exp_scores)

            chosen_idx = self.rng.choices(range(len(displayed_candidates)), weights=probs, k=1)[0]
            chosen_fact_id = displayed_candidates[chosen_idx]
            chosen_pos = chosen_idx + 1

            self.logger.log_curriculum_choice(
                step_num=step_num,
                remaining_facts=displayed_candidates,
                chosen_fact=chosen_fact_id,
                model_reasoning=f"Logprob selection (temp={self.temperature}): {scores}",
                fallback_used=False,
                displayed_order=displayed_candidates,
                displayed_position=chosen_pos,
                parse_strategy="logprob",
                selection_method="logprob",
                scores=scores,
            )
            self._update_state(chosen_fact_id)
            return chosen_fact_id

        # -----------------------------------------------------------
        # Method 2: Free-text generation (v1 default)
        # -----------------------------------------------------------
        # Isolate RNG states and save model mode
        was_training = getattr(self.model, "training", False)
        prev_torch_state = torch.get_rng_state()
        prev_cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        self.model.eval()
        try:
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1024
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            input_len = inputs["input_ids"].shape[1] if isinstance(inputs, dict) and "input_ids" in inputs else 0
            if hasattr(outputs, "shape") and len(outputs.shape) == 2:
                gen_tokens = outputs[0][input_len:]
                response = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            elif isinstance(outputs, str):
                response = outputs.strip()
            else:
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        finally:
            if was_training:
                self.model.train()
            # Restore global torch RNG state so training RNG stream is unperturbed
            torch.set_rng_state(prev_torch_state)
            if prev_cuda_state is not None:
                torch.cuda.set_rng_state_all(prev_cuda_state)

        # Parse the choice robustly
        chosen_idx, parse_strategy, fallback_used = self._parse_choice(
            response, displayed_candidates, step_num
        )
        chosen_fact_id = displayed_candidates[chosen_idx]
        chosen_pos = chosen_idx + 1

        self.logger.log_curriculum_choice(
            step_num=step_num,
            remaining_facts=displayed_candidates,
            chosen_fact=chosen_fact_id,
            model_reasoning=response,
            fallback_used=fallback_used,
            displayed_order=displayed_candidates,
            displayed_position=chosen_pos,
            parse_strategy=parse_strategy,
            selection_method="generate",
        )

        self._update_state(chosen_fact_id)
        return chosen_fact_id

    def _score_candidates_logprob(
        self, prompt: str, candidates: List[str]
    ) -> Dict[str, float]:
        """
        Compute mean token log-likelihood of '<fact_id>: <title>' given the prompt.
        """
        scores: Dict[str, float] = {}
        if not hasattr(self.model, "forward") and not hasattr(self.model, "__call__"):
            return {fid: 0.0 for fid in candidates}

        try:
            prompt_enc = self.tokenizer(prompt, return_tensors="pt")
            prompt_len = prompt_enc.input_ids.shape[1]
        except Exception:
            return {fid: 0.0 for fid in candidates}

        for fid in candidates:
            title = next(f["title"] for f in self.fact_pool if f["id"] == fid)
            target_str = f" {fid}: {title}"
            try:
                full_enc = self.tokenizer(prompt + target_str, return_tensors="pt")
                full_ids = full_enc.input_ids.to(self.device)
                seq_len = full_ids.shape[1]
                if seq_len <= prompt_len:
                    scores[fid] = -100.0
                    continue

                with torch.no_grad():
                    outputs = self.model(full_ids)
                    if not hasattr(outputs, "logits"):
                        scores[fid] = 0.0
                        continue
                    logits = outputs.logits  # [1, seq_len, vocab_size]
                    target_logits = logits[:, prompt_len - 1 : -1, :]
                    target_ids = full_ids[:, prompt_len:]

                    log_probs = torch.log_softmax(target_logits, dim=-1)
                    token_log_probs = torch.gather(
                        log_probs, 2, target_ids.unsqueeze(-1)
                    ).squeeze(-1)
                    mean_ll = token_log_probs.mean().item()
                    scores[fid] = mean_ll
            except Exception:
                scores[fid] = -100.0

        return scores

    def _parse_choice(
        self,
        response: str,
        displayed_candidates: List[str],
        step_num: int,
    ) -> Tuple[int, str, bool]:
        """
        Extract the model's chosen candidate index from its response.

        Priority order (WP2):
          1. Leading fact id match at start -> 'id'
          2. Leading digit match at start -> 'leading_digit'
          3. Fact ID occurrence anywhere in reply (earliest occurrence wins) -> 'id'
          4. Standalone digit anywhere in reply -> 'anywhere_digit'
          5. Fallback policy ('first' -> index 0, 'random' -> random candidate) -> 'fallback'
        """
        stripped = response.strip()
        num_candidates = len(displayed_candidates)

        # 1. Leading fact id at start: e.g. '[fact_005]' or 'fact_005'
        leading_id = re.match(r'^\s*\[?(fact_\w+)\]?', stripped)
        if leading_id and leading_id.group(1) in displayed_candidates:
            return displayed_candidates.index(leading_id.group(1)), "id", False

        # 2. Leading candidate digit at start: e.g. '3:', '3.', '3 -', '3 '
        leading_digit = re.match(r'^\s*\[?(\d+)\]?(?:[:\.\)\s-]|$)', stripped)
        if leading_digit:
            val = int(leading_digit.group(1))
            if 1 <= val <= num_candidates:
                return val - 1, "leading_digit", False

        # 3. Fact ID occurrence anywhere in reply: earliest occurrence in reply wins
        earliest_pos = None
        earliest_idx = None
        for idx, fid in enumerate(displayed_candidates):
            m = re.search(r'\b' + re.escape(fid) + r'\b', stripped)
            if m:
                pos = m.start()
                if earliest_pos is None or pos < earliest_pos:
                    earliest_pos = pos
                    earliest_idx = idx

        if earliest_idx is not None:
            return earliest_idx, "id", False

        # 4. Standalone digit anywhere in reply
        digit_matches = re.finditer(r'(?<![A-Za-z0-9])(\d+)(?![A-Za-z0-9])', stripped)
        for m in digit_matches:
            val = int(m.group(1))
            if 1 <= val <= num_candidates:
                return val - 1, "anywhere_digit", False

        # 5. Fallback policy
        print(
            f"  [Curriculum step {step_num}] FALLBACK: could not parse response "
            f"'{stripped[:80]}'. Policy: {self.fallback_policy}."
        )
        if self.fallback_policy == "random":
            fallback_idx = self.rng.randrange(num_candidates)
        else:  # "first"
            fallback_idx = 0
        return fallback_idx, "fallback", True

    def get_full_order_by_running(self) -> List[str]:
        """
        Pre-generate the full curriculum order by running selection for all steps.
        Call this BEFORE the main self-edit loop to log the complete order upfront.
        """
        print("\n[Curriculum] Pre-generating self-selected order for all facts...")
        full_order = []
        saved_rng_state = self.rng.getstate()
        self.studied = []
        self.remaining = [f["id"] for f in self.fact_pool]
        for step in range(len(self.fact_pool)):
            chosen = self.select_next(step)
            full_order.append(chosen)
        self.studied = []
        self.remaining = [f["id"] for f in self.fact_pool]
        self.rng.setstate(saved_rng_state)
        return full_order

    def _update_state(self, chosen_fact_id: str):
        self.remaining.remove(chosen_fact_id)
        self.studied.append(chosen_fact_id)
