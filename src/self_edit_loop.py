"""
src/self_edit_loop.py

Self-edit loop wrappers for all 4 experimental configs.

Architecture note (from reading SEAL's own code):
  SEAL's original continual_self_edits.py uses vLLM + ZMQ as separate processes,
  then merges adapters into the base model after each step. For our research
  (focused on comparing planned vs self-generated curriculum under orthogonality),
  we use a simpler in-process approach with HuggingFace + PEFT:
    1. Same model loaded in memory throughout
    2. LoRA adapters NOT merged — kept separate for orthogonality computation
    3. This is explicitly stated in our paper's Methodology (vs SEAL's merge approach)

  This is a deliberate, documented simplification for a 12-week honours project.
  The key insight we're testing (does orthogonality transfer to self-generated order?)
  is testable with this setup. The exact vLLM/merge implementation is left as future work.

Base class: SEALSelfEditLoop  (Config C2 — vanilla LoRA)
Subclass:   OLoRASEALLoop     (Config C3 planned / C4 self-generated)
Subclass:   LoRISEALLoop      (Config C5 — LoRI mechanism)
"""
import os
import json
import torch
from datetime import datetime
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from torch.optim import AdamW

from .evaluator import evaluate_all_facts
from .logger import RunLogger
from .orthogonality import OLoRAOrthogonalityPenalty


# ─────────────────────────────────────────────────────────────────
#  Config C2 — Vanilla LoRA (SEAL baseline)
# ─────────────────────────────────────────────────────────────────

class SEALSelfEditLoop:
    """
    Config C2: SEAL's standard self-edit loop with vanilla LoRA.
    This replicates (and should reproduce) SEAL's reported forgetting behaviour.
    """

    CONFIG_NAME = "C2_vanilla_lora"

    def __init__(self, config: dict):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n{'='*60}")
        print(f"Initialising {self.CONFIG_NAME}")
        print(f"  Model:   {config['model_name']}")
        print(f"  Device:  {self.device}")
        print(f"  Run ID:  {config['run_id']}")
        print(f"{'='*60}\n")

        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # Load base model in bf16 — 24GB VRAM handles Qwen2.5-1.5B easily
        self.base_model = AutoModelForCausalLM.from_pretrained(
            config["model_name"],
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )

        # Inject LoRA adapters
        lora_config = LoraConfig(
            r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        self.model = get_peft_model(self.base_model, lora_config)
        self.model.print_trainable_parameters()

        self.logger = RunLogger(config["run_id"], config["results_dir"])
        self.logger.log_config(config)

    # ------------------------------------------------------------------ #
    #  Self-edit step: passage → synthetic data → LoRA SFT → eval        #
    # ------------------------------------------------------------------ #

    def generate_synthetic_data(self, passage: str, title: str) -> str:
        """
        Step 1 of SEAL: the model reads a passage and rewrites it as
        self-generated training data (Q&A pairs).
        """
        prompt = (
            f"Read the following passage carefully, then write 4 question-answer pairs "
            f"that test specific factual knowledge from this passage. "
            f"Make each question precise — answerable ONLY from this passage.\n\n"
            f"Title: {title}\n"
            f"Passage: {passage}\n\n"
            f"Format each as:\n"
            f"Q: [specific factual question]\n"
            f"A: [precise answer]\n\n"
            f"Training pairs:"
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.4,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        synthetic = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        return synthetic

    def self_edit_step(self, fact: dict, step_num: int):
        """One complete self-edit: generate → SFT → (subclass hooks for extra loss)."""
        print(f"\n{'─'*60}")
        print(f"  Step {step_num:02d}: {fact['id']} — {fact['title']}")
        print(f"{'─'*60}")

        # 1. Generate synthetic training data
        synthetic_data = self.generate_synthetic_data(fact["passage"], fact["title"])
        self.logger.log_synthetic_data(step_num, fact["id"], synthetic_data)
        print(f"  Generated {len(synthetic_data)} chars of synthetic data")

        # 2. SFT step (overridden in subclasses)
        self._sft_step(synthetic_data, fact["id"], step_num)

    def _sft_step(self, synthetic_text: str, fact_id: str, step_num: int):
        """Standard LoRA SFT — no extra constraints."""
        self.model.train()
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.get("lr", 2e-4),
            weight_decay=0.01,
        )

        # Combine passage + synthetic Q&A as training text
        inputs = self.tokenizer(
            synthetic_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length",
        ).to(self.device)

        labels = inputs["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        total_loss = 0.0
        n_epochs = self.config.get("sft_epochs", 3)
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / n_epochs
        print(f"  SFT loss: {avg_loss:.4f}")
        self.model.eval()
        self.logger.log_sft_loss(step_num, fact_id, {"sft_loss": avg_loss})

    # ------------------------------------------------------------------ #
    #  Main experiment driver                                             #
    # ------------------------------------------------------------------ #

    def run(
        self,
        fact_pool: List[dict],
        ordered_fact_ids: List[str],
        baseline_accuracy: dict,
    ) -> Dict[int, Dict[str, float]]:
        """
        Run the full self-edit sequence and return the accuracy matrix.

        Returns:
            {step_num: {fact_id: accuracy}} — your Figure 1 heatmap data.
        """
        accuracy_matrix: Dict[int, Dict[str, float]] = {}

        print(f"\n{'='*60}")
        print(f"Starting {self.CONFIG_NAME} run")
        print(f"  Facts: {len(ordered_fact_ids)}")
        print(f"  Order: {ordered_fact_ids}")
        print(f"{'='*60}\n")

        for step_num, fact_id in enumerate(ordered_fact_ids):
            fact = next((f for f in fact_pool if f["id"] == fact_id), None)
            if fact is None:
                print(f"⚠️  Fact {fact_id} not found — skipping")
                continue

            # Self-edit on this fact
            self.self_edit_step(fact, step_num)

            # Evaluate on ALL facts studied so far (builds forgetting curve)
            print(f"\n  Evaluating after step {step_num}...")
            studied_so_far = ordered_fact_ids[: step_num + 1]
            accuracy_matrix[step_num] = evaluate_all_facts(
                self.model, self.tokenizer, fact_pool, studied_so_far, self.device
            )
            self.logger.log_accuracy_matrix(step_num, accuracy_matrix[step_num])

            # Save checkpoint every 5 steps
            if step_num % 5 == 0:
                self._save_checkpoint(step_num)

        self.logger.finalize(ordered_fact_ids)
        return accuracy_matrix

    def _save_checkpoint(self, step_num: int):
        ckpt_path = os.path.join(
            self.config["results_dir"], self.config["run_id"],
            f"checkpoint_step_{step_num:03d}"
        )
        self.model.save_pretrained(ckpt_path)
        print(f"  💾 Checkpoint saved: step {step_num}")


# ─────────────────────────────────────────────────────────────────
#  Config C3 / C4 — O-LoRA (adds orthogonality penalty)
# ─────────────────────────────────────────────────────────────────

class OLoRASEALLoop(SEALSelfEditLoop):
    """
    Config C3 (planned curriculum) and Config C4 (self-generated curriculum).
    Identical to SEALSelfEditLoop EXCEPT the SFT step adds the O-LoRA penalty:
        L_total = L_SFT + λ · Σ ||A_t^T @ A_i||²_F
    """

    CONFIG_NAME = "OLoRA_SEAL"

    def __init__(self, config: dict):
        super().__init__(config)
        self.orthogonality = OLoRAOrthogonalityPenalty(
            lambda_orth=config.get("lambda_orth", 0.1)
        )
        print(f"  O-LoRA λ = {config.get('lambda_orth', 0.1)}")

    def _sft_step(self, synthetic_text: str, fact_id: str, step_num: int):
        """O-LoRA SFT: SFT loss + orthogonality penalty."""
        self.model.train()
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.get("lr", 2e-4),
            weight_decay=0.01,
        )

        inputs = self.tokenizer(
            synthetic_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length",
        ).to(self.device)
        labels = inputs["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        n_epochs = self.config.get("sft_epochs", 3)
        final_sft, final_orth, final_total = 0.0, 0.0, 0.0

        for epoch in range(n_epochs):
            optimizer.zero_grad()

            # Standard SFT loss
            outputs = self.model(**inputs, labels=labels)
            sft_loss = outputs.loss

            # O-LoRA orthogonality penalty
            current_As = self.orthogonality.get_A_matrices_from_model(self.model)
            orth_penalty = self.orthogonality.compute_penalty(current_As)

            total_loss = sft_loss + orth_penalty
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            final_sft = sft_loss.item()
            final_orth = orth_penalty.item() if hasattr(orth_penalty, "item") else float(orth_penalty)
            final_total = total_loss.item()

        print(f"  O-LoRA losses → SFT: {final_sft:.4f} | Orth: {final_orth:.6f} | Total: {final_total:.4f}")
        print(f"  Num prev adapters in history: {self.orthogonality.num_stored_adapters}")

        # Save this adapter to history AFTER training completes
        with torch.no_grad():
            current_As = self.orthogonality.get_A_matrices_from_model(self.model)
        self.orthogonality.save_current_adapter(current_As)

        # Log subspace overlaps (paper material)
        with torch.no_grad():
            current_As_for_log = self.orthogonality.get_A_matrices_from_model(self.model)
            overlaps = self.orthogonality.compute_current_overlaps(current_As_for_log)
        self.logger.log_adapter_subspace_overlap(step_num, overlaps)

        self.model.eval()
        self.logger.log_sft_loss(step_num, fact_id, {
            "sft_loss": final_sft,
            "orth_penalty": final_orth,
            "total_loss": final_total,
            "num_prev_adapters": self.orthogonality.num_stored_adapters,
        })
        self.logger.log_orthogonality_stats(step_num, final_orth, self.orthogonality.num_stored_adapters)


# ─────────────────────────────────────────────────────────────────
#  Config C5 — LoRI (frozen A, sparse B)
# ─────────────────────────────────────────────────────────────────

class LoRISEALLoop(SEALSelfEditLoop):
    """
    Config C5: LoRI mechanism — Reduced Interference LoRA.
    Reference: Zheng et al. (2025), github.com/juzhengz/LoRI

    Changes from vanilla LoRA:
      1. LoRA A matrices are frozen after init (random projection basis)
      2. B matrices use a top-k gradient mask (sparsity) per self-edit step
    """

    CONFIG_NAME = "C5_lori"

    def __init__(self, config: dict):
        super().__init__(config)
        self.sparsity = config.get("lori_sparsity", 0.8)  # fraction of B to freeze
        self._freeze_A_matrices()
        print(f"  LoRI sparsity = {self.sparsity}")

    def _freeze_A_matrices(self):
        """Freeze all LoRA A matrices — they act as fixed random projections."""
        frozen_count = 0
        for name, param in self.model.named_parameters():
            if "lora_A" in name:
                param.requires_grad_(False)
                frozen_count += 1
        print(f"  LoRI: {frozen_count} LoRA A matrices frozen (fixed random bases)")

    def _sft_step(self, synthetic_text: str, fact_id: str, step_num: int):
        """LoRI SFT: standard loss, but apply sparse gradient mask to B matrices."""
        self.model.train()
        optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.get("lr", 2e-4),
            weight_decay=0.01,
        )

        inputs = self.tokenizer(
            synthetic_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length",
        ).to(self.device)
        labels = inputs["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        n_epochs = self.config.get("sft_epochs", 3)
        b_masks: dict = {}  # built after first backward pass

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()

            # Build B masks on first epoch using gradient magnitude
            if epoch == 0:
                b_masks = self._build_b_masks()

            # Apply masks — zero out low-gradient B entries
            self._apply_b_masks(b_masks)

            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()

        print(f"  LoRI SFT loss: {loss.item():.4f} | Sparsity: {self.sparsity}")
        self.model.eval()
        self.logger.log_sft_loss(step_num, fact_id, {
            "sft_loss": loss.item(),
            "lori_sparsity": self.sparsity,
        })

    def _build_b_masks(self) -> dict:
        """Top-k gradient mask: keep (1 - sparsity) fraction of B entries with largest gradients."""
        masks = {}
        for name, param in self.model.named_parameters():
            if "lora_B" in name and param.grad is not None:
                grad_mag = param.grad.abs()
                threshold = torch.quantile(grad_mag.float().flatten(), self.sparsity)
                masks[name] = (grad_mag >= threshold).float()
        return masks

    def _apply_b_masks(self, masks: dict):
        """Zero out gradients for masked B entries."""
        for name, param in self.model.named_parameters():
            if "lora_B" in name and name in masks and param.grad is not None:
                param.grad.mul_(masks[name])
