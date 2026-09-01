"""
src/orthogonality.py

O-LoRA Orthogonality Penalty Implementation.

Reference:
    Wang et al. (2023) "Orthogonal Subspace Learning for Language Model Continual Learning"
    EMNLP Findings 2023, arXiv:2310.14152

The penalty is:
    L_orth = lambda * Σ_{i=1}^{t-1} ||A_t^T · A_i||^2_F

Where:
    - A_t  = the current self-edit step's LoRA A matrices
    - A_i  = all previous self-edit steps' LoRA A matrices
    - ||·||_F = Frobenius norm

This gets ADDED to the SFT loss during each self-edit training step.
"""
import torch
import torch.nn as nn
from typing import List


class OLoRAOrthogonalityPenalty:
    """
    Tracks all previously learned LoRA A matrices and computes
    an orthogonality penalty against any new adapter being trained.

    Usage:
        1. Create one instance per experiment run.
        2. Call compute_penalty(current_As) inside the training loop
           and ADD the result to your SFT loss.
        3. After each full self-edit step, call save_current_adapter(current_As)
           so future steps penalize against this one.
    """

    def __init__(self, lambda_orth: float = 0.1):
        """
        Args:
            lambda_orth: Penalty weight. Paper uses 0.1.
                         If forgetting is severe, try 0.5 or 1.0.
                         If plasticity (learning new facts) suffers, reduce to 0.05.
        """
        self.lambda_orth = lambda_orth
        # Each entry: list of A matrices (one per LoRA layer) from one past self-edit
        self.previous_A_matrices: List[List[torch.Tensor]] = []

    def compute_penalty(self, current_A_matrices: List[torch.Tensor]) -> torch.Tensor:
        """
        Computes the orthogonality penalty.

        Args:
            current_A_matrices: List of current LoRA A tensors (one per layer,
                                 must be attached to computation graph for grad).

        Returns:
            Scalar penalty tensor (with grad). Returns 0.0 if no history yet.
        """
        if not self.previous_A_matrices:
            # First step — no previous adapters to be orthogonal to
            return torch.tensor(0.0, device=current_A_matrices[0].device, requires_grad=True)

        total_penalty = torch.zeros(1, device=current_A_matrices[0].device)

        for prev_step_As in self.previous_A_matrices:
            step_penalty = torch.zeros(1, device=current_A_matrices[0].device)
            for curr_A, prev_A in zip(current_A_matrices, prev_step_As):
                if curr_A.shape != prev_A.shape:
                    # Shapes should match (same model), but guard just in case
                    continue
                # A_t^T @ A_i  — shape: (r × r) if A is (r × d)
                overlap = torch.mm(curr_A, prev_A.t())   # curr (r,d) @ prev (d,r) → (r,r)
                step_penalty = step_penalty + torch.norm(overlap, p="fro") ** 2
            total_penalty = total_penalty + step_penalty

        return self.lambda_orth * total_penalty.squeeze()

    def save_current_adapter(self, current_A_matrices: List[torch.Tensor]):
        """
        Save this step's A matrices AFTER the full SFT step completes.
        Call this ONCE per self-edit step, after optimizer.step() is done.

        Detaches + clones — no gradient flows through stored matrices.
        """
        saved = [A.detach().clone() for A in current_A_matrices]
        self.previous_A_matrices.append(saved)
        print(f"  [O-LoRA] Saved adapter #{len(self.previous_A_matrices)} to history "
              f"({len(saved)} layers stored)")

    def get_A_matrices_from_model(self, model) -> List[torch.Tensor]:
        """
        Extract all LoRA A matrices from a PEFT model.
        These are the trainable low-rank matrices added to each attention layer.
        """
        A_matrices = []
        for name, param in model.named_parameters():
            if "lora_A" in name and param.requires_grad:
                A_matrices.append(param)
        return A_matrices

    def compute_current_overlaps(self, current_A_matrices: List[torch.Tensor]) -> List[float]:
        """
        For logging only: returns ||A_t^T @ A_i||_F for each past adapter.
        Call this AFTER training to measure how orthogonal the learned adapter is.
        """
        overlaps = []
        for i, prev_step_As in enumerate(self.previous_A_matrices):
            total_overlap = 0.0
            for curr_A, prev_A in zip(current_A_matrices, prev_step_As):
                if curr_A.shape == prev_A.shape:
                    with torch.no_grad():
                        overlap = torch.mm(curr_A, prev_A.t())
                        total_overlap += torch.norm(overlap, p="fro").item()
            overlaps.append({"prev_adapter_idx": i, "frobenius_overlap": total_overlap})
        return overlaps

    @property
    def num_stored_adapters(self) -> int:
        return len(self.previous_A_matrices)
