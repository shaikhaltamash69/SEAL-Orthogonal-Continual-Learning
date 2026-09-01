"""
scripts/test_orthogonality.py

Unit test for the O-LoRA orthogonality penalty.
RUN THIS BEFORE attempting Config C3 or C4 experiments.

If this test fails → DO NOT proceed to experiments. Debug orthogonality.py first.
If this test passes → the math is wired correctly, proceed to C3.

Expected behaviour:
  - Initial overlap between two random A matrices: HIGH (they're correlated)
  - After 200 gradient steps minimizing only L_orth: overlap DROPS significantly
  - The penalty is driving the subspaces apart

Pass criterion: final overlap < 20% of initial overlap
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.orthogonality import OLoRAOrthogonalityPenalty


def test_penalty_drives_orthogonality():
    print("\n" + "=" * 60)
    print("O-LoRA Orthogonality Penalty — Unit Test")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("⚠️  CUDA not available — running on CPU (slower but valid for unit test)")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    penalty_module = OLoRAOrthogonalityPenalty(lambda_orth=1.0)  # strong λ for clear test signal

    # Simulate Task 1 LoRA A matrices (2 layers, rank=16, dim=64)
    A1_layer1 = torch.randn(16, 64).to(device)
    A1_layer2 = torch.randn(16, 64).to(device)
    penalty_module.save_current_adapter([A1_layer1, A1_layer2])

    # Task 2: starts correlated with Task 1 (high overlap)
    A2_layer1 = A1_layer1.clone() + 0.1 * torch.randn_like(A1_layer1)
    A2_layer2 = A1_layer2.clone() + 0.1 * torch.randn_like(A1_layer2)
    A2_layer1.requires_grad_(True)
    A2_layer2.requires_grad_(True)

    # Measure initial overlap
    with torch.no_grad():
        initial_overlap = (
            torch.norm(torch.mm(A2_layer1, A1_layer1.t()), p="fro").item()
            + torch.norm(torch.mm(A2_layer2, A1_layer2.t()), p="fro").item()
        )
    print(f"\nInitial overlap (should be HIGH): {initial_overlap:.4f}")

    # Optimize A2 to minimize ONLY the orthogonality penalty
    optimizer = torch.optim.Adam([A2_layer1, A2_layer2], lr=0.01)
    print(f"\nTraining A2 to minimize orthogonality penalty (200 steps)...")
    print(f"{'Step':>6} | {'Penalty':>10} | {'Overlap':>10}")
    print("-" * 32)

    for step in range(200):
        optimizer.zero_grad()
        penalty = penalty_module.compute_penalty([A2_layer1, A2_layer2])
        penalty.backward()
        optimizer.step()

        if step % 40 == 0:
            with torch.no_grad():
                current_overlap = (
                    torch.norm(torch.mm(A2_layer1, A1_layer1.t()), p="fro").item()
                    + torch.norm(torch.mm(A2_layer2, A1_layer2.t()), p="fro").item()
                )
            print(f"{step:>6} | {penalty.item():>10.4f} | {current_overlap:>10.4f}")

    # Final overlap
    with torch.no_grad():
        final_overlap = (
            torch.norm(torch.mm(A2_layer1, A1_layer1.t()), p="fro").item()
            + torch.norm(torch.mm(A2_layer2, A1_layer2.t()), p="fro").item()
        )

    print(f"\nFinal overlap:   {final_overlap:.4f}")
    print(f"Initial overlap: {initial_overlap:.4f}")
    print(f"Reduction:       {(1 - final_overlap/initial_overlap)*100:.1f}%")

    # Pass criterion: final overlap < 20% of initial
    passed = final_overlap < (0.20 * initial_overlap)
    if passed:
        print(f"\n[PASS] Orthogonality penalty is working correctly.")
        print(f"   Proceed to Config C3 experiments.")
    else:
        print(f"\n[FAIL] Penalty did not drive sufficient orthogonality.")
        print(f"   Expected: final < {0.20 * initial_overlap:.4f}")
        print(f"   Got:      {final_overlap:.4f}")
        print(f"   Debug orthogonality.py before running C3/C4.")

    return passed


def test_compute_penalty_zero_at_first_step():
    """Before any adapters are saved, penalty must be zero."""
    print("\n" + "-" * 40)
    print("Test: penalty = 0 at step 0 (no history)")
    penalty_module = OLoRAOrthogonalityPenalty(lambda_orth=0.5)
    dummy_A = [torch.randn(16, 64, requires_grad=True)]
    penalty = penalty_module.compute_penalty(dummy_A)
    assert penalty.item() == 0.0, f"Expected 0.0, got {penalty.item()}"
    print("[PASS] penalty is 0.0 before any adapters saved")


def test_penalty_increases_with_num_adapters():
    """Penalty should grow as more adapters are saved."""
    print("\n" + "-" * 40)
    print("Test: penalty increases with # stored adapters")
    penalty_module = OLoRAOrthogonalityPenalty(lambda_orth=0.1)
    curr_A = [torch.randn(16, 64, requires_grad=True).cuda()] if torch.cuda.is_available() else [torch.randn(16, 64, requires_grad=True)]
    penalties = []
    for i in range(5):
        penalty_module.save_current_adapter(curr_A)
        penalty = penalty_module.compute_penalty(curr_A)
        penalties.append(penalty.item())
        print(f"  After {i+1} stored adapters: penalty = {penalty.item():.6f}")
    assert penalties[-1] > penalties[0], "Penalty should grow with history length"
    print("[PASS] penalty grows with history")


if __name__ == "__main__":
    test_compute_penalty_zero_at_first_step()
    test_penalty_increases_with_num_adapters()
    success = test_penalty_drives_orthogonality()
    sys.exit(0 if success else 1)
