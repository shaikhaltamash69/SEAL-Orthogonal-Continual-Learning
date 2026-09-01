"""
scripts/smoke_test.py

Phase 0 final checkpoint — runs a SINGLE self-edit cycle end-to-end
on 2 dummy facts to confirm the entire pipeline works before real experiments.

This is NOT for measuring results — just proving nothing crashes.

Run: python scripts/smoke_test.py

Expected output:
  - Model loads
  - Synthetic data generated
  - LoRA SFT completes
  - Accuracy evaluated
  - run_log.json written to results/SMOKE_TEST_*/
"""
import json
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.self_edit_loop import SEALSelfEditLoop, OLoRASEALLoop

# ── Minimal dummy facts (2 facts, 2 QA pairs each) ─────────────────────────
# These are fictional to avoid any pretraining contamination in the test
DUMMY_FACTS = [
    {
        "id": "smoke_001",
        "title": "The Velantian Telescope Discovery",
        "passage": (
            "In March 2025, the fictional Velantian Space Observatory detected an exoplanet "
            "designated VEL-7b orbiting a red dwarf star 42 light-years from Earth. "
            "The planet has a radius 1.3 times that of Earth and completes one orbit every "
            "17 days. Lead researcher Dr. Amara Singh announced the discovery at the "
            "fictional Velantian Astronomical Conference in Geneva. The observatory used "
            "a novel thermal imaging array to confirm the planet's presence."
        ),
        "qa_pairs": [
            {
                "question": "What is the designation of the exoplanet discovered by the Velantian Observatory?",
                "answer": "VEL-7b",
                "distractors": ["VEL-8c", "VEL-5a", "VEL-7a"]
            },
            {
                "question": "How many days does VEL-7b take to complete one orbit?",
                "answer": "17 days",
                "distractors": ["12 days", "23 days", "31 days"]
            }
        ],
        "domain": "science",
        "source_url": "fictional",
        "date_of_event": "2025-03-01"
    },
    {
        "id": "smoke_002",
        "title": "The Bremvor Bridge Construction",
        "passage": (
            "The fictional Bremvor Suspension Bridge, completed in August 2025, spans "
            "3,847 meters across the Kelton Strait in northern Scandinavia. Built by "
            "engineering firm Halvorsen & Partners at a cost of 4.2 billion euros, "
            "it is the longest bridge in fictional Scandinavia. Chief engineer Petra "
            "Lindqvist oversaw the 7-year construction project. The bridge can withstand "
            "wind speeds of up to 280 km/h and carries both road and rail traffic."
        ),
        "qa_pairs": [
            {
                "question": "How long is the Bremvor Suspension Bridge in meters?",
                "answer": "3,847 meters",
                "distractors": ["2,950 meters", "4,120 meters", "3,500 meters"]
            },
            {
                "question": "Who was the chief engineer of the Bremvor Bridge?",
                "answer": "Petra Lindqvist",
                "distractors": ["Amara Singh", "Erik Halvorsen", "Maria Johansson"]
            }
        ],
        "domain": "technology",
        "source_url": "fictional",
        "date_of_event": "2025-08-01"
    }
]

DUMMY_BASELINE = {
    "smoke_001": {"qa_accuracy_before_training": 0.0},
    "smoke_002": {"qa_accuracy_before_training": 0.0}
}


def run_smoke_test():
    print("=" * 60)
    print("SEAL Pipeline Smoke Test (2 dummy facts)")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")

    # Write dummy data temporarily
    os.makedirs("data", exist_ok=True)
    if not os.path.exists("data/fact_pool.json"):
        with open("data/fact_pool.json", "w") as f:
            json.dump(DUMMY_FACTS, f, indent=2)
        print("[INFO] Wrote dummy fact pool to data/fact_pool.json for smoke test")
        print("       Replace this with your real fact pool before real experiments!\n")

    with open("data/fact_pool.json", "w") as f:
        json.dump(DUMMY_FACTS, f, indent=2)
    with open("data/baseline_accuracy.json", "w") as f:
        json.dump(DUMMY_BASELINE, f, indent=2)

    config = {
        "config_name": "SMOKE_TEST",
        "run_id": "SMOKE_TEST_vanilla",
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "loop_type": "vanilla",
        "lora_r": 8,          # small rank for speed
        "lora_alpha": 16,
        "lr": 2e-4,
        "sft_epochs": 1,      # 1 epoch only for smoke test
        "results_dir": "results/",
    }

    print("[TEST 1] Vanilla LoRA self-edit loop...")
    try:
        loop = SEALSelfEditLoop(config)
        accuracy_matrix = loop.run(
            DUMMY_FACTS,
            ["smoke_001", "smoke_002"],
            DUMMY_BASELINE
        )
        print(f"  Final accuracy matrix: {json.dumps(accuracy_matrix, indent=4)}")
        print("[TEST 1] PASS - Vanilla LoRA pipeline works\n")
    except Exception as e:
        print(f"[TEST 1] FAIL: {e}")
        raise

    print("[TEST 2] O-LoRA self-edit loop (orthogonality penalty)...")
    olora_config = {**config, "run_id": "SMOKE_TEST_olora", "loop_type": "olora", "lambda_orth": 0.1}
    try:
        loop2 = OLoRASEALLoop(olora_config)
        accuracy_matrix2 = loop2.run(
            DUMMY_FACTS,
            ["smoke_001", "smoke_002"],
            DUMMY_BASELINE
        )
        print(f"  Final accuracy matrix: {json.dumps(accuracy_matrix2, indent=4)}")
        print("[TEST 2] PASS - O-LoRA pipeline works\n")
    except Exception as e:
        print(f"[TEST 2] FAIL: {e}")
        raise

    print("=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("The full pipeline runs end-to-end.")
    print()
    print("Next steps:")
    print("  1. Replace data/fact_pool.json with your real 30-50 post-2024 facts")
    print("  2. Run: python verify_novelty.py")
    print("  3. Run: python scripts/run_config.py --config configs/config_c2_vanilla.json --seed 42")
    print("=" * 60)


if __name__ == "__main__":
    run_smoke_test()
