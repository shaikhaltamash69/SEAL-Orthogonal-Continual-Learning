"""
scripts/run_multi_seeds.py

Runs multi-seed iterations (seeds 1, 2, 3) across configs C2, C3, C4, C5
to generate error bars and compute the Wilcoxon Signed-Rank Test.

Usage:
    python scripts/run_multi_seeds.py --seeds 1 2 3
"""
import argparse
import os
import sys
import subprocess

CONFIGS = [
    "configs/config_c2_vanilla.json",
    "configs/config_c3_olora_planned.json",
    "configs/config_c4_olora_selfgen.json",
    "configs/config_c5_lori_selfgen.json",
]


def run_multi_seeds(seeds: list):
    print("=" * 60)
    print(f"Multi-Seed Experiment Runner: Seeds {seeds}")
    print("=" * 60)

    for seed in seeds:
        for config_path in CONFIGS:
            print(f"\n[RUNNING] {config_path} | Seed {seed}...")
            cmd = [
                sys.executable, "scripts/run_config.py",
                "--config", config_path,
                "--seed", str(seed)
            ]
            res = subprocess.run(cmd)
            if res.returncode != 0:
                print(f"❌ Failed: {config_path} seed {seed}")

    print("\n" + "=" * 60)
    print("Multi-seed execution complete!")
    print("Re-running figures and statistical tests...")
    print("=" * 60)
    subprocess.run([sys.executable, "scripts/generate_figures.py"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3], help="Seeds to run")
    args = parser.parse_args()
    run_multi_seeds(args.seeds)
