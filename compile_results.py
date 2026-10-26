"""
compile_results.py  (WP1 rewrite)

Compile multi-seed results from results/ and print summary tables.

Key changes from v1:
  - Imports config registry from scripts/config_registry.py (single source of truth)
  - Only includes runs with end_time AND full step count (== pool size)
  - If multiple complete runs exist for (config, seed), uses the latest by folder name
    and prints a warning
  - For self-generated configs, computes and checks the fallback rate
    (exits non-zero if above 0.2 unless --allow-high-fallback is passed)
  - Exits non-zero if any expected (config, seed) is missing
  - Prints counts as questions (5/38), not just fractions
  - For 3 seeds shows each value and min-max, never std if all identical
  - Always prints an untrained model row from the baseline file
  - Reports CTG in questions and percentage points, with seed ranges,
    and prints WITHIN SEED RANGE when the gap is smaller than either arm's range
  - --protocol {1,2} filters on config.protocol_version (absent means 1)

Usage:
    python compile_results.py
    python compile_results.py --protocol 1
    python compile_results.py --allow-high-fallback
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scripts.config_registry import (
    FOLDER_PREFIX_TO_CONFIG,
    SEEDS_V1,
    all_configs,
    config_for_run_id,
)


RESULTS_DIR = ROOT / "results"
BASELINE_PATH = ROOT / "data" / "baseline_accuracy.json"
POOL_PATH = ROOT / "data" / "fact_pool.json"

SELF_GEN_CURRICULA = {"self_generated"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_pool_size() -> int:
    with open(POOL_PATH) as f:
        return len(json.load(f))


def load_baseline() -> dict[str, float]:
    """
    Load baseline accuracy per fact.
    Handles both:
      - Simple flat dict: {fact_id: float}
      - Nested dict: {fact_id: {qa_accuracy_before_training: float, ...}}
    Returns {fact_id: accuracy_float}.
    """
    if not BASELINE_PATH.exists():
        return {}
    with open(BASELINE_PATH) as f:
        raw = json.load(f)
    result = {}
    for fact_id, val in raw.items():
        if isinstance(val, (int, float)):
            result[fact_id] = float(val)
        elif isinstance(val, dict):
            result[fact_id] = float(val.get("qa_accuracy_before_training", 0.0))
    return result


def fallback_rate(log: dict) -> float | None:
    """
    Return the fraction of curriculum-choice steps where fallback_used=True.
    Returns None if there are no curriculum choices (planned curriculum).
    """
    choices = [
        step.get("curriculum_choice", {})
        for step in log.get("steps", {}).values()
        if "curriculum_choice" in step
    ]
    if not choices:
        return None
    n_fallback = sum(1 for c in choices if c.get("fallback_used", False))
    return n_fallback / len(choices)


def scan_results(protocol: int, pool_size: int) -> dict[str, dict[int, dict]]:
    """
    Scan results/ and return {config_id: {seed: {run_id, acc, fallback_rate}}}.
    Only includes complete, valid runs for the requested protocol.
    Warns when multiple complete runs exist for a (config, seed) and takes latest.
    """
    found: dict[str, dict[int, list]] = {}

    for entry in RESULTS_DIR.iterdir():
        if not entry.is_dir() or entry.name == "archive":
            continue
        run_id = entry.name
        cfg_id = config_for_run_id(run_id)
        if cfg_id is None:
            continue

        log_path = entry / "run_log.json"
        if not log_path.exists():
            continue

        with open(log_path) as f:
            log = json.load(f)

        # Protocol filter: absent protocol_version counts as v1
        run_protocol = log.get("config", {}).get("protocol_version", 1)
        if run_protocol != protocol:
            continue

        # Completeness check
        if "end_time" not in log:
            continue
        n_steps = len(log.get("steps", {}))
        if n_steps != pool_size:
            continue

        seed = log.get("config", {}).get("seed")
        if seed is None:
            continue

        # Compute final accuracy
        steps = log.get("steps", {})
        last_step = str(max(int(k) for k in steps.keys()))
        final_acc = steps[last_step].get("mean_accuracy", 0.0)

        fb = fallback_rate(log)

        found.setdefault(cfg_id, {}).setdefault(seed, []).append(
            {"run_id": run_id, "acc": final_acc, "fallback_rate": fb}
        )

    # For each (config, seed), keep only the latest run_id
    result: dict[str, dict[int, dict]] = {}
    for cfg_id, seed_map in found.items():
        result[cfg_id] = {}
        for seed, runs in seed_map.items():
            if len(runs) > 1:
                runs.sort(key=lambda r: r["run_id"])
                print(
                    f"[WARN] Multiple complete runs for ({cfg_id}, seed={seed}): "
                    + ", ".join(r["run_id"] for r in runs)
                    + f" — using latest: {runs[-1]['run_id']}"
                )
            result[cfg_id][seed] = runs[-1]

    return result


def questions_str(acc: float, n_questions: int) -> str:
    """Format accuracy as 'N/38 (PP.PP%)'."""
    q = round(acc * n_questions)
    return f"{q}/{n_questions} ({acc*100:.2f}%)"


def ctg_str(acc_planned: float, acc_selfgen: float, n_q: int) -> str:
    diff = acc_planned - acc_selfgen
    q_diff = round(diff * n_q)
    sign = "+" if diff >= 0 else ""
    return f"{sign}{q_diff}/{n_q} ({sign}{diff*100:.2f}pp)"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compile SEAL multi-seed results.")
    parser.add_argument(
        "--protocol", type=int, default=1, choices=[1, 2],
        help="Protocol version to filter on (default: 1).",
    )
    parser.add_argument(
        "--allow-high-fallback", action="store_true",
        help="Do not exit non-zero when fallback rate > 0.2 for self-gen runs.",
    )
    args = parser.parse_args()

    pool_size = load_pool_size()
    n_questions = pool_size * 2  # 2 QA pairs per fact
    seeds = SEEDS_V1

    cfgs = all_configs(args.protocol)
    baseline = load_baseline()

    # Untrained baseline row: mean accuracy across all facts in baseline file
    if baseline:
        baseline_acc = sum(baseline.values()) / len(baseline)
    else:
        baseline_acc = None

    # Scan results
    found = scan_results(args.protocol, pool_size)

    exit_code = 0

    # ── Check completeness ────────────────────────────────────────────────────
    missing = []
    for cfg_id in cfgs:
        for seed in seeds:
            if cfg_id not in found or seed not in found.get(cfg_id, {}):
                missing.append(f"({cfg_id}, seed={seed})")
    if missing:
        print("[ERROR] Missing expected (config, seed) pairs:")
        for m in missing:
            print(f"  {m}")
        print("  Re-run missing experiments before compiling.")
        exit_code = 1

    # ── Check fallback rates ──────────────────────────────────────────────────
    HIGH_FALLBACK_THRESHOLD = 0.2
    high_fallback_cfgs = []
    for cfg_id, seed_map in found.items():
        cfg_info = cfgs.get(cfg_id, {})
        if cfg_info.get("curriculum") not in SELF_GEN_CURRICULA:
            continue
        for seed, run_data in seed_map.items():
            fb = run_data.get("fallback_rate")
            if fb is not None and fb > HIGH_FALLBACK_THRESHOLD:
                high_fallback_cfgs.append(
                    f"  ({cfg_id}, seed={seed}): fallback_rate={fb:.2%} > {HIGH_FALLBACK_THRESHOLD:.0%}"
                )
    if high_fallback_cfgs and not args.allow_high_fallback:
        print(
            "[ERROR] High fallback rate (> 20%) in self-generated curriculum runs.\n"
            "  These results are NOT valid self-generated curricula.\n"
            "  Fix the curriculum parser and re-run, OR pass --allow-high-fallback."
        )
        for line in high_fallback_cfgs:
            print(line)
        exit_code = 1

    # ── Print results table ───────────────────────────────────────────────────
    print()
    print("=" * 80)
    print(f"  SEAL RESULTS  (protocol v{args.protocol}, {n_questions} evaluation questions)")
    print("=" * 80)
    print(
        f"  {'ID':<5} {'Method':<25} "
        + "".join(f"  s={s:<6}" for s in seeds)
        + f"  {'Min':>7}  {'Max':>7}"
    )
    print("  " + "-" * 76)

    summary: dict[str, dict] = {}

    # Untrained model row
    if baseline_acc is not None:
        b_q = round(baseline_acc * n_questions)
        b_row = f"  {'---':<5} {'Untrained model':<25}"
        b_row += f"  {b_q}/{n_questions} ({baseline_acc*100:.2f}%)" + "  (same pre-training state)"
        print(b_row)
        print("  " + "-" * 76)

    for cfg_id, cfg_info in cfgs.items():
        label = cfg_info["label"]
        seed_accs = {}
        for seed in seeds:
            run = found.get(cfg_id, {}).get(seed)
            if run:
                seed_accs[seed] = run["acc"]

        vals = [seed_accs.get(s) for s in seeds]
        vals_clean = [v for v in vals if v is not None]

        if vals_clean:
            mean_acc = sum(vals_clean) / len(vals_clean)
            min_acc = min(vals_clean)
            max_acc = max(vals_clean)
        else:
            mean_acc = min_acc = max_acc = 0.0

        summary[cfg_id] = {
            "mean": mean_acc, "min": min_acc, "max": max_acc,
            "per_seed": seed_accs,
            "label": label,
        }

        row = f"  {cfg_id:<5} {label:<25}"
        for s in seeds:
            v = seed_accs.get(s)
            if v is not None:
                q = round(v * n_questions)
                row += f"  {q:>2}/{n_questions:<3}"
            else:
                row += f"  {'--':>6} "
        min_q = round(min_acc * n_questions)
        max_q = round(max_acc * n_questions)
        all_same = (min_q == max_q)
        row += f"  {min_q:>3}/{n_questions}" if not all_same else f"  {'(all same)':>7}"
        row += f"  {max_q:>3}/{n_questions}" if not all_same else ""
        print(row)

    print()
    print("  CURRICULUM TRANSFER GAPS (CTG = planned_acc - selfgen_acc):")
    print("  " + "-" * 60)

    def _ctg_analysis(planned_id: str, selfgen_id: str, method: str):
        if planned_id not in summary or selfgen_id not in summary:
            print(f"  {method}: missing data.")
            return
        p = summary[planned_id]
        s = summary[selfgen_id]
        diff_mean = p["mean"] - s["mean"]
        q_diff = round(diff_mean * n_questions)
        sign = "+" if diff_mean >= 0 else ""

        # Range-based significance check
        p_range = p["max"] - p["min"]
        s_range = s["max"] - s["min"]
        max_range = max(p_range, s_range)
        within_range = abs(diff_mean) <= max_range

        flag = "  << WITHIN SEED RANGE (not significant)" if within_range else ""
        direction = "planned better" if diff_mean > 0 else ("self-gen better" if diff_mean < 0 else "tied")
        print(
            f"  {method} CTG ({planned_id} - {selfgen_id}): "
            f"{sign}{q_diff}/{n_questions} ({sign}{diff_mean*100:.2f}pp) [{direction}]{flag}"
        )
        print(
            f"    Planned  {planned_id}: {p['min']*100:.2f}%–{p['max']*100:.2f}% (range {p_range*100:.2f}pp)"
        )
        print(
            f"    Self-gen {selfgen_id}: {s['min']*100:.2f}%–{s['max']*100:.2f}% (range {s_range*100:.2f}pp)"
        )

    _ctg_analysis("C3", "C4", "O-LoRA")
    print()
    _ctg_analysis("C6", "C5", "LoRI  ")

    # Baseline improvement
    if baseline_acc is not None and "C2" in summary:
        c2_mean = summary["C2"]["mean"]
        print()
        print(f"  Improvement over C2 baseline (mean={round(c2_mean*n_questions)}/{n_questions}):")
        for cfg_id in ["C3", "C4", "C5", "C6"]:
            if cfg_id in summary:
                d = summary[cfg_id]["mean"] - c2_mean
                q_d = round(d * n_questions)
                sign = "+" if d >= 0 else ""
                print(f"    {cfg_id}: {sign}{q_d}/{n_questions} ({sign}{d*100:.2f}pp)")
    print("=" * 80)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out = {}
    for cfg_id, data in summary.items():
        out[cfg_id] = {
            "mean": data["mean"],
            "min": data["min"],
            "max": data["max"],
            "per_seed": {str(k): v for k, v in data["per_seed"].items()},
        }
    if baseline_acc is not None:
        out["untrained"] = {"mean": baseline_acc}
    out_path = RESULTS_DIR / "final_results_summary.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {out_path}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
