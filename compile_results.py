"""
compile_results.py  (WP1 / WP4 updated)

Compile multi-seed results from results/ and print summary tables.

Key features:
  - Imports config registry from scripts/config_registry.py (single source of truth)
  - Only includes runs with end_time and complete step count
  - Warns on multiple complete runs for (config, seed) and takes the latest
  - Computes fallback rates and exits non-zero if fallback rate > 0.20 (override: --allow-high-fallback)
  - Exits non-zero if any expected (config, seed) is missing
  - Prints accuracy as question counts (e.g. 5/38) and shows per-seed values + min-max ranges
  - WP4: Computes and prints learned accuracy, final accuracy, and forgetting (learned - final),
    split by ALL facts, NEW facts (prior acc = 0), and PRE-KNOWN facts (prior acc > 0)
  - WP4: Always prints Untrained Model baseline row in every table
  - CTG reporting with seed-range overlap check (WITHIN SEED RANGE)
  - Protocol selection via --protocol {1, 2}

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
    SEEDS_V2,
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
    Scan results/ and return {config_id: {seed: {run_id, acc, fallback_rate, metrics}}}.
    Only includes complete, valid runs for the requested protocol.
    Warns when multiple complete runs exist for a (config, seed) and takes latest.
    Computes WP4 metrics (learned, final, forgetting) for all, new, and pre-known facts.
    """
    baseline = load_baseline()
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

        try:
            with open(log_path) as f:
                log = json.load(f)
        except Exception:
            continue

        # Protocol filter: absent protocol_version counts as v1
        run_protocol = log.get("config", {}).get("protocol_version", 1)
        if run_protocol != protocol:
            continue

        # Completeness check
        if "end_time" not in log:
            continue
        steps = log.get("steps", {})
        if len(steps) != pool_size:
            continue

        seed = log.get("config", {}).get("seed")
        if seed is None:
            continue

        order = log.get("curriculum_order_used", [])
        if not order:
            # Fallback to step numerical order
            sorted_step_keys = sorted(steps.keys(), key=lambda k: int(k))
            order = [steps[k].get("fact_id") for k in sorted_step_keys if "fact_id" in steps[k]]

        last_step_key = str(max(int(k) for k in steps.keys()))
        last_acc_matrix = steps[last_step_key].get("accuracy_matrix", {})
        final_acc = steps[last_step_key].get("mean_accuracy", 0.0)

        # WP4: Compute per-fact learned, final, and forgetting
        learned_dict = {}
        final_dict = {}
        forget_dict = {}
        for s_idx, fid in enumerate(order):
            s_key = str(s_idx)
            s_acc_matrix = steps.get(s_key, {}).get("accuracy_matrix", {})
            l_acc = s_acc_matrix.get(fid, 0.0)
            f_acc = last_acc_matrix.get(fid, 0.0)
            learned_dict[fid] = l_acc
            final_dict[fid] = f_acc
            forget_dict[fid] = l_acc - f_acc

        new_facts = [fid for fid in order if baseline.get(fid, 0.0) == 0.0]
        pre_facts = [fid for fid in order if baseline.get(fid, 0.0) > 0.0]

        def _calc_split(fact_subset: list[str]) -> dict[str, float]:
            if not fact_subset:
                return {"learned": 0.0, "final": 0.0, "forgetting": 0.0}
            m_l = sum(learned_dict[f] for f in fact_subset) / len(fact_subset)
            m_f = sum(final_dict[f] for f in fact_subset) / len(fact_subset)
            return {"learned": m_l, "final": m_f, "forgetting": m_l - m_f}

        metrics = {
            "all": _calc_split(order),
            "new": _calc_split(new_facts),
            "pre": _calc_split(pre_facts),
        }

        fb = fallback_rate(log)

        found.setdefault(cfg_id, {}).setdefault(seed, []).append({
            "run_id": run_id,
            "acc": final_acc,
            "fallback_rate": fb,
            "metrics": metrics,
        })

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


def pp_str(diff: float, n_q: int) -> str:
    """Format difference as '+N/38 (+PP.PPpp)'."""
    q = round(diff * n_q)
    sign = "+" if diff >= 0 else ""
    return f"{sign}{q}/{n_q} ({sign}{diff*100:.2f}pp)"


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
    n_questions_all = pool_size * 2
    seeds = SEEDS_V1 if args.protocol == 1 else SEEDS_V2

    cfgs = all_configs(args.protocol)
    baseline = load_baseline()

    # Determine fact split counts
    new_facts_count = sum(1 for v in baseline.values() if v == 0.0) if baseline else 16
    pre_facts_count = sum(1 for v in baseline.values() if v > 0.0) if baseline else 3
    n_q_new = new_facts_count * 2
    n_q_pre = pre_facts_count * 2

    # Baseline accuracy splits
    if baseline:
        base_acc_all = sum(baseline.values()) / len(baseline)
        base_acc_new = 0.0
        pre_vals = [v for v in baseline.values() if v > 0.0]
        base_acc_pre = sum(pre_vals) / len(pre_vals) if pre_vals else 0.0
    else:
        base_acc_all = base_acc_new = base_acc_pre = None

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
    HIGH_FALLBACK_THRESHOLD = 0.20
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

    # ── Summary Data Structure ────────────────────────────────────────────────
    summary: dict[str, dict] = {}

    for cfg_id, cfg_info in cfgs.items():
        label = cfg_info["label"]
        seed_runs = found.get(cfg_id, {})
        seed_final_accs = {s: seed_runs[s]["acc"] for s in seeds if s in seed_runs}

        vals_clean = list(seed_final_accs.values())
        mean_acc = sum(vals_clean) / len(vals_clean) if vals_clean else 0.0
        min_acc = min(vals_clean) if vals_clean else 0.0
        max_acc = max(vals_clean) if vals_clean else 0.0

        # WP4 metric splits per seed
        splits_per_seed = {}
        for s in seeds:
            if s in seed_runs:
                splits_per_seed[s] = seed_runs[s].get("metrics", {})

        summary[cfg_id] = {
            "mean": mean_acc,
            "min": min_acc,
            "max": max_acc,
            "per_seed": seed_final_accs,
            "label": label,
            "splits": splits_per_seed,
        }

    # ── Table 1: Final Accuracy (Overview) ────────────────────────────────────
    print()
    print("=" * 85)
    print(f"  SEAL RESULTS: FINAL ACCURACY (Protocol v{args.protocol}, {n_questions_all} questions)")
    print("=" * 85)
    print(
        f"  {'ID':<5} {'Method':<25} "
        + "".join(f"  s={s:<6}" for s in seeds)
        + f"  {'Min':>7}  {'Max':>7}"
    )
    print("  " + "-" * 81)

    # Untrained model row
    if base_acc_all is not None:
        b_q = round(base_acc_all * n_questions_all)
        b_row = f"  {'---':<5} {'Untrained model':<25}"
        b_row += f"  {b_q}/{n_questions_all} ({base_acc_all*100:.2f}%)" + "  (same pre-training state)"
        print(b_row)
        print("  " + "-" * 81)

    for cfg_id in cfgs:
        if cfg_id not in summary:
            continue
        data = summary[cfg_id]
        label = data["label"]
        seed_accs = data["per_seed"]
        row = f"  {cfg_id:<5} {label:<25}"
        for s in seeds:
            v = seed_accs.get(s)
            if v is not None:
                q = round(v * n_questions_all)
                row += f"  {q:>2}/{n_questions_all:<3}"
            else:
                row += f"  {'--':>6} "
        min_q = round(data["min"] * n_questions_all)
        max_q = round(data["max"] * n_questions_all)
        all_same = (min_q == max_q)
        row += f"  {min_q:>3}/{n_questions_all}" if not all_same else f"  {'(all same)':>7}"
        row += f"  {max_q:>3}/{n_questions_all}" if not all_same else ""
        print(row)

    # ── Table 2: WP4 Continual Learning Metrics by Fact Split ─────────────────
    def _print_split_table(split_key: str, split_title: str, n_q: int, base_val: float | None):
        print()
        print("-" * 85)
        print(f"  SPLIT: {split_title} ({n_q} questions)")
        print("-" * 85)
        print(f"  {'Config':<5} {'Method':<24} {'Learned (mean [min, max])':<27} {'Final (mean)':<14} {'Forgetting':<14}")
        print("  " + "-" * 81)

        # Untrained row
        if base_val is not None:
            b_q_str = questions_str(base_val, n_q)
            print(f"  {'---':<5} {'Untrained model':<24} {b_q_str:<27} {b_q_str:<14} {'0/0 (0.00pp)':<14}")
            print("  " + "-" * 81)

        for cfg_id in cfgs:
            if cfg_id not in summary:
                continue
            data = summary[cfg_id]
            label = data["label"]
            splits = data["splits"]

            learned_vals = [splits[s][split_key]["learned"] for s in seeds if s in splits and split_key in splits[s]]
            final_vals = [splits[s][split_key]["final"] for s in seeds if s in splits and split_key in splits[s]]
            forget_vals = [splits[s][split_key]["forgetting"] for s in seeds if s in splits and split_key in splits[s]]

            if not learned_vals:
                continue

            m_l = sum(learned_vals) / len(learned_vals)
            min_l, max_l = min(learned_vals), max(learned_vals)
            m_f = sum(final_vals) / len(final_vals)
            m_fg = sum(forget_vals) / len(forget_vals)

            learned_display = f"{round(m_l*n_q)}/{n_q} ({m_l*100:.1f}%) [{round(min_l*n_q)}-{round(max_l*n_q)}]"
            final_display = f"{round(m_f*n_q)}/{n_q} ({m_f*100:.1f}%)"
            forget_display = pp_str(m_fg, n_q)

            print(f"  {cfg_id:<5} {label:<24} {learned_display:<27} {final_display:<14} {forget_display:<14}")

    _print_split_table("all", f"ALL FACTS ({pool_size} facts)", n_questions_all, base_acc_all)
    _print_split_table("new", f"NEW FACTS ONLY ({new_facts_count} facts, prior accuracy = 0%)", n_q_new, base_acc_new)
    _print_split_table("pre", f"PRE-KNOWN FACTS ({pre_facts_count} facts, prior accuracy > 0%)", n_q_pre, base_acc_pre)

    # ── Table 3: CTG Analysis ─────────────────────────────────────────────────
    print()
    print("=" * 85)
    print("  CURRICULUM TRANSFER GAPS (CTG = planned - self_gen)")
    print("=" * 85)

    def _ctg_analysis(planned_id: str, selfgen_id: str, method: str, split_key: str = "all", n_q: int = n_questions_all):
        if planned_id not in summary or selfgen_id not in summary:
            return
        p_splits = summary[planned_id]["splits"]
        s_splits = summary[selfgen_id]["splits"]

        p_finals = [p_splits[s][split_key]["final"] for s in seeds if s in p_splits]
        s_finals = [s_splits[s][split_key]["final"] for s in seeds if s in s_splits]

        if not p_finals or not s_finals:
            return

        p_mean = sum(p_finals) / len(p_finals)
        s_mean = sum(s_finals) / len(s_finals)
        diff_mean = p_mean - s_mean

        p_range = max(p_finals) - min(p_finals)
        s_range = max(s_finals) - min(s_finals)
        max_range = max(p_range, s_range)
        within_range = abs(diff_mean) <= max_range

        flag = "  << WITHIN SEED RANGE (not statistically significant)" if within_range else ""
        direction = "planned better" if diff_mean > 0 else ("self-gen better" if diff_mean < 0 else "tied")
        q_diff = round(diff_mean * n_q)
        sign = "+" if diff_mean >= 0 else ""

        print(
            f"  {method} CTG ({planned_id} - {selfgen_id}) [{split_key.upper()}]: "
            f"{sign}{q_diff}/{n_q} ({sign}{diff_mean*100:.2f}pp) [{direction}]{flag}"
        )
        print(
            f"    Planned  {planned_id}: {min(p_finals)*100:.1f}% - {max(p_finals)*100:.1f}% (range {p_range*100:.1f}pp)"
        )
        print(
            f"    Self-gen {selfgen_id}: {min(s_finals)*100:.1f}% - {max(s_finals)*100:.1f}% (range {s_range*100:.1f}pp)"
        )

    print("  Overall (All Facts):")
    _ctg_analysis("C3", "C4", "O-LoRA", "all", n_questions_all)
    print()
    _ctg_analysis("C6", "C5", "LoRI  ", "all", n_questions_all)

    print("\n  On Genuinely New Facts Only:")
    _ctg_analysis("C3", "C4", "O-LoRA", "new", n_q_new)
    print()
    _ctg_analysis("C6", "C5", "LoRI  ", "new", n_q_new)
    print("=" * 85)

    # ── Save JSON Summary ─────────────────────────────────────────────────────
    out = {}
    for cfg_id, data in summary.items():
        out[cfg_id] = {
            "mean": data["mean"],
            "min": data["min"],
            "max": data["max"],
            "per_seed": {str(k): v for k, v in data["per_seed"].items()},
            "splits": data["splits"],
        }
    if base_acc_all is not None:
        out["untrained"] = {
            "mean": base_acc_all,
            "all": base_acc_all,
            "new": base_acc_new,
            "pre": base_acc_pre,
        }
    out_path = RESULTS_DIR / "final_results_summary.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {out_path}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
