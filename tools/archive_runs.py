"""
tools/archive_runs.py

WP1: Classify and archive invalid/obsolete run folders from results/.

Categories
----------
  valid           end_time present AND step count == pool size (19).
  aborted         end_time missing OR step count < pool size.
  broken_selection dated 20260901 or 20260902 (self-selection always fell back;
                  the planned C3 run from that date was also valid but is
                  superseded by the Oct run).
  smoke           name starts with SMOKE_TEST.

Non-valid runs are moved with `git mv` to results/archive/<category>/ and
MANIFEST.md is updated.

Usage (dry-run, default):
    python tools/archive_runs.py

Usage (apply moves):
    python tools/archive_runs.py --apply

Anything ambiguous is printed but NOT moved -- inspect manually.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ARCHIVE = RESULTS / "archive"
POOL_PATH = ROOT / "data" / "fact_pool.json"

# Date stamps whose self-generated selection always fell back
BROKEN_SELECTION_DATES = {"20260901", "20260902"}


def classify_run(run_dir: Path, pool_size: int) -> tuple[str, str]:
    """
    Return (category, reason) for a run folder.
    category ∈ {valid, aborted, broken_selection, smoke, ambiguous}
    """
    name = run_dir.name

    # smoke
    if name.startswith("SMOKE_TEST"):
        return "smoke", "Name starts with SMOKE_TEST"

    # broken selection — check date stamp in the folder name
    parts = name.split("_")
    for part in parts:
        if len(part) == 8 and part.isdigit() and part[:6] in {d[:6] for d in BROKEN_SELECTION_DATES}:
            date_part = part
            if date_part in BROKEN_SELECTION_DATES:
                return "broken_selection", f"Dated {date_part}: self-selection always fell back"

    log_path = run_dir / "run_log.json"
    if not log_path.exists():
        return "aborted", "run_log.json missing"

    try:
        with open(log_path) as f:
            log = json.load(f)
    except json.JSONDecodeError as e:
        return "ambiguous", f"run_log.json is malformed: {e}"

    has_end = "end_time" in log
    n_steps = len(log.get("steps", {}))

    if has_end and n_steps == pool_size:
        return "valid", f"Complete: {n_steps}/{pool_size} steps with end_time"
    elif has_end and n_steps != pool_size:
        return "aborted", f"end_time present but only {n_steps}/{pool_size} steps"
    else:
        return "aborted", f"No end_time, {n_steps}/{pool_size} steps"


def git_mv(src: Path, dst: Path, dry_run: bool) -> bool:
    """Run `git mv src dst`. Returns True on success."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "mv", str(src), str(dst)]
    if dry_run:
        print(f"  [DRY-RUN] {' '.join(cmd)}")
        return True
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] git mv failed: {result.stderr.strip()}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="WP1: classify and archive invalid SEAL run folders."
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually perform git mv moves. Default is dry-run (prints only).",
    )
    parser.add_argument(
        "--results-dir", default=str(RESULTS),
        help="Path to results/ directory.",
    )
    args = parser.parse_args()
    dry_run = not args.apply
    results_dir = Path(args.results_dir)

    # Load pool size
    with open(POOL_PATH) as f:
        pool_size = len(json.load(f))
    print(f"[Archive] Pool size: {pool_size}")
    print(f"[Archive] Dry-run: {dry_run} (pass --apply to move files)")
    print()

    run_dirs = sorted(
        [d for d in results_dir.iterdir() if d.is_dir() and d.name != "archive"],
    )

    categories: dict[str, list[dict]] = {
        "valid": [], "aborted": [], "broken_selection": [], "smoke": [], "ambiguous": [],
    }

    for run_dir in run_dirs:
        cat, reason = classify_run(run_dir, pool_size)
        categories[cat].append({"run_id": run_dir.name, "reason": reason})

    # Print summary
    for cat in ["valid", "aborted", "broken_selection", "smoke", "ambiguous"]:
        items = categories[cat]
        print(f"[{cat.upper()}] ({len(items)} runs)")
        for item in items:
            print(f"  {item['run_id']:<50}  {item['reason']}")
        print()

    # Move non-valid runs (skip valid and ambiguous)
    to_move = [
        (cat, item)
        for cat in ["aborted", "broken_selection", "smoke"]
        for item in categories[cat]
    ]

    if not to_move:
        print("[Archive] Nothing to move.")
        return

    manifest_lines: list[str] = []
    moved = 0
    for cat, item in to_move:
        src = results_dir / item["run_id"]
        dst = results_dir / "archive" / cat / item["run_id"]
        if dst.exists():
            print(f"  [SKIP] Already archived: {dst}")
            continue
        ok = git_mv(src, dst, dry_run)
        if ok:
            moved += 1
            manifest_lines.append(
                f"| {item['run_id']} | {cat} | {item['reason']} | "
                f"{datetime.now().strftime('%Y-%m-%d')} |"
            )

    # Update MANIFEST.md
    manifest_path = results_dir / "archive" / "MANIFEST.md"
    if dry_run:
        print(f"\n[DRY-RUN] Would write {len(manifest_lines)} entries to {manifest_path}")
    else:
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        existing = ""
        if manifest_path.exists():
            existing = manifest_path.read_text(encoding="utf-8")
        if not existing.strip():
            header = (
                "# Archive Manifest\n\n"
                "Runs moved out of `results/` by `tools/archive_runs.py`.\n\n"
                "| run_id | category | reason | archived_date |\n"
                "|---|---|---|---|\n"
            )
            existing = header
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(existing.rstrip("\n") + "\n")
            for line in manifest_lines:
                f.write(line + "\n")
        print(f"\n[Archive] MANIFEST updated: {manifest_path}")

    action = "Would move" if dry_run else "Moved"
    print(f"\n[Archive] {action} {moved}/{len(to_move)} runs.")
    if dry_run:
        print("[Archive] Re-run with --apply to perform the moves.")
    if categories["ambiguous"]:
        print(f"\n[Archive] WARNING: {len(categories['ambiguous'])} ambiguous run(s) — inspect manually:")
        for item in categories["ambiguous"]:
            print(f"  {item['run_id']}: {item['reason']}")


if __name__ == "__main__":
    main()
