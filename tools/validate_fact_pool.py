"""
tools/validate_fact_pool.py

Validates the continual learning fact pool schema, quality, and metadata.

Checks:
  - Unique fact IDs
  - Required fields: id, title, passage, qa_pairs, domain, source_url, date_of_event
  - At least --min-questions (default: 4) per fact
  - date_of_event after / on or after --min-date (default: 2025-01-01)
  - Domain distribution and random adjacent-pair expectation (sum_d n_d*(n_d - 1) / N)
  - Warnings for domains with < 5 facts or total pool < 60 facts
  - Exits non-zero if any validation errors are detected.

Usage:
  python tools/validate_fact_pool.py
  python tools/validate_fact_pool.py --pool data/fact_pool.json --min-questions 4 --min-date 2025-01-01
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import os
import sys
from typing import Dict, List, Tuple

REQUIRED_FIELDS = [
    "id",
    "title",
    "passage",
    "qa_pairs",
    "domain",
    "source_url",
    "date_of_event",
]


def validate_pool(
    pool: List[dict],
    min_questions: int = 4,
    min_date_str: str = "2025-01-01",
) -> Tuple[List[str], List[str], Dict[str, int], float]:
    """
    Validate a list of fact dicts.

    Returns:
      errors: List of fatal validation error messages
      warnings: List of non-fatal shortfall/warning messages
      domain_counts: Counter dict of domain -> count
      random_expectation: Theoretical expected adjacent same-domain pairs
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not pool:
        errors.append("Fact pool is empty (0 facts loaded).")
        return errors, warnings, {}, 0.0

    # 1. Unique IDs
    seen_ids = set()
    duplicate_ids = set()
    for fact in pool:
        fid = fact.get("id")
        if not fid:
            errors.append(f"Fact missing 'id' attribute: {fact.get('title', 'Untitled')}")
        elif fid in seen_ids:
            duplicate_ids.add(fid)
        else:
            seen_ids.add(fid)
    if duplicate_ids:
        errors.append(f"Duplicate fact IDs found: {sorted(duplicate_ids)}")

    # Parse min_date
    try:
        min_date = datetime.strptime(min_date_str, "%Y-%m-%d").date()
    except ValueError:
        errors.append(f"Invalid --min-date format: '{min_date_str}' (expected YYYY-MM-DD)")
        min_date = None

    # 2. Field presence & values
    for fact in pool:
        fid = fact.get("id", "UNKNOWN_ID")
        for field in REQUIRED_FIELDS:
            if field not in fact or fact[field] is None:
                errors.append(f"[{fid}] Missing required field: '{field}'")
            elif isinstance(fact[field], str) and not fact[field].strip():
                errors.append(f"[{fid}] Empty string for required field: '{field}'")

        # Check qa_pairs
        qa_pairs = fact.get("qa_pairs")
        if not isinstance(qa_pairs, list):
            errors.append(f"[{fid}] 'qa_pairs' must be a list of QA objects")
        else:
            if len(qa_pairs) < min_questions:
                errors.append(
                    f"[{fid}] Has {len(qa_pairs)} question(s); required at least {min_questions}"
                )
            for idx, qa in enumerate(qa_pairs):
                if not isinstance(qa, dict):
                    errors.append(f"[{fid}] qa_pairs[{idx}] must be a dict")
                    continue
                if not qa.get("question") or not str(qa.get("question")).strip():
                    errors.append(f"[{fid}] qa_pairs[{idx}] missing or empty 'question'")
                if not qa.get("answer") or not str(qa.get("answer")).strip():
                    errors.append(f"[{fid}] qa_pairs[{idx}] missing or empty 'answer'")

        # Check date_of_event
        date_str = fact.get("date_of_event")
        if date_str and min_date is not None:
            try:
                event_date = datetime.strptime(str(date_str).strip(), "%Y-%m-%d").date()
                if event_date < min_date:
                    errors.append(
                        f"[{fid}] date_of_event {event_date} is before min_date {min_date}"
                    )
            except ValueError:
                errors.append(
                    f"[{fid}] Invalid date_of_event format '{date_str}' (expected YYYY-MM-DD)"
                )

    # 3. Domain counts & stats
    domains = [f.get("domain", "unknown") for f in pool]
    domain_counts = dict(Counter(domains))

    n_total = len(pool)
    if n_total > 0:
        sum_sq = sum(c * (c - 1) for c in domain_counts.values())
        random_expectation = sum_sq / n_total
    else:
        random_expectation = 0.0

    # 4. Warnings: domain imbalance and pool size
    if n_total < 60:
        warnings.append(
            f"Pool has only {n_total} facts (target is >= 60 facts for full benchmark)."
        )

    for domain, count in sorted(domain_counts.items(), key=lambda x: x[1]):
        if count < 5:
            warnings.append(
                f"Domain '{domain}' has only {count} fact(s) (< 5 required per domain for balance)."
            )

    return errors, warnings, domain_counts, random_expectation


def main():
    parser = argparse.ArgumentParser(description="Validate fact pool against SEAL schema and quality guidelines.")
    parser.add_argument("--pool", type=str, default="data/fact_pool.json",
                        help="Path to fact pool JSON file (default: data/fact_pool.json)")
    parser.add_argument("--min-questions", type=int, default=4,
                        help="Minimum number of QA pairs per fact (default: 4)")
    parser.add_argument("--min-date", type=str, default="2025-01-01",
                        help="Earliest allowed date_of_event YYYY-MM-DD (default: 2025-01-01)")
    args = parser.parse_args()

    print("=" * 70)
    print("SEAL Fact Pool Validation")
    print(f"  Pool file:      {args.pool}")
    print(f"  Min questions:  {args.min_questions}")
    print(f"  Min event date: {args.min_date}")
    print("=" * 70)

    if not os.path.exists(args.pool):
        print(f"\n[FATAL] Fact pool file not found: {args.pool}")
        sys.exit(1)

    try:
        with open(args.pool, "r", encoding="utf-8") as f:
            pool = json.load(f)
    except Exception as e:
        print(f"\n[FATAL] Failed to read/parse JSON from {args.pool}: {e}")
        sys.exit(1)

    errors, warnings, domain_counts, random_expectation = validate_pool(
        pool=pool,
        min_questions=args.min_questions,
        min_date_str=args.min_date,
    )

    print(f"\nPool Summary:")
    print(f"  Total facts:          {len(pool)}")
    print(f"  Domains ({len(domain_counts)}):")
    for domain, count in sorted(domain_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"    - {domain:15s}: {count}")
    print(f"  Random expectation (adjacent same-domain pairs): {random_expectation:.4f}")

    if warnings:
        print(f"\nShortfall Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"  [WARN] {w}")

    if errors:
        print(f"\nValidation Errors ({len(errors)}):")
        for e in errors[:25]:
            print(f"  [ERROR] {e}")
        if len(errors) > 25:
            print(f"  ... and {len(errors) - 25} more error(s).")
        print(f"\nResult: FAILED with {len(errors)} error(s).")
        sys.exit(1)
    else:
        print(f"\nResult: PASSED schema and quality validation.")
        sys.exit(0)


if __name__ == "__main__":
    main()
