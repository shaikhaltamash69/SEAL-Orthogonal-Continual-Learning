"""
tests/test_wp5_eval_and_data_quality.py

Unit tests for Work Package 5:
  1. Strict answer matching in src/evaluator.py:
     - expected 'Paris' vs output 'Parisian' -> False
     - expected 'Paris' vs output 'Paris, France' -> True
     - numeric normalisation: '1,000' vs '1000' -> True
     - answer aliases hit -> True
     - contains mode vs strict mode behavior
  2. Pool validator in tools/validate_fact_pool.py:
     - runs on active fact_pool.json and reports shortfalls without crashing
     - correctly identifies duplicate IDs, missing fields, insufficient questions, old dates
     - passes clean synthetic pools with 0 errors
  3. Novelty guard in scripts/run_config.py:
     - rejects facts flagged keep_in_pool: false
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.evaluator import match_answer, normalize_answer
from tools.validate_fact_pool import validate_pool


def test_normalize_answer():
    # Punctuation stripped, lowercase, whitespace collapsed, number comma stripped
    raw = "  Nobel Prize,  2024: 1,000!  "
    norm = normalize_answer(raw)
    assert norm == "nobel prize 2024 1000"


def test_strict_match_parisian_rejected():
    # Expected 'Paris' should NOT match 'Parisian' under strict mode
    expected = "Paris"
    output = "The answer is Parisian architecture."
    assert not match_answer(generated=output, expected_answer=expected, eval_match="strict")

    # But under contains mode, it would match (substring)
    assert match_answer(generated=output, expected_answer=expected, eval_match="contains")


def test_strict_match_phrase_accepted():
    # Expected 'Paris' should match 'Paris, France'
    expected = "Paris"
    output = "The capital is Paris, France."
    assert match_answer(generated=output, expected_answer=expected, eval_match="strict")


def test_numeric_normalization_match():
    # '1,000' should match '1000'
    expected = "1,000"
    output = "Approximately 1000 participants joined."
    assert match_answer(generated=output, expected_answer=expected, eval_match="strict")

    expected2 = "1000"
    output2 = "The total capacity was 1,000 seats."
    assert match_answer(generated=output2, expected_answer=expected2, eval_match="strict")


def test_answer_aliases_match():
    expected = "USA"
    aliases = ["United States", "United States of America", "US"]
    output = "The team represents the United States."
    assert match_answer(
        generated=output,
        expected_answer=expected,
        answer_aliases=aliases,
        eval_match="strict",
    )


def test_validator_runs_on_current_pool_without_crashing():
    pool_path = Path("data/fact_pool.json")
    assert pool_path.exists()
    with open(pool_path, "r", encoding="utf-8") as f:
        pool = json.load(f)

    errors, warnings, domain_counts, random_exp = validate_pool(
        pool=pool,
        min_questions=4,
        min_date_str="2025-01-01",
    )

    # Current pool has 19 facts, 2 questions each, late 2024 dates
    assert len(pool) == 19
    assert len(errors) > 0  # reports the shortfalls
    assert any("required at least 4" in e for e in errors)
    assert any("before min_date 2025-01-01" in e for e in errors)
    assert len(warnings) > 0
    assert any("Pool has only 19 facts" in w for w in warnings)
    assert abs(random_exp - 4.8421) < 0.01


def test_validator_synthetic_clean_pool():
    clean_pool = [
        {
            "id": f"fact_{i:03d}",
            "title": f"Fact Title {i}",
            "passage": f"Passage text for fact {i}.",
            "qa_pairs": [
                {"question": f"Q{j}?", "answer": f"A{j}"} for j in range(4)
            ],
            "domain": f"domain_{i % 5}",
            "source_url": "https://example.com",
            "date_of_event": "2025-05-10",
        }
        for i in range(60)
    ]

    errors, warnings, domain_counts, random_exp = validate_pool(
        pool=clean_pool,
        min_questions=4,
        min_date_str="2025-01-01",
    )
    assert errors == []
    assert warnings == []
    assert len(domain_counts) == 5
    assert all(c == 12 for c in domain_counts.values())


def test_validator_detects_malformed_pool():
    bad_pool = [
        {
            "id": "fact_001",
            "title": "Dup 1",
            "passage": "Text",
            "qa_pairs": [{"question": "Q?", "answer": "A"}],
            "domain": "sports",
            "source_url": "https://example.com",
            "date_of_event": "2023-01-01",
        },
        {
            "id": "fact_001",  # duplicate
            "title": "Dup 2",
            # missing passage, domain, source_url
            "qa_pairs": [],
            "date_of_event": "invalid-date",
        },
    ]
    errors, warnings, domain_counts, random_exp = validate_pool(bad_pool, min_questions=2)
    err_text = " ".join(errors)
    assert "Duplicate fact IDs" in err_text
    assert "Missing required field: 'passage'" in err_text
    assert "Invalid date_of_event format" in err_text


def test_run_config_novelty_gate_rejects_disallowed_fact(tmp_path):
    # Create temporary baseline file with fact_001 marked keep_in_pool: false
    fake_baseline = {
        "fact_001": {
            "fact_id": "fact_001",
            "qa_accuracy_before_training": 1.0,
            "keep_in_pool": False,
        }
    }
    baseline_file = tmp_path / "test_baseline.json"
    with open(baseline_file, "w") as f:
        json.dump(fake_baseline, f)

    fake_config = {
        "config_name": "test_novelty_gate",
        "baseline_accuracy_path": str(baseline_file),
        "curriculum": "planned",
    }
    config_file = tmp_path / "test_config.json"
    with open(config_file, "w") as f:
        json.dump(fake_config, f)

    # Invoking run_config.py should exit with code 1
    res = subprocess.run(
        [sys.executable, "scripts/run_config.py", "--config", str(config_file), "--seed", "42"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 1
    assert "flagged 'keep_in_pool: false'" in res.stdout
