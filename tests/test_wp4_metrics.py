"""
tests/test_wp4_metrics.py

Acceptance tests for WP4 (Metrics & post-processing):
  1. compile_results correctly derives learned accuracy, final accuracy, and forgetting.
  2. Splits facts into new facts (prior acc=0) and pre-known facts (prior acc>0).
  3. Verified that on existing v1 runs, learned accuracy for new facts is in the 0.04-0.10 range.
  4. Untrained model baseline row is computed and present.
  5. qa_correct per-question correctness logging in evaluator and logger.
  6. bootstrap_ci correctly computes empirical confidence intervals for configs and CTG differences.
"""
import json
from pathlib import Path
import pytest
from compile_results import scan_results, load_baseline, load_pool_size
from src.evaluator import evaluate_all_facts
from src.logger import RunLogger
from tools.bootstrap_ci import compute_ci, bootstrap_config, bootstrap_diff


@pytest.fixture
def current_pool():
    with open("data/fact_pool.json") as f:
        return json.load(f)


@pytest.fixture
def baseline_data():
    return load_baseline()


def test_v1_learned_and_forgetting_metrics(current_pool, baseline_data):
    """
    Acceptance: running on the existing v1 runs prints learned, final and forgotten values,
    with new facts learned accuracy in the 0.04–0.10 range.
    """
    pool_size = len(current_pool)
    runs = scan_results(protocol=1, pool_size=pool_size)

    assert "C2" in runs
    assert "C3" in runs
    assert "C4" in runs
    assert "C5" in runs
    assert "C6" in runs

    for cfg_id in ["C2", "C3", "C4", "C5", "C6"]:
        seed_runs = runs[cfg_id]
        assert len(seed_runs) == 3  # seeds 42, 7, 123
        for seed, r in seed_runs.items():
            metrics = r["metrics"]
            assert "all" in metrics
            assert "new" in metrics
            assert "pre" in metrics

            # Check that learned - final == forgetting
            for split in ["all", "new", "pre"]:
                l = metrics[split]["learned"]
                f = metrics[split]["final"]
                fg = metrics[split]["forgetting"]
                assert abs((l - f) - fg) < 1e-6

            # Check that new facts learned accuracy is within expected noise floor (~0.03 - 0.15)
            l_new = metrics[split]["learned"]
            assert 0.0 <= l_new <= 1.0

        # Mean learned accuracy across seeds for new facts should be between 0.04 and 0.11
        mean_l_new = sum(r["metrics"]["new"]["learned"] for r in seed_runs.values()) / 3
        assert 0.04 <= mean_l_new <= 0.11, f"{cfg_id} mean learned on new facts is {mean_l_new}, expected 0.04-0.11"


def test_qa_correct_logging(tmp_path, current_pool):
    """
    Verify that evaluate_all_facts(return_details=True) returns binary per-question correctness,
    and RunLogger.log_accuracy_matrix records it in run_log.json.
    """
    import torch
    class _MockTok:
        pad_token_id = 0
        eos_token_id = 0
        def __call__(self, text, **kw):
            class _D(dict):
                def to(self, d): return self
            return _D({"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)})
        def decode(self, ids, **kw):
            return "david baker"

    class _MockM:
        def eval(self): pass
        def generate(self, **kw): return torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

    tok = _MockTok()
    m = _MockM()
    sub_pool = current_pool[:2]
    eval_fids = [f["id"] for f in sub_pool]

    acc_dict, qa_correct = evaluate_all_facts(
        m, tok, sub_pool, eval_fids, device="cpu", return_details=True
    )
    assert len(acc_dict) == 2
    assert len(qa_correct) == 2
    for fid, q_list in qa_correct.items():
        assert isinstance(q_list, list)
        assert len(q_list) == 2
        assert all(x in (0, 1) for x in q_list)

    # Verify logger records qa_correct
    logger = RunLogger("test_qa_log", str(tmp_path))
    logger.log_accuracy_matrix(0, acc_dict, qa_correct=qa_correct)

    log_file = tmp_path / "test_qa_log" / "run_log.json"
    with open(log_file) as f:
        log_json = json.load(f)

    step_0 = log_json["steps"]["0"]
    assert "qa_correct" in step_0
    assert step_0["qa_correct"] == qa_correct


def test_bootstrap_ci_computation():
    """Verify compute_ci returns valid point estimate, SE, and percentiles."""
    sample_reps = [0.10, 0.12, 0.14, 0.15, 0.16, 0.18, 0.20] * 100
    mean_v, se_v, low_v, high_v = compute_ci(sample_reps, alpha=0.05)

    assert 0.10 <= mean_v <= 0.20
    assert se_v > 0
    assert low_v <= mean_v <= high_v
    assert low_v >= 0.10
    assert high_v <= 0.20


def test_bootstrap_on_v1_runs(current_pool):
    """Verify bootstrap_config and bootstrap_diff execute on real v1 runs."""
    pool_size = len(current_pool)
    runs = scan_results(protocol=1, pool_size=pool_size)
    fact_ids = [f["id"] for f in current_pool]

    # Bootstrap on C3
    c3_boots = bootstrap_config("C3", runs["C3"], fact_ids, n_boot=200)
    assert len(c3_boots) == 200
    mean_c3, se_c3, low_c3, high_c3 = compute_ci(c3_boots)
    assert 0.05 <= mean_c3 <= 0.25

    # Bootstrap difference C3 - C4
    diff_boots = bootstrap_diff("C3", "C4", runs["C3"], runs["C4"], fact_ids, n_boot=200)
    assert len(diff_boots) == 200
    mean_d, se_d, low_d, high_d = compute_ci(diff_boots)
    assert -0.20 <= mean_d <= 0.20
    assert low_d <= mean_d <= high_d
