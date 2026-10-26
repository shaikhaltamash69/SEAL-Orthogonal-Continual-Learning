"""
tests/test_wp3_control_curricula.py

Acceptance tests for WP3 (Control curricula):
  1. DispersedCurriculum yields fewer adjacent same-domain pairs than planned order for current pool.
  2. RandomCurriculum is reproducible given seed and produces valid permutation.
  3. order_stats matches formula: random_expectation = Σ_d n_d(n_d - 1) / N (~4.84 for pool).
  4. order_stats logged at finalize.
  5. Config registry and v2 configs (C2-C10) are valid and runnable.
"""
import json
from collections import Counter
from pathlib import Path
import pytest
from src.curriculum import PlannedCurriculum, RandomCurriculum, DispersedCurriculum
from src.logger import RunLogger
from scripts.config_registry import all_configs, CONFIGS_V1, CONFIGS_V2


@pytest.fixture
def current_pool():
    pool_path = Path("data/fact_pool.json")
    with open(pool_path) as f:
        return json.load(f)


def count_adjacent_same_domain(fact_ids, fact_pool):
    dom_map = {f["id"]: f.get("domain", "unknown") for f in fact_pool}
    return sum(
        1 for i in range(len(fact_ids) - 1)
        if dom_map.get(fact_ids[i]) == dom_map.get(fact_ids[i + 1])
    )


def test_dispersed_curriculum_reduces_adjacent_pairs(current_pool):
    """
    Acceptance: DispersedCurriculum yields fewer adjacent same-domain pairs
    than the planned order for the current pool.
    """
    planned = PlannedCurriculum(current_pool).get_full_order()
    dispersed = DispersedCurriculum(current_pool).get_full_order()

    planned_adj = count_adjacent_same_domain(planned, current_pool)
    dispersed_adj = count_adjacent_same_domain(dispersed, current_pool)

    # In current 19-fact pool, planned has 8 adjacent pairs, dispersed has only 1!
    assert dispersed_adj < planned_adj
    assert planned_adj == 8
    assert dispersed_adj == 1
    assert set(dispersed) == set(planned)
    assert len(dispersed) == len(current_pool)


def test_random_curriculum_seed_reproducibility(current_pool):
    """RandomCurriculum is reproducible given a seed, and different across seeds."""
    c1 = RandomCurriculum(current_pool, seed=42).get_full_order()
    c2 = RandomCurriculum(current_pool, seed=42).get_full_order()
    c3 = RandomCurriculum(current_pool, seed=123).get_full_order()

    assert c1 == c2
    assert c1 != c3
    assert set(c1) == {f["id"] for f in current_pool}
    assert len(c1) == len(current_pool)


def test_order_stats_formula(current_pool):
    """
    Acceptance: order_stats matches formula:
      random_expectation = Σ_d n_d(n_d - 1) / N
    For current pool (N=19, sports=8, science=6, politics=3, economics=1, technology=1):
      (8*7 + 6*5 + 3*2 + 0 + 0) / 19 = 92 / 19 ≈ 4.8421
    """
    dom_map = {f["id"]: f.get("domain", "unknown") for f in current_pool}
    counts = Counter(dom_map.values())
    N = len(current_pool)
    expected_val = sum(cnt * (cnt - 1) for cnt in counts.values()) / N

    assert round(expected_val, 2) == 4.84


def test_order_stats_logged_at_finalize(current_pool, tmp_path):
    """RunLogger.finalize writes order_stats to run_log.json."""
    logger = RunLogger("test_order_stats", str(tmp_path))
    ordered = DispersedCurriculum(current_pool).get_full_order()
    logger.finalize(ordered, fact_pool=current_pool)

    log_file = tmp_path / "test_order_stats" / "run_log.json"
    assert log_file.exists()
    with open(log_file) as f:
        log_data = json.load(f)

    assert "order_stats" in log_data
    assert log_data["order_stats"]["adjacent_same_domain"] == 1
    assert log_data["order_stats"]["random_expectation"] == 4.8421


def test_v2_configs_registered_and_valid():
    """All v2 configs (C2-C10) exist on disk, are valid JSON, and have protocol_version == 2."""
    configs_v2 = all_configs(protocol=2)
    assert len(configs_v2) == 9  # C2 through C10
    expected_keys = {"C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"}
    assert set(configs_v2.keys()) == expected_keys

    for cfg_id, meta in configs_v2.items():
        cfg_file = meta["config_file"]
        assert cfg_file.exists(), f"{cfg_id} file {cfg_file} missing"
        with open(cfg_file) as f:
            data = json.load(f)
        assert data.get("protocol_version") == 2
        assert data.get("curriculum") in {"planned", "self_generated", "random", "dispersed"}
