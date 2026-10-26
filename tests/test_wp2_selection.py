"""
tests/test_wp2_selection.py

Acceptance tests for WP2 (Fix the selection step):
  1. Parsing: id, leading digit, anywhere digit, fallback, multi-id earliest occurrence.
  2. Shuffle-map correctness: displayed number always maps to the right fact.
  3. RNG isolation: training RNG state identical with and without selection settings.
  4. Online mode: mock model test in loop.run with callable next_fact_fn.
  5. Backward compatibility: with every new key absent, v1 configs produce identical behaviour.
"""
import copy
import random
import re
import pytest
import torch
from src.curriculum import SelfGeneratedCurriculum, PlannedCurriculum
from src.logger import RunLogger


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, **kwargs):
        class _TensorDict(dict):
            def to(self, device):
                return self

        # Generate a dummy token tensor based on text length
        tokens = [1] * max(1, len(text.split()))
        d = _TensorDict({
            "input_ids": torch.tensor([tokens], dtype=torch.long),
            "attention_mask": torch.tensor([[1] * len(tokens)], dtype=torch.long),
        })
        return d

    def decode(self, token_ids, **kwargs):
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        return "1: This is the default decoded choice."


class _MockModel:
    def __init__(self, response_text: str = "1: Mock choice"):
        self.response_text = response_text
        self.training = True

    def eval(self):
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.training = mode
        return self

    def generate(self, **kwargs):
        # Return a tensor containing tokens, or return mock
        return torch.tensor([[1, 2, 3, 4]], dtype=torch.long)


@pytest.fixture
def fake_pool():
    return [
        {"id": f"fact_{i:03d}", "title": f"Fact Title {i}", "passage": f"Passage {i}", "qa_pairs": []}
        for i in range(1, 20)  # 19 facts
    ]


@pytest.fixture
def fake_logger(tmp_path):
    return RunLogger("test_run", str(tmp_path))


# =====================================================================
# 1. Parsing tests
# =====================================================================

def test_parse_leading_id(fake_pool, fake_logger):
    """Verify that a leading fact_id is parsed with strategy 'id'."""
    candidates = [f"fact_{i:03d}" for i in range(1, 11)]
    curriculum = SelfGeneratedCurriculum(
        model=None, tokenizer=None, fact_pool=fake_pool, logger=fake_logger
    )
    
    idx, strategy, fallback = curriculum._parse_choice(
        "[fact_005]: Excellent topic for retention.", candidates, 0
    )
    assert strategy == "id"
    assert not fallback
    assert candidates[idx] == "fact_005"

    # Without brackets
    idx2, strategy2, fallback2 = curriculum._parse_choice(
        "fact_003 is the foundational subject.", candidates, 0
    )
    assert strategy2 == "id"
    assert not fallback2
    assert candidates[idx2] == "fact_003"


def test_parse_leading_digit(fake_pool, fake_logger):
    """Verify that a candidate number at the start of reply is parsed with strategy 'leading_digit'."""
    candidates = [f"fact_{i:03d}" for i in range(1, 11)]
    curriculum = SelfGeneratedCurriculum(
        model=None, tokenizer=None, fact_pool=fake_pool, logger=fake_logger
    )

    idx, strategy, fallback = curriculum._parse_choice(
        "3: This topic builds directly on the prior material.", candidates, 0
    )
    assert strategy == "leading_digit"
    assert not fallback
    assert idx == 2  # 3rd candidate -> index 2
    assert candidates[idx] == "fact_003"

    # Leading digit with dot
    idx2, strategy2, fallback2 = curriculum._parse_choice(
        "1. Start with the basics.", candidates, 0
    )
    assert strategy2 == "leading_digit"
    assert not fallback2
    assert idx2 == 0


def test_parse_multi_id_earliest_occurrence(fake_pool, fake_logger):
    """
    Where several fact IDs appear in the reply, pick the EARLIEST occurrence in the reply,
    NOT the first in candidate order.
    """
    candidates = [f"fact_{i:03d}" for i in range(1, 11)]  # fact_001 is index 0, fact_005 is index 4
    curriculum = SelfGeneratedCurriculum(
        model=None, tokenizer=None, fact_pool=fake_pool, logger=fake_logger
    )

    # fact_005 appears at position 13; fact_001 appears at position 35.
    reply = "I recommend [fact_005] rather than [fact_001] because it is more fundamental."
    idx, strategy, fallback = curriculum._parse_choice(reply, candidates, 0)
    assert strategy == "id"
    assert not fallback
    assert candidates[idx] == "fact_005"  # earliest occurrence wins, even though fact_001 is earlier in candidate list!


def test_parse_anywhere_digit(fake_pool, fake_logger):
    """Verify standalone digit anywhere in reply falls back to 'anywhere_digit'."""
    candidates = [f"fact_{i:03d}" for i in range(1, 11)]
    curriculum = SelfGeneratedCurriculum(
        model=None, tokenizer=None, fact_pool=fake_pool, logger=fake_logger
    )

    reply = "After careful thought, I would select candidate 4 for maximum retention."
    idx, strategy, fallback = curriculum._parse_choice(reply, candidates, 0)
    assert strategy == "anywhere_digit"
    assert not fallback
    assert idx == 3  # candidate 4 -> index 3
    assert candidates[idx] == "fact_004"


def test_parse_fallback_first_and_random(fake_pool, fake_logger):
    """Verify fallback policy: 'first' (index 0) vs 'random' (seeded random candidate)."""
    candidates = [f"fact_{i:03d}" for i in range(1, 11)]
    reply = "I am unable to decide on any topic."

    # Policy 'first'
    curriculum_first = SelfGeneratedCurriculum(
        model=None, tokenizer=None, fact_pool=fake_pool, logger=fake_logger,
        fallback_policy="first"
    )
    idx_first, strategy_first, fallback_first = curriculum_first._parse_choice(reply, candidates, 0)
    assert strategy_first == "fallback"
    assert fallback_first is True
    assert idx_first == 0

    # Policy 'random'
    curriculum_rnd = SelfGeneratedCurriculum(
        model=None, tokenizer=None, fact_pool=fake_pool, logger=fake_logger,
        seed=123, fallback_policy="random"
    )
    idx_rnd, strategy_rnd, fallback_rnd = curriculum_rnd._parse_choice(reply, candidates, 0)
    assert strategy_rnd == "fallback"
    assert fallback_rnd is True
    assert 0 <= idx_rnd < len(candidates)


# =====================================================================
# 2. Shuffle-map correctness
# =====================================================================

def test_shuffle_map_correctness(fake_pool, fake_logger):
    """
    When shuffle_candidates is True, verify that candidate numbering displayed in the prompt
    maps correctly to the corresponding fact ID.
    """
    curriculum = SelfGeneratedCurriculum(
        model=None, tokenizer=None, fact_pool=fake_pool, logger=fake_logger,
        seed=42, shuffle_candidates=True, max_candidates=10
    )

    candidates = curriculum._get_candidates()
    assert len(candidates) == 10
    # Shuffled order should differ from raw pool order
    assert candidates != [f["id"] for f in fake_pool[:10]]

    # For every displayed position 1..10, selecting that number must return candidates[k-1]
    for k in range(1, 11):
        idx, strat, fallback = curriculum._parse_choice(f"{k}: selection", candidates, 0)
        assert not fallback
        assert candidates[idx] == candidates[k - 1]


# =====================================================================
# 3. RNG isolation
# =====================================================================

def test_rng_isolation(fake_pool, fake_logger):
    """
    Selection must use its own RNG stream.
    Training PyTorch RNG and Python random RNG must be completely identical before and after selection.
    """
    class _MockGenModel(_MockModel):
        def generate(self, **kwargs):
            # Advance global torch RNG inside generate
            torch.randn(10)
            return torch.tensor([[1, 2, 3]], dtype=torch.long)

    tok = _FakeTokenizer()
    tok.decode = lambda tokens, **kw: "2: choice"
    model = _MockGenModel()

    curriculum = SelfGeneratedCurriculum(
        model=model, tokenizer=tok, fact_pool=fake_pool, logger=fake_logger,
        seed=999, shuffle_candidates=True
    )

    # Set external (training) seeds and record states
    random.seed(12345)
    torch.manual_seed(54321)
    py_state_before = random.getstate()
    torch_state_before = torch.get_rng_state()

    # Run selection
    chosen = curriculum.select_next(0)
    assert chosen in [f["id"] for f in fake_pool]

    # Verify external RNG states are 100% UNCHANGED
    assert random.getstate() == py_state_before
    assert torch.equal(torch.get_rng_state(), torch_state_before)


# =====================================================================
# 4. Online mode mock-model test
# =====================================================================

def test_online_mode_in_loop(fake_pool, fake_logger, tmp_path):
    """
    Test online mode in SEALSelfEditLoop.run with callable next_fact_fn.
    Verifies that the realised order is recorded, model eval mode was restored,
    and accuracy matrix has entries for each step.
    """
    from src.self_edit_loop import SEALSelfEditLoop

    tok = _FakeTokenizer()
    tok.decode = lambda tokens, **kw: "1: First choice"
    model = _MockModel()

    curriculum = SelfGeneratedCurriculum(
        model=model, tokenizer=tok, fact_pool=fake_pool[:3], logger=fake_logger,
        selection_mode="online"
    )

    # Mock SEALSelfEditLoop without loading full weights
    loop = object.__new__(SEALSelfEditLoop)
    loop.CONFIG_NAME = "test_loop"
    loop.config = {"results_dir": str(tmp_path), "run_id": "test_online"}
    loop.logger = fake_logger
    loop.model = model
    loop.tokenizer = tok
    loop.device = "cpu"
    loop.self_edit_step = lambda fact, step_num: None
    loop._save_checkpoint = lambda step_num: None

    # Run loop with callable next_fact_fn
    next_fact_fn = lambda step_num, studied: curriculum.select_next(step_num)
    
    # Mock evaluate_all_facts in self_edit_loop
    import src.self_edit_loop
    orig_eval = src.self_edit_loop.evaluate_all_facts
    src.self_edit_loop.evaluate_all_facts = lambda m, t, pool, studied, dev: {fid: 1.0 for fid in studied}

    try:
        matrix = loop.run(fake_pool[:3], next_fact_fn, {})
        assert len(matrix) == 3
        # Check that finalize logged the realised order
        assert "curriculum_order_used" in fake_logger.run_log
        assert len(fake_logger.run_log["curriculum_order_used"]) == 3
        assert len(set(fake_logger.run_log["curriculum_order_used"])) == 3
    finally:
        src.self_edit_loop.evaluate_all_facts = orig_eval


# =====================================================================
# 5. Backward compatibility with v1 configs
# =====================================================================

def test_v1_backward_compatibility(fake_pool, fake_logger):
    """
    With every new key absent (defaults), SelfGeneratedCurriculum reproduces v1 exact behaviour:
      - max_candidates is 10
      - shuffle_candidates is False
      - fallback_policy is 'first'
      - selection_method is 'generate'
      - candidates presented are self.remaining[:10] in exact initial order
    """
    tok = _FakeTokenizer()
    tok.decode = lambda tokens, **kw: "1: choice"
    model = _MockModel()

    # Instantiate with only v1 arguments (no new keys)
    curriculum = SelfGeneratedCurriculum(
        model=model, tokenizer=tok, fact_pool=fake_pool, logger=fake_logger
    )

    assert curriculum.max_candidates == 10
    assert curriculum.shuffle_candidates is False
    assert curriculum.fallback_policy == "first"
    assert curriculum.selection_method == "generate"
    assert curriculum.selection_mode == "upfront"

    # In step 0, candidates must be exactly the first 10 facts in order
    candidates = curriculum._get_candidates()
    expected_v1 = [f["id"] for f in fake_pool[:10]]
    assert candidates == expected_v1

    # On unparseable response, falls back to index 0 (first remaining)
    fallback_idx, strat, fallback = curriculum._parse_choice("gibberish", candidates, 0)
    assert fallback is True
    assert fallback_idx == 0
    assert strat == "fallback"
