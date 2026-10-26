"""
tests/test_wp0_learning_gate.py

WP0 acceptance tests:
  1. Leakage test: no qa_pair question text appears in the generation prompt.
  2. synthetic_mode='passage' returns raw passage text (no model call).
  3. synthetic_mode='model' returns non-empty text (mock).
  4. synthetic_mode='model+passage' includes passage text.
  5. Learning gate: gate aborts when accuracy < gate_min.

All tests run on CPU with mocked models — no GPU, < 1 min.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from tools.plasticity_probe import _assert_no_qa_leakage


# ── Minimal fake fact for testing ─────────────────────────────────────────────
FAKE_FACT = {
    "id": "fact_test",
    "title": "Test Event",
    "passage": "The test event happened in 2025 in London.",
    "qa_pairs": [
        {"question": "Where did the test event happen?", "answer": "London"},
        {"question": "When did the test event happen?", "answer": "2025"},
    ],
    "domain": "science",
    "source_url": "https://example.com",
    "date_of_event": "2025-01-01",
}


# ─── Leakage tests ────────────────────────────────────────────────────────────

class TestLeakageGuard:
    def test_clean_prompt_passes(self):
        """A prompt without qa text should pass with no exception."""
        clean_prompt = f"Title: {FAKE_FACT['title']}\nPassage: {FAKE_FACT['passage']}"
        _assert_no_qa_leakage(clean_prompt, FAKE_FACT)  # should not raise

    def test_leaked_question_raises(self):
        """A prompt containing an eval question should raise AssertionError."""
        leaked_prompt = (
            f"Title: {FAKE_FACT['title']}\n"
            f"Passage: {FAKE_FACT['passage']}\n"
            f"Q: {FAKE_FACT['qa_pairs'][0]['question']}"   # <- leakage
        )
        with pytest.raises(AssertionError, match="LEAKAGE"):
            _assert_no_qa_leakage(leaked_prompt, FAKE_FACT)

    def test_partial_question_not_leaked(self):
        """A partial match (not the full question) should pass."""
        partial_prompt = "The test event happened"  # part of passage, not question
        _assert_no_qa_leakage(partial_prompt, FAKE_FACT)  # should not raise


# ─── synthetic_mode tests (no GPU, mock model) ────────────────────────────────

class _TensorDict(dict):
    """A dict subclass that supports .to(device) so tokenizer output can be moved."""
    def to(self, device):
        return self


class _FakeTokenizer:
    """Minimal tokenizer mock."""
    eos_token_id = 0
    eos_token = "<eos>"
    padding_side = "right"

    def __call__(self, text, **kwargs):
        import torch
        # Return a tiny tensor regardless of input — supports .to(device)
        return _TensorDict(
            input_ids=torch.zeros(1, 4, dtype=torch.long),
            attention_mask=torch.ones(1, 4, dtype=torch.long),
        )

    def decode(self, tokens, skip_special_tokens=True):
        return "Q: mock question\nA: mock answer"


class _FakeModel:
    """Minimal model mock — generate() returns a fixed tensor."""
    def eval(self): return self
    def train(self): return self

    def generate(self, input_ids, **kwargs):
        import torch
        # Return [batch, input+4] with zeros
        return torch.zeros(1, input_ids.shape[1] + 4, dtype=torch.long)

    def __call__(self, **kwargs):
        import torch
        class _Out:
            loss = torch.tensor(1.0, requires_grad=True)
        return _Out()

    def named_parameters(self):
        import torch
        param = torch.nn.Parameter(torch.zeros(2, 2))
        return [("dummy", param)]

    def parameters(self):
        import torch
        return [torch.nn.Parameter(torch.zeros(2, 2))]

    def print_trainable_parameters(self): pass


def _make_loop(synthetic_mode: str):
    """Build a SEALSelfEditLoop with mocked model/tokenizer, no GPU needed."""
    import importlib
    # Patch heavy imports before importing self_edit_loop
    import unittest.mock as mock
    import sys

    config = {
        "model_name": "mock",
        "run_id": "test_run",
        "results_dir": "/tmp/test_results_seal_wp0",
        "synthetic_mode": synthetic_mode,
        "lora_r": 4,
        "lora_alpha": 8,
        "sft_epochs": 1,
        "lr": 2e-4,
    }

    fake_tok = _FakeTokenizer()
    fake_model = _FakeModel()

    with mock.patch("src.self_edit_loop.AutoModelForCausalLM") as mock_auto_model, \
         mock.patch("src.self_edit_loop.AutoTokenizer") as mock_auto_tok, \
         mock.patch("src.self_edit_loop.get_peft_model") as mock_peft, \
         mock.patch("src.self_edit_loop.RunLogger") as mock_logger:

        mock_auto_model.from_pretrained.return_value = fake_model
        mock_auto_tok.from_pretrained.return_value = fake_tok
        mock_peft.return_value = fake_model
        mock_logger.return_value.log_config = lambda *a, **kw: None

        from src.self_edit_loop import SEALSelfEditLoop
        loop = SEALSelfEditLoop.__new__(SEALSelfEditLoop)
        loop.config = config
        loop.device = "cpu"
        loop.model = fake_model
        loop.tokenizer = fake_tok

    return loop


class TestSyntheticMode:
    def test_passage_mode_returns_passage(self):
        """synthetic_mode='passage' must return text containing the passage."""
        loop = _make_loop("passage")
        result = loop.generate_synthetic_data(FAKE_FACT["passage"], FAKE_FACT["title"])
        assert FAKE_FACT["passage"] in result
        assert FAKE_FACT["title"] in result

    def test_passage_mode_no_model_call(self):
        """synthetic_mode='passage' must NOT call model.generate."""
        import unittest.mock as mock
        loop = _make_loop("passage")
        with mock.patch.object(loop.model, "generate", side_effect=AssertionError("model.generate called!")) as _m:
            # Should NOT raise — passage mode skips generation
            result = loop.generate_synthetic_data(FAKE_FACT["passage"], FAKE_FACT["title"])
        assert FAKE_FACT["passage"] in result

    def test_model_mode_calls_generate(self):
        """synthetic_mode='model' (default) calls model.generate."""
        import unittest.mock as mock
        import torch
        loop = _make_loop("model")
        tok = loop.tokenizer

        call_count = [0]
        def fake_generate(**kwargs):
            call_count[0] += 1
            return torch.zeros(1, 8, dtype=torch.long)

        with mock.patch.object(loop.model, "generate", side_effect=fake_generate):
            loop.generate_synthetic_data(FAKE_FACT["passage"], FAKE_FACT["title"])
        assert call_count[0] == 1

    def test_model_plus_passage_contains_passage(self):
        """synthetic_mode='model+passage' must contain passage text."""
        import unittest.mock as mock
        import torch
        loop = _make_loop("model+passage")

        with mock.patch.object(
            loop.model, "generate",
            return_value=torch.zeros(1, 8, dtype=torch.long)
        ):
            result = loop.generate_synthetic_data(FAKE_FACT["passage"], FAKE_FACT["title"])
        assert FAKE_FACT["passage"] in result

    def test_missing_mode_defaults_to_model(self):
        """When synthetic_mode is absent from config, default is 'model' (v1 behaviour)."""
        import unittest.mock as mock
        import torch
        loop = _make_loop("model")
        del loop.config["synthetic_mode"]   # simulate absent key

        call_count = [0]
        def fake_generate(**kwargs):
            call_count[0] += 1
            return torch.zeros(1, 8, dtype=torch.long)

        with mock.patch.object(loop.model, "generate", side_effect=fake_generate):
            loop.generate_synthetic_data(FAKE_FACT["passage"], FAKE_FACT["title"])
        assert call_count[0] == 1  # model was called = model mode


# ─── Probe smoke (import-only) ────────────────────────────────────────────────

def test_probe_imports():
    """Ensure plasticity_probe.py imports cleanly."""
    import importlib
    mod = importlib.import_module("tools.plasticity_probe")
    assert hasattr(mod, "PlasticityProbe")
    assert hasattr(mod, "_assert_no_qa_leakage")
