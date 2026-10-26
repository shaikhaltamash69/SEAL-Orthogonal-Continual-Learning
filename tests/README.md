# SEAL Unit Test Suite

This directory contains fast, CPU-only unit tests covering all Phase 1 work packages (WP0 through WP5).
All model interactions are mocked, allowing the complete suite to execute in under 15 seconds without a GPU.

## Running Tests

Run all unit tests:

```bash
pytest -v
```

Run tests for a specific work package:

```bash
# WP0: Learning gate, plasticity probe, leakage protection
pytest tests/test_wp0_learning_gate.py -v

# WP1: Run archival, aggregation validation, config registry
pytest tests/test_wp1_archive_and_compile.py -v

# WP2: Selection candidate capping, shuffling, RNG isolation, online mode
pytest tests/test_wp2_selection.py -v

# WP3: Control curricula (random, dispersed), order stats, v2 configs
pytest tests/test_wp3_control_curricula.py -v

# WP4: Learned vs final accuracy, forgetting metrics, bootstrap CI
pytest tests/test_wp4_metrics.py -v

# WP5: Strict matching, novelty gate, fact pool validator
pytest tests/test_wp5_eval_and_data_quality.py -v
```

## Test Files

- `test_wp0_learning_gate.py`: Verifies zero leakage from evaluation QA into training prompts, `synthetic_mode` options, and probe structure.
- `test_wp1_archive_and_compile.py`: Validates run classification, non-zero exit on corrupted runs or high fallback rates, and config registry lookups.
- `test_wp2_selection.py`: Tests candidate sampling, reply parsing strategies (earliest ID occurrence, digits, fallback), RNG stream isolation, and mock online selection.
- `test_wp3_control_curricula.py`: Tests `DispersedCurriculum` domain interleaving, `RandomCurriculum` reproducibility, and order statistics.
- `test_wp4_metrics.py`: Tests learned vs final accuracy derivation, new vs pre-known fact splits, question-level correctness logging, and bootstrap CI resampling.
- `test_wp5_eval_and_data_quality.py`: Tests normalized regex strict answer matching (`Paris` vs `Parisian`), numeric normalization, answer aliases, pool validator, and novelty gating.
