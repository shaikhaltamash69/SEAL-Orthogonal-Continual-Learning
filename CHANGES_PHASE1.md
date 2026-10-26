# Phase 1 Implementation Summary: SEAL Repository

This document summarizes all architectural changes, tools, validation gates, control curricula, metric pipelines, and test suites implemented under the **Phase 1 Change Request** for repository `SEAL-Orthogonal-Continual-Learning` (originating from commit `77b4320`).

---

## 1. Overview of Commits

All changes were implemented incrementally with dedicated unit tests passing prior to each commit:

| Work Package | Commit Hash | Commit Message |
|---|---|---|
| **WP0** | `14e98ec` | `WP0: add plasticity probe, synthetic modes, leakage assertion, and optional learning gate` |
| **WP1** | `9a4aede` | `WP1: archive invalid runs, strict compile results aggregation, and centralized config registry` |
| **WP2** | `c655370` | `WP2: fix selection step with candidate cap, shuffling, own RNG isolation, enhanced parsing priorities, fallback policies, logprob scoring, and online selection mode` |
| **WP3** | `65f23d0` | `WP3: add random and dispersed control curricula, order statistics logging, and Protocol 2 configs` |
| **WP4** | `bbdbc76` | `WP4: post-processing learned/final/forgetting metrics, untrained baseline row, qa_correct logging, and bootstrap CI tool` |
| **WP5** | `c5230d3` | `WP5: strict evaluation matching, novelty filter with max-prior-accuracy, pool validator, and authoring guidelines` |
| **WP6** | `a815a89` | `WP6: environment inspection tool, clean requirements, script organization, and README documentation` |

---

## 2. Changes by Work Package

### Work Package 0: Learning-Efficacy Gate & Plasticity Probe

- **What Changed**:
  - Added standalone hyperparameter sweep tool `tools/plasticity_probe.py` to evaluate immediate learning on isolated facts before continual learning. Sweeps `sft_epochs` ∈ {3, 5, 10}, `lr` ∈ {2e-4, 5e-4, 1e-3}, and `synthetic_mode` ∈ {`model`, `passage`, `model+passage`}.
  - Added `synthetic_mode` configuration key to `SEALSelfEditLoop.generate_synthetic_data` (`src/self_edit_loop.py`):
    - `model` (default, backward-compatible v1 behavior),
    - `passage` (trains directly on raw reference passage text),
    - `model+passage` (concatenates model synthetic generation with reference passage).
  - Implemented strict data leakage prevention asserting that no text from held-out evaluation `qa_pairs` is ever exposed in generation prompts.
  - Implemented pre-run plasticity gate in `scripts/run_config.py` via `learning_gate: true` and `learning_gate_min` (default: 0.5; default off).
- **Files Modified / Created**:
  - Created: `tools/plasticity_probe.py`
  - Modified: `src/self_edit_loop.py`, `scripts/run_config.py`
  - Created: `tests/test_wp0_learning_gate.py`
- **Tests Added**:
  - `TestLeakageGuard`: clean prompt passes, leaked question text raises `AssertionError`, partial non-leakage passes.
  - `TestSyntheticMode`: verifies `passage`, `model`, `model+passage`, and default fallback behavior.
- **Skipped / Note**: Winning hyperparameter setting is not chosen autonomously; probe results table is provided for the user to select.

---

### Work Package 1: Results Archival, Strict Aggregation & Config Registry

- **What Changed**:
  - Created run classification and archival tool `tools/archive_runs.py`. Archived 16 broken, aborted, or smoke test runs into `results/archive/` while retaining valid October 2026 runs in `results/`. Documented all moved runs in `results/archive/MANIFEST.md`.
  - Rewrote `compile_results.py`:
    - Filters only complete runs containing `end_time` and full step counts; takes newest run if duplicates exist.
    - Exits non-zero if fallback rate > 0.20 for self-generated configs (overrideable via `--allow-high-fallback`).
    - Exits non-zero if any expected (config, seed) run is missing.
    - Displays score counts as questions (`N/38`) and reports individual seed values alongside min–max ranges.
    - Always prints an untrained base model row from the baseline accuracy file.
    - Computes CTG in questions and percentage points, printing `WITHIN SEED RANGE` when the difference does not exceed the larger arm's seed spread.
    - Added `--protocol {1, 2}` filtering based on `protocol_version`.
  - Created `scripts/config_registry.py` as the single source of truth for config IDs, files, labels, and protocols.
- **Files Modified / Created**:
  - Created: `tools/archive_runs.py`, `results/archive/MANIFEST.md`, `scripts/config_registry.py`
  - Modified: `compile_results.py`
  - Created: `tests/test_wp1_archive_and_compile.py`
- **Tests Added**:
  - Run classifier unit tests: valid runs, missing `end_time`, incomplete steps, September 2026 broken runs, smoke tests.
  - Registry lookup tests.
  - Non-zero exit code validation on corrupted logs and `--allow-high-fallback` override.

---

### Work Package 2: Selection Step Fixes & Online Mode

- **What Changed**:
  - Rewrote `SelfGeneratedCurriculum` in `src/curriculum.py`:
    - `max_candidates`: defaults to 10 (v1 compatibility); `null` exposes all remaining candidates. When capped, candidates are sampled randomly using the selection RNG.
    - `shuffle_candidates`: defaults to `false`. When `true`, shuffles candidate display order every step while maintaining a bidirectional ID mapping.
    - RNG stream isolation: selection uses dedicated `random.Random(seed)` / `torch.Generator`, preventing curriculum decisions from perturbing training RNG.
    - Parsing priority: (1) fact ID / number at start of reply; (2) earliest occurring fact ID in the reply text; (3) standalone digit anywhere; (4) fallback. Logs `parse_strategy`.
    - `fallback_policy`: `first` (v1 default) or `random` (seeded random candidate). Always logs `fallback_used: true`.
    - `selection_method`: `generate` (free text) or `logprob` (evaluates mean token log-likelihood per candidate with softmax temperature sampling).
    - `selection_mode`: `upfront` (v1 default) or `online` (picks next fact inside the training loop after evaluating adapter states).
  - Extended `RunLogger.log_curriculum_choice` (`src/logger.py`) with metadata (`displayed_order`, `displayed_position`, `parse_strategy`, `selection_method`, `scores`).
  - Added support in `SEALSelfEditLoop.run` (`src/self_edit_loop.py`) for callable `next_fact_fn(step_num, studied_ids) -> fact_id`. Documented adapter interaction during online selection.
- **Files Modified / Created**:
  - Modified: `src/curriculum.py`, `src/logger.py`, `src/self_edit_loop.py`, `scripts/run_config.py`
  - Created: `tests/test_wp2_selection.py`
- **Tests Added**:
  - Parsing strategies: leading ID, leading digit, multi-ID earliest occurrence, anywhere digit, fallback policies.
  - Bidirectional shuffle map correctness.
  - RNG isolation between training and selection.
  - Mock online mode execution.
  - Backward compatibility verifying identical behavior when new keys are absent.

---

### Work Package 3: Control Curricula & Protocol 2 Configs

- **What Changed**:
  - Added `RandomCurriculum(fact_pool, seed)` to `src/curriculum.py` for reproducible seeded random permutations.
  - Added `DispersedCurriculum(fact_pool)` to `src/curriculum.py` for deterministic topic-interleaved ordering (round-robin across domains, largest domain first, ties broken by ID). Reduces adjacent same-domain pairs from 8 (human-planned) down to 1.
  - Extended `scripts/run_config.py` to accept `"curriculum": "random" | "dispersed"`.
  - Added `order_stats` to run logs at `finalize`: computes `adjacent_same_domain` count and theoretical random expectation $\sum_d n_d(n_d - 1) / N \approx 4.8421$.
  - Created 9 Protocol 2 configurations in `configs/v2/`:
    - `C2`: Vanilla LoRA (Protocol 2)
    - `C3`: O-LoRA Planned
    - `C4`: O-LoRA Self-Generated (logprob scoring, candidate shuffling, all candidates)
    - `C5`: LoRI Self-Generated (logprob scoring, candidate shuffling, all candidates)
    - `C6`: LoRI Planned
    - `C7`: O-LoRA Seeded Random Control
    - `C8`: O-LoRA Topic-Dispersed Control
    - `C9`: LoRI Seeded Random Control
    - `C10`: LoRI Topic-Dispersed Control
  - Registered all v2 configurations in `scripts/config_registry.py`.
  - Updated `scripts/run_multi_seeds.py` with `--protocol 2` and `--configs` flags.
- **Files Modified / Created**:
  - Modified: `src/curriculum.py`, `src/logger.py`, `scripts/run_config.py`, `scripts/config_registry.py`, `scripts/run_multi_seeds.py`
  - Created: `configs/v2/config_c*.json` (9 files)
  - Created: `tests/test_wp3_control_curricula.py`
- **Tests Added**:
  - Dispersed curriculum adjacent pair minimization.
  - Random curriculum seed reproducibility.
  - Order statistics calculation against theoretical expectation.
  - Verification of all registered v2 config files.

---

### Work Package 4: Learned/Final/Forgetting Metrics & Clustered Bootstrap CI

- **What Changed**:
  - Updated `src/evaluator.py` (`evaluate_all_facts`): supports `return_details=True` returning per-question correctness arrays (`qa_correct: [0/1, ...]`).
  - Updated `RunLogger.log_accuracy_matrix` to record question-level details.
  - Updated `compile_results.py` to compute and display:
    - **learned accuracy**: accuracy on fact $t$ at the exact step it was trained,
    - **final accuracy**: accuracy on fact $t$ at the final sequence step,
    - **forgetting**: learned − final accuracy,
    - each split across **all facts**, **genuinely new facts** (prior accuracy 0), and **pre-known facts** (prior accuracy > 0).
    - Untrained model row displayed in every table from baseline accuracy data.
  - Created `tools/bootstrap_ci.py`: clustered two-stage bootstrap confidence interval estimator (resampling facts with replacement, then questions within facts) computing 95% intervals for config accuracy and CTG differences.
- **Files Modified / Created**:
  - Modified: `src/evaluator.py`, `src/logger.py`, `src/self_edit_loop.py`, `compile_results.py`
  - Created: `tools/bootstrap_ci.py`, `tests/test_wp4_metrics.py`
- **Tests Added**:
  - Metric derivation on legacy v1 run logs (verified learned accuracy on new facts is ~4–10%).
  - Logging of `qa_correct` matrices.
  - Bootstrap resampling CI computation and CTG interval estimation.

---

### Work Package 5: Strict Evaluation Matching, Novelty Gating & Pool Validator

- **What Changed**:
  - Implemented normalized strict answer matching in `src/evaluator.py`:
    - `normalize_answer`: NFKD Unicode normalization, lowercasing, punctuation stripping, whitespace collapsing, and comma removal in numbers (`1,000` ≡ `1000`).
    - `match_answer`: supports `eval_match: "contains"` (v1 default) and `"strict"` (whole-word/phrase regex boundary check). Supports optional per-question `answer_aliases`.
  - Updated `verify_novelty.py` with `--max-prior-accuracy` (default: `0.0` strict instead of 0.5), `--model` selection, and per-model output writing (`data/baseline_accuracy_<model-slug>.json`).
  - Added `baseline_accuracy_path` config key (default: `data/baseline_accuracy.json`) to `scripts/run_config.py`. Enforces novelty gate: aborts run if any fact in the pool is flagged `keep_in_pool: false`.
  - Created fact pool validator `tools/validate_fact_pool.py`:
    - Verifies unique IDs and required fields (`id`, `title`, `passage`, `qa_pairs`, `domain`, `source_url`, `date_of_event`).
    - Checks minimum QA pairs (`--min-questions`, default: 4) and event dates (`--min-date`, default: 2025-01-01).
    - Computes domain counts, random expectation, and warns for domains with < 5 facts or pools with < 60 facts.
    - Exits non-zero on validation errors without crashing.
  - Added `data/README.md` with schema specifications and authoring tasks.
- **Files Modified / Created**:
  - Modified: `src/evaluator.py`, `src/self_edit_loop.py`, `verify_novelty.py`, `scripts/run_config.py`
  - Created: `tools/validate_fact_pool.py`, `data/README.md`, `tests/test_wp5_eval_and_data_quality.py`
- **Tests Added**:
  - Strict matching: `Paris` rejected for `Parisian`, accepted for `Paris, France`.
  - Numeric normalization: `1,000` matches `1000`.
  - Answer alias matching.
  - Validator execution on active pool (reports shortfalls cleanly) and synthetic pools.
  - Novelty gate refusal in `scripts/run_config.py`.
- **Skipped / Note**: Facts and questions were not authored autonomously per ground rules.

---

### Work Package 6: Repo Tidy & Reproducibility

- **What Changed**:
  - Created `tools/print_env.py` to inspect runtime dependencies, CUDA, and hardware, writing `results/ENVIRONMENT.md`.
  - Integrated `get_environment_info()` into `scripts/run_config.py` to record active environment in run log configs.
  - Cleaned `requirements.txt` to contain direct project dependencies only. Removed unimported packages: `wandb`, `lm-eval`, `jupyter`, `notebook`, `ipykernel`, `evaluate`, `datasets`.
  - Generated `requirements.lock.txt` using `pip freeze` from the live environment.
  - Replaced hardcoded PyTorch badge with `<!-- TODO: set from results/ENVIRONMENT.md -->` in `README.md`.
  - Moved scratch and report scripts to `tools/` via `git mv`: `check_status.py`, `inspect_live.py`, `inspect_logs.py`, `verify_issues.py`, `generate_doc.py`, `generate_synopsis_report.py`.
  - Kept `verify_novelty.py` and `verify_gpu.py` at root as referenced in papers and documentation. Documented upstream `SEAL/` directory.
  - Moved root-level `.docx` files to `docs/` via `git mv`.
  - Investigated git history for missing `fact_004` (`git log -S fact_004 --all`). Documented in `data/README.md: fact_004 absent; reason unknown; author to fill in`.
  - Created `tests/README.md` documenting test invocation commands.
  - Updated root `README.md` with comprehensive documentation and directory layout.
- **Files Modified / Created / Moved**:
  - Created: `tools/print_env.py`, `results/ENVIRONMENT.md`, `requirements.lock.txt`, `tests/README.md`
  - Modified: `requirements.txt`, `README.md`, `data/README.md`, `scripts/run_config.py`
  - Moved: 6 scratch scripts to `tools/`, 4 `.docx` files to `docs/`

---

## 3. Test Suite Verification

The complete test suite runs on CPU in under 15 seconds:

```bash
pytest -v
```

**Results:**
- `49 passed, 1 warning in 12.46s`
- 0 failures, 0 regressions.

---

## 4. User Action Items & TODOs

Per ground rules, the following tasks require authoring, hyperparameter selection, or GPU execution by the user:

1. **Author Additional Facts & QA Pairs**:
   - The current pool contains 19 facts with 2 QA pairs each.
   - Author an additional 2–3 questions per fact (target: 4–5 questions each) with canonical answers and `answer_aliases`.
   - Expand total pool to $\ge 60$ facts with post-2025 event dates (`date_of_event >= 2025-01-01`).
   - Balance domain representation ($\ge 10$ facts per domain across sports, science, politics, economics, technology).
   - Verify pool with `python tools/validate_fact_pool.py`.

2. **Select Plasticity Hyperparameters (WP0)**:
   - Run the plasticity sweep on GPU:
     ```bash
     python tools/plasticity_probe.py --fact-id fact_001
     ```
   - Review immediate accuracy table across epochs (3, 5, 10), learning rates (2e-4, 5e-4, 1e-3), and synthetic modes (`model`, `passage`, `model+passage`).
   - Update `sft_epochs`, `lr`, and `synthetic_mode` in `configs/v2/` configs accordingly.

3. **Verify Novelty on Expanded Pool (WP5)**:
   - After authoring new facts, run:
     ```bash
     python verify_novelty.py --model Qwen/Qwen2.5-1.5B-Instruct --max-prior-accuracy 0.0
     ```
   - Ensure pre-known facts are filtered or refined.

4. **Verify PyTorch Version Badge (WP6)**:
   - Check `results/ENVIRONMENT.md` (active environment: PyTorch `2.6.0+cu124`).
   - Update `<!-- TODO: set from results/ENVIRONMENT.md -->` in `README.md` with desired badge label.

5. **Decision on `.gitignore` for Archived Logs**:
   - `results/archive/` contains raw logs from superseded and aborted runs.
   - Let Antigravity know if you wish to add `results/archive/**/*.json` to `.gitignore` to prevent repository bloat, or keep them tracked for historical auditability.
