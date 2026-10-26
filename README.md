# Autonomous LLM Continual Learning: O-LoRA & LoRI under Self-Generated Curricula

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
<!-- TODO: set from results/ENVIRONMENT.md -->
[![PEFT 0.20](https://img.shields.io/badge/PEFT-0.20-green.svg)](https://huggingface.co/docs/peft/index)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An empirical research framework evaluating **Parameter-Efficient Continual Learning (O-LoRA & LoRI)** inside **Autonomous Self-Editing Language Models (SEAL)** across verified post-2024 factual updates.

---

## 📌 Abstract & Research Focus

While techniques like Subspace-Orthogonal LoRA (O-LoRA) and Reduced Interference LoRA (LoRI) reduce catastrophic forgetting in LLMs, existing literature evaluates them strictly on static, human-planned task sequences. In autonomous agent architectures (e.g. SEAL), models generate their own training data and self-select their edit sequence.

This repository systematically quantifies the **Curriculum Transfer Gap (CTG)**: how memory protection behaves when moving from human-planned to model-directed self-generated curricula.

> [!NOTE]
> The upstream reference repository from the original SEAL paper is included as [`SEAL/`](SEAL/) for comparative baseline reference and historical inspection; it is kept unchanged.

---

## 📁 Repository Structure

```
├── configs/               # Experimental configurations
│   ├── config_c2_vanilla.json          # Protocol 1: C2 Vanilla LoRA (Self-Generated)
│   ├── config_c3_olora_planned.json    # Protocol 1: C3 O-LoRA (Planned)
│   ├── config_c4_olora_selfgen.json    # Protocol 1: C4 O-LoRA (Self-Generated)
│   ├── config_c5_lori_selfgen.json     # Protocol 1: C5 LoRI (Self-Generated)
│   ├── config_c6_lori_planned.json     # Protocol 1: C6 LoRI (Planned)
│   └── v2/                             # Protocol 2 configurations (C2–C10)
│       ├── config_c2_vanilla.json      # Protocol 2: C2 with strict matching & candidate shuffle
│       ├── config_c3_olora_planned.json# Protocol 2: C3 O-LoRA planned
│       ├── config_c4_olora_selfgen.json# Protocol 2: C4 O-LoRA logprob self-generated
│       ├── config_c5_lori_selfgen.json # Protocol 2: C5 LoRI logprob self-generated
│       ├── config_c6_lori_planned.json # Protocol 2: C6 LoRI planned
│       ├── config_c7_olora_random.json # Protocol 2: C7 O-LoRA seeded random control
│       ├── config_c8_olora_dispersed.json# Protocol 2: C8 O-LoRA topic-dispersed control
│       ├── config_c9_lori_random.json  # Protocol 2: C9 LoRI seeded random control
│       └── config_c10_lori_dispersed.json# Protocol 2: C10 LoRI topic-dispersed control
├── data/                  # Continual learning data & baselines
│   ├── fact_pool.json     # Active 19-fact pool
│   ├── fact_pool_template.json # Template for authoring new facts
│   ├── baseline_accuracy.json  # Base model accuracy across pool
│   └── README.md          # Authoring guidelines, schema & shortfalls
├── docs/                  # Project reports, manuscripts, and synopsis documents
├── figures/               # PDF & PNG paper figures (300 DPI)
├── results/               # Active experimental run outputs
│   ├── ENVIRONMENT.md     # Auto-generated runtime environment specifications
│   └── archive/           # Archived runs (aborted, legacy, smoke tests) with MANIFEST.md
├── scripts/               # Core execution and orchestration scripts
│   ├── config_registry.py # Central registry mapping config IDs to paths and labels
│   ├── run_config.py      # Main CLI runner for individual experimental configs
│   ├── run_multi_seeds.py # Multi-seed orchestration runner (Protocol 1 and 2)
│   ├── generate_figures.py# Paper figures generator
│   ├── test_orthogonality.py# O-LoRA orthogonality penalty test
│   └── smoke_test.py      # End-to-end integration test
├── src/                   # Core library components
│   ├── curriculum.py      # Planned, Random, Dispersed & Self-Generated curriculum engines
│   ├── evaluator.py       # SQuAD-style QA evaluator (contains & normalized strict regex)
│   ├── logger.py          # Append-safe structured JSON run logger
│   ├── orthogonality.py   # O-LoRA projection and penalty computation
│   └── self_edit_loop.py  # Continual learning loops (Vanilla, O-LoRA, LoRI)
├── tests/                 # Fast CPU unit test suite (pytest)
│   └── README.md          # Test documentation and run instructions
├── tools/                 # Diagnostics, validation, and analysis utilities
│   ├── archive_runs.py    # Classifies and archives incomplete/broken runs
│   ├── bootstrap_ci.py    # Clustered bootstrap confidence intervals
│   ├── check_status.py    # Live training process status checker
│   ├── inspect_live.py    # Live run progress inspector
│   ├── inspect_logs.py    # Run log JSON inspector
│   ├── plasticity_probe.py# Sweeps learning hyperparameters on isolated facts
│   ├── print_env.py       # Inspects environment and generates results/ENVIRONMENT.md
│   ├── validate_fact_pool.py # Validates fact pool schema, questions, and dates
│   └── verify_issues.py   # Historical log validator
├── verify_gpu.py          # GPU & CUDA capability verifier
├── verify_novelty.py      # Base model zero-shot accuracy verifier
├── compile_results.py     # Results compiler and validation aggregator
├── requirements.txt       # Direct production dependencies
└── requirements.lock.txt  # Fully pinned dependencies from active environment
```

---

## ⚡ Quickstart

### 1. Environment Setup

```bash
git clone https://github.com/shaikhaltamash69/SEAL-Orthogonal-Continual-Learning.git
cd SEAL-Orthogonal-Continual-Learning

# Inspect environment and generate results/ENVIRONMENT.md
python tools/print_env.py

# Install dependencies
pip install -r requirements.txt

# Verify CUDA and GPU capability
python verify_gpu.py
```

### 2. Fact Pool Quality & Plasticity Diagnostics

Before running full multi-seed continual learning sequences:

```bash
# 1. Validate fact pool schema and report shortfalls
python tools/validate_fact_pool.py

# 2. Check base model novelty on untrained pool (strict 0.0 prior accuracy)
python verify_novelty.py --model Qwen/Qwen2.5-1.5B-Instruct --max-prior-accuracy 0.0

# 3. Sweep plasticity hyperparameters on a single fact to ensure effective learning
python tools/plasticity_probe.py --fact-id fact_001 --smoke
```

---

## 🔬 Running Continual Learning Experiments

### Protocol 1 (Baseline & Historical Replication)

Runs the original 5 configurations (`C2` to `C6`):

```bash
# Run a single config
python scripts/run_config.py --config configs/config_c2_vanilla.json --seed 42

# Run all Protocol 1 configurations across seeds 42, 7, 123
python scripts/run_multi_seeds.py --protocol 1
```

### Protocol 2 (Enhanced Methodology & Control Arms)

Protocol 2 activates strict answer matching, candidate shuffling, isolated selection RNG, logprob candidate scoring, and includes seeded random (`C7`, `C9`) and topic-dispersed (`C8`, `C10`) control arms:

```bash
# Run a single Protocol 2 config
python scripts/run_config.py --config configs/v2/config_c4_olora_selfgen.json --seed 42

# Run all Protocol 2 configs across 3 seeds
python scripts/run_multi_seeds.py --protocol 2

# Or run specific configs
python scripts/run_multi_seeds.py --protocol 2 --configs C7 C8 C9 C10 --seeds 42 7
```

---

## 📊 Aggregating Results & Computing Metrics

Aggregate completed runs and compute learned accuracy, final accuracy, forgetting, and the Curriculum Transfer Gap:

```bash
# Compile Protocol 1 results
python compile_results.py --protocol 1

# Compile Protocol 2 results
python compile_results.py --protocol 2

# Calculate 95% bootstrap confidence intervals for CTG
python tools/bootstrap_ci.py --config-arm1 C3 --config-arm2 C4 --metric final_accuracy
```

---

## 🧪 Unit Tests

Run the CPU-only test suite covering all work packages (runs in < 15s):

```bash
pytest -v
```

See [`tests/README.md`](tests/README.md) for details.

---

## 📜 Citation / Paper

For complete methodology, mathematical formulations, and qualitative reasoning analysis, see [`paper_final.tex`](paper_final.tex).

```bibtex
@article{altamash2026seal_olora,
  title={Evaluating Subspace Orthogonality and Sparse Low-Rank Adaptation under Model-Directed Curricula in Autonomous Self-Editing Language Models},
  author={Shaikh, Altamash},
  journal={Honours Research Thesis},
  year={2026}
}
```
