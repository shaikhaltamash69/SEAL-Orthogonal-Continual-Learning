# SEAL Data Directory

This directory contains the continual learning fact pool and pre-training baseline accuracy records.

## Files

- `fact_pool.json`: The active 19-fact pool evaluated in Protocol 1.
- `fact_pool_template.json`: Schema template for contributing new facts.
- `baseline_accuracy.json`: Legacy baseline accuracy evaluations on the untrained base model (`Qwen2.5-1.5B-Instruct`).
- `baseline_accuracy_<model-slug>.json`: Model-specific baseline accuracy records generated via `verify_novelty.py`.

---

## Fact Pool Schema

Each entry in `fact_pool.json` must be a JSON object with the following fields:

```json
{
  "id": "fact_001",
  "title": "Descriptive title of the factual event",
  "passage": "A factual reference paragraph used for synthetic QA generation or training.",
  "qa_pairs": [
    {
      "question": "Held-out evaluation question 1?",
      "answer": "Canonical answer",
      "answer_aliases": ["Alternative spelling 1", "Acceptable synonym"],
      "distractors": ["Wrong option 1", "Wrong option 2"]
    }
  ],
  "domain": "science",
  "source_url": "https://...",
  "date_of_event": "2025-01-15"
}
```

### Validation Tool

Run the fact pool validator to check schema adherence and quality criteria:

```bash
python tools/validate_fact_pool.py
```

Optional parameters:
- `--pool <path>`: Path to fact pool JSON (default: `data/fact_pool.json`)
- `--min-questions <N>`: Required QA pairs per fact (default: 4)
- `--min-date <YYYY-MM-DD>`: Earliest allowed date of event (default: 2025-01-01)

---

## TODO: facts and questions to author

The current `fact_pool.json` contains 19 facts with only 2 QA pairs each, dated in late 2024. As verified by `tools/validate_fact_pool.py`, the following authoring tasks are required from the user before executing full Protocol 2 benchmark runs:

1. **Expand Questions per Fact**:
   - Each fact currently has 2 questions. Author an additional 2 to 3 QA pairs per fact (bringing total to 4–5 questions per fact) to support robust bootstrap confidence intervals.
   - Author canonical `answer` and optional `answer_aliases` for each question.
   - Avoid questions where the answer is trivial or easily guessable without studying the passage.

2. **Expand Fact Pool Size**:
   - Expand the pool from 19 facts to at least 60 facts to provide sufficient statistical power for continual learning sequences.
   - Focus on recent events occurring after January 1, 2025 (`date_of_event >= 2025-01-01`) to minimise base model prior knowledge.

3. **Balance Topic Domains**:
   - The current pool is heavily skewed towards `sports` (8) and `science` (6), with `politics` (3), `economics` (1), and `technology` (1) severely under-represented.
   - Author at least 10–12 facts per domain across all 5 domains to enable clean cross-domain continual learning and dispersed curriculum analysis.

4. **Verify Novelty on Untrained Models**:
   - After authoring new facts, run:
     ```bash
     python verify_novelty.py --model Qwen/Qwen2.5-1.5B-Instruct --max-prior-accuracy 0.0
     ```
   - Any fact where the base model achieves > 0.0 accuracy must either be revised with more specific questions or excluded (`keep_in_pool: false`).

5. **fact_004**:
   - `fact_004 absent; reason unknown; author to fill in` (searched git history via `git log -S fact_004 --all`; no earlier commits contained `fact_004`).
