# SEAL Project: Critical Issues & Fix Plan
# Generated: 2026-10-01

CONFIRMED_ISSUES = """
==============================================================
SEAL PROJECT — VERIFIED ISSUES FROM AUTOMATED LOG ANALYSIS
==============================================================

[CRITICAL-1] Curriculum parser bug — 0 of 19 model choices were real
  - _parse_choice() uses re.search(r'^(\d+)', ...) — requires NUMBER at LINE START
  - Qwen2.5 responded with "NUMBER:" literally (following the prompt template) in 13/19 steps
  - In remaining 5 steps, model responded with "[fact_XXX]" format — also unparseable
  - Result: ALL 19 steps in C2, C4, C5 defaulted to index 0 = first remaining fact in insertion order
  - This makes C2, C3, C4, C5 ALL use the SAME insertion order
  - The "Curriculum Transfer Gap" is therefore NOT a curriculum effect — it is pure run variance

[CRITICAL-2] r=16/alpha=32 used in all runs, paper says r=8/alpha=16
  - Confirmed from configs: lora_r=16, lora_alpha=32 in all 4 config files
  - Paper Section IV-D says "r=8, alpha=16"
  - 2x rank means 2x capacity and 4x compute per adapter — non-trivial discrepancy

[HIGH-3] LoRI sparsity keeps top 20% NOT top 80% as paper states
  - Code: threshold = torch.quantile(grad_mag, 0.8) -> keeps entries >= 80th percentile = TOP 20%
  - Paper says "top-80% gradient masking" which is ambiguous
  - Need to decide: does "top-80%" mean keep top 80% (set sparsity=0.2) or keep top 20% (sparsity=0.8)?
  - LoRI paper (Zhang et al.) keeps the SPARSE subset — i.e. most gradients zeroed, few updated
  - So sparsity=0.8 (top 20% kept) is closer to the original LoRI design
  - Paper text needs to say "80% of B gradients zeroed per step (top-20% gradient magnitude retained)"

[HIGH-4] Single seed — no statistical confidence
  - C5 beats C3 by exactly 1 QA answer (7/38 vs 6/38 = 0.0263 difference)
  - C5 beats C3 by 2.63 percentage points, well within single-run noise
  - "Decisively rejects H0", "state-of-the-art" are unsupportable without error bars
  - Minimum fix: run 3 seeds, report mean +/- sd

[MEDIUM-5] QA evaluation: 2 pairs per fact (38 total), paper says 1 pair
  - 0.1579 = 6/38, 0.0789 = 3/38, 0.1842 = 7/38 — all fit denominator 38
  - Paper says "one held-out (q_i, a_i) pair" — incorrect, it's 2

[MEDIUM-6] CTG rounding inconsistency
  - True gap = 3/38 = 0.07895...
  - Paper text says 0.0790 (from 0.1579 - 0.0789 = 0.0790)
  - Figure caption says 0.0789
  - Should consistently report 3/38 = 0.0789 (to 4 d.p.)

[MEDIUM-7] No LoRI-Planned (C6) config
  - "LoRI eliminates the Curriculum Transfer Gap" cannot be verified
  - Need C6: LoRI + PlannedCurriculum to compute LoRI's own CTG

[LOW-8] Wrong references (as flagged by Claude)
  - [7] SEAL: should be Zweiger et al. "Self-Adapting Language Models"
  - [6] LoRI: should be Zhang, You, Panda, Goldstein (not Chen/Zhang/Liu/Wang)
  - [5] O-LoRA: pages 10658-10671, not 8658-8677 (EMNLP main, not Findings)

==============================================================
FIX PLAN — WHAT TO DO IN PRIORITY ORDER
==============================================================

PHASE 1 — Code Fixes (makes the experiment honest)
  1. Fix _parse_choice() to use re.search(r'\d+') anywhere in response
     AND also handle "[fact_XXX]" format by matching fact_id directly
  2. Add a C6 config: lori + planned curriculum
  3. Align paper text: r=16/alpha=32, 2 QA pairs per fact, sparsity description

PHASE 2 — Rerun Experiments (3 seeds)
  1. Run C2, C3, C4, C5, C6 with seeds 42, 7, 123
  2. The new C4 and C5 will have genuinely different orderings from C3
  3. Collect mean +/- sd for each config

PHASE 3 — Honest Paper Rewrite
  1. Report r=16/alpha=32 (what was actually used)
  2. Report 2 QA pairs per fact (38 evaluation points)
  3. Remove "decisively rejects H0" — say "pilot study with n=1 seed"
  4. Scope LoRI CTG claim: "C6 (LoRI Planned) vs C5 (LoRI Self-Gen)" after rerun
  5. Fix all 5 wrong references
  6. Fix rounding: consistently use 3/38 = 0.0789
  7. Scope "SEAL" name to avoid collision with Zweiger et al.

QUICK WINS (can do RIGHT NOW without rerunning)
  A. Fix _parse_choice() — takes 10 minutes, makes future reruns honest
  B. Add C6 config file — takes 5 minutes
  C. Fix paper hyperparams (r=16, alpha=32) and QA count (2 per fact)
  D. Fix references
  E. Remove statistical overclaiming language
"""

print(CONFIRMED_ISSUES)
