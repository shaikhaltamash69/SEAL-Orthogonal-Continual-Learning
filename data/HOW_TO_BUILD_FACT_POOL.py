"""
HOW TO BUILD YOUR FACT POOL (fact_pool.json)
=============================================

GOAL: 30-50 facts that Qwen2.5-1.5B (trained on data up to ~early 2024) does NOT know.
Use events, records, discoveries from mid-2024 onwards.

STRUCTURE: Each fact needs:
  - id: unique string like "fact_001"
  - title: short descriptive title (used in curriculum selection prompts)
  - passage: 150-300 words of factual text. Be SPECIFIC: include names, numbers, dates.
  - qa_pairs: 2-3 QA pairs. Questions must be answerable ONLY from this passage.
  - domain: category label
  - source_url: where you got it (for your own reference/audit)
  - date_of_event: YYYY-MM-DD

SOURCES TO USE (all free, reliable, verifiable):
  Science:    https://www.nature.com/news | https://phys.org | Nobel Prize .org
  Sports:     https://www.espn.com | official sports federation websites
  Tech:       Official company blog posts (not Wikipedia)
  Politics:   Government press releases, BBC News
  Economics:  World Bank data.worldbank.org | IMF imf.org/en/News

QUALITY RULES:
  1. The passage must be 150-300 words minimum
  2. Each QA answer must be a short, exact string (not a long paragraph)
  3. Avoid questions whose answers could be guessed without reading the passage
  4. Prefer facts with SPECIFIC NUMBERS (measurement, count, date, cost) as answers
     - "42 light-years" or "3,847 meters" is better than "far away"
  5. After writing each fact, ask the base model the QA before training (verify_novelty.py)

CATEGORIES TO FILL (aim for 6-8 facts per domain):
  science    - lab discoveries, space missions, medical breakthroughs
  sports     - world records, championship results, athlete milestones
  technology - product launches, AI milestones, infrastructure projects
  politics   - election results, policy decisions, international agreements
  economics  - market records, trade data, company milestones

TIPS:
  - Avoid anything about ChatGPT/LLMs that the model may have seen in pretraining
  - Best facts: Nobel Prize 2024 winners, Paris Olympics 2024 specific results,
    specific scientific papers published late 2024+, specific government actions
  - One passage can cover multiple related facts (combine if needed)
  - The more OBSCURE and SPECIFIC the number, the more reliable the novelty test
"""
