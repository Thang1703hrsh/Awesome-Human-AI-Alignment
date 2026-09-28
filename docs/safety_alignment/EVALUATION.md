# Evaluation protocol

Safety alignment needs at least two axes. A single reward score cannot show whether a model became safe by refusing
everything, and a safety rate alone cannot show whether useful behavior survived.

## Standard model-based report

The common report follows SafeRLHF, SafeDPO and BSO:

1. Generate one answer for every test prompt.
2. Score helpfulness with `PKU-Alignment/beaver-7b-v1.0-reward` (or the unified reward checkpoint).
3. Score cost with `PKU-Alignment/beaver-7b-v1.0-cost`.
4. Report mean reward, mean cost and `safe_rate = mean(cost <= 0)`.
5. When an SFT/reference generation is supplied, also report helpfulness and harmlessness win rates.

```bash
BASELINE_MODEL=PKU-Alignment/alpaca-7b-reproduced \
  scripts/safety_alignment/tools/evaluate.sh output/safedpo pku
cat evaluation/safedpo/pku/scored/summary.json
```

Available profiles:

| Profile | Dataset | Purpose |
|---|---|---|
| `pku` | `PKU-Alignment/PKU-SafeRLHF-30K`, test | common safety/helpfulness frontier |
| `beaver` | `PKU-Alignment/BeaverTails-Evaluation` | held-out SafeRLHF safety prompts |
| `do_not_answer` | `LibrAI/do-not-answer` | MidPO-style harmful instruction evaluation |
| `wildguard` | `allenai/wildguardmix`, `wildguardtest` | MidPO out-of-distribution harmful prompts |
| `xstest` | `walledai/XSTest`, test | 250 benign and 200 unsafe prompts for over-refusal/safety |

Use the unified scorer checkpoints when matching SafeDPO/BSO exactly:

```bash
REWARD_MODEL=PKU-Alignment/beaver-7b-unified-reward \
COST_MODEL=PKU-Alignment/beaver-7b-unified-cost \
BASELINE_MODEL=PKU-Alignment/alpaca-7b-reproduced \
  scripts/safety_alignment/tools/evaluate.sh output/bso pku
```

## Blinded LLM judge

The papers commonly complement score models with GPT pairwise assessment. Candidate and reference order must be
hidden and randomized. `hai-align safety judge` implements this and reads `OPENAI_API_KEY` only from the environment:

```bash
hai-align safety judge evaluation/safedpo/pku/paired.jsonl \
  --model YOUR_JUDGE_MODEL --output-dir evaluation/safedpo/pku/judge
```

It evaluates helpfulness and safety separately. The reported paper-style win rate is `wins / (wins + losses)`; ties
are retained in the raw JSONL but excluded from that denominator. Store the judge model identifier and raw judgments
with every result because judge versions are not interchangeable.

## Method-specific additions

- **SafeDPO:** evaluate XSTest's 250 benign prompts for partial/full refusal and its 200 unsafe prompts for safety.
  The paper uses an LLM classifier for both; do not substitute a string-matching refusal heuristic in reported results.
- **MidPO:** report PKU, Do-Not-Answer and harmful WildGuardTest subsets; include both Beaver score models and a
  blinded LLM judge.
- **CPO:** preserve control tags during generation and report MT-Bench (helpfulness), HaluEval 2.0 (honesty) and
  HackaPrompt (harmlessness). These are not reducible to the common PKU report.
- **BFPO:** run its pinned lm-evaluation-harness tasks for utility plus AdvBench/ALERT/RealToxicityPrompts for safety.
- **MORLHF/MODPO/CAN:** sweep all paper preference/dual weights and plot the entire safety-helpfulness frontier rather
  than selecting a single favorable checkpoint.

The common evaluator is deliberately dependency-light. Method-specific harnesses are kept as documented external
evaluations because copying a pinned benchmark implementation into this package would make its metrics silently drift.

