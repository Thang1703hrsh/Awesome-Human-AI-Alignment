# Evaluator benchmarks and inference-time methods

This page covers the three benchmarks the survey uses as evidence (RewardBench 2, JudgeBench, AlpacaEval 2) and the
three native inference-time methods (CAA, best-of-N with a reward model, ARGS). Scoring is dependency-free; loading
data needs `datasets` and `huggingface_hub`, and model scoring needs the `inference` extra.

```bash
python -m pip install -e ".[inference]" datasets
hai-align bench list
```

## Benchmarks

Every scorer follows the benchmark's official code and was checked against it on randomized inputs (agreement to
floating-point precision). Proportions carry Wilson 95% intervals, as in the survey's result tables, and every
summary records the dataset, split, revision, evaluated model, and library version.

| Benchmark | Evaluates | Protocol |
|---|---|---|
| RewardBench 2 (`allenai/reward-bench-2`) | reward model or pointwise judge | best of four; a correct completion tied at the top with k−1 others earns 1/k; Ties uses the official 0.30/0.30/0.20/0.20/0.01 composite; overall = unweighted mean of six subsets |
| JudgeBench (`ScalerLab/JudgeBench`, `gpt`/`claude`) | LLM judge or reward model | judges see both orders and a pair is correct only when the two-trial vote is positive; reward models score each response once (exact ties go to B, as in the source) |
| AlpacaEval 2 (`tatsu-lab/alpaca_eval`) | policy | official package with `weighted_alpaca_eval_gpt4_turbo` for raw and length-controlled win rates; a native backend for any pairwise judge |

```bash
hai-align bench rewardbench2 --reward-model Skywork/Skywork-Reward-V2-Qwen3-0.6B --output-dir eval/rb2
hai-align bench judgebench --reward-model Skywork/Skywork-Reward-V2-Qwen3-0.6B --output-dir eval/jb
hai-align bench judgebench --judge-model gpt-4o --provider openai --output-dir eval/jb-gpt4o   # vanilla prompt
hai-align bench alpacaeval generate outputs/my-model --output eval/ae/outputs.json
hai-align bench alpacaeval score eval/ae/outputs.json --output-dir eval/ae               # needs `pip install alpaca-eval`
```

Pin `--revision` to a dataset commit for reproducible numbers. Python API:

```python
from human_alignment.benchmarks import load_rewardbench2, evaluate_rewardbench2
from human_alignment.integrations.reward_model import TransformersRewardModel

rm = TransformersRewardModel.from_pretrained("Skywork/Skywork-Reward-V2-Qwen3-0.6B")
rows, summary = evaluate_rewardbench2(load_rewardbench2(), rm)   # any (prompt, response) -> float works
print(summary["average"], summary["subsets"]["Ties"])
```

Limits: the AlpacaEval native backend's `length_controlled_minimal` is an unregularised fit of the package's minimal
length model without the instruction-difficulty term. It is a diagnostic; leaderboard-comparable LC win rates come
from the official backend. RewardBench 2 generative-judge prompts are not bundled: pass any pointwise scorer.

## Inference-time methods

All three leave the base model's weights unchanged and implement the `InferenceMechanism` interface
(`generate(model, request) -> InferenceResult`).

| Method | Source | What it changes |
|---|---|---|
| `ContrastiveActivationAddition` | Rimsky et al. (2024), `nrimsky/CAA` | adds `multiplier · v` to one decoder layer's output, from the end of the instruction onward; `v` = mean activation difference between positive and negative completions |
| `RewardModelBestOfN` | best-of-N selection | samples N responses and returns the highest-reward one |
| `ARGSDecoding` | Khanov et al. (2024), `deeplearning-wisc/args` | scores the policy's top-k next tokens as `logit + w · reward(prefix + token)`; greedy or sampled |

```python
from human_alignment.mechanisms.inference import ContrastiveActivationAddition, ARGSDecoding

caa = ContrastiveActivationAddition.from_examples(
    model, tokenizer,
    [("Is it OK to lie to my boss?", " No, honesty matters.", " Sure, if it helps you.")],
    layer=13, multiplier=4.0,
)
print(caa.generate_text("Should I hide this mistake?"))

args = ARGSDecoding(model, tokenizer, reward_model=rm, weight=1.5, topk=10)
print(args.decode("How do I stay safe online?")[0])
```

Differences from the sources: CAA reads activations at position −1 by default (the source uses −2, the answer letter
of `(A`/`(B`; pass `position=-2` for multiple-choice data) and expects left padding for batched prompts. ARGS scores
candidates as text with the reward model's own template, so the reward model need not share the policy tokenizer;
the source scores token ids directly, which is faster but requires a shared tokenizer.
