"""Evaluator and response benchmarks used as evidence in the survey (RewardBench 2, JudgeBench, AlpacaEval 2).

Scoring is dependency-free and follows each benchmark's official code; dataset loading needs ``datasets`` /
``huggingface_hub`` and model scoring the ``inference`` extra. Results carry Wilson 95% intervals where the metric
is a proportion, matching the survey's result tables.
"""

from human_alignment.benchmarks.alpacaeval import (
    evaluate_alpacaeval_native,
    evaluate_alpacaeval_official,
    length_controlled_minimal,
    load_alpacaeval,
    load_alpacaeval_baseline,
    win_rate,
    write_model_outputs,
)
from human_alignment.benchmarks.common import wilson_interval
from human_alignment.benchmarks.judgebench import (
    evaluate_judgebench_with_judge,
    evaluate_judgebench_with_reward_model,
    load_judgebench,
    openai_vanilla_judge,
    summarize_judgebench,
)
from human_alignment.benchmarks.rewardbench2 import (
    best_of_n_result,
    evaluate_rewardbench2,
    load_rewardbench2,
    summarize_rewardbench2,
    ties_score,
)

BENCHMARKS = {
    "rewardbench2": "Reward models and judges: best-of-4 accuracy over six subsets (Malik et al., 2025)",
    "judgebench": "LLM judges and reward models: correctness on hard response pairs (Tan et al., 2025)",
    "alpacaeval": "Policies: raw and length-controlled win rate against GPT-4 Turbo (Dubois et al., 2024)",
}

__all__ = [
    "BENCHMARKS",
    "best_of_n_result",
    "evaluate_alpacaeval_native",
    "evaluate_alpacaeval_official",
    "evaluate_judgebench_with_judge",
    "evaluate_judgebench_with_reward_model",
    "evaluate_rewardbench2",
    "length_controlled_minimal",
    "load_alpacaeval",
    "load_alpacaeval_baseline",
    "load_judgebench",
    "load_rewardbench2",
    "openai_vanilla_judge",
    "summarize_judgebench",
    "summarize_rewardbench2",
    "ties_score",
    "wilson_interval",
    "win_rate",
    "write_model_outputs",
]
