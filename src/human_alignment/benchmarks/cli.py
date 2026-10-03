"""``hai-align bench``: run RewardBench 2, JudgeBench and AlpacaEval 2 under their official protocols."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from human_alignment.benchmarks.common import write_report


def _reward_model(args):
    from human_alignment.integrations.reward_model import TransformersRewardModel

    return TransformersRewardModel.from_pretrained(
        args.reward_model, dtype=args.dtype, max_length=args.max_length, batch_size=args.batch_size
    )


def _provenance(dataset: str, split: str, args) -> dict:
    """Dataset, revision, limit and library version stored with every benchmark summary."""
    from human_alignment import __version__

    return {
        "dataset": dataset,
        "split": split,
        "revision": getattr(args, "revision", None) or "latest",
        "limit": getattr(args, "limit", None),
        "library_version": __version__,
    }


def _add_reward_model_args(p: argparse.ArgumentParser, required: bool) -> None:
    p.add_argument("--reward-model", required=required, help="sequence-classification reward model (HF id or path)")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=8)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hai-align bench")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list available benchmarks")

    rb = sub.add_parser("rewardbench2", help="best-of-4 accuracy of a reward model")
    _add_reward_model_args(rb, required=True)
    rb.add_argument("--subset", action="append", help="restrict to a subset (repeatable)")
    rb.add_argument("--limit", type=int)
    rb.add_argument("--output-dir", required=True)

    jb = sub.add_parser("judgebench", help="judge a reward model or an LLM judge on JudgeBench")
    jb.add_argument("--split", choices=("gpt", "claude"), default="gpt")
    _add_reward_model_args(jb, required=False)
    jb.add_argument("--judge-model", help="LLM judge behind an OpenAI-compatible API (vanilla prompt)")
    jb.add_argument("--provider", choices=("deepseek", "nvidia", "openai"), default="openai")
    jb.add_argument("--base-url")
    jb.add_argument("--api-keys", help="comma-separated keys (prefer JUDGE_API_KEYS or the provider variable)")
    jb.add_argument("--single-game", action="store_true", help="judge one order only (not the official protocol)")
    jb.add_argument("--limit", type=int)
    jb.add_argument("--output-dir", required=True)

    ae = sub.add_parser("alpacaeval", help="generate AlpacaEval 2 outputs or score them")
    ae_sub = ae.add_subparsers(dest="ae_cmd", required=True)
    gen = ae_sub.add_parser("generate", help="write model outputs in the AlpacaEval format")
    gen.add_argument("model")
    gen.add_argument("--output", required=True)
    gen.add_argument("--generator", help="name recorded in the outputs (default: model path)")
    gen.add_argument("--limit", type=int)
    gen.add_argument("--max-new-tokens", type=int, default=2048)
    gen.add_argument("--temperature", type=float, default=0.0)
    gen.add_argument("--top-p", type=float, default=1.0)
    gen.add_argument("--batch-size", type=int, default=8)
    gen.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    score = ae_sub.add_parser("score", help="score outputs with the official alpaca_eval package")
    score.add_argument("outputs")
    score.add_argument("--output-dir", required=True)
    score.add_argument("--annotators-config", default="weighted_alpaca_eval_gpt4_turbo")

    for p in (rb, jb, gen):
        p.add_argument("--revision", help="dataset revision (commit) to pin; recorded in the summary")
    args = parser.parse_args(argv)
    if args.cmd == "list":
        from human_alignment.benchmarks import BENCHMARKS

        for name, description in BENCHMARKS.items():
            print(f"{name:14s} {description}")
        return 0

    if args.cmd == "rewardbench2":
        from human_alignment.benchmarks.rewardbench2 import evaluate_rewardbench2, load_rewardbench2

        rows = load_rewardbench2(args.subset, args.limit, args.revision)
        scored, summary = evaluate_rewardbench2(rows, _reward_model(args))
        summary.update(_provenance("allenai/reward-bench-2", "test", args), model=args.reward_model)
        write_report(args.output_dir, "rewardbench2", scored, summary)
        print(json.dumps(summary, indent=2))
        return 0

    if args.cmd == "judgebench":
        from human_alignment.benchmarks import judgebench as J

        if bool(args.reward_model) == bool(args.judge_model):
            parser.error("judgebench needs exactly one of --reward-model or --judge-model")
        pairs = J.load_judgebench(args.split, args.limit, args.revision)
        if args.reward_model:
            rows, summary = J.evaluate_judgebench_with_reward_model(pairs, _reward_model(args))
            summary["judge"] = args.reward_model
        else:
            from human_alignment.safety.eval.rubric_judge import make_clients

            clients, _ = make_clients(args.provider, args.api_keys, args.base_url)
            judge = J.openai_vanilla_judge(clients[0], args.judge_model)
            rows, summary = J.evaluate_judgebench_with_judge(pairs, judge, reverse_order=not args.single_game)
            summary["judge"] = args.judge_model
        summary.update(_provenance(J.DATASET, args.split, args))
        write_report(args.output_dir, f"judgebench_{args.split}", rows, summary)
        print(json.dumps(summary, indent=2))
        return 0

    if args.cmd == "alpacaeval" and args.ae_cmd == "generate":
        from human_alignment.benchmarks.alpacaeval import load_alpacaeval, write_model_outputs
        from human_alignment.benchmarks.generation import chat_generate

        rows = load_alpacaeval(args.limit, args.revision)
        outputs = chat_generate(
            args.model,
            [r["instruction"] for r in rows],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            batch_size=args.batch_size,
            dtype=args.dtype,
        )
        rows = [{**r, "output": o} for r, o in zip(rows, outputs)]
        print(write_model_outputs(rows, args.output, args.generator or Path(args.model).name))
        return 0

    if args.cmd == "alpacaeval" and args.ae_cmd == "score":
        from human_alignment.benchmarks.alpacaeval import evaluate_alpacaeval_official

        result = evaluate_alpacaeval_official(args.outputs, args.output_dir, args.annotators_config)
        print(result[0].to_string() if isinstance(result, tuple) else result)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
