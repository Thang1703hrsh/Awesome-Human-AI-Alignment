"""Command line interface for training, resource preparation and evaluation."""

from __future__ import annotations

import argparse
import logging
import sys

from human_alignment.safety.registry import METHODS, get_runner
from human_alignment.safety.schema import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hai-align safety")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run one experiment config")
    run.add_argument("config")
    run.add_argument(
        "overrides",
        nargs="*",
        help="dotted overrides, e.g. train.learning_rate=1e-6 data.limit=64",
    )
    show = sub.add_parser("show", help="print the resolved config")
    show.add_argument("config")
    show.add_argument("overrides", nargs="*")
    sub.add_parser("list", help="list methods")
    prepare = sub.add_parser(
        "prepare", help="download external inputs for a method family"
    )
    prepare.add_argument("profile", help="method/profile name, or all")
    prepare.add_argument("--root", default="downloads")
    prepare.add_argument("--include-evaluation", action="store_true")
    prepare.add_argument(
        "--token",
        default=None,
        help="Hugging Face token (prefer HF_TOKEN in the environment)",
    )
    prepare.add_argument(
        "--dry-run", action="store_true", help="print destinations without downloading"
    )
    doctor = sub.add_parser(
        "doctor", help="check packages, CUDA and expected project paths"
    )
    doctor.add_argument("--root", default=".")
    generate = sub.add_parser(
        "generate", help="generate responses for an evaluation dataset"
    )
    generate.add_argument("model")
    generate.add_argument("--dataset", default="PKU-Alignment/PKU-SafeRLHF-30K")
    generate.add_argument("--dataset-config")
    generate.add_argument("--split", default="test")
    generate.add_argument("--prompt-column")
    generate.add_argument("--output", required=True)
    generate.add_argument("--limit", type=int)
    generate.add_argument("--batch-size", type=int, default=8)
    generate.add_argument("--max-new-tokens", type=int, default=256)
    generate.add_argument("--temperature", type=float, default=0.0)
    generate.add_argument("--top-p", type=float, default=1.0)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    generate.add_argument(
        "--keep-column",
        action="append",
        default=[],
        help="copy a dataset column into each output row (repeatable), e.g. --keep-column label",
    )
    score = sub.add_parser(
        "score", help="score generated JSONL with Beaver reward and cost models"
    )
    score.add_argument("generations")
    score.add_argument("--output-dir", required=True)
    score.add_argument("--reward-model", default="PKU-Alignment/beaver-7b-v1.0-reward")
    score.add_argument("--cost-model", default="PKU-Alignment/beaver-7b-v1.0-cost")
    score.add_argument("--batch-size", type=int, default=8)
    score.add_argument("--max-length", type=int, default=512)
    score.add_argument("--threshold", type=float, default=0.0)
    pair = sub.add_parser(
        "pair", help="join candidate and reference generation JSONL by prompt"
    )
    pair.add_argument("candidate")
    pair.add_argument("reference")
    pair.add_argument("--output", required=True)
    judge = sub.add_parser(
        "judge", help="blinded helpfulness/safety LLM judge on paired JSONL"
    )
    judge.add_argument("paired_generations")
    judge.add_argument("--output-dir", required=True)
    judge.add_argument(
        "--model", required=True, help="judge model available through the OpenAI API"
    )
    judge.add_argument("--seed", type=int, default=42)
    for name, help_text in (
        ("xstest-judge", "XSTest over-refusal / harmlessness LLM judge (BSO authors' protocol)"),
        ("rubric-judge", "pointwise helpfulness score and safe/unsafe label LLM judge"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("generations", help="generation JSONL from `generate`")
        p.add_argument("--output-dir", required=True)
        p.add_argument("--provider", choices=("deepseek", "nvidia", "openai"), default="deepseek")
        p.add_argument("--model", help="judge model (default: the provider preset)")
        p.add_argument("--base-url", help="override the provider's OpenAI-compatible endpoint")
        p.add_argument(
            "--api-keys",
            help="comma-separated keys used round-robin (prefer JUDGE_API_KEYS or the provider variable)",
        )
        p.add_argument("--workers", type=int, default=8)
        p.add_argument(
            "--exclude-unparsed",
            action="store_true",
            help="drop unparseable judgments from the means instead of scoring them 0 as the source does",
        )
        if name == "xstest-judge":
            p.add_argument("--label-column", default="label")
            p.add_argument("--max-per-split", type=int, help="judge at most N safe and N unsafe rows")
        else:
            p.add_argument("--mode", choices=("helpfulness", "safety", "both"), default="both")
            p.add_argument("--temperature", type=float, default=0.7)
            p.add_argument("--max-samples", type=int)
            p.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if args.cmd == "list":
        for name, spec in METHODS.items():
            print(f"{name:28s} {spec.description}  [{spec.source}]")
        return 0
    if args.cmd == "prepare":
        from human_alignment.safety.resources import download_resources

        paths = download_resources(
            args.profile, args.root, args.include_evaluation, args.token, args.dry_run
        )
        for path in paths:
            print(path)
        return 0
    if args.cmd == "doctor":
        from human_alignment.safety.doctor import collect_diagnostics, diagnostics_text

        report = collect_diagnostics(args.root)
        print(diagnostics_text(report))
        return int(
            any(v is None for k, v in report["packages"].items() if k != "deepspeed")
        )
    if args.cmd == "generate":
        from human_alignment.safety.eval.generate import generate_rows, load_prompt_rows

        rows = load_prompt_rows(
            args.dataset,
            args.split,
            args.dataset_config,
            args.prompt_column,
            args.limit,
            args.keep_column,
        )
        print(
            generate_rows(
                args.model,
                rows,
                args.output,
                args.batch_size,
                args.max_new_tokens,
                args.temperature,
                args.top_p,
                args.seed,
                dtype=args.dtype,
            )
        )
        return 0
    if args.cmd == "score":
        import json
        from human_alignment.safety.eval.scoring import score_file

        result = score_file(
            args.generations,
            args.output_dir,
            args.reward_model,
            args.cost_model,
            args.batch_size,
            args.max_length,
            args.threshold,
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.cmd == "pair":
        from human_alignment.safety.eval.pairwise import attach_reference

        print(attach_reference(args.candidate, args.reference, args.output))
        return 0
    if args.cmd == "judge":
        import json
        from human_alignment.safety.eval.judge import judge_file

        print(
            json.dumps(
                judge_file(
                    args.paired_generations, args.output_dir, args.model, args.seed
                ),
                indent=2,
            )
        )
        return 0
    if args.cmd in ("xstest-judge", "rubric-judge"):
        import json
        from human_alignment.safety.eval.rubric_judge import judge_file as rubric_judge_file

        common = dict(
            provider=args.provider,
            model=args.model,
            api_keys=args.api_keys,
            base_url=args.base_url,
            workers=args.workers,
            exclude_unparsed=args.exclude_unparsed,
        )
        if args.cmd == "xstest-judge":
            options = dict(label_column=args.label_column, max_per_split=args.max_per_split)
            protocol = "xstest"
        else:
            options = dict(
                mode=args.mode,
                temperature=args.temperature,
                max_samples=args.max_samples,
                seed=args.seed,
            )
            protocol = "rubric"
        summary = rubric_judge_file(
            protocol, args.generations, args.output_dir, **common, **options
        )
        print(json.dumps(summary, indent=2))
        return 0
    cfg = load_config(args.config, args.overrides)
    if args.cmd == "show":
        from human_alignment.safety.schema import _config_to_dict
        import yaml

        print(yaml.safe_dump(_config_to_dict(cfg), sort_keys=False))
        return 0
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    out = get_runner(cfg.method)(cfg)
    peak = (
        f", peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB"
        if torch.cuda.is_available()
        else ""
    )
    print(f"done: {out}{peak}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

