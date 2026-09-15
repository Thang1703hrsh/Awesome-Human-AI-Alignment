"""Command-line access to training, generation, and taxonomy metadata."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from human_alignment.api import align, load_checkpoint, run_config
from human_alignment.config import CheckpointConfig
from human_alignment.catalog import categories, load_methods, validate_catalog
from human_alignment.registry import default_registry
from human_alignment.types import AlignmentStage


def _catalog_validate(_: argparse.Namespace) -> int:
    report = validate_catalog()
    print(
        json.dumps(
            {
                "valid": report.valid,
                "dimensions": report.dimension_count,
                "terminal_categories": report.category_count,
                "errors": report.errors,
            },
            indent=2,
        )
    )
    return 0 if report.valid else 1


def _catalog_list(args: argparse.Namespace) -> int:
    rows = categories()
    if args.dimension:
        rows = tuple(row for row in rows if row["dimension"] == args.dimension)
    for row in rows:
        print(f"{row['id']:<38} {row['name']} [{row['dimension']}]")
    return 0


def _method_list(args: argparse.Namespace) -> int:
    rows = load_methods()
    if args.category:
        rows = [row for row in rows if args.category in row["taxonomy_categories"]]
    for row in rows:
        print(f"{row['id']:<28} {row['name']} ({row['implementation_status']})")
    return 0


def _registry_list(args: argparse.Namespace) -> int:
    stage = AlignmentStage(args.stage) if args.stage else None
    for registered in default_registry().list(stage=stage):
        metadata = registered.metadata
        categories_text = ",".join(metadata.taxonomy_categories)
        print(f"{metadata.id:<28} {metadata.stage.value:<10} {categories_text}")
    return 0


def _train(args: argparse.Namespace) -> int:
    if args.config:
        run = run_config(args.config)
    else:
        missing = [
            name
            for name, value in (
                ("method", args.method),
                ("model", args.model),
                ("dataset", args.dataset),
            )
            if not value
        ]
        if missing:
            raise SystemExit("Missing required arguments: " + ", ".join(missing))
        run = align(
            args.method,
            model=args.model,
            dataset=args.dataset,
            output_dir=args.output,
        )
    print(json.dumps({"method": run.method, "output_dir": run.output_dir, "metrics": run.metrics}))
    return 0


def _generate(args: argparse.Namespace) -> int:
    run = load_checkpoint(
        args.model,
        config=CheckpointConfig(
            adapter_path=args.adapter,
            device_map=args.device_map,
            torch_dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        ),
    )
    print(run.generate(args.prompt, max_new_tokens=args.max_new_tokens))
    return 0


def _prepare_distillation(args: argparse.Namespace) -> int:
    from human_alignment.workflows import prepare_preference_distillation_dataset

    result = prepare_preference_distillation_dataset(
        args.dataset,
        objective=args.objective,
        output_dir=args.output,
        teacher_model=args.teacher,
        reference_model=args.reference,
        student_model=args.student,
        tokenizer=args.tokenizer,
        split=args.split,
        top_k=args.top_k,
        num_responses=args.num_responses,
        teacher_score_mode=args.teacher_score_mode,
        generation_backend=args.generation_backend,
    )
    print(json.dumps({"objective": result.objective, "dataset": str(result.path)}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hai-align", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    catalog_parser = commands.add_parser("catalog", help="Inspect taxonomy metadata")
    catalog_commands = catalog_parser.add_subparsers(dest="catalog_command", required=True)
    validate_parser = catalog_commands.add_parser("validate")
    validate_parser.set_defaults(handler=_catalog_validate)
    list_parser = catalog_commands.add_parser("list")
    choices = sorted({row["dimension"] for row in categories()})
    list_parser.add_argument("--dimension", choices=choices)
    list_parser.set_defaults(handler=_catalog_list)

    methods_parser = commands.add_parser("methods", help="List software methods")
    methods_parser.add_argument("--category")
    methods_parser.set_defaults(handler=_method_list)

    registry_parser = commands.add_parser("registry", help="List registered factories")
    registry_parser.add_argument("--stage", choices=[stage.value for stage in AlignmentStage])
    registry_parser.set_defaults(handler=_registry_list)

    train_parser = commands.add_parser("train", help="Train an alignment method")
    train_parser.add_argument("method", nargs="?")
    train_parser.add_argument("--model")
    train_parser.add_argument("--dataset")
    train_parser.add_argument("--output")
    train_parser.add_argument("--config")
    train_parser.set_defaults(handler=_train)

    prepare_parser = commands.add_parser("prepare", help="Prepare alignment supervision")
    prepare_commands = prepare_parser.add_subparsers(dest="prepare_command", required=True)
    distillation_parser = prepare_commands.add_parser(
        "distillation", help="Prepare DCKD, TVKD, ADPA, CTPD, PPD, or VPD data"
    )
    distillation_parser.add_argument(
        "objective", choices=("dckd", "tvkd", "adpa", "ctpd", "ppd", "vpd")
    )
    distillation_parser.add_argument(
        "--generation-backend",
        choices=("transformers", "vllm"),
        default="transformers",
    )
    distillation_parser.add_argument("--dataset", required=True)
    distillation_parser.add_argument("--output", required=True)
    distillation_parser.add_argument("--teacher")
    distillation_parser.add_argument("--reference")
    distillation_parser.add_argument("--student")
    distillation_parser.add_argument("--tokenizer")
    distillation_parser.add_argument("--split")
    distillation_parser.add_argument("--top-k", type=int, default=50)
    distillation_parser.add_argument("--num-responses", type=int, default=4)
    distillation_parser.add_argument(
        "--teacher-score-mode",
        choices=("sequence_logprob", "pairwise"),
        default="sequence_logprob",
    )
    distillation_parser.set_defaults(handler=_prepare_distillation)

    generate_parser = commands.add_parser("generate", help="Generate from a saved model")
    generate_parser.add_argument("--model", required=True)
    generate_parser.add_argument("--prompt", required=True)
    generate_parser.add_argument("--max-new-tokens", type=int, default=128)
    generate_parser.add_argument("--adapter")
    generate_parser.add_argument("--device-map")
    generate_parser.add_argument("--dtype")
    generate_parser.add_argument("--trust-remote-code", action="store_true")
    generate_parser.set_defaults(handler=_generate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
