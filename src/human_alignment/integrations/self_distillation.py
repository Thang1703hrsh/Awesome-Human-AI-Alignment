"""Transformers backend for Preference-Based Self-Distillation (PBSD).

Ported from the PBSD reference code (github.com/LucasXinYu/PBSD, MIT):
prompts from ``data_collator.py`` and rollout, scoring, and gating from
``pbsd_trainer.py``. One model is both teacher and student. The teacher is the
frozen base model (LoRA adapters disabled) conditioned on the reference
solution; the student is the trainable model that sees only the problem.
Generation uses ``model.generate``; the reference's vLLM rollout path is not
included.
"""

from __future__ import annotations

import copy
import json
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from human_alignment.config import PBSDConfig
from human_alignment.exceptions import BackendUnavailableError, ConfigurationError, DatasetFormatError
from human_alignment.results import AlignmentRun
from human_alignment.supervision.answer_verification import extract_boxed_answer, is_correct

_BOXED_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
_INDEPENDENT_TRANSITION = (
    "\n\nAfter reading the reference solution above, make sure you truly understand "
    "the reasoning behind each step — do not copy or paraphrase it. Now, using your "
    "own words and independent reasoning, derive the same final answer to the problem above. "
    "Think step by step, explore different approaches, and don't be afraid to backtrack "
    "or reconsider if something doesn't work out:\n"
)
_REFERENCE_TRANSITION = (
    "\n\nWrite a clear, correct, SELF-CONTAINED step-by-step solution to the problem, "
    "as if you are solving it yourself and teaching a student. The correct solution "
    "above shows the right method and answer — follow its approach and reach the same "
    "answer — but your write-up MUST stand on its own: do NOT mention, cite, quote, or "
    "refer to 'the reference', 'the reference solution', 'the solution above', or the "
    "fact that you were given any solution. Do NOT critique or verify it. Just present "
    "the solution directly and concisely. Put the final answer within \\boxed{}:\n"
)


def _thinking(mode: str, default: bool) -> bool:
    return {"enabled": True, "disabled": False}.get(mode, default)


def pbsd_prompts(row: Mapping[str, Any], tokenizer: Any, config: PBSDConfig) -> tuple[str, str, str]:
    """Return the chat-templated student prompt, teacher prompt, and gold answer."""

    problem, solution = row["problem"], row["solution"]
    tooluse = config.task == "tooluse"
    gold = row.get("Answer")
    if gold is None or (isinstance(gold, str) and not gold.strip()):
        gold = solution if tooluse else extract_boxed_answer(solution)
    gold = "" if gold is None else str(gold)

    student_message = problem if tooluse else f"{problem}\n\n{_BOXED_INSTRUCTION}"
    transition = _REFERENCE_TRANSITION if config.teacher_use_reference else _INDEPENDENT_TRANSITION
    reference = (
        "Here is a reference solution to this problem:\n"
        f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
    )
    if tooluse:
        teacher_message = f"{problem}\n\n{reference}{transition}\n"
    else:
        pin = ""
        if config.teacher_use_reference:
            pin = (f"\n\nThe correct final answer is exactly: {gold}. Your solution "
                   f"MUST end by boxing this exact answer.")
        teacher_message = (
            f"Problem: {problem}\n\n{reference}{transition}{pin}\n{_BOXED_INSTRUCTION}"
        )

    def render(message: str, thinking: bool) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": message}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )

    return (
        render(student_message, _thinking(config.student_think_mode, False)),
        render(teacher_message, _thinking(config.teacher_think_mode, True)),
        gold,
    )


class PBSDCollator:
    """Left-pad student/teacher prompts; right-pad optional precomputed teacher rollouts."""

    def __init__(self, tokenizer: Any, config: PBSDConfig) -> None:
        self.tokenizer = tokenizer
        self.config = config

    def _encode(self, text: str) -> list[int]:
        # Chat templates already render special tokens such as BOS.
        kwargs: dict[str, Any] = {"add_special_tokens": False}
        if self.config.max_length is not None:
            kwargs.update(truncation=True, max_length=self.config.max_length)
        return list(self.tokenizer(text, **kwargs)["input_ids"])

    def _left_pad(self, sequences: Sequence[list[int]], torch: Any) -> tuple[Any, Any]:
        width = max(len(ids) for ids in sequences)
        pad = self.tokenizer.pad_token_id
        ids = torch.full((len(sequences), width), pad, dtype=torch.long)
        mask = torch.zeros((len(sequences), width), dtype=torch.long)
        for row, sequence in enumerate(sequences):
            if sequence:
                ids[row, width - len(sequence):] = torch.tensor(sequence, dtype=torch.long)
                mask[row, width - len(sequence):] = 1
        return ids, mask

    def __call__(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        import torch

        student, teacher, gold, trajectories = [], [], [], []
        for row in rows:
            if "problem" not in row or "solution" not in row:
                raise DatasetFormatError("PBSD rows require problem and solution columns")
            student_prompt, teacher_prompt, answer = pbsd_prompts(row, self.tokenizer, self.config)
            student.append(self._encode(student_prompt))
            teacher.append(self._encode(teacher_prompt))
            gold.append(answer)
            trajectories.append(row.get("teacher_trajectory"))
        if not all(student) or not all(teacher):
            raise DatasetFormatError("PBSD prompts must not be empty after tokenization")
        student_ids, student_mask = self._left_pad(student, torch)
        teacher_ids, teacher_mask = self._left_pad(teacher, torch)
        batch = {
            "student_prompt_ids": student_ids,
            "student_prompt_attention_mask": student_mask,
            "teacher_prompt_ids": teacher_ids,
            "teacher_prompt_attention_mask": teacher_mask,
            "ground_truth_answers": gold,
        }
        # As in the reference, precomputed best-of-N teacher rollouts are used only
        # when every row in the batch has one; each ends with EOS.
        if all(value is not None and str(value).strip() for value in trajectories):
            eos = self.tokenizer.eos_token_id
            encoded = [self._encode(str(value)) + [eos] for value in trajectories]
            width = max(len(ids) for ids in encoded)
            batch["teacher_completion_ids"] = torch.tensor(
                [ids + [self.tokenizer.pad_token_id] * (width - len(ids)) for ids in encoded],
                dtype=torch.long,
            )
        return batch


def completion_mask(completions: Any, eos_token_ids: Sequence[int], pad_token_id: int) -> Any:
    """Keep each completion through its first EOS and drop padding.

    The reference re-includes the first PAD position, which equals the stopping
    EOS when ``pad == eos`` but scores one padding token when they differ (as in
    Qwen3). Here the stopping EOS is always kept and right padding never is, also
    for a rollout truncated at ``max_new_tokens`` and padded to the batch width.
    """

    import torch

    is_eos = torch.zeros_like(completions, dtype=torch.bool)
    for token_id in eos_token_ids:
        is_eos |= completions.eq(int(token_id))
    seen_before = is_eos.long().cumsum(-1) - is_eos.long()
    return (seen_before.eq(0) & (completions.ne(int(pad_token_id)) | is_eos)).long()


def sequence_log_probs(
    model: Any,
    prompt_ids: Any,
    prompt_mask: Any,
    completions: Sequence[Any],
    eos_token_ids: Sequence[int],
    pad_token_id: int,
    normalize_by_length: bool,
) -> list[Any]:
    """Score completions after one left-padded prompt batch, as ``score_two_completions_logprob``."""

    import torch

    width = max(int(item.shape[1]) for item in completions)
    padded = [torch.nn.functional.pad(item, (0, width - int(item.shape[1])), value=pad_token_id)
              for item in completions]
    masks = [completion_mask(item, eos_token_ids, pad_token_id) for item in padded]
    count = len(padded)
    input_ids = torch.cat([torch.cat([prompt_ids, item], 1) for item in padded], 0)
    attention = torch.cat([torch.cat([prompt_mask, mask], 1) for mask in masks], 0)
    # Left padding would shift RoPE positions; derive positions from the mask.
    positions = attention.long().cumsum(-1) - 1
    positions.masked_fill_(attention == 0, 1)
    logits = model(input_ids=input_ids, attention_mask=attention, position_ids=positions,
                   use_cache=False).logits
    prompt_length = prompt_ids.shape[1]
    logits = logits[:, prompt_length - 1 : prompt_length + width - 1].float()
    targets = torch.cat(padded, 0)
    mask = torch.cat(masks, 0).to(logits.dtype)
    token_logps = logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1) * mask
    scores = token_logps.sum(-1)
    if normalize_by_length:
        scores = scores / mask.sum(-1).clamp(min=1)
    return list(scores.chunk(count))


def _make_trainer_class(transformers: Any) -> type[Any]:
    import torch

    from human_alignment.mechanisms.training.distillation_losses import pbsd_loss

    class PBSDTrainer(transformers.Trainer):
        def __init__(self, *args: Any, pbsd_config: PBSDConfig, teacher_model: Any = None,
                     **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.pbsd_config = pbsd_config
            self.model_accepts_loss_kwargs = False
            self.teacher_model = None
            if teacher_model is not None:
                teacher_model.requires_grad_(False)
                teacher_model.eval()
                self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)
            self._pbsd_metrics: dict[str, list[float]] = {}

        def _teacher(self, model: Any) -> tuple[Any, Any]:
            if self.teacher_model is not None:
                return self.teacher_model, nullcontext()
            return model, self.accelerator.unwrap_model(model).disable_adapter()

        def _eos_ids(self, model: Any) -> list[int]:
            ids = {int(self.processing_class.eos_token_id)}
            configured = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
            if configured is not None:
                ids.update(int(value) for value in
                           (configured if isinstance(configured, (list, tuple)) else [configured]))
            return sorted(ids)

        def _generate(self, model: Any, ids: Any, mask: Any, max_new_tokens: int) -> Any:
            cfg = self.pbsd_config
            options: dict[str, Any] = dict(
                do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p,
                max_new_tokens=max_new_tokens, pad_token_id=self.processing_class.pad_token_id,
                eos_token_id=self._eos_ids(model), use_cache=True,
            )
            if cfg.top_k > 0:
                options["top_k"] = cfg.top_k
            output = model.generate(input_ids=ids, attention_mask=mask, **options)
            return output[:, ids.shape[1]:]

        def compute_loss(self, model: Any, inputs: Mapping[str, Any], return_outputs: bool = False,
                         num_items_in_batch: Any = None) -> Any:
            if return_outputs:
                raise ValueError("PBSD does not return model outputs")
            cfg = self.pbsd_config
            unwrapped = self.accelerator.unwrap_model(model)
            student_ids = inputs["student_prompt_ids"]
            student_mask = inputs["student_prompt_attention_mask"]
            teacher_ids = inputs["teacher_prompt_ids"]
            teacher_mask = inputs["teacher_prompt_attention_mask"]
            teacher, teacher_context = self._teacher(model)
            eos_ids = self._eos_ids(unwrapped)
            pad = int(self.processing_class.pad_token_id)

            with torch.no_grad():
                student_rollout = self._generate(unwrapped, student_ids, student_mask,
                                                 cfg.student_max_new_tokens)
                teacher_rollout = inputs.get("teacher_completion_ids")
                if teacher_rollout is None:
                    teacher_model, context = self._teacher(model)
                    generator = (self.accelerator.unwrap_model(teacher_model)
                                 if self.teacher_model is not None else unwrapped)
                    with context:
                        teacher_rollout = self._generate(generator, teacher_ids, teacher_mask,
                                                         cfg.teacher_max_new_tokens)
            rollouts = (teacher_rollout, student_rollout)
            with torch.no_grad(), teacher_context:
                teacher_on_teacher, teacher_on_student = sequence_log_probs(
                    teacher, teacher_ids, teacher_mask, rollouts, eos_ids, pad,
                    cfg.normalize_logprob_by_length)
            student_on_teacher, student_on_student = sequence_log_probs(
                model, student_ids, student_mask, rollouts, eos_ids, pad,
                cfg.normalize_logprob_by_length)

            delta = ((teacher_on_teacher - student_on_teacher)
                     - (teacher_on_student - student_on_student)).detach()
            gate = torch.ones_like(delta)
            if cfg.correctness_gate:
                # Keep only pairs where the teacher rollout is correct and the student's is not.
                gold = inputs["ground_truth_answers"]
                teacher_text = self.processing_class.batch_decode(teacher_rollout, skip_special_tokens=True)
                student_text = self.processing_class.batch_decode(student_rollout, skip_special_tokens=True)
                teacher_ok = torch.tensor([float(is_correct(text, answer, cfg.task))
                                           for text, answer in zip(teacher_text, gold)], device=delta.device)
                student_ok = torch.tensor([float(is_correct(text, answer, cfg.task))
                                           for text, answer in zip(student_text, gold)], device=delta.device)
                gate = teacher_ok * (1.0 - student_ok)
                self._record("pbsd/teacher_acc", teacher_ok.mean())
                self._record("pbsd/student_acc", student_ok.mean())
            loss = pbsd_loss(teacher_on_teacher, student_on_teacher, teacher_on_student,
                             student_on_student, beta=cfg.beta, loss_type=cfg.pairwise_loss_type,
                             gate=gate)
            self._record("pbsd/delta_mean", delta.mean())
            self._record("pbsd/teacher_win_rate", delta.gt(0).float().mean())
            self._record("pbsd/gate_frac", gate.mean())
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite PBSD loss")
            return loss

        def _record(self, name: str, value: Any) -> None:
            self._pbsd_metrics.setdefault(name, []).append(float(value))

        def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
            for name, values in self._pbsd_metrics.items():
                logs[name] = sum(values) / len(values)
            self._pbsd_metrics = {}
            super().log(logs, *args, **kwargs)

    return PBSDTrainer


class PBSDBackend:
    """Train PBSD with Transformers and PEFT; one model serves as teacher and student."""

    def train(self, *, method: str, model: Any, dataset: Any, config: PBSDConfig,
              options: Mapping[str, Any] | None = None) -> AlignmentRun:
        if method != "pbsd":
            raise ConfigurationError(f"Unsupported self-distillation method: {method}")
        if not isinstance(config, PBSDConfig):
            raise ConfigurationError("PBSD requires PBSDConfig")
        try:
            import accelerate  # noqa: F401
            import torch
            import transformers
        except ImportError as exc:
            raise BackendUnavailableError(
                "PBSD requires `pip install human-ai-alignment[self_distillation]`."
            ) from exc
        from human_alignment.integrations.transformers import make_generator

        opts = dict(options or {})
        tokenizer = opts.pop("tokenizer", None)
        model_kwargs = dict(opts.pop("model_kwargs", {}))
        split = opts.pop("dataset_split", "train")
        if opts:
            raise ConfigurationError(f"Unknown PBSD backend options: {', '.join(sorted(opts))}")
        if tokenizer is None:
            if not isinstance(model, (str, Path)):
                raise ConfigurationError("Pass tokenizer when model is an object")
            tokenizer = str(model)
        if isinstance(tokenizer, (str, Path)):
            tokenizer = transformers.AutoTokenizer.from_pretrained(str(tokenizer))
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.eos_token_id is None:
            raise ConfigurationError("PBSD requires a tokenizer with eos_token_id")
        tokenizer.padding_side = "left"

        policy = (transformers.AutoModelForCausalLM.from_pretrained(str(model), **model_kwargs)
                  if isinstance(model, (str, Path)) else model)
        teacher = None
        if config.use_lora:
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError as exc:
                raise BackendUnavailableError("PBSD LoRA training requires `pip install peft`") from exc
            # The reference disables all dropout, including LoRA dropout.
            policy = get_peft_model(policy, LoraConfig(
                r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=0.0,
                target_modules=list(config.lora_target_modules), task_type="CAUSAL_LM"))
        else:
            # Without adapters, a frozen copy of the initial model is the same fixed teacher.
            teacher = copy.deepcopy(policy)
        for module in policy.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0

        rows = dataset.data
        if isinstance(rows, (str, Path)):
            import datasets

            if Path(rows).is_dir():
                rows = datasets.load_from_disk(str(rows))
                if hasattr(rows, "keys"):
                    rows = rows[split]
            else:
                rows = datasets.load_dataset(str(rows), split=split)
        if not len(rows):
            raise ConfigurationError("PBSD dataset must not be empty")

        kwargs = config.trainer_kwargs()
        kwargs.pop("max_length", None)
        kwargs.update(remove_unused_columns=False, label_names=[])
        args = transformers.TrainingArguments(**kwargs)
        if args.deepspeed or args.fsdp:
            raise ConfigurationError("The PBSD backend supports a single device or DDP")
        transformers.set_seed(config.seed)
        collator = PBSDCollator(tokenizer, config)
        collator([rows[0]])  # Fail on malformed rows before optimizer allocation.
        trainer = _make_trainer_class(transformers)(
            model=policy, args=args, train_dataset=rows, processing_class=tokenizer,
            data_collator=collator, pbsd_config=config, teacher_model=teacher)
        output = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
        trained = trainer.accelerator.unwrap_model(trainer.model)

        def save(path: str) -> None:
            if trainer.is_world_process_zero():
                destination = Path(path)
                destination.mkdir(parents=True, exist_ok=True)
                trained.save_pretrained(destination)
                tokenizer.save_pretrained(destination)
                (destination / "pbsd_config.json").write_text(
                    json.dumps(asdict(config), indent=2, default=str), encoding="utf-8")

        save(config.output_dir)
        return AlignmentRun(
            model=trained, tokenizer=tokenizer, output_dir=config.output_dir,
            metrics={key: float(value) for key, value in output.metrics.items()
                     if isinstance(value, (int, float))},
            metadata={"backend": "pbsd", "method": method, "lora": config.use_lora},
            _generator=make_generator(trained, tokenizer), _saver=save,
        )
