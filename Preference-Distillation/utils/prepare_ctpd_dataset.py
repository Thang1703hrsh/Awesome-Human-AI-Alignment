"""Prepare offline parent-token mappings and CTPD weights.

This follows the CTPD pipeline at the level needed by this codebase:
build shared parent-token spans between the student and teacher tokenizers,
score those spans with the positive and negative teacher models, and export
the parent lists/weights consumed by scripts/run_distill_dpo.py.
"""

import argparse
import os
from pathlib import Path

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer


def _apply_prompt_template(tokenizer, messages):
    if isinstance(messages, list):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return str(messages)


def _assistant_response_text(tokenizer, message):
    if isinstance(message, dict):
        return tokenizer.apply_chat_template([message], tokenize=False, add_special_tokens=False)
    return str(message)


def _extract_prompt_and_response(example, side, tokenizer):
    value = example[side]
    if isinstance(value, list) and value and isinstance(value[-1], dict):
        prompt_messages = value[:-1]
        response = _assistant_response_text(tokenizer, value[-1])
        return prompt_messages, response

    prompt = example.get("prompt", "")
    response = _assistant_response_text(tokenizer, value)
    return prompt, response


def _encode_response(tokenizer, text):
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return list(encoded["input_ids"]), [tuple(offset) for offset in encoded["offset_mapping"]]


def _parent_spans(student_offsets, teacher_offsets):
    if not student_offsets or not teacher_offsets:
        return []

    end = max(student_offsets[-1][1], teacher_offsets[-1][1])
    boundary_counts = {0: 2, end: 2}
    for start, stop in student_offsets:
        boundary_counts[start] = boundary_counts.get(start, 0) + 1
        boundary_counts[stop] = boundary_counts.get(stop, 0) + 1
    for start, stop in teacher_offsets:
        boundary_counts[start] = boundary_counts.get(start, 0) + 1
        boundary_counts[stop] = boundary_counts.get(stop, 0) + 1

    boundaries = [0]
    for _, stop in student_offsets:
        if boundary_counts.get(stop, 0) >= 2 and stop != boundaries[-1]:
            boundaries.append(int(stop))

    if boundaries[-1] != end:
        boundaries.append(int(end))

    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


def _indices_in_span(offsets, span):
    start, stop = span
    return [
        idx
        for idx, (tok_start, tok_stop) in enumerate(offsets)
        if tok_stop <= stop and tok_stop > start
    ]


def _score_token_logps(model, input_ids, prompt_len, response_len):
    with torch.no_grad():
        logits = model(input_ids=input_ids.unsqueeze(0)).logits[0].log_softmax(-1)

    scores = []
    for response_idx in range(response_len):
        position = prompt_len + response_idx
        if position == 0 or position >= logits.shape[0]:
            scores.append(0.0)
            continue
        token_id = input_ids[position]
        scores.append(float(logits[position - 1, token_id]))
    return scores


def _sum_by_parent(token_scores, parent_indices):
    values = []
    for indices in parent_indices:
        values.append(sum(token_scores[index] for index in indices if index < len(token_scores)))
    return values


def _prepare_side(example, side, teacher_tokenizer, student_tokenizer, teacher, reference, device):
    prompt_value, response = _extract_prompt_and_response(example, side, student_tokenizer)
    teacher_prompt = _apply_prompt_template(student_tokenizer, prompt_value)
    student_prompt = _apply_prompt_template(student_tokenizer, prompt_value)

    teacher_prompt_ids = teacher_tokenizer(teacher_prompt, add_special_tokens=False)["input_ids"]
    student_prompt_ids = student_tokenizer(student_prompt, add_special_tokens=False)["input_ids"]

    teacher_response_ids, teacher_offsets = _encode_response(teacher_tokenizer, response)
    _, student_offsets = _encode_response(student_tokenizer, response)

    spans = _parent_spans(student_offsets, teacher_offsets)
    teacher_parent_indices = [_indices_in_span(teacher_offsets, span) for span in spans]
    student_parent_indices = [
        [len(student_prompt_ids) + index for index in _indices_in_span(student_offsets, span)]
        for span in spans
    ]

    teacher_input_ids = torch.tensor(
        teacher_prompt_ids + teacher_response_ids,
        dtype=torch.long,
        device=device,
    )
    teacher_scores = _score_token_logps(
        teacher,
        teacher_input_ids,
        len(teacher_prompt_ids),
        len(teacher_response_ids),
    )
    reference_scores = _score_token_logps(
        reference,
        teacher_input_ids,
        len(teacher_prompt_ids),
        len(teacher_response_ids),
    )

    teacher_parent_scores = _sum_by_parent(teacher_scores, teacher_parent_indices)
    reference_parent_scores = _sum_by_parent(reference_scores, teacher_parent_indices)

    deltas = torch.tensor(teacher_parent_scores) - torch.tensor(reference_parent_scores)
    mu = 1.0 if side == "chosen" else -1.0
    weights = torch.exp((mu * deltas).clamp(-0.5, 1.5))
    weights = (torch.round(weights * 100) / 100).tolist()
    if weights:
        weights[-1] = 0.0

    return student_parent_indices, weights


def prepare_example(example, teacher_tokenizer, student_tokenizer, teacher, reference, device):
    result = dict(example)
    for side in ("chosen", "rejected"):
        parent_list, weights = _prepare_side(
            example,
            side,
            teacher_tokenizer,
            student_tokenizer,
            teacher,
            reference,
            device,
        )
        result[f"{side}_ctpd_parent_list"] = parent_list
        result[f"{side}_ctpd_weight"] = weights
    return result


def load_split(path, split):
    try:
        return load_dataset(path, split=split)
    except Exception:
        return load_from_disk(os.path.join(path, split))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--student-tokenizer", required=True)
    parser.add_argument("--save-to", required=True)
    parser.add_argument("--splits", nargs="+", default=["train_prefs", "test_prefs"])
    args = parser.parse_args()

    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    student_tokenizer = AutoTokenizer.from_pretrained(args.student_tokenizer)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    reference = AutoModelForCausalLM.from_pretrained(
        args.reference,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    device = next(teacher.parameters()).device

    output = DatasetDict()
    for split in args.splits:
        dataset = load_split(args.dataset, split)
        output_split = "train" if "train" in split else "test"
        output[output_split] = dataset.map(
            lambda row: prepare_example(
                row,
                teacher_tokenizer,
                student_tokenizer,
                teacher,
                reference,
                device,
            ),
            desc=f"Preparing CTPD {split}",
        )

    Path(args.save_to).parent.mkdir(parents=True, exist_ok=True)
    output.save_to_disk(args.save_to)


if __name__ == "__main__":
    main()
