#!/usr/bin/env python
# coding=utf-8
from typing import Dict

import os
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from accelerate import PartialState
from vllm import LLM, SamplingParams
from datasets import Dataset


def process(row, tokenizer):
    responses_chat = []
    for res in row['responses']:
        responses_chat.append(
            [
                {
                    "role": "user",
                    "content": row["prompt"]
                },
                {
                    "role": "assistant",
                    "content": res
                }
            ]
        )

    prompt_chat = tokenizer.apply_chat_template(responses_chat[0][:-1], tokenize=False, add_generation_prompt=True)
    responses_chat = [tokenizer.apply_chat_template(item, tokenize=False) for item in responses_chat]
    for i, res in enumerate(responses_chat):
        assert res[:len(prompt_chat)] == prompt_chat
        responses_chat[i] = res[len(prompt_chat):]

    row['prompt'] = prompt_chat
    row['responses'] = responses_chat
    return row

def build_tokenized_answer(tokenizer, prompt, answer):
    """
    Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`.
    It does ensure `enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]`.
    Reference:
        https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
    """

    full_tokenized = tokenizer(prompt + answer, add_special_tokens=False)
    prompt_input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
    answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids) :]

    # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
    full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])

    # Prepare input tokens for token by token comparison
    full_input_ids = np.array(full_tokenized["input_ids"])

    if len(full_input_ids) != len(full_concat_input_ids):
        raise ValueError("Prompt input ids and answer input ids should have the same length.")

    # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
    # can be merged together when tokenizing prompt+answer. This could result
    # on the last token from the prompt being different when tokenized on its own
    # vs when done as prompt+answer.
    response_token_ids_start_idx = len(prompt_input_ids)

    # If tokenized prompt is different than both prompt+answer, then it means the
    # last token has changed due to merging.
    if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
        response_token_ids_start_idx -= 1

    prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
    prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

    if len(prompt_input_ids) != len(prompt_attention_mask):
        raise ValueError("Prompt input ids and attention mask should have the same length.")

    answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
    answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

    return dict(
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
        input_ids=answer_input_ids,
        attention_mask=answer_attention_mask,
    )

def tokenize_row(
    feature,
    tokenizer,
    truncation_mode="keep_end",
    max_prompt_length=1800,
    max_length=2048,
    label_pad_token_id=-100
) -> Dict:
    """Tokenize a single row from a PD specific dataset.

    At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
    in case the prompt + chosen or prompt + rejected responses is/are too long. First
        we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

    We also create the labels for the chosen/rejected responses, which are of length equal to
        the sum of the length of the prompt and the chosen/rejected response, with
        label_pad_token_id  for the prompt tokens.
    """
    batch = {}
    prompt = feature["prompt"]
    responses = feature['responses']

    # Check issues below for more details
    #  1. https://github.com/huggingface/trl/issues/907
    #  2. https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
    #  3. https://github.com/LianjiaTech/BELLE/issues/337

    if not isinstance(prompt, str):
        raise ValueError(f"prompt should be an str but got {type(prompt)}")
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)
    prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

    responses_tokens = []
    for res in responses:
        if not isinstance(res, str):
            raise ValueError(f"rejected should be an str but got {type(res)}")
        responses_tokens.append(build_tokenized_answer(tokenizer, prompt, res))
    # Last prompt token might get merged by tokenizer and
    # it should not be included for generation if that happens
    prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])

    prompt_len_input_ids = min([len(res["prompt_input_ids"]) for res in responses_tokens])

    for k, v in prompt_tokens.items():
        prompt_tokens[k] = v[:prompt_len_input_ids]

    # Make sure prompts only have one different token at most an
    # and length only differs by 1 at most
    num_diff_tokens = len(set([tuple(res["prompt_input_ids"]) for res in responses_tokens]))
    num_diff_len = len(set([len(res["prompt_input_ids"]) for res in responses_tokens]))
    if num_diff_tokens > 1 or num_diff_len > 1:
        raise ValueError(
            "Chosen and rejected prompt_input_ids might only differ on the "
            "last token due to tokenizer merge ops."
        )
    
    # add BOS token to head of prompt. Avoid adding if it's already there
    # prompt_tokens, chosen_tokens, rejected_tokens = add_bos_token_if_needed(
    #     self.tokenizer.bos_token_id,
    #     prompt_len_input_ids,
    #     prompt_tokens,
    #     chosen_prompt_len_input_ids,
    #     chosen_tokens,
    #     rejected_prompt_len_input_ids,
    #     rejected_tokens,
    # )

    # add EOS token to end of answer. Avoid adding if it's already there
    # chosen_tokens, rejected_tokens = add_eos_token_if_needed(
    #     self.tokenizer.eos_token_id, chosen_tokens, rejected_tokens
    # )

    # longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))
    longer_response_length = max(len(res["input_ids"]) for res in responses_tokens)

    for answer_tokens in [*responses_tokens, prompt_tokens]:
        if len(answer_tokens["prompt_input_ids"]) + longer_response_length > max_length:
            if truncation_mode == "keep_start":
                for k in ["prompt_input_ids", "prompt_attention_mask"]:
                    answer_tokens[k] = answer_tokens[k][: max_prompt_length]
            elif truncation_mode == "keep_end":
                for k in ["prompt_input_ids", "prompt_attention_mask"]:
                    answer_tokens[k] = answer_tokens[k][-max_prompt_length :]
            else:
                raise ValueError(f"Unknown truncation mode: {truncation_mode}")

    # if that's still too long, truncate the response
    for answer_tokens in responses_tokens:
        if len(answer_tokens["prompt_input_ids"]) + longer_response_length > max_length:
            for k in ["input_ids", "attention_mask"]:
                answer_tokens[k] = answer_tokens[k][: max_length - max_prompt_length]

    # Create labels
    responses_sequence_tokens = []
    for res in responses_tokens:
        res_seq_tokens = {
            k: res[f"prompt_{k}"] + res[k] for k in ["input_ids", "attention_mask"]
        }
        res_seq_tokens["labels"] = res_seq_tokens["input_ids"][:]
        res_seq_tokens["labels"][: len(res["prompt_input_ids"])] = [
            label_pad_token_id
        ] * len(res["prompt_input_ids"])
        responses_sequence_tokens.append(res_seq_tokens)

    batch = {
        **prompt_tokens,
        'responses_input_ids': [],
        'responses_attention_mask': [],
        'responses_labels': [],
    }
    for res in responses_sequence_tokens:
        for k, v in res.items():
            batch[f'responses_{k}'].append(v)

    return batch
