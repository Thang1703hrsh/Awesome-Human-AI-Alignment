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

from data_gen.gen.common import process, tokenize_row


def json_map(row, tokenizer):
    chat_dict = [
        {
            "role": "user",
            "content": row['prompt']
        },
        {
            "role": "assistant",
            "content": row['output']
        }
    ]
    chat = tokenizer.apply_chat_template(chat_dict, tokenize=False)
    prompt_input = tokenizer.apply_chat_template(chat_dict[:1], tokenize=False, add_generation_prompt=True)
    assert chat[:len(prompt_input)] == prompt_input
    output = chat[len(prompt_input):]
    
    return {
        "input": prompt_input,
        "output": output
    }


def main(model_name, temperature, input_file, output_file):
    # if 'gemma' in model_name:
    #     os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
    model = LLM(
        model_name,
        tensor_parallel_size=torch.cuda.device_count(),
        enable_chunked_prefill=True,
        max_model_len=2500,
        gpu_memory_utilization=.6
    )
    tokenizer = model.get_tokenizer()

    # load data
    with open(input_file, encoding="utf-8") as f:
        data_sampling = json.load(f)

    dataset = Dataset.from_list(data_sampling)

    with PartialState().local_main_process_first():
        dataset = dataset.map(
            process,
            fn_kwargs={
                "tokenizer": tokenizer,
                # "pairwise": training_args.loss_type == "simpo"
            },
            num_proc=4,
        )
        # tokenize the dataset
        dataset = dataset.map(
            tokenize_row,
            fn_kwargs={
                "tokenizer": tokenizer,
            },
            num_proc=4,
        )

    sampling_params = SamplingParams(
        max_tokens=1,
        prompt_logprobs=1,
        temperature=temperature)
    data_prob = []
    for i, row in tqdm(enumerate(data_sampling), total=len(data_sampling), desc="Processing rows"):
        results = model.generate(
            prompt_token_ids=dataset[i]['responses_input_ids'],
            sampling_params=sampling_params,
            use_tqdm=False)
        row_new = {**row, 'logp': [], 'avg_logp': []}
        for j in range(len(results)):
            logps = [logp[label].logprob for logp, label in zip(results[j].prompt_logprobs[1:], dataset[i]['responses_labels'][j][1:]) if label != -100]
            logps_sum = sum(logps)
            logps_avg = logps_sum / len(logps)
            row_new['logp'].append(logps_sum)
            row_new['avg_logp'].append(logps_avg)
        data_prob.append(row_new)

    # sampling
    # # stat the prob as score
    # result_scores = []
    # for i in range(0, len(results), len(CMB)):
    #     ress = results[i:i+len(CMB)]
    #     cands = tot_cands[i//len(CMB)]
    #     row = data_sampling[i//len(CMB)]
    #     prob_pair = []
    #     for res in ress:
    #         _p = stat_prob(res.outputs[0].logprobs[0], tokenizer, options='AB')
    #         prob_pair.append([_p['A'], _p['B']])
    #     cands_prob = [1e-8] * len(cands)
    #     for cmb, probs in zip(CMB, prob_pair):
    #         for i, p in zip(cmb, probs):
    #             cands_prob[i] += p
    #     avg_probs = [p / (num_options-1) for p in cands_prob]
    #     avg_scores = np.log(avg_probs).tolist()
    #     result_scores.append(avg_scores)

    # # save to file
    # record, cnt = [], 0
    # for i, (row, cands) in enumerate(zip(data_sampling, tot_cands)):
    #     record.append({
    #         'prompt': row['prompt'],
    #         'responses': cands,
    #         'scores': result_scores[cnt]
    #     })
    #     cnt += 1

    # output_filename = '.'.join(input_file.split('.')[:-1]) + f'_prob_sl.json'
    with open(output_file, 'w', encoding="utf-8") as f:
        json.dump(data_prob, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LLM Log Probability Calculation")
    parser.add_argument('--model_name', type=str, required=True, help='Model name for the tokenizer and model')
    parser.add_argument('--temperature', type=float, required=True, help='Temperature for sampling')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSON file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the input JSON file')
    args = parser.parse_args()
    main(args.model_name, args.temperature, args.input_file, args.output_file)
