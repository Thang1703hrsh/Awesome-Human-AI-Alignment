#!/usr/bin/env python
# coding=utf-8
import os
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from itertools import combinations
from string import ascii_uppercase
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from data_gen.gen.utlis import get_template


def compare_top_k(dict_1, dict_2, k):
    sorted_dict_1 = sorted(dict_1.items(), key=lambda item: item[1], reverse=True)
    sorted_dict_2 = sorted(dict_2.items(), key=lambda item: item[1], reverse=True)

    top_k_dict_1 = [item[0] for item in sorted_dict_1[:k]]
    top_k_dict_2 = [item[0] for item in sorted_dict_2[:k]]

    return top_k_dict_1 == top_k_dict_2


def generate_option_caps(n) -> str:
    if n < 1 or n > 26:
        return "n should be within the range 1-26"

    result = []
    for i in range(1, n):
        result.append(f"{ascii_uppercase[i-1]}")
    result.append(f"or {ascii_uppercase[n-1]}")
    return f"({', '.join(result)})"


def truncate_and_add_ellipsis(text, max_tokens, tokenizer):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        truncated_text = tokenizer.decode(truncated_tokens)
        return truncated_text + "..."
    return text


def generate_options(strings, max_tokens, tokenizer):
    if len(strings) > 26:
        return "Number of strings exceeds alphabet range (1-26)"

    formatted_strings = []
    for i, s in enumerate(strings):
        # Truncate each response if it exceeds the maximum token count
        truncated_s = truncate_and_add_ellipsis(s, max_tokens, tokenizer)
        formatted_strings.append(f"{ascii_uppercase[i]}) \"\"\"\n{truncated_s}\n\"\"\"")
    return "\n".join(formatted_strings)


def stat_prob(logprob: dict, tokenizer: AutoTokenizer, options: list[str], epsilon=1e-8):
    probs = {opt: epsilon for opt in options}
    for k, v in logprob.items():
        token = tokenizer.decode(k)
        for opt in options:
            if opt in token and len(token) < 3:
                probs[opt] += np.exp(v.logprob)
    return probs


def shuffle_options(cands):
    indices = list(range(len(cands)))
    random.shuffle(indices)
    shuffled_cands = [cands[i] for i in indices]
    # origin index: shuffle index
    reverse_indices = {i: indices.index(i) for i in range(len(cands))}
    return shuffled_cands, reverse_indices


def main(model_name, temperature, input_file, output_file, num_options):
    CMB = list(combinations(range(num_options), 2))

    model = LLM(
        model_name,
        tensor_parallel_size=torch.cuda.device_count(),
        max_model_len=8192,
        gpu_memory_utilization=.95
    )
    tokenizer = model.get_tokenizer()
    template = get_template(tokenizer)

    # load data
    with open(input_file, encoding="utf-8") as f:
        data_sampling = json.load(f)

    data_mcq, tot_cands = [], []
    for row in tqdm(data_sampling):
        # get different responses
        cands = row['responses']
        tot_cands.append(cands)

        for i, j in CMB:
            mcq_cmb = template.safe_substitute(
                identifier_list_str=generate_option_caps(2),
                instruction=row['prompt'],
                candidate_response_list_str=generate_options([cands[i], cands[j]], 2500, tokenizer)
            )
            if not mcq_cmb.startswith(tokenizer.bos_token):
                mcq_cmb = tokenizer.bos_token + mcq_cmb
            data_mcq.append(mcq_cmb)

    # sampling
    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=temperature,
        logprobs=20)
    results = model.generate(data_mcq, sampling_params=sampling_params)

    # stat the prob as score
    result_scores = []
    for i in range(0, len(results), len(CMB)):
        ress = results[i:i+len(CMB)]
        cands = tot_cands[i//len(CMB)]
        row = data_sampling[i//len(CMB)]
        prob_pair = []
        for res in ress:
            try:
                _p = stat_prob(res.outputs[0].logprobs[0], tokenizer, options='AB')
            except IndexError:
                breakpoint()
            prob_pair.append([_p['A'], _p['B']])
        cands_prob = [1e-8] * len(cands)
        for cmb, probs in zip(CMB, prob_pair):
            for i, p in zip(cmb, probs):
                cands_prob[i] += p
        avg_probs = [p / (num_options-1) for p in cands_prob]
        avg_scores = np.log(avg_probs).tolist()
        result_scores.append(avg_scores)

    # save to file
    record, cnt = [], 0
    for i, (row, cands) in enumerate(zip(data_sampling, tot_cands)):
        record.append({
            'prompt': row['prompt'],
            'responses': cands,
            'scores': result_scores[cnt]
        })
        cnt += 1

    # output_filename = '.'.join(input_file.split('.')[:-1]) + f'_prob.json'
    # with open(output_filename, 'w', encoding="utf-8") as f:
    #     json.dump(record, f, indent=4)
    with open(output_file, 'w', encoding="utf-8") as f:
        json.dump(record, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LLM Log Probability Calculation")
    parser.add_argument('--model_name', type=str, required=True, help='Model name for the tokenizer and model')
    parser.add_argument('--temperature', type=float, required=True, help='Temperature for sampling')
    parser.add_argument('--num_options', type=int, default=2, help='Number of options for multiple choice questions')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSON file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the input JSON file')
    args = parser.parse_args()
    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    main(args.model_name, args.temperature, args.input_file, args.output_file, args.num_options)
