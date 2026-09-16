import argparse
import json
import os
import random
from typing import List, Dict

import torch
from datasets import load_from_disk, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def is_share_gpt_format(inputs: List):
    """Check if input follows the ShareGPT format."""
    return isinstance(inputs[0], List) and isinstance(inputs[0][0], dict) and (
            "role" in inputs[0][0] and "content" in inputs[0][0]
    )


def is_prompts(inputs: List):
    """Check if input is a list of prompts."""
    return isinstance(inputs[0], str)


def transform_prompt_to_sharegpt(inputs: List[str]):
    """Convert plain prompts to ShareGPT format."""
    return [[{"role": "user", "content": prompt}] for prompt in inputs]


def load_input(data_path: str, prompt_key: str, dataset_split=None):
    """Load dataset from JSON, JSONL, or Hugging Face dataset format."""
    if data_path.endswith("json"):
        dataset = json.load(open(data_path, "r"))
        dataset = [row[prompt_key] for row in dataset] if prompt_key else dataset
    elif data_path.endswith("jsonl"):
        dataset = [json.loads(row)[prompt_key] for row in open(data_path, "r")] if prompt_key else [json.loads(row) for
                                                                                                    row in
                                                                                                    open(data_path,
                                                                                                         "r")]
    else:
        try:
            dataset = load_from_disk(data_path)
        except (ValueError, FileNotFoundError):
            dataset = load_dataset(data_path)
        if dataset_split:
            dataset = dataset[dataset_split]
        dataset = dataset[prompt_key] if prompt_key else dataset
    return dataset if is_share_gpt_format(dataset) else transform_prompt_to_sharegpt(dataset)


def remove_last_assistant_response(inputs: List[List[Dict[str, str]]]):
    """Remove the last response if its role is 'assistant'."""
    return [conv[:-1] if conv and conv[-1]['role'] == "assistant" else conv for conv in tqdm(inputs)]


def str2bool(v):
    """Convert string to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, required=True,
                        help="Dataset path (JSON, JSONL, or Hugging Face format)")
    parser.add_argument('--dataset_split', type=str, required=False)
    parser.add_argument('-p', '--prompt_key', type=str, default=None, help="Key for prompts in dataset")
    parser.add_argument('-o', '--out_dir', type=str, default=None, help="Output directory")
    parser.add_argument('-m', '--model', type=str, required=True, help="Path to the LLM model")
    parser.add_argument('-s', '--max_samples', type=int, default=None, help="Maximum number of samples to process")
    parser.add_argument('-l', '--max_length', type=int, default=2048, help="Maximum generation length")
    parser.add_argument('--apply_template', type=str2bool, default=True)
    parser.add_argument('--temperature', type=float, default=0, help="Sampling temperature")
    parser.add_argument('--top_p', type=float, default=1, help="Top-p sampling")
    parser.add_argument('--top_k', type=int, default=-1, help="Top-k sampling")
    parser.add_argument('--num_outputs', type=int, default=1, help="Number of outputs per prompt")
    args = parser.parse_args()

    raw_dataset = load_input(args.data, args.prompt_key, args.dataset_split)
    if args.max_samples:
        raw_dataset = random.sample(raw_dataset, min(len(raw_dataset), args.max_samples))
    prompts = remove_last_assistant_response(raw_dataset)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    llm = LLM(args.model, dtype="bfloat16", tensor_parallel_size=torch.cuda.device_count(), max_seq_len_to_capture=2048)
    sampling_params = SamplingParams(n=args.num_outputs, top_k=args.top_k, temperature=args.temperature,
                                     top_p=args.top_p, max_tokens=args.max_length)

    prompts_with_format = [tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in
                           prompts]
    outputs = llm.generate(prompts_with_format, sampling_params)
    outputs = [[o.text for o in output.outputs] if output.outputs else [] for output in outputs]

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        save_path = os.path.join(args.out_dir, f"{os.path.basename(args.data)}-{args.dataset_split}.jsonl")
        with open(save_path, "w", encoding="utf-8") as fp:
            for prompt, response in zip(prompts_with_format, outputs):
                fp.write(json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False) + "\n")
        print(f"Saved to: {save_path}")
