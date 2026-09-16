import json
import os
import torch
import argparse
import numpy as np
from vllm import LLM, SamplingParams
from datasets import load_dataset, load_from_disk
from jinja2.exceptions import TemplateError


def apply_chat_template(tokenizer, usr_prompt):
    prompt = tokenizer.apply_chat_template(
        [
            {'role': 'user', 'content': usr_prompt}
        ],
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt

def generate_text(args):
    # Load dataset
    if args.local:
        dataset = load_from_disk(args.dataset)[args.dataset_split]
    else:
        dataset = load_dataset(args.dataset, split=args.dataset_split)
    outputs = []

    # Load tokenizer and model
    # if 'gemma' in args.model_name.lower():
    #     os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
    llm = LLM(model=args.model_name, seed=args.seed, tensor_parallel_size=torch.cuda.device_count())
    tokenizer = llm.get_tokenizer()

    prompts = [
        apply_chat_template(tokenizer, usr_prompt)
        for usr_prompt in dataset['prompt']
    ]
    sampling_params = SamplingParams(
        temperature=args.temperature, 
        top_p=args.top_p, 
        max_tokens=args.max_tokens, 
        seed=args.seed
    )
    chats = llm.generate(prompts, sampling_params)
    for i, chat in enumerate(chats):
        outputs.append({
            "prompt": dataset['prompt'][i],
            "input": prompts[i],
            "output": chat.outputs[0].text
        })

    # Ensure the output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Save outputs to JSON
    output_file = os.path.join(args.output_dir, f"output_{args.seed}.json")
    with open(output_file, 'w') as f:
        json.dump(outputs, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Distributed Text Generation")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--model_name", type=str, required=True, help="Model name from Hugging Face model hub")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name from Hugging Face datasets library")
    parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum length of generated text")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling top_p")
    parser.add_argument("--local", action="store_true", help="Run locally")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the generated outputs, e.g., './outputs/'")
    args = parser.parse_args()

    generate_text(args)

if __name__ == "__main__":
    main()