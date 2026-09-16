import argparse
import json
import os
from tqdm import tqdm
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    pipeline
)

def generate_multiple_answers_batch(
    model,
    tokenizer,
    prompts,
    num_return_sequences=2,
    num_beams=2,
    do_sample=False,
    temperature=1.0,
    top_p=1.0,
    max_new_tokens=128,
):
    inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(model.device)
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )

    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=gen_config)

    decoded = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
    grouped = [
        decoded[i * num_return_sequences:(i + 1) * num_return_sequences]
        for i in range(len(prompts))
    ]
    return grouped

def get_reward_from_pipeline_batch(rm_pipe, rm_tokenizer, prompt, answers):
    inputs = []
    for answer in answers:
        chat = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer}
        ]
        input_text = rm_tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=False
        ).replace(rm_tokenizer.bos_token, "")
        inputs.append(input_text)

    outputs = rm_pipe(
        inputs,
        return_all_scores=True,
        function_to_apply="none",
        batch_size=len(inputs)
    )
    rewards = [float(out[0]["score"]) for out in outputs]
    return rewards

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="argilla/dpo-mix-7k")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_file", type=str, default="offline_generation.jsonl")
    parser.add_argument("--student_model_name_or_path", type=str, required=True)
    parser.add_argument("--num_return_sequences", type=int, default=2)
    parser.add_argument("--num_beams", type=int, default=2)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--reward_model_name_or_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset_name} / split={args.dataset_split}")
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    print("Loading student model...")
    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model_name_or_path)
    student_model = AutoModelForCausalLM.from_pretrained(args.student_model_name_or_path, device_map='auto')
    student_model.eval()

    print("Loading reward model...")
    rm_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_name_or_path)
    rm_pipe = pipeline(
        "sentiment-analysis",
        model=args.reward_model_name_or_path,
        tokenizer=rm_tokenizer,
        device_map="auto",
        model_kwargs={"torch_dtype": torch.bfloat16}
    )

    print(f"Start offline generation on {len(dataset)} samples...")
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    count = 0
    with open(args.output_file, "w", encoding="utf-8") as f_out:
        for i in tqdm(range(0, len(dataset), args.batch_size)):
            batch = dataset.select(range(i, min(i + args.batch_size, len(dataset))))  # ✅ 핵심 수정!

            prompts = []
            for example in batch:
                if "prompt" in example:
                    prompts.append(example["prompt"])
                elif "question" in example:
                    prompts.append(example["question"])
                elif "chosen" in example and isinstance(example["chosen"], list):
                    for message in example["chosen"]:
                        if message["role"] == "user":
                            prompts.append(message["content"])
                            break
                    else:
                        prompts.append("")
                else:
                    prompts.append("")

            all_answers = generate_multiple_answers_batch(
                model=student_model,
                tokenizer=student_tokenizer,
                prompts=prompts,
                num_return_sequences=args.num_return_sequences,
                num_beams=args.num_beams,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
            )

            for prompt, answers in zip(prompts, all_answers):
                if len(answers) < 2:
                    answers = answers + answers

                rewards = get_reward_from_pipeline_batch(rm_pipe, rm_tokenizer, prompt, answers)

                idx_sorted = sorted(range(len(rewards)), key=lambda i: rewards[i], reverse=True)
                chosen_idx, rejected_idx = idx_sorted[0], idx_sorted[1]

                out_data = {
                    "prompt": prompt,
                    "chosen": answers[chosen_idx],
                    "rejected": answers[rejected_idx],
                    "chosen_reward": rewards[chosen_idx],
                    "rejected_reward": rewards[rejected_idx],
                }
                f_out.write(json.dumps(out_data, ensure_ascii=False) + "\n")
                count += 1

    print(f"Done. Saved {count} samples to {args.output_file}")

if __name__ == "__main__":
    main()