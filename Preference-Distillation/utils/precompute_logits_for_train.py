import argparse
import json
import os
from datetime import timedelta
from functools import partial
from typing import List, Dict

import torch
from accelerate import InitProcessGroupKwargs, Accelerator
from datasets import load_from_disk, load_dataset, Dataset
from datasets.exceptions import DatasetGenerationError
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM

# Initialize Accelerator with a custom timeout for process groups
kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1080000))
accelerator = Accelerator(kwargs_handlers=[kwargs])


def custom_collate_fn(batch, pad_token_id):
    """
    Custom collate function to pad input sequences to the maximum length in the batch.

    Args:
        batch (List[Dict]): A list of samples from the dataset.
        pad_token_id (int): The token ID used for padding.

    Returns:
        Dict: A dictionary containing padded input_ids, attention_mask, labels, and indices.
    """
    # Determine the maximum sequence length in the batch
    max_length = max(len(item['input_ids']) for item in batch)

    return {
        "index": [item["index"] for item in batch],
        "input_ids": [
            item['input_ids'] + [pad_token_id] * (max_length - len(item['input_ids']))
            for item in batch
        ],
        "attention_mask": [
            item['attention_mask'] + [0] * (max_length - len(item['attention_mask']))
            for item in batch
        ],
        "labels": [
            item['labels'] + [-100] * (max_length - len(item['labels']))
            for item in batch
        ]
    }


def compress_probs(logits: torch.Tensor, top_k: int = 50, vocab_size: int = 32000) -> List[Dict[str, List[float]]]:
    """
    Compress logits into top-k probabilities and the sum of remaining probabilities.

    Args:
        logits (torch.Tensor): Logits output from the model.
        top_k (int, optional): Number of top probabilities to retain. Defaults to 50.
        vocab_size (int, optional): Size of the vocabulary. Defaults to 32000.

    Returns:
        List[Dict[str, List[float]]]: A list of dictionaries containing top-k indices, values, and remaining probability sum.
    """
    # Apply softmax to convert logits to probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Get top-k probabilities and their corresponding indices
    top_k_values, top_k_indices = torch.topk(probs, top_k, dim=-1)

    # Calculate the sum of the remaining probabilities
    remaining_probs_sum = 1 - top_k_values.sum(dim=-1)

    compressed_probs = []
    for i in range(logits.size(0)):
        compressed_probs.append({
            'indices': top_k_indices[i].tolist(),
            'values': top_k_values[i].tolist(),
            'remaining_probs_sum': remaining_probs_sum[i].item()
        })
    return compressed_probs


def encode_prompt(conversation: List[Dict[str, str]], tokenizer: PreTrainedTokenizer,
                  user_begin: str, user_end: str) -> List[int]:
    """
    Encode the conversation into input IDs using the tokenizer's chat template.

    Args:
        conversation (List[Dict[str, str]]): The conversation history.
        tokenizer (PreTrainedTokenizer): The tokenizer to encode the text.
        user_begin (str): The beginning token/string for user messages.
        user_end (str): The ending token/string for user messages.

    Returns:
        List[int]: The encoded input IDs for the prompt.
    """
    # Apply the chat template without tokenization and generation prompts
    text = tokenizer.apply_chat_template(
        conversation[:-1], tokenize=False, add_generation_prompt=False
    )
    # Remove the user_begin and user_end tokens from the text
    text = text[len(user_begin): -len(user_end)]
    # Encode the cleaned text
    return tokenizer.encode(text)


def preprocess_conversation(
        conversation: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        user_begin: str = "",
        user_end: str = "",
        assistant_begin: str = "",
        assistant_end: str = "",
) -> Dict[str, List[int]]:
    """
    Preprocess the conversation by encoding it and preparing input IDs, labels, and attention masks.

    Args:
        conversation (List[Dict[str, str]]): The conversation history.
        tokenizer (PreTrainedTokenizer): The tokenizer to encode the text.
        user_begin (str, optional): The beginning token/string for user messages. Defaults to "".
        user_end (str, optional): The ending token/string for user messages. Defaults to "".
        assistant_begin (str, optional): The beginning token/string for assistant messages. Defaults to "".
        assistant_end (str, optional): The ending token/string for assistant messages. Defaults to "".

    Returns:
        Dict[str, List[int]]: A dictionary containing input_ids, labels, and attention_mask.
    """
    # Encode the special tokens
    user_begin_encoded = tokenizer.encode(user_begin, add_special_tokens=False)
    user_end_encoded = tokenizer.encode(user_end, add_special_tokens=False)
    assistant_begin_encoded = tokenizer.encode(assistant_begin, add_special_tokens=False)
    assistant_end_encoded = tokenizer.encode(assistant_end, add_special_tokens=False)

    # Encode the conversation prompt
    prompt_encoded = encode_prompt(conversation, tokenizer, user_begin, user_end)
    # Encode the assistant's response
    response_encoded = tokenizer.encode(conversation[-1]['content'], add_special_tokens=False)

    # Combine the encoded parts to form the complete prompt with the assistant's beginning token
    prompt_with_template = user_begin_encoded + prompt_encoded + user_end_encoded + assistant_begin_encoded

    return {
        "input_ids": prompt_with_template + response_encoded,
        "labels": [-100] * len(prompt_with_template) + response_encoded,
        "attention_mask": [1] * len(prompt_with_template + response_encoded),
    }


class LogitsExtractor:
    """
    A class to extract logits from a pre-trained causal language model.
    """

    def __init__(self, load_from: str = None):
        """
        Initialize the LogitsExtractor with a specified model.

        Args:
            load_from (str, optional): Path or identifier of the model to load. Defaults to a predefined path.
        """
        self.device = "cuda"  # Use GPU for computation
        self.model = AutoModelForCausalLM.from_pretrained(
            load_from,
            #device_map=self.device,
            device_map = 'auto',
            trust_remote_code=True,
            torch_dtype=torch.float16
        )

    def __call__(self, model_input: Dict[str, List[int]]) -> List[Dict]:
        """
        Process a batch of inputs and extract compressed logits.

        Args:
            model_input (Dict[str, List[int]]): A batch of input data containing input_ids, labels, and attention_mask.

        Returns:
            List[Dict]: A list of dictionaries containing indices, compressed probabilities, and labels.
        """
        # Convert input data to tensors and move to the appropriate device
        input_ids = torch.tensor(model_input['input_ids'], dtype=torch.long, device=self.device)
        labels = torch.tensor(model_input['labels'], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(model_input['attention_mask'], dtype=torch.long, device=self.device)

        # Forward pass through the model
        output = self.model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            use_cache=False
        )

        # Extract logits and adjust labels for next-token prediction
        logits = output.logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()

        compressed_output = []
        for i in range(logits.size(0)):
            # Filter out positions where labels are -100 (ignored)
            valid_logits = logits[i][labels[i] != -100]
            valid_labels = labels[i][labels[i] != -100]

            compressed_output.append({
                "index": model_input["index"][i],
                "compressed_probs": compress_probs(valid_logits),
                "labels": valid_labels.cpu().numpy().tolist(),
            })
        return compressed_output


class ConversationDataset(TorchDataset):
    """
    A custom dataset class for handling conversation data.
    """

    def __init__(self, dataset: List[Dict], conversation_key: str, tokenizer: PreTrainedTokenizer,
                 user_begin: str, user_end: str, assistant_begin: str, assistant_end: str):
        """
        Initialize the ConversationDataset.

        Args:
            dataset (List[Dict]): The dataset containing conversation data.
            conversation_key (str): The key to access conversation data within each sample.
            tokenizer (PreTrainedTokenizer): The tokenizer for encoding conversations.
            user_begin (str): Beginning token/string for user messages.
            user_end (str): Ending token/string for user messages.
            assistant_begin (str): Beginning token/string for assistant messages.
            assistant_end (str): Ending token/string for assistant messages.
        """
        # Sort the dataset based on the length of concatenated conversation content
        self.dataset = sorted(
            dataset,
            key=lambda row: len("".join([turn['content'] for turn in row[conversation_key]]))
        )
        self.conversation_key = conversation_key
        self.tokenizer = tokenizer
        self.user_begin = user_begin
        self.user_end = user_end
        self.assistant_begin = assistant_begin
        self.assistant_end = assistant_end

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        """
        Retrieve and preprocess a sample by index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Dict[str, List[int]]: The preprocessed conversation data.
        """
        row = self.dataset[idx]
        preprocessed_conversation = preprocess_conversation(
            conversation=row[self.conversation_key],
            tokenizer=self.tokenizer,
            user_begin=self.user_begin,
            user_end=self.user_end,
            assistant_begin=self.assistant_begin,
            assistant_end=self.assistant_end
        )
        # Assign the index for later reference
        preprocessed_conversation['index'] = row['index']
        return preprocessed_conversation


class DynamicDataLoader:
    """
    A dynamic data loader that creates batches based on maximum token constraints.
    """

    def __init__(self, dataset: "ConversationDataset", max_tokens_per_batch: int,
                 collate_fn, max_batch_size: int):
        """
        Initialize the DynamicDataLoader.

        Args:
            dataset (ConversationDataset): The dataset to load data from.
            max_tokens_per_batch (int): Maximum number of tokens allowed per batch.
            collate_fn (callable): Function to collate individual samples into a batch.
            max_batch_size (int): Maximum number of samples allowed per batch.
        """
        self.dataset = dataset
        self.max_tokens_per_batch = max_tokens_per_batch
        self.collate_fn = collate_fn
        self.max_batch_size = max_batch_size

    def __iter__(self):
        """
        Iterate over the dataset and yield batches that respect the token and batch size constraints.

        Yields:
            Dict: A batch of data processed by the collate function.
        """
        num_accumulated_tokens = 0
        accumulated_batch: List[Dict] = []

        # Iterate through the dataset with a progress bar
        for i in tqdm(range(len(self.dataset)), desc="Loading data"):
            sample = self.dataset[i]
            length = len(sample['input_ids'])  # Number of tokens in the sample

            # If a single sample exceeds the maximum token limit, yield a trimmed version of it
            if num_accumulated_tokens == 0 and length > self.max_tokens_per_batch:
                trimmed_sample = {
                    k: v[:self.max_tokens_per_batch * 2] for k, v in sample.items() if isinstance(v, List)
                }
                trimmed_sample["index"] = sample["index"]
                yield self.collate_fn([trimmed_sample])
                continue  # Skip to the next sample

            # Check if adding the current sample would exceed token or batch size limits
            if (num_accumulated_tokens + length > self.max_tokens_per_batch or
                    len(accumulated_batch) >= self.max_batch_size):
                # Yield the accumulated batch
                yield self.collate_fn(accumulated_batch)
                # Reset counters and start a new batch with the current sample
                num_accumulated_tokens = length
                accumulated_batch = [sample]
            else:
                # Accumulate the current sample into the batch
                accumulated_batch.append(sample)
                num_accumulated_tokens += length

        # Yield any remaining samples as the last batch
        if accumulated_batch:
            yield self.collate_fn(accumulated_batch)


def main():
    """Main function to execute the data processing and logits extraction."""
    parser = argparse.ArgumentParser(description="Process conversations and extract logits.")
    parser.add_argument('--data', type=str, required=True,
                        help="Path to the dataset (JSON or Hugging Face dataset path).")
    parser.add_argument("--max-tokens-per-batch", type=int, default=2048,
                        help="Maximum number of tokens per batch.")
    parser.add_argument("--max-batch-size", type=int, default=1,
                        help="Maximum number of samples per batch.")
    parser.add_argument('--split', type=str, default="train",
                        help="Dataset split to use (e.g., train, validation).")
    parser.add_argument('--debug-mode', type=int, default=None,
                        help="Enable debug mode with a limited number of samples.")
    parser.add_argument('--model', type=str, required=True,
                        help="Model name or path for logits extraction.")
    parser.add_argument('--pad-token-id', type=int, default=None,
                        help="Token ID used for padding. If not set, the tokenizer's pad_token_id is used.")
    parser.add_argument('--conversation-key', type=str, required=True,
                        help="Key to access conversation data within the dataset.")
    parser.add_argument('--save-to', type=str, required=True,
                        help="Directory to save the processed results.")
    parser.add_argument('--user-begin', default="", type=lambda s: s.replace('\\n', '\n'),
                        help="String indicating the beginning of user messages.")
    parser.add_argument('--user-end', default="", type=lambda s: s.replace('\\n', '\n'),
                        help="String indicating the end of user messages.")
    parser.add_argument('--assistant-begin', default="", type=lambda s: s.replace('\\n', '\n'),
                        help="String indicating the beginning of assistant messages.")
    parser.add_argument('--assistant-end', default="", type=lambda s: s.replace('\\n', '\n'),
                        help="String indicating the end of assistant messages.")
    args = parser.parse_args()

    try:
        dataset: Dataset = load_dataset(args.data, split=args.split)
        if len(dataset) < 10:
            raise ValueError
    except (DatasetGenerationError, ValueError):
        dataset_path = os.path.join(args.data, args.split)
        dataset: Dataset = load_from_disk(dataset_path)

    if args.debug_mode is not None:
        dataset = dataset.select(range(args.debug_mode))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pad_token_id: int = args.pad_token_id if args.pad_token_id is not None else tokenizer.pad_token_id
    logits_extractor = LogitsExtractor(load_from=args.model)

    dataset_list = [{
        'index': row['index'] if 'index' in row else i,
        args.conversation_key: row[args.conversation_key]
    } for i, row in enumerate(dataset)]

    with accelerator.split_between_processes(dataset_list) as dataset_split:
        conversation_dataset = ConversationDataset(
            dataset=dataset_split,
            conversation_key=args.conversation_key,
            tokenizer=tokenizer,
            user_begin=args.user_begin,
            user_end=args.user_end,
            assistant_begin=args.assistant_begin,
            assistant_end=args.assistant_end
        )

        dataloader = DynamicDataLoader(
            dataset=conversation_dataset,
            max_tokens_per_batch=args.max_tokens_per_batch,
            collate_fn=partial(custom_collate_fn, pad_token_id=pad_token_id),
            max_batch_size=args.max_batch_size
        )

        os.makedirs(args.save_to, exist_ok=True)
        output_file = os.path.join(args.save_to, f"results_rank_{accelerator.process_index}.jsonl")

        with open(output_file, 'w') as f:
            for batch in tqdm(dataloader, desc="Processing batches"):
                with torch.no_grad():
                    output = logits_extractor(batch)
                    for o in output:
                        f.write(json.dumps(o) + '\n')
                torch.cuda.empty_cache()
                accelerator.free_memory()

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        import glob
        from collections import defaultdict

        all_results = []
        for file in glob.glob(os.path.join(args.save_to, "results_rank_*.jsonl")):
            with open(file, 'r') as f:
                for line in f:
                    all_results.append(json.loads(line))

        # Group results by index
        grouped_results = defaultdict(lambda: {"compressed_probs": [], "labels": []})
        for res in all_results:
            idx = res["index"]
            grouped_results[idx]["compressed_probs"].extend(res["compressed_probs"])
            grouped_results[idx]["labels"].extend(res["labels"])

        # Restore order based on dataset index
        compressed_probs_list = []
        label_list = []

        for i in range(len(dataset)):
            res = grouped_results.get(i)
            if res is None:
                raise ValueError(f"Missing result for index {i}")
            compressed_probs_list.append(res["compressed_probs"])
            label_list.append(res["labels"])

        # Final check
        if len(compressed_probs_list) != len(dataset):
            raise ValueError(f"Final result length mismatch: {len(compressed_probs_list)} vs {len(dataset)}")

        dataset = dataset.add_column(f"{args.conversation_key}_compressed_probs", compressed_probs_list)
        dataset = dataset.add_column(f"{args.conversation_key}_labels", label_list)
        dataset.save_to_disk(args.save_to)

if __name__ == '__main__':
    main()
