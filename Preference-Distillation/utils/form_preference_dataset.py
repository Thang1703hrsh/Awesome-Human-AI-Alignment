import argparse
import json
from typing import List, Dict

from datasets import load_from_disk, load_dataset, DatasetDict
from datasets.exceptions import DatasetGenerationError


def load_responses_jsonl(jsonl_path: str):
    return [json.loads(row)["response"] for row in open(jsonl_path, "r").readlines()]


def add_response_at_end(messages: List[Dict[str, str]], responses: List[str]):
    rst = []
    for message, response in zip(messages, responses):
        rst.append(message[: -1] + [{"role": "assistant", "content": response[0]}])
    return rst


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original-dataset', required=True)
    parser.add_argument('--chosen-train', type=str, default=None, required=False)
    parser.add_argument('--chosen-test', type=str, default=None, required=False)
    parser.add_argument('--rejected-train', type=str, default=None, required=False)
    parser.add_argument('--rejected-test', type=str, default=None, required=False)
    parser.add_argument('--output-dir', type=str, default=None, required=False)
    args = parser.parse_args()

    try:
        raw_dataset: DatasetDict = load_dataset(args.original_dataset)
    except DatasetGenerationError:
        raw_dataset: DatasetDict = load_from_disk(args.original_dataset)

    # chosen-train
    if args.chosen_train is not None:
        chosen_train = add_response_at_end(raw_dataset["train"]["chosen"], load_responses_jsonl(args.chosen_train))
        raw_dataset['train'] = raw_dataset['train'].remove_columns(["chosen"])
        raw_dataset['train'] = raw_dataset['train'].add_column("chosen", chosen_train)

    # chosen-test
    if args.chosen_test is not None:
        chosen_test = add_response_at_end(raw_dataset["test"]["chosen"], load_responses_jsonl(args.chosen_test))
        raw_dataset['test'] = raw_dataset['test'].remove_columns(["chosen"])
        raw_dataset['test'] = raw_dataset['test'].add_column("chosen", chosen_test)

    # rejected-train
    if args.rejected_train is not None:
        rejected_train = add_response_at_end(raw_dataset["train"]["rejected"], load_responses_jsonl(args.rejected_train))
        raw_dataset['train'] = raw_dataset['train'].remove_columns(["rejected"])
        raw_dataset['train'] = raw_dataset['train'].add_column("rejected", rejected_train)

    # rejected-test
    if args.rejected_test is not None:
        rejected_test = add_response_at_end(raw_dataset["test"]["rejected"], load_responses_jsonl(args.rejected_test))
        raw_dataset['test'] = raw_dataset['test'].remove_columns(["rejected"])
        raw_dataset['test'] = raw_dataset['test'].add_column("rejected", rejected_test)

    # save to DatasetDict
    raw_dataset.save_to_disk(args.output_dir)
