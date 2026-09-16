import argparse
from datasets import load_from_disk, Dataset, load_dataset, DatasetDict
from datasets.exceptions import DatasetGenerationError
from tqdm import tqdm
import os


def merge_logits(chosen_teacher_logits, rejected_teacher_logits, split, output_dataset: Dataset):
    chosen_logits = []
    rejected_logits = []
    chosen_labels = []
    rejected_labels = []
    count = 0

    # Create a pool of worker processes to process data in parallel
    for row_chosen, row_rejected in tqdm(zip(chosen_teacher_logits[split], rejected_teacher_logits[split]), desc=f"Processing {split}"):
        count = count + 1
        chosen_logits.append(row_chosen["chosen_compressed_probs"])
        chosen_labels.append(row_chosen["chosen_labels"])
        rejected_logits.append(row_rejected["rejected_compressed_probs"])
        rejected_labels.append(row_rejected["rejected_labels"])
    output_dataset = output_dataset.add_column("chosen_compressed_probs", chosen_logits)
    output_dataset = output_dataset.add_column("rejected_compressed_probs", rejected_logits)
    output_dataset = output_dataset.add_column("chosen_labels", chosen_labels)
    output_dataset = output_dataset.add_column("rejected_labels", rejected_labels)
    return output_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dataset-dict', type=str)
    parser.add_argument('--teacher-chosen-logp-train', type=str)
    parser.add_argument('--teacher-rejected-logp-train', type=str)
    parser.add_argument('--teacher-chosen-logp-test', type=str)
    parser.add_argument('--teacher-rejected-logp-test', type=str)
    parser.add_argument('--save-to', type=str)
    args = parser.parse_args()

    # Load datasets
    chosen_teacher_logp = {
        "train": load_from_disk(args.teacher_chosen_logp_train).to_iterable_dataset(),
        "test": load_from_disk(args.teacher_chosen_logp_test).to_iterable_dataset()}
    rejected_teacher_logp = {
        "train": load_from_disk(args.teacher_rejected_logp_train).to_iterable_dataset(),
        "test": load_from_disk(args.teacher_rejected_logp_test).to_iterable_dataset()}

    # Initialize the result dataset dictionary
    try:
        rst: DatasetDict = load_dataset(args.input_dataset_dict)
    except DatasetGenerationError:
        dataset_path = os.path.join(args.input_dataset_dict)
        rst: DatasetDict = load_from_disk(dataset_path)

    # Process both train and test splits in parallel
    source_splits = {"train": "train_prefs", "test": "test_prefs"}
    for output_split, input_split in source_splits.items():
        rst[output_split] = merge_logits(
            chosen_teacher_logp,
            rejected_teacher_logp,
            output_split,
            rst[input_split],
        )

    for split in source_splits.values():
        if split in rst:
            del rst[split]

    # Save the result
    rst.save_to_disk(args.save_to)
