import argparse
import os
from math import log
from multiprocessing import Pool, cpu_count

from datasets import load_from_disk, DatasetDict, Dataset, load_dataset
from datasets.exceptions import DatasetGenerationError
from tqdm import tqdm


def get_margin_logp_token(pa, pb):
    indices = set(pa['indices']) | set(pb['indices'])
    probs_a, probs_b = {i: max(min(pa['values']), 1e-7) for i in indices}, {i: max(min(pa['values']), 1e-7) for i in
                                                                            indices}
    for i, v in zip(pa['indices'], pa['values']):
        probs_a[i] = max(v, 1e-7)
    for i, v in zip(pb['indices'], pb['values']):
        probs_b[i] = max(v, 1e-7)
    log_probs_margin = {i: log(probs_a[i] / probs_b[i]) for i in pa['indices']}
    log_probs_margin = sorted(log_probs_margin.items(), reverse=True, key=lambda x: x[1])
    return {'indices': [i[0] for i in log_probs_margin], 'values': [i[1] for i in log_probs_margin]}


def add_chosen_column(row, chosen_data, key: str):
    # Assuming row is a dict containing data from new_dataset
    # You can access the corresponding chosen value by using row['index']
    # or by any other criteria that connects the two datasets
    chosen_value = chosen_data[row['index']]  # example, replace with appropriate logic
    row[key] = chosen_value[key]
    return row


def is_prefix(list1, list2):
    # Checks if one list is a prefix of the other. The function compares the shorter
    # list to the first few elements of the longer list.
    short_list, long_list = (list1, list2) if len(list1) <= len(list2) else (list2, list1)
    return short_list == long_list[:len(short_list)]


# Function to parallelize the processing of a dataset split
def process_split(args, dpo_teacher_logits, ref_teacher_logits, split, output_dataset: Dataset):
    tmp = []
    tag = []
    labels = []
    count = 0

    # Create a pool of worker processes to process data in parallel
    with Pool(cpu_count()) as pool:
        for row_dpo, row_ref in tqdm(zip(dpo_teacher_logits[split], ref_teacher_logits[split]),
                                     desc=f"Processing {split}"):
            count = count + 1
            tag.append(is_prefix(row_dpo[args.label_key], row_ref[args.label_key]))
            tmp.append(pool.starmap(get_margin_logp_token, zip(row_dpo[args.logits_key], row_ref[args.logits_key])))
            labels.append(row_dpo[args.label_key] if len(row_dpo[args.label_key]) >= len(row_ref[args.label_key]) else row_ref[args.label_key])
    output_dataset = output_dataset.add_column(args.output_key, tmp)
    output_dataset = output_dataset.add_column("is_valid", tag)
    output_dataset = output_dataset.add_column(args.label_key, labels)
    output_dataset = output_dataset.filter(lambda row: row['is_valid'])
    print(f"save {sum(tag)} / {count}")
    return output_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dataset-dict', type=str)
    parser.add_argument('--dpo-teacher-logp-train', type=str, default=None, required=True)
    parser.add_argument('--ref-teacher-logp-train', type=str, default=None, required=True)
    parser.add_argument('--dpo-teacher-logp-test', type=str, default=None)
    parser.add_argument('--ref-teacher-logp-test', type=str, default=None)
    parser.add_argument('--label-key', type=str)
    parser.add_argument('--teacher-chosen-dataset-dir', type=str, default=None)
    parser.add_argument('--save-to', type=str)
    parser.add_argument('--logits-key', type=str, default="rejected_compressed_probs")
    parser.add_argument('--output-key', type=str, default="rejected_margin_logp_every")
    args = parser.parse_args()

    # Load datasets
    dpo_teacher_logits = {
        "train": load_from_disk(args.dpo_teacher_logp_train).to_iterable_dataset()}
    ref_teacher_logits = {
        "train": load_from_disk(args.ref_teacher_logp_train).to_iterable_dataset()}

    if args.dpo_teacher_logp_test is not None:
        dpo_teacher_logits["test"] = load_from_disk(args.dpo_teacher_logp_test).to_iterable_dataset()
    if args.ref_teacher_logp_test is not None:
        ref_teacher_logits["test"] = load_from_disk(args.ref_teacher_logp_test).to_iterable_dataset()

    # Initialize the result dataset dictionary
    try:
        rst = load_dataset(args.input_dataset_dict)
    except (DatasetGenerationError, FileNotFoundError, ValueError):
        rst = load_from_disk(os.path.join(args.input_dataset_dict))

    # Process both train and test splits in parallel
    for split in dpo_teacher_logits.keys():
        new_dataset = process_split(args, dpo_teacher_logits, ref_teacher_logits, split, rst[split])
        rst[split] = new_dataset

    # Save the result
    rst.save_to_disk(args.save_to)
