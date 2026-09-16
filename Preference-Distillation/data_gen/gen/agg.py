import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from Levenshtein import distance as levenshtein_distance
from multiprocessing import Pool, cpu_count


def compute_edit_distance_matrix(texts):
    n = len(texts)
    dist_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            dist_matrix[i][j] = levenshtein_distance(texts[i], texts[j])
            dist_matrix[j][i] = dist_matrix[i][j]
    return dist_matrix


def get_most_different_texts(texts, k):
    n = len(texts)
    if k > n:
        raise ValueError("k should be less than or equal to the number of texts")

    dist_matrix = compute_edit_distance_matrix(texts)
    selected_indices = [0]  # Start with the first text
    remaining_indices = set(range(1, n))

    while len(selected_indices) < k:
        max_min_dist = -1
        next_index = -1
        for i in remaining_indices:
            min_dist = min(dist_matrix[i][j] for j in selected_indices)
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                next_index = i
        selected_indices.append(next_index)
        remaining_indices.remove(next_index)

    return selected_indices


def process_sample(sample_data, args_n):
    prompt, all_texts = sample_data
    gen_text, gen_input = [], []

    for data in all_texts:
        gen_text.append(data["output"])
        gen_input.append(data["input"])

    if args_n != len(gen_text):
        gen_indices = get_most_different_texts(gen_text, args_n)
        gen_text = [gen_text[i] for i in gen_indices]
        gen_input = [gen_input[i] for i in gen_indices]

    if len(set(gen_text)) != args_n:
        return None

    return {
        "prompt": prompt,
        "input": gen_input,
        "responses": gen_text,
    }


# Create a helper function for the pool
def process_sample_helper(sample):
    return process_sample(sample[0], sample[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation_file_dir", type=str, help="Directory containing the generation files", default="datasets/gemma2_ultrafeedback")
    parser.add_argument('-n', type=int, default=2, help='Number of responses')
    args = parser.parse_args()

    # Load all data
    all_data = []
    for file_name in os.listdir(args.generation_file_dir):
        if file_name.startswith("output") and file_name.endswith(".json"):
            generation_file = os.path.join(args.generation_file_dir, file_name)
            with open(generation_file, 'r') as f:
                output_data = json.load(f)
                all_data.append(output_data)

    # Ensure all_data is loaded
    num_samples = len(all_data[0])
    all_res = []
    num_identical = 0

    # Prepare data for multiprocessing: Combine the prompt with all text data for each sample
    sample_data = [(all_data[0][i]["prompt"], [data[i] for data in all_data]) for i in range(num_samples)]

    # Create a multiprocessing pool using all CPU cores
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_sample_helper, [(sample, args.n) for sample in sample_data]), total=num_samples))

    # Filter out None values from results
    all_res = [res for res in results if res is not None]
    num_identical = num_samples - len(all_res)

    print(f"Filtered out {num_identical} samples with identical generated responses")

    # Save the results to a JSON file
    output_file_path = os.path.join(args.generation_file_dir, f'agg_outputs_n{args.n}.json')
    with open(output_file_path, 'w') as f:
        json.dump(all_res, f, indent=4)

    print(f"Processed outputs saved to {output_file_path}")


if __name__ == "__main__":
    main()