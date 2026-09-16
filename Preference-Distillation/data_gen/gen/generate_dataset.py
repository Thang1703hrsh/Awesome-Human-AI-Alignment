import json
import torch
from datasets import Dataset
from tqdm import tqdm
import argparse

def main(prob_file, prob_sl_file, output_dir, seed=42, test_size=0.02):
    # 读取文件
    with open(prob_file, 'r', encoding='utf-8') as f:
        prob_data = json.load(f)
    try:
        with open(prob_sl_file, 'r', encoding='utf-8') as f:
            prob_sl_data = json.load(f)
    except FileNotFoundError:
        prob_sl_data = [None for _ in prob_data]

    dataset = []
    for prob, prob_sl in tqdm(zip(prob_data, prob_sl_data)):
        example = prob.copy()
        if prob_sl is not None:
            for key, value in prob_sl.items():
                if key in example:
                    if example[key] != value:
                        raise ValueError(f"Conflict found for key '{key}': {example[key]} != {value}")
                example[key] = value
        else:
            example['logp'] = example['scores'].copy()
            example['avg_logp'] = example['scores'].copy()
        dataset.append(example)

    # 转换为HuggingFace Dataset格式
    dataset = Dataset.from_list(dataset)
    dataset = dataset.rename_columns(
        {
            "scores": "scores_mcq",
            "logp": "scores_logp",
            "avg_logp": "scores_avglogp",
        }
    )

    # 划分训练集和测试集
    dataset = dataset.train_test_split(test_size=test_size, seed=seed)

    # 保存处理好的数据集
    dataset.save_to_disk(output_dir)

if __name__ == "__main__":
    # 使用argparse处理命令行参数
    parser = argparse.ArgumentParser(description="Process and merge JSON data with Dataset")
    parser.add_argument('--prob_file', type=str, required=True, help="Path to the main probability file (prob.gemma.json)")
    parser.add_argument('--prob_sl_file', type=str, required=True, help="Path to the secondary probability file (prob.sl.json)")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the processed dataset")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for train-test split")
    parser.add_argument('--test_size', type=float, default=0.02, help="Test size fraction for train-test split")
    
    args = parser.parse_args()

    # 调用main函数
    main(args.prob_file, args.prob_sl_file, args.output_dir, args.seed, args.test_size)