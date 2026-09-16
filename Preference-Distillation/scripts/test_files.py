import sys
sys.path.append("/home/minchan.kwon/ADPA")
from alignment import DataArguments, get_tokenizer
from typing import Any, Callable, Literal, Optional, Union, List
from utils.compress_logits import load_input_and_target_probs_fast
from transformers import AutoTokenizer
import random
import inspect
import logging
import os
import random
import warnings
from collections import defaultdict
from contextlib import nullcontext, contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Union, List

import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F
import transformers
from accelerate import PartialState
from accelerate.utils import is_deepspeed_available
from accelerate.utils import tqdm
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from datasets.exceptions import DatasetGenerationError
from packaging import version
from peft import PeftConfig
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, set_seed, TrainingArguments
from transformers import (
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    is_wandb_available,
)
from transformers.data.data_collator import DataCollatorMixin
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from transformers.utils import is_peft_available
from trl import FDivergenceConstants, FDivergenceType, create_reference_model, SyncRefModelCallback
from trl.models import PreTrainedModelWrapper
from trl.trainer.utils import (
    cap_exp,
    disable_dropout_in_model,
    pad,
    pad_to_length,
    peft_module_casting_to_bf16, RunningMoments,
)

from alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    get_checkpoint,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)
from alignment.data import maybe_insert_system_message, is_openai_format
from utils.compress_logits import load_input_and_target_probs_fast, load_input_and_target_probs_soft_kl,load_input_and_target_probs_filtered_soft_kl
from utils.compress_logits import soft_margin_ce_loss

from typing import Optional, List, Union, Dict, Any
def mix_datasets(
        dataset_mixer: dict,
        splits: Optional[List[str]] = None,
        configs: Optional[List[str]] = None,
        columns_to_keep: Optional[List[str]] = None,
        shuffle=True,
) -> DatasetDict:
    """
    Loads and mixes datasets according to proportions specified in `dataset_mixer`.

    Args:
        dataset_mixer (`dict`):
            Dictionary containing the dataset names and their training proportions. By default, all test proportions are 1.
        splits (Optional[List[str]], *optional*, defaults to `None`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        configs (Optional[List[str]], *optional*, defaults to `None`):
            List of dataset config names. If given must be the same length as 'dataset_mixer' keys.
        columns_to_keep (Optional[List[str]], *optional*, defaults to `None`):
            Column names to keep in the dataset. Useful in the datamixer to avoid schema conflicts,
            and for cpt this should be (at least) the text column.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.
    """
    splits = ["train", "test"] if splits is None else splits
    configs = [None] * len(dataset_mixer) if not configs else configs
    columns_to_keep = [] if columns_to_keep is None else columns_to_keep

    if configs is not None and len(configs) != len(dataset_mixer):
        raise ValueError("The number of given dataset config names must be the same as the given number of datasets.")

    raw_datasets = DatasetDict()
    raw_train_datasets = []
    raw_val_datasets = []
    fracs = []
    for (ds, frac), ds_config in zip(dataset_mixer.items(), configs):
        fracs.append(frac)
        for split in splits:
            try:
                # Try first if dataset on a Hub repo
                dataset = load_dataset(ds, ds_config, split=split)
                if len(dataset) < 10:
                    raise ValueError
            except (DatasetGenerationError, ValueError):
                # If not, check local dataset
                dataset_path = os.path.join(ds, split)
                if not os.path.exists(dataset_path):
                    continue
                dataset = load_from_disk(dataset_path)

            # Rename
            rename_dict = {}
            if "teacher_chosen_probs" in columns_to_keep and "chosen_compressed_probs" in dataset.column_names:
                rename_dict["chosen_compressed_probs"] = "teacher_chosen_probs"
            if "teacher_rejected_probs" in columns_to_keep and "rejected_compressed_probs" in dataset.column_names:
                rename_dict["rejected_compressed_probs"] = "teacher_rejected_probs"
            dataset = dataset.rename_columns(rename_dict)

            # Remove redundant columns to avoid schema conflicts on load
            dataset = dataset.remove_columns([col for col in dataset.column_names if col not in columns_to_keep])
            if "train" in split:
                raw_train_datasets.append(dataset)
            elif "test" in split or "valid" in split or "eval" in split:
                raw_val_datasets.append(dataset)
            else:
                raise ValueError(f"Split type {split} not recognized as one of test or train.")

    if any(frac < 0 for frac in fracs):
        raise ValueError("Dataset fractions cannot be negative.")

    if len(raw_train_datasets) > 0:
        train_subsets = []
        for dataset, frac in zip(raw_train_datasets, fracs):
            train_subset = dataset.select(range(int(frac * len(dataset))))
            train_subsets.append(train_subset)
        if shuffle:
            raw_datasets["train"] = concatenate_datasets(train_subsets).shuffle(seed=42)
        else:
            raw_datasets["train"] = concatenate_datasets(train_subsets)
    # No subsampling for test datasets to enable fair comparison across models
    if len(raw_val_datasets) > 0:
        if shuffle:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets).shuffle(seed=42)
        else:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets)

    if len(raw_datasets) == 0:
        raise ValueError(
            f"Dataset {dataset_mixer} not recognized with splits {splits}. Check the dataset has been correctly formatted."
        )

    return raw_datasets
def get_datasets(
        data_config: DataArguments | dict,
        splits: Optional[List[str]] = None,
        configs: Optional[List[str]] = None,
        columns_to_keep: Optional[List[str]] = None,
        shuffle: bool = True,
):
    if type(data_config) is DataArguments:
        dataset_mixer = data_config.dataset_mixer
    elif isinstance(data_config, dict):
        dataset_mixer = data_config
    else:
        raise ValueError(f"Data config {data_config} not recognized.")

    raw_datasets = mix_datasets(
        dataset_mixer,
        splits=splits,
        configs=configs,
        columns_to_keep=columns_to_keep,
        shuffle=shuffle,
    )
    return raw_datasets

from alignment import ModelArguments  # 너 코드 구조에 따라 경로 조정해야 할 수 있어

model_args = ModelArguments(
    model_name_or_path="gpt2",  # 또는 네가 쓰는 모델 경로
    tokenizer_name_or_path="/home/minchan.kwon/ADPA/model/student_adpa_0epoch",  # tokenizer만 쓰는 거니까 여기도 모델과 맞춰줘
)

def apply_chat_template(
        example,
        tokenizer,
        task: Literal["sft", "generation", "rm", "dpo"],
        auto_insert_empty_system_msg: bool = True,
):
    if all(k in example.keys() for k in ("chosen", "rejected")):
        if not is_openai_format(example["chosen"]) or not is_openai_format(example["rejected"]):
            raise ValueError(
                f"Could not format example as dialogue for `{task}` task! Require OpenAI format for all messages"
            )

        # For DPO/ORPO, the inputs are triples of (prompt, chosen, rejected), where `chosen` and `rejected` are the final turn of a dialogue
        # We therefore need to extract the N-1 turns to form the prompt
        prompt_messages = example["chosen"][:-1]
        # Now we extract the final turn to define chosen/rejected responses
        chosen_message = example["chosen"][-1]
        rejected_message = example["rejected"][-1]

        # Prepend a system message if the first message is not a system message
        if auto_insert_empty_system_msg:
            maybe_insert_system_message(prompt_messages, tokenizer)

        example["text_prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False,
                                                               add_generation_prompt=True)
        example["text_chosen"] = chosen_message['content'] + tokenizer.eos_token
        example["text_rejected"] = rejected_message['content'] + tokenizer.eos_token
    else:
        raise ValueError(
            f"Could not format example as dialogue for `{task}` task! Require either the "
            f"`[chosen, rejected]` or `[prompt, chosen, rejected]` keys but found {list(example.keys())}"
        )
    return example


# ✅ Step 1. DataArguments 세팅
data_args = DataArguments(
    dataset_mixer={"/home/minchan.kwon/ADPA/data/llama3.2-1b-deita-dpomix/adpa_dataset_0epoch": 1.0},  # 예: {"data/my_dataset": 1.0}
    
    dataset_splits=["train"],
    dataset_configs=[None],
    auto_insert_empty_system_msg=True,
    preprocessing_num_workers=1
)

# ✅ Step 2. 필요한 컬럼만 유지
columns_to_keep = ["messages", "chosen", "rejected", "prompt", "completion", "label",
                   "chosen_labels", "rejected_labels", "rejected_margin_logp_every"]

# ✅ Step 3. 데이터셋 로딩
raw_datasets = get_datasets(data_args, columns_to_keep=columns_to_keep)

# ✅ Step 4. tokenizer 로드
tokenizer = get_tokenizer(model_args, data_args)  # 모델 인자가 None인 경우, 기본 tokenizer 로드

# ✅ Step 5. Chat Template 적용
formatted_dataset = raw_datasets["train"].map(
    apply_chat_template,
    fn_kwargs={
        "tokenizer": tokenizer,
        "task": "dpo",
        "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
    },
    remove_columns=[col for col in raw_datasets["train"].column_names if col not in columns_to_keep],
    desc="Apply chat formatting",
)

# ✅ Step 6. 샘플 확인
sample = formatted_dataset[1]  # 또는 random.choice(formatted_dataset)

from pprint import pprint
import torch

labels = torch.tensor(sample["rejected_labels"])
margin_distributions = sample["rejected_margin_logp_every"]

token_adv_map = {}

for t in range(len(labels)):
    token_id = labels[t].item()
    if token_id == -100:
        continue  # 무시

    token_str = tokenizer.decode([token_id])
    compressed = margin_distributions[t]

    if token_id in compressed["indices"]:
        idx = compressed["indices"].index(token_id)
        adv = compressed["values"][idx]
    else:
        adv = None  # top-k 안에 없음

    token_adv_map[token_str] = adv

print("\n📊 Label token과 Advantage 값:")
for token, adv in token_adv_map.items():
    print(f"{token!r:20} : {adv}")


import matplotlib.pyplot as plt

# 📈 Advantage 시각화
tokens = list(token_adv_map.keys())
advantages = [token_adv_map[token] if token_adv_map[token] is not None else 0 for token in tokens]

plt.figure(figsize=(max(10, len(tokens) * 0.5), 6))
plt.bar(tokens, advantages)
plt.xlabel("Tokens")
plt.ylabel("Advantage")
plt.title("Token-wise Advantage Distribution")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# 그래프 저장
plt.savefig("advantage_distribution_1.png")
plt.close()

# ➕ 필요하면 여기서 load_input_and_target_probs_fast를 바로 써볼 수 있어
# 예시:
# import torch
# input_probs = torch.rand(len(sample["rejected_labels"]), 32000)  # 임시 input probs
# labels = torch.tensor(sample["rejected_labels"])
# target_probs = load_input_and_target_probs_fast([sample["rejected_margin_logp_every"]], input_probs.unsqueeze(0), labels.unsqueeze(0))