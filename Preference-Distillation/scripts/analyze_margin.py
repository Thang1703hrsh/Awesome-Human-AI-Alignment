import sys
sys.path.append("/home/minchan.kwon/ADPA")
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

logger = logging.getLogger(__name__)

if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

if is_wandb_available():
    import wandb

if is_deepspeed_available():
    import deepspeed
def get_compressed_labels(labels: torch.Tensor, compressed_probs) -> torch.Tensor:
    """
    labels: [B, T] – original token indices (e.g. 50256 vocab)
    compressed_probs: compressed_probs[b][j] – where j is response token idx

    returns: labels_compressed: [B, T] – index into compressed vocab (V')
    """
    device = labels.device
    B, T = labels.shape
    labels_compressed = torch.full_like(labels, fill_value=-100)

    for b in range(B):
        label_start = torch.where(labels[b] != -100)[0]
        if label_start.numel() == 0:
            continue
        label_start = label_start[0].item()

        for t in range(T):
            if labels[b, t] == -100:
                continue
            j = t - label_start
            if j < 0 or j >= len(compressed_probs[b]):
                continue
            compressed = compressed_probs[b][j]
            if labels[b, t].item() in compressed["indices"]:
                idx_in_compressed = compressed["indices"].index(labels[b, t].item())
                labels_compressed[b, t] = idx_in_compressed
            else:
                # 안에 없다면 special index: V' - 1 ("remaining_probs")
                labels_compressed[b, t] = len(compressed["indices"])

    return labels_compressed
def get_label_to_compressed_map(labels: torch.Tensor) -> List[List[int]]:
    """
    For each label in the batch, return the mapping from label positions to compressed_probs index.
    That is, output[i][t] = j such that compressed_probs[i][j] corresponds to labels[i][t].

    Returns:
        List[List[int]]: shape = [B, T], with -1 if that label is -100
    """
    batch_size, seq_len = labels.shape
    mapping = []

    for i in range(batch_size):
        row_map = [-1] * seq_len
        valid_indices = torch.where(labels[i] != -100)[0]
        if valid_indices.numel() == 0:
            mapping.append(row_map)
            continue

        label_start = valid_indices[0].item()
        for j, t in enumerate(valid_indices):
            row_map[t.item()] = j  # compressed_probs[i][j] <-> labels[i][t]
        mapping.append(row_map)

    return mapping
@dataclass
class CustomDPOConfig(TrainingArguments):
    # hyper-parameters used in Alignment HandBook
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": ("The Hub model branch to push the model to.")},
    )
    logging_first_step: bool = field(
        default=True,
        metadata={"help": ("Whether to log and evaluate the first global_step or not.")},
    )
    optim: Optional[str] = field(default="rmsprop")
    remove_unused_columns: bool = field(default=False)

    # hyper-parameters used in ADPA
    sft_on_chosen: Optional[float] = field(default=0)
    sft_on_rejected: Optional[float] = field(default=0)
    dpo_weight: Optional[float] = field(default=0)
    distillation_weight: Optional[float] = field(default=0)
    chosen_distil_weight: Optional[float] = field(default=0)
    rejected_distil_weight: Optional[float] = field(default=0)
    adpa_weight: Optional[float] = field(default=0)
    qadapter_alpha_tilde: Optional[float] = field(default=1.0, metadata={"help": "Weight for qadapter loss"})
    qadapter_alpha: Optional[float] = field(default=1.0, metadata={"help": "Weight for qadapter loss"})
    qadapter_distil_weight : Optional[float] = field(default=0)
    dpkd_distil_weight: Optional[float] = field(default=0)
    qadapter_reward_normalization: Optional[bool] = field(default=False, metadata={"help": "Whether to normalize the reward for qadapter"})
    kl_student_weight: Optional[float] = field(default=0.0, metadata={"help": "Weight for KL(student || old_student) loss"})
    kl_student_target: Optional[str] = field(default="rejected", metadata={"help": "Apply student KL on 'chosen', 'rejected', or 'both'"})
    qadapter_loss_type : Optional[str] = field(
        default="dpo",
    )
    adpa_loss_type: Optional[str] = field(
        default="reverse_ce",  # 또는 "soft_kl"
        metadata={"help": "Type of ADPA loss: reverse_ce or soft_kl"},
    )
    adpa_temperature: Optional[float] = field(
        default=1.0,
        metadata={"help": "Temperature for softmax in soft_kl ADPA loss."}
    )
    adpa_importance_sampling: Optional[bool] = field(default=False)
    kl_penalty_weight: Optional[float] = field(default=0)

    # Copy From DPOConfig
    learning_rate: float = 1e-6
    beta: float = 0.1
    label_smoothing: float = 0.0
    loss_type: Literal[
        "sigmoid",
        "hinge",
        "ipo",
        "exo_pair",
        "nca_pair",
        "robust",
        "bco_pair",
        "sppo_hard",
        "aot",
        "aot_pair",
        "discopop",
        "apo_zero",
        "apo_down",
    ] = "sigmoid"
    use_weighting: bool = False
    label_pad_token_id: int = -100
    padding_value: Optional[int] = None
    truncation_mode: str = "keep_end"
    max_length: Optional[int] = None
    max_rejected_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    is_encoder_decoder: Optional[bool] = None
    disable_dropout: bool = True
    generate_during_eval: bool = False
    precompute_ref_log_probs: bool = False
    dataset_num_proc: Optional[int] = None
    model_init_kwargs: Optional[dict[str, Any]] = None
    ref_model_init_kwargs: Optional[dict[str, Any]] = None
    model_adapter_name: Optional[str] = None
    ref_adapter_name: Optional[str] = None
    reference_free: bool = False
    force_use_ref_model: bool = False
    f_divergence_type: FDivergenceType = FDivergenceType.REVERSE_KL
    f_alpha_divergence_coef: float = 1.0
    sync_ref_model: bool = False
    ref_model_mixup_alpha: float = 0.9
    ref_model_sync_steps: int = 64
    rpo_alpha: Optional[float] = None
    discopop_tau: float = 0.05
    use_num_logits_to_keep: bool = False
    simpo_gamma: Optional[float] = 0
    dpo_length_normalization: Optional[bool] = False

    def __post_init__(self):
        return super().__post_init__()


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
) -> DatasetDict:
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


@dataclass
class DistillDataCollator(DataCollatorMixin):
    pad_token_id: int
    return_tensors: str = "pt"

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        # Convert to tensor
        prompt_input_ids = [torch.tensor(example["prompt_input_ids"]) for example in examples]
        prompt_attention_mask = [torch.ones_like(input_ids) for input_ids in prompt_input_ids]
        chosen_input_ids = [torch.tensor(example["chosen_input_ids"]) for example in examples]
        chosen_attention_mask = [torch.ones_like(input_ids) for input_ids in chosen_input_ids]
        chosen_labels = [torch.tensor(example["chosen_labels"]) for example in examples]
        rejected_input_ids = [torch.tensor(example["rejected_input_ids"]) for example in examples]
        rejected_attention_mask = [torch.ones_like(input_ids) for input_ids in rejected_input_ids]
        rejected_labels = [torch.tensor(example["rejected_labels"]) for example in examples]

        # Pad
        output = {}
        output["prompt_input_ids"] = pad(prompt_input_ids, padding_value=self.pad_token_id, padding_side="left")
        output["prompt_attention_mask"] = pad(prompt_attention_mask, padding_value=0, padding_side="left")
        output["chosen_input_ids"] = pad(chosen_input_ids, padding_value=self.pad_token_id)
        output["chosen_attention_mask"] = pad(chosen_attention_mask, padding_value=0)
        output["chosen_labels"] = pad(chosen_labels, padding_value=-100)
        output["rejected_input_ids"] = pad(rejected_input_ids, padding_value=self.pad_token_id)
        output["rejected_attention_mask"] = pad(rejected_attention_mask, padding_value=0)
        output["rejected_labels"] = pad(rejected_labels, padding_value=-100)

        # For DPO Training
        if "ref_chosen_logps" in examples[0]:
            output["ref_chosen_logps"] = torch.tensor([example["ref_chosen_logps"] for example in examples])
        if "ref_rejected_logps" in examples[0]:
            output["ref_rejected_logps"] = torch.tensor([example["ref_rejected_logps"] for example in examples])

        # For Distillation Training
        if "teacher_chosen_probs" in examples[0]:
            output["teacher_chosen_probs"] = [example["teacher_chosen_probs"] for example in examples]
        if "teacher_rejected_probs" in examples[0]:
            output["teacher_rejected_probs"] = [example["teacher_rejected_probs"] for example in examples]

        # For ADPA Training
        if "rejected_margin_logp_every" in examples[0]:
            output["rejected_margin_logp_every"] = [example["rejected_margin_logp_every"] for example in examples]

        return output


class DistillTrainer(Trainer):
    def __init__(
            self,
            model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
            ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
            args: Optional[CustomDPOConfig] = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
            processing_class: Optional[
                Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
            ] = None,
            model_init: Optional[Callable[[], PreTrainedModel]] = None,
            compute_metrics: Optional[Callable[[EvalLoopOutput], dict]] = None,
            callbacks: Optional[list[TrainerCallback]] = None,
            optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
            peft_config: Optional[dict] = None,
    ):
        if args.dpo_weight == 0:
            ref_model = None
        else:
            if "ref_chosen_logps" in train_dataset.column_names and "ref_rejected_logps" in train_dataset.column_names:
                ref_model = None
            elif args.reference_free:
                ref_model = None

        if model is None:
            raise ValueError("No model provided. Please provide a model to train.")

        if not isinstance(model, str) and ref_model is model:
            raise ValueError(
                "`model` and `ref_model` cannot be the same object. If you want `ref_model` to be the "
                "same as `model`, you must mass a copy of it, or `None` if you use peft."
            )

        if args.model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError(
                "You passed model_init_kwargs to the DPOTrainer/DPOConfig, but your model is already instantiated."
            )
        else:
            model_init_kwargs = args.model_init_kwargs
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if torch_dtype is not None:
                # Convert to `torch.dtype` if an str is passed
                if isinstance(torch_dtype, str) and torch_dtype != "auto":
                    torch_dtype = getattr(torch, torch_dtype)
                if torch_dtype != "auto" and not isinstance(torch_dtype, torch.dtype):
                    raise ValueError(
                        f"Invalid `torch_dtype` passed to the DPOConfig. Expected a string with either `torch.dtype` or 'auto', but got {torch_dtype}."
                    )
                model_init_kwargs["torch_dtype"] = torch_dtype

        if args.ref_model_init_kwargs is None or self.args.dpo_weight == 0:
            ref_model_init_kwargs = {}
        elif not isinstance(ref_model, str):
            raise ValueError(
                "You passed ref_model_init_kwargs to the DPOTrainer/DPOConfig, but your ref_model is already instantiated."
            )
        else:
            ref_model_init_kwargs = args.ref_model_init_kwargs
            torch_dtype = ref_model_init_kwargs.get("torch_dtype")
            if torch_dtype is not None:
                # Convert to `torch.dtype` if an str is passed
                if isinstance(torch_dtype, str) and torch_dtype != "auto":
                    torch_dtype = getattr(torch, torch_dtype)
                if torch_dtype != "auto" and not isinstance(torch_dtype, torch.dtype):
                    raise ValueError(
                        f"Invalid `torch_dtype` passed to the DPOConfig. Expected a string with either `torch.dtype` or 'auto', but got {torch_dtype}."
                    )
                ref_model_init_kwargs["torch_dtype"] = torch_dtype

        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if isinstance(ref_model, str):
            ref_model = AutoModelForCausalLM.from_pretrained(ref_model, **ref_model_init_kwargs)

        # Initialize this variable to False. This helps tracking the case when `peft_module_casting_to_bf16`
        # has been called in order to properly call autocast if needed.
        self._peft_has_been_casted_to_bf16 = False

        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_config, we merge and unload it first
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()

            if ref_model is not None and not args.force_use_ref_model:
                raise ValueError(
                    "You passed both a ref_model and a peft_config. For training PEFT adapters with DPO there is no need to pass a reference"
                    " model. Please pass `ref_model=None` in case you want to train PEFT adapters, or pass a ref_model with `force_use_ref_model=True` in DPOTrainer's init."
                    " if you want to use a different ref_model."
                )

            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )

                prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                if _support_gc_kwargs:
                    prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)
            elif getattr(args, "gradient_checkpointing", False):
                # For backward compatibility with older versions of transformers
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:

                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)

                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            # get peft model with the given config
            model = get_peft_model(model, peft_config)
            if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(model)
                # If args.bf16 we need to explicitly call `generate` with torch amp autocast context manager
                self._peft_has_been_casted_to_bf16 = True

        # For models that use gradient_checkpointing, we need to attach a hook that enables input
        # to explicitly have `requires_grad=True`, otherwise training will either silently
        # fail or completely fail.
        elif getattr(args, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if args.generate_during_eval and not is_wandb_available():
            raise ValueError(
                "`generate_during_eval=True` requires Weights and Biases to be installed."
                " Please install `wandb` to resolve."
            )

        # We only support Decoder-Only Text LLM
        self.is_encoder_decoder = False
        self.is_vision_model = False
        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        self.model_adapter_name = args.model_adapter_name
        self.ref_adapter_name = args.ref_adapter_name
        self.reference_free = args.reference_free

        self.ref_model = ref_model
        if args.dpo_weight == 0:
            self.ref_model = None
        else:
            if "ref_chosen_logps" in train_dataset.column_names and "ref_rejected_logps" in train_dataset.column_names:
                self.ref_model = None
            elif self.is_peft_model or args.precompute_ref_log_probs or args.reference_free:
                self.ref_model = None
            elif ref_model is None:
                print('!!Createing reference model!!')
                self.ref_model = create_reference_model(model)


        if processing_class is None:
            raise ValueError("processing_class must be specified to tokenize a DPO dataset.")

        if args.padding_value is not None:
            self.padding_value = args.padding_value
        else:
            if hasattr(processing_class, "pad_token_id") and processing_class.pad_token_id is not None:
                self.padding_value = processing_class.pad_token_id
            elif hasattr(processing_class, "tokenizer") and processing_class.tokenizer.pad_token_id is not None:
                self.padding_value = processing_class.tokenizer.pad_token_id
            else:
                raise ValueError(
                    "Can't find `pad_token_id` in the `processing_class`. "
                    "Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`) "
                    "before instantiating the trainer."
                )

        if data_collator is None:
            data_collator = DistillDataCollator(pad_token_id=self.padding_value)

        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        self.max_length = args.max_length
        self.max_rejected_length = args.max_length
        self.generate_during_eval = args.generate_during_eval
        self.label_pad_token_id = args.label_pad_token_id
        self.max_prompt_length = args.max_prompt_length
        self.truncation_mode = args.truncation_mode
        self.precompute_ref_log_probs = args.precompute_ref_log_probs
        self.use_num_logits_to_keep = args.use_num_logits_to_keep

        # Since ref_logs are precomputed on the first call to get_train/eval_dataloader
        # keep track of first called to avoid computation of future calls
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False

        if (
                args.loss_type in ["hinge", "ipo", "bco_pair", "sppo_hard", "nca_pair", "apo_zero", "apo_down"]
                and args.label_smoothing > 0
        ):
            warnings.warn(
                f"You are using the {args.loss_type} loss type that does not support label smoothing. The "
                "`label_smoothing` parameter will be ignored. Set `label_smoothing` to `0.0` to remove this warning.",
                UserWarning,
            )
        if args.loss_type == "kto_pair":
            raise ValueError("Support for kto_pair has been removed in DPOTrainer. Please use KTOTrainer.")

        self.beta = args.beta
        self.simpo_gamma = args.simpo_gamma
        self.label_smoothing = args.label_smoothing
        self.loss_type = args.loss_type
        self.aux_loss_enabled = getattr(model.config, "output_router_logits", False)
        self.use_weighting = args.use_weighting
        self.aux_loss_coef = getattr(model.config, "router_aux_loss_coef", 0.0)
        if self.aux_loss_enabled and self.aux_loss_coef == 0.0:
            warnings.warn(
                "You set `output_router_logits` to `True` in the model config, but `router_aux_loss_coef` is set to "
                "`0.0`, meaning the auxiliary loss will not be used. Either set `router_aux_loss_coef` to a value "
                "greater than `0.0`, or set `output_router_logits` to `False` if you don't want to use the auxiliary "
                "loss.",
                UserWarning,
            )

        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        self.f_divergence_type = args.f_divergence_type
        self.f_divergence_params = {FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY: args.f_alpha_divergence_coef}
        self.dataset_num_proc = args.dataset_num_proc

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in DPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys are "prompt_input_ids", "chosen_input_ids", and
        # "rejected_input_ids". As a result, the trainer issues the warning: "Could not estimate the number of tokens
        # of the input, floating-point operations will not be computed." To suppress this warning, we set the
        # "estimate_tokens" key in the model's "warnings_issued" dictionary to True. This acts as a flag to indicate
        # that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Compute that only on the main process for faster data processing.
        # see: https://github.com/huggingface/trl/pull/1255
        with PartialState().local_main_process_first():
            if args.chosen_distil_weight > 0:
                train_dataset = train_dataset.filter(lambda row: len(row["chosen_labels"]) > 0 and len(row["teacher_chosen_probs"]) > 0, num_proc=self.dataset_num_proc)
                if eval_dataset is not None:
                    eval_dataset = eval_dataset.filter(lambda row: len(row["chosen_labels"]) > 0 and len(row["teacher_chosen_probs"]) > 0, num_proc=self.dataset_num_proc)
            if args.rejected_distil_weight > 0:
                train_dataset = train_dataset.filter(lambda row: len(row["rejected_labels"]) > 0 and len(row["teacher_rejected_probs"]) > 0, num_proc=self.dataset_num_proc)
                if eval_dataset is not None:
                    eval_dataset = eval_dataset.filter(lambda row: len(row["rejected_labels"]) > 0 and len(row["teacher_rejected_probs"]) > 0, num_proc=self.dataset_num_proc)

            # tokenize the dataset, lower writer batch size to avoid OOM (frequent in vision models)
            fn_kwargs = {
                "processing_class": processing_class,
                "max_prompt_length": args.max_prompt_length,
                "max_length": args.max_length,
                "max_rejected_length": args.max_rejected_length,
            }
            train_dataset = train_dataset.map(
                self.tokenize_row,
                fn_kwargs=fn_kwargs,
                num_proc=self.dataset_num_proc,
                writer_batch_size=10,
                desc="Tokenizing train dataset",
            )
            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(
                    self.tokenize_row,
                    fn_kwargs=fn_kwargs,
                    num_proc=self.dataset_num_proc,
                    writer_batch_size=10,
                    desc="Tokenizing eval dataset",
                )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

        # Deepspeed Zero-3 does not support precompute_ref_log_probs
        if self.is_deepspeed_enabled:
            if self.accelerator.state.deepspeed_plugin.zero_stage == 3 and self.precompute_ref_log_probs:
                raise ValueError(
                    "You cannot use `precompute_ref_log_probs=True` with Deepspeed ZeRO-3. Please set `precompute_ref_log_probs=False`."
                )

        if self.ref_model is None:
            # if not (self.is_peft_model or self.precompute_ref_log_probs):
            #     raise ValueError(
            #         "No reference model and model is not a Peft model. Try setting `precompute_ref_log_probs=True`"
            #     )
            if args.sync_ref_model:
                raise ValueError(
                    "You currently cannot use `ref_model=None` with TR-DPO method. Please provide `ref_model`."
                )
        else:
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            if self.precompute_ref_log_probs:
                raise ValueError(
                    "You cannot use `precompute_ref_log_probs=True` with TR-DPO method. Please set `precompute_ref_log_probs=False`."
                )

            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        if self.loss_type == "bco_pair":
            self.running = RunningMoments(self.accelerator)
            
            
    def save_margin_inference(self, save_path="margin_inference.png"):
        import matplotlib.pyplot as plt

        dataloader = self.get_train_dataloader()  # <-- 핵심: train_dataloader는 tokenize + collate 모두 됨
        margins = []
        indices = []

        for batch in tqdm(dataloader, desc="Computing margins"):
            batch = self._prepare_inputs(batch)
            with torch.no_grad():
                output = self.concatenated_forward(self.model, batch)
                margin = (output["chosen_logps"] - output["rejected_logps"]).cpu().tolist()
                batch_indices = list(range(len(margin)))
            margins.extend(margin)
            indices.extend(batch_indices)

        sorted_pairs = sorted(zip(indices, margins), key=lambda x: x[1])
        sorted_indices, sorted_margins = zip(*sorted_pairs)

        plt.figure()
        plt.scatter(sorted_indices, sorted_margins)
        plt.xlabel('Sample Index')
        plt.ylabel('Margin (chosen_logp - rejected_logp)')
        plt.title('Margin Inference Sorted by Ascending Margin')
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"✅ Saved margin plot to {save_path}")
        
        
        
    @staticmethod
    def tokenize_row(features, processing_class, max_prompt_length, max_length, max_rejected_length):
        tokenizer = processing_class  # the processing class is a tokenizer
        prompt_input_ids = tokenizer(features["prompt"], add_special_tokens=False)["input_ids"]

        if "chosen_labels" in features.keys():
            chosen_labels = features["chosen_labels"]
        else:
            chosen_labels = tokenizer(features["chosen"], add_special_tokens=False)["input_ids"]
        if "rejected_labels" in features.keys():
            rejected_labels = features["rejected_labels"]
        else:
            rejected_labels = tokenizer(features["rejected"], add_special_tokens=False)["input_ids"]

        # Add BOS token
        if tokenizer.bos_token_id != prompt_input_ids[0]:
            prompt_input_ids = [tokenizer.bos_token_id] + prompt_input_ids

        # Add EOS token
        if tokenizer.eos_token_id not in chosen_labels[-5:]:
            chosen_labels = chosen_labels + [tokenizer.eos_token_id]
        if tokenizer.eos_token_id not in rejected_labels[-5:]:
            rejected_labels = rejected_labels + [tokenizer.eos_token_id]

        # Truncate prompt and completion sequences
        if max_rejected_length is not None:
            rejected_labels = rejected_labels[:max_rejected_length]
        completion_length = max(len(prompt_input_ids + chosen_labels), len(prompt_input_ids + rejected_labels))
        if max_length is not None and completion_length > max_length:
            if max_prompt_length is not None and len(prompt_input_ids) > max_prompt_length:
                prompt_input_ids = prompt_input_ids[-max_prompt_length:]
            max_response_len = max_length - len(prompt_input_ids)
            chosen_labels = chosen_labels[:max_response_len]
            rejected_labels = rejected_labels[:max_response_len]

        chosen_input_ids = prompt_input_ids + chosen_labels
        rejected_input_ids = prompt_input_ids + rejected_labels
        output = {
            "prompt_input_ids": prompt_input_ids,
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": [1] * len(chosen_input_ids),
            "chosen_labels": [-100] * len(prompt_input_ids) + chosen_labels,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": [1] * len(rejected_input_ids),
            "rejected_labels": [-100] * len(prompt_input_ids) + rejected_labels,
        }

        if "ref_chosen_logps" in features.keys() and "ref_rejected_logps" in features.keys():
            output["ref_chosen_logps"] = features["ref_chosen_logps"]
            output["ref_rejected_logps"] = features["ref_rejected_logps"]
        if "teacher_chosen_probs" in features.keys():
            output["teacher_chosen_probs"] = features["teacher_chosen_probs"]
        if "teacher_rejected_probs" in features.keys():
            output["teacher_rejected_probs"] = features["teacher_rejected_probs"]
        if "rejected_margin_logp_every" in features.keys():
            output["rejected_margin_logp_every"] = features["rejected_margin_logp_every"]

        return output

    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In DPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by `DPODataCollatorWithPadding`, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt_input_ids", "chosen_input_ids", "rejected_input_ids"]

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_train_dataloader to precompute `ref_log_probs`.
        """

        if self.precompute_ref_log_probs and not self._precomputed_train_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_train_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

            ref_chosen_logps = []
            ref_rejected_logps = []
            for padded_batch in tqdm(iterable=data_loader, desc="Train dataset reference log probs"):
                ref_chosen_logp, ref_rejected_logp = self.compute_ref_log_probs(padded_batch)
                ref_chosen_logp, ref_rejected_logp = self.accelerator.gather_for_metrics(
                    (ref_chosen_logp, ref_rejected_logp)
                )
                ref_chosen_logps.append(ref_chosen_logp.cpu())
                ref_rejected_logps.append(ref_rejected_logp.cpu())

                # Unnecessary cache clearing to avoid OOM
                torch.cuda.empty_cache()
                self.accelerator.free_memory()

            all_ref_chosen_logps = torch.cat(ref_chosen_logps).float().numpy()
            all_ref_rejected_logps = torch.cat(ref_rejected_logps).float().numpy()

            self.train_dataset = self.train_dataset.add_column(name="ref_chosen_logps", column=all_ref_chosen_logps)
            self.train_dataset = self.train_dataset.add_column(
                name="ref_rejected_logps", column=all_ref_rejected_logps
            )

            self._precomputed_train_ref_log_probs = True

        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_eval_dataloader to precompute `ref_log_probs`.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if self.precompute_ref_log_probs and not self._precomputed_eval_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_eval_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))

            ref_chosen_logps = []
            ref_rejected_logps = []
            for padded_batch in tqdm(iterable=data_loader, desc="Eval dataset reference log probs"):
                ref_chosen_logp, ref_rejected_logp = self.compute_ref_log_probs(padded_batch)
                ref_chosen_logp, ref_rejected_logp = self.accelerator.gather_for_metrics(
                    (ref_chosen_logp, ref_rejected_logp)
                )
                ref_chosen_logps.append(ref_chosen_logp.cpu())
                ref_rejected_logps.append(ref_rejected_logp.cpu())

            all_ref_chosen_logps = torch.cat(ref_chosen_logps).float().numpy()
            all_ref_rejected_logps = torch.cat(ref_rejected_logps).float().numpy()

            eval_dataset = eval_dataset.add_column(name="ref_chosen_logps", column=all_ref_chosen_logps)
            eval_dataset = eval_dataset.add_column(name="ref_rejected_logps", column=all_ref_rejected_logps)

            # Save calculated ref_chosen_logps and ref_rejected_logps to the eval_dataset for subsequent runs
            if self.eval_dataset is not None:
                self.eval_dataset = eval_dataset
            self._precomputed_eval_ref_log_probs = True

        return super().get_eval_dataloader(eval_dataset=eval_dataset)

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        with self.accelerator.unwrap_model(
                self.model
        ).disable_adapter() if self.is_peft_model and not self.ref_adapter_name else nullcontext():
            if self.ref_adapter_name:
                self.model.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.set_adapter(self.model_adapter_name or "default")

    def compute_ref_log_probs(self, batch: dict[str, torch.LongTensor]) -> dict:
        """Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset."""
        compte_ref_context_manager = amp.autocast("cuda") if self._peft_has_been_casted_to_bf16 else nullcontext()
        with torch.no_grad(), compte_ref_context_manager:
            if self.ref_model is None:
                #print('no compute ref_logp')
                with self.null_ref_context():
                    ref_model_output = self.concatenated_forward(self.model, batch)
            else:
                #print('compute ref_logp')
                ref_model_output = self.concatenated_forward(self.ref_model, batch)
                
        #print(ref_model_output['rejected_logps'].size())
        return ref_model_output["chosen_logps"], ref_model_output["rejected_logps"]
    
    def compute_ref_probs(self, batch: dict[str, torch.LongTensor]) -> dict:
        """Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset."""
        compte_ref_context_manager = amp.autocast("cuda") if self._peft_has_been_casted_to_bf16 else nullcontext()
        with torch.no_grad(), compte_ref_context_manager:
            if self.ref_model is None:
                #print('no compute ref_logp')
                with self.null_ref_context():
                    ref_model_output = self.concatenated_forward(self.model, batch)
            else:
                #print('compute ref_logp')
                ref_model_output = self.concatenated_forward(self.ref_model, batch)
                
        #print(ref_model_output['rejected_logps'].size())
        return ref_model_output["policy_chosen_probs"], ref_model_output["policy_rejected_probs"]

    @staticmethod
    def concatenated_inputs(
            batch: dict[str, Union[list, torch.LongTensor]], padding_value: int
    ) -> dict[str, torch.LongTensor]:
        """
        Concatenate the `chosen` and `rejected` inputs from the batch into a single tensor for both the prompt
        and completion sequences.

        Args:
            batch (`dict[str, Union[list, torch.LongTensor]]`):
                A batch of input data. The batch must contain the following keys:

                - `"prompt_input_ids"`: Tensor of shape `(batch_size, prompt_length)` representing the prompt input IDs.
                - `"chosen_input_ids"`: Tensor of shape `(batch_size, chosen_length)` representing the chosen completion input IDs.
                - `"rejected_input_ids"`: Tensor of shape `(batch_size, rejected_length)` representing the rejected completion input IDs.\

            padding_value (`int`):
                The padding value to use for the concatenated completion sequences (`chosen_input_ids` and
                `rejected_input_ids`).

        Returns:
            `dict[str, torch.LongTensor]`: A dictionary containing:

                - `"prompt_input_ids"`: Concatenated prompt input IDs of shape `(2 * batch_size, prompt_length)`.
                - `"completion_input_ids"`: Concatenated chosen and rejected completion input IDs of shape `(2 * batch_size, max_length)`.
                - `"prompt_attention_mask"`: Concatenated prompt attention masks of shape `(2 * batch_size, prompt_length)`.
                - `"completion_attention_mask"`: Concatenated chosen and rejected attention masks of shape `(2 * batch_size, max_length)`.\

        Notes:
            The completion input IDs and attention masks are padded to the maximum completion length of the chosen
            or rejected sequences.
        """
        output = {}

        # For the prompt, the input_ids are the same for both the chosen and rejected responses
        output["prompt_input_ids"] = torch.cat([batch["prompt_input_ids"], batch["prompt_input_ids"]], dim=0)
        output["prompt_attention_mask"] = torch.cat(
            [batch["prompt_attention_mask"], batch["prompt_attention_mask"]], dim=0
        )

        # Concatenate the chosen and rejected completions
        max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])
        output["completion_input_ids"] = torch.cat(
            (
                pad_to_length(batch["chosen_input_ids"], max_length, pad_value=padding_value),
                pad_to_length(batch["rejected_input_ids"], max_length, pad_value=padding_value),
            ),
        )
        output["completion_attention_mask"] = torch.cat(
            (
                pad_to_length(batch["chosen_attention_mask"], max_length, pad_value=0),
                pad_to_length(batch["rejected_attention_mask"], max_length, pad_value=0),
            ),
        )
        output["completion_labels"] = torch.cat(
            (
                pad_to_length(batch["chosen_labels"], max_length, pad_value=-100),
                pad_to_length(batch["rejected_labels"], max_length, pad_value=-100),
            )
        )
        return output

    def dpo_loss(
            self,
            chosen_logps: torch.FloatTensor,
            rejected_logps: torch.FloatTensor,
            ref_chosen_logps: torch.FloatTensor,
            ref_rejected_logps: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            chosen_logps (`torch.FloatTensor`):
                Log probabilities of the model for the chosen responses. Shape: `(batch_size,)`.
            rejected_logps (`torch.FloatTensor`):
                Log probabilities of the model for the rejected responses. Shape: `(batch_size,)`.
            ref_chosen_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the chosen responses. Shape: `(batch_size,)`.
            ref_rejected_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the rejected responses. Shape: `(batch_size,)`.

        Returns:
            A tuple of three tensors: `(losses, chosen_rewards, rejected_rewards)`.
            The losses tensor contains the DPO loss for each example in the batch.
            The `chosen_rewards` and `rejected_rewards` tensors contain the rewards for the chosen and rejected
            responses, respectively.
        """
        device = self.accelerator.device

        # Get the log ratios for the chosen and rejected responses
        chosen_logratios = chosen_logps.to(device) - (not self.reference_free) * ref_chosen_logps.to(device)
        rejected_logratios = rejected_logps.to(device) - (not self.reference_free) * ref_rejected_logps.to(device)

        if self.f_divergence_type == FDivergenceType.ALPHA_DIVERGENCE.value:
            # The alpha-divergence formula: (1 - u^-alpha) / alpha
            # The divergence difference between the chosen and rejected sample is:
            #     (1 - u[w]^-alpha) / alpha - (1 - u[l]^-alpha) / alpha
            #        = (u[l]^-alpha - u[w]^-alpha) / alpha
            # where u[w] and u[l] are the policy/reference probability ratios
            # for the chosen and rejected samples, respectively.
            alpha_coef = FDivergenceConstants.ALPHA_DIVERGENCE_COEF_DEFAULT
            if self.f_divergence_params and FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY in self.f_divergence_params:
                alpha_coef = float(self.f_divergence_params[FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY])
            logits = (cap_exp(rejected_logratios * -alpha_coef) - cap_exp(chosen_logratios * -alpha_coef)) / alpha_coef
        else:
            logratios = chosen_logps - rejected_logps
            if self.reference_free:
                ref_logratios = torch.tensor([0], dtype=logratios.dtype, device=logratios.device)
            else:
                ref_logratios = ref_chosen_logps - ref_rejected_logps

            logratios = logratios.to(self.accelerator.device)
            ref_logratios = ref_logratios.to(self.accelerator.device)
            logits = logratios - ref_logratios - self.args.simpo_gamma

            if self.f_divergence_type == FDivergenceType.JS_DIVERGENCE.value:
                # The js-divergence formula: log(2 * u / (1 + u))
                # The divergence difference between the chosen and rejected sample is:
                #     log(2 * u[w] / (1 + u[w])) - log(2 * u[l] / (1 + u[l]))
                #       = log(u[w]) - log(u[l]) - (log(1 + u[w]) - log(1 + u[l]))
                # where u[w] and u[l] are the policy/reference probability ratios
                # for the chosen and rejected samples, respectively.
                logits -= F.softplus(chosen_logratios) - F.softplus(rejected_logratios)

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the
        # labels and calculates a conservative DPO loss.
        if self.loss_type == "sigmoid":
            losses = (
                    -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                    - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )

        elif self.loss_type == "robust":
            losses = (
                             -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                             + F.logsigmoid(-self.beta * logits) * self.label_smoothing
                     ) / (1 - 2 * self.label_smoothing)

        elif self.loss_type == "exo_pair":
            # eqn (16) of the EXO paper: https://huggingface.co/papers/2402.00856
            import math

            if self.label_smoothing == 0:
                self.label_smoothing = 1e-3
            losses = (self.beta * logits).sigmoid() * (
                    F.logsigmoid(self.beta * logits) - math.log(1 - self.label_smoothing)
            ) + (-self.beta * logits).sigmoid() * (F.logsigmoid(-self.beta * logits) - math.log(self.label_smoothing))

        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)

        elif self.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.beta)) ** 2

        elif self.loss_type == "bco_pair":
            chosen_logratios = chosen_logps - ref_chosen_logps
            rejected_logratios = rejected_logps - ref_rejected_logps
            chosen_rewards = self.beta * chosen_logratios
            rejected_rewards = self.beta * rejected_logratios
            rewards = torch.cat((chosen_rewards, rejected_rewards), 0).mean().detach()
            self.running.update(rewards)
            delta = self.running.mean
            losses = -F.logsigmoid((self.beta * chosen_logratios) - delta) - F.logsigmoid(
                -(self.beta * rejected_logratios - delta)
            )

        elif self.loss_type == "sppo_hard":
            # In the paper (https://huggingface.co/papers/2405.00675), SPPO employs a soft probability approach,
            # estimated using the PairRM score. The probability calculation is conducted outside of the trainer class.
            # The version described here is the hard probability version, where P in Equation (4.7) of Algorithm 1 is
            # set to 1 for the winner and 0 for the loser.
            a = chosen_logps - ref_chosen_logps
            b = rejected_logps - ref_rejected_logps
            losses = (a - 0.5 / self.beta) ** 2 + (b + 0.5 / self.beta) ** 2

        elif self.loss_type == "nca_pair":
            chosen_rewards = (chosen_logps - ref_chosen_logps) * self.beta
            rejected_rewards = (rejected_logps - ref_rejected_logps) * self.beta
            losses = (
                    -F.logsigmoid(chosen_rewards)
                    - 0.5 * F.logsigmoid(-chosen_rewards)
                    - 0.5 * F.logsigmoid(-rejected_rewards)
            )

        elif self.loss_type == "aot_pair":
            chosen_logratios = chosen_logps - ref_chosen_logps
            rejected_logratios = rejected_logps - ref_rejected_logps
            chosen_logratios_sorted, _ = torch.sort(chosen_logratios, dim=0)
            rejected_logratios_sorted, _ = torch.sort(rejected_logratios, dim=0)
            delta = chosen_logratios_sorted - rejected_logratios_sorted
            losses = (
                    -F.logsigmoid(self.beta * delta) * (1 - self.label_smoothing)
                    - F.logsigmoid(-self.beta * delta) * self.label_smoothing
            )

        elif self.loss_type == "aot":
            logratios = chosen_logps - rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps
            logratios_sorted, _ = torch.sort(logratios, dim=0)
            ref_logratios_sorted, _ = torch.sort(ref_logratios, dim=0)
            delta = logratios_sorted - ref_logratios_sorted
            losses = (
                    -F.logsigmoid(self.beta * delta) * (1 - self.label_smoothing)
                    - F.logsigmoid(-self.beta * delta) * self.label_smoothing
            )

        elif self.loss_type == "apo_zero":
            # Eqn (7) of the APO paper (https://huggingface.co/papers/2408.06266)
            # Use this loss when you believe the chosen outputs are better than your model's default output
            losses_chosen = 1 - F.sigmoid(self.beta * chosen_logratios)  # Increase chosen likelihood
            losses_rejected = F.sigmoid(self.beta * rejected_logratios)  # Decrease rejected likelihood
            losses = losses_chosen + losses_rejected

        elif self.loss_type == "apo_down":
            # Eqn (8) of the APO paper (https://huggingface.co/papers/2408.06266)
            # Use this loss when you believe the chosen outputs are worse than your model's default output.
            # Decrease chosen likelihood and decrease rejected likelihood more
            losses_chosen = F.sigmoid(self.beta * chosen_logratios)
            losses_rejected = 1 - F.sigmoid(self.beta * (chosen_logratios - rejected_logratios))
            losses = losses_chosen + losses_rejected

        elif self.loss_type == "discopop":
            # Eqn (5) of the DiscoPOP paper (https://huggingface.co/papers/2406.08414)
            # This loss was discovered with LLM discovery
            logratios = chosen_logps - rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps
            logits = logratios - ref_logratios
            logits = logits * self.beta
            # Modulate the mixing coefficient based on the log ratio magnitudes
            log_ratio_modulation = torch.sigmoid(logits / self.args.discopop_tau)
            logistic_component = -F.logsigmoid(logits)
            exp_component = torch.exp(-logits)
            # Blend between logistic and exponential component based on log ratio modulation
            losses = logistic_component * (1 - log_ratio_modulation) + exp_component * log_ratio_modulation

        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'exo_pair', "
                "'nca_pair', 'robust', 'bco_pair', 'sppo_hard', 'aot', 'aot_pair', 'discopop', 'apo_zero', 'apo_down']"
            )

        chosen_rewards = self.beta * (chosen_logps.to(device) - ref_chosen_logps.to(device)).detach()
        rejected_rewards = self.beta * (rejected_logps.to(device) - ref_rejected_logps.to(device)).detach()

        return losses, chosen_rewards, rejected_rewards

    def concatenated_forward(self, model: nn.Module, batch: dict[str, Union[list, torch.LongTensor]]):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        num_examples = batch["prompt_input_ids"].shape[0]

        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.padding_value)

        model_kwargs = {}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True
        input_ids = concatenated_batch["completion_input_ids"]
        attention_mask = concatenated_batch["completion_attention_mask"]
        labels = concatenated_batch["completion_labels"]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **model_kwargs)

        # Offset the logits by one to align with the labels
        logits = outputs.logits[:, :-1, :]
        labels = labels[:, 1:].clone()
        loss_mask = labels != -100

        # Compute the log probabilities of the labels
        labels_0 = labels.clone()
        labels_0[~loss_mask] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels_0.unsqueeze(2)).squeeze(2)
        per_token_logps[~loss_mask] = 0
        all_logps = per_token_logps.sum(-1)

        output = {}

        if self.use_weighting:
            with torch.no_grad():
                # Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827
                logprobs = F.log_softmax(logits, dim=-1)
                weights_adjustment_factor = torch.logsumexp(2 * logprobs, dim=-1)  # same as sum(probs**2) in log space
                per_token_logps_adjusted = per_token_logps - weights_adjustment_factor
                all_weights = (per_token_logps_adjusted * loss_mask).sum(-1) / loss_mask.sum(-1)
                chosen_weights = all_weights[:num_examples]
                rejected_weights = all_weights[num_examples:]
                output["policy_weights"] = torch.clamp(torch.exp(chosen_weights + rejected_weights), max=1)

        # calculate the NLL loss
        chosen_logits = logits[:num_examples]
        chosen_labels = labels[:num_examples]
        output['chosen_labels'] = chosen_labels
        output['rejected_labels'] = labels[num_examples:]
        output["nll_loss"] = F.cross_entropy(
            torch.flatten(chosen_logits, end_dim=1), torch.flatten(chosen_labels, end_dim=1), ignore_index=-100
        )

        # storge the probability
        probs = F.softmax(logits, dim=-1)
        output["policy_chosen_probs"] = probs[:num_examples]
        output["policy_rejected_probs"] = probs[num_examples:]
        output["policy_rejected_logits"] = logits[num_examples:]  # shape: [B_rejected, T, V]
        if self.loss_type == "ipo" or self.args.dpo_length_normalization:
            all_logps = all_logps / loss_mask.sum(-1)

        output["chosen_logps"] = all_logps[:num_examples]
        output["rejected_logps"] = all_logps[num_examples:]
        output["mean_chosen_logits"] = logits[:num_examples][loss_mask[:num_examples]].mean()
        output["mean_rejected_logits"] = logits[num_examples:][loss_mask[num_examples:]].mean()
        return output

    @staticmethod
    def get_kl_loss(input_probs, target_probs, labels):
        mask = (labels != -100).to(input_probs.device)
        input_probs[input_probs < 1e-7] = 1e-7
        fkl = F.kl_div(
            input=input_probs[mask].log(),
            target=target_probs[mask],
            reduction='none'
        ).sum(-1).mean()
        return fkl
    # 이 부분이 policy gradient 계산 부분
    @staticmethod
    def get_reverse_ce(input_probs, target_score, labels):
        mask = (labels != -100).to(input_probs.device)
        input_probs = input_probs[mask]
        target_score = target_score[mask]
        value_reward_p = (input_probs * target_score).sum(-1).mean()
        return - value_reward_p
    

    def get_batch_loss_metrics(
            self,
            model,
            batch: dict,
            train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        losses = torch.tensor(0, device=batch["chosen_input_ids"].device, dtype=torch.float)
        prefix = "eval_" if train_eval == "eval" else ""

        # PART1: NLL loss on CHOSEN
        model_output = self.concatenated_forward(model, batch)
        if torch.isnan(model_output["nll_loss"]):
            print(model_output.keys())
            print("❗️ NaN 발생!")
            print("chosen_logits:", model_output["policy_chosen_probs"])
            print("chosen_labels:", model_output["chosen_labels"])
            print("valid label 개수:", (model_output["chosen_labels"] != -100).sum().item())
            print("logits contains NaN:", torch.isnan(model_output["policy_chosen_probs"]).any().item())
            print("logits contains Inf:", torch.isinf(model_output["policy_chosen_probs"]).any().item())
        losses = losses + self.args.sft_on_chosen * model_output["nll_loss"].mean()
        metrics[f"{prefix}loss/chosen-sft"] = model_output["nll_loss"].detach().mean().cpu().item()

        #PART1.5: NLL loss on REJECTED
        if self.args.sft_on_rejected > 0:
            rejected_logits = model_output["policy_rejected_logits"]
            rejected_labels = model_output["rejected_labels"]
            rejected_loss = F.cross_entropy(
                torch.flatten(rejected_logits, end_dim=1), torch.flatten(rejected_labels, end_dim=1), ignore_index=-100
            )
            losses = losses + self.args.sft_on_rejected * rejected_loss.mean()
            metrics[f"{prefix}loss/rejected-sft"] = rejected_loss.detach().mean().cpu().item()
            

        # PART2: DPO loss
        ref_chosen_logps, ref_rejected_logps = None, None
        if self.args.dpo_weight > 0.01:
            if "ref_chosen_logps" in batch.keys() and "ref_rejected_logps" in batch.keys():
                ref_chosen_logps = batch["ref_chosen_logps"]
                ref_rejected_logps = batch["ref_rejected_logps"]
            else:
                ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch)
            dpo_losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                model_output["chosen_logps"], model_output["rejected_logps"], ref_chosen_logps, ref_rejected_logps
            )
            if self.use_weighting:
                dpo_losses = dpo_losses * model_output["policy_weights"]
            losses = losses + self.args.dpo_weight * dpo_losses.mean()
            reward_accuracies = (chosen_rewards > rejected_rewards).float()
            metrics[f"{prefix}loss/dpo"] = dpo_losses.detach().mean().cpu().item()
            metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
            metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
            metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
            metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()

        #PART2.5 : Reference regularizer
        if self.args.kl_student_weight>0:
            #print(self.ref_model)
# KL 정규화: ref logps가 batch에 없으면 직접 계산
            if self.args.kl_student_target in ("chosen", "both"):
                if "ref_chosen_logps" in batch:
                    ref_chosen_logps = batch["ref_chosen_logps"]
                else:
                    ref_chosen_logps, _ = self.compute_ref_log_probs(batch)
                chosen_logps = model_output["chosen_logps"]
                kl_chosen_loss = ref_chosen_logps - chosen_logps
                losses = losses + self.args.kl_student_weight * kl_chosen_loss.mean()
                metrics[f"{prefix}loss/kl-chosen"] = kl_chosen_loss.detach().mean().cpu().item()
            if self.args.kl_student_target in ("rejected", "both"):
                if "ref_rejected_logps" in batch:
                    ref_rejected_logps = batch["ref_rejected_logps"]
                else:
                    _, ref_rejected_logps = self.compute_ref_log_probs(batch)
                rejected_logps = model_output["rejected_logps"]
                kl_rejected_loss = ref_rejected_logps - rejected_logps
                losses = losses + self.args.kl_student_weight * kl_rejected_loss.mean()
                metrics[f"{prefix}loss/kl-rejected"] = kl_rejected_loss.detach().mean().cpu().item()

        # KL Penalty
        if self.args.kl_penalty_weight > 0:
            if ref_rejected_logps is None:
                if "ref_rejected_logps" in batch.keys():
                    ref_rejected_logps = batch["ref_rejected_logps"]
                else:
                    _, ref_rejected_logps = self.compute_ref_log_probs(batch)
            #print(ref_rejected_logps.size(), 'KL-penaly')
            # Because `rejected` sampled from REF student, we use forward KL here.
            kl_penalty_loss = ref_rejected_logps - model_output["rejected_logps"]
            #print(kl_penalty_loss.size(), 'KL_penalty loss')
            losses = losses + self.args.kl_penalty_weight * kl_penalty_loss.mean()
            metrics[f"{prefix}loss/kl-penalty"] = kl_penalty_loss.detach().mean().cpu().item()



        if self.args.dpkd_distil_weight > 0:
            # [B, T, V']: soft target probs
            policy_chosen_probs, teacher_chosen_probs = load_input_and_target_probs_fast(
                compressed_probs=batch["teacher_chosen_probs"],
                input_probs=model_output["policy_chosen_probs"],
                labels=model_output["chosen_labels"]
            )
            policy_rejected_probs, teacher_rejected_probs = load_input_and_target_probs_fast(
                compressed_probs=batch["teacher_rejected_probs"],
                input_probs=model_output["policy_rejected_probs"],
                labels=model_output["rejected_labels"]
            )

            dpkd_losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                policy_chosen_probs, policy_rejected_probs, teacher_chosen_probs, teacher_rejected_probs
            )
            if self.use_weighting:
                dpkd_losses = dpkd_losses * model_output["policy_weights"]
            losses = losses + self.args.dpkd_distil_weight * dpkd_losses.mean()
            reward_accuracies = (chosen_rewards > rejected_rewards).float()
            metrics[f"{prefix}loss/dpkd"] = dpkd_losses.detach().mean().cpu().item()
            metrics[f"{prefix}dpkd_rewards/chosen"] = chosen_rewards.mean().cpu()
            metrics[f"{prefix}dpkd_rewards/rejected"] = rejected_rewards.mean().cpu()
            metrics[f"{prefix}dpkd_rewards/accuracies"] = reward_accuracies.mean().cpu()
            metrics[f"{prefix}dpkd_rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()



        # PART3: DISTILLATION LOSS
        if self.args.distillation_weight > 0:
            distill_loss = torch.tensor(0, device=batch["chosen_input_ids"].device, dtype=torch.float)
            qadapter_loss = torch.tensor(0,device=batch['chosen_input_ids'].device,dtype=torch.float)
            if self.args.chosen_distil_weight > 0.001:
                policy_chosen_probs, teacher_chosen_probs = load_input_and_target_probs_fast(
                    compressed_probs=batch["teacher_chosen_probs"],
                    input_probs=model_output["policy_chosen_probs"],
                    labels=model_output["chosen_labels"])
                chosen_fkl_loss = self.get_kl_loss(
                    input_probs=policy_chosen_probs,
                    target_probs=teacher_chosen_probs,
                    labels=model_output["chosen_labels"])
                distill_loss += self.args.chosen_distil_weight * chosen_fkl_loss
                metrics[f"{prefix}loss/chosen-fkl"] = chosen_fkl_loss.detach().mean().cpu().item()
            if self.args.rejected_distil_weight > 0.001:
                policy_rejected_probs, teacher_rejected_probs = load_input_and_target_probs_fast(
                    compressed_probs=batch["teacher_rejected_probs"],
                    input_probs=model_output["policy_rejected_probs"],
                    labels=model_output["rejected_labels"])
                rejected_fkl_loss = self.get_kl_loss(
                    input_probs=policy_rejected_probs,
                    target_probs=teacher_rejected_probs,
                    labels=model_output["rejected_labels"])
                distill_loss += self.args.rejected_distil_weight * rejected_fkl_loss
                metrics[f"{prefix}loss/rejected-fkl"] = rejected_fkl_loss.detach().mean().cpu().item()
                
                
            if self.args.qadapter_distil_weight > 0:
                # [B, T, V']: soft target probs
                policy_rejected_probs, teacher_rejected_probs = load_input_and_target_probs_fast(
                    compressed_probs=batch["teacher_rejected_probs"],
                    input_probs=model_output["policy_rejected_probs"],
                    labels=model_output["rejected_labels"]
                )
                policy_chosen_probs, teacher_chosen_probs = load_input_and_target_probs_fast(
                    compressed_probs=batch["teacher_chosen_probs"],
                    input_probs=model_output["policy_chosen_probs"],
                    labels=model_output["chosen_labels"]
                )

                # Concatenate chosen/rejected
                policy_probs = torch.cat([policy_chosen_probs, policy_rejected_probs], dim=0)  # [2B, T, V']
                teacher_probs = torch.cat([teacher_chosen_probs, teacher_rejected_probs], dim=0)
                labels = torch.cat([model_output["chosen_labels"], model_output["rejected_labels"]], dim=0)  # [2B, T]
                mask = (labels != -100)  # [2B, T]

                # Q-values: log of student probs
                q_values = torch.log(torch.clamp(policy_probs, min=1e-8))  # [2B, T, V']
                log_pi_star = torch.log(torch.clamp(teacher_probs, min=1e-8))  # [2B, T, V']

                # V(s): logsumexp over vocab dimension
                alpha_tilde = self.args.qadapter_alpha_tilde
                alpha_0 = self.args.qadapter_alpha
                gamma = 1.0
                beta = 0.0

                values = alpha_tilde * torch.logsumexp((q_values + alpha_0 * log_pi_star) / alpha_tilde, dim=-1)  # [2B, T]

                # Q(s,a): gather only ground truth tokens
                labels_chosen_compressed = get_compressed_labels(model_output["chosen_labels"], batch["teacher_chosen_probs"])
                labels_rejected_compressed = get_compressed_labels(model_output["rejected_labels"], batch["teacher_rejected_probs"])
                labels_compressed = torch.cat([labels_chosen_compressed, labels_rejected_compressed], dim=0)  # [2B, T]
                # Step 2: Create mask and safe index for gather
                valid_mask = labels_compressed != -100  # [2B, T]
                labels_safe = labels_compressed.clone()
                labels_safe[~valid_mask] = 0
                labels_exp = labels_safe.unsqueeze(-1)  # [2B, T, 1]

                # Step 3: Gather q-values
                q_value_ai = torch.gather(q_values, dim=-1, index=labels_exp).squeeze(-1)  # [2B, T]
                q_value_ai = q_value_ai * valid_mask  # mask out invalid positions
                #print("q_values.shape:", q_values.shape)
                #print("labels_exp.max():", labels_exp.max().item())
                #print("labels_exp.min():", labels_exp.min().item())
                # reward = Q - gamma * V
                reward = (q_value_ai - gamma * values) * mask  # [2B, T]
                reward = reward * valid_mask  # 마스킹 적용
                if self.args.qadapter_reward_normalization:
                    lengths = mask.sum(dim=1).float()  # [2B]
                    reward_sum = reward.sum(dim=1) / lengths.clamp(min=1.0)  # 길이 정규화
                else:
                    reward_sum = reward.sum(dim=1)

                # DPO-style preference loss
                bs = reward_sum.shape[0] // 2
                if self.args.qadapter_loss_type == "dpo":
                    # 기존 DPO-style loss
                    logits = torch.stack([reward_sum[:bs], reward_sum[bs:]], dim=-1)
                    targets = torch.zeros(bs, dtype=torch.long, device=logits.device)
                    pref_loss = F.cross_entropy(logits, targets)
                    qadapter_loss = pref_loss

                elif self.args.qadapter_loss_type == "pg":
                    # 새로 추가한 policy gradient 방식
                    log_policy_probs = torch.log(torch.clamp(policy_probs, min=1e-8))
                    log_pi_a = torch.gather(log_policy_probs, dim=-1, index=labels_exp).squeeze(-1)
                    log_pi_a = log_pi_a * valid_mask
                    pg_term = - (log_pi_a * reward)
                    pg_term = pg_term * mask * valid_mask
                    qadapter_loss = pg_term.sum() / mask.sum()

                metrics[f"{prefix}loss/qadapter_total"] = qadapter_loss.detach().cpu().item()
                metrics[f"{prefix}reward/qadapter_mean"] = reward_sum.detach().mean().cpu().item()
                metrics[f"{prefix}reward/qadapter_margin"] = (reward_sum[:bs] - reward_sum[bs:]).mean().cpu().item()
                metrics[f"{prefix}loss/qadapter_accuracy"] = (reward_sum[:bs] > reward_sum[bs:]).float().mean().cpu().item()
                
            losses = losses + self.args.distillation_weight * distill_loss.mean() + self.args.qadapter_distil_weight * qadapter_loss.mean()

        # PART4: Advantage-Guided Distillation Preference Alignment  model_output["rejected_labels"][model_output["rejected_labels"] != -100], [i['indices'] for i in batch["rejected_margin_logp_every"][0]]
        #여기가 ADPA 들어가는 부분. 이쪽을 수정하면 좋을 것 같은데.
        # PART4: Advantage-Guided Distillation Preference Alignment
        # PART4: Advantage-Guided Distillation Preference Alignment
        if self.args.adpa_weight > 0:
            #print(batch.keys())
            policy_rejected_probs, rejected_margin_logp = load_input_and_target_probs_fast(
                compressed_probs=batch["rejected_margin_logp_every"],
                input_probs=model_output["policy_rejected_probs"],
                labels=model_output["rejected_labels"],
                )
            batch_size, seq_len, top_k = policy_rejected_probs.size()
            vocab_size = top_k
            if self.args.adpa_loss_type == "reverse_ce":
                adpa_loss = self.get_reverse_ce(
                    policy_rejected_probs, rejected_margin_logp, model_output["rejected_labels"]
                )

            elif self.args.adpa_loss_type == "reverse_ce_with_ref":
                
                if "ref_rejected_logps" in batch:
                    ref_rejected_logps = batch["ref_rejected_logps"]
                else:
                    _, ref_rejected_logps = self.compute_ref_probs(batch)
                #print(ref_rejected_logps.size())
                ref_rejected_probs, _ = load_input_and_target_probs_fast(
                    compressed_probs=batch["rejected_margin_logp_every"],
                    input_probs=ref_rejected_logps,
                    labels=model_output["rejected_labels"],
                    )
                lambda_zero = -0.5
                adpa_loss = self.get_reverse_ce(
                    policy_rejected_probs , rejected_margin_logp + lambda_zero*ref_rejected_probs, model_output["rejected_labels"]
                )

            elif self.args.adpa_loss_type == "simple_adv":
                new_policy = policy_rejected_probs.detach().clone()
                new_margin = rejected_margin_logp  - new_policy
                # new_margin을 2로 나눠서 mean으로 만들어줌
                #new_margin = new_margin / 2.0
                adpa_loss = self.get_reverse_ce(
                    policy_rejected_probs, new_margin, model_output["rejected_labels"]
                )

            #여기
            elif self.args.adpa_loss_type == "q-sft":
                labels = model_output["rejected_labels"].clone()            # [B, T]
                student_probs = model_output["policy_rejected_probs"].clone()  # [B, T, V]
                with torch.no_grad():
                    compressed_probs = batch["rejected_margin_logp_every"]
                    device = student_probs.device
                    batch_size, seq_len, vocab_size = student_probs.shape

                    # logit 초기값 (정답 제외 나머지 토큰들)
                    base_logit = -1.0
                    target_logits = torch.full((batch_size, seq_len, vocab_size), fill_value=base_logit, device=device)

                    # 위치 매핑: label[b][t] → compressed_probs[b][j]에서의 j 구함
                    label2comp_map = get_label_to_compressed_map(labels)

                    for b in range(batch_size):
                        for t in range(seq_len):
                            label = labels[b, t].item()
                            if label == -100:
                                continue

                            j = label2comp_map[b][t]
                            if j == -1 or j >= len(compressed_probs[b]):
                                continue

                            compressed = compressed_probs[b][j]
                            if label in compressed["indices"]:
                                idx = compressed["indices"].index(label)
                                logit_val = compressed["values"][idx]
                                target_logits[b, t, label] = logit_val
                            else:
                                # label이 포함 안 되어 있으면 그대로 base_logit 유지
                                pass

                    # softmax → 확률 분포
                    target_probs = torch.softmax(target_logits, dim=-1).clone().detach()

                # KL(student || target)
                adpa_loss = self.get_kl_loss(
                    input_probs=student_probs,
                    target_probs=target_probs,
                    labels=labels
                )
                                        
                    
                    
                


            elif self.args.adpa_loss_type == "kl_from_logits":
                #teacher_probs = F.softmax(rejected_margin_logp / self.args.adpa_temperature, dim=-1)
                #teacher_probs = torch.exp(rejected_margin_logp)
                #teacher_probs = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True)
                adpa_loss = self.get_kl_loss(
                    policy_rejected_probs, rejected_margin_logp, model_output["rejected_labels"]
                )
                
                
            elif self.args.adpa_loss_type == "rough_pg":
                batch_size, seq_len, top_k = policy_rejected_probs.size()
                # 1) mask를 이용해 -100 토큰 필터링
                mask = (model_output["rejected_labels"] != -100)
                # 2) margin(=rejected_margin_logp)은 [B, T], 그 중 valid 토큰만 합산 -> shape: [B]
                advantage_summed = rejected_margin_logp[mask].view(batch_size, -1).sum(dim=1)
                # 3) 시퀀스 전체 로그우도는 model_output["rejected_logps"] : [B]
                seq_logprob = model_output["rejected_logps"]  # (already sum of valid tokens)
                # 4) 로스 = - E[ advantage_summed * seq_logprob ]
                #    advantage_summed와 seq_logprob를 곱한 뒤 음수 부호를 취해 mean
                adpa_loss = - (advantage_summed * seq_logprob).mean()



            elif self.args.adpa_loss_type == "soft_kl_plus_student":
                batch_size, seq_len, top_k = policy_rejected_probs.size()
                vocab_size = model_output["policy_rejected_probs"].shape[-1]

                # [1] teacher margin -> teacher top-k (indices, values) => target_probs 초기화
                student_probs = model_output["policy_rejected_probs"].clone()
                labels = model_output["rejected_labels"]

                # 토치 텐서 준비
                topk_indices = torch.zeros(batch_size, seq_len, top_k, dtype=torch.long, device=student_probs.device)
                topk_values = torch.zeros(batch_size, seq_len, top_k, dtype=torch.float, device=student_probs.device)

                for b in range(batch_size):
                    margin_logp_seq = batch["rejected_margin_logp_every"][b]
                    for t in range(len(margin_logp_seq)):
                        if t >= seq_len:
                            continue 
                        compressed = margin_logp_seq[t]
                        k = min(top_k, len(compressed["indices"]))
                        topk_indices[b, t, :k] = torch.tensor(compressed["indices"][:k], dtype=torch.long, device=student_probs.device)
                        topk_values[b, t, :k] = torch.tensor(compressed["values"][:k], dtype=torch.float, device=student_probs.device)

                # [2] teacher 분포를 vocab_size에 맞춰 sparse->dense로 확장
                teacher_logits = torch.full((batch_size, seq_len, vocab_size), -1e9, device=student_probs.device)
                # -1e9로 초기화 -> teacher margin이 0인 위치는 매우 작은 로그값
                for b in range(batch_size):
                    margin_logp_seq = batch["rejected_margin_logp_every"][b]
                    seq_len_local = len(margin_logp_seq)
                    for t in range(seq_len_local):
                        if t >= seq_len:
                            continue 
                        k = min(top_k, len(margin_logp_seq[t]["indices"]))
                        teacher_logits[b, t, topk_indices[b, t, :k]] = topk_values[b, t, :k]

                # [3] student_probs -> student_logits
                #    log() 시에 0이 들어있으면 -inf가 나올 수 있으니 작은 값을 깔아둔다.
                EPS = 1e-7
                student_probs_clamped = student_probs.clamp(min=EPS)  # avoid log(0)
                student_logits = student_probs_clamped.log().detach()

                # [4] 혼합(logits) = teacher_logits + lam * student_logits
                #    lam, temperature 등은 hyperparam (필요시 self.args에 추가)
                lam = 1.0
                temperature = self.args.adpa_temperature
                combined_logits = teacher_logits + lam * student_logits
                # softmax
                target_probs = F.softmax(combined_logits / temperature, dim=-1).clone().detach()

                # [5] KL: KL( student_probs || target_probs )
                adpa_loss = self.get_kl_loss(
                    input_probs=student_probs,
                    target_probs=target_probs,
                    labels=model_output["rejected_labels"]
                )
                
            else:
                raise ValueError(f"Unsupported adpa_loss_type: {self.args.adpa_loss_type}")

            metrics[f"{prefix}loss/adpa"] = adpa_loss.detach().mean().cpu().item()

            if self.args.adpa_importance_sampling:
                if ref_rejected_logps is None:
                    if "ref_rejected_logps" in batch.keys():
                        ref_rejected_logps = batch["ref_rejected_logps"]
                    else:
                        _, ref_rejected_logps = self.compute_ref_log_probs(batch)

                importance_weight = torch.exp(model_output["rejected_logps"] - ref_rejected_logps)
                adpa_loss = adpa_loss * importance_weight
                metrics[f"{prefix}loss/importance_weight"] = importance_weight.detach().mean().cpu().item()

            losses = losses + self.args.adpa_weight * adpa_loss.mean()


        metrics[f"{prefix}logps/chosen"] = model_output["chosen_logps"].detach().mean().cpu()
        metrics[f"{prefix}logps/rejected"] = model_output["rejected_logps"].detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = model_output["mean_chosen_logits"].detach().cpu()
        metrics[f"{prefix}logits/rejected"] = model_output["mean_rejected_logits"].detach().cpu()
        if self.aux_loss_enabled:
            metrics[f"{prefix}aux_loss"] = model_output["aux_loss"].detach().cpu()

        return losses.mean(), metrics

    def compute_loss(
            self,
            model: Union[PreTrainedModel, nn.Module],
            inputs: dict[str, Union[torch.Tensor, Any]],
            return_outputs=False,
            num_items_in_batch=None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        compute_loss_context_manager = amp.autocast("cuda") if self._peft_has_been_casted_to_bf16 else nullcontext()
        with compute_loss_context_manager:
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # Make sure to move the loss to the device the original accumulating loss is at back in the `Trainer` class:
        loss = loss.to(self.args.device)
        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return loss, metrics

        return loss

    def generate_from_model_and_ref(self, model, batch: dict[str, torch.LongTensor]) -> tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        # If one uses `generate_during_eval` with peft + bf16, we need to explicitly call generate with
        # the torch cuda amp context manager as some hidden states are silently casted to full precision.
        generate_context_manager = amp.autocast("cuda") if self._peft_has_been_casted_to_bf16 else nullcontext()

        with generate_context_manager:
            policy_output = model.generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.max_length,
                do_sample=True,
                pad_token_id=self.processing_class.pad_token_id,
            )

            # if ref_output in batch use that otherwise use the reference model
            if "ref_output" in batch:
                ref_output = batch["ref_output"]
            else:
                if self.ref_model is None:
                    with self.null_ref_context():
                        ref_output = self.model.generate(
                            input_ids=batch["prompt_input_ids"],
                            attention_mask=batch["prompt_attention_mask"],
                            max_length=self.max_length,
                            do_sample=True,
                            pad_token_id=self.processing_class.pad_token_id,
                        )
                else:
                    ref_output = self.ref_model.generate(
                        input_ids=batch["prompt_input_ids"],
                        attention_mask=batch["prompt_attention_mask"],
                        max_length=self.max_length,
                        do_sample=True,
                        pad_token_id=self.processing_class.pad_token_id,
                    )

        policy_output = pad_to_length(policy_output, self.max_length, self.processing_class.pad_token_id)
        policy_output_decoded = self.processing_class.batch_decode(policy_output, skip_special_tokens=True)

        ref_output = pad_to_length(ref_output, self.max_length, self.processing_class.pad_token_id)
        ref_output_decoded = self.processing_class.batch_decode(ref_output, skip_special_tokens=True)

        return policy_output_decoded, ref_output_decoded

    def prediction_step(
            self,
            model: Union[PreTrainedModel, nn.Module],
            inputs: dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[list[str]] = None,
    ):
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = amp.autocast("cuda") if self._peft_has_been_casted_to_bf16 else nullcontext()

        with torch.no_grad(), prediction_context_manager:
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return loss.detach(), None, None

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = tuple(v.unsqueeze(dim=0) for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1).to(self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)

    def store_metrics(self, metrics: dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[list[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Overriding built-in evaluation loop to store metrics for each batch.
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """

        # Sample and save to game log if requested (for one batch to save time)
        if self.generate_during_eval:
            # Generate random indices within the range of the total number of samples
            num_samples = len(dataloader.dataset)
            random_indices = random.sample(range(num_samples), k=self.args.eval_batch_size)

            # Use dataloader.dataset.select to get the random batch without iterating over the DataLoader
            random_batch_dataset = dataloader.dataset.select(random_indices)
            random_batch = self.data_collator(random_batch_dataset)
            random_batch = self._prepare_inputs(random_batch)

            policy_output_decoded, ref_output_decoded = self.generate_from_model_and_ref(self.model, random_batch)

            self.log(
                {
                    "game_log": wandb.Table(
                        columns=["Prompt", "Policy", "Ref Model"],
                        rows=[
                            [prompt, pol[len(prompt):], ref[len(prompt):]]
                            for prompt, pol, ref in zip(
                                random_batch["prompt"], policy_output_decoded, ref_output_decoded
                            )
                        ],
                    )
                }
            )
            self.state.log_history.pop()

        # Base evaluation
        initial_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        return initial_output

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`dict[str, float]`):
                The values to log.
            start_time (`float` or `None`, *optional*, defaults to `None`):
                Start time of the training.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]

        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            return super().log(logs, start_time)
        else:  # transformers<=4.46
            return super().log(logs)


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, CustomDPOConfig))
    model_args, data_args, training_args = parser.parse()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    #last_checkpoint = get_checkpoint(training_args)
    last_checkpoint = None
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Load datasets
    ###############
    columns_to_keep = ["messages", "chosen", "rejected", "prompt", "completion", "label", "chosen_labels", "rejected_labels"]
    if training_args.chosen_distil_weight > 0 and training_args.distillation_weight > 0:
        columns_to_keep += ["teacher_chosen_probs"]
    if training_args.rejected_distil_weight > 0 and training_args.distillation_weight > 0:
        columns_to_keep += ["teacher_rejected_probs"]
    if training_args.adpa_weight > 0:
        columns_to_keep += ["rejected_margin_logp_every"]

    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=columns_to_keep
    )
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = [c for c in raw_datasets["train"].features if c not in [
        "chosen_labels", "rejected_labels", "ref_chosen_logps", "ref_rejected_logps",
        "teacher_chosen_probs", "teacher_rejected_probs", "rejected_margin_logp_every"]]

    #####################################
    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)

    #####################
    # Apply chat template
    #####################
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "dpo",
            "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
        },
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

    # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    for split in ["train", "test"]:
        if split in raw_datasets.keys():
            raw_datasets[split] = raw_datasets[split].rename_columns(
                {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
            )

    # Log a few random samples from the training set:
    for index in random.sample(range(len(raw_datasets["train"])), 3):
        logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")
        logger.info(f"Chosen sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['chosen']}")
        logger.info(f"Rejected sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['rejected']}")

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = model_args.model_name_or_path
    if is_adapter_model(model, model_args.model_revision) is True:
        logger.info(f"Loading SFT adapter for {model_args.model_name_or_path=}")
        peft_config = PeftConfig.from_pretrained(model_args.model_name_or_path, revision=model_args.model_revision)
        model_kwargs = dict(
            revision=model_args.base_model_revision,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
            device_map=get_kbit_device_map() if quantization_config is not None else None,
            quantization_config=quantization_config,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            **model_kwargs,
        )
        model = PeftModel.from_pretrained(
            base_model,
            model_args.model_name_or_path,
            revision=model_args.model_revision,
        )
        model_kwargs = None

    ref_model = model
    ref_model_kwargs = model_kwargs

    if model_args.use_peft is True:
        ref_model = None
        ref_model_kwargs = None

    #########################
    # Instantiate DPO trainer
    #########################
    trainer = DistillTrainer(
        model,
        ref_model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"] if "test" in raw_datasets.keys() else None,
        peft_config=get_peft_config(model_args),
    )
    # epoch=0 이면 인퍼런스만 수행하고 종료
    if training_args.num_train_epochs == 0:
        trainer.save_margin_inference(save_path="margin_inference.png")
        return
    ###############
    # Training loop
    ###############
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(data_args.dataset_mixer.keys()),
        "dataset_tags": list(data_args.dataset_mixer.keys()),
        "tags": ["alignment-handbook"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(raw_datasets["test"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    logger.info("*** Training complete! ***")


if __name__ == "__main__":
    main()
