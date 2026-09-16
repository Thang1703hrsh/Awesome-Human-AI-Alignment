
import inspect
import random
import warnings
from itertools import permutations
from collections import defaultdict
from contextlib import nullcontext
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, DataCollator, PreTrainedModel, PreTrainedTokenizerBase, Trainer
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from transformers.utils import is_torch_fx_proxy
from deepspeed.accelerator import get_accelerator

from simpo_config import SimPOConfig

from dataclasses import dataclass
from typing import Dict, Literal, Optional

from transformers import TrainingArguments

from trl.trainer.utils import (
    pad,
    DPODataCollatorWithPadding,
    disable_dropout_in_model,
    pad_to_length,
    peft_module_casting_to_bf16,
    trl_sanitze_kwargs_for_tagging,
)

from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
import wandb


@dataclass
class PDDataCollatorWithPadding:
    r"""
    DPO DataCollator class that pads the tokenized inputs to the maximum length of the batch.
    Args:
        pad_token_id (`int` defaults to 0):
            The tokenizer's pad_token_id.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            Whether or not you model has an encoder_decoder architecture.
    """

    pad_token_id: int = 0
    label_pad_token_id: int = -100
    is_encoder_decoder: Optional[bool] = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # first, pad everything to the same length
        padded_batch = {}
        for k in features[0].keys():
            if k.endswith(("_input_ids", "_attention_mask", "_labels", "_pixel_values")):
                if self.is_encoder_decoder:
                    raise NotImplementedError("Encoder-decoder models are not supported yet.")
                else:
                    # Set padding value based on the key
                    if k.endswith("_input_ids"):
                        if self.pad_token_id is None:
                            raise ValueError(
                                "Padding is enabled, but the tokenizer is not configured with a padding token."
                                " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                                " before calling the trainer."
                            )
                        padding_value = self.pad_token_id
                    elif k.endswith("_labels"):
                        padding_value = self.label_pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif k.endswith("_pixel_values"):
                        padding_value = 0  # TODO: check if this is correct
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    # Set padding side based on the key
                    if k in ["prompt_input_ids", "prompt_attention_mask"]:
                        padding_side = "left"
                    else:
                        padding_side = "right"

                    # Set the dtype
                    if k.endswith("_pixel_values"):
                        dtype = torch.float32  # will be downcasted if necessary by the Trainer
                    else:
                        dtype = torch.int64

                    # Convert to tensor and pad
                    if k.startswith('responses_'):
                        n_feat = len(features[0][k])
                        features_flat = [v for ex in features for v in ex[k]]
                        to_pad = [torch.tensor(ex, dtype=dtype) for ex in features_flat]
                        padded_batch[k] = pad(to_pad, padding_value=padding_value, padding_side="left")
                        padded_batch[k] = padded_batch[k].reshape(len(features), n_feat, padded_batch[k].shape[-1])
                    else:
                        to_pad = [torch.tensor(ex[k], dtype=dtype) for ex in features]
                        padded_batch[k] = pad(to_pad, padding_value=padding_value, padding_side=padding_side)
            elif k.startswith("scores") or k.endswith('repr'):
                # the cached reference model logprobs
                padded_batch[k] = torch.tensor([ex[k] for ex in features])
            else:
                padded_batch[k] = [ex[k] for ex in features]

        return padded_batch


class SimPOTrainer(Trainer):
    r"""
    Initialize SimPOTrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        args (`SimPOConfig`):
            The SimPO config arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        peft_config (`Dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return
            a dictionary string to metric values.
    """

    _tag_names = ["trl", "simpo"]

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: Optional[SimPOConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional[Dict] = None,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
    ):
        if args.model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_kwargs to the SimPOTrainer. But your model is already instantiated.")
        else:
            model_init_kwargs = args.model_init_kwargs
            model_init_kwargs["torch_dtype"] = (
                model_init_kwargs["torch_dtype"]
                if model_init_kwargs["torch_dtype"] in ["auto", None]
                else getattr(torch, model_init_kwargs["torch_dtype"])
            )

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the SimPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        # Initialize this variable to False. This helps tracking the case when `peft_module_casting_to_bf16`
        # has been called in order to properly call autocast if needed.
        self._peft_has_been_casted_to_bf16 = False

        if peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif peft_config is not None:
            # if model is a peft model and we have a peft_config, we merge and unload it first
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()

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

        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif args.is_encoder_decoder is None:
            raise ValueError("When no model is provided, you need to pass the parameter is_encoder_decoder.")
        else:
            self.is_encoder_decoder = args.is_encoder_decoder

        if self.is_encoder_decoder:
            self.decoder_start_token_id = model.config.decoder_start_token_id
            self.pad_token_id = model.config.pad_token_id

        if tokenizer is None:
            raise ValueError("tokenizer must be specified to tokenize a SimPO dataset.")
        if args.max_length is None:
            warnings.warn(
                "`max_length` is not set in the SimPOConfig's init"
                " it will default to `512` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_length = 512
        else:
            max_length = args.max_length
        if args.max_prompt_length is None:
            warnings.warn(
                "`max_prompt_length` is not set in the SimPOConfig's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_prompt_length = 128
        else:
            max_prompt_length = args.max_prompt_length

        if args.max_target_length is None and self.is_encoder_decoder:
            warnings.warn(
                "When using an encoder decoder architecture, you should set `max_target_length` in the SimPOConfig's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_target_length = 128
        else:
            max_target_length = args.max_target_length

        if data_collator is None:
            # data_collator = DPODataCollatorWithPadding(
            #     pad_token_id=tokenizer.pad_token_id,
            #     label_pad_token_id=args.label_pad_token_id,
            #     is_encoder_decoder=self.is_encoder_decoder,
            # )
            data_collator = PDDataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id,
                label_pad_token_id=args.label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        if args.disable_dropout:
            disable_dropout_in_model(model)

        self.max_length = max_length
        self.generate_during_eval = args.generate_during_eval
        self.label_pad_token_id = args.label_pad_token_id
        self.padding_value = args.padding_value if args.padding_value is not None else tokenizer.pad_token_id
        self.max_prompt_length = max_prompt_length
        self.max_batch_token_size = args.max_batch_token_size
        self.truncation_mode = args.truncation_mode
        self.max_target_length = max_target_length
        self.tokenizer_ = tokenizer
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma

        if args.loss_type in ["hinge"] and args.label_smoothing > 0:
            warnings.warn(
                "You are using a loss type that does not support label smoothing. Ignoring label_smoothing parameter."
            )

        self.gamma_beta_ratio = args.gamma_beta_ratio
        self.sft_weight = args.sft_weight
        self.label_smoothing = args.label_smoothing
        self.loss_type = args.loss_type

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # Compute that only on the main process for faster data processing.
        # see: https://github.com/huggingface/trl/pull/1255
        with PartialState().local_main_process_first():
            # tokenize the dataset
            train_dataset = train_dataset.map(self.tokenize_row, num_proc=args.dataset_num_proc)
            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(self.tokenize_row, num_proc=args.dataset_num_proc)
            # # filter the dataset
            # train_dataset = train_dataset.filter(self.filter, num_proc=args.dataset_num_proc)
            # if eval_dataset is not None:
            #     eval_dataset = eval_dataset.filter(self.filter, num_proc=args.dataset_num_proc)


        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

    # def filter(self, example):
    #     for i in example['responses_attention_mask']:
    #         if len(i) > 1200:
    #             return False
    #     return True
        # batch_token_size = sum(sum(i) for i in example['responses_attention_mask'])
        # return batch_token_size <= self.max_batch_token_size

    def build_tokenized_answer(self, prompt, answer):
        """
        Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`.
        It does ensure `enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]`.
        Reference:
            https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
        """

        full_tokenized = self.tokenizer_(prompt + answer, add_special_tokens=False)
        prompt_input_ids = self.tokenizer_(prompt, add_special_tokens=False)["input_ids"]

        answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
        answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids) :]

        # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
        full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])

        # Prepare input tokens for token by token comparison
        full_input_ids = np.array(full_tokenized["input_ids"])

        if len(full_input_ids) != len(full_concat_input_ids):
            raise ValueError("Prompt input ids and answer input ids should have the same length.")

        # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
        # can be merged together when tokenizing prompt+answer. This could result
        # on the last token from the prompt being different when tokenized on its own
        # vs when done as prompt+answer.
        response_token_ids_start_idx = len(prompt_input_ids)

        # If tokenized prompt is different than both prompt+answer, then it means the
        # last token has changed due to merging.
        if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
            response_token_ids_start_idx -= 1

        prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
        prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

        if len(prompt_input_ids) != len(prompt_attention_mask):
            raise ValueError("Prompt input ids and attention mask should have the same length.")

        answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
        answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

        return dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            input_ids=answer_input_ids,
            attention_mask=answer_attention_mask,
        )

    # def tokenize_row(self, feature, model: Optional[Union[PreTrainedModel, nn.Module]] = None) -> Dict:
    #     """Tokenize a single row from a SimPO specific dataset.

    #     At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
    #     in case the prompt + chosen or prompt + rejected responses is/are too long. First
    #         we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

    #     We also create the labels for the chosen/rejected responses, which are of length equal to
    #         the sum of the length of the prompt and the chosen/rejected response, with
    #         label_pad_token_id  for the prompt tokens.
    #     """
    #     batch = {}
    #     prompt = feature["prompt"]
    #     chosen = feature["chosen"]
    #     rejected = feature["rejected"]

    #     if not self.is_encoder_decoder:
    #         # Check issues below for more details
    #         #  1. https://github.com/huggingface/trl/issues/907
    #         #  2. https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
    #         #  3. https://github.com/LianjiaTech/BELLE/issues/337

    #         if not isinstance(prompt, str):
    #             raise ValueError(f"prompt should be an str but got {type(prompt)}")
    #         prompt_tokens = self.tokenizer_(prompt, add_special_tokens=False)
    #         prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

    #         if not isinstance(chosen, str):
    #             raise ValueError(f"chosen should be an str but got {type(chosen)}")
    #         chosen_tokens = self.build_tokenized_answer(prompt, chosen)

    #         if not isinstance(rejected, str):
    #             raise ValueError(f"rejected should be an str but got {type(rejected)}")
    #         rejected_tokens = self.build_tokenized_answer(prompt, rejected)

    #         # Last prompt token might get merged by tokenizer and
    #         # it should not be included for generation if that happens
    #         prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])

    #         chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
    #         rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
    #         prompt_len_input_ids = min(chosen_prompt_len_input_ids, rejected_prompt_len_input_ids)

    #         for k, v in prompt_tokens.items():
    #             prompt_tokens[k] = v[:prompt_len_input_ids]

    #         # Make sure prompts only have one different token at most an
    #         # and length only differs by 1 at most
    #         num_diff_tokens = sum(
    #             [a != b for a, b in zip(chosen_tokens["prompt_input_ids"], rejected_tokens["prompt_input_ids"])]
    #         )
    #         num_diff_len = abs(chosen_prompt_len_input_ids - rejected_prompt_len_input_ids)
    #         if num_diff_tokens > 1 or num_diff_len > 1:
    #             raise ValueError(
    #                 "Chosen and rejected prompt_input_ids might only differ on the "
    #                 "last token due to tokenizer merge ops."
    #             )

    #         # add BOS token to head of prompt. Avoid adding if it's already there
    #         bos_token_id = self.tokenizer_.bos_token_id
    #         if prompt_len_input_ids == 0 or bos_token_id != prompt_tokens["prompt_input_ids"][0]:
    #             prompt_tokens["prompt_input_ids"] = [bos_token_id] + prompt_tokens["prompt_input_ids"]
    #             prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
    #         if chosen_prompt_len_input_ids == 0 or bos_token_id != chosen_tokens["prompt_input_ids"][0]:
    #             chosen_tokens["prompt_input_ids"] = [bos_token_id] + chosen_tokens["prompt_input_ids"]
    #             chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
    #         if rejected_prompt_len_input_ids == 0 or bos_token_id != rejected_tokens["prompt_input_ids"][0]:
    #             rejected_tokens["prompt_input_ids"] = [bos_token_id] + rejected_tokens["prompt_input_ids"]
    #             rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]

    #         # add EOS token to end of answer. Avoid adding if it's already there
    #         eos_token_id = self.tokenizer_.eos_token_id
    #         if len(chosen_tokens["input_ids"]) == 0 or eos_token_id != chosen_tokens["input_ids"][-1]:
    #             chosen_tokens["input_ids"].append(eos_token_id)
    #             chosen_tokens["attention_mask"].append(1)
    #         if len(rejected_tokens["input_ids"]) == 0 or eos_token_id != rejected_tokens["input_ids"][-1]:
    #             rejected_tokens["input_ids"].append(eos_token_id)
    #             rejected_tokens["attention_mask"].append(1)

    #         longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

    #         # if combined sequence is too long, truncate the prompt
    #         for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
    #             if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
    #                 if self.truncation_mode == "keep_start":
    #                     for k in ["prompt_input_ids", "prompt_attention_mask"]:
    #                         answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
    #                 elif self.truncation_mode == "keep_end":
    #                     for k in ["prompt_input_ids", "prompt_attention_mask"]:
    #                         answer_tokens[k] = answer_tokens[k][-self.max_prompt_length :]
    #                 else:
    #                     raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

    #         # if that's still too long, truncate the response
    #         for answer_tokens in [chosen_tokens, rejected_tokens]:
    #             if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
    #                 for k in ["input_ids", "attention_mask"]:
    #                     answer_tokens[k] = answer_tokens[k][: self.max_length - self.max_prompt_length]

    #         # Create labels
    #         chosen_sequence_tokens = {
    #             k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
    #         }
    #         rejected_sequence_tokens = {
    #             k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
    #         }
    #         chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
    #         chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
    #             self.label_pad_token_id
    #         ] * len(chosen_tokens["prompt_input_ids"])
    #         rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
    #         rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [
    #             self.label_pad_token_id
    #         ] * len(rejected_tokens["prompt_input_ids"])

    #         for k, toks in {
    #             "chosen_": chosen_sequence_tokens,
    #             "rejected_": rejected_sequence_tokens,
    #             "": prompt_tokens,
    #         }.items():
    #             for type_key, tokens in toks.items():
    #                 if type_key == "token_type_ids":
    #                     continue
    #                 batch[f"{k}{type_key}"] = tokens

    #     else:
    #         chosen_tokens = self.tokenizer_(
    #             chosen, truncation=True, max_length=self.max_target_length, add_special_tokens=True
    #         )
    #         rejected_tokens = self.tokenizer_(
    #             rejected, truncation=True, max_length=self.max_target_length, add_special_tokens=True
    #         )
    #         prompt_tokens = self.tokenizer_(
    #             prompt, truncation=True, max_length=self.max_prompt_length, add_special_tokens=True
    #         )

    #         batch["chosen_labels"] = chosen_tokens["input_ids"]
    #         batch["rejected_labels"] = rejected_tokens["input_ids"]
    #         batch["prompt_input_ids"] = prompt_tokens["input_ids"]
    #         batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]

    #         if model is not None and hasattr(model, "prepare_decoder_input_ids_from_labels"):
    #             batch["rejected_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
    #                 labels=torch.tensor(batch["rejected_labels"])
    #             )
    #             batch["chosen_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
    #                 labels=torch.tensor(batch["chosen_labels"])
    #             )

    #     return batch

    def tokenize_row(self, feature, model: Optional[Union[PreTrainedModel, nn.Module]] = None) -> Dict:
        """Tokenize a single row from a PD specific dataset.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
        in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}
        prompt = feature["prompt"]
        responses = feature['responses']

        if not self.is_encoder_decoder:
            # Check issues below for more details
            #  1. https://github.com/huggingface/trl/issues/907
            #  2. https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
            #  3. https://github.com/LianjiaTech/BELLE/issues/337

            if not isinstance(prompt, str):
                raise ValueError(f"prompt should be an str but got {type(prompt)}")
            prompt_tokens = self.tokenizer_(prompt, add_special_tokens=False)
            prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

            responses_tokens = []
            for res in responses:
                if not isinstance(res, str):
                    raise ValueError(f"rejected should be an str but got {type(res)}")
                responses_tokens.append(self.build_tokenized_answer(prompt, res))
            # Last prompt token might get merged by tokenizer and
            # it should not be included for generation if that happens
            prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])

            prompt_len_input_ids = min([len(res["prompt_input_ids"]) for res in responses_tokens])

            for k, v in prompt_tokens.items():
                prompt_tokens[k] = v[:prompt_len_input_ids]

            # Make sure prompts only have one different token at most an
            # and length only differs by 1 at most
            num_diff_tokens = len(set([tuple(res["prompt_input_ids"]) for res in responses_tokens]))
            num_diff_len = len(set([len(res["prompt_input_ids"]) for res in responses_tokens]))
            if num_diff_tokens > 1 or num_diff_len > 1:
                raise ValueError(
                    "Chosen and rejected prompt_input_ids might only differ on the "
                    "last token due to tokenizer merge ops."
                )
            
            # add BOS token to head of prompt. Avoid adding if it's already there
            # prompt_tokens, chosen_tokens, rejected_tokens = add_bos_token_if_needed(
            #     self.tokenizer_.bos_token_id,
            #     prompt_len_input_ids,
            #     prompt_tokens,
            #     chosen_prompt_len_input_ids,
            #     chosen_tokens,
            #     rejected_prompt_len_input_ids,
            #     rejected_tokens,
            # )

            # add EOS token to end of answer. Avoid adding if it's already there
            # chosen_tokens, rejected_tokens = add_eos_token_if_needed(
            #     self.tokenizer_.eos_token_id, chosen_tokens, rejected_tokens
            # )

            # longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))
            longer_response_length = max(len(res["input_ids"]) for res in responses_tokens)

            for answer_tokens in [*responses_tokens, prompt_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    if self.truncation_mode == "keep_start":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
                    elif self.truncation_mode == "keep_end":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][-self.max_prompt_length :]
                    else:
                        raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

            # if that's still too long, truncate the response
            for answer_tokens in responses_tokens:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    for k in ["input_ids", "attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][: self.max_length - self.max_prompt_length]

            # Create labels
            responses_sequence_tokens = []
            for res in responses_tokens:
                res_seq_tokens = {
                    k: res[f"prompt_{k}"] + res[k] for k in ["input_ids", "attention_mask"]
                }
                res_seq_tokens["labels"] = res_seq_tokens["input_ids"][:]
                res_seq_tokens["labels"][: len(res["prompt_input_ids"])] = [
                    self.label_pad_token_id
                ] * len(res["prompt_input_ids"])
                responses_sequence_tokens.append(res_seq_tokens)

            batch = {
                **prompt_tokens,
                'responses_input_ids': [],
                'responses_attention_mask': [],
                'responses_labels': [],
            }
            for res in responses_sequence_tokens:
                for k, v in res.items():
                    batch[f'responses_{k}'].append(v)
        else:
            raise NotImplementedError

        return batch

    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
            is_encoder_decoder: Whether the model is an encoder-decoder model.
            label_pad_token_id: The label pad token id.
            padding_value: The padding value to use for the concatenated inputs_ids.
            device: The device for the concatenated inputs.

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        concatenated_batch = {}

        if is_encoder_decoder:
            max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
        else:
            max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)

        if is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1).to(device=device)
            concatenated_batch["concatenated_attention_mask"] = (
                batch["prompt_attention_mask"].repeat(2, 1).to(device=device)
            )

        return concatenated_batch

    # def simpo_loss(
    #     self,
    #     policy_chosen_logps: torch.FloatTensor,
    #     policy_rejected_logps: torch.FloatTensor,
    # ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    #     """Compute the SimPO loss for a batch of policy model log probabilities.

    #     Args:
    #         policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
    #         policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

    #     Returns:
    #         A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
    #         The losses tensor contains the SimPO loss for each example in the batch.
    #         The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    #     """
    #     pi_logratios = policy_chosen_logps - policy_rejected_logps
    #     pi_logratios = pi_logratios.to(self.accelerator.device)
    #     logits = pi_logratios - self.gamma_beta_ratio

    #     if self.loss_type == "sigmoid":
    #         losses = (
    #             -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
    #             - F.logsigmoid(-self.beta * logits) * self.label_smoothing
    #         )
    #     elif self.loss_type == "hinge":
    #         losses = torch.relu(1 - self.beta * logits)
    #     else:
    #         raise ValueError(
    #             f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge']"
    #         )

    #     chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
    #     rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()

    #     return losses, chosen_rewards, rejected_rewards

    @staticmethod
    def plackett_luce_logprob_with_rankings(scores, rankings):
        """
        计算给定分数矩阵和特定排名的 log 概率。

        Args:
            scores (torch.Tensor): 大小为 [b, n] 的分数矩阵, b 为 batch size, n 为分数个数。
            rankings (torch.Tensor): 大小为 [b, n] 的排名矩阵, 表示每个样本的特定排名。

        Returns:
            log_probs (torch.Tensor): 大小为 [b, n] 的 log 概率矩阵, 表示每个位置的 log 概率。
        """
        b, n = scores.shape
        # 初始化 log_probs，用于存储每个位置的 log 概率
        log_probs = scores.new_zeros((b, n))

        # 创建一个掩码矩阵，用于跟踪每个样本中未被选中的元素
        available_items_mask = torch.ones((b, n), dtype=torch.bool, device=scores.device)

        for k in range(n):
            # 当前选中元素的索引（需要确保索引为 long 类型）
            idx = rankings[:, k].unsqueeze(1).long()  # [b, 1]

            # 获取当前选中元素的分数
            chosen_scores = torch.gather(scores, 1, idx)  # [b, 1]

            # 计算分母部分：未被选中元素的分数的 exp 求和
            denom_scores = scores.masked_fill(~available_items_mask, float('-inf'))  # [b, n]
            denom = denom_scores.logsumexp(dim=1, keepdim=True)  # [b, 1]

            # 计算当前位置的 log 概率
            log_probs[:, k:k+1] = chosen_scores - denom  # [b, 1]

            # 更新 available_items_mask，将当前选中元素置为 False
            available_items_mask.scatter_(1, idx, False)

        return log_probs

    @staticmethod
    def plackett_luce_logprob(scores):
        """
        计算给定分数矩阵的排名概率分布。
        
        Args:
            scores (torch.Tensor): 大小为 [b, n] 的分数矩阵, b 为 batch size, n 为分数个数。

        Returns:
            rankings_probs (torch.Tensor): 大小为 [b, n!] 的排名概率矩阵，表示每个样本的排名分布。
        """
        b, n = scores.shape

        # 排列所有可能的排名，并使用 new_tensor 确保在同一设备上
        all_rankings = scores.new_tensor(list(permutations(range(n))), dtype=torch.long)  # 使用 scores.new_tensor
        num_rankings = all_rankings.shape[0]

        # 扩展 scores 和 rankings 用于批处理计算
        scores_expanded = scores.unsqueeze(1).expand(b, num_rankings, n)  # [b, n!, n]
        rankings_expanded = all_rankings.unsqueeze(0).expand(b, num_rankings, n)  # [b, n!, n]

        # 使用 new_zeros 来初始化 rankings_probs
        log_rankings_probs = scores.new_zeros((b, num_rankings))

        for k in range(n):
            # 根据排名提取当前第 k 位的分数
            chosen_items = torch.gather(scores_expanded, 2, rankings_expanded[:, :, k].unsqueeze(2)).squeeze(2)
            
            # 通过构建掩码来选择未被选中的元素，使用 new_ones 来创建掩码
            mask = rankings_expanded[:, :, :k]  # 已选择的元素在前 k-1 位置
            available_items_mask = scores.new_ones((b, num_rankings, n), dtype=torch.bool).scatter(2, mask, False)
            
            # 计算分母时忽略已选项
            denom = scores_expanded.masked_fill(~available_items_mask, -1e8).logsumexp(-1)
            
            # 累积概率分数
            log_rankings_probs += chosen_items - denom

        return log_rankings_probs


    @staticmethod
    def jenson_shannon_divergence(
        n1_logits: torch.FloatTensor,
        n2_logits: torch.FloatTensor,
        log: bool = True,
    ) -> torch.FloatTensor:
        if not log:
            n1_logprobs = n1_logits.log_softmax(-1)
            n2_logprobs = n2_logits.log_softmax(-1)
        else:
            n1_logprobs, n2_logprobs = n1_logits, n2_logits
        m_logprobs = torch.cat((n1_logprobs, n2_logprobs), 0).logsumexp(0).unsqueeze(0) + n1_logprobs.new_tensor(0.5).log()

        loss = F.kl_div(n1_logprobs, m_logprobs, reduction="none", log_target=True).sum(-1)
        loss += F.kl_div(n2_logprobs, m_logprobs, reduction="none", log_target=True).sum(-1)

        return 0.5 * loss

    def pd_loss(
        self,
        student_logps: torch.FloatTensor,
        teacher_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the PD loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the PD loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        batch_size = student_logps.shape[0]
        pref_idx = teacher_logps.argmax(1)
        dispref_idx = teacher_logps.argmin(1)

        if self.loss_type == "pd":
            # N = 2
            # n_indices = max_variance_selection_indices(ref_logps, N)
            student_scores = (self.beta * student_logps).to(self.accelerator.device)
            teacher_scores = (self.beta * teacher_logps).to(self.accelerator.device)
            # cal preference distribution
            student_pref = self.plackett_luce_logprob(student_scores)
            teacher_pref = self.plackett_luce_logprob(teacher_scores)
            # policy_pref = policy_scores.log_softmax(dim=-1)
            # ref_pref = ref_scores.log_softmax(dim=-1)
            # cal js div
            loss = self.jenson_shannon_divergence(student_pref, teacher_pref, log=True).mean()

        elif self.loss_type == "rd":
            rankings = teacher_logps.argsort(-1, descending=True)
            student_scores = (self.beta * student_logps).to(self.accelerator.device)
            loss = - self.plackett_luce_logprob_with_rankings(student_scores, rankings).sum(-1).mean()

        elif self.loss_type == "simpo":
            logits = (student_logps[range(batch_size), pref_idx] - student_logps[range(batch_size), dispref_idx]).to(self.accelerator.device)
            logits = logits - 0.3

            loss = -F.logsigmoid(self.beta * logits).mean()

        policy_rewards = self.beta * (student_logps.to(self.accelerator.device)).detach()

        chosen_rewards = policy_rewards[:, pref_idx]
        rejected_rewards = policy_rewards[:, dispref_idx]

        return loss, chosen_rewards, rejected_rewards

    @staticmethod
    def pad_convert(tensor, padding_value=-100, mode='left-to-right'):
        """
        将一个填充张量在 left-pad 和 right-pad 之间转换。

        参数:
        - tensor (torch.Tensor): 输入张量，形状为 [b, s]。
        - padding_value (int): 用于填充的值，默认为 -100。
        - mode (str): 转换模式，'left-to-right' 表示从 left-pad 到 right-pad，
                    'right-to-left' 表示从 right-pad 到 left-pad。

        返回:
        - torch.Tensor: 转换后的张量。
        """
        # 获取有效数据长度
        valid_lengths = (tensor != padding_value).sum(dim=1)
        
        # 创建结果张量，并用 padding_value 填充
        result = torch.full_like(tensor, padding_value).to(tensor.device)

        # 生成索引矩阵
        s = tensor.size(1)
        arange = torch.arange(s).expand(tensor.size(0), s).to(tensor.device)

        if mode == 'left-to-right':
            # 从 left-pad 转为 right-pad
            mask = arange < valid_lengths.unsqueeze(1)
            result[mask] = tensor[tensor != padding_value]
        elif mode == 'right-to-left':
            # 从 right-pad 转为 left-pad
            mask = arange >= (s - valid_lengths.unsqueeze(1))
            result[mask] = tensor[tensor != padding_value]
        else:
            raise ValueError("Invalid mode. Use 'left-to-right' or 'right-to-left'.")
        
        return result

    # def concatenated_forward(
    #     self,
    #     model: nn.Module,
    #     batch: Dict[str, Union[List, torch.LongTensor]],
    #     output_repr: bool = False,
    # ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    #     """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

    #     We do this to avoid doing two forward passes, because it's faster for FSDP.
    #     """
    #     model_kwargs = {}

    #     batch_size, n_res, seq_len = batch['responses_input_ids'].shape
    #     input_ids_flat = batch['responses_input_ids'].reshape(batch_size * n_res, -1)
    #     mask_flat = batch['responses_attention_mask'].reshape(batch_size * n_res, -1)
    #     labels_flat = batch['responses_labels'].reshape(batch_size * n_res, -1)

    #     outputs = model(
    #         input_ids=input_ids_flat,
    #         attention_mask=mask_flat,
    #         output_hidden_states=output_repr,
    #         **model_kwargs,
    #     )

    #     # drop the same tokens in the beginning of each response
    #     # labels_right_pad = self.pad_convert(labels_flat, padding_value=self.label_pad_token_id, mode='left-to-right')
    #     # mask_different = (labels_right_pad != labels_right_pad[0]).any(dim=0)
    #     # cumulative_true = torch.cumsum(mask_different, dim=0) > 0
    #     # mask_different = torch.where(cumulative_true, torch.tensor(True, device=mask_different.device), mask_different)
    #     # mask_index = -123
    #     # labels_right_pad_mask = torch.where(mask_different, labels_right_pad, mask_index)
    #     # labels_left_pad_mask = self.pad_convert(labels_right_pad_mask, padding_value=self.label_pad_token_id, mode='right-to-left')
    #     # labels_flat = torch.where(labels_left_pad_mask==mask_index, self.label_pad_token_id, labels_left_pad_mask)

    #     all_logits = outputs.logits
    #     reprs = outputs.hidden_states[-1][range(batch_size * n_res), mask_flat.sum(-1) - 1]
    #     all_logps = self.get_batch_logps(
    #         all_logits,
    #         labels_flat,
    #         average_log_prob=self.loss_type not in ['plad'],
    #         is_encoder_decoder=self.is_encoder_decoder,
    #         label_pad_token_id=self.label_pad_token_id,
    #     )
    #     logits = all_logits.reshape(batch_size, n_res, seq_len, -1)
    #     logps = all_logps.reshape(batch_size, n_res)
    #     reprs = reprs.reshape(batch_size, n_res, -1)

    #     if output_repr:
    #         return logits, logps, reprs
    #     return logits, logps,

    def concatenated_forward(
        self,
        model: nn.Module,
        batch: Dict[str, Union[List, torch.LongTensor]],
        output_repr: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        To prevent memory overflow, process the inputs in smaller inner batches if necessary.
        """
        model_kwargs = {}
        inner_batch_size = 4  # Set the inner batch size

        batch_size, n_res, seq_len = batch['responses_input_ids'].shape
        total_size = batch_size * n_res

        input_ids_flat = batch['responses_input_ids'].reshape(total_size, -1)
        mask_flat = batch['responses_attention_mask'].reshape(total_size, -1)
        labels_flat = batch['responses_labels'].reshape(total_size, -1)

        logits_list = []
        logps_list = []
        reprs_list = []

        if total_size <= inner_batch_size:
            # Process all inputs at once
            outputs = model(
                input_ids=input_ids_flat,
                attention_mask=mask_flat,
                output_hidden_states=output_repr,
                **model_kwargs,
            )

            all_logits = outputs.logits
            reprs = outputs.hidden_states[-1][range(total_size), mask_flat.sum(-1) - 1]
            all_logps = self.get_batch_logps(
                all_logits,
                labels_flat,
                average_log_prob=self.loss_type not in ['plad'],
                is_encoder_decoder=self.is_encoder_decoder,
                label_pad_token_id=self.label_pad_token_id,
            )

            logits = all_logits.reshape(batch_size, n_res, seq_len, -1)
            logps = all_logps.reshape(batch_size, n_res)
            reprs = reprs.reshape(batch_size, n_res, -1)

        else:
            # Process inputs in smaller inner batches
            for start_idx in range(0, total_size, inner_batch_size):
                end_idx = min(start_idx + inner_batch_size, total_size)
                input_ids_chunk = input_ids_flat[start_idx:end_idx]
                mask_chunk = mask_flat[start_idx:end_idx]
                labels_chunk = labels_flat[start_idx:end_idx]

                outputs = model(
                    input_ids=input_ids_chunk,
                    attention_mask=mask_chunk,
                    output_hidden_states=output_repr,
                    **model_kwargs,
                )

                all_logits = outputs.logits
                reprs = outputs.hidden_states[-1][range(end_idx - start_idx), mask_chunk.sum(-1) - 1]
                all_logps = self.get_batch_logps(
                    all_logits,
                    labels_chunk,
                    average_log_prob=self.loss_type not in ['plad'],
                    is_encoder_decoder=self.is_encoder_decoder,
                    label_pad_token_id=self.label_pad_token_id,
                )

                logits_list.append(all_logits)
                logps_list.append(all_logps)
                if output_repr:
                    reprs_list.append(reprs)

            # Concatenate all chunks
            all_logits = torch.cat(logits_list, dim=0)
            all_logps = torch.cat(logps_list, dim=0)
            logits = all_logits.reshape(batch_size, n_res, seq_len, -1)
            logps = all_logps.reshape(batch_size, n_res)

            if output_repr:
                all_repr = torch.cat(reprs_list, dim=0)
                reprs = all_repr.reshape(batch_size, n_res, -1)

        if output_repr:
            return logits, logps, reprs
        return logits, logps

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = True,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)


    def get_batch_samples(self, epoch_iterator, num_batches):
        batch_samples = []
        num_items_in_batch = None
        for _ in range(num_batches):
            try:
                batch_samples += [next(epoch_iterator)]
            except StopIteration:
                break
        if len(batch_samples) > 0 and 'responses_labels' in batch_samples[0]:
            # For now we don't support object detection
            try:
                num_items_in_batch = sum(
                    [data_batch['responses_labels'][..., 1:].ne(-100).sum().item() for data_batch in batch_samples]
                )
            except TypeError:
                pass
        return batch_samples, num_items_in_batch

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        num_items_in_batch: Optional[int] = None,
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the SimPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        prefix = "eval_" if train_eval == "eval" else ""

        forward_output = self.concatenated_forward(model, batch, output_repr=True)
        logits, logps = forward_output[:2]
        scores = batch['scores_avglogp'] * self.alpha + batch['scores_mcq'] * (1 - self.alpha) if self.loss_type == "pd" else batch['scores_mcq']
        loss_output = self.pd_loss(logps, scores)
        loss, chosen_rewards, rejected_rewards = loss_output[:3]

        if num_items_in_batch is not None:
            loss = loss * (batch['responses_labels'][..., 1:].ne(-100).sum().item() / num_items_in_batch)

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()

        return loss, metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[int] = None,
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )

        # compute_loss_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        # with compute_loss_context_manager():
        loss, metrics = self.get_batch_loss_metrics(model, inputs, num_items_in_batch, train_eval="train")

        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with torch.no_grad(), prediction_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = tuple(v.unsqueeze(dim=0) for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1).to(self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
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

            policy_output_decoded = self.get_batch_samples(self.model, random_batch)

            self.log(
                {
                    "game_log": wandb.Table(
                        columns=["Prompt", "Policy"],
                        rows=[
                            [prompt, pol[len(prompt) :]]
                            for prompt, pol in zip(random_batch["prompt"], policy_output_decoded)
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

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)

    @wraps(Trainer.push_to_hub)
    def push_to_hub(self, commit_message: Optional[str] = "End of training", blocking: bool = True, **kwargs) -> str:
        """
        Overwrite the `push_to_hub` method in order to force-add the tag "simpo" when pushing the
        model on the Hub. Please refer to `~transformers.Trainer.push_to_hub` for more details.
        """
        kwargs = trl_sanitze_kwargs_for_tagging(model=self.model, tag_names=self._tag_names, kwargs=kwargs)

        return super().push_to_hub(commit_message=commit_message, blocking=blocking, **kwargs)
