"""Group Relative Policy Optimization with callable reward functions."""

from human_alignment.config import GRPOConfig
from human_alignment.mechanisms.training.base import TrainingMethodBase


class GRPO(TrainingMethodBase):
    method_id = "grpo"
    backend_method = "grpo"
    dataset_kind = "prompt"
    config_type = GRPOConfig
    supervision_categories = ("programmatic_verifiable_feedback",)
    mechanism_categories = ("reinforcement_learning",)
    default_output_dir = "outputs/grpo"
