"""Supervised fine-tuning with instruction or demonstration data."""

from human_alignment.config import SFTConfig
from human_alignment.mechanisms.training.base import TrainingMethodBase


class SFT(TrainingMethodBase):
    method_id = "sft"
    backend_method = "sft"
    dataset_kind = "instruction"
    config_type = SFTConfig
    supervision_categories = ("demonstrations_preferences",)
    mechanism_categories = ("supervised_alignment",)
    default_output_dir = "outputs/sft"
