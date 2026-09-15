"""Simple Preference Optimization through TRL's length-normalized objective."""

from human_alignment.config import SimPOConfig
from human_alignment.mechanisms.training.base import TrainingMethodBase


class SimPO(TrainingMethodBase):
    method_id = "simpo"
    backend_method = "dpo"
    dataset_kind = "preference"
    config_type = SimPOConfig
    supervision_categories = ("demonstrations_preferences",)
    mechanism_categories = ("preference_optimization",)
    default_output_dir = "outputs/simpo"
