"""Direct Preference Optimization with a simple public API."""

from human_alignment.config import DPOConfig
from human_alignment.mechanisms.training.base import TrainingMethodBase


class DPO(TrainingMethodBase):
    method_id = "dpo"
    backend_method = "dpo"
    dataset_kind = "preference"
    config_type = DPOConfig
    supervision_categories = ("demonstrations_preferences",)
    mechanism_categories = ("preference_optimization",)
    default_output_dir = "outputs/dpo"
