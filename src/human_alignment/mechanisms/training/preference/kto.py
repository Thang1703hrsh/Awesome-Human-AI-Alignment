"""Kahneman-Tversky Optimization for desirable and undesirable examples."""

from human_alignment.config import KTOConfig
from human_alignment.mechanisms.training.base import TrainingMethodBase


class KTO(TrainingMethodBase):
    method_id = "kto"
    backend_method = "kto"
    dataset_kind = "kto"
    config_type = KTOConfig
    supervision_categories = ("demonstrations_preferences",)
    mechanism_categories = ("preference_optimization",)
    default_output_dir = "outputs/kto"
