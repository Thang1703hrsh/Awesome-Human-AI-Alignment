"""PPO facade for TRL's experimental PPO backend."""

from human_alignment.config import PPOConfig
from human_alignment.mechanisms.training.base import TrainingMethodBase


class PPO(TrainingMethodBase):
    method_id = "ppo"
    backend_method = "ppo"
    dataset_kind = "prompt"
    config_type = PPOConfig
    supervision_categories = ("reliable_scalable_oversight",)
    mechanism_categories = ("reinforcement_learning",)
    default_output_dir = "outputs/ppo"
