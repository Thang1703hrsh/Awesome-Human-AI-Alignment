"""Public method facades; backend dependencies remain lazy."""

from human_alignment.config import IPOConfig, PreferenceOptimizationConfig
from human_alignment.mechanisms.training.base import TrainingMethodBase
from human_alignment.mechanisms.training.preference.dpo import DPO


class IPO(DPO):
    method_id = "ipo"
    config_type = IPOConfig
    default_output_dir = "outputs/ipo"


class NativePreferenceMethod(TrainingMethodBase):
    dataset_kind = "preference"
    config_type = PreferenceOptimizationConfig
    supervision_categories = ("demonstrations_preferences",)
    mechanism_categories = ("preference_optimization",)

    @staticmethod
    def _backend():
        from human_alignment.integrations.preference import PreferenceBackend

        return PreferenceBackend()


class BPO(NativePreferenceMethod):
    """Bregman preference optimization (Kim et al., 2025)."""
    method_id = backend_method = "bpo"
    default_output_dir = "outputs/bpo"


class TDPO(NativePreferenceMethod):
    method_id = backend_method = "tdpo"
    default_output_dir = "outputs/tdpo"


class TISDPO(NativePreferenceMethod):
    method_id = backend_method = "tis_dpo"
    default_output_dir = "outputs/tis_dpo"


class TIDPO(NativePreferenceMethod):
    """Released TDPO-based TI-DPO variant, including attribution and triplet loss."""
    method_id = backend_method = "ti_dpo"
    default_output_dir = "outputs/ti_dpo"


class TBPOQ(NativePreferenceMethod):
    method_id = backend_method = "tbpo_q"
    default_output_dir = "outputs/tbpo_q"


class TBPOA(NativePreferenceMethod):
    method_id = backend_method = "tbpo_a"
    default_output_dir = "outputs/tbpo_a"
