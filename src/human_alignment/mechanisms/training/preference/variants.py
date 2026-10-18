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


class BetaDPO(NativePreferenceMethod):
    """beta-DPO (Wu et al., 2024): batch-level dynamic beta with beta-guided filtering."""
    method_id = backend_method = "beta_dpo"
    default_output_dir = "outputs/beta_dpo"


class EpsilonDPO(NativePreferenceMethod):
    """epsilon-DPO (Lee et al., 2025): per-pair KL penalty control via logit perturbation."""
    method_id = backend_method = "epsilon_dpo"
    default_output_dir = "outputs/epsilon_dpo"


class LogitsSAM(NativePreferenceMethod):
    """logits-SAM DPO (Luo et al., 2026): sharpness-aware minimization on the output layer."""
    method_id = backend_method = "logits_sam"
    default_output_dir = "outputs/logits_sam"


class RaDPO(NativePreferenceMethod):
    """Ra-DPO (Zhang et al., 2025): DPO with a nested-risk (CVaR) sequential risk ratio."""
    method_id = backend_method = "ra_dpo"
    default_output_dir = "outputs/ra_dpo"


class ADPO(NativePreferenceMethod):
    """Autoregressive DPO (Oi et al., 2026): Bradley-Terry comparisons per response segment."""
    method_id = backend_method = "adpo"
    default_output_dir = "outputs/adpo"
