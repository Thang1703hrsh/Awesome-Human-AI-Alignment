"""beta-DPO, epsilon-DPO, logits-SAM, Ra-DPO, and ADPO objectives."""

import importlib.util
import types
import unittest

from human_alignment import ADPO, BetaDPO, EpsilonDPO, LogitsSAM, PreferenceOptimizationConfig, RaDPO
from human_alignment.exceptions import ConfigurationError

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class FacadeTests(unittest.TestCase):
    def test_methods_sit_under_preference_optimization(self):
        for cls in (ADPO, BetaDPO, EpsilonDPO, LogitsSAM, RaDPO):
            self.assertEqual(cls.mechanism_categories, ("preference_optimization",))
            self.assertEqual(cls.backend_method, cls.method_id)

    def test_configuration_is_validated(self):
        for invalid in ({"beta_dpo_filter_ratio": 1.0}, {"epsilon": 0.0}, {"sam_rho": 0.0},
                        {"ra_dpo_variant": 3}, {"adpo_granularity": "token"}, {"adpo_segments": 0}):
            with self.assertRaises(ConfigurationError):
                PreferenceOptimizationConfig(**invalid)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
class LossTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch
        torch.manual_seed(0)
        self.logits = torch.randn(4, 6, 11)
        self.reference = self.logits + 0.2 * torch.randn(4, 6, 11)
        self.labels = torch.randint(0, 11, (4, 6))
        self.labels[:, :2] = -100

    def test_beta_dpo_uniform_when_policy_equals_reference(self):
        from human_alignment.mechanisms.training.preference.losses import beta_dpo_loss

        zeros = self.torch.zeros(5)
        state = {}
        losses = beta_dpo_loss(zeros, zeros, state, beta=0.1, filter_ratio=0.2)
        self.assertTrue(self.torch.isfinite(losses).all())
        self.assertAlmostEqual(losses.mean().item(), self.torch.log(self.torch.tensor(2.0)).item(), places=6)
        self.assertEqual(int(losses.gt(0).sum()), 4)

    def test_epsilon_beta_updates_once_per_optimizer_step(self):
        from human_alignment.mechanisms.training.preference.trainer import NativePreferenceTrainer

        sync = types.SimpleNamespace(sync_gradients=False)
        fake = types.SimpleNamespace(
            loss_state={"beta": 0.1, "step_sum": 0.0},
            args=types.SimpleNamespace(gradient_accumulation_steps=2),
            preference_config=PreferenceOptimizationConfig(epsilon=0.01),
            accelerator=types.SimpleNamespace(gradient_state=sync, device="cpu", gather=lambda x: x))
        fake.loss_state["last_steps"] = self.torch.tensor([1, 1])
        NativePreferenceTrainer._update_epsilon_beta(fake)
        self.assertEqual(fake.loss_state["beta"], 0.1)
        sync.sync_gradients = True
        fake.loss_state["last_steps"] = self.torch.tensor([1, -1])
        NativePreferenceTrainer._update_epsilon_beta(fake)
        self.assertAlmostEqual(fake.loss_state["beta"], 0.1 / (1 + 0.5 * 0.01))
        self.assertEqual(fake.loss_state["step_sum"], 0.0)

    def test_ra_dpo_quantile_matches_torch_quantile(self):
        from human_alignment.mechanisms.training.preference.losses import _linear_quantile

        values = self.torch.randn(3, 5, 17)
        for q in (0.05, 0.5, 0.95):
            self.assertTrue(self.torch.allclose(
                _linear_quantile(values, q), self.torch.quantile(values, q, dim=-1, keepdim=True)))

    def test_ra_dpo_reduces_to_dpo_without_risk(self):
        import torch.nn.functional as F

        from human_alignment.mechanisms.training.preference.losses import preference_loss, token_statistics

        config = PreferenceOptimizationConfig(beta=0.1)
        losses = preference_loss("ra_dpo", self.logits, self.logits.clone(), self.labels, config)
        self.assertTrue(self.torch.allclose(losses, -F.logsigmoid(self.torch.zeros(2))))
        ratio, *_ = token_statistics(self.logits, self.reference, self.labels)
        self.assertEqual(preference_loss("ra_dpo", self.logits, self.reference, self.labels, config).shape, (2,))

    def test_adpo_segments_partition_response_tokens(self):
        from human_alignment.mechanisms.training.preference.losses import adpo_segments

        ratio = self.torch.tensor([[1., 2., 3., 4., 5.], [1., 1., 0., 0., 0.]])
        mask = self.torch.tensor([[True] * 5, [True, True, False, False, False]])
        self.assertEqual(adpo_segments(ratio, mask, "adaptive", 2).tolist(), [[6., 9.], [1., 1.]])
        self.assertEqual(adpo_segments(ratio, mask, "adaptive", 2, reduce="mean").tolist(),
                         [[2., 4.5], [1., 1.]])
        self.assertEqual(adpo_segments(ratio, mask, "static", static_k=2).tolist(),
                         [[3., 7., 5.], [2., 0., 0.]])

    def test_adpo_pass_averages_over_released_width(self):
        import torch.nn.functional as F

        from human_alignment.mechanisms.training.preference.losses import preference_loss, token_statistics

        config = PreferenceOptimizationConfig(beta=1.0, adpo_granularity="pass")
        ratio, _, _, mask = token_statistics(self.logits, self.reference, self.labels)
        z = F.pad((ratio * mask)[:2] - (ratio * mask)[2:], (1, 0))
        expected = -F.logsigmoid(z).mean(-1)
        self.assertTrue(self.torch.allclose(
            preference_loss("adpo", self.logits, self.reference, self.labels, config), expected))


if __name__ == "__main__":
    unittest.main()
