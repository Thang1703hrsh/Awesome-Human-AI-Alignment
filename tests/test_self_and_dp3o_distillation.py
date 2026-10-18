import importlib.util
import unittest

from human_alignment import DP3O, PBSD, PBSDConfig, PreferenceOptimizationConfig
from human_alignment.exceptions import ConfigurationError
from human_alignment.supervision.answer_verification import (
    extract_boxed_answer,
    is_correct,
    is_tooluse_correct,
)

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
MATH_VERIFY_AVAILABLE = importlib.util.find_spec("math_verify") is not None


class ConfigAndTaxonomyTests(unittest.TestCase):
    def test_new_methods_sit_under_alignment_distillation(self):
        self.assertEqual(DP3O.mechanism_categories, ("alignment_distillation",))
        self.assertEqual(PBSD.mechanism_categories, ("alignment_distillation",))

    def test_dp3o_configuration_is_validated(self):
        self.assertEqual(PreferenceOptimizationConfig().dp3o_alpha, 0.2)
        with self.assertRaises(ConfigurationError):
            PreferenceOptimizationConfig(dp3o_alpha=1.5)
        with self.assertRaises(ConfigurationError):
            PreferenceOptimizationConfig(dp3o_reward_type="oracle")

    def test_pbsd_defaults_follow_reference_launcher(self):
        config = PBSDConfig()
        self.assertEqual((config.beta, config.learning_rate, config.lora_r, config.lora_alpha),
                         (0.1, 1e-5, 64, 128))
        self.assertEqual((config.temperature, config.top_p, config.top_k), (1.1, 0.95, 20))
        for invalid in ({"pairwise_loss_type": "hinge"}, {"task": "code"},
                        {"teacher_think_mode": "maybe"}, {"top_p": 0.0}):
            with self.assertRaises(ConfigurationError):
                PBSDConfig(**invalid)


class AnswerVerificationTests(unittest.TestCase):
    def test_boxed_extraction_uses_last_box_after_thinking(self):
        self.assertEqual(extract_boxed_answer(r"\boxed{\frac{1}{2}}"), r"\frac{1}{2}")
        self.assertEqual(extract_boxed_answer(r"<think>\boxed{3}</think>\boxed{4}"), "4")
        self.assertIsNone(extract_boxed_answer(r"\boxed{unclosed"))

    def test_tooluse_matches_gold_action(self):
        self.assertTrue(is_tooluse_correct("Plan.\nAction: search[ apple ]", "Action: search[apple]"))
        self.assertFalse(is_tooluse_correct("Action: search[pear]", "Action: search[apple]"))

    @unittest.skipUnless(MATH_VERIFY_AVAILABLE, "math-verify is required")
    def test_math_matches_symbolically(self):
        self.assertTrue(is_correct(r"so \boxed{\frac{1}{2}}", "0.5"))
        self.assertTrue(is_correct(r"\boxed{3:4}", r"\frac{3}{4}"))
        self.assertFalse(is_correct("no box", "7"))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
class LossTests(unittest.TestCase):
    def test_dp3o_mixes_hard_and_soft_targets(self):
        import torch
        import torch.nn.functional as F

        from human_alignment.mechanisms.training.preference.losses import dp3o_loss

        chosen, rejected = torch.tensor([2.0, -1.0]), torch.tensor([0.5, 1.0])
        diff = torch.tensor([3.0, -2.0])
        logits = 0.1 * (chosen - rejected)
        p = torch.sigmoid(diff)
        soft = -p * F.logsigmoid(logits) - (1 - p) * F.logsigmoid(-logits)
        expected = 0.2 * -F.logsigmoid(logits) + 0.8 * soft
        self.assertTrue(torch.allclose(dp3o_loss(chosen, rejected, diff, beta=0.1, alpha=0.2), expected))
        self.assertTrue(torch.allclose(dp3o_loss(chosen, rejected, None, beta=0.1, alpha=1.0),
                                       -F.logsigmoid(logits)))
        with self.assertRaises(ValueError):
            dp3o_loss(chosen, rejected, None, alpha=0.5)

    def test_collator_attaches_reward_differences(self):
        from human_alignment.mechanisms.training.preference.data import PreferenceCollator

        class Tokenizer:
            bos_token_id, eos_token_id, pad_token_id = None, 9, 0

            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [len(word) for word in text.split()]}

        collator = PreferenceCollator(Tokenizer(), 32, 8, reward_keys=("proxy_chosen_rewards",
                                                                        "proxy_rejected_rewards"))
        row = {"prompt": "p q", "chosen": "a b", "rejected": "c",
               "proxy_chosen_rewards": [1.5], "proxy_rejected_rewards": -0.5}
        self.assertEqual(collator([row])["reward_diff"].tolist(), [2.0])
        with self.assertRaises(ValueError):
            collator([{k: v for k, v in row.items() if k != "proxy_rejected_rewards"}])

    def test_pbsd_loss_is_teacher_referenced_dpo_with_gate(self):
        import torch
        import torch.nn.functional as F

        from human_alignment.mechanisms.training.distillation_losses import pbsd_loss

        tt, st, ts, ss = (torch.tensor(v) for v in ([-5., -2.], [-7., -3.], [-9., -1.], [-6., -2.]))
        delta = (tt - st) - (ts - ss)
        expected = -F.logsigmoid(-0.1 * delta)[0]
        gate = torch.tensor([1.0, 0.0])
        self.assertAlmostEqual(pbsd_loss(tt, st, ts, ss, gate=gate).item(), expected.item(), places=6)
        self.assertEqual(pbsd_loss(tt, st, ts, ss, gate=torch.zeros(2)).item(), 0.0)

    def test_completion_mask_keeps_through_first_eos(self):
        import torch

        from human_alignment.integrations.self_distillation import completion_mask

        # PAD (0) differs from EOS (2): stop at EOS, never score padding, even after truncation.
        completions = torch.tensor([[5, 6, 2, 0, 0], [5, 6, 7, 0, 0], [5, 6, 7, 8, 9]])
        self.assertEqual(completion_mask(completions, [2], 0).tolist(),
                         [[1, 1, 1, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
        # PAD equals EOS: the first EOS is kept, later EOS padding is not.
        self.assertEqual(completion_mask(torch.tensor([[5, 2, 2, 2]]), [2], 2).tolist(), [[1, 1, 0, 0]])


class PBSDPromptTests(unittest.TestCase):
    class Tokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
            return f"<think={enable_thinking}>" + messages[0]["content"]

    def test_teacher_sees_reference_and_student_does_not(self):
        from human_alignment.integrations.self_distillation import pbsd_prompts

        row = {"problem": "What is 1+1?", "solution": r"1+1=\boxed{2}"}
        student, teacher, gold = pbsd_prompts(row, self.Tokenizer(), PBSDConfig())
        self.assertEqual(gold, "2")
        self.assertTrue(student.startswith("<think=False>What is 1+1?"))
        self.assertNotIn("Reference Solution", student)
        self.assertTrue(teacher.startswith("<think=True>Problem: What is 1+1?"))
        self.assertIn("=== Reference Solution Begin ===\n1+1=\\boxed{2}", teacher)
        _, pinned, _ = pbsd_prompts(row, self.Tokenizer(), PBSDConfig(teacher_use_reference=True))
        self.assertIn("The correct final answer is exactly: 2.", pinned)


if __name__ == "__main__":
    unittest.main()
