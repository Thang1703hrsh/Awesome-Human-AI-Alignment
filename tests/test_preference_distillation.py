import importlib.util
import tempfile
import unittest
from unittest.mock import patch

from human_alignment import (
    ADPA,
    DCKD,
    PreferenceDistillation,
    PreferenceDistillationConfig,
    PreferenceDistillationExample,
    TVKD,
    VPD,
    align,
    collect_preference_distillation_data,
)
from human_alignment.exceptions import ConfigurationError, DatasetFormatError
from human_alignment.results import AlignmentRun
from human_alignment.supervision.datasets import prepare_dataset


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class PreferenceDistillationAPITests(unittest.TestCase):
    def test_typed_example_checks_score_count(self):
        with self.assertRaises(DatasetFormatError):
            PreferenceDistillationExample("prompt", ("a", "b"), (1.0,))

    def test_distillation_dataset_accepts_ranked_responses(self):
        bundle = prepare_dataset(
            [PreferenceDistillationExample("prompt", ("a", "b"), (1.0, 0.0))],
            "preference_distillation",
        )
        self.assertEqual(bundle.kind, "preference_distillation")
        self.assertEqual(bundle.data[0]["teacher_scores"], (1.0, 0.0))

    def test_empty_optional_scores_allow_online_teacher(self):
        from human_alignment.integrations.preference_distillation import _normalize_row

        class Tokenizer:
            chat_template = None

        row = {
            "prompt": "prompt",
            "responses": ["a", "b"],
            "teacher_scores": (),
        }
        normalized = _normalize_row(
            row,
            Tokenizer(),
            PreferenceDistillationConfig(objective="vpd"),
        )
        self.assertIsNone(normalized["teacher_scores"])

    def test_local_huggingface_directory_is_deferred_to_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = prepare_dataset(directory, "preference_distillation")
        self.assertEqual(bundle.data, directory)

    def test_invalid_objective_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            PreferenceDistillationConfig(objective="unknown")
        with self.assertRaises(ConfigurationError):
            PreferenceDistillationConfig(value_function="unknown")

    def test_method_defaults_preserve_migrated_recipes(self):
        dckd = DCKD()
        self.assertEqual(dckd.config.sft_weight, 1.0)
        self.assertEqual(dckd.config.chosen_kd_weight, 0.1)
        tvkd = TVKD()
        self.assertEqual(tvkd.config.value_function, "entropy")
        self.assertEqual(tvkd.config.chosen_kd_weight, 0.0001)
        adpa = ADPA()
        self.assertEqual(adpa.config.objective_weight, 1.6)
        self.assertEqual(adpa.config.sft_weight, 1.0)
        self.assertEqual(VPD().config.beta, 10.0)
        generic_dckd = PreferenceDistillation(config={"objective": "dckd"})
        self.assertEqual(generic_dckd.config.chosen_kd_weight, 0.1)

    def test_collects_multi_response_supervision(self):
        examples = collect_preference_distillation_data(
            ["prompt"],
            lambda prompt, count: [f"{prompt}-{index}" for index in range(count)],
            num_responses=3,
            scorer=lambda prompt, responses: range(len(responses)),
        )
        self.assertEqual(len(examples[0].responses), 3)
        self.assertEqual(examples[0].teacher_scores, (0.0, 1.0, 2.0))

    @patch(
        "human_alignment.integrations.preference_distillation."
        "PreferenceDistillationBackend.train"
    )
    def test_vpd_has_simple_public_api(self, train):
        train.return_value = AlignmentRun(model="trained")
        run = VPD(
            model="student",
            dataset=[PreferenceDistillationExample("prompt", ("a", "b"), (1.0, 0.0))],
        ).train()
        self.assertEqual(run.method, "vpd")
        self.assertEqual(train.call_args.kwargs["config"].objective, "vpd")
        self.assertEqual(run.output_dir, "outputs/vpd")

    @patch(
        "human_alignment.integrations.preference_distillation."
        "PreferenceDistillationBackend.train"
    )
    def test_align_dispatches_vpd(self, train):
        train.return_value = AlignmentRun(model="trained")
        run = align(
            "vpd",
            model="student",
            dataset=[{"prompt": "p", "responses": ["a", "b"], "teacher_scores": [1, 0]}],
        )
        self.assertEqual(run.method, "vpd")


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for loss tests")
class PreferenceDistillationLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch

    def test_vpd_prefers_teacher_order(self):
        from human_alignment.mechanisms.training.distillation_losses import vpd_loss

        teacher = self.torch.tensor([[2.0, 1.0, 0.0]])
        aligned = vpd_loss(self.torch.tensor([[2.0, 1.0, 0.0]]), teacher)
        reversed_loss = vpd_loss(self.torch.tensor([[0.0, 1.0, 2.0]]), teacher)
        self.assertLess(aligned.item(), reversed_loss.item())

    def test_ppd_is_zero_for_identical_distributions(self):
        from human_alignment.mechanisms.training.distillation_losses import ppd_loss

        scores = self.torch.tensor([[0.5, 0.0, -0.5]], requires_grad=True)
        loss = ppd_loss(scores, scores.detach()).mean()
        self.assertAlmostEqual(loss.item(), 0.0, places=6)
        loss.backward()
        self.assertIsNotNone(scores.grad)

    def test_compressed_kl_preserves_residual_mass(self):
        from human_alignment.mechanisms.training.distillation_losses import (
            compressed_forward_kl,
        )

        student = self.torch.tensor([[[0.6, 0.3, 0.1]]], requires_grad=True)
        labels = self.torch.tensor([[0]])
        teacher = [[{"indices": [0, 1], "values": [0.6, 0.3], "remaining_probs_sum": 0.1}]]
        loss = compressed_forward_kl(student, labels, teacher)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)
        loss.backward()
        self.assertIsNotNone(student.grad)

    def test_probability_compression_preserves_mass(self):
        from human_alignment.mechanisms.training.distillation_losses import (
            compress_probabilities,
        )

        compressed = compress_probabilities(self.torch.tensor([[2.0, 1.0, 0.0]]), top_k=2)
        mass = sum(compressed[0]["values"]) + compressed[0]["remaining_probs_sum"]
        self.assertAlmostEqual(mass, 1.0, places=6)

    def test_adpa_and_tvkd_losses_are_differentiable(self):
        from human_alignment.mechanisms.training.distillation_losses import (
            advantage_expectation_loss,
            tvkd_loss,
        )

        chosen = self.torch.tensor([[[0.7, 0.2, 0.1]]], requires_grad=True)
        rejected = self.torch.tensor([[[0.2, 0.7, 0.1]]], requires_grad=True)
        labels = self.torch.tensor([[0]])
        compressed = [[{"indices": [0, 1], "values": [0.7, 0.2], "remaining_probs_sum": 0.1}]]
        advantage = [[{"indices": [0, 1], "values": [1.0, -1.0]}]]
        loss = advantage_expectation_loss(rejected, labels, advantage)
        loss = loss + tvkd_loss(
            chosen,
            rejected,
            labels,
            labels,
            compressed,
            compressed,
        ).mean()
        self.assertTrue(self.torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(chosen.grad)
        self.assertIsNotNone(rejected.grad)


class PADFilterTests(unittest.TestCase):
    def test_confidence_filter_keeps_rows_with_mixed_selection(self):
        from math import log

        from human_alignment.integrations.preference_distillation import (
            _filter_mcq_confidence,
        )

        rows = [
            {"scores_mcq": [log(0.9), log(0.2)]},
            {"scores_mcq": [log(0.9), log(0.8)]},
            {"scores_mcq": [log(0.5), log(0.4)]},
            {"scores_mcq": [log(0.51), log(0.5)]},
        ]
        self.assertEqual(_filter_mcq_confidence(rows), [rows[0], rows[3]])
        with self.assertRaises(DatasetFormatError):
            _filter_mcq_confidence([{"teacher_scores": [0.0, 1.0]}])

    def test_confidence_filter_is_limited_to_pad_objectives(self):
        PreferenceDistillationConfig(objective="ppd", mcq_confidence_filter=True)
        with self.assertRaises(ConfigurationError):
            PreferenceDistillationConfig(objective="adpa", mcq_confidence_filter=True)


class _MCQTokenizer:
    """Tokenizer stub whose model reads the most recent MCQ query."""

    vocab = ["A", " A", "B", "None", "zz"]
    bos_token = None

    def __init__(self):
        self.last_query = ""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"

    def encode(self, text, add_special_tokens=False):
        return [4] * len(text.split())

    def decode(self, ids):
        if isinstance(ids, int):
            return self.vocab[ids]
        return " ".join("zz" for _ in ids)

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        import torch

        self.last_query = text
        return {"input_ids": torch.zeros((1, 1), dtype=torch.long)}


class _MCQModel:
    """Prefers whichever option contains the word 'good'."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids, use_cache=False):
        import torch

        query = self.tokenizer.last_query
        option_a = query.split('A) """', 1)[1].split('"""', 1)[0]
        option_b = query.split('B) """', 1)[1].split('"""', 1)[0]
        if ("good" in option_a) == ("good" in option_b):
            p_a = p_b = 0.4
        else:
            p_a, p_b = (0.7, 0.2) if "good" in option_a else (0.2, 0.7)
        probabilities = [p_a / 2, p_a / 2, p_b, 0.0, 1.0 - p_a - p_b]
        logits = torch.tensor(probabilities).clamp_min(1e-12).log()
        return type("Output", (), {"logits": logits.view(1, 1, -1)})()


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for PAD MCQ scoring")
class PADMCQScoringTests(unittest.TestCase):
    def test_scores_follow_reference_pairwise_aggregation(self):
        import torch

        from human_alignment.integrations.distillation_data import _pad_mcq_scores

        tokenizer = _MCQTokenizer()
        scores = _pad_mcq_scores(
            _MCQModel(tokenizer), tokenizer, "Question?", ["good", "bad", "good"], torch
        )
        eps = 1e-8
        # Pairs (0,1), (0,2), (1,2) with A as the first index; every pair adds eps per side.
        totals = [eps + (eps + 0.7) + (eps + 0.4), eps + (eps + 0.2) + (eps + 0.2),
                  eps + (eps + 0.4) + (eps + 0.7)]
        from math import log

        expected = [log(total / 2) for total in totals]
        for value, reference in zip(scores, expected):
            self.assertAlmostEqual(value, reference, places=6)
        self.assertIn("Output only the identifier of the best response (A, or B).",
                      tokenizer.last_query)
        self.assertIn("### Query\nQuestion?", tokenizer.last_query)


if __name__ == "__main__":
    unittest.main()
