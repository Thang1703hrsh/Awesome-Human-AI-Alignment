import unittest
from unittest.mock import patch

from human_alignment import DPO, DPOConfig, PreferenceExample, align
from human_alignment.exceptions import ConfigurationError
from human_alignment.mechanisms.training import DPO as TrainingDPO
from human_alignment.mechanisms.training.preference import DPO as PreferenceDPO
from human_alignment.results import AlignmentRun


class TrainingMethodTests(unittest.TestCase):
    def setUp(self):
        self.examples = [PreferenceExample("prompt", "chosen", "rejected")]

    def test_public_dpo_imports_resolve_to_preference_branch(self):
        self.assertIs(DPO, PreferenceDPO)
        self.assertIs(TrainingDPO, PreferenceDPO)

    @patch("human_alignment.integrations.trl.TRLBackend.train")
    def test_dpo_has_simple_public_api(self, train):
        train.return_value = AlignmentRun(model={"trained": True}, metrics={"loss": 0.1})
        run = DPO(
            model="model-id",
            dataset=self.examples,
            config=DPOConfig(output_dir="outputs/test"),
        ).train()
        self.assertTrue(run.model["trained"])
        self.assertEqual(run.method, "dpo")
        self.assertIn("preference_optimization", run.taxonomy_trace.mechanisms)
        self.assertEqual(train.call_args.kwargs["dataset"].kind, "preference")

    @patch("human_alignment.integrations.trl.TRLBackend.train")
    def test_align_dispatches_through_registry(self, train):
        train.return_value = AlignmentRun(model="trained")
        run = align(
            "dpo",
            model="model-id",
            dataset=self.examples,
            output_dir="outputs/test",
        )
        self.assertEqual(run.method, "dpo")

    def test_align_rejects_pipeline_only_adapter(self):
        with self.assertRaises(ConfigurationError):
            align(
                "alignment_distillation",
                model="model-id",
                dataset=self.examples,
            )


if __name__ == "__main__":
    unittest.main()
