import unittest
from types import SimpleNamespace

from human_alignment.config import DPOConfig, SFTConfig
from human_alignment.integrations.trl import TRLBackend


class FakeConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeTrainer:
    last_instance = None

    def __init__(self, model, args, train_dataset, **options):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.options = options
        self.processing_class = None
        self.saved_to = None
        self.train_options = None
        FakeTrainer.last_instance = self

    def train(self, **options):
        self.train_options = options
        return SimpleNamespace(metrics={"loss": 0.25, "text": "ignored"})

    def save_model(self, path):
        self.saved_to = path


class TRLIntegrationTests(unittest.TestCase):
    def test_dpo_arguments_are_forwarded_once(self):
        trl = SimpleNamespace(DPOConfig=FakeConfig, DPOTrainer=FakeTrainer)
        config = DPOConfig(output_dir="outputs/dpo", beta=0.2, loss_type="sigmoid")
        run = TRLBackend._dpo(trl, "model", ["row"], config, {})
        trainer = FakeTrainer.last_instance
        self.assertEqual(trainer.args.kwargs["beta"], 0.2)
        self.assertEqual(trainer.args.kwargs["loss_type"], "sigmoid")
        self.assertEqual(run.metrics["loss"], 0.25)
        self.assertEqual(trainer.saved_to, "outputs/dpo")

    def test_sft_arguments_are_forwarded(self):
        trl = SimpleNamespace(SFTConfig=FakeConfig, SFTTrainer=FakeTrainer)
        config = SFTConfig(output_dir="outputs/sft", packing=True)
        TRLBackend._sft(trl, "model", ["row"], config, {})
        trainer = FakeTrainer.last_instance
        self.assertTrue(trainer.args.kwargs["packing"])

    def test_training_can_resume_from_checkpoint(self):
        trl = SimpleNamespace(DPOConfig=FakeConfig, DPOTrainer=FakeTrainer)
        config = DPOConfig(resume_from_checkpoint="outputs/dpo/checkpoint-10")
        TRLBackend._dpo(trl, "model", ["row"], config, {})
        self.assertEqual(
            FakeTrainer.last_instance.train_options["resume_from_checkpoint"],
            "outputs/dpo/checkpoint-10",
        )


if __name__ == "__main__":
    unittest.main()
