import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from human_alignment import (
    AlignmentRun,
    CheckpointConfig,
    DistillationPreparationConfig,
    EvaluationCase,
    InteractionContext,
    load_checkpoint,
    prepare_preference_distillation_dataset,
)
from human_alignment.exceptions import ConfigurationError, DatasetFormatError
from human_alignment.results import AssuranceReport
from human_alignment.supervision.preference_distillation import (
    attach_generated_response,
    build_adpa_record,
    common_parent_groups,
    merge_compressed_probabilities,
)


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_config_validates_adapter_merge(self):
        with self.assertRaises(ConfigurationError):
            CheckpointConfig(merge_adapter=True)
        with self.assertRaises(ConfigurationError):
            CheckpointConfig(task="unknown")

    @patch("human_alignment.integrations.transformers.load_transformers_model")
    def test_load_checkpoint_forwards_typed_configuration(self, loader):
        loader.return_value = AlignmentRun(model="loaded")
        run = load_checkpoint(
            "checkpoint",
            config={"device_map": "auto", "torch_dtype": "bfloat16"},
        )
        self.assertEqual(run.model, "loaded")
        self.assertEqual(loader.call_args.kwargs["config"].device_map, "auto")

    def test_loaded_run_can_be_evaluated(self):
        class Evaluator:
            def assess(self, model, cases, *, baseline_model=None):
                return AssuranceReport("test", {"count": float(len(cases))}, True)

        run = AlignmentRun(model="checkpoint")
        report = run.evaluate(
            Evaluator(), [EvaluationCase(InteractionContext("prompt"))]
        )
        self.assertTrue(report.passed)


class PreparationTests(unittest.TestCase):
    def test_pair_builders_preserve_records(self):
        record = {"prompt": "p", "chosen": "a", "rejected": "b", "id": 1}
        distribution = [{"indices": [0], "values": [0.9], "remaining_probs_sum": 0.1}]
        merged = merge_compressed_probabilities(record, distribution, distribution)
        self.assertEqual(merged["id"], 1)
        self.assertIn("teacher_chosen_probs", merged)
        advantage = build_adpa_record(record, distribution, distribution)
        self.assertAlmostEqual(advantage["rejected_margin_logp_every"][0]["values"][0], 0.0)

    def test_generated_chat_response_replaces_only_assistant_turn(self):
        record = {
            "rejected": [
                {"role": "user", "content": "p"},
                {"role": "assistant", "content": "old"},
            ]
        }
        updated = attach_generated_response(record, "new")
        self.assertEqual(updated["rejected"][-1]["content"], "new")
        self.assertEqual(updated["rejected"][0]["content"], "p")

    def test_parent_groups_use_shared_boundaries(self):
        student, teacher = common_parent_groups(
            [(0, 2), (2, 4)], [(0, 1), (1, 4)]
        )
        self.assertEqual(student, [[0, 1]])
        self.assertEqual(teacher, [[0, 1]])
        with self.assertRaises(DatasetFormatError):
            common_parent_groups([], [(0, 1)])

    def test_preparation_configuration_is_validated(self):
        with self.assertRaises(ConfigurationError):
            DistillationPreparationConfig("vpd", "output", num_responses=1)
        with self.assertRaises(ConfigurationError):
            DistillationPreparationConfig(
                "vpd", "output", teacher_score_mode="unsupported"
            )

    @patch("human_alignment.integrations.distillation_data.DistillationDataBackend.prepare")
    def test_public_preparation_workflow(self, prepare):
        prepare.return_value = {"train": []}
        with tempfile.TemporaryDirectory() as directory:
            result = prepare_preference_distillation_dataset(
                [{"prompt": "p", "responses": ["a", "b"]}],
                objective="vpd",
                output_dir=directory,
                teacher_model="teacher",
            )
            self.assertEqual(result.path, Path(directory))
            self.assertEqual(prepare.call_args.kwargs["objective"], "vpd")


if __name__ == "__main__":
    unittest.main()
