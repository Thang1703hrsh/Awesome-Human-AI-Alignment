import unittest

from human_alignment import AlignmentSpecification, AlignmentTarget, InteractionContext
from human_alignment.exceptions import DatasetFormatError
from human_alignment.specification import Stakeholder
from human_alignment.supervision import PreferenceExample, prepare_dataset


class SupervisionAndTrainingTests(unittest.TestCase):
    def test_preference_examples_are_normalized(self):
        bundle = prepare_dataset(
            [PreferenceExample("prompt", "chosen", "rejected")],
            "preference",
        )
        self.assertEqual(bundle.data[0]["chosen"], "chosen")
        self.assertEqual(bundle.origin, "memory")

    def test_missing_preference_field_is_rejected(self):
        with self.assertRaises(DatasetFormatError):
            prepare_dataset([{"prompt": "p", "chosen": "c"}], "preference")

    def test_missing_local_file_is_not_treated_as_remote_dataset(self):
        with self.assertRaises(DatasetFormatError):
            prepare_dataset("missing-preferences.jsonl", "preference")

    def test_dynamic_target_applies_context_rule(self):
        specification = AlignmentSpecification(
            AlignmentTarget(
                "target",
                "description",
                context_rules={"high_risk": "require human approval"},
            ),
            stakeholders=(Stakeholder("user", "affected party"),),
        )
        context = InteractionContext("prompt", metadata={"high_risk": True})
        resolved = specification.resolve(context)
        self.assertIn("require human approval", resolved.target.constraints)


if __name__ == "__main__":
    unittest.main()
