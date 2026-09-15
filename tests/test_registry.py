import unittest

from human_alignment.exceptions import RegistrationError
from human_alignment.registry import (
    ImplementationStatus,
    MethodMetadata,
    MethodRegistry,
    default_registry,
)
from human_alignment.types import AlignmentStage


class RegistryTests(unittest.TestCase):
    def test_default_registry_exposes_dpo(self):
        registered = default_registry().get("dpo")
        self.assertEqual(registered.metadata.primary_category, "preference_optimization")
        self.assertEqual(registered.metadata.implementation_status, ImplementationStatus.NATIVE)

    def test_multi_label_category_filter(self):
        methods = default_registry().list(taxonomy_category="demonstrations_preferences")
        self.assertIn("dpo", {method.metadata.id for method in methods})

    def test_duplicate_registration_is_rejected(self):
        registry = MethodRegistry()
        metadata = MethodMetadata(
            "x",
            "X",
            "human_feedback",
            ("human_feedback",),
            AlignmentStage.SUPERVISION,
            ImplementationStatus.ADAPTER,
        )
        registry.register(metadata)
        with self.assertRaises(RegistrationError):
            registry.register(metadata)


if __name__ == "__main__":
    unittest.main()
