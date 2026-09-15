import unittest

from human_alignment.mechanisms.inference import (
    BestOfNSearch,
    ControlDecision,
    HumanControl,
    InferenceSteering,
    IterativeRefinement,
)
from human_alignment.types import GenerationRequest, InteractionContext


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.request = GenerationRequest(InteractionContext("draft"))

    def test_steering_transforms_request(self):
        mechanism = InferenceSteering(
            lambda model, request: request.context.input,
            lambda model, request: GenerationRequest(InteractionContext("steered")),
        )
        result = mechanism.generate(None, self.request)
        self.assertEqual(result.generations[0].output, "steered")

    def test_best_of_n_is_exposed_from_search_module(self):
        from human_alignment.mechanisms.inference.search import BestOfNSearch as SearchClass

        self.assertIs(BestOfNSearch, SearchClass)

    def test_refinement_revises_output(self):
        mechanism = IterativeRefinement(
            lambda model, request: "draft",
            lambda model, output, request: "be clearer",
            lambda model, output, critique, request: f"{output}: revised",
            iterations=2,
        )
        result = mechanism.generate(None, self.request)
        self.assertEqual(result.generations[0].output, "draft: revised: revised")

    def test_control_can_replace_rejected_action(self):
        mechanism = HumanControl(
            lambda model, request: "unsafe action",
            lambda output, request: ControlDecision(False, "safe fallback", "rejected"),
        )
        result = mechanism.generate(None, self.request)
        self.assertEqual(result.generations[0].output, "safe fallback")
        self.assertFalse(result.generations[0].metadata["approved"])


if __name__ == "__main__":
    unittest.main()
