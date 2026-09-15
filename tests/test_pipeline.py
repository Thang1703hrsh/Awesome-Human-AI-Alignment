import unittest

from human_alignment import (
    AlignmentPipeline,
    AlignmentSpecification,
    AlignmentTarget,
    EvaluationCase,
    FeedbackRepresentation,
    InteractionContext,
)
from human_alignment.assurance import BehavioralEvaluation
from human_alignment.mechanisms.training import AlignmentDistillation
from human_alignment.supervision import Demonstration, HumanFeedback


class PipelineTests(unittest.TestCase):
    def test_dependency_free_pipeline(self):
        context = InteractionContext("prompt")
        specification = AlignmentSpecification(AlignmentTarget("target", "description"))
        provider = HumanFeedback(
            lambda context, specification: Demonstration("answer"),
            FeedbackRepresentation.DEMONSTRATION,
        )
        method = AlignmentDistillation(
            lambda model, signals, specification: (
                {**model, "aligned": True},
                {"loss": 0.2},
            )
        )
        evaluation = BehavioralEvaluation(
            {"alignment": lambda model, case: model["aligned"]},
            {"alignment": 1.0},
        )
        result = AlignmentPipeline(
            specification,
            supervision=(provider,),
            training=(method,),
            assurance=(evaluation,),
        ).run(
            {"aligned": False},
            contexts=(context,),
            evaluation_cases=(EvaluationCase(context),),
        )
        self.assertTrue(result.model["aligned"])
        self.assertEqual(result.training[0].metrics["loss"], 0.2)
        self.assertTrue(result.assurance[0].passed)


if __name__ == "__main__":
    unittest.main()
