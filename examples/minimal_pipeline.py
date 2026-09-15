"""Dependency-free lifecycle composition using a toy distillation callback."""

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


context = InteractionContext("Explain alignment", id="example")
specification = AlignmentSpecification(
    AlignmentTarget("helpful-assistant", "Provide a concise and accurate response.")
)
feedback = HumanFeedback(
    lambda context, specification: Demonstration("Alignment connects behavior to targets."),
    FeedbackRepresentation.DEMONSTRATION,
)
distillation = AlignmentDistillation(
    lambda model, signals, specification: (
        {**model, "aligned": True, "signals": len(signals)},
        {"loss": 0.1},
    )
)
evaluation = BehavioralEvaluation(
    {"aligned": lambda model, case: model.get("aligned", False)},
    {"aligned": 1.0},
)

result = AlignmentPipeline(
    specification,
    supervision=(feedback,),
    training=(distillation,),
    assurance=(evaluation,),
).run(
    {"name": "toy-model"},
    contexts=(context,),
    evaluation_cases=(EvaluationCase(context),),
)

print(result.training[0].metrics)
print(result.assurance[0])
