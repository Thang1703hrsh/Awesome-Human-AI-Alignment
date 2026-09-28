"""Offline example: python recipes/specification/demo.py (after editable install)."""
from human_alignment.types import AlignmentTarget, InteractionContext
from human_alignment.specification import (
    InstructionConstraint, TaskSpecification, UserProfile,
    PersonalizedSpecification, UncertaintySpecification,
)
from human_alignment.registry import default_registry


def main():
    target = AlignmentTarget("assistance", "Answer usefully", constraints=("Be truthful",))
    context = InteractionContext("Reply briefly", metadata={"user_id": "alice"})
    personalized = PersonalizedSpecification(target, {
        "alice": UserProfile("alice", ("Use concise language",))}).resolve(context)
    uncertain = UncertaintySpecification(personalized.target, lambda _: .2).resolve(context)
    task = TaskSpecification(uncertain.target, (
        InstructionConstraint("length", "word_count", {"min": 1, "max": 10}),))
    print("Resolved constraints:", task.resolve(context).target.constraints)
    print("Compliance:", task.evaluate("Here is a concise answer."))
    print("Reliability:", default_registry().create("ifeval_pp").evaluate(
        {"a": [True, True], "b": [True, False]}, k=2))


if __name__ == "__main__":
    main()
