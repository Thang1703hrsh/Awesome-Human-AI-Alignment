"""Observable supervision signals, sources, and training datasets."""

from human_alignment.supervision.datasets import (
    DatasetBundle,
    InstructionExample,
    KTOExample,
    PreferenceExample,
    PreferenceDistillationExample,
    PromptExample,
    collect_preference_distillation_data,
    prepare_dataset,
)
from human_alignment.supervision.preference_distillation import (
    attach_generated_response,
    build_adpa_record,
    common_parent_groups,
    merge_compressed_probabilities,
)
from human_alignment.supervision.signals import (
    Critique,
    Demonstration,
    Preference,
    ProcessFeedback,
    TrajectoryFeedback,
)
from human_alignment.supervision.sources import (
    AIFeedback,
    FeedbackProvider,
    HumanFeedback,
    VerifiableFeedback,
)

HumanFeedbackProvider = HumanFeedback
AIFeedbackProvider = AIFeedback
VerifiableFeedbackProvider = VerifiableFeedback
SupervisionProvider = FeedbackProvider

__all__ = [
    "AIFeedback",
    "AIFeedbackProvider",
    "attach_generated_response",
    "build_adpa_record",
    "common_parent_groups",
    "Critique",
    "DatasetBundle",
    "Demonstration",
    "FeedbackProvider",
    "HumanFeedback",
    "HumanFeedbackProvider",
    "InstructionExample",
    "KTOExample",
    "merge_compressed_probabilities",
    "Preference",
    "PreferenceExample",
    "PreferenceDistillationExample",
    "ProcessFeedback",
    "PromptExample",
    "SupervisionProvider",
    "TrajectoryFeedback",
    "VerifiableFeedback",
    "VerifiableFeedbackProvider",
    "collect_preference_distillation_data",
    "prepare_dataset",
]
