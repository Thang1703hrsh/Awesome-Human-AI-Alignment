"""Lazy factories for specification-related reference components."""
from functools import partial
from importlib import import_module


COMPONENTS = {
    "flan_mixture": ("human_alignment.supervision.specification_data", "FlanMixture"),
    "parrot": ("human_alignment.supervision.specification_data", "ParrotTurns"),
    "good_plan": ("human_alignment.assurance.specification", "AssistanceOutcomeEvaluation"),
    "ifeval_pp": ("human_alignment.assurance.specification", "IFEvalReliability"),
    "mosaic": ("human_alignment.assurance.specification", "MosaicEvaluation"),
    "reasonif": ("human_alignment.assurance.specification", "ReasonIFEvaluation"),
    "fb_bench": ("human_alignment.assurance.specification", "FeedbackResponsiveness"),
    "vpl": ("human_alignment.mechanisms.training.specification_losses", "VPLObjective"),
    "oppu": ("human_alignment.specification.personalization", "OPPU"),
    "prism": ("human_alignment.supervision.specification_data", "PRISMPreferences"),
    "pad": ("human_alignment.mechanisms.inference.personalized", "PAD"),
    "prose": ("human_alignment.specification.personalization", "PROSE"),
    "fpps": ("human_alignment.mechanisms.inference.personalized", "FPPS"),
    "personalized_benchmark": ("human_alignment.assurance.specification", "PersonalizedBenchmark"),
    "interaction_alignment": ("human_alignment.specification.personalization", "InteractionAlignment"),
    "distributional_preference": ("human_alignment.mechanisms.training.specification_losses", "DistributionalPreferenceLearning"),
    "active_preference": ("human_alignment.specification.uncertainty", "ActivePreferenceLearning"),
    "preference_stability": ("human_alignment.assurance.specification", "PreferenceStability"),
    "copr": ("human_alignment.mechanisms.training.specification_losses", "COPRObjective"),
    "moral_change": ("human_alignment.assurance.specification", "MoralChangeAnalysis"),
    "pilaf": ("human_alignment.specification.uncertainty", "PILAF"),
    "wdpo": ("human_alignment.mechanisms.training.specification_losses", "WDPOObjective"),
    "kldpo": ("human_alignment.mechanisms.training.specification_losses", "KLDPOObjective"),
}


def _create(module, name, **parameters):
    return getattr(import_module(module), name)(**parameters)


def factories():
    return {key: partial(_create, module, name) for key, (module, name) in COMPONENTS.items()}
