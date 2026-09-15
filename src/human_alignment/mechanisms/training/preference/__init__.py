"""Preference-optimization methods."""

from human_alignment.mechanisms.training.preference.dpo import DPO
from human_alignment.mechanisms.training.preference.kto import KTO
from human_alignment.mechanisms.training.preference.simpo import SimPO

__all__ = ["DPO", "KTO", "SimPO"]
