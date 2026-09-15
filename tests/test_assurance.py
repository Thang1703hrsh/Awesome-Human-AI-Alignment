import unittest

from human_alignment.assurance import (
    AlignmentPreservation,
    EvaluatorReliability,
    MonitoringAudit,
    RobustnessEvaluation,
)
from human_alignment.types import EvaluationCase, InteractionContext


class AssuranceTests(unittest.TestCase):
    def setUp(self):
        self.cases = (
            EvaluationCase(InteractionContext("a"), expected=1.0),
            EvaluationCase(InteractionContext("b"), expected=0.0),
        )

    def test_evaluator_reliability(self):
        report = EvaluatorReliability(
            lambda model, case: case.expected, max_mae=0.0
        ).assess(None, self.cases)
        self.assertTrue(report.passed)
        self.assertEqual(report.metrics["binary_agreement"], 1.0)

    def test_robustness_detects_drop(self):
        def perturb(case):
            return EvaluationCase(case.context, case.expected, {"shifted": True})

        report = RobustnessEvaluation(
            lambda model, case: 0.5 if case.metadata.get("shifted") else 1.0,
            {"distribution_shift": perturb},
            max_drop=0.2,
        ).assess(None, self.cases)
        self.assertFalse(report.passed)

    def test_preservation_compares_models(self):
        report = AlignmentPreservation(
            {"safety": lambda model, case: model["safety"]}, max_drop=0.05
        ).assess({"safety": 0.8}, self.cases, baseline_model={"safety": 0.9})
        self.assertFalse(report.passed)

    def test_monitoring_retains_audit_events(self):
        monitor = MonitoringAudit({"safe": lambda case: case.expected})
        report = monitor.assess(None, self.cases)
        self.assertEqual(len(monitor.events), 2)
        self.assertFalse(report.passed)


if __name__ == "__main__":
    unittest.main()
