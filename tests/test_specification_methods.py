import math
import unittest

from human_alignment.registry import default_registry
from human_alignment.catalog import validate_catalog
from human_alignment.specification import (
    ConstraintChecker, InstructionConstraint, TaskSpecification,
    PersonalizedSpecification, UserProfile, UncertaintySpecification,
)
from human_alignment.specification.personalization import OPPU, PROSE, InteractionAlignment
from human_alignment.specification.uncertainty import ActivePreferenceLearning, PILAF
from human_alignment.specification.methods import COMPONENTS
from human_alignment.types import AlignmentTarget, InteractionContext
from human_alignment.assurance.specification import (
    IFEvalReliability, MosaicEvaluation, ReasonIFEvaluation, FeedbackResponsiveness,
    AssistanceOutcomeEvaluation, UserComparison, PersonalizedBenchmark,
    TimedPreference, PreferenceStability, MoralChangeAnalysis,
)
from human_alignment.supervision.specification_data import FlanMixture, ParrotTurns, PRISMPreferences

try:
    import torch
except ImportError:
    torch = None


class SpecificationTests(unittest.TestCase):
    def setUp(self):
        self.target = AlignmentTarget("assist", "Help", constraints=("Be truthful",))

    def test_catalog_factories(self):
        self.assertTrue(validate_catalog().valid)
        registry = default_registry()
        self.assertEqual(len(COMPONENTS), 23)  # 22 papers; WDPO/KLDPO are separate.
        for name in COMPONENTS:
            self.assertIsNotNone(registry.get(name).factory)
        self.assertIsInstance(registry.create("ifeval_pp"), IFEvalReliability)

    def test_task_resolve_and_checks(self):
        task = TaskSpecification(self.target, (
            InstructionConstraint("intro", "starts_with", {"text": "Hello"}),
            InstructionConstraint("length", "word_count", {"min": 2, "max": 3}),
        ))
        self.assertTrue(all(task.evaluate("Hello world").values()))
        self.assertFalse(task.evaluate("Hello")["length:final"])
        self.assertIn("Be truthful", task.resolve(InteractionContext("Q")).target.constraints)

    def test_bad_constraints(self):
        checker = ConstraintChecker()
        with self.assertRaises(ValueError):
            checker.check("x", InstructionConstraint("a", "unknown"))
        with self.assertRaises(ValueError):
            checker.check("x", InstructionConstraint("a", "word_count", {"min": -1}))
        with self.assertRaises(TypeError):
            ConstraintChecker({"x": lambda *_: 1}).check("a", InstructionConstraint("a", "x"))
        self.assertFalse(checker.check('{"x": NaN}', InstructionConstraint("j", "json_object")))

    def test_reliable_k_requires_all_cousins(self):
        metric = IFEvalReliability()
        result = metric.evaluate({"a": [True, True], "b": [True, False]}, 2)
        self.assertEqual(result["reliable@2"], .5)
        self.assertEqual(result["prompt_accuracy"], .75)
        with self.assertRaises(ValueError):
            metric.evaluate({"a": [True]}, 2)

    def test_reasoning_unknown_not_pass(self):
        task = TaskSpecification(self.target, (InstructionConstraint("x", "contains", {"text": "yes"}, "both"),))
        result = ReasonIFEvaluation().evaluate([(task, "yes", None)])
        self.assertIsNone(result["reasoning"])
        self.assertEqual(result["final"], 1)
        self.assertEqual(result["joint_compliance"], 0)
        self.assertEqual(result["unobserved_constraints"], 1)

    def test_mosaic(self):
        task = TaskSpecification(self.target, (InstructionConstraint("x", "contains", {"text": "yes"}),))
        result = MosaicEvaluation().evaluate([(task, "yes"), (task, "no")])
        self.assertEqual(result["kind"]["contains"], .5)
        self.assertEqual(result["position"][0], .5)

    def test_feedback_and_utility_distinct(self):
        result = FeedbackResponsiveness().evaluate([False, True], [True, False])
        self.assertEqual(result["correction_rate"], 1)
        self.assertEqual(result["preservation_rate"], 0)
        self.assertEqual(AssistanceOutcomeEvaluation().evaluate([1, 2], [2, 1])["pearson_correlation"], -1)

    def test_profile_requires_explicit_user(self):
        spec = PersonalizedSpecification(self.target, {"a": UserProfile("a", ("Concise",))})
        result = spec.resolve(InteractionContext("Q", metadata={"user_id": "a"}))
        self.assertEqual(result.target.constraints, ("Be truthful", "Concise"))
        with self.assertRaises(KeyError):
            spec.resolve(InteractionContext("Q"))

    def test_oppu_isolation(self):
        calls = []
        method = OPPU(lambda _: object(), lambda model, rows, user: calls.append(user) or model)
        models = method.fit_users({"a": [{"user_id": "a"}], "b": [{"user_id": "b"}]})
        self.assertIsNot(models["a"], models["b"])
        shared = object()
        with self.assertRaises(ValueError):
            OPPU(lambda _: shared, lambda m, *_: m).fit_users({"a": [{}], "b": [{}]})
        with self.assertRaises(ValueError):
            method.fit_users({"a": [{"user_id": "b"}]})

    def test_prose_verifies_all_samples(self):
        calls = []
        def backend(request):
            calls.append(request["task"])
            if request["task"] == "verify_preferences":
                return {"supported": True, "critique": "Supported by sample"}
            return {"preferences": ["Brief"]}
        profile = PROSE(backend).infer("a", ["sample one", "sample two"])
        self.assertEqual(profile.preferences, ("Brief",))
        self.assertEqual(calls.count("verify_preferences"), 2)

    def test_interaction_isolation_and_atomic_validation(self):
        requests = []
        def backend(request):
            requests.append(request)
            return {"response": "ok", "preferences": ["Brief"]}
        method = InteractionAlignment(backend)
        method.respond("a", "secret A")
        method.respond("b", "B")
        self.assertEqual(len(requests[-1]["history"]), 1)
        method.backend = lambda _: {"preferences": ["Changed"], "response": ""}
        with self.assertRaises(ValueError):
            method.respond("a", "invalid")
        self.assertEqual(len(method.histories["a"]), 2)

    def test_personalized_ranks_do_not_pool_users(self):
        result = PersonalizedBenchmark().evaluate([
            UserComparison("a", "X", "Y", 1), UserComparison("b", "X", "Y", 0)])
        for metric in ("elo", "bradley_terry"):
            self.assertGreater(result["a"][metric]["X"], result["a"][metric]["Y"])
            self.assertLess(result["b"][metric]["X"], result["b"][metric]["Y"])

    def test_stability_requires_matched_context(self):
        rows = [TimedPreference("a", "q", 0, "yes"), TimedPreference("a", "q", 1, "no"),
                TimedPreference("b", "q", 2, "yes")]
        self.assertEqual(PreferenceStability().evaluate(rows)["transitions"], 1)
        self.assertIn("not identified", MoralChangeAnalysis().evaluate(rows)["interpretation"])
        with self.assertRaises(ValueError):
            PreferenceStability().evaluate(rows[:1])

    def test_uncertainty_validation(self):
        spec = UncertaintySpecification(self.target, lambda _: .4)
        self.assertEqual(spec.resolve(InteractionContext("q")).target.uncertainty, .4)
        with self.assertRaises(ValueError):
            UncertaintySpecification(self.target, lambda _: float("nan")).resolve(InteractionContext("q"))

    def test_active_selects_high_gap_not_low(self):
        rows = [{"policy_logps": [-2, -2], "reference_logps": [-2, -2], "sample_logps": [-10]},
                {"policy_logps": [-1, -5], "reference_logps": [-2, -2], "sample_logps": [-1]}]
        self.assertEqual(ActivePreferenceLearning().select(rows, 1), [1])
        self.assertEqual(ActivePreferenceLearning().select(rows, 1, entropy_pool_size=1), [0])

    def test_pilaf_logits_and_repeatability(self):
        self.assertEqual(PILAF(beta=.5).logits([2., 0.], [0., 2.], 1), [3., -1.])
        args = ([0], lambda _: [1., 0.], lambda _: [0., 1.])
        self.assertEqual(PILAF(seed=4).sample_pair(*args, max_new_tokens=3),
                         PILAF(seed=4).sample_pair(*args, max_new_tokens=3))

    def test_flan_and_parrot(self):
        mixture = FlanMixture({"x": [{"inputs": "q", "targets": "a"}]}, {"x": 1})
        self.assertEqual(mixture.sample(2), mixture.sample(2))
        rows = ParrotTurns().convert([{"messages": [
            {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"},
            {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}]}])
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(rows[1].prompt), 3)

    def test_prism_selected_history_not_highest_score(self):
        row = {"user_id": "a", "conversation_id": "c", "conversation_history": [
            {"turn": 0, "role": "user", "content": "q"},
            {"turn": 0, "role": "model", "content": "winner", "score": 90, "if_chosen": False},
            {"turn": 0, "role": "model", "content": "selected", "score": 80, "if_chosen": True},
            {"turn": 1, "role": "user", "content": "q2"},
            {"turn": 1, "role": "model", "content": "a", "score": 90},
            {"turn": 1, "role": "model", "content": "b", "score": 80}]}
        rows = PRISMPreferences().convert([row])
        self.assertEqual(rows[1].prompt[1]["content"], "selected")
        self.assertEqual(rows[1].metadata["user_id"], "a")


@unittest.skipIf(torch is None, "Requires optional torch")
class SpecificationTensorTests(unittest.TestCase):
    def test_vpl_elbo_and_gradient(self):
        from human_alignment.mechanisms.training.specification_losses import VPLObjective
        margins = torch.zeros(2, 3, requires_grad=True)
        mu = torch.zeros(3, 2, requires_grad=True)
        loss = VPLObjective().loss(margins, mu, torch.zeros_like(mu))
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)
        loss.backward()
        self.assertTrue((margins.grad < 0).all())

    def test_distributional_likelihoods(self):
        from human_alignment.mechanisms.training.specification_losses import DistributionalPreferenceLearning
        method = DistributionalPreferenceLearning()
        z, one = torch.zeros(2), torch.ones(2)
        self.assertAlmostEqual(method.gaussian_loss(z, z, one, one).item(), math.log(2), places=6)
        logits = torch.zeros(2, 4, requires_grad=True)
        loss = method.categorical_loss(logits, logits)
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertLess(method.categorical_loss(torch.tensor([[0., 10.]]), torch.tensor([[10., 0.]])).item(), .001)

    def test_kl_dro_envelope(self):
        from human_alignment.mechanisms.training.specification_losses import KLDPOObjective
        losses = torch.tensor([1., 3.], requires_grad=True)
        KLDPOObjective().loss(losses).backward()
        self.assertTrue(torch.allclose(losses.grad, losses.detach().softmax(0)))

    def test_wdpo_input_gradient_and_optimization(self):
        from human_alignment.mechanisms.training.specification_losses import WDPOObjective
        x = torch.tensor([[1.], [2.]], requires_grad=True)
        weight = torch.nn.Parameter(torch.tensor(2.))
        losses = (weight*x).square().flatten()
        loss = WDPOObjective(radius=.1).loss(losses, [x])
        expected = losses.mean().item() + .1*math.sqrt((8**2+16**2)/2)
        self.assertAlmostEqual(loss.item(), expected, places=5)
        optimizer = torch.optim.SGD([weight], lr=.01)
        loss.backward()
        optimizer.step()
        self.assertLess(weight.item(), 2.)

    def test_copr_replay_and_dual(self):
        from human_alignment.mechanisms.training.specification_losses import COPRObjective
        method = COPRObjective()
        logps = torch.tensor([[-1., -2.]], requires_grad=True)
        self.assertEqual(method.fit_loss(logps, logps).item(), 0.)
        loss = method.loss(torch.tensor(2.), torch.tensor([3.]), torch.tensor([1.]), torch.tensor([0.]))
        self.assertEqual(loss.item(), 2.)
        self.assertGreater(method.dual_update(torch.tensor([0.]), torch.tensor([1.])).item(), 0)

    def test_pad_base_topk_first(self):
        from human_alignment.mechanisms.inference.personalized import PAD
        token = PAD(top_k=2).next_token(torch.tensor([3., 2., 0.]), torch.tensor([0., 5., 100.]))
        self.assertEqual(token.item(), 1)

    def test_fpps_mixed(self):
        from human_alignment.mechanisms.inference.personalized import FPPS
        output = FPPS().steer(torch.zeros(2, 2), torch.ones(2), torch.tensor([2., 0.]), [.8, .2])
        self.assertTrue(torch.allclose(output, torch.tensor([[-1., -1.], [-.6, 0.]])))


if __name__ == "__main__":
    unittest.main()
