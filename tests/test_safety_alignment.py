import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

from human_alignment import SafetyAlignment, align, safety_methods
from human_alignment.catalog import validate_catalog
from human_alignment.cli import main as cli_main
from human_alignment.results import AlignmentRun
from human_alignment.safety.api import load_safety_config
from human_alignment.safety.data.pairs import mocan_logits, safedpo_transform
from human_alignment.safety.eval.judge import parse_judgment, summarize_judgments
from human_alignment.safety.eval.pairwise import attach_reference
from human_alignment.safety.eval.scoring import summarize_scores
from human_alignment.safety.losses.preference import bso_loss, dpo_loss, safedpo_loss
from human_alignment.safety.resources import profile_resources


ROOT = Path(__file__).resolve().parents[1]
RECIPES = ROOT / "recipes" / "safety_alignment" / "methods"


class SafetyAlignmentIntegrationTests(unittest.TestCase):
    def test_all_migrated_methods_are_available(self):
        methods = safety_methods()
        self.assertEqual(len(methods), 25)
        self.assertIn("saferlhf", methods)
        self.assertIn("safedpo", methods)
        self.assertIn("midpo_router", methods)

    def test_recipe_inheritance_and_overrides(self):
        config = load_safety_config(
            RECIPES / "safedpo.yaml",
            ["method_args.delta=5", "train.output_dir=outputs/test-safedpo"],
        )
        self.assertEqual(config.method, "safedpo")
        self.assertEqual(config.method_args["delta"], 5)
        self.assertEqual(config.train["output_dir"], "outputs/test-safedpo")
        self.assertEqual(config.data.selection, "safedpo")

    def test_every_migrated_recipe_resolves_without_the_source_tree(self):
        recipes = sorted(RECIPES.glob("*.yaml"))
        self.assertEqual(len(recipes), 27)
        for recipe in recipes:
            with self.subTest(recipe=recipe.name):
                config = load_safety_config(recipe)
                deepspeed = config.train.get("deepspeed")
                if deepspeed:
                    self.assertTrue((ROOT / deepspeed).is_file(), deepspeed)

    def test_main_registry_exposes_safety_methods_lazily(self):
        method = align(
            "safedpo",
            model="student",
            dataset="preferences",
            output_dir="outputs/safedpo",
            train=False,
        )
        self.assertIsInstance(method, SafetyAlignment)
        self.assertEqual(method.selected_method, "safedpo")
        self.assertEqual(validate_catalog().errors, ())

    @patch("human_alignment.safety.api.run_safety_config")
    def test_safety_facade_returns_standard_alignment_run(self, run_config):
        run_config.return_value = Path("outputs/safedpo")
        run = SafetyAlignment(
            model="student",
            dataset="preferences",
            output_dir="outputs/safedpo",
            method="safedpo",
        ).train()
        self.assertIsInstance(run, AlignmentRun)
        self.assertEqual(run.method, "safedpo")
        self.assertEqual(run.taxonomy_trace.specification, ("safety_alignment",))
        self.assertEqual(run.taxonomy_trace.mechanisms, ("preference_optimization",))
        resolved = run_config.call_args.args[0]
        self.assertEqual(resolved.model.policy, "student")
        self.assertEqual(resolved.data.dataset, "preferences")

    def test_safety_cli_is_nested_under_main_cli(self):
        output = io.StringIO()
        with redirect_stdout(output):
            result = cli_main(["safety", "list"])
        self.assertEqual(result, 0)
        self.assertIn("safedpo", output.getvalue())
        self.assertIn("saferlhf", output.getvalue())


class SafetyObjectiveTests(unittest.TestCase):
    def test_dpo_and_safe_objectives_are_differentiable(self):
        chosen = torch.tensor([1.0, 0.5], requires_grad=True)
        rejected = torch.tensor([0.0, 0.25], requires_grad=True)
        ref_chosen = torch.zeros(2)
        ref_rejected = torch.zeros(2)
        safe = torch.tensor([True, True])
        unsafe = torch.tensor([False, True])

        dpo = dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=0.1)
        safe_dpo = safedpo_loss(
            chosen,
            rejected,
            ref_chosen,
            ref_rejected,
            safe,
            unsafe,
            beta=0.1,
            delta=5.0,
        )
        bso = bso_loss(
            chosen,
            rejected,
            ref_chosen,
            ref_rejected,
            ~safe,
            ~unsafe,
            beta=0.1,
            safety_penalty=3.0,
            generator="sba",
            lam=0.2,
            s=4.0,
        )
        loss = dpo.losses.mean() + safe_dpo.losses.mean() + bso.losses.mean()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(chosen.grad)
        self.assertIsNotNone(rejected.grad)

    def test_safedpo_transform_swaps_or_drops_unsafe_preferences(self):
        swapped = safedpo_transform(
            {
                "prompt": "p",
                "chosen": "unsafe",
                "rejected": "safe",
                "chosen_safe": False,
                "rejected_safe": True,
            }
        )
        self.assertEqual(swapped["chosen"], "safe")
        self.assertIsNone(
            safedpo_transform(
                {
                    "prompt": "p",
                    "chosen": "a",
                    "rejected": "b",
                    "chosen_safe": False,
                    "rejected_safe": False,
                }
            )
        )

    def test_mocan_combines_helpfulness_and_negative_cost(self):
        logits = mocan_logits([0.0], [1.0], [2.0], [0.0], lam=0.5)
        torch.testing.assert_close(logits, torch.tensor([2.0]))


class SafetyEvaluationTests(unittest.TestCase):
    def test_resource_profiles_are_deduplicated(self):
        resources = profile_resources("midpo", include_evaluation=True)
        keys = {(r.kind, r.repo_id, r.revision) for r in resources}
        self.assertEqual(len(resources), len(keys))

    def test_score_and_judge_summaries(self):
        summary = summarize_scores(
            [
                {
                    "reward": 2.0,
                    "cost": -1.0,
                    "reference_reward": 1.0,
                    "reference_cost": 0.0,
                },
                {
                    "reward": 0.0,
                    "cost": 2.0,
                    "reference_reward": 1.0,
                    "reference_cost": 1.0,
                },
            ]
        )
        self.assertEqual(summary["safe_rate"], 0.5)
        self.assertEqual(summary["joint_win_rate"], 0.5)
        parsed = parse_judgment(
            '```json\n{"winner":"A","score_a":8,"score_b":4,"reason":"better"}\n```'
        )
        self.assertEqual(parsed["winner"], "A")
        judged = summarize_judgments(
            [
                {"helpfulness_winner": "candidate", "safety_winner": "tie"},
                {"helpfulness_winner": "reference", "safety_winner": "candidate"},
            ]
        )
        self.assertEqual(judged["safety"]["win_rate_excluding_ties"], 1.0)

    def test_pairing_generation_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate.jsonl"
            reference = root / "reference.jsonl"
            output = root / "paired.jsonl"
            candidate.write_text(
                json.dumps({"prompt": "p", "response": "candidate"}) + "\n",
                encoding="utf-8",
            )
            reference.write_text(
                json.dumps({"prompt": "p", "response": "reference"}) + "\n",
                encoding="utf-8",
            )
            attach_reference(candidate, reference, output)
            self.assertEqual(json.loads(output.read_text())["reference_response"], "reference")


if __name__ == "__main__":
    unittest.main()
