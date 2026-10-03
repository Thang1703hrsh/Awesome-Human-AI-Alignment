"""Scoring rules of the RewardBench 2, JudgeBench and AlpacaEval 2 evaluators (torch-free).

Expected values are worked out by hand from the official code (``reward-bench/scripts/run_v2.py`` and
``rewardbench/utils.py``, ``JudgeBench/utils/metrics.py``, ``alpaca_eval/metrics/helpers.py``); during development
the functions were also checked against those sources on randomized inputs.
"""

import io
import math
import unittest
from contextlib import redirect_stdout

from human_alignment.benchmarks import alpacaeval as AE
from human_alignment.benchmarks import judgebench as JB
from human_alignment.benchmarks import rewardbench2 as RB
from human_alignment.benchmarks.common import score_pairs, wilson_interval


class RewardBench2Tests(unittest.TestCase):
    def test_best_of_four_penalises_ties_at_the_top(self):
        self.assertEqual(RB.best_of_n_result([3.0, 1.0, 2.0, 0.0]), 1.0)
        self.assertEqual(RB.best_of_n_result([3.0, 3.0, 2.0, 0.0]), 0.5)
        self.assertEqual(RB.best_of_n_result([1.0, 1.0, 1.0, 1.0]), 0.25)
        self.assertEqual(RB.best_of_n_result([1.0, 3.0, 2.0, 0.0]), 0.0)

    def test_ties_composite(self):
        rows = [
            {"id": "ref:0", "num_correct": 1, "scores": [5.0, 1.0, 0.0]},   # margin 4
            {"id": "tied:0", "num_correct": 2, "scores": [5.0, 4.0, 1.0]},  # spread 1, margin 3
            {"id": "ref:1", "num_correct": 1, "scores": [0.0, 1.0]},        # inaccurate, margin -1
            {"id": "tied:1", "num_correct": 2, "scores": [3.0, 1.0, 2.0]},  # spread 2, margin -1
        ]
        out = RB.ties_score(rows)
        self.assertEqual((out["ref_accuracy"], out["tied_accuracy"]), (0.5, 0.5))
        self.assertEqual(out["correctness_preferred"], 0.5)       # 3 > 1, -1 < 2
        self.assertEqual(out["correctness_preferred_hard"], 0.5)  # min(4,3)=3 > 1, min(-1,-1) < 2
        margin = (math.tanh(3 / 1 - 1) + math.tanh(-1 / 2 - 1)) / 2
        self.assertAlmostEqual(out["correctness_margin_score"], margin)
        self.assertAlmostEqual(out["score"], 0.3 * 0.5 + 0.3 * 0.5 + 0.2 * 0.5 + 0.2 * 0.5 + 0.01 * margin)

    def test_zero_spread_follows_numpy_division(self):
        rows = [
            {"id": "ref:0", "num_correct": 1, "scores": [2.0, 1.0]},
            {"id": "tied:0", "num_correct": 2, "scores": [2.0, 2.0, 1.0]},  # spread 0, margin 1 -> tanh(inf) = 1
        ]
        self.assertEqual(RB.ties_score(rows)["correctness_margin_score"], 1.0)

    def test_summary_is_an_unweighted_mean_over_subsets(self):
        rows = [{"subset": "Focus", "scores": [1.0, 0.0, 0.0, 0.0]} for _ in range(3)]
        rows += [{"subset": "Math", "scores": [0.0, 1.0, 0.0, 0.0]}]
        summary = RB.summarize_rewardbench2(rows)
        self.assertEqual(summary["subsets"]["Focus"]["score"], 1.0)
        self.assertEqual(summary["average"], 0.5)  # not 3/4
        self.assertFalse(summary["complete"])

    def test_evaluate_scores_every_completion(self):
        rows = [{"id": "1", "subset": "Safety", "prompt": "p", "completions": ["good", "bad", "bad", "worse"],
                 "num_correct": 1}]
        scored, summary = RB.evaluate_rewardbench2(rows, lambda prompt, response: float(response == "good"))
        self.assertEqual(scored[0]["scores"], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(summary["subsets"]["Safety"]["score"], 1.0)

    def test_hub_lists_may_arrive_as_strings(self):
        self.assertEqual(RB._as_list('["a", "b"]'), ["a", "b"])
        self.assertEqual(RB._as_list("['a', \"b\"]"), ["a", "b"])


class JudgeBenchTests(unittest.TestCase):
    def test_double_game_vote(self):
        self.assertGreater(JB.vote("A>B", "A>B", "A>B"), 0)
        self.assertGreater(JB.vote("A>B", "A>B", "A=B"), 0)   # one right, one tie: correct
        self.assertGreater(JB.vote("A>B", None, "A>B"), 0)    # failed trial counts 0
        self.assertEqual(JB.vote("A>B", "A>B", "B>A"), 0)     # inconsistent: not correct
        self.assertLess(JB.vote("A>B", "B>A", "A=B"), 0)

    def test_judge_protocol_flips_the_second_order(self):
        pairs = [{"pair_id": "1", "source": "livebench-math", "label": "A>B", "question": "q",
                  "response_A": "right", "response_B": "wrong"}]
        always_first = lambda q, a, b: "A>B"  # pure position bias
        rows, summary = JB.evaluate_judgebench_with_judge(pairs, always_first)
        self.assertEqual(rows[0]["decisions"], ["A>B", "B>A"])
        self.assertEqual(summary["overall"]["accuracy"], 0.0)
        self.assertEqual(summary["overall"]["inconsistent"], 1)
        content = lambda q, a, b: "A>B" if a == "right" else "B>A"
        _, summary = JB.evaluate_judgebench_with_judge(pairs, content)
        self.assertEqual(summary["categories"]["math"]["accuracy"], 1.0)

    def test_reward_model_single_game_breaks_ties_towards_b(self):
        pairs = [{"pair_id": str(i), "source": src, "label": label, "question": "q",
                  "response_A": a, "response_B": b}
                 for i, (src, label, a, b) in enumerate([("mmlu-pro-law", "A>B", "x", "x"),
                                                         ("livecodebench", "A>B", "long answer", "x")])]
        rows, summary = JB.evaluate_judgebench_with_reward_model(pairs, lambda q, r: float(len(r)))
        self.assertEqual([r["decision"] for r in rows], ["B>A", "A>B"])
        self.assertEqual(summary["protocol"], "single game")
        self.assertEqual(summary["categories"]["knowledge"]["accuracy"], 0.0)

    def test_vanilla_parser_is_exact(self):
        self.assertEqual(JB.parse_vanilla(" Output (a) "), "A>B")
        self.assertEqual(JB.parse_vanilla("Output (b)"), "B>A")
        self.assertIsNone(JB.parse_vanilla("I think Output (a)"))
        self.assertEqual(JB.category("livebench-reasoning"), "reasoning")


class AlpacaEvalTests(unittest.TestCase):
    def test_win_rate_conventions(self):
        out = AE.win_rate([2.0, 1.0, 1.5, 0, 1.75, None, 3.0])
        # 0 -> draw (1.5); None and 3.0 are dropped; shifted values 1, 0, .5, .5, .75
        self.assertAlmostEqual(out["win_rate"], 100 * 2.75 / 5)
        self.assertEqual((out["n_wins"], out["n_wins_base"], out["n_draws"]), (2, 1, 2))
        self.assertEqual((out["n_total"], out["n_dropped"]), (5, 2))
        self.assertAlmostEqual(out["discrete_win_rate"], 100 * (1 + 0 + 0.5 + 0.5 + 1) / 5)

    def test_length_controlled_minimal_removes_a_pure_length_effect(self):
        # The judge prefers the longer output and most model outputs are longer than the baseline's (raw 70%);
        # the length model attributes part of that advantage to length, so the equal-length rate is lower.
        offsets = (10, 40, 90, 160)
        model = ["x" * (100 + d) for d in offsets] * 3 + ["x" * (100 - d) for d in offsets]
        base = ["y" * 100] * len(model)
        prefs = [1.9 if len(m) > 100 else 1.1 for m in model]
        out = AE.length_controlled_minimal(prefs, model, base)
        self.assertGreater(AE.win_rate(prefs)["win_rate"], 60.0)
        self.assertLess(out["lc_minimal_win_rate"], AE.win_rate(prefs)["win_rate"])
        self.assertLess(out["length_coef"], 0)  # longer baseline -> lower model preference

    def test_length_controlled_minimal_leaves_length_independent_preferences_unchanged(self):
        model = ["x" * n for n in (10, 200, 30, 400, 50, 600)]
        base = ["y" * 100] * len(model)
        out = AE.length_controlled_minimal([1.7] * len(model), model, base)
        self.assertAlmostEqual(out["lc_minimal_win_rate"], 70.0, places=6)
        self.assertAlmostEqual(out["length_coef"], 0.0, places=6)

    def test_native_backend_pairs_outputs_with_the_reference(self):
        rows = [{"instruction": "i1", "output": "aaaa"}, {"instruction": "i2", "output": "b"}]
        baseline = {"i1": {"output": "zz"}, "i2": {"output": "zzzz"}}
        judge = lambda inst, ref, out: 2.0 if len(out) > len(ref) else 1.0
        judged, summary = AE.evaluate_alpacaeval_native(rows, baseline, judge)
        self.assertEqual([r["preference"] for r in judged], [2.0, 1.0])
        self.assertEqual(summary["win_rate"], 50.0)


class CommonTests(unittest.TestCase):
    def test_wilson_matches_the_survey_tables(self):
        low, high = wilson_interval(178, 350)  # JudgeBench vanilla GPT-4o: 50.86% -> 45.6–56.1
        self.assertEqual((round(100 * low, 1), round(100 * high, 1)), (45.6, 56.1))
        self.assertIsNone(wilson_interval(0, 0))

    def test_score_pairs_prefers_batching(self):
        class Batched:
            def score_batch(self, pairs):
                return [len(r) for _, r in pairs]

        self.assertEqual(score_pairs(Batched(), [("p", "ab"), ("p", "c")]), [2.0, 1.0])

    def test_registry_and_cli_expose_the_benchmarks(self):
        from human_alignment.cli import main
        from human_alignment.registry import default_registry

        registry = default_registry()
        for method in ("rewardbench2", "judgebench", "alpacaeval", "caa", "reward_best_of_n", "args"):
            self.assertIsNotNone(registry.get(method).factory)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            self.assertEqual(main(["bench", "list"]), 0)
        self.assertIn("rewardbench2", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
