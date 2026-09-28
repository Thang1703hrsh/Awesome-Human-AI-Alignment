import math
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from human_alignment.exceptions import DatasetFormatError
from human_alignment.registry import default_registry
from human_alignment.supervision import (
    ConstitutionalAI, GEval, JSONFeedbackBackend, LLMJudge, PaperFeedbackDataset,
    RLAIF, SelfInstruct, SelfRewarding, edit_masks, fine_grained_reward_loss,
    preference_reward_loss, prepare_dataset,
)
from human_alignment.types import FeedbackSource, InteractionContext


def convert(name, rows):
    return PaperFeedbackDataset(name, rows).convert()


class PaperDataTests(unittest.TestCase):
    def test_cli_conversion_and_no_overwrite(self):
        from human_alignment.cli import main

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            destination = Path(directory) / "pairs.jsonl"
            source.write_text(json.dumps([{"prompt": "q", "original": "bad",
                                           "corrected": "good"}]), encoding="utf-8")
            args = ["prepare", "feedback", "fine_grained_feedback", "--dataset", str(source),
                    "--output", str(destination), "--annotation-source", "ai"]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            result = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(result["chosen"], "good")
            self.assertEqual(result["metadata"]["source"], "ai")
            before = destination.read_bytes()
            with self.assertRaises(FileExistsError):
                main(args)
            self.assertEqual(destination.read_bytes(), before)

    def test_summary_choice_and_provenance(self):
        signals = convert("summarization_feedback", [{"info": {"post": "post"},
            "summaries": [{"text": "bad"}, {"text": "good"}], "choice": 1, "split": "valid2"}])
        row = prepare_dataset(signals, "preference").data[0]
        self.assertEqual((row["chosen"], row["rejected"]), ("good", "bad"))
        self.assertEqual(row["metadata"]["provenance"]["split"], "valid2")
        self.assertEqual(row["metadata"]["source"], "human")

    def test_hh_shared_history(self):
        prompt = "\n\nHuman: one\n\nAssistant: first\n\nHuman: two\n\nAssistant:"
        signal = convert("hh_rlhf", [{"chosen": prompt + " good", "rejected": prompt + " bad"}])[0]
        self.assertEqual(signal.context.input, prompt)
        self.assertEqual(signal.payload.options, (" good", " bad"))
        with self.assertRaises(DatasetFormatError):
            convert("hh_rlhf", [{"chosen": prompt + " good", "rejected": "wrong"}])

    def test_openassistant_lower_rank_wins_and_full_history(self):
        rows = [
            {"message_id": "p", "parent_id": None, "role": "prompter", "text": "q"},
            {"message_id": "a", "parent_id": "p", "role": "assistant", "text": "a"},
            {"message_id": "p2", "parent_id": "a", "role": "prompter", "text": "q2"},
            {"message_id": "bad", "parent_id": "p2", "role": "assistant", "text": "bad", "rank": 1},
            {"message_id": "good", "parent_id": "p2", "role": "assistant", "text": "good", "rank": 0},
            {"message_id": "del", "parent_id": "p2", "role": "assistant", "text": "deleted", "rank": 0, "deleted": True},
        ]
        signal, = convert("openassistant", rows)
        self.assertEqual(len(signal.context.input), 3)
        self.assertEqual(signal.payload.options[0][0]["content"], "good")
        rows[0]["parent_id"] = "p2"
        with self.assertRaises(DatasetFormatError):
            convert("openassistant", rows)

    def test_helpsteer_uses_helpfulness_not_verbosity(self):
        rows = [dict(prompt="q", response="good", helpfulness=4, correctness=4, coherence=4,
                     complexity=0, verbosity=0),
                dict(prompt="q", response="bad", helpfulness=1, correctness=1, coherence=1,
                     complexity=4, verbosity=4)]
        self.assertEqual(convert("helpsteer2", rows)[0].payload.options[0], "good")
        rows[1]["helpfulness"] = 4
        self.assertEqual(convert("helpsteer2", rows), [])
        rows[1]["helpfulness"] = float("nan")
        with self.assertRaises(DatasetFormatError):
            convert("helpsteer2", rows)

    def test_instructgpt_demonstrations_and_tied_rankings(self):
        rows = [{"prompt": "q", "completion": "a"},
                {"prompt": "p", "responses": ["a", "b", "c"], "ranks": [0, 0, 1]}]
        signals = convert("instructgpt_feedback", rows)
        self.assertEqual(len(signals), 3)
        self.assertEqual(prepare_dataset(signals[:1], "instruction").data[0]["completion"], "a")

    def test_trajectory_tie_preserved(self):
        signal, = convert("human_preferences", [{"segments": [[1, 2], [3, 4]],
                                                 "preference_probability": 0.5}])
        self.assertEqual(signal.payload["preference_probability"], 0.5)
        with self.assertRaises(DatasetFormatError):
            prepare_dataset([signal], "preference")

    def test_minimal_edits_and_images(self):
        signals = convert("fine_grained_feedback", [{"prompt": "q", "original": "bad", "corrected": "good"}])
        self.assertEqual(signals[0].payload.options, ("good", "bad"))
        signal, = convert("rlhf_v", [{"text": '{"question":"q","chosen":"yes","rejected":"no"}',
                                     "image_path": "image.jpg"}])
        self.assertEqual(signal.context.input, {"text": "q", "image": "image.jpg"})

    def test_mm_rlhf_rank_direction_and_ties(self):
        signals = convert("mm_rlhf", [{"question": "q", "video": "v.mp4",
            "models_output": ["best", "also best", "bad"], "final_ranking": [1, 1, 3]}])
        self.assertEqual(len(signals), 2)
        self.assertTrue(all(s.payload.options[1] == "bad" for s in signals))
        self.assertEqual(signals[0].context.input["video"], "v.mp4")

    def test_wildfeedback_hybrid_and_required_feedback(self):
        row = {"prompt": "q", "chosen": "good", "rejected": "bad", "user_feedback": "too vague"}
        self.assertEqual(convert("wildfeedback", [row])[0].source, FeedbackSource.HYBRID)
        del row["user_feedback"]
        with self.assertRaises(DatasetFormatError):
            convert("wildfeedback", [row])

    def test_ultrafeedback_aspect_scores(self):
        def completion(text, score):
            return {"response": text, "overall_score": 10,
                    "annotations": {k: {"Rating": str(score)} for k in
                        ("instruction_following", "truthfulness", "honesty", "helpfulness")}}
        signal, = convert("ultrafeedback", [{"instruction": "q",
                                             "completions": [completion("bad", 1), completion("good", 5)]}])
        self.assertEqual(signal.payload.options, ("good", "bad"))
        self.assertEqual(signal.source, FeedbackSource.AI)

    def test_dataset_provider_rejects_ignored_contexts(self):
        with self.assertRaises(DatasetFormatError):
            PaperFeedbackDataset("hh_rlhf", []).collect([InteractionContext("q")], None)

    def test_registry_supervision_factories(self):
        registry = default_registry()
        for category in ("human_feedback", "ai_feedback"):
            methods = registry.list(taxonomy_category=category)
            self.assertGreater(len(methods), 1)
            self.assertTrue(all(m.factory is not None for m in methods))
        self.assertEqual(registry.create("helpsteer2", records=[]).convert(), [])


class AIFeedbackTests(unittest.TestCase):
    def test_order_consistent_judging(self):
        seen = []
        def backend(request):
            seen.append(request["responses"])
            return {"winner": "A" if request["responses"][0] == "good" else "B"}
        signal = LLMJudge(backend).compare(InteractionContext("q"), ["bad", "good"])
        self.assertEqual(signal.payload.preferred_index, 1)
        self.assertEqual(seen, [("bad", "good"), ("good", "bad")])

    def test_position_bias_and_ties_abstain(self):
        for winner in ("A", "tie"):
            self.assertIsNone(LLMJudge(lambda r: {"winner": winner}).compare(
                InteractionContext("q"), ["one", "two"]))

    def test_rlaif_soft_labels_and_explicit_hard_conversion(self):
        provider = RLAIF(lambda r: {"probability_a": 0.8 if r["responses"][0] == "a" else 0.3})
        contexts = [InteractionContext("q", metadata={"responses": ["a", "b"]})]
        signals = provider.collect(contexts, None)
        self.assertAlmostEqual(signals[0].payload["probability_a"], 0.75)
        self.assertEqual(provider.pairwise_preferences(contexts)[0].payload.preferred_index, 0)

    def test_constitutional_revision_chain(self):
        def backend(r):
            if r["task"] == "constitutional_critique":
                return {"critique": r["principle"]}
            return {"response": r["response"] + "+"}
        signal, = ConstitutionalAI(backend, ["p1", "p2"]).collect(
            [InteractionContext("q", metadata={"response": "a"})], None)
        self.assertEqual(signal.payload.output, "a++")
        self.assertEqual(len(signal.provenance["revisions"]), 2)

    def test_geval_weighted_score_and_zero_mass(self):
        def backend(r):
            return {"steps": "check"} if r["task"] == "evaluation_steps" else {
                "score_probabilities": {"1": 0.1, "5": 0.3}}
        signal, = GEval(backend, "quality").collect(
            [InteractionContext("q", metadata={"response": "a"})], None)
        self.assertAlmostEqual(signal.payload["score"], 4.0)
        provider = GEval(lambda r: {"steps": "check", "score_probabilities": {"1": 0}}, "quality")
        with self.assertRaises(DatasetFormatError):
            provider.collect([InteractionContext("q", metadata={"response": "a"})], None)

    def test_self_instruct_dedup_and_stages(self):
        tasks = []
        def backend(r):
            tasks.append(r["task"])
            return {"instructions": ["Explain gravity", "Write a poem", "Write a poem"],
                    "is_classification": False,
                    "instances": [{"input": "", "output": "poem"}]}
        signals = SelfInstruct(backend, ["Explain gravity"]).generate(2)
        self.assertEqual(len(signals), 1)
        self.assertIn("classify_instruction", tasks)

    def test_self_rewarding_refreshes_model_every_round(self):
        versions, trained = [], []
        def factory(model):
            versions.append(model)
            def backend(r):
                return {"responses": ["good", "bad"]} if r["task"] == "generate_candidates" else {
                    "score": 5 if r["response"] == "good" else 0}
            return backend
        def train(model, signals, iteration):
            trained.append(signals[0].payload.options)
            return model + 1
        model, history = SelfRewarding(factory, train, 2).run(0, [InteractionContext("q")], rounds=2)
        self.assertEqual((model, versions, len(history)), (2, [0, 1], 2))
        self.assertEqual(trained, [("good", "bad"), ("good", "bad")])

    def test_json_backend_rejects_fake_probabilities(self):
        backend = JSONFeedbackBackend(lambda prompt: '{"winner":"A"}')
        self.assertEqual(backend({"task": "pairwise_judge"})["winner"], "A")
        with self.assertRaises(DatasetFormatError):
            backend({"task": "score_distribution"})
        with self.assertRaises(DatasetFormatError):
            JSONFeedbackBackend(lambda p: "not json")({"task": "generate"})

    def test_bad_scores_rejected(self):
        with self.assertRaises(DatasetFormatError):
            RLAIF(lambda r: {"probability_a": float("nan")}).compare(
                InteractionContext("q"), ["a", "b"])


class RewardLossTests(unittest.TestCase):
    def test_edit_mask_insert_delete(self):
        self.assertEqual(edit_masks([1, 2, 3], [1, 4, 3, 5]),
                         ((False, True, False), (False, True, False, True)))

    def test_losses_soft_ties_padding_and_gradients(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional torch not installed")
        chosen = torch.tensor([[2., 4., 999.]], requires_grad=True)
        rejected = torch.tensor([[1., 1., 999.]], requires_grad=True)
        mask = torch.tensor([[1., 1., 0.]])
        loss = fine_grained_reward_loss(chosen, rejected, mask, mask)
        self.assertAlmostEqual(loss.item(), math.log1p(math.exp(-2)), places=6)
        loss.backward()
        self.assertEqual(chosen.grad[0, 2].item(), 0)
        self.assertLess(chosen.grad[0, 0].item(), 0)
        tie = preference_reward_loss(chosen, chosen, mask, mask, [0.5])
        self.assertAlmostEqual(tie.item(), math.log(2), places=6)
        with self.assertRaises(ValueError):
            fine_grained_reward_loss(chosen, rejected, mask * 0, mask)


if __name__ == "__main__":
    unittest.main()
