"""CAA, reward-model best-of-N and ARGS on a tiny random Llama with an in-memory character tokenizer."""

import unittest

try:
    import torch
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, LlamaForSequenceClassification, PreTrainedTokenizerFast
except ModuleNotFoundError as exc:
    raise unittest.SkipTest(f"Requires optional torch/transformers: {exc.name}") from exc

from human_alignment.benchmarks.rewardbench2 import evaluate_rewardbench2
from human_alignment.integrations.reward_model import TransformersRewardModel
from human_alignment.mechanisms.inference import (
    ARGSDecoding,
    ContrastiveActivationAddition,
    RewardModelBestOfN,
    compute_caa_vector,
)
from human_alignment.mechanisms.inference.activation_steering import decoder_layers
from human_alignment.types import GenerationRequest, InteractionContext

CHARS = "abcdefghijklmnopqrstuvwxyz ?."


def char_tokenizer():
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, **{c: i + 3 for i, c in enumerate(CHARS)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex("."), "isolated")
    tok.decoder = decoders.Fuse()
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>",
                                   model_input_names=["input_ids", "attention_mask"])


def tiny_config(tokenizer, **kw):
    return LlamaConfig(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                       pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, **kw)


class InferenceMethodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.tok = char_tokenizer()
        cls.model = LlamaForCausalLM(tiny_config(cls.tok)).eval()
        cls.request = GenerationRequest(InteractionContext("why is the sky blue?"))

    def test_caa_vector_is_the_mean_activation_difference(self):
        examples = [("tell me", " yes", " no"), ("say", " good", " bad")]
        vector = compute_caa_vector(self.model, self.tok, examples, layer=0)
        expected = []
        with torch.no_grad():
            for prompt, pos, neg in examples:
                h = [self.model(**self.tok(prompt + c, return_tensors="pt"), output_hidden_states=True)
                     .hidden_states[1][0, -1] for c in (pos, neg)]  # hidden_states[1] = output of layer 0
                expected.append(h[0] - h[1])
        torch.testing.assert_close(vector, torch.stack(expected).mean(0))

    def test_steering_touches_only_the_instruction_end_and_is_removed(self):
        vector = torch.randn(32) * 5
        caa = ContrastiveActivationAddition(self.model, self.tok, vector, layer=0, multiplier=1.0)
        ids = self.tok("hello there", return_tensors="pt")
        with torch.no_grad():
            plain = self.model(**ids).logits
            with caa.steering():
                steered = self.model(**ids).logits
        torch.testing.assert_close(steered[0, :-1], plain[0, :-1])
        self.assertFalse(torch.allclose(steered[0, -1], plain[0, -1]))
        self.assertEqual(len(decoder_layers(self.model)[0]._forward_hooks), 0)

    def test_zero_multiplier_reproduces_plain_generation(self):
        caa = ContrastiveActivationAddition(self.model, self.tok, torch.randn(32) * 5, layer=1, max_new_tokens=8)
        ids = self.tok("why", return_tensors="pt")
        with torch.no_grad():
            plain = self.model.generate(**ids, max_new_tokens=8, do_sample=False, pad_token_id=0)
        expected = self.tok.decode(plain[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        self.assertEqual(caa.generate_text("why", multiplier=0.0), expected)
        self.assertNotEqual(caa.generate_text("why", multiplier=50.0), expected)

    def test_best_of_n_returns_the_highest_reward_candidate(self):
        bon = RewardModelBestOfN(self.model, self.tok, lambda p, r: r.count("a") - 0.01 * len(r), n=6,
                                 max_new_tokens=10, temperature=1.5)
        torch.manual_seed(1)
        result = bon.generate(None, self.request)
        scores = result.metadata["candidate_scores"]
        self.assertEqual(len(scores), 6)
        self.assertEqual(result.generations[0].score, max(scores))

    def test_args_with_zero_weight_is_greedy_decoding(self):
        args = ARGSDecoding(self.model, self.tok, lambda p, r: 0.0, weight=0.0, topk=5, max_new_tokens=8)
        ids = self.tok("why is the sky blue?", return_tensors="pt")
        with torch.no_grad():
            greedy = self.model.generate(**ids, max_new_tokens=8, do_sample=False, pad_token_id=0)
        expected = self.tok.decode(greedy[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        self.assertEqual(args.decode("why is the sky blue?")[0], expected)

    def test_args_follows_the_reward_when_weighted(self):
        args = ARGSDecoding(self.model, self.tok, lambda p, r: 10.0 * r.count("z"), weight=100.0,
                            topk=len(self.tok), max_new_tokens=6)
        output, rewards = args.decode("why")
        self.assertEqual(output, "zzzzzz")
        self.assertEqual(rewards, [10.0 * k for k in range(1, 7)])
        sampled = [ARGSDecoding(self.model, self.tok, lambda p, r: 0.0, weight=0.0, topk=5, max_new_tokens=6,
                                method="sampling", temperature=1.0, seed=3).decode("why")[0] for _ in range(2)]
        self.assertEqual(sampled[0], sampled[1])


class RewardModelTests(unittest.TestCase):
    def test_batched_scores_match_single_scores_and_feed_rewardbench(self):
        torch.manual_seed(0)
        tok = char_tokenizer()
        rm = TransformersRewardModel(LlamaForSequenceClassification(tiny_config(tok, num_labels=1)).eval(), tok,
                                     batch_size=3)
        pairs = [("q?", "a"), ("q?", "a much longer answer."), ("hi", "ok"), ("hi", "no")]
        batched = rm.score_batch(pairs)
        single = [rm.score(p, r) for p, r in pairs]
        for b, s in zip(batched, single):
            self.assertAlmostEqual(b, s, places=4)
        rows = [{"id": "1", "subset": "Focus", "prompt": "q?", "completions": ["a", "b", "c", "d"], "num_correct": 1}]
        scored, summary = evaluate_rewardbench2(rows, rm)
        self.assertEqual(len(scored[0]["scores"]), 4)
        self.assertIn(summary["subsets"]["Focus"]["score"], (0.0, 0.25, 1 / 3, 0.5, 1.0))


if __name__ == "__main__":
    unittest.main()
