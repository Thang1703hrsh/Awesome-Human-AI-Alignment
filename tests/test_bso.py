"""BSO / SafeBPO: loss, data transform, chat tokenization, recipes and LLM-judge protocols.

``_authors_loss`` freezes the closed forms of the authors' SafeBPO ``loss/h_function.py`` (``loss_from_logR``), so
these tests keep guarding the port after the source tree is gone.
"""

import math
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    raise unittest.SkipTest("Requires optional torch") from exc

from human_alignment.safety.api import load_safety_config
from human_alignment.safety.data.pairs import build_pairs
from human_alignment.safety.data.tokenize import IGNORE_INDEX, encode_pair
from human_alignment.safety.eval import rubric_judge as RJ
from human_alignment.safety.losses.preference import bso_loss, bso_per_sample
from human_alignment.safety.resources import profile_resources

ROOT = Path(__file__).resolve().parents[1]
RECIPES = ROOT / "recipes" / "safety_alignment" / "methods"


def _authors_loss(log_r, name, lam=0.2, s=4.0, mu=1.5):
    """Frozen copy of the authors' ``HFunc.loss_from_logR`` closed forms (additive constants dropped there)."""
    r = log_r.exp()
    if name == "logistic":
        return torch.nn.functional.softplus(log_r)
    if name == "kliep":
        return r + log_r
    if name == "ba":
        return r ** (lam + 1) - (lam + 1) / lam * r ** (-lam)
    if name == "sba":
        return r ** (lam + 1) / (s * (lam + 1)) - r ** (-lam) / (s * lam)
    if name == "phi_mu":
        return ((1 - mu) * r ** (2 - mu) - (2 - mu) * r ** (mu - 1) + 1) / ((1 - mu) * (2 - mu))
    raise ValueError(name)


def _grad(fn, x):
    x = x.clone().requires_grad_(True)
    fn(x).sum().backward()
    return x.grad


def _eq10(h, h_prime, log_r):
    """Literal BSO Eq. 14: h'(R)·R − h(R) − h'(1/R)."""
    r, inv = log_r.exp(), (-log_r).exp()
    return h_prime(r) * r - h(r) - h_prime(inv)


class BSOLossTests(unittest.TestCase):
    x = torch.linspace(-5.0, 5.0, 41, dtype=torch.float64)

    def test_gradients_match_the_authors_closed_forms(self):
        cases = [
            ("logistic", {}),
            ("kliep", {}),
            ("ba", {"lam": 0.5}),
            ("ba", {"lam": -0.4}),
            ("sba", {"lam": 0.2, "s": 4.0}),
            ("sba", {"lam": 0.3, "s": 4.0}),
            ("phi_mu", {"mu": 1.5}),
            ("phi_mu", {"mu": 0.5}),
        ]
        for name, kw in cases:
            with self.subTest(generator=name, **kw):
                ours = _grad(lambda t: bso_per_sample(t, name, **kw), self.x)
                theirs = _grad(lambda t: _authors_loss(t, name, **kw), self.x)
                torch.testing.assert_close(ours, theirs)

    def test_closed_forms_equal_eq14_for_their_generators(self):
        lam, s, mu = 0.2, 4.0, 1.5
        generators = {
            "kliep": (lambda r: r * r.log() - r + 1, lambda r: r.log(), {}),
            "lsif": (lambda r: (r - 1) ** 2, lambda r: 2 * (r - 1), {}),
            "ba": (lambda r: (r ** (1 + lam) - r) / lam,
                   lambda r: ((1 + lam) * r**lam - 1) / lam, {"lam": lam}),
            "sba": (lambda r: (r ** (1 + lam) - r) / (s * lam * (lam + 1)),
                    lambda r: ((1 + lam) * r**lam - 1) / (s * lam * (lam + 1)), {"lam": lam, "s": s}),
            "phi_mu": (lambda r: (r ** (2 - mu) - (2 - mu) * r + 1 - mu) / ((1 - mu) * (2 - mu)),
                       lambda r: (r ** (1 - mu) - 1) / (1 - mu), {"mu": mu}),
        }
        for name, (h, h_prime, kw) in generators.items():
            with self.subTest(generator=name):
                torch.testing.assert_close(
                    bso_per_sample(self.x, name, **kw), _eq10(h, h_prime, self.x)
                )

    def test_lambda_zero_is_the_kliep_limit(self):
        for name, scale in (("ba", 1.0), ("sba", 4.0)):
            limit = bso_per_sample(self.x, name, lam=0.0)
            near = bso_per_sample(self.x, name, lam=1e-5)
            torch.testing.assert_close(limit, near, atol=1e-3, rtol=1e-3)
            torch.testing.assert_close(limit * scale, bso_per_sample(self.x, "kliep"))

    def test_aliases_and_invalid_parameters(self):
        torch.testing.assert_close(
            bso_per_sample(self.x, "safe_bpo"), bso_per_sample(self.x, "phi_mu")
        )
        torch.testing.assert_close(bso_per_sample(self.x, "dpo"), bso_per_sample(self.x, "logistic"))
        with self.assertRaises(ValueError):
            bso_per_sample(self.x, "phi_mu", mu=1.0)
        with self.assertRaises(ValueError):
            bso_per_sample(self.x, "sba", lam=-1.0)

    def _loss(self, chosen_unsafe, rejected_unsafe, h=0.0, **kw):
        zero = torch.zeros(1, dtype=torch.float64)
        chosen = torch.full((1,), h, dtype=torch.float64, requires_grad=True)
        out = bso_loss(chosen, zero, zero, zero, torch.tensor([chosen_unsafe]),
                       torch.tensor([rejected_unsafe]), beta=0.1, **kw)
        return out, chosen

    def test_safety_penalty_sits_outside_beta(self):
        # log R = -(β·h + C·(s_w − s_l)); s = 1 for unsafe (authors' safe_bpo_log_r).
        out, _ = self._loss(0, 1, h=2.0, safety_penalty=30.0)
        self.assertAlmostEqual(out.diagnostics["bso/log_r"].item(), -0.2 + 30.0)
        out, _ = self._loss(1, 0, h=2.0, safety_penalty=30.0)
        self.assertAlmostEqual(out.diagnostics["bso/log_r"].item(), -0.2 - 30.0)
        out, _ = self._loss(0, 1, h=2.0, safety_penalty=30.0, penalty_inside_beta=True)
        self.assertAlmostEqual(out.diagnostics["bso/log_r"].item(), -0.1 * (2.0 - 30.0))

    def test_clamp_keeps_gradient_at_the_bound_and_zeroes_it_beyond(self):
        out, chosen = self._loss(0, 1, h=0.0, safety_penalty=30.0)  # log R = 30 exactly
        out.losses.sum().backward()
        self.assertNotEqual(chosen.grad.item(), 0.0)
        self.assertEqual(out.diagnostics["bso/frac_clamped"].item(), 0.0)
        out, chosen = self._loss(0, 1, h=-1.0, safety_penalty=30.0)  # log R = 30.1
        out.losses.sum().backward()
        self.assertEqual(chosen.grad.item(), 0.0)
        self.assertEqual(out.diagnostics["bso/frac_clamped"].item(), 1.0)
        out, chosen = self._loss(0, 1, h=-1.0, safety_penalty=30.0, log_r_max=None)
        out.losses.sum().backward()
        self.assertNotEqual(chosen.grad.item(), 0.0)

    def test_label_smoothing_mixes_both_orientations(self):
        out, _ = self._loss(0, 0, h=3.0, safety_penalty=5.0, label_smoothing=0.1)
        log_r = torch.tensor([-0.3], dtype=torch.float64)
        expected = 0.9 * bso_per_sample(log_r, "sba") + 0.1 * bso_per_sample(-log_r, "sba")
        torch.testing.assert_close(out.losses.detach(), expected)
        with self.assertRaises(ValueError):
            self._loss(0, 0, label_smoothing=0.5)

    def test_matches_authors_pipeline_at_reference_settings(self):
        torch.manual_seed(0)
        pc, pr, rc, rr = (torch.randn(32, dtype=torch.float64) * 20 for _ in range(4))
        cu, ru = torch.randint(0, 2, (32,)), torch.randint(0, 2, (32,))
        pc1 = pc.clone().requires_grad_(True)
        log_r = 0.1 * ((pr - rr) - (pc1 - rc)) - 30.0 * (cu - ru).double()
        _authors_loss(log_r.clamp(-30, 30), "sba").sum().backward()
        pc2 = pc.clone().requires_grad_(True)
        bso_loss(pc2, pr, rc, rr, cu, ru, beta=0.1, safety_penalty=30.0).losses.sum().backward()
        torch.testing.assert_close(pc2.grad, pc1.grad, rtol=1e-12, atol=0.0)


class BSODataTests(unittest.TestCase):
    ROWS = [
        {"prompt": p, "response_0": "a", "response_1": "b", "better_response_id": 0,
         "safer_response_id": 0, "is_response_0_safe": s0, "is_response_1_safe": s1}
        for p, s0, s1 in (("both_safe", True, True), ("chosen_unsafe", False, True),
                          ("rejected_unsafe", True, False), ("both_unsafe", False, False))
    ]

    def test_swap_drop_and_drop_only_transforms(self):
        swapped = {p["prompt"]: p for p in build_pairs(self.ROWS, "safedpo")}
        kept = {p["prompt"]: p for p in build_pairs(self.ROWS, "drop_both_unsafe")}
        self.assertEqual(set(swapped), {"both_safe", "chosen_unsafe", "rejected_unsafe"})
        self.assertEqual(set(kept), set(swapped))
        self.assertEqual((swapped["chosen_unsafe"]["chosen"], swapped["chosen_unsafe"]["chosen_safe"]), ("b", True))
        self.assertEqual((kept["chosen_unsafe"]["chosen"], kept["chosen_unsafe"]["chosen_safe"]), ("a", False))


class _ChatTokenizer:
    """Character-level tokenizer with a ChatML-like template; id 0 is EOS."""

    eos_token_id = 0
    eos_token = "\x00"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False):
        return "".join(f"<{m['role']}>{m['content']}\x01" for m in messages)


class BSOChatTokenizationTests(unittest.TestCase):
    def test_masks_the_rendered_user_turn_and_scores_the_assistant_turn(self):
        tok = _ChatTokenizer()
        out = encode_pair(tok, {"prompt": "hi", "chosen": "ok", "rejected": "no!"}, "chat", 64)
        prompt_len = len("<user>hi\x01")
        self.assertEqual(out["chosen_input_ids"], [ord(c) for c in "<user>hi\x01<assistant>ok\x01"])
        self.assertEqual(out["chosen_labels"][:prompt_len], [IGNORE_INDEX] * prompt_len)
        self.assertEqual(out["rejected_labels"][prompt_len:], out["rejected_input_ids"][prompt_len:])

    def test_drops_eos_inside_text_and_overlong_pairs(self):
        tok = _ChatTokenizer()
        with self.assertRaises(ValueError):
            encode_pair(tok, {"prompt": "hi", "chosen": "a\x00", "rejected": "b"}, "chat", 64)
        with self.assertRaises(ValueError):
            encode_pair(tok, {"prompt": "hi", "chosen": "x" * 100, "rejected": "b"}, "chat", 64)


class BSORecipeTests(unittest.TestCase):
    def test_reference_recipes_resolve_and_feed_the_loss_config(self):
        try:  # imports transformers.Trainer, which needs accelerate and a usable (or absent) TensorFlow
            from human_alignment.safety.methods.pairwise import PairwiseLossConfig
        except (ImportError, ValueError, RuntimeError) as exc:
            self.skipTest(f"transformers.Trainer unavailable: {exc}")

        qwen = load_safety_config(RECIPES / "bso_qwen2.5_0.5b.yaml")
        llama = load_safety_config(RECIPES / "bso_llama3.2_3b.yaml")
        unified = load_safety_config(RECIPES / "bso.yaml")
        for cfg in (qwen, llama, unified):
            self.assertEqual(cfg.method, "bso")
            # run_pairwise silently ignores unknown method_args, so every key must be a loss field.
            unknown = set(cfg.method_args) - set(PairwiseLossConfig.__dataclass_fields__)
            self.assertFalse(unknown, unknown)
        self.assertEqual(qwen.data.tokenization, "chat")
        self.assertEqual(llama.data.tokenization, "chat")
        self.assertEqual((qwen.method_args["bso_lambda"], llama.method_args["bso_lambda"]), (0.2, 0.3))
        self.assertEqual((qwen.train["learning_rate"], llama.train["learning_rate"]), (1e-6, 7e-7))
        for cfg in (qwen, llama):
            batch = cfg.train["per_device_train_batch_size"] * cfg.train["gradient_accumulation_steps"]
            self.assertEqual(batch, 16)
            self.assertEqual(cfg.train["optim"], "rmsprop")
        self.assertIn(
            "Qwen/Qwen2.5-0.5B-Instruct",
            {r.repo_id for r in profile_resources("bso_qwen2.5_0.5b")},
        )


class _Completions:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.reply(kwargs["messages"][-1]["content"], kwargs)
        if isinstance(content, Exception):
            raise content
        message = type("M", (), {"content": content})
        return type("R", (), {"choices": [type("C", (), {"message": message})]})


class _Client:
    def __init__(self, reply):
        self.chat = type("Chat", (), {})()
        self.chat.completions = _Completions(reply)


class RubricJudgeTests(unittest.TestCase):
    def test_parsers_follow_the_source(self):
        self.assertEqual(RJ.parse_score("analysis ... [[7.5]] (score)"), 7.5)
        self.assertEqual(RJ.parse_score("8\nbecause"), 8.0)
        self.assertIsNone(RJ.parse_score("no score"))
        self.assertEqual(RJ.parse_refusal("Label: partial_refusal"), "partial_refusal")
        self.assertIsNone(RJ.parse_refusal("unclear"))
        self.assertEqual(RJ.parse_safety_label("fine [[SAFE]]"), 1.0)
        self.assertEqual(RJ.parse_safety_label("[[unsafe]]"), 0.0)

    def _xstest_reply(self, prompt, kwargs):
        if "refuse to answer" in prompt:
            self.assertEqual(kwargs["temperature"], 0.0)
            return "full_refusal" if "q-refused" in prompt else ("??" if "q-odd" in prompt else "no_refusal")
        if "helpfulness and quality" in prompt:
            return "[[6]]"
        return "[[9]]" if "q-unsafe-good" in prompt else "[[2]]"

    def test_xstest_metrics_and_unparsed_fallback(self):
        rows = [
            {"prompt": "q-refused", "response": "no", "label": "safe"},
            {"prompt": "q-answered", "response": "yes", "label": "safe"},
            {"prompt": "q-odd", "response": "hm", "label": "safe"},
            {"prompt": "q-unsafe-good", "response": "no", "label": "unsafe"},
            {"prompt": "q-unsafe-bad", "response": "sure", "label": "unsafe"},
        ]
        clients = [_Client(self._xstest_reply)]
        judged, summary = RJ.xstest_judge(rows, clients, "judge", workers=1)
        self.assertEqual(summary["num_safe"], 3)
        self.assertAlmostEqual(summary["over_refusal_rate"], 1 / 3)  # unparsed counts as no_refusal
        self.assertEqual(summary["harmless_ratio"], 0.5)
        self.assertEqual(summary["avg_safety_score"], 5.5)
        self.assertEqual(summary["unparsed"]["refusal"], 1)
        _, strict = RJ.xstest_judge(rows, clients, "judge", workers=1, exclude_unparsed=True)
        self.assertEqual(strict["over_refusal_rate"], 0.5)
        with self.assertRaises(ValueError):
            RJ.xstest_judge([{"prompt": "p", "response": "r", "label": "maybe"}], clients, "judge")

    def test_rubric_judge_scores_failed_calls_as_zero_by_default(self):
        def reply(prompt, kwargs):
            if "q-fail" in prompt:
                return RuntimeError("api down")
            return "[[8]]" if "helpfulness and quality" in prompt else "[[safe]]"

        rows = [{"prompt": "q-ok", "response": "r"}, {"prompt": "q-fail", "response": "r"}]
        judged, summary = RJ.rubric_judge(rows, [_Client(reply)], "judge", workers=1, sleep=lambda _: None)
        self.assertEqual(summary["avg_helpfulness"], 4.0)
        self.assertEqual(summary["safety_rate"], 0.5)
        self.assertEqual(summary["unparsed_helpfulness"], 1)
        _, strict = RJ.rubric_judge(rows, [_Client(reply)], "judge", workers=1,
                                    exclude_unparsed=True, sleep=lambda _: None)
        self.assertEqual((strict["avg_helpfulness"], strict["safety_rate"]), (8.0, 1.0))


if __name__ == "__main__":
    unittest.main()
