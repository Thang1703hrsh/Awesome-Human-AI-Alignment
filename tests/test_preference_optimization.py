import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from human_alignment import IPO, IPOConfig, PreferenceOptimizationConfig, align
from human_alignment.catalog import validate_catalog
from human_alignment.config import load_config
from human_alignment.exceptions import ConfigurationError
from human_alignment.mechanisms.training.preference.data import PreferenceCollator
from human_alignment.mechanisms.training.preference.losses import (
    bregman_loss, importance_weights, preference_loss, rank_weights,
    tdpo_loss, token_statistics, triplet_loss,
)
from human_alignment.results import AlignmentRun


class PreferenceLossTests(unittest.TestCase):
    def test_bregman_closed_forms_and_gradients(self):
        z = torch.tensor([-0.7, 0.1, 1.1], dtype=torch.float64, requires_grad=True)
        r = z.exp()
        lam, scale = 0.2, 4.
        # Independent generator definition rather than reusing the closed forms.
        generators = {
            "kliep": (lambda r: r * r.log() - r, lambda r: r.log()),
            "lsif": (lambda r: r.square(), lambda r: 2 * r),
            "ba": (lambda r: (r ** (1 + lam) - r) / lam,
                   lambda r: ((1 + lam) * r ** lam - 1) / lam),
            "sba": (lambda r: (r ** (1 + lam) - r) / (scale * lam * (1 + lam)),
                    lambda r: ((1 + lam) * r ** lam - 1) / (scale * lam * (1 + lam))),
        }
        for name, (h, derivative) in generators.items():
            with self.subTest(generator=name):
                risk = derivative(r) * r - h(r) - derivative(1 / r)
                actual = bregman_loss(z, name, lam, scale)
                # Constants differ between conventions; derivatives must agree.
                expected_grad, = torch.autograd.grad(risk.sum(), z, retain_graph=True)
                actual_grad, = torch.autograd.grad(actual.sum(), z, retain_graph=True)
                torch.testing.assert_close(expected_grad, actual_grad)
        torch.testing.assert_close(bregman_loss(z, "logistic"), -torch.nn.functional.logsigmoid(-z))

    def test_tdpo2_detaches_only_chosen_kl(self):
        values = [torch.tensor([v], requires_grad=True) for v in (1., 0.2, 0.3, 0.5)]
        loss = tdpo_loss(*values, beta=0.3, alpha=0.7)
        expected = -torch.nn.functional.logsigmoid(torch.tensor(0.3 * (0.8 - 0.7 * 0.2)))
        torch.testing.assert_close(loss.squeeze(), expected)
        grads = torch.autograd.grad(loss.sum(), values, allow_unused=True)
        self.assertIsNone(grads[2])
        self.assertGreater(grads[3].abs().item(), 0)
        self.assertIsNotNone(torch.autograd.grad(tdpo_loss(*values, tdpo2=False).sum(), values[2])[0])

    def test_causal_mask_and_reference_are_frozen(self):
        torch.manual_seed(4)
        p = torch.randn(2, 5, 7, requires_grad=True)
        r = torch.randn_like(p, requires_grad=True)
        labels = torch.tensor([[-100, -100, 2, 3, -100], [-100, 1, 2, 3, 4]])
        ratio, fkl, rkl, mask = token_statistics(p, r, labels)
        expected = p[0, 1].log_softmax(-1)[2] - r[0, 1].log_softmax(-1)[2]
        torch.testing.assert_close(ratio[0, 1], expected)
        self.assertEqual(ratio[0, 0].item(), 0)
        self.assertTrue((fkl >= -1e-6).all() and (rkl >= -1e-6).all())
        (ratio.sum() + fkl.sum() + rkl.sum()).backward()
        self.assertIsNone(r.grad)
        torch.testing.assert_close(p.grad[:, -1], torch.zeros_like(p.grad[:, -1]))
        with self.assertRaises(ValueError):
            token_statistics(p, r, torch.full_like(labels, -100))

    def test_uniform_tis_weights_reduce_to_dpo(self):
        torch.manual_seed(5)
        p, r = torch.randn(4, 5, 7), torch.randn(4, 5, 7)
        labels = torch.tensor([[-100, -100, 2, 3, 4]] * 4)
        cfg = PreferenceOptimizationConfig()
        ratio, _, _, _ = token_statistics(p, r, labels)
        margin = ratio.sum(-1)
        expected = torch.nn.functional.softplus(-cfg.beta * (margin[:2] - margin[2:]))
        actual = preference_loss("tis_dpo", p, r, labels, cfg, weights=torch.ones_like(labels).float())
        torch.testing.assert_close(actual, expected)
        with self.assertRaises(ValueError):
            preference_loss("tis_dpo", p, r, labels, cfg)

    def test_tbpo_masks_unequal_responses_and_trains_baseline(self):
        torch.manual_seed(6)
        p = torch.randn(2, 6, 7, requires_grad=True)
        r = torch.randn_like(p)
        labels = torch.tensor([[-100, -100, 2, 3, 4, 1], [-100, -100, 2, 4, -100, -100]])
        b = torch.randn(2, 6, requires_grad=True)
        cfg = PreferenceOptimizationConfig()
        for method in ("tbpo_q", "tbpo_a"):
            result = preference_loss(method, p, r, labels, cfg, baselines=b)
            changed = p.detach().clone()
            changed[0, 3:] += torch.randn_like(changed[0, 3:]) * 10
            # Only the two overlapping response positions contribute.
            torch.testing.assert_close(result, preference_loss(method, changed, r, labels, cfg, baselines=b))
        preference_loss("tbpo_q", p, r, labels, cfg, baselines=b).sum().backward()
        self.assertGreater(b.grad.abs().sum().item(), 0)
        self.assertGreater(p.grad.abs().sum().item(), 0)

    def test_importance_normalization_and_rank_direction(self):
        scores = torch.tensor([[1., 3., 2., 99.]])
        mask = torch.tensor([[True, True, True, False]])
        weights = importance_weights(scores, mask)
        torch.testing.assert_close(weights.sum(), torch.tensor(3.))
        self.assertEqual(weights[0, -1], 0)
        torch.testing.assert_close(rank_weights(scores, mask), torch.tensor([[0.7, 1.3, 1., 0.]]))

    def test_triplet_backpropagates_through_anchor(self):
        anchor = torch.tensor([[0.8, 0.7]], requires_grad=True)
        pos, neg = torch.zeros_like(anchor), torch.ones_like(anchor)
        mask = torch.ones_like(anchor, dtype=torch.bool)
        triplet_loss(anchor, pos, neg, mask, mask, mask).sum().backward()
        self.assertGreater(anchor.grad.abs().sum().item(), 0)


class PreferenceAPITests(unittest.TestCase):
    def test_registry_and_recipes(self):
        root = Path(__file__).resolve().parents[1]
        for method in ("ipo", "bpo", "tdpo", "tis_dpo", "ti_dpo", "tbpo_q", "tbpo_a"):
            with self.subTest(method=method):
                obj = align(method, model="m", dataset="d", train=False)
                self.assertEqual(obj.method_id, method)
                recipe = load_config(root / "recipes" / "preference_optimization" / f"{method}.toml")
                obj.config_type(**recipe.parameters)
        self.assertTrue(validate_catalog().valid)

    def test_ipo_forwards_ipo_loss(self):
        with patch("human_alignment.integrations.trl.TRLBackend.train", return_value=AlignmentRun(model="m")) as train:
            IPO(model="m", dataset=[{"prompt": "p", "chosen": "a", "rejected": "b"}]).train()
        self.assertEqual(train.call_args.kwargs["method"], "dpo")
        self.assertEqual(train.call_args.kwargs["config"].loss_type, "ipo")
        with self.assertRaises(ConfigurationError):
            IPOConfig(loss_type="sigmoid")

    def test_invalid_native_configuration(self):
        for kwargs in ({"beta": 0}, {"importance_mix": 2}, {"generator": "unknown"},
                       {"triplet_weight": -1}, {"bregman_scale": math.nan},
                       {"max_length": 128, "max_prompt_length": 128}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ConfigurationError):
                PreferenceOptimizationConfig(**kwargs)


@unittest.skipUnless(importlib.util.find_spec("accelerate") and importlib.util.find_spec("transformers"),
                     "Requires optional preference training dependencies")
class PreferenceTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        vocab = {t: i for i, t in enumerate(["<pad>", "<eos>", "<bos>", "<unk>", "Q", "good", "bad", "answer", "yes", "no"])}
        tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        cls.tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="<pad>",
            eos_token="<eos>", bos_token="<bos>", unk_token="<unk>")

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def model(self):
        from transformers import GPT2Config, GPT2LMHeadModel
        return GPT2LMHeadModel(GPT2Config(vocab_size=10, n_positions=32, n_embd=8,
            n_layer=1, n_head=1, bos_token_id=2, eos_token_id=1, pad_token_id=0))

    def config(self, path, **extra):
        return PreferenceOptimizationConfig(output_dir=str(path), max_length=16, max_prompt_length=4,
            batch_size=1, learning_rate=1e-3, anchor_max_new_tokens=3,
            extra_args={"max_steps": 1, "save_strategy": "no", "use_cpu": True,
                        "disable_tqdm": True, "dataloader_pin_memory": False, **extra})

    def test_all_native_methods_train_and_export(self):
        from transformers import AutoModelForCausalLM
        rows = [{"prompt": "Q", "chosen": "good answer", "rejected": "bad answer",
                 "chosen_weights": [0.7, 1.3], "rejected_weights": [1.3, 0.7]}]
        for method in ("bpo", "tdpo", "tis_dpo", "ti_dpo", "tbpo_q", "tbpo_a"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                model = self.model()
                before = model.transformer.wte.weight.detach().clone()
                reference = copy.deepcopy(model)
                run = align(method, model=model, dataset=rows, config=self.config(directory),
                    backend_options={"tokenizer": self.tokenizer, "reference_model": reference})
                self.assertTrue(torch.isfinite(torch.tensor(run.metrics["train_loss"])))
                self.assertFalse(torch.equal(before, model.transformer.wte.weight))
                torch.testing.assert_close(before, reference.transformer.wte.weight)
                loaded = AutoModelForCausalLM.from_pretrained(directory)
                torch.testing.assert_close(loaded.transformer.wte.weight, model.transformer.wte.weight)
                self.assertEqual(json.loads((Path(directory) / "preference_config.json").read_text())["method"], method)
                if method == "tbpo_q":
                    self.assertTrue((Path(directory) / "baseline_head.safetensors").exists())

    def test_tis_estimates_weights_from_contrastive_models(self):
        rows = [{"prompt": "Q", "chosen": "good answer", "rejected": "bad answer"}]
        with tempfile.TemporaryDirectory() as directory:
            run = align("tis_dpo", model=self.model(), dataset=rows, config=self.config(directory),
                backend_options={"tokenizer": self.tokenizer, "positive_model": self.model(),
                                 "negative_model": self.model()})
            self.assertTrue(math.isfinite(run.metrics["train_loss"]))

    def test_checkpoint_restores_baseline_and_optimizer(self):
        rows = [{"prompt": "Q", "chosen": "good answer", "rejected": "bad answer"}]
        with tempfile.TemporaryDirectory() as directory:
            initial = self.model()
            reference = copy.deepcopy(initial)
            cfg = self.config(directory, save_strategy="steps", save_steps=1)
            align("tbpo_q", model=initial, dataset=rows, config=cfg,
                  backend_options={"tokenizer": self.tokenizer, "reference_model": reference})
            checkpoint = Path(directory) / "checkpoint-1"
            state = torch.load(checkpoint / "pytorch_model.bin", weights_only=True)
            self.assertIn("baseline.weight", state)
            self.assertTrue((checkpoint / "optimizer.pt").exists())
            from dataclasses import replace
            resumed = replace(cfg, resume_from_checkpoint=str(checkpoint),
                              extra_args={**cfg.extra_args, "max_steps": 2})
            run = align("tbpo_q", model=self.model(), dataset=rows, config=resumed,
                        backend_options={"tokenizer": self.tokenizer, "reference_model": reference})
            self.assertTrue(math.isfinite(run.metrics["train_loss"]))

    def test_attribution_preserves_mode_and_parameter_gradients(self):
        from human_alignment.mechanisms.training.preference.trainer import attribution
        model = self.model().train()
        batch = PreferenceCollator(self.tokenizer, 16, 4)(
            [{"prompt": "Q", "chosen": "good answer", "rejected": "bad"}])
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        weights = attribution(model, batch["chosen_input_ids"], batch["chosen_attention_mask"],
                              batch["chosen_labels"], self.config("unused"))
        self.assertTrue(model.training)
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(all(torch.equal(p.grad, torch.ones_like(p)) for p in model.parameters()))

    def test_accumulation_matches_full_batch_update(self):
        rows = [{"prompt": "Q", "chosen": "good answer", "rejected": "bad answer"},
                {"prompt": "Q", "chosen": "yes", "rejected": "no"}]
        from dataclasses import replace
        initial = self.model()
        results = []
        for batch_size, accumulation in ((2, 1), (1, 2)):
            with tempfile.TemporaryDirectory() as directory:
                cfg = replace(self.config(directory), batch_size=batch_size,
                              gradient_accumulation_steps=accumulation)
                run = align("tdpo", model=copy.deepcopy(initial), dataset=rows, config=cfg,
                            backend_options={"tokenizer": self.tokenizer})
                results.append({k: v.detach().clone() for k, v in run.model.state_dict().items()})
        for key in results[0]:
            torch.testing.assert_close(results[0][key], results[1][key], atol=1e-6, rtol=1e-5)

    def test_weight_alignment_and_padding(self):
        collator = PreferenceCollator(self.tokenizer, 16, 4)
        row = {"prompt": "Q", "chosen": "good answer", "rejected": "bad",
               "chosen_weights": [1., 2.], "rejected_weights": [1.]}
        batch = collator([row])
        self.assertEqual(batch["chosen_weights"].shape, batch["chosen_labels"].shape)
        self.assertEqual(batch["rejected_labels"][0, -1], -100)
        with self.assertRaises(ValueError):
            collator([{**row, "chosen_weights": [1.]}])
