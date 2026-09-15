import unittest
from pathlib import Path

from human_alignment.config import DPOConfig, load_config
from human_alignment.exceptions import ConfigurationError


class ConfigTests(unittest.TestCase):
    def test_dpo_example_loads(self):
        config = load_config("examples/dpo.toml")
        self.assertEqual(config.method, "dpo")
        self.assertEqual(config.model, "Qwen/Qwen3-0.6B")
        self.assertEqual(config.parameters["beta"], 0.1)

    def test_invalid_training_values_fail_early(self):
        with self.assertRaises(ConfigurationError):
            DPOConfig(batch_size=0)

    def test_method_options_load_from_recipe(self):
        config = load_config("recipes/preference_distillation/ctpd.toml")
        self.assertEqual(config.method_options["reference_model"], "models/teacher_dpo")

    def test_all_distillation_recipes_load(self):
        paths = Path("recipes/preference_distillation").glob("*.toml")
        recipes = [load_config(path) for path in paths]
        self.assertEqual(len(recipes), 9)
        self.assertIn("dckd", {recipe.method for recipe in recipes})


if __name__ == "__main__":
    unittest.main()
