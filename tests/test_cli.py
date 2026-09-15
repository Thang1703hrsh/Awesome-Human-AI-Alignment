import unittest

from human_alignment.cli import build_parser


class CLITests(unittest.TestCase):
    def test_distillation_preparation_command(self):
        args = build_parser().parse_args(
            [
                "prepare",
                "distillation",
                "vpd",
                "--dataset",
                "data",
                "--teacher",
                "teacher",
                "--output",
                "prepared",
                "--teacher-score-mode",
                "pairwise",
            ]
        )
        self.assertEqual(args.objective, "vpd")
        self.assertEqual(args.teacher_score_mode, "pairwise")

    def test_checkpoint_generation_options(self):
        args = build_parser().parse_args(
            [
                "generate",
                "--model",
                "checkpoint",
                "--adapter",
                "adapter",
                "--prompt",
                "prompt",
                "--device-map",
                "auto",
            ]
        )
        self.assertEqual(args.adapter, "adapter")
        self.assertEqual(args.device_map, "auto")


if __name__ == "__main__":
    unittest.main()
