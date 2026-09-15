"""Minimal VPD example. Requires `pip install -e .[distillation]`."""

from human_alignment import VPD, PreferenceDistillationConfig, PreferenceDistillationExample


examples = [
    PreferenceDistillationExample(
        prompt="What is Human-AI alignment?",
        responses=(
            "It studies how AI behavior can remain consistent with intended human targets.",
            "It is another term for increasing model size.",
        ),
        teacher_scores=(1.0, 0.0),
    )
]

run = VPD(
    model="Qwen/Qwen3-0.6B",
    dataset=examples,
    config=PreferenceDistillationConfig(
        objective="vpd",
        output_dir="outputs/qwen-vpd",
        epochs=1,
        batch_size=1,
        max_length=1024,
        max_prompt_length=512,
    ),
).train()

print(run.metrics)
