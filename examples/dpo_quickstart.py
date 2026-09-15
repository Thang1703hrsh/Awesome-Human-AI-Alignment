"""Train DPO from preference pairs. Requires `pip install -e .[dpo]`."""

from human_alignment import DPO, DPOConfig, PreferenceExample


examples = [
    PreferenceExample(
        prompt="Explain Human-AI alignment briefly.",
        chosen="It concerns whether AI behavior is consistent with human objectives.",
        rejected="It only means making models larger.",
    ),
    PreferenceExample(
        prompt="How should an assistant handle uncertainty?",
        chosen="It should communicate uncertainty and avoid unsupported claims.",
        rejected="It should always answer confidently.",
    ),
]

run = DPO(
    model="Qwen/Qwen3-0.6B",
    dataset=examples,
    config=DPOConfig(
        output_dir="outputs/qwen-dpo",
        epochs=1,
        batch_size=2,
        beta=0.1,
    ),
).train()

print(run.metrics)
print(run.generate("What is Human-AI alignment?"))
