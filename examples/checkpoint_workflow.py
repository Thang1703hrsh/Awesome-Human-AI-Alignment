"""Load an aligned checkpoint for generation or assurance evaluation."""

from human_alignment import CheckpointConfig, load_checkpoint


run = load_checkpoint(
    "outputs/qwen-dpo",
    config=CheckpointConfig(device_map="auto", torch_dtype="bfloat16"),
)

print(run.generate("Explain Human-AI alignment briefly."))

# The same object can be passed to an assurance method:
# report = run.evaluate(evaluator, evaluation_cases, baseline_model=base_model)
