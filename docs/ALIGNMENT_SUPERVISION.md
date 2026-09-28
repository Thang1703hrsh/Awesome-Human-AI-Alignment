# Human and AI feedback

This module implements the **supervision contribution** of the 17 active papers
in the survey figure. Dataset adapters and model-backed reference workflows are
registered under the supervision stage. They are not claims of full paper
reproduction, original-model availability, or matching published benchmark scores.
CriticGPT, Rule Based Rewards, and Self-Refine were commented out in the requested
list and are not part of this addition.

## Paper-to-code map and sources

| Paper / official source | Registry ID | Implemented scope |
|---|---|---|
| [Christiano et al.](https://arxiv.org/abs/1706.03741) | `human_preferences` | Trajectory comparisons with soft/tie labels; differentiable summed-reward Bradley-Terry loss. No Atari/MuJoCo environment or active-query loop. |
| [Learning to Summarize](https://github.com/openai/summarize-from-feedback) | `summarization_feedback` | Official comparison-record adapter, preserving split and prompt metadata. |
| [InstructGPT](https://arxiv.org/abs/2203.02155) | `instructgpt_feedback` | User-provided demonstrations and rankings; original private training data is not bundled. |
| [Helpful and Harmless RLHF](https://github.com/anthropics/hh-rlhf) | `hh_rlhf` | Split shared transcript at the last assistant boundary, retaining multi-turn history. |
| [OpenAssistant](https://huggingface.co/datasets/OpenAssistant/oasst1) | `openassistant` | Reconstruct flat message ancestry; compare ranked assistant siblings; reject missing/cyclic ancestry. |
| [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) | `helpsteer2` | Group by prompt, construct helpfulness preference pairs, preserve all five ratings. Complexity/verbosity are not treated as quality scores. |
| [Xu et al., Fine-grained Supervision](https://aclanthology.org/2024.acl-short.62/) | `fine_grained_feedback` | Minimal-edit pairs, edit masks, exact mean-token reward loss (Eq. 1–3). Token-reward PPO integration is not included. |
| [RLHF-V](https://github.com/RLHF-V/RLHF-V) | `rlhf_v` | Official dataset `text` JSON and image ingestion. No Muffin model or specialized dense-DPO trainer. |
| [MM-RLHF](https://github.com/Kwai-YuanQi/MM-RLHF) | `mm_rlhf` | Official `models_output` / `final_ranking` adapter with image/video and critique metadata. No MM-DPO or multimodal reward-model training. |
| [WildFeedback](https://huggingface.co/datasets/microsoft/WildFeedback) | `wildfeedback` | Explicit normalized preference interface retaining in-situ user feedback and hybrid provenance. Raw SAT/DSAT extraction is not implemented. |
| [Constitutional AI](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback) | `constitutional_ai` | Sequential principle-based critique/revision for SFT; separate constitutional preference labeling for reward training. |
| [RLAIF vs. RLHF](https://arxiv.org/abs/2309.00267) | `rlaif` | Order-averaged soft preference labels; explicit optional hard-label conversion. No direct-RLAIF online policy optimizer. |
| [Self-Instruct](https://github.com/yizhongw/self-instruct) | `self_instruct` | Seed bootstrapping, classification detection, instance generation, ROUGE-L similarity filtering and deduplication. |
| [UltraFeedback](https://github.com/OpenBMB/UltraFeedback) | `ultrafeedback` | Four-aspect ratings aggregated into strict pairs; original annotations retained. |
| [LLM-as-a-Judge](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) | `llm_judge` | Pairwise judging with reversed-order consistency; abstains on ties/inconsistent winners. No MT-Bench runner. |
| [G-Eval](https://github.com/nlpyang/geval) | `g_eval` | Generate evaluation steps, then probability-weighted scoring. Requires score probabilities from a real backend. |
| [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020) | `self_rewarding` | Generate/score candidates using the current model, train on best/worst pairs, refresh the backend each round. |

Paper links were checked on 2026-09-28. Implementations are original code based on
the linked methods and schemas; upstream repositories are not vendored. Obtain
datasets/models separately under their respective licenses. Dataset adapters are
marked `adapter`; partial research workflows are marked `reference` in the catalog.
Human and AI source abstractions themselves are also registered.

The bibliography currently contains duplicate `xu2024finegrained` entries referring
to different works. This module follows **Dehong Xu et al., ACL 2024, 673–680**,
the title requested in the figure, not the other paper sharing that key. Its
experiments use AI edits; set `annotation_source="ai"` for those records. The same
supervision format also supports human edits. Bibliography entries were not changed.

## Convert data for existing trainers

```python
from human_alignment.registry import default_registry
from human_alignment.supervision import prepare_dataset

provider = default_registry().create("summarization_feedback", records=[{
    "info": {"post": "A short source document."},
    "summaries": [{"text": "Incorrect summary"}, {"text": "Faithful summary"}],
    "choice": 1,
    "split": "train",
}])
signals = provider.convert()
dataset = prepare_dataset(signals, "preference").data
# DPO(model=..., dataset=dataset, output_dir=...).train()
```

The data includes `metadata.source`, `confidence`, and `provenance`. Split/filter
upstream records before converting; adapters do not merge train/evaluation splits
automatically. Dataset providers can also be used in `AlignmentPipeline` with
`contexts=()`. Filtering happens on the supplied records, not pipeline contexts.

```bash
hai-align prepare feedback hh_rlhf --dataset hh_train.jsonl --output prepared/hh.jsonl
hai-align methods --category human_feedback
hai-align methods --category ai_feedback
```

CLI input is local JSON/JSONL and output is a **new** JSONL file. It refuses to
overwrite an existing file. An empty set of strict pairs is reported as an empty
dataset error. For demonstrations, use `--kind instruction`. Mixed demonstration
and preference inputs should be separated first. Soft trajectory/RLAIF labels
require the corresponding reward objective, not the hard-pair export command.

Supported row schemas (in addition to the upstream schemas in the table):

```python
# human_preferences: probability the FIRST segment is preferred; 0.5 = tie
{"segments": [["state-action-1"], ["state-action-2"]], "preference_probability": 0.5}
# instructgpt_feedback: either completion or responses + ranks (smaller is better)
{"prompt": "q", "completion": "a"}
{"prompt": "q", "responses": ["a", "b"], "ranks": [0, 1]}
# fine_grained_feedback
{"prompt": "q", "original": "incorrect", "corrected": "corrected"}
# wildfeedback: NORMALIZED schema, not an automatic parser of upstream raw logs
{"prompt": "q", "chosen": "revised", "rejected": "original", "user_feedback": "Be concise"}
```

OpenAssistant expects the flat message export, not nested trees. RLHF-V accepts
the official `text` JSON string plus `image` or `image_path`. MM-RLHF preserves
`image` / `video` references. Multimodal prompts remain structured mappings and
**must be passed to a multimodal processor/trainer**; the existing text trainers
are not made multimodal by these adapters. Use media paths for JSON export, not
decoded PIL images. Tied rankings produce no strict pair. WildFeedback is marked
hybrid because human interactions can be processed/augmented by AI.

## Use a local model or an API-backed generator

```python
from human_alignment import load_checkpoint
from human_alignment.supervision import JSONFeedbackBackend, LLMJudge, prepare_dataset
from human_alignment.types import InteractionContext

model = load_checkpoint("path/to/local/instruction-model")
backend = JSONFeedbackBackend(
    lambda prompt: model.generate(prompt, max_new_tokens=512)
)
judge = LLMJudge(backend, model_name="local-checkpoint")
contexts = [InteractionContext("Explain gravity", metadata={
    "responses": ["Mass attracts mass.", "Gravity is sound."]
})]
signals = judge.collect(contexts, specification=None)
if signals:  # ties or disagreement after swapping produce no pair
    dataset = prepare_dataset(signals, "preference").data
```

`JSONFeedbackBackend` supplies task prompts and parses JSON. No API keys, paid
calls, model downloads, or training run occur at import time. Any `str -> str`
completion callable may be used. Invalid model outputs raise `DatasetFormatError`;
the caller owns retry, persistence, batching and rate-limit policy.

For `RLAIF` and `GEval`, supply `probability_backend(request)` returning
`{"probability_a": p}` or `{"score_probabilities": {"1": p1, ...}}` respectively.
Use normalized label-token likelihoods or empirical sampled score frequencies.
Do not substitute model-written confidence for token probabilities. G-Eval
renormalizes retained score probability mass and records that mass in provenance.
The library does not implement tokenizer-specific label likelihood extraction.

`ConstitutionalAI(backend, principles=[...]).collect(contexts, None)` emits final
revised demonstrations; contexts may provide an initial `metadata.response`.
Its `.preferences(...)` judges two supplied responses under the constitution;
it does not blindly label every revision as preferred. `SelfInstruct(backend,
seeds=[...]).generate(rounds=...)` emits synthetic demonstrations. Their prompts
and filtering are reference implementations, not exact original experiment recipes.

`SelfRewarding(backend_factory, train_round).run(model, contexts, rounds=...)`
returns `(updated_model, history)`. The factory must bind both generation and
judging to the supplied current model. The training callback can normalize signals
with `prepare_dataset` and invoke the existing DPO API, returning its resulting
model/checkpoint. The generic pipeline's `.collect()` interface is intentionally
not used for this iterative training workflow. Self-rewarding uses original
quality prompts here, not the exact paper's additive scoring rubric.

## Reward objectives

`preference_reward_loss(left, right, left_mask, right_mask, probabilities)` sums
per-step rewards for trajectory comparisons. `fine_grained_reward_loss(...)`
averages valid response-token rewards separately on each side. Both preserve
gradients and ignore padding. They require torch only when called. Models must
output token rewards; these functions do not construct a reward model or run PPO.

`edit_masks(original_token_ids, corrected_token_ids)` identifies changes in both
sequences, including insertions/deletions. It is an optional alignment utility;
the exact mean-reward loss does not discard unchanged-token contributions or
assume corrected and original sequences have equal lengths.

## Checks

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -p test_paper_supervision.py -v
python -m human_alignment.cli catalog validate
```

Tests cover pair direction, shared history, ties, invalid scores, image/video
retention, source provenance, judge order effects, constitutional revisions,
self-instruct deduplication, G-Eval probability aggregation, self-rewarding model
refresh, and differentiable reward losses. They use deterministic backend fixtures;
paper-scale training and live judge quality are not validated by these tests.
