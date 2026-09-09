# Awesome Human–AI Alignment

A curated collection of research on **Human–AI Alignment**, organized around a taxonomy of how alignment objectives are specified, how supervision is provided, how alignment is implemented, and how aligned behavior is evaluated and maintained.

**4 dimensions · 8 intermediate groups · 23 terminal categories · 145 unique papers**

## 🔔 News

- **[09/09/2026]** 🚀 Initial release of the Human–AI Alignment taxonomy and curated paper collection.

## 📝 Introduction

This repository accompanies an ongoing survey of **Human–AI Alignment**. Rather than organizing the literature around a single training paradigm such as RLHF or preference optimization, we use a broader taxonomy covering the full alignment pipeline: **specification, supervision, mechanisms, and assurance**.

The taxonomy is intentionally **non-exclusive**. A paper may contribute to more than one branch—for example, a work can introduce a preference-optimization mechanism while also addressing safety or multimodal alignment. The taxonomy figure therefore shows only representative works for readability, while the lists below provide the comprehensive categorized collection.

Within each terminal category, papers are listed from newer to older work where publication-year metadata is available.

## 📊 Taxonomy

Below is the taxonomy summarizing the Human–AI Alignment landscape:

![Human–AI Alignment Taxonomy](./Figure/Taxonomy.pdf)

📄 [View the high-resolution taxonomy PDF](./Figure/Taxonomy.pdf)

The taxonomy is organized around four complementary questions:

| Dimension | Question | Sections |
|---|---|---|
| **Alignment Specification** | What should an AI system align to, and whose objectives or values should count? | [I](#section-i-task-and-assistance-alignment) · [II](#section-ii-safety-alignment) · [III](#section-iii-personalized-alignment) · [IV](#section-iv-pluralistic-and-societal-alignment) · [V](#section-v-uncertainty-and-drift) |
| **Alignment Supervision** | Where do alignment signals come from, and how are they expressed and scaled? | [VI](#section-vi-human-feedback) · [VII](#section-vii-ai-feedback) · [VIII](#section-viii-programmatic-and-verifiable-feedback) · [IX](#section-ix-demonstrations-and-preferences) · [X](#section-x-critique-and-process-feedback) · [XI](#section-xi-reliable-and-scalable-oversight) |
| **Alignment Mechanisms** | How are alignment signals translated into model behavior during training or inference? | [XII](#section-xii-reward-and-verifier-modeling) · [XIII](#section-xiii-supervised-alignment) · [XIV](#section-xiv-preference-optimization) · [XV](#section-xv-reinforcement-learning) · [XVI](#section-xvi-alignment-distillation) · [XVII](#section-xvii-steering-search-and-refinement) · [XVIII](#section-xviii-human-control) |
| **Alignment Assurance** | How do we evaluate, stress-test, preserve, interpret, and monitor alignment? | [XIX](#section-xix-behavioral-and-evaluator-evaluation) · [XX](#section-xx-adversarial-and-distribution-robustness) · [XXI](#section-xxi-alignment-preservation) · [XXII](#section-xxii-mechanistic-and-theoretical-evidence) · [XXIII](#section-xxiii-monitoring-and-auditing) |

### Cross-cutting descriptors

The tree is complemented by descriptors that cut across branches:

- **System and modality:** text, multimodal, agentic, embodied.
- **Feedback granularity:** response, span, step, token.
- **Optimization granularity:** sequence, segment, token.
- **Learning regime:** offline, online, static, continual.

Feedback granularity and optimization granularity are treated separately: the unit at which feedback is collected does not necessarily match the unit at which an optimization objective is applied.

## Alignment Specification

What should an AI system align to, and whose objectives or values should count?

| Group | Branch | Section |
|---|---|---|
| Alignment Objectives | Task & Assistance Alignment | [Section I](#section-i-task-and-assistance-alignment) |
| Alignment Objectives | Safety Alignment | [Section II](#section-ii-safety-alignment) |
| Values & Stakeholders | Personalized Alignment | [Section III](#section-iii-personalized-alignment) |
| Values & Stakeholders | Pluralistic & Societal Alignment | [Section IV](#section-iv-pluralistic-and-societal-alignment) |
| Values & Stakeholders | Uncertainty & Drift | [Section V](#section-v-uncertainty-and-drift) |

<a id="section-i-task-and-assistance-alignment"></a>

## Section I: Task & Assistance Alignment

* UltraFeedback: Boosting Language Models with Scaled AI Feedback [[Paper]](https://proceedings.mlr.press/v235/cui24f.html) ![](https://img.shields.io/badge/year-2024-red)
* HelpSteer 2: Open-source Dataset for Training Top-Performing Reward Models [[Paper]](https://doi.org/10.52202/079017-0047) ![](https://img.shields.io/badge/year-2024-red)
* Scaling Instruction-Finetuned Language Models [[Paper]](https://www.jmlr.org/papers/v25/23-0870.html) ![](https://img.shields.io/badge/year-2024-red)
* RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models [[Paper]](https://aclanthology.org/2024.acl-long.585/) ![](https://img.shields.io/badge/year-2024-red)
* Aligning Large Multimodal Models with Factually Augmented RLHF [[Paper]](https://aclanthology.org/2024.findings-acl.775/) ![](https://img.shields.io/badge/year-2024-red)
* Self-Instruct: Aligning Language Models with Self-Generated Instructions [[Paper]](https://aclanthology.org/2023.acl-long.754/) ![](https://img.shields.io/badge/year-2023-red)
* LIMA: Less Is More for Alignment [[Paper]](https://papers.neurips.cc/paper_files/paper/2023/hash/ac662d74829e4407ce1d126477f4a03a-Abstract-Conference.html) ![](https://img.shields.io/badge/year-2023-red)
* Flan Collection: The Flan Collection: Designing Data and Methods for Effective Instruction Tuning [[Paper]](https://proceedings.mlr.press/v202/longpre23a.html) ![](https://img.shields.io/badge/year-2023-red)
* SteerLM: Attribute Conditioned SFT as an (User-Steerable) Alternative to RLHF [[Paper]](https://aclanthology.org/2023.findings-emnlp.754/) ![](https://img.shields.io/badge/year-2023-red)
* Fine-Grained Human Feedback Gives Better Rewards for Language Model Training [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b8c90b65739ae8417e61eadb521f63d5-Abstract.html) ![](https://img.shields.io/badge/year-2023-red)
* DPO: Direct Preference Optimization: Your Language Model Is Secretly a Reward Model [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html) ![](https://img.shields.io/badge/year-2023-red)
* Enabling Large Language Models to Generate Text with Citations [[Paper]](https://aclanthology.org/2023.emnlp-main.398/) ![](https://img.shields.io/badge/year-2023-red)
* FLAN: Finetuned Language Models Are Zero-Shot Learners [[Paper]](https://research.google/pubs/finetuned-language-models-are-zero-shot-learners/) ![](https://img.shields.io/badge/year-2022-red)
* T0: Multitask Prompted Training Enables Zero-Shot Task Generalization [[Paper]](https://arxiv.org/abs/2110.08207) ![](https://img.shields.io/badge/year-2022-red)
* Natural Instructions: Cross-Task Generalization via Natural Language Crowdsourcing Instructions [[Paper]](https://aclanthology.org/2022.acl-long.244/) ![](https://img.shields.io/badge/year-2022-red)
* Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks [[Paper]](https://aclanthology.org/2022.emnlp-main.340/) ![](https://img.shields.io/badge/year-2022-red)
* InstructGPT: Training Language Models to Follow Instructions with Human Feedback [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract.html) ![](https://img.shields.io/badge/year-2022-red)
* Helpful and Harmless Assistant: Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback [[Paper]](https://arxiv.org/abs/2204.05862) ![](https://img.shields.io/badge/year-2022-red)
* Sparrow: Improving Alignment of Dialogue Agents via Targeted Human Judgements [[Paper]](https://arxiv.org/abs/2209.14375) ![](https://img.shields.io/badge/year-2022-red)
* General Language Assistant: A General Language Assistant as a Laboratory for Alignment [[Paper]](https://arxiv.org/abs/2112.00861) ![](https://img.shields.io/badge/year-2021-red)
* WebGPT: Browser-Assisted Question-Answering with Human Feedback [[Paper]](https://arxiv.org/abs/2112.09332) ![](https://img.shields.io/badge/year-2021-red)

<a id="section-ii-safety-alignment"></a>

## Section II: Safety Alignment

* Safe RLHF: Safe Reinforcement Learning from Human Feedback [[Paper]](https://proceedings.iclr.cc/paper_files/paper/2024/hash/dd1577afd396928ed64216f3f1fd5556-Abstract-Conference.html) ![](https://img.shields.io/badge/year-2024-red)
* Rule Based Rewards for Language Model Safety [[Scholar]](https://scholar.google.com/scholar?q=Rule+Based+Rewards+for+Language+Model+Safety) ![](https://img.shields.io/badge/year-2024-red)
* Circuit Breakers: Improving Alignment and Robustness with Circuit Breakers [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2024/hash/97ca7168c2c333df5ea61ece3b3276e1-Abstract-Conference.html) ![](https://img.shields.io/badge/year-2024-red)
* BeaverTails: Towards Improved Safety Alignment of LLM via a Human-Preference Dataset [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/4dbb61cb68671edc4ca3712d70083b9f-Abstract.html) ![](https://img.shields.io/badge/year-2023-red)
* Helpful and Harmless Assistant: Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback [[Paper]](https://arxiv.org/abs/2204.05862) ![](https://img.shields.io/badge/year-2022-red)
* Constitutional AI: Harmlessness from AI Feedback [[Paper]](https://arxiv.org/abs/2212.08073) ![](https://img.shields.io/badge/year-2022-red)
* General Language Assistant: A General Language Assistant as a Laboratory for Alignment [[Paper]](https://arxiv.org/abs/2112.00861) ![](https://img.shields.io/badge/year-2021-red)

<a id="section-iii-personalized-alignment"></a>

## Section III: Personalized Alignment

* Persona-judge: Personalized Alignment of Large Language Models via Token-level Self-judgment [[Paper]](https://aclanthology.org/2025.findings-acl.260/) ![](https://img.shields.io/badge/year-2025-red)
* Personalized Alignment Survey: A Survey on Personalized Alignment—The Missing Piece for Large Language Models in Real-World Applications [[Paper]](https://aclanthology.org/2025.findings-acl.277/) ![](https://img.shields.io/badge/year-2025-red)
* Personalized Language Modeling from Personalized Human Feedback [[Paper]](https://arxiv.org/abs/2402.05133) ![](https://img.shields.io/badge/year-2024-red)
* Personalizing RLHF: Personalizing Reinforcement Learning from Human Feedback [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5e1c255653eb98cef13f45b2d337c882-Abstract-Conference.html) ![](https://img.shields.io/badge/year-2024-red)
* Personalized Soups: Personalized Large Language Model Alignment via Post-hoc Parameter Merging [[Paper]](https://arxiv.org/abs/2310.11564) ![](https://img.shields.io/badge/year-2023-red)

<a id="section-iv-pluralistic-and-societal-alignment"></a>

## Section IV: Pluralistic & Societal Alignment

* Steerable Pluralism: Pluralistic Alignment via Few-Shot Comparative Regression [[Paper]](https://doi.org/10.1609/aies.v8i1.36527) ![](https://img.shields.io/badge/year-2025-red)
* Distributional Preference Learning: Understanding and Accounting for Hidden Context in RLHF [[Scholar]](https://scholar.google.com/scholar?q=Distributional+Preference+Learning%3A+Understanding+and+Accounting+for+Hidden+Context+in+RLHF) ![](https://img.shields.io/badge/year-2024-red)
* Collective Constitutional AI: Aligning a Language Model with Public Input [[Paper]](https://arxiv.org/abs/2406.07814) ![](https://img.shields.io/badge/year-2024-red)
* PRISM: The PRISM Alignment Dataset: What Participatory, Representative and Individualised Human Feedback Reveals About the Subjective and Multicultural Alignment of Large Language Models [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2024/hash/be2e1b68b44f2419e19f6c35a1b8cf35-Abstract-Datasets_and_Benchmarks_Track.html) ![](https://img.shields.io/badge/year-2024-red)
* CultureLLM: Incorporating Cultural Differences into Large Language Models [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2024/hash/9a16935bf54c4af233e25d998b7f4a2c-Abstract-Conference.html) ![](https://img.shields.io/badge/year-2024-red)
* Modular Pluralism: Pluralistic Alignment via Multi-LLM Collaboration [[Paper]](https://aclanthology.org/2024.emnlp-main.240/) ![](https://img.shields.io/badge/year-2024-red)
* Pluralistic Alignment: Position: A Roadmap to Pluralistic Alignment [[Scholar]](https://scholar.google.com/scholar?q=Position%3A+A+Roadmap+to+Pluralistic+Alignment) ![](https://img.shields.io/badge/year-2024-red)
* Rule Based Rewards for Language Model Safety [[Scholar]](https://scholar.google.com/scholar?q=Rule+Based+Rewards+for+Language+Model+Safety) ![](https://img.shields.io/badge/year-2024-red)
* AI Control: Improving Safety Despite Intentional Subversion [[Paper]](https://proceedings.mlr.press/v235/greenblatt24a.html) ![](https://img.shields.io/badge/year-2024-red)
* Finding Agreement: Fine-Tuning Language Models to Find Agreement among Humans with Diverse Preferences [[Paper]](https://proceedings.neurips.cc/paper/2022/hash/f978c8f3b5f399cae464e85f72e28503-Abstract-Conference.html) ![](https://img.shields.io/badge/year-2022-red)
* Constitutional AI: Harmlessness from AI Feedback [[Paper]](https://arxiv.org/abs/2212.08073) ![](https://img.shields.io/badge/year-2022-red)
* The Moral Machine Experiment [[Paper]](https://doi.org/10.1038/s41586-018-0637-6) ![](https://img.shields.io/badge/year-2018-red)

<a id="section-v-uncertainty-and-drift"></a>

## Section V: Uncertainty & Drift

* COPR: Continual Human Preference Learning via Optimal Policy Regularization [[Paper]](https://aclanthology.org/2025.findings-acl.281/) ![](https://img.shields.io/badge/year-2025-red)
* Lifelong Safety Alignment for Language Models [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1ca2f4528792b31eab7a3e7f6a1c130b-Abstract-Conference.html) ![](https://img.shields.io/badge/year-2025-red)
* Inverse Reward Design [[Scholar]](https://scholar.google.com/scholar?q=Inverse+Reward+Design) ![](https://img.shields.io/badge/year-2017-red)
* Off-Switch Game: The Off-Switch Game [[Paper]](https://doi.org/10.24963/ijcai.2017/32) ![](https://img.shields.io/badge/year-2017-red)
* Cooperative IRL: Cooperative Inverse Reinforcement Learning [[Scholar]](https://scholar.google.com/scholar?q=Cooperative+Inverse+Reinforcement+Learning) ![](https://img.shields.io/badge/year-2016-red)
* Apprenticeship Learning via IRL: Apprenticeship Learning via Inverse Reinforcement Learning [[Scholar]](https://scholar.google.com/scholar?q=Apprenticeship+Learning+via+Inverse+Reinforcement+Learning) ![](https://img.shields.io/badge/year-2004-red)
* Inverse RL: Algorithms for Inverse Reinforcement Learning [[Scholar]](https://scholar.google.com/scholar?q=Algorithms+for+Inverse+Reinforcement+Learning) ![](https://img.shields.io/badge/year-2000-red)

## Alignment Supervision

Where do alignment signals come from, and how are they expressed and scaled?

| Group | Branch | Section |
|---|---|---|
| Feedback Source | Human Feedback | [Section VI](#section-vi-human-feedback) |
| Feedback Source | AI Feedback | [Section VII](#section-vii-ai-feedback) |
| Feedback Source | Programmatic & Verifiable Feedback | [Section VIII](#section-viii-programmatic-and-verifiable-feedback) |
| Feedback & Oversight | Demonstrations & Preferences | [Section IX](#section-ix-demonstrations-and-preferences) |
| Feedback & Oversight | Critique & Process Feedback | [Section X](#section-x-critique-and-process-feedback) |
| Feedback & Oversight | Reliable & Scalable Oversight | [Section XI](#section-xi-reliable-and-scalable-oversight) |

<a id="section-vi-human-feedback"></a>

## Section VI: Human Feedback

* MM-RLHF: The Next Step Forward in Multimodal LLM Alignment [[Paper]](https://icml.cc/virtual/2025/poster/45124) ![](https://img.shields.io/badge/year-2025-red)
* RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-Grained Correctional Human Feedback [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/html/Yu_RLHF-V_Towards_Trustworthy_MLLMs_via_Behavior_Alignment_from_Fine-grained_Correctional_CVPR_2024_paper.html) ![](https://img.shields.io/badge/year-2024-red)
* Fine-Grained Human Feedback Gives Better Rewards for Language Model Training [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b8c90b65739ae8417e61eadb521f63d5-Abstract.html) ![](https://img.shields.io/badge/year-2023-red)
* InstructGPT: Training Language Models to Follow Instructions with Human Feedback [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract.html) ![](https://img.shields.io/badge/year-2022-red)
* Helpful and Harmless Assistant: Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback [[Paper]](https://arxiv.org/abs/2204.05862) ![](https://img.shields.io/badge/year-2022-red)
* Sparrow: Improving Alignment of Dialogue Agents via Targeted Human Judgements [[Paper]](https://arxiv.org/abs/2209.14375) ![](https://img.shields.io/badge/year-2022-red)
* Learning to Summarize with Human Feedback [[Paper]](https://proceedings.neurips.cc/paper/2020/hash/1f89885d556929e98d3ef9b86448f951-Abstract.html) ![](https://img.shields.io/badge/year-2020-red)
* Fine-Tuning LMs from Human Preferences: Fine-Tuning Language Models from Human Preferences [[Paper]](https://arxiv.org/abs/1909.08593) ![](https://img.shields.io/badge/year-2019-red)
* Deep RL from Human Preferences: Deep Reinforcement Learning from Human Preferences [[Paper]](https://proceedings.neurips.cc/paper/2017/hash/d5e2c0adad503c91f91df240d0cd4e49-Abstract.html) ![](https://img.shields.io/badge/year-2017-red)

<a id="section-vii-ai-feedback"></a>

## Section VII: AI Feedback

* RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback [[Paper]](https://proceedings.mlr.press/v235/lee24t.html) ![](https://img.shields.io/badge/year-2024-red)
* Self-Rewarding LMs: Self-Rewarding Language Models [[Scholar]](https://scholar.google.com/scholar?q=Self-Rewarding+Language+Models) ![](https://img.shields.io/badge/year-2024-red)
* DLMA: Direct Large Language Model Alignment Through Self-Rewarding Contrastive Prompt Distillation [[Paper]](https://aclanthology.org/2024.acl-long.523/) ![](https://img.shields.io/badge/year-2024-red)
* UltraFeedback: Boosting Language Models with Scaled AI Feedback [[Paper]](https://proceedings.mlr.press/v235/cui24f.html) ![](https://img.shields.io/badge/year-2024-red)
* VLFeedback: A Large-Scale AI Feedback Dataset for Large Vision-Language Models Alignment [[Paper]](https://aclanthology.org/2024.emnlp-main.358/) ![](https://img.shields.io/badge/year-2024-red)
* Constitutional AI: Harmlessness from AI Feedback [[Paper]](https://arxiv.org/abs/2212.08073) ![](https://img.shields.io/badge/year-2022-red)

<a id="section-viii-programmatic-and-verifiable-feedback"></a>

## Section VIII: Programmatic & Verifiable Feedback

* DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning [[Paper]](https://arxiv.org/abs/2501.12948) ![](https://img.shields.io/badge/year-2025-red)
* Rule Based Rewards for Language Model Safety [[Scholar]](https://scholar.google.com/scholar?q=Rule+Based+Rewards+for+Language+Model+Safety) ![](https://img.shields.io/badge/year-2024-red)
* Math-Shepherd: Verify and Reinforce LLMs Step-by-Step without Human Annotations [[Paper]](https://aclanthology.org/2024.acl-long.510/) ![](https://img.shields.io/badge/year-2024-red)
* DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models [[Paper]](https://arxiv.org/abs/2402.03300) ![](https://img.shields.io/badge/year-2024-red)

<a id="section-ix-demonstrations-and-preferences"></a>

## Section IX: Demonstrations & Preferences

* UltraFeedback: Boosting Language Models with Scaled AI Feedback [[Paper]](https://proceedings.mlr.press/v235/cui24f.html) ![](https://img.shields.io/badge/year-2024-red)
* HelpSteer 2: Open-source Dataset for Training Top-Performing Reward Models [[Paper]](https://doi.org/10.52202/079017-0047) ![](https://img.shields.io/badge/year-2024-red)
* RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-Grained Correctional Human Feedback [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/html/Yu_RLHF-V_Towards_Trustworthy_MLLMs_via_Behavior_Alignment_from_Fine-grained_Correctional_CVPR_2024_paper.html) ![](https://img.shields.io/badge/year-2024-red)
* Self-Instruct: Aligning Language Models with Self-Generated Instructions [[Paper]](https://aclanthology.org/2023.acl-long.754/) ![](https://img.shields.io/badge/year-2023-red)
* LIMA: Less Is More for Alignment [[Paper]](https://papers.neurips.cc/paper_files/paper/2023/hash/ac662d74829e4407ce1d126477f4a03a-Abstract-Conference.html) ![](https://img.shields.io/badge/year-2023-red)
* Flan Collection: The Flan Collection: Designing Data and Methods for Effective Instruction Tuning [[Paper]](https://proceedings.mlr.press/v202/longpre23a.html) ![](https://img.shields.io/badge/year-2023-red)
* Fine-Grained Human Feedback Gives Better Rewards for Language Model Training [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b8c90b65739ae8417e61eadb521f63d5-Abstract.html) ![](https://img.shields.io/badge/year-2023-red)
* FLAN: Finetuned Language Models Are Zero-Shot Learners [[Paper]](https://research.google/pubs/finetuned-language-models-are-zero-shot-learners/) ![](https://img.shields.io/badge/year-2022-red)
* Natural Instructions: Cross-Task Generalization via Natural Language Crowdsourcing Instructions [[Paper]](https://aclanthology.org/2022.acl-long.244/) ![](https://img.shields.io/badge/year-2022-red)
* Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks [[Paper]](https://aclanthology.org/2022.emnlp-main.340/) ![](https://img.shields.io/badge/year-2022-red)
* InstructGPT: Training Language Models to Follow Instructions with Human Feedback [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract.html) ![](https://img.shields.io/badge/year-2022-red)
* Learning to Summarize with Human Feedback [[Paper]](https://proceedings.neurips.cc/paper/2020/hash/1f89885d556929e98d3ef9b86448f951-Abstract.html) ![](https://img.shields.io/badge/year-2020-red)
* Deep RL from Human Preferences: Deep Reinforcement Learning from Human Preferences [[Paper]](https://proceedings.neurips.cc/paper/2017/hash/d5e2c0adad503c91f91df240d0cd4e49-Abstract.html) ![](https://img.shields.io/badge/year-2017-red)

<a id="section-x-critique-and-process-feedback"></a>

## Section X: Critique & Process Feedback

* CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing [[Scholar]](https://scholar.google.com/scholar?q=CRITIC%3A+Large+Language+Models+Can+Self-Correct+with+Tool-Interactive+Critiquing) ![](https://img.shields.io/badge/year-2024-red)
* UltraFeedback: Boosting Language Models with Scaled AI Feedback [[Paper]](https://proceedings.mlr.press/v235/cui24f.html) ![](https://img.shields.io/badge/year-2024-red)
* Let's Verify Step by Step [[Paper]](https://proceedings.iclr.cc/paper_files/paper/2024/hash/aca97732e30bcf1303bc22ac3924fd16-Abstract-Conference.html) ![](https://img.shields.io/badge/year-2024-red)
* RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-Grained Correctional Human Feedback [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/html/Yu_RLHF-V_Towards_Trustworthy_MLLMs_via_Behavior_Alignment_from_Fine-grained_Correctional_CVPR_2024_paper.html) ![](https://img.shields.io/badge/year-2024-red)
* Self-Refine: Iterative Refinement with Self-Feedback [[Paper]](https://arxiv.org/abs/2303.17651) ![](https://img.shields.io/badge/year-2023-red)
* Reflexion: Language Agents with Verbal Reinforcement Learning [[Scholar]](https://scholar.google.com/scholar?q=Reflexion%3A+Language+Agents+with+Verbal+Reinforcement+Learning) ![](https://img.shields.io/badge/year-2023-red)
* Fine-Grained Human Feedback Gives Better Rewards for Language Model Training [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b8c90b65739ae8417e61eadb521f63d5-Abstract.html) ![](https://img.shields.io/badge/year-2023-red)
* Constitutional AI: Harmlessness from AI Feedback [[Paper]](https://arxiv.org/abs/2212.08073) ![](https://img.shields.io/badge/year-2022-red)
* Process- and Outcome-Based Feedback: Solving Math Word Problems with Process- and Outcome-Based Feedback [[Paper]](https://arxiv.org/abs/2211.14275) ![](https://img.shields.io/badge/year-2022-red)

<a id="section-xi-reliable-and-scalable-oversight"></a>

## Section XI: Reliable & Scalable Oversight

* RewardBench: Evaluating Reward Models for Language Modeling [[Paper]](https://aclanthology.org/2025.findings-naacl.96/) ![](https://img.shields.io/badge/year-2025-red)
* Barriers and Pathways to Human-AI Alignment: A Game-Theoretic Approach [[Paper]](https://arxiv.org/abs/2502.05934) ![](https://img.shields.io/badge/year-2025-red)
* Active Preference Learning for Large Language Models [[Scholar]](https://scholar.google.com/scholar?q=Active+Preference+Learning+for+Large+Language+Models) ![](https://img.shields.io/badge/year-2024-red)
* RIME: Robust Preference-based Reinforcement Learning with Noisy Preferences [[Scholar]](https://scholar.google.com/scholar?q=RIME%3A+Robust+Preference-based+Reinforcement+Learning+with+Noisy+Preferences) ![](https://img.shields.io/badge/year-2024-red)
* Data-Quality-Aware Reward Modeling: Reward Modeling Requires Automatic Adjustment Based on Data Quality [[Paper]](https://doi.org/10.18653/v1/2024.findings-emnlp.234) ![](https://img.shields.io/badge/year-2024-red)
* Weak-to-Strong Generalization: Eliciting Strong Capabilities with Weak Supervision [[Paper]](https://proceedings.mlr.press/v235/burns24b.html) ![](https://img.shields.io/badge/year-2024-red)
* LLM Debate: Debating with More Persuasive LLMs Leads to More Truthful Answers [[Paper]](https://proceedings.mlr.press/v235/khan24a.html) ![](https://img.shields.io/badge/year-2024-red)
* Prover-Verifier Games Improve Legibility of LLM Outputs [[Paper]](https://arxiv.org/abs/2407.13692) ![](https://img.shields.io/badge/year-2024-red)
* AI Control: Improving Safety Despite Intentional Subversion [[Paper]](https://proceedings.mlr.press/v235/greenblatt24a.html) ![](https://img.shields.io/badge/year-2024-red)
* Self-Exploring Language Models: Active Preference Elicitation for Online Alignment [[Paper]](https://arxiv.org/abs/2405.19332) ![](https://img.shields.io/badge/year-2024-red)
* Reward Model Overoptimization: Scaling Laws for Reward Model Overoptimization [[Paper]](https://proceedings.mlr.press/v202/gao23h.html) ![](https://img.shields.io/badge/year-2023-red)
* Fine-Grained Human Feedback Gives Better Rewards for Language Model Training [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b8c90b65739ae8417e61eadb521f63d5-Abstract.html) ![](https://img.shields.io/badge/year-2023-red)
* AI Safety via Debate [[Paper]](https://arxiv.org/abs/1805.00899) ![](https://img.shields.io/badge/year-2018-red)
* Iterated Amplification: Supervising Strong Learners by Amplifying Weak Experts [[Paper]](https://arxiv.org/abs/1810.08575) ![](https://img.shields.io/badge/year-2018-red)

## Alignment Mechanisms

How are alignment signals translated into model behavior during training or inference?

| Group | Branch | Section |
|---|---|---|
| Training-Time Alignment | Reward & Verifier Modeling | [Section XII](#section-xii-reward-and-verifier-modeling) |
| Training-Time Alignment | Supervised Alignment | [Section XIII](#section-xiii-supervised-alignment) |
| Training-Time Alignment | Preference Optimization | [Section XIV](#section-xiv-preference-optimization) |
| Training-Time Alignment | Reinforcement Learning | [Section XV](#section-xv-reinforcement-learning) |
| Training-Time Alignment | Alignment Distillation | [Section XVI](#section-xvi-alignment-distillation) |
| Inference-Time Alignment | Steering, Search & Refinement | [Section XVII](#section-xvii-steering-search-and-refinement) |
| Inference-Time Alignment | Human Control | [Section XVIII](#section-xviii-human-control) |

<a id="section-xii-reward-and-verifier-modeling"></a>

## Section XII: Reward & Verifier Modeling

* BiPRM: The Bidirectional Process Reward Model [[Paper]](https://aclanthology.org/2026.acl-long.572/) ![](https://img.shields.io/badge/year-2026-red)
* LLaVA-Critic: Learning to Evaluate Multimodal Models [[Paper]](https://doi.org/10.1109/CVPR52734.2025.01271) ![](https://img.shields.io/badge/year-2025-red)
* Safe RLHF: Safe Reinforcement Learning from Human Feedback [[Paper]](https://proceedings.iclr.cc/paper_files/paper/2024/hash/dd1577afd396928ed64216f3f1fd5556-Abstract-Conference.html) ![](https://img.shields.io/badge/year-2024-red)
* HelpSteer 2: Open-source Dataset for Training Top-Performing Reward Models [[Paper]](https://doi.org/10.52202/079017-0047) ![](https://img.shields.io/badge/year-2024-red)
* Let's Verify Step by Step [[Paper]](https://proceedings.iclr.cc/paper_files/paper/2024/hash/aca97732e30bcf1303bc22ac3924fd16-Abstract-Conference.html) ![](https://img.shields.io/badge/year-2024-red)
* Math-Shepherd: Verify and Reinforce LLMs Step-by-Step without Human Annotations [[Paper]](https://aclanthology.org/2024.acl-long.510/) ![](https://img.shields.io/badge/year-2024-red)
* Prover-Verifier Games Improve Legibility of LLM Outputs [[Paper]](https://arxiv.org/abs/2407.13692) ![](https://img.shields.io/badge/year-2024-red)
* Fine-Grained Human Feedback Gives Better Rewards for Language Model Training [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b8c90b65739ae8417e61eadb521f63d5-Abstract.html) ![](https://img.shields.io/badge/year-2023-red)
* InstructGPT: Training Language Models to Follow Instructions with Human Feedback [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract.html) ![](https://img.shields.io/badge/year-2022-red)
* Helpful and Harmless Assistant: Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback [[Paper]](https://arxiv.org/abs/2204.05862) ![](https://img.shields.io/badge/year-2022-red)
* Sparrow: Improving Alignment of Dialogue Agents via Targeted Human Judgements [[Paper]](https://arxiv.org/abs/2209.14375) ![](https://img.shields.io/badge/year-2022-red)
* Process- and Outcome-Based Feedback: Solving Math Word Problems with Process- and Outcome-Based Feedback [[Paper]](https://arxiv.org/abs/2211.14275) ![](https://img.shields.io/badge/year-2022-red)
* Learning to Summarize with Human Feedback [[Paper]](https://proceedings.neurips.cc/paper/2020/hash/1f89885d556929e98d3ef9b86448f951-Abstract.html) ![](https://img.shields.io/badge/year-2020-red)
* Fine-Tuning LMs from Human Preferences: Fine-Tuning Language Models from Human Preferences [[Paper]](https://arxiv.org/abs/1909.08593) ![](https://img.shields.io/badge/year-2019-red)
* Deep RL from Human Preferences: Deep Reinforcement Learning from Human Preferences [[Paper]](https://proceedings.neurips.cc/paper/2017/hash/d5e2c0adad503c91f91df240d0cd4e49-Abstract.html) ![](https://img.shields.io/badge/year-2017-red)

<a id="section-xiii-supervised-alignment"></a>

## Section XIII: Supervised Alignment

* Scaling Instruction-Finetuned Language Models [[Paper]](https://www.jmlr.org/papers/v25/23-0870.html) ![](https://img.shields.io/badge/year-2024-red)
* MetaAligner: Towards Generalizable Multi-Objective Alignment of Language Models [[Paper]](https://doi.org/10.52202/079017-1086) ![](https://img.shields.io/badge/year-2024-red)
* Self-Instruct: Aligning Language Models with Self-Generated Instructions [[Paper]](https://aclanthology.org/2023.acl-long.754/) ![](https://img.shields.io/badge/year-2023-red)
* LIMA: Less Is More for Alignment [[Paper]](https://papers.neurips.cc/paper_files/paper/2023/hash/ac662d74829e4407ce1d126477f4a03a-Abstract-Conference.html) ![](https://img.shields.io/badge/year-2023-red)
* Flan Collection: The Flan Collection: Designing Data and Methods for Effective Instruction Tuning [[Paper]](https://proceedings.mlr.press/v202/longpre23a.html) ![](https://img.shields.io/badge/year-2023-red)
* SteerLM: Attribute Conditioned SFT as an (User-Steerable) Alternative to RLHF [[Paper]](https://aclanthology.org/2023.findings-emnlp.754/) ![](https://img.shields.io/badge/year-2023-red)
* FLAN: Finetuned Language Models Are Zero-Shot Learners [[Paper]](https://research.google/pubs/finetuned-language-models-are-zero-shot-learners/) ![](https://img.shields.io/badge/year-2022-red)
* T0: Multitask Prompted Training Enables Zero-Shot Task Generalization [[Paper]](https://arxiv.org/abs/2110.08207) ![](https://img.shields.io/badge/year-2022-red)
* Natural Instructions: Cross-Task Generalization via Natural Language Crowdsourcing Instructions [[Paper]](https://aclanthology.org/2022.acl-long.244/) ![](https://img.shields.io/badge/year-2022-red)
* Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks [[Paper]](https://aclanthology.org/2022.emnlp-main.340/) ![](https://img.shields.io/badge/year-2022-red)
* InstructGPT: Training Language Models to Follow Instructions with Human Feedback [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract.html) ![](https://img.shields.io/badge/year-2022-red)

<a id="section-xiv-preference-optimization"></a>

## Section XIV: Preference Optimization

* SPPO: Self-Play Preference Optimization for Language Model Alignment [[Paper]](https://arxiv.org/abs/2405.00675) ![](https://img.shields.io/badge/year-2025-red)
* SDPO: Segment-Level Direct Preference Optimization for Social Agents [[Paper]](https://aclanthology.org/2025.acl-long.607/) ![](https://img.shields.io/badge/year-2025-red)
* MMedPO: Aligning Medical Vision-Language Models with Clinical-Aware Multimodal Preference Optimization [[Paper]](https://mlanthology.org/icml/2025/zhu2025icml-mmedpo/) ![](https://img.shields.io/badge/year-2025-red)
* KTO: Model Alignment as Prospect Theoretic Optimization [[Paper]](https://arxiv.org/abs/2402.01306) ![](https://img.shields.io/badge/year-2024-red)
* ORPO: Monolithic Preference Optimization without Reference Model [[Paper]](https://aclanthology.org/2024.emnlp-main.626/) ![](https://img.shields.io/badge/year-2024-red)
* SimPO: Simple Preference Optimization with a Reference-Free Reward [[Paper]](https://papers.neurips.cc/paper_files/paper/2024/hash/e099c1c9699814af0be873a175361713-Abstract-Conference.html) ![](https://img.shields.io/badge/year-2024-red)
* TDPO: Token-Level Direct Preference Optimization [[Paper]](https://www.microsoft.com/en-us/research/publication/token-level-direct-preference-optimization/) ![](https://img.shields.io/badge/year-2024-red)
* Direct Nash Optimization: Teaching Language Models to Self-Improve with General Preferences [[Paper]](https://arxiv.org/abs/2404.03715) ![](https://img.shields.io/badge/year-2024-red)
* Self-Exploring Language Models: Active Preference Elicitation for Online Alignment [[Paper]](https://arxiv.org/abs/2405.19332) ![](https://img.shields.io/badge/year-2024-red)
* RRHF: Rank Responses to Align Language Models with Human Feedback [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/23e6f78bdec844a9f7b6c957de2aae91-Abstract-Conference.html) ![](https://img.shields.io/badge/year-2023-red)
* SLiC-HF: Sequence Likelihood Calibration with Human Feedback [[Paper]](https://arxiv.org/abs/2305.10425) ![](https://img.shields.io/badge/year-2023-red)
* DPO: Direct Preference Optimization: Your Language Model Is Secretly a Reward Model [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html) ![](https://img.shields.io/badge/year-2023-red)
* IPO & ΨPO: A General Theoretical Paradigm to Understand Learning from Human Preferences [[Paper]](https://arxiv.org/abs/2310.12036) ![](https://img.shields.io/badge/year-2023-red)
* Beyond Hallucinations: Enhancing LVLMs through Hallucination-Aware Direct Preference Optimization [[Paper]](https://arxiv.org/abs/2311.16839) ![](https://img.shields.io/badge/year-2023-red)

<a id="section-xv-reinforcement-learning"></a>

## Section XV: Reinforcement Learning

* DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning [[Paper]](https://arxiv.org/abs/2501.12948) ![](https://img.shields.io/badge/year-2025-red)
* MM-RLHF: The Next Step Forward in Multimodal LLM Alignment [[Paper]](https://icml.cc/virtual/2025/poster/45124) ![](https://img.shields.io/badge/year-2025-red)
* Safe RLHF: Safe Reinforcement Learning from Human Feedback [[Paper]](https://proceedings.iclr.cc/paper_files/paper/2024/hash/dd1577afd396928ed64216f3f1fd5556-Abstract-Conference.html) ![](https://img.shields.io/badge/year-2024-red)
* RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback [[Paper]](https://proceedings.mlr.press/v235/lee24t.html) ![](https://img.shields.io/badge/year-2024-red)
* DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models [[Paper]](https://arxiv.org/abs/2402.03300) ![](https://img.shields.io/badge/year-2024-red)
* Rule Based Rewards for Language Model Safety [[Scholar]](https://scholar.google.com/scholar?q=Rule+Based+Rewards+for+Language+Model+Safety) ![](https://img.shields.io/badge/year-2024-red)
* Aligning Large Multimodal Models with Factually Augmented RLHF [[Paper]](https://aclanthology.org/2024.findings-acl.775/) ![](https://img.shields.io/badge/year-2024-red)
* Multi-turn Reinforcement Learning from Preference Human Feedback [[Scholar]](https://scholar.google.com/scholar?q=Multi-turn+Reinforcement+Learning+from+Preference+Human+Feedback) ![](https://img.shields.io/badge/year-2024-red)
* SAIL: Self-Improving Efficient Online Alignment of Large Language Models [[Paper]](https://arxiv.org/abs/2406.15567) ![](https://img.shields.io/badge/year-2024-red)
* RLHF Survey: A Survey of Reinforcement Learning from Human Feedback [[Paper]](https://arxiv.org/abs/2312.14925) ![](https://img.shields.io/badge/year-2023-red)
* InstructGPT: Training Language Models to Follow Instructions with Human Feedback [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract.html) ![](https://img.shields.io/badge/year-2022-red)
* Process- and Outcome-Based Feedback: Solving Math Word Problems with Process- and Outcome-Based Feedback [[Paper]](https://arxiv.org/abs/2211.14275) ![](https://img.shields.io/badge/year-2022-red)
* Learning to Summarize with Human Feedback [[Paper]](https://proceedings.neurips.cc/paper/2020/hash/1f89885d556929e98d3ef9b86448f951-Abstract.html) ![](https://img.shields.io/badge/year-2020-red)
* Deep RL from Human Preferences: Deep Reinforcement Learning from Human Preferences [[Paper]](https://proceedings.neurips.cc/paper/2017/hash/d5e2c0adad503c91f91df240d0cd4e49-Abstract.html) ![](https://img.shields.io/badge/year-2017-red)

<a id="section-xvi-alignment-distillation"></a>

## Section XVI: Alignment Distillation

* ADPA: Advantage-Guided Distillation for Preference Alignment in Small Language Models [[Paper]](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2f891d026c7ba978168621842bc6fe73-Abstract-Conference.html) ![](https://img.shields.io/badge/year-2025-red)
* PAD: Capturing Nuanced Preferences: Preference-Aligned Distillation for Small Language Models [[Paper]](https://aclanthology.org/2025.findings-acl.822/) ![](https://img.shields.io/badge/year-2025-red)
* AlignDistil: Token-Level Language Model Alignment as Adaptive Policy Distillation [[Paper]](https://aclanthology.org/2025.acl-long.972/) ![](https://img.shields.io/badge/year-2025-red)
* CycleAlign: Iterative Distillation from Black-Box LLM to White-Box Models for Better Human Alignment [[Paper]](https://aclanthology.org/2024.findings-acl.869/) ![](https://img.shields.io/badge/year-2024-red)
* PLaD: Preference-Based Large Language Model Distillation with Pseudo-Preference Pairs [[Paper]](https://aclanthology.org/2024.findings-acl.923/) ![](https://img.shields.io/badge/year-2024-red)
* DPKD: Direct Preference Knowledge Distillation for Large Language Models [[Paper]](https://arxiv.org/abs/2406.19774) ![](https://img.shields.io/badge/year-2024-red)

<a id="section-xvii-steering-search-and-refinement"></a>

## Section XVII: Steering, Search & Refinement

* LLaVA-Critic: Learning to Evaluate Multimodal Models [[Paper]](https://doi.org/10.1109/CVPR52734.2025.01271) ![](https://img.shields.io/badge/year-2025-red)
* Contrastive Activation Addition: Steering Language Models with Activation Engineering [[Paper]](https://doi.org/10.18653/v1/2024.acl-long.828) ![](https://img.shields.io/badge/year-2024-red)
* CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing [[Scholar]](https://scholar.google.com/scholar?q=CRITIC%3A+Large+Language+Models+Can+Self-Correct+with+Tool-Interactive+Critiquing) ![](https://img.shields.io/badge/year-2024-red)
* Math-Shepherd: Verify and Reinforce LLMs Step-by-Step without Human Annotations [[Paper]](https://aclanthology.org/2024.acl-long.510/) ![](https://img.shields.io/badge/year-2024-red)
* Inference-Time Intervention: Eliciting Truthful Answers from a Language Model [[Paper]](https://doi.org/10.52202/075280-1797) ![](https://img.shields.io/badge/year-2023-red)
* Representation Engineering: A Top-Down Approach to AI Transparency [[Paper]](https://arxiv.org/abs/2310.01405) ![](https://img.shields.io/badge/year-2023-red)
* Activation Addition: Steering Language Models Without Optimization [[Paper]](https://arxiv.org/abs/2308.10248) ![](https://img.shields.io/badge/year-2023-red)
* Self-Refine: Iterative Refinement with Self-Feedback [[Paper]](https://arxiv.org/abs/2303.17651) ![](https://img.shields.io/badge/year-2023-red)
* Reflexion: Language Agents with Verbal Reinforcement Learning [[Scholar]](https://scholar.google.com/scholar?q=Reflexion%3A+Language+Agents+with+Verbal+Reinforcement+Learning) ![](https://img.shields.io/badge/year-2023-red)
* WebGPT: Browser-Assisted Question-Answering with Human Feedback [[Paper]](https://arxiv.org/abs/2112.09332) ![](https://img.shields.io/badge/year-2021-red)

<a id="section-xviii-human-control"></a>

## Section XVIII: Human Control

* Corrigibility Transformation: Constructing Goals That Accept Updates [[Paper]](https://arxiv.org/abs/2510.15395) ![](https://img.shields.io/badge/year-2025-red)
* AI Control: Improving Safety Despite Intentional Subversion [[Paper]](https://proceedings.mlr.press/v235/greenblatt24a.html) ![](https://img.shields.io/badge/year-2024-red)
* STaR-GATE: Teaching Language Models to Ask Clarifying Questions [[Scholar]](https://scholar.google.com/scholar?q=STaR-GATE%3A+Teaching+Language+Models+to+Ask+Clarifying+Questions) ![](https://img.shields.io/badge/year-2024-red)
* Learning to Defer: Consistent Estimators for Learning to Defer to an Expert [[Scholar]](https://scholar.google.com/scholar?q=Consistent+Estimators+for+Learning+to+Defer+to+an+Expert) ![](https://img.shields.io/badge/year-2020-red)
* Off-Switch Game: The Off-Switch Game [[Paper]](https://doi.org/10.24963/ijcai.2017/32) ![](https://img.shields.io/badge/year-2017-red)
* Cooperative IRL: Cooperative Inverse Reinforcement Learning [[Scholar]](https://scholar.google.com/scholar?q=Cooperative+Inverse+Reinforcement+Learning) ![](https://img.shields.io/badge/year-2016-red)
* Safely Interruptible Agents [[Scholar]](https://scholar.google.com/scholar?q=Safely+Interruptible+Agents) ![](https://img.shields.io/badge/year-2016-red)
* Corrigibility [[Paper]](https://cdn.aaai.org/ocs/ws/ws0067/10124-45900-1-PB.pdf) ![](https://img.shields.io/badge/year-2015-red)

## Alignment Assurance

How do we evaluate, stress-test, preserve, interpret, and monitor alignment?

| Group | Branch | Section |
|---|---|---|
| Evaluation & Robustness | Behavioral & Evaluator Evaluation | [Section XIX](#section-xix-behavioral-and-evaluator-evaluation) |
| Evaluation & Robustness | Adversarial & Distribution Robustness | [Section XX](#section-xx-adversarial-and-distribution-robustness) |
| Evaluation & Robustness | Alignment Preservation | [Section XXI](#section-xxi-alignment-preservation) |
| System Assurance | Mechanistic & Theoretical Evidence | [Section XXII](#section-xxii-mechanistic-and-theoretical-evidence) |
| System Assurance | Monitoring & Auditing | [Section XXIII](#section-xxiii-monitoring-and-auditing) |

<a id="section-xix-behavioral-and-evaluator-evaluation"></a>

## Section XIX: Behavioral & Evaluator Evaluation

* RewardBench: Evaluating Reward Models for Language Modeling [[Paper]](https://aclanthology.org/2025.findings-naacl.96/) ![](https://img.shields.io/badge/year-2025-red)
* PRMBench: A Fine-Grained and Challenging Benchmark for Process-Level Reward Models [[Paper]](https://aclanthology.org/2025.acl-long.1230/) ![](https://img.shields.io/badge/year-2025-red)
* RAG-RewardBench: Benchmarking Reward Models in Retrieval-Augmented Generation [[Paper]](https://aclanthology.org/2025.findings-acl.877/) ![](https://img.shields.io/badge/year-2025-red)
* XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models [[Paper]](https://arxiv.org/abs/2308.01263) ![](https://img.shields.io/badge/year-2024-red)
* Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference [[Scholar]](https://scholar.google.com/scholar?q=Chatbot+Arena%3A+An+Open+Platform+for+Evaluating+LLMs+by+Human+Preference) ![](https://img.shields.io/badge/year-2024-red)
* HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal [[Paper]](https://www.microsoft.com/en-us/research/publication/harmbench-a-standardized-evaluation-framework-for-automated-red-teaming-and-robust-refusal/) ![](https://img.shields.io/badge/year-2024-red)
* HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models [[Scholar]](https://scholar.google.com/scholar?q=HallusionBench%3A+An+Advanced+Diagnostic+Suite+for+Entangled+Language+Hallucination+and+Visual+Illusion+in+Large+Vision-Language+Models) ![](https://img.shields.io/badge/year-2024-red)
* SafetyBench: Evaluating the Safety of Large Language Models [[Paper]](https://aclanthology.org/2024.acl-long.830/) ![](https://img.shields.io/badge/year-2024-red)
* RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models [[Paper]](https://aclanthology.org/2024.acl-long.585/) ![](https://img.shields.io/badge/year-2024-red)
* MT-Bench & Chatbot Arena: Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html) ![](https://img.shields.io/badge/year-2023-red)
* IFEval: Instruction-Following Evaluation for Large Language Models [[Paper]](https://arxiv.org/abs/2311.07911) ![](https://img.shields.io/badge/year-2023-red)
* FActScore: Fine-Grained Atomic Evaluation of Factual Precision in Long Form Text Generation [[Paper]](https://aclanthology.org/2023.emnlp-main.741/) ![](https://img.shields.io/badge/year-2023-red)
* AlpacaEval: An Automatic Evaluator of Instruction-Following Models [[Paper]](https://github.com/tatsu-lab/alpaca_eval) ![](https://img.shields.io/badge/year-2023-red)
* Enabling Large Language Models to Generate Text with Citations [[Paper]](https://aclanthology.org/2023.emnlp-main.398/) ![](https://img.shields.io/badge/year-2023-red)
* TruthfulQA: Measuring How Models Mimic Human Falsehoods [[Paper]](https://aclanthology.org/2022.acl-long.229/) ![](https://img.shields.io/badge/year-2022-red)

<a id="section-xx-adversarial-and-distribution-robustness"></a>

## Section XX: Adversarial & Distribution Robustness

* When Context Flips, Safety Breaks: Diagnosing Brittle Safety in Aligned Language Models [[Paper]](https://arxiv.org/abs/2605.27851) ![](https://img.shields.io/badge/year-2026-red)
* HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal [[Paper]](https://www.microsoft.com/en-us/research/publication/harmbench-a-standardized-evaluation-framework-for-automated-red-teaming-and-robust-refusal/) ![](https://img.shields.io/badge/year-2024-red)
* StrongREJECT: A StrongREJECT for Empty Jailbreaks [[Paper]](https://arxiv.org/abs/2402.10260) ![](https://img.shields.io/badge/year-2024-red)
* Circuit Breakers: Improving Alignment and Robustness with Circuit Breakers [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2024/hash/97ca7168c2c333df5ea61ece3b3276e1-Abstract-Conference.html) ![](https://img.shields.io/badge/year-2024-red)
* XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models [[Paper]](https://arxiv.org/abs/2308.01263) ![](https://img.shields.io/badge/year-2024-red)
* Universal Adversarial Attacks: Universal and Transferable Adversarial Attacks on Aligned Language Models [[Paper]](https://arxiv.org/abs/2307.15043) ![](https://img.shields.io/badge/year-2023-red)
* HELM: Holistic Evaluation of Language Models [[Scholar]](https://scholar.google.com/scholar?q=Holistic+Evaluation+of+Language+Models) ![](https://img.shields.io/badge/year-2023-red)
* Extreme-Risk Evaluation: Model Evaluation for Extreme Risks [[Paper]](https://arxiv.org/abs/2305.15324) ![](https://img.shields.io/badge/year-2023-red)
* Red Teaming LMs with LMs: Red Teaming Language Models with Language Models [[Paper]](https://doi.org/10.18653/v1/2022.emnlp-main.225) ![](https://img.shields.io/badge/year-2022-red)
* Goal Misgeneralization in Deep RL: Goal Misgeneralization in Deep Reinforcement Learning [[Paper]](https://proceedings.mlr.press/v162/langosco22a.html) ![](https://img.shields.io/badge/year-2022-red)
* Goal Misgeneralization: Why Correct Specifications Aren't Enough for Correct Goals [[Paper]](https://arxiv.org/abs/2210.01790) ![](https://img.shields.io/badge/year-2022-red)
* Concrete Problems in AI Safety [[Paper]](https://arxiv.org/abs/1606.06565) ![](https://img.shields.io/badge/year-2016-red)

<a id="section-xxi-alignment-preservation"></a>

## Section XXI: Alignment Preservation

* Alignment Midtraining for Animals [[Paper]](https://arxiv.org/abs/2604.13076) ![](https://img.shields.io/badge/year-2026-red)
* COPR: Continual Human Preference Learning via Optimal Policy Regularization [[Paper]](https://aclanthology.org/2025.findings-acl.281/) ![](https://img.shields.io/badge/year-2025-red)
* Lifelong Safety Alignment for Language Models [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1ca2f4528792b31eab7a3e7f6a1c130b-Abstract-Conference.html) ![](https://img.shields.io/badge/year-2025-red)
* Layer-Aware Representation Filtering: Purifying Finetuning Data to Preserve LLM Safety Alignment [[Paper]](https://aclanthology.org/2025.emnlp-main.406/) ![](https://img.shields.io/badge/year-2025-red)
* Fine-Tuning Aligned LMs Compromises Safety: Fine-Tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To! [[Paper]](https://proceedings.iclr.cc/paper_files/paper/2024/hash/83b7da3ed13f06c13ce82235c8eedf35-Abstract-Conference.html) ![](https://img.shields.io/badge/year-2024-red)
* Circuit Breakers: Improving Alignment and Robustness with Circuit Breakers [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2024/hash/97ca7168c2c333df5ea61ece3b3276e1-Abstract-Conference.html) ![](https://img.shields.io/badge/year-2024-red)

<a id="section-xxii-mechanistic-and-theoretical-evidence"></a>

## Section XXII: Mechanistic & Theoretical Evidence

* Sustaining AI Safety: Control-Theoretic External Impossibility, Intrinsic Necessity, and Structural Requirements [[Paper]](https://arxiv.org/abs/2605.12963) ![](https://img.shields.io/badge/year-2026-red)
* Corrigibility Transformation: Constructing Goals That Accept Updates [[Paper]](https://arxiv.org/abs/2510.15395) ![](https://img.shields.io/badge/year-2025-red)
* Contrastive Activation Addition: Steering Language Models with Activation Engineering [[Paper]](https://doi.org/10.18653/v1/2024.acl-long.828) ![](https://img.shields.io/badge/year-2024-red)
* Discovering Latent Knowledge in Language Models Without Supervision [[Scholar]](https://scholar.google.com/scholar?q=Discovering+Latent+Knowledge+in+Language+Models+Without+Supervision) ![](https://img.shields.io/badge/year-2023-red)
* Inference-Time Intervention: Eliciting Truthful Answers from a Language Model [[Paper]](https://doi.org/10.52202/075280-1797) ![](https://img.shields.io/badge/year-2023-red)
* Representation Engineering: A Top-Down Approach to AI Transparency [[Paper]](https://arxiv.org/abs/2310.01405) ![](https://img.shields.io/badge/year-2023-red)
* Off-Switch Game: The Off-Switch Game [[Paper]](https://doi.org/10.24963/ijcai.2017/32) ![](https://img.shields.io/badge/year-2017-red)
* Corrigibility [[Paper]](https://cdn.aaai.org/ocs/ws/ws0067/10124-45900-1-PB.pdf) ![](https://img.shields.io/badge/year-2015-red)

<a id="section-xxiii-monitoring-and-auditing"></a>

## Section XXIII: Monitoring & Auditing

* EvalSafetyGap: A Hybrid Survey and Conceptual Framework for LLM Evaluation-Safety Failures [[Paper]](https://arxiv.org/abs/2606.30219) ![](https://img.shields.io/badge/year-2026-red)
* RewardBench: Evaluating Reward Models for Language Modeling [[Paper]](https://aclanthology.org/2025.findings-naacl.96/) ![](https://img.shields.io/badge/year-2025-red)
* AI Control: Improving Safety Despite Intentional Subversion [[Paper]](https://proceedings.mlr.press/v235/greenblatt24a.html) ![](https://img.shields.io/badge/year-2024-red)
* Test-Set Contamination: Proving Test Set Contamination in Black-Box Language Models [[Scholar]](https://scholar.google.com/scholar?q=Proving+Test+Set+Contamination+in+Black-Box+Language+Models) ![](https://img.shields.io/badge/year-2024-red)
* HELM: Holistic Evaluation of Language Models [[Scholar]](https://scholar.google.com/scholar?q=Holistic+Evaluation+of+Language+Models) ![](https://img.shields.io/badge/year-2023-red)
* Model-Written Evaluations: Discovering Language Model Behaviors with Model-Written Evaluations [[Paper]](https://aclanthology.org/2023.findings-acl.847/) ![](https://img.shields.io/badge/year-2023-red)
* Extreme-Risk Evaluation: Model Evaluation for Extreme Risks [[Paper]](https://arxiv.org/abs/2305.15324) ![](https://img.shields.io/badge/year-2023-red)

---

## General Survey Context

Broad surveys that span multiple branches are listed here rather than being forced into a single technical category.

* AI Alignment: A Contemporary Survey [[Paper]](https://doi.org/10.1145/3770749) ![](https://img.shields.io/badge/year-2025-red)
* Aligning Multimodal LLM with Human Preference: A Survey [[Paper]](https://arxiv.org/abs/2503.14504) ![](https://img.shields.io/badge/year-2025-red)
* A Survey on Human Preference Learning for Large Language Models [[Paper]](https://arxiv.org/abs/2406.11191) ![](https://img.shields.io/badge/year-2024-red)
* Towards a Unified View of Preference Learning for Large Language Models: A Survey [[Paper]](https://arxiv.org/abs/2409.02795) ![](https://img.shields.io/badge/year-2024-red)
* Bridging the Gap: A Survey on Integrating (Human) Feedback for Natural Language Generation [[Paper]](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00626/118795/) ![](https://img.shields.io/badge/year-2023-red)
* Large Language Model Alignment: A Survey [[Paper]](https://arxiv.org/abs/2309.15025) ![](https://img.shields.io/badge/year-2023-red)
