# Awesome Human–AI Alignment

A curated collection of research on specifying, supervising, learning and evaluating aligned AI systems.

**4 dimensions · 23 terminal categories · 145 bibliography entries**

## Overview

The collection follows the complete taxonomy in [`taxonomy.tex`](./taxonomy.tex). Categories use descriptive names shared with the figure and its navigation links. Papers can appear in several categories. The four taxonomy figures include every paper assigned to each category; the lists below account for every entry in [`human_ai_alignment_refs.bib`](./human_ai_alignment_refs.bib), including general survey context.

## Taxonomy and navigation

| Dimension | Category |
|---|---|
| Alignment Specification | [Task & Assistance Alignment](#task-assistance-alignment) |
| Alignment Specification | [Safety Alignment](#safety-alignment) |
| Alignment Specification | [Personalized Alignment](#personalized-alignment) |
| Alignment Specification | [Pluralistic & Societal Alignment](#pluralistic-societal-alignment) |
| Alignment Specification | [Uncertainty & Drift](#uncertainty-drift) |
| Alignment Supervision | [Human Feedback](#human-feedback) |
| Alignment Supervision | [AI Feedback](#ai-feedback) |
| Alignment Supervision | [Programmatic Feedback](#programmatic-feedback) |
| Alignment Supervision | [Demonstrations & Evaluations](#demonstrations-evaluations) |
| Alignment Supervision | [Critique & Process Feedback](#critique-process-feedback) |
| Alignment Supervision | [Reliable & Scalable Oversight](#reliable-scalable-oversight) |
| Alignment Mechanisms | [Reward & Verifier Modeling](#reward-verifier-modeling) |
| Alignment Mechanisms | [Supervised Alignment](#supervised-alignment) |
| Alignment Mechanisms | [Preference & Ranking Optimization](#preference-ranking-optimization) |
| Alignment Mechanisms | [Reinforcement Learning](#reinforcement-learning) |
| Alignment Mechanisms | [Alignment Distillation](#alignment-distillation) |
| Alignment Mechanisms | [Steering, Search & Refinement](#steering-search-refinement) |
| Alignment Mechanisms | [Human Control](#human-control) |
| Alignment Assurance | [Behavior & Evaluator Assessment](#behavior-evaluator-assessment) |
| Alignment Assurance | [Adversarial & Distribution Robustness](#adversarial-distribution-robustness) |
| Alignment Assurance | [Alignment Preservation](#alignment-preservation) |
| Alignment Assurance | [Mechanistic & Theoretical Evidence](#mechanistic-theoretical-evidence) |
| Alignment Assurance | [Monitoring & Auditing](#monitoring-auditing) |

**Programmatic Feedback** covers rule-based rewards, executable checks and verifiable outcomes. **Demonstrations & Evaluations** includes demonstrations, preference comparisons, binary labels, scalar ratings and multi-attribute judgments. **Behavior & Evaluator Assessment** covers both model behavior and the reliability of reward models, judges and process verifiers.

Cross-cutting descriptors: **system** (text, multimodal, embodied and tool-using agents); **feedback granularity** (response, span, step and token); **optimization granularity** (sequence, segment and token); **learning regime** (offline, online, static and continual). Feedback granularity and optimization granularity are distinct.

Paper titles, author names, years and links follow the supplied BibTeX records. *Scholar search* marks a record without a direct URL or DOI. The category mapping is an editorial synthesis, not a claim that every paper solves alignment or provides a deployment guarantee.

## Alignment Specification

### Alignment Objectives

<a id="task-assistance-alignment"></a>

#### Task & Assistance Alignment

- **FLAN** — [Finetuned Language Models Are Zero-Shot Learners](https://research.google/pubs/finetuned-language-models-are-zero-shot-learners/) — Wei et al. (2022) · [`wei2022flan`](./human_ai_alignment_refs.bib)
- **T0** — [Multitask Prompted Training Enables Zero-Shot Task Generalization](https://arxiv.org/abs/2110.08207) — Sanh et al. (2022) · [`sanh2022t0`](./human_ai_alignment_refs.bib)
- **Natural Instructions** — [Cross-Task Generalization via Natural Language Crowdsourcing Instructions](https://aclanthology.org/2022.acl-long.244/) — Mishra et al. (2022) · [`mishra2022natural`](./human_ai_alignment_refs.bib)
- **Super-NaturalInstructions** — [Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks](https://aclanthology.org/2022.emnlp-main.340/) — Wang et al. (2022) · [`wang2022superni`](./human_ai_alignment_refs.bib)
- **InstructGPT** — [Training Language Models to Follow Instructions with Human Feedback](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract.html) — Ouyang et al. (2022) · [`ouyang2022instructgpt`](./human_ai_alignment_refs.bib)
- **Self-Instruct** — [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://aclanthology.org/2023.acl-long.754/) — Wang et al. (2023) · [`wang2023selfinstruct`](./human_ai_alignment_refs.bib)
- **LIMA** — [LIMA: Less Is More for Alignment](https://papers.neurips.cc/paper_files/paper/2023/hash/ac662d74829e4407ce1d126477f4a03a-Abstract-Conference.html) — Zhou et al. (2023) · [`zhou2023lima`](./human_ai_alignment_refs.bib)
- **Flan Collection** — [The Flan Collection: Designing Data and Methods for Effective Instruction Tuning](https://proceedings.mlr.press/v202/longpre23a.html) — Longpre et al. (2023) · [`longpre2023flancollection`](./human_ai_alignment_refs.bib)
- **SteerLM** — [SteerLM: Attribute Conditioned SFT as an (User-Steerable) Alternative to RLHF](https://aclanthology.org/2023.findings-emnlp.754/) — Dong et al. (2023) · [`dong2023steerlm`](./human_ai_alignment_refs.bib)
- **General Language Assistant** — [A General Language Assistant as a Laboratory for Alignment](https://arxiv.org/abs/2112.00861) — Askell et al. (2021) · [`askell2021assistant`](./human_ai_alignment_refs.bib)
- **Helpful and Harmless Assistant** — [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862) — Bai et al. (2022) · [`bai2022hh`](./human_ai_alignment_refs.bib)
- **Sparrow** — [Improving Alignment of Dialogue Agents via Targeted Human Judgements](https://arxiv.org/abs/2209.14375) — Glaese et al. (2022) · [`glaese2022sparrow`](./human_ai_alignment_refs.bib)
- **WebGPT** — [WebGPT: Browser-Assisted Question-Answering with Human Feedback](https://arxiv.org/abs/2112.09332) — Nakano et al. (2021) · [`nakano2021webgpt`](./human_ai_alignment_refs.bib)
- **Fine-Grained Human Feedback** — [Fine-Grained Human Feedback Gives Better Rewards for Language Model Training](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b8c90b65739ae8417e61eadb521f63d5-Abstract.html) — Wu et al. (2023) · [`wu2023finegrained`](./human_ai_alignment_refs.bib)
- **UltraFeedback** — [UltraFeedback: Boosting Language Models with Scaled AI Feedback](https://proceedings.mlr.press/v235/cui24f.html) — Cui et al. (2024) · [`cui2024ultrafeedback`](./human_ai_alignment_refs.bib)
- **HelpSteer2** — [HelpSteer 2: Open-source Dataset for Training Top-Performing Reward Models](https://doi.org/10.52202/079017-0047) — Wang et al. (2024) · [`wang2024helpsteer2`](./human_ai_alignment_refs.bib)
- **DPO** — [Direct Preference Optimization: Your Language Model Is Secretly a Reward Model](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html) — Rafailov et al. (2023) · [`rafailov2023dpo`](./human_ai_alignment_refs.bib)
- **Scaling Instruction-Finetuned Language Models** — [Scaling Instruction-Finetuned Language Models](https://www.jmlr.org/papers/v25/23-0870.html) — Chung et al. (2024) · [`chung2024flan`](./human_ai_alignment_refs.bib)
- **Enabling Large Language Models to Generate Text with Citations** — [Enabling Large Language Models to Generate Text with Citations](https://aclanthology.org/2023.emnlp-main.398/) — Gao et al. (2023) · [`gao2023citations`](./human_ai_alignment_refs.bib)
- **RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models** — [RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models](https://aclanthology.org/2024.acl-long.585/) — Niu et al. (2024) · [`niu2024ragtruth`](./human_ai_alignment_refs.bib)
- **Aligning Large Multimodal Models with Factually Augmented RLHF** — [Aligning Large Multimodal Models with Factually Augmented RLHF](https://aclanthology.org/2024.findings-acl.775/) — Sun et al. (2024) · [`sun2024factrlhf`](./human_ai_alignment_refs.bib)

<a id="safety-alignment"></a>

#### Safety Alignment

- **General Language Assistant** — [A General Language Assistant as a Laboratory for Alignment](https://arxiv.org/abs/2112.00861) — Askell et al. (2021) · [`askell2021assistant`](./human_ai_alignment_refs.bib)
- **Helpful and Harmless Assistant** — [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862) — Bai et al. (2022) · [`bai2022hh`](./human_ai_alignment_refs.bib)
- **Constitutional AI** — [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Bai et al. (2022) · [`bai2022constitutional`](./human_ai_alignment_refs.bib)
- **BeaverTails** — [BeaverTails: Towards Improved Safety Alignment of LLM via a Human-Preference Dataset](https://proceedings.neurips.cc/paper_files/paper/2023/hash/4dbb61cb68671edc4ca3712d70083b9f-Abstract.html) — Ji et al. (2023) · [`ji2023beavertails`](./human_ai_alignment_refs.bib)
- **Safe RLHF** — [Safe RLHF: Safe Reinforcement Learning from Human Feedback](https://proceedings.iclr.cc/paper_files/paper/2024/hash/dd1577afd396928ed64216f3f1fd5556-Abstract-Conference.html) — Dai et al. (2024) · [`dai2024saferlhf`](./human_ai_alignment_refs.bib)
- **Rule-Based Rewards** — [Rule Based Rewards for Language Model Safety](https://scholar.google.com/scholar?q=Rule+Based+Rewards+for+Language+Model+Safety) — Mu et al. (2024) · *Scholar search* · [`mu2024rulebased`](./human_ai_alignment_refs.bib)
- **Circuit Breakers** — [Improving Alignment and Robustness with Circuit Breakers](https://proceedings.neurips.cc/paper_files/paper/2024/hash/97ca7168c2c333df5ea61ece3b3276e1-Abstract-Conference.html) — Zou et al. (2024) · [`zou2024circuitbreakers`](./human_ai_alignment_refs.bib)

### Values & Stakeholders

<a id="personalized-alignment"></a>

#### Personalized Alignment

- **Personalized Soups** — [Personalized Soups: Personalized Large Language Model Alignment via Post-hoc Parameter Merging](https://arxiv.org/abs/2310.11564) — Jang et al. (2023) · [`jang2023personalizedsoups`](./human_ai_alignment_refs.bib)
- **Personalized Language Modeling** — [Personalized Language Modeling from Personalized Human Feedback](https://arxiv.org/abs/2402.05133) — Li et al. (2024) · [`li2024personalized`](./human_ai_alignment_refs.bib)
- **Personalizing RLHF** — [Personalizing Reinforcement Learning from Human Feedback](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5e1c255653eb98cef13f45b2d337c882-Abstract-Conference.html) — Poddar et al. (2024) · [`poddar2024personalizing`](./human_ai_alignment_refs.bib)
- **Persona-Judge** — [Persona-judge: Personalized Alignment of Large Language Models via Token-level Self-judgment](https://aclanthology.org/2025.findings-acl.260/) — Zhang et al. (2025) · [`zhang2025personajudge`](./human_ai_alignment_refs.bib)
- **Personalized Alignment Survey** — [A Survey on Personalized Alignment—The Missing Piece for Large Language Models in Real-World Applications](https://aclanthology.org/2025.findings-acl.277/) — Guan et al. (2025) · [`guan2025personalizedsurvey`](./human_ai_alignment_refs.bib)

<a id="pluralistic-societal-alignment"></a>

#### Pluralistic & Societal Alignment

- **Finding Agreement** — [Fine-Tuning Language Models to Find Agreement among Humans with Diverse Preferences](https://proceedings.neurips.cc/paper/2022/hash/f978c8f3b5f399cae464e85f72e28503-Abstract-Conference.html) — Bakker et al. (2022) · [`bakker2022finetuning`](./human_ai_alignment_refs.bib)
- **Distributional Preference Learning** — [Distributional Preference Learning: Understanding and Accounting for Hidden Context in RLHF](https://scholar.google.com/scholar?q=Distributional+Preference+Learning%3A+Understanding+and+Accounting+for+Hidden+Context+in+RLHF) — Siththaranjan et al. (2024) · *Scholar search* · [`siththaranjan2024distributional`](./human_ai_alignment_refs.bib)
- **Collective Constitutional AI** — [Collective Constitutional AI: Aligning a Language Model with Public Input](https://arxiv.org/abs/2406.07814) — Huang et al. (2024) · [`huang2024collective`](./human_ai_alignment_refs.bib)
- **PRISM** — [The PRISM Alignment Dataset: What Participatory, Representative and Individualised Human Feedback Reveals About the Subjective and Multicultural Alignment of Large Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/be2e1b68b44f2419e19f6c35a1b8cf35-Abstract-Datasets_and_Benchmarks_Track.html) — Kirk et al. (2024) · [`kirk2024prism`](./human_ai_alignment_refs.bib)
- **CultureLLM** — [CultureLLM: Incorporating Cultural Differences into Large Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/9a16935bf54c4af233e25d998b7f4a2c-Abstract-Conference.html) — Li et al. (2024) · [`li2024culturellm`](./human_ai_alignment_refs.bib)
- **Modular Pluralism** — [Modular Pluralism: Pluralistic Alignment via Multi-LLM Collaboration](https://aclanthology.org/2024.emnlp-main.240/) — Feng et al. (2024) · [`feng2024modular`](./human_ai_alignment_refs.bib)
- **Pluralistic Alignment** — [Position: A Roadmap to Pluralistic Alignment](https://scholar.google.com/scholar?q=Position%3A+A+Roadmap+to+Pluralistic+Alignment) — Sorensen et al. (2024) · *Scholar search* · [`sorensen2024pluralistic`](./human_ai_alignment_refs.bib)
- **Steerable Pluralism** — [Steerable Pluralism: Pluralistic Alignment via Few-Shot Comparative Regression](https://doi.org/10.1609/aies.v8i1.36527) — Adams et al. (2025) · [`adams2025steerable`](./human_ai_alignment_refs.bib)
- **Constitutional AI** — [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Bai et al. (2022) · [`bai2022constitutional`](./human_ai_alignment_refs.bib)
- **Rule-Based Rewards** — [Rule Based Rewards for Language Model Safety](https://scholar.google.com/scholar?q=Rule+Based+Rewards+for+Language+Model+Safety) — Mu et al. (2024) · *Scholar search* · [`mu2024rulebased`](./human_ai_alignment_refs.bib)
- **AI Control** — [AI Control: Improving Safety Despite Intentional Subversion](https://proceedings.mlr.press/v235/greenblatt24a.html) — Greenblatt et al. (2024) · [`greenblatt2024control`](./human_ai_alignment_refs.bib)
- **The Moral Machine Experiment** — [The Moral Machine Experiment](https://doi.org/10.1038/s41586-018-0637-6) — Awad et al. (2018) · [`awad2018moral`](./human_ai_alignment_refs.bib)

<a id="uncertainty-drift"></a>

#### Uncertainty & Drift

- **Inverse RL** — [Algorithms for Inverse Reinforcement Learning](https://scholar.google.com/scholar?q=Algorithms+for+Inverse+Reinforcement+Learning) — Ng et al. (2000) · *Scholar search* · [`ng2000irl`](./human_ai_alignment_refs.bib)
- **Apprenticeship Learning via IRL** — [Apprenticeship Learning via Inverse Reinforcement Learning](https://scholar.google.com/scholar?q=Apprenticeship+Learning+via+Inverse+Reinforcement+Learning) — Abbeel et al. (2004) · *Scholar search* · [`abbeel2004apprenticeship`](./human_ai_alignment_refs.bib)
- **Cooperative IRL** — [Cooperative Inverse Reinforcement Learning](https://scholar.google.com/scholar?q=Cooperative+Inverse+Reinforcement+Learning) — Hadfield-Menell et al. (2016) · *Scholar search* · [`hadfieldmenell2016cirl`](./human_ai_alignment_refs.bib)
- **Inverse Reward Design** — [Inverse Reward Design](https://scholar.google.com/scholar?q=Inverse+Reward+Design) — Hadfield-Menell et al. (2017) · *Scholar search* · [`hadfieldmenell2017ird`](./human_ai_alignment_refs.bib)
- **Off-Switch Game** — [The Off-Switch Game](https://doi.org/10.24963/ijcai.2017/32) — Hadfield-Menell et al. (2017) · [`hadfieldmenell2017offswitch`](./human_ai_alignment_refs.bib)
- **COPR** — [COPR: Continual Human Preference Learning via Optimal Policy Regularization](https://aclanthology.org/2025.findings-acl.281/) — Zhang et al. (2025) · [`zhang2025copr`](./human_ai_alignment_refs.bib)
- **Lifelong Safety Alignment** — [Lifelong Safety Alignment for Language Models](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1ca2f4528792b31eab7a3e7f6a1c130b-Abstract-Conference.html) — Wang et al. (2025) · [`wang2025lifelong`](./human_ai_alignment_refs.bib)

## Alignment Supervision

### Feedback Source

<a id="human-feedback"></a>

#### Human Feedback

- **Deep RL from Human Preferences** — [Deep Reinforcement Learning from Human Preferences](https://proceedings.neurips.cc/paper/2017/hash/d5e2c0adad503c91f91df240d0cd4e49-Abstract.html) — Christiano et al. (2017) · [`christiano2017preferences`](./human_ai_alignment_refs.bib)
- **Fine-Tuning LMs from Human Preferences** — [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593) — Ziegler et al. (2019) · [`ziegler2019finetuning`](./human_ai_alignment_refs.bib)
- **Learning to Summarize** — [Learning to Summarize with Human Feedback](https://proceedings.neurips.cc/paper/2020/hash/1f89885d556929e98d3ef9b86448f951-Abstract.html) — Stiennon et al. (2020) · [`stiennon2020summarize`](./human_ai_alignment_refs.bib)
- **InstructGPT** — [Training Language Models to Follow Instructions with Human Feedback](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract.html) — Ouyang et al. (2022) · [`ouyang2022instructgpt`](./human_ai_alignment_refs.bib)
- **Helpful and Harmless Assistant** — [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862) — Bai et al. (2022) · [`bai2022hh`](./human_ai_alignment_refs.bib)
- **Sparrow** — [Improving Alignment of Dialogue Agents via Targeted Human Judgements](https://arxiv.org/abs/2209.14375) — Glaese et al. (2022) · [`glaese2022sparrow`](./human_ai_alignment_refs.bib)
- **Fine-Grained Human Feedback** — [Fine-Grained Human Feedback Gives Better Rewards for Language Model Training](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b8c90b65739ae8417e61eadb521f63d5-Abstract.html) — Wu et al. (2023) · [`wu2023finegrained`](./human_ai_alignment_refs.bib)
- **RLHF-V** — [RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-Grained Correctional Human Feedback](https://openaccess.thecvf.com/content/CVPR2024/html/Yu_RLHF-V_Towards_Trustworthy_MLLMs_via_Behavior_Alignment_from_Fine-grained_Correctional_CVPR_2024_paper.html) — Yu et al. (2024) · [`yu2024rlhfv`](./human_ai_alignment_refs.bib)
- **MM-RLHF: The Next Step Forward in Multimodal LLM Alignment** — [MM-RLHF: The Next Step Forward in Multimodal LLM Alignment](https://icml.cc/virtual/2025/poster/45124) — Zhang et al. (2025) · [`zhang2025mmrlhf`](./human_ai_alignment_refs.bib)

<a id="ai-feedback"></a>

#### AI Feedback

- **Constitutional AI** — [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Bai et al. (2022) · [`bai2022constitutional`](./human_ai_alignment_refs.bib)
- **RLAIF** — [RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://proceedings.mlr.press/v235/lee24t.html) — Lee et al. (2024) · [`lee2024rlaif`](./human_ai_alignment_refs.bib)
- **Self-Rewarding LMs** — [Self-Rewarding Language Models](https://scholar.google.com/scholar?q=Self-Rewarding+Language+Models) — Yuan et al. (2024) · *Scholar search* · [`yuan2024selfrewarding`](./human_ai_alignment_refs.bib)
- **DLMA** — [Direct Large Language Model Alignment Through Self-Rewarding Contrastive Prompt Distillation](https://aclanthology.org/2024.acl-long.523/) — Liu et al. (2024) · [`liu2024dlma`](./human_ai_alignment_refs.bib)
- **UltraFeedback** — [UltraFeedback: Boosting Language Models with Scaled AI Feedback](https://proceedings.mlr.press/v235/cui24f.html) — Cui et al. (2024) · [`cui2024ultrafeedback`](./human_ai_alignment_refs.bib)
- **VLFeedback** — [VLFeedback: A Large-Scale AI Feedback Dataset for Large Vision-Language Models Alignment](https://aclanthology.org/2024.emnlp-main.358/) — Li et al. (2024) · [`li2024vlfeedback`](./human_ai_alignment_refs.bib)

<a id="programmatic-feedback"></a>

#### Programmatic Feedback

- **Rule-Based Rewards** — [Rule Based Rewards for Language Model Safety](https://scholar.google.com/scholar?q=Rule+Based+Rewards+for+Language+Model+Safety) — Mu et al. (2024) · *Scholar search* · [`mu2024rulebased`](./human_ai_alignment_refs.bib)
- **Math-Shepherd** — [Math-Shepherd: Verify and Reinforce LLMs Step-by-Step without Human Annotations](https://aclanthology.org/2024.acl-long.510/) — Wang et al. (2024) · [`wang2024mathshepherd`](./human_ai_alignment_refs.bib)
- **DeepSeekMath** — [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) — Shao et al. (2024) · [`shao2024deepseekmath`](./human_ai_alignment_refs.bib)
- **DeepSeek-R1** — [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — DeepSeek-AI (2025) · [`deepseek2025r1`](./human_ai_alignment_refs.bib)

### Feedback & Oversight

<a id="demonstrations-evaluations"></a>

#### Demonstrations & Evaluations

- **FLAN** — [Finetuned Language Models Are Zero-Shot Learners](https://research.google/pubs/finetuned-language-models-are-zero-shot-learners/) — Wei et al. (2022) · [`wei2022flan`](./human_ai_alignment_refs.bib)
- **Natural Instructions** — [Cross-Task Generalization via Natural Language Crowdsourcing Instructions](https://aclanthology.org/2022.acl-long.244/) — Mishra et al. (2022) · [`mishra2022natural`](./human_ai_alignment_refs.bib)
- **Super-NaturalInstructions** — [Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks](https://aclanthology.org/2022.emnlp-main.340/) — Wang et al. (2022) · [`wang2022superni`](./human_ai_alignment_refs.bib)
- **InstructGPT** — [Training Language Models to Follow Instructions with Human Feedback](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract.html) — Ouyang et al. (2022) · [`ouyang2022instructgpt`](./human_ai_alignment_refs.bib)
- **Self-Instruct** — [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://aclanthology.org/2023.acl-long.754/) — Wang et al. (2023) · [`wang2023selfinstruct`](./human_ai_alignment_refs.bib)
- **LIMA** — [LIMA: Less Is More for Alignment](https://papers.neurips.cc/paper_files/paper/2023/hash/ac662d74829e4407ce1d126477f4a03a-Abstract-Conference.html) — Zhou et al. (2023) · [`zhou2023lima`](./human_ai_alignment_refs.bib)
- **Flan Collection** — [The Flan Collection: Designing Data and Methods for Effective Instruction Tuning](https://proceedings.mlr.press/v202/longpre23a.html) — Longpre et al. (2023) · [`longpre2023flancollection`](./human_ai_alignment_refs.bib)
- **Deep RL from Human Preferences** — [Deep Reinforcement Learning from Human Preferences](https://proceedings.neurips.cc/paper/2017/hash/d5e2c0adad503c91f91df240d0cd4e49-Abstract.html) — Christiano et al. (2017) · [`christiano2017preferences`](./human_ai_alignment_refs.bib)
- **Learning to Summarize** — [Learning to Summarize with Human Feedback](https://proceedings.neurips.cc/paper/2020/hash/1f89885d556929e98d3ef9b86448f951-Abstract.html) — Stiennon et al. (2020) · [`stiennon2020summarize`](./human_ai_alignment_refs.bib)
- **Fine-Grained Human Feedback** — [Fine-Grained Human Feedback Gives Better Rewards for Language Model Training](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b8c90b65739ae8417e61eadb521f63d5-Abstract.html) — Wu et al. (2023) · [`wu2023finegrained`](./human_ai_alignment_refs.bib)
- **UltraFeedback** — [UltraFeedback: Boosting Language Models with Scaled AI Feedback](https://proceedings.mlr.press/v235/cui24f.html) — Cui et al. (2024) · [`cui2024ultrafeedback`](./human_ai_alignment_refs.bib)
- **HelpSteer2** — [HelpSteer 2: Open-source Dataset for Training Top-Performing Reward Models](https://doi.org/10.52202/079017-0047) — Wang et al. (2024) · [`wang2024helpsteer2`](./human_ai_alignment_refs.bib)
- **RLHF-V** — [RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-Grained Correctional Human Feedback](https://openaccess.thecvf.com/content/CVPR2024/html/Yu_RLHF-V_Towards_Trustworthy_MLLMs_via_Behavior_Alignment_from_Fine-grained_Correctional_CVPR_2024_paper.html) — Yu et al. (2024) · [`yu2024rlhfv`](./human_ai_alignment_refs.bib)

<a id="critique-process-feedback"></a>

#### Critique & Process Feedback

- **Constitutional AI** — [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Bai et al. (2022) · [`bai2022constitutional`](./human_ai_alignment_refs.bib)
- **Self-Refine** — [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651) — Madaan et al. (2023) · [`madaan2023selfrefine`](./human_ai_alignment_refs.bib)
- **Reflexion** — [Reflexion: Language Agents with Verbal Reinforcement Learning](https://scholar.google.com/scholar?q=Reflexion%3A+Language+Agents+with+Verbal+Reinforcement+Learning) — Shinn et al. (2023) · *Scholar search* · [`shinn2023reflexion`](./human_ai_alignment_refs.bib)
- **CRITIC** — [CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing](https://scholar.google.com/scholar?q=CRITIC%3A+Large+Language+Models+Can+Self-Correct+with+Tool-Interactive+Critiquing) — Gou et al. (2024) · *Scholar search* · [`gou2024critic`](./human_ai_alignment_refs.bib)
- **UltraFeedback** — [UltraFeedback: Boosting Language Models with Scaled AI Feedback](https://proceedings.mlr.press/v235/cui24f.html) — Cui et al. (2024) · [`cui2024ultrafeedback`](./human_ai_alignment_refs.bib)
- **Fine-Grained Human Feedback** — [Fine-Grained Human Feedback Gives Better Rewards for Language Model Training](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b8c90b65739ae8417e61eadb521f63d5-Abstract.html) — Wu et al. (2023) · [`wu2023finegrained`](./human_ai_alignment_refs.bib)
- **Let's Verify Step by Step** — [Let's Verify Step by Step](https://proceedings.iclr.cc/paper_files/paper/2024/hash/aca97732e30bcf1303bc22ac3924fd16-Abstract-Conference.html) — Lightman et al. (2024) · [`lightman2024verify`](./human_ai_alignment_refs.bib)
- **Process- and Outcome-Based Feedback** — [Solving Math Word Problems with Process- and Outcome-Based Feedback](https://arxiv.org/abs/2211.14275) — Uesato et al. (2022) · [`uesato2022process`](./human_ai_alignment_refs.bib)
- **RLHF-V** — [RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-Grained Correctional Human Feedback](https://openaccess.thecvf.com/content/CVPR2024/html/Yu_RLHF-V_Towards_Trustworthy_MLLMs_via_Behavior_Alignment_from_Fine-grained_Correctional_CVPR_2024_paper.html) — Yu et al. (2024) · [`yu2024rlhfv`](./human_ai_alignment_refs.bib)

<a id="reliable-scalable-oversight"></a>

#### Reliable & Scalable Oversight

- **Reward Model Overoptimization** — [Scaling Laws for Reward Model Overoptimization](https://proceedings.mlr.press/v202/gao23h.html) — Gao et al. (2023) · [`gao2023overoptimization`](./human_ai_alignment_refs.bib)
- **Active Preference Learning** — [Active Preference Learning for Large Language Models](https://scholar.google.com/scholar?q=Active+Preference+Learning+for+Large+Language+Models) — Muldrew et al. (2024) · *Scholar search* · [`muldrew2024active`](./human_ai_alignment_refs.bib)
- **RIME** — [RIME: Robust Preference-based Reinforcement Learning with Noisy Preferences](https://scholar.google.com/scholar?q=RIME%3A+Robust+Preference-based+Reinforcement+Learning+with+Noisy+Preferences) — Cheng et al. (2024) · *Scholar search* · [`cheng2024rime`](./human_ai_alignment_refs.bib)
- **Data-Quality-Aware Reward Modeling** — [Reward Modeling Requires Automatic Adjustment Based on Data Quality](https://doi.org/10.18653/v1/2024.findings-emnlp.234) — Wang et al. (2024) · [`wang2024rmquality`](./human_ai_alignment_refs.bib)
- **Fine-Grained Human Feedback** — [Fine-Grained Human Feedback Gives Better Rewards for Language Model Training](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b8c90b65739ae8417e61eadb521f63d5-Abstract.html) — Wu et al. (2023) · [`wu2023finegrained`](./human_ai_alignment_refs.bib)
- **RewardBench** — [RewardBench: Evaluating Reward Models for Language Modeling](https://aclanthology.org/2025.findings-naacl.96/) — Lambert et al. (2025) · [`lambert2025rewardbench`](./human_ai_alignment_refs.bib)
- **AI Safety via Debate** — [AI Safety via Debate](https://arxiv.org/abs/1805.00899) — Irving et al. (2018) · [`irving2018debate`](./human_ai_alignment_refs.bib)
- **Iterated Amplification** — [Supervising Strong Learners by Amplifying Weak Experts](https://arxiv.org/abs/1810.08575) — Christiano et al. (2018) · [`christiano2018amplification`](./human_ai_alignment_refs.bib)
- **Weak-to-Strong** — [Weak-to-Strong Generalization: Eliciting Strong Capabilities with Weak Supervision](https://proceedings.mlr.press/v235/burns24b.html) — Burns et al. (2024) · [`burns2024weakstrong`](./human_ai_alignment_refs.bib)
- **LLM Debate** — [Debating with More Persuasive LLMs Leads to More Truthful Answers](https://proceedings.mlr.press/v235/khan24a.html) — Khan et al. (2024) · [`khan2024debate`](./human_ai_alignment_refs.bib)
- **Prover-Verifier Games** — [Prover-Verifier Games Improve Legibility of LLM Outputs](https://arxiv.org/abs/2407.13692) — Kirchner et al. (2024) · [`kirchner2024proververifier`](./human_ai_alignment_refs.bib)
- **AI Control** — [AI Control: Improving Safety Despite Intentional Subversion](https://proceedings.mlr.press/v235/greenblatt24a.html) — Greenblatt et al. (2024) · [`greenblatt2024control`](./human_ai_alignment_refs.bib)
- **Barriers and Pathways to Human–AI Alignment** — [Barriers and Pathways to Human-AI Alignment: A Game-Theoretic Approach](https://arxiv.org/abs/2502.05934) — Nayebi (2025) · [`nayebi2025barriers`](./human_ai_alignment_refs.bib)
- **Self-Exploring Language Models: Active Preference Elicitation for Online Alignment** — [Self-Exploring Language Models: Active Preference Elicitation for Online Alignment](https://arxiv.org/abs/2405.19332) — Zhang et al. (2024) · [`zhang2024selm`](./human_ai_alignment_refs.bib)

## Alignment Mechanisms

### Training-Time Alignment

<a id="reward-verifier-modeling"></a>

#### Reward & Verifier Modeling

- **Deep RL from Human Preferences** — [Deep Reinforcement Learning from Human Preferences](https://proceedings.neurips.cc/paper/2017/hash/d5e2c0adad503c91f91df240d0cd4e49-Abstract.html) — Christiano et al. (2017) · [`christiano2017preferences`](./human_ai_alignment_refs.bib)
- **Fine-Tuning LMs from Human Preferences** — [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593) — Ziegler et al. (2019) · [`ziegler2019finetuning`](./human_ai_alignment_refs.bib)
- **Learning to Summarize** — [Learning to Summarize with Human Feedback](https://proceedings.neurips.cc/paper/2020/hash/1f89885d556929e98d3ef9b86448f951-Abstract.html) — Stiennon et al. (2020) · [`stiennon2020summarize`](./human_ai_alignment_refs.bib)
- **InstructGPT** — [Training Language Models to Follow Instructions with Human Feedback](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract.html) — Ouyang et al. (2022) · [`ouyang2022instructgpt`](./human_ai_alignment_refs.bib)
- **Helpful and Harmless Assistant** — [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862) — Bai et al. (2022) · [`bai2022hh`](./human_ai_alignment_refs.bib)
- **Sparrow** — [Improving Alignment of Dialogue Agents via Targeted Human Judgements](https://arxiv.org/abs/2209.14375) — Glaese et al. (2022) · [`glaese2022sparrow`](./human_ai_alignment_refs.bib)
- **Fine-Grained Human Feedback** — [Fine-Grained Human Feedback Gives Better Rewards for Language Model Training](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b8c90b65739ae8417e61eadb521f63d5-Abstract.html) — Wu et al. (2023) · [`wu2023finegrained`](./human_ai_alignment_refs.bib)
- **Safe RLHF** — [Safe RLHF: Safe Reinforcement Learning from Human Feedback](https://proceedings.iclr.cc/paper_files/paper/2024/hash/dd1577afd396928ed64216f3f1fd5556-Abstract-Conference.html) — Dai et al. (2024) · [`dai2024saferlhf`](./human_ai_alignment_refs.bib)
- **HelpSteer2** — [HelpSteer 2: Open-source Dataset for Training Top-Performing Reward Models](https://doi.org/10.52202/079017-0047) — Wang et al. (2024) · [`wang2024helpsteer2`](./human_ai_alignment_refs.bib)
- **Process- and Outcome-Based Feedback** — [Solving Math Word Problems with Process- and Outcome-Based Feedback](https://arxiv.org/abs/2211.14275) — Uesato et al. (2022) · [`uesato2022process`](./human_ai_alignment_refs.bib)
- **Let's Verify Step by Step** — [Let's Verify Step by Step](https://proceedings.iclr.cc/paper_files/paper/2024/hash/aca97732e30bcf1303bc22ac3924fd16-Abstract-Conference.html) — Lightman et al. (2024) · [`lightman2024verify`](./human_ai_alignment_refs.bib)
- **Math-Shepherd** — [Math-Shepherd: Verify and Reinforce LLMs Step-by-Step without Human Annotations](https://aclanthology.org/2024.acl-long.510/) — Wang et al. (2024) · [`wang2024mathshepherd`](./human_ai_alignment_refs.bib)
- **Prover-Verifier Games** — [Prover-Verifier Games Improve Legibility of LLM Outputs](https://arxiv.org/abs/2407.13692) — Kirchner et al. (2024) · [`kirchner2024proververifier`](./human_ai_alignment_refs.bib)
- **LLaVA-Critic** — [LLaVA-Critic: Learning to Evaluate Multimodal Models](https://doi.org/10.1109/CVPR52734.2025.01271) — Xiong et al. (2025) · [`xiong2025llavacritic`](./human_ai_alignment_refs.bib)
- **BiPRM** — [The Bidirectional Process Reward Model](https://aclanthology.org/2026.acl-long.572/) — Zhang et al. (2026) · [`zhang2026biprm`](./human_ai_alignment_refs.bib)

<a id="supervised-alignment"></a>

#### Supervised Alignment

- **FLAN** — [Finetuned Language Models Are Zero-Shot Learners](https://research.google/pubs/finetuned-language-models-are-zero-shot-learners/) — Wei et al. (2022) · [`wei2022flan`](./human_ai_alignment_refs.bib)
- **T0** — [Multitask Prompted Training Enables Zero-Shot Task Generalization](https://arxiv.org/abs/2110.08207) — Sanh et al. (2022) · [`sanh2022t0`](./human_ai_alignment_refs.bib)
- **Natural Instructions** — [Cross-Task Generalization via Natural Language Crowdsourcing Instructions](https://aclanthology.org/2022.acl-long.244/) — Mishra et al. (2022) · [`mishra2022natural`](./human_ai_alignment_refs.bib)
- **Super-NaturalInstructions** — [Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks](https://aclanthology.org/2022.emnlp-main.340/) — Wang et al. (2022) · [`wang2022superni`](./human_ai_alignment_refs.bib)
- **InstructGPT** — [Training Language Models to Follow Instructions with Human Feedback](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract.html) — Ouyang et al. (2022) · [`ouyang2022instructgpt`](./human_ai_alignment_refs.bib)
- **Self-Instruct** — [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://aclanthology.org/2023.acl-long.754/) — Wang et al. (2023) · [`wang2023selfinstruct`](./human_ai_alignment_refs.bib)
- **LIMA** — [LIMA: Less Is More for Alignment](https://papers.neurips.cc/paper_files/paper/2023/hash/ac662d74829e4407ce1d126477f4a03a-Abstract-Conference.html) — Zhou et al. (2023) · [`zhou2023lima`](./human_ai_alignment_refs.bib)
- **Flan Collection** — [The Flan Collection: Designing Data and Methods for Effective Instruction Tuning](https://proceedings.mlr.press/v202/longpre23a.html) — Longpre et al. (2023) · [`longpre2023flancollection`](./human_ai_alignment_refs.bib)
- **SteerLM** — [SteerLM: Attribute Conditioned SFT as an (User-Steerable) Alternative to RLHF](https://aclanthology.org/2023.findings-emnlp.754/) — Dong et al. (2023) · [`dong2023steerlm`](./human_ai_alignment_refs.bib)
- **Scaling Instruction-Finetuned Language Models** — [Scaling Instruction-Finetuned Language Models](https://www.jmlr.org/papers/v25/23-0870.html) — Chung et al. (2024) · [`chung2024flan`](./human_ai_alignment_refs.bib)
- **MetaAligner: Towards Generalizable Multi-Objective Alignment of Language Models** — [MetaAligner: Towards Generalizable Multi-Objective Alignment of Language Models](https://doi.org/10.52202/079017-1086) — Yang et al. (2024) · [`yang2024metaaligner`](./human_ai_alignment_refs.bib)

<a id="preference-ranking-optimization"></a>

#### Preference & Ranking Optimization

- **RRHF** — [RRHF: Rank Responses to Align Language Models with Human Feedback](https://proceedings.neurips.cc/paper_files/paper/2023/hash/23e6f78bdec844a9f7b6c957de2aae91-Abstract-Conference.html) — Yuan et al. (2023) · [`yuan2023rrhf`](./human_ai_alignment_refs.bib)
- **SLiC-HF** — [SLiC-HF: Sequence Likelihood Calibration with Human Feedback](https://arxiv.org/abs/2305.10425) — Zhao et al. (2023) · [`zhao2023slichf`](./human_ai_alignment_refs.bib)
- **DPO** — [Direct Preference Optimization: Your Language Model Is Secretly a Reward Model](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html) — Rafailov et al. (2023) · [`rafailov2023dpo`](./human_ai_alignment_refs.bib)
- **IPO & ΨPO** — [A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036) — Azar et al. (2023) · [`azar2023psipo`](./human_ai_alignment_refs.bib)
- **KTO** — [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306) — Ethayarajh et al. (2024) · [`ethayarajh2024kto`](./human_ai_alignment_refs.bib)
- **ORPO** — [ORPO: Monolithic Preference Optimization without Reference Model](https://aclanthology.org/2024.emnlp-main.626/) — Hong et al. (2024) · [`hong2024orpo`](./human_ai_alignment_refs.bib)
- **SimPO** — [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://papers.neurips.cc/paper_files/paper/2024/hash/e099c1c9699814af0be873a175361713-Abstract-Conference.html) — Meng et al. (2024) · [`meng2024simpo`](./human_ai_alignment_refs.bib)
- **TDPO** — [Token-Level Direct Preference Optimization](https://www.microsoft.com/en-us/research/publication/token-level-direct-preference-optimization/) — Zeng et al. (2024) · [`zeng2024tdpo`](./human_ai_alignment_refs.bib)
- **Direct Nash Optimization** — [Direct Nash Optimization: Teaching Language Models to Self-Improve with General Preferences](https://arxiv.org/abs/2404.03715) — Rosset et al. (2024) · [`rosset2024dno`](./human_ai_alignment_refs.bib)
- **SPPO** — [Self-Play Preference Optimization for Language Model Alignment](https://arxiv.org/abs/2405.00675) — Wu et al. (2025) · [`wu2025sppo`](./human_ai_alignment_refs.bib)
- **SDPO** — [SDPO: Segment-Level Direct Preference Optimization for Social Agents](https://aclanthology.org/2025.acl-long.607/) — Kong et al. (2025) · [`kong2025sdpo`](./human_ai_alignment_refs.bib)
- **Beyond Hallucinations: Enhancing LVLMs through Hallucination-Aware Direct Preference Optimization** — [Beyond Hallucinations: Enhancing LVLMs through Hallucination-Aware Direct Preference Optimization](https://arxiv.org/abs/2311.16839) — Zhao et al. (2023) · [`zhao2023hadpo`](./human_ai_alignment_refs.bib)
- **MMedPO: Aligning Medical Vision-Language Models with Clinical-Aware Multimodal Preference Optimization** — [MMedPO: Aligning Medical Vision-Language Models with Clinical-Aware Multimodal Preference Optimization](https://mlanthology.org/icml/2025/zhu2025icml-mmedpo/) — Zhu et al. (2025) · [`zhu2025mmedpo`](./human_ai_alignment_refs.bib)
- **Self-Exploring Language Models: Active Preference Elicitation for Online Alignment** — [Self-Exploring Language Models: Active Preference Elicitation for Online Alignment](https://arxiv.org/abs/2405.19332) — Zhang et al. (2024) · [`zhang2024selm`](./human_ai_alignment_refs.bib)

<a id="reinforcement-learning"></a>

#### Reinforcement Learning

- **Deep RL from Human Preferences** — [Deep Reinforcement Learning from Human Preferences](https://proceedings.neurips.cc/paper/2017/hash/d5e2c0adad503c91f91df240d0cd4e49-Abstract.html) — Christiano et al. (2017) · [`christiano2017preferences`](./human_ai_alignment_refs.bib)
- **Learning to Summarize** — [Learning to Summarize with Human Feedback](https://proceedings.neurips.cc/paper/2020/hash/1f89885d556929e98d3ef9b86448f951-Abstract.html) — Stiennon et al. (2020) · [`stiennon2020summarize`](./human_ai_alignment_refs.bib)
- **InstructGPT** — [Training Language Models to Follow Instructions with Human Feedback](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract.html) — Ouyang et al. (2022) · [`ouyang2022instructgpt`](./human_ai_alignment_refs.bib)
- **Process- and Outcome-Based Feedback** — [Solving Math Word Problems with Process- and Outcome-Based Feedback](https://arxiv.org/abs/2211.14275) — Uesato et al. (2022) · [`uesato2022process`](./human_ai_alignment_refs.bib)
- **Safe RLHF** — [Safe RLHF: Safe Reinforcement Learning from Human Feedback](https://proceedings.iclr.cc/paper_files/paper/2024/hash/dd1577afd396928ed64216f3f1fd5556-Abstract-Conference.html) — Dai et al. (2024) · [`dai2024saferlhf`](./human_ai_alignment_refs.bib)
- **RLAIF** — [RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://proceedings.mlr.press/v235/lee24t.html) — Lee et al. (2024) · [`lee2024rlaif`](./human_ai_alignment_refs.bib)
- **DeepSeekMath** — [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) — Shao et al. (2024) · [`shao2024deepseekmath`](./human_ai_alignment_refs.bib)
- **Rule-Based Rewards** — [Rule Based Rewards for Language Model Safety](https://scholar.google.com/scholar?q=Rule+Based+Rewards+for+Language+Model+Safety) — Mu et al. (2024) · *Scholar search* · [`mu2024rulebased`](./human_ai_alignment_refs.bib)
- **DeepSeek-R1** — [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — DeepSeek-AI (2025) · [`deepseek2025r1`](./human_ai_alignment_refs.bib)
- **RLHF Survey** — [A Survey of Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2312.14925) — Kaufmann et al. (2023) · [`kaufmann2023survey`](./human_ai_alignment_refs.bib)
- **Aligning Large Multimodal Models with Factually Augmented RLHF** — [Aligning Large Multimodal Models with Factually Augmented RLHF](https://aclanthology.org/2024.findings-acl.775/) — Sun et al. (2024) · [`sun2024factrlhf`](./human_ai_alignment_refs.bib)
- **MM-RLHF: The Next Step Forward in Multimodal LLM Alignment** — [MM-RLHF: The Next Step Forward in Multimodal LLM Alignment](https://icml.cc/virtual/2025/poster/45124) — Zhang et al. (2025) · [`zhang2025mmrlhf`](./human_ai_alignment_refs.bib)
- **Multi-turn Reinforcement Learning from Preference Human Feedback** — [Multi-turn Reinforcement Learning from Preference Human Feedback](https://scholar.google.com/scholar?q=Multi-turn+Reinforcement+Learning+from+Preference+Human+Feedback) — Shani et al. (2024) · *Scholar search* · [`shani2024multiturn`](./human_ai_alignment_refs.bib)
- **SAIL: Self-Improving Efficient Online Alignment of Large Language Models** — [SAIL: Self-Improving Efficient Online Alignment of Large Language Models](https://arxiv.org/abs/2406.15567) — Ding et al. (2024) · [`ding2024sail`](./human_ai_alignment_refs.bib)

<a id="alignment-distillation"></a>

#### Alignment Distillation

- **CycleAlign** — [CycleAlign: Iterative Distillation from Black-Box LLM to White-Box Models for Better Human Alignment](https://aclanthology.org/2024.findings-acl.869/) — Hong et al. (2024) · [`hong2024cyclealign`](./human_ai_alignment_refs.bib)
- **PLaD** — [PLaD: Preference-Based Large Language Model Distillation with Pseudo-Preference Pairs](https://aclanthology.org/2024.findings-acl.923/) — Zhang et al. (2024) · [`zhang2024plad`](./human_ai_alignment_refs.bib)
- **DPKD** — [Direct Preference Knowledge Distillation for Large Language Models](https://arxiv.org/abs/2406.19774) — Li et al. (2024) · [`li2024dpkd`](./human_ai_alignment_refs.bib)
- **ADPA** — [Advantage-Guided Distillation for Preference Alignment in Small Language Models](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2f891d026c7ba978168621842bc6fe73-Abstract-Conference.html) — Gao et al. (2025) · [`gao2025adpa`](./human_ai_alignment_refs.bib)
- **PAD** — [Capturing Nuanced Preferences: Preference-Aligned Distillation for Small Language Models](https://aclanthology.org/2025.findings-acl.822/) — Gu et al. (2025) · [`gu2025pad`](./human_ai_alignment_refs.bib)
- **AlignDistil** — [AlignDistil: Token-Level Language Model Alignment as Adaptive Policy Distillation](https://aclanthology.org/2025.acl-long.972/) — Zhang et al. (2025) · [`zhang2025aligndistil`](./human_ai_alignment_refs.bib)

### Inference-Time Alignment

<a id="steering-search-refinement"></a>

#### Steering, Search & Refinement

- **Inference-Time Intervention** — [Inference-Time Intervention: Eliciting Truthful Answers from a Language Model](https://doi.org/10.52202/075280-1797) — Li et al. (2023) · [`li2023iti`](./human_ai_alignment_refs.bib)
- **Representation Engineering** — [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405) — Zou et al. (2023) · [`zou2023repreng`](./human_ai_alignment_refs.bib)
- **Activation Addition** — [Activation Addition: Steering Language Models Without Optimization](https://arxiv.org/abs/2308.10248) — Turner et al. (2023) · [`turner2023activationaddition`](./human_ai_alignment_refs.bib)
- **Contrastive Activation Addition** — [Steering Language Models with Activation Engineering](https://doi.org/10.18653/v1/2024.acl-long.828) — Rimsky et al. (2024) · [`rimsky2024caa`](./human_ai_alignment_refs.bib)
- **WebGPT** — [WebGPT: Browser-Assisted Question-Answering with Human Feedback](https://arxiv.org/abs/2112.09332) — Nakano et al. (2021) · [`nakano2021webgpt`](./human_ai_alignment_refs.bib)
- **Self-Refine** — [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651) — Madaan et al. (2023) · [`madaan2023selfrefine`](./human_ai_alignment_refs.bib)
- **Reflexion** — [Reflexion: Language Agents with Verbal Reinforcement Learning](https://scholar.google.com/scholar?q=Reflexion%3A+Language+Agents+with+Verbal+Reinforcement+Learning) — Shinn et al. (2023) · *Scholar search* · [`shinn2023reflexion`](./human_ai_alignment_refs.bib)
- **CRITIC** — [CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing](https://scholar.google.com/scholar?q=CRITIC%3A+Large+Language+Models+Can+Self-Correct+with+Tool-Interactive+Critiquing) — Gou et al. (2024) · *Scholar search* · [`gou2024critic`](./human_ai_alignment_refs.bib)
- **Math-Shepherd** — [Math-Shepherd: Verify and Reinforce LLMs Step-by-Step without Human Annotations](https://aclanthology.org/2024.acl-long.510/) — Wang et al. (2024) · [`wang2024mathshepherd`](./human_ai_alignment_refs.bib)
- **LLaVA-Critic** — [LLaVA-Critic: Learning to Evaluate Multimodal Models](https://doi.org/10.1109/CVPR52734.2025.01271) — Xiong et al. (2025) · [`xiong2025llavacritic`](./human_ai_alignment_refs.bib)

<a id="human-control"></a>

#### Human Control

- **Cooperative IRL** — [Cooperative Inverse Reinforcement Learning](https://scholar.google.com/scholar?q=Cooperative+Inverse+Reinforcement+Learning) — Hadfield-Menell et al. (2016) · *Scholar search* · [`hadfieldmenell2016cirl`](./human_ai_alignment_refs.bib)
- **Safely Interruptible Agents** — [Safely Interruptible Agents](https://scholar.google.com/scholar?q=Safely+Interruptible+Agents) — Orseau et al. (2016) · *Scholar search* · [`orseau2016interruptible`](./human_ai_alignment_refs.bib)
- **Off-Switch Game** — [The Off-Switch Game](https://doi.org/10.24963/ijcai.2017/32) — Hadfield-Menell et al. (2017) · [`hadfieldmenell2017offswitch`](./human_ai_alignment_refs.bib)
- **Corrigibility** — [Corrigibility](https://cdn.aaai.org/ocs/ws/ws0067/10124-45900-1-PB.pdf) — Soares et al. (2015) · [`soares2015corrigibility`](./human_ai_alignment_refs.bib)
- **Corrigibility Transformation** — [Corrigibility Transformation: Constructing Goals That Accept Updates](https://arxiv.org/abs/2510.15395) — Hudson (2025) · [`hudson2025corrigibility`](./human_ai_alignment_refs.bib)
- **Learning to Defer** — [Consistent Estimators for Learning to Defer to an Expert](https://scholar.google.com/scholar?q=Consistent+Estimators+for+Learning+to+Defer+to+an+Expert) — Mozannar et al. (2020) · *Scholar search* · [`mozannar2020defer`](./human_ai_alignment_refs.bib)
- **AI Control** — [AI Control: Improving Safety Despite Intentional Subversion](https://proceedings.mlr.press/v235/greenblatt24a.html) — Greenblatt et al. (2024) · [`greenblatt2024control`](./human_ai_alignment_refs.bib)
- **STaR-GATE** — [STaR-GATE: Teaching Language Models to Ask Clarifying Questions](https://scholar.google.com/scholar?q=STaR-GATE%3A+Teaching+Language+Models+to+Ask+Clarifying+Questions) — Andukuri et al. (2024) · *Scholar search* · [`andukuri2024stargate`](./human_ai_alignment_refs.bib)

## Alignment Assurance

### Evaluation & Robustness

<a id="behavior-evaluator-assessment"></a>

#### Behavior & Evaluator Assessment

- **TruthfulQA** — [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://aclanthology.org/2022.acl-long.229/) — Lin et al. (2022) · [`lin2022truthfulqa`](./human_ai_alignment_refs.bib)
- **MT-Bench & Chatbot Arena** — [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://proceedings.neurips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html) — Zheng et al. (2023) · [`zheng2023mtbench`](./human_ai_alignment_refs.bib)
- **IFEval** — [Instruction-Following Evaluation for Large Language Models](https://arxiv.org/abs/2311.07911) — Zhou et al. (2023) · [`zhou2023ifeval`](./human_ai_alignment_refs.bib)
- **FActScore** — [FActScore: Fine-Grained Atomic Evaluation of Factual Precision in Long Form Text Generation](https://aclanthology.org/2023.emnlp-main.741/) — Min et al. (2023) · [`min2023factscore`](./human_ai_alignment_refs.bib)
- **XSTest** — [XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models](https://arxiv.org/abs/2308.01263) — Rottger et al. (2024) · [`rottger2024xstest`](./human_ai_alignment_refs.bib)
- **Chatbot Arena** — [Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference](https://scholar.google.com/scholar?q=Chatbot+Arena%3A+An+Open+Platform+for+Evaluating+LLMs+by+Human+Preference) — Chiang et al. (2024) · *Scholar search* · [`chiang2024arena`](./human_ai_alignment_refs.bib)
- **HarmBench** — [HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal](https://www.microsoft.com/en-us/research/publication/harmbench-a-standardized-evaluation-framework-for-automated-red-teaming-and-robust-refusal/) — Mazeika et al. (2024) · [`mazeika2024harmbench`](./human_ai_alignment_refs.bib)
- **HallusionBench** — [HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models](https://scholar.google.com/scholar?q=HallusionBench%3A+An+Advanced+Diagnostic+Suite+for+Entangled+Language+Hallucination+and+Visual+Illusion+in+Large+Vision-Language+Models) — Guan et al. (2024) · *Scholar search* · [`guan2024hallusionbench`](./human_ai_alignment_refs.bib)
- **RewardBench** — [RewardBench: Evaluating Reward Models for Language Modeling](https://aclanthology.org/2025.findings-naacl.96/) — Lambert et al. (2025) · [`lambert2025rewardbench`](./human_ai_alignment_refs.bib)
- **PRMBench** — [PRMBench: A Fine-Grained and Challenging Benchmark for Process-Level Reward Models](https://aclanthology.org/2025.acl-long.1230/) — Song et al. (2025) · [`song2025prmbench`](./human_ai_alignment_refs.bib)
- **AlpacaEval: An Automatic Evaluator of Instruction-Following Models** — [AlpacaEval: An Automatic Evaluator of Instruction-Following Models](https://github.com/tatsu-lab/alpaca_eval) — Li et al. (2023) · [`li2023alpacaeval`](./human_ai_alignment_refs.bib)
- **SafetyBench: Evaluating the Safety of Large Language Models** — [SafetyBench: Evaluating the Safety of Large Language Models](https://aclanthology.org/2024.acl-long.830/) — Zhang et al. (2024) · [`zhang2024safetybench`](./human_ai_alignment_refs.bib)
- **RAG-RewardBench: Benchmarking Reward Models in Retrieval-Augmented Generation** — [RAG-RewardBench: Benchmarking Reward Models in Retrieval-Augmented Generation](https://aclanthology.org/2025.findings-acl.877/) — Jin et al. (2025) · [`ragrewardbench2025`](./human_ai_alignment_refs.bib)
- **RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models** — [RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models](https://aclanthology.org/2024.acl-long.585/) — Niu et al. (2024) · [`niu2024ragtruth`](./human_ai_alignment_refs.bib)
- **Enabling Large Language Models to Generate Text with Citations** — [Enabling Large Language Models to Generate Text with Citations](https://aclanthology.org/2023.emnlp-main.398/) — Gao et al. (2023) · [`gao2023citations`](./human_ai_alignment_refs.bib)

<a id="adversarial-distribution-robustness"></a>

#### Adversarial & Distribution Robustness

- **Red Teaming LMs with LMs** — [Red Teaming Language Models with Language Models](https://doi.org/10.18653/v1/2022.emnlp-main.225) — Perez et al. (2022) · [`perez2022redteaming`](./human_ai_alignment_refs.bib)
- **Universal Adversarial Attacks** — [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043) — Zou et al. (2023) · [`zou2023jailbreak`](./human_ai_alignment_refs.bib)
- **HarmBench** — [HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal](https://www.microsoft.com/en-us/research/publication/harmbench-a-standardized-evaluation-framework-for-automated-red-teaming-and-robust-refusal/) — Mazeika et al. (2024) · [`mazeika2024harmbench`](./human_ai_alignment_refs.bib)
- **StrongREJECT** — [A StrongREJECT for Empty Jailbreaks](https://arxiv.org/abs/2402.10260) — Souly et al. (2024) · [`souly2024strongreject`](./human_ai_alignment_refs.bib)
- **Circuit Breakers** — [Improving Alignment and Robustness with Circuit Breakers](https://proceedings.neurips.cc/paper_files/paper/2024/hash/97ca7168c2c333df5ea61ece3b3276e1-Abstract-Conference.html) — Zou et al. (2024) · [`zou2024circuitbreakers`](./human_ai_alignment_refs.bib)
- **XSTest** — [XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models](https://arxiv.org/abs/2308.01263) — Rottger et al. (2024) · [`rottger2024xstest`](./human_ai_alignment_refs.bib)
- **Concrete Problems in AI Safety** — [Concrete Problems in AI Safety](https://arxiv.org/abs/1606.06565) — Amodei et al. (2016) · [`amodei2016concrete`](./human_ai_alignment_refs.bib)
- **Goal Misgeneralization in Deep RL** — [Goal Misgeneralization in Deep Reinforcement Learning](https://proceedings.mlr.press/v162/langosco22a.html) — Di Langosco et al. (2022) · [`langosco2022goal`](./human_ai_alignment_refs.bib)
- **Goal Misgeneralization** — [Goal Misgeneralization: Why Correct Specifications Aren't Enough for Correct Goals](https://arxiv.org/abs/2210.01790) — Shah et al. (2022) · [`shah2022goal`](./human_ai_alignment_refs.bib)
- **When Context Flips, Safety Breaks** — [When Context Flips, Safety Breaks: Diagnosing Brittle Safety in Aligned Language Models](https://arxiv.org/abs/2605.27851) — Choi et al. (2026) · [`choi2026context`](./human_ai_alignment_refs.bib)
- **HELM** — [Holistic Evaluation of Language Models](https://scholar.google.com/scholar?q=Holistic+Evaluation+of+Language+Models) — Liang et al. (2023) · *Scholar search* · [`liang2023helm`](./human_ai_alignment_refs.bib)
- **Extreme-Risk Evaluation** — [Model Evaluation for Extreme Risks](https://arxiv.org/abs/2305.15324) — Shevlane et al. (2023) · [`shevlane2023extremerisks`](./human_ai_alignment_refs.bib)

<a id="alignment-preservation"></a>

#### Alignment Preservation

- **Fine-Tuning Aligned LMs Compromises Safety** — [Fine-Tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!](https://proceedings.iclr.cc/paper_files/paper/2024/hash/83b7da3ed13f06c13ce82235c8eedf35-Abstract-Conference.html) — Qi et al. (2024) · [`qi2024finetuning`](./human_ai_alignment_refs.bib)
- **Circuit Breakers** — [Improving Alignment and Robustness with Circuit Breakers](https://proceedings.neurips.cc/paper_files/paper/2024/hash/97ca7168c2c333df5ea61ece3b3276e1-Abstract-Conference.html) — Zou et al. (2024) · [`zou2024circuitbreakers`](./human_ai_alignment_refs.bib)
- **COPR** — [COPR: Continual Human Preference Learning via Optimal Policy Regularization](https://aclanthology.org/2025.findings-acl.281/) — Zhang et al. (2025) · [`zhang2025copr`](./human_ai_alignment_refs.bib)
- **Lifelong Safety Alignment** — [Lifelong Safety Alignment for Language Models](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1ca2f4528792b31eab7a3e7f6a1c130b-Abstract-Conference.html) — Wang et al. (2025) · [`wang2025lifelong`](./human_ai_alignment_refs.bib)
- **Alignment Midtraining for Animals** — [Alignment Midtraining for Animals](https://arxiv.org/abs/2604.13076) — Brazilek et al. (2026) · [`brazilek2026alignment`](./human_ai_alignment_refs.bib)
- **Layer-Aware Representation Filtering: Purifying Finetuning Data to Preserve LLM Safety Alignment** — [Layer-Aware Representation Filtering: Purifying Finetuning Data to Preserve LLM Safety Alignment](https://aclanthology.org/2025.emnlp-main.406/) — Li et al. (2025) · [`li2025larf`](./human_ai_alignment_refs.bib)

### System Assurance

<a id="mechanistic-theoretical-evidence"></a>

#### Mechanistic & Theoretical Evidence

- **Discovering Latent Knowledge** — [Discovering Latent Knowledge in Language Models Without Supervision](https://scholar.google.com/scholar?q=Discovering+Latent+Knowledge+in+Language+Models+Without+Supervision) — Burns et al. (2023) · *Scholar search* · [`burns2023latent`](./human_ai_alignment_refs.bib)
- **Inference-Time Intervention** — [Inference-Time Intervention: Eliciting Truthful Answers from a Language Model](https://doi.org/10.52202/075280-1797) — Li et al. (2023) · [`li2023iti`](./human_ai_alignment_refs.bib)
- **Representation Engineering** — [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405) — Zou et al. (2023) · [`zou2023repreng`](./human_ai_alignment_refs.bib)
- **Contrastive Activation Addition** — [Steering Language Models with Activation Engineering](https://doi.org/10.18653/v1/2024.acl-long.828) — Rimsky et al. (2024) · [`rimsky2024caa`](./human_ai_alignment_refs.bib)
- **Sustaining AI Safety** — [Sustaining AI Safety: Control-Theoretic External Impossibility, Intrinsic Necessity, and Structural Requirements](https://arxiv.org/abs/2605.12963) — Mazzu (2026) · [`mazzu2026sustaining`](./human_ai_alignment_refs.bib)
- **Corrigibility Transformation** — [Corrigibility Transformation: Constructing Goals That Accept Updates](https://arxiv.org/abs/2510.15395) — Hudson (2025) · [`hudson2025corrigibility`](./human_ai_alignment_refs.bib)
- **Corrigibility** — [Corrigibility](https://cdn.aaai.org/ocs/ws/ws0067/10124-45900-1-PB.pdf) — Soares et al. (2015) · [`soares2015corrigibility`](./human_ai_alignment_refs.bib)
- **Off-Switch Game** — [The Off-Switch Game](https://doi.org/10.24963/ijcai.2017/32) — Hadfield-Menell et al. (2017) · [`hadfieldmenell2017offswitch`](./human_ai_alignment_refs.bib)

<a id="monitoring-auditing"></a>

#### Monitoring & Auditing

- **HELM** — [Holistic Evaluation of Language Models](https://scholar.google.com/scholar?q=Holistic+Evaluation+of+Language+Models) — Liang et al. (2023) · *Scholar search* · [`liang2023helm`](./human_ai_alignment_refs.bib)
- **Model-Written Evaluations** — [Discovering Language Model Behaviors with Model-Written Evaluations](https://aclanthology.org/2023.findings-acl.847/) — Perez et al. (2023) · [`perez2023modelwritten`](./human_ai_alignment_refs.bib)
- **Extreme-Risk Evaluation** — [Model Evaluation for Extreme Risks](https://arxiv.org/abs/2305.15324) — Shevlane et al. (2023) · [`shevlane2023extremerisks`](./human_ai_alignment_refs.bib)
- **EvalSafetyGap** — [EvalSafetyGap: A Hybrid Survey and Conceptual Framework for LLM Evaluation-Safety Failures](https://arxiv.org/abs/2606.30219) — Uluirmak et al. (2026) · [`uluirmak2026evalsafetygap`](./human_ai_alignment_refs.bib)
- **AI Control** — [AI Control: Improving Safety Despite Intentional Subversion](https://proceedings.mlr.press/v235/greenblatt24a.html) — Greenblatt et al. (2024) · [`greenblatt2024control`](./human_ai_alignment_refs.bib)
- **Test-Set Contamination** — [Proving Test Set Contamination in Black-Box Language Models](https://scholar.google.com/scholar?q=Proving+Test+Set+Contamination+in+Black-Box+Language+Models) — Oren et al. (2024) · *Scholar search* · [`oren2024contamination`](./human_ai_alignment_refs.bib)
- **RewardBench** — [RewardBench: Evaluating Reward Models for Language Modeling](https://aclanthology.org/2025.findings-naacl.96/) — Lambert et al. (2025) · [`lambert2025rewardbench`](./human_ai_alignment_refs.bib)

## General survey context

- **Bridging the Gap: A Survey on Integrating (Human) Feedback for Natural Language Generation** — [Bridging the Gap: A Survey on Integrating (Human) Feedback for Natural Language Generation](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00626/118795/) — Fernandes et al. (2023) · [`fernandes2024feedback`](./human_ai_alignment_refs.bib)
- **AI Alignment: A Contemporary Survey** — [AI Alignment: A Contemporary Survey](https://doi.org/10.1145/3770749) — Ji et al. (2025) · [`ji2025alignment`](./human_ai_alignment_refs.bib)
- **Large Language Model Alignment: A Survey** — [Large Language Model Alignment: A Survey](https://arxiv.org/abs/2309.15025) — Shen et al. (2023) · [`shen2023alignment`](./human_ai_alignment_refs.bib)
- **A Survey on Human Preference Learning for Large Language Models** — [A Survey on Human Preference Learning for Large Language Models](https://arxiv.org/abs/2406.11191) — Jiang et al. (2024) · [`jiang2024preferencesurvey`](./human_ai_alignment_refs.bib)
- **Towards a Unified View of Preference Learning for Large Language Models: A Survey** — [Towards a Unified View of Preference Learning for Large Language Models: A Survey](https://arxiv.org/abs/2409.02795) — Gao et al. (2024) · [`gao2024unifiedpreference`](./human_ai_alignment_refs.bib)
- **Aligning Multimodal LLM with Human Preference: A Survey** — [Aligning Multimodal LLM with Human Preference: A Survey](https://arxiv.org/abs/2503.14504) — Yu et al. (2025) · [`yu2025multimodalsurvey`](./human_ai_alignment_refs.bib)

## Using the taxonomy

`main.tex` includes the taxonomy with `\input{taxonomy}`. Keep this inclusion only once to avoid duplicate figure labels. Compile with the existing BibTeX database. Each TikZ category uses its descriptive node key; its paper panel uses the same key prefixed with `papers-`.

The existing `Figure/Taxonomy.png` depicts an older version and is not displayed here. The taxonomy now spans four figures. Category assignments in this README and `taxonomy.tex` have been checked for agreement.

## Contributing

1. Add or correct the record in `human_ai_alignment_refs.bib`, preferably with a publication URL or DOI.
2. Add the paper to the appropriate category in this README using its descriptive category name and BibTeX key.
3. Update the matching figure panel as well, retaining the descriptive node key and using `\taxpaper{Short name}{bibtex-key}`.
4. Explain the proposed category assignment; distinguish empirical methods, benchmarks, surveys and conditional theoretical results.

## Source files

- [`taxonomy.tex`](./taxonomy.tex): four editable taxonomy figures, complete category references and descriptive node names.
- [`human_ai_alignment_refs.bib`](./human_ai_alignment_refs.bib): all bibliographic records.
- [`main.tex`](./main.tex): manuscript source.

## Completeness check

All 145 unique BibTeX keys are represented in the taxonomy and this README. The 23 terminal categories contain 139 unique records; six additional general surveys appear in the context panel. Multiple category assignments are intentional. This verifies coverage and citation consistency against the supplied files, not independent bibliographic accuracy or a fresh full-text review.
