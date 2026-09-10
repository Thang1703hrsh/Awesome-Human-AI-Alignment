import subprocess
import shutil
import re
from pathlib import Path

# =========================================================
# LATEX TAXONOMY -- STANDALONE VERSION
# =========================================================
# Lưu ý:
# - Không dùng figure*, adjustbox, caption, label.
# - Không dùng \citep vì đây là file standalone.
# - Bibkey trong source được tự động đổi thành [Author et al., Year] khi xuất.
# - Có thêm refsUltraDense cho các node chứa rất nhiều paper.
# =========================================================

latex_code = r"""
\documentclass[margin=5pt]{standalone}

\usepackage{tikz}
\usetikzlibrary{calc,backgrounds}
\usepackage{xcolor}

% =========================================================
% COLORS
% =========================================================

\definecolor{categoryyellow}{RGB}{255,255,204}
\definecolor{categoryblue}{RGB}{207,226,251}
\definecolor{categorygreen}{RGB}{226,240,220}
\definecolor{referencegray}{RGB}{248,248,248}
\definecolor{referenceblue}{RGB}{0,0,128}

% Standalone citation representation
\newcommand{\taxpaper}[2]{#1~{\color{referenceblue}[#2]}}

\begin{document}

\begin{tikzpicture}[
    x=1pt,
    y=-1pt,
    box/.style={
        draw=black!70,
        line width=.55pt,
        rounded corners=2pt,
        inner xsep=3pt,
        inner ysep=3pt,
        outer sep=0pt,
        align=center,
        font=\fontsize{9}{10.5}\selectfont
    },
    refs/.style={
        box,
        fill=referencegray,
        align=left,
        text width=377pt,
        minimum width=383pt,
        minimum height=29pt,
        font=\fontsize{8}{9.5}\selectfont
    },
    refsDense/.style={
        refs,
        font=\fontsize{6.8}{8.0}\selectfont
    },
    refsUltraDense/.style={
        refs,
        font=\fontsize{6.2}{7.2}\selectfont
    },
    branch/.style={
        draw=black!60,
        line width=.55pt,
        line cap=round,
        line join=round
    }
]

\def\RowGap{3pt}
\def\GroupGap{7pt}
\def\PillarGap{12pt}

% =========================================================
% I. ALIGNMENT SPECIFICATION
% =========================================================

\node[refsDense,anchor=north] (refTaskAssistance) at (536.5,0) {
\taxpaper{Scaling Instruction-Finetuned LMs}{chung2024flan};
\taxpaper{UltraFeedback}{cui2024ultrafeedback};
\taxpaper{HelpSteer2}{wang2024helpsteer2};
\taxpaper{Parrot}{sun2024parrot};
\taxpaper{LIONs}{yu2024lions};
\taxpaper{Magpie}{xu2025magpie};
\taxpaper{UltraIF}{an2025ultraif};
\taxpaper{A Good Plan is Hard to Find}{balepur2025goodplan};
\taxpaper{SFTMix}{xiao2026sftmix}.
};

\node[
    box,
    fill=categorygreen,
    text width=106pt,
    minimum width=112pt,
    minimum height=29pt
] (taskAssistance) at (277,0 |- refTaskAssistance.center)
{Task \& Assistance Alignment};

\node[refsDense,anchor=north]
(refSafetyAlignment)
at ([yshift=-\RowGap]refTaskAssistance.south) {
\taxpaper{Safe RLHF}{dai2024saferlhf};
\taxpaper{Circuit Breakers}{zou2024circuitbreakers};
\taxpaper{Rule-Based Rewards}{mu2024rulebased};
\taxpaper{Vaccine}{huang2024vaccine};
\taxpaper{Deliberative Alignment}{guan2024deliberative};
\taxpaper{Lifelong Safety Alignment}{wang2025lifelong};
\taxpaper{LARF}{li2025larf};
\taxpaper{AlphaAlign}{zhang2026alphaalign};
\taxpaper{Safety at One Shot}{zhang2026safetyoneshot};
\taxpaper{LASA}{yang2026lasa}.
};

\node[
    box,
    fill=categorygreen,
    text width=106pt,
    minimum width=112pt,
    minimum height=29pt
] (safetyAlignment) at (277,0 |- refSafetyAlignment.center)
{Safety Alignment};

\coordinate (alignmentObjectivesMid)
at ($(taskAssistance.center)!0.5!(safetyAlignment.center)$);

\node[
    box,
    fill=categoryblue,
    text width=82pt,
    minimum width=88pt,
    minimum height=29pt
] (alignmentObjectives) at (163,0 |- alignmentObjectivesMid)
{Alignment Objectives};


\node[refsDense,anchor=north]
(refPersonalizedAlignment)
at ([yshift=-\GroupGap]refSafetyAlignment.south) {
\taxpaper{Personalizing RLHF}{poddar2024personalizing};
\taxpaper{OPPU}{tan2024oppu};
\taxpaper{PRISM}{kirk2024prism};
\taxpaper{PAD}{chen2025pad};
\taxpaper{PROSE}{aroca2025prose};
\taxpaper{RLPA}{zhao2025rlpa};
\taxpaper{UserAlign}{padurean2025useralign};
\taxpaper{AlignX}{li2026alignx};
\taxpaper{C-BPO}{ma2026cbpo};
\taxpaper{Profile-to-PEFT}{tan2026profilepeft}.
};

\node[
    box,
    fill=categorygreen,
    text width=106pt,
    minimum width=112pt,
    minimum height=29pt
] (personalizedAlignment)
at (277,0 |- refPersonalizedAlignment.center)
{Personalized Alignment};


\node[refsDense,anchor=north]
(refPluralisticSocietal)
at ([yshift=-\RowGap]refPersonalizedAlignment.south) {
\taxpaper{Roadmap to Pluralistic Alignment}{sorensen2024pluralistic};
\taxpaper{CultureSPA}{xu2025culturespa};
\taxpaper{Pairwise Calibrated Rewards}{halpern2025pairwise};
\taxpaper{Heterogeneous Preference Alignment}{shirali2025heterogeneous};
\taxpaper{Steerable Pluralistic CoT}{zhang2025pluralisticcot};
\taxpaper{Operationalizing Pluralistic Values}{ali2026pluralistic};
\taxpaper{FGD-Align}{pan2026fgdalign};
\taxpaper{ALPHA}{bhattacharyya2026alpha};
\taxpaper{Cross-Cultural Consensus}{zhang2026crosscultural};
\taxpaper{DVMap}{zhu2026dvmap}.
};

\node[
    box,
    fill=categorygreen,
    text width=106pt,
    minimum width=112pt,
    minimum height=29pt
] (pluralisticSocietal)
at (277,0 |- refPluralisticSocietal.center)
{Pluralistic \& Societal Alignment};


\node[refsDense,anchor=north]
(refUncertaintyDrift)
at ([yshift=-\RowGap]refPluralisticSocietal.south) {
\taxpaper{Distributional Preference Learning}{siththaranjan2024distributional};
\taxpaper{Active Preference Learning}{muldrew2024active};
\taxpaper{PerpCorrect}{kong2024perpcorrect};
\taxpaper{Stability of Moral Preferences}{boerstler2024stability};
\taxpaper{COPR}{zhang2025copr};
\taxpaper{UGDA}{sun2025ugda};
\taxpaper{Uncertainty-Aware RM Routing}{xu2025uncertaintyjudge};
\taxpaper{Moral Change or Noise?}{keswani2026moralchange};
\taxpaper{Reward Model Routing}{wu2026rmrouting}.
};

\node[
    box,
    fill=categorygreen,
    text width=106pt,
    minimum width=112pt,
    minimum height=29pt
] (uncertaintyDrift)
at (277,0 |- refUncertaintyDrift.center)
{Uncertainty \& Drift};

\coordinate (valuesStakeholdersMid)
at ($(personalizedAlignment.center)!0.5!(uncertaintyDrift.center)$);

\node[
    box,
    fill=categoryblue,
    text width=82pt,
    minimum width=88pt,
    minimum height=29pt
] (valuesStakeholders)
at (163,0 |- valuesStakeholdersMid)
{Values \& Stakeholders};

\coordinate (specificationMid)
at ($(alignmentObjectives.center)!0.5!(valuesStakeholders.center)$);

\node[
    box,
    fill=categoryyellow,
    text width=64pt,
    minimum width=70pt,
    minimum height=29pt
] (pillarSpecification)
at (70,0 |- specificationMid)
{\textbf{Alignment\\Specification}};


% =========================================================
% II. ALIGNMENT SUPERVISION
% =========================================================

\node[refsDense,anchor=north]
(refHumanFeedback)
at ([yshift=-\PillarGap]refUncertaintyDrift.south) {
\taxpaper{HelpSteer2}{wang2024helpsteer2};
\taxpaper{RLHF-V}{yu2024rlhfv};
\taxpaper{Fine-grained Supervision}{xu2024finegrained};
\taxpaper{ALT}{lloret2024alt};
\taxpaper{DITTO}{shaikh2025ditto};
\taxpaper{BCO}{jung2025bco};
\taxpaper{Hybrid Preferences}{miranda2025hybrid};
\taxpaper{MM-RLHF}{zhang2025mmrlhf};
\taxpaper{WildFeedback}{shi2026wildfeedback}.
};

\node[
    box,
    fill=categorygreen,
    text width=106pt,
    minimum width=112pt,
    minimum height=29pt
] (humanFeedback)
at (277,0 |- refHumanFeedback.center)
{Human Feedback};


\node[refsDense,anchor=north]
(refAIFeedback)
at ([yshift=-\RowGap]refHumanFeedback.south) {
\taxpaper{RLAIF vs. RLHF}{lee2024rlaif};
\taxpaper{UltraFeedback}{cui2024ultrafeedback};
\taxpaper{Self-Rewarding LMs}{yuan2024selfrewarding};
\taxpaper{DLMA}{liu2024dlma};
\taxpaper{VLFeedback}{li2024vlfeedback};
\taxpaper{Diverse AI Feedback}{yu2025daif};
\taxpaper{LLM-as-a-Judge Alignment}{ye2025conj};
\taxpaper{LLaVA-Critic}{xiong2025llavacritic};
\taxpaper{Self-Alignment Optimization}{yin2026sao};
\taxpaper{Check Your Work}{cook2026checklist}.
};

\node[
    box,
    fill=categorygreen,
    text width=106pt,
    minimum width=112pt,
    minimum height=29pt
] (aiFeedback)
at (277,0 |- refAIFeedback.center)
{AI Feedback};


\node[refsDense,anchor=north]
(refProgrammaticVerifiable)
at ([yshift=-\RowGap]refAIFeedback.south) {
\taxpaper{Math-Shepherd}{wang2024mathshepherd};
\taxpaper{Rule-Based Rewards}{mu2024rulebased};
\taxpaper{V-STaR}{hosseini2024vstar};
\taxpaper{DeepSeek-R1}{deepseek2025r1};
\taxpaper{Rewarding Progress}{setlur2025rewarding};
\taxpaper{RLEF}{gehring2025rlef};
\taxpaper{Generative Verifiers}{zhang2025genrm};
\taxpaper{Crossing the Reward Bridge}{su2026rewardbridge};
\taxpaper{Knowledge-to-Verification}{yuan2026k2v}.
};

\node[
    box,
    fill=categorygreen,
    text width=106pt,
    minimum width=112pt,
    minimum height=29pt
] (programmaticVerifiable)
at (277,0 |- refProgrammaticVerifiable.center)
{Programmatic \& Verifiable Feedback};

\coordinate (feedbackSourceMid)
at ($(humanFeedback.center)!0.5!(programmaticVerifiable.center)$);

\node[
    box,
    fill=categoryblue,
    text width=82pt,
    minimum width=88pt,
    minimum height=29pt
] (feedbackSource)
at (163,0 |- feedbackSourceMid)
{Feedback Source};


\node[refsDense,anchor=north]
(refDemonstrationsPreferences)
at ([yshift=-\GroupGap]refProgrammaticVerifiable.south) {
\taxpaper{HelpSteer2}{wang2024helpsteer2};
\taxpaper{UltraFeedback}{cui2024ultrafeedback};
\taxpaper{Active Preference Learning}{muldrew2024active};
\taxpaper{Reward from Demonstrations}{li2024demoreward};
\taxpaper{DITTO}{shaikh2025ditto};
\taxpaper{Joint Demo \& Preference Learning}{li2025joint};
\taxpaper{BCO}{jung2025bco};
\taxpaper{PUGC}{tan2025pugc};
\taxpaper{Preference Data Sweet Spot}{xiao2025sweetspot};
\taxpaper{WildFeedback}{shi2026wildfeedback}.
};

\node[
    box,
    fill=categorygreen,
    text width=106pt,
    minimum width=112pt,
    minimum height=29pt
] (demonstrationsPreferences)
at (277,0 |- refDemonstrationsPreferences.center)
{Demonstrations \& Preferences};


\node[refsDense,anchor=north]
(refCritiqueProcess)
at ([yshift=-\RowGap]refDemonstrationsPreferences.south) {
\taxpaper{Let's Verify Step by Step}{lightman2024verify};
\taxpaper{Critic-RM}{yu2025criticrm};
\taxpaper{Rewarding Progress}{setlur2025rewarding};
\taxpaper{OpenPRM}{zhang2025openprm};
\taxpaper{CTRL}{xie2025ctrl};
\taxpaper{DG-PRM}{yin2025dgprm};
\taxpaper{RCO}{yu2025rco};
\taxpaper{Critique-RL}{xi2026critiquerl};
\taxpaper{BiPRM}{zhang2026biprm};
\taxpaper{Agent-RRM}{fan2026agentrrm}.
};

\node[
    box,
    fill=categorygreen,
    text width=106pt,
    minimum width=112pt,
    minimum height=29pt
] (critiqueProcess)
at (277,0 |- refCritiqueProcess.center)
{Critique \& Process Feedback};


\node[refsDense,anchor=north]
(refReliableScalable)
at ([yshift=-\RowGap]refCritiqueProcess.south) {
\taxpaper{Weak-to-Strong}{burns2024weakstrong};
\taxpaper{Debate Helps Weak-to-Strong}{lang2025debate};
\taxpaper{Neural Interactive Proofs}{hammond2025neuralproofs};
\taxpaper{Scaling Laws for Oversight}{engels2025scaling};
\taxpaper{AI Debate Aids Assessment}{rahman2025aidebate};
\taxpaper{Great Models Think Alike}{goel2025oversight};
\taxpaper{Partitioned Human Supervision}{yin2026partitioned};
\taxpaper{Peer Prediction}{qiu2026peerprediction};
\taxpaper{Selective Weak-to-Strong}{lang2026selective};
\taxpaper{Agentic Oversight}{ranaldi2026agentic}.
};

\node[
    box,
    fill=categorygreen,
    text width=106pt,
    minimum width=112pt,
    minimum height=29pt
] (reliableScalable)
at (277,0 |- refReliableScalable.center)
{Reliable \& Scalable Oversight};

\coordinate (feedbackOversightMid)
at ($(demonstrationsPreferences.center)!0.5!(reliableScalable.center)$);

\node[
    box,
    fill=categoryblue,
    text width=82pt,
    minimum width=88pt,
    minimum height=29pt
] (feedbackOversight)
at (163,0 |- feedbackOversightMid)
{Feedback \& Oversight};

\coordinate (supervisionMid)
at ($(feedbackSource.center)!0.5!(feedbackOversight.center)$);

\node[
    box,
    fill=categoryyellow,
    text width=64pt,
    minimum width=70pt,
    minimum height=29pt
] (pillarSupervision)
at (70,0 |- supervisionMid)
{\textbf{Alignment\\Supervision}};


% =========================================================
% III. ALIGNMENT MECHANISMS
% =========================================================

\node[refsDense,anchor=north]
(refRewardVerifier)
at ([yshift=-\PillarGap]refReliableScalable.south) {
\taxpaper{Generative Verifiers}{zhang2025genrm};
\taxpaper{GRAM}{wang2025gram};
\taxpaper{Reward Reasoning Models}{guo2025rrm};
\taxpaper{Agentic Reward Modeling}{peng2025agenticrm};
\taxpaper{DG-PRM}{yin2025dgprm};
\taxpaper{RM-R1}{chen2026rmr1};
\taxpaper{PRISM}{zhou2026prismrm};
\taxpaper{OpenRubrics}{liu2026openrubrics};
\taxpaper{BiPRM}{zhang2026biprm};
\taxpaper{Rationale-Consistent RM}{wang2026rationale}.
};

\node[
    box,
    fill=categorygreen,
    text width=106pt,
    minimum width=112pt,
    minimum height=29pt
] (rewardVerifier)
at (277,0 |- refRewardVerifier.center)
{Reward \& Verifier Modeling};


\node[refsDense,anchor=north]
(refSupervisedAlignment)
at ([yshift=-\RowGap]refRewardVerifier.south) {
\taxpaper{Scaling Instruction-Finetuned LMs}{chung2024flan};
\taxpaper{Instruction Backtranslation}{li2024backtranslation};
\taxpaper{DEITA}{liu2024deita};
\taxpaper{LESS}{xia2024less};
\taxpaper{Magpie}{xu2025magpie};
\taxpaper{GRAPE}{zhang2025grape};
\taxpaper{MAIN}{yang2025main};
\taxpaper{SFTMix}{xiao2026sftmix};
\taxpaper{OASIS}{lee2026oasis}.
};

\node[
    box,
    fill=categorygreen,
    text width=106pt,
    minimum width=112pt,
    minimum height=29pt
] (supervisedAlignment)
at (277,0 |- refSupervisedAlignment.center)
{Supervised Alignment};


\node[refsUltraDense,anchor=north]
(refPreferenceOptimization)
at ([yshift=-\RowGap]refSupervisedAlignment.south) {
\taxpaper{HyPO}{yuan2026hypo};
\taxpaper{Causal DPO}{le2026causaldpo};
\taxpaper{SPPO}{wu2025sppo};
\taxpaper{SDPO}{kong2025sdpo};
\taxpaper{MMedPO}{zhu2025mmedpo};
\taxpaper{AlphaDPO}{wu2025alphadpo};
\taxpaper{WSPO}{zhu2025wspo};
\taxpaper{MPO}{wang2025mpo};
\taxpaper{Anchored PO}{doosterlinck2025apo};
\taxpaper{SGDPO}{zhu2025sgdpo};
\taxpaper{T-REG}{zhou2025treg};
\taxpaper{Bregman PO}{kim2025bregmanpo};
\taxpaper{PRO}{guo2025pro};
\taxpaper{TDPO}{zeng2024tdpo};
\taxpaper{rDPO}{chowdhury2024rdpo};
\taxpaper{Nash-MD}{munos2024nash};
\taxpaper{Behavior-PO}{xu2024behaviorbpo};
\taxpaper{CPO}{xu2024cpo};
\taxpaper{KTO}{ethayarajh2024kto};
\taxpaper{ORPO}{hong2024orpo};
\taxpaper{SimPO}{meng2024simpo};
\taxpaper{Direct Nash Optimization}{rosset2024dno};
\taxpaper{Self-Exploring LMs}{zhang2024selm};
\taxpaper{SAIL}{ding2024sail};
\taxpaper{RIME}{cheng2024rime};
\taxpaper{Active Preference Learning}{muldrew2024active};
\taxpaper{Multi-turn RLHF}{shani2024multiturn};
\taxpaper{MetaAligner}{yang2024metaaligner}.
};

\node[
    box,
    fill=categorygreen,
    text width=106pt,
    minimum width=112pt,
    minimum height=29pt
] (preferenceOptimization)
at (277,0 |- refPreferenceOptimization.center)
{Preference Optimization};


\node[refsDense,anchor=north]
(refReinforcementLearning)
at ([yshift=-\RowGap]refPreferenceOptimization.south) {
\taxpaper{Safe RLHF}{dai2024saferlhf};
\taxpaper{ReMax}{li2024remax};
\taxpaper{Back to Basics}{ahmadian2024reinforce};
\taxpaper{Multi-turn RLHF}{shani2024multiturn};
\taxpaper{DeepSeek-R1}{deepseek2025r1};
\taxpaper{RLCF}{viswanathan2025rlcf};
\taxpaper{MM-RLHF}{zhang2025mmrlhf};
\taxpaper{Rectified Policy Optimization}{peng2025repo};
\taxpaper{AlphaAlign}{zhang2026alphaalign};
\taxpaper{RLVRR}{jiang2026rlvrr}.
};

\node[
    box,
    fill=categorygreen,
    text width=106pt,
    minimum width=112pt,
    minimum height=29pt
] (reinforcementLearning)
at (277,0 |- refReinforcementLearning.center)
{Reinforcement Learning};


\node[refsDense,anchor=north]
(refAlignmentDistillation)
at ([yshift=-\RowGap]refReinforcementLearning.south) {
\taxpaper{CycleAlign}{hong2024cyclealign};
\taxpaper{PLaD}{zhang2024plad};
\taxpaper{Self-Rewarding Distillation}{liu2024dlma};
\taxpaper{ADPA}{gao2025adpa};
\taxpaper{AlignDistil}{zhang2025aligndistil};
\taxpaper{PAD}{gu2025pad};
\taxpaper{CTPD}{nguyen2026ctpd}.
};

\node[
    box,
    fill=categorygreen,
    text width=106pt,
    minimum width=112pt,
    minimum height=29pt
] (alignmentDistillation)
at (277,0 |- refAlignmentDistillation.center)
{Alignment Distillation};

\coordinate (trainingAlignmentMid)
at ($(rewardVerifier.center)!0.5!(alignmentDistillation.center)$);

\node[
    box,
    fill=categoryblue,
    text width=82pt,
    minimum width=88pt,
    minimum height=29pt
] (trainingAlignment)
at (163,0 |- trainingAlignmentMid)
{Training-Time Alignment};


\node[refsDense,anchor=north]
(refSteeringSearchRefinement)
at ([yshift=-\GroupGap]refAlignmentDistillation.south) {
\taxpaper{CAA}{rimsky2024caa};
\taxpaper{CRITIC}{gou2024critic};
\taxpaper{ARGS}{khanov2024args};
\taxpaper{Instruction Activation Steering}{stolfo2025steering};
\taxpaper{Nudging}{fei2025nudging};
\taxpaper{DARWIN}{hung2025darwin};
\taxpaper{S$^2$R}{ma2025s2r};
\taxpaper{FineSteer}{weng2026finesteer};
\taxpaper{PID Steering}{nguyen2026pidsteering};
\taxpaper{ReQueR}{zhou2026requer}.
};

\node[
    box,
    fill=categorygreen,
    text width=106pt,
    minimum width=112pt,
    minimum height=29pt
] (steeringSearchRefinement)
at (277,0 |- refSteeringSearchRefinement.center)
{Steering, Search \& Refinement};


\node[refsDense,anchor=north]
(refHumanControl)
at ([yshift=-\RowGap]refSteeringSearchRefinement.south) {
\taxpaper{AI Control}{greenblatt2024control};
\taxpaper{STaR-GATE}{andukuri2024stargate};
\taxpaper{Agent-Gated Shared Autonomy}{xue2025agsa};
\taxpaper{Off-Switch Game}{garber2025offswitch};
\taxpaper{Authenticated Delegation}{south2025delegation};
\taxpaper{ARIA}{he2025aria};
\taxpaper{Structured Clarification}{suri2026clarification};
\taxpaper{Meaningful Human Oversight}{zhu2026oversight}.
};

\node[
    box,
    fill=categorygreen,
    text width=106pt,
    minimum width=112pt,
    minimum height=29pt
] (humanControl)
at (277,0 |- refHumanControl.center)
{Human Control};

\coordinate (inferenceAlignmentMid)
at ($(steeringSearchRefinement.center)!0.5!(humanControl.center)$);

\node[
    box,
    fill=categoryblue,
    text width=82pt,
    minimum width=88pt,
    minimum height=29pt
] (inferenceAlignment)
at (163,0 |- inferenceAlignmentMid)
{Inference-Time Alignment};

\coordinate (mechanismsMid)
at ($(trainingAlignment.center)!0.5!(inferenceAlignment.center)$);

\node[
    box,
    fill=categoryyellow,
    text width=64pt,
    minimum width=70pt,
    minimum height=29pt
] (pillarMechanisms)
at (70,0 |- mechanismsMid)
{\textbf{Alignment\\Mechanisms}};


% =========================================================
% IV. ALIGNMENT ASSURANCE
% =========================================================

\node[refsDense,anchor=north]
(refBehavioralEvaluator)
at ([yshift=-\PillarGap]refHumanControl.south) {
\taxpaper{Chatbot Arena}{chiang2024arena};
\taxpaper{HarmBench}{mazeika2024harmbench};
\taxpaper{WildBench}{lin2025wildbench};
\taxpaper{JudgeBench}{tan2025judgebench};
\taxpaper{PRMBench}{song2025prmbench};
\taxpaper{RewardBench 2}{malik2026rewardbench2};
\taxpaper{IF-RewardBench}{wen2026ifrewardbench};
\taxpaper{LLMs Get Lost}{laban2026lost};
\taxpaper{Multi-Crit}{xiong2026multicrit};
\taxpaper{Plan-RewardBench}{wang2026planrewardbench}.
};

\node[
    box,
    fill=categorygreen,
    text width=106pt,
    minimum width=112pt,
    minimum height=29pt
] (behavioralEvaluator)
at (277,0 |- refBehavioralEvaluator.center)
{Behavioral \& Evaluator Evaluation};


\node[refsDense,anchor=north]
(refAdversarialDistribution)
at ([yshift=-\RowGap]refBehavioralEvaluator.south) {
\taxpaper{HarmBench}{mazeika2024harmbench};
\taxpaper{Circuit Breakers}{zou2024circuitbreakers};
\taxpaper{Fine-Tuning Compromises Safety}{qi2024finetuning};
\taxpaper{BackdoorAlign}{wang2024backdooralign};
\taxpaper{Simple Adaptive Attacks}{andriushchenko2025adaptive};
\taxpaper{Jailbreak Antidote}{shen2025antidote};
\taxpaper{Lifelong Safety Alignment}{wang2025lifelong};
\taxpaper{Into the Gray Zone}{hung2026grayzone};
\taxpaper{Safety Alignment Fails to Generalize}{long2026languagegames};
\taxpaper{SafetyMem}{wang2026safetymem}.
};

\node[
    box,
    fill=categorygreen,
    text width=106pt,
    minimum width=112pt,
    minimum height=29pt
] (adversarialDistribution)
at (277,0 |- refAdversarialDistribution.center)
{Adversarial \& Distribution Robustness};


\node[refsDense,anchor=north]
(refAlignmentPreservation)
at ([yshift=-\RowGap]refAdversarialDistribution.south) {
\taxpaper{Keeping LLMs Aligned}{lyu2024keeping};
\taxpaper{Vaccine}{huang2024vaccine};
\taxpaper{Lisa}{huang2024lisa};
\taxpaper{SaLoRA}{li2025salora};
\taxpaper{RSN-Tune}{zhao2025safetyneurons};
\taxpaper{LARF}{li2025larf};
\taxpaper{COPR}{zhang2025copr};
\taxpaper{SOT}{wang2026sot};
\taxpaper{SafeMERGE}{djuhera2026safemerge};
\taxpaper{Continual Safety Alignment}{bach2026continual}.
};

\node[
    box,
    fill=categorygreen,
    text width=106pt,
    minimum width=112pt,
    minimum height=29pt
] (alignmentPreservation)
at (277,0 |- refAlignmentPreservation.center)
{Alignment Preservation};

\coordinate (evaluationRobustnessMid)
at ($(behavioralEvaluator.center)!0.5!(alignmentPreservation.center)$);

\node[
    box,
    fill=categoryblue,
    text width=82pt,
    minimum width=88pt,
    minimum height=29pt
] (evaluationRobustness)
at (163,0 |- evaluationRobustnessMid)
{Evaluation \& Robustness};


\node[refsDense,anchor=north]
(refMechanisticTheoretical)
at ([yshift=-\GroupGap]refAlignmentPreservation.south) {
\taxpaper{Fundamental Limits}{wolf2024limitations};
\taxpaper{Learning Dynamics}{im2024dynamics};
\taxpaper{Refusal Direction}{arditi2024refusal};
\taxpaper{Makes \& Breaks Safety FT}{jain2024safetyfinetuning};
\taxpaper{RM Overoptimization Laws}{rafailov2024overoptimization};
\taxpaper{Safety Neurons}{chen2025safetyneurons};
\taxpaper{Template-Anchored Safety}{leong2025template};
\taxpaper{Self-Jailbreaking}{yong2026selfjailbreaking};
\taxpaper{Intrinsic Alignment Barriers}{nayebi2026barriers};
\taxpaper{The Alignment Game}{falahati2026alignmentgame}.
};

\node[
    box,
    fill=categorygreen,
    text width=106pt,
    minimum width=112pt,
    minimum height=29pt
] (mechanisticTheoretical)
at (277,0 |- refMechanisticTheoretical.center)
{Mechanistic \& Theoretical Evidence};


\node[refsDense,anchor=north]
(refMonitoringAuditing)
at ([yshift=-\RowGap]refMechanisticTheoretical.south) {
\taxpaper{Black-Box Audits}{casper2024blackbox};
\taxpaper{Test-Set Contamination}{oren2024contamination};
\taxpaper{AI Control}{greenblatt2024control};
\taxpaper{Behavioral Shift Auditing}{richter2025auditing};
\taxpaper{CoT Red-Handed}{arnav2025cot};
\taxpaper{High-Stakes Probes}{mckenzie2025highstakes};
\taxpaper{Streaming Monitor}{li2025streaming};
\taxpaper{Watch the Weights}{zhong2026weights};
\taxpaper{Strategic Dishonesty}{panfilov2026dishonesty};
\taxpaper{AgenticEval}{wang2026agenticeval}.
};

\node[
    box,
    fill=categorygreen,
    text width=106pt,
    minimum width=112pt,
    minimum height=29pt
] (monitoringAuditing)
at (277,0 |- refMonitoringAuditing.center)
{Monitoring \& Auditing};

\coordinate (systemAssuranceMid)
at ($(mechanisticTheoretical.center)!0.5!(monitoringAuditing.center)$);

\node[
    box,
    fill=categoryblue,
    text width=82pt,
    minimum width=88pt,
    minimum height=29pt
] (systemAssurance)
at (163,0 |- systemAssuranceMid)
{System Assurance};

\coordinate (assuranceMid)
at ($(evaluationRobustness.center)!0.5!(systemAssurance.center)$);

\node[
    box,
    fill=categoryyellow,
    text width=64pt,
    minimum width=70pt,
    minimum height=29pt
] (pillarAssurance)
at (70,0 |- assuranceMid)
{\textbf{Alignment\\Assurance}};


% =========================================================
% ROOT
% =========================================================

\coordinate (rootMid)
at ($(pillarSpecification.center)!0.5!(pillarAssurance.center)$);

\node[
    box,
    fill=categoryyellow,
    rotate=90,
    minimum height=29pt,
    font=\bfseries\fontsize{10}{11}\selectfont
] (humanAIAlignment)
at (5,0 |- rootMid)
{Human--AI Alignment};


% =========================================================
% CONNECTOR MACRO
% =========================================================

\def\Connect#1#2#3#4{%
    \coordinate (route) at ($(#1.east)!0.5!(#2.west)$);
    \draw[branch]
        (#1.east)
        --
        (route |- #1.east);
    \draw[branch]
        (route |- #2.west)
        --
        (route |- #3.west);
    \foreach \child in {#4}
        \draw[branch]
            (route |- \child.west)
            --
            (\child.west);
}


% =========================================================
% CONNECTORS
% =========================================================

\begin{scope}[on background layer]

% Root -> Pillars
\coordinate (rootRoute) at ($(humanAIAlignment.south)!0.5!(pillarSpecification.west)$);
\draw[branch] (humanAIAlignment.south) -- (rootRoute |- humanAIAlignment.south);
\draw[branch] (rootRoute |- pillarSpecification.west) -- (rootRoute |- pillarAssurance.west);

\foreach \pillar in {pillarSpecification,pillarSupervision,pillarMechanisms,pillarAssurance}{
    \draw[branch] (rootRoute |- \pillar.west) -- (\pillar.west);
}

% =========================================================
% I. ALIGNMENT SPECIFICATION
% =========================================================

\Connect{pillarSpecification}{alignmentObjectives}{valuesStakeholders}{alignmentObjectives,valuesStakeholders}

\Connect{alignmentObjectives}{taskAssistance}{safetyAlignment}{taskAssistance,safetyAlignment}

\draw[branch] (taskAssistance.east) -- (refTaskAssistance.west);
\draw[branch] (safetyAlignment.east) -- (refSafetyAlignment.west);

\Connect{valuesStakeholders}{personalizedAlignment}{uncertaintyDrift}{personalizedAlignment,pluralisticSocietal,uncertaintyDrift}

\draw[branch] (personalizedAlignment.east) -- (refPersonalizedAlignment.west);
\draw[branch] (pluralisticSocietal.east) -- (refPluralisticSocietal.west);
\draw[branch] (uncertaintyDrift.east) -- (refUncertaintyDrift.west);

% =========================================================
% II. ALIGNMENT SUPERVISION
% =========================================================

\Connect{pillarSupervision}{feedbackSource}{feedbackOversight}{feedbackSource,feedbackOversight}

\Connect{feedbackSource}{humanFeedback}{programmaticVerifiable}{humanFeedback,aiFeedback,programmaticVerifiable}

\draw[branch] (humanFeedback.east) -- (refHumanFeedback.west);
\draw[branch] (aiFeedback.east) -- (refAIFeedback.west);
\draw[branch] (programmaticVerifiable.east) -- (refProgrammaticVerifiable.west);

\Connect{feedbackOversight}{demonstrationsPreferences}{reliableScalable}{demonstrationsPreferences,critiqueProcess,reliableScalable}

\draw[branch] (demonstrationsPreferences.east) -- (refDemonstrationsPreferences.west);
\draw[branch] (critiqueProcess.east) -- (refCritiqueProcess.west);
\draw[branch] (reliableScalable.east) -- (refReliableScalable.west);

% =========================================================
% III. ALIGNMENT MECHANISMS
% =========================================================

\Connect{pillarMechanisms}{trainingAlignment}{inferenceAlignment}{trainingAlignment,inferenceAlignment}

\Connect{trainingAlignment}{rewardVerifier}{alignmentDistillation}{rewardVerifier,supervisedAlignment,preferenceOptimization,reinforcementLearning,alignmentDistillation}

\draw[branch] (rewardVerifier.east) -- (refRewardVerifier.west);
\draw[branch] (supervisedAlignment.east) -- (refSupervisedAlignment.west);
\draw[branch] (preferenceOptimization.east) -- (refPreferenceOptimization.west);
\draw[branch] (reinforcementLearning.east) -- (refReinforcementLearning.west);
\draw[branch] (alignmentDistillation.east) -- (refAlignmentDistillation.west);

\Connect{inferenceAlignment}{steeringSearchRefinement}{humanControl}{steeringSearchRefinement,humanControl}

\draw[branch] (steeringSearchRefinement.east) -- (refSteeringSearchRefinement.west);
\draw[branch] (humanControl.east) -- (refHumanControl.west);

% =========================================================
% IV. ALIGNMENT ASSURANCE
% =========================================================

\Connect{pillarAssurance}{evaluationRobustness}{systemAssurance}{evaluationRobustness,systemAssurance}

\Connect{evaluationRobustness}{behavioralEvaluator}{alignmentPreservation}{behavioralEvaluator,adversarialDistribution,alignmentPreservation}

\draw[branch] (behavioralEvaluator.east) -- (refBehavioralEvaluator.west);
\draw[branch] (adversarialDistribution.east) -- (refAdversarialDistribution.west);
\draw[branch] (alignmentPreservation.east) -- (refAlignmentPreservation.west);

\Connect{systemAssurance}{mechanisticTheoretical}{monitoringAuditing}{mechanisticTheoretical,monitoringAuditing}

\draw[branch] (mechanisticTheoretical.east) -- (refMechanisticTheoretical.west);
\draw[branch] (monitoringAuditing.east) -- (refMonitoringAuditing.west);

\end{scope}

\end{tikzpicture}

\end{document}
"""

# =========================================================
# AUTHOR--YEAR CITATION CONVERSION
# =========================================================
#
# Giữ nguyên các dòng taxonomy ở dạng:
#
#   \taxpaper{HelpSteer2}{wang2024helpsteer2}
#
# nhưng trước khi ghi Taxonomy.tex, Python sẽ đổi thành:
#
#   \taxpaper{HelpSteer2}{Wang et al., 2024}
#
# Vì vậy PDF/PNG sẽ hiển thị:
#
#   HelpSteer2 [Wang et al., 2024]
#
# Không cần BibTeX/Biber và không cần sửa thủ công hàng trăm node.
# =========================================================

# Một số họ/tổ chức không thể viết đúng chỉ bằng .capitalize().
AUTHOR_NAME_OVERRIDES = {
    "deepseek": "DeepSeek-AI",
    "doosterlinck": "D'Oosterlinck",
    "mckenzie": "McKenzie",
}

def bibkey_to_author_year(bibkey: str) -> str:
    """
    Chuyển bibkey kiểu:
        wang2024helpsteer2
        zhang2026biprm
        deepseek2025r1
    thành:
        Wang et al., 2024
        Zhang et al., 2026
        DeepSeek-AI et al., 2025

    Quy ước:
    - Phần trước năm được xem là họ của first author.
    - Năm lấy trực tiếp từ bibkey.
    - Dùng 'et al.' vì các paper trong taxonomy hầu hết là multi-author.
    """

    match = re.match(r"^([A-Za-z]+)((?:19|20)\d{2})(.*)$", bibkey)

    if match is None:
        # Nếu gặp key không theo cấu trúc authorYYYY..., giữ nguyên key
        # thay vì làm hỏng quá trình compile.
        print(
            f"[WARNING] Không suy ra được author/year từ bibkey: {bibkey}. "
            "Giữ nguyên bibkey này."
        )
        return bibkey

    author_key = match.group(1).lower()
    year = match.group(2)

    if author_key in AUTHOR_NAME_OVERRIDES:
        author = AUTHOR_NAME_OVERRIDES[author_key]
    else:
        author = author_key[0].upper() + author_key[1:]

    return f"{author} et al., {year}"


def convert_taxpaper_citations(source: str) -> str:
    """
    Chỉ thay đối số thứ hai của \\taxpaper{title}{bibkey}.

    Ví dụ:
        \\taxpaper{HelpSteer2}{wang2024helpsteer2}
    ->
        \\taxpaper{HelpSteer2}{Wang et al., 2024}

    Phần title và toàn bộ cấu trúc TikZ được giữ nguyên.
    """

    pattern = re.compile(
        r"(\\taxpaper\{(?:[^{}]|\{[^{}]*\})*\})\{([^{}]+)\}"
    )

    def replace(match: re.Match) -> str:
        paper_part = match.group(1)
        bibkey = match.group(2).strip()
        citation = bibkey_to_author_year(bibkey)
        return f"{paper_part}{{{citation}}}"

    converted, count = pattern.subn(replace, source)

    print(f"Đã chuyển {count} citation sang dạng Author et al., Year.")

    return converted


# Tạo bản LaTeX dùng để xuất.
# Biến latex_code gốc vẫn giữ nguyên bibkey để dễ chỉnh taxonomy.
rendered_latex_code = convert_taxpaper_citations(latex_code)


# =========================================================
# OUTPUT SETTINGS
# =========================================================

OUTPUT_NAME = "Taxonomy"
PNG_DPI = 300

# Nếu convert.py nằm trong folder Figure:
#   output sẽ nằm ngay trong Figure.
#
# Nếu convert.py nằm ở project root:
#   tự tạo folder Figure.
SCRIPT_DIR = Path(__file__).resolve().parent

if SCRIPT_DIR.name.lower() == "figure":
    OUTPUT_DIR = SCRIPT_DIR
else:
    OUTPUT_DIR = SCRIPT_DIR / "Figure"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

tex_path = OUTPUT_DIR / f"{OUTPUT_NAME}.tex"
pdf_path = OUTPUT_DIR / f"{OUTPUT_NAME}.pdf"
png_path = OUTPUT_DIR / f"{OUTPUT_NAME}.png"


# =========================================================
# HELPER
# =========================================================

def run_command(command, cwd=None):
    """
    Chạy command và hiện đầy đủ output nếu xảy ra lỗi.
    """
    print()
    print("> " + " ".join(str(x) for x in command))

    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace"
    )

    print(result.stdout)

    if result.returncode != 0:
        raise RuntimeError(
            f"Lệnh thất bại với exit code {result.returncode}:\n"
            + " ".join(str(x) for x in command)
        )

    return result


# =========================================================
# CHECK REQUIRED PROGRAMS
# =========================================================

pdflatex = shutil.which("pdflatex")

if pdflatex is None:
    raise RuntimeError(
        "\nKhông tìm thấy pdflatex.\n"
        "Hãy cài MiKTeX hoặc TeX Live và đảm bảo pdflatex có trong PATH."
    )


# Ưu tiên pdftoppm, nếu không có thì thử pdftocairo.
pdftoppm = shutil.which("pdftoppm")
pdftocairo = shutil.which("pdftocairo")

if pdftoppm is None and pdftocairo is None:
    raise RuntimeError(
        "\nKhông tìm thấy pdftoppm hoặc pdftocairo.\n"
        "Hãy cài Poppler và thêm thư mục bin của Poppler vào PATH."
    )


# =========================================================
# REMOVE OLD OUTPUT
# =========================================================

for old_file in [pdf_path, png_path]:
    if old_file.exists():
        try:
            old_file.unlink()
        except PermissionError:
            raise RuntimeError(
                f"\nKhông thể ghi đè file:\n{old_file}\n\n"
                "Hãy đóng file nếu nó đang được mở bằng Acrobat, "
                "Edge, Chrome hoặc phần mềm xem ảnh."
            )


# =========================================================
# WRITE STANDALONE LATEX
# =========================================================

tex_path.write_text(
    rendered_latex_code.strip() + "\n",
    encoding="utf-8"
)

print("=" * 70)
print("ĐÃ TẠO FILE LATEX")
print("=" * 70)
print(tex_path)


# =========================================================
# COMPILE LATEX -> PDF
# =========================================================
#
# Rất quan trọng:
# Compile trong OUTPUT_DIR và chỉ truyền tên Taxonomy.tex.
# Cách này tránh lỗi đường dẫn trên Windows và lỗi Figure/Figure.
# =========================================================

print()
print("=" * 70)
print("BIÊN DỊCH LATEX -> PDF")
print("=" * 70)

run_command(
    [
        pdflatex,
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        tex_path.name
    ],
    cwd=OUTPUT_DIR
)


if not pdf_path.exists():
    raise RuntimeError(
        f"\npdflatex chạy xong nhưng không tìm thấy PDF:\n{pdf_path}"
    )

print()
print(f"PDF OK: {pdf_path}")


# =========================================================
# PDF -> PNG
# =========================================================

print()
print("=" * 70)
print(f"CHUYỂN PDF -> PNG ({PNG_DPI} DPI)")
print("=" * 70)

# pdftoppm / pdftocairo tự thêm .png.
# Vì vậy chỉ truyền prefix "Taxonomy".
png_prefix = OUTPUT_DIR / OUTPUT_NAME

if pdftoppm is not None:

    run_command(
        [
            pdftoppm,
            "-png",
            "-singlefile",
            "-r",
            str(PNG_DPI),
            pdf_path.name,
            png_prefix.name
        ],
        cwd=OUTPUT_DIR
    )

else:

    run_command(
        [
            pdftocairo,
            "-png",
            "-singlefile",
            "-r",
            str(PNG_DPI),
            pdf_path.name,
            png_prefix.name
        ],
        cwd=OUTPUT_DIR
    )


if not png_path.exists():
    raise RuntimeError(
        f"\nKhông tìm thấy PNG sau khi chuyển đổi:\n{png_path}"
    )

print()
print(f"PNG OK: {png_path}")


# =========================================================
# REMOVE TEMPORARY LATEX FILES
# =========================================================

temporary_extensions = [
    ".aux",
    ".log",
    ".out",
    ".toc",
    ".fls",
    ".fdb_latexmk",
    ".synctex.gz"
]

for extension in temporary_extensions:
    temp_file = OUTPUT_DIR / f"{OUTPUT_NAME}{extension}"

    if temp_file.exists():
        try:
            temp_file.unlink()
        except PermissionError:
            pass


# =========================================================
# FINISHED
# =========================================================

print()
print("=" * 70)
print("XUẤT TAXONOMY HOÀN TẤT")
print("=" * 70)
print(f"TEX : {tex_path}")
print(f"PDF : {pdf_path}")
print(f"PNG : {png_path}")
print(f"DPI : {PNG_DPI}")
print("=" * 70)