import os
import subprocess

# Mã LaTeX đã được làm sạch, bọc bằng standalone
latex_code = r"""
\documentclass[margin=5pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{calc,backgrounds}
\usepackage{xcolor}

\definecolor{categoryyellow}{RGB}{255,255,204}
\definecolor{categoryblue}{RGB}{207,226,251}
\definecolor{categorygreen}{RGB}{226,240,220}
\definecolor{referencegray}{RGB}{248,248,248}
\definecolor{referenceblue}{RGB}{0,0,128}

\newcommand{\taxpaper}[2]{#1~{\color{referenceblue}[#2]}}

\begin{document}
\begin{tikzpicture}[x=1pt,y=-1pt,box/.style={draw=black!70,line width=.55pt,rounded corners=2pt,inner xsep=3pt,inner ysep=3pt,outer sep=0pt,align=center,font=\fontsize{9}{10.5}\selectfont},refs/.style={box,fill=referencegray,align=left,text width=317pt,minimum width=323pt,minimum height=29pt,font=\fontsize{8}{9.5}\selectfont},refsDense/.style={refs,font=\fontsize{7.2}{8.4}\selectfont},branch/.style={draw=black!60,line width=.55pt,line cap=round,line join=round}]
\def\RowGap{3pt}\def\GroupGap{7pt}\def\PillarGap{12pt}

% =========================================================
% I. ALIGNMENT SPECIFICATION
% =========================================================
\node[refs,anchor=north] (refTaskAssistance) at (506.5,0) {\taxpaper{HelpSteer2}{wang2024helpsteer2}; \taxpaper{UltraFeedback}{cui2024ultrafeedback}; \taxpaper{Scaling Instruction-Finetuned LMs}{chung2024flan}; \taxpaper{InstructGPT}{ouyang2022instructgpt}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (taskAssistance) at (277,0 |- refTaskAssistance.center) {Task \& Assistance Alignment};
\node[refs,anchor=north] (refSafetyAlignment) at ([yshift=-\RowGap]refTaskAssistance.south) {\taxpaper{Lifelong Safety Alignment}{wang2025lifelong}; \taxpaper{Circuit Breakers}{zou2024circuitbreakers}; \taxpaper{Rule-Based Rewards}{mu2024rulebased}; \taxpaper{Safe RLHF}{dai2024saferlhf}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (safetyAlignment) at (277,0 |- refSafetyAlignment.center) {Safety Alignment};
\coordinate (alignmentObjectivesMid) at ($(taskAssistance.center)!0.5!(safetyAlignment.center)$);
\node[box,fill=categoryblue,text width=82pt,minimum width=88pt,minimum height=29pt] (alignmentObjectives) at (163,0 |- alignmentObjectivesMid) {Alignment Objectives};

\node[refs,anchor=north] (refPersonalizedAlignment) at ([yshift=-\GroupGap]refSafetyAlignment.south) {\taxpaper{Persona-Judge}{zhang2025personajudge}; \taxpaper{Personalized Alignment Survey}{guan2025personalizedsurvey}; \taxpaper{Personalizing RLHF}{poddar2024personalizing}; \taxpaper{Personalized Language Modeling}{li2024personalized}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (personalizedAlignment) at (277,0 |- refPersonalizedAlignment.center) {Personalized Alignment};
\node[refs,anchor=north] (refPluralisticSocietal) at ([yshift=-\RowGap]refPersonalizedAlignment.south) {\taxpaper{Steerable Pluralism}{adams2025steerable}; \taxpaper{Collective Constitutional AI}{huang2024collective}; \taxpaper{PRISM}{kirk2024prism}; \taxpaper{Modular Pluralism}{feng2024modular}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (pluralisticSocietal) at (277,0 |- refPluralisticSocietal.center) {Pluralistic \& Societal Alignment};
\node[refs,anchor=north] (refUncertaintyDrift) at ([yshift=-\RowGap]refPluralisticSocietal.south) {\taxpaper{COPR}{zhang2025copr}; \taxpaper{Lifelong Safety Alignment}{wang2025lifelong}; \taxpaper{Distributional Preference Learning}{siththaranjan2024distributional}; \taxpaper{Inverse Reward Design}{hadfieldmenell2017ird}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (uncertaintyDrift) at (277,0 |- refUncertaintyDrift.center) {Uncertainty \& Drift};
\coordinate (valuesStakeholdersMid) at ($(personalizedAlignment.center)!0.5!(uncertaintyDrift.center)$);
\node[box,fill=categoryblue,text width=82pt,minimum width=88pt,minimum height=29pt] (valuesStakeholders) at (163,0 |- valuesStakeholdersMid) {Values \& Stakeholders};
\coordinate (specificationMid) at ($(alignmentObjectives.center)!0.5!(valuesStakeholders.center)$);
\node[box,fill=categoryyellow,text width=64pt,minimum width=70pt,minimum height=29pt] (pillarSpecification) at (70,0 |- specificationMid) {\textbf{Alignment\\Specification}};

% =========================================================
% II. ALIGNMENT SUPERVISION
% =========================================================
\node[refs,anchor=north] (refHumanFeedback) at ([yshift=-\PillarGap]refUncertaintyDrift.south) {\taxpaper{MM-RLHF}{zhang2025mmrlhf}; \taxpaper{RLHF-V}{yu2024rlhfv}; \taxpaper{HelpSteer2}{wang2024helpsteer2}; \taxpaper{Fine-Grained Human Feedback}{wu2023finegrained}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (humanFeedback) at (277,0 |- refHumanFeedback.center) {Human Feedback};
\node[refs,anchor=north] (refAIFeedback) at ([yshift=-\RowGap]refHumanFeedback.south) {\taxpaper{VLFeedback}{li2024vlfeedback}; \taxpaper{UltraFeedback}{cui2024ultrafeedback}; \taxpaper{RLAIF}{lee2024rlaif}; \taxpaper{Self-Rewarding LMs}{yuan2024selfrewarding}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (aiFeedback) at (277,0 |- refAIFeedback.center) {AI Feedback};
\node[refs,anchor=north] (refProgrammaticVerifiable) at ([yshift=-\RowGap]refAIFeedback.south) {\taxpaper{DeepSeek-R1}{deepseek2025r1}; \taxpaper{Math-Shepherd}{wang2024mathshepherd}; \taxpaper{Rule-Based Rewards}{mu2024rulebased}; \taxpaper{DeepSeekMath}{shao2024deepseekmath}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (programmaticVerifiable) at (277,0 |- refProgrammaticVerifiable.center) {Programmatic \& Verifiable Feedback};
\coordinate (feedbackSourceMid) at ($(humanFeedback.center)!0.5!(programmaticVerifiable.center)$);
\node[box,fill=categoryblue,text width=82pt,minimum width=88pt,minimum height=29pt] (feedbackSource) at (163,0 |- feedbackSourceMid) {Feedback Source};

\node[refs,anchor=north] (refDemonstrationsPreferences) at ([yshift=-\GroupGap]refProgrammaticVerifiable.south) {\taxpaper{HelpSteer2}{wang2024helpsteer2}; \taxpaper{UltraFeedback}{cui2024ultrafeedback}; \taxpaper{RLHF-V}{yu2024rlhfv}; \taxpaper{Self-Instruct}{wang2023selfinstruct}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (demonstrationsPreferences) at (277,0 |- refDemonstrationsPreferences.center) {Demonstrations \& Preferences};
\node[refs,anchor=north] (refCritiqueProcess) at ([yshift=-\RowGap]refDemonstrationsPreferences.south) {\taxpaper{LLaVA-Critic}{xiong2025llavacritic}; \taxpaper{Let's Verify Step by Step}{lightman2024verify}; \taxpaper{Math-Shepherd}{wang2024mathshepherd}; \taxpaper{CRITIC}{gou2024critic}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (critiqueProcess) at (277,0 |- refCritiqueProcess.center) {Critique \& Process Feedback};
\node[refs,anchor=north] (refReliableScalable) at ([yshift=-\RowGap]refCritiqueProcess.south) {\taxpaper{Weak-to-Strong}{burns2024weakstrong}; \taxpaper{AI Control}{greenblatt2024control}; \taxpaper{LLM Debate}{khan2024debate}; \taxpaper{Prover-Verifier Games}{kirchner2024proververifier}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (reliableScalable) at (277,0 |- refReliableScalable.center) {Reliable \& Scalable Oversight};
\coordinate (feedbackOversightMid) at ($(demonstrationsPreferences.center)!0.5!(reliableScalable.center)$);
\node[box,fill=categoryblue,text width=82pt,minimum width=88pt,minimum height=29pt] (feedbackOversight) at (163,0 |- feedbackOversightMid) {Feedback \& Oversight};
\coordinate (supervisionMid) at ($(feedbackSource.center)!0.5!(feedbackOversight.center)$);
\node[box,fill=categoryyellow,text width=64pt,minimum width=70pt,minimum height=29pt] (pillarSupervision) at (70,0 |- supervisionMid) {\textbf{Alignment\\Supervision}};

% =========================================================
% III. ALIGNMENT MECHANISMS
% =========================================================
\node[refs,anchor=north] (refRewardVerifier) at ([yshift=-\PillarGap]refReliableScalable.south) {\taxpaper{BiPRM}{zhang2026biprm}; \taxpaper{LLaVA-Critic}{xiong2025llavacritic}; \taxpaper{PRMBench}{song2025prmbench}; \taxpaper{RewardBench}{lambert2025rewardbench}; \taxpaper{HelpSteer2}{wang2024helpsteer2}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (rewardVerifier) at (277,0 |- refRewardVerifier.center) {Reward \& Verifier Modeling};
\node[refs,anchor=north] (refSupervisedAlignment) at ([yshift=-\RowGap]refRewardVerifier.south) {\taxpaper{Scaling Instruction-Finetuned LMs}{chung2024flan}; \taxpaper{SteerLM}{dong2023steerlm}; \taxpaper{LIMA}{zhou2023lima}; \taxpaper{Self-Instruct}{wang2023selfinstruct}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (supervisedAlignment) at (277,0 |- refSupervisedAlignment.center) {Supervised Alignment};
\node[refsDense,anchor=north] (refPreferenceOptimization) at ([yshift=-\RowGap]refSupervisedAlignment.south) {\taxpaper{SPPO}{wu2025sppo}; \taxpaper{SDPO}{kong2025sdpo}; \taxpaper{MMedPO}{zhu2025mmedpo}; \taxpaper{KTO}{ethayarajh2024kto}; \taxpaper{ORPO}{hong2024orpo}; \taxpaper{SimPO}{meng2024simpo}; \taxpaper{TDPO}{zeng2024tdpo}; \taxpaper{Direct Nash Optimization}{rosset2024dno}; \taxpaper{Self-Exploring LMs}{zhang2024selm}; \taxpaper{DPO}{rafailov2023dpo}; \taxpaper{IPO/$\Psi$PO}{azar2023psipo}; \taxpaper{HA-DPO}{zhao2023hadpo}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (preferenceOptimization) at (277,0 |- refPreferenceOptimization.center) {Preference Optimization};
\node[refs,anchor=north] (refReinforcementLearning) at ([yshift=-\RowGap]refPreferenceOptimization.south) {\taxpaper{DeepSeek-R1}{deepseek2025r1}; \taxpaper{MM-RLHF}{zhang2025mmrlhf}; \taxpaper{Safe RLHF}{dai2024saferlhf}; \taxpaper{RLAIF}{lee2024rlaif}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (reinforcementLearning) at (277,0 |- refReinforcementLearning.center) {Reinforcement Learning};
\node[refs,anchor=north] (refAlignmentDistillation) at ([yshift=-\RowGap]refReinforcementLearning.south) {\taxpaper{AlignDistil}{zhang2025aligndistil}; \taxpaper{PAD}{gu2025pad}; \taxpaper{ADPA}{gao2025adpa}; \taxpaper{DPKD}{li2024dpkd}; \taxpaper{PLaD}{zhang2024plad}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (alignmentDistillation) at (277,0 |- refAlignmentDistillation.center) {Alignment Distillation};
\coordinate (trainingAlignmentMid) at ($(rewardVerifier.center)!0.5!(alignmentDistillation.center)$);
\node[box,fill=categoryblue,text width=82pt,minimum width=88pt,minimum height=29pt] (trainingAlignment) at (163,0 |- trainingAlignmentMid) {Training-Time Alignment};

\node[refs,anchor=north] (refSteeringSearchRefinement) at ([yshift=-\GroupGap]refAlignmentDistillation.south) {\taxpaper{LLaVA-Critic}{xiong2025llavacritic}; \taxpaper{CRITIC}{gou2024critic}; \taxpaper{Contrastive Activation Addition}{rimsky2024caa}; \taxpaper{Self-Refine}{madaan2023selfrefine}; \taxpaper{Inference-Time Intervention}{li2023iti}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (steeringSearchRefinement) at (277,0 |- refSteeringSearchRefinement.center) {Steering, Search \& Refinement};
\node[refs,anchor=north] (refHumanControl) at ([yshift=-\RowGap]refSteeringSearchRefinement.south) {\taxpaper{Corrigibility Transformation}{hudson2025corrigibility}; \taxpaper{AI Control}{greenblatt2024control}; \taxpaper{STaR-GATE}{andukuri2024stargate}; \taxpaper{Learning to Defer}{mozannar2020defer}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (humanControl) at (277,0 |- refHumanControl.center) {Human Control};
\coordinate (inferenceAlignmentMid) at ($(steeringSearchRefinement.center)!0.5!(humanControl.center)$);
\node[box,fill=categoryblue,text width=82pt,minimum width=88pt,minimum height=29pt] (inferenceAlignment) at (163,0 |- inferenceAlignmentMid) {Inference-Time Alignment};
\coordinate (mechanismsMid) at ($(trainingAlignment.center)!0.5!(inferenceAlignment.center)$);
\node[box,fill=categoryyellow,text width=64pt,minimum width=70pt,minimum height=29pt] (pillarMechanisms) at (70,0 |- mechanismsMid) {\textbf{Alignment\\Mechanisms}};

% =========================================================
% IV. ALIGNMENT ASSURANCE
% =========================================================
\node[refs,anchor=north] (refBehavioralEvaluator) at ([yshift=-\PillarGap]refHumanControl.south) {\taxpaper{PRMBench}{song2025prmbench}; \taxpaper{RewardBench}{lambert2025rewardbench}; \taxpaper{SafetyBench}{zhang2024safetybench}; \taxpaper{HarmBench}{mazeika2024harmbench}; \taxpaper{HallusionBench}{guan2024hallusionbench}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (behavioralEvaluator) at (277,0 |- refBehavioralEvaluator.center) {Behavioral \& Evaluator Evaluation};
\node[refs,anchor=north] (refAdversarialDistribution) at ([yshift=-\RowGap]refBehavioralEvaluator.south) {\taxpaper{When Context Flips, Safety Breaks}{choi2026context}; \taxpaper{StrongREJECT}{souly2024strongreject}; \taxpaper{Circuit Breakers}{zou2024circuitbreakers}; \taxpaper{HarmBench}{mazeika2024harmbench}; \taxpaper{XSTest}{rottger2024xstest}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (adversarialDistribution) at (277,0 |- refAdversarialDistribution.center) {Adversarial \& Distribution Robustness};
\node[refs,anchor=north] (refAlignmentPreservation) at ([yshift=-\RowGap]refAdversarialDistribution.south) {\taxpaper{Alignment Midtraining}{brazilek2026alignment}; \taxpaper{Lifelong Safety Alignment}{wang2025lifelong}; \taxpaper{LARF}{li2025larf}; \taxpaper{COPR}{zhang2025copr}; \taxpaper{Fine-Tuning Aligned LMs Compromises Safety}{qi2024finetuning}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (alignmentPreservation) at (277,0 |- refAlignmentPreservation.center) {Alignment Preservation};
\coordinate (evaluationRobustnessMid) at ($(behavioralEvaluator.center)!0.5!(alignmentPreservation.center)$);
\node[box,fill=categoryblue,text width=82pt,minimum width=88pt,minimum height=29pt] (evaluationRobustness) at (163,0 |- evaluationRobustnessMid) {Evaluation \& Robustness};

\node[refs,anchor=north] (refMechanisticTheoretical) at ([yshift=-\GroupGap]refAlignmentPreservation.south) {\taxpaper{Sustaining AI Safety}{mazzu2026sustaining}; \taxpaper{Corrigibility Transformation}{hudson2025corrigibility}; \taxpaper{Contrastive Activation Addition}{rimsky2024caa}; \taxpaper{Representation Engineering}{zou2023repreng}; \taxpaper{Inference-Time Intervention}{li2023iti}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (mechanisticTheoretical) at (277,0 |- refMechanisticTheoretical.center) {Mechanistic \& Theoretical Evidence};
\node[refs,anchor=north] (refMonitoringAuditing) at ([yshift=-\RowGap]refMechanisticTheoretical.south) {\taxpaper{EvalSafetyGap}{uluirmak2026evalsafetygap}; \taxpaper{RewardBench}{lambert2025rewardbench}; \taxpaper{AI Control}{greenblatt2024control}; \taxpaper{Test-Set Contamination}{oren2024contamination}; \taxpaper{Model-Written Evaluations}{perez2023modelwritten}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (monitoringAuditing) at (277,0 |- refMonitoringAuditing.center) {Monitoring \& Auditing};
\coordinate (systemAssuranceMid) at ($(mechanisticTheoretical.center)!0.5!(monitoringAuditing.center)$);
\node[box,fill=categoryblue,text width=82pt,minimum width=88pt,minimum height=29pt] (systemAssurance) at (163,0 |- systemAssuranceMid) {System Assurance};
\coordinate (assuranceMid) at ($(evaluationRobustness.center)!0.5!(systemAssurance.center)$);
\node[box,fill=categoryyellow,text width=64pt,minimum width=70pt,minimum height=29pt] (pillarAssurance) at (70,0 |- assuranceMid) {\textbf{Alignment\\Assurance}};

% =========================================================
% ROOT AND CONNECTORS
% =========================================================
\coordinate (rootMid) at ($(pillarSpecification.center)!0.5!(pillarAssurance.center)$);
\node[box,fill=categoryyellow,rotate=90,minimum height=29pt,font=\bfseries\fontsize{10}{11}\selectfont] (humanAIAlignment) at (5,0 |- rootMid) {Human--AI Alignment};

\def\Connect#1#2#3#4{\coordinate (route) at ($(#1.east)!0.5!(#2.west)$);\draw[branch] (#1.east) -- (route |- #1.east);\draw[branch] (route |- #2.west) -- (route |- #3.west);\foreach \child in {#4}\draw[branch] (route |- \child.west) -- (\child.west);}

\begin{scope}[on background layer]
\coordinate (rootRoute) at ($(humanAIAlignment.south)!0.5!(pillarSpecification.west)$);
\draw[branch] (humanAIAlignment.south) -- (rootRoute |- humanAIAlignment.south);
\draw[branch] (rootRoute |- pillarSpecification.west) -- (rootRoute |- pillarAssurance.west);
\foreach \pillar in {pillarSpecification,pillarSupervision,pillarMechanisms,pillarAssurance}\draw[branch] (rootRoute |- \pillar.west) -- (\pillar.west);

\Connect{pillarSpecification}{alignmentObjectives}{valuesStakeholders}{alignmentObjectives,valuesStakeholders}
\Connect{alignmentObjectives}{taskAssistance}{safetyAlignment}{taskAssistance,safetyAlignment}
\draw[branch] (taskAssistance.east) -- (refTaskAssistance.west);
\draw[branch] (safetyAlignment.east) -- (refSafetyAlignment.west);
\Connect{valuesStakeholders}{personalizedAlignment}{uncertaintyDrift}{personalizedAlignment,pluralisticSocietal,uncertaintyDrift}
\draw[branch] (personalizedAlignment.east) -- (refPersonalizedAlignment.west);
\draw[branch] (pluralisticSocietal.east) -- (refPluralisticSocietal.west);
\draw[branch] (uncertaintyDrift.east) -- (refUncertaintyDrift.west);

\Connect{pillarSupervision}{feedbackSource}{feedbackOversight}{feedbackSource,feedbackOversight}
\Connect{feedbackSource}{humanFeedback}{programmaticVerifiable}{humanFeedback,aiFeedback,programmaticVerifiable}
\draw[branch] (humanFeedback.east) -- (refHumanFeedback.west);
\draw[branch] (aiFeedback.east) -- (refAIFeedback.west);
\draw[branch] (programmaticVerifiable.east) -- (refProgrammaticVerifiable.west);
\Connect{feedbackOversight}{demonstrationsPreferences}{reliableScalable}{demonstrationsPreferences,critiqueProcess,reliableScalable}
\draw[branch] (demonstrationsPreferences.east) -- (refDemonstrationsPreferences.west);
\draw[branch] (critiqueProcess.east) -- (refCritiqueProcess.west);
\draw[branch] (reliableScalable.east) -- (refReliableScalable.west);

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

with open('Figure.tex', 'w', encoding='utf-8') as f:
    f.write(latex_code.strip())

# Compile to PDF
subprocess.run(['pdflatex', '-interaction=nonstopmode', 'Figure.tex'], check=True)

# Dọn dẹp file thừa (tùy chọn)
for ext in [".aux", ".log"]:
    if os.path.exists(f"Figure{ext}"):
        os.remove(f"Figure{ext}")

print("Quá trình xuất hoàn tất!")