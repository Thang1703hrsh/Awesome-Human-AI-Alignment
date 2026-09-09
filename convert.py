import os
import subprocess
import fitz  # Thư viện PyMuPDF

# Mã LaTeX của bạn
latex_code = r"""
\documentclass[margin=5pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{calc,backgrounds}
\usepackage{xcolor}

% Định nghĩa các màu sắc sử dụng trong biểu đồ
\definecolor{categoryyellow}{RGB}{255,255,204}
\definecolor{categoryblue}{RGB}{207,226,251}
\definecolor{categorygreen}{RGB}{226,240,220}
\definecolor{referencegray}{RGB}{248,248,248}
\definecolor{referenceblue}{RGB}{0,0,128}

% Định nghĩa lại lệnh \taxpaper
\newcommand{\taxpaper}[2]{#1~{\color{referenceblue}[#2]}}

\begin{document}

\begin{tikzpicture}[x=1pt,y=-1pt,box/.style={draw=black!70,line width=.55pt,rounded corners=2pt,inner xsep=3pt,inner ysep=3pt,outer sep=0pt,align=center,font=\fontsize{9}{10.5}\selectfont},refs/.style={box,fill=referencegray,align=left,text width=317pt,minimum width=323pt,minimum height=29pt,font=\fontsize{8}{9.5}\selectfont},branch/.style={draw=black!60,line width=.55pt,line cap=round,line join=round}]
\def\RowGap{3pt}\def\GroupGap{7pt}\def\PillarGap{12pt}

% =========================================================
% I. ALIGNMENT SPECIFICATION
% =========================================================
\node[refs,anchor=north] (refTaskAssistance) at (506.5,0) {\taxpaper{FLAN}{wei2022flan}; \taxpaper{InstructGPT}{ouyang2022instructgpt}; \taxpaper{WebGPT}{nakano2021webgpt}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (taskAssistance) at (277,0 |- refTaskAssistance.center) {Task \& Assistance Alignment};
\node[refs,anchor=north] (refSafetyAlignment) at ([yshift=-\RowGap]refTaskAssistance.south) {\taxpaper{Constitutional AI}{bai2022constitutional}; \taxpaper{Safe RLHF}{dai2024saferlhf}; \taxpaper{Rule-Based Rewards}{mu2024rulebased}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (safetyAlignment) at (277,0 |- refSafetyAlignment.center) {Safety Alignment};
\coordinate (alignmentObjectivesMid) at ($(taskAssistance.center)!0.5!(safetyAlignment.center)$);
\node[box,fill=categoryblue,text width=82pt,minimum width=88pt,minimum height=29pt] (alignmentObjectives) at (163,0 |- alignmentObjectivesMid) {Alignment Objectives};

\node[refs,anchor=north] (refPersonalizedAlignment) at ([yshift=-\GroupGap]refSafetyAlignment.south) {\taxpaper{Personalizing RLHF}{poddar2024personalizing}; \taxpaper{Personalized Soups}{jang2023personalizedsoups}; \taxpaper{Persona-Judge}{zhang2025personajudge}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (personalizedAlignment) at (277,0 |- refPersonalizedAlignment.center) {Personalized Alignment};
\node[refs,anchor=north] (refPluralisticSocietal) at ([yshift=-\RowGap]refPersonalizedAlignment.south) {\taxpaper{Collective Constitutional AI}{huang2024collective}; \taxpaper{PRISM}{kirk2024prism}; \taxpaper{Pluralistic Alignment}{sorensen2024pluralistic}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (pluralisticSocietal) at (277,0 |- refPluralisticSocietal.center) {Pluralistic \& Societal Alignment};
\node[refs,anchor=north] (refUncertaintyDrift) at ([yshift=-\RowGap]refPluralisticSocietal.south) {\taxpaper{Cooperative IRL}{hadfieldmenell2016cirl}; \taxpaper{Inverse Reward Design}{hadfieldmenell2017ird}; \taxpaper{COPR}{zhang2025copr}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (uncertaintyDrift) at (277,0 |- refUncertaintyDrift.center) {Uncertainty \& Drift};
\coordinate (valuesStakeholdersMid) at ($(personalizedAlignment.center)!0.5!(uncertaintyDrift.center)$);
\node[box,fill=categoryblue,text width=82pt,minimum width=88pt,minimum height=29pt] (valuesStakeholders) at (163,0 |- valuesStakeholdersMid) {Values \& Stakeholders};
\coordinate (specificationMid) at ($(alignmentObjectives.center)!0.5!(valuesStakeholders.center)$);
\node[box,fill=categoryyellow,text width=64pt,minimum width=70pt,minimum height=29pt] (pillarSpecification) at (70,0 |- specificationMid) {\textbf{Alignment\\Specification}};

% =========================================================
% II. ALIGNMENT SUPERVISION
% =========================================================
\node[refs,anchor=north] (refHumanFeedback) at ([yshift=-\PillarGap]refUncertaintyDrift.south) {\taxpaper{Deep RL from Human Preferences}{christiano2017preferences}; \taxpaper{Learning to Summarize}{stiennon2020summarize}; \taxpaper{RLHF-V}{yu2024rlhfv}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (humanFeedback) at (277,0 |- refHumanFeedback.center) {Human Feedback};
\node[refs,anchor=north] (refAIFeedback) at ([yshift=-\RowGap]refHumanFeedback.south) {\taxpaper{Constitutional AI}{bai2022constitutional}; \taxpaper{RLAIF}{lee2024rlaif}; \taxpaper{VLFeedback}{li2024vlfeedback}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (aiFeedback) at (277,0 |- refAIFeedback.center) {AI Feedback};
\node[refs,anchor=north] (refProgrammaticVerifiable) at ([yshift=-\RowGap]refAIFeedback.south) {\taxpaper{Rule-Based Rewards}{mu2024rulebased}; \taxpaper{Math-Shepherd}{wang2024mathshepherd}; \taxpaper{DeepSeek-R1}{deepseek2025r1}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (programmaticVerifiable) at (277,0 |- refProgrammaticVerifiable.center) {Programmatic \& Verifiable Feedback};
\coordinate (feedbackSourceMid) at ($(humanFeedback.center)!0.5!(programmaticVerifiable.center)$);
\node[box,fill=categoryblue,text width=82pt,minimum width=88pt,minimum height=29pt] (feedbackSource) at (163,0 |- feedbackSourceMid) {Feedback Source};

\node[refs,anchor=north] (refDemonstrationsPreferences) at ([yshift=-\GroupGap]refProgrammaticVerifiable.south) {\taxpaper{Self-Instruct}{wang2023selfinstruct}; \taxpaper{UltraFeedback}{cui2024ultrafeedback}; \taxpaper{HelpSteer2}{wang2024helpsteer2}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (demonstrationsPreferences) at (277,0 |- refDemonstrationsPreferences.center) {Demonstrations \& Preferences};
\node[refs,anchor=north] (refCritiqueProcess) at ([yshift=-\RowGap]refDemonstrationsPreferences.south) {\taxpaper{Fine-Grained Human Feedback}{wu2023finegrained}; \taxpaper{Let's Verify Step by Step}{lightman2024verify}; \taxpaper{RLHF-V}{yu2024rlhfv}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (critiqueProcess) at (277,0 |- refCritiqueProcess.center) {Critique \& Process Feedback};
\node[refs,anchor=north] (refReliableScalable) at ([yshift=-\RowGap]refCritiqueProcess.south) {\taxpaper{Active Preference Learning}{muldrew2024active}; \taxpaper{AI Safety via Debate}{irving2018debate}; \taxpaper{Weak-to-Strong}{burns2024weakstrong}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (reliableScalable) at (277,0 |- refReliableScalable.center) {Reliable \& Scalable Oversight};
\coordinate (feedbackOversightMid) at ($(demonstrationsPreferences.center)!0.5!(reliableScalable.center)$);
\node[box,fill=categoryblue,text width=82pt,minimum width=88pt,minimum height=29pt] (feedbackOversight) at (163,0 |- feedbackOversightMid) {Feedback \& Oversight};
\coordinate (supervisionMid) at ($(feedbackSource.center)!0.5!(feedbackOversight.center)$);
\node[box,fill=categoryyellow,text width=64pt,minimum width=70pt,minimum height=29pt] (pillarSupervision) at (70,0 |- supervisionMid) {\textbf{Alignment\\Supervision}};

% =========================================================
% III. ALIGNMENT MECHANISMS
% =========================================================
\node[refs,anchor=north] (refRewardVerifier) at ([yshift=-\PillarGap]refReliableScalable.south) {\taxpaper{HelpSteer2}{wang2024helpsteer2}; \taxpaper{Let's Verify Step by Step}{lightman2024verify}; \taxpaper{BiPRM}{zhang2026biprm}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (rewardVerifier) at (277,0 |- refRewardVerifier.center) {Reward \& Verifier Modeling};
\node[refs,anchor=north] (refSupervisedAlignment) at ([yshift=-\RowGap]refRewardVerifier.south) {\taxpaper{FLAN}{wei2022flan}; \taxpaper{Self-Instruct}{wang2023selfinstruct}; \taxpaper{LIMA}{zhou2023lima}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (supervisedAlignment) at (277,0 |- refSupervisedAlignment.center) {Supervised Alignment};
\node[refs,anchor=north] (refPreferenceOptimization) at ([yshift=-\RowGap]refSupervisedAlignment.south) {\taxpaper{DPO}{rafailov2023dpo}; \taxpaper{TDPO}{zeng2024tdpo}; \taxpaper{SPPO}{wu2025sppo}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (preferenceOptimization) at (277,0 |- refPreferenceOptimization.center) {Preference Optimization};
\node[refs,anchor=north] (refReinforcementLearning) at ([yshift=-\RowGap]refPreferenceOptimization.south) {\taxpaper{InstructGPT}{ouyang2022instructgpt}; \taxpaper{Safe RLHF}{dai2024saferlhf}; \taxpaper{DeepSeek-R1}{deepseek2025r1}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (reinforcementLearning) at (277,0 |- refReinforcementLearning.center) {Reinforcement Learning};
\node[refs,anchor=north] (refAlignmentDistillation) at ([yshift=-\RowGap]refReinforcementLearning.south) {\taxpaper{PLaD}{zhang2024plad}; \taxpaper{PAD}{gu2025pad}; \taxpaper{AlignDistil}{zhang2025aligndistil}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (alignmentDistillation) at (277,0 |- refAlignmentDistillation.center) {Alignment Distillation};
\coordinate (trainingAlignmentMid) at ($(rewardVerifier.center)!0.5!(alignmentDistillation.center)$);
\node[box,fill=categoryblue,text width=82pt,minimum width=88pt,minimum height=29pt] (trainingAlignment) at (163,0 |- trainingAlignmentMid) {Training-Time Alignment};

\node[refs,anchor=north] (refSteeringSearchRefinement) at ([yshift=-\GroupGap]refAlignmentDistillation.south) {\taxpaper{Inference-Time Intervention}{li2023iti}; \taxpaper{Self-Refine}{madaan2023selfrefine}; \taxpaper{CRITIC}{gou2024critic}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (steeringSearchRefinement) at (277,0 |- refSteeringSearchRefinement.center) {Steering, Search \& Refinement};
\node[refs,anchor=north] (refHumanControl) at ([yshift=-\RowGap]refSteeringSearchRefinement.south) {\taxpaper{Corrigibility}{soares2015corrigibility}; \taxpaper{Learning to Defer}{mozannar2020defer}; \taxpaper{AI Control}{greenblatt2024control}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (humanControl) at (277,0 |- refHumanControl.center) {Human Control};
\coordinate (inferenceAlignmentMid) at ($(steeringSearchRefinement.center)!0.5!(humanControl.center)$);
\node[box,fill=categoryblue,text width=82pt,minimum width=88pt,minimum height=29pt] (inferenceAlignment) at (163,0 |- inferenceAlignmentMid) {Inference-Time Alignment};
\coordinate (mechanismsMid) at ($(trainingAlignment.center)!0.5!(inferenceAlignment.center)$);
\node[box,fill=categoryyellow,text width=64pt,minimum width=70pt,minimum height=29pt] (pillarMechanisms) at (70,0 |- mechanismsMid) {\textbf{Alignment\\Mechanisms}};

% =========================================================
% IV. ALIGNMENT ASSURANCE
% =========================================================
\node[refs,anchor=north] (refBehavioralEvaluator) at ([yshift=-\PillarGap]refHumanControl.south) {\taxpaper{IFEval}{zhou2023ifeval}; \taxpaper{RewardBench}{lambert2025rewardbench}; \taxpaper{PRMBench}{song2025prmbench}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (behavioralEvaluator) at (277,0 |- refBehavioralEvaluator.center) {Behavioral \& Evaluator Evaluation};
\node[refs,anchor=north] (refAdversarialDistribution) at ([yshift=-\RowGap]refBehavioralEvaluator.south) {\taxpaper{Universal Adversarial Attacks}{zou2023jailbreak}; \taxpaper{HarmBench}{mazeika2024harmbench}; \taxpaper{Goal Misgeneralization in Deep RL}{langosco2022goal}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (adversarialDistribution) at (277,0 |- refAdversarialDistribution.center) {Adversarial \& Distribution Robustness};
\node[refs,anchor=north] (refAlignmentPreservation) at ([yshift=-\RowGap]refAdversarialDistribution.south) {\taxpaper{Fine-Tuning Aligned LMs Compromises Safety}{qi2024finetuning}; \taxpaper{Lifelong Safety Alignment}{wang2025lifelong}; \taxpaper{LARF}{li2025larf}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (alignmentPreservation) at (277,0 |- refAlignmentPreservation.center) {Alignment Preservation};
\coordinate (evaluationRobustnessMid) at ($(behavioralEvaluator.center)!0.5!(alignmentPreservation.center)$);
\node[box,fill=categoryblue,text width=82pt,minimum width=88pt,minimum height=29pt] (evaluationRobustness) at (163,0 |- evaluationRobustnessMid) {Evaluation \& Robustness};

\node[refs,anchor=north] (refMechanisticTheoretical) at ([yshift=-\GroupGap]refAlignmentPreservation.south) {\taxpaper{Discovering Latent Knowledge}{burns2023latent}; \taxpaper{Inference-Time Intervention}{li2023iti}; \taxpaper{Sustaining AI Safety}{mazzu2026sustaining}.};
\node[box,fill=categorygreen,text width=106pt,minimum width=112pt,minimum height=29pt] (mechanisticTheoretical) at (277,0 |- refMechanisticTheoretical.center) {Mechanistic \& Theoretical Evidence};
\node[refs,anchor=north] (refMonitoringAuditing) at ([yshift=-\RowGap]refMechanisticTheoretical.south) {\taxpaper{Model-Written Evaluations}{perez2023modelwritten}; \taxpaper{AI Control}{greenblatt2024control}; \taxpaper{Test-Set Contamination}{oren2024contamination}.};
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

def generate_taxonomy():
    tex_filename = "taxonomy.tex"
    pdf_filename = "taxonomy.pdf"
    png_filename = "taxonomy.png"

    # Bước 1: Ghi chuỗi mã LaTeX ra file .tex
    print("1. Đang ghi file LaTeX...")
    with open(tex_filename, "w", encoding="utf-8") as f:
        f.write(latex_code.strip())

    # Bước 2: Biên dịch file .tex thành .pdf bằng pdflatex
    print("2. Đang biên dịch PDF...")
    try:
        # Tùy chọn -interaction=nonstopmode để không bị dừng nếu có cảnh báo nhỏ
        subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_filename], check=True, stdout=subprocess.DEVNULL)
        print(f"   => Đã tạo thành công {pdf_filename}")
    except subprocess.CalledProcessError:
        print("   [Lỗi] Không thể biên dịch LaTeX. Vui lòng kiểm tra lại trình biên dịch (pdflatex) trên máy.")
        return
    except FileNotFoundError:
        print("   [Lỗi] Không tìm thấy lệnh 'pdflatex'. Vui lòng cài đặt TeX Live hoặc MiKTeX và thêm vào biến môi trường Path.")
        return

    # Bước 3: Chuyển đổi file .pdf thành .png bằng PyMuPDF
    print("3. Đang chuyển đổi PDF sang PNG...")
    try:
        # Mở file PDF
        doc = fitz.open(pdf_filename)
        # Chỉ lấy trang đầu tiên (do standalone chỉ sinh ra 1 trang)
        page = doc.load_page(0)
        # Thiết lập độ phân giải cao (dpi=300)
        pix = page.get_pixmap(dpi=300)
        # Lưu file PNG
        pix.save(png_filename)
        doc.close()
        print(f"   => Đã tạo thành công {png_filename}")
        
    except Exception as e:
        print(f"   [Lỗi] Có vấn đề trong quá trình xuất PNG: {e}")
        return

    # Bước 4: Dọn dẹp các tệp tạm do LaTeX sinh ra (Tùy chọn)
    print("4. Đang dọn dẹp các tệp phụ trợ...")
    for ext in [".aux", ".log", ".tex"]:
        temp_file = tex_filename.replace(".tex", ext)
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
    print("\nHoàn tất! Cả file Taxonomy.pdf và Taxonomy.png đã sẵn sàng trong thư mục hiện tại.")

if __name__ == "__main__":
    generate_taxonomy()