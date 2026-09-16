import argparse
import json
import os
from typing import List
import wandb
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from datasets import load_dataset
from tqdm import tqdm  # 추가

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    pipeline,
    default_data_collator,
)


################################################################
# DPO LOSS 구현
################################################################
def dpo_loss(
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    chosen_reward: torch.Tensor,
    rejected_reward: torch.Tensor,
    beta: float = 0.1
) -> torch.Tensor:
    """
    DPO Loss:
    L = - log σ( β * (R_c - R_r) + (log p(c|x) - log p(r|x)) )
    """
    reward_diff = chosen_reward - rejected_reward  # [batch]
    logp_diff = chosen_logps - rejected_logps      # [batch]
    inside_term = beta * reward_diff + logp_diff
    loss = -F.logsigmoid(inside_term)
    return loss.mean()


################################################################
# Dataset
################################################################
class OfflineDPODataset(Dataset):
    """
    오프라인으로 만들어둔 JSONL (prompt, chosen, rejected, chosen_reward, rejected_reward)을
    Dataset으로 로딩하여, DPO 학습에 사용.
    """
    def __init__(self, dataset):
        """
        dataset: HF Dataset (json) 형태
        """
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "prompt": item["prompt"],
            "chosen": item["chosen"],
            "rejected": item["rejected"],
            "chosen_reward": torch.tensor(item["chosen_reward"], dtype=torch.float32),
            "rejected_reward": torch.tensor(item["rejected_reward"], dtype=torch.float32),
        }


################################################################
# Custom Collator
################################################################
def custom_dpo_collator(batch):
    """
    Trainer에 전달될 배치 구성을 정의.
    """
    return {
        "prompt": [item["prompt"] for item in batch],
        "chosen": [item["chosen"] for item in batch],
        "rejected": [item["rejected"] for item in batch],
        "chosen_reward": torch.stack([item["chosen_reward"] for item in batch]),
        "rejected_reward": torch.stack([item["rejected_reward"] for item in batch]),
    }


################################################################
# DPOTrainer
################################################################
class DPOTrainer(Trainer):
    """
    Trainer를 상속하여, train/eval 시 (prompt, chosen, rejected, chosen_reward, rejected_reward)
    로부터 DPO Loss를 계산하고, 평가 단계에서는 실제 student 모델의 응답에 대해
    reward를 계산하여 eval_reward metric을 반환합니다.
    """
    def __init__(self, processing_class=None, rm_pipe=None, rm_tokenizer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processing_class = processing_class  # student_tokenizer
        self.rm_pipe = rm_pipe                    # reward model pipeline
        self.rm_tokenizer = rm_tokenizer          # reward model tokenizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        prompts = inputs["prompt"]
        chosen_texts = inputs["chosen"]
        rejected_texts = inputs["rejected"]
        chosen_rewards = inputs["chosen_reward"]
        rejected_rewards = inputs["rejected_reward"]

        chosen_logps = []
        rejected_logps = []

        for p, c, r in zip(prompts, chosen_texts, rejected_texts):
            chosen_logps.append(self._calc_log_probs(model, p, c))
            rejected_logps.append(self._calc_log_probs(model, p, r))

        chosen_logps = torch.stack(chosen_logps)
        rejected_logps = torch.stack(rejected_logps)  # (batch,)

        loss = dpo_loss(chosen_logps, rejected_logps, chosen_rewards, rejected_rewards, beta=0.1)
        return (loss, None) if return_outputs else loss

    def _calc_log_probs(self, model, prompt, answer):
        tokenizer = self.processing_class
        device = next(model.parameters()).device

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        full_ids = prompt_ids + answer_ids

        # 빈 prompt/answer인 경우 처리
        if len(full_ids) == 0 or len(answer_ids) == 0:
            return torch.tensor(0.0, device=device)

        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            out = model(input_ids, labels=input_ids)
        seq_len = input_ids.shape[1]
        total_nll = out.loss * seq_len
        ratio = len(answer_ids) / seq_len
        approx_answer_logp_val = -total_nll * ratio

        # backward 연산을 위해 requires_grad=True로 계산
        approx_answer_logp = torch.tensor(approx_answer_logp_val.item(), device=device, requires_grad=True)
        return approx_answer_logp

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        기본 prediction_step을 우회하여, student 모델이 eval_dataset의 prompt에 대해 생성한 답변을
        reward 모델로 평가한 average reward를 eval_reward metric으로 반환합니다.
        """
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        self.model.eval()
        metrics = evaluate_student_model_on_prompts(
            self.model,
            self.processing_class,
            self.rm_pipe,
            self.rm_tokenizer,
            eval_dataset,
        )
        result = {f"{metric_key_prefix}_reward": metrics["avg_reward"]}
        return result


################################################################
# (수정된) Student Model 평가 함수
# - 배치 단위로 prompt를 묶어서 한번에 generate
# - reward 모델 inference도 배치 단위로 처리
# - tqdm으로 진행상황 표시
################################################################
def evaluate_student_model_on_prompts(
    student_model,
    student_tokenizer,
    rm_pipe,
    rm_tokenizer,
    eval_dataset,
    max_new_tokens=64,
    batch_size=8,
):
    """
    eval_dataset 안의 prompt를 batch_size씩 묶어서, student_model이 답변을 생성한 뒤
    reward model로부터 점수를 얻어 평균값을 구함.

    반환 예시: {"avg_reward": float 값}
    """
    student_model.eval()
    device = student_model.device

    # 전체 prompt 수집
    all_prompts = [ex["prompt"] for ex in eval_dataset]
    all_gen_texts = []

    # -------------------------------
    # 1) Student 모델로부터 답변(batch 단위) 생성
    # -------------------------------
    for i in tqdm(range(0, len(all_prompts), batch_size), desc="Generating answers"):
        batch_prompts = all_prompts[i:i+batch_size]

        # 토크나이징 (padding, truncation)
        batch_inputs = student_tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}



        with torch.no_grad():
            # generate 호출 시 attention_mask도 같이 넣기
            batch_outputs = student_model.generate(
                input_ids=batch_inputs["input_ids"],
                attention_mask=batch_inputs["attention_mask"],  # ✅ 이거 추가!
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # 각 결과를 디코딩
        for out_ids in batch_outputs:
            gen_text = student_tokenizer.decode(out_ids, skip_special_tokens=True)
            all_gen_texts.append(gen_text)

    # -------------------------------
    # 2) Reward 모델에 넣을 텍스트(batch 단위) 구성
    # -------------------------------
    rm_input_texts = []
    for prompt, gen_text in zip(all_prompts, all_gen_texts):
        # RM용 chat 템플릿 - 모델에 맞춰 수정 가능
        chat = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": gen_text},
        ]
        # 아래는 예시로 rm_tokenizer에 있는 apply_chat_template을 사용한다고 가정
        input_text = rm_tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=False
        ).replace(rm_tokenizer.bos_token, "")
        rm_input_texts.append(input_text)

    all_rewards = []

    # -------------------------------
    # 3) Reward 모델(batch 단위)로 스코어 얻기
    # -------------------------------
    for i in tqdm(range(0, len(rm_input_texts), batch_size), desc="Scoring with Reward Model"):
        batch_texts = rm_input_texts[i:i+batch_size]

        pipe_outputs = rm_pipe(
            batch_texts,
            return_all_scores=True,
            function_to_apply="none",
            batch_size=len(batch_texts),  # pipeline에 들어갈 실제 batch size
        )
        # pipeline 결과에서 score만 추출
        for out in pipe_outputs:
            reward_val = float(out[0]["score"])
            all_rewards.append(reward_val)

    avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    print("Average Reward:", avg_reward)
    # WandB에 로깅
    wandb.log({"avg_reward": avg_reward})
    return {"avg_reward": avg_reward}


################################################################
# compute_metrics
# Trainer가 evaluate()를 호출할 때 사용되는 메트릭 계산 함수
################################################################
def make_compute_metrics_fn(
    student_model,
    student_tokenizer,
    rm_pipe,
    rm_tokenizer,
    eval_dataset,
):
    """
    Trainer에게 전달할 compute_metrics 함수를 만들어 반환.
    """
    def compute_metrics(eval_preds):
        """
        eval_preds: HF Trainer가 전달하는 (predictions, labels) 튜플(혹은 dict)
                    DPOTrainer에서는 사실 크게 의미 없는 값일 수 있으므로 무시 가능.
        """
        # Student 모델로 eval_dataset Prompt에 대해 응답 생성 -> Reward 평균 계산
        result = evaluate_student_model_on_prompts(
            student_model,
            student_tokenizer,
            rm_pipe,
            rm_tokenizer,
            eval_dataset,
        )
        # Trainer가 인식할 수 있도록 키를 "eval_reward"로 지정
        # (metric_for_best_model="eval_reward")
        metrics = {
            "eval_reward": result["avg_reward"]
        }
        return metrics

    return compute_metrics


################################################################
# main
################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="train_offline.jsonl")
    parser.add_argument("--eval_file", type=str, default="test_offline.jsonl")
    parser.add_argument("--student_model_name_or_path", type=str, required=True)
    parser.add_argument("--reward_model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./dpo_outputs")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    args = parser.parse_args()

    # Weights & Biases 초기화
    wandb.init(project="DPO-Training", config=args)

    # 1) Student 모델 로드
    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model_name_or_path)
    student_tokenizer.pad_token = student_tokenizer.eos_token
    student_model = AutoModelForCausalLM.from_pretrained(
        args.student_model_name_or_path,
        torch_dtype=torch.bfloat16,
    ).cuda()
    student_model.train()

    # 2) Reward 모델 로드 (pipeline 방식)
    rm_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_name_or_path)
    rm_pipe = pipeline(
        "sentiment-analysis",
        model=args.reward_model_name_or_path,
        tokenizer=rm_tokenizer,
        device_map="auto",
        model_kwargs={"torch_dtype": torch.bfloat16},
    )

    # 3) Offline Dataset 로드
    print(f"Loading offline train dataset from {args.train_file}")
    train_dataset = load_dataset("json", data_files={"train": args.train_file})["train"]
    dpo_train_dataset = OfflineDPODataset(train_dataset)

    eval_dataset = None
    dpo_eval_dataset = None
    if args.eval_file and os.path.exists(args.eval_file):
        print(f"Loading offline eval dataset from {args.eval_file}")
        eval_dataset = load_dataset("json", data_files={"eval": args.eval_file})["eval"]
        dpo_eval_dataset = OfflineDPODataset(eval_dataset)
    else:
        print("No eval file found or path invalid. Skipping eval dataset.")

    # 4) TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        logging_steps=50,
        bf16=True,  # GPU가 bf16 지원 시
        gradient_accumulation_steps=4,
        ddp_find_unused_parameters=False,
        save_steps=200,

        # === 추가: epoch마다 평가/체크포인트 저장 ===
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_reward",
        greater_is_better=True,

        remove_unused_columns=False,
    )

    # 5) Trainer 생성
    trainer = DPOTrainer(
        model=student_model,
        processing_class=student_tokenizer,
        rm_pipe=rm_pipe,
        rm_tokenizer=rm_tokenizer,
        args=training_args,
        train_dataset=dpo_train_dataset,
        eval_dataset=dpo_eval_dataset,
        data_collator=custom_dpo_collator,
        compute_metrics=make_compute_metrics_fn(
            student_model, student_tokenizer, rm_pipe, rm_tokenizer,
            dpo_eval_dataset if dpo_eval_dataset else [],
        )
    )

    if dpo_eval_dataset:
        print("==== Initial Evaluation Before Training ====")
        initial_metrics = trainer.evaluate()
        print("Initial Evaluation Metrics:", initial_metrics)

    # 6) DPO Training
    trainer.train()

    # 최종 모델 저장
    trainer.save_model(args.output_dir)
    print("DPO Training Done.")

    # (선택) 마지막으로 Best 모델에 대한 평가 로깅
    if dpo_eval_dataset:
        best_model = AutoModelForCausalLM.from_pretrained(args.output_dir).cuda()
        final_result = evaluate_student_model_on_prompts(
            best_model,
            student_tokenizer,
            rm_pipe,
            rm_tokenizer,
            dpo_eval_dataset
        )
        print("==== Final Evaluation (Best Model) ====")
        print(final_result)
        wandb.log({"final_eval_reward": final_result["avg_reward"]})


if __name__ == "__main__":
    main()