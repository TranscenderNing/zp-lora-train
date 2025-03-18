from unsloth import FastLanguageModel
import torch
import wandb
import sys
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from datasets import load_dataset
import time
import json





prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
    Write a response that appropriately completes the request.
    Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

    ### Instruction:
    假设您擅长识别诈骗案例，请根据以下情节分析并判断该案例是否为诈骗行为。如果是，请输出1；否则，请输出0。只需输出1或0。

    ### Question:
    {}

    ### Response:
    <think>{}"""


train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
    Write a response that appropriately completes the request.
    Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

    ### Instruction:
    假设您擅长识别诈骗案例，请根据以下情节分析并判断该案例是否为诈骗行为。如果是，请输出1；否则，请输出0。只需输出1或0。

    ### Question:
    {}

    ### Response:
    <think>
    {}
    </think>
    {}"""

wandb.login(key="af2d64fbef52fe17acd762501c685c7c720d8cf3")
run = wandb.init(
    project="my fint-tune on deepseek r1 with zp data",
    job_type="training",
    anonymous="allow",
)


max_seq_length = 2048
dtype = None
load_in_4bit = True
epochs = 3
learning_rate = 3e-4

data_dir = "/home/ldn/baidu/reft-pytorch-codes/zp/data/zp_data/train"
model_path = "/home/ldn/baidu/reft-pytorch-codes/zp/models/DeepSeek-R1-Distill-Llama-8B"

new_model_local = f"models/DeepSeek-R1-8b-zp-COT-10080-{epochs}-{learning_rate}"
new_merge_model_local = f"models/DeepSeek-R1-8b-zp-COT-merged-10080-{epochs}-{learning_rate}"

def train():
    # ## Loading the model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # ## Loading and processing the dataset
    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN
    print("eos token:")
    print(EOS_TOKEN)

    def formatting_prompts_func(examples):
        inputs = examples["Question"]
        cots = examples["Complex_CoT"]
        outputs = examples["Response"]
        texts = []
        for input, cot, output in zip(inputs, cots, outputs):
            text = train_prompt_style.format(input, cot, output) + EOS_TOKEN
            texts.append(text)
        return {
            "text": texts,
        }

    dataset = load_dataset(
        data_dir, split="train"
    )  # 这里同样去huggingface上面下载数据集，然后放到本地

    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
    )
    print("dataset:")
    print(len(dataset))
    print(dataset["text"][0])

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        args=TrainingArguments(
            num_train_epochs=epochs,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            # Use num_train_epochs = 1, warmup_ratio for full training runs!
            warmup_steps=5,
            # max_steps=60,
            learning_rate=learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
        ),
    )

    trainer_stats = trainer.train()
    print(trainer_stats)

    wandb.finish()

    # ## Saving the model locally

    model.save_pretrained(new_model_local)  # Local saving
    tokenizer.save_pretrained(new_model_local)
    model.save_pretrained_merged(
        new_merge_model_local,
        tokenizer,
        save_method="merged_16bit",
    )


def main():
    # 记录开始时间
    start_time = time.time()

    # 调用训练函数
    train()

    # 记录结束时间
    end_time = time.time()

    # 计算并打印训练函数执行时间（单位：秒）
    elapsed_time = end_time - start_time
    print(f"Train function executed in {elapsed_time / 60:.2f} minutes.")


if __name__ == "__main__":
    main()
