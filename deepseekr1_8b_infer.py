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


max_seq_length = 2048
dtype = None
load_in_4bit = True


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

test_file="/home/ldn/baidu/reft-pytorch-codes/zp/data/zp_data/test/test_data.json"
# model_name="/home/ldn/baidu/reft-pytorch-codes/zp/models/DeepSeek-R1-Distill-Llama-8B" # 这里改成你本地模型，以我的为例，我已经huggingface上的模型文件下载到本地。
# model_name = "/home/ldn/baidu/reft-pytorch-codes/zp/models/DeepSeek-R1-8b-zp-COT-10080"
model_name = "/home/ldn/baidu/reft-pytorch-codes/zp/models/DeepSeek-R1-8b-zp-COT-merged-10080-3-0.0002"
output_file="DeepSeek-R1-Distill-Llama-8B_eval_reults_e3.json"

def deepseekr1_infer_batch(
    batch_size=8,
):
    with open(test_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(len(data))
    test_data = {"question": [], "label": []}
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        test_data["question"].append(
            [prompt_style.format(item["Question"], "") for item in batch]
        )  # 将当前批次的question添加到test_data中: [item["question"] for item in batch],
        test_data["label"].append(
            [item["label"] for item in batch]
        )  # 将当前批次的label添加到test_data中: [item["label"] for item in batch]})  # 将当前批次添加到批次列表中

    max_seq_length = 2048
    dtype = None
    load_in_4bit = True
    # ## Loading the model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,  # 这里改成你本地模型，以我的为例，我已经huggingface上的模型文件下载到本地。
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token


    FastLanguageModel.for_inference(model)  # Unsloth has 2x faster inference!

    json_records = []
    i = 0
    for questions, labels in zip(test_data["question"], test_data["label"]):
        i += 1
        print(f"正在处理第{i}个batch")
        print(len(questions))
        print(questions)
        print("="*100)
        print(len(labels))
        print(labels)
        inputs = tokenizer(questions, return_tensors="pt", padding=True, max_length="longest").to("cuda")
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=1200,
            use_cache=True,
        )
        response = tokenizer.batch_decode(outputs)
        print('response len', len(response))
        print(response)
        for question, label, model_res in zip(questions, labels, response):
            print(model_res)
            model_res = model_res.split("### Response:")[1]
            print("用户输入：")
            print(questions)
            print("模型输出")
            print(model_res)
            json_records.append(
                {"input": question, "output": model_res, "label": label}
            )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(json_records, f, ensure_ascii=False, indent=4)




def main():
    # 记录开始时间
    start_time = time.time()
    deepseekr1_infer_batch()
    # 记录结束时间
    end_time = time.time()

    # 计算并打印训练函数执行时间（单位：秒）
    elapsed_time = end_time - start_time
    print(f"test function executed in {elapsed_time / 60:.2f} minutes.")


if __name__ == "__main__":
    main()
