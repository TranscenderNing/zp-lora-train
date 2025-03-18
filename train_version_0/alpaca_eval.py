from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import BitsAndBytesConfig
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import csv
import json
import functools
import argparse 

from datetime import datetime
import uuid
import os
import time
from peft import PeftModel


# 系统消息
DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。"""


# 模板
# TEMPLATE = (
#     "[INST] <<SYS>>\n"
#     "{system_prompt}\n"
#     "<</SYS>>\n\n"
#     "判断一段话是否为诈骗话术，输出0或1，这段话为-->{input_text} [/INST]"
# )
TEMPLATE = (
    "[INST] <<SYS>>\n"
    "{system_prompt}\n"
    "<</SYS>>\n\n"
    "{instruction} [/INST]"
)


# 结果保存路径
save_batch_result_dir = '/home/ldn/baidu/reft-pytorch-codes/zp/origin-train/data/eval/results/'
# 预测错误样本保存路径
save_error_result_dir = '/home/ldn/baidu/reft-pytorch-codes/zp/origin-train/data/eval/err'
# 上传样本信息路径
messages_dir = '/home/ldn/baidu/reft-pytorch-codes/zp/origin-train/data/eval/test_data.json'

result_filename = save_batch_result_dir  + 'predict.csv'
error_filename = save_error_result_dir + 'predict_error.csv'

# 获取当前时间
def getCurrentTime():
    timestamp = time.time()
    dt_object = datetime.fromtimestamp(timestamp)
    formatted_time = dt_object.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_time


# testing log
def write_log_to_csv(filename, error_filename, epoch_profile, batch_logs, sample_logs, error_logs):
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(epoch_profile)
        writer.writerow([])  # 添加空行分隔表格
        writer.writerows(batch_logs)
        writer.writerow([])
        writer.writerows(sample_logs)
    print(f"save {len(sample_logs) - 1} results to {filename}.")
    if len(error_logs) > 0 :
        with open(error_filename, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(error_logs)
        print(f"save {len(error_logs) - 1} errors to {error_filename}. error_log:")
        for row in error_logs[1:]:
            print(row)
        print("=" * 100)
    else :
        print("there is no errors.")


# 格式化数据集格式
def generate_prompt(instruction, system_prompt=DEFAULT_SYSTEM_PROMPT):
    return TEMPLATE.format_map({'instruction': instruction,'system_prompt': system_prompt})


# alpaca dataloader
def collote_fn_alpaca(batch_samples,tokenizer):
    batch_sentence_1 = []
    batch_label = []
    ids = []
    for sample in batch_samples:
        batch_sentence_1.append(sample['text'])
        batch_label.append(int(sample['label']))
        ids.append(sample['id'])
        
    X = tokenizer(batch_sentence_1,return_tensors="pt", padding=True, truncation=True, )
    y = torch.tensor(batch_label)
    return X, y, ids


# alpaca 数据集
class ZpAlpacaData(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        with open(data_file, mode='r') as f:
            data = f.read()
        samples = json.loads(data)
        
        Data = {}
        for idx, sample in enumerate(samples):
            formatted_prompt = generate_prompt(instruction=f"{sample['instruction']}{sample['input']}")
            Data[idx] = {'text': formatted_prompt, 'label': sample['output'], 'id':sample['id']}
        
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class ChineseAlpace2:
    def __init__(self, base_model = None, batch_size = 8) -> None:
        print("[info]: 开始加载 alpaca tokenizer and alpaca model")
        self.device = "cuda"
        self.batch_size = batch_size
        # 加载model及tokenizer
        if base_model is None:
            base_model = "../../models/chinese-alpaca-2-7b-ZPdata1-1epochs"
        self.tokenizer = LlamaTokenizer.from_pretrained(base_model, legacy=True, padding_side='left', pad_token='<s>')
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=False,
                load_in_8bit=False,
                bnb_4bit_compute_dtype=torch.float16
            )
        )
        self.model = PeftModel.from_pretrained(model, base_model)
        self.model.eval()
        print("[info]: 加载 alpaca tokenizer and alpaca model 完毕")

    def run(self, input_text):
        formatted_inputs = [self.generate_prompt(i) for i in input_text]
        with torch.no_grad():
            X = self.tokenizer(formatted_inputs,return_tensors="pt", padding=True, truncation=True, ).to(self.device)
            output = self.model.generate(**X, max_new_tokens=10,return_dict_in_generate=True, output_scores=True)
            input_length = X.input_ids.shape[1]
            generated_tokens = output.sequences[:, input_length:]
            output = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            print(output)
        return output

    def generate_prompt(input_text, system_prompt=DEFAULT_SYSTEM_PROMPT):
        return TEMPLATE.format_map({'input_text': input_text,'system_prompt': system_prompt})

    def batch_evaluation_alpaca(self, test_dataloader, size, filename):
        with torch.no_grad():
            batch_id = 0  # 用于生成记录一个批次的文件名
            nums_batch = size // self.batch_size
            if size % self.batch_size != 0:  # 判断是否有余数
                nums_batch += 1  # 如果有余数，向上取整
            epoch_start_time = time.time()
            epoch_start_time_str = getCurrentTime()
            sample_logs = [['batch id', 'tid','content','perdicted lable','original lable','confidence','sampletimecost']] 
            batch_logs = [['batch id','batch tokens','batch_samples','batch_time_cost(s)']]
            error_logs = [['tid','content','perdicted lable','original lable','confidence']] 
            for X, y, ids in test_dataloader:
                print(f'第{batch_id+1}个batch,共{nums_batch}个batch')
                correct = 0
                confidence = []
                labels_arr = []
                starttime = getCurrentTime()
                start_time = time.time()
                X, y = X.to(self.device), y.to(self.device)
                # 原始文本
                # time
                Xtext = self.tokenizer.batch_decode(X.input_ids,skip_special_tokens=True)
                labels_arr += np.array(y.cpu()).tolist()
                for idx in range(X.input_ids.shape[0]):
                    print(f"{ids[idx]} | {Xtext[idx].split('-->')[1].split('[/INST]')[0]}")
                # 模型输出
                # time
                output = self.model.generate(**X, max_new_tokens=5,return_dict_in_generate=True, output_scores=True)
                
                ## 统计输出概率
                transition_scores = self.model.compute_transition_scores(output.sequences, output.scores, normalize_logits=True)
                input_length = X.input_ids.shape[1]
                # print('input_length',input_length)
                generated_tokens = output.sequences[:, input_length:]
                # output = tokenizer.batch_decode(output.sequences,)
                # print(output)
                n = generated_tokens.shape[0]
                # time
                for idx in range(n):
                    print(f" {generated_tokens[idx][1]:5d} | {self.tokenizer.decode(generated_tokens[idx][1]):8s} | {np.exp(transition_scores[idx][1].cpu().numpy()):.2f} ")
                    confidence.append(np.exp(transition_scores[idx][1].cpu().numpy()))
                
                # time
                output = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                result = []
                for str1 in output:
                    result.append(int(str1))

                end_time = time.time()
                execution_time = (end_time - start_time)
            
                # time
                tokens = [input_length] * self.batch_size
                temp_list= zip(ids, Xtext, result, labels_arr, confidence, tokens)
                for id, content, pred_label, original_label, confidence, tokens in temp_list:
                    sample_logs.append(
                        [
                            batch_id, 
                            id,
                            content.split('-->')[1].split('[/INST]')[0],
                            pred_label,
                            original_label,
                            confidence,
                            execution_time
                        ]
                    )
                    if pred_label != original_label:
                        error_logs.append(
                            [
                                id,
                                content.split('-->')[1].split('[/INST]')[0],
                                pred_label,
                                original_label,
                                confidence,
                            ]
                        )
                batch_logs.append([batch_id, self.batch_size*input_length, self.batch_size, execution_time])
                batch_id += 1
                
            epoch_end_time = time.time()
            random_uuid = uuid.uuid4()
            epoch_profile = [
                ['epoch id', 'epoch tokens', 'epoch samples', 'epoch batches', 'epoch start time', 'epoch end time', 'epoch time cost(s)'],
                [random_uuid, size*input_length, size, int(nums_batch), epoch_start_time_str , getCurrentTime(), (epoch_end_time-epoch_start_time)],
            ]
            # result_filename = save_batch_result_dir + filename[:-5] +'.csv'
            # error_filename = save_error_result_dir + filename[:-5] +'_error.csv'
            # time
            write_log_to_csv(result_filename, error_filename, epoch_profile, batch_logs, sample_logs, error_logs)   


def main(args):
    batch_size = args.batchsize
    modelpath = args.modelpath
    # begin
    model = ChineseAlpace2(batch_size=batch_size, base_model=modelpath)
    # end
    if not os.path.exists(messages_dir):
        print(f"{messages_dir} not exists, make this dir.")
        os.mkdir(messages_dir)
    if not os.path.exists(save_error_result_dir):
        print(f"{save_error_result_dir} not exists, make this dir.")
        os.mkdir(save_error_result_dir)

    # 处理文件
    filepath = messages_dir
    print("-" * 100)
    print(f"process {filepath}.")
    test_data = ZpAlpacaData(filepath)
    size = len(test_data)
    print(f"total messages: {size}")
    partial_collate_fn = functools.partial(collote_fn_alpaca,tokenizer=model.tokenizer)
    # time
    test_dataloader = DataLoader(test_data, batch_size=batch_size, collate_fn=partial_collate_fn)
    # time
    model.batch_evaluation_alpaca(test_dataloader, size, filepath)
    print(f"complate process {filepath}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', default=8, type=int, help='batch size')
    parser.add_argument('--modelpath', default="/home/ldn/baidu/reft-pytorch-codes/zp/origin-train/models/chinese-alpaca-2-7b-ZPdata1-1epochs-2025-0310", type=str, help='model path')
    args = parser.parse_args()
    main(args)