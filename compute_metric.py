import csv
import pandas as pd
import numpy as np
import json
import argparse
import os

from collections import defaultdict

ROUND_N = 4

predictions_file = "/home/ldn/baidu/reft-pytorch-codes/zp/DeepSeek-R1-Distill-Llama-8B_eval_reults.json"

predictions_file = "/home/ldn/baidu/reft-pytorch-codes/zp/DeepSeek-R1-Distill-Llama-8B_eval_reults_0.json"

predictions_file = "/home/ldn/baidu/reft-pytorch-codes/zp/DeepSeek-R1-Distill-Llama-8B_eval_reults_finetune.json"


predictions_file = "/home/ldn/baidu/reft-pytorch-codes/zp/DeepSeek-R1-Distill-Llama-8B_eval_reults_e3.json"
# 统计测试结果
def count_test():
    TP = FP = TN = FN = 0
    with open(predictions_file, mode='r', encoding='utf-8') as f:
        predicts = json.load(fp=f)
    
    print(len(predicts))
    # 统计各项数据
    for idx, eval_res in enumerate(iterable=predicts):
        try:
            perdicted = eval_res["output"].split("</think>\n\n")[1][0]
        except:
            print(idx)
            perdicted = None
        # perdicted = eval_res["output"].split("</think>\n\n")[1][0]
        original = str(eval_res["label"])
        if perdicted == '1':
            if original == '1':
                TP += 1
            else:
                FP += 1
        else:
            if original == '1':
                FN += 1
            else:
                TN += 1
    # print(TP, FP, TN, FN)
    # print(p_confidences, n_confidences)

    # 计算准确率，召回率，精确率，f1值
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    precision = 0 if (TP+FP) == 0 else TP/(TP+FP)
    recall = 0 if (TP+FN) == 0 else TP/(TP+FN)
    f1 = 0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)



    result= {
        'TP': TP,
        'FP': FP,
        'TN': TN,
        'FN': FN,
        'accuracy': round(accuracy, ROUND_N),
        'recall': round(recall, ROUND_N),
        'f1': round(f1, ROUND_N),}
    print(result)





if __name__ == '__main__':
    count_test() 
