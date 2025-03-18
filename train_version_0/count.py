import csv
import pandas as pd
import numpy as np
import json
import argparse
import os

from collections import defaultdict

ROUND_N = 4

parser = argparse.ArgumentParser()
parser.add_argument("--file_path", type=str, required=True, help="test_log file dir")

# 读取csv文件，遇到空行结束
def read_table(reader):
    tables = []
    for row in reader:
            if not row:  # 遇到空行表示当前表结束
                return reader, tables
            else:
                tables.append(row)
    return reader, tables


# 读取test_log，获取epoch, batch, predicts三张表
def read_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        reader, epoch = read_table(reader)
        reader, batch = read_table(reader)
        _, predicts = read_table(reader)
    return epoch, batch, predicts


# 计算平均值和标准差
def cal_mean_std(lst):
    np_array = np.array(lst)
    np_array = np_array.astype(float)
    mean = np.mean(np_array)
    std = np.std(np_array)
    mean = mean if not np.isnan(mean) else -1
    std = std if not np.isnan(std) else -1
    return mean, std


# 统计测试结果
def count_test(predicts):
    TP = FP = TN = FN = 0
    p_confidences = []
    n_confidences = []

    # 统计各项数据
    for i in range(1, len(predicts)):
        perdicted, original, confidence = predicts[i][3:6]
        if perdicted == '1':
            p_confidences.append(confidence)
            if original == '1':
                TP += 1
            else:
                FP += 1
        else:
            n_confidences.append(confidence)
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
    # print(accuracy, precision, recall, f1)

    # 计算正负样本置信度的平均值，标准差
    p_confidences_mean, p_confidences_std =  cal_mean_std(p_confidences)
    n_confidences_mean, n_confidences_std =  cal_mean_std(n_confidences)

    return {
        'TP': TP,
        'FP': FP,
        'TN': TN,
        'FN': FN,
        'accuracy': round(accuracy, ROUND_N),
        'recall': round(recall, ROUND_N),
        'f1': round(f1, ROUND_N),
        'p_confidences_mean': round(p_confidences_mean, ROUND_N),
        'p_confidences_std': round(p_confidences_std, ROUND_N),
        'n_confidences_mean': round(n_confidences_mean, ROUND_N),
        'n_confidences_std': round(n_confidences_std, ROUND_N),}


# 统计性能结果
def count_performance(epoch, batch, predicts):
    epoch_tokens, epoch_samples = epoch[1][1:3]
    epoch_timecost = epoch[1][-1]
    MTPS = float(epoch_tokens) / float(epoch_timecost)
    MSPS = float(epoch_samples) / float(epoch_timecost)

    batch_timecost = [b[-1] for b in batch[1:]]
    batch_timecost_mean, batch_timecost_std = cal_mean_std(batch_timecost)

    predicts_timecost = [p[-1] for p in predicts[1:]]
    predicts_timecost_mean, predicts_timecost_std = cal_mean_std(predicts_timecost)

    return {
        'MTPS': int(MTPS),
        'MSPS': int(MSPS),
        'batch_timecost_mean': round(batch_timecost_mean, ROUND_N),
        'batch_timecost_std': round(batch_timecost_std, ROUND_N),
        'predicts_timecost_mean': round(predicts_timecost_mean, ROUND_N),
        'predicts_timecost_std': round(predicts_timecost_std, ROUND_N),}


# 统计测试结果及性能结果
def count(file_path):
    epoch, batch, predicts = read_csv(file_path)
    test_result =  count_test(predicts)
    perfomance = count_performance(epoch, batch, predicts)
    return test_result, perfomance


if __name__ == '__main__':
    # file_path = '/home/public/zp-codes/logs/Results/testinglog-202310191115.csv'  
    # args = parser.parse_args()
    file_paths = ["/home/ldn/baidu/reft-pytorch-codes/zp/origin-train/data/eval/results/predict.csv"]


    # 逐个处理测试日志
    test_results, perfomances = [], []
    for file_path in file_paths:
        print('-' * 100)
        print(f'处理测试日志 -  {file_path} ...')
        test_result, perfomance = count(file_path)
        print(test_result, perfomance)
        test_results.append(test_result)
        perfomances.append(perfomance)

    # 合并结果
    test_result = defaultdict(int)
    perfomance = defaultdict(int)
    for test_result_ in test_results:
        for k, v in test_result_.items():
            test_result[k] += v
    for perfomance_ in perfomances:
        for k, v in perfomance_.items():
            perfomance[k] += v
    n = len(file_paths)
    print(n)
    for k, v in test_result.items():
        if k in ['TP', 'FP', 'FN', 'TN']:
            continue
        test_result[k] /= n
        test_result[k] = round(test_result[k], ROUND_N)
    for k, v in perfomance.items():
        perfomance[k] /= n
        if k in ['MTPS', 'MSPS']:
            perfomance[k] = int(perfomance[k])
        perfomance[k] = round(perfomance[k], ROUND_N) 
    print('-' * 100)
    # print(test_result, perfomance)

    save_file = os.path.join('/home/ldn/baidu/reft-pytorch-codes/zp/origin-train/data/eval/results/result.json')
    print(f'保存结果到文件 -  {save_file} ...')
    with open(save_file, 'w') as f:
        json.dump({'test_result': test_result, 'perfomance': perfomance}, f, indent=4)