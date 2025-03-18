# zp-lora-train
zp lora train
# 环境
conda activate xxx
# unloth训练
CUDA_VISIBLE_DEVICES=2 nohup python deepseekr1_8b_finetune.py > train_10080.log 2>&1 &

# 推理
CUDA_VISIBLE_DEVICES=1 nohup python deepseekr1_8b_infer.py > eval_2520.log 2>&1 &




# 计算指标
python compute_metric.py