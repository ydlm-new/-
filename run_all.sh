#!/bin/bash
# 102 Flower Classification - 运行所有实验
# 在CPU环境下，每个实验约需30-60分钟

DATA_DIR="."
EPOCHS=10
BATCH_SIZE=32

echo "============================================"
echo "实验1: Baseline (Pretrained ResNet-18)"
echo "============================================"
python train.py --experiment baseline --epochs $EPOCHS --batch_size $BATCH_SIZE --data_dir $DATA_DIR --fc_lr 0.01 --backbone_lr 0.001

echo ""
echo "============================================"
echo "实验2a: 超参数对比 - 较小学习率"
echo "============================================"
python train.py --experiment baseline_lr1 --epochs $EPOCHS --batch_size $BATCH_SIZE --data_dir $DATA_DIR

echo ""
echo "============================================"
echo "实验2b: 超参数对比 - 中等学习率 (同Baseline)"
echo "============================================"
python train.py --experiment baseline_lr2 --epochs $EPOCHS --batch_size $BATCH_SIZE --data_dir $DATA_DIR

echo ""
echo "============================================"
echo "实验2c: 超参数对比 - 较大学习率"
echo "============================================"
python train.py --experiment baseline_lr3 --epochs $EPOCHS --batch_size $BATCH_SIZE --data_dir $DATA_DIR

echo ""
echo "============================================"
echo "实验3: From Scratch (无预训练)"
echo "============================================"
python train.py --experiment scratch --epochs $EPOCHS --batch_size $BATCH_SIZE --data_dir $DATA_DIR --fc_lr 0.01

echo ""
echo "============================================"
echo "实验4: SE-ResNet-18 (注意力机制)"
echo "============================================"
python train.py --experiment se_resnet --epochs $EPOCHS --batch_size $BATCH_SIZE --data_dir $DATA_DIR --fc_lr 0.01 --backbone_lr 0.001

echo ""
echo "============================================"
echo "所有实验完成！"
echo "运行 swanlab watch 查看可视化结果"
echo "============================================"
