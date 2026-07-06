#!/bin/bash
# 电阻边分类任务 - 二分类
# 训练集: ssram+digtime+timing_ctrl
# 测试集: sandwich+ultra8t+array_128_32_8t

# ============== 并行运行所有任务 ==============

# RCG 项目 (6种模型)
(cd rcg && python main.py --task classification --task_level edge --data_type r --net_only 1 --num_classes 2 --model pna --lr 0.0001 --batch_size 128 --num_gnn_layers 4 --num_head_layers 4 --num_heads 2 --dropout 0.3 --hid_dim 144 --num_workers 0 --gpu 0) &
(cd rcg && python main.py --task classification --task_level edge --data_type r --net_only 1 --num_classes 2 --model gcn --lr 0.0001 --batch_size 128 --num_gnn_layers 4 --num_head_layers 4 --num_heads 2 --dropout 0.3 --hid_dim 144 --num_workers 0 --gpu 0) &
(cd rcg && python main.py --task classification --task_level edge --data_type r --net_only 1 --num_classes 2 --model gat --lr 0.0001 --batch_size 128 --num_gnn_layers 4 --num_head_layers 4 --num_heads 2 --dropout 0.3 --hid_dim 144 --num_workers 0 --gpu 0) &
(cd rcg && python main.py --task classification --task_level edge --data_type r --net_only 1 --num_classes 2 --model sage --lr 0.0001 --batch_size 128 --num_gnn_layers 4 --num_head_layers 4 --num_heads 2 --dropout 0.3 --hid_dim 144 --num_workers 0 --gpu 1) &
(cd rcg && python main.py --task classification --task_level edge --data_type r --net_only 1 --num_classes 2 --model sgformer --lr 0.0001 --batch_size 128 --num_gnn_layers 4 --num_head_layers 4 --num_heads 2 --dropout 0.3 --hid_dim 144 --num_workers 0 --gpu 1) &
(cd rcg && python main.py --task classification --task_level edge --data_type r --net_only 1 --num_classes 2 --model polynormer --lr 0.0001 --batch_size 128 --num_gnn_layers 4 --num_head_layers 4 --num_heads 2 --dropout 0.3 --hid_dim 144 --num_workers 0 --gpu 1) &

# Paragraph 项目
(cd paragraph && python main.py --task classification --task_level edge --data_type r --net_only 1 --num_classes 2 --class_boundaries 0.5 --lr 0.0001 --batch_size 128 --num_gnn_layers 4 --num_head_layers 2 --dropout 0.4 --hid_dim 64 --num_workers 0 --gpu 2) &

# Cirgps 项目
(cd Cirgps && python main.py --task classification --task_level edge --data_type r --net_only 1 --num_classes 2 --class_boundaries 0.5 --lr 0.0001 --batch_size 128 --num_gnn_layers 4 --num_head_layers 2 --dropout 0.3 --hid_dim 144 --num_workers 0 --gpu 2) &

# CircuitGCL 项目
(cd CircuitGCL && python main.py --task classification --task_level edge --data_type r --net_only 1 --num_classes 2 --lr 0.0001 --batch_size 128 --num_gnn_layers 4 --num_head_layers 2 --dropout 0.3 --hid_dim 144 --num_workers 0 --gpu 2) &

# 等待所有后台任务完成
wait

echo "所有任务完成!"
