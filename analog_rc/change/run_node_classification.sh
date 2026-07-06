#!/bin/bash

# 节点分类任务 - 7种模型
# 日志保存在 sh_logs/node_classification/

mkdir -p sh_logs/node_classification

echo "========== 节点分类任务 =========="

# SAGE
echo "[1/7] Running SAGE..."
python main.py --task_level node --task classification --model sage --train_cases "1+5+7" --test_cases "15+17" --num_classes 5 --hid_dim 128 --num_layers 4 --dropout 0.3 --lr 0.00005 --epochs 200 --batch_size 64 --num_neighbors 128 --num_hops 4 --activation prelu --gpu 0 > sh_logs/node_classification/sage.log 2>&1
echo "SAGE completed"

# GCN
echo "[2/7] Running GCN..."
python main.py --task_level node --task classification --model gcn --train_cases "1+5+7" --test_cases "15+17" --num_classes 5 --hid_dim 128 --num_layers 4 --dropout 0.3 --lr 0.00005 --epochs 200 --batch_size 64 --num_neighbors 128 --num_hops 4 --activation prelu --gpu 0 > sh_logs/node_classification/gcn.log 2>&1
echo "GCN completed"

# GAT
echo "[3/7] Running GAT..."
python main.py --task_level node --task classification --model gat --train_cases "1+5+7" --test_cases "15+17" --num_classes 5 --hid_dim 128 --num_layers 4 --dropout 0.3 --lr 0.00005 --epochs 200 --batch_size 64 --num_neighbors 128 --num_hops 4 --activation prelu --gpu 0 > sh_logs/node_classification/gat.log 2>&1
echo "GAT completed"

# GINE
echo "[4/7] Running GINE..."
python main.py --task_level node --task classification --model gine --train_cases "1+5+7" --test_cases "15+17" --num_classes 5 --hid_dim 128 --num_layers 4 --dropout 0.3 --lr 0.00005 --epochs 200 --batch_size 64 --num_neighbors 128 --num_hops 4 --activation prelu --gpu 0 > sh_logs/node_classification/gine.log 2>&1
echo "GINE completed"

# PNA
echo "[5/7] Running PNA..."
python main.py --task_level node --task classification --model pna --train_cases "1+5+7" --test_cases "15+17" --num_classes 5 --hid_dim 128 --num_layers 4 --dropout 0.3 --lr 0.00005 --epochs 200 --batch_size 64 --num_neighbors 128 --num_hops 4 --activation prelu --pna_towers 4 --gpu 0 > sh_logs/node_classification/pna.log 2>&1
echo "PNA completed"

# SGFormer
echo "[6/7] Running SGFormer..."
python main.py --task_level node --task classification --model sgformer --train_cases "1+5+7" --test_cases "15+17" --num_classes 5 --hid_dim 128 --num_layers 4 --dropout 0.3 --lr 0.00005 --epochs 200 --batch_size 64 --num_neighbors 128 --num_hops 4 --activation prelu --trans_num_layers 2 --trans_num_heads 1 --trans_dropout 0.5 --gnn_num_layers 3 --gnn_dropout 0.5 --graph_weight 0.5 --gpu 0 > sh_logs/node_classification/sgformer.log 2>&1
echo "SGFormer completed"

# Polynormer
echo "[7/7] Running Polynormer..."
python main.py --task_level node --task classification --model polynormer --train_cases "1+5+7" --test_cases "15+17" --num_classes 5 --hid_dim 128 --num_layers 4 --dropout 0.3 --lr 0.00005 --epochs 200 --batch_size 64 --num_neighbors 128 --num_hops 4 --activation prelu --local_layers 7 --global_layers 2 --in_dropout 0.15 --global_dropout 0.5 --poly_heads 1 --beta 0.9 --gpu 0 > sh_logs/node_classification/polynormer.log 2>&1
echo "Polynormer completed"

echo "========== 节点分类任务全部完成 =========="
