#!/bin/bash

# 边回归任务 - 7种模型并行执行
# 日志保存在 sh_logs/edge_regression/

mkdir -p sh_logs/edge_regression

echo "========== 边回归任务 (并行执行) =========="
echo "启动时间: $(date '+%Y-%m-%d %H:%M:%S')"

# 设置环境变量禁用进度条，避免日志文件过大
export TQDM_DISABLE=1

# 并行启动7个模型，每个使用不同的GPU

# SAGE (GPU 0)
echo "[1/7] Starting SAGE..."
TQDM_DISABLE=1 python main.py --task_level edge --edge_sample_rate 0.6 --task regression --model sage --train_cases "1+5+7+10+11+15+23+29+39+42+44+45+55+58+71+72+74+75+78" --test_cases "15+17+23" --hid_dim 96 --num_layers 4 --dropout 0.4 --lr 0.00005 --epochs 200 --batch_size 64 --num_neighbors 64 --num_hops 3 --activation leakyrelu --gpu 0 > sh_logs/edge_regression/sage.log 2>&1 &
PID_SAGE=$!

# GCN (GPU 0)
echo "[2/7] Starting GCN..."
TQDM_DISABLE=1 python main.py --task_level edge --edge_sample_rate 0.6 --task regression --model gcn --train_cases "1+5+7+10+11+15+23+29+39+42+44+45+55+58+71+72+74+75+78" --test_cases "15+17+23" --hid_dim 96 --num_layers 4 --dropout 0.4 --lr 0.00005 --epochs 200 --batch_size 64 --num_neighbors 64 --num_hops 3 --activation leakyrelu --gpu 0 > sh_logs/edge_regression/gcn.log 2>&1 &
PID_GCN=$!

# GAT (GPU 1)
echo "[3/7] Starting GAT..."
TQDM_DISABLE=1 python main.py --task_level edge --edge_sample_rate 0.6 --task regression --model gat --train_cases "1+5+7+10+11+15+23+29+39+42+44+45+55+58+71+72+74+75+78" --test_cases "15+17+23" --hid_dim 96 --num_layers 4 --dropout 0.4 --lr 0.00005 --epochs 200 --batch_size 64 --num_neighbors 64 --num_hops 3 --activation leakyrelu --gpu 1 > sh_logs/edge_regression/gat.log 2>&1 &
PID_GAT=$!

# # GINE (GPU 1)
# echo "[4/7] Starting GINE..."
# TQDM_DISABLE=1 python main.py --task_level edge --edge_sample_rate 0.6 --task regression --model gine --train_cases "1+5+7+10+11+15+23+29+39+42+44+45+55+58+71+72+74+75+78" --test_cases "15+17+23" --hid_dim 96 --num_layers 4 --dropout 0.4 --lr 0.00005 --epochs 200 --batch_size 64 --num_neighbors 64 --num_hops 3 --activation leakyrelu --gpu 1 > sh_logs/edge_regression/gine.log 2>&1 &
# PID_GINE=$!

# PNA (GPU 2)
echo "[5/7] Starting PNA..."
TQDM_DISABLE=1 python main.py --task_level edge --edge_sample_rate 0.6 --task regression --model pna --train_cases "1+5+7+10+11+15+23+29+39+42+44+45+55+58+71+72+74+75+78" --test_cases "15+17+23" --hid_dim 96 --num_layers 4 --dropout 0.4 --lr 0.00005 --epochs 200 --batch_size 64 --num_neighbors 64 --num_hops 3 --activation leakyrelu --pna_towers 4 --gpu 2 > sh_logs/edge_regression/pna.log 2>&1 &
PID_PNA=$!

# SGFormer (GPU 2)
echo "[6/7] Starting SGFormer..."
TQDM_DISABLE=1 python main.py --task_level edge --edge_sample_rate 0.6 --task regression --model sgformer --train_cases "1+5+7+10+11+15+23+29+39+42+44+45+55+58+71+72+74+75+78" --test_cases "15+17+23" --hid_dim 96 --num_layers 4 --dropout 0.4 --lr 0.00005 --epochs 200 --batch_size 64 --num_neighbors 64 --num_hops 3 --activation leakyrelu --trans_num_layers 2 --trans_num_heads 1 --trans_dropout 0.5 --gnn_num_layers 3 --gnn_dropout 0.5 --graph_weight 0.5 --gpu 2 > sh_logs/edge_regression/sgformer.log 2>&1 &
PID_SGFORMER=$!

# Polynormer (GPU 3)
echo "[7/7] Starting Polynormer..."
TQDM_DISABLE=1 python main.py --task_level edge --edge_sample_rate 0.6 --task regression --model polynormer --train_cases "1+5+7+10+11+15+23+29+39+42+44+45+55+58+71+72+74+75+78" --test_cases "15+17+23" --hid_dim 96 --num_layers 4 --dropout 0.4 --lr 0.00005 --epochs 200 --batch_size 64 --num_neighbors 64 --num_hops 3 --activation leakyrelu --local_layers 7 --global_layers 2 --in_dropout 0.15 --global_dropout 0.5 --poly_heads 1 --beta 0.9 --gpu 3 > sh_logs/edge_regression/polynormer.log 2>&1 &
PID_POLYNORMER=$!

echo ""
echo "所有任务已启动，等待完成..."
echo "进程ID: SAGE=$PID_SAGE, GCN=$PID_GCN, GAT=$PID_GAT, GINE=$PID_GINE, PNA=$PID_PNA, SGFormer=$PID_SGFORMER, Polynormer=$PID_POLYNORMER"
echo "日志文件大小会更小（已禁用进度条）"
echo ""

# 等待所有后台任务完成
wait $PID_SAGE
echo "[✓] SAGE completed"

wait $PID_GCN
echo "[✓] GCN completed"

wait $PID_GAT
echo "[✓] GAT completed"

wait $PID_GINE
echo "[✓] GINE completed"

wait $PID_PNA
echo "[✓] PNA completed"

wait $PID_SGFORMER
echo "[✓] SGFormer completed"

wait $PID_POLYNORMER
echo "[✓] Polynormer completed"

echo ""
echo "========== 边回归任务全部完成 =========="
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
