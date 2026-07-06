"""
Main entry point for RC graph training.

Usage:
    # 节点任务 - 回归
    python main.py --task_level node --task regression --train_cases "1+5+7" --test_cases "15+17"
    
    # 节点任务 - 分类
    python main.py --task_level node --task classification --train_cases "1+5+7" --test_cases "15+17"
    
    # 边任务 - 回归
    python main.py --task_level edge --task regression --train_cases "1+5+7" --test_cases "15+17"
    
    # 自定义分类边界 (默认 0.2,0.4,0.6,0.8 → 5类)
    python main.py --task classification --class_boundaries "0.33,0.67"  # 3类
"""
import argparse
import torch
import numpy as np
import random
import os
import logging
from datetime import datetime

from dataset import RCDataset
from dataloader import create_dataloaders
from model import GNNModel
from train import train_model


def setup_main_logger(args):
    """设置主日志记录器"""
    log_dir = f'logs/{args.task_level}_{args.task}'
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'{log_dir}/{args.model}_{timestamp}.log'
    
    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger, log_file


def parse_args():
    parser = argparse.ArgumentParser(description="RC Graph Training")
    
    # Dataset
    parser.add_argument("--data_dir", type=str, default="../data/", 
                        help="Directory containing .pt files")
    parser.add_argument("--train_cases", type=str, default="1+2+3+6+8+9+10+11+12+15+16+17+18", 
                        help="Case IDs for training (e.g., '17')")
    parser.add_argument("--test_cases", type=str, default="5+14+20", 
                        help="Case IDs for testing (e.g., '15+17')")
    parser.add_argument("--no_cache", type=int, default=0, help="Disable data caching (force reprocess)")
    
    # Task
    parser.add_argument("--task_level", type=str, default="node", choices=["node", "edge"], 
                        help="Task level: 'node' for net node prediction, 'edge' for pin-pair_to-pin edge prediction")
    parser.add_argument("--task", type=str, default="regression", choices=["regression", "classification"])
    parser.add_argument("--num_classes", type=int, default=5, help="Number of classes for classification") #只需要改这个，分界会自动改
    parser.add_argument("--class_boundaries", type=str, default="",
                        help="Classification boundaries for normalized labels (0-1), e.g. '0.33,0.67'. If empty, auto-generate from num_classes")
    
    # Model
    parser.add_argument("--model", type=str, default="sage", 
                        choices=["gcn", "sage", "gat", "gine", "pna", "CustomGatedGCN", "CustomGCNConv", "CustomGINEConv", "sgformer", "polynormer"],
                        help="GNN model type")
    parser.add_argument("--activation", type=str, default="prelu", 
                        choices=["relu", "elu", "tanh", "leakyrelu", "prelu"], 
                        help="Activation function")
    parser.add_argument("--use_node_attr", type=int, default=1, 
                        help="Use node attributes in addition to type embedding")
    parser.add_argument("--hid_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of GNN layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    
    # PNA 参数
    parser.add_argument("--pna_towers", type=int, default=2, help="Number of towers (for pna)")
    
    # SGFormer 参数
    parser.add_argument("--trans_num_layers", type=int, default=2, help="Number of transformer layers (for sgformer)")
    parser.add_argument("--trans_num_heads", type=int, default=1, help="Number of transformer heads (for sgformer)")
    parser.add_argument("--trans_dropout", type=float, default=0.5, help="Transformer dropout (for sgformer)")
    parser.add_argument("--gnn_num_layers", type=int, default=3, help="Number of GNN layers in SGFormer")
    parser.add_argument("--gnn_dropout", type=float, default=0.5, help="GNN dropout (for sgformer)")
    parser.add_argument("--graph_weight", type=float, default=0.5, help="Weight balance global and gnn (for sgformer)")
    parser.add_argument("--aggregate", type=str, default="add", help="Aggregate type (for sgformer)")
    
    # Polynormer 参数
    parser.add_argument("--local_layers", type=int, default=7, help="Number of local attention layers (for polynormer)")
    parser.add_argument("--global_layers", type=int, default=2, help="Number of global attention layers (for polynormer)")
    parser.add_argument("--in_dropout", type=float, default=0.15, help="Input dropout (for polynormer)")
    parser.add_argument("--global_dropout", type=float, default=0.5, help="Global dropout (for polynormer)")
    parser.add_argument("--poly_heads", type=int, default=1, help="Number of attention heads (for polynormer)")
    parser.add_argument("--beta", type=float, default=0.9, help="Aggregate type beta (for polynormer)")
    parser.add_argument("--local_attn", type=int, default=0, help="Whether use local attention (for polynormer)")
    
    # Training
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.00005, help="Learning rate")
    parser.add_argument("--num_neighbors", type=int, default=128, help="Number of neighbors per hop")
    parser.add_argument("--num_hops", type=int, default=4, help="Number of hops")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    
    # Sampling
    parser.add_argument("--edge_sample_rate", type=float, default=0.6, 
                        help="Edge sampling rate for edge-level tasks (0.0-1.0, default 1.0 = use all edges)")
    parser.add_argument("--node_sample_rate", type=float, default=1.0,
                        help="Node sampling rate for node-level tasks (0.0-1.0, default 1.0 = use all nodes)")
    
    # Loss function
    # 回归: mse, mae, weighted_mse, weighted_mae, focal_mse, huber
    # 分类: ce, weighted_ce, focal, label_smoothing
    parser.add_argument("--reg_loss", type=str, default="mse",
                        choices=["mse", "mae", "weighted_mse", "weighted_mae", "focal_mse", "huber"],
                        help="Loss for regression task")
    parser.add_argument("--cls_loss", type=str, default="ce",
                        choices=["ce", "weighted_ce", "focal", "label_smoothing"],
                        help="Loss for classification task")
    # --loss_alpha: 用于 weighted_mse, weighted_mae
    #   weight = 1 + alpha * y，alpha越大高值样本权重越大，建议范围 0.5~5.0
    parser.add_argument("--loss_alpha", type=float, default=1.0, help="Alpha for weighted loss")
    # --loss_gamma: 用于 focal_mse, focal
    #   gamma越大越关注难样本（预测误差大的样本），建议范围 1.0~5.0
    parser.add_argument("--loss_gamma", type=float, default=2.0, help="Gamma for focal loss")
    # --loss_delta: 用于 huber
    #   误差<delta用MSE，>delta用MAE，对异常值鲁棒，建议范围 0.1~2.0
    parser.add_argument("--loss_delta", type=float, default=1.0, help="Delta for huber loss")
    # --label_smoothing: 用于 label_smoothing
    #   平滑系数，0表示不平滑，0.1表示10%概率分给其他类，建议范围 0.05~0.2
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing factor")
    
    # Normalization
    parser.add_argument("--normalize", type=str, default="log", choices=["minmax", "log"],
                        help="Label normalization method: minmax (y/700) or log (log(1+y)/log(701))")
    
    # Device
    parser.add_argument("--gpu", type=int, default=1, help="GPU index (-1 for CPU)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # 解析分类边界并设置
    # 如果用户指定了 class_boundaries，使用它；否则根据 num_classes 自动生成均匀边界
    if args.class_boundaries:
        class_boundaries = [float(x.strip()) for x in args.class_boundaries.split(',')]
        args.num_classes = len(class_boundaries) + 1
    else:
        # 根据 num_classes 自动生成均匀边界
        class_boundaries = [i / args.num_classes for i in range(1, args.num_classes)]
    RCDataset.DEFAULT_CLASS_BOUNDARIES = class_boundaries
    
    # 设置归一化方式
    RCDataset.NORMALIZE_METHOD = args.normalize
    
    logger, log_file = setup_main_logger(args)
    
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    
    logger.info(f"{'=' * 60}")
    logger.info(f"RC Graph Training")
    logger.info(f"{'=' * 60}")
    logger.info(f"日志保存到: {log_file}")
    logger.info(f"Using device: {device}")
    logger.info(f"Task: {args.task_level} level {args.task}")
    logger.info(f"归一化方式: {args.normalize}")
    logger.info(f"损失函数: {args.reg_loss if args.task == 'regression' else args.cls_loss}")
    if args.task == 'classification':
        logger.info(f"分类边界: {class_boundaries} ({args.num_classes} 类)")
    logger.info(f"")
    logger.info(f"训练配置:")
    logger.info(f"  模型: {args.model}")
    logger.info(f"  激活函数: {args.activation}")
    logger.info(f"  隐藏维度: {args.hid_dim}")
    logger.info(f"  层数: {args.num_layers}")
    logger.info(f"  Dropout: {args.dropout}")
    logger.info(f"  学习率: {args.lr}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch Size: {args.batch_size}")
    logger.info(f"  Num Neighbors: {args.num_neighbors}")
    logger.info(f"  Num Hops: {args.num_hops}")
    if args.task_level == 'edge':
        logger.info(f"  边采样率: {args.edge_sample_rate:.2%}")
    else:
        logger.info(f"  节点采样率: {args.node_sample_rate:.2%}")
    logger.info(f"{'=' * 60}")
    
    # Load datasets
    logger.info(f"\n{'=' * 60}")
    logger.info(f"数据预处理")
    logger.info(f"{'=' * 60}")
    
    train_ids = [c.strip() for c in args.train_cases.split('+') if c.strip()]
    test_ids = [c.strip() for c in args.test_cases.split('+') if c.strip()]
    
    logger.info(f"训练集 cases: {train_ids}")
    logger.info(f"测试集 cases: {test_ids}")
    logger.info(f"")
    
    # 加载训练图
    logger.info(f"--- 加载训练数据 ---")
    train_graphs = []
    total_train_nodes = 0
    total_train_edges = 0
    total_train_targets = 0
    use_cache = not args.no_cache
    
    for cid in train_ids:
        filepath = os.path.join(args.data_dir, f"case{cid}_RC.pt")
        if os.path.exists(filepath):
            logger.info(f"  加载 case{cid}_RC.pt ...")
            graph = RCDataset.load_and_process(filepath, args.task_level, use_cache=use_cache)
            train_graphs.append(graph)
            
            if args.task_level == 'node':
                target_count = graph.target_node_mask.sum().item() if hasattr(graph, 'target_node_mask') else 0
                valid_count = graph.train_node_mask.sum().item() if hasattr(graph, 'train_node_mask') else 0
                logger.info(f"    Case {cid}: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges")
                logger.info(f"             net节点: {target_count}, 有效训练节点(0-700): {valid_count}")
                total_train_targets += valid_count
            else:
                target_count = graph.edge_label_y.size(0) if hasattr(graph, 'edge_label_y') else 0
                logger.info(f"    Case {cid}: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges")
                logger.info(f"             有效目标边(pair_to且0-700): {target_count}")
                total_train_targets += target_count
            
            total_train_nodes += graph.num_nodes
            total_train_edges += graph.edge_index.shape[1]
        else:
            logger.warning(f"    Warning: {filepath} not found, skipping")
    
    logger.info(f"")
    logger.info(f"  训练集统计:")
    logger.info(f"    总节点数: {total_train_nodes}")
    logger.info(f"    总边数: {total_train_edges}")
    logger.info(f"    总目标数: {total_train_targets}")
    
    # 加载测试图
    logger.info(f"")
    logger.info(f"--- 加载测试数据 ---")
    test_graphs = []
    total_test_nodes = 0
    total_test_edges = 0
    total_test_targets = 0
    
    for cid in test_ids:
        filepath = os.path.join(args.data_dir, f"case{cid}_RC.pt")
        if os.path.exists(filepath):
            logger.info(f"  加载 case{cid}_RC.pt ...")
            graph = RCDataset.load_and_process(filepath, args.task_level, use_cache=use_cache)
            test_graphs.append((cid, graph))
            
            if args.task_level == 'node':
                target_count = graph.target_node_mask.sum().item() if hasattr(graph, 'target_node_mask') else 0
                valid_count = graph.train_node_mask.sum().item() if hasattr(graph, 'train_node_mask') else 0
                logger.info(f"    Case {cid}: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges")
                logger.info(f"             net节点: {target_count}, 有效测试节点(0-700): {valid_count}")
                total_test_targets += valid_count
            else:
                target_count = graph.edge_label_y.size(0) if hasattr(graph, 'edge_label_y') else 0
                logger.info(f"    Case {cid}: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges")
                logger.info(f"             有效目标边(pair_to且0-700): {target_count}")
                total_test_targets += target_count
            
            total_test_nodes += graph.num_nodes
            total_test_edges += graph.edge_index.shape[1]
        else:
            logger.warning(f"    Warning: {filepath} not found, skipping")
    
    logger.info(f"")
    logger.info(f"  测试集统计:")
    logger.info(f"    总节点数: {total_test_nodes}")
    logger.info(f"    总边数: {total_test_edges}")
    logger.info(f"    总目标数: {total_test_targets}")
    
    if not train_graphs:
        raise ValueError("No training graphs loaded!")
    
    # 如果使用 PNA，需要预计算 degree histogram
    if args.model == 'pna':
        from model import compute_degree_histogram
        logger.info(f"")
        logger.info(f"--- 计算 PNA degree histogram ---")
        deg = compute_degree_histogram(train_graphs)
        args.deg = deg
        logger.info(f"  Max degree: {len(deg) - 1}")
        logger.info(f"  Degree histogram 计算完成")

    # Create dataloaders
    logger.info(f"")
    logger.info(f"--- 创建 DataLoaders ---")
    train_loader, val_loader, test_loaders = create_dataloaders(
        train_graphs, test_graphs, args
    )
    logger.info(f"  DataLoaders 创建完成")
    
    # Create model
    logger.info(f"")
    logger.info(f"--- 创建模型 ---")
    model = GNNModel(args).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Model: {args.model.upper()}")
    logger.info(f"  Parameters: {num_params:,}")
    logger.info(f"{'=' * 60}")
    
    # Train
    logger.info(f"\n开始训练...")
    train_model(args, model, train_loader, val_loader, test_loaders, device)


if __name__ == "__main__":
    main()
