import argparse
import torch
import numpy as np
from sram_dataset import performat_SramDataset
from downstream_train import downstream_train
import os
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RCG")
    # Task setting
    parser.add_argument("--task_level", type=str, default="edge", help="Task level. 'node' or 'edge'.")
    parser.add_argument("--task", type=str, default="regression", help="Task type. 'classification' or 'regression'.")
    
    # Data type setting: c (capacitance) or r (resistance)
    parser.add_argument("--data_type", type=str, default="c", choices=['c', 'r'],
                        help="数据类型: c(电容/capacitance) 或 r(电阻/resistance)")
    
    # Dataset setting
    parser.add_argument("--data_dir", type=str, default=None, help="Root directory containing dataset files. Auto-set based on data_type if not specified.")
    parser.add_argument("--train_dataset", type=str, default="ssram+digtime+timing_ctrl", help="Dataset names for training (e.g., sandwich+ultra8t)")
    parser.add_argument("--test_dataset", type=str, default="sandwich+ultra8t+array_128_32_8t", help="Dataset names for testing")
    parser.add_argument('--neg_edge_ratio', type=float, default=0.0, help='The ratio of negative edges.')
    parser.add_argument('--net_only', type=int, default=1, help='Only use net nodes for node level task or not.')

    # Graph sampling setting
    parser.add_argument("--small_dataset_sample_rates", type=float, default=0.7, help="The sample rate for small dataset.")
    parser.add_argument("--large_dataset_sample_rates", type=float, default=0.1, help='Target edge num of large dataset.')
    parser.add_argument("--num_hops", type=int, default=4, help="Number of hops in subgraph sampling.")
    parser.add_argument('--num_neighbors', type=int, default=64, help='The number of neighbors in subgraph sampling.')
    
    # Training setting
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument("--num_workers", type=int, default=0, help="The number of workers in data loaders.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index. Default: -1, using cpu.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="The batch size.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")

    ## Downstream GNN setting
    parser.add_argument("--model", type=str, default='gcn', 
        choices=['clustergcn', 'resgatedgcn', 'gat', 'gcn', 'sage', 'gine', 'pna',
                 'sgformer', 'polynormer', 'CustomGatedGCN', 'CustomGCNConv', 'CustomGINEConv'],
        help="The gnn model.")
    parser.add_argument("--num_gnn_layers", type=int, default=4, help="Number of GNN layers.")
    parser.add_argument("--num_head_layers", type=int, default=2, help="Number of head layers.")
    parser.add_argument("--hid_dim", type=int, default=144, help="Hidden layer dim.")
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout for neural networks.')
    parser.add_argument('--use_bn', type=int, default=1, help='0 or 1. Batch norm for neural networks.')
    parser.add_argument('--act_fn', default='leakyrelu', choices=['relu', 'elu', 'tanh', 'leakyrelu', 'prelu'], help='Activation function')
    parser.add_argument('--use_stats', type=int, default=1, help='0 or 1. Circuit statistics features.')

    # Custom layer settings
    parser.add_argument('--residual', type=int, default=1, help='Whether to use residuals in custom layers')
    parser.add_argument('--ffn', type=int, default=1, help='Whether to use ffn in custom layers')
    
    # SGFormer/Polynormer settings
    parser.add_argument("--num_heads", type=int, default=2, help='The number of heads for SGFormer')
    parser.add_argument("--global_layers", type=int, default=2, help='Global layers for Polynormer')

    # Regression setting
    parser.add_argument('--src_dst_agg', type=str, default='concat',
        choices=['concat', 'add', 'pooladd', 'poolmean', 'globalattn'], help="The way to aggregate nodes.")
    parser.add_argument("--regress_loss", type=str, default='mse', 
        choices=['mse', 'gai', 'bmc', 'bni', 'lds'], help="The loss function for edge regression.")
    
    # Classification setting
    parser.add_argument('--class_loss', type=str, default='cross_entropy',
        choices=['bsmCE', 'focal', 'cross_entropy'], help='The loss function for classification.')
    parser.add_argument('--num_classes', type=int, default=2, help='The number of classes.')
    parser.add_argument('--class_boundaries', type=list, default=[0.6], help='The boundaries for classification. 0.6 is optimal for global percentile normalization.')

    # Balanced MSE setting
    parser.add_argument("--noise_sigma", type=float, default=0.001, help="The simga_noise of Balanced MSE.")
    
    # LDS setting
    parser.add_argument('--lds_kernel', type=str, default='gaussian', choices=['gaussian', 'triang', 'laplace'])
    parser.add_argument('--lds_ks', type=int, default=9, help='LDS kernel size')
    parser.add_argument('--lds_sigma', type=float, default=0.02, help='LDS kernel sigma')
    
    # t-SNE visualization setting
    parser.add_argument('--plot_tsne', type=int, default=0,  help='Whether to plot t-SNE visualization of train/val/test splits.')
    parser.add_argument('--tsne_max_samples', type=int, default=100000, help='Max samples per graph for t-SNE (default: 10000, set to -1 for all nodes).')
    parser.add_argument('--tsne_perplexity', type=int, default=30, help='Perplexity for t-SNE.')
    
    parser.add_argument('--log_dir', type=str, default='logs', help='The directory to save the log file.')

    args = parser.parse_args()

    # 根据 data_type 自动设置数据目录
    if args.data_dir is None:
        if args.data_type == 'r':
            args.data_dir = '../sram_r/'
        else:
            args.data_dir = '../sram/'
    
    print(f"数据类型: {'电阻(resistance)' if args.data_type == 'r' else '电容(capacitance)'}")
    print(f"数据目录: {args.data_dir}")

    # Syncronize all random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(args.gpu))
        print('Using GPU: {}'.format(args.gpu))
    else:
        device = torch.device("cpu")

    print(f"============= PID = {os.getpid()} ============= ")
    print(args)

    # Load Dataset
    dataset = performat_SramDataset(
        dataset_dir=args.data_dir,
        train_cases=args.train_dataset,
        test_cases=args.test_dataset,
        neg_edge_ratio=args.neg_edge_ratio,
        to_undirected=True,
        small_dataset_sample_rates=args.small_dataset_sample_rates,
        large_dataset_sample_rates=args.large_dataset_sample_rates,
        task_level=args.task_level,
        net_only=args.net_only,
        class_boundaries=args.class_boundaries,
        data_type=args.data_type  # 新增: 传递数据类型
    )

    # Training
    downstream_train(args, dataset, device)
