import argparse
import torch
import numpy as np
from sram_dataset import performat_SramDataset
from downstream_train import downstream_link_pred
import os
import random
import gc
import sys
import datetime

if __name__ == "__main__":
    # STEP 0: Parse Arguments ======================================================================= #
    parser = argparse.ArgumentParser(description="CircuitGPS_simple")
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument("--train_dataset", type=str, default="ssram+digtime+timing_ctrl", help="Dataset names for training (e.g., sandwich+ultra8t)")
    parser.add_argument("--test_dataset", type=str, default="sandwich+ultra8t+array_128_32_8t", help="Dataset names for testing")
    parser.add_argument("--data_dir", type=str, default=None, help="Root directory containing dataset files. Auto-set based on data_type if not specified.")
    parser.add_argument("--data_type", type=str, default="c", choices=['c', 'r'],
                        help="数据类型: c(电容/capacitance) 或 r(电阻/resistance)")
    parser.add_argument("--task", type=str, default="regression", help="Task type. 'classification' or 'regression'.")
    parser.add_argument("--task_level", type=str, default="edge", help="Task level. 'node' or 'edge'.")
    parser.add_argument("--net_only", type=int, default=1, help="Only use net nodes for node level task.")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes for classification.")
    parser.add_argument("--class_boundaries", type=str, default="0.6", help="Class boundaries for classification. 0.6 is optimal for global percentile normalization.")
    parser.add_argument("--max_dist", type=int, default=350, help="The max values in DSPD.")
    parser.add_argument("--num_workers", type=int, default=0, help="The number of workers in data loaders.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index. Default: -1, using cpu.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="The batch size.") 
    parser.add_argument("--lr", type=float, default=0.00005, help="Learning rate.")
    parser.add_argument("--num_gnn_layers", type=int, default=4, help="Number of GNN layers.")
    parser.add_argument("--num_head_layers", type=int, default=2, help="Number of head layers.")
    parser.add_argument("--hid_dim", type=int, default=64, help="Hidden layer dim.")
    parser.add_argument('--dropout', type=float, default=0.4, help='dropout for neural networks.')
    parser.add_argument('--use_bn', type=int, default=1, help='Batch norm for neural networks.')
    parser.add_argument('--act_fn', default='relu', help='Activation function')
    parser.add_argument('--src_dst_agg', default='concat', help='The way to aggregate nodes. Can be "concat" or "add" or "pool".')
    parser.add_argument('--num_hops',type=int,default=3,help='Number of hops.')
    parser.add_argument('--to_undirected', type=int, default=0, help='Whether to convert the graph to undirected graph.')
    parser.add_argument('--small_dataset_sample_rates', type=float, default=1.0, help='Sampling rate for small datasets (ssram, digtime, timing_ctrl, array_128_32_8t).')
    parser.add_argument('--large_dataset_sample_rates', type=float, default=0.1, help='Sampling rate for large datasets (sandwich, ultra8t).')
    parser.add_argument('--use_ensemble', type=int, default=0, help='Whether to use ensemble model for predictions.')
    parser.add_argument('--num_ensemble', type=int, default=3, help='Number of models in the ensemble.')
    parser.add_argument('--ensemble_thresholds', type=str, default='0.33,0.66', 
                        help='Comma-separated max prediction thresholds for Algorithm 2 ensemble strategy.')
    args = parser.parse_args()
    
    # 解析class_boundaries
    args.class_boundaries = [float(x) for x in args.class_boundaries.split(',')]
    
    # 根据 data_type 自动设置数据目录
    if args.data_dir is None:
        if args.data_type == 'r':
            args.data_dir = '../sram_r/'
        else:
            args.data_dir = '../sram/'
    
    print(f"数据类型: {'电阻(resistance)' if args.data_type == 'r' else '电容(capacitance)'}")
    print(f"数据目录: {args.data_dir}")
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.use_ensemble:
        args.ensemble_thresholds = [float(x) for x in args.ensemble_thresholds.split(',')]
        print(f"Using ensemble model with {args.num_ensemble} models and thresholds: {args.ensemble_thresholds}")

    if args.gpu != -1 and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(args.gpu))
        # 清理GPU缓存
        torch.cuda.empty_cache()
        # 限制GPU显存使用为12GB（一半）
        torch.cuda.set_per_process_memory_fraction(0.5, device=args.gpu)
        print(f'Using GPU: {args.gpu}, 已限制GPU显存使用为50%')
    else:
        device = torch.device("cpu")
        
    # 启用垃圾回收和设置较低的阈值
    gc.enable()
    gc.set_threshold(100, 5, 5)
    gc.collect()

    # 优化batch size，避免内存问题
    if args.batch_size > 128:
        print(f"Warning: 降低batch_size从{args.batch_size}到128以避免内存问题")
        args.batch_size = 128
    
    # 创建日志目录和文件
    os.makedirs("./logs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"./logs/{args.train_dataset}_to_{args.test_dataset}_{timestamp}.txt"
    
    # 重定向stdout到日志文件
    log_file = open(log_filename, 'w')
    original_stdout = sys.stdout
    
    # 使用Tee类来同时输出到控制台和文件
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    sys.stdout = Tee(original_stdout, log_file)
    
    print(f"日志文件已创建：{log_filename}")
    print(f"============= PID = {os.getpid()} ============= ")
    print("参数配置:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    # 根据数据集名称确定采样率
    train_names = args.train_dataset.split('+')
    train_sample_rates = [
        args.large_dataset_sample_rates if name in ['sandwich', 'ultra8t'] 
        else args.small_dataset_sample_rates 
        for name in train_names
    ]
    
    test_names = args.test_dataset.split('+')
    test_sample_rates = [
        args.large_dataset_sample_rates if name in ['sandwich', 'ultra8t'] 
        else args.small_dataset_sample_rates 
        for name in test_names
    ]
    
    train_dataset = performat_SramDataset(
        name=args.train_dataset, 
        dataset_dir=args.data_dir, 
        neg_edge_ratio=0.0,
        to_undirected=args.to_undirected,
        sample_rates=train_sample_rates,
        task_type=args.task,
        task_level=args.task_level,
        net_only=args.net_only,
        class_boundaries=args.class_boundaries,
        data_type=args.data_type,  # 新增: 传递数据类型
    )
    
    test_dataset = performat_SramDataset(
        name=args.test_dataset, 
        dataset_dir=args.data_dir, 
        neg_edge_ratio=0.0,
        to_undirected=args.to_undirected,
        sample_rates=test_sample_rates,
        task_type=args.task,
        task_level=args.task_level,
        net_only=args.net_only,
        class_boundaries=args.class_boundaries,
        data_type=args.data_type,  # 新增: 传递数据类型
    )
    
    dataset = {
        'train': train_dataset,
        'test': test_dataset
    }

    if args.task_level == 'node':
        from downstream_train import downstream_node_pred
        downstream_node_pred(args, dataset, device)
    else:
        downstream_link_pred(args, dataset, device)
    
    # 恢复标准输出并关闭日志文件
    sys.stdout = original_stdout
    log_file.close()
    print(f"所有输出已保存到 {log_filename}")