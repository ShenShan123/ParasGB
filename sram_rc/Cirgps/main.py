import argparse
import torch
import numpy as np
from sram_dataset import performat_SramDataset, adaption_for_sgrl
from downstream_train import downstream_train
import os
import random
import sys
import gc
import datetime
import psutil

if __name__ == "__main__":
    # STEP 0: Parse Arguments ======================================================================= #
    parser = argparse.ArgumentParser(description="CircuitGPS_simple")
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument("--train_dataset", type=str, default="ssram+digtime+timing_ctrl", help="Dataset names for training (e.g., sandwich+ultra8t)")
    parser.add_argument("--test_dataset", type=str, default="sandwich+ultra8t+array_128_32_8t", help="Dataset names for testing")
    parser.add_argument("--data_dir", type=str, default=None, help="Root directory containing dataset files. Auto-set based on data_type if not specified.")
    parser.add_argument("--data_type", type=str, default="c", choices=['c', 'r'],
                        help="数据类型: c(电容/capacitance) 或 r(电阻/resistance)")
    parser.add_argument("--add_tar_edge", type=int, default=0, help="0 or 1. Inject target edges into the graph.")
    parser.add_argument("--small_dataset_sample_rates", type=float, default=1.0, help="Sampling rate for small datasets (ssram, digtime, timing_ctrl, array_128_32_8t).")
    parser.add_argument("--large_dataset_sample_rates", type=float, default=0.1, help="Sampling rate for large datasets (sandwich, ultra8t).")
    parser.add_argument('--u', type=float, default=0.5, help='Parameter u for experimentation.')
    # parser.add_argument('--num_layers', type=int, default=1, help='num_layers')
    # parser.add_argument('--num_hop', type=int, default=1, help='num_hop')
    # parser.add_argument('--trials', type=int, default=20, help='trials')
    # CirGPS arguments
    parser.add_argument("--task", type=str, default="regression", help="Task type. 'classification' or 'regression'.")
    parser.add_argument("--task_level", type=str, default="edge", help="Task level. 'node' or 'edge'.")
    parser.add_argument("--net_only", type=int, default=1, help="0 or 1. Only use net nodes for node task.")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of classes for classification.")
    parser.add_argument("--class_boundaries", type=str, default="0.6", help="Class boundaries for classification. 0.6 is optimal for global percentile normalization.")
    parser.add_argument("--use_pe", type=int, default=1, help="Positional encoding. Defualt: True.")
    parser.add_argument("--num_hops", type=int, default=3, help="Number of hops in subgraph sampling.")
    parser.add_argument("--max_dist", type=int, default=350, help="The max values in DSPD.")
    parser.add_argument("--num_workers", type=int, default=0, help="The number of workers in data loaders.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index. Default: -1, using cpu.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="The batch size.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--model", type=str, default='clustergcn', help="The gnn model. Could be 'clustergcn', 'resgatedgcn', 'gat', 'gcn', 'sage', 'gine'.")
    parser.add_argument("--num_gnn_layers", type=int, default=4, help="Number of GNN layers.")
    parser.add_argument("--num_head_layers", type=int, default=2, help="Number of head layers.")
    parser.add_argument("--hid_dim", type=int, default=144, help="Hidden layer dim.")
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout for neural networks.')
    parser.add_argument('--use_bn', type=int, default=1, help='0 or 1. Batch norm for neural networks.')
    parser.add_argument('--act_fn', default='relu', help='Activation function')
    parser.add_argument('--src_dst_agg', type=str, default='concat', help='The way to aggregate nodes. Can be `concat` or `add` or `pooladd` or `poolmean`.')
    parser.add_argument('--use_stats', type=int, default=0, help='0 or 1. Circuit statistics features.')
    # 内存管理参数
    parser.add_argument('--mem_fraction', type=float, default=0.5, help='GPU显存使用比例限制 (0-1)')
    parser.add_argument('--pe_chunk_size', type=int, default=2000, help='PE计算时的分块大小')
    parser.add_argument('--skip_pe_calculation', type=int, default=0, help='是否跳过PE计算 (0或1)')
    parser.add_argument('--low_memory_mode', type=int, default=0, help='启用低内存模式 (0或1)')
    args = parser.parse_args()

    # 根据 data_type 自动设置数据目录
    if args.data_dir is None:
        if args.data_type == 'r':
            args.data_dir = '../sram_r/'
        else:
            args.data_dir = '../sram/'
    
    print(f"数据类型: {'电阻(resistance)' if args.data_type == 'r' else '电容(capacitance)'}")
    print(f"数据目录: {args.data_dir}")

    # 创建日志目录
    os.makedirs("./logs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"./logs/{args.train_dataset}_to_{args.test_dataset}_{timestamp}.txt"
    
    # 将所有print输出重定向到日志文件
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

    # 内存使用监控函数
    def print_memory_usage():
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_gb = memory_info.rss / (1024 ** 3)
        print(f"当前内存使用: {memory_gb:.2f} GB")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_mem_alloc = torch.cuda.memory_allocated(i) / (1024 ** 3)
                gpu_mem_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                print(f"GPU {i} 显存使用: {gpu_mem_alloc:.2f} GB (已分配), {gpu_mem_reserved:.2f} GB (已预留)")

    # Syncronize all random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 内存优化：启用垃圾回收
    gc.enable()
    gc.set_threshold(100, 5, 5)
    print("已启用主动垃圾回收机制")
    
    # 检查并打印当前内存使用情况
    print("初始内存状态:")
    print_memory_usage()

    # Check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(args.gpu))
        print('Using GPU: {}'.format(args.gpu))
        # 清理GPU缓存
        torch.cuda.empty_cache()
        # 限制GPU显存使用
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction, device=args.gpu)
        print(f'已限制GPU显存使用为{args.mem_fraction*100}%')
    else:
        device = torch.device("cpu")

    print(f"============= PID = {os.getpid()} ============= ")
    print(f"Parameter u: {args.u}")
    print("参数配置:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    
    # 解析 class_boundaries
    class_boundaries = [float(x) for x in args.class_boundaries.split(',')]
    
    # 强制进行一次垃圾回收
    gc.collect()
    
    # 低内存模式下调整参数
    if args.low_memory_mode:
        print("启用低内存模式")
        # 减少batch_size
        original_batch_size = args.batch_size
        args.batch_size = min(args.batch_size, 128)
        print(f"调整batch_size: {original_batch_size} -> {args.batch_size}")
        
        # 减少num_workers
        original_num_workers = args.num_workers
        args.num_workers = min(args.num_workers, 8)
        print(f"调整num_workers: {original_num_workers} -> {args.num_workers}")
        
        # 如果低内存模式启用且没有显式指定PE计算块大小，设置较小的默认值
        if args.pe_chunk_size == 2000:
            args.pe_chunk_size = 1000
            print(f"调整PE计算块大小为: {args.pe_chunk_size}")
            
        # 传递内存管理相关参数给PE计算函数
        os.environ["PE_CHUNK_SIZE"] = str(args.pe_chunk_size)
        os.environ["SKIP_PE_CALCULATION"] = str(args.skip_pe_calculation)
        os.environ["LOW_MEMORY_MODE"] = str(args.low_memory_mode)
    
    # STEP 1: Load Dataset =================================================================== #
    print("开始加载数据集...")
    
    # 定义大小数据集
    large_datasets = ['sandwich', 'ultra8t']
    small_datasets = ['ssram', 'digtime', 'timing_ctrl', 'array_128_32_8t']
    
    # 为训练集确定采样率
    train_names = args.train_dataset.split('+')
    train_sample_rates = []
    for name in train_names:
        if name in large_datasets:
            train_sample_rates.append(args.large_dataset_sample_rates)
        elif name in small_datasets:
            train_sample_rates.append(args.small_dataset_sample_rates)
        else:
            print(f"警告: 未知数据集 {name}, 使用默认采样率 1.0")
            train_sample_rates.append(1.0)
    
    print(f"训练集采样率: {dict(zip(train_names, train_sample_rates))}")
    
    train_dataset = performat_SramDataset(
        name=args.train_dataset, 
        dataset_dir=args.data_dir, 
        add_target_edges=args.add_tar_edge,
        neg_edge_ratio=0.0,
        to_undirected=True,
        sample_rates=train_sample_rates,
        task_type=args.task,
        task_level=args.task_level,
        net_only=bool(args.net_only),
        class_boundaries=class_boundaries,
        data_type=args.data_type,  # 新增: 传递数据类型
    )
    
    print("训练集加载完成，检查内存使用:")
    print_memory_usage()
    gc.collect()
    
    # 为测试集确定采样率
    test_names = args.test_dataset.split('+')
    test_sample_rates = []
    for name in test_names:
        if name in large_datasets:
            test_sample_rates.append(args.large_dataset_sample_rates)
        elif name in small_datasets:
            test_sample_rates.append(args.small_dataset_sample_rates)
        else:
            print(f"警告: 未知数据集 {name}, 使用默认采样率 1.0")
            test_sample_rates.append(1.0)
    
    print(f"测试集采样率: {dict(zip(test_names, test_sample_rates))}")
    
    test_dataset = performat_SramDataset(
        name=args.test_dataset, 
        dataset_dir=args.data_dir, 
        add_target_edges=args.add_tar_edge,
        neg_edge_ratio=0.0,
        to_undirected=True,
        sample_rates=test_sample_rates,
        task_type=args.task,
        task_level=args.task_level,
        net_only=bool(args.net_only),
        class_boundaries=class_boundaries,
        data_type=args.data_type,  # 新增: 传递数据类型
    )
    
    print("测试集加载完成，检查内存使用:")
    print_memory_usage()
    gc.collect()
    
    # STEP 2-3: If you do graph contrastive learning, you should add the code here =========== #
    
    # STEP 4: Training Epochs ================================================================ #
    # No graph contrastive learning, no initail embeddings
    # embeds = torch.zeros(( train_graph.num_nodes, hid_dim))
    dataset = {
        'train': train_dataset,
        'test': test_dataset
    }
    
    print("开始训练过程...")
    if args.task_level == 'node':
        from downstream_train import downstream_node_train
        downstream_node_train(args, dataset, device)
    else:
        downstream_train(args, dataset, device) 
    
    # 关闭输出文件并恢复标准输出
    sys.stdout = original_stdout
    log_file.close()
    print(f"所有输出已保存到日志文件 {log_filename}") 