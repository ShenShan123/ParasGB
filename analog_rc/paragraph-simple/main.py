import argparse
import torch
import numpy as np
from sram_dataset import performat_SramDataset
from downstream_train import downstream_link_pred, downstream_node_pred
import os
import random
import gc
import sys
import datetime

if __name__ == "__main__":
    # STEP 0: Parse Arguments ======================================================================= #
    parser = argparse.ArgumentParser(description="CircuitGPS_simple")
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument("--data_dir", type=str, default="../data", help="数据目录路径")
    parser.add_argument("--train_dataset", type=str, default="1+2+3+6+8+9+10+11+12+15+16+17+18", help="Name of training dataset (case IDs).")
    parser.add_argument("--test_dataset", type=str, default="5+14+20", help="Name of test dataset (case IDs).")
    parser.add_argument("--task", type=str, default="regression", help="Task type. 'classification' or 'regression'.")
    parser.add_argument("--task_level", type=str, default="node", choices=['edge', 'node'], help="Task level. 'edge' for edge prediction, 'node' for node prediction.")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of classes for node classification task.")
    parser.add_argument("--max_dist", type=int, default=350, help="The max values in DSPD.")
    parser.add_argument("--num_workers", type=int, default=4, help="The number of workers in data loaders.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index. Default: -1, using cpu.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="The batch size.")
    parser.add_argument("--lr", type=float, default=0.00005, help="Learning rate.")
    parser.add_argument("--num_gnn_layers", type=int, default=4, help="Number of GNN layers.")  # The number of FC layers were set to 4 for the net parasitic capacitance model
    parser.add_argument("--num_head_layers", type=int, default=2, help="Number of head layers.")
    parser.add_argument("--hid_dim", type=int, default=64, help="Hidden layer dim.")  #the dimension of embedding is set to F = 32
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout for neural networks.')
    parser.add_argument('--use_bn', type=int, default=0, help='Batch norm for neural networks.')
    parser.add_argument('--act_fn', default='relu', help='Activation function')
    parser.add_argument('--src_dst_agg', default='concat', help='The way to aggregate nodes. Can be "concat" or "add" or "pool".')
    parser.add_argument('--num_hops',type=int,default=4,help='Number of hops.')
    parser.add_argument('--to_undirected', type=int, default=1, help='Whether to convert the graph to undirected graph.')
    parser.add_argument('--train_sample_rate', type=float, default=1.0, help='Sampling rate for training datasets.')
    parser.add_argument('--test_sample_rate', type=float, default=1.0, help='Sampling rate for testing datasets.')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.gpu != -1 and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(args.gpu))
        torch.cuda.empty_cache()
        print(f'Using GPU: {args.gpu}')
    else:
        device = torch.device("cpu")
        
    # 启用垃圾回收
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

    # 根据 task_level 确定实际的 task_type
    if args.task_level == 'node':
        task_type = f'node_{args.task}'  # 'node_regression' or 'node_classification'
    else:
        task_type = args.task  # 'classification' or 'regression'

    train_dataset = performat_SramDataset(
        name=args.train_dataset, 
        dataset_dir=args.data_dir, 
        to_undirected=args.to_undirected,
        sample_rates=args.train_sample_rate,
        task_type=task_type,
        num_classes=args.num_classes,
    )
    
    test_dataset = performat_SramDataset(
        name=args.test_dataset, 
        dataset_dir=args.data_dir, 
        to_undirected=args.to_undirected,
        sample_rates=args.test_sample_rate,
        task_type=task_type,
        num_classes=args.num_classes,
    )
    
    dataset = {
        'train': train_dataset,
        'test': test_dataset
    }

    # 根据 task_level 选择训练函数
    if args.task_level == 'node':
        downstream_node_pred(args, dataset, device)
    else:
        downstream_link_pred(args, dataset, device)
    
    # 恢复标准输出并关闭日志文件
    sys.stdout = original_stdout
    log_file.close()
    print(f"所有输出已保存到 {log_filename}")