"""
批量运行对比实验脚本
支持的项目: rcg(6种模型), CircuitGCL, Cirgps, paragraph-simple
支持的任务: node_regression, node_classification, edge_regression, edge_classification

用法:
    python run_baselines.py                              # 运行所有(串行)
    python run_baselines.py --projects rcg_gcn        # 只运行 rcg_gcn
    python run_baselines.py --tasks node_regression      # 只运行节点回归
    python run_baselines.py --dry-run                    # 只打印命令不执行
    python run_baselines.py --auto-gpu                   # 自动选择空闲GPU
    python run_baselines.py --parallel                   # 多GPU并行执行
    python run_baselines.py --parallel --gpus 0,1,2      # 指定GPU并行执行
"""
import os
import subprocess
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ==================================================================================
# GPU 自动选择功能
# ==================================================================================

def get_available_gpus() -> list:
    """获取所有可用GPU及其空闲显存"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return []
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(',')
                gpu_id = int(parts[0].strip())
                free_mem = int(parts[1].strip())
                gpus.append((gpu_id, free_mem))
        return gpus
    except:
        return []

def get_free_gpu(min_free_memory_mb: int = 4000) -> int:
    """
    自动选择空闲GPU
    Args:
        min_free_memory_mb: 最小空闲显存要求(MB)
    Returns:
        GPU索引，如果没有可用GPU返回-1(使用CPU)
    """
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            print("[警告] nvidia-smi 执行失败，使用默认GPU")
            return GPU
        
        best_gpu = -1
        max_free = 0
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(',')
                gpu_id = int(parts[0].strip())
                free_mem = int(parts[1].strip())
                if free_mem > max_free:
                    max_free = free_mem
                    best_gpu = gpu_id
        
        if max_free < min_free_memory_mb:
            print(f"[警告] 所有GPU空闲显存不足 {min_free_memory_mb}MB，最大空闲: {max_free}MB")
            print(f"[警告] 仍使用GPU {best_gpu}，可能会OOM")
        
        if best_gpu >= 0:
            print(f"[自动选择] GPU {best_gpu} (空闲显存: {max_free}MB)")
        return best_gpu
    except FileNotFoundError:
        print("[警告] 未找到 nvidia-smi，使用默认GPU配置")
        return GPU
    except Exception as e:
        print(f"[警告] GPU检测失败: {e}，使用默认GPU配置")
        return GPU

# ==================================================================================
# 全局配置 - 在这里修改参数
# ==================================================================================

# 基础路径
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = str(BASE_DIR / "data")
RESULTS_DIR = BASE_DIR / "results"

# 数据集配置
TRAIN_CASES = "1+5+7+15+23+29+39+42+44+58+71+72+74"
TEST_CASES = "11+78+55"

# 通用配置
GPU = 0                    # GPU 索引，-1 表示 CPU，设为 "auto" 可自动选择
AUTO_GPU = True          # 是否自动选择空闲GPU
MIN_FREE_MEMORY_MB = 6000  # 自动选择时的最小空闲显存要求(MB)
EPOCHS = 200               # 训练轮数
NUM_CLASSES = 5            # 分类任务类别数
TIMEOUT = None             # 超时时间(秒)，None表示不限制

# 要运行的项目 (可以注释掉不想运行的)
PROJECTS_TO_RUN = [
    # rcg 的 6 种模型
    "rcg_gcn",
    "rcg_sage",
    "rcg_gat",
    "rcg_pna",
    "rcg_sgformer",
    "rcg_polynormer",
    # 其他项目
    "circuitgcl",
    "cirgps",
    "paragraph",
]

# 要运行的任务 (可以注释掉不想运行的)
TASKS_TO_RUN = [
    "node_regression",
    "node_classification",
    "edge_regression",
    "edge_classification",
]

# ==================================================================================
# rcg_gcn 参数配置 (回归任务)
# ==================================================================================
rcg_GCN_REG_CONFIG = {
    "dir": "rcg",
    "script": "main.py",
    "model": "gcn",
    # 模型参数
    "edge_sample_rate": 0.6,
    "activation": "leakyrelu",          # relu, elu, tanh, leakyrelu, prelu
    "hid_dim": 96,
    "num_layers": 4,
    "dropout": 0.4,
    "use_node_attr": 1,
    # 采样参数
    "num_neighbors": 64,
    "num_hops": 3,
    # 训练参数
    "lr": 0.00005,
    "batch_size": 64,
    # 回归损失函数: mse, mae, weighted_mse, weighted_mae, focal_mse, huber
    "reg_loss": "mse",
    "loss_alpha": 1.0,                  # weighted_mse/mae 的权重系数
    "loss_gamma": 2.0,                  # focal_mse 的 gamma
    "loss_delta": 1.0,                  # huber 的 delta
    # 归一化
    "normalize": "log",                 # minmax, log
}

# ==================================================================================
# rcg_gcn 参数配置 (分类任务)
# ==================================================================================
rcg_GCN_CLS_CONFIG = {
    "dir": "rcg",
    "script": "main.py",
    "model": "gcn",
    "edge_sample_rate": 0.6,
    # 模型参数
    "activation": "leakyrelu",          # relu, elu, tanh, leakyrelu, prelu
    "hid_dim": 96,
    "num_layers": 4,
    "dropout": 0.4,
    "use_node_attr": 1,
    # 采样参数
    "num_neighbors": 64,
    "num_hops": 3,
    # 训练参数
    "lr": 0.00005,
    "batch_size": 64,
    # 分类损失函数: ce, weighted_ce, focal, label_smoothing
    "cls_loss": "ce",
    "loss_gamma": 2.0,                  # focal 的 gamma
    "label_smoothing": 0.1,             # label_smoothing 的平滑系数
    # 归一化
    "normalize": "log",                 # minmax, log
}

# ==================================================================================
# rcg_sage 参数配置 (回归任务)
# ==================================================================================
rcg_SAGE_REG_CONFIG = {
    "dir": "rcg",
    "script": "main.py",
    "model": "sage",
    "edge_sample_rate": 0.6,
    # 模型参数
    "activation": "leakyrelu",
    "hid_dim": 96,
    "num_layers": 4,
    "dropout": 0.4,
    "use_node_attr": 1,
    # 采样参数
    "num_neighbors": 64,
    "num_hops": 3,
    # 训练参数
    "lr": 0.00005,
    "batch_size": 64,
    # 回归损失函数
    "reg_loss": "mse",
    "loss_alpha": 1.0,
    "loss_gamma": 2.0,
    "loss_delta": 1.0,
    # 归一化
    "normalize": "log",
}

# ==================================================================================
# rcg_sage 参数配置 (分类任务)
# ==================================================================================
rcg_SAGE_CLS_CONFIG = {
    "dir": "rcg",
    "script": "main.py",
    "model": "sage",
    "edge_sample_rate": 0.6,
    # 模型参数
    "activation": "leakyrelu",
    "hid_dim": 96,
    "num_layers": 4,
    "dropout": 0.4,
    "use_node_attr": 1,
    # 采样参数
    "num_neighbors": 64,
    "num_hops": 3,
    # 训练参数
    "lr": 0.00005,
    "batch_size": 64,
    # 分类损失函数
    "cls_loss": "ce",
    "loss_gamma": 2.0,
    "label_smoothing": 0.1,
    # 归一化
    "normalize": "log",
}

# ==================================================================================
# rcg_gat 参数配置 (回归任务)
# ==================================================================================
rcg_GAT_REG_CONFIG = {
    "dir": "rcg",
    "script": "main.py",
    "model": "gat",
    "edge_sample_rate": 0.6,
    # 模型参数
    "activation": "leakyrelu",
    "hid_dim": 96,
    "num_layers": 4,
    "dropout": 0.4,
    "use_node_attr": 1,
    # 采样参数
    "num_neighbors": 64,
    "num_hops": 3,
    # 训练参数
    "lr": 0.00005,
    "batch_size": 64,
    # 回归损失函数
    "reg_loss": "mse",
    "loss_alpha": 1.0,
    "loss_gamma": 2.0,
    "loss_delta": 1.0,
    # 归一化
    "normalize": "log",
}

# ==================================================================================
# rcg_gat 参数配置 (分类任务)
# ==================================================================================
rcg_GAT_CLS_CONFIG = {
    "dir": "rcg",
    "script": "main.py",
    "model": "gat",
    "edge_sample_rate": 0.6,
    # 模型参数
    "activation": "leakyrelu",
    "hid_dim": 96,
    "num_layers": 4,
    "dropout": 0.4,
    "use_node_attr": 1,
    # 采样参数
    "num_neighbors": 64,
    "num_hops": 3,
    # 训练参数
    "lr": 0.00005,
    "batch_size": 64,
    # 分类损失函数
    "cls_loss": "ce",
    "loss_gamma": 2.0,
    "label_smoothing": 0.1,
    # 归一化
    "normalize": "log",
}

# ==================================================================================
# rcg_pna 参数配置 (回归任务)
# ==================================================================================
rcg_PNA_REG_CONFIG = {
    "dir": "rcg",
    "script": "main.py",
    "model": "pna",
    "edge_sample_rate": 0.6,
    # 模型参数
    "activation": "leakyrelu",
    "hid_dim": 96,
    "num_layers": 4,
    "dropout": 0.4,
    "use_node_attr": 1,
    # PNA 专用参数
    "pna_towers": 4,
    # 采样参数
    "num_neighbors": 64,
    "num_hops": 3,
    # 训练参数
    "lr": 0.00005,
    "batch_size": 64,
    # 回归损失函数
    "reg_loss": "mse",
    "loss_alpha": 1.0,
    "loss_gamma": 2.0,
    "loss_delta": 1.0,
    # 归一化
    "normalize": "log",
}

# ==================================================================================
# rcg_pna 参数配置 (分类任务)
# ==================================================================================
rcg_PNA_CLS_CONFIG = {
    "dir": "rcg",
    "script": "main.py",
    "model": "pna",
    "edge_sample_rate": 0.6,
    # 模型参数
    "activation": "leakyrelu",
    "hid_dim": 96,
    "num_layers": 4,
    "dropout": 0.4,
    "use_node_attr": 1,
    # PNA 专用参数
    "pna_towers": 4,
    # 采样参数
    "num_neighbors": 64,
    "num_hops": 3,
    # 训练参数
    "lr": 0.00005,
    "batch_size": 64,
    # 分类损失函数
    "cls_loss": "ce",
    "loss_gamma": 2.0,
    "label_smoothing": 0.1,
    # 归一化
    "normalize": "log",
}

# ==================================================================================
# rcg_sgformer 参数配置 (回归任务)
# ==================================================================================
rcg_SGFORMER_REG_CONFIG = {
    "dir": "rcg",
    "script": "main.py",
    "model": "sgformer",
    "edge_sample_rate": 0.6,
    # 模型参数
    "activation": "leakyrelu",
    "hid_dim": 96,
    "num_layers": 4,
    "dropout": 0.4,
    "use_node_attr": 1,
    # SGFormer 专用参数
    "trans_num_layers": 2,
    "trans_num_heads": 1,
    "trans_dropout": 0.5,
    "gnn_num_layers": 3,
    "gnn_dropout": 0.5,
    "graph_weight": 0.5,
    "aggregate": "add",
    # 采样参数
    "num_neighbors": 64,
    "num_hops": 3,
    # 训练参数
    "lr": 0.00005,
    "batch_size": 64,
    # 回归损失函数
    "reg_loss": "mse",
    "loss_alpha": 1.0,
    "loss_gamma": 2.0,
    "loss_delta": 1.0,
    # 归一化
    "normalize": "log",
}

# ==================================================================================
# rcg_sgformer 参数配置 (分类任务)
# ==================================================================================
rcg_SGFORMER_CLS_CONFIG = {
    "dir": "rcg",
    "script": "main.py",
    "model": "sgformer",
    "edge_sample_rate": 0.6,
    # 模型参数
    "activation": "leakyrelu",
    "hid_dim": 96,
    "num_layers": 4,
    "dropout": 0.4,
    "use_node_attr": 1,
    # SGFormer 专用参数
    "trans_num_layers": 2,
    "trans_num_heads": 1,
    "trans_dropout": 0.5,
    "gnn_num_layers": 3,
    "gnn_dropout": 0.5,
    "graph_weight": 0.5,
    "aggregate": "add",
    # 采样参数
    "num_neighbors": 64,
    "num_hops": 3,
    # 训练参数
    "lr": 0.00005,
    "batch_size": 64,
    # 分类损失函数
    "cls_loss": "ce",
    "loss_gamma": 2.0,
    "label_smoothing": 0.1,
    # 归一化
    "normalize": "log",
}

# ==================================================================================
# rcg_polynormer 参数配置 (回归任务)
# ==================================================================================
rcg_POLYNORMER_REG_CONFIG = {
    "dir": "rcg",
    "script": "main.py",
    "model": "polynormer",
    "edge_sample_rate": 0.6,
    # 模型参数
    "activation": "leakyrelu",
    "hid_dim": 96,
    "num_layers": 4,
    "dropout": 0.4,
    "use_node_attr": 1,
    # Polynormer 专用参数
    "local_layers": 7,
    "global_layers": 2,
    "in_dropout": 0.15,
    "global_dropout": 0.5,
    "poly_heads": 1,
    "beta": 0.9,
    "local_attn": 0,
    # 采样参数
    "num_neighbors": 64,
    "num_hops": 3,
    # 训练参数
    "lr": 0.00005,
    "batch_size": 64,
    # 回归损失函数
    "reg_loss": "mse",
    "loss_alpha": 1.0,
    "loss_gamma": 2.0,
    "loss_delta": 1.0,
    # 归一化
    "normalize": "log",
}

# ==================================================================================
# rcg_polynormer 参数配置 (分类任务)
# ==================================================================================
rcg_POLYNORMER_CLS_CONFIG = {
    "dir": "rcg",
    "script": "main.py",
    "model": "polynormer",
    "edge_sample_rate": 0.6,
    # 模型参数
    "activation": "leakyrelu",
    "hid_dim": 96,
    "num_layers": 4,
    "dropout": 0.4,
    "use_node_attr": 1,
    # Polynormer 专用参数
    "local_layers": 7,
    "global_layers": 2,
    "in_dropout": 0.15,
    "global_dropout": 0.5,
    "poly_heads": 1,
    "beta": 0.9,
    "local_attn": 0,
    # 采样参数
    "num_neighbors": 64,
    "num_hops": 3,
    # 训练参数
    "lr": 0.00005,
    "batch_size": 64,
    # 分类损失函数
    "cls_loss": "ce",
    "loss_gamma": 2.0,
    "label_smoothing": 0.1,
    # 归一化
    "normalize": "log",
}

# ==================================================================================
# CircuitGCL 项目参数配置
# ==================================================================================
CIRCUITGCL_CONFIG = {
    "dir": "CircuitGCL",
    "script": "main.py",
    # 模型参数
    "model": "sage",                    # clustergcn, resgatedgcn, gat, gcn, sage, gine
    "act_fn": "leakyrelu",                  # relu, elu, tanh, leakyrelu, prelu
    "hid_dim": 96,
    "num_gnn_layers": 4,
    "num_head_layers": 2,
    "dropout": 0.4,
    "use_bn": 1,
    "use_stats": 1,
    # 采样参数
    "num_neighbors": 64,
    "num_hops": 3,
    "sample_rate": 0.6,
    # 训练参数
    "lr": 0.00005,
    "batch_size": 64,
    # SGRL 对比学习 (0=关闭, 1=开启)
    "sgrl": 1,
    "cl_epochs": 50,
    "cl_hid_dim": 256,
    # 损失函数
    "regress_loss": "mse",              # mse, gai, bmc, bni, lds
    "class_loss": "bsmCE",              # bsmCE, focal, cross_entropy
    # 边聚合方式
    "src_dst_agg": "concat",            # concat, add, pooladd, poolmean
}
# ==================================================================================
# Cirgps 项目参数配置
# ==================================================================================
CIRGPS_CONFIG = {
    "dir": "Cirgps",
    "script": "main.py",
    # 模型参数
    "model": "clustergcn",              # clustergcn, resgatedgcn, gat, gcn, sage, gine
    "act_fn": "relu",
    "hid_dim": 144,
    "num_gnn_layers": 4,
    "num_head_layers": 2,
    "dropout": 0.3,
    "use_bn": 1,
    "use_stats": 1,
    # 位置编码
    "use_pe": 1,
    "max_dist": 350,
    # 采样参数
    "num_hops": 2,
    "train_sample_rate": 1.0,
    "test_sample_rate": 1.0,
    # 训练参数
    "lr": 0.0001,
    "batch_size": 32,
    # 边聚合方式
    "src_dst_agg": "concat",            # concat, add, pooladd, poolmean
    # 内存管理
    "mem_fraction": 0.5,
    "low_memory_mode": 0,
}

# ==================================================================================
# paragraph-simple 项目参数配置
# ==================================================================================
PARAGRAPH_CONFIG = {
    "dir": "paragraph-simple",
    "script": "main.py",
    # 模型参数
    "act_fn": "relu",
    "hid_dim": 64,
    "num_gnn_layers": 4,
    "num_head_layers": 2,
    "dropout": 0.3,
    "use_bn": 0,
    # 采样参数
    "num_hops": 4,
    "train_sample_rate": 1.0,
    "test_sample_rate": 1.0,
    "to_undirected": 1,
    # 训练参数
    "lr": 0.00005,
    "batch_size": 32,
    # 边聚合方式
    "src_dst_agg": "concat",            # concat, add, pool
}


# ==================================================================================
# 以下是运行逻辑，一般不需要修改
# ==================================================================================

def build_rcg_cmd(task: str, cfg: dict, gpu: int = None) -> str:
    """构建 rcg 项目的命令"""
    task_level, task_type = task.split("_")
    use_gpu = gpu if gpu is not None else GPU
    
    cmd = f"python -u {cfg['script']}"
    cmd += f" --task_level {task_level} --task {task_type}"
    cmd += f" --data_dir \"{DATA_DIR}\""
    cmd += f" --train_cases \"{TRAIN_CASES}\" --test_cases \"{TEST_CASES}\""
    cmd += f" --model {cfg['model']} --activation {cfg['activation']}"
    cmd += f" --hid_dim {cfg['hid_dim']} --num_layers {cfg['num_layers']}"
    cmd += f" --dropout {cfg['dropout']} --use_node_attr {cfg['use_node_attr']}"
    cmd += f" --num_neighbors {cfg['num_neighbors']} --num_hops {cfg['num_hops']}"
    cmd += f" --epochs {EPOCHS} --batch_size {cfg['batch_size']} --lr {cfg['lr']}"
    cmd += f" --normalize {cfg['normalize']}"
    cmd += f" --gpu {use_gpu}"
    
    # 边任务添加采样率参数
    if task_level == "edge" and "edge_sample_rate" in cfg:
        cmd += f" --edge_sample_rate {cfg['edge_sample_rate']}"
    
    # PNA 专用参数
    if cfg['model'] == 'pna' and 'pna_towers' in cfg:
        cmd += f" --pna_towers {cfg['pna_towers']}"
    
    # SGFormer 专用参数
    if cfg['model'] == 'sgformer':
        cmd += f" --trans_num_layers {cfg.get('trans_num_layers', 2)}"
        cmd += f" --trans_num_heads {cfg.get('trans_num_heads', 1)}"
        cmd += f" --trans_dropout {cfg.get('trans_dropout', 0.5)}"
        cmd += f" --gnn_num_layers {cfg.get('gnn_num_layers', 3)}"
        cmd += f" --gnn_dropout {cfg.get('gnn_dropout', 0.5)}"
        cmd += f" --graph_weight {cfg.get('graph_weight', 0.5)}"
        cmd += f" --aggregate {cfg.get('aggregate', 'add')}"
    
    # Polynormer 专用参数
    if cfg['model'] == 'polynormer':
        cmd += f" --local_layers {cfg.get('local_layers', 7)}"
        cmd += f" --global_layers {cfg.get('global_layers', 2)}"
        cmd += f" --in_dropout {cfg.get('in_dropout', 0.15)}"
        cmd += f" --global_dropout {cfg.get('global_dropout', 0.5)}"
        cmd += f" --poly_heads {cfg.get('poly_heads', 1)}"
        cmd += f" --beta {cfg.get('beta', 0.9)}"
        cmd += f" --local_attn {cfg.get('local_attn', 0)}"
    
    if task_type == "regression":
        cmd += f" --reg_loss {cfg['reg_loss']}"
    else:
        cmd += f" --cls_loss {cfg['cls_loss']} --num_classes {NUM_CLASSES}"
    
    return cmd


def build_circuitgcl_cmd(task: str, gpu: int = None) -> str:
    """构建 CircuitGCL 项目的命令"""
    cfg = CIRCUITGCL_CONFIG
    task_level, task_type = task.split("_")
    use_gpu = gpu if gpu is not None else GPU
    
    # 边任务采样率设为0.5，节点任务使用配置值
    sample_rate = 0.5 if task_level == "edge" else cfg['sample_rate']
    
    cmd = f"python {cfg['script']}"
    cmd += f" --task_level {task_level} --task {task_type}"
    cmd += f" --dataset_dir \"{DATA_DIR}\""
    cmd += f" --train_dataset \"{TRAIN_CASES}\" --test_dataset \"{TEST_CASES}\""
    cmd += f" --model {cfg['model']} --act_fn {cfg['act_fn']}"
    cmd += f" --hid_dim {cfg['hid_dim']} --num_gnn_layers {cfg['num_gnn_layers']}"
    cmd += f" --num_head_layers {cfg['num_head_layers']}"
    cmd += f" --dropout {cfg['dropout']} --use_bn {cfg['use_bn']} --use_stats {cfg['use_stats']}"
    cmd += f" --num_neighbors {cfg['num_neighbors']} --num_hops {cfg['num_hops']}"
    cmd += f" --sample_rate {sample_rate}"
    cmd += f" --epochs {EPOCHS} --batch_size {cfg['batch_size']} --lr {cfg['lr']}"
    cmd += f" --sgrl {cfg['sgrl']}"
    cmd += f" --src_dst_agg {cfg['src_dst_agg']}"
    cmd += f" --gpu {use_gpu}"
    
    if task_type == "regression":
        cmd += f" --regress_loss {cfg['regress_loss']}"
    else:
        cmd += f" --class_loss {cfg['class_loss']} --num_classes {NUM_CLASSES}"
    
    return cmd


def build_cirgps_cmd(task: str, gpu: int = None) -> str:
    """构建 Cirgps 项目的命令"""
    cfg = CIRGPS_CONFIG
    task_level, task_type = task.split("_")
    use_gpu = gpu if gpu is not None else GPU
    
    # 边任务采样率设为0.5，节点任务使用配置值
    train_sample_rate = 0.5 if task_level == "edge" else cfg['train_sample_rate']
    test_sample_rate = 0.5 if task_level == "edge" else cfg['test_sample_rate']
    
    cmd = f"python {cfg['script']}"
    cmd += f" --task_level {task_level} --task {task_type}"
    cmd += f" --data_dir \"{DATA_DIR}\""
    cmd += f" --train_dataset \"{TRAIN_CASES}\" --test_dataset \"{TEST_CASES}\""
    cmd += f" --model {cfg['model']} --act_fn {cfg['act_fn']}"
    cmd += f" --hid_dim {cfg['hid_dim']} --num_gnn_layers {cfg['num_gnn_layers']}"
    cmd += f" --num_head_layers {cfg['num_head_layers']}"
    cmd += f" --dropout {cfg['dropout']} --use_bn {cfg['use_bn']} --use_stats {cfg['use_stats']}"
    cmd += f" --use_pe {cfg['use_pe']} --max_dist {cfg['max_dist']}"
    cmd += f" --num_hops {cfg['num_hops']}"
    cmd += f" --train_sample_rate {train_sample_rate} --test_sample_rate {test_sample_rate}"
    cmd += f" --epochs {EPOCHS} --batch_size {cfg['batch_size']} --lr {cfg['lr']}"
    cmd += f" --src_dst_agg {cfg['src_dst_agg']}"
    cmd += f" --mem_fraction {cfg['mem_fraction']} --low_memory_mode {cfg['low_memory_mode']}"
    cmd += f" --gpu {use_gpu}"
    
    if task_type == "classification":
        cmd += f" --num_classes {NUM_CLASSES}"
    
    return cmd


def build_paragraph_cmd(task: str, gpu: int = None) -> str:
    """构建 paragraph-simple 项目的命令"""
    cfg = PARAGRAPH_CONFIG
    task_level, task_type = task.split("_")
    use_gpu = gpu if gpu is not None else GPU
    
    # 边任务采样率设为0.5，节点任务使用配置值
    train_sample_rate = 0.5 if task_level == "edge" else cfg['train_sample_rate']
    test_sample_rate = 0.5 if task_level == "edge" else cfg['test_sample_rate']
    
    cmd = f"python {cfg['script']}"
    cmd += f" --task_level {task_level} --task {task_type}"
    cmd += f" --data_dir \"{DATA_DIR}\""
    cmd += f" --train_dataset \"{TRAIN_CASES}\" --test_dataset \"{TEST_CASES}\""
    cmd += f" --act_fn {cfg['act_fn']}"
    cmd += f" --hid_dim {cfg['hid_dim']} --num_gnn_layers {cfg['num_gnn_layers']}"
    cmd += f" --num_head_layers {cfg['num_head_layers']}"
    cmd += f" --dropout {cfg['dropout']} --use_bn {cfg['use_bn']}"
    cmd += f" --num_hops {cfg['num_hops']}"
    cmd += f" --train_sample_rate {train_sample_rate} --test_sample_rate {test_sample_rate}"
    cmd += f" --to_undirected {cfg['to_undirected']}"
    cmd += f" --epochs {EPOCHS} --batch_size {cfg['batch_size']} --lr {cfg['lr']}"
    cmd += f" --src_dst_agg {cfg['src_dst_agg']}"
    cmd += f" --gpu {use_gpu}"
    
    if task_type == "classification":
        cmd += f" --num_classes {NUM_CLASSES}"
    
    return cmd


# 项目配置映射 (区分回归和分类)
PROJECT_CONFIGS = {
    # rcg 的 6 种模型 - 回归
    "rcg_gcn_reg": ("rcg", rcg_GCN_REG_CONFIG),
    "rcg_sage_reg": ("rcg", rcg_SAGE_REG_CONFIG),
    "rcg_gat_reg": ("rcg", rcg_GAT_REG_CONFIG),
    "rcg_pna_reg": ("rcg", rcg_PNA_REG_CONFIG),
    "rcg_sgformer_reg": ("rcg", rcg_SGFORMER_REG_CONFIG),
    "rcg_polynormer_reg": ("rcg", rcg_POLYNORMER_REG_CONFIG),
    # rcg 的 6 种模型 - 分类
    "rcg_gcn_cls": ("rcg", rcg_GCN_CLS_CONFIG),
    "rcg_sage_cls": ("rcg", rcg_SAGE_CLS_CONFIG),
    "rcg_gat_cls": ("rcg", rcg_GAT_CLS_CONFIG),
    "rcg_pna_cls": ("rcg", rcg_PNA_CLS_CONFIG),
    "rcg_sgformer_cls": ("rcg", rcg_SGFORMER_CLS_CONFIG),
    "rcg_polynormer_cls": ("rcg", rcg_POLYNORMER_CLS_CONFIG),
    # 其他项目
    "circuitgcl": ("CircuitGCL", CIRCUITGCL_CONFIG),
    "cirgps": ("Cirgps", CIRGPS_CONFIG),
    "paragraph": ("paragraph-simple", PARAGRAPH_CONFIG),
}


def get_rcg_config(project: str, task: str):
    """根据项目名和任务类型获取 rcg 配置"""
    # 从 project 中提取模型名 (如 rcg_gcn -> gcn)
    model_name = project.replace("rcg_", "")
    # 根据任务类型选择配置
    is_cls = "classification" in task
    config_key = f"rcg_{model_name}_{'cls' if is_cls else 'reg'}"
    return PROJECT_CONFIGS[config_key][1]


def build_cmd(project: str, task: str, gpu: int = None) -> str:
    """根据项目名构建命令"""
    if project.startswith("rcg_"):
        cfg = get_rcg_config(project, task)
        return build_rcg_cmd(task, cfg, gpu)
    elif project == "circuitgcl":
        return build_circuitgcl_cmd(task, gpu)
    elif project == "cirgps":
        return build_cirgps_cmd(task, gpu)
    elif project == "paragraph":
        return build_paragraph_cmd(task, gpu)
    else:
        raise ValueError(f"未知项目: {project}")


def get_work_dir(project: str) -> str:
    """获取项目工作目录"""
    if project.startswith("rcg_"):
        return "rcg"
    elif project == "circuitgcl":
        return "CircuitGCL"
    elif project == "cirgps":
        return "Cirgps"
    elif project == "paragraph":
        return "paragraph-simple"
    else:
        raise ValueError(f"未知项目: {project}")


def run_experiment(project: str, task: str, timestamp: str, dry_run: bool = False, gpu: int = None, quiet: bool = False) -> dict:
    """运行单个实验"""
    work_dir = BASE_DIR / get_work_dir(project)
    
    # 按任务类型分目录保存日志
    task_dir = RESULTS_DIR / task
    task_dir.mkdir(parents=True, exist_ok=True)
    log_file = task_dir / f"{project}_{timestamp}.log"
    
    use_gpu = gpu if gpu is not None else GPU
    
    result = {"project": project, "task": task, "status": "未运行", "log_file": str(log_file), "gpu": use_gpu}
    
    if not work_dir.exists():
        if not quiet:
            print(f"[GPU {use_gpu}] [错误] {project}-{task}: 目录不存在")
        result["status"] = "目录不存在"
        return result
    
    cmd = build_cmd(project, task, use_gpu)
    
    if not quiet:
        print(f"[GPU {use_gpu}] 开始: {project} - {task}")
    
    if dry_run:
        result["status"] = "dry-run"
        return result
    
    try:
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"项目: {project}\n任务: {task}\nGPU: {use_gpu}\n命令: {cmd}\n时间: {datetime.now()}\n")
            f.write("=" * 60 + "\n\n")
            f.flush()
            
            # 设置环境变量禁用tqdm进度条，减少日志大小和乱码
            env = os.environ.copy()
            env['TQDM_DISABLE'] = '1'
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONUNBUFFERED'] = '1'
            
            process = subprocess.run(cmd, shell=True, cwd=work_dir, stdout=f, stderr=subprocess.STDOUT, timeout=TIMEOUT, env=env)
        
        if process.returncode == 0:
            if not quiet:
                print(f"[GPU {use_gpu}] ✓ 完成: {project} - {task}")
            result["status"] = "成功"
        else:
            if not quiet:
                print(f"[GPU {use_gpu}] ✗ 失败: {project} - {task} (返回码:{process.returncode})")
            result["status"] = f"失败(返回码:{process.returncode})"
            
    except subprocess.TimeoutExpired:
        if not quiet:
            print(f"[GPU {use_gpu}] ⏱ 超时: {project} - {task}")
        result["status"] = "超时"
    except Exception as e:
        if not quiet:
            print(f"[GPU {use_gpu}] ✗ 错误: {project} - {task}")
        result["status"] = f"错误: {e}"
    
    return result


def print_summary(results: list):
    """打印运行结果汇总"""
    print("\n" + "=" * 80)
    print("运行结果汇总")
    print("=" * 80)
    print(f"\n{'项目':<20} {'任务':<25} {'GPU':<6} {'状态':<15}")
    print("-" * 70)
    for r in results:
        gpu_str = str(r.get('gpu', '-'))
        print(f"{r['project']:<20} {r['task']:<25} {gpu_str:<6} {r['status']:<15}")
    
    success = sum(1 for r in results if r["status"] == "成功")
    fail = sum(1 for r in results if "失败" in r["status"] or "错误" in r["status"])
    print("-" * 70)
    print(f"总计: {len(results)} | 成功: {success} | 失败: {fail}")
    print(f"日志保存在: {RESULTS_DIR}")
    print("=" * 80)


def run_parallel(experiments: list, gpus: list, timestamp: str, dry_run: bool = False) -> list:
    """
    多GPU并行执行实验
    Args:
        experiments: [(project, task), ...] 实验列表
        gpus: 可用GPU列表
        timestamp: 时间戳
        dry_run: 是否只打印不执行
    Returns:
        结果列表
    """
    import time
    from queue import Queue
    
    results = []
    results_lock = threading.Lock()
    completed = [0]
    total = len(experiments)
    
    # 任务队列
    task_queue = Queue()
    for exp in experiments:
        task_queue.put(exp)
    
    print(f"\n并行模式: {len(gpus)} GPU {gpus}, {total} 个实验")
    print("-" * 50)
    
    def worker(gpu_id):
        """每个GPU一个worker线程"""
        while True:
            try:
                project, task = task_queue.get_nowait()
            except:
                break  # 队列空了，退出
            
            result = run_experiment(project, task, timestamp, dry_run, gpu_id, quiet=False)
            
            with results_lock:
                results.append(result)
                completed[0] += 1
                print(f"[进度] {completed[0]}/{total} 完成")
            
            task_queue.task_done()
    
    # 为每个GPU启动一个线程
    threads = []
    for gpu_id in gpus:
        t = threading.Thread(target=worker, args=(gpu_id,))
        t.start()
        threads.append(t)
    
    # 等待所有线程完成
    for t in threads:
        t.join()
    
    return results


def main():
    global GPU
    import argparse
    parser = argparse.ArgumentParser(description="批量运行对比实验")
    parser.add_argument("--projects", type=str, default="", help="要运行的项目，逗号分隔")
    parser.add_argument("--tasks", type=str, default="", help="要运行的任务，逗号分隔")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令不执行")
    parser.add_argument("--auto-gpu", action="store_true", help="自动选择空闲GPU")
    parser.add_argument("--gpu", type=int, default=None, help="指定GPU索引，覆盖配置文件")
    parser.add_argument("--parallel", type=int, default=1, help="多GPU并行执行")
    parser.add_argument("--gpus", type=str, default="1,2,3", help="并行时使用的GPU列表，如 0,1,2")
    args = parser.parse_args()
    
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 确定要运行的项目和任务
    projects = [p.strip() for p in args.projects.split(",") if p.strip()] if args.projects else PROJECTS_TO_RUN
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()] if args.tasks else TASKS_TO_RUN
    
    # 验证项目名
    valid_projects = ["rcg_gcn", "rcg_sage", "rcg_gat", "rcg_pna", 
                      "rcg_sgformer", "rcg_polynormer", "circuitgcl", "cirgps", "paragraph"]
    for p in projects:
        if p not in valid_projects:
            print(f"[错误] 未知项目: {p}")
            print(f"可用项目: {valid_projects}")
            return
    
    # 构建实验列表
    experiments = [(p, t) for p in projects for t in tasks]
    
    print("=" * 50)
    print("批量运行对比实验")
    print("=" * 50)
    print(f"项目: {len(projects)}个, 任务: {len(tasks)}个, 共: {len(experiments)}个实验")
    print(f"训练集: {TRAIN_CASES}")
    print(f"测试集: {TEST_CASES}")
    
    # 并行模式
    if args.parallel:
        # 确定使用的GPU
        if args.gpus:
            gpus = [int(g.strip()) for g in args.gpus.split(",")]
        else:
            # 自动检测所有可用GPU
            available = get_available_gpus()
            if available:
                gpus = [g[0] for g in available]
            else:
                gpus = [0]
        
        print(f"模式: 并行 (GPU: {gpus})")
        print("=" * 50)
        
        results = run_parallel(experiments, gpus, timestamp, args.dry_run)
    else:
        # 串行模式 - GPU选择逻辑
        if args.gpu is not None:
            GPU = args.gpu
        elif args.auto_gpu or AUTO_GPU:
            GPU = get_free_gpu(MIN_FREE_MEMORY_MB)
        
        print(f"模式: 串行 (GPU: {GPU})")
        print("=" * 50)
        
        results = []
        for i, (project, task) in enumerate(experiments, 1):
            print(f"\n[{i}/{len(experiments)}] {project} - {task}")
            result = run_experiment(project, task, timestamp, args.dry_run, GPU)
            results.append(result)
    
    print_summary(results)


if __name__ == "__main__":
    main()
