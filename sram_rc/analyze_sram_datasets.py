"""
SRAM数据集综合分析脚本
包含四个功能：
1. 电路统计信息 (circuit statistics)
2. 标签分布绘图 (label distribution) - 预处理前后分开保存
3. 图相似度计算 (graph similarity - NetLSD)
4. t-SNE可视化 (t-SNE visualization of train/val/test splits)

支持两种数据集：
- sram: 电容数据 (cc_p2n, cc_p2p, cc_n2n边类型) -> imgs/sram/
- sram_r: 电阻数据 (r_p2p边类型) -> imgs/sram_r/

输出目录结构:
    imgs/
    ├── sram/
    │   ├── edge/                    # 耦合电容边标签
    │   │   ├── {name}_raw.png       # 预处理前
    │   │   └── {name}_normalized.png # 预处理后
    │   ├── node/                    # 对地电容节点标签
    │   │   ├── {name}_raw.png
    │   │   └── {name}_normalized.png
    │   ├── all_edge_raw.png
    │   ├── all_edge_normalized.png
    │   ├── all_node_raw.png
    │   └── all_node_normalized.png
    └── sram_r/
        ├── edge/                    # 等效电阻边标签
        │   ├── {name}_raw.png
        │   └── {name}_normalized.png
        ├── all_edge_raw.png
        └── all_edge_normalized.png
"""
import torch
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import eigsh
from sklearn.manifold import TSNE
import os

# ==================== 配置 ====================
SRAM_DIR = "D:/desktop/github_push/sram_rc/sram"
SRAM_R_DIR = "D:/desktop/github_push/sram_rc/sram_r"
DATASETS = ['ssram', 'digtime', 'timing_ctrl', 'array_128_32_8t', 'sandwich', 'ultra8t']
DATASETS_R = ['ssram', 'digtime', 'timing_ctrl', 'array_128_32_8t', 'sandwich', 'ultra8t']
BAR_COLOR = '#E8A838'  # 统一柱状图颜色

# ==================== 绑图参数 ====================
PLOT_DPI = 300                     # 图片保存DPI
PLOT_FONTSIZE_LABEL = 30           # 坐标轴标签字体大小
PLOT_FONTSIZE_TICK = 16            # 刻度字体大小
PLOT_FONTSIZE_TITLE = 14           # 标题字体大小
PLOT_FONTSIZE_LEGEND = 14          # 图例字体大小

# 图相似度计算参数
SIMILARITY_SAMPLE_RATE = 0.1       # 大图采样率 (10%)
SIMILARITY_SAMPLE_THRESHOLD = 500000  # 节点数超过此值时启用采样
SIMILARITY_TIMESCALES = 250        # NetLSD时间尺度数量

# t-SNE可视化参数
TSNE_SAMPLE_THRESHOLD = 100000     # 节点数阈值，超过此值则采样
TSNE_SAMPLE_RATE = 0.2             # 采样率 (10%)

# ==================== 通用函数 ====================
def load_raw_graph(name, data_dir=SRAM_DIR):
    """加载原始图数据"""
    raw_path = os.path.join(data_dir, f"{name}.pt")
    if not os.path.exists(raw_path):
        print(f"文件不存在: {raw_path}")
        return None
    hg = torch.load(raw_path, weights_only=False)
    if isinstance(hg, list):
        hg = hg[0]
    return hg

# ==================== 1. 电路统计信息 ====================
def analyze_circuit_statistics(save_path="./imgs/sram/circuit_statistics.xlsx"):
    """分析每个图的结构特征，生成Excel表格"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"[1/3] 计算电路统计信息...")
    print(f"{'='*60}")
    
    results = []
    
    for name in DATASETS:
        print(f"  处理 {name}...")
        hg = load_raw_graph(name)
        if hg is None:
            continue
        
        # 节点统计
        num_device = hg['device'].num_nodes if 'device' in hg.node_types else (hg['dev'].num_nodes if 'dev' in hg.node_types else 0)
        num_pin = hg['pin'].num_nodes if 'pin' in hg.node_types else 0
        num_net = hg['net'].num_nodes if 'net' in hg.node_types else 0
        total_nodes = num_device + num_pin + num_net
        
        # 边统计
        num_dev_pin, num_pin_net = 0, 0
        num_cc_p2n, num_cc_p2p, num_cc_n2n = 0, 0, 0
        
        for etype in hg.edge_types:
            edge_name = etype[1]
            count = hg[etype].edge_index.shape[1]
            
            if edge_name == 'device-pin':
                num_dev_pin = count
            elif edge_name == 'pin-net':
                num_pin_net = count
            elif edge_name == 'cc_p2n':
                num_cc_p2n = count
            elif edge_name == 'cc_p2p':
                num_cc_p2p = count
            elif edge_name == 'cc_n2n':
                num_cc_n2n = count
        
        total_edges = num_dev_pin + num_pin_net + num_cc_p2n + num_cc_p2p + num_cc_n2n
        total_cc_edges = num_cc_p2n + num_cc_p2p + num_cc_n2n
        
        # 节点标签统计
        if 'net' in hg.node_types and hasattr(hg['net'], 'y') and hg['net'].y is not None:
            net_y = hg['net'].y.squeeze()
            if net_y.numel() > 0:
                node_label_mean = net_y.mean().item()
                node_label_std = net_y.std().item()
                node_label_min = net_y.min().item()
                node_label_max = net_y.max().item()
            else:
                node_label_mean = node_label_std = node_label_min = node_label_max = 0
        else:
            node_label_mean = node_label_std = node_label_min = node_label_max = 0
        
        # 边标签统计
        edge_labels = []
        for etype in hg.edge_types:
            if etype[1] in ['cc_p2n', 'cc_p2p', 'cc_n2n']:
                if hasattr(hg[etype], 'y') and hg[etype].y is not None:
                    edge_labels.append(hg[etype].y)
        
        if edge_labels:
            all_edge_labels = torch.cat(edge_labels).squeeze()
            if all_edge_labels.numel() > 0:
                edge_label_mean = all_edge_labels.mean().item()
                edge_label_std = all_edge_labels.std().item()
                edge_label_min = all_edge_labels.min().item()
                edge_label_max = all_edge_labels.max().item()
            else:
                edge_label_mean = edge_label_std = edge_label_min = edge_label_max = 0
        else:
            edge_label_mean = edge_label_std = edge_label_min = edge_label_max = 0
        
        def sci_fmt(x):
            if x == 0:
                return '0'
            return f'{x:.2e}'
        
        results.append({
            'dataset': name,
            'total_nodes': total_nodes,
            'num_device': num_device,
            'num_pin': num_pin,
            'num_net': num_net,
            'total_edges': total_edges,
            'num_dev_pin': num_dev_pin,
            'num_pin_net': num_pin_net,
            'total_cc_edges': total_cc_edges,
            'num_cc_p2n': num_cc_p2n,
            'num_cc_p2p': num_cc_p2p,
            'num_cc_n2n': num_cc_n2n,
            'node_label_mean': sci_fmt(node_label_mean),
            'node_label_std': sci_fmt(node_label_std),
            'node_label_min': sci_fmt(node_label_min),
            'node_label_max': sci_fmt(node_label_max),
            'edge_label_mean': sci_fmt(edge_label_mean),
            'edge_label_std': sci_fmt(edge_label_std),
            'edge_label_min': sci_fmt(edge_label_min),
            'edge_label_max': sci_fmt(edge_label_max),
        })
    
    df = pd.DataFrame(results).set_index('dataset')
    df.to_excel(save_path)
    print(f"  统计信息已保存到: {save_path}")
    return df

def analyze_resistance_statistics(save_path="./imgs/sram_r/resistance_statistics.xlsx"):
    """分析sram_r数据集的电阻统计信息"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"[1/3] 计算电阻统计信息 (sram_r)...")
    print(f"{'='*60}")
    
    results = []
    
    for name in DATASETS_R:
        print(f"  处理 {name} (sram_r)...")
        hg = load_raw_graph(name, data_dir=SRAM_R_DIR)
        if hg is None:
            continue
        
        num_device = hg['device'].num_nodes if 'device' in hg.node_types else 0
        num_pin = hg['pin'].num_nodes if 'pin' in hg.node_types else 0
        num_net = hg['net'].num_nodes if 'net' in hg.node_types else 0
        total_nodes = num_device + num_pin + num_net
        
        num_dev_pin, num_pin_net, num_r_p2p = 0, 0, 0
        
        for etype in hg.edge_types:
            edge_name = etype[1]
            count = hg[etype].edge_index.shape[1]
            
            if edge_name == 'device-pin':
                num_dev_pin = count
            elif edge_name == 'pin-net':
                num_pin_net = count
            elif edge_name == 'r_p2p':
                num_r_p2p = count
        
        total_edges = num_dev_pin + num_pin_net + num_r_p2p
        
        edge_labels = []
        for etype in hg.edge_types:
            if etype[1] == 'r_p2p':
                if hasattr(hg[etype], 'y') and hg[etype].y is not None:
                    edge_labels.append(hg[etype].y)
        
        if edge_labels:
            all_edge_labels = torch.cat(edge_labels).squeeze()
            if all_edge_labels.numel() > 0:
                edge_label_mean = all_edge_labels.mean().item()
                edge_label_std = all_edge_labels.std().item()
                edge_label_min = all_edge_labels.min().item()
                edge_label_max = all_edge_labels.max().item()
            else:
                edge_label_mean = edge_label_std = edge_label_min = edge_label_max = 0
        else:
            edge_label_mean = edge_label_std = edge_label_min = edge_label_max = 0
        
        def sci_fmt(x):
            if x == 0:
                return '0'
            return f'{x:.2e}'
        
        results.append({
            'dataset': name,
            'total_nodes': total_nodes,
            'num_device': num_device,
            'num_pin': num_pin,
            'num_net': num_net,
            'total_edges': total_edges,
            'num_dev_pin': num_dev_pin,
            'num_pin_net': num_pin_net,
            'num_r_p2p': num_r_p2p,
            'resistance_mean': sci_fmt(edge_label_mean),
            'resistance_std': sci_fmt(edge_label_std),
            'resistance_min': sci_fmt(edge_label_min),
            'resistance_max': sci_fmt(edge_label_max),
        })
    
    df = pd.DataFrame(results).set_index('dataset')
    df.to_excel(save_path)
    print(f"  电阻统计信息已保存到: {save_path}")
    return df


# ==================== 2. 标签分布绘图 ====================

def plot_distribution(labels, save_path, color=BAR_COLOR, xlabel='value', 
                      xlim=None, use_log_scale=False):
    """绘制单个分布图（统一格式）
    
    Args:
        labels: 标签数据 (numpy array)
        save_path: 保存路径
        color: 柱状图颜色
        xlabel: x轴标签
        xlim: x轴范围 (tuple)，None表示自动
        use_log_scale: 是否使用对数刻度
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
    ax.set_facecolor('lightgray')
    
    if use_log_scale and len(labels) > 0:
        labels = labels[labels > 0]
        if len(labels) > 0:
            ax.hist(labels, bins=50, density=True, color=color, 
                    edgecolor=color, alpha=0.9, linewidth=0.5)
            ax.set_xscale('log')
    else:
        ax.hist(labels, bins=50, density=True, color=color, 
                edgecolor=color, alpha=0.9, linewidth=0.5)
    
    ax.set_xlabel(xlabel, fontsize=PLOT_FONTSIZE_LABEL)
    ax.set_ylabel('density', fontsize=PLOT_FONTSIZE_LABEL)
    ax.tick_params(axis='both', which='major', labelsize=PLOT_FONTSIZE_TICK)
    
    if xlim is not None:
        ax.set_xlim(xlim)
    
    ax.grid(True, alpha=0.3, color='white')
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.12)
    plt.savefig(save_path, dpi=PLOT_DPI, facecolor='white')
    plt.close()

# ==================== 全局百分位参数（电阻归一化用）====================
# 这些值需要先运行 compute_global_resistance_percentiles() 计算得到
GLOBAL_RESISTANCE_LOG_P1 = None  # 全局 log10(R) 的 1% 百分位
GLOBAL_RESISTANCE_LOG_P99 = None  # 全局 log10(R) 的 99% 百分位

def compute_global_resistance_percentiles():
    """计算所有电阻数据集的全局百分位，用于统一归一化"""
    global GLOBAL_RESISTANCE_LOG_P1, GLOBAL_RESISTANCE_LOG_P99
    
    print(f"\n{'='*60}")
    print(f"计算全局电阻百分位...")
    print(f"{'='*60}")
    
    all_log_vals = []
    
    for name in DATASETS_R:
        hg = load_raw_graph(name, data_dir=SRAM_R_DIR)
        if hg is None:
            continue
        
        for edge_type in hg.edge_types:
            if edge_type[1] == 'r_p2p':
                if hasattr(hg[edge_type], 'y') and hg[edge_type].y is not None:
                    edge_y = hg[edge_type].y.cpu().numpy().flatten()
                    # 过滤零值
                    nonzero_vals = edge_y[edge_y > 0]
                    if len(nonzero_vals) > 0:
                        log_vals = np.log10(nonzero_vals)
                        all_log_vals.append(log_vals)
                        print(f"  {name}: {len(nonzero_vals)} 非零样本")
    
    if all_log_vals:
        combined_log = np.concatenate(all_log_vals)
        GLOBAL_RESISTANCE_LOG_P1 = np.percentile(combined_log, 1)
        GLOBAL_RESISTANCE_LOG_P99 = np.percentile(combined_log, 99)
        
        print(f"\n全局百分位计算完成:")
        print(f"  总样本数: {len(combined_log):,}")
        print(f"  log10(R) P1:  {GLOBAL_RESISTANCE_LOG_P1:.4f}")
        print(f"  log10(R) P99: {GLOBAL_RESISTANCE_LOG_P99:.4f}")
        print(f"  对应原始值: R_P1 = {10**GLOBAL_RESISTANCE_LOG_P1:.2f} Ω, R_P99 = {10**GLOBAL_RESISTANCE_LOG_P99:.2f} Ω")
        
        return GLOBAL_RESISTANCE_LOG_P1, GLOBAL_RESISTANCE_LOG_P99
    
    return None, None

# ==================== 归一化函数 ====================
def normalize_coupling_capacitance(edge_y, return_raw=False):
    """归一化耦合电容（cc_p2n, cc_p2p, cc_n2n边）"""
    valid_mask = (edge_y > 1e-21) & (edge_y < 1e-15)
    filtered = edge_y[valid_mask]
    normalized = torch.log10(filtered * 1e21) / 6
    normalized = torch.clamp(normalized, 0.0, 1.0)
    
    if return_raw:
        return normalized.numpy(), filtered.numpy()
    return normalized.numpy()

def normalize_ground_capacitance(node_y, return_raw=False):
    """归一化对地电容（net节点标签）"""
    valid_mask = (node_y > 1e-21) & (node_y < 1e-15)
    filtered = node_y[valid_mask]
    normalized = torch.log10(filtered * 1e20) / 6
    normalized = torch.clamp(normalized, 0.0, 1.0)
    
    if return_raw:
        return normalized.numpy(), filtered.numpy()
    return normalized.numpy()

def normalize_resistance(edge_y, return_raw=False, use_global=True):
    """归一化等效电阻（r_p2p边）- 使用全局百分位归一化
    
    Args:
        edge_y: 电阻值 tensor
        return_raw: 是否返回原始值
        use_global: 是否使用全局百分位（推荐True，保证训练/测试集一致）
    
    归一化方案：
    1. 过滤零值
    2. Log10变换压缩数量级差异
    3. 使用全局1%-99%百分位裁剪极端值（所有数据集统一）
    4. 线性归一化到[0,1]
    
    公式: clip((log10(R) - GLOBAL_P1) / (GLOBAL_P99 - GLOBAL_P1), 0, 1)
    """
    edge_y_np = edge_y.cpu().numpy().flatten()
    
    # 1. 过滤零值
    nonzero_mask = edge_y_np > 0
    filtered_raw = edge_y_np[nonzero_mask]
    
    if len(filtered_raw) == 0:
        if return_raw:
            return np.array([]), np.array([])
        return np.array([])
    
    # 2. Log10变换
    log_vals = np.log10(filtered_raw)
    
    # 3. 获取百分位参数
    if use_global and GLOBAL_RESISTANCE_LOG_P1 is not None and GLOBAL_RESISTANCE_LOG_P99 is not None:
        # 使用全局百分位
        log_p1 = GLOBAL_RESISTANCE_LOG_P1
        log_p99 = GLOBAL_RESISTANCE_LOG_P99
    else:
        # 回退到局部百分位（不推荐）
        log_p1 = np.percentile(log_vals, 1)
        log_p99 = np.percentile(log_vals, 99)
    
    # 4. 裁剪并归一化到[0,1]
    if log_p99 > log_p1:
        normalized = (log_vals - log_p1) / (log_p99 - log_p1)
        normalized = np.clip(normalized, 0.0, 1.0)
    else:
        normalized = np.zeros_like(log_vals)
    
    if return_raw:
        return normalized, filtered_raw
    return normalized

def plot_label_distributions():
    """绘制sram数据集的标签分布（耦合电容和对地电容）"""
    print(f"\n{'='*60}")
    print(f"[2/3] 绘制标签分布图 (sram 电容数据)...")
    print(f"{'='*60}")
    
    # 创建目录
    os.makedirs('imgs/sram/edge', exist_ok=True)
    os.makedirs('imgs/sram/node', exist_ok=True)
    
    all_edge_raw, all_edge_norm = [], []
    all_node_raw, all_node_norm = [], []
    
    for name in DATASETS:
        print(f"  加载 {name}...")
        hg = load_raw_graph(name)
        if hg is None:
            continue
        
        # 提取耦合电容边标签
        edge_labels = []
        for edge_type in hg.edge_types:
            if edge_type[1] in ['cc_p2n', 'cc_p2p', 'cc_n2n']:
                if hasattr(hg[edge_type], 'y') and hg[edge_type].y is not None:
                    edge_labels.append(hg[edge_type].y)
        
        if edge_labels:
            edge_y = torch.cat(edge_labels).squeeze()
            norm_labels, raw_labels = normalize_coupling_capacitance(edge_y, return_raw=True)
            
            # 保存单个数据集的图（分开保存）
            plot_distribution(raw_labels, f'imgs/sram/edge/{name}_raw.png',
                             xlabel='raw value (F)', use_log_scale=True)
            plot_distribution(norm_labels, f'imgs/sram/edge/{name}_normalized.png',
                             xlabel='normalized label', xlim=(0, 1))
            
            all_edge_raw.append(raw_labels)
            all_edge_norm.append(norm_labels)
            print(f"    耦合电容边数量（过滤后）: {len(norm_labels)}")
        
        # 提取对地电容节点标签
        if 'net' in hg.node_types and hasattr(hg['net'], 'y') and hg['net'].y is not None:
            node_y = hg['net'].y.squeeze()
            norm_labels, raw_labels = normalize_ground_capacitance(node_y, return_raw=True)
            
            # 保存单个数据集的图（分开保存）
            plot_distribution(raw_labels, f'imgs/sram/node/{name}_raw.png',
                             xlabel='raw value (F)', use_log_scale=True)
            plot_distribution(norm_labels, f'imgs/sram/node/{name}_normalized.png',
                             xlabel='normalized label', xlim=(0, 1))
            
            all_node_raw.append(raw_labels)
            all_node_norm.append(norm_labels)
            print(f"    对地电容节点数量（过滤后）: {len(norm_labels)}")
    
    # 绘制合并分布图（分开保存）
    if all_edge_raw:
        combined_raw = np.concatenate(all_edge_raw)
        combined_norm = np.concatenate(all_edge_norm)
        plot_distribution(combined_raw, 'imgs/sram/all_edge_raw.png',
                         xlabel='raw value (F)', use_log_scale=True)
        plot_distribution(combined_norm, 'imgs/sram/sram_all_edge_labels.png',
                         xlabel='normalized label', xlim=(0, 1))
    
    if all_node_raw:
        combined_raw = np.concatenate(all_node_raw)
        combined_norm = np.concatenate(all_node_norm)
        plot_distribution(combined_raw, 'imgs/sram/all_node_raw.png',
                         xlabel='raw value (F)', use_log_scale=True)
        plot_distribution(combined_norm, 'imgs/sram/all_node_normalized.png',
                         xlabel='normalized label', xlim=(0, 1))
    
    print(f"  分布图已保存到 imgs/sram/ 目录")

def plot_resistance_distributions():
    """绘制sram_r数据集的电阻标签分布（使用全局百分位归一化）"""
    print(f"\n{'='*60}")
    print(f"[2/3] 绘制电阻标签分布图 (sram_r)...")
    print(f"{'='*60}")
    
    # 先计算全局百分位
    compute_global_resistance_percentiles()
    
    os.makedirs('imgs/sram_r/edge', exist_ok=True)
    
    all_edge_raw, all_edge_norm = [], []
    
    for name in DATASETS_R:
        print(f"  加载 {name} (sram_r)...")
        hg = load_raw_graph(name, data_dir=SRAM_R_DIR)
        if hg is None:
            continue
        
        # 提取等效电阻边标签
        edge_labels = []
        for edge_type in hg.edge_types:
            if edge_type[1] == 'r_p2p':
                if hasattr(hg[edge_type], 'y') and hg[edge_type].y is not None:
                    edge_labels.append(hg[edge_type].y)
        
        if edge_labels:
            edge_y = torch.cat(edge_labels).squeeze()
            # 使用全局百分位归一化
            norm_labels, raw_labels = normalize_resistance(edge_y, return_raw=True, use_global=True)
            
            # 保存单个数据集的图（分开保存）
            plot_distribution(raw_labels, f'imgs/sram_r/edge/{name}_raw.png',
                             xlabel='raw value (Ω)', use_log_scale=True)
            plot_distribution(norm_labels, f'imgs/sram_r/edge/{name}_normalized.png',
                             xlabel='normalized label', xlim=(0, 1))
            
            all_edge_raw.append(raw_labels)
            all_edge_norm.append(norm_labels)
            print(f"    等效电阻边数量（过滤后）: {len(norm_labels)}")
        else:
            print(f"    警告: {name} 没有找到r_p2p边类型")
    
    if not all_edge_norm:
        print("  没有找到有效的电阻数据")
        return
    
    # 绘制合并分布图（分开保存）
    combined_raw = np.concatenate(all_edge_raw)
    combined_norm = np.concatenate(all_edge_norm)
    plot_distribution(combined_raw, 'imgs/sram_r/all_edge_raw.png',
                     xlabel='raw value (Ω)', use_log_scale=True)
    plot_distribution(combined_norm, 'imgs/sram_r/sram_resistance_labels.png',
                     xlabel='normalized label', xlim=(0, 1))
    
    print(f"  电阻分布图已保存到 imgs/sram_r/ 目录")


# ==================== 3. 图相似度计算 (NetLSD) ====================
def hetero_to_networkx(hg, sample_rate=1.0):
    """将异构图转换为NetworkX图（支持采样）"""
    G = nx.Graph()
    node_offset, current_offset = {}, 0
    node_mapping = {}
    
    total_nodes = sum(hg[ntype].num_nodes for ntype in hg.node_types)
    
    if sample_rate < 1.0:
        print(f"      图采样: {total_nodes} 节点 -> 采样率 {sample_rate:.3f}")
    
    for ntype in hg.node_types:
        num_nodes = hg[ntype].num_nodes
        
        if sample_rate < 1.0:
            num_sampled = max(1, int(num_nodes * sample_rate))
            sampled_nodes = np.random.choice(num_nodes, num_sampled, replace=False)
            sampled_nodes = sorted(sampled_nodes)
        else:
            sampled_nodes = np.arange(num_nodes)
        
        node_offset[ntype] = current_offset
        for local_id, original_id in enumerate(sampled_nodes):
            node_mapping[(ntype, original_id)] = current_offset + local_id
        
        current_offset += len(sampled_nodes)
    
    G.add_nodes_from(range(current_offset))
    
    for etype in hg.edge_types:
        src_type, _, dst_type = etype
        ei = hg[etype].edge_index.cpu().numpy()
        
        valid_edges = []
        for i in range(ei.shape[1]):
            src_key = (src_type, int(ei[0, i]))
            dst_key = (dst_type, int(ei[1, i]))
            
            if src_key in node_mapping and dst_key in node_mapping:
                valid_edges.append((node_mapping[src_key], node_mapping[dst_key]))
        
        if valid_edges:
            G.add_edges_from(valid_edges)
        
        del valid_edges, ei
    
    print(f"      采样后: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    return G

def compute_netlsd(G, timescales=250):
    """计算NetLSD描述符"""
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    
    n = G.number_of_nodes()
    if n < 2:
        return np.zeros(timescales, dtype=np.float32)
    
    L = nx.normalized_laplacian_matrix(G).toarray()
    eigenvalues = np.linalg.eigvalsh(L)
    
    t_values = np.logspace(-2, 2, timescales)
    hkt = np.zeros(timescales, dtype=np.float32)
    for i, t in enumerate(t_values):
        hkt[i] = np.sum(np.exp(-t * eigenvalues))
    
    hkt = hkt / n
    
    del L, eigenvalues
    return hkt

def compute_graph_similarity():
    """计算图相似度并绘制热力图"""
    print(f"\n{'='*60}")
    print(f"[3/3] 计算图相似度 (NetLSD)...")
    print(f"  配置: DPI={SIMILARITY_DPI}, timescales={SIMILARITY_TIMESCALES}")
    print(f"  采样策略: 节点>{SIMILARITY_SAMPLE_THRESHOLD} 时采样 {SIMILARITY_SAMPLE_RATE*100:.1f}%")
    print(f"{'='*60}")
    
    os.makedirs('imgs/sram', exist_ok=True)
    
    print("  步骤1: 计算NetLSD特征...")
    valid_datasets = []
    feature_cache_dir = 'imgs/.cache'
    os.makedirs(feature_cache_dir, exist_ok=True)
    
    for name in DATASETS:
        print(f"    处理 {name}...")
        hg = load_raw_graph(name)
        if hg is None:
            continue
        
        try:
            total_nodes = sum(hg[ntype].num_nodes for ntype in hg.node_types)
            print(f"      原始: {total_nodes} 节点")
            
            if total_nodes > sample_threshold:
                actual_sample_rate = sample_rate
                print(f"      大图采样: {sample_rate*100:.1f}%")
            else:
                actual_sample_rate = 1.0
                print(f"      小图不采样")
            
            G = hetero_to_networkx(hg, sample_rate=actual_sample_rate)
            del hg
            
            descriptor = compute_netlsd(G, timescales=timescales)
            del G
            
            np.save(f'{feature_cache_dir}/{name}_netlsd.npy', descriptor)
            valid_datasets.append(name)
            del descriptor
            
        except Exception as e:
            print(f"      计算失败: {e}")
    
    if len(valid_datasets) < 2:
        print("  有效数据集不足")
        return None
    
    n = len(valid_datasets)
    print(f"\n  步骤2: 计算相似度矩阵 ({n}x{n})...")
    
    similarity = np.zeros((n, n), dtype=np.float32)
    
    for i in range(n):
        feat_i = np.load(f'{feature_cache_dir}/{valid_datasets[i]}_netlsd.npy').astype(np.float32)
        
        for j in range(i, n):
            feat_j = np.load(f'{feature_cache_dir}/{valid_datasets[j]}_netlsd.npy').astype(np.float32)
            
            distance = np.sqrt(np.sum((feat_i - feat_j) ** 2))
            sim = 1.0 / (1.0 + distance)
            
            similarity[i, j] = sim
            similarity[j, i] = sim
            
            del feat_j
        
        del feat_i
        
        if (i + 1) % 2 == 0 or i == n - 1:
            print(f"    进度: {i+1}/{n}")
    
    print("\n  步骤3: 保存结果...")
    df_sim = pd.DataFrame(similarity, index=valid_datasets, columns=valid_datasets)
    df_sim.to_excel('imgs/sram/graph_similarity_netlsd.xlsx')
    
    print("  步骤4: 绘制热力图...")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
    
    im = ax.imshow(similarity, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(valid_datasets, fontsize=PLOT_FONTSIZE_TICK)
    ax.set_yticklabels(valid_datasets, fontsize=PLOT_FONTSIZE_TICK)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Similarity', rotation=-90, va="bottom", fontsize=PLOT_FONTSIZE_TICK)
    ax.set_title('NetLSD Similarity', fontsize=PLOT_FONTSIZE_TITLE)
    
    plt.tight_layout()
    plt.savefig('imgs/sram/graph_similarity_heatmap.png', dpi=PLOT_DPI, bbox_inches='tight')
    plt.close('all')
    
    print("  步骤5: 清理缓存...")
    import shutil
    shutil.rmtree(feature_cache_dir)
    
    print(f"  相似度结果已保存到 imgs/sram/ 目录")
    return df_sim

# ==================== 4. 按电路规模的t-SNE可视化 ====================
def plot_tsne_by_circuit_scale(data_dir=SRAM_DIR, 
                                sample_threshold=TSNE_SAMPLE_THRESHOLD,
                                sample_rate=TSNE_SAMPLE_RATE,
                                save_path='imgs/sram/sram_tsne_circuit_scale.png'):
    """
    按电路规模（L/XL/XXL）绘制t-SNE可视化图
    
    规模划分：
    - L: digtime, timing_ctrl
    - XL: ssram, array_128_32_8t  
    - XXL: ultra8t, sandwich
    
    Args:
        data_dir: 数据目录
        sample_threshold: 节点数阈值，超过此值则采样
        sample_rate: 采样率（当节点数超过阈值时使用）
        save_path: 保存路径
    """
    print(f"\n{'='*60}")
    print(f"绘制按电路规模的t-SNE可视化...")
    print(f"  采样策略: 节点数 > {sample_threshold} 时采样 {sample_rate*100:.0f}%")
    print(f"{'='*60}")
    
    # 电路规模分组
    scale_groups = {
        'L': ['digtime', 'timing_ctrl'],
        'XL': ['ssram', 'array_128_32_8t'],
        'XXL': ['ultra8t', 'sandwich']
    }
    
    # 颜色配置
    colors = {
        'L': '#1f77b4',    # 蓝色
        'XL': '#ff7f0e',   # 橙色
        'XXL': '#2ca02c'   # 绿色
    }
    
    features_list = []
    labels_list = []
    
    for scale_name, circuits in scale_groups.items():
        print(f"\n  [{scale_name}] 电路: {circuits}")
        
        for circuit_name in circuits:
            print(f"    加载 {circuit_name}...")
            hg = load_raw_graph(circuit_name, data_dir=data_dir)
            if hg is None:
                continue
            
            # 提取节点特征
            node_features = None
            for ntype in ['net', 'device', 'dev', 'pin']:
                if ntype in hg.node_types:
                    node_data = hg[ntype]
                    if hasattr(node_data, 'x') and node_data.x is not None:
                        node_features = node_data.x.cpu().numpy()
                        num_nodes = len(node_features)
                        print(f"      使用 {ntype} 节点: {num_nodes} 个")
                        break
            
            if node_features is None:
                continue
            
            # 采样：节点数超过阈值时按采样率采样
            if num_nodes > sample_threshold:
                num_samples = int(num_nodes * sample_rate)
                sampled_idx = np.random.choice(num_nodes, num_samples, replace=False)
                node_features = node_features[sampled_idx]
                print(f"      采样: {num_samples} 个节点 ({sample_rate*100:.0f}%)")
            
            features_list.append(node_features)
            labels_list.extend([scale_name] * len(node_features))
    
    if not features_list:
        print("  错误: 没有找到有效的节点特征")
        return None
    
    # 合并特征
    all_features = np.vstack(features_list)
    all_labels = np.array(labels_list)
    
    print(f"\n  总计样本数: {len(all_labels)}")
    for scale in ['L', 'XL', 'XXL']:
        count = np.sum(all_labels == scale)
        print(f"    {scale}: {count}")
    
    # t-SNE降维 - 调整参数让点更密集
    print(f"\n  执行t-SNE降维...")
    tsne = TSNE(n_components=2, perplexity=50, max_iter=2000, 
                learning_rate=200, early_exaggeration=12,
                random_state=42, verbose=1)
    embeddings = tsne.fit_transform(all_features)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('#f8f8f8')  # 浅灰背景
    
    # 按照 XXL -> XL -> L 的顺序绘制
    for scale_name in ['XXL', 'XL', 'L']:
        mask = all_labels == scale_name
        if mask.sum() > 0:
            ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
                      c=colors[scale_name], label=scale_name,
                      s=40, alpha=0.7, edgecolors='white', linewidths=0.3)
    
    # 图例设置 - 放大，放在图内
    legend = ax.legend(loc='upper right', fontsize=28, markerscale=3, 
                      frameon=True, fancybox=False, framealpha=0.95,
                      edgecolor='gray', borderpad=1)
    legend.get_frame().set_facecolor('white')
    
    # 去掉坐标轴和边框
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # 设置坐标轴范围 - 更紧凑
    x_margin = (embeddings[:, 0].max() - embeddings[:, 0].min()) * 0.02
    y_margin = (embeddings[:, 1].max() - embeddings[:, 1].min()) * 0.02
    ax.set_xlim(embeddings[:, 0].min() - x_margin, embeddings[:, 0].max() + x_margin)
    ax.set_ylim(embeddings[:, 1].min() - y_margin, embeddings[:, 1].max() + y_margin)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n  t-SNE图已保存到: {save_path}")
    return embeddings


# ==================== 主函数 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("SRAM数据集综合分析工具")
    print("=" * 60)
    
    # 1. 统计信息
    # analyze_circuit_statistics()
    # analyze_resistance_statistics()
    
    # 2. 标签分布
    # plot_label_distributions()
    # plot_resistance_distributions()
    
    # 3. 图相似度
    # compute_graph_similarity()
    
    # 4. 按电路规模的t-SNE可视化
    plot_tsne_by_circuit_scale()
    
    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)
