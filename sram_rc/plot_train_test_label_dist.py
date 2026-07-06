"""
绘制训练集和测试集的分类标签分布
按照现有的数据划分方式（ssram+digtime+timing_ctrl 作为训练集，sandwich+ultra8t+array_128_32_8t 作为测试集）

对比两种归一化方式：
1. 局部百分位：每个数据集用自己的P1/P99（问题所在）
2. 全局百分位：所有数据集用统一的P1/P99（解决方案）
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# ==================== 配置 ====================
SRAM_R_DIR = 'sram_r'

# 训练集配置 - 用于对比分析
TRAIN_CONFIGS = {
    'with_digtime': ['ssram', 'digtime', 'timing_ctrl'],
    'without_digtime': ['ssram', 'timing_ctrl'],
}
TRAIN_DATASETS = ['ssram', 'digtime', 'timing_ctrl']  # 默认配置
TEST_DATASETS = ['sandwich', 'ultra8t', 'array_128_32_8t']
ALL_DATASETS = ['ssram', 'digtime', 'timing_ctrl', 'sandwich', 'ultra8t', 'array_128_32_8t']

# 分类边界（归一化后的值）
# CLASS_BOUNDARIES = [0.5]  # 二分类 - 0.5边界
CLASS_BOUNDARIES = [0.6]  # 二分类 - 全局百分位下最均衡的边界
# CLASS_BOUNDARIES = [0.2, 0.4, 0.6, 0.8]  # 五分类

# 绘图参数
PLOT_DPI = 300
PLOT_FONTSIZE_LABEL = 18
PLOT_FONTSIZE_TICK = 14
PLOT_FONTSIZE_TITLE = 16
PLOT_FONTSIZE_LEGEND = 14
BAR_COLOR_TRAIN = '#4472C4'
BAR_COLOR_TEST = '#C55A5A'

# 全局百分位（运行时计算）
GLOBAL_LOG_P1 = None
GLOBAL_LOG_P99 = None

# ==================== 函数 ====================
def load_raw_graph(name, data_dir=SRAM_R_DIR):
    """加载原始图数据"""
    raw_path = os.path.join(data_dir, f"{name}.pt")
    if not os.path.exists(raw_path):
        print(f"文件不存在: {raw_path}")
        return None
    hg = torch.load(raw_path, weights_only=False)
    if isinstance(hg, list):
        hg = hg[0]
    return hg

def compute_global_percentiles():
    """计算所有数据集的全局百分位"""
    global GLOBAL_LOG_P1, GLOBAL_LOG_P99
    
    print("\n计算全局百分位...")
    all_log_vals = []
    
    for name in ALL_DATASETS:
        hg = load_raw_graph(name)
        if hg is None:
            continue
        
        for edge_type in hg.edge_types:
            if edge_type[1] == 'r_p2p':
                if hasattr(hg[edge_type], 'y') and hg[edge_type].y is not None:
                    edge_y = hg[edge_type].y.cpu().numpy().flatten()
                    nonzero_vals = edge_y[edge_y > 0]
                    if len(nonzero_vals) > 0:
                        log_vals = np.log10(nonzero_vals)
                        all_log_vals.append(log_vals)
    
    if all_log_vals:
        combined_log = np.concatenate(all_log_vals)
        GLOBAL_LOG_P1 = np.percentile(combined_log, 1)
        GLOBAL_LOG_P99 = np.percentile(combined_log, 99)
        
        print(f"  总样本数: {len(combined_log):,}")
        print(f"  全局 log10(R) P1:  {GLOBAL_LOG_P1:.4f} (对应 {10**GLOBAL_LOG_P1:.2f} Ω)")
        print(f"  全局 log10(R) P99: {GLOBAL_LOG_P99:.4f} (对应 {10**GLOBAL_LOG_P99:.2f} Ω)")
        
        return GLOBAL_LOG_P1, GLOBAL_LOG_P99
    
    return None, None

def normalize_resistance_local(edge_y):
    """归一化等效电阻 - 使用局部百分位（每个数据集独立计算）"""
    edge_y_np = edge_y.cpu().numpy().flatten()
    
    nonzero_mask = edge_y_np > 0
    filtered_raw = edge_y_np[nonzero_mask]
    
    if len(filtered_raw) == 0:
        return np.array([])
    
    log_vals = np.log10(filtered_raw)
    
    # 局部百分位 - 问题所在！
    log_p1 = np.percentile(log_vals, 1)
    log_p99 = np.percentile(log_vals, 99)
    
    if log_p99 > log_p1:
        normalized = (log_vals - log_p1) / (log_p99 - log_p1)
        normalized = np.clip(normalized, 0.0, 1.0)
    else:
        normalized = np.zeros_like(log_vals)
    
    return normalized

def normalize_resistance_global(edge_y):
    """归一化等效电阻 - 使用全局百分位（所有数据集统一）"""
    global GLOBAL_LOG_P1, GLOBAL_LOG_P99
    
    edge_y_np = edge_y.cpu().numpy().flatten()
    
    nonzero_mask = edge_y_np > 0
    filtered_raw = edge_y_np[nonzero_mask]
    
    if len(filtered_raw) == 0:
        return np.array([])
    
    log_vals = np.log10(filtered_raw)
    
    # 使用全局百分位
    if GLOBAL_LOG_P1 is not None and GLOBAL_LOG_P99 is not None:
        log_p1 = GLOBAL_LOG_P1
        log_p99 = GLOBAL_LOG_P99
    else:
        # 回退到局部
        log_p1 = np.percentile(log_vals, 1)
        log_p99 = np.percentile(log_vals, 99)
    
    if log_p99 > log_p1:
        normalized = (log_vals - log_p1) / (log_p99 - log_p1)
        normalized = np.clip(normalized, 0.0, 1.0)
    else:
        normalized = np.zeros_like(log_vals)
    
    return normalized

def get_class_labels(normalized_values, boundaries):
    """根据边界将归一化值转换为分类标签"""
    boundaries_tensor = torch.tensor(boundaries)
    values_tensor = torch.tensor(normalized_values)
    class_labels = torch.bucketize(values_tensor, boundaries_tensor)
    return class_labels.numpy()

def collect_labels(dataset_names, use_global=False, data_dir=SRAM_R_DIR):
    """收集指定数据集的所有标签
    
    Args:
        dataset_names: 数据集名称列表
        use_global: 是否使用全局百分位归一化
        data_dir: 数据目录
    """
    all_normalized = []
    
    normalize_fn = normalize_resistance_global if use_global else normalize_resistance_local
    
    for name in dataset_names:
        print(f"  加载 {name}...")
        hg = load_raw_graph(name, data_dir)
        if hg is None:
            continue
        
        # 提取 r_p2p 边标签
        edge_labels = []
        for edge_type in hg.edge_types:
            if edge_type[1] == 'r_p2p':
                if hasattr(hg[edge_type], 'y') and hg[edge_type].y is not None:
                    edge_labels.append(hg[edge_type].y)
        
        if edge_labels:
            edge_y = torch.cat(edge_labels).squeeze()
            normalized = normalize_fn(edge_y)
            all_normalized.append(normalized)
            print(f"    样本数（过滤后）: {len(normalized)}")
    
    if all_normalized:
        return np.concatenate(all_normalized)
    return np.array([])

def plot_class_distribution(train_labels, test_labels, boundaries, save_path, title_suffix='', train_names=None):
    """绘制训练集和测试集的分类标签分布对比图"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if train_names is None:
        train_names = TRAIN_DATASETS
    
    num_classes = len(boundaries) + 1
    class_names = [f'Class {i}' for i in range(num_classes)]
    
    # 统计每个类别的数量
    train_counts = np.bincount(train_labels, minlength=num_classes)
    test_counts = np.bincount(test_labels, minlength=num_classes)
    
    # 计算百分比
    train_pct = train_counts / train_counts.sum() * 100
    test_pct = test_counts / test_counts.sum() * 100
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(num_classes)
    width = 0.6
    
    # 训练集
    ax1 = axes[0]
    bars1 = ax1.bar(x, train_counts, width, color=BAR_COLOR_TRAIN, edgecolor='white', linewidth=1)
    ax1.set_xlabel('Class', fontsize=PLOT_FONTSIZE_LABEL)
    ax1.set_ylabel('Count', fontsize=PLOT_FONTSIZE_LABEL)
    ax1.set_title(f'Training Set\n({"+".join(train_names)})\n{title_suffix}', fontsize=PLOT_FONTSIZE_TITLE)
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, fontsize=PLOT_FONTSIZE_TICK)
    ax1.tick_params(axis='y', labelsize=PLOT_FONTSIZE_TICK)
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加数量和百分比标签
    for bar, count, pct in zip(bars1, train_counts, train_pct):
        height = bar.get_height()
        ax1.annotate(f'{count:,}\n({pct:.1f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12)
    
    # 测试集
    ax2 = axes[1]
    bars2 = ax2.bar(x, test_counts, width, color=BAR_COLOR_TEST, edgecolor='white', linewidth=1)
    ax2.set_xlabel('Class', fontsize=PLOT_FONTSIZE_LABEL)
    ax2.set_ylabel('Count', fontsize=PLOT_FONTSIZE_LABEL)
    ax2.set_title(f'Test Set\n({"+".join(TEST_DATASETS)})\n{title_suffix}', fontsize=PLOT_FONTSIZE_TITLE)
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names, fontsize=PLOT_FONTSIZE_TICK)
    ax2.tick_params(axis='y', labelsize=PLOT_FONTSIZE_TICK)
    ax2.grid(axis='y', alpha=0.3)
    
    # 添加数量和百分比标签
    for bar, count, pct in zip(bars2, test_counts, test_pct):
        height = bar.get_height()
        ax2.annotate(f'{count:,}\n({pct:.1f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n分类标签分布图已保存到: {save_path}")
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"分类边界: {boundaries}")
    print(f"类别数: {num_classes}")
    print(f"{'='*60}")
    print(f"\n训练集 ({'+'.join(train_names)}):")
    print(f"  总样本数: {train_counts.sum():,}")
    for i, (count, pct) in enumerate(zip(train_counts, train_pct)):
        print(f"  Class {i}: {count:,} ({pct:.1f}%)")
    
    print(f"\n测试集 ({'+'.join(TEST_DATASETS)}):")
    print(f"  总样本数: {test_counts.sum():,}")
    for i, (count, pct) in enumerate(zip(test_counts, test_pct)):
        print(f"  Class {i}: {count:,} ({pct:.1f}%)")

def plot_normalized_distribution(train_normalized, test_normalized, boundaries, save_path, title_suffix='', train_names=None):
    """绘制训练集和测试集的归一化值分布对比图（带分类边界线）"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if train_names is None:
        train_names = TRAIN_DATASETS
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制直方图
    ax.hist(train_normalized, bins=50, density=True, alpha=0.7, 
            color=BAR_COLOR_TRAIN, label=f'Train ({"+".join(train_names)})', edgecolor='white')
    ax.hist(test_normalized, bins=50, density=True, alpha=0.7, 
            color=BAR_COLOR_TEST, label=f'Test ({"+".join(TEST_DATASETS)})', edgecolor='white')
    
    # 绘制分类边界线
    for b in boundaries:
        ax.axvline(x=b, color='red', linestyle='--', linewidth=2, label=f'Boundary={b}')
    
    ax.set_xlabel('Normalized Value', fontsize=PLOT_FONTSIZE_LABEL)
    ax.set_ylabel('Density', fontsize=PLOT_FONTSIZE_LABEL)
    ax.set_title(f'Normalized Label Distribution (Train vs Test)\n{title_suffix}', fontsize=PLOT_FONTSIZE_TITLE)
    ax.set_xlim(0, 1)
    ax.tick_params(axis='both', labelsize=PLOT_FONTSIZE_TICK)
    ax.legend(fontsize=PLOT_FONTSIZE_LEGEND)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"归一化值分布图已保存到: {save_path}")

# ==================== 主函数 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("绘制训练集/测试集分类标签分布")
    print("对比局部百分位 vs 全局百分位")
    print("对比有/无 digtime 数据集")
    print("=" * 60)
    
    # 先计算全局百分位
    compute_global_percentiles()
    
    # 收集测试集数据（只需要一次）
    print("\n收集测试集数据（全局百分位）...")
    test_normalized_global = collect_labels(TEST_DATASETS, use_global=True)
    test_class_global = get_class_labels(test_normalized_global, CLASS_BOUNDARIES)
    
    # ========== 对比分析：有/无 digtime ==========
    print("\n" + "=" * 60)
    print("对比分析：有/无 digtime 数据集对训练集分布的影响")
    print("=" * 60)
    
    for config_name, train_datasets in TRAIN_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"配置: {config_name}")
        print(f"训练集: {train_datasets}")
        print(f"{'='*60}")
        
        # 收集训练集数据
        print(f"\n收集训练集数据（全局百分位）...")
        train_normalized = collect_labels(train_datasets, use_global=True)
        train_class = get_class_labels(train_normalized, CLASS_BOUNDARIES)
        
        # 绘制分布图
        suffix = 'no_digtime' if 'without' in config_name else 'with_digtime'
        
        plot_class_distribution(
            train_class, test_class_global, 
            CLASS_BOUNDARIES,
            f'imgs/sram_r/train_test_class_distribution_{suffix}.png',
            title_suffix=f'(Global Percentile - {config_name})',
            train_names=train_datasets
        )
        
        plot_normalized_distribution(
            train_normalized, test_normalized_global,
            CLASS_BOUNDARIES,
            f'imgs/sram_r/train_test_normalized_distribution_{suffix}.png',
            title_suffix=f'(Global Percentile - {config_name})',
            train_names=train_datasets
        )
        
        # 测试不同边界值
        print(f"\n测试不同边界值 ({config_name}):")
        for boundary in [0.5, 0.6, 0.7]:
            train_cls = get_class_labels(train_normalized, [boundary])
            test_cls = get_class_labels(test_normalized_global, [boundary])
            
            train_c0 = (train_cls == 0).sum() / len(train_cls) * 100
            train_c1 = (train_cls == 1).sum() / len(train_cls) * 100
            test_c0 = (test_cls == 0).sum() / len(test_cls) * 100
            test_c1 = (test_cls == 1).sum() / len(test_cls) * 100
            
            # 计算训练集和测试集分布差异
            diff_c0 = abs(train_c0 - test_c0)
            diff_c1 = abs(train_c1 - test_c1)
            
            print(f"  边界={boundary}:")
            print(f"    训练集: Class0={train_c0:.1f}%, Class1={train_c1:.1f}%")
            print(f"    测试集: Class0={test_c0:.1f}%, Class1={test_c1:.1f}%")
            print(f"    分布差异: ΔClass0={diff_c0:.1f}%, ΔClass1={diff_c1:.1f}%")
    
    # ========== 单独分析每个数据集的分布 ==========
    print("\n" + "=" * 60)
    print("单独分析每个数据集的分布（全局百分位，边界=0.6）")
    print("=" * 60)
    
    for name in ALL_DATASETS:
        normalized = collect_labels([name], use_global=True)
        if len(normalized) > 0:
            class_labels = get_class_labels(normalized, [0.6])
            c0_pct = (class_labels == 0).sum() / len(class_labels) * 100
            c1_pct = (class_labels == 1).sum() / len(class_labels) * 100
            print(f"  {name:20s}: {len(normalized):>10,} 样本, Class0={c0_pct:5.1f}%, Class1={c1_pct:5.1f}%")
    
    print("\n" + "=" * 60)
    print("完成! 请对比两种训练集配置的分布图")
    print("=" * 60)
