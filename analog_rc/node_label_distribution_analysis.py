"""
分析节点任务的电容标签分布 (net节点上的y)
"""
import torch
import numpy as np
import os

data_dir = 'data'
case_ids = ['1', '5', '7', '10', '11', '15', '17', '23', '29', '39', '42', '44', '45', '55', '58', '71', '72', '74', '75', '78']

all_node_labels = []

print("=== 逐个Case分析 (net节点电容) ===")
for cid in case_ids:
    filepath = os.path.join(data_dir, f'case{cid}_RC.pt')
    if not os.path.exists(filepath):
        print(f"Case {cid}: 文件不存在")
        continue
    
    hg = torch.load(filepath, weights_only=False)
    if isinstance(hg, list):
        hg = hg[0]
    
    # 检查 net 节点的 y 属性
    if 'net' in hg.node_types and hasattr(hg['net'], 'y'):
        net_y = hg['net'].y.numpy().flatten()
        all_node_labels.extend(net_y.tolist())
        print(f'Case {cid}: {len(net_y)} net nodes, y: min={net_y.min():.4e}, max={net_y.max():.4e}, mean={net_y.mean():.4e}')
    else:
        print(f'Case {cid}: 没有 net 节点或没有 y 属性')

if len(all_node_labels) == 0:
    print("\n没有找到节点标签数据!")
else:
    all_labels = np.array(all_node_labels)
    print(f'\n=== 总体统计 ===')
    print(f'总net节点数: {len(all_labels)}')
    print(f'范围: [{all_labels.min():.4e}, {all_labels.max():.4e}]')
    print(f'均值: {all_labels.mean():.4e}')
    print(f'中位数: {np.median(all_labels):.4e}')
    print(f'标准差: {all_labels.std():.4e}')

    # 分位数分析
    print(f'\n=== 分位数分析 ===')
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f'  {p}%: {np.percentile(all_labels, p):.4e}')

    # 区间分布 (电容值通常在 1e-15 ~ 1e-12 范围)
    print(f'\n=== 区间分布 ===')
    bins = [0, 1e-16, 1e-15, 1e-14, 1e-13, 5e-13, 8e-13, 1e-12, 1e-11, 1e-10]
    for i in range(len(bins)-1):
        count = np.sum((all_labels >= bins[i]) & (all_labels < bins[i+1]))
        pct = count / len(all_labels) * 100
        print(f'  [{bins[i]:.0e}, {bins[i+1]:.0e}): {count} ({pct:.2f}%)')

    # 零值统计
    zero_count = np.sum(all_labels == 0)
    print(f'\n零值节点: {zero_count} ({zero_count/len(all_labels)*100:.2f}%)')
    
    # 非零值统计
    nonzero = all_labels[all_labels > 0]
    if len(nonzero) > 0:
        print(f'\n=== 非零值统计 ===')
        print(f'非零节点数: {len(nonzero)}')
        print(f'非零范围: [{nonzero.min():.4e}, {nonzero.max():.4e}]')
        print(f'非零均值: {nonzero.mean():.4e}')
        print(f'非零中位数: {np.median(nonzero):.4e}')
    
    # 超过阈值的统计 (当前代码用 8e-13 作为 MAX_NODE_LABEL)
    threshold = 8e-13
    over_threshold = np.sum(all_labels > threshold)
    print(f'\n超过 {threshold:.0e} 的节点: {over_threshold} ({over_threshold/len(all_labels)*100:.2f}%)')
    
    # 小电容统计
    under_1e14 = np.sum((all_labels > 0) & (all_labels < 1e-14))
    print(f'小电容 (0, 1e-14): {under_1e14} ({under_1e14/len(all_labels)*100:.2f}%)')
