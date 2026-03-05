"""
DataLoader creation for node and edge level tasks.
支持硬编码位置索引的数据加载。
"""
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from torch_geometric.data import Batch


def create_dataloaders(train_graphs, test_graphs, args):
    """
    Create train, val, and test dataloaders.
    """
    if len(train_graphs) == 1:
        train_graph = train_graphs[0]
    else:
        train_graph = merge_graphs(train_graphs)
    
    if args.task_level == 'node':
        return create_node_loaders(train_graph, test_graphs, args)
    else:
        return create_edge_loaders(train_graph, test_graphs, args)


def merge_graphs(graphs):
    """
    合并多个图为一个批次图。
    """
    batch = Batch.from_data_list(graphs)
    
    if hasattr(graphs[0], 'target_node_mask'):
        masks = []
        for g in graphs:
            masks.append(g.target_node_mask)
        batch.target_node_mask = torch.cat(masks, dim=0)
    
    if hasattr(graphs[0], 'train_node_mask'):
        masks = []
        for g in graphs:
            masks.append(g.train_node_mask)
        batch.train_node_mask = torch.cat(masks, dim=0)
    
    if hasattr(graphs[0], 'valid_label_mask'):
        masks = []
        for g in graphs:
            masks.append(g.valid_label_mask)
        batch.valid_label_mask = torch.cat(masks, dim=0)
    
    if hasattr(graphs[0], 'target_node_type_id'):
        batch.target_node_type_id = graphs[0].target_node_type_id
    
    # 合并边任务的edge_label_index和edge_label_y
    if hasattr(graphs[0], 'edge_label_index') and hasattr(graphs[0], 'edge_label_y'):
        edge_label_indices = []
        edge_label_ys = []
        node_offset = 0
        
        for g in graphs:
            adjusted_index = g.edge_label_index + node_offset
            edge_label_indices.append(adjusted_index)
            edge_label_ys.append(g.edge_label_y)
            node_offset += g.num_nodes
        
        batch.edge_label_index = torch.cat(edge_label_indices, dim=1)
        batch.edge_label_y = torch.cat(edge_label_ys, dim=0)
    
    return batch


def create_node_loaders(train_graph, test_graphs, args):
    """
    创建节点级任务的NeighborLoader。
    
    支持节点采样: 通过 args.node_sample_rate 控制采样比例 (0.0-1.0)
    """
    if hasattr(train_graph, 'train_node_mask'):
        valid_nodes = torch.where(train_graph.train_node_mask)[0]
    elif hasattr(train_graph, 'target_node_mask'):
        valid_nodes = torch.where(train_graph.target_node_mask)[0]
    elif hasattr(train_graph, 'target_node_type_id'):
        target_type_id = train_graph.target_node_type_id
        if isinstance(target_type_id, torch.Tensor):
            target_type_id = target_type_id.item() if target_type_id.numel() == 1 else target_type_id[0].item()
        valid_nodes = torch.where(train_graph.node_type == target_type_id)[0]
    else:
        valid_nodes = torch.arange(train_graph.num_nodes)
    
    # 节点采样: 如果 node_sample_rate < 1.0，则随机采样部分节点
    node_sample_rate = getattr(args, 'node_sample_rate', 1.0)
    num_valid_nodes = len(valid_nodes)
    
    if node_sample_rate < 1.0 and node_sample_rate > 0:
        sample_size = int(num_valid_nodes * node_sample_rate)
        sample_size = max(1, sample_size)  # 至少保留1个节点
        
        # 随机采样节点索引
        np.random.seed(42)  # 保证可复现
        sampled_indices = np.random.choice(num_valid_nodes, size=sample_size, replace=False)
        valid_nodes = valid_nodes[sampled_indices]
        
        print(f"  节点采样: {num_valid_nodes} -> {sample_size} (采样率: {node_sample_rate:.2%})")
    
    all_nodes = valid_nodes.numpy()
    train_idx, val_idx = train_test_split(all_nodes, test_size=0.2, shuffle=True, random_state=42)
    
    train_idx = torch.tensor(train_idx, dtype=torch.long)
    val_idx = torch.tensor(val_idx, dtype=torch.long)
    
    print(f"  有效训练节点 (net且label合适): {len(valid_nodes)}, Train: {len(train_idx)}, Val: {len(val_idx)}")
    
    train_loader = NeighborLoader(
        train_graph,
        num_neighbors=[args.num_neighbors] * args.num_hops,
        input_nodes=train_idx,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    
    val_loader = NeighborLoader(
        train_graph,
        num_neighbors=[args.num_neighbors] * args.num_hops,
        input_nodes=val_idx,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    
    test_loaders = {}
    for name, test_graph in test_graphs:
        if hasattr(test_graph, 'train_node_mask'):
            test_nodes = torch.where(test_graph.train_node_mask)[0]
        elif hasattr(test_graph, 'target_node_mask'):
            test_nodes = torch.where(test_graph.target_node_mask)[0]
        elif hasattr(test_graph, 'target_node_type_id'):
            target_type_id = test_graph.target_node_type_id
            if isinstance(target_type_id, torch.Tensor):
                target_type_id = target_type_id.item() if target_type_id.numel() == 1 else target_type_id[0].item()
            test_nodes = torch.where(test_graph.node_type == target_type_id)[0]
        else:
            test_nodes = torch.arange(test_graph.num_nodes)
        
        test_loaders[name] = NeighborLoader(
            test_graph,
            num_neighbors=[args.num_neighbors] * args.num_hops,
            input_nodes=test_nodes,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        print(f"  Test {name}: {len(test_nodes)} 有效节点 (net且label在过滤范围)")
    
    return train_loader, val_loader, test_loaders


def create_edge_loaders(train_graph, test_graphs, args):
    """
    创建边级任务的LinkNeighborLoader。
    edge_label_y格式: [E, 2], 第0列回归，第1列分类
    
    支持边采样: 通过 args.edge_sample_rate 控制采样比例 (0.0-1.0)
    """
    edge_label_index = train_graph.edge_label_index
    edge_label_y = train_graph.edge_label_y  # [E, 2]
    
    num_edges = edge_label_y.size(0)
    if num_edges == 0:
        raise ValueError("No target edges found for edge-level task!")
    
    # 边采样: 如果 edge_sample_rate < 1.0，则随机采样部分边
    edge_sample_rate = getattr(args, 'edge_sample_rate', 1.0)
    if edge_sample_rate < 1.0 and edge_sample_rate > 0:
        sample_size = int(num_edges * edge_sample_rate)
        sample_size = max(1, sample_size)  # 至少保留1条边
        
        # 随机采样边索引
        np.random.seed(42)  # 保证可复现
        sampled_indices = np.random.choice(num_edges, size=sample_size, replace=False)
        sampled_indices = np.sort(sampled_indices)
        
        edge_label_index = edge_label_index[:, sampled_indices]
        edge_label_y = edge_label_y[sampled_indices]
        
        print(f"  边采样: {num_edges} -> {sample_size} (采样率: {edge_sample_rate:.2%})")
        num_edges = sample_size
    
    all_edges = np.arange(num_edges)
    train_idx, val_idx = train_test_split(all_edges, test_size=0.2, shuffle=True, random_state=42)
    
    train_idx = torch.tensor(train_idx, dtype=torch.long)
    val_idx = torch.tensor(val_idx, dtype=torch.long)
    
    print(f"  有效目标边 (pair_to且label在0-700): {num_edges}, Train: {len(train_idx)}, Val: {len(val_idx)}")
    
    # LinkNeighborLoader的edge_label需要是2D tensor [E, 2]
    train_loader = LinkNeighborLoader(
        train_graph,
        num_neighbors=[args.num_neighbors] * args.num_hops,
        edge_label_index=edge_label_index[:, train_idx],
        edge_label=edge_label_y[train_idx],  # [E_train, 2]
        subgraph_type='bidirectional',
        disjoint=True,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    
    val_loader = LinkNeighborLoader(
        train_graph,
        num_neighbors=[args.num_neighbors] * args.num_hops,
        edge_label_index=edge_label_index[:, val_idx],
        edge_label=edge_label_y[val_idx],  # [E_val, 2]
        subgraph_type='bidirectional',
        disjoint=True,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    
    test_loaders = {}
    for name, test_graph in test_graphs:
        test_edge_label_index = test_graph.edge_label_index
        test_edge_label_y = test_graph.edge_label_y
        
        if test_edge_label_y.size(0) == 0:
            print(f"  Warning: Test {name} has no target edges, skipping")
            continue
        
        test_loaders[name] = LinkNeighborLoader(
            test_graph,
            num_neighbors=[args.num_neighbors] * args.num_hops,
            edge_label_index=test_edge_label_index,
            edge_label=test_edge_label_y,  # [E_test, 2]
            subgraph_type='bidirectional',
            disjoint=True,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        print(f"  Test {name}: {test_edge_label_y.size(0)} 有效目标边")
    
    return train_loader, val_loader, test_loaders


def plot_tsne_visualization(dataset, num_train, task_level='node', 
                            save_path='imgs/tsne_visualization.png',
                            max_samples_per_graph=100000, perplexity=30, random_state=42):
    """
    绘制训练集/验证集/测试集的 t-SNE 可视化图
    
    根据数据集划分（与 create_dataloaders 保持一致）：
    - 训练集 (Train): 来自训练图的80%节点/边
    - 验证集 (Validation): 来自训练图的20%节点/边  
    - 测试集 (Test): 来自测试图的所有节点/边
    
    Args:
        dataset: 完整数据集
        num_train: 训练图的数量
        task_level: 'node' 或 'edge'
        save_path: 保存路径
        max_samples_per_graph: 每个图最大采样数量 (-1 表示使用所有样本)
        perplexity: t-SNE perplexity 参数
        random_state: 随机种子
        
    使用示例:
        from change.dataloader import plot_tsne_visualization
        from change.dataset import RCGraphDataset
        
        # 加载数据集
        train_dataset = [1, 5, 7, 10, 11, 15, 17, 23, 29, 39]
        test_dataset = [42, 44, 45, 55, 58, 71, 72, 74, 75, 78]
        
        dataset = RCGraphDataset(
            root='../data',
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            task_level='node',
            task_type='regression'
        )
        
        # 绘制节点任务的 t-SNE 可视化
        plot_tsne_visualization(
            dataset=dataset,
            num_train=len(train_dataset),
            task_level='node',
            save_path='imgs/node/tsne_visualization.png',
            max_samples_per_graph=10000,
            perplexity=30
        )
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    
    print("\n" + "="*60)
    print("绘制 t-SNE 可视化图...")
    print("="*60)
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else 'imgs', exist_ok=True)
    
    features_list = []
    labels_list = []  # 0=train, 1=val, 2=test
    
    # ========== 收集训练图的特征（划分为训练集和验证集）==========
    print(f"\n  [训练/验证集] 共 {num_train} 个训练图:")
    for i in range(num_train):
        graph = dataset[i]
        graph_name = graph.name if hasattr(graph, 'name') else f'train_graph_{i}'
        
        if task_level == 'node':
            node_feat = graph.node_attr.cpu().numpy()
            num_nodes = len(node_feat)
            
            # 采样（如果 max_samples_per_graph == -1，则使用所有节点）
            if max_samples_per_graph == -1 or num_nodes <= max_samples_per_graph:
                sampled_idx = np.arange(num_nodes)
            else:
                sampled_idx = np.random.choice(num_nodes, max_samples_per_graph, replace=False)
            
            # 80% 训练，20% 验证
            np.random.shuffle(sampled_idx)
            split_point = int(len(sampled_idx) * 0.8)
            train_idx = sampled_idx[:split_point]
            val_idx = sampled_idx[split_point:]
            
            features_list.append(node_feat[train_idx])
            labels_list.extend([0] * len(train_idx))
            
            features_list.append(node_feat[val_idx])
            labels_list.extend([1] * len(val_idx))
            
            print(f"    {graph_name}: {len(train_idx)} train + {len(val_idx)} val 节点")
            
        elif task_level == 'edge':
            node_feat = graph.node_attr.cpu().numpy()
            edge_label_index = graph.edge_label_index.cpu().numpy()
            num_edges = edge_label_index.shape[1]
            
            # 采样（如果 max_samples_per_graph == -1，则使用所有边）
            if max_samples_per_graph == -1 or num_edges <= max_samples_per_graph:
                sampled_idx = np.arange(num_edges)
            else:
                sampled_idx = np.random.choice(num_edges, max_samples_per_graph, replace=False)
            
            # 计算边特征（两端节点特征的平均）
            def get_edge_features(indices):
                src_feat = node_feat[edge_label_index[0, indices]]
                dst_feat = node_feat[edge_label_index[1, indices]]
                return (src_feat + dst_feat) / 2
            
            # 80% 训练，20% 验证
            np.random.shuffle(sampled_idx)
            split_point = int(len(sampled_idx) * 0.8)
            train_idx = sampled_idx[:split_point]
            val_idx = sampled_idx[split_point:]
            
            features_list.append(get_edge_features(train_idx))
            labels_list.extend([0] * len(train_idx))
            
            features_list.append(get_edge_features(val_idx))
            labels_list.extend([1] * len(val_idx))
            
            print(f"    {graph_name}: {len(train_idx)} train + {len(val_idx)} val 边")
    
    # ========== 收集测试图的特征 ==========
    num_test = len(dataset) - num_train
    print(f"\n  [测试集] 共 {num_test} 个测试图:")
    for i in range(num_train, len(dataset)):
        graph = dataset[i]
        graph_name = graph.name if hasattr(graph, 'name') else f'test_graph_{i}'
        
        if task_level == 'node':
            node_feat = graph.node_attr.cpu().numpy()
            num_nodes = len(node_feat)
            
            # 采样
            if max_samples_per_graph == -1 or num_nodes <= max_samples_per_graph:
                sampled_idx = np.arange(num_nodes)
            else:
                sampled_idx = np.random.choice(num_nodes, max_samples_per_graph, replace=False)
            
            features_list.append(node_feat[sampled_idx])
            labels_list.extend([2] * len(sampled_idx))
            
            print(f"    {graph_name}: {len(sampled_idx)} test 节点")
            
        elif task_level == 'edge':
            node_feat = graph.node_attr.cpu().numpy()
            edge_label_index = graph.edge_label_index.cpu().numpy()
            num_edges = edge_label_index.shape[1]
            
            # 采样
            if max_samples_per_graph == -1 or num_edges <= max_samples_per_graph:
                sampled_idx = np.arange(num_edges)
            else:
                sampled_idx = np.random.choice(num_edges, max_samples_per_graph, replace=False)
            
            # 计算边特征
            src_feat = node_feat[edge_label_index[0, sampled_idx]]
            dst_feat = node_feat[edge_label_index[1, sampled_idx]]
            edge_feat = (src_feat + dst_feat) / 2
            
            features_list.append(edge_feat)
            labels_list.extend([2] * len(sampled_idx))
            
            print(f"    {graph_name}: {len(sampled_idx)} test 边")
    
    # ========== 合并特征 ==========
    all_features = np.vstack(features_list)
    all_labels = np.array(labels_list)
    
    train_count = np.sum(all_labels == 0)
    val_count = np.sum(all_labels == 1)
    test_count = np.sum(all_labels == 2)
    print(f"\n  总计: Train={train_count}, Val={val_count}, Test={test_count}, 总样本={len(all_labels)}")
    
    # ========== 特征标准化 ==========
    from sklearn.preprocessing import StandardScaler
    print(f"  特征维度: {all_features.shape[1]}")
    print(f"  特征标准化中...")
    scaler = StandardScaler()
    all_features_scaled = scaler.fit_transform(all_features)
    
    # ========== t-SNE 降维 ==========
    print("  执行 t-SNE 降维（这可能需要几分钟）...")
    tsne = TSNE(n_components=3, perplexity=min(perplexity, len(all_labels)-1), 
                random_state=random_state, max_iter=1000, verbose=1)
    embeddings = tsne.fit_transform(all_features_scaled)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 颜色配置（与论文图一致）
    colors = ['#1f4e79', '#c55a11', '#bf9000']  # 深蓝, 红褐, 黄褐
    labels_name = ['Train', 'Validation', 'Test']
    markers = ['o', 'o', 'o']
    
    # 按照 test -> val -> train 的顺序绘制，让训练集在最上层
    for i in [2, 1, 0]:
        mask = all_labels == i
        if mask.sum() > 0:
            ax.scatter(embeddings[mask, 0], embeddings[mask, 1], 
                      c=colors[i], label=labels_name[i], 
                      marker=markers[i], s=30, alpha=0.7, edgecolors='none')
    
    # 图例设置 - 放在图内上方，避免与标题重叠
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.98), 
              ncol=3, fontsize=14, markerscale=2, frameon=True,
              fancybox=False, shadow=False, framealpha=0.9)
    
    # 去掉坐标轴刻度和标签
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # 设置坐标轴范围，让点更密集
    x_margin = (embeddings[:, 0].max() - embeddings[:, 0].min()) * 0.05
    y_margin = (embeddings[:, 1].max() - embeddings[:, 1].min()) * 0.05
    ax.set_xlim(embeddings[:, 0].min() - x_margin, embeddings[:, 0].max() + x_margin)
    ax.set_ylim(embeddings[:, 1].min() - y_margin, embeddings[:, 1].max() + y_margin)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  t-SNE 图已保存到: {save_path}")
    print("="*60 + "\n")


if __name__ == "__main__":
    """
    直接运行此模块生成 t-SNE 可视化图:
        python dataloader.py --task_level node
        python dataloader.py --task_level edge
    """
    import os
    import argparse
    from dataset import RCDataset
    
    parser = argparse.ArgumentParser(description='绘制训练/验证/测试集的 t-SNE 可视化')
    parser.add_argument('--task_level', type=str, default='node', choices=['node', 'edge'],
                        help='任务类型: node 或 edge')
    parser.add_argument('--task_type', type=str, default='regression', choices=['regression', 'classification'],
                        help='任务类型: regression 或 classification')
    parser.add_argument('--max_samples', type=int, default=100000,
                        help='每个图最大采样数量 (-1 表示全部)')
    parser.add_argument('--perplexity', type=int, default=30,
                        help='t-SNE perplexity 参数')
    parser.add_argument('--save_path', type=str, default=None,
                        help='保存路径 (默认: imgs/{task_level}/tsne_visualization.png)')
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='数据目录路径')
    args = parser.parse_args()
    
    # 数据集划分 (可根据需要修改)
    train_cases = [1, 5, 7, 10, 15, 23, 29, 39, 42, 44, 58, 71, 72, 74]
    test_cases = [11,78,55]
    
    print(f"\n{'='*60}")
    print(f"加载数据集...")
    print(f"  训练图: {train_cases}")
    print(f"  测试图: {test_cases}")
    print(f"  任务级别: {args.task_level}")
    print(f"{'='*60}")
    
    # 加载数据集 (使用 RCDataset.load_and_process 逐个加载)
    class SimpleDataset:
        """简单的数据集包装类，用于 t-SNE 可视化"""
        def __init__(self, graphs):
            self.graphs = graphs
        
        def __len__(self):
            return len(self.graphs)
        
        def __getitem__(self, idx):
            return self.graphs[idx]
    
    all_graphs = []
    
    # 加载训练图
    print(f"\n--- 加载训练图 ---")
    for cid in train_cases:
        filepath = os.path.join(args.data_dir, f"case{cid}_RC.pt")
        if os.path.exists(filepath):
            print(f"  加载 case{cid}_RC.pt ...")
            graph = RCDataset.load_and_process(filepath, args.task_level, use_cache=True)
            graph.name = f"case{cid}"
            all_graphs.append(graph)
        else:
            print(f"  Warning: {filepath} not found, skipping")
    
    num_train = len(all_graphs)
    
    # 加载测试图
    print(f"\n--- 加载测试图 ---")
    for cid in test_cases:
        filepath = os.path.join(args.data_dir, f"case{cid}_RC.pt")
        if os.path.exists(filepath):
            print(f"  加载 case{cid}_RC.pt ...")
            graph = RCDataset.load_and_process(filepath, args.task_level, use_cache=True)
            graph.name = f"case{cid}"
            all_graphs.append(graph)
        else:
            print(f"  Warning: {filepath} not found, skipping")
    
    dataset = SimpleDataset(all_graphs)
    
    # 设置保存路径
    if args.save_path is None:
        save_path = f'imgs/{args.task_level}/tsne_visualization.png'
    else:
        save_path = args.save_path
    
    # 绘制 t-SNE 可视化
    plot_tsne_visualization(
        dataset=dataset,
        num_train=num_train,
        task_level=args.task_level,
        save_path=save_path,
        max_samples_per_graph=args.max_samples,
        perplexity=args.perplexity,
        random_state=42
    )
