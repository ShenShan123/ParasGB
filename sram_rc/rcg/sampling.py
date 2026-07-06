import torch
import networkx as nx
from torch_geometric.utils import to_dense_adj, to_networkx
from torch_geometric.data import Data, Batch
from torch_geometric.loader import LinkNeighborLoader,NeighborLoader
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import os
import pickle
from collections import Counter
from torch_geometric.data import HeteroData
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
NET = 0
DEV = 1
PIN = 2


def plot_tsne_visualization(dataset, num_train, task_level='node', 
                            save_path='imgs/tsne_visualization.png',
                            max_samples_per_graph=10000, perplexity=30, random_state=42):
    """
    绘制训练集/验证集/测试集的 t-SNE 可视化图
    
    根据用户设置的 train_dataset 和 test_dataset 划分：
    - 训练集 (Train): 来自训练图的80%节点/边
    - 验证集 (Validation): 来自训练图的20%节点/边  
    - 测试集 (Test): 来自测试图的所有节点/边
    
    Args:
        dataset: 完整数据集
        num_train: 训练图的数量
        task_level: 'node' 或 'edge'
        save_path: 保存路径
        max_samples_per_graph: 每个图最大采样数量
        perplexity: t-SNE perplexity 参数
        random_state: 随机种子
    """
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
            if num_nodes > max_samples_per_graph:
                sampled_idx = np.random.choice(num_nodes, max_samples_per_graph, replace=False)
            else:
                sampled_idx = np.arange(num_nodes)
            
            features_list.append(node_feat[sampled_idx])
            labels_list.extend([2] * len(sampled_idx))
            
            print(f"    {graph_name}: {len(sampled_idx)} test 节点")
            
        elif task_level == 'edge':
            node_feat = graph.node_attr.cpu().numpy()
            edge_label_index = graph.edge_label_index.cpu().numpy()
            num_edges = edge_label_index.shape[1]
            
            # 采样
            if num_edges > max_samples_per_graph:
                sampled_idx = np.random.choice(num_edges, max_samples_per_graph, replace=False)
            else:
                sampled_idx = np.arange(num_edges)
            
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
    
    # ========== t-SNE 降维 ==========
    print("  执行 t-SNE 降维（这可能需要几分钟）...")
    tsne = TSNE(n_components=3, perplexity=min(perplexity, len(all_labels)-1), 
                random_state=random_state, max_iter=1000, verbose=1)
    embeddings = tsne.fit_transform(all_features)
    
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
    
    # 去掉标题
    # ax.set_title(f't-SNE Visualization of Train/Val/Test {task_name}s', fontsize=14, pad=20)
    
    # 设置坐标轴范围，让点更密集
    x_margin = (embeddings[:, 0].max() - embeddings[:, 0].min()) * 0.05
    y_margin = (embeddings[:, 1].max() - embeddings[:, 1].min()) * 0.05
    ax.set_xlim(embeddings[:, 0].min() - x_margin, embeddings[:, 0].max() + x_margin)
    ax.set_ylim(embeddings[:, 1].min() - y_margin, embeddings[:, 1].max() + y_margin)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  t-SNE 图已保存到: {save_path}")
    print("="*50 + "\n")


def dataset_sampling(args, dataset, train_idx=None, val_idx=None):
    """ 
    Sampling subgraphs for each graph in dataset
    Args:
        args (argparse.Namespace): The arguments
        dataset (torch_geometric.data.InMemoryDataset): The dataset
    Return:
        train_loader, val_loader, test_loaders
    dataset (torch_geometric.data.InMemoryDataset): The dataset
    train_idx (Tensor or list, optional): 从 API 传入的训练节点/边索引
    val_idx   (Tensor or list, optional): 从 API 传入的验证节点/边索引
    """
    # Get number of training graphs from dataset
    num_train = getattr(dataset, 'num_train', 1)
    
    # 获取所有训练图的名称
    train_names = getattr(dataset, 'train_names', [])
    test_names = getattr(dataset, 'test_names', [])
    
    print(f"Training graphs ({num_train}): {train_names}")
    print(f"Test graphs ({len(test_names)}): {test_names}")
    
    # 合并所有训练图
    if num_train > 1:
        train_graphs = [dataset[i] for i in range(num_train)]
        train_graph = Batch.from_data_list(train_graphs)
        train_graph.name = "merged_train_graph"
        print(f"Merged {num_train} training graphs into one graph with {train_graph.num_nodes} nodes")
    else:
        train_graph = dataset[0]
        print(f"Using graph '{train_graph.name}' as primary training graph")
        
    if args.task_level == 'node':
        # train_graph.y 可能是：
        #  • 二维 tensor ([N,2], [原始值, 类别 id])
        #  • 一维 tensor ([N], 只有类别 id)
        # print(f"train_graph.y.dim():{train_graph.y.dim()}")
        # assert 0
        if train_graph.y.dim() > 1:
            # 分类时用第二列
            if args.net_only:
                mask = train_graph.node_type == NET
                class_labels = train_graph.y[mask, 1]
            else:
                class_labels = train_graph.y[:, 1]
        else:
            # 只有类别 id，仅仅用于api函数
            if args.net_only:
                mask = train_graph.node_type == NET
                class_labels = train_graph.y[mask]
            else:
                class_labels = train_graph.y

        if train_idx is not None and val_idx is not None:
            train_node_ind = train_idx
            val_node_ind   = val_idx
        else:
            all_nodes = np.arange(train_graph.y.size(0))
            
            # 根据数据集大小确定采样率
            is_large_dataset = any(name in ['sandwich', 'ultra8t'] for name in train_names)
            sample_rate = args.large_dataset_sample_rates if is_large_dataset else args.small_dataset_sample_rates
            
            # 如果采样率 < 1.0，则进行节点采样
            if sample_rate < 1.0:
                num_samples = int(len(all_nodes) * sample_rate)
                all_nodes = np.random.choice(all_nodes, num_samples, replace=False)
                print(f"训练集节点采样: 从 {train_graph.y.size(0)} 个节点中采样 {num_samples} 个 (采样率: {sample_rate})")
            
            train_node_ind, val_node_ind = train_test_split(
                all_nodes, test_size=0.2, shuffle=True
            )
        # convert to tensor
        train_node_ind = torch.tensor(train_node_ind, dtype=torch.long)
        val_node_ind = torch.tensor(val_node_ind, dtype=torch.long)
    
        train_loader = NeighborLoader(
            train_graph,
            num_neighbors=args.num_hops * [args.num_neighbors],
            input_nodes=train_node_ind,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_loader = NeighborLoader(
            train_graph,
            num_neighbors=args.num_hops * [args.num_neighbors],
            input_nodes=val_node_ind,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
       
        test_loaders = {}
        # 测试集：从索引 num_train 开始到末尾
        for i in range(num_train, len(dataset)):
            test_graph = dataset[i]
            graph_name = test_graph.name if hasattr(test_graph, 'name') else f'test_graph_{i}'
            
            print(f"Setting up test loader for graph '{graph_name}' (index {i})")

            all_test_nodes = torch.arange(test_graph.num_nodes)
            
            # 根据数据集名称确定采样率
            is_large_dataset = graph_name in ['sandwich', 'ultra8t']
            sample_rate = args.large_dataset_sample_rates if is_large_dataset else args.small_dataset_sample_rates
            
            sampled_size = max(1, int(test_graph.num_nodes * sample_rate))
            perm = torch.randperm(test_graph.num_nodes)
            test_input_nodes = all_test_nodes[perm[:sampled_size]]
            print(f"测试集 {graph_name}: 从 {test_graph.num_nodes} 个节点中采样 {sampled_size} 个 (采样率: {sample_rate})")

            test_loaders[graph_name] = NeighborLoader(
                test_graph,
                num_neighbors=args.num_hops * [args.num_neighbors],
                input_nodes=test_input_nodes,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
        
        print(f"Created {len(test_loaders)} test loaders: {list(test_loaders.keys())}")
            
    elif args.task_level == 'edge':
        # class_labels = train_graph.edge_label[:,1]
        # print(f"train_graph.edge_label.dim()：{train_graph.edge_label.dim()}")
        # assert 0
        if train_graph.edge_label.dim() > 1:
            # 分类任务：第二列是类别 id
            class_labels = train_graph.edge_label[:, 1]
            
        else: #仅用于api函数
            # 回归任务：一维 tensor
            class_labels = train_graph.edge_label
        
        train_ind, val_ind = train_test_split(
            np.arange(train_graph.edge_label.size(0)), 
            test_size=0.2, shuffle=True,
        )
        train_ind = torch.tensor(train_ind, dtype=torch.long)
        val_ind = torch.tensor(val_ind, dtype=torch.long)

        train_edge_label_index = train_graph.edge_label_index[:, train_ind]
        train_edge_label = train_graph.edge_label[train_ind]

        train_loader = LinkNeighborLoader(
            train_graph,
            num_neighbors=args.num_hops * [args.num_neighbors],
            edge_label_index=train_edge_label_index,
            edge_label=train_edge_label,
            subgraph_type='bidirectional',
            disjoint=True,
            batch_size=args.batch_size,
            shuffle=False, 
            num_workers=args.num_workers,
        )

        val_edge_label_index = train_graph.edge_label_index[:, val_ind]
        val_edge_label = train_graph.edge_label[val_ind]

        val_loader = LinkNeighborLoader(
            train_graph,
            num_neighbors=args.num_hops * [args.num_neighbors],
            edge_label_index=val_edge_label_index,
            edge_label=val_edge_label,
            subgraph_type='bidirectional',
            disjoint=True,
            batch_size=args.batch_size,
            shuffle=False, 
            num_workers=args.num_workers,
        )

        test_loaders = {}

        # 测试集：从索引 num_train 开始到末尾
        for i in range(num_train, len(dataset)):
            test_graph = dataset[i]
            graph_name = test_graph.name if hasattr(test_graph, 'name') else f'test_graph_{i}'
            
            print(f"Setting up test loader for graph '{graph_name}' (index {i})")
           
            test_input_edge_labels = test_graph.edge_label
            test_edge_label_index = test_graph.edge_label_index
            test_edge_label = test_input_edge_labels

            test_loaders[graph_name] = LinkNeighborLoader(
                test_graph,
                num_neighbors=args.num_hops * [args.num_neighbors],
                edge_label_index=test_edge_label_index,
                edge_label=test_edge_label,
                subgraph_type='bidirectional',
                disjoint=True,
                batch_size=args.batch_size,
                shuffle=False, 
                num_workers=args.num_workers,
            )
        
        print(f"Created {len(test_loaders)} test loaders: {list(test_loaders.keys())}")
            
    else:
        raise ValueError(f"Invalid task level: {args.task_level}")
    
    if isinstance(class_labels, torch.Tensor):
        class_labels = class_labels.cpu().numpy().tolist()
    label_counts = Counter(class_labels)
    print("label_counts", label_counts)
    max_label = max(label_counts, key=label_counts.get)
    print(f"The most common label in the training set is: {max_label}, with {label_counts[max_label]} samples")

    # 绘制 t-SNE 可视化（如果启用）
    if getattr(args, 'plot_tsne', False):
        # 生成保存路径（包含训练集和测试集名称）
        train_dataset_name = '+'.join(train_names) if train_names else 'train'
        test_dataset_name = '+'.join(test_names) if test_names else 'test'
        tsne_save_path = f'imgs/tsne_{args.task_level}_{train_dataset_name}_vs_{test_dataset_name}.png'
        
        plot_tsne_visualization(
            dataset=dataset,
            num_train=num_train,
            task_level=args.task_level,
            save_path=tsne_save_path,
            max_samples_per_graph=getattr(args, 'tsne_max_samples', 100000),
            perplexity=getattr(args, 'tsne_perplexity', 30)
        )
    # assert 0
    return (train_loader, val_loader, test_loaders, max_label)

