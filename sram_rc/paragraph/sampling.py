import torch
import numpy as np
from torch_geometric.data import Batch
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
from sklearn.model_selection import train_test_split

def dataset_sampling(args, dataset):
    """ 
    Args:
        args (argparse.Namespace): The arguments
        dataset (dict): Dictionary containing 'train' and 'test' datasets
    Return:
        train_loader, val_loader, test_loaders
    """
    # 获取训练数据集
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    
    # 合并训练数据集中的图
    num_train = len(train_dataset.names)
    if num_train > 1:
        train_graphs = [train_dataset[i] for i in range(num_train)]
        merged_graph = Batch.from_data_list(train_graphs)
        merged_graph.name = "merged_train_graph"
        print(f"Merged {num_train} training graphs into one graph with {merged_graph.num_nodes} nodes")
    else:
        merged_graph = train_dataset[0]
        print(f"Using graph '{merged_graph.name}' as primary training graph")
    
    # 获取训练和验证集拆分
    train_ind, val_ind = train_test_split(
        np.arange(merged_graph.edge_label.size(0)), 
        test_size=0.2, shuffle=True, 
    )
    train_ind = torch.tensor(train_ind, dtype=torch.long)
    val_ind = torch.tensor(val_ind, dtype=torch.long)

    train_edge_label_index = merged_graph.edge_label_index[:, train_ind]
    train_edge_label = merged_graph.edge_label[train_ind]

    # 创建训练数据加载器
    train_loader = LinkNeighborLoader(
        merged_graph,
        num_neighbors=args.num_hops * [-1],  # -1 represents all neighbors.
        edge_label_index=train_edge_label_index,
        edge_label=train_edge_label,
        subgraph_type='bidirectional',
        disjoint=True,
        batch_size=args.batch_size,
        shuffle=True,  # shuffle in training.
        num_workers=args.num_workers,
    )

    # 创建验证数据加载器
    val_edge_label_index = merged_graph.edge_label_index[:, val_ind]
    val_edge_label = merged_graph.edge_label[val_ind]
   
    val_loader = LinkNeighborLoader(
        merged_graph,
        num_neighbors=args.num_hops * [-1],
        edge_label_index=val_edge_label_index,
        edge_label=val_edge_label,
        subgraph_type='bidirectional',
        disjoint=True,
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers,
    )

    # 创建测试数据加载器
    test_loaders = {}
    
    # 为每个测试数据集创建数据加载器
    for i in range(len(test_dataset.names)):
        test_graph = test_dataset[i]
        graph_name = test_graph.name
       
        test_loaders[graph_name] = LinkNeighborLoader(
            test_graph,
            num_neighbors=args.num_hops * [-1],
            edge_label_index=test_graph.edge_label_index,
            edge_label=test_graph.edge_label,
            subgraph_type='bidirectional',
            disjoint=True,
            batch_size=args.batch_size,
            shuffle=False,  
            num_workers=args.num_workers,
        )

    return train_loader, val_loader, test_loaders


def dataset_node_sampling(args, dataset):
    """ 
    节点级任务的采样函数
    Args:
        args (argparse.Namespace): The arguments
        dataset (dict): Dictionary containing 'train' and 'test' datasets
    Return:
        train_loader, val_loader, test_loaders
    """
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    
    # 合并训练数据集中的图
    num_train = len(train_dataset.names)
    if num_train > 1:
        train_graphs = [train_dataset[i] for i in range(num_train)]
        merged_graph = Batch.from_data_list(train_graphs)
        merged_graph.name = "merged_train_graph"
        print(f"Merged {num_train} training graphs into one graph with {merged_graph.num_nodes} nodes")
    else:
        merged_graph = train_dataset[0]
        print(f"Using graph '{merged_graph.name}' as primary training graph")
    
    # 应用采样率（根据数据集大小）
    all_train_indices = np.arange(merged_graph.num_nodes)
    # 判断是否为大数据集
    is_large_dataset = any(name in ['sandwich', 'ultra8t'] for name in train_dataset.names)
    sample_rate = args.large_dataset_sample_rates if is_large_dataset else args.small_dataset_sample_rates
    if sample_rate < 1.0:
        num_samples = int(len(all_train_indices) * sample_rate)
        all_train_indices = np.random.choice(all_train_indices, num_samples, replace=False)
        print(f"节点采样: 从 {merged_graph.num_nodes} 个节点中采样 {num_samples} 个 (采样率: {sample_rate})")
    
    # 获取训练和验证集拆分
    train_ind, val_ind = train_test_split(
        all_train_indices, 
        test_size=0.2, shuffle=True, 
    )
    train_ind = torch.tensor(train_ind, dtype=torch.long)
    val_ind = torch.tensor(val_ind, dtype=torch.long)

    # 创建训练数据加载器
    train_loader = NeighborLoader(
        merged_graph,
        num_neighbors=args.num_hops * [-1],
        input_nodes=train_ind,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # 创建验证数据加载器
    val_loader = NeighborLoader(
        merged_graph,
        num_neighbors=args.num_hops * [-1],
        input_nodes=val_ind,
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers,
    )

    # 创建测试数据加载器
    test_loaders = {}
    
    for i in range(len(test_dataset.names)):
        test_graph = test_dataset[i]
        graph_name = test_graph.name
        
        # 应用采样率（根据数据集大小）
        is_large_dataset = graph_name in ['sandwich', 'ultra8t']
        sample_rate = args.large_dataset_sample_rates if is_large_dataset else args.small_dataset_sample_rates
        if sample_rate < 1.0:
            num_test_samples = int(test_graph.num_nodes * sample_rate)
            test_nodes = torch.tensor(
                np.random.choice(test_graph.num_nodes, num_test_samples, replace=False),
                dtype=torch.long
            )
            print(f"测试集 {graph_name}: 从 {test_graph.num_nodes} 个节点中采样 {num_test_samples} 个 (采样率: {sample_rate})")
        else:
            test_nodes = torch.arange(test_graph.num_nodes)
       
        test_loaders[graph_name] = NeighborLoader(
            test_graph,
            num_neighbors=args.num_hops * [-1],
            input_nodes=test_nodes,
            batch_size=args.batch_size,
            shuffle=False,  
            num_workers=args.num_workers,
        )

    return train_loader, val_loader, test_loaders