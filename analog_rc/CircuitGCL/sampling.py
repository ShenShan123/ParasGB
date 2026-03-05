import torch
import networkx as nx
from torch_geometric.utils import to_dense_adj, to_networkx
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader,NeighborLoader
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import pickle
from collections import Counter
from torch_geometric.data import HeteroData
# 节点类型常量 (与 RC 数据格式一致)
DEV = 0
PIN = 1
NET = 2

def dataset_sampling(args, dataset):
    """ 
    Sampling subgraphs for each graph in dataset
    Args:
        args (argparse.Namespace): The arguments
        dataset (torch_geometric.data.InMemoryDataset): The dataset
    Return:
        train_loader, val_loader, test_loaders
    """
    
    # 获取训练集索引 (第一个训练集用于训练)
    train_idx = 0
    for i, name in enumerate(dataset.names):
        if name in dataset.train_names:
            train_idx = i
            break
    
    train_graph = dataset[train_idx]
    
    if args.task_level == 'node':
        if args.net_only:
            mask = train_graph.node_type == NET 
            class_labels = train_graph.y[mask, 1]
        else:
            class_labels = train_graph.y[:, 1]
        # get all node indices
        all_nodes = np.arange(train_graph.y.size(0))
                
        # split training and validation set
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
        for i, name in enumerate(dataset.names):
            # 跳过训练集
            if name in dataset.train_names:
                continue
                
            test_graph = dataset[i]
            graph_name = test_graph.name

            all_test_nodes = torch.arange(test_graph.num_nodes)
            
            # 采样测试节点
            sampled_size = max(1, int(test_graph.num_nodes * args.sample_rate))
            perm = torch.randperm(test_graph.num_nodes)
            test_input_nodes = all_test_nodes[perm[:sampled_size]]

            test_loaders[graph_name] = NeighborLoader(
                test_graph,
                num_neighbors=args.num_hops * [args.num_neighbors],
                input_nodes=test_input_nodes,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
    
    elif args.task_level == 'edge':
        class_labels = train_graph.edge_label[:,1]
        ## get split for validation
        train_ind, val_ind = train_test_split(
            np.arange(train_graph.edge_label.size(0)), 
            test_size=0.2, shuffle=True,
        )
        train_ind = torch.tensor(train_ind, dtype=torch.long)
        val_ind = torch.tensor(val_ind, dtype=torch.long)

        train_edge_label_index = train_graph.edge_label_index[:, train_ind]
        train_edge_label = train_graph.edge_label[train_ind]

        ## Create the dataloaders for training dataset
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

        ## Create the dataloaders for validation dataset
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

        ## 测试集
        for i, name in enumerate(dataset.names):
            # 跳过训练集
            if name in dataset.train_names:
                continue
                
            test_graph = dataset[i]
            graph_name = test_graph.name
           
            test_input_edge_labels = test_graph.edge_label
            test_edge_label_index = test_graph.edge_label_index
            test_edge_label = test_input_edge_labels

            ## Create the dataloaders for each test dataset
            test_loaders[graph_name] = \
                LinkNeighborLoader(
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
            
    else:
        raise ValueError(f"Invalid task level: {args.task_level}")
    
    if isinstance(class_labels, torch.Tensor):
        class_labels = class_labels.cpu().numpy().tolist()
    label_counts = Counter(class_labels)
    print("label_counts", label_counts)
    max_label = max(label_counts, key=label_counts.get)
    print(f"The most common label in the training set is: {max_label}, with {label_counts[max_label]} samples")

    return (train_loader, val_loader, test_loaders, max_label)

  