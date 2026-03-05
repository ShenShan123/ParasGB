import torch
import networkx as nx
from torch_geometric.utils import to_dense_adj, to_networkx
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import pickle
import glob
from torch_geometric.data import Batch


def get_single_spd(data, anchor_index, max_dist):
    """
    Compute shortest path distances from a single anchor node to all other nodes.
    For node tasks, we store the same distance in both columns.
    
    Args:
        data (torch_geometric.data.Data): Input graph data
        anchor_index (int): Index of anchor node
        max_dist (int): Maximum distance value
    
    Returns:
        torch.Tensor: Tensor of shortest path distances (shape: [num_nodes, 2])
    """
    num_nodes = data.num_nodes
    
    # Initialize distance matrix with max_dist
    distances = torch.full((num_nodes, 2), max_dist, dtype=torch.long)
    
    # Convert to NetworkX graph
    G = to_networkx(data, to_undirected=True)
    
    if anchor_index not in G:
        return distances
    
    try:
        shortest_lengths = nx.single_source_shortest_path_length(G, anchor_index)
        
        for node, dist in shortest_lengths.items():
            d = dist if dist < max_dist else max_dist
            # Store same distance in both columns for node task
            distances[node, 0] = d
            distances[node, 1] = d
    except Exception as e:
        print(f"Error computing SPD from node {anchor_index}: {e}")
    
    return distances


def get_double_spd(data, anchor_indices, max_dist):
    """
    Compute shortest path distances from multiple anchor nodes to all other nodes
    in an undirected graph using NetworkX.
    
    Args:
        data (torch_geometric.data.Data): Input graph data
        anchor_indices (torch.Tensor or list): Indices of anchor nodes (shape: [M])
    
    Returns:
        torch.Tensor: Tensor of shortest path distances (shape: [num_nodes, M])
    """
    num_nodes = data.num_nodes
    M = len(anchor_indices)
    
    # 检查图的大小，如果过大可能导致内存问题
    if num_nodes > 10000:
        print(f"警告: 大型图 ({num_nodes} 节点)，尝试优化内存使用...")
        
        # 初始化距离矩阵，默认为最大距离
        distances = torch.full((num_nodes, M), max_dist, dtype=torch.long)
        
        # 检查是否可以使用PyTorch Sparse张量来减少内存使用
        try:
            # 将PyG边索引转换为COO稀疏矩阵
            edge_index = data.edge_index
            # 创建邻接矩阵
            import scipy.sparse as sp
            adj = sp.coo_matrix(
                (torch.ones(edge_index.size(1)), 
                 (edge_index[0].numpy(), edge_index[1].numpy())),
                shape=(num_nodes, num_nodes)
            )
            # 转换为CSR格式以加速计算
            adj_csr = adj.tocsr()
            
            # 转换为无向图
            adj_csr = adj_csr.maximum(adj_csr.transpose())
            
            # 使用scipy的最短路径算法
            from scipy.sparse.csgraph import shortest_path
            
            # 批处理计算，防止内存溢出
            batch_size = 500  # 每批处理的节点数
            
            # 转换到list如果是tensor
            if isinstance(anchor_indices, torch.Tensor):
                anchor_indices = anchor_indices.tolist()
                
            for i, anchor in enumerate(anchor_indices):
                # 只计算从锚点到所有其他节点的最短路径
                dists = shortest_path(
                    adj_csr, directed=False, 
                    indices=[anchor], return_predecessors=False
                ).flatten()
                
                # 填充距离矩阵
                for node, dist in enumerate(dists):
                    # 如果距离是无穷大或超过max_dist，设为max_dist
                    if np.isinf(dist) or dist >= max_dist:
                        distances[node, i] = max_dist
                    else:
                        distances[node, i] = int(dist)
                        
            # 垃圾回收
            import gc
            gc.collect()
            
            return distances
            
        except Exception as e:
            print(f"稀疏矩阵计算失败: {e}，回退到NetworkX方法")
            
            
    # 对于较小的图，或者如果稀疏矩阵方法失败，使用原始的NetworkX方法
    # Convert PyG data to NetworkX undirected graph once
    import gc
    G = to_networkx(data, to_undirected=True)
    
    # 主动释放内存
    if hasattr(data, 'edge_index'):
        del data.edge_index
    gc.collect()
    
    # Initialize distance matrix with -1 (unreachable)
    distances = torch.full((num_nodes, M), max_dist, dtype=torch.long)

    # Convert to list if given as tensor
    if isinstance(anchor_indices, torch.Tensor):
        anchor_indices = anchor_indices.tolist()

    for i, anchor in enumerate(anchor_indices):
        if anchor not in G:
            raise ValueError(f"Anchor node {anchor} not found in graph")
            
        # Get shortest paths using BFS
        try:
            shortest_lengths = nx.single_source_shortest_path_length(G, anchor)
            
            # Fill distances for this anchor column
            for node, dist in shortest_lengths.items():
                distances[node, i] = dist if dist < max_dist else max_dist
        except Exception as e:
            print(f"计算节点{anchor}的最短路径时发生错误: {e}")
            # 发生错误时，保持默认的max_dist值
    
    # 主动清理内存
    del G
    gc.collect()
    
    return distances

def pe_encoding_for_graph(
        args, graph, edge_label_index, edge_label, processed_pe_path=None,
    ):
    """
    With a given graph in dataset, do subgraph sampling and 
    then calculate the DSPD for the sampled subgraph.
    Args:
        args (argparse.Namespace): The arguments
        graph (torch_geometric.data.Data): The graph
        graph_name (str): The name of the graph
        edge_label_index (torch.Tensor): The edge label index
        edge_label (torch.Tensor): The edge label
        processed_pe_path (str): The path to save the DSPD per batch
    Return:
        loader: The loader with 'batch_size' for mini-batch training
        batch_dspd_list: The DSPDs of batches coming from the loader.
    """
    num_neighbors = -1
    path_exist = os.path.exists(processed_pe_path)
    

    ## If we do not use PE, just return the loader and an empty list
    if (not args.use_pe) or path_exist:
        ## The actual loader used in mini-batch training
        loader = LinkNeighborLoader(
            graph,
            num_neighbors=args.num_hops * [num_neighbors],
            edge_label_index=edge_label_index,
            edge_label=edge_label,
            subgraph_type='bidirectional',
            disjoint=True,
            batch_size=args.batch_size,
            shuffle=False, 
            num_workers=0,  # 设置为0，避免多进程问题
        )
        if path_exist and args.use_pe:
            print("Found existing file of dspd_per_batch!")
            print(f"Loading from {processed_pe_path}")

            with open (processed_pe_path, 'rb') as fp:
                dspd_per_batch = pickle.load(fp)
        
        else:
            dspd_per_batch = [None] * ((edge_label.size(0) + args.batch_size - 1) // args.batch_size)
        return loader, dspd_per_batch
    
    ## 分块处理以减少内存使用
    ## 计算总共需要处理的边数
    total_edges = edge_label_index.size(1)
    ## 设定每批处理的边数，防止内存溢出
    chunk_size = min(2000, total_edges)  # 可调整，根据可用内存大小
    num_chunks = (total_edges + chunk_size - 1) // chunk_size
    
    print(f"将PE计算分为{num_chunks}个块进行处理，每块{chunk_size}条边")
    
    dspd_per_subg = []
    gid_per_subg = []
    
    ## 按块处理PE计算
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, total_edges)
        
        chunk_edge_label_index = edge_label_index[:, start_idx:end_idx]
        chunk_edge_label = edge_label[start_idx:end_idx]
        
        ## Create a LinkNeighborLoader for subgraph sampling.
        ## For each edge_label_index, we sample a 'num_hops' subgraph.
        ## NOTE: This loader is only used for PE calculation.
        chunk_loader = LinkNeighborLoader(
            graph,
            num_neighbors=args.num_hops * [num_neighbors],
            edge_label_index=chunk_edge_label_index,
            edge_label=chunk_edge_label,
            subgraph_type='bidirectional',
            disjoint=True,
            batch_size=1, ## batch_size is always 1
            shuffle=False, 
            num_workers=1,  # 减少worker数量以减少内存消耗
        )
        
        ## 主动进行垃圾回收
        import gc
        gc.collect()
        
        ## Calculate the SPD for each batch
        for subgraph in tqdm(
            chunk_loader, 
            desc=f"块 {chunk_idx+1}/{num_chunks}: 子图采样和DSPD计算"
        ):
            try:
                spd = get_double_spd(
                    subgraph,
                    ## src and dst nodes in edge_label_index are always 
                    ## the first 2 nodes in the subgraph.
                    anchor_indices=[0, 1], max_dist=args.max_dist,
                )
                dspd_per_subg.append(spd)
                assert dspd_per_subg[-1].size(0) == subgraph.num_nodes
                gid_per_subg.append(subgraph.n_id)
            except Exception as e:
                print(f"DSPD计算错误: {e}")
                # 对于失败的计算，使用默认值
                dummy_dspd = torch.full((subgraph.num_nodes, 2), args.max_dist, dtype=torch.long)
                dspd_per_subg.append(dummy_dspd)
                gid_per_subg.append(subgraph.n_id)
        
        # 释放loader
        del chunk_loader
        gc.collect()
        
        # 每处理完一块保存一次中间结果
        if chunk_idx > 0 and chunk_idx % 5 == 0:
            print(f"保存中间结果到{processed_pe_path}.part{chunk_idx}")
            with open(f"{processed_pe_path}.part{chunk_idx}", 'wb') as fp:
                pickle.dump((dspd_per_subg, gid_per_subg), fp)

    ## The actual loader used in mini-batch training
    loader = LinkNeighborLoader(
        graph,
        num_neighbors=args.num_hops * [num_neighbors],
        edge_label_index=edge_label_index,
        edge_label=edge_label,
        subgraph_type='bidirectional',
        disjoint=True,
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=0,  # 设置为0，避免多进程问题
    )

    ## 计算需要多少批次
    num_batches = (edge_label.size(0) + args.batch_size - 1) // args.batch_size
    dspd_per_batch = [None] * num_batches
    
    ## match the DSPDs of subgraphs back to the data batches
    batch_counter = 0
    for b, batch in enumerate(
        tqdm(loader, desc='将DSPD映射回批次', leave=False)
    ):
        try:
            batched_dspd = torch.empty(
                (batch.num_nodes, 2), dtype=torch.long).fill_(args.max_dist)
            ## For each batrch, we have:
            ## batch.edge_label.size(0) == batch.edge_label_index.size(1)
            ## batch.batch.max()+1 == batch.input_id.size(0) == \
            num_subgraphs = batch.input_id.size(0)

            for i in range(num_subgraphs):
                subg_node_mask = batch.batch == i
                ## global subgraph id is the id of the sampled 'edge_label_index'
                global_subg_id = batch.input_id[i]
                # 确保索引在范围内
                if global_subg_id < len(dspd_per_subg):
                    batched_dspd[subg_node_mask] = dspd_per_subg[global_subg_id]
            
            ## store the dspd for each batch
            dspd_per_batch[batch_counter] = batched_dspd
            batch_counter += 1
        except Exception as e:
            print(f"批次{b}映射错误: {e}")
            # 对于失败的批次，使用默认值
            dspd_per_batch[batch_counter] = torch.full((batch.num_nodes, 2), args.max_dist, dtype=torch.long)
            batch_counter += 1
    
    ## save dspd_per_batch to file
    print(f"Saving dspd_per_batch to {processed_pe_path}")
    try:
        with open(processed_pe_path, 'wb') as fp:
            pickle.dump(dspd_per_batch, fp)
    except Exception as e:
        print(f"保存DSPD失败: {e}")
        print("尝试分块保存...")
        # 如果完整保存失败，尝试分块保存
        chunk_size = len(dspd_per_batch) // 4 + 1
        for i in range(0, len(dspd_per_batch), chunk_size):
            chunk = dspd_per_batch[i:i+chunk_size]
            chunk_path = f"{processed_pe_path}.{i//chunk_size}"
            with open(chunk_path, 'wb') as fp:
                pickle.dump(chunk, fp)

    return loader, dspd_per_batch

def dataset_sampling_and_pe_calculation(args, train_dataset, test_dataset):
    """ 
    Sampling subgraphs for each graph in dataset and 
    calculate the PE for each sampled subgraph.
    Args:
        args (argparse.Namespace): The arguments
        train_dataset (torch_geometric.data.InMemoryDataset): The training dataset
        test_dataset (torch_geometric.data.InMemoryDataset): The testing dataset
    Return:
        train_loader, val_loader, test_loaders, 
        train_subgraph_dspd_list, valid_subgraph_dspd_list, test_subgraph
    """
    # 合并训练数据集中的图
    num_train = len(train_dataset)
    if num_train > 1:
        train_graphs = [train_dataset[i] for i in range(num_train)]
        merged_graph = Batch.from_data_list(train_graphs)
        merged_graph.name = "merged_train_graph"
        print(f"Merged {num_train} training graphs into one graph with {merged_graph.num_nodes} nodes")
    else:
        merged_graph = train_dataset[0]
        print(f"Using graph '{merged_graph.name}' as primary training graph")
    
    ## get split for validation
    train_ind, val_ind = train_test_split(
        np.arange(merged_graph.edge_label.size(0)), 
        test_size=0.2, shuffle=True, #stratify=stratify,
    )
    train_ind = torch.tensor(train_ind, dtype=torch.long)
    val_ind = torch.tensor(val_ind, dtype=torch.long)

    train_edge_label_index = merged_graph.edge_label_index[:, train_ind]
    train_edge_label = merged_graph.edge_label[train_ind]
    dspd_name = f'_h{args.num_hops}_seed{args.seed}_train.dspd'
    processed_pe_path = os.path.join(
        os.path.dirname(train_dataset.processed_paths[0]), 
        dspd_name
    )
    train_loader, train_dspd_list = pe_encoding_for_graph(
        args, merged_graph, train_edge_label_index, train_edge_label, processed_pe_path
    )

    val_edge_label_index = merged_graph.edge_label_index[:, val_ind]
    val_edge_label = merged_graph.edge_label[val_ind]
    dspd_name = f'_h{args.num_hops}_seed{args.seed}_val.dspd'
    processed_pe_path = os.path.join(
        os.path.dirname(train_dataset.processed_paths[0]), 
        dspd_name
    )
    val_loader, valid_dspd_list = pe_encoding_for_graph(
        args, merged_graph, val_edge_label_index, val_edge_label, processed_pe_path
    )

    ## test data come from the rest datasets
    test_loaders = {}
    test_dspd_dict = {}
    for graph_idx in range(len(test_dataset)):
        test_graph = test_dataset[graph_idx]
        test_edge_label_index = test_graph.edge_label_index
        test_edge_label = test_graph.edge_label
        dspd_name = f'_h{args.num_hops}_seed{args.seed}_test_{graph_idx}.dspd'
        processed_pe_path = os.path.join(
            os.path.dirname(test_dataset.processed_paths[0]), 
            dspd_name
        )
        test_loader, test_dspd_list = pe_encoding_for_graph(
            args, test_graph, test_edge_label_index, test_edge_label, processed_pe_path
        )
        # 使用图名称作为key，如果没有名称则使用索引
        test_name = test_graph.name if hasattr(test_graph, 'name') else f'test_{graph_idx}'
        test_loaders[test_name] = (test_loader, test_dspd_list)
        test_dspd_dict[graph_idx] = test_dspd_list

    return (
        train_loader, val_loader, test_loaders,
        train_dspd_list, valid_dspd_list, test_dspd_dict,
    )


def pe_encoding_for_node_graph(
        args, graph, input_nodes, node_labels, processed_pe_path=None,
    ):
    """
    For node tasks: do subgraph sampling and calculate the SPD for each sampled subgraph.
    For node tasks, DSPD stores the same distance in both columns (distance to the target node).
    
    Args:
        args (argparse.Namespace): The arguments
        graph (torch_geometric.data.Data): The graph
        input_nodes (torch.Tensor): The input node indices
        node_labels (torch.Tensor): The node labels
        processed_pe_path (str): The path to save the SPD per batch
    Return:
        loader: The loader with 'batch_size' for mini-batch training
        batch_spd_list: The SPDs of batches coming from the loader.
    """
    num_neighbors = 128
    path_exist = os.path.exists(processed_pe_path) if processed_pe_path else False
    
    if (not args.use_pe) or path_exist:
        loader = NeighborLoader(
            graph,
            num_neighbors=args.num_hops * [num_neighbors],
            input_nodes=input_nodes,
            batch_size=args.batch_size,
            shuffle=False, 
            num_workers=0,
        )
        if path_exist and args.use_pe:
            print(f"Loading SPD from {processed_pe_path}")
            with open(processed_pe_path, 'rb') as fp:
                spd_per_batch = pickle.load(fp)
        else:
            spd_per_batch = [None] * ((len(input_nodes) + args.batch_size - 1) // args.batch_size)
        return loader, spd_per_batch
    
    # Calculate SPD for each node
    total_nodes = len(input_nodes)
    chunk_size = min(1000, total_nodes)  # 减小chunk_size以降低内存使用
    num_chunks = (total_nodes + chunk_size - 1) // chunk_size
    
    print(f"将节点PE计算分为{num_chunks}个块进行处理")
    
    spd_per_subg = []
    gid_per_subg = []
    
    import gc
    import torch
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, total_nodes)
        
        chunk_input_nodes = input_nodes[start_idx:end_idx]
        
        chunk_loader = NeighborLoader(
            graph,
            num_neighbors=args.num_hops * [num_neighbors],
            input_nodes=chunk_input_nodes,
            batch_size=1,
            shuffle=False, 
            num_workers=0,  # 使用0避免多进程内存问题
        )
        
        gc.collect()
        
        for subgraph in tqdm(
            chunk_loader, 
            desc=f"块 {chunk_idx+1}/{num_chunks}: 节点子图采样和SPD计算"
        ):
            try:
                # For node task, anchor is the first node (target node)
                spd = get_single_spd(subgraph, anchor_index=0, max_dist=args.max_dist)
                spd_per_subg.append(spd)
                gid_per_subg.append(subgraph.n_id)
            except Exception as e:
                print(f"SPD计算错误: {e}")
                dummy_spd = torch.full((subgraph.num_nodes, 2), args.max_dist, dtype=torch.long)
                spd_per_subg.append(dummy_spd)
                gid_per_subg.append(subgraph.n_id)
        
        del chunk_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Create actual loader
    loader = NeighborLoader(
        graph,
        num_neighbors=args.num_hops * [num_neighbors],
        input_nodes=input_nodes,
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=0,
    )
    
    num_batches = (len(input_nodes) + args.batch_size - 1) // args.batch_size
    spd_per_batch = [None] * num_batches
    
    batch_counter = 0
    for b, batch in enumerate(tqdm(loader, desc='将SPD映射回批次', leave=False)):
        try:
            batched_spd = torch.empty((batch.num_nodes, 2), dtype=torch.long).fill_(args.max_dist)
            num_subgraphs = batch.input_id.size(0) if hasattr(batch, 'input_id') else batch.batch.max().item() + 1
            
            for i in range(num_subgraphs):
                subg_node_mask = batch.batch == i
                global_subg_id = batch.input_id[i] if hasattr(batch, 'input_id') else i
                if global_subg_id < len(spd_per_subg):
                    batched_spd[subg_node_mask] = spd_per_subg[global_subg_id]
            
            spd_per_batch[batch_counter] = batched_spd
            batch_counter += 1
        except Exception as e:
            print(f"批次{b}映射错误: {e}")
            spd_per_batch[batch_counter] = torch.full((batch.num_nodes, 2), args.max_dist, dtype=torch.long)
            batch_counter += 1
    
    if processed_pe_path:
        print(f"Saving spd_per_batch to {processed_pe_path}")
        try:
            with open(processed_pe_path, 'wb') as fp:
                pickle.dump(spd_per_batch, fp)
        except Exception as e:
            print(f"保存SPD失败: {e}")
    
    return loader, spd_per_batch


def dataset_node_sampling_and_pe_calculation(args, train_dataset, test_dataset):
    """ 
    Sampling subgraphs for node tasks and calculate the PE.
    
    Args:
        args (argparse.Namespace): The arguments
        train_dataset: The training dataset
        test_dataset: The testing dataset
    Return:
        train_loader, val_loader, test_loaders,
        train_spd_list, valid_spd_list, test_spd_dict
    """
    # Merge training graphs
    num_train = len(train_dataset)
    if num_train > 1:
        train_graphs = [train_dataset[i] for i in range(num_train)]
        merged_graph = Batch.from_data_list(train_graphs)
        merged_graph.name = "merged_train_graph"
        print(f"Merged {num_train} training graphs into one graph with {merged_graph.num_nodes} nodes")
    else:
        merged_graph = train_dataset[0]
        print(f"Using graph '{merged_graph.name}' as primary training graph")
    
    # 获取所有节点索引
    all_train_indices = np.arange(merged_graph.num_nodes)
    
    # 根据数据集大小确定采样率
    is_large_dataset = any(name in ['sandwich', 'ultra8t'] for name in train_dataset.names)
    sample_rate = args.large_dataset_sample_rates if is_large_dataset else args.small_dataset_sample_rates
    
    # 如果采样率 < 1.0，则进行节点采样
    if sample_rate < 1.0:
        num_samples = int(len(all_train_indices) * sample_rate)
        all_train_indices = np.random.choice(all_train_indices, num_samples, replace=False)
        print(f"训练集节点采样: 从 {merged_graph.num_nodes} 个节点中采样 {num_samples} 个 (采样率: {sample_rate})")
    
    # Train/val split
    train_ind, val_ind = train_test_split(
        all_train_indices, 
        test_size=0.2, shuffle=True,
    )
    train_ind = torch.tensor(train_ind, dtype=torch.long)
    val_ind = torch.tensor(val_ind, dtype=torch.long)

    train_labels = merged_graph.y[train_ind]
    spd_name = f'_h{args.num_hops}_seed{args.seed}_node_train.spd'
    processed_pe_path = os.path.join(
        os.path.dirname(train_dataset.processed_paths[0]), 
        spd_name
    )
    train_loader, train_spd_list = pe_encoding_for_node_graph(
        args, merged_graph, train_ind, train_labels, processed_pe_path
    )

    val_labels = merged_graph.y[val_ind]
    spd_name = f'_h{args.num_hops}_seed{args.seed}_node_val.spd'
    processed_pe_path = os.path.join(
        os.path.dirname(train_dataset.processed_paths[0]), 
        spd_name
    )
    val_loader, valid_spd_list = pe_encoding_for_node_graph(
        args, merged_graph, val_ind, val_labels, processed_pe_path
    )

    # Test loaders
    test_loaders = []
    test_spd_dict = {}
    for graph_idx in range(len(test_dataset)):
        test_graph = test_dataset[graph_idx]
        graph_name = test_graph.name if hasattr(test_graph, 'name') else f'test_{graph_idx}'
        
        # 根据数据集名称确定采样率
        is_large_dataset = graph_name in ['sandwich', 'ultra8t']
        sample_rate = args.large_dataset_sample_rates if is_large_dataset else args.small_dataset_sample_rates
        
        # 对测试集进行采样
        if sample_rate < 1.0:
            num_test_samples = int(test_graph.num_nodes * sample_rate)
            test_nodes = torch.tensor(
                np.random.choice(test_graph.num_nodes, num_test_samples, replace=False),
                dtype=torch.long
            )
            print(f"测试集 {graph_name}: 从 {test_graph.num_nodes} 个节点中采样 {num_test_samples} 个 (采样率: {sample_rate})")
        else:
            test_nodes = torch.arange(test_graph.num_nodes)
        
        test_labels = test_graph.y[test_nodes] if sample_rate < 1.0 else test_graph.y
        spd_name = f'_h{args.num_hops}_seed{args.seed}_node_test_{graph_idx}.spd'
        processed_pe_path = os.path.join(
            os.path.dirname(test_dataset.processed_paths[0]), 
            spd_name
        )
        test_loader, test_spd_list = pe_encoding_for_node_graph(
            args, test_graph, test_nodes, test_labels, processed_pe_path
        )
        test_loaders.append(test_loader)
        test_spd_dict[graph_idx] = test_spd_list

    return (
        train_loader, val_loader, test_loaders,
        train_spd_list, valid_spd_list, test_spd_dict,
    )
