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
            num_workers=args.num_workers,  # 使用命令行参数统一配置
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
            num_workers=args.num_workers,  # 使用命令行参数统一配置
            persistent_workers=False if args.num_workers == 0 else True,  # 根据worker数量决定
        )
        
        ## 主动进行垃圾回收
        import gc
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
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
                
                # 及时释放子图内存
                del subgraph, spd
                
            except Exception as e:
                print(f"DSPD计算错误: {e}")
                # 对于失败的计算，使用默认值
                dummy_dspd = torch.full((subgraph.num_nodes, 2), args.max_dist, dtype=torch.long)
                dspd_per_subg.append(dummy_dspd)
                gid_per_subg.append(subgraph.n_id)
                del subgraph, dummy_dspd
        
        # 释放loader
        del chunk_loader
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 每处理完一块保存一次中间结果（更频繁地保存）
        if chunk_idx > 0 and chunk_idx % 3 == 0:  # 从5改为3，更频繁保存
            print(f"保存中间结果到{processed_pe_path}.part{chunk_idx}")
            with open(f"{processed_pe_path}.part{chunk_idx}", 'wb') as fp:
                pickle.dump((dspd_per_subg, gid_per_subg), fp)
            # 保存后清理部分内存
            gc.collect()

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
        num_workers=args.num_workers,  # 使用命令行参数统一配置
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
    ''''''
    # 合并训练数据集中的图
    train_graphs = []
    total_edge_labels = 0
    total_nodes = 0
    total_edges = 0
    
    # 收集所有的训练图
    for i in range(len(train_dataset)):
        train_graph = train_dataset[i]
        train_graphs.append(train_graph)
        total_edge_labels += train_graph.edge_label.size(0)
        total_nodes += train_graph.num_nodes
        total_edges += train_graph.edge_index.size(1)

    # 创建合并后的图
    merged_graph = train_graphs[0].__class__()
    
    # 合并节点特征
    merged_graph.x = torch.cat([g.x for g in train_graphs], dim=0)
    
    # 合并节点类型
    node_type_list = []
    for g in train_graphs:
        if hasattr(g, 'node_type'):
            node_type_list.append(g.node_type)
    if node_type_list:
        merged_graph.node_type = torch.cat(node_type_list, dim=0)
    
    # 合并边索引，需要调整索引
    edge_index_list = []
    edge_type_list = []
    node_offset = 0
    for g in train_graphs:
        edge_index = g.edge_index.clone()
        edge_index += node_offset
        edge_index_list.append(edge_index)
        edge_type_list.append(g.edge_type)
        node_offset += g.num_nodes
    merged_graph.edge_index = torch.cat(edge_index_list, dim=1)
    merged_graph.edge_type = torch.cat(edge_type_list, dim=0)
    
    # 合并边标签索引
    edge_label_index_list = []
    node_offset = 0
    for g in train_graphs:
        edge_label_index = g.edge_label_index.clone()
        edge_label_index += node_offset
        edge_label_index_list.append(edge_label_index)
        node_offset += g.num_nodes
    merged_graph.edge_label_index = torch.cat(edge_label_index_list, dim=1)
    
    # 合并边标签
    merged_graph.edge_label = torch.cat([g.edge_label for g in train_graphs], dim=0)
    
    # 合并节点属性
    merged_graph.node_attr = torch.cat([g.node_attr for g in train_graphs], dim=0)
    
        
    # 合并节点类型映射
    if hasattr(train_graphs[0], '_n2type'):
        merged_graph._n2type = train_graphs[0]._n2type.copy()
        merged_graph._num_ntypes = train_graphs[0]._num_ntypes
    
    # 合并边类型映射
    if hasattr(train_graphs[0], '_e2type'):
        merged_graph._e2type = train_graphs[0]._e2type.copy()
        merged_graph._num_etypes = train_graphs[0]._num_etypes
    
  
    
    
    # 设置其他属性
    merged_graph.num_nodes = total_nodes
    merged_graph.name = "merged_train_graph"
    
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
    test_loaders = []
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
        test_loaders.append(test_loader)
        test_dspd_dict[graph_idx] = test_dspd_list

    return (
        train_loader, val_loader, test_loaders,
        train_dspd_list, valid_dspd_list, test_dspd_dict,
    )


def node_pe_encoding_for_graph(
        args, graph, node_label_index, node_label, processed_pe_path=None,
    ):
    """
    节点任务的子图采样和 DSPD 计算
    对于节点任务，使用 anchor_indices=[0, 0]，即目标节点作为两个锚点
    
    Args:
        args: 参数
        graph: 图数据
        node_label_index: 目标节点索引
        node_label: 节点标签
        processed_pe_path: DSPD 保存路径
    Return:
        loader: NeighborLoader
        dspd_per_batch: 每个批次的 DSPD
    """
    num_neighbors = -1
    path_exist = os.path.exists(processed_pe_path) if processed_pe_path else False
    
    # 创建实际训练用的 loader (需要先创建，因为后面要遍历它来获取正确的批次大小)
    loader = NeighborLoader(
        graph,
        num_neighbors=args.num_hops * [num_neighbors],
        input_nodes=node_label_index,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,  # 使用命令行参数统一配置
    )
    loader.node_labels = node_label
    
    # 如果不使用 PE 或已有缓存，直接返回
    if (not args.use_pe) or path_exist:
        if path_exist and args.use_pe:
            print(f"Found existing DSPD file: {processed_pe_path}")
            with open(processed_pe_path, 'rb') as fp:
                dspd_per_batch = pickle.load(fp)
        else:
            # 不使用 PE 时，为每个批次创建默认的 DSPD
            dspd_per_batch = []
            for batch in loader:
                dspd_per_batch.append(torch.full((batch.num_nodes, 2), args.max_dist, dtype=torch.long))
            # 重新创建 loader (因为上面遍历消耗了)
            loader = NeighborLoader(
                graph,
                num_neighbors=args.num_hops * [num_neighbors],
                input_nodes=node_label_index,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,  # 使用命令行参数统一配置
            )
            loader.node_labels = node_label
        
        return loader, dspd_per_batch
    
    # 计算 DSPD - 直接按批次计算，避免映射问题
    print(f"节点任务 PE 计算: 直接按批次计算 DSPD")
    
    import gc
    
    dspd_per_batch = []
    
    # 遍历 loader，为每个批次计算 DSPD
    for batch_idx, batch in enumerate(tqdm(loader, desc='计算每个批次的 DSPD')):
        try:
            batched_dspd = torch.full(
                (batch.num_nodes, 2), args.max_dist, dtype=torch.long
            )
            
            # 获取批次中的子图数量
            num_subgraphs = batch.batch.max().item() + 1
            
            for i in range(num_subgraphs):
                # 获取当前子图的节点掩码
                subg_node_mask = batch.batch == i
                subg_num_nodes = subg_node_mask.sum().item()
                
                # 提取子图
                subg_node_indices = torch.where(subg_node_mask)[0]
                subg_edge_mask = (batch.edge_index[0].unsqueeze(1) == subg_node_indices).any(dim=1) & \
                                 (batch.edge_index[1].unsqueeze(1) == subg_node_indices).any(dim=1)
                
                # 创建子图的边索引 (重新映射节点索引)
                node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(subg_node_indices)}
                subg_edge_index = batch.edge_index[:, subg_edge_mask]
                remapped_edge_index = torch.tensor([
                    [node_mapping.get(src.item(), 0) for src in subg_edge_index[0]],
                    [node_mapping.get(dst.item(), 0) for dst in subg_edge_index[1]]
                ], dtype=torch.long)
                
                # 创建子图数据
                subgraph = Data(
                    edge_index=remapped_edge_index,
                    num_nodes=subg_num_nodes
                )
                
                # 计算 DSPD (目标节点是子图中的第一个节点，索引为 0)
                spd = get_double_spd(
                    subgraph,
                    anchor_indices=[0, 0],
                    max_dist=args.max_dist,
                )
                
                # 将 DSPD 填充到批次的对应位置
                batched_dspd[subg_node_mask] = spd
                
            dspd_per_batch.append(batched_dspd)
            
        except Exception as e:
            print(f"批次 {batch_idx} DSPD 计算错误: {e}")
            dspd_per_batch.append(torch.full(
                (batch.num_nodes, 2), args.max_dist, dtype=torch.long
            ))
        
        # 定期垃圾回收
        if batch_idx % 50 == 0:
            gc.collect()
    
    # 重新创建 loader (因为上面遍历消耗了)
    loader = NeighborLoader(
        graph,
        num_neighbors=args.num_hops * [num_neighbors],
        input_nodes=node_label_index,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,  # 使用命令行参数统一配置
    )
    loader.node_labels = node_label
    
    # 保存 DSPD
    if processed_pe_path:
        print(f"保存 DSPD 到 {processed_pe_path}")
        try:
            with open(processed_pe_path, 'wb') as fp:
                pickle.dump(dspd_per_batch, fp)
        except Exception as e:
            print(f"保存 DSPD 失败: {e}")
    
    return loader, dspd_per_batch


def node_dataset_sampling_and_pe_calculation(args, train_dataset, test_dataset):
    """
    节点任务的数据采样和 PE 计算
    
    标签格式保持 [N, 2]:
        - node_label[:, 0]: 归一化后的连续值 (用于回归)
        - node_label[:, 1]: 离散类别索引 (用于分类)
    
    标签列的选择在模型内部进行，这里保持完整的 [N, 2] 格式
    """
    # 合并训练数据集
    train_graphs = []
    total_nodes = 0
    
    for i in range(len(train_dataset)):
        train_graph = train_dataset[i]
        train_graphs.append(train_graph)
        total_nodes += train_graph.num_nodes
    
    # 创建合并后的图
    merged_graph = train_graphs[0].__class__()
    
    # 合并节点特征
    merged_graph.x = torch.cat([g.x for g in train_graphs], dim=0)
    
    # 合并 node_type
    if hasattr(train_graphs[0], 'node_type'):
        merged_graph.node_type = torch.cat([g.node_type for g in train_graphs], dim=0)
    
    # 合并 node_attr
    if hasattr(train_graphs[0], 'node_attr'):
        merged_graph.node_attr = torch.cat([g.node_attr for g in train_graphs], dim=0)
    
    # 合并边索引和边类型
    edge_index_list = []
    edge_type_list = []
    node_offset = 0
    for g in train_graphs:
        edge_index = g.edge_index.clone()
        edge_index += node_offset
        edge_index_list.append(edge_index)
        if hasattr(g, 'edge_type'):
            edge_type_list.append(g.edge_type)
        node_offset += g.num_nodes
    merged_graph.edge_index = torch.cat(edge_index_list, dim=1)
    if edge_type_list:
        merged_graph.edge_type = torch.cat(edge_type_list, dim=0)
    
    # 合并节点标签索引和标签 (保持 [N, 2] 格式)
    node_label_index_list = []
    node_label_list = []
    node_offset = 0
    for g in train_graphs:
        if hasattr(g, 'node_label_index') and g.node_label_index.size(0) > 0:
            node_label_index = g.node_label_index.clone()
            node_label_index += node_offset
            node_label_index_list.append(node_label_index)
            node_label_list.append(g.node_label)  # 保持 [N, 2] 格式
        node_offset += g.num_nodes
    
    merged_graph.node_label_index = torch.cat(node_label_index_list, dim=0)
    merged_graph.node_label = torch.cat(node_label_list, dim=0)  # [N, 2]
    merged_graph.num_nodes = total_nodes
    merged_graph.name = "merged_train_graph"
    
    # 合并类型映射
    if hasattr(train_graphs[0], '_n2type'):
        merged_graph._n2type = train_graphs[0]._n2type.copy()
        merged_graph._num_ntypes = train_graphs[0]._num_ntypes
    if hasattr(train_graphs[0], '_e2type'):
        merged_graph._e2type = train_graphs[0]._e2type.copy()
        merged_graph._num_etypes = train_graphs[0]._num_etypes
    
    print(f"合并后训练图: nodes={merged_graph.num_nodes}, target_nodes={merged_graph.node_label_index.size(0)}")
    print(f"节点标签格式: {merged_graph.node_label.shape} (第0列=回归, 第1列=分类)")
    
    # 训练/验证集划分
    num_target_nodes = merged_graph.node_label_index.size(0)
    train_ind, val_ind = train_test_split(
        np.arange(num_target_nodes),
        test_size=0.2, shuffle=True, random_state=args.seed
    )
    train_ind = torch.tensor(train_ind, dtype=torch.long)
    val_ind = torch.tensor(val_ind, dtype=torch.long)
    
    # 保持完整的 [N, 2] 标签格式 (在模型内部根据任务类型选择列)
    train_node_labels = merged_graph.node_label[train_ind]  # [N_train, 2]
    val_node_labels = merged_graph.node_label[val_ind]  # [N_val, 2]
    
    print(f"训练集: {train_node_labels.size(0)} 个节点")
    print(f"验证集: {val_node_labels.size(0)} 个节点")
    
    # 训练集
    train_node_indices = merged_graph.node_label_index[train_ind]
    dspd_name = f'_h{args.num_hops}_seed{args.seed}_node_train.dspd'
    processed_pe_path = os.path.join(
        os.path.dirname(train_dataset.processed_paths[0]),
        dspd_name
    )
    train_loader, train_dspd_list = node_pe_encoding_for_graph(
        args, merged_graph, train_node_indices, train_node_labels, processed_pe_path
    )
    
    # 验证集
    val_node_indices = merged_graph.node_label_index[val_ind]
    dspd_name = f'_h{args.num_hops}_seed{args.seed}_node_val.dspd'
    processed_pe_path = os.path.join(
        os.path.dirname(train_dataset.processed_paths[0]),
        dspd_name
    )
    val_loader, valid_dspd_list = node_pe_encoding_for_graph(
        args, merged_graph, val_node_indices, val_node_labels, processed_pe_path
    )
    
    # 测试集 (保持 [N, 2] 格式)
    test_loaders = []
    test_dspd_dict = {}
    for graph_idx in range(len(test_dataset)):
        test_graph = test_dataset[graph_idx]
        test_node_indices = test_graph.node_label_index
        test_node_labels = test_graph.node_label  # [N, 2]
        
        dspd_name = f'_h{args.num_hops}_seed{args.seed}_node_test_{graph_idx}.dspd'
        processed_pe_path = os.path.join(
            os.path.dirname(test_dataset.processed_paths[0]),
            dspd_name
        )
        test_loader, test_dspd_list = node_pe_encoding_for_graph(
            args, test_graph, test_node_indices, test_node_labels, processed_pe_path
        )
        test_loaders.append(test_loader)
        test_dspd_dict[graph_idx] = test_dspd_list
    
    return (
        train_loader, val_loader, test_loaders,
        train_dspd_list, valid_dspd_list, test_dspd_dict,
    )
