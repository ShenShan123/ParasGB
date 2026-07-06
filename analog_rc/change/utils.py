"""
工具函数模块。
"""
import torch
import os
from typing import List


def analyze_node_features(data_dir: str, case_ids: List[str] = None, save_path: str = "./stastic/node_features_analysis.txt"):
    """
    分析每个图的三种节点类型的节点特征，打印每个维度的值范围、是否离散、含义。
    
    Args:
        data_dir: 数据目录路径
        case_ids: 要分析的case ID列表，None则分析目录下所有case
        save_path: 保存结果的txt文件路径
    
    Usage:
        from utils import analyze_node_features
        analyze_node_features("D:/desktop/github_push/rcg_v2/data")
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    f = open(save_path, 'w', encoding='utf-8')
    
    def log(msg=""):
        print(msg)
        f.write(msg + "\n")
    
    # 特征名称和含义定义
    FEATURE_INFO = {
        'device': {
            'names': ['is_mos', 'is_res', 'is_cap', 'w', 'l', 'm', 'nf', 'ad', 'as', 'pd', 'ps', 
                      'nrd', 'nrs', 'sa', 'sb', 'is_pmos'],
            'meanings': [
                '是否为MOS管',
                '是否为电阻',
                '是否为电容',
                '沟道宽度 (Width)',
                '沟道长度 (Length)', 
                '并联倍数 (Multiplier)',
                '指数 (Number of Fingers)',
                '漏极面积 (Drain Area)',
                '源极面积 (Source Area)',
                '漏极周长 (Drain Perimeter)',
                '源极周长 (Source Perimeter)',
                '漏极方块电阻数',
                '源极方块电阻数',
                '应力参数 SA',
                '应力参数 SB',
                '是否为PMOS (1=PMOS, 0=NMOS)'
            ]
        },
        'net': {
            'names': ['is_power', 'is_ground', 'is_signal', 'is_top_level', 'num_devices', 
                      'gate_count', 'sd_count', 'bulk_count', 'total_L', 'total_W'],
            'meanings': [
                '是否为电源网络 (VDD)',
                '是否为地网络 (VSS/GND)',
                '是否为信号网络',
                '是否为顶层网络',
                '连接的器件数量',
                '连接到栅极的数量',
                '连接到源/漏的数量',
                '连接到衬底的数量',
                '连接器件的总沟道长度',
                '连接器件的总沟道宽度'
            ]
        },
        'pin': {
            'names': ['is_drain', 'is_gate', 'is_source', 'is_bulk', 'is_cap_terminal', 'is_res_terminal'],
            'meanings': [
                '是否连接到MOS漏极 (D)',
                '是否连接到MOS栅极 (G)',
                '是否连接到MOS源极 (S)',
                '是否连接到MOS衬底 (B)',
                '是否连接到电容端子',
                '是否连接到电阻端子 (或无活动连接标志)'
            ]
        }
    }
    
    # 获取所有case文件
    if case_ids is None:
        files = [ff for ff in os.listdir(data_dir) if ff.endswith('_RC.pt')]
        case_ids = sorted([ff.replace('case', '').replace('_RC.pt', '') for ff in files], key=lambda x: int(x))
    
    log(f"分析 {len(case_ids)} 个case的节点特征...")
    log(f"Case IDs: {case_ids}\n")
    
    # 用于汇总所有case的特征数据
    all_features = {
        'dev': {},  # {dim: [所有值]}
        'pin': {},
        'net': {}
    }
    
    # ==================== 第一部分：每个图的详细信息 ====================
    for cid in case_ids:
        filepath = os.path.join(data_dir, f"case{cid}_RC.pt")
        if not os.path.exists(filepath):
            log(f"Warning: {filepath} not found, skipping\n")
            continue
        
        # 直接加载原始异构图
        hg = torch.load(filepath, weights_only=False)
        if isinstance(hg, list):
            hg = hg[0]
        
        log("=" * 100)
        log(f"Case {cid}")
        log("=" * 100)
        
        # 分析每种节点类型
        for type_name in ['dev', 'pin', 'net']:
            if type_name not in hg.node_types:
                log(f"\n  {type_name.upper()}: 不存在")
                continue
            
            if not hasattr(hg[type_name], 'x'):
                log(f"\n  {type_name.upper()}: 无特征")
                continue
            
            x = hg[type_name].x
            num_nodes = x.size(0)
            num_dims = x.size(1)
            info = FEATURE_INFO.get(type_name if type_name != 'dev' else 'device', {'names': [], 'meanings': []})
            
            log(f"\n  {type_name.upper()} ({num_nodes} 节点, {num_dims} 维)")
            log(f"  {'维度':<4} {'名称':<12} {'类型':<8} {'范围/值':<50} {'含义'}")
            log(f"  " + "-" * 110)
            
            for dim in range(num_dims):
                col = x[:, dim]
                min_val = col.min().item()
                max_val = col.max().item()
                unique_vals = torch.unique(col)
                num_unique = len(unique_vals)
                
                # 收集数据用于汇总
                if dim not in all_features[type_name]:
                    all_features[type_name][dim] = []
                all_features[type_name][dim].extend(col.tolist())
                
                # 判断是否离散（唯一值<=100且为整数）
                is_integer = torch.all(col == col.long().float()).item()
                is_discrete = num_unique <= 100 and is_integer
                
                # 获取特征名称和含义
                feat_name = info['names'][dim] if dim < len(info['names']) else f'feat_{dim}'
                feat_meaning = info['meanings'][dim] if dim < len(info['meanings']) else '未知'
                
                if is_discrete:
                    type_str = "离散"
                    # 显示离散值列表，标出最大值
                    vals_list = sorted([int(v) for v in unique_vals.tolist()])
                    max_discrete = vals_list[-1]
                    if len(vals_list) <= 20:
                        range_str = f"{vals_list}"
                    else:
                        range_str = f"[0..{max_discrete}] ({num_unique}个值, max={max_discrete})"
                else:
                    type_str = "连续"
                    range_str = f"[{min_val:.16e}, {max_val:.16e}]"
                
                log(f"  {dim:<4} {feat_name:<12} {type_str:<8} {range_str:<50} {feat_meaning}")
        
        log()
    
    # ==================== 第二部分：所有图的汇总统计 ====================
    log("\n")
    log("=" * 100)
    log("全局汇总 (所有Case)")
    log("=" * 100)
    
    for type_name in ['dev', 'pin', 'net']:
        info = FEATURE_INFO.get(type_name if type_name != 'dev' else 'device', {'names': [], 'meanings': []})
        
        if not all_features[type_name]:
            log(f"\n  {type_name.upper()}: 无数据")
            continue
        
        num_dims = max(all_features[type_name].keys()) + 1
        total_nodes = len(all_features[type_name].get(0, []))
        
        log(f"\n  {type_name.upper()} (共 {total_nodes} 节点, {num_dims} 维)")
        log(f"  {'维度':<4} {'名称':<12} {'类型':<8} {'范围/值':<60} {'含义'}")
        log(f"  " + "-" * 120)
        
        for dim in range(num_dims):
            if dim not in all_features[type_name]:
                continue
            
            values = torch.tensor(all_features[type_name][dim])
            min_val = values.min().item()
            max_val = values.max().item()
            unique_vals = torch.unique(values)
            num_unique = len(unique_vals)
            
            # 判断是否离散（唯一值<=100且为整数）
            is_integer = torch.all(values == values.long().float()).item()
            is_discrete = num_unique <= 100 and is_integer
            
            # 获取特征名称和含义
            feat_name = info['names'][dim] if dim < len(info['names']) else f'feat_{dim}'
            feat_meaning = info['meanings'][dim] if dim < len(info['meanings']) else '未知'
            
            if is_discrete:
                type_str = "离散"
                vals_list = sorted([int(v) for v in unique_vals.tolist()])
                max_discrete = vals_list[-1]
                if len(vals_list) <= 20:
                    range_str = f"{vals_list}"
                else:
                    range_str = f"[0..{max_discrete}] ({num_unique}个值, max={max_discrete})"
            else:
                type_str = "连续"
                range_str = f"[{min_val:.16e}, {max_val:.16e}]"
            
            log(f"  {dim:<4} {feat_name:<12} {type_str:<8} {range_str:<60} {feat_meaning}")
    
    log("\n")
    log("=" * 100)
    log("分析完成")
    log("=" * 100)
    
    f.close()
    print(f"\n结果已保存到: {save_path}")


def analyze_graph_structure(data_dir: str, case_ids: List[str] = None, save_path: str = "./stastic/graph_structure_analysis.xlsx", save_to_file: bool = True):
    """
    分析每个图的结构特征，生成两个Excel表格：
    1. 基础统计表：节点数、边数、标签统计
    2. 结构特征表：度统计、密度、聚类系数等
    
    Args:
        data_dir: 数据目录路径
        case_ids: 要分析的case ID列表，None则分析目录下所有case
        save_path: 保存Excel的路径
        save_to_file: 是否保存到文件（默认True）
    """
    import numpy as np
    import pandas as pd
    import networkx as nx
    
    # 只有在需要保存文件时才创建目录
    if save_to_file and save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 获取所有case文件
    if case_ids is None:
        files = [f for f in os.listdir(data_dir) if f.endswith('_RC.pt')]
        case_ids = sorted([f.replace('case', '').replace('_RC.pt', '') for f in files], key=lambda x: int(x))
    
    print(f"分析 {len(case_ids)} 个case的图结构...")
    
    basic_results = []  # 基础统计
    struct_results = []  # 结构特征
    
    for cid in case_ids:
        filepath = os.path.join(data_dir, f"case{cid}_RC.pt")
        if not os.path.exists(filepath):
            print(f"  Warning: {filepath} not found, skipping")
            continue
        
        # 加载原始异构图
        hg = torch.load(filepath, weights_only=False)
        if isinstance(hg, list):
            hg = hg[0]
        
        # 节点统计
        num_device = hg['dev'].num_nodes if 'dev' in hg.node_types else 0
        num_pin = hg['pin'].num_nodes if 'pin' in hg.node_types else 0
        num_net = hg['net'].num_nodes if 'net' in hg.node_types else 0
        total_nodes = num_device + num_pin + num_net
        
        # 边统计
        edge_counts = {}
        total_edges = 0
        for etype in hg.edge_types:
            count = hg[etype].edge_index.shape[1]
            edge_counts[etype] = count
            total_edges += count
        
        num_pair_to = edge_counts.get(('pin', 'pair_to', 'pin'), 0)
        num_pin_net = edge_counts.get(('pin', 'belongs_to', 'net'), 0)
        num_dev_pin = edge_counts.get(('dev', 'connects_to', 'pin'), 0)
        num_dev_net = edge_counts.get(('dev', 'connects_to', 'net'), 0)
        
        # 构建 NetworkX 图用于高级分析
        G = nx.Graph()
        G.add_nodes_from(range(total_nodes))
        
        all_edges = []
        for etype in hg.edge_types:
            ei = hg[etype].edge_index
            all_edges.append(ei)
            edges = ei.t().numpy().tolist()
            G.add_edges_from(edges)
        
        # 度统计
        if all_edges:
            combined_edges = torch.cat(all_edges, dim=1)
            all_nodes_in_edges = torch.cat([combined_edges[0], combined_edges[1]])
            degrees = torch.bincount(all_nodes_in_edges, minlength=total_nodes).float()
            avg_degree = degrees.mean().item()
            max_degree = degrees.max().item()
            min_degree = degrees.min().item()
            std_degree = degrees.std().item()
        else:
            avg_degree = max_degree = min_degree = std_degree = 0
        
        # 图密度
        if total_nodes > 1:
            density = 2 * total_edges / (total_nodes * (total_nodes - 1))
        else:
            density = 0
        
        # 聚类系数
        try:
            clustering_coef = nx.average_clustering(G)
        except:
            clustering_coef = 0
        
        # 传递性
        try:
            transitivity = nx.transitivity(G)
        except:
            transitivity = 0
        
        # 连通分量
        num_components = nx.number_connected_components(G)
        if num_components > 0:
            largest_cc = max(nx.connected_components(G), key=len)
            largest_cc_size = len(largest_cc)
            largest_cc_ratio = largest_cc_size / total_nodes
        else:
            largest_cc_size = 0
            largest_cc_ratio = 0
        
        # 度相关性
        try:
            assortativity = nx.degree_assortativity_coefficient(G)
        except:
            assortativity = 0
        
        # 节点标签统计
        if 'net' in hg.node_types and hasattr(hg['net'], 'y'):
            net_y = hg['net'].y.numpy().flatten()
            node_label_mean = np.mean(net_y)
            node_label_std = np.std(net_y)
            node_label_min = np.min(net_y)
            node_label_max = np.max(net_y)
        else:
            node_label_mean = node_label_std = node_label_min = node_label_max = 0
        
        # 边标签统计 (边标签存储在 .y 中)
        if ('pin', 'pair_to', 'pin') in hg.edge_types and hasattr(hg[('pin', 'pair_to', 'pin')], 'y'):
            edge_y = hg[('pin', 'pair_to', 'pin')].y.numpy().flatten()
            edge_label_mean = np.mean(edge_y)
            edge_label_std = np.std(edge_y)
            edge_label_min = np.min(edge_y)
            edge_label_max = np.max(edge_y)
        else:
            edge_label_mean = edge_label_std = edge_label_min = edge_label_max = 0
        
        # 基础统计表
        # 节点标签使用科学计数法 (电容值 ~1e-13)，边标签保持原样 (电阻值 0-700)
        basic_results.append({
            'case_id': cid,
            'total_nodes': total_nodes,
            'num_device': num_device,
            'num_pin': num_pin,
            'num_net': num_net,
            'total_edges': total_edges,
            'num_pair_to': num_pair_to,
            'num_pin_net': num_pin_net,
            'num_dev_pin': num_dev_pin,
            'num_dev_net': num_dev_net,
            'node_label_mean': f'{node_label_mean:.2e}',
            'node_label_std': f'{node_label_std:.2e}',
            'node_label_min': f'{node_label_min:.2e}',
            'node_label_max': f'{node_label_max:.2e}',
            'edge_label_mean': round(edge_label_mean, 2),
            'edge_label_std': round(edge_label_std, 2),
            'edge_label_min': round(edge_label_min, 2),
            'edge_label_max': round(edge_label_max, 2),
        })
        
        # 结构特征表
        struct_results.append({
            'case_id': cid,
            'avg_degree': round(avg_degree, 2),
            'max_degree': int(max_degree),
            'min_degree': int(min_degree),
            'std_degree': round(std_degree, 2),
            'density': round(density, 6),
            'clustering_coef': round(clustering_coef, 4),
            'transitivity': round(transitivity, 4),
            'num_components': num_components,
            'largest_cc_size': largest_cc_size,
            'largest_cc_ratio': round(largest_cc_ratio, 4),
            'assortativity': round(assortativity, 4) if not np.isnan(assortativity) else 0,
        })
        
        print(f"  Case {cid}: {total_nodes} nodes, {total_edges} edges")
    
    # 保存为Excel (两个sheet) - 只有当 save_to_file=True 时才保存
    if save_to_file and save_path:
        df_basic = pd.DataFrame(basic_results)
        df_struct = pd.DataFrame(struct_results)
        
        with pd.ExcelWriter(save_path) as writer:
            df_basic.to_excel(writer, sheet_name='Basic_Stats', index=False)
            df_struct.to_excel(writer, sheet_name='Structure_Features', index=False)
        
        print(f"\n结果已保存到: {save_path}")
        print(f"  - Sheet 'Basic_Stats': 节点/边/标签统计")
        print(f"  - Sheet 'Structure_Features': 度/密度/聚类等结构特征")
    
    return basic_results, struct_results


def compute_graph_similarity(data_dir: str, case_ids: List[str] = None, save_path: str = "./stastic/graph_similarity.xlsx"):
    """
    计算图之间的相似度矩阵，基于多个结构特征。
    
    使用的特征：
    - 节点数比例 (device/pin/net)
    - 平均度、度标准差
    - 图密度
    - 聚类系数
    - 度相关性
    
    Args:
        data_dir: 数据目录路径
        case_ids: 要分析的case ID列表
        save_path: 保存相似度矩阵的路径
    """
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from scipy.spatial.distance import pdist, squareform
    
    # 先获取图结构特征 - 不保存到文件
    basic_results, struct_results = analyze_graph_structure(data_dir, case_ids, save_path=None, save_to_file=False)
    
    if not struct_results:
        print("没有数据可分析")
        return
    
    # 提取用于相似度计算的特征（只用结构特征，不用规模特征）
    feature_names = [
        'avg_degree', 'std_degree',
        'density', 'clustering_coef', 'transitivity', 'assortativity'
    ]
    
    case_ids_list = [r['case_id'] for r in struct_results]
    features = np.array([[r[f] for f in feature_names] for r in struct_results])
    
    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 计算欧氏距离矩阵
    distances = squareform(pdist(features_scaled, metric='euclidean'))
    
    # 转换为相似度 (1 / (1 + distance))
    similarity = 1 / (1 + distances)
    
    # 保存距离矩阵
    df_dist = pd.DataFrame(distances, index=case_ids_list, columns=case_ids_list)
    df_sim = pd.DataFrame(similarity, index=case_ids_list, columns=case_ids_list)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with pd.ExcelWriter(save_path) as writer:
        df_dist.to_excel(writer, sheet_name='Distance')
        df_sim.to_excel(writer, sheet_name='Similarity')
    
    print(f"\n相似度矩阵已保存到: {save_path}")
    
    # 找出最相似的图对
    print("\n" + "=" * 60)
    print("最相似的图对 (距离最小):")
    print("=" * 60)
    
    n = len(case_ids_list)
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((case_ids_list[i], case_ids_list[j], distances[i, j]))
    
    pairs.sort(key=lambda x: x[2])
    
    for i, (c1, c2, dist) in enumerate(pairs[:10]):
        print(f"  {i+1}. Case {c1} <-> Case {c2}: distance = {dist:.4f}")
    
    # 找出最不相似的图对
    print("\n最不相似的图对 (距离最大):")
    for i, (c1, c2, dist) in enumerate(pairs[-5:]):
        print(f"  {i+1}. Case {c1} <-> Case {c2}: distance = {dist:.4f}")
    
    return df_dist, df_sim


def compute_graph_similarity_wl_kernel(data_dir: str, case_ids: List[str] = None, 
                                        save_path: str = "./stastic/graph_similarity_wl.xlsx",
                                        n_iter: int = 3):  # 降低默认迭代次数从5到3
    """
    使用 Weisfeiler-Lehman (WL) Graph Kernel 计算图相似度。
    
    论文依据: Shervashidze et al., "Weisfeiler-Lehman Graph Kernels", JMLR 2011
    
    WL Kernel 通过迭代地聚合邻居标签来捕获图的结构信息，
    是最经典和广泛使用的图核方法之一。
    
    Args:
        data_dir: 数据目录路径
        case_ids: 要分析的case ID列表
        save_path: 保存相似度矩阵的路径
        n_iter: WL迭代次数 (默认3，降低以节省内存)
    
    Returns:
        similarity_matrix: 相似度矩阵 DataFrame
    """
    import numpy as np
    import pandas as pd
    import networkx as nx
    from collections import Counter
    import hashlib
    
    # 获取所有case文件
    if case_ids is None:
        files = [f for f in os.listdir(data_dir) if f.endswith('_RC.pt')]
        case_ids = sorted([f.replace('case', '').replace('_RC.pt', '') for f in files], key=lambda x: int(x))
    
    print(f"使用 WL Kernel 计算 {len(case_ids)} 个图的相似度...")
    print(f"论文: Shervashidze et al., 'Weisfeiler-Lehman Graph Kernels', JMLR 2011")
    
    graphs = []
    valid_case_ids = []
    
    for cid in case_ids:
        filepath = os.path.join(data_dir, f"case{cid}_RC.pt")
        if not os.path.exists(filepath):
            continue
        
        hg = torch.load(filepath, weights_only=False)
        if isinstance(hg, list):
            hg = hg[0]
        
        # 构建 NetworkX 图
        G = nx.Graph()
        
        # 添加节点和类型标签
        node_offset = 0
        node_labels = {}
        
        for ntype in ['dev', 'pin', 'net']:
            if ntype in hg.node_types:
                num_nodes = hg[ntype].num_nodes
                for i in range(num_nodes):
                    node_id = node_offset + i
                    G.add_node(node_id)
                    node_labels[node_id] = ntype  # 用节点类型作为初始标签
                node_offset += num_nodes
        
        # 添加边
        for etype in hg.edge_types:
            ei = hg[etype].edge_index
            edges = ei.t().numpy().tolist()
            G.add_edges_from(edges)
        
        nx.set_node_attributes(G, node_labels, 'label')
        graphs.append(G)
        valid_case_ids.append(cid)
        print(f"  Case {cid}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    def wl_subtree_kernel(G1, G2, n_iter=3):
        """计算两个图之间的 WL subtree kernel (优化版本，使用哈希避免内存溢出)"""
        
        def get_wl_labels(G, n_iter):
            """获取 WL 迭代后的标签直方图 (使用哈希代替字符串拼接)"""
            # 初始标签：使用节点类型
            labels = {n: hash(str(G.nodes[n].get('label', 0))) for n in G.nodes()}
            all_histograms = [Counter(labels.values())]
            
            for iteration in range(n_iter):
                new_labels = {}
                for node in G.nodes():
                    # 收集邻居标签并排序
                    neighbor_labels = sorted([labels[n] for n in G.neighbors(node)])
                    
                    # 使用哈希组合标签，避免字符串拼接导致的内存爆炸
                    # 将当前标签和邻居标签组合
                    combined = str(labels[node]) + '_' + '_'.join(map(str, neighbor_labels[:50]))  # 限制邻居数量
                    
                    # 使用MD5哈希压缩标签
                    new_labels[node] = int(hashlib.md5(combined.encode()).hexdigest()[:16], 16)
                
                labels = new_labels
                all_histograms.append(Counter(labels.values()))
            
            return all_histograms
        
        hist1 = get_wl_labels(G1, n_iter)
        hist2 = get_wl_labels(G2, n_iter)
        
        # 计算核值 (所有迭代的直方图内积之和)
        kernel_value = 0
        for h1, h2 in zip(hist1, hist2):
            all_keys = set(h1.keys()) | set(h2.keys())
            for key in all_keys:
                kernel_value += h1.get(key, 0) * h2.get(key, 0)
        
        return kernel_value
    
    # 计算核矩阵
    n = len(graphs)
    kernel_matrix = np.zeros((n, n))
    
    print(f"\n计算 {n}x{n} 核矩阵...")
    for i in range(n):
        for j in range(i, n):
            k = wl_subtree_kernel(graphs[i], graphs[j], n_iter)
            kernel_matrix[i, j] = k
            kernel_matrix[j, i] = k
        if (i + 1) % 5 == 0:
            print(f"  进度: {i+1}/{n}")
    
    # 归一化核矩阵得到相似度
    diag = np.sqrt(np.diag(kernel_matrix))
    similarity = kernel_matrix / np.outer(diag, diag)
    np.fill_diagonal(similarity, 1.0)
    
    # 保存结果
    df_sim = pd.DataFrame(similarity, index=valid_case_ids, columns=valid_case_ids)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_sim.to_excel(save_path)
    
    print(f"\nWL Kernel 相似度矩阵已保存到: {save_path}")
    
    # 绘制热力图
    _plot_similarity_heatmap(df_sim, valid_case_ids, save_path, title="WL Kernel Similarity")
    
    _print_similarity_pairs(df_sim, valid_case_ids)
    
    return df_sim


def compute_graph_similarity_spectral(data_dir: str, case_ids: List[str] = None,
                                       save_path: str = "./stastic/graph_similarity_spectral.xlsx",
                                       k: int = 50):
    """
    使用 Spectral (谱) 方法计算图相似度。
    
    论文依据: 
    - Chung, "Spectral Graph Theory", 1997
    - Wilson & Zhu, "A Study of Graph Spectra for Comparing Graphs", BMVC 2008
    
    基于 Laplacian 矩阵的特征值分布来比较图结构。
    谱方法能捕获图的全局拓扑特性。
    
    Args:
        data_dir: 数据目录路径
        case_ids: 要分析的case ID列表
        save_path: 保存相似度矩阵的路径
        k: 使用的特征值数量 (默认50)
    
    Returns:
        similarity_matrix: 相似度矩阵 DataFrame
    """
    import numpy as np
    import pandas as pd
    import networkx as nx
    from scipy.spatial.distance import pdist, squareform
    
    if case_ids is None:
        files = [f for f in os.listdir(data_dir) if f.endswith('_RC.pt')]
        case_ids = sorted([f.replace('case', '').replace('_RC.pt', '') for f in files], key=lambda x: int(x))
    
    print(f"使用 Spectral 方法计算 {len(case_ids)} 个图的相似度...")
    print(f"论文: Chung, 'Spectral Graph Theory', 1997")
    print(f"      Wilson & Zhu, 'A Study of Graph Spectra', BMVC 2008")
    
    spectral_features = []
    valid_case_ids = []
    
    for cid in case_ids:
        filepath = os.path.join(data_dir, f"case{cid}_RC.pt")
        if not os.path.exists(filepath):
            continue
        
        hg = torch.load(filepath, weights_only=False)
        if isinstance(hg, list):
            hg = hg[0]
        
        # 构建 NetworkX 图
        G = nx.Graph()
        total_nodes = sum(hg[ntype].num_nodes for ntype in hg.node_types if ntype in ['dev', 'pin', 'net'])
        G.add_nodes_from(range(total_nodes))
        
        for etype in hg.edge_types:
            ei = hg[etype].edge_index
            edges = ei.t().numpy().tolist()
            G.add_edges_from(edges)
        
        # 计算归一化 Laplacian 的特征值
        try:
            # 获取最大连通分量
            if nx.is_connected(G):
                largest_cc = G
            else:
                largest_cc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
            
            # 计算归一化 Laplacian 特征值
            L = nx.normalized_laplacian_matrix(largest_cc).toarray()
            eigenvalues = np.linalg.eigvalsh(L)
            eigenvalues = np.sort(eigenvalues)
            
            # 取前 k 个特征值作为特征向量
            if len(eigenvalues) < k:
                # 填充零
                padded = np.zeros(k)
                padded[:len(eigenvalues)] = eigenvalues
                eigenvalues = padded
            else:
                eigenvalues = eigenvalues[:k]
            
            spectral_features.append(eigenvalues)
            valid_case_ids.append(cid)
            print(f"  Case {cid}: {largest_cc.number_of_nodes()} nodes (largest CC)")
            
        except Exception as e:
            print(f"  Case {cid}: 计算失败 - {e}")
    
    # 计算谱距离
    features = np.array(spectral_features)
    distances = squareform(pdist(features, metric='euclidean'))
    
    # 转换为相似度
    similarity = 1 / (1 + distances)
    
    # 保存结果
    df_sim = pd.DataFrame(similarity, index=valid_case_ids, columns=valid_case_ids)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_sim.to_excel(save_path)
    
    print(f"\nSpectral 相似度矩阵已保存到: {save_path}")
    
    # 绘制热力图
    _plot_similarity_heatmap(df_sim, valid_case_ids, save_path, title="Spectral Similarity")
    
    _print_similarity_pairs(df_sim, valid_case_ids)
    
    return df_sim


def compute_graph_similarity_netlsd(data_dir: str, case_ids: List[str] = None,
                                     save_path: str = "./stastic/graph_similarity_netlsd.xlsx",
                                     timescales: int = 250):
    """
    使用 NetLSD (Network Laplacian Spectral Descriptor) 计算图相似度。
    
    论文依据: Tsitsulin et al., "NetLSD: Hearing the Shape of a Graph", KDD 2018
    
    NetLSD 基于 Laplacian 矩阵的热核迹 (heat kernel trace)，
    是一种高效且理论上有保证的图相似度度量方法。
    
    Args:
        data_dir: 数据目录路径
        case_ids: 要分析的case ID列表
        save_path: 保存相似度矩阵的路径
        timescales: 热扩散时间尺度数量 (默认250)
    
    Returns:
        similarity_matrix: 相似度矩阵 DataFrame
    """
    import numpy as np
    import pandas as pd
    import networkx as nx
    from scipy.spatial.distance import pdist, squareform
    
    if case_ids is None:
        files = [f for f in os.listdir(data_dir) if f.endswith('_RC.pt')]
        case_ids = sorted([f.replace('case', '').replace('_RC.pt', '') for f in files], key=lambda x: int(x))
    
    print(f"使用 NetLSD 计算 {len(case_ids)} 个图的相似度...")
    print(f"论文: Tsitsulin et al., 'NetLSD: Hearing the Shape of a Graph', KDD 2018")
    
    def compute_netlsd(G, timescales=250):
        """计算图的 NetLSD 描述符"""
        # 获取最大连通分量
        if not nx.is_connected(G):
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        
        n = G.number_of_nodes()
        if n < 2:
            return np.zeros(timescales)
        
        # 计算归一化 Laplacian 特征值
        L = nx.normalized_laplacian_matrix(G).toarray()
        eigenvalues = np.linalg.eigvalsh(L)
        
        # 计算热核迹 (heat kernel trace)
        # h(t) = sum_i exp(-t * lambda_i)
        t_values = np.logspace(-2, 2, timescales)
        hkt = np.zeros(timescales)
        
        for i, t in enumerate(t_values):
            hkt[i] = np.sum(np.exp(-t * eigenvalues))
        
        # 归一化
        hkt = hkt / n
        
        return hkt
    
    netlsd_features = []
    valid_case_ids = []
    
    for cid in case_ids:
        filepath = os.path.join(data_dir, f"case{cid}_RC.pt")
        if not os.path.exists(filepath):
            continue
        
        hg = torch.load(filepath, weights_only=False)
        if isinstance(hg, list):
            hg = hg[0]
        
        # 构建 NetworkX 图
        G = nx.Graph()
        total_nodes = sum(hg[ntype].num_nodes for ntype in hg.node_types if ntype in ['dev', 'pin', 'net'])
        G.add_nodes_from(range(total_nodes))
        
        for etype in hg.edge_types:
            ei = hg[etype].edge_index
            edges = ei.t().numpy().tolist()
            G.add_edges_from(edges)
        
        try:
            descriptor = compute_netlsd(G, timescales)
            netlsd_features.append(descriptor)
            valid_case_ids.append(cid)
            print(f"  Case {cid}: {G.number_of_nodes()} nodes")
        except Exception as e:
            print(f"  Case {cid}: 计算失败 - {e}")
    
    # 计算距离 (论文推荐使用 L2 距离)
    features = np.array(netlsd_features)
    distances = squareform(pdist(features, metric='euclidean'))
    
    # 转换为相似度
    similarity = 1 / (1 + distances)
    
    # 保存结果
    df_sim = pd.DataFrame(similarity, index=valid_case_ids, columns=valid_case_ids)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_sim.to_excel(save_path)
    
    print(f"\nNetLSD 相似度矩阵已保存到: {save_path}")
    
    # 绘制热力图
    _plot_similarity_heatmap(df_sim, valid_case_ids, save_path, title="NetLSD Similarity")
    
    _print_similarity_pairs(df_sim, valid_case_ids)
    
    return df_sim


def _plot_similarity_heatmap(df_sim, case_ids, save_path, title="Graph Similarity"):
    """
    绘制相似度矩阵热力图，使用绿色-紫色配色方案
    
    Args:
        df_sim: 相似度矩阵 DataFrame
        case_ids: case ID 列表
        save_path: 保存路径 (Excel路径，会自动改为.png)
        title: 图标题
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    
    similarity = df_sim.values
    
    # 使用单向配色方案：红-黄-绿反转 (高相似度=绿色，低相似度=红色)
    cmap = 'RdYlGn_r'
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(10, 7))
    
    im = ax.imshow(similarity, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    
    # 设置刻度
    ax.set_xticks(np.arange(len(case_ids)))
    ax.set_yticks(np.arange(len(case_ids)))
    ax.set_xticklabels(case_ids, fontsize=8)
    ax.set_yticklabels(case_ids, fontsize=8)
    
    # 旋转x轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Similarity', rotation=-90, va="bottom", fontsize=10)
    
    ax.set_title(title, fontsize=12)
    
    plt.tight_layout()
    
    # 保存图片
    img_path = save_path.replace('.xlsx', '.png')
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  热力图已保存到: {img_path}")


def _print_similarity_pairs(df_sim, case_ids):
    """打印最相似和最不相似的图对"""
    import numpy as np
    
    similarity = df_sim.values
    n = len(case_ids)
    pairs = []
    
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((case_ids[i], case_ids[j], similarity[i, j]))
    
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    print("\n" + "=" * 60)
    print("最相似的图对:")
    print("=" * 60)
    for i, (c1, c2, sim) in enumerate(pairs[:10]):
        print(f"  {i+1}. Case {c1} <-> Case {c2}: similarity = {sim:.4f}")
    
    print("\n最不相似的图对:")
    for i, (c1, c2, sim) in enumerate(pairs[-5:]):
        print(f"  {i+1}. Case {c1} <-> Case {c2}: similarity = {sim:.4f}")


def compute_all_graph_similarities(data_dir: str, case_ids: List[str] = None,
                                    save_dir: str = "./stastic",
                                    skip_wl: bool = False):
    """
    使用多种方法计算图相似度，并生成综合报告。
    
    方法:
    1. WL Kernel (Shervashidze et al., JMLR 2011) - 可选跳过
    2. Spectral (Chung, 1997; Wilson & Zhu, BMVC 2008)
    3. NetLSD (Tsitsulin et al., KDD 2018)
    
    Args:
        data_dir: 数据目录路径
        case_ids: 要分析的case ID列表
        save_dir: 保存结果的目录
        skip_wl: 是否跳过WL Kernel计算 (对大图可能很慢，默认False)
    
    Returns:
        dict: 包含各方法的相似度矩阵
    """
    import pandas as pd
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 70)
    print("图相似度综合分析")
    print("=" * 70)
    
    results = {}
    
    # 1. WL Kernel (可选)
    if not skip_wl:
        print("\n" + "=" * 70)
        print("[1/3] Weisfeiler-Lehman Kernel (大图可能较慢)")
        print("=" * 70)
        try:
            results['wl'] = compute_graph_similarity_wl_kernel(
                data_dir, case_ids, 
                save_path=os.path.join(save_dir, "graph_similarity_wl.xlsx"),
                n_iter=3  # 降低迭代次数
            )
        except MemoryError:
            print("  [警告] WL Kernel 计算内存不足，已跳过")
            print("  [提示] 可设置 skip_wl=True 跳过此方法")
    else:
        print("\n[跳过] WL Kernel (skip_wl=True)")
    
    # 2. Spectral
    print("\n" + "=" * 70)
    print(f"[{2 if not skip_wl else 1}/3] Spectral Method")
    print("=" * 70)
    results['spectral'] = compute_graph_similarity_spectral(
        data_dir, case_ids,
        save_path=os.path.join(save_dir, "graph_similarity_spectral.xlsx")
    )
    
    # 3. NetLSD
    print("\n" + "=" * 70)
    print(f"[{3 if not skip_wl else 2}/3] NetLSD")
    print("=" * 70)
    results['netlsd'] = compute_graph_similarity_netlsd(
        data_dir, case_ids,
        save_path=os.path.join(save_dir, "graph_similarity_netlsd.xlsx")
    )
    
    # 生成综合报告
    print("\n" + "=" * 70)
    print("综合分析完成")
    print("=" * 70)
    print(f"\n结果文件:")
    print(f"  1. {save_dir}/graph_similarity_wl.xlsx      - WL Kernel")
    print(f"  2. {save_dir}/graph_similarity_spectral.xlsx - Spectral")
    print(f"  3. {save_dir}/graph_similarity_netlsd.xlsx   - NetLSD")
    
    print(f"\n论文引用:")
    print(f"  [1] Shervashidze et al., 'Weisfeiler-Lehman Graph Kernels', JMLR 2011")
    print(f"  [2] Chung, 'Spectral Graph Theory', 1997")
    print(f"  [3] Tsitsulin et al., 'NetLSD: Hearing the Shape of a Graph', KDD 2018")
    
    return results


def plot_graph_tsne(data_dir: str, case_ids: List[str] = None, 
                    save_path: str = "./stastic/graph_tsne.png",
                    perplexity: int = 5):
    """
    使用 t-SNE 将图的结构特征降维到2D并可视化。
    每个图（case）用不同颜色的点表示。
    
    Args:
        data_dir: 数据目录路径
        case_ids: 要分析的case ID列表
        save_path: 保存图片的路径
        perplexity: t-SNE 的 perplexity 参数 (默认5，因为样本数较少)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    import networkx as nx
    
    # 获取所有case文件
    if case_ids is None:
        files = [f for f in os.listdir(data_dir) if f.endswith('_RC.pt')]
        case_ids = sorted([f.replace('case', '').replace('_RC.pt', '') for f in files], key=lambda x: int(x))
    
    print(f"==================== t-SNE 图特征可视化 ====================")
    print(f"分析 {len(case_ids)} 个图...")
    
    features_list = []
    valid_case_ids = []
    
    for cid in case_ids:
        filepath = os.path.join(data_dir, f"case{cid}_RC.pt")
        if not os.path.exists(filepath):
            print(f"  [跳过] case{cid} 不存在")
            continue
        
        hg = torch.load(filepath, weights_only=False)
        if isinstance(hg, list):
            hg = hg[0]
        
        # 提取图结构特征
        try:
            # 节点统计
            num_dev = hg['dev'].num_nodes if 'dev' in hg.node_types else 0
            num_pin = hg['pin'].num_nodes if 'pin' in hg.node_types else 0
            num_net = hg['net'].num_nodes if 'net' in hg.node_types else 0
            total_nodes = num_dev + num_pin + num_net
            
            # 边统计
            total_edges = 0
            for etype in hg.edge_types:
                total_edges += hg[etype].edge_index.shape[1]
            
            # 构建 NetworkX 图计算高级特征
            G = nx.Graph()
            G.add_nodes_from(range(total_nodes))
            for etype in hg.edge_types:
                ei = hg[etype].edge_index
                edges = ei.t().numpy().tolist()
                G.add_edges_from(edges)
            
            # 度统计
            degrees = [d for n, d in G.degree()]
            avg_degree = np.mean(degrees) if degrees else 0
            std_degree = np.std(degrees) if degrees else 0
            max_degree = max(degrees) if degrees else 0
            
            # 图密度
            density = nx.density(G)
            
            # 聚类系数
            try:
                clustering = nx.average_clustering(G)
            except:
                clustering = 0
            
            # 传递性
            try:
                transitivity = nx.transitivity(G)
            except:
                transitivity = 0
            
            # 连通分量
            num_components = nx.number_connected_components(G)
            
            # 节点类型比例
            dev_ratio = num_dev / total_nodes if total_nodes > 0 else 0
            pin_ratio = num_pin / total_nodes if total_nodes > 0 else 0
            net_ratio = num_net / total_nodes if total_nodes > 0 else 0
            
            # 组合特征向量
            feature_vec = [
                np.log1p(total_nodes),  # 节点数 (log)
                np.log1p(total_edges),  # 边数 (log)
                dev_ratio,              # device比例
                pin_ratio,              # pin比例
                net_ratio,              # net比例
                avg_degree,             # 平均度
                std_degree,             # 度标准差
                max_degree,             # 最大度
                density,                # 图密度
                clustering,             # 聚类系数
                transitivity,           # 传递性
                num_components,         # 连通分量数
            ]
            
            features_list.append(feature_vec)
            valid_case_ids.append(cid)
            print(f"  Case {cid}: nodes={total_nodes}, edges={total_edges}")
            
        except Exception as e:
            print(f"  Case {cid}: 特征提取失败 - {e}")
    
    if len(features_list) < 2:
        print("  [错误] 有效图数量不足，无法进行 t-SNE")
        return
    
    # 标准化特征
    features = np.array(features_list)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # t-SNE 降维
    # perplexity 必须小于样本数
    actual_perplexity = min(perplexity, len(features_list) - 1)
    print(f"\n  运行 t-SNE (perplexity={actual_perplexity})...")
    
    tsne = TSNE(n_components=2, perplexity=actual_perplexity, random_state=42, 
                max_iter=1000, learning_rate='auto', init='pca')
    features_2d = tsne.fit_transform(features_scaled)
    
    # 绘制散点图
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    
    # 使用不同颜色
    colors = plt.cm.tab20(np.linspace(0, 1, len(valid_case_ids)))
    
    for i, (cid, color) in enumerate(zip(valid_case_ids, colors)):
        ax.scatter(features_2d[i, 0], features_2d[i, 1], 
                   c=[color], s=150, label=f'case{cid}', 
                   edgecolors='black', linewidths=0.5, alpha=0.8)
        # 添加标签
        ax.annotate(f'{cid}', (features_2d[i, 0], features_2d[i, 1]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('Graph Feature Distribution (t-SNE)', fontsize=14)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  [保存] t-SNE 可视化图: {save_path}")
    print(f"==================== t-SNE 可视化完成 ====================")
    
    return features_2d, valid_case_ids


def plot_combined_label_distribution(data_dir: str, case_ids: list, save_dir: str = None):
    """
    将所有图的标签分布画在一张图上。
    按节点任务和边任务分别保存，归一化前后各一张，共四张图：
    - imgs/node/all_node_labels_original.png - 节点标签原始分布
    - imgs/node/all_node_labels_normalized.png - 节点标签归一化后分布
    - imgs/edge/all_edge_labels_original.png - 边标签原始分布
    - imgs/edge/all_edge_labels_normalized.png - 边标签归一化后分布
    
    样式：灰色背景，橙黄色柱状图，y轴为density
    
    Args:
        data_dir: 数据目录路径，如 '../data'
        case_ids: case ID 列表，如 [1, 5, 7, 10]
        save_dir: 保存根目录，默认为 'imgs'
    """
    import matplotlib.pyplot as plt
    import numpy as np

    def _add_mean_median_lines(ax, values):
        """在直方图上标注均值和中位数两条竖线。"""
        if values is None or len(values) == 0:
            return

        mean_val = float(np.mean(values))
        median_val = float(np.median(values))

        ax.axvline(mean_val, color='#D62728', linestyle='--', linewidth=2,
                   label=f'mean={mean_val:.3g}')
        ax.axvline(median_val, color='#1F77B4', linestyle='-.', linewidth=2,
                   label=f'median={median_val:.3g}')
        ax.legend(fontsize=16, frameon=True, loc='upper right')
    
    # 标签范围常量
    MAX_EDGE_LABEL = 700.0  # 电阻最大值 (Ω)
    MAX_NODE_LABEL = 8e-13  # 电容最大值 (F)
    
    if save_dir is None:
        save_dir = 'imgs'
    
    # 创建节点和边任务的子目录
    node_save_dir = os.path.join(save_dir, 'node')
    edge_save_dir = os.path.join(save_dir, 'edge')
    os.makedirs(node_save_dir, exist_ok=True)
    os.makedirs(edge_save_dir, exist_ok=True)
    
    all_node_labels = []
    all_edge_labels = []
    
    print(f"==================== 收集所有图的标签 ====================")
    
    for case_id in case_ids:
        filepath = os.path.join(data_dir, f'case{case_id}_RC.pt')
        if not os.path.exists(filepath):
            print(f"  [跳过] {filepath} 不存在")
            continue
        
        print(f"  [加载] case{case_id}...")
        hg = torch.load(filepath)
        if isinstance(hg, list):
            hg = hg[0]
        
        # 收集节点标签 (net节点的电容)
        if 'net' in hg.node_types and hasattr(hg['net'], 'y'):
            net_y = hg['net'].y
            if hasattr(hg['net'], 'x'):
                net_x = hg['net'].x
                is_power_net = (net_x[:, 0] == 1.0) | (net_x[:, 1] == 1.0)
                net_y = net_y[~is_power_net]
            valid_mask = (net_y > 0) & (net_y <= MAX_NODE_LABEL)
            valid_y = net_y[valid_mask.squeeze()]
            all_node_labels.append(valid_y.cpu().numpy().flatten())
            print(f"    节点标签: {len(valid_y)} 个有效值")
        
        # 收集边标签 (pair_to边的电阻)
        pair_to_etype = ('pin', 'pair_to', 'pin')
        if pair_to_etype in hg.edge_types and hasattr(hg[pair_to_etype], 'y'):
            edge_y = hg[pair_to_etype].y
            valid_mask = (edge_y > 0) & (edge_y <= MAX_EDGE_LABEL)
            valid_y = edge_y[valid_mask.squeeze()]
            all_edge_labels.append(valid_y.cpu().numpy().flatten())
            print(f"    边标签: {len(valid_y)} 个有效值")
    
    # 合并所有标签
    if all_node_labels:
        combined_node_labels = np.concatenate(all_node_labels)
        print(f"\n  总节点标签数: {len(combined_node_labels)}")
    else:
        combined_node_labels = np.array([])
    
    if all_edge_labels:
        combined_edge_labels = np.concatenate(all_edge_labels)
        print(f"  总边标签数: {len(combined_edge_labels)}")
    else:
        combined_edge_labels = np.array([])
    
    # 设置样式：灰色背景，橙黄色柱状图
    bar_color = '#E8A838'  # 橙黄色
    
    # ==================== 节点标签分布图 ====================
    if len(combined_node_labels) > 0:
        # 1. 原始节点标签分布
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
        ax.set_facecolor('lightgray')
        ax.hist(combined_node_labels, bins=50, density=True, color=bar_color, edgecolor=bar_color, alpha=0.9)
        _add_mean_median_lines(ax, combined_node_labels)
        ax.set_xlabel('label', fontsize=21)
        ax.set_ylabel('density', fontsize=21)
        ax.tick_params(axis='both', which='major', labelsize=21)
        ax.grid(True, alpha=0.3, color='white')
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        plt.tight_layout()
        save_path = os.path.join(node_save_dir, 'all_node_labels_original.png')
        plt.savefig(save_path, dpi=300, facecolor='white')
        plt.close()
        print(f"\n  [保存] 节点标签原始分布图: {save_path}")
        
        # 2. 归一化后节点标签分布
        normalized_node = (np.log1p(combined_node_labels * 1e15) / 
                          np.log1p(MAX_NODE_LABEL * 1e15)).clip(0, 1)
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
        ax.set_facecolor('lightgray')
        ax.hist(normalized_node, bins=50, density=True, color=bar_color, edgecolor=bar_color, alpha=0.9)
        _add_mean_median_lines(ax, normalized_node)
        ax.set_xlabel('normalized label', fontsize=21)
        ax.set_ylabel('density', fontsize=21)
        ax.tick_params(axis='both', which='major', labelsize=21)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3, color='white')
        plt.tight_layout()
        save_path = os.path.join(node_save_dir, 'analog_all_node_labels.png')
        plt.savefig(save_path, dpi=300, facecolor='white')
        plt.close()
        print(f"  [保存] 节点标签归一化分布图: {save_path}")
    
    # ==================== 边标签分布图 ====================
    if len(combined_edge_labels) > 0:
        # 1. 原始边标签分布
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
        ax.set_facecolor('lightgray')
        ax.hist(combined_edge_labels, bins=50, density=True, color=bar_color, edgecolor=bar_color, alpha=0.9)
        _add_mean_median_lines(ax, combined_edge_labels)
        ax.set_xlabel('label', fontsize=21)
        ax.set_ylabel('density', fontsize=21)
        ax.tick_params(axis='both', which='major', labelsize=21)
        ax.grid(True, alpha=0.3, color='white')
        plt.tight_layout()
        save_path = os.path.join(edge_save_dir, 'all_edge_labels_original.png')
        plt.savefig(save_path, dpi=300, facecolor='white')
        plt.close()
        print(f"  [保存] 边标签原始分布图: {save_path}")
        
        # 2. 归一化后边标签分布
        normalized_edge = (np.log1p(combined_edge_labels) / 
                          np.log1p(MAX_EDGE_LABEL)).clip(0, 1)
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
        ax.set_facecolor('lightgray')
        ax.hist(normalized_edge, bins=50, density=True, color=bar_color, edgecolor=bar_color, alpha=0.9)
        _add_mean_median_lines(ax, normalized_edge)
        ax.set_xlabel('normalized label', fontsize=21)
        ax.set_ylabel('density', fontsize=21)
        ax.tick_params(axis='both', which='major', labelsize=21)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3, color='white')
        plt.tight_layout()
        save_path = os.path.join(edge_save_dir, 'analog_all_edge_labels.png')
        plt.savefig(save_path, dpi=300, facecolor='white')
        plt.close()
        print(f"  [保存] 边标签归一化分布图: {save_path}")
    
    print(f"\n==================== 组合标签分布图绘制完成 ====================")


def plot_tsne_by_analog_circuit_scale(data_dir: str = "../data/",
                                       sample_threshold: int = 5000,
                                       sample_rate: float = 0.1,
                                       save_path: str = 'imgs/analog_tsne_circuit_scale.png'):
    """
    按模拟电路规模（XS/S/M）绘制t-SNE可视化图
    
    规模划分：
    - XS: 19, 13, 4, 9, 10, 11
    - S: 16, 2, 1, 12, 6, 3, 17, 8, 14
    - M: 18, 15, 20, 5, 7
    
    Args:
        data_dir: 数据目录
        sample_threshold: 节点数阈值，超过此值则采样
        sample_rate: 采样率（当节点数超过阈值时使用）
        save_path: 保存路径
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    
    print(f"\n{'='*60}")
    print(f"绘制按模拟电路规模的t-SNE可视化...")
    print(f"  采样策略: 节点数 > {sample_threshold} 时采样 {sample_rate*100:.0f}%")
    print(f"{'='*60}")
    
    # 电路规模分组 (根据图片中的划分)
    scale_groups = {
        'XS': [75, 45, 10, 29, 39, 42],
        'S': [71, 5, 1, 44, 15, 7, 72, 23, 55],
        'M': [74, 58, 78, 11, 17]
    }
    
    # 颜色配置
    colors = {
        'XS': '#1f77b4',   # 蓝色
        'S': '#ff7f0e',    # 橙色
        'M': '#2ca02c'     # 绿色
    }
    
    features_list = []
    labels_list = []
    
    for scale_name, case_ids in scale_groups.items():
        print(f"\n  [{scale_name}] 电路: {case_ids}")
        
        for case_id in case_ids:
            filepath = os.path.join(data_dir, f'case{case_id}_RC.pt')
            if not os.path.exists(filepath):
                print(f"    [跳过] case{case_id} 不存在")
                continue
            
            print(f"    加载 case{case_id}...")
            hg = torch.load(filepath, weights_only=False)
            if isinstance(hg, list):
                hg = hg[0]
            
            # 提取节点特征 (优先使用 net 节点)
            node_features = None
            for ntype in ['net', 'dev', 'pin']:
                if ntype in hg.node_types:
                    node_data = hg[ntype]
                    if hasattr(node_data, 'x') and node_data.x is not None:
                        node_features = node_data.x.cpu().numpy()
                        num_nodes = len(node_features)
                        print(f"      使用 {ntype} 节点: {num_nodes} 个")
                        break
            
            if node_features is None:
                print(f"      [跳过] 无有效节点特征")
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
    for scale in ['XS', 'S', 'M']:
        count = np.sum(all_labels == scale)
        print(f"    {scale}: {count}")
    
    # t-SNE降维
    print(f"\n  执行t-SNE降维...")
    tsne = TSNE(n_components=2, perplexity=50, max_iter=2000, learning_rate=200,
                early_exaggeration=12, random_state=42, verbose=1)
    embeddings = tsne.fit_transform(all_features)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('#f8f8f8')  # 浅灰背景
    
    # 按照 M -> S -> XS 的顺序绘制 (大的先画，小的后画覆盖)
    for scale_name in ['M', 'S', 'XS']:
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
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n  t-SNE图已保存到: {save_path}")
    
    return embeddings


def plot_tsne_by_analog_circuit_scale_edge(data_dir: str = "../data/",
                                            sample_threshold: int = 10000,
                                            sample_rate: float = 0.05,
                                            save_path: str = 'imgs/analog_tsne_circuit_scale_edge.png',
                                            aggregation: str = 'concat'):
    """
    按模拟电路规模（XS/S/M）绘制边级别的t-SNE可视化图
    
    边特征通过聚合两端pin节点的特征得到：
    - concat: 拼接两端节点特征 [src_feat, dst_feat]
    - mean: 两端节点特征的均值
    - sum: 两端节点特征的和
    - hadamard: 两端节点特征的逐元素乘积
    
    Args:
        data_dir: 数据目录
        sample_threshold: 边数阈值，超过此值则采样
        sample_rate: 采样率
        save_path: 保存路径
        aggregation: 聚合方式 ('concat', 'mean', 'sum', 'hadamard')
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    
    print(f"\n{'='*60}")
    print(f"绘制按模拟电路规模的边级别t-SNE可视化...")
    print(f"  聚合方式: {aggregation}")
    print(f"  采样策略: 边数 > {sample_threshold} 时采样 {sample_rate*100:.0f}%")
    print(f"{'='*60}")
    
    # 电路规模分组
    scale_groups = {
        'XS': [75, 45, 10, 29, 39, 42],
        'S': [71, 5, 1, 44, 15, 7, 72, 23, 55],
        'M': [74, 58, 78, 11, 17]
    }
    
    # 颜色配置
    colors = {
        'XS': '#1f77b4',   # 蓝色
        'S': '#ff7f0e',    # 橙色
        'M': '#2ca02c'     # 绿色
    }
    
    features_list = []
    labels_list = []
    
    for scale_name, case_ids in scale_groups.items():
        print(f"\n  [{scale_name}] 电路: {case_ids}")
        
        for case_id in case_ids:
            filepath = os.path.join(data_dir, f'case{case_id}_RC.pt')
            if not os.path.exists(filepath):
                print(f"    [跳过] case{case_id} 不存在")
                continue
            
            print(f"    加载 case{case_id}...")
            hg = torch.load(filepath, weights_only=False)
            if isinstance(hg, list):
                hg = hg[0]
            
            # 获取 pin-pair_to-pin 边
            pair_to_etype = ('pin', 'pair_to', 'pin')
            if pair_to_etype not in hg.edge_types:
                print(f"      [跳过] 无 pair_to 边")
                continue
            
            edge_index = hg[pair_to_etype].edge_index
            num_edges = edge_index.shape[1]
            
            # 获取 pin 节点特征
            if 'pin' not in hg.node_types or not hasattr(hg['pin'], 'x'):
                print(f"      [跳过] 无 pin 节点特征")
                continue
            
            pin_features = hg['pin'].x.cpu().numpy()
            
            # 构建边特征：聚合两端节点特征
            src_idx = edge_index[0].cpu().numpy()
            dst_idx = edge_index[1].cpu().numpy()
            
            src_feat = pin_features[src_idx]
            dst_feat = pin_features[dst_idx]
            
            if aggregation == 'concat':
                edge_features = np.concatenate([src_feat, dst_feat], axis=1)
            elif aggregation == 'mean':
                edge_features = (src_feat + dst_feat) / 2
            elif aggregation == 'sum':
                edge_features = src_feat + dst_feat
            elif aggregation == 'hadamard':
                edge_features = src_feat * dst_feat
            else:
                edge_features = np.concatenate([src_feat, dst_feat], axis=1)
            
            print(f"      边数: {num_edges}, 边特征维度: {edge_features.shape[1]}")
            
            # 采样
            if num_edges > sample_threshold:
                num_samples = int(num_edges * sample_rate)
                sampled_idx = np.random.choice(num_edges, num_samples, replace=False)
                edge_features = edge_features[sampled_idx]
                print(f"      采样: {num_samples} 条边 ({sample_rate*100:.0f}%)")
            
            features_list.append(edge_features)
            labels_list.extend([scale_name] * len(edge_features))
    
    if not features_list:
        print("  错误: 没有找到有效的边特征")
        return None
    
    # 合并特征
    all_features = np.vstack(features_list)
    all_labels = np.array(labels_list)
    
    print(f"\n  总计样本数: {len(all_labels)}")
    for scale in ['XS', 'S', 'M']:
        count = np.sum(all_labels == scale)
        print(f"    {scale}: {count}")
    
    # t-SNE降维
    print(f"\n  执行t-SNE降维...")
    tsne = TSNE(n_components=2, perplexity=50, max_iter=2000, learning_rate=200,
                early_exaggeration=12, random_state=42, verbose=1)
    embeddings = tsne.fit_transform(all_features)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('#f8f8f8')
    
    for scale_name in ['M', 'S', 'XS']:
        mask = all_labels == scale_name
        if mask.sum() > 0:
            ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
                       c=colors[scale_name], label=scale_name,
                       s=40, alpha=0.7, edgecolors='white', linewidths=0.3)
    
    legend = ax.legend(loc='upper right', fontsize=28, markerscale=3,
                       frameon=True, fancybox=False, framealpha=0.95,
                       edgecolor='gray', borderpad=1)
    legend.get_frame().set_facecolor('white')
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    x_margin = (embeddings[:, 0].max() - embeddings[:, 0].min()) * 0.02
    y_margin = (embeddings[:, 1].max() - embeddings[:, 1].min()) * 0.02
    ax.set_xlim(embeddings[:, 0].min() - x_margin, embeddings[:, 0].max() + x_margin)
    ax.set_ylim(embeddings[:, 1].min() - y_margin, embeddings[:, 1].max() + y_margin)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n  边级别t-SNE图已保存到: {save_path}")
    
    return embeddings


if __name__ == "__main__":
    data_dir = "../data/"
    
    # 节点和边各个特征维度分析
    # analyze_node_features(data_dir)
    
    # 图结构信息统计
    # analyze_graph_structure(data_dir)
    
    # # 原来的计算图相似度的方法
    # compute_graph_similarity(data_dir)
    
    # 使用三种方法计算图相似度 (跳过WL Kernel以节省时间和内存)
    # compute_all_graph_similarities(data_dir, skip_wl=False)
    
    # # 绘制组合标签分布图
    case_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    plot_combined_label_distribution(data_dir, case_ids, save_dir='imgs')
    
    # # 绘制 t-SNE 可视化
    # plot_graph_tsne(data_dir, case_ids, save_path='./stastic/graph_tsne.png')
    
    # 绘制模拟电路按规模的 t-SNE 可视化 (XS/S/M) - 节点级别
    # plot_tsne_by_analog_circuit_scale(
    #     data_dir=data_dir,
    #     sample_threshold=10000,
    #     sample_rate=1.0,
    #     save_path='imgs/analog_tsne_circuit_scale.png'
    # )
    
    # 绘制模拟电路按规模的 t-SNE 可视化 (XS/S/M) - 边级别
    # plot_tsne_by_analog_circuit_scale_edge(
    #     data_dir=data_dir,
    #     sample_threshold=200000,   # 边数超过1万时采样
    #     sample_rate=0.3,         # 采样5%
    #     save_path='imgs/analog_tsne_circuit_scale_edge.png',
    #     aggregation='concat'
    # )




