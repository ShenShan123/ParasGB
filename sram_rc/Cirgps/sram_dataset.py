import torch
from sklearn.model_selection import train_test_split
import numpy as np
import os
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected
import logging
import time
from pathlib import Path
from utils import (
    get_pos_neg_edges, add_tar_edges_to_g, get_balanced_edges, 
    collated_data_separate)
from torch.utils.data import Dataset
from torch_geometric.data import Batch

class SealSramDataset(InMemoryDataset):
    def __init__(
        self,
        name, #add
        root, #add
        add_target_edges=False,
        neg_edge_ratio=1.0,
        to_undirected=True,
        sample_rates=[1.0],
        task_type='classification',
        task_level='edge',
        net_only=False,
        class_boundaries=[0.2, 0.4, 0.6, 0.8],
        transform=None, 
        pre_transform=None,
        data_type='c'  # 新增: 'c' (电容) 或 'r' (电阻)
    ) -> None:
        """ The SRAM dataset. 
        It can be a combination of several large circuit graphs or millions of sampled subgraphs.
        Args:
            name (str): The name of the dataset.
            root (str): The root directory of the dataset.
            add_target_edges (bool): Whether to add target (labeled) edges to the graph.
            neg_edge_ratio (float): The ratio of negative edges to positive edges.
            to_undirected (bool): Whether to convert the graph to an undirected graph.
            sample_rates (list): The sampling rates of target edges for each dataset.
            task_type (str): The task type. It can be 'classification' or 'regression'.
            task_level (str): The task level. It can be 'node' or 'edge'.
            net_only (bool): Whether to only use net nodes for node task.
            class_boundaries (list): The boundaries for classification buckets.
            data_type (str): 'c' for capacitance, 'r' for resistance.
        """
        self.name = 'sram_r' if data_type == 'r' else 'sram'
        self.data_type = data_type  # 新增: 保存数据类型
        print(f"数据类型: {'电阻(resistance)' if data_type == 'r' else '电容(capacitance)'}")

        ## split the dataset according to '+' in the name
        if '+' in name:
            self.names = name.split('+')
        else:
            self.names = [name]
            
        print(f"SealSramDataset includes {self.names} circuits")

        self.sample_rates = sample_rates
        assert len(self.names) == len(self.sample_rates), \
            f"len of dataset:{len(self.names)}, len of sample_rate: {len(self.sample_rates)}"
        
        self.folder = root  # 直接使用root作为数据目录
        self.add_target_edges = add_target_edges
        self.neg_edge_ratio = neg_edge_ratio
        self.to_undirected = to_undirected
        ## data_lengths can be the number of total datasets or the number of subgraphs
        self.data_lengths = {}
        ## offset index for each graph
        self.data_offsets = {}

        self.task_type = task_type
        self.task_level = task_level
        self.net_only = net_only
        self.class_boundaries = torch.tensor(class_boundaries)
        self.max_net_node_feat = torch.ones((1, 17))
        self.max_dev_node_feat = torch.ones((1, 17))
        
        super().__init__(self.folder, transform, pre_transform)
        data_list = []

        for i, name in enumerate(self.names):
            ## If a processed data file exsit, we load it directly
            loaded_data, loaded_slices = torch.load(self.processed_paths[i])

            self.data_offsets[name] = len(data_list)
            if self.task_level == 'node':
                self.data_lengths[name] = loaded_data.y.size(0)
            else:
                self.data_lengths[name] = loaded_data.edge_label.size(0)
            if loaded_slices is not None:
                data_list += collated_data_separate(loaded_data, loaded_slices)
            else:
                data_list.append(loaded_data)
            
            print(f"load processed {name}, "+
                  f"len(data_list)={self.data_lengths[name]}, "+
                  f"data_offset={self.data_offsets[name]} ")
    
        ## combine multiple graphs into data list
        self.data, self.slices = self.collate(data_list)

    def norm_nfeat(self, ntypes):
        """
        This is only used in regression task. Only `DEVICE` and `NET` nodes have circuit statistics.
        Args:
            ntypes (list): The node types {0, 1} to be normalized
        """
        # 确保数据已加载
        if self._data is None:
            raise ValueError("Dataset not loaded properly, self._data is None")

        for ntype in ntypes:
            node_mask = self._data.node_type == ntype
            max_node_feat, _ = self._data.node_attr[node_mask].max(dim=0, keepdim=True)
            max_node_feat[max_node_feat == 0.0] = 1.0

            print(f"normalizing node_attr {ntype}: {max_node_feat} ...")
            self._data.node_attr[node_mask] /= max_node_feat
        
        if self.task_level == 'edge':
            if self.data_type == 'r':
                # 电阻任务: Log变换 + 全局百分位归一化
                edge_labels_np = self._data.edge_label.cpu().numpy().flatten()
                log_vals = np.log10(edge_labels_np)
                
                # 动态计算全局1%-99%百分位
                log_p1 = np.percentile(log_vals, 1)
                log_p99 = np.percentile(log_vals, 99)
                
                # 归一化到 [0, 1]
                if log_p99 > log_p1:
                    normalized = (log_vals - log_p1) / (log_p99 - log_p1)
                    normalized = np.clip(normalized, 0.0, 1.0)
                else:
                    normalized = np.zeros_like(log_vals)
                
                self._data.edge_label = torch.tensor(normalized, dtype=torch.float32)
                print(f"电阻归一化完成: 全局百分位 P1={10**log_p1:.2f}Ω, P99={10**log_p99:.2f}Ω")
            else:
                # 电容任务: log 变换归一化
                self._data.edge_label = torch.log10(self._data.edge_label * 1e21) / 6
                self._data.edge_label[self._data.edge_label < 0] = 0.0
                self._data.edge_label[self._data.edge_label > 1] = 1.0
            
            # 分类标签
            edge_label_c = torch.bucketize(self._data.edge_label, self.class_boundaries)
            
            # 存储格式: [regression_label, classification_label]
            self._data.edge_label = torch.stack(
                [self._data.edge_label, edge_label_c.float()], dim=1
            )
        elif self.task_level == 'node':
            # if self.data_type == 'r':
            #     # 电阻任务: 百分位过滤 + Min-Max 归一化
            #     node_labels_np = self._data.y.cpu().numpy().flatten()
            #     upper_bound = np.percentile(node_labels_np, 90)
            #     lower_bound = np.percentile(node_labels_np, 10)
                
            #     self._data.y = torch.clamp(self._data.y, lower_bound, upper_bound)
                
            #     min_val = self._data.y.min()
            #     max_val = self._data.y.max()
            #     if max_val > min_val:
            #         self._data.y = (self._data.y - min_val) / (max_val - min_val)
            # else:
                # 电容任务: log 变换归一化
            self._data.y = torch.log10(self._data.y * 1e20) / 6
            self._data.y[self._data.y < 0] = 0.0
            self._data.y[self._data.y > 1] = 1.0
            
            # 分类标签
            node_label_c = torch.bucketize(self._data.y.squeeze(), self.class_boundaries)
            
            # 存储格式: [regression_label, classification_label]
            self._data.y = torch.stack(
                [self._data.y.squeeze(), node_label_c.float()], dim=1
            )
        
        self._data_list = None

    def set_cl_embeds(self, embeds):
        """
        Set dataset attribute `x` as the embeddings learned by SGRL.
        Args:
            embeds (torch.Tensor): The embeddings [N, cl_hid_dim] learned by SGRL.
        """
        if self._data is None or self.slices is None:
            self.data, self.slices = self.collate(self._data_list)
            self._data_list = None

        self._data.x = embeds
        self._data_list = None
        print("Setting CL embeddings to x...")
        print('self._data', self._data)
        # print('self.slices', self.slices)

    def sram_graph_load(self, name, raw_path):
        """
        In the loaded circuit graph, ground capacitance values are stored in "tar_node_y' attribute of the node. 
        There are to edge sets. 
        The first is the connections existing in circuit topology, attribute names start with "edge". 
        For example, "g.edge_index" and "g.edge_type".
        The second is edges to be predicted, which are parasitic coupling edges. Their attribute names start with "tar_edge". 
        For example, "g.tar_edge_index" and "g.tar_edge_type".
        Coupling capacitance values are stored in the 'tar_edge_y' attribute of the edge.
        Args:
            name (str): The name of the dataset.
            raw_path (str): The path of the raw data file.
        Returns:
            g (torch_geometric.data.Data): The processed homo graph data.
        """
        logging.info(f"raw_path: {raw_path}")
        hg = torch.load(raw_path)
        if isinstance(hg, list):
            hg = hg[0]
        # print("hg", hg)
        power_net_ids = torch.tensor([0, 1])
        
        if name == "sandwich":
            # VDD VSS TTVDD
            power_net_ids = torch.tensor([0, 1, 1422])
        elif name == "ultra8t":
            # VDD VSS SRMVDD
            power_net_ids = torch.tensor([0, 1, 377])
        elif name == "sram_sp_8192w":
            # VSSE VDDCE VDDPE
            power_net_ids = torch.tensor([0, 1, 2])
        elif name == "ssram":
            # VDD VSS VVDD
            power_net_ids = torch.tensor([0, 1, 352])
        elif name == "digtime":
            power_net_ids = torch.tensor([0, 1])
        elif name == "timing_ctrl":
            power_net_ids = torch.tensor([0, 1])
        elif name == "array_128_32_8t":
            power_net_ids = torch.tensor([0, 1])
        
        """ graph transform """ 
        ### remove the power pins
        subset_dict = {}
        for ntype in hg.node_types:
            subset_dict[ntype] = torch.ones(hg[ntype].num_nodes, dtype=torch.bool)
            if ntype == 'net':
                subset_dict[ntype][power_net_ids] = False

        hg = hg.subgraph(subset_dict)
        
        # 根据 data_type 选择边类型
        if self.data_type == 'r':
            # 电阻任务: 使用 r_p2p 边
            hg = hg.edge_type_subgraph([
                ('device', 'device-pin', 'pin'),
                ('pin', 'pin-net', 'net'),
                ('pin', 'r_p2p', 'pin'),  # 电阻边
            ])
        else:
            # 电容任务: 使用 cc_* 边
            hg = hg.edge_type_subgraph([
                ## circuit connections in schematics
                ('device', 'device-pin', 'pin'),
                ('pin', 'pin-net', 'net'),
                ## parasitic coupling edges: pin2net, pin2pin, net2net
                ('pin', 'cc_p2n', 'net'),
                ('pin', 'cc_p2p', 'pin'),
                ('net', 'cc_n2n', 'net'),
            ])

        print(hg)

        ### transform hetero g into homo g
        g = hg.to_homogeneous()
        g.name = name
        assert hasattr(g, 'node_type')
        assert hasattr(g, 'edge_type')
        edge_offset = 0
        tar_edge_y = []
        tar_node_y = []
        g._n2type = {}
        node_feat = []
        max_feat_dim = 17
        ## padding y for device nodes (only net and pin nodes has capacitance)
        hg['device'].y = torch.ones((hg['device'].x.shape[0], 1)) * 1e-30
        
        for n, ntype in enumerate(hg.node_types):
            g._n2type[ntype] = n
            feat = hg[ntype].x
            feat = torch.nn.functional.pad(feat, (0, max_feat_dim-feat.size(1)))
            node_feat.append(feat)
            tar_node_y.append(hg[ntype].y)
        
        ## There is 'node_type' attribute after transforming hg to g.
        ## The 'node_type' is used as default node feature, g.x.
        g.x = g.node_type.view(-1, 1)
        ## circuit statistic features
        g.node_attr = torch.cat(node_feat, dim=0)
        ## lumped ground capacitance on net/pin nodes
        g.tar_node_y = torch.cat(tar_node_y, dim=0)

        del g.y
        g._e2type = {}

        for e, (edge, store) in enumerate(hg.edge_items()):
            g._e2type[edge] = e
            if self.data_type == 'r':
                # 电阻任务: 匹配 'r' 且不含 'device'
                if 'r' in edge[1] and 'device' not in edge[1]:  # r_p2p
                    tar_edge_y.append(store['y']) 
                else:
                    edge_offset += store['edge_index'].shape[1]
            else:
                # 电容任务: 匹配 'cc'
                if 'cc' in edge[1]:
                    tar_edge_y.append(store['y'])
                else:
                    edge_offset += store['edge_index'].shape[1]
        
        g._num_ntypes = len(g._n2type)
        g._num_etypes = len(g._e2type)
        logging.info(f"g._n2type {g._n2type}")
        logging.info(f"g._e2type {g._e2type}")

        tar_edge_index = g.edge_index[:, edge_offset:]
        tar_edge_type = g.edge_type[edge_offset:]
        tar_edge_y = torch.cat(tar_edge_y)

        # testing
        # for i in range(tar_edge_type.min(), tar_edge_type.max()+1):
        #     mask = tar_edge_type == i
        #     print("tar_edge_type", tar_edge_type[mask][0], "tar_edge_y", tar_edge_y[mask][0])
        # assert 0

        ## 根据 data_type 使用不同的过滤方式
        if self.data_type == 'r':
            # 电阻任务: 过滤零值
            legel_edge_mask = tar_edge_y > 0
        else:
            # 电容任务: 固定范围过滤
            legel_edge_mask = (tar_edge_y < 1e-15) & (tar_edge_y > 1e-21)
        # tar_edge_src_y = g.tar_node_y[tar_edge_index[0, :]].squeeze()
        # tar_edge_dst_y = g.tar_node_y[tar_edge_index[1, :]].squeeze()
        # legel_node_mask = (tar_edge_src_y < 1e-13) & (tar_edge_src_y > 1e-23)
        # legel_node_mask &= (tar_edge_dst_y < 1e-13) & (tar_edge_dst_y > 1e-23)

        ## remove the target edges with extreme capacitance values
        g.tar_edge_y = tar_edge_y[legel_edge_mask]# & legel_node_mask]
        g.tar_edge_index = tar_edge_index[:, legel_edge_mask]# & legel_node_mask]
        g.tar_edge_type = tar_edge_type[legel_edge_mask]# & legel_node_mask]
        # logging.info(f"we filter out the edges with Cc > 1e-15 and Cc < 1e-21 " + 
        #              f"{legel_edge_mask.size(0)-legel_edge_mask.sum()}")
        # logging.info(f"we filter out the edges with src/dst Cg > 1e-13 and Cg < 1e-23 " +
        #              f"{legel_node_mask.size(0)-legel_node_mask.sum()}")

        ## Calculate target edge type distributions (Cc_p2n : Cc_p2p : Cc_n2n)
        _, g.tar_edge_dist = g.tar_edge_type.unique(return_counts=True)
        
        ## remove target edges from the original g
        g.edge_type = g.edge_type[0:edge_offset]
        g.edge_index = g.edge_index[:, 0:edge_offset]

        ## convert to undirected edges
        if self.to_undirected:
            g.edge_index, g.edge_type = to_undirected(
                g.edge_index, g.edge_type, g.num_nodes, reduce='mean'
            )
        
        return g

    def single_g_process(self, idx: int):
        logging.info(f"processing dataset {self.names[idx]} "+ 
                     f"with sample_rate {self.sample_rates[idx]}...")
        ## we can load multiple graphs
        graph = self.sram_graph_load(self.names[idx], self.raw_paths[idx])
        logging.info(f"loaded graph {graph}")
        
        if self.task_level == 'node':
            # 节点级任务处理
            aug_graph = graph
            
            # 设置节点标签
            if self.net_only:
                # 仅使用net节点
                net_mask = graph.node_type == graph._n2type.get('net', 0)
                aug_graph.y = torch.zeros((graph.num_nodes, 1))
                aug_graph.y[net_mask] = graph.tar_node_y[net_mask]
            else:
                aug_graph.y = graph.tar_node_y
            
            # 删除不需要的属性
            if hasattr(aug_graph, 'tar_node_y'):
                del aug_graph.tar_node_y
            if hasattr(aug_graph, 'tar_edge_index'):
                del aug_graph.tar_edge_index
            if hasattr(aug_graph, 'tar_edge_type'):
                del aug_graph.tar_edge_type
            if hasattr(aug_graph, 'tar_edge_y'):
                del aug_graph.tar_edge_y
            
            torch.save((aug_graph, None), self.processed_paths[idx])
            return aug_graph.y.size(0)
        
        else:
            # 边级任务处理 (原有逻辑)
            ## generate negative edges for the loaded graph
            neg_edge_index, neg_edge_type = get_pos_neg_edges(
                graph, neg_ratio=self.neg_edge_ratio)
            
            if self.add_target_edges:
                ## self.data is actually the augmented graph
                aug_graph = add_tar_edges_to_g(graph, neg_edge_index, neg_edge_type)
            else:
                aug_graph = graph
            
            ## sample a portion of pos/neg edges
            (
                pos_edge_index, pos_edge_type, pos_edge_y,
                neg_edge_index, neg_edge_type
            ) = get_balanced_edges(
                graph, neg_edge_index, neg_edge_type, 
                self.neg_edge_ratio, self.sample_rates[idx]
            )

            ## Now we deal with the edge-level task
            if self.task_type == 'classification':
                # 5分类任务：使用电容值，后续在norm_nfeat中分桶
                links = pos_edge_index  # [2, Np] 只使用正边
                labels = pos_edge_y  # 使用电容值作为标签
            elif self.task_type == 'regression':
                ## We only consider the positive edges in the regression task.
                links = pos_edge_index  # [2, Np]
                labels = pos_edge_y
            else:
                raise ValueError(f"No defination of task {self.task_type} in this version!")
            
            ## remove the redundant attributes in this version
            del aug_graph.tar_node_y
            del aug_graph.tar_edge_index
            del aug_graph.tar_edge_type
            del aug_graph.tar_edge_y

            ## To use LinkNeighborLoader, the target links rename to edge_label_index
            ## target edge labels rename to edge_label
            aug_graph.edge_label_index = links
            aug_graph.edge_label = labels

            torch.save((aug_graph, None), self.processed_paths[idx])
            return aug_graph.edge_label.size(0)

    def process(self):
        data_lens_for_split = []
        p = Path(self.processed_dir)
        ## we can have multiple graphs
        for i, name in enumerate(self.names):
            ## if there is a processed file, we skip the self.single_g_process()
            if os.path.exists(self.processed_paths[i]):
                logging.info(f"Found process file of {name} in {self.processed_paths[i]}, skipping process()")
                continue 
                        
            data_lens_for_split.append(
                self.single_g_process(i)
            )

    @property
    def raw_file_names(self):
        raw_file_names = []
        for name in self.names:
            raw_file_names.append(name+'.pt')
        
        return raw_file_names
    
    @property
    def raw_dir(self) -> str:
        # Raw files are directly in the folder (e.g., './sram_rc/sram/')
        return self.folder
    
    @property
    def processed_dir(self) -> str:
        # 根据 data_type 选择处理目录
        suffix = '_r' if self.data_type == 'r' else ''
        base_name = 'processed_cirgps' + suffix
        
        if self.task_level == 'node':
            return os.path.join(self.root, base_name, f'node_{self.task_type}')
        else:
            return os.path.join(self.root, base_name, self.task_type)

    @property
    def processed_file_names(self):
        processed_names = []
        for i, name in enumerate(self.names):
            fname = name
            if self.sample_rates[i] < 1.0:
                fname += f"_s{self.sample_rates[i]}"
            if self.task_level == 'edge' and self.add_target_edges:
                fname += "_aug"
            if self.task_level == 'edge' and self.neg_edge_ratio < 1.0:
                fname += f"_nr{self.neg_edge_ratio:.1f}"
            if self.task_level == 'node':
                fname += "_node"
                if self.net_only:
                    fname += "_netonly"
            processed_names.append(fname+"_processed.pt")
        return processed_names

class LinkPredictionDataset(Dataset):
    """This is self implemented dataset for link prediction task.
    No use in this version.
    """
    def __init__(self, node_embeddings, graph, test=False):
        assert node_embeddings.size(0) == graph.num_nodes
        # links size is [2, num_links]
        self.len = graph.edge_label_index.size(1)
        self.x_data = node_embeddings
        self.y_data = graph.edge_label.long()
        self.links = graph.edge_label_index
        if test:
            self.test_idx = np.arange(self.len)
        else:
            self.train_idx, self.val_idx = self.get_idx_split()

    def __getitem__(self, index):
        # links size is [2, num_links]
        src = self.links[0, index]
        dst = self.links[1, index]
        y = self.y_data[index]
        return self.x_data[src], self.x_data[dst], y

    def __len__(self):
        return self.len
    
    def get_idx_split(self):
        return train_test_split(
            np.arange(self.len), 
            test_size=0.2, 
            random_state=123, 
            shuffle=True,
            #stratify=stratify,
        )
    
def collate_fn(dataList):
    """ It is only used by self implemented 'LinkPredictionDataset'.
    """
    num_items = len(dataList)
    dim = dataList[0][0].size(-1)
    src_data = torch.zeros((num_items, dim), dtype=torch.float32)
    dst_data = torch.zeros((num_items, dim), dtype=torch.float32)
    y = torch.zeros((num_items), dtype=torch.long)
    for i in range(num_items):
        src_data[i], dst_data[i], y[i] = dataList[i]
    batch = Data()
    batch.x = torch.stack([src_data, dst_data], dim=0)
    batch.y = y
    return batch

def adaption_for_sgrl(dataset):
    """
    It is only used for contrastive learning (SGRL).
    """
    data_list = []

    for i, name in enumerate(dataset.names):
        single_graph = Data(
            x=dataset[i].node_type, edge_index=dataset[i].edge_index, 
            edge_attr=dataset[i].edge_type
        )
        data_list.append(single_graph)

    # Create a big graph concate all graphs from the `data_list`
    batch = Batch.from_data_list(data_list)

    ## Change the size of x
    batch.x = batch.x.view(-1, 1)
    ## Rename the `edge_attr` to `edge_type`
    batch.edge_type = batch.edge_attr

    ## remove the redundant attributes
    del batch.edge_attr

    print("attributes in big batch", batch)
    print("batch.ptr", batch.ptr)

    return batch

def performat_SramDataset(dataset_dir, name, 
                          add_target_edges, 
                          neg_edge_ratio, to_undirected, 
                          sample_rates, task_type,
                          task_level='edge', net_only=False,
                          class_boundaries=[0.2, 0.4, 0.6, 0.8],
                          data_type='c'  # 新增: 'c' (电容) 或 'r' (电阻)
                          ):
    start = time.perf_counter()
    
    # sample_rates 现在应该是一个列表
    if not isinstance(sample_rates, list):
        # 如果传入的是单个值,转换为列表
        num_datasets = len(name.split('+'))
        sample_rates = [sample_rates] * num_datasets

    dataset = SealSramDataset(
        name=name, root=dataset_dir,
        add_target_edges=add_target_edges,
        neg_edge_ratio=neg_edge_ratio,
        to_undirected=to_undirected,
        sample_rates=sample_rates,
        task_type=task_type,
        task_level=task_level,
        net_only=net_only,
        class_boundaries=class_boundaries,
        data_type=data_type,  # 新增: 传递数据类型
    )

    elapsed = time.perf_counter() - start
    timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
            + f'{elapsed:.2f}'[-3:]
    print(f"PID = {os.getpid()}")
    print(f"Building dataset {name} took {timestr}")
    print('Dataloader: Loading success.')

    return dataset