import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader


def merge_graph_attributes(merged_graph, graphs):
    merged_graph.x = torch.cat([g.x for g in graphs], dim=0)

    if hasattr(graphs[0], "node_type"):
        merged_graph.node_type = torch.cat([g.node_type for g in graphs], dim=0)

    if hasattr(graphs[0], "node_attr"):
        merged_graph.node_attr = torch.cat([g.node_attr for g in graphs], dim=0)

    if getattr(graphs[0], "y", None) is not None:
        merged_graph.y = torch.cat([g.y for g in graphs], dim=0)

    if getattr(graphs[0], "target_node_mask", None) is not None:
        merged_graph.target_node_mask = torch.cat([g.target_node_mask for g in graphs], dim=0)


def dataset_sampling(args, dataset):
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    train_graphs = []
    total_nodes = 0
    for i in range(len(train_dataset.names)):
        train_graph = train_dataset[i]
        train_graphs.append(train_graph)
        total_nodes += train_graph.num_nodes

    merged_graph = train_graphs[0].__class__()
    merge_graph_attributes(merged_graph, train_graphs)

    edge_index_list = []
    edge_label_index_list = []
    edge_type_list = []
    node_offset = 0
    for g in train_graphs:
        edge_index = g.edge_index.clone()
        edge_index += node_offset
        edge_index_list.append(edge_index)
        if hasattr(g, "edge_type"):
            edge_type_list.append(g.edge_type)

        edge_label_index = g.edge_label_index.clone()
        edge_label_index += node_offset
        edge_label_index_list.append(edge_label_index)
        node_offset += g.num_nodes

    merged_graph.edge_index = torch.cat(edge_index_list, dim=1)
    if edge_type_list:
        merged_graph.edge_type = torch.cat(edge_type_list, dim=0)
    merged_graph.edge_label_index = torch.cat(edge_label_index_list, dim=1)
    merged_graph.edge_label = torch.cat([g.edge_label for g in train_graphs], dim=0)
    merged_graph.num_nodes = total_nodes
    merged_graph.name = "merged_train_graph"

    train_ind, val_ind = train_test_split(
        np.arange(merged_graph.edge_label.size(0)),
        test_size=0.2,
        shuffle=True,
    )
    train_ind = torch.tensor(train_ind, dtype=torch.long)
    val_ind = torch.tensor(val_ind, dtype=torch.long)

    train_loader = LinkNeighborLoader(
        merged_graph,
        num_neighbors=args.num_hops * [-1],
        edge_label_index=merged_graph.edge_label_index[:, train_ind],
        edge_label=merged_graph.edge_label[train_ind],
        subgraph_type="bidirectional",
        disjoint=True,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = LinkNeighborLoader(
        merged_graph,
        num_neighbors=args.num_hops * [-1],
        edge_label_index=merged_graph.edge_label_index[:, val_ind],
        edge_label=merged_graph.edge_label[val_ind],
        subgraph_type="bidirectional",
        disjoint=True,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    test_loaders = {}
    for i in range(len(test_dataset.names)):
        test_graph = test_dataset[i]
        graph_name = test_graph.name
        test_loaders[graph_name] = LinkNeighborLoader(
            test_graph,
            num_neighbors=args.num_hops * [-1],
            edge_label_index=test_graph.edge_label_index,
            edge_label=test_graph.edge_label,
            subgraph_type="bidirectional",
            disjoint=True,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    return train_loader, val_loader, test_loaders


def node_dataset_sampling(args, dataset):
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    train_graphs = []
    total_nodes = 0
    for i in range(len(train_dataset.names)):
        train_graph = train_dataset[i]
        train_graphs.append(train_graph)
        total_nodes += train_graph.num_nodes

    merged_graph = train_graphs[0].__class__()
    merge_graph_attributes(merged_graph, train_graphs)

    edge_index_list = []
    edge_type_list = []
    node_offset = 0
    for g in train_graphs:
        edge_index = g.edge_index.clone()
        edge_index += node_offset
        edge_index_list.append(edge_index)
        if hasattr(g, "edge_type"):
            edge_type_list.append(g.edge_type)
        node_offset += g.num_nodes
    merged_graph.edge_index = torch.cat(edge_index_list, dim=1)
    if edge_type_list:
        merged_graph.edge_type = torch.cat(edge_type_list, dim=0)

    node_label_index_list = []
    node_label_list = []
    node_offset = 0
    for g in train_graphs:
        if hasattr(g, "node_label_index") and g.node_label_index.size(0) > 0:
            node_label_index = g.node_label_index.clone()
            node_label_index += node_offset
            node_label_index_list.append(node_label_index)
            node_label_list.append(g.node_label)
        node_offset += g.num_nodes

    merged_graph.node_label_index = torch.cat(node_label_index_list, dim=0)
    merged_graph.node_label = torch.cat(node_label_list, dim=0)
    merged_graph.num_nodes = total_nodes
    merged_graph.name = "merged_train_graph"

    print(
        f"merged train graph: nodes={merged_graph.num_nodes}, "
        f"edges={merged_graph.edge_index.size(1)}, "
        f"target_nodes={merged_graph.node_label_index.size(0)}"
    )

    num_target_nodes = merged_graph.node_label_index.size(0)
    train_ind, val_ind = train_test_split(
        np.arange(num_target_nodes),
        test_size=0.2,
        shuffle=True,
        random_state=42,
    )
    train_ind = torch.tensor(train_ind, dtype=torch.long)
    val_ind = torch.tensor(val_ind, dtype=torch.long)

    train_node_indices = merged_graph.node_label_index[train_ind]
    val_node_indices = merged_graph.node_label_index[val_ind]

    train_loader = NeighborLoader(
        merged_graph,
        num_neighbors=args.num_hops * [-1],
        input_nodes=train_node_indices,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = NeighborLoader(
        merged_graph,
        num_neighbors=args.num_hops * [-1],
        input_nodes=val_node_indices,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    test_loaders = {}
    for i in range(len(test_dataset.names)):
        test_graph = test_dataset[i]
        graph_name = test_graph.name

        if hasattr(test_graph, "node_label_index") and test_graph.node_label_index.size(0) > 0:
            test_loader = NeighborLoader(
                test_graph,
                num_neighbors=args.num_hops * [-1],
                input_nodes=test_graph.node_label_index,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
            test_loaders[graph_name] = test_loader
            print(f"test graph {graph_name}: target_nodes={test_graph.node_label_index.size(0)}")

    return train_loader, val_loader, test_loaders
