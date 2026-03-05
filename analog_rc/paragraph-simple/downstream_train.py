import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score,
    mean_absolute_error, mean_squared_error,
    mean_squared_error as root_mean_squared_error, r2_score,
)
from torch_geometric.data import Data

import time
import os
from tqdm import tqdm
from model import GraphHead, NodeHead
from sampling import dataset_sampling, node_dataset_sampling



class Logger (object):
    """ 
    Logger for printing message during training and evaluation. 
    Adapted from GraphGPS 
    """
    
    def __init__(self, task='classification'):
        super().__init__()
        # Whether to run comparison tests of alternative score implementations.
        self.test_scores = False
        self._iter = 0
        self._true = []
        self._pred = []
        self._loss = 0.0
        self._size_current = 0
        self.task = task

    def _get_pred_int(self, pred_score):
        # 检查是否为NumPy数组
        if isinstance(pred_score, np.ndarray):
            if len(pred_score.shape) == 1 or pred_score.shape[1] == 1:
                return (pred_score > 0.5).astype(np.int64)
            else:
                return np.argmax(pred_score, axis=1)
        else:
            # PyTorch张量
            if len(pred_score.shape) == 1 or pred_score.shape[1] == 1:
                return (pred_score > 0.5).long()
            else:
                return pred_score.max(dim=1)[1]

    def update_stats(self, true, pred, batch_size, loss):
        self._true.append(true)
        self._pred.append(pred)
        self._size_current += batch_size
        self._loss += loss * batch_size
        self._iter += 1

    def write_epoch(self, split=""):
        true, pred_score = torch.cat(self._true), torch.cat(self._pred)
        true = true.cpu().numpy()
        pred_score = pred_score.cpu().numpy()
        reformat = lambda x: round(float(x), 4)

        if self.task == 'classification':
            pred_int = self._get_pred_int(pred_score)
            # 如果pred_int已经是numpy数组，就不需要再调用.numpy()
            if not isinstance(pred_int, np.ndarray):
                pred_int = pred_int.numpy()

            # 检查是否为多分类
            num_classes = len(np.unique(true))
            is_multiclass = num_classes > 2

            if is_multiclass:
                # 多分类指标
                try:
                    # 对于多分类，使用 macro 平均
                    r_a_score = roc_auc_score(true, pred_score, multi_class='ovr', average='macro')
                except ValueError:
                    r_a_score = 0.0
                
                res = {
                    'loss': reformat(self._loss / self._size_current),
                    'accuracy': reformat(accuracy_score(true, pred_int)),
                    'precision': reformat(precision_score(true, pred_int, average='macro', zero_division=0)),
                    'recall': reformat(recall_score(true, pred_int, average='macro', zero_division=0)),
                    'f1': reformat(f1_score(true, pred_int, average='macro', zero_division=0)),
                    'auc': reformat(r_a_score),
                }
            else:
                # 二分类指标
                try:
                    r_a_score = roc_auc_score(true, pred_score)
                except ValueError:
                    r_a_score = 0.0

                res = {
                    'loss': reformat(self._loss / self._size_current),
                    'accuracy': reformat(accuracy_score(true, pred_int)),
                    'precision': reformat(precision_score(true, pred_int, zero_division=0)),
                    'recall': reformat(recall_score(true, pred_int, zero_division=0)),
                    'f1': reformat(f1_score(true, pred_int, zero_division=0)),
                    'auc': reformat(r_a_score),
                }
        else:
            res = {
                'loss': reformat(self._loss / self._size_current),
                'mae': reformat(mean_absolute_error(true, pred_score)),
                'mse': reformat(mean_squared_error(true, pred_score)),
                'rmse': reformat(root_mean_squared_error(true, pred_score)),
                'r2': reformat(r2_score(true, pred_score)),
            }

        # Just print the results to screen
        print(split, res)
        return res

def compute_loss(pred, true, task):
    """Compute loss and prediction score. 
    This version only supports binary classification.
    Args:
        pred (torch.tensor): Unnormalized prediction
        true (torch.tensor): Groud truth label
        task (str): The task type, 'classification' or 'regression'
    Returns: Loss, normalized prediction score
    """

    ## default manipulation for pred and true
    ## can be skipped if special loss computation is needed
    pred = pred.squeeze(-1) if pred.ndim > 1 else pred
    true = true.squeeze(-1) if true.ndim > 1 else true

    if task == 'classification':
        bce_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')

        ## multiclass
        if pred.ndim > 1 and true.ndim == 1:
            pred = F.log_softmax(pred, dim=-1)
            return F.nll_loss(pred, true), pred
        ## binary or multilabel
        else:
            true = true.float()
            return bce_loss(pred, true), torch.sigmoid(pred)
        
    elif task == 'regression':
        mse_loss = torch.nn.MSELoss(reduction='mean')
        return mse_loss(pred, true), pred
    
    else:
        raise ValueError(f"Task type {task} not supported!")

@torch.no_grad()
def eval_epoch(loader, model, device, 
               split='val', task='classification'):
    """ 
    evaluate the model on the validation or test set
    Args:
        loader (torch.utils.data.DataLoader): The data loader
        model (torch.nn.Module): The model
        device (torch.device): The device to run the model on
        split (str): The split name, 'val' or 'test'
        task (str): The edge-level task type, 'classification' or 'regression'
    """
    model.eval()
    time_start = time.time()
    logger = Logger(task=task)

    for i, batch in enumerate(tqdm(loader, desc="eval_"+split, leave=False)):
        ## copy dspd tensor to the batch
        pred, true = model(batch.to(device))
        loss, pred_score = compute_loss(pred, true, task)
        _true = true.detach().to('cpu', non_blocking=True)
        _pred = pred_score.detach().to('cpu', non_blocking=True)
        logger.update_stats(true=_true,
                            pred=_pred,
                            batch_size=_true.squeeze().size(0),
                            loss=loss.detach().cpu().item(),
                            )
    logger.write_epoch(split)

def train(args, model, optimizer, 
          train_loader, val_loader, test_loaders, 
          device):
    """
    Train the head model for link prediction task
    """
    # 获取数据集名称
    dataset_name = args.train_dataset.split('+')[0] if '+' in args.train_dataset else args.train_dataset
    
    # Reset optimizer
    optimizer.zero_grad()
    
    # 创建日志目录
    log_dir = f"./logs/{dataset_name}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_log_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    
    # 创建模型保存目录
    model_dir = f"./models/{dataset_name}"
    os.makedirs(model_dir, exist_ok=True)
    
    with open(log_file, 'w') as f:
        f.write(f"Paragraph-Simple 训练日志\n")
        f.write(f"数据集: {dataset_name}\n\n")
        
        best_val_loss = float('inf')
        best_epoch = 0
        
        for epoch in range(args.epochs):
            logger = Logger(task=args.task)
            model.train()
    
            for i, batch in enumerate(tqdm(train_loader, desc=f'Epoch:{epoch}')):
                batch = batch.to(device)
                optimizer.zero_grad()
                
                # 前向传播
                y_pred, y = model(batch)
                loss, pred = compute_loss(y_pred, y, args.task)
                _true = y.detach().to('cpu', non_blocking=True)
                _pred = pred.detach().to('cpu', non_blocking=True)

                loss.backward()
                optimizer.step()
                
                logger.update_stats(true=_true,
                                    pred=_pred, 
                                    batch_size=_true.size(0),
                                    loss=loss.detach().cpu().item())
                
            # 获取训练结果
            train_results = logger.write_epoch("Train")
            f.write(f"Epoch {epoch+1}/{args.epochs} - 训练: ")
            for k, v in train_results.items():
                f.write(f"{k}={v:.6f} ")
            f.write("\n")
            
            # 验证
            val_logger = Logger(task=args.task)
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    y_pred, y = model(batch)
                    loss, pred = compute_loss(y_pred, y, args.task)
                    
                    val_logger.update_stats(
                        true=y.detach().to('cpu', non_blocking=True),
                        pred=pred.detach().to('cpu', non_blocking=True),
                        batch_size=y.size(0),
                        loss=loss.detach().cpu().item()
                    )
            
            val_results = val_logger.write_epoch("Val")
            f.write(f"Epoch {epoch+1}/{args.epochs} - 验证: ")
            for k, v in val_results.items():
                f.write(f"{k}={v:.6f} ")
            f.write("\n")
            
            # 判断当前模型是否是最佳模型（使用loss判断，loss越小越好）
            # 第0个epoch强制保存，之后只有loss更小时才保存
            is_best = (epoch == 0) or (val_results["loss"] < best_val_loss)
            if is_best:
                best_val_loss = val_results["loss"]
                best_epoch = epoch
            
            # 保存最佳模型
            if is_best:
                best_model_path = os.path.join(model_dir, "best_model.pth")
                torch.save(model.state_dict(), best_model_path)
                
                f.write(f"【新的最佳模型！】Epoch {epoch+1}\n")
                for test_name, test_loader in test_loaders.items():
                    test_logger = Logger(task=args.task)
                    model.eval()
                    with torch.no_grad():
                        for batch in test_loader:
                            batch = batch.to(device)
                            y_pred, y = model(batch)
                            loss, pred = compute_loss(y_pred, y, args.task)
                            
                            test_logger.update_stats(
                                true=y.detach().to('cpu', non_blocking=True),
                                pred=pred.detach().to('cpu', non_blocking=True),
                                batch_size=y.size(0),
                                loss=loss.detach().cpu().item()
                            )
                    
                    test_results = test_logger.write_epoch(f"Test_{test_name}")
                    f.write(f"测试集 {test_name}: ")
                    for k, v in test_results.items():
                        f.write(f"{k}={v:.6f} ")
                    f.write("\n")
                
                f.write("\n")
        
        # 训练结束，记录最终结果
        f.write(f"\n训练完成!\n")
        f.write(f"最佳模型在第 {best_epoch+1} 轮, 验证Loss = {best_val_loss:.6f}\n")
    
    print(f"训练完成! 最佳模型在第 {best_epoch+1} 轮")
    print(f"训练日志已保存到 {log_file}")
    
    # 加载最佳模型
    best_model_path = os.path.join(model_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"已加载最佳模型: {best_model_path}")
    
    return model

# 新数据集节点类型: dev=0, pin=1, net=2
DEV = 0
PIN = 1
NET = 2

def downstream_link_pred(args, dataset, device):
    """ downstream task training for link prediction (边任务)
    """
    # 对训练和测试数据集分别进行处理
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    
    if args.task == 'regression':
        ## 规范化回归任务的特征
        # 新数据集: dev=0, pin=1, net=2
        train_dataset.norm_nfeat([DEV, NET])  # 0=DEV, 2=NET
        test_dataset.norm_nfeat([DEV, NET])
    
    ## 子图采样
    (
        train_loader, val_loader, test_loaders
    ) = dataset_sampling(args, dataset)
    
    # 创建单模型
    model = GraphHead(
        args.hid_dim, 1, num_layers=args.num_gnn_layers, 
        num_head_layers=args.num_head_layers, 
        use_bn=args.use_bn, drop_out=args.dropout, activation=args.act_fn, 
        src_dst_agg=args.src_dst_agg, max_dist=args.max_dist,
        task=args.task
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    start = time.time()

    ## Start training
    train(args, model, optimizer, 
          train_loader, val_loader, test_loaders, 
          device)

    elapsed = time.time() - start
    timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed))

    print(f"Done! Training took {timestr}")


def downstream_node_pred(args, dataset, device):
    """ downstream task training for node prediction (节点任务)
    """
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    
    # 归一化节点特征
    print("正在归一化训练集...")
    train_dataset.norm_nfeat([DEV, PIN, NET])
    print("正在归一化测试集...")
    test_dataset.norm_nfeat([DEV, PIN, NET])
    
    ## 节点采样
    (
        train_loader, val_loader, test_loaders
    ) = node_dataset_sampling(args, dataset)
    
    # 确定输出维度
    if args.task == 'classification':
        dim_out = args.num_classes
    else:
        dim_out = 1
    
    # 创建节点任务模型
    model = NodeHead(
        args.hid_dim, dim_out=dim_out, 
        num_layers=args.num_gnn_layers, 
        num_head_layers=args.num_head_layers, 
        use_bn=args.use_bn, drop_out=args.dropout, 
        activation=args.act_fn,
        task=args.task  # 传递 'regression' 或 'classification'
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    start = time.time()

    ## Start training
    train_node(args, model, optimizer, 
               train_loader, val_loader, test_loaders, 
               device)

    elapsed = time.time() - start
    timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed))

    print(f"Done! Node task training took {timestr}")


def train_node(args, model, optimizer, 
               train_loader, val_loader, test_loaders, 
               device):
    """
    Train the model for node prediction task (节点任务训练)
    """
    dataset_name = args.train_dataset.split('+')[0] if '+' in args.train_dataset else args.train_dataset
    
    optimizer.zero_grad()
    
    # 创建日志目录
    log_dir = f"./logs/{dataset_name}_node"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_log_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    
    # 创建模型保存目录
    model_dir = f"./models/{dataset_name}_node"
    os.makedirs(model_dir, exist_ok=True)
    
    # 获取标签
    train_labels = train_loader.node_labels.to(device)
    val_labels = val_loader.node_labels.to(device)
    
    # 确定基础任务类型 (classification 或 regression)
    base_task = args.task  # 'classification' 或 'regression'
    
    with open(log_file, 'w') as f:
        f.write(f"Paragraph-Simple 节点任务训练日志\n")
        f.write(f"数据集: {dataset_name}\n")
        f.write(f"任务类型: node_{base_task}\n\n")
        
        best_val_loss = float('inf')
        best_epoch = 0
        
        for epoch in range(args.epochs):
            logger = Logger(task=base_task)
            model.train()
            
            # 计算当前 batch 的标签索引
            label_idx = 0
            
            for i, batch in enumerate(tqdm(train_loader, desc=f'Epoch:{epoch}')):
                batch = batch.to(device)
                optimizer.zero_grad()
                
                # 获取当前 batch 的标签
                batch_size = batch.batch_size if hasattr(batch, 'batch_size') else batch.x.size(0)
                batch_labels = train_labels[label_idx:label_idx + batch_size]
                label_idx += batch_size
                
                # 前向传播
                y_pred, y = model(batch, batch_labels)
                loss, pred = compute_loss(y_pred, y, base_task)
                _true = y.detach().to('cpu', non_blocking=True)
                _pred = pred.detach().to('cpu', non_blocking=True)

                loss.backward()
                optimizer.step()
                
                logger.update_stats(true=_true,
                                    pred=_pred, 
                                    batch_size=_true.size(0),
                                    loss=loss.detach().cpu().item())
            
            # 获取训练结果
            train_results = logger.write_epoch("Train")
            f.write(f"Epoch {epoch+1}/{args.epochs} - 训练: ")
            for k, v in train_results.items():
                f.write(f"{k}={v:.6f} ")
            f.write("\n")
            
            # 验证
            val_results = eval_node_epoch(val_loader, val_labels, model, device, 'Val', base_task)
            f.write(f"Epoch {epoch+1}/{args.epochs} - 验证: ")
            for k, v in val_results.items():
                f.write(f"{k}={v:.6f} ")
            f.write("\n")
            
            # 判断当前模型是否是最佳模型（使用loss判断，loss越小越好）
            # 第0个epoch强制保存，之后只有loss更小时才保存
            is_best = (epoch == 0) or (val_results["loss"] < best_val_loss)
            if is_best:
                best_val_loss = val_results["loss"]
                best_epoch = epoch
            
            if is_best:
                # 保存最佳模型
                best_model_path = os.path.join(model_dir, "best_model.pth")
                torch.save(model.state_dict(), best_model_path)
                
                f.write(f"【新的最佳模型！】Epoch {epoch+1}\n")
                
                # 测试
                for test_name, test_loader in test_loaders.items():
                    test_labels = test_loader.node_labels.to(device)
                    test_results = eval_node_epoch(test_loader, test_labels, model, device, f'Test_{test_name}', base_task)
                    f.write(f"测试集 {test_name}: ")
                    for k, v in test_results.items():
                        f.write(f"{k}={v:.6f} ")
                    f.write("\n")
                
                f.write("\n")
        
        f.write(f"\n训练完成!\n")
        f.write(f"最佳模型在第 {best_epoch+1} 轮, 验证Loss = {best_val_loss:.6f}\n")
    
    print(f"训练完成! 最佳模型在第 {best_epoch+1} 轮")
    print(f"训练日志已保存到 {log_file}")
    
    return model


@torch.no_grad()
def eval_node_epoch(loader, labels, model, device, split='val', task='regression'):
    """
    评估节点任务模型
    Args:
        task: 'classification' 或 'regression'
    """
    model.eval()
    logger = Logger(task=task)
    
    label_idx = 0
    for batch in tqdm(loader, desc=f"eval_{split}", leave=False):
        batch = batch.to(device)
        
        batch_size = batch.batch_size if hasattr(batch, 'batch_size') else batch.x.size(0)
        batch_labels = labels[label_idx:label_idx + batch_size]
        label_idx += batch_size
        
        pred, true = model(batch, batch_labels)
        loss, pred_score = compute_loss(pred, true, task)
        
        _true = true.detach().to('cpu', non_blocking=True)
        _pred = pred_score.detach().to('cpu', non_blocking=True)
        
        logger.update_stats(true=_true,
                            pred=_pred,
                            batch_size=_true.size(0),
                            loss=loss.detach().cpu().item())
    
    return logger.write_epoch(split)