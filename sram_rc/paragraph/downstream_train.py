import types  
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score,
    mean_absolute_error, mean_squared_error,
    mean_squared_error as root_mean_squared_error, r2_score,
)
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data

import time
import os
import sys
sys.path.append('..')
from tqdm import tqdm
from model import GraphHead
from Ensemble_model import EnsembleModel
from sampling import dataset_sampling
from loss import balanced_softmax_loss, get_sample_per_class


class Logger (object):
    """ 
    Logger for printing message during training and evaluation. 
    Adapted from GraphGPS 
    """
    
    def __init__(self, task='classification'):
        super().__init__()
        self.test_scores = False
        self._iter = 0
        self._true = []
        self._pred = []
        self._loss = 0.0
        self._size_current = 0
        self.task = task

    def _get_pred_int(self, pred_score):
        """Convert prediction scores to class labels.
        
        Args:
            pred_score: For binary classification, shape [N] or [N, 1] with probabilities
                       For multi-class classification, shape [N, num_classes] with probabilities
        
        Returns:
            Class labels as integers
        """
        if len(pred_score.shape) == 1 or pred_score.shape[1] == 1:
            # Binary classification: threshold at 0.5
            return (pred_score > 0.5).astype(int)
        else:
            # Multi-class classification: argmax over classes
            return pred_score.argmax(axis=1)

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
            
            # 调试信息
            # print(f"\n[DEBUG {split}]")
            # print(f"  true.shape: {true.shape}")
            # print(f"  pred_score.shape: {pred_score.shape}")
            # print(f"  unique classes in true: {np.unique(true)}")
            # print(f"  unique classes in pred: {np.unique(pred_int)}")
            # if len(np.unique(true)) <= 10:
            #     print(f"  class distribution in true: {np.bincount(true.astype(int))}")
            # if len(np.unique(pred_int)) <= 10:
            #     print(f"  class distribution in pred: {np.bincount(pred_int.astype(int))}")
            
            # 根据预测输出的维度判断是否多分类
            num_classes = pred_score.shape[1] if pred_score.ndim > 1 else 1
            is_multiclass = num_classes > 2
            # print(f"  num_classes: {num_classes}, is_multiclass: {is_multiclass}")

            try:
                if is_multiclass:
                    # 多分类使用 macro 平均
                    r_a_score = roc_auc_score(true, pred_score, multi_class='ovr', average='macro')
                else:
                    r_a_score = roc_auc_score(true, pred_score)
            except ValueError:
                r_a_score = 0.0

            # 多分类使用 macro 平均
            avg = 'macro' if is_multiclass else 'binary'
            res = {
                'loss': reformat(self._loss / self._size_current),
                'accuracy': reformat(accuracy_score(true, pred_int)),
                'precision': reformat(precision_score(true, pred_int, average=avg, zero_division=0)),
                'recall': reformat(recall_score(true, pred_int, average=avg, zero_division=0)),
                'f1': reformat(f1_score(true, pred_int, average=avg, zero_division=0)),
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

        print(split, res)
        return res


def compute_loss(pred, true, task, sample_per_class=None):
    """Compute loss and prediction score."""
    pred = pred.squeeze(-1) if pred.ndim > 1 and pred.size(-1) == 1 else pred
    true = true.squeeze(-1) if true.ndim > 1 else true

    if task == 'classification':
        if pred.ndim > 1 and pred.size(-1) > 1:
            # 多分类：确保标签是整数类型
            true = true.long()
            # 使用 BalancedSoftmax（仅多分类）
            if sample_per_class is not None:
                loss = balanced_softmax_loss(pred, true, sample_per_class)
            else:
                loss = F.cross_entropy(pred, true)
            pred_probs = torch.softmax(pred, dim=-1)
            return loss, pred_probs
        else:
            # 二分类：使用 BCEWithLogitsLoss，不使用 BalancedSoftmax
            bce_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
            true = true.float()
            loss = bce_loss(pred, true)
            return loss, torch.sigmoid(pred)
        
    elif task == 'regression':
        mse_loss = torch.nn.MSELoss(reduction='mean')
        return mse_loss(pred, true), pred
    
    else:
        raise ValueError(f"Task type {task} not supported!")


@torch.no_grad()
def eval_epoch(loader, model, device, 
               split='val', task='classification'):
    """evaluate the model on the validation or test set"""
    model.eval()
    logger = Logger(task=task)

    for i, batch in enumerate(tqdm(loader, desc="eval_"+split, leave=False)):
        pred, y = model(batch.to(device))
        # 边级任务: y 格式为 [regression_label, classification_label]
        if task == 'regression':
            y_true = y[:, 0] if y.dim() > 1 and y.size(1) == 2 else y
        else:
            y_true = y[:, 1].long() if y.dim() > 1 and y.size(1) == 2 else y.long()
        
        loss, pred_score = compute_loss(pred, y_true, task)
        _true = y_true.detach().to('cpu', non_blocking=True)
        _pred = pred_score.detach().to('cpu', non_blocking=True)
        logger.update_stats(true=_true,
                            pred=_pred,
                            batch_size=_true.squeeze().size(0),
                            loss=loss.detach().cpu().item(),
                            )
    logger.write_epoch(split)


def train(args, model, optimizers, 
          train_loader, val_loader, test_loaders, 
          device, sample_per_class=None):
    """Train the head model for link prediction task
    
    Args:
        sample_per_class: 每个类别的样本数量，用于 BalancedSoftmax（仅分类任务）
    """
    dataset_name = args.train_dataset.split('+')[0] if '+' in args.train_dataset else args.train_dataset
    
    if isinstance(optimizers, list):
        for opt in optimizers:
            opt.zero_grad()
    else:
        optimizers.zero_grad()
    
    log_dir = f"./logs/{dataset_name}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_log_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    
    with open(log_file, 'w') as f:
        f.write(f"Paragraph-Simple 训练日志\n")
        f.write(f"数据集: {dataset_name}\n\n")
        
        best_val_metrics = {"loss": float('inf')}
        best_epoch = 0
        
        for epoch in range(args.epochs):
            logger = Logger(task=args.task)
            model.train()
    
            for i, batch in enumerate(tqdm(train_loader, desc=f'Epoch:{epoch}')):
                batch = batch.to(device)
                
                if args.use_ensemble and isinstance(model, EnsembleModel):
                    loss, _pred, _true = model.train_step(batch, optimizers)
                else:
                    if isinstance(optimizers, list):
                        for opt in optimizers:
                            opt.zero_grad()
                    else:
                        optimizers.zero_grad()
                    
                    y_pred, y = model(batch)
                    # 边级任务: y 格式为 [regression_label, classification_label]
                    if args.task == 'regression':
                        y_true = y[:, 0] if y.dim() > 1 and y.size(1) == 2 else y
                    else:
                        y_true = y[:, 1].long() if y.dim() > 1 and y.size(1) == 2 else y.long()
                    
                    loss, pred = compute_loss(y_pred, y_true, args.task, sample_per_class=sample_per_class)
                    _true = y_true.detach().to('cpu', non_blocking=True)
                    _pred = pred.detach().to('cpu', non_blocking=True)
    
                    loss.backward()
                    
                    if isinstance(optimizers, list):
                        for opt in optimizers:
                            opt.step()
                    else:
                        optimizers.step()
                
                logger.update_stats(true=_true,
                                    pred=_pred, 
                                    batch_size=_true.size(0),
                                    loss=loss.detach().cpu().item(),
                                   )
                
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
                    # 边级任务: y 格式为 [regression_label, classification_label]
                    if args.task == 'regression':
                        y_true = y[:, 0] if y.dim() > 1 and y.size(1) == 2 else y
                    else:
                        y_true = y[:, 1].long() if y.dim() > 1 and y.size(1) == 2 else y.long()
                    
                    loss, pred = compute_loss(y_pred, y_true, args.task, sample_per_class=sample_per_class)
                    
                    val_logger.update_stats(
                        true=y_true.detach().to('cpu', non_blocking=True),
                        pred=pred.detach().to('cpu', non_blocking=True),
                        batch_size=y_true.size(0),
                        loss=loss.detach().cpu().item()
                    )
            
            val_results = val_logger.write_epoch("Val")
            f.write(f"Epoch {epoch+1}/{args.epochs} - 验证: ")
            for k, v in val_results.items():
                f.write(f"{k}={v:.6f} ")
            f.write("\n")
            
            is_best = False
            if val_results["loss"] < best_val_metrics["loss"]:
                is_best = True
                best_val_metrics["loss"] = val_results["loss"]
                best_epoch = epoch
            
            if is_best:
                f.write(f"【新的最佳模型！】Epoch {epoch+1}\n")
                for test_name, test_loader in test_loaders.items():
                    test_logger = Logger(task=args.task)
                    model.eval()
                    with torch.no_grad():
                        for batch in tqdm(test_loader, desc=f"Test_{test_name}", leave=False):
                            batch = batch.to(device)
                            y_pred, y = model(batch)
                            # 边级任务: y 格式为 [regression_label, classification_label]
                            if args.task == 'regression':
                                y_true = y[:, 0] if y.dim() > 1 and y.size(1) == 2 else y
                            else:
                                y_true = y[:, 1].long() if y.dim() > 1 and y.size(1) == 2 else y.long()
                            
                            loss, pred = compute_loss(y_pred, y_true, args.task, sample_per_class=sample_per_class)
                            
                            test_logger.update_stats(
                                true=y_true.detach().to('cpu', non_blocking=True),
                                pred=pred.detach().to('cpu', non_blocking=True),
                                batch_size=y_true.size(0),
                                loss=loss.detach().cpu().item()
                            )
                    
                    test_results = test_logger.write_epoch(f"Test_{test_name}")
                    f.write(f"测试集 {test_name}: ")
                    for k, v in test_results.items():
                        f.write(f"{k}={v:.6f} ")
                    f.write("\n")
                
                f.write("\n")
        
        f.write(f"\n训练完成!\n")
        f.write(f"最佳模型在第 {best_epoch+1} 轮, 验证Loss = {best_val_metrics['loss']:.6f}\n")
    
    print(f"训练完成! 最佳模型在第 {best_epoch+1} 轮")
    print(f"训练日志已保存到 {log_file}")
    
    return model


NET = 0
DEV = 1
PIN = 2


def downstream_node_pred(args, dataset, device):
    """downstream task training for node prediction"""
    from sampling import dataset_node_sampling
    
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    
    # 始终调用 norm_nfeat 来处理标签格式 [regression, classification]
    train_dataset.norm_nfeat([0, 1])
    test_dataset.norm_nfeat([0, 1])
    
    (
        train_loader, val_loader, test_loaders
    ) = dataset_node_sampling(args, dataset)
    
    # 根据任务类型设置输出维度
    dim_out = args.num_classes if args.task == 'classification' else 1
    
    model = GraphHead(
        args.hid_dim, dim_out, num_layers=args.num_gnn_layers, 
        num_head_layers=args.num_head_layers, 
        use_bn=args.use_bn, drop_out=args.dropout, activation=args.act_fn, 
        src_dst_agg=args.src_dst_agg, max_dist=args.max_dist,
        task=args.task, task_level='node'
    )
    model = model.to(device)
    optimizers = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    start = time.time()
    
    train_node(args, model, optimizers, 
               train_loader, val_loader, test_loaders, 
               device)
    
    model.to(device)
    elapsed = time.time() - start
    timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed))
    print(f"Done! Training took {timestr}")


def train_node(args, model, optimizers, 
               train_loader, val_loader, test_loaders, 
               device):
    """Train the model for node prediction task"""
    dataset_name = args.train_dataset.split('+')[0] if '+' in args.train_dataset else args.train_dataset
    
    if isinstance(optimizers, list):
        for opt in optimizers:
            opt.zero_grad()
    else:
        optimizers.zero_grad()
    
    log_dir = f"./logs/{dataset_name}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"node_training_log_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    
    with open(log_file, 'w') as f:
        f.write(f"Paragraph-Simple Node Task 训练日志\n")
        f.write(f"数据集: {dataset_name}\n\n")
        
        best_val_metrics = {"loss": float('inf')}
        best_epoch = 0
        
        for epoch in range(args.epochs):
            logger = Logger(task=args.task)
            model.train()
    
            for i, batch in enumerate(tqdm(train_loader, desc=f'Epoch:{epoch}')):
                batch = batch.to(device)
                
                if isinstance(optimizers, list):
                    for opt in optimizers:
                        opt.zero_grad()
                else:
                    optimizers.zero_grad()
                
                y_pred, y = model(batch)
                if args.task == 'regression':
                    y_true = y[:, 0] if y.dim() > 1 else y
                else:
                    y_true = y[:, 1].long() if y.dim() > 1 else y.long()
                
                loss, pred = compute_loss(y_pred, y_true, args.task)
                _true = y_true.detach().to('cpu', non_blocking=True)
                _pred = pred.detach().to('cpu', non_blocking=True)

                loss.backward()
                
                if isinstance(optimizers, list):
                    for opt in optimizers:
                        opt.step()
                else:
                    optimizers.step()
                
                logger.update_stats(true=_true,
                                    pred=_pred, 
                                    batch_size=_true.size(0),
                                    loss=loss.detach().cpu().item(),
                                   )
                
            train_results = logger.write_epoch("Train")
            f.write(f"Epoch {epoch+1}/{args.epochs} - 训练: ")
            for k, v in train_results.items():
                f.write(f"{k}={v:.6f} ")
            f.write("\n")
            
            val_results = eval_node_epoch(val_loader, model, device, args.task, split='val')
            f.write(f"Epoch {epoch+1}/{args.epochs} - 验证: ")
            for k, v in val_results.items():
                f.write(f"{k}={v:.6f} ")
            f.write("\n")
            
            is_best = False
            if val_results["loss"] < best_val_metrics["loss"]:
                is_best = True
                best_val_metrics["loss"] = val_results["loss"]
                best_epoch = epoch
            
            if is_best:
                f.write(f"【新的最佳模型！】Epoch {epoch+1}\n")
                for test_name, test_loader in test_loaders.items():
                    test_results = eval_node_epoch(test_loader, model, device, args.task, split=f'test_{test_name}')
                    f.write(f"测试集 {test_name}: ")
                    for k, v in test_results.items():
                        f.write(f"{k}={v:.6f} ")
                    f.write("\n")
                f.write("\n")
        
        f.write(f"\n训练完成!\n")
        f.write(f"最佳模型在第 {best_epoch+1} 轮, 验证Loss = {best_val_metrics['loss']:.6f}\n")
    
    print(f"训练完成! 最佳模型在第 {best_epoch+1} 轮")
    print(f"训练日志已保存到 {log_file}")
    return model


@torch.no_grad()
def eval_node_epoch(loader, model, device, task, split='val'):
    """评估节点级任务"""
    model.eval()
    logger = Logger(task=task)

    for batch in tqdm(loader, desc="eval_"+split, leave=False):
        batch = batch.to(device)
        pred, y = model(batch)
        
        if task == 'regression':
            y_true = y[:, 0] if y.dim() > 1 else y
        else:
            y_true = y[:, 1].long() if y.dim() > 1 else y.long()
        
        loss, pred_score = compute_loss(pred, y_true, task)
        _true = y_true.detach().to('cpu', non_blocking=True)
        _pred = pred_score.detach().to('cpu', non_blocking=True)
        logger.update_stats(true=_true,
                            pred=_pred,
                            batch_size=_true.size(0),
                            loss=loss.detach().cpu().item(),
                            )
    return logger.write_epoch(split)


def get_sample_per_class(labels, num_classes, device):
    """计算每个类别的样本数量，用于 BalancedSoftmax
    
    Args:
        labels: 类别标签 tensor
        num_classes: 类别数量
        device: 设备
    
    Returns:
        sample_per_class: 每个类别的样本数量 tensor
    """
    counts = torch.zeros(num_classes, device=device)
    labels = labels.long()
    for i in range(num_classes):
        counts[i] = (labels == i).sum().float()
    
    # 避免除零
    counts = counts.clamp(min=1.0)
    
    print(f"类别分布: {counts.cpu().numpy()}")
    
    return counts


def downstream_link_pred(args, dataset, device):
    """downstream task training for link prediction"""
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    
    # 始终调用 norm_nfeat 来处理标签格式 [regression, classification]
    train_dataset.norm_nfeat([0, 1])
    test_dataset.norm_nfeat([0, 1])
    
    (
        train_loader, val_loader, test_loaders
    ) = dataset_sampling(args, dataset)
    
    # 计算每个类别的样本数量（仅分类任务且多分类，用于 BalancedSoftmax）
    sample_per_class = None
    if args.task == 'classification' and args.num_classes > 2:
        # 从训练数据集获取所有边标签
        train_data = train_dataset._data
        if hasattr(train_data, 'edge_label') and train_data.edge_label is not None:
            # edge_label 格式为 [regression_label, classification_label]
            if train_data.edge_label.dim() > 1 and train_data.edge_label.size(1) == 2:
                class_labels = train_data.edge_label[:, 1].long()
            else:
                class_labels = train_data.edge_label.long()
            sample_per_class = get_sample_per_class(class_labels, args.num_classes, device)
    
    # 根据任务类型设置输出维度
    dim_out = args.num_classes if args.task == 'classification' else 1
    
    if args.use_ensemble:
        print(f"Creating ensemble model with {args.num_ensemble} sub-models")
        model = EnsembleModel(args, device, thresholds=args.ensemble_thresholds)
        
        optimizers = [
            torch.optim.Adam(model.models[i].parameters(), lr=args.lr) 
            for i in range(len(model.models))
        ]
    else:
        model = GraphHead(
            args.hid_dim, dim_out, num_layers=args.num_gnn_layers, 
            num_head_layers=args.num_head_layers, 
            use_bn=args.use_bn, drop_out=args.dropout, activation=args.act_fn, 
            src_dst_agg=args.src_dst_agg, max_dist=args.max_dist,
            task=args.task
        )
        model = model.to(device)
        optimizers = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    start = time.time()

    train(args, model, optimizers, 
          train_loader, val_loader, test_loaders, 
          device, sample_per_class=sample_per_class)

    model.to(device)
    elapsed = time.time() - start
    timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed))

    print(f"Done! Training took {timestr}")
