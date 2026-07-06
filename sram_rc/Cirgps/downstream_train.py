import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score,
    mean_absolute_error, mean_squared_error,
    root_mean_squared_error, r2_score,
)

import time
import os
import sys
sys.path.append('..')
from tqdm import tqdm
from model import GraphHead
from sampling_and_pe import dataset_sampling_and_pe_calculation
from loss import balanced_softmax_loss, get_sample_per_class

NET = 0
DEV = 1
PIN = 2


def downstream_node_train(args, dataset, device):
    """ downstream task training for node prediction
    Args:
        args (argparse.Namespace): The arguments
        dataset (dict): Dictionary containing 'train' and 'test' datasets
        device (torch.device): The device to train the model on
    """
    from sampling_and_pe import dataset_node_sampling_and_pe_calculation
    
    # 归一化特征和标签
    dataset['train'].norm_nfeat([NET, DEV])
    dataset['test'].norm_nfeat([NET, DEV])
    
    model = GraphHead(args)
    
    # 节点级采样和PE计算
    (
        train_loader, val_loader, test_loaders,
        train_spd_list, valid_spd_list, test_spd_dict,
    ) = dataset_node_sampling_and_pe_calculation(args, dataset['train'], dataset['test'])

    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.task == 'classification':
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        print(f"Task is {args.task}, using CrossEntropyLoss for {args.num_classes}-class classification")
    else:
        criterion = torch.nn.MSELoss(reduction='mean')
    
    start = time.time()

    # 训练
    train_node(args, model, optimizer, criterion,
               train_loader, val_loader, test_loaders, 
               train_spd_list, valid_spd_list, 
               test_spd_dict, device)
    
    elapsed = time.time() - start
    timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed))
    print(f"Done! Node task training took {timestr}")


def train_node(args, model, optimizer, criterion,
               train_loader, val_loader, test_loaders, 
               train_batched_spd, val_batched_spd, 
               test_batched_spd_dict, device):
    """
    Train the model for node prediction task
    """
    import psutil
    import gc
    
    def print_memory_usage():
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_gb = memory_info.rss / (1024 ** 3)
        print(f"当前内存使用: {memory_gb:.2f} GB")
        return memory_gb
    
    dataset_name = args.train_dataset.split('+')[0] if '+' in args.train_dataset else args.train_dataset
    
    optimizer.zero_grad()
    
    best_results = {
        'best_val_mse': 1e9, 'best_val_loss': 1e9, 
        'best_epoch': 0, 'test_results': [], 'test_names': []
    }
    
    result_file = open('node_test_results.txt', 'w')
    result_file.write(f"节点任务训练\n")
    result_file.write(f"训练数据集: {args.train_dataset}\n")
    result_file.write(f"测试数据集: {args.test_dataset}\n\n")
    
    print("训练开始前内存使用情况:")
    initial_mem = print_memory_usage()
    
    for epoch in range(args.epochs):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")
        
        logger = Logger(task=args.task)
        model.train()

        for i, batch in enumerate(tqdm(train_loader, desc=f'Epoch:{epoch}')):
            optimizer.zero_grad()
            
            # 处理SPD大小不匹配的情况
            spd = train_batched_spd[i]
            if spd is not None:
                if spd.size(0) != batch.num_nodes:
                    # 如果大小不匹配，截断或填充
                    if spd.size(0) > batch.num_nodes:
                        spd = spd[:batch.num_nodes]
                    else:
                        pad = torch.full((batch.num_nodes - spd.size(0), spd.size(1)), args.max_dist, dtype=spd.dtype)
                        spd = torch.cat([spd, pad], dim=0)
                batch.dspd = spd
            
            y_pred, y = model(batch.to(device))
            
            # 节点任务使用 y[:, 0] 作为回归标签, y[:, 1] 作为分类标签
            if args.task == 'regression':
                y_true = y[:, 0] if y.dim() > 1 else y
            else:
                y_true = y[:, 1].long() if y.dim() > 1 else y.long()
            
            loss, pred = compute_loss(args, y_pred, y_true, criterion=criterion)
            _true = y_true.detach().to('cpu', non_blocking=True)
            _pred = pred.detach().to('cpu', non_blocking=True)
            
            loss.backward()
            optimizer.step()
            
            logger.update_stats(true=_true,
                                pred=_pred, 
                                batch_size=_true.size(0),
                                loss=loss.detach().cpu().item(),
                               )
            
        train_results = logger.write_epoch("Train")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # 验证
        val_results = eval_node_epoch(args, val_loader, val_batched_spd, model, device, 
                                      split='val', criterion=criterion)
        
        is_best = False
        if args.task == 'classification':
            # 使用loss作为最佳模型判断标准
            is_best = val_results["loss"] < best_results.get("best_val_loss", float('inf'))
            if is_best:
                best_results["best_val_loss"] = val_results["loss"]
                best_results["best_epoch"] = epoch
        else:
            is_best = val_results["mse"] < best_results["best_val_mse"]
            if is_best:
                best_results["best_val_mse"] = val_results["mse"]
                best_results["best_epoch"] = epoch
        
        if is_best:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"【新的最佳模型】Epoch {epoch}")
            result_file.write(f"Epoch {epoch}, 验证集结果: {val_results}\n")
            test_results = []
            test_names = []
            for test_idx, (test_loader, test_batched_spd) in enumerate(zip(test_loaders, test_batched_spd_dict.values())):
                test_result = eval_node_epoch(args, test_loader, test_batched_spd, model, device, 
                                              split=f'test_{test_idx}', criterion=criterion)
                test_results.append(test_result)
                test_names.append(f"test_{test_idx}")
                print(f"test_{test_idx}", test_result)
                result_file.write(f"测试集 {test_idx} 结果: {test_result}\n")
            result_file.write("\n")
            
            best_results["test_results"] = test_results
            best_results["test_names"] = test_names
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    result_file.close()
    return best_results


@torch.no_grad()
def eval_node_epoch(args, loader, batched_spd, model, device, 
                    split='val', criterion=None):
    """评估节点级任务"""
    model.eval()
    logger = Logger(task=args.task)

    for i, batch in enumerate(tqdm(loader, desc="eval_"+split, leave=False)):
        # 处理SPD大小不匹配的情况
        spd = batched_spd[i]
        if spd is not None:
            if spd.size(0) != batch.num_nodes:
                if spd.size(0) > batch.num_nodes:
                    spd = spd[:batch.num_nodes]
                else:
                    pad = torch.full((batch.num_nodes - spd.size(0), spd.size(1)), args.max_dist, dtype=spd.dtype)
                    spd = torch.cat([spd, pad], dim=0)
            batch.dspd = spd
        pred, y = model(batch.to(device))
        
        if args.task == 'regression':
            y_true = y[:, 0] if y.dim() > 1 else y
        else:
            y_true = y[:, 1].long() if y.dim() > 1 else y.long()
        
        loss, pred_score = compute_loss(args, pred, y_true, criterion=criterion)
        _true = y_true.detach().to('cpu', non_blocking=True)
        _pred = pred_score.detach().to('cpu', non_blocking=True)
        logger.update_stats(true=_true,
                            pred=_pred,
                            batch_size=_true.size(0),
                            loss=loss.detach().cpu().item(),
                            )
    return logger.write_epoch(split)


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
            
            # 统一使用 macro 平均，与其他项目保持一致
            try:
                if pred_score.ndim > 1 and pred_score.shape[1] > 1:
                    r_a_score = roc_auc_score(true, pred_score, multi_class='ovr', average='macro')
                else:
                    r_a_score = roc_auc_score(true, pred_score)
            except ValueError:
                r_a_score = 0.0

            res = {
                'loss': round(self._loss / self._size_current, 8),
                'accuracy': reformat(accuracy_score(true, pred_int)),
                'precision': reformat(precision_score(true, pred_int, average='macro', zero_division=0)),
                'recall': reformat(recall_score(true, pred_int, average='macro', zero_division=0)),
                'f1': reformat(f1_score(true, pred_int, average='macro', zero_division=0)),
                'auc': reformat(r_a_score),
            }
        else:
            res = {
                'loss': round(self._loss / self._size_current, 8),
                'mae': reformat(mean_absolute_error(true, pred_score)),
                'mse': reformat(mean_squared_error(true, pred_score)),
                'rmse': reformat(root_mean_squared_error(true, pred_score)),
                'r2': reformat(r2_score(true, pred_score)),
            }

        print(split, res)
        return res


def compute_loss(args, pred, true, criterion, sample_per_class=None):
    """Compute loss and prediction score."""
    assert criterion, "Loss function is not provided!"
    pred = pred.squeeze(-1) if pred.ndim > 1 and pred.size(-1) == 1 else pred
    true = true.squeeze(-1) if true.ndim > 1 else true

    if args.task == 'classification':
        # 分类任务：确保标签是整数类型
        true = true.long()
        # 使用 BalancedSoftmax 或普通 CrossEntropyLoss
        if sample_per_class is not None:
            loss = balanced_softmax_loss(pred, true, sample_per_class)
        else:
            loss = criterion(pred, true)
        pred_probs = torch.softmax(pred, dim=-1)
        return loss, pred_probs
        
    elif args.task == 'regression':
        return criterion(pred, true), pred
    
    else:
        raise ValueError(f"Task type {args.task} not supported!")


@torch.no_grad()
def eval_epoch(args, loader, batched_dspd, model, device, 
               split='val', criterion=None, sample_per_class=None):
    """evaluate the model on the validation or test set"""
    model.eval()
    logger = Logger(task=args.task)

    for i, batch in enumerate(tqdm(loader, desc="eval_"+split, leave=False)):
        # 处理DSPD大小不匹配的情况
        dspd = batched_dspd[i]
        if dspd is not None:
            if dspd.size(0) != batch.num_nodes:
                if dspd.size(0) > batch.num_nodes:
                    dspd = dspd[:batch.num_nodes]
                else:
                    pad = torch.full((batch.num_nodes - dspd.size(0), dspd.size(1)), args.max_dist, dtype=dspd.dtype)
                    dspd = torch.cat([dspd, pad], dim=0)
            batch.dspd = dspd
        pred, true = model(batch.to(device))
        # 边级任务: true 格式为 [regression_label, classification_label]
        if args.task == 'classification' and true.dim() > 1 and true.size(1) == 2:
            true = true[:, 1].long()
        loss, pred_score = compute_loss(args, pred, true, criterion=criterion, sample_per_class=sample_per_class)
        _true = true.detach().to('cpu', non_blocking=True)
        _pred = pred_score.detach().to('cpu', non_blocking=True)
        logger.update_stats(true=_true,
                            pred=_pred,
                            batch_size=_true.size(0),
                            loss=loss.detach().cpu().item(),
                            )
    return logger.write_epoch(split)


def train(args, model, optimizier, criterion,
          train_loader, val_loader, test_loaders, 
          train_batched_dspd, val_batched_dspd, 
          test_batched_dspd_dict, device, sample_per_class=None):
    """Train the head model for link prediction task"""
    import psutil
    import gc
    
    def print_memory_usage():
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_gb = memory_info.rss / (1024 ** 3)
        print(f"当前内存使用: {memory_gb:.2f} GB")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_mem_alloc = torch.cuda.memory_allocated(i) / (1024 ** 3)
                gpu_mem_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                print(f"GPU {i} 显存使用: {gpu_mem_alloc:.2f} GB (已分配), {gpu_mem_reserved:.2f} GB (已预留)")
        return memory_gb
        
    dataset_name = args.train_dataset.split('+')[0] if '+' in args.train_dataset else args.train_dataset
    
    optimizier.zero_grad()
    
    best_results = {
        'best_val_mse': 1e9, 'best_val_loss': 1e9, 
        'best_epoch': 0, 'test_results': [], 'test_names': []
    }
    
    result_file = open('test_results.txt', 'w')
    result_file.write(f"训练参数 u: {args.u}\n")
    result_file.write(f"训练数据集: {args.train_dataset}\n")
    result_file.write(f"测试数据集: {args.test_dataset}\n")
    result_file.write(f"小数据集采样率: {args.small_dataset_sample_rates}\n")
    result_file.write(f"大数据集采样率: {args.large_dataset_sample_rates}\n\n")
    
    print("训练开始前内存使用情况:")
    initial_mem = print_memory_usage()
    
    mem_warning_threshold = 0.9
    mem_critical_threshold = 2.0
    
    for epoch in range(args.epochs):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")
        print("训练前内存状态:")
        current_mem = print_memory_usage()
        
        if current_mem > initial_mem * mem_warning_threshold:
            print(f"内存使用增加到 {current_mem:.2f} GB，执行垃圾回收...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        logger = Logger(task=args.task)
        model.train()

        for i, batch in enumerate(tqdm(train_loader, desc=f'Epoch:{epoch}')):
            optimizier.zero_grad()
            
            # 处理DSPD大小不匹配的情况
            dspd = train_batched_dspd[i]
            if dspd is not None:
                if dspd.size(0) != batch.num_nodes:
                    if dspd.size(0) > batch.num_nodes:
                        dspd = dspd[:batch.num_nodes]
                    else:
                        pad = torch.full((batch.num_nodes - dspd.size(0), dspd.size(1)), args.max_dist, dtype=dspd.dtype)
                        dspd = torch.cat([dspd, pad], dim=0)
                batch.dspd = dspd
            
            y_pred, y = model(batch.to(device))
            
            # 边级任务: 检查 edge_label 格式
            if args.task == 'classification':
                if hasattr(batch, 'edge_label') and batch.edge_label.dim() > 1 and batch.edge_label.size(1) == 2:
                    # edge_label 格式为 [regression_label, classification_label]
                    y_true = batch.edge_label[:, 1].long()
                elif y.dim() > 1 and y.size(1) == 2:
                    y_true = y[:, 1].long()
                else:
                    y_true = y
            else:
                y_true = y
            loss, pred = compute_loss(args, y_pred, y_true, criterion=criterion, sample_per_class=sample_per_class)
            _true = y_true.detach().to('cpu', non_blocking=True)
            _pred = y_pred.detach().to('cpu', non_blocking=True)
            
            loss.backward()
            optimizier.step()
            
            logger.update_stats(true=_true,
                                pred=pred.detach().to('cpu', non_blocking=True), 
                                batch_size=_true.size(0),
                                loss=loss.detach().cpu().item(),
                               )
                               
            if i > 0 and i % 100 == 0:
                current_mem = print_memory_usage()
                if current_mem > initial_mem * mem_critical_threshold:
                    print(f"批次 {i}: 内存使用过高，执行垃圾回收...")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
        print("Train results this epoch")
        train_results = logger.write_epoch()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print("Validate after epoch")
        val_results = eval_epoch(args, val_loader, val_batched_dspd, model, device, 
                               split='val', criterion=criterion, sample_per_class=sample_per_class)
        
        if args.task == 'classification':
            is_best = val_results["loss"] < best_results.get("best_val_loss", float('inf'))
            if is_best:
                best_results["best_val_loss"] = val_results["loss"]
                best_results["best_epoch"] = epoch
        else:
            is_best = val_results["mse"] < best_results["best_val_mse"]
            if is_best:
                best_results["best_val_mse"] = val_results["mse"]
                best_results["best_epoch"] = epoch
        
        if is_best:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            result_file.write(f"Epoch {epoch}, 验证集结果: {val_results}\n")
            test_results = []
            test_names = []
            for test_name, (test_loader, test_batched_dspd) in test_loaders.items():
                test_result = eval_epoch(args, test_loader, test_batched_dspd, model, device, 
                                      split=f'test_{test_name}', criterion=criterion, sample_per_class=sample_per_class)
                test_results.append(test_result)
                test_names.append(test_name)
                print(f"test_{test_name}", test_result)
                result_file.write(f"测试集 {test_name} 结果: {test_result}\n")
            result_file.write("\n")
            
            best_results["test_results"] = test_results
            best_results["test_names"] = test_names
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    result_file.close()
    return best_results


def downstream_train(args, dataset, device):
    """downstream task training for link prediction"""
    
    # 归一化特征和标签（包括分桶生成分类标签）
    NET = 0
    DEV = 1
    dataset['train'].norm_nfeat([NET, DEV])
    dataset['test'].norm_nfeat([NET, DEV])
    
    model = GraphHead(args)
    
    (
        train_loader, val_loader, test_loaders,
        train_dspd_list, valid_dspd_list, test_dspd_dict,
    ) = dataset_sampling_and_pe_calculation(args, dataset['train'], dataset['test'])

    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    optimizier = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 根据任务类型设置 criterion 和 sample_per_class
    sample_per_class = None
    if args.task == 'classification':
        # 分类任务统一使用 CrossEntropyLoss（模型输出 num_classes 个 logits）
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        print(f"Task is {args.task}, using CrossEntropyLoss for {args.num_classes}-class classification")
        
        # 计算每个类别的样本数量（用于 BalancedSoftmax）
        train_data = dataset['train']._data
        if hasattr(train_data, 'edge_label') and train_data.edge_label is not None:
            if train_data.edge_label.dim() > 1 and train_data.edge_label.size(1) == 2:
                class_labels = train_data.edge_label[:, 1].long()
            else:
                class_labels = train_data.edge_label.long()
            sample_per_class = get_sample_per_class(class_labels, args.num_classes, device)
    else:
        # 回归任务
        criterion = torch.nn.MSELoss(reduction='mean')
    
    start = time.time()

    train(args, model, optimizier, criterion,
          train_loader, val_loader, test_loaders, 
          train_dspd_list, valid_dspd_list, 
          test_dspd_dict, device, sample_per_class=sample_per_class)
    
    elapsed = time.time() - start
    timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed))
    print(f"Done! Training took {timestr}")
