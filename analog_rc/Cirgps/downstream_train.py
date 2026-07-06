import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score,
    mean_absolute_error, mean_squared_error,
    root_mean_squared_error, r2_score,
)

# from torch.utils.data.sampler import SubsetRandomSampler
# from sram_dataset import LinkPredictionDataset
# from sram_dataset import collate_fn, adaption_for_sgrl
# from torch_geometric.data import Batch

import time
import os
from tqdm import tqdm
from model import GraphHead, NodeHead
from sampling_and_pe import dataset_sampling_and_pe_calculation, node_dataset_sampling_and_pe_calculation

# 节点类型 (与 newgraph.py 一致)
DEV = 0
PIN = 1
NET = 2

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
        if len(pred_score.shape) == 1 or pred_score.shape[1] == 1:
            return (pred_score > 0.5).astype(int)
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
        true = true.numpy()
        pred_score = pred_score.numpy()
        reformat = lambda x: round(float(x), 4)

        if self.task == 'classification':
            # 检查是否是多分类任务 (pred_score 已经是类别索引)
            if len(pred_score.shape) == 1:
                # pred_score 已经是预测的类别索引
                pred_int = pred_score.astype(int)
            else:
                pred_int = self._get_pred_int(pred_score)
            
            # 确保 true 也是整数类型
            true = true.astype(int)
            
            # 检查类别数量
            num_classes = max(true.max(), pred_int.max()) + 1
            
            if num_classes > 2:
                # 多分类任务
                res = {
                    'loss': round(self._loss / self._size_current, 8),
                    'accuracy': reformat(accuracy_score(true, pred_int)),
                    'precision_macro': reformat(precision_score(true, pred_int, average='macro', zero_division=0)),
                    'recall_macro': reformat(recall_score(true, pred_int, average='macro', zero_division=0)),
                    'f1_macro': reformat(f1_score(true, pred_int, average='macro', zero_division=0)),
                }
            else:
                # 二分类任务
                try:
                    r_a_score = roc_auc_score(true, pred_score)
                except ValueError:
                    r_a_score = 0.0

                res = {
                    'loss': round(self._loss / self._size_current, 8),
                    'accuracy': reformat(accuracy_score(true, pred_int)),
                    'precision': reformat(precision_score(true, pred_int, zero_division=0)),
                    'recall': reformat(recall_score(true, pred_int, zero_division=0)),
                    'f1': reformat(f1_score(true, pred_int, zero_division=0)),
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

        # 结果打印到屏幕
        print(split, res)
        return res

def compute_loss(args, pred, true, criterion):
    """Compute loss and prediction score. 
    Args:
        args (argparse.Namespace): The arguments
        pred (torch.tensor): Unnormalized prediction
        true (torch.tensor): Ground truth label (已根据任务类型选择好)
        criterion (torch.nn.Module): The loss function
    Returns: Loss, normalized prediction score
    """
    assert criterion, "Loss function is not provided!"
    
    if args.task == 'classification':
        # 多分类任务
        # pred: [batch_size, num_classes], true: [batch_size] (类别索引)
        true = true.long()
        loss = criterion(pred, true)
        # 返回预测的类别
        pred_class = pred.argmax(dim=1)
        return loss, pred_class
        
    elif args.task == 'regression':
        pred = pred.squeeze(-1) if pred.ndim > 1 else pred
        true = true.squeeze(-1) if true.ndim > 1 else true
        true = true.float()
        return criterion(pred, true), pred
    
    else:
        raise ValueError(f"Task type {args.task} not supported!")

@torch.no_grad()
def eval_epoch(args, loader, batched_dspd, model, device, 
               split='val', criterion=None):
    """ 
    evaluate the model on the validation or test set
    Args:
        args (argparse.Namespace): The arguments
        loader (torch.utils.data.DataLoader): The data loader
        model (torch.nn.Module): The model
        device (torch.device): The device to run the model on
        split (str): The split name, 'val' or 'test'
    """
    model.eval()
    time_start = time.time()
    logger = Logger(task=args.task)

    for i, batch in enumerate(tqdm(loader, desc="eval_"+split, leave=False)):
        ## copy dspd tensor to the batch
        batch.dspd = batched_dspd[i]
        pred, true = model(batch.to(device))
        loss, pred_score = compute_loss(args, pred, true, criterion=criterion)
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
          test_batched_dspd_dict, device):
    """
    Train the head model for link prediction task
    Args:
        args (argparse.Namespace): The arguments
        head_model (torch.nn.Module): The head model
        optimizier (torch.optim.Optimizer): The optimizer
        criterion (torch.nn.Module): The loss function
        train_loader (torch.utils.data.DataLoader): The training data loader
        val_loader (torch.utils.data.DataLoader): The validation data loader  
        test_laders (list): A list of test data loaders
        train_batched_dspd (list): The list of batched DSPD tensors for training
        val_batched_dspd (list): The list of batched DSPD tensors for validation
        test_batched_dspd_dict (dict): The dictionary of batched DSPD tensors for test datasets
        device (torch.device): The device to train the model on
    """
    # 导入内存监控工具
    import psutil
    import gc
    
    # 内存监控函数
    def print_memory_usage():
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_gb = memory_info.rss / (1024 ** 3)
        # print(f"当前内存使用: {memory_gb:.2f} GB")
        # if torch.cuda.is_available():
        #     for i in range(torch.cuda.device_count()):
        #         gpu_mem_alloc = torch.cuda.memory_allocated(i) / (1024 ** 3)
        #         gpu_mem_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
        #         print(f"GPU {i} 显存使用: {gpu_mem_alloc:.2f} GB (已分配), {gpu_mem_reserved:.2f} GB (已预留)")
        return memory_gb
    
    optimizier.zero_grad()
    
    best_results = {
        'best_val_mse': 1e9, 'best_val_loss': 1e9, 
        'best_epoch': 0, 'test_results': [], 'test_names': []
    }
    
    # 创建结果文件
    result_file = open('test_results.txt', 'w')
    result_file.write(f"训练参数 u: {args.u}\n")
    result_file.write(f"训练数据集: {args.train_dataset}, 采样率: {args.train_sample_rate}\n")
    result_file.write(f"测试数据集: {args.test_dataset}, 采样率: {args.test_sample_rate}\n\n")
    
    # 检查初始内存
    print("训练开始前内存使用情况:")
    initial_mem = print_memory_usage()
    
    # 设置内存检查阈值
    mem_warning_threshold = 0.9  # 90%的初始内存使用时发出警告
    mem_critical_threshold = 2.0  # 2倍初始内存使用时主动减少内存使用
    
    for epoch in range(args.epochs):
        # 检查训练前的内存使用
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")
        print("训练前内存状态:")
        current_mem = print_memory_usage()
        
        # 如果内存使用超过警告阈值，进行垃圾回收
        if current_mem > initial_mem * mem_warning_threshold:
            print(f"内存使用增加到 {current_mem:.2f} GB，执行垃圾回收...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 检查是否超过临界阈值
            if current_mem > initial_mem * mem_critical_threshold:
                print("内存使用超过临界值，尝试减少不必要的数据...")
                # 在极端情况下，可以考虑减小训练数据或减少缓存
                
        logger = Logger(task=args.task)
        model.train()

        for i, batch in enumerate(tqdm(train_loader, desc=f'Epoch:{epoch}')):
            optimizier.zero_grad()
            ## copy dspd tensor to the data batch
            batch.dspd = train_batched_dspd[i]
            
            ## Get the prediction from the model
            y_pred, y = model(batch.to(device))
            loss, pred = compute_loss(args, y_pred, y, criterion=criterion)
            _true = y.detach().to('cpu', non_blocking=True)
            _pred = pred.detach().to('cpu', non_blocking=True)
            
            loss.backward()
            optimizier.step()
            
            logger.update_stats(true=_true,
                                pred=pred.detach().to('cpu', non_blocking=True), 
                                batch_size=_true.size(0),
                                loss=loss.detach().cpu().item(),
                               )
                               
            # 每100个批次检查一次内存使用情况
            if i > 0 and i % 100 == 0:
                # 检查内存使用
                current_mem = print_memory_usage()
                if current_mem > initial_mem * mem_critical_threshold:
                    print(f"批次 {i}: 内存使用过高，执行垃圾回收...")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
        ## Get train results this epoch
        print("Train results this epoch")
        train_results = logger.write_epoch()

        # 验证前清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # validate
        print("Validate after epoch")
        val_results = eval_epoch(args, val_loader, val_batched_dspd, model, device, 
                               split='val', criterion=criterion)
        
        # 统一使用 loss 来判断最佳模型 (越小越好)
        is_best = val_results["loss"] < best_results["best_val_loss"]
        if is_best:
            best_results["best_val_loss"] = val_results["loss"]
            best_results["best_epoch"] = epoch
        
        # Test when getting best val results
        if is_best:
            # 测试前清理内存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            result_file.write(f"Epoch {epoch}, 验证集结果: {val_results}\n")
            test_results = []
            test_names = []
            for test_idx, test_loader in enumerate(test_loaders):
                test_batched_dspd = test_batched_dspd_dict[test_idx]
                test_result = eval_epoch(args, test_loader, test_batched_dspd, model, device, 
                                      split=f'test_{test_idx}', criterion=criterion)
                test_results.append(test_result)
                test_names.append(f"test_{test_idx}")
                result_file.write(f"测试集 {test_idx} 结果: {test_result}\n")
            result_file.write("\n")
            
            best_results["test_results"] = test_results
            best_results["test_names"] = test_names
            
        # Epoch结束后强制进行垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    result_file.close()
    return best_results

def _old_downstream_train(args, dataset, device):
    """ downstream task training for link prediction (deprecated, use downstream_train instead)
    Args:
        args (argparse.Namespace): The arguments
        dataset (dict): Dictionary containing 'train' and 'test' datasets
        device (torch.device): The device to train the model on
    """
    # 归一化节点特征和边标签 (关键步骤!)
    print("正在归一化训练集...")
    dataset['train'].norm_nfeat([DEV, PIN, NET])
    print("正在归一化测试集...")
    dataset['test'].norm_nfeat([DEV, PIN, NET])
    
    model = GraphHead(args)

    
    ## Subgraph sampling for each dataset graph & PE calculation
    (
        train_loader, val_loader, test_loaders,
        train_dspd_list, valid_dspd_list, test_dspd_dict,
    ) = dataset_sampling_and_pe_calculation(args, dataset['train'], dataset['test'])

    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    optimizier = torch.optim.Adam(model.parameters(),lr=args.lr)

    if args.task == 'classification':
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        print(f"Task is {args.task}, using BCEWithLogitsLoss")

    else:
        criterion = torch.nn.MSELoss(reduction='mean')
    
    start = time.time()

    ## Start training, go go go!
    train(args, model, optimizier, criterion,
          train_loader, val_loader, test_loaders, 
          train_dspd_list, valid_dspd_list, 
          test_dspd_dict, device)
    
    elapsed = time.time() - start
    timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed))
    print(f"Done! Training took {timestr}")


# ==================== 节点任务相关函数 ====================

@torch.no_grad()
def node_eval_epoch(args, loader, batched_dspd, model, device, 
                    split='val', criterion=None):
    """节点任务的评估函数"""
    model.eval()
    logger = Logger(task=args.task)
    
    for i, batch in enumerate(tqdm(loader, desc=f"eval_{split}", leave=False)):
        batch.dspd = batched_dspd[i]
        node_labels = loader.node_labels[i * args.batch_size:(i + 1) * args.batch_size]
        
        pred, true = model(batch.to(device), node_labels.to(device))
        loss, pred_score = compute_loss(args, pred, true, criterion=criterion)
        
        _true = true.detach().to('cpu', non_blocking=True)
        _pred = pred_score.detach().to('cpu', non_blocking=True)
        logger.update_stats(
            true=_true, pred=_pred,
            batch_size=_true.size(0),
            loss=loss.detach().cpu().item(),
        )
    
    return logger.write_epoch(split)


def node_train(args, model, optimizer, criterion,
               train_loader, val_loader, test_loaders,
               train_batched_dspd, val_batched_dspd,
               test_batched_dspd_dict, device):
    """节点任务的训练函数"""
    import psutil
    import gc
    
    def print_memory_usage():
        process = psutil.Process(os.getpid())
        memory_gb = process.memory_info().rss / (1024 ** 3)
        print(f"当前内存使用: {memory_gb:.2f} GB")
        return memory_gb
    
    optimizer.zero_grad()
    
    best_results = {
        'best_val_mse': 1e9, 'best_val_loss': 1e9,
        'best_epoch': 0, 'test_results': [], 'test_names': []
    }
    
    result_file = open('test_results.txt', 'w')
    result_file.write(f"节点任务训练\n")
    result_file.write(f"训练数据集: {args.train_dataset}\n")
    result_file.write(f"测试数据集: {args.test_dataset}\n\n")
    
    print("训练开始前内存使用情况:")
    initial_mem = print_memory_usage()
    
    for epoch in range(args.epochs):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")
        
        # 确保使用正确的任务类型
        task_for_logger = args.task
        print(f"Logger task type: {task_for_logger}")
        logger = Logger(task=task_for_logger)
        model.train()
        
        for i, batch in enumerate(tqdm(train_loader, desc=f'Epoch:{epoch}')):
            optimizer.zero_grad()
            batch.dspd = train_batched_dspd[i]
            
            # 获取当前批次的标签
            start_idx = i * args.batch_size
            end_idx = min((i + 1) * args.batch_size, train_loader.node_labels.size(0))
            node_labels = train_loader.node_labels[start_idx:end_idx]
            
            y_pred, y = model(batch.to(device), node_labels.to(device))
            loss, pred = compute_loss(args, y_pred, y, criterion=criterion)
            
            _true = y.detach().to('cpu', non_blocking=True)
            _pred = pred.detach().to('cpu', non_blocking=True)
            
            loss.backward()
            optimizer.step()
            
            logger.update_stats(
                true=_true, pred=_pred,
                batch_size=_true.size(0),
                loss=loss.detach().cpu().item(),
            )
        
        print("Train results this epoch")
        train_results = logger.write_epoch()
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 验证
        print("Validate after epoch")
        val_results = node_eval_epoch(
            args, val_loader, val_batched_dspd, model, device,
            split='val', criterion=criterion
        )
        
        if args.task == 'classification':
            # 多分类任务使用 accuracy 作为最佳指标
            is_best = val_results.get("accuracy", 0) > best_results.get("best_val_accuracy", 0)
            if is_best:
                best_results["best_val_accuracy"] = val_results["accuracy"]
                best_results["best_epoch"] = epoch
        else:
            is_best = val_results["mse"] < best_results["best_val_mse"]
            if is_best:
                best_results["best_val_mse"] = val_results["mse"]
                best_results["best_epoch"] = epoch
        
        if is_best:
            result_file.write(f"Epoch {epoch}, 验证集结果: {val_results}\n")
            test_results = []
            test_names = []
            
            for test_idx, test_loader in enumerate(test_loaders):
                test_batched_dspd = test_batched_dspd_dict[test_idx]
                test_result = node_eval_epoch(
                    args, test_loader, test_batched_dspd, model, device,
                    split=f'test_{test_idx}', criterion=criterion
                )
                test_results.append(test_result)
                test_names.append(f"test_{test_idx}")
                result_file.write(f"测试集 {test_idx} 结果: {test_result}\n")
            result_file.write("\n")
            
            best_results["test_results"] = test_results
            best_results["test_names"] = test_names
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    result_file.close()
    return best_results


def downstream_train(args, dataset, device):
    """下游任务训练入口函数
    
    根据 task_level 选择边任务或节点任务
    """
    task_level = getattr(args, 'task_level', 'edge')
    num_classes = getattr(args, 'num_classes', 5)
    
    # 归一化节点特征 (标签在 process 阶段已处理好)
    print("正在归一化训练集...")
    dataset['train'].norm_nfeat([DEV, PIN, NET], num_classes=num_classes)
    print("正在归一化测试集...")
    dataset['test'].norm_nfeat([DEV, PIN, NET], num_classes=num_classes)
    
    if task_level == 'node':
        # 节点任务
        print("=== 节点任务训练 ===")
        model = NodeHead(args)
        
        (
            train_loader, val_loader, test_loaders,
            train_dspd_list, valid_dspd_list, test_dspd_dict,
        ) = node_dataset_sampling_and_pe_calculation(args, dataset['train'], dataset['test'])
        
        model = model.to(device)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        if args.task == 'classification':
            criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            print(f"Task is {args.task}, using CrossEntropyLoss, num_classes={num_classes}")
        else:
            criterion = torch.nn.MSELoss(reduction='mean')
            print(f"Task is {args.task}, using MSELoss")
        
        start = time.time()
        
        node_train(args, model, optimizer, criterion,
                   train_loader, val_loader, test_loaders,
                   train_dspd_list, valid_dspd_list,
                   test_dspd_dict, device)
        
        elapsed = time.time() - start
        timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed))
        print(f"Done! Training took {timestr}")
    
    else:
        # 边任务
        print("=== 边任务训练 ===")
        model = GraphHead(args)
        
        (
            train_loader, val_loader, test_loaders,
            train_dspd_list, valid_dspd_list, test_dspd_dict,
        ) = dataset_sampling_and_pe_calculation(args, dataset['train'], dataset['test'])
        
        model = model.to(device)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        if args.task == 'classification':
            criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            print(f"Task is {args.task}, using CrossEntropyLoss, num_classes={num_classes}")
        else:
            criterion = torch.nn.MSELoss(reduction='mean')
            print(f"Task is {args.task}, using MSELoss")
        
        start = time.time()
        
        train(args, model, optimizer, criterion,
              train_loader, val_loader, test_loaders,
              train_dspd_list, valid_dspd_list,
              test_dspd_dict, device)
        
        elapsed = time.time() - start
        timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed))
        print(f"Done! Training took {timestr}")