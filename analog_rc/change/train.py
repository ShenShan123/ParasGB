"""
Training and evaluation functions.
支持节点任务(net节点)和边任务(pin-pair_to-pin边)。
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import os
import logging
from datetime import datetime

from loss import get_criterion


def train_model(args, model, train_loader, val_loader, test_loaders, device):
    """Main training loop."""
    
    # 获取已存在的logger（由main.py创建）
    logger = logging.getLogger('main')
    
    # 创建模型保存目录
    if args.task == 'regression':
        model_dir = 'model/regression'
    else:
        model_dir = 'model/classification'
    os.makedirs(model_dir, exist_ok=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
    
    # 获取损失函数
    criterion = get_criterion(args, device, train_loader)
    logger.info(f"损失函数: {type(criterion).__name__}")
    
    best_val_loss = float('inf')
    best_val_metrics = {}
    best_epoch = 0
    best_test_results = {}
    
    # 使用tqdm显示epoch进度
    epoch_pbar = tqdm(range(args.epochs), desc="Training Progress", position=0)
    
    for epoch in epoch_pbar:
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, args)
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, args)
        
        scheduler.step()
        
        # 更新进度条描述
        if args.task == 'regression':
            epoch_pbar.set_postfix({
                'Train Loss': f'{train_loss:.4f}',
                'Val Loss': f'{val_metrics["loss"]:.4f}',
                'Val R2': f'{val_metrics["r2"]:.4f}'
            })
            log_msg = (f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_metrics['loss']:.6f} | Val MAE: {val_metrics['mae']:.4f} | "
                      f"Val R2: {val_metrics['r2']:.4f}")
        else:
            epoch_pbar.set_postfix({
                'Train Loss': f'{train_loss:.4f}',
                'Val Loss': f'{val_metrics["loss"]:.4f}',
                'Val Acc': f'{val_metrics["accuracy"]:.4f}'
            })
            log_msg = (f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_metrics['loss']:.6f} | Val Acc: {val_metrics['accuracy']:.4f} | "
                      f"Val F1: {val_metrics['f1']:.4f}")
        
        # 记录到日志文件（不打印到控制台，避免干扰进度条）
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.stream.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {log_msg}\n")
                handler.stream.flush()
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_val_metrics = val_metrics.copy()  # 保存最佳验证集指标
            best_epoch = epoch
            
            # Evaluate on test sets
            best_test_results = {}
            for name, test_loader in test_loaders.items():
                test_metrics = evaluate(model, test_loader, criterion, device, args)
                best_test_results[name] = test_metrics
                if args.task == 'regression':
                    test_log = f"  Test {name}: Loss={test_metrics['loss']:.6f}, MAE={test_metrics['mae']:.4f}, MSE={test_metrics['mse']:.6f}, RMSE={test_metrics['rmse']:.4f}, R2={test_metrics['r2']:.4f}"
                else:
                    test_log = f"  Test {name}: Loss={test_metrics['loss']:.6f}, Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}"
                for handler in logger.handlers:
                    if isinstance(handler, logging.FileHandler):
                        handler.stream.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {test_log}\n")
                        handler.stream.flush()
            
            # Save model
            model_name = f"{args.model}_h{args.hid_dim}_l{args.num_layers}_d{args.dropout}_lr{args.lr}"
            model_path = f"{model_dir}/best_{args.task_level}_{model_name}.pt"
            torch.save(model.state_dict(), model_path)
            save_log = f"  Model saved to {model_path}"
            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.stream.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {save_log}\n")
                    handler.stream.flush()
    
    # 训练结束，打印最终结果
    logger.info(f"\n{'=' * 60}")
    logger.info(f"训练完成!")
    logger.info(f"最佳结果 (Epoch {best_epoch}):")
    logger.info(f"-" * 40)
    logger.info(f"验证集 (Val):")
    if args.task == 'regression':
        logger.info(f"  Loss: {best_val_metrics['loss']:.6f}")
        logger.info(f"  MAE:  {best_val_metrics['mae']:.4f}")
        logger.info(f"  MSE:  {best_val_metrics['mse']:.6f}")
        logger.info(f"  RMSE: {best_val_metrics['rmse']:.4f}")
        logger.info(f"  R2:   {best_val_metrics['r2']:.4f}")
    else:
        logger.info(f"  Loss: {best_val_metrics['loss']:.6f}")
        logger.info(f"  Acc:  {best_val_metrics['accuracy']:.4f}")
        logger.info(f"  F1:   {best_val_metrics['f1']:.4f}")
    
    logger.info(f"-" * 40)
    logger.info(f"测试集 (Test):")
    for name, metrics in best_test_results.items():
        logger.info(f"  [{name}]:")
        if args.task == 'regression':
            logger.info(f"    Loss: {metrics['loss']:.6f}")
            logger.info(f"    MAE:  {metrics['mae']:.4f}")
            logger.info(f"    MSE:  {metrics['mse']:.6f}")
            logger.info(f"    RMSE: {metrics['rmse']:.4f}")
            logger.info(f"    R2:   {metrics['r2']:.4f}")
        else:
            logger.info(f"    Loss: {metrics['loss']:.6f}")
            logger.info(f"    Acc:  {metrics['accuracy']:.4f}")
            logger.info(f"    F1:   {metrics['f1']:.4f}")
    logger.info(f"{'=' * 60}")


def train_epoch(model, loader, optimizer, criterion, device, args):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    num_samples = 0
    
    # 使用tqdm显示batch进度
    batch_pbar = tqdm(loader, desc="Training", leave=False, position=1)
    
    for batch in batch_pbar:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        pred, y = model(batch)
        
        # 跳过空批次
        if pred.numel() == 0:
            continue
        
        if args.task == 'regression':
            pred = pred.view(-1)
            y = y.view(-1)
            
            loss = criterion(pred, y)
        else:
            # 确保y是1D张量
            y = y.view(-1).long()
            loss = criterion(pred, y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * pred.size(0)
        num_samples += pred.size(0)
        num_batches += 1
        
        # 更新batch进度条
        batch_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    return total_loss / max(num_samples, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, args):
    """Evaluate model on a dataset."""
    model.eval()
    
    all_preds = []
    all_labels = []
    total_loss = 0
    num_samples = 0
    
    # 使用tqdm显示评估进度
    eval_pbar = tqdm(loader, desc="Evaluating", leave=False, position=1)
    
    for batch in eval_pbar:
        batch = batch.to(device)
        pred, y = model(batch)
        
        # 跳过空批次
        if pred.numel() == 0:
            continue
        
        if args.task == 'regression':
            pred = pred.view(-1)
            y = y.view(-1)
            
            loss = criterion(pred, y)
            all_preds.append(pred.cpu())
            all_labels.append(y.cpu())
        else:
            # 确保y是1D张量
            y = y.view(-1).long()
            loss = criterion(pred, y)
            pred_class = pred.argmax(dim=1)
            all_preds.append(pred_class.cpu())
            all_labels.append(y.cpu())
        
        total_loss += loss.item() * pred.size(0)
        num_samples += pred.size(0)
    
    if not all_preds:
        return {'loss': float('inf'), 'mae': float('inf'), 'mse': float('inf'), 
                'rmse': float('inf'), 'r2': -float('inf')}
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    metrics = {'loss': total_loss / max(num_samples, 1)}
    
    if args.task == 'regression':
        metrics['mae'] = mean_absolute_error(all_labels, all_preds)
        metrics['mse'] = mean_squared_error(all_labels, all_preds)
        metrics['rmse'] = metrics['mse'] ** 0.5
        metrics['r2'] = r2_score(all_labels, all_preds) if len(all_labels) > 1 else 0.0
    else:
        metrics['accuracy'] = accuracy_score(all_labels, all_preds)
        metrics['f1'] = f1_score(all_labels, all_preds, average='macro')
    
    return metrics
