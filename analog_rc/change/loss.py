"""
损失函数模块。
支持多种回归和分类损失函数，用于处理标签不均衡问题。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ==================== 自定义损失函数 ====================

class WeightedMSELoss(nn.Module):
    """加权MSE：对高值样本赋予更大权重。w = 1 + alpha * y"""
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, pred, target):
        weights = 1.0 + self.alpha * target
        return (weights * (pred - target) ** 2).mean()


class WeightedMAELoss(nn.Module):
    """加权MAE：对高值样本赋予更大权重。"""
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, pred, target):
        weights = 1.0 + self.alpha * target
        return (weights * torch.abs(pred - target)).mean()


class FocalMSELoss(nn.Module):
    """Focal MSE：聚焦难样本。loss = |error|^gamma * MSE"""
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, pred, target):
        error = torch.abs(pred - target)
        return (error ** self.gamma * (pred - target) ** 2).mean()


class HuberLoss(nn.Module):
    """Huber损失：对异常值鲁棒（兼容低版本PyTorch）。"""
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, pred, target):
        error = torch.abs(pred - target)
        quadratic = torch.clamp(error, max=self.delta)
        linear = error - quadratic
        return (0.5 * quadratic ** 2 + self.delta * linear).mean()


class FocalLoss(nn.Module):
    """Focal Loss：聚焦难分类样本。FL = -(1-p)^gamma * log(p)"""
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha) if isinstance(alpha, (list, np.ndarray)) else alpha
    
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha.to(pred.device)[target] if isinstance(self.alpha, torch.Tensor) else self.alpha
            focal_weight = alpha_t * focal_weight
        return (focal_weight * ce_loss).mean()


class LabelSmoothingCE(nn.Module):
    """标签平滑交叉熵（兼容低版本PyTorch）。"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_classes = pred.size(1)
        log_probs = F.log_softmax(pred, dim=1)
        with torch.no_grad():
            smooth_labels = torch.zeros_like(log_probs).fill_(self.smoothing / (n_classes - 1))
            smooth_labels.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return (-smooth_labels * log_probs).sum(dim=1).mean()


# ==================== 工具函数 ====================

def get_class_weights(train_loader, num_classes, device):
    """计算类别权重（频率越低权重越大）。"""
    class_counts = torch.zeros(num_classes)
    for batch in train_loader:
        labels = None
        # 尝试从不同的属性获取标签
        if hasattr(batch, 'y'):
            if batch.y.dim() == 2:
                labels = batch.y[:, 1].long()
            elif batch.y.dim() == 1:
                labels = batch.y.long()
        elif hasattr(batch, 'edge_label'):
            if batch.edge_label.dim() == 2:
                labels = batch.edge_label[:, 1].long()
            elif batch.edge_label.dim() == 1:
                labels = batch.edge_label.long()
        
        if labels is None:
            continue
            
        for c in range(num_classes):
            class_counts[c] += (labels == c).sum().item()
    
    class_counts = class_counts.clamp(min=1)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * num_classes
    return weights.to(device)


def get_criterion(args, device, train_loader=None):
    """
    获取损失函数。
    
    回归: mse, mae, weighted_mse, weighted_mae, focal_mse, huber
    分类: ce, weighted_ce, focal, label_smoothing
    """
    if args.task == 'regression':
        loss_type = getattr(args, 'reg_loss', 'mse')
        losses = {
            'mse': nn.MSELoss(),
            'mae': nn.L1Loss(),
            'huber': HuberLoss(delta=getattr(args, 'loss_delta', 1.0)),
            'weighted_mse': WeightedMSELoss(alpha=getattr(args, 'loss_alpha', 1.0)),
            'weighted_mae': WeightedMAELoss(alpha=getattr(args, 'loss_alpha', 1.0)),
            'focal_mse': FocalMSELoss(gamma=getattr(args, 'loss_gamma', 2.0)),
        }
        return losses.get(loss_type, nn.MSELoss())
    
    else:  # classification
        loss_type = getattr(args, 'cls_loss', 'ce')
        weights = get_class_weights(train_loader, args.num_classes, device) if loss_type == 'weighted_ce' and train_loader else None
        if weights is not None:
            print(f"类别权重: {weights.cpu().numpy()}")
        
        losses = {
            'ce': nn.CrossEntropyLoss(),
            'weighted_ce': nn.CrossEntropyLoss(weight=weights) if weights is not None else nn.CrossEntropyLoss(),
            'focal': FocalLoss(gamma=getattr(args, 'loss_gamma', 2.0)),
            'label_smoothing': LabelSmoothingCE(smoothing=getattr(args, 'label_smoothing', 0.1)),
        }
        return losses.get(loss_type, nn.CrossEntropyLoss())
