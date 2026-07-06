"""Loss 模块"""
import torch
import torch.nn.functional as F


def balanced_softmax_loss(logits, labels, sample_per_class, reduction='mean'):
    """Balanced Softmax Cross Entropy Loss"""
    if isinstance(sample_per_class, list):
        sample_per_class = torch.tensor(sample_per_class).type_as(logits)
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    return F.cross_entropy(input=logits, target=labels, reduction=reduction)


def get_sample_per_class(labels, num_classes, device):
    """计算每个类别的样本数量"""
    counts = torch.zeros(num_classes, device=device)
    labels = labels.long().to(device)
    for i in range(num_classes):
        counts[i] = (labels == i).sum().float()
    counts = counts.clamp(min=1.0)
    print(f"类别分布: {counts.cpu().numpy()}")
    return counts
