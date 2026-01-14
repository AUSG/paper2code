import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, num_classes=10):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, pred, target):
        log_probs = F.log_softmax(pred, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None, num_classes=10):
        super(WeightedCrossEntropyLoss, self).__init__()
        if class_weights is None:
            self.class_weights = torch.ones(num_classes)
        else:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    def forward(self, inputs, targets):
        if inputs.is_cuda:
            self.class_weights = self.class_weights.cuda()
        
        weights = self.class_weights[targets]
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        weighted_loss = weights * ce_loss
        
        return weighted_loss.mean()


def get_cat_dog_focused_weights(num_classes=10, cat_idx=3, dog_idx=5, weight=2.0):
    weights = torch.ones(num_classes)
    weights[cat_idx] = weight
    weights[dog_idx] = weight
    return weights.tolist()

