import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np

# Source: https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/78109
class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logit, target):
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val + \
               ((-max_val).exp() + (-logit - max_val).exp()).log()

        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size())==2:
            loss = loss.sum(dim=1)
        return loss

# https://www.kaggle.com/backaggle/imet-fastai-starter-focal-and-fbeta-loss#Create-learner-with-densenet121-and-FocalLoss
class FbetaLoss(nn.Module):
    def __init__(self, beta=2):
        super(FbetaLoss, self).__init__()
        self.small_value = 1e-6
        self.beta = beta

    def forward(self, logits, labels):
        batch_size = logits.size()[0]
        p = torch.sigmoid(logits)
        l = labels
        num_pos = torch.sum(p, 1) + self.small_value
        num_pos_hat = torch.sum(l, 1) + self.small_value
        tp = torch.sum(l * p, 1)
        precise = tp / num_pos
        recall = tp / num_pos_hat
        fs = (1 + self.beta * self.beta) * precise * recall / (self.beta * self.beta * precise + recall + self.small_value)
        loss = fs.sum() / batch_size
        return (1 - loss).expand(1)

class CombineLoss(nn.Module):
    def __init__(self, gamma=2, beta=2):
        super(CombineLoss, self).__init__()
        self.fbeta_loss = FbetaLoss(beta=beta)
        self.focal_loss = FocalLoss(gamma=gamma)
        
    def forward(self, logits, labels):
        loss_beta = self.fbeta_loss(logits, labels)
        loss_focal = self.focal_loss(logits, labels)
        return 0.5 * loss_beta + 0.5 * loss_focal

class CombineLoss2(nn.Module):
    def __init__(self, beta=2):
        super(CombineLoss2, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.fbeta_loss = FbetaLoss(beta=beta)
        
    def forward(self, logits, labels):
        loss_bce = self.bce_loss(logits, labels)
        loss_beta = self.fbeta_loss(logits, labels)
        return 0.9 * loss_bce + 0.1 * loss_beta

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)