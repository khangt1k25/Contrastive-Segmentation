# Code referenced from https://github.com/facebookresearch/astmt

from cProfile import label
from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module


class BalancedCrossEntropyLoss(Module):
    """
    Balanced Cross Entropy Loss with optional ignore regions
    """

    def __init__(self, size_average=True, batch_average=True, pos_weight=None):
        super(BalancedCrossEntropyLoss, self).__init__()
        self.size_average = size_average
        self.batch_average = batch_average
        self.pos_weight = pos_weight

    def forward(self, output, labels, void_pixels=None):
        assert (output.size() == labels.size())

        # Weighting of the loss, default is HED-style
        if self.pos_weight is None:
            num_labels_pos = torch.sum(labels)
            num_labels_neg = torch.sum(1.0 - labels)
            num_total = num_labels_pos + num_labels_neg
            w = num_labels_neg / num_total
        else:
            w = self.pos_weight

        output_gt_zero = torch.ge(output, 0).float()
        loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
            1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))

        loss_pos_pix = -torch.mul(labels, loss_val)
        loss_neg_pix = -torch.mul(1.0 - labels, loss_val)

        if void_pixels is not None and not self.pos_weight:
            w_void = torch.le(void_pixels, 0.5).float()
            loss_pos_pix = torch.mul(w_void, loss_pos_pix)
            loss_neg_pix = torch.mul(w_void, loss_neg_pix)
            num_total = num_total - torch.ge(void_pixels, 0.5).float().sum()
            w = num_labels_neg / num_total

        loss_pos = torch.sum(loss_pos_pix)
        loss_neg = torch.sum(loss_neg_pix)

        final_loss = w * loss_pos + (1 - w) * loss_neg

        if self.size_average:
            final_loss /= float(np.prod(labels.size()))
        elif self.batch_average:
            final_loss /= labels.size()[0]

        return final_loss


class Regression_loss(Module):
    def __init__(self):
        super(Regression_loss, self).__init__()
    def forward(self, output, labels):
        # x = F.normalize(output, dim=1)
        # y = F.normalize(labels, dim=1)
        return (2 - 2 * (output * labels).sum(dim=-1)).mean()

class Clustering_loss(Module):
    def __init__(self):
        super(Clustering_loss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.temperature = 1.0

    def forward(self, c_q, c_k):
        '''
        c_q, c_k: Bxclusters
        '''
        B=c_q.shape[0]

        p_q = c_q.sum(0).view(-1)
        p_q /= p_q.sum()
        ne_loss = torch.log(p_q.size(0)) + (p_q * torch.log(p_q)).sum() 
        

        logits = torch.matmul(c_q.t(), c_k)
        labels = torch.arange(logits.shape[0]).to(c_q.device)
        
        logits /=  self.temperature

        cluster_loss = self.criterion(logits, labels)/B
        
        return cluster_loss, ne_loss
