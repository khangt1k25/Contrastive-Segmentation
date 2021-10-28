# Code referenced from https://github.com/facebookresearch/astmt
import sys
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
        




class ConsistencyLoss(Module):
    """
    Consistency Loss  
    """

    def __init__(self, type='l2'):
        super(ConsistencyLoss, self).__init__()
        self.type = type
        assert(self.type in ['l2norm', 'l2'])

    def forward(self, output, labels, mask=None):
        assert (output.size() == labels.size())
        if mask is None:
            mask = torch.ones_like(labels, requires_grad=False)

        if self.type == 'l2norm':
            output = F.normalize(output, dim=-1)
            labels = F.normalize(labels, dim=-1)
            mask = mask.unsqueeze(-1)
            z = (output-labels)*mask
            z = z.reshape((-1, z.shape[-1]))
            return ((z**2).sum(dim=-1)).mean()
        
        elif self.type == 'l2':
            mask = mask.unsqueeze(-1)
            z = (output-labels)*mask
            z = z.reshape((-1, z.shape[-1]))
            return ((z**2).sum(dim=-1)).mean()


        

class AttentionLoss(Module):
    def __init__(self):
        super(AttentionLoss, self).__init__()
    def forward(self, x, mask, sal):
        bsz, dim, H, W = x.shape
        x = x.reshape(bsz, dim, -1) # B x dim x H.W

        mask = mask.reshape(bsz, -1)
        mask = torch.softmax(mask, dim=1)

        x_mean = torch.bmm(x, mask.unsqueeze(2)).squeeze()
        x_mean = nn.functional.normalize(x_mean, dim=1)

        sal = sal.reshape(bsz, -1)
        sal = 1. - sal
        mask = mask * sal
      
        attention_loss = ((mask-sal)**2).sum(dim=1).mean()
        return x_mean, attention_loss


def IIC_Loss(y1, y2 ,C = 20,  lamb = 1., EPS= sys.float_info.epsilon):
    p_i_j = y1.unsqueeze(2) * y2.unsqueeze(1)  # bn, k, k

    p_i_j = p_i_j.mean(dim=0)

    
    # Symmetric
    p_i_j = (p_i_j + p_i_j.t()) / 2.0
    assert p_i_j.size() == (C, C)


    # Nan problem fix:
    mask = ((p_i_j > EPS).data).type(torch.float32)
    p_i_j = p_i_j * mask + EPS * (1 - mask)

    # Computing the marginals
    p_i = p_i_j.sum(dim=1).view(C, 1).expand(C, C)
    p_j = p_i_j.sum(dim=0).view(1, C).expand(C, C)

    # Compute the mutual information
    loss = -p_i_j * (torch.log(p_i_j) - lamb * torch.log(p_j) - lamb * torch.log(p_i))

    loss = torch.mean(loss)

    return loss 