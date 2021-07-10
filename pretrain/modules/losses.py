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



class CatInstConsistency:
    def __init__(self, reduction="mean", cons_type="neg_log_dot_prod"):
        assert reduction in ("none", "mean", "sum"), f"reduction={reduction}!"
        self.reduction = reduction

        possible_cons_types = ("xent", "jsd", "l2", "l1",
                               "neg_dot_prod", "neg_log_dot_prod")
        assert cons_type in possible_cons_types, \
            f"cons_type must be in {possible_cons_types}. Found {cons_type}!"
        self.cons_type = cons_type

    def get_cons(self, p_1, p_2):
        cons_type = self.cons_type

        if cons_type == "neg_log_dot_prod":
            cons = -(p_1 * p_2).sum(-1).log()
        elif cons_type == "neg_dot_prod":
            cons = -(p_1 * p_2).sum(-1)
        elif cons_type == "xent":
            cons = -(p_1.detach() * p_2.log()).sum(-1) \
                   -(p_2.detach() * p_1.log()).sum(-1)
            cons = cons * 0.5
        elif cons_type == "jsd":
            p_avg = 0.5 * (p_1 + p_2)

            cons = 0.5 * ((p_1 * (p_1.log() - p_avg.log())).sum(-1) +
                          (p_2 * (p_2.log() - p_avg.log())).sum(-1))
        elif cons_type == "l2":
            cons = (p_1 - p_2).pow(2).sum(-1)
        elif cons_type == "l1":
            cons = (p_1 - p_2).abs().sum(-1)
        else:
            raise ValueError(f"Do not support cons_type={cons_type}!")

        return cons

    def __call__(self, p_1, p_2):
        loss = self.get_cons(p_1, p_2)

        if self.reduction == "mean":
            loss = loss.mean(0)
        elif self.reduction == "sum":
            loss = loss.sum(0)

        return loss



def IIC_Loss(y, y_neighbor,C = 20,  lamb = 1., EPS= sys.float_info.epsilon):
    p_i_j = y.unsqueeze(2) * y_neighbor.unsqueeze(1)  # bn, k, k

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

    loss = torch.sum(loss)

    return loss 