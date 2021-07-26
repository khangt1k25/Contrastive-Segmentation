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


class ConInstContrast:
    def __init__(self, num_samples, temperature, device, reduction="mean"):
        super(ConInstContrast, self).__init__()
        self.num_samples = num_samples
        self.temperature = temperature
        self.device = device
        self.reduction = reduction

        # This mask is different from the mask in version 2
        # It also mask 2 sub-diagonals
        self.mask = self.get_mask(num_samples)
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)

    def get_mask(self, num_samples):
        B = num_samples

        mask = torch.full((2 * B, 2 * B), True, dtype=torch.bool,
                          device=self.device, requires_grad=False)
        mask = mask.fill_diagonal_(False)

        for i in range(B):
            mask[i, B + i] = False
            mask[B + i, i] = False

        return mask


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

class CatInstContrast(ConInstContrast):
    def __init__(self, num_samples, device, reduction="mean",
                 critic_type="log_dot_prod"):
        super(CatInstContrast, self).__init__(num_samples, None, device, reduction)
        self.critic_type = critic_type

    def critic(self, p, normalized):
        if self.critic_type == "log_dot_prod":
            return torch.matmul(p, p.t()).log()

        elif self.critic_type == "dot_prod":
            return torch.matmul(p, p.t())

        elif self.critic_type == "neg_l2" or self.critic_type == "nsse":
            return -(p.unsqueeze(1) - p.unsqueeze(0)).pow(2).sum(-1)

        elif self.critic_type == "neg_jsd":
            p1 = p.unsqueeze(1)
            p2 = p.unsqueeze(0)
            p_avg = 0.5 * (p1 + p2)

            out = -0.5 * ((p1 * (p1.log() - p_avg.log())).sum(-1) +
                          (p2 * (p2.log() - p_avg.log())).sum(-1))

            return out

        else:
            raise ValueError(f"Do not support critic_type={self.critic_type}!")

    def __call__(self, p_1, p_2, return_sim_matrix=False):
        return super(CatInstContrast, self).__call__(
            p_1, p_2, normalized=True,
            return_sim_matrix=return_sim_matrix)



class ConsistencyLoss(Module):
    """
    Consistency Loss  
    """

    def __init__(self, type='l2norm'):
        super(ConsistencyLoss, self).__init__()
        self.type = type
        assert(self.type in ['l2norm', 'l2', 'negativecosine'])

    def forward(self, output, labels, mask):
        assert (output.size() == labels.size())

        if self.type == 'l2norm':
            output = F.normalize(output, dim=-1)
            labels = F.normalize(labels, dim=-1)
            mask = mask.unsqueeze(-1)
            x = output * mask
            y = labels * mask
            x =  x.reshape((-1, x.shape[-1]))
            y =  y.reshape((-1, y.shape[-1]))

            return (((x-y) ** 2).sum(dim=-1)).mean()
        
        elif self.type == 'l2':
            mask = mask.unsqueeze(-1)
            x = output * mask
            y = labels * mask
            x =  x.reshape((-1, x.shape[-1]))
            y =  y.reshape((-1, y.shape[-1]))

            return (((x-y) ** 2).sum(dim=-1)).mean()


        




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