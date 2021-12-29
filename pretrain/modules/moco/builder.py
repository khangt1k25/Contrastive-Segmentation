#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
#
# MoCo implementation based upon https://arxiv.org/abs/1911.05722
# 
# Pixel-wise contrastive loss based upon our paper


import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.common_config import get_model, get_filter, get_predictionHead
from modules.losses import BalancedCrossEntropyLoss, Regression_loss

class ContrastiveModel(nn.Module):
    def __init__(self, p):
        """
        p: configuration dict
        """
        super(ContrastiveModel, self).__init__()

        self.K = p['moco_kwargs']['K'] 
        self.m = p['moco_kwargs']['m'] 
        self.T = p['moco_kwargs']['T']

        # create the model 
        self.model_q = get_model(p)
        self.model_k = get_model(p)

        self.filter = get_filter(p)
        self.bg2fg_head = get_predictionHead(p)

        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.dim = p['model_kwargs']['ndim']
        self.register_buffer("queue", torch.randn(self.dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # balanced cross-entropy loss
        self.bce = BalancedCrossEntropyLoss(size_average=True)
        self.reg = Regression_loss()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, sal_q, sal_k):
        """
        Input:
            images: a batch of images (B x 3 x H x W) 
            sal: a batch of saliency masks (B x H x W)
        Output:
            logits, targets
        """
        batch_size = im_q.size(0)

        q, q_bg = self.model_q(im_q)         # queries: B x dim x H x W
        q = nn.functional.normalize(q, dim=1)
        q_reshape = q.reshape(batch_size, self.dim, -1) # B x dim x H.W
        q = q.permute((0, 2, 3, 1))          # queries: B x H x W x dim 
        q = torch.reshape(q, [-1, self.dim]) # queries: pixels x dim
        
        with torch.no_grad():
            sal_q_filter = self.filter(sal_q) * sal_q
            sal_q_flat = sal_q_filter.reshape(batch_size, -1, 1).type(q.dtype) # B x H.W x 1
        
        q_mean = torch.bmm(q_reshape, sal_q_flat).squeeze() # B x dim
        q_mean = nn.functional.normalize(q_mean, dim=1)

        # q_bg_mean = torch.bmm(q_reshape, 1.-sal_q.reshape(batch_size, -1, 1).type(q.dtype)).squeeze()
        # q_bg_mean = nn.functional.normalize(q_bg_mean, dim=1)


        # compute saliency loss
        sal_loss = self.bce(q_bg, sal_q)
   
        with torch.no_grad():
            offset = torch.arange(0, 2 * batch_size, 2).to(sal_q.device)
            sal_q = (sal_q + torch.reshape(offset, [-1, 1, 1]))*sal_q # all bg's to 0
            sal_q = sal_q.view(-1)
            mask_indexes = torch.nonzero((sal_q)).view(-1).squeeze()
            sal_q = torch.index_select(sal_q, index=mask_indexes, dim=0) // 2
        
        # compute key prototypes
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k, _ = self.model_k(im_k)  # keys: N x C x H x W
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            
            # prototypes k
            k = k.reshape(batch_size, self.dim, -1) # B x dim x H.W
            
            
            sal_k_filter = self.filter(sal_k) * sal_k

            sal_k = sal_k_filter.reshape(batch_size, -1, 1).type(q.dtype) # B x H.W x 1
            
            # sal_k = sal_k.reshape(batch_size, -1, 1).type(k.dtype) # B x H.W x 1
            
            prototypes_foreground = torch.bmm(k, sal_k).squeeze() # B x dim
            prototypes_foreground = nn.functional.normalize(prototypes_foreground, dim=1)



            # prototypes_background = torch.bmm(k, 1.-sal_k.reshape(batch_size, -1, 1).type(k.dtype)).squeeze()
            # prototypes_background = nn.functional.normalize(prototypes_background, dim=1)     

        # q: pixels x dim
        # k: pixels x dim
        # prototypes_k: proto x dim
        q = torch.index_select(q, index=mask_indexes, dim=0)
        l_batch = torch.matmul(q, prototypes_foreground.t())   # shape: pixels x proto
        negatives = self.queue.clone().detach()     # shape: dim x negatives
        l_mem = torch.matmul(q, negatives)          # shape: pixels x negatives (Memory bank)
        logits = torch.cat([l_batch, l_mem], dim=1) # pixels x (proto + negatives)

        ## Superpixel 
        l_positive = torch.matmul(q_mean, prototypes_foreground.t())
        l_negative = torch.matmul(q_mean, negatives)
        mean_logits = torch.cat([l_positive, l_negative], dim=1)
        mean_labels = torch.arange(mean_logits.shape[0]).to(q.device)
        bg_logits = torch.zeros([])
        bg_labels = torch.zeros([])

        ## Background: type1 
        # bg_positive = torch.einsum('ij, ij->i', q_bg_mean, prototypes_background)
        # bg_negative = torch.matmul(q_bg_mean, prototypes_foreground.t())
        # bg_logits = torch.cat([bg_positive.unsqueeze(1), bg_negative], dim=1)
        # bg_labels = torch.zeros(bg_logits.shape[0], dtype=torch.long).to(q.device)
        
        ## Background: type2 (combine with superpixel)
        # l_positive = torch.matmul(q_mean, prototypes_foreground.t()) # BxB
        # l_negative = torch.matmul(q_mean, negatives) # Bx mem
        # l_bg_negative = torch.matmul(q_mean, prototypes_background.t()) #BxB
        # mean_logits = torch.cat([l_positive, l_negative, l_bg_negative], dim=1)
        # mean_labels = torch.arange(mean_logits.shape[0]).to(q.device)
        # bg_logits = torch.zeros([])
        # bg_labels = torch.zeros([])
        
        ## Background: type3 
        # bg_positive = torch.matmul(q_bg_mean, prototypes_background.t()).t()
        # bg_positive = bg_positive.reshape(-1, 1) # (B^2, 1)
        # bg_negative = torch.matmul(q_bg_mean, prototypes_foreground.t())
        # bg_negative = torch.cat([bg_negative]*batch_size, dim=0) # (B^2, negatives)
        # bg_logits = torch.cat([bg_positive, bg_negative], dim=1) 
        # bg_labels = torch.zeros(bg_logits.shape[0], dtype=torch.long).to(q.device)

        ## Background: type4

        # predictedfg = self.bg2fg_head(q_bg_mean)
        # predictedfg = nn.functional.normalize(predictedfg, dim=1) 
        # bg2 = self.reg(predictedfg, prototypes_foreground)
        
        # bg2 = self.reg(predictedfg, prototypes_background)


        # apply temperature
        logits /= self.T
        mean_logits /= self.T
        bg_logits /= self.T

        # dequeue and enqueue
        self._dequeue_and_enqueue(prototypes_foreground) 

        return logits, sal_q, mean_logits, mean_labels, bg_logits, bg_labels, sal_loss


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
