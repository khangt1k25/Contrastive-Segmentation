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

from utils.common_config import get_model, get_filter
from modules.losses import BalancedCrossEntropyLoss

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
        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient


        # create the queue
        self.dim = p['model_kwargs']['ndim']
        self.register_buffer("obj_queue", torch.randn(self.dim, self.K))
        self.obj_queue = nn.functional.normalize(self.obj_queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


        # balanced cross-entropy loss
        self.bce = BalancedCrossEntropyLoss(size_average=True)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys_obj):

        batch_size = keys_obj.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.obj_queue[:, ptr:ptr + batch_size] = keys_obj.T

        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr




    def forward(self, im_q, sal_q, im_k, sal_k, classfier):
    
        batch_size, dim, H, W = im_q.shape
    
        q, q_bg = self.model_q(im_q)         # queries: B x dim x H x W
        q = nn.functional.normalize(q, dim=1)
        
        # compute saliency loss
        sal_loss = self.bce(q_bg, sal_q)


        cluster = classfier(q) # BxCxHxW
        cluster = cluster.permute((0, 2, 3, 1)) # BxHxWxC
        cluster = torch.reshape(cluster, [-1, cluster.shape[-1]]) # BHW x C
        with torch.no_grad():
            pseudo_labels = torch.argmax(cluster, dim=1).long() # BHW
                
        

        with torch.no_grad():
            offset = torch.arange(0, 2 * batch_size, 2).to(sal_q.device)
            sal_q = (sal_q + torch.reshape(offset, [-1, 1, 1]))*sal_q # all bg's to 0
            sal_q = sal_q.view(-1)
            mask_indexes = torch.nonzero((sal_q)).view(-1).squeeze()
            sal_q = torch.index_select(sal_q, index=mask_indexes, dim=0) // 2
        

        with torch.no_grad():
            self._momentum_update_key_encoder()  # update the key encoder
            k, _ = self.model_k(im_k)  # keys: B x dim x H x W
            k = nn.functional.normalize(k, dim=1)
            k = k.reshape(batch_size, self.dim, -1) # B x dim x H.W
            sal_k = sal_k.reshape(batch_size, -1, 1).type(q.dtype)
            prototypes_obj = torch.bmm(k, sal_k).squeeze() # B x dim
            prototypes_obj = nn.functional.normalize(prototypes_obj, dim=1) 


        q = q.permute((0, 2, 3, 1))          # queries: B x H x W x dim 
        q = torch.reshape(q, [-1, self.dim]) # queries: BHW x dim

        # Select the object pixels
        q = torch.index_select(q, index=mask_indexes, dim=0)  # pixels x dim
        
        

        cluster = torch.index_select(cluster, index=mask_indexes, dim=0) # pixels x cluster
        
        opt = 2 # 1: push away other cluster, 2: do not push
        if opt == 1:
            pseudo_labels = torch.index_select(pseudo_labels, index=mask_indexes, dim=0)    # pixels
        elif opt == 2:
            cluster = torch.max(cluster, cluster, dim=1).values.unsqueeze(1) # pixels x 1
            pseudo_labels = torch.zeros(cluster.shape[0]) # pixels


        batch_logits = torch.matmul(q, prototypes_obj.t()) #pixels x B
        mask = torch.ones_like(batch_logits).scatter_(1, sal_q.unsqueeze(1), 0.)
        batch_logits = batch_logits[mask.bool()].view(batch_logits.shape[0], -1)
        


        bank_obj = self.obj_queue.clone().detach()         # dim x negatives
        bank_logits =  torch.matmul(q, bank_obj)          # pixels x negatives
        

        logits  = torch.cat([cluster, batch_logits, bank_logits], dim=1) # pixels x (cluster+ negatives)
        
        self._dequeue_and_enqueue(prototypes_obj) 


        logits /= self.T
        
        return logits, pseudo_labels.long(), sal_loss





