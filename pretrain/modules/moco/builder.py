#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
#
# MoCo implementation based upon https://arxiv.org/abs/1911.05722
# 
# Pixel-wise contrastive loss based upon our paper


from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import compute_negative_euclidean

from utils.common_config import get_model
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


    def mc_forward(self, im_q, im_k, sal_q, sal_k):

        batch_size = im_q.size(0)

        q, q_bg = self.model_q(im_q)         # queries: B x dim x H x W
        q = nn.functional.normalize(q, dim=1)
        q = q.permute((0, 2, 3, 1))          # queries: B x H x W x dim 
        q = torch.reshape(q, [-1, self.dim]) # queries: pixels x dim

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
            k, _ = self.model_k(im_k)  # keys: N x C x H x W
            k = nn.functional.normalize(k, dim=1)
            # prototypes k
            k = k.reshape(batch_size, self.dim, -1) # B x dim x H.W
            sal_k = sal_k.reshape(batch_size, -1, 1).type(k.dtype) # B x H.W x 1
            prototypes_foreground = torch.bmm(k, sal_k).squeeze() # B x dim
            prototypes = nn.functional.normalize(prototypes_foreground, dim=1)        

        # q: pixels x dim
        # k: pixels x dim
        # prototypes_k: proto x dim
        q = torch.index_select(q, index=mask_indexes, dim=0)
        l_batch = torch.matmul(q, prototypes.t())   # shape: pixels x proto
        negatives = self.obj_queue.clone().detach()     # shape: dim x negatives
        l_mem = torch.matmul(q, negatives)          # shape: pixels x negatives (Memory bank)
        logits = torch.cat([l_batch, l_mem], dim=1) # pixels x (proto + negatives)

        # apply temperature
        logits /= self.T
        
        # dequeue and enqueue
        self._dequeue_and_enqueue(prototypes) 

        return logits, sal_q, sal_loss

    def forward(self, im_q, sal_q, im_k, sal_k, classifier=None, centroids=None, im_randaug=None, sal_randaug=None, loader=None, index=None):
            
        batch_size, dim, H, W = im_q.shape
        


        q, q_bg = self.model_q(im_q)         # queries: B x dim x H x W
        q = nn.functional.normalize(q, dim=1)

        if classifier:
            cluster = classifier(q)  # cosine
            cluster = cluster.permute((0, 2, 3, 1))
            cluster = torch.reshape(cluster, [-1, cluster.shape[-1]]) # BHW x C
            

            randaug, _ = self.model_q(im_randaug)
            randaug = nn.functional.normalize(randaug, dim=1)
            randaug = classifier(randaug) #consine
            randaug = randaug.permute((0, 2, 3, 1)) # BxCxHxW
            randaug = torch.reshape(randaug, [-1, randaug.shape[-1]]) # BHW x C


        # compute saliency loss
        sal_loss = self.bce(q_bg, sal_q)
        


        with torch.no_grad():
            self._momentum_update_key_encoder()  # update the key encoder
            k, _ = self.model_k(im_k)  # keys: B x C x H x W
            k = nn.functional.normalize(k, dim=1) #  B x C x H x W

            kernel = 3
            padding = int((kernel-1)//2)
            num_neigbor = F.avg_pool2d(sal_q.float(), kernel_size=kernel, stride=1, padding=padding) # BxHxW
            feat_neigbor = loader.dataset.apply_eqv(deepcopy(index), deepcopy(k)) # B x C x Hx W   
            feat_neigbor = feat_neigbor * sal_q # BxCxHxW
            feat_neigbor = F.avg_pool2d(feat_neigbor, kernel_size=kernel, stride=1, padding=padding, divisor_override=1) # sum_pooling: BxdimxHxW
            feat_neigbor = feat_neigbor * num_neigbor # BxCxHxW
            feat_neigbor = feat_neigbor.permute((0, 2, 3, 1))          # B x H x W x dim 
            feat_neigbor = torch.reshape(feat_neigbor, [-1, self.dim]) # BHW x dim
            

            



            pseudo_label = classifier(k) # B x C x H x W        
            pseudo_maxval = torch.softmax(pseudo_label/0.1, dim=1) # B x C x H x W
            pseudo_label = pseudo_label.topk(1, dim=1)[1].squeeze().long()
            pseudo_maxval = pseudo_maxval.topk(1, dim=1)[0].squeeze().detach()         
            threshold = 0.9
            pseudo_maxval = (pseudo_maxval > threshold).float()
            
            

            pseudo_label_query = loader.dataset.apply_eqv(deepcopy(index), deepcopy(pseudo_label)).flatten()  # BHW
            pseudo_label_randaug = loader.dataset.apply_randaug(deepcopy(index), deepcopy(pseudo_label), is_feat=1).flatten() # BHW
            pseudo_maxval = loader.dataset.apply_randaug(deepcopy(index), pseudo_maxval, is_feat=1).flatten()# BHW


            
            k = k.reshape(batch_size, self.dim, -1) # B x dim x H.W
            sal_k = sal_k.reshape(batch_size, -1, 1).type(q.dtype)
            
            prototypes_obj = torch.bmm(k, sal_k).squeeze() # B x dim
            prototypes_obj = nn.functional.normalize(prototypes_obj, dim=1) 
        
        with torch.no_grad():
            offset = torch.arange(0, 2 * batch_size, 2).to(sal_q.device)
            sal_q = (sal_q + torch.reshape(offset, [-1, 1, 1]))*sal_q # all bg's to 0
            sal_q = sal_q.view(-1)
            mask_indexes = torch.nonzero((sal_q)).view(-1).squeeze()
            sal_q = torch.index_select(sal_q, index=mask_indexes, dim=0) // 2
        
        with torch.no_grad():
            offset2 = torch.arange(0, 2 * batch_size, 2).to(sal_randaug.device)
            sal_randaug = (sal_randaug + torch.reshape(offset2, [-1, 1, 1]))*sal_randaug # all bg's to 0
            sal_randaug = sal_randaug.view(-1)
            mask_indexes2 = torch.nonzero((sal_randaug)).view(-1).squeeze()

        negatives = self.obj_queue.clone().detach()         # dim x K
        # compute pixel-level loss
        q = q.permute((0, 2, 3, 1))          # B x H x W x dim 
        q = torch.reshape(q, [-1, self.dim]) # BHW x dim
        q = torch.index_select(q, index=mask_indexes, dim=0)  # pixels x dim
        
        with torch.no_grad():
            feat_neigbor = torch.index_select(feat_neigbor, index=mask_indexes, dim=0) # pixels x dim
        

       

    
        pos = (q * feat_neigbor).sum(1).view(-1, 1)
        l_batch = torch.matmul(q, prototypes_obj.t())   # shape: pixels x proto
        pixels, proto = l_batch.shape[0], l_batch.shape[1]
        mask_batch = torch.ones_like(l_batch).scatter_(1, sal_q.unsqueeze(1), 0.)
        l_batch = l_batch[mask_batch.bool()].view(pixels, proto-1) #pixels x (proto -1)
        l_mem = torch.matmul(q, negatives)          # shape: pixels x negatives (Memory bank)
        logits = torch.cat([pos ,l_batch, l_mem], dim=1) # pixels x (1+ proto-1 + negatives)


        # compute cluster loss : Not use bg
        if classifier:
            cluster = torch.index_select(cluster, index=mask_indexes, dim=0) # pixels x C
            with torch.no_grad():
                pseudo_label_query =  torch.index_select(pseudo_label_query, index=mask_indexes, dim=0).long().detach() # pixels
            cluster /= 0.1


            randaug = torch.index_select(randaug, index=mask_indexes2, dim=0)
            with torch.no_grad():
                
                pseudo_label_randaug = torch.index_select(pseudo_label_randaug, index=mask_indexes2, dim=0).long().detach()
                pseudo_maxval = torch.index_select(pseudo_maxval, index=mask_indexes2, dim=0).float().detach()
            
            randaug /= 0.1

        self._dequeue_and_enqueue(prototypes_obj) 
        

        logits /= self.T
        
        if classifier:
            return logits, torch.zeros(size=(pixels, )).long().to(sal_q.device), cluster, pseudo_label_query, randaug, pseudo_label_randaug, pseudo_maxval, sal_loss
        else:
            return logits, sal_q, sal_loss

        

